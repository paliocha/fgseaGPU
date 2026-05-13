// fsgea_multilevel.hpp — adaptive multilevel splitting Monte Carlo for very
// small GSEA p-values.
//
// The algorithm (Korotkevich, Sukhov, Sergushichev 2021) layers level
// thresholds T_0 < T_1 < ... < T_L: at each level we keep the top half of a
// sample population whose ES exceeds T_l, then mix them via an MCMC kernel
// that preserves the constraint |ES| >= T_l. The final p-value telescopes:
//
//     P(|ES| >= obs)  =  prod_l P(|ES| >= T_{l+1} | |ES| >= T_l)
//                     ~  (1/2)^L * (fraction at the last level exceeding obs)
//
// Each MCMC chain is sequential (each move conditions on the previous
// state), so the parallelism here comes from running pathways concurrently
// via std::execution::par_unseq. The GPU path doesn't help — this is a
// fundamentally serial-per-chain algorithm and that's the price of being
// able to probe p-values down to 1e-50.

#pragma once

#include "fsgea_core.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace fsgea::multilevel {

struct Config {
    std::int64_t sampleSize{101};   // chains kept per level; must be odd
    double       eps{1e-50};        // stop building levels once p < eps
    double       moveScale{1.0};    // MCMC iterations per move scaling factor
    std::int64_t seed{42};
    bool         oneSided{false};   // true => use scoreType=Pos exclusively
};

struct Result {
    double pval;
    double log2err;
    bool   floored;   // true if we hit the eps floor before bracketing obs
};

namespace detail {

// Asymptotic digamma + recurrence. We avoid Boost so the package is
// dependency-free on the C++ side. Accurate to ~1e-12 for x > 0.
inline double digamma(double x) {
    double r = 0.0;
    while (x < 6.0) { r -= 1.0 / x; x += 1.0; }
    double const t  = 1.0 / x;
    double const t2 = t * t;
    return r + std::log(x) - 0.5 * t
             - t2 * (1.0/12.0 - t2 * (1.0/120.0 - t2 / 252.0));
}

inline double trigamma(double x) {
    double r = 0.0;
    while (x < 6.0) { r += 1.0 / (x * x); x += 1.0; }
    double const t  = 1.0 / x;
    double const t2 = t * t;
    return r + t + 0.5 * t2
             + t * t2 * (1.0/6.0 - t2 * (1.0/30.0 - t2 / 42.0));
}

// E[log X] for X ~ Beta(a, b). Used for the final-level correction.
inline double betaMeanLog(double a, double b) {
    return digamma(a) - digamma(a + b);
}

// Approximate base-2 standard deviation of log p-value across L levels with
// N samples each. The classical multilevel splitting variance bound.
inline double log2errEstimate(std::int64_t levels, std::int64_t N) {
    if (levels <= 0 || N <= 1) return 0.0;
    double const halfN = static_cast<double>((N + 1) / 2);
    double const var   = static_cast<double>(levels) *
                         (trigamma(halfN) - trigamma(static_cast<double>(N + 1)));
    return std::sqrt(var) / std::log(2.0);
}

inline ScoreType collapseToPositive(ScoreType s) noexcept {
    return s == ScoreType::Neg ? ScoreType::Pos : s;
}

} // namespace detail

class EsRuler {
public:
    EsRuler(std::span<double const> stats,
            std::int64_t k,
            double gseaParam,
            ScoreType scoreType,
            Config cfg)
      : stats_(stats), k_(k),
        gseaParam_(gseaParam),
        signedScore_(scoreType),
        cfg_(cfg)
    {
        if (cfg_.sampleSize < 3 || (cfg_.sampleSize % 2) == 0) {
            throw std::invalid_argument(
                "multilevel sampleSize must be odd and >= 3");
        }
        // For scoreType == Neg we negate stats and treat the problem as the
        // positive-side tail; the answer is symmetric. For Std we score by
        // |running-sum extremum| which equals the positive-side ES of a
        // suitable signed walk; equivalent here.
        if (scoreType == ScoreType::Neg) {
            negatedStats_.assign(stats.begin(), stats.end());
            for (auto& s : negatedStats_) s = -s;
            std::ranges::reverse(negatedStats_); // keep "decreasing" order
            stats_ = std::span<double const>(negatedStats_);
        }
        scoreType_ = ScoreType::Pos;
    }

    // Build levels up to (and including) the one that brackets |obsEs|, or
    // until the p-value drops below cfg_.eps.
    void extend(double absObsEs) {
        auto const n = static_cast<std::int64_t>(stats_.size());
        if (k_ <= 0 || k_ >= n) return;

        std::mt19937_64 rng(static_cast<std::uint64_t>(cfg_.seed));
        auto const N = static_cast<std::size_t>(cfg_.sampleSize);

        // Initial uniform-random sample of N gene sets and their ES values.
        std::vector<std::vector<std::int32_t>> chains(N);
        std::vector<double> es(N);
        for (std::size_t i = 0; i < N; ++i) {
            chains[i].resize(static_cast<std::size_t>(k_));
            sampleWithoutReplacement(n, k_,
                std::span<std::int32_t>(chains[i]), rng);
            es[i] = calcEs(stats_, std::span<std::int32_t const>(chains[i]),
                           gseaParam_, scoreType_).es;
        }

        thresholds_.clear();
        // Each MCMC step does ~k moves; scale by user knob and pathway size.
        auto const mcmcSteps = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(
                   std::ceil(cfg_.moveScale * static_cast<double>(k_))));

        // Build levels until the median of |ES| exceeds the observed value,
        // or until we've descended below the eps p-value floor.
        double const floorL = -std::log2(cfg_.eps); // max levels before floor
        std::int64_t level = 0;
        while (true) {
            // Sort by ES ascending; median is at index N/2.
            std::vector<std::size_t> order(N);
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
                return es[a] < es[b];
            });

            double const medianEs = es[order[N / 2]];
            thresholds_.push_back(medianEs);

            if (medianEs >= absObsEs) {
                // Final level brackets observed — we'll use this population.
                finalEs_.clear();
                finalEs_.reserve(N);
                for (std::size_t idx : order) finalEs_.push_back(es[idx]);
                return;
            }
            if (static_cast<double>(level + 1) >= floorL) {
                floored_ = true;
                finalEs_.clear();
                finalEs_.reserve(N);
                for (std::size_t idx : order) finalEs_.push_back(es[idx]);
                return;
            }

            // Top half survives; bottom half gets replaced with copies of
            // randomly chosen survivors, then mixed by MCMC.
            std::size_t const surviveStart = N / 2 + 1; // strictly above median
            std::size_t const survivors    = N - surviveStart;
            if (survivors == 0) {
                // Degenerate: all values tied at median; can't split further.
                floored_ = true;
                finalEs_.clear();
                finalEs_.reserve(N);
                for (std::size_t idx : order) finalEs_.push_back(es[idx]);
                return;
            }

            std::uniform_int_distribution<std::size_t> pick(0, survivors - 1);
            for (std::size_t j = 0; j < surviveStart; ++j) {
                std::size_t const src = order[surviveStart + pick(rng)];
                std::size_t const dst = order[j];
                chains[dst] = chains[src];
                es[dst]     = es[src];
            }

            // MCMC mix each chain under the constraint |ES| >= medianEs.
            for (std::size_t j = 0; j < N; ++j) {
                for (std::int64_t step = 0; step < mcmcSteps; ++step) {
                    mcmcMove(chains[j], es[j], medianEs, rng);
                }
            }

            ++level;
        }
    }

    // Convert the built ruler into a (p-value, log2err) result for `absObs`.
    [[nodiscard]] Result pvalue(double absObsEs) const {
        auto const L = static_cast<std::int64_t>(thresholds_.size());
        if (L == 0) return {1.0, 0.0, false};

        auto const N = static_cast<std::int64_t>(cfg_.sampleSize);
        std::int64_t exceed = 0;
        for (double e : finalEs_) if (e >= absObsEs) ++exceed;

        // Telescope: (1/2)^(L-1) * conditional fraction at the last level.
        // Add Bayesian smoothing (Beta(exceed+1, N-exceed+1) mean in log
        // space, which is the standard trick to avoid log(0) when exceed=0).
        double const logCondP =
            detail::betaMeanLog(static_cast<double>(exceed + 1),
                                static_cast<double>(N - exceed + 1));
        double const logP = -(L - 1) * std::log(2.0) + logCondP;
        double const p    = std::exp(logP);
        double const err  = detail::log2errEstimate(L, N);
        bool   const fl   = floored_ || p < cfg_.eps;
        return {std::clamp(p, cfg_.eps, 1.0), err, fl};
    }

private:
    std::span<double const>   stats_;
    std::vector<double>       negatedStats_;
    std::int64_t              k_;
    double                    gseaParam_;
    ScoreType                 signedScore_;  // as requested by caller
    ScoreType                 scoreType_;    // collapsed to Pos internally
    Config                    cfg_;
    std::vector<double>       thresholds_;
    std::vector<double>       finalEs_;
    bool                      floored_{false};

    // Single Metropolis–Hastings move: pick a member, replace with a uniform
    // non-member; accept if the new ES still satisfies the level constraint.
    void mcmcMove(std::vector<std::int32_t>& chain,
                  double& chainEs,
                  double threshold,
                  std::mt19937_64& rng) const
    {
        auto const n = static_cast<std::int64_t>(stats_.size());
        std::uniform_int_distribution<std::int32_t>
            posDist(0, static_cast<std::int32_t>(k_ - 1));
        std::uniform_int_distribution<std::int32_t>
            geneDist(0, static_cast<std::int32_t>(n - 1));

        std::int32_t const slot = posDist(rng);
        std::int32_t const oldGene = chain[static_cast<std::size_t>(slot)];

        // Sample a new gene not already in the chain.
        std::int32_t newGene;
        do {
            newGene = geneDist(rng);
        } while (std::ranges::find(chain, newGene) != chain.end());

        // Replace and re-sort the chain (k is typically small; O(k) insert).
        chain[static_cast<std::size_t>(slot)] = newGene;
        std::ranges::sort(chain);

        double const newEs = calcEs(
            stats_, std::span<std::int32_t const>(chain),
            gseaParam_, scoreType_).es;

        if (newEs >= threshold) {
            chainEs = newEs;
            return;
        }
        // Reject — restore.
        auto it = std::ranges::find(chain, newGene);
        *it = oldGene;
        std::ranges::sort(chain);
        // chainEs unchanged
    }
};

// Run multilevel on many pathways concurrently. `pathwayPositions[i]` must be
// sorted ascending and zero-based; gives the positions of gene-set members in
// the ranked `stats` vector.
struct PathwayMlResult {
    double es;
    double pval;
    double log2err;
    bool   floored;
};

inline std::vector<PathwayMlResult> runMultilevel(
    std::span<double const> stats,
    std::vector<std::vector<std::int32_t>> const& pathwayPositions,
    double gseaParam,
    ScoreType scoreType,
    Config cfg)
{
    std::vector<PathwayMlResult> out(pathwayPositions.size());
    std::vector<std::size_t> idx(pathwayPositions.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
        [&](std::size_t i) {
            auto const& pos = pathwayPositions[i];
            auto obs = calcEs(stats,
                              std::span<std::int32_t const>(pos),
                              gseaParam, scoreType);

            Config local = cfg;
            // Per-pathway seed derivation so parallel work is deterministic.
            local.seed = static_cast<std::int64_t>(
                static_cast<std::uint64_t>(cfg.seed) ^
                (static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ULL));

            ScoreType const oriented =
                (scoreType == ScoreType::Std && obs.es < 0)
                    ? ScoreType::Neg
                    : (scoreType == ScoreType::Std ? ScoreType::Pos : scoreType);

            EsRuler ruler(stats,
                          static_cast<std::int64_t>(pos.size()),
                          gseaParam,
                          oriented,
                          local);
            ruler.extend(std::abs(obs.es));
            auto r = ruler.pvalue(std::abs(obs.es));

            out[i] = {obs.es, r.pval, r.log2err, r.floored};
        });
    return out;
}

} // namespace fsgea::multilevel
