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
// via fsgea::par. The GPU path doesn't help — this is a
// fundamentally serial-per-chain algorithm and that's the price of being
// able to probe p-values down to 1e-50.

#pragma once

#include "fsgea_core.hpp"
#include "fsgea_exec.hpp"

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

} // namespace detail

class EsRuler {
public:
    EsRuler(std::span<double const> stats,
            std::int64_t k,
            double gseaParam,
            ScoreType scoreType,
            Config cfg)
      : k_(k), gseaParam_(gseaParam), cfg_(cfg)
    {
        if (cfg_.sampleSize < 3 || (cfg_.sampleSize % 2) == 0) {
            throw std::invalid_argument(
                "multilevel sampleSize must be odd and >= 3");
        }
        // The algorithm always works against the positive-side tail. For
        // scoreType == Neg we flip the rank vector so the negative tail
        // becomes the positive tail of an equivalent problem.
        if (scoreType == ScoreType::Neg) {
            negatedStats_.assign(stats.rbegin(), stats.rend());
            for (auto& s : negatedStats_) s = -s;
            stats_ = std::span<double const>(negatedStats_);
        } else {
            stats_ = stats;
        }
    }

    // Build levels up to (and including) the one that brackets |obsEs|, or
    // until the p-value drops below cfg_.eps.
    void extend(double absObsEs) {
        auto const n = static_cast<std::int64_t>(stats_.size());
        if (k_ <= 0 || k_ >= n) return;

        std::mt19937_64 rng(static_cast<std::uint64_t>(cfg_.seed));
        auto const N = static_cast<std::size_t>(cfg_.sampleSize);

        // Per-chain membership bitmap, allocated once. The inner MCMC loop
        // tests "is this gene already in the chain?" billions of times for
        // large pathways — a bitmap is two orders of magnitude faster than
        // std::ranges::find on a small sorted vector.
        std::vector<std::vector<char>> member(N,
            std::vector<char>(static_cast<std::size_t>(n), 0));

        std::vector<std::vector<std::int32_t>> chains(N);
        std::vector<double> es(N);
        for (std::size_t i = 0; i < N; ++i) {
            chains[i].resize(static_cast<std::size_t>(k_));
            sampleWithoutReplacement(n, k_,
                std::span<std::int32_t>(chains[i]), rng);
            for (auto g : chains[i]) member[i][static_cast<std::size_t>(g)] = 1;
            es[i] = calcEs(stats_, std::span<std::int32_t const>(chains[i]),
                           gseaParam_, kInternalScore).es;
        }

        thresholds_.clear();
        // Each MCMC step does ~k moves; scale by user knob and pathway size.
        auto const mcmcSteps = std::max<std::int64_t>(
            1, static_cast<std::int64_t>(
                   std::ceil(cfg_.moveScale * static_cast<double>(k_))));

        auto snapshotFinal = [&](std::vector<std::size_t> const& order) {
            finalEs_.assign(order.size(), 0.0);
            for (std::size_t r = 0; r < order.size(); ++r)
                finalEs_[r] = es[order[r]];
        };

        // Build levels until the median of |ES| exceeds the observed value,
        // or until we've descended below the eps p-value floor.
        double const floorL = -std::log2(cfg_.eps); // max levels before floor
        std::int64_t level = 0;
        while (true) {
            std::vector<std::size_t> order(N);
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&](std::size_t a, std::size_t b) {
                return es[a] < es[b];
            });

            double const medianEs = es[order[N / 2]];
            thresholds_.push_back(medianEs);

            if (medianEs >= absObsEs) {        // bracketed observed
                snapshotFinal(order);
                return;
            }
            if (static_cast<double>(level + 1) >= floorL) {  // hit eps floor
                floored_ = true;
                snapshotFinal(order);
                return;
            }

            std::size_t const surviveStart = N / 2 + 1;     // strictly above median
            std::size_t const survivors    = N - surviveStart;
            if (survivors == 0) {              // all-tied: cannot split further
                floored_ = true;
                snapshotFinal(order);
                return;
            }

            std::uniform_int_distribution<std::size_t> pick(0, survivors - 1);
            for (std::size_t j = 0; j < surviveStart; ++j) {
                std::size_t const src = order[surviveStart + pick(rng)];
                std::size_t const dst = order[j];
                chains[dst] = chains[src];
                es[dst]     = es[src];
                std::ranges::fill(member[dst], 0);
                for (auto g : chains[dst])
                    member[dst][static_cast<std::size_t>(g)] = 1;
            }

            for (std::size_t j = 0; j < N; ++j) {
                for (std::int64_t step = 0; step < mcmcSteps; ++step) {
                    mcmcMove(chains[j], member[j], es[j], medianEs, rng);
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
    static constexpr ScoreType kInternalScore = ScoreType::Pos;

    std::span<double const>   stats_;        // points at negatedStats_ for Neg
    std::vector<double>       negatedStats_; // owns flipped stats when Neg
    std::int64_t              k_;
    double                    gseaParam_;
    Config                    cfg_;
    std::vector<double>       thresholds_;
    std::vector<double>       finalEs_;
    bool                      floored_{false};

    // Single Metropolis–Hastings move: pick a member, swap it with a
    // uniform non-member, accept if the new ES still satisfies the level
    // constraint. The membership bitmap collapses the "is this gene already
    // in the chain?" check from O(k) find to O(1) lookup.
    void mcmcMove(std::vector<std::int32_t>& chain,
                  std::vector<char>&         member,
                  double&                    chainEs,
                  double                     threshold,
                  std::mt19937_64&           rng) const
    {
        auto const n = static_cast<std::int64_t>(stats_.size());
        std::uniform_int_distribution<std::int32_t>
            slotDist(0, static_cast<std::int32_t>(k_ - 1));
        std::uniform_int_distribution<std::int32_t>
            geneDist(0, static_cast<std::int32_t>(n - 1));

        std::int32_t const slot    = slotDist(rng);
        std::int32_t const oldGene = chain[static_cast<std::size_t>(slot)];

        std::int32_t newGene;
        do {
            newGene = geneDist(rng);
        } while (member[static_cast<std::size_t>(newGene)]);

        // After replacing chain[slot] with newGene, only one element is
        // potentially misplaced. Bubble it into its sorted position — O(d)
        // moves where d is the displacement, typically far below the
        // O(k log k) cost of a full sort. Returns the final index of the
        // newly-inserted gene.
        auto bubbleInsert = [&](std::int32_t pos, std::int32_t value) {
            chain[static_cast<std::size_t>(pos)] = value;
            while (pos > 0 &&
                   chain[static_cast<std::size_t>(pos)] <
                   chain[static_cast<std::size_t>(pos - 1)]) {
                std::swap(chain[static_cast<std::size_t>(pos)],
                          chain[static_cast<std::size_t>(pos - 1)]);
                --pos;
            }
            while (pos + 1 < static_cast<std::int32_t>(k_) &&
                   chain[static_cast<std::size_t>(pos)] >
                   chain[static_cast<std::size_t>(pos + 1)]) {
                std::swap(chain[static_cast<std::size_t>(pos)],
                          chain[static_cast<std::size_t>(pos + 1)]);
                ++pos;
            }
            return pos;
        };

        member[static_cast<std::size_t>(oldGene)] = 0;
        member[static_cast<std::size_t>(newGene)] = 1;
        std::int32_t const newPos = bubbleInsert(slot, newGene);

        double const newEs = calcEs(
            stats_, std::span<std::int32_t const>(chain),
            gseaParam_, kInternalScore).es;

        if (newEs >= threshold) {
            chainEs = newEs;
            return;
        }
        // Reject: bubble the oldGene back in place of newGene.
        member[static_cast<std::size_t>(newGene)] = 0;
        member[static_cast<std::size_t>(oldGene)] = 1;
        (void) bubbleInsert(newPos, oldGene);
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

[[nodiscard]] inline std::vector<PathwayMlResult> runMultilevel(
    std::span<double const> stats,
    std::vector<std::vector<std::int32_t>> const& pathwayPositions,
    double gseaParam,
    ScoreType scoreType,
    Config cfg)
{
    std::vector<PathwayMlResult> out(pathwayPositions.size());
    std::vector<std::size_t> idx(pathwayPositions.size());
    std::iota(idx.begin(), idx.end(), 0);

    auto const orient = [scoreType](double es) {
        if (scoreType != ScoreType::Std) return scoreType;
        return es >= 0 ? ScoreType::Pos : ScoreType::Neg;
    };
    std::for_each(fsgea::par, idx.begin(), idx.end(),
        [&](std::size_t i) {
            auto const& pos = pathwayPositions[i];
            auto const obs = calcEs(stats,
                                    std::span<std::int32_t const>(pos),
                                    gseaParam, scoreType);

            Config local = cfg;
            local.seed = static_cast<std::int64_t>(fsgea::splitmix(
                static_cast<std::uint64_t>(cfg.seed) ^
                static_cast<std::uint64_t>(i)));

            EsRuler ruler(stats,
                          static_cast<std::int64_t>(pos.size()),
                          gseaParam, orient(obs.es), local);
            ruler.extend(std::abs(obs.es));
            auto const r = ruler.pvalue(std::abs(obs.es));
            out[i] = {obs.es, r.pval, r.log2err, r.floored};
        });
    return out;
}

} // namespace fsgea::multilevel
