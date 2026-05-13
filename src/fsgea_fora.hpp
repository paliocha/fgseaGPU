// fsgea_fora.hpp — over-representation analysis via the hypergeometric tail.
//
// For each pathway we test whether the overlap with a query gene set is
// larger than expected under uniform sampling from the universe. The test
// statistic is the upper-tail hypergeometric probability:
//
//     pval = P(X >= q) ,  X ~ Hypergeometric(N, m, k)
//
// where N is the universe size, m the pathway size, k the query size, and
// q the observed overlap. We compute it in log space via lgamma to stay
// stable for large universes.
//
// Embarrassingly parallel across pathways — std::execution::par_unseq.

#pragma once

#include "fsgea_qvalue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fsgea::fora {

struct Input {
    std::int64_t universeSize{};                               // N
    std::int64_t querySize{};                                  // k
    std::vector<std::string>                  pathwayNames;    // P
    std::vector<std::vector<std::int32_t>>    pathwayMembers;  // each: positions in universe
    std::vector<std::int32_t>                 queryMembers;    // positions in universe
    std::int64_t minSize{1};
    std::int64_t maxSize{std::numeric_limits<std::int64_t>::max()};
};

struct Result {
    std::string pathway;
    double      pval{1.0};       // upper-tail hypergeometric P(X >= q)
    double      midP{1.0};       // P(X >= q) - 0.5 * P(X = q)
    double      qvalue{1.0};     // Storey q-value over mid-p (discrete adj.)
    double      pi0Used{1.0};    // pi_0 estimate used for q-values
    double      foldEnrichment{};
    std::int64_t overlap{};
    std::int64_t size{};
    std::vector<std::int32_t> overlapGenes; // zero-based positions in universe
};

namespace detail {

// log Binomial(n, k) via lgamma. Defined for 0 <= k <= n.
inline double logChoose(std::int64_t n, std::int64_t k) {
    if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
    return std::lgamma(static_cast<double>(n + 1))
         - std::lgamma(static_cast<double>(k + 1))
         - std::lgamma(static_cast<double>(n - k + 1));
}

// log P(X = q) under Hypergeometric(N, m, k).
inline double logHyperPmf(std::int64_t q, std::int64_t N,
                          std::int64_t m, std::int64_t k)
{
    return logChoose(m, q) + logChoose(N - m, k - q) - logChoose(N, k);
}

// P(X >= q) summed in log-space. Stable for q far in the tail.
inline double upperTail(std::int64_t q, std::int64_t N,
                        std::int64_t m, std::int64_t k)
{
    auto const qMin = std::max<std::int64_t>(0, k - (N - m));
    auto const qMax = std::min(m, k);
    if (q <= qMin) return 1.0;
    if (q >  qMax) return 0.0;

    // Sum exp(logPmf(i)) for i = q..qMax, log-sum-exp style with a running
    // max for numerical stability.
    std::vector<double> logs;
    logs.reserve(static_cast<std::size_t>(qMax - q + 1));
    double maxLog = -std::numeric_limits<double>::infinity();
    for (std::int64_t i = q; i <= qMax; ++i) {
        double lp = logHyperPmf(i, N, m, k);
        logs.push_back(lp);
        if (lp > maxLog) maxLog = lp;
    }
    if (!std::isfinite(maxLog)) return 0.0;

    double acc = 0.0;
    for (double lp : logs) acc += std::exp(lp - maxLog);
    return std::min(1.0, std::exp(maxLog) * acc);
}

} // namespace detail

inline std::vector<Result> run(Input const& in) {
    if (in.universeSize <= 0) {
        throw std::invalid_argument("fora: universeSize must be positive");
    }

    // Sort the query for fast intersection.
    auto queryOrder = in.queryMembers;
    std::ranges::sort(queryOrder);

    // Filter pathways by size and prepare records.
    std::vector<std::size_t> kept;
    kept.reserve(in.pathwayMembers.size());
    for (std::size_t p = 0; p < in.pathwayMembers.size(); ++p) {
        auto sz = static_cast<std::int64_t>(in.pathwayMembers[p].size());
        if (sz >= in.minSize && sz <= in.maxSize) kept.push_back(p);
    }
    std::vector<Result> out(kept.size());
    std::vector<std::size_t> outIdx(kept.size());
    std::iota(outIdx.begin(), outIdx.end(), 0);

    std::for_each(std::execution::par_unseq, outIdx.begin(), outIdx.end(),
        [&](std::size_t i) {
            std::size_t const pIdx = kept[i];
            auto const& members = in.pathwayMembers[pIdx];

            // Intersect sorted query with sorted pathway members.
            auto pathwaySorted = members;
            std::ranges::sort(pathwaySorted);
            std::vector<std::int32_t> overlap;
            overlap.reserve(std::min(pathwaySorted.size(), queryOrder.size()));
            std::ranges::set_intersection(pathwaySorted, queryOrder,
                                          std::back_inserter(overlap));

            std::int64_t const q = static_cast<std::int64_t>(overlap.size());
            std::int64_t const m = static_cast<std::int64_t>(members.size());
            std::int64_t const N = in.universeSize;
            std::int64_t const k = in.querySize;

            double const pval    = detail::upperTail(q, N, m, k);
            double const pmfAtQ  = std::isfinite(detail::logHyperPmf(q, N, m, k))
                ? std::exp(detail::logHyperPmf(q, N, m, k)) : 0.0;
            double const midP    = std::clamp(pval - 0.5 * pmfAtQ, 0.0, 1.0);

            double const expected =
                static_cast<double>(k) * static_cast<double>(m)
                / static_cast<double>(N);
            double const fold = expected > 0
                ? static_cast<double>(q) / expected
                : std::numeric_limits<double>::infinity();

            Result& r = out[i];
            r.pathway        = in.pathwayNames[pIdx];
            r.pval           = pval;
            r.midP           = midP;
            r.foldEnrichment = fold;
            r.overlap        = q;
            r.size           = m;
            r.overlapGenes   = std::move(overlap);
        });

    // Discrete-aware q-values: Storey-Tibshirani on the mid-p sequence
    // (Heyse 2011 / Lancaster mid-p correction). Falls back gracefully for
    // small m: storey() switches to min-over-lambda below 4 tests.
    std::vector<double> midPs;
    midPs.reserve(out.size());
    for (auto const& r : out) midPs.push_back(r.midP);
    auto qres = fsgea::qvalue::storey(midPs);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].qvalue  = qres.qvalues[i];
        out[i].pi0Used = qres.pi0;
    }
    return out;
}

} // namespace fsgea::fora
