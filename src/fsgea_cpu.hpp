// fsgea_cpu.hpp — C++23 parallel CPU backend.
//
// Uses std::execution::par_unseq for the outer permutation loop and operates
// on flat row-major buffers so the same kernel can be JIT'd to vectorised
// instructions on x86_64, AArch64, etc.

#pragma once

#include "fsgea_core.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace fsgea::cpu {

// Compute B random permutation enrichment scores for a single pathway size k.
// `stats` is the ranked statistic vector (length n, decreasing order).
// Returns a vector of length B of (signed) ES values.
inline std::vector<double> permEsBatch(
    std::span<double const> stats,
    std::int64_t k,
    std::int64_t B,
    double gseaParam,
    ScoreType scoreType,
    std::uint64_t seed)
{
    auto const n = static_cast<std::int64_t>(stats.size());
    std::vector<double> out(static_cast<std::size_t>(B));

    // Each permutation gets its own splitmix-derived seed so parallel
    // execution is deterministic given the master seed.
    auto splitmix = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    };

    std::vector<std::int64_t> perms(static_cast<std::size_t>(B));
    std::iota(perms.begin(), perms.end(), 0);

    std::for_each(std::execution::par_unseq, perms.begin(), perms.end(),
        [&](std::int64_t b) {
            std::mt19937_64 rng(splitmix(seed ^ static_cast<std::uint64_t>(b)));
            std::vector<std::int32_t> pos(static_cast<std::size_t>(k));
            sampleWithoutReplacement(n, k, std::span<std::int32_t>(pos), rng);
            auto r = calcEs(stats, pos, gseaParam, scoreType);
            out[static_cast<std::size_t>(b)] = r.es;
        });
    return out;
}

// Per-pathway observed ES + leading edge using literal definition.
struct ObservedEs {
    double es;
    std::vector<std::int32_t> leadingEdge; // sorted positions in stats
};

inline ObservedEs observedEs(
    std::span<double const> stats,
    std::span<std::int32_t const> positions, // sorted asc
    double gseaParam,
    ScoreType scoreType)
{
    auto r = calcEs(stats, positions, gseaParam, scoreType);
    if (r.leadingEdgeEnd < 0) {
        // Either the gene set is empty/degenerate, or every running-sum
        // extremum is exactly zero. Both are pathological: the leading edge
        // is undefined and any downstream interpretation would be misleading.
        throw std::domain_error(
            "fsgea: observed enrichment score is zero for this gene set; "
            "the leading edge is undefined. Check that the gene set has "
            "non-zero overlap with `stats` and that not all member statistics "
            "are zero.");
    }

    ObservedEs res{r.es, {}};
    // Leading-edge subset: members up to and including the extremum, taken
    // from whichever end matches the sign of ES. For positive ES, that's the
    // first (leadingEdgeEnd + 1) positions of `positions`. For negative ES,
    // it's the last (k - leadingEdgeEnd) positions.
    if (r.es >= 0) {
        res.leadingEdge.assign(
            positions.begin(),
            positions.begin() + r.leadingEdgeEnd + 1);
    } else {
        res.leadingEdge.assign(
            positions.begin() + r.leadingEdgeEnd,
            positions.end());
    }
    return res;
}

} // namespace fsgea::cpu
