// fsgea_cpu.hpp — C++23 parallel CPU backend.
//
// Uses fsgea::par for the outer permutation loop and operates
// on flat row-major buffers so the same kernel can be JIT'd to vectorised
// instructions on x86_64, AArch64, etc.

#pragma once

#include "fsgea_core.hpp"
#include "fsgea_exec.hpp"

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

    // Per-permutation splitmix-derived seed so par_unseq is deterministic.
    std::vector<std::int64_t> perms(static_cast<std::size_t>(B));
    std::iota(perms.begin(), perms.end(), 0);

    fsgea::for_each(perms.begin(), perms.end(),
        [&](std::int64_t b) {
            std::mt19937_64 rng(fsgea::splitmix(
                seed ^ static_cast<std::uint64_t>(b)));
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
    std::span<std::int32_t const> positions, // sorted ascending
    double gseaParam,
    ScoreType scoreType)
{
    auto const r = calcEs(stats, positions, gseaParam, scoreType);
    if (!r.leadingEdgeEnd) {
        // Degenerate gene set or all-zero member statistics: the leading
        // edge is undefined and any downstream interpretation would be
        // misleading. Fail loud.
        throw std::domain_error(
            "fsgea: observed enrichment score is zero for this gene set; "
            "the leading edge is undefined. Check that the gene set has "
            "non-zero overlap with `stats` and that not all member statistics "
            "are zero.");
    }

    auto const tip = *r.leadingEdgeEnd;
    // For positive ES the leading edge is the prefix up to the extremum; for
    // negative ES it's the suffix starting at the extremum.
    auto const begin = r.es >= 0 ? positions.begin() : positions.begin() + tip;
    auto const end   = r.es >= 0 ? positions.begin() + tip + 1 : positions.end();
    return {r.es, std::vector<std::int32_t>(begin, end)};
}

} // namespace fsgea::cpu
