// fsgea_core.hpp — shared types and the CPU reference implementation.
//
// The GSEA enrichment score for a pre-ranked statistic vector r of length n
// and a gene set S of size k at sorted positions p_0 < p_1 < ... < p_{k-1} is
//
//     ES = extremum over i of  S_i
//     S_i = sum_{j<=i} ( sign_pos * |r_{p_j}|^w / NR
//                       - 1/(n - k) * (p_j - p_{j-1} - 1) )
//
// with NR = sum_j |r_{p_j}|^w and w = gseaParam. Sign convention follows
// Korotkevich et al.: "std" returns the signed maximum-magnitude excursion,
// "pos" only the running maximum, "neg" only the running minimum negated.
//
// All hot kernels operate on contiguous spans so they can be lifted onto
// any execution policy or device backend without copying.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fsgea {

enum class ScoreType : std::uint8_t { Std, Pos, Neg };

constexpr ScoreType parseScoreType(std::string_view s) {
    if (s == "std") return ScoreType::Std;
    if (s == "pos") return ScoreType::Pos;
    if (s == "neg") return ScoreType::Neg;
    throw std::invalid_argument("scoreType must be one of \"std\", \"pos\", \"neg\"");
}

// Per-pathway result of a single ES evaluation.
struct EsResult {
    double es{};         // signed enrichment score
    std::int32_t leadingEdgeEnd{-1}; // index into the *sorted-by-rank* position
                                     // vector at which |running sum| was maximal
};

// Compute the (signed) enrichment score for a single gene set against a fixed
// ranked vector. `positions` must be sorted ascending and zero-based. The
// algorithm is the literal definition above and runs in O(k).
template <std::ranges::random_access_range Stats,
          std::ranges::random_access_range Positions>
[[nodiscard]] constexpr EsResult calcEs(
    Stats const& stats,
    Positions const& positions,
    double gseaParam,
    ScoreType scoreType) noexcept(false)
{
    auto const n = static_cast<std::int64_t>(std::ranges::size(stats));
    auto const k = static_cast<std::int64_t>(std::ranges::size(positions));
    if (k == 0 || k >= n) return {};

    // Numerator NR = sum |r|^w over set members.
    double nr = 0.0;
    for (auto pos : positions) {
        nr += std::pow(std::abs(static_cast<double>(stats[pos])), gseaParam);
    }
    if (nr <= 0.0) return {};

    double const negStep = 1.0 / static_cast<double>(n - k);
    double cur = 0.0;
    double best = 0.0;
    std::int32_t bestIdx = -1;
    std::int64_t prev = -1;
    std::int32_t i = 0;

    for (auto pos : positions) {
        // negative drift accumulated between previous and current set member
        cur -= negStep * static_cast<double>(pos - prev - 1);
        // member contribution
        double const w = std::pow(std::abs(static_cast<double>(stats[pos])), gseaParam) / nr;
        cur += w;

        switch (scoreType) {
            case ScoreType::Std:
                if (std::abs(cur) > std::abs(best)) { best = cur; bestIdx = i; }
                break;
            case ScoreType::Pos:
                if (cur > best) { best = cur; bestIdx = i; }
                break;
            case ScoreType::Neg:
                if (cur < best) { best = cur; bestIdx = i; }
                break;
        }
        prev = pos;
        ++i;
    }
    return {best, bestIdx};
}

// Result for an entire fgsea() call.
struct PathwayResult {
    std::string pathway;
    double pval{1.0};
    double padj{1.0};
    double es{};
    double nes{};
    std::int64_t size{};
    std::int64_t nMoreExtreme{};
    std::vector<std::int32_t> leadingEdge; // indices into `stats` (zero-based)
};

// Permutation null estimate for one pathway. `perm_es` holds the |ES| values
// observed under uniform random gene sets of size `k`. The signed observed ES
// is `obs_es`. Returns (raw two-sided p-value, NES, number-more-extreme).
struct PermSummary { double pval; double nes; std::int64_t nMoreExtreme; };

[[nodiscard]] inline PermSummary summarisePermutations(
    double obsEs,
    std::span<double const> permEs,
    ScoreType scoreType) noexcept
{
    if (permEs.empty()) return {1.0, 0.0, 0};

    std::int64_t nMore = 0;
    double meanPos = 0.0, meanNeg = 0.0;
    std::int64_t nPos = 0, nNeg = 0;

    for (double e : permEs) {
        if (e >= 0) { meanPos += e; ++nPos; }
        else        { meanNeg += e; ++nNeg; }
    }
    if (nPos > 0) meanPos /= static_cast<double>(nPos);
    if (nNeg > 0) meanNeg /= static_cast<double>(nNeg);

    double const denom = obsEs >= 0 ? meanPos : -meanNeg;
    double const nes = denom > 0.0 ? obsEs / denom : 0.0;

    if (scoreType == ScoreType::Std) {
        for (double e : permEs) {
            bool const sameSign = (e >= 0) == (obsEs >= 0);
            if (sameSign && std::abs(e) >= std::abs(obsEs)) ++nMore;
        }
        std::int64_t const denomN = obsEs >= 0 ? nPos : nNeg;
        double const p = denomN > 0
            ? static_cast<double>(nMore + 1) / static_cast<double>(denomN + 1)
            : 1.0;
        return {p, nes, nMore};
    }
    // pos / neg: one-sided
    for (double e : permEs) if (e >= obsEs) ++nMore;
    double const p = static_cast<double>(nMore + 1)
                   / static_cast<double>(static_cast<std::int64_t>(permEs.size()) + 1);
    return {p, nes, nMore};
}

// Draw k distinct integers from [0, n) into out[0..k). Floyd's algorithm —
// O(k) and stable across reproducible seeds.
template <class Rng>
inline void sampleWithoutReplacement(std::int64_t n, std::int64_t k,
                                     std::span<std::int32_t> out, Rng& rng)
{
    // Floyd's combinatorial sampling. Avoids allocating an O(n) table.
    std::vector<char> used(static_cast<std::size_t>(n), 0); // bitmap-equivalent
    std::int64_t i = 0;
    for (std::int64_t j = n - k; j < n; ++j) {
        std::uniform_int_distribution<std::int64_t> dist(0, j);
        std::int64_t t = dist(rng);
        if (used[static_cast<std::size_t>(t)]) {
            out[i++] = static_cast<std::int32_t>(j);
            used[static_cast<std::size_t>(j)] = 1;
        } else {
            out[i++] = static_cast<std::int32_t>(t);
            used[static_cast<std::size_t>(t)] = 1;
        }
    }
    std::ranges::sort(out.subspan(0, static_cast<std::size_t>(k)));
}

} // namespace fsgea
