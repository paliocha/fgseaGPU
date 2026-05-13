// fsgea_phenotype_gpu.hpp — LibTorch backend for phenotype-permutation GSEA.
//
// Three GPU passes per mini-batch of B permutations:
//
//   1. METRIC. For each (gene, perm) compute a Welch-style two-class
//      statistic. We avoid an explicit per-perm masking loop by encoding
//      labels as a 0/1 indicator matrix M ∈ {0, 1}^{B×S} (one row per
//      permutation, columns indexed by sample) and using:
//
//          sumA  = exprs · Mᵀ                    [G × B]
//          sumAsq = exprs² · Mᵀ                  [G × B]
//          sumB  = exprs.sum(1, keepdim=True) − sumA
//          sumBsq = (exprs²).sum(1, keepdim=True) − sumAsq
//
//      Means and variances follow by element-wise arithmetic. The metric
//      formula is then evaluated entirely as fused element-wise ops.
//
//   2. RANK. argsort the [G × B] stats matrix descending along dim 0 to
//      get `order` ∈ {0,..G-1}^{G×B}; compute the inverse permutation
//      `rank` via scatter; gather the descending-sorted stats matrix.
//
//   3. ES WALK. For each pathway (loop on host; pathways have wildly
//      varying sizes, so padding doesn't pay): look up per-perm positions
//      of the pathway's genes in `rank`, sort them along the k axis to get
//      `positions` ∈ {0,..G-1}^{k×B}, gather member stats from the sorted
//      matrix, build a [G × B] step tensor (-1/(G-k) everywhere, member
//      weights scattered on top), cumsum along dim 0, reduce to a signed
//      ES per permutation.
//
// Memory cost: dominant tensors are O(G · B). At G=20 k, B=10 k, FP64
// that's 1.6 GB — comfortable on an RTX Pro 6000 Blackwell (96 GB).
// For much larger B the dispatcher chunks along the batch dim under a
// configurable budget (same pattern as fsgea_dispatch.hpp).

// Included from fsgea_phenotype.hpp AFTER the Input/Metric types are
// defined. Don't include this header directly.

#pragma once

#include "fsgea_core.hpp"
#include "fsgea_gpu.hpp"   // for gpu::Device + asTorchDevice

#ifdef FSGEA_WITH_TORCH
#  include <torch/torch.h>
#endif

#include <span>
#include <vector>

namespace fsgea::phenotype::gpu {

#ifdef FSGEA_WITH_TORCH

namespace detail {

// Build a [B, S] mask of label permutations. The original 0/1 label vector
// is shuffled independently in each row via argsort-of-noise. Counts of
// class A and B are preserved per row by construction (it's a permutation
// of the same multiset of labels).
inline torch::Tensor permuteLabels(
    torch::Tensor const& labels,   // [S] int64, values in {0, 1}
    std::int64_t B,
    torch::Device device,
    std::int64_t seed)
{
    auto const S = labels.size(0);
    auto gen = at::detail::createCPUGenerator(static_cast<std::uint64_t>(seed));
    auto noise = torch::rand({B, S}, gen,
        torch::TensorOptions().dtype(torch::kFloat32)).to(device);
    auto const perm_idx = noise.argsort(/*dim=*/1);   // [B, S], int64
    // labels.unsqueeze(0).expand({B, S}) is shape-compatible; gather picks
    // the permuted labels per row.
    return labels.unsqueeze(0).expand({B, S}).gather(/*dim=*/1, perm_idx);
}

// Per-gene class statistic, vectorised across [G × B].
//
// `exprs` is [G, S] float64 on device; `mask_a` is [B, S] of 1.0/0.0 — the
// per-permutation indicator of class A. Returns a [G, B] tensor of metric
// values evaluated for every (gene, permutation) pair.
inline torch::Tensor scoreAllGenes(
    torch::Tensor const& exprs,        // [G, S]
    torch::Tensor const& mask_a,       // [B, S] (float)
    Metric metric)
{
    auto const opts = exprs.options();

    // n_A is constant across permutations (Fisher-Yates preserves counts).
    auto const nA = mask_a.sum(/*dim=*/1, /*keepdim=*/true);            // [B,1]
    auto const nB = static_cast<double>(mask_a.size(1)) - nA;           // [B,1]

    // Sums per (gene, perm). The transpose puts permutation on the
    // contracted side so the matmul produces a [G, B] result directly.
    auto const mT       = mask_a.t().contiguous();                       // [S, B]
    auto const sumA     = torch::matmul(exprs,            mT);           // [G, B]
    auto const exprs_sq = exprs * exprs;
    auto const sumAsq   = torch::matmul(exprs_sq,         mT);           // [G, B]
    auto const sumRow   = exprs   .sum(/*dim=*/1, /*keepdim=*/true);     // [G, 1]
    auto const sumRowSq = exprs_sq.sum(/*dim=*/1, /*keepdim=*/true);     // [G, 1]
    auto const sumB     = sumRow   - sumA;                               // [G, B]
    auto const sumBsq   = sumRowSq - sumAsq;                             // [G, B]

    auto const nA_g = nA.transpose(0, 1);  // [1, B] for broadcasting against [G, B]
    auto const nB_g = nB.transpose(0, 1);  // [1, B]

    auto const meanA = sumA / nA_g.clamp_min(1);
    auto const meanB = sumB / nB_g.clamp_min(1);
    auto const denomA = (nA_g - 1).clamp_min(1);
    auto const denomB = (nB_g - 1).clamp_min(1);
    auto const varA = (sumAsq - nA_g * meanA.square()) / denomA;
    auto const varB = (sumBsq - nB_g * meanB.square()) / denomB;

    constexpr double EPS = 1e-12;
    switch (metric) {
        case Metric::SignalToNoise: {
            auto const sdA = varA.clamp_min(0).sqrt()
                .clamp_min(meanA.abs() * 0.2).clamp_min(EPS);
            auto const sdB = varB.clamp_min(0).sqrt()
                .clamp_min(meanB.abs() * 0.2).clamp_min(EPS);
            return (meanA - meanB) / (sdA + sdB);
        }
        case Metric::TTest: {
            auto const se = (varA.clamp_min(EPS) / nA_g.clamp_min(1) +
                             varB.clamp_min(EPS) / nB_g.clamp_min(1))
                            .clamp_min(EPS).sqrt();
            return (meanA - meanB) / se;
        }
        case Metric::DiffOfClasses:
            return meanA - meanB;
        case Metric::RatioOfClasses:
            return meanA / torch::where(meanB.abs() < EPS,
                                        torch::full_like(meanB, EPS), meanB);
        case Metric::Log2RatioOfClasses:
            return (meanA.clamp_min(EPS) / meanB.clamp_min(EPS)).log2();
    }
    return torch::zeros_like(meanA);
}

// ES walk for one pathway across all B permutations. `stats_sorted` is
// [G, B] descending-sorted; `rank` is the inverse-permutation matrix
// (rank[g, b] = position of gene g in perm b's ranking).
inline torch::Tensor esBatchOnePathway(
    torch::Tensor const& stats_sorted,         // [G, B] float64
    torch::Tensor const& rank,                 // [G, B] int64
    torch::Tensor const& pathway_genes_int64,  // [k]   int64
    double gseaParam,
    ScoreType scoreType)
{
    auto const G = stats_sorted.size(0);
    auto const B = stats_sorted.size(1);
    auto const k = pathway_genes_int64.size(0);
    auto const opts = stats_sorted.options();

    if (k <= 0 || k >= G) return torch::zeros({B}, opts);

    // [k, B] positions, sorted ascending within each column.
    auto positions = rank.index_select(/*dim=*/0, pathway_genes_int64);
    positions = std::get<0>(positions.sort(/*dim=*/0));

    auto const member_stats = stats_sorted.gather(/*dim=*/0, positions); // [k, B]
    auto const member_w     = member_stats.abs().pow(gseaParam);          // [k, B]
    auto const NR           = member_w.sum(/*dim=*/0, /*keepdim=*/true)
                                      .clamp_min(1e-300);                 // [1, B]

    double const negStep = 1.0 / static_cast<double>(G - k);
    auto step = torch::full({G, B}, -negStep, opts);
    step.scatter_(/*dim=*/0, positions, member_w / NR);

    auto const cs = step.cumsum(/*dim=*/0);
    switch (scoreType) {
        case ScoreType::Pos: return std::get<0>(cs.max(/*dim=*/0));
        case ScoreType::Neg: return std::get<0>(cs.min(/*dim=*/0));
        case ScoreType::Std: {
            auto const hi = std::get<0>(cs.max(/*dim=*/0));
            auto const lo = std::get<0>(cs.min(/*dim=*/0));
            return torch::where(hi.abs() >= lo.abs(), hi, lo);
        }
    }
    return torch::zeros({B}, opts);
}

} // namespace detail

// Run one mini-batch of B phenotype permutations on `device` and return a
// [B, P] tensor of ES values (P = number of kept pathways).
inline torch::Tensor runBatch(
    Input const& in,
    torch::Tensor const& exprs_t,                            // [G, S]
    torch::Tensor const& labels_t,                           // [S] int64
    std::vector<torch::Tensor> const& pathways_t,            // P pathways, each [k_p] int64
    std::int64_t B,
    std::int64_t seed,
    torch::Device device)
{
    auto const G = exprs_t.size(0);

    // 1. Permute labels per row.
    auto const labels_perm = detail::permuteLabels(labels_t, B, device, seed)
                                  .to(torch::kFloat64);  // [B, S] in {0.0, 1.0}

    // 2. Per-(gene, perm) statistic.
    auto const stats = detail::scoreAllGenes(exprs_t, labels_perm, in.metric); // [G, B]

    // 3. Descending argsort along the gene dim and its inverse.
    auto const order = std::get<1>(stats.sort(/*dim=*/0, /*descending=*/true)); // [G, B]
    auto const stats_sorted = stats.gather(/*dim=*/0, order);                   // [G, B]

    auto const ar = torch::arange(G, torch::TensorOptions()
                                    .dtype(torch::kInt64).device(device))
                       .unsqueeze(1).expand({G, B});
    auto rank = torch::empty_like(order);
    rank.scatter_(/*dim=*/0, order, ar);                                        // [G, B]

    // 4. Per-pathway ES walk; results into [B, P].
    auto const P = static_cast<std::int64_t>(pathways_t.size());
    auto out = torch::empty({B, P}, stats.options());
    for (std::int64_t i = 0; i < P; ++i) {
        auto es = detail::esBatchOnePathway(
            stats_sorted, rank, pathways_t[static_cast<std::size_t>(i)],
            in.gseaParam, in.scoreType);
        out.select(/*dim=*/1, i).copy_(es);
    }
    return out;
}

#endif // FSGEA_WITH_TORCH

} // namespace fsgea::phenotype::gpu
