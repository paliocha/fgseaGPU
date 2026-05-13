// fsgea_dispatch.hpp — orchestrates CPU vs GPU execution.
//
// The dispatcher owns the policy decisions: which device to land on, how to
// split the permutation batch under a memory budget, how to collate per-
// pathway results into the data.table that R sees. Backends only know how
// to do one batch of work.

#pragma once

#include "fsgea_core.hpp"
#include "fsgea_cpu.hpp"
#include "fsgea_gpu.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace fsgea {

struct FgseaInput {
    std::vector<double>                       stats;           // length n, decreasing
    std::vector<std::string>                  pathwayNames;    // length P
    std::vector<std::vector<std::int32_t>>    pathwayPositions; // each sorted asc, zero-based
    std::int64_t  nperm{1000};
    double        gseaParam{1.0};
    ScoreType     scoreType{ScoreType::Std};
    std::int64_t  minSize{1};
    std::int64_t  maxSize{std::numeric_limits<std::int64_t>::max()};
    std::int64_t  seed{42};
    std::string   deviceHint{"auto"};
    std::int64_t  gpuMemoryBudgetBytes{static_cast<std::int64_t>(2) << 30}; // 2 GiB default
};

namespace detail {

inline std::int64_t chunkBatchSize(std::int64_t n, std::int64_t bytesBudget) {
    // 8 bytes/elem for float64; we hold one [B, n] step tensor plus a [B, n]
    // cumsum tensor simultaneously, so ~16 bytes per (b, j) cell. Add slack.
    constexpr std::int64_t bytesPerCell = 24;
    std::int64_t maxB = bytesBudget / (bytesPerCell * std::max<std::int64_t>(n, 1));
    return std::max<std::int64_t>(maxB, 1);
}

inline std::uint64_t splitmix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace detail

[[nodiscard]] inline std::vector<PathwayResult> runFgsea(FgseaInput const& in) {
    auto const n = static_cast<std::int64_t>(in.stats.size());
    if (n < 2) throw std::invalid_argument("stats must contain at least two entries");

    // Filter pathways to size range.
    std::vector<std::size_t> keep;
    keep.reserve(in.pathwayNames.size());
    for (std::size_t p = 0; p < in.pathwayNames.size(); ++p) {
        auto sz = static_cast<std::int64_t>(in.pathwayPositions[p].size());
        if (sz >= in.minSize && sz <= in.maxSize && sz > 0 && sz < n) {
            keep.push_back(p);
        }
    }
    if (keep.empty()) return {};

    std::vector<PathwayResult> results(keep.size());

    [[maybe_unused]] auto const device = gpu::resolveDevice(in.deviceHint);
#ifdef FSGEA_WITH_TORCH
    bool const useGpu = (device != gpu::Device::CPU);
    std::optional<torch::Tensor> statsTensor;
    if (useGpu) {
        auto td = gpu::asTorchDevice(device);
        statsTensor = torch::from_blob(
            const_cast<double*>(in.stats.data()),
            {n}, torch::kFloat64).clone().to(td);
    }
#endif

    std::span<double const> statsSpan(in.stats);

    for (std::size_t i = 0; i < keep.size(); ++i) {
        auto pIdx = keep[i];
        auto const& positions = in.pathwayPositions[pIdx];
        auto k = static_cast<std::int64_t>(positions.size());

        // Observed pass (cheap, always on CPU — it's O(k))
        auto obs = cpu::observedEs(statsSpan,
                                   std::span<std::int32_t const>(positions),
                                   in.gseaParam, in.scoreType);

        // Permutation null
        std::vector<double> permEs;
        permEs.reserve(static_cast<std::size_t>(in.nperm));

#ifdef FSGEA_WITH_TORCH
        if (useGpu) {
            auto td = gpu::asTorchDevice(device);
            auto chunk = detail::chunkBatchSize(n, in.gpuMemoryBudgetBytes);
            std::int64_t done = 0;
            while (done < in.nperm) {
                auto B = std::min<std::int64_t>(chunk, in.nperm - done);
                std::int64_t chunkSeed = static_cast<std::int64_t>(
                    detail::splitmix(static_cast<std::uint64_t>(in.seed) ^
                                     (static_cast<std::uint64_t>(pIdx) << 17) ^
                                     static_cast<std::uint64_t>(done)));
                auto es = gpu::permEsBatchTorch(
                    *statsTensor, k, B, in.gseaParam, in.scoreType, chunkSeed)
                    .to(torch::kCPU).contiguous();
                auto a = es.accessor<double, 1>();
                for (std::int64_t b = 0; b < B; ++b) permEs.push_back(a[b]);
                done += B;
            }
        } else
#endif
        {
            std::uint64_t seed = detail::splitmix(
                static_cast<std::uint64_t>(in.seed) ^
                (static_cast<std::uint64_t>(pIdx) << 17));
            permEs = cpu::permEsBatch(statsSpan, k, in.nperm,
                                      in.gseaParam, in.scoreType, seed);
        }

        auto summary = summarisePermutations(obs.es, permEs, in.scoreType);

        auto& r = results[i];
        r.pathway      = in.pathwayNames[pIdx];
        r.es           = obs.es;
        r.nes          = summary.nes;
        r.pval         = summary.pval;
        r.size         = k;
        r.nMoreExtreme = summary.nMoreExtreme;
        r.leadingEdge  = std::move(obs.leadingEdge);
    }

    // BH adjusted p-values across kept pathways.
    {
        std::vector<std::pair<double, std::size_t>> pIdx;
        pIdx.reserve(results.size());
        for (std::size_t i = 0; i < results.size(); ++i)
            pIdx.emplace_back(results[i].pval, i);
        std::ranges::sort(pIdx);
        auto m = static_cast<double>(pIdx.size());
        double prev = 1.0;
        for (auto it = pIdx.rbegin(); it != pIdx.rend(); ++it) {
            std::size_t rank = static_cast<std::size_t>(std::distance(it, pIdx.rend()));
            double adj = std::min(prev, it->first * m / static_cast<double>(rank));
            results[it->second].padj = adj;
            prev = adj;
        }
    }

    return results;
}

} // namespace fsgea
