// fgsea_dispatch.h — orchestrates CPU vs GPU execution.
//
// The dispatcher owns the policy decisions: which device to land on, how to
// split the permutation batch under a memory budget, how to collate per-
// pathway results into the data.table that R sees. Backends only know how
// to do one batch of work.

#pragma once

#include "fgsea_core.h"
#include "fgsea_cpu.h"
#include "fgsea_gpu.h"
#include "fgsea_qvalue.h"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace fgsea {

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
    // The M*B output buffer used by the batched GPU path adds 8*M bytes per
    // permutation row — negligible relative to the [B, n] tensors for any
    // realistic n, so we don't reserve a slice of the budget for it here.
    constexpr std::int64_t bytesPerCell = 24;
    std::int64_t maxB = bytesBudget / (bytesPerCell * std::max<std::int64_t>(n, 1));
    return std::max<std::int64_t>(maxB, 1);
}

} // namespace detail

[[nodiscard]] inline std::vector<PathwayResult> runFgsea(FgseaInput const& in) {
    auto const n = static_cast<std::int64_t>(in.stats.size());
    if (n < 2)        throw std::invalid_argument("stats must contain at least two entries");
    if (in.nperm < 1) throw std::invalid_argument("nperm must be >= 1");
    if (in.gseaParam < 0.0) throw std::invalid_argument("gseaParam must be >= 0");

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
#ifdef FGSEA_WITH_TORCH
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

    // Observed ES (cheap, O(k) per pathway — always CPU). Compute up front so
    // we can defer per-pathway summarisation until after the batched GPU pass.
    struct ObsRecord {
        double                       es;
        std::vector<std::int32_t>    leadingEdge;
        std::int64_t                 k;
    };
    std::vector<ObsRecord> obsCache(keep.size());
    for (std::size_t i = 0; i < keep.size(); ++i) {
        auto pIdx = keep[i];
        auto const& positions = in.pathwayPositions[pIdx];
        auto obs = cpu::observedEs(statsSpan,
                                   std::span<std::int32_t const>(positions),
                                   in.gseaParam, in.scoreType);
        obsCache[i].es          = obs.es;
        obsCache[i].leadingEdge = std::move(obs.leadingEdge);
        obsCache[i].k           = static_cast<std::int64_t>(positions.size());
    }

    // Permutation null storage, one vector per kept pathway.
    std::vector<std::vector<double>> permEsAll(keep.size());
    for (auto& v : permEsAll) v.reserve(static_cast<std::size_t>(in.nperm));

    // Bucket kept pathways by their set size k. The permutation null under the
    // simple-permutation model depends only on (n, k, stats, gseaParam,
    // scoreType) — so all same-k pathways share one Monte Carlo null. We draw
    // it once per bucket and copy into each member's permEs vector. Big win
    // on real workloads: MSigDB hallmark sets have ~50 distinct sizes covering
    // 50× more pathways, so the kernel work drops by that ratio. This is also
    // what upstream fgseaSimple does.
    std::map<std::int64_t, std::vector<std::size_t>> by_k;
    for (std::size_t i = 0; i < keep.size(); ++i) {
        by_k[obsCache[i].k].push_back(i);
    }

    for (auto const& kv : by_k) {
        auto const k_u  = kv.first;
        auto const& grp = kv.second;

        std::vector<double> shared_null;
        shared_null.reserve(static_cast<std::size_t>(in.nperm));

#ifdef FGSEA_WITH_TORCH
        if (useGpu) {
            auto const chunk = detail::chunkBatchSize(n, in.gpuMemoryBudgetBytes);
            std::int64_t done = 0;
            while (done < in.nperm) {
                auto const B = std::min<std::int64_t>(chunk, in.nperm - done);
                std::int64_t chunkSeed = static_cast<std::int64_t>(
                    fgsea::splitmix(static_cast<std::uint64_t>(in.seed) ^
                                     (static_cast<std::uint64_t>(k_u) << 17) ^
                                     static_cast<std::uint64_t>(done)));
                auto es = gpu::permEsBatchTorch(
                    *statsTensor, k_u, B, in.gseaParam, in.scoreType, chunkSeed)
                    .to(torch::kCPU).contiguous();
                auto a = es.accessor<double, 1>();
                for (std::int64_t b = 0; b < B; ++b) shared_null.push_back(a[b]);
                done += B;
            }
        } else
#endif
        {
            std::uint64_t seed = fgsea::splitmix(
                static_cast<std::uint64_t>(in.seed) ^
                (static_cast<std::uint64_t>(k_u) << 17));
            shared_null = cpu::permEsBatch(statsSpan, k_u, in.nperm,
                                           in.gseaParam, in.scoreType, seed);
        }

        for (auto i : grp) permEsAll[i] = shared_null;
    }

    for (std::size_t i = 0; i < keep.size(); ++i) {
        auto pIdx   = keep[i];
        auto const& obs = obsCache[i];
        auto summary = summarisePermutations(obs.es, permEsAll[i], in.scoreType);

        auto& r = results[i];
        r.pathway      = in.pathwayNames[pIdx];
        r.es           = obs.es;
        r.nes          = summary.nes;
        r.pval         = summary.pval;
        r.size         = obs.k;
        r.nMoreExtreme = summary.nMoreExtreme;
        r.leadingEdge  = obsCache[i].leadingEdge;
    }

    // Storey-Tibshirani q-values across kept pathways.
    {
        std::vector<double> pvals;
        pvals.reserve(results.size());
        for (auto const& r : results) pvals.push_back(r.pval);
        auto q = fgsea::qvalue::storey(pvals);
        for (std::size_t i = 0; i < results.size(); ++i) {
            results[i].padj    = q.qvalues[i];
            results[i].pi0Used = q.pi0;
        }
    }

    return results;
}

} // namespace fgsea
