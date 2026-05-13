// fsgea_phenotype.hpp — phenotype-permutation GSEA.
//
// Unlike the preranked pipeline (fsgea_dispatch.hpp), here the input is a
// raw expression matrix plus a two-class label vector. Each permutation
// shuffles the labels, recomputes a per-gene rank metric (signal-to-noise,
// Welch's t, or a class-mean variant), re-ranks the genes, and walks the
// enrichment score on that fresh ranking. The pipeline therefore has three
// hot stages:
//
//   1. Per-permutation rank metric across G genes × S samples (the heavy
//      kernel — what cudaGSEA spends most of its paper on).
//   2. Per-permutation argsort along the gene dimension to produce the
//      ranking used by the walk.
//   3. Per-(permutation × pathway) ES walk — same algebra as the preranked
//      kernel, just with stats varying per row instead of being shared.
//
// CPU engine (this file): `std::execution::par_unseq` outer-loop over
// permutations; each thread carries a per-gene Welford accumulator pair
// and an O(G) scratch arena. Reproducible from a single master seed via
// splitmix-derived per-permutation seeds.
//
// GPU engine (fsgea_phenotype_gpu.hpp, gated by FSGEA_WITH_TORCH): the
// metric stage becomes a `[G, S] @ [B, S]ᵀ → [G, B]` matmul (plus the same
// shape for `exprs²`), the argsort runs along dim 0, and the ES walk reuses
// the same scatter/cumsum/amax kernel as `fsgea_gpu.hpp::permEsBatchTorch`
// but with stats varying per column.

#pragma once

#include "fsgea_core.hpp"
#include "fsgea_cpu.hpp"
#include "fsgea_gpu.hpp"     // for gpu::resolveDevice/torchAvailable
#include "fsgea_qvalue.hpp"

#include <algorithm>
#include <cmath>
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

namespace fsgea::phenotype {

enum class Metric : std::uint8_t {
    SignalToNoise,      // (μ_A − μ_B) / (σ_A + σ_B), with GSEA-style sd floor
    TTest,              // Welch's two-sample t-statistic
    DiffOfClasses,      // μ_A − μ_B
    RatioOfClasses,     // μ_A / μ_B
    Log2RatioOfClasses  // log2(μ_A / μ_B)
};

[[nodiscard]] constexpr Metric parseMetric(std::string_view s) {
    if (s == "s2n")        return Metric::SignalToNoise;
    if (s == "ttest")      return Metric::TTest;
    if (s == "diff")       return Metric::DiffOfClasses;
    if (s == "ratio")      return Metric::RatioOfClasses;
    if (s == "log2_ratio") return Metric::Log2RatioOfClasses;
    throw std::invalid_argument(
        "phenotype metric must be one of "
        "\"s2n\", \"ttest\", \"diff\", \"ratio\", \"log2_ratio\"");
}

struct Input {
    // Row-major expression matrix: exprs[g * n_samples + s] = expression of
    // gene g in sample s. The R glue is responsible for the transpose from
    // R's column-major layout.
    std::vector<double>                       exprs;
    std::int64_t                              n_genes{};
    std::int64_t                              n_samples{};
    // Two-class labels in {0, 1}. Class A = 0, Class B = 1; sign convention
    // is "positive ES ⇒ enriched in A".
    std::vector<std::int8_t>                  labels;
    std::vector<std::string>                  pathway_names;
    // Gene-set membership as row indices into `exprs` (0-based, sorted).
    std::vector<std::vector<std::int32_t>>    pathway_genes;
    std::int64_t  nperm{1000};
    Metric        metric{Metric::SignalToNoise};
    double        gseaParam{1.0};
    ScoreType     scoreType{ScoreType::Std};
    std::int64_t  minSize{1};
    std::int64_t  maxSize{std::numeric_limits<std::int64_t>::max()};
    std::int64_t  seed{42};
    std::string   deviceHint{"auto"};
};

namespace detail {

// Welford online accumulator: single-pass numerically stable mean and
// variance. Sample variance uses (n−1) Bessel correction.
struct Welford {
    double mean{};
    double M2{};
    std::int64_t n{};

    constexpr void push(double x) noexcept {
        ++n;
        double const d = x - mean;
        mean += d / static_cast<double>(n);
        M2   += d * (x - mean);
    }
    [[nodiscard]] constexpr double variance() const noexcept {
        return n > 1 ? M2 / static_cast<double>(n - 1) : 0.0;
    }
    [[nodiscard]] double stddev() const noexcept {
        return std::sqrt(variance());
    }
};

[[nodiscard]] inline double computeMetric(
    Metric m, Welford const& a, Welford const& b) noexcept
{
    constexpr double EPS = 1e-12;
    switch (m) {
        case Metric::SignalToNoise: {
            // GSEA convention: floor σ at max(0.2·|μ|, ε) to avoid blow-up on
            // near-constant rows. Matches Broad GSEA's default behaviour.
            double const sd_a = std::max(a.stddev(),
                                std::max(0.2 * std::abs(a.mean), EPS));
            double const sd_b = std::max(b.stddev(),
                                std::max(0.2 * std::abs(b.mean), EPS));
            return (a.mean - b.mean) / (sd_a + sd_b);
        }
        case Metric::TTest: {
            double const v_a = std::max(a.variance(), EPS);
            double const v_b = std::max(b.variance(), EPS);
            double const se  = std::sqrt(
                v_a / std::max<std::int64_t>(a.n, 1) +
                v_b / std::max<std::int64_t>(b.n, 1));
            return (a.mean - b.mean) / std::max(se, EPS);
        }
        case Metric::DiffOfClasses:
            return a.mean - b.mean;
        case Metric::RatioOfClasses:
            return a.mean / (b.mean < 0 ? std::min(b.mean, -EPS)
                                        : std::max(b.mean,  EPS));
        case Metric::Log2RatioOfClasses:
            return std::log2(std::max(a.mean, EPS) /
                             std::max(b.mean, EPS));
    }
    return 0.0;
}

inline std::uint64_t splitmix(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Per-permutation working state. Reused via thread-local arenas would be
// nice but `par_unseq` doesn't let us hold thread-local refs cleanly, so we
// allocate per call; the cost is one O(G) vector per permutation, dwarfed
// by the S·G inner loop.
struct PermScratch {
    std::vector<std::int8_t>   labels_perm;
    std::vector<double>        stats;
    std::vector<std::int32_t>  order;
    std::vector<std::int32_t>  rank;
    std::vector<double>        stats_sorted;
    std::vector<std::int32_t>  positions;   // reused per pathway

    void resize(std::int64_t G, std::int64_t S) {
        labels_perm.resize(static_cast<std::size_t>(S));
        stats       .resize(static_cast<std::size_t>(G));
        order       .resize(static_cast<std::size_t>(G));
        rank        .resize(static_cast<std::size_t>(G));
        stats_sorted.resize(static_cast<std::size_t>(G));
    }
};

// Score every gene under one label assignment. Linear in S·G; the inner
// loop is contiguous in memory (exprs is row-major by gene).
inline void scoreAllGenes(
    Input const& in,
    std::span<std::int8_t const> labels_perm,
    std::span<double> out_stats)
{
    auto const G = in.n_genes;
    auto const S = in.n_samples;
    for (std::int64_t g = 0; g < G; ++g) {
        Welford a, b;
        auto const row = in.exprs.data() + g * S;
        for (std::int64_t s = 0; s < S; ++s) {
            (labels_perm[static_cast<std::size_t>(s)] == 0 ? a : b)
                .push(row[s]);
        }
        out_stats[static_cast<std::size_t>(g)] =
            computeMetric(in.metric, a, b);
    }
}

inline void argsortDesc(std::span<double const> stats,
                        std::span<std::int32_t> out)
{
    std::iota(out.begin(), out.end(), std::int32_t{0});
    std::ranges::sort(out, [&](std::int32_t a, std::int32_t b) {
        return stats[static_cast<std::size_t>(a)] >
               stats[static_cast<std::size_t>(b)];
    });
}

inline void invertOrder(std::span<std::int32_t const> order,
                        std::span<std::int32_t> rank)
{
    for (std::size_t i = 0; i < order.size(); ++i) {
        rank[static_cast<std::size_t>(order[i])] =
            static_cast<std::int32_t>(i);
    }
}

} // namespace detail

} // namespace fsgea::phenotype

// GPU helpers depend on the Input/Metric types defined above, so include
// here at file scope rather than inside the namespace block.
#ifdef FSGEA_WITH_TORCH
#  include "fsgea_phenotype_gpu.hpp"
#endif

namespace fsgea::phenotype {

[[nodiscard]] inline std::vector<PathwayResult> runPhenotype(Input const& in)
{
    // --- input validation -------------------------------------------------
    if (in.n_genes <= 0 || in.n_samples <= 0) {
        throw std::invalid_argument("phenotype: empty exprs matrix");
    }
    if (static_cast<std::int64_t>(in.labels.size()) != in.n_samples) {
        throw std::invalid_argument(
            "phenotype: labels length must equal n_samples");
    }
    if (in.nperm < 1) {
        throw std::invalid_argument("phenotype: nperm must be >= 1");
    }
    if (in.gseaParam < 0.0) {
        throw std::invalid_argument("phenotype: gseaParam must be >= 0");
    }
    bool has_a = false, has_b = false;
    for (auto v : in.labels) {
        if      (v == 0) has_a = true;
        else if (v == 1) has_b = true;
        else throw std::invalid_argument(
                "phenotype: labels must be 0 or 1");
    }
    if (!has_a || !has_b) {
        throw std::invalid_argument(
            "phenotype: labels must contain both classes");
    }

    // Filter pathways by size in [minSize, maxSize] and trim to genes that
    // actually exist in exprs (the R glue already maps gene IDs to row
    // indices and drops misses, so we just check bounds and size here).
    std::vector<std::size_t> kept;
    kept.reserve(in.pathway_names.size());
    for (std::size_t p = 0; p < in.pathway_names.size(); ++p) {
        auto const sz = static_cast<std::int64_t>(in.pathway_genes[p].size());
        if (sz >= in.minSize && sz <= in.maxSize &&
            sz > 0 && sz < in.n_genes) {
            kept.push_back(p);
        }
    }
    if (kept.empty()) return {};

    auto const G = in.n_genes;
    auto const S = in.n_samples;
    auto const P = static_cast<std::int64_t>(kept.size());

    // --- observed pass ---------------------------------------------------
    detail::PermScratch obs;
    obs.resize(G, S);
    detail::scoreAllGenes(in,
        std::span<std::int8_t const>(in.labels),
        std::span<double>(obs.stats));
    detail::argsortDesc(obs.stats, obs.order);
    detail::invertOrder(obs.order, obs.rank);
    for (std::int64_t i = 0; i < G; ++i) {
        obs.stats_sorted[static_cast<std::size_t>(i)] =
            obs.stats[static_cast<std::size_t>(obs.order[i])];
    }

    std::vector<double> obs_es(static_cast<std::size_t>(P));
    std::vector<std::vector<std::int32_t>> obs_le(static_cast<std::size_t>(P));
    for (std::size_t i = 0; i < kept.size(); ++i) {
        auto const& genes = in.pathway_genes[kept[i]];
        std::vector<std::int32_t> positions;
        positions.reserve(genes.size());
        for (auto g : genes)
            positions.push_back(obs.rank[static_cast<std::size_t>(g)]);
        std::ranges::sort(positions);

        auto const r = calcEs(
            std::span<double const>(obs.stats_sorted),
            std::span<std::int32_t const>(positions),
            in.gseaParam, in.scoreType);
        obs_es[i] = r.es;

        if (r.leadingEdgeEnd) {
            auto const tip = *r.leadingEdgeEnd;
            auto const begin = r.es >= 0 ? positions.begin()
                                         : positions.begin() + tip;
            auto const end   = r.es >= 0 ? positions.begin() + tip + 1
                                         : positions.end();
            obs_le[i].reserve(static_cast<std::size_t>(end - begin));
            for (auto it = begin; it != end; ++it) {
                obs_le[i].push_back(
                    obs.order[static_cast<std::size_t>(*it)]);
            }
        }
    }

    // --- permutation null ------------------------------------------------
    // Flat row-major matrix of ES values: perm_es[b * P + i].
    std::vector<double> perm_es(static_cast<std::size_t>(in.nperm * P));

    [[maybe_unused]] auto const dev =
        fsgea::gpu::resolveDevice(in.deviceHint);

#ifdef FSGEA_WITH_TORCH
    bool const use_gpu = (dev != fsgea::gpu::Device::CPU);
    if (use_gpu) {
        auto const td = fsgea::gpu::asTorchDevice(dev);
        auto const opts64 = torch::TensorOptions().dtype(torch::kFloat64);

        // Stage host tensors once.
        auto exprs_t = torch::from_blob(
                const_cast<double*>(in.exprs.data()),
                {in.n_genes, in.n_samples}, opts64).clone().to(td);

        std::vector<std::int64_t> labels_i64(
            in.labels.begin(), in.labels.end());
        auto labels_t = torch::from_blob(
                labels_i64.data(),
                {in.n_samples},
                torch::TensorOptions().dtype(torch::kInt64))
                .clone().to(td);

        std::vector<torch::Tensor> pathways_t;
        pathways_t.reserve(kept.size());
        for (auto p : kept) {
            std::vector<std::int64_t> g64(
                in.pathway_genes[p].begin(),
                in.pathway_genes[p].end());
            pathways_t.push_back(
                torch::from_blob(g64.data(),
                                 {static_cast<std::int64_t>(g64.size())},
                                 torch::TensorOptions().dtype(torch::kInt64))
                .clone().to(td));
        }

        // Chunk along the permutation dimension to keep peak VRAM under a
        // sensible budget; 2000 perms × 20k genes × ~6 tensors × 8B ≈ 2 GB,
        // well within an RTX Pro 6000 Blackwell.
        std::int64_t constexpr CHUNK = 2000;
        for (std::int64_t done = 0; done < in.nperm; done += CHUNK) {
            std::int64_t const B = std::min<std::int64_t>(CHUNK, in.nperm - done);
            std::int64_t const chunkSeed = static_cast<std::int64_t>(
                detail::splitmix(static_cast<std::uint64_t>(in.seed) ^
                                 static_cast<std::uint64_t>(done)));
            auto es = fsgea::phenotype::gpu::runBatch(
                in, exprs_t, labels_t, pathways_t, B, chunkSeed, td);
            // es is [B, P] on device; copy to host into perm_es slice.
            auto es_cpu = es.to(torch::kCPU).contiguous();
            auto a = es_cpu.accessor<double, 2>();
            for (std::int64_t b = 0; b < B; ++b) {
                for (std::int64_t i = 0; i < P; ++i) {
                    perm_es[static_cast<std::size_t>(
                        (done + b) * P + i)] = a[b][i];
                }
            }
        }
    } else
#endif
    {
    auto idx = std::ranges::iota_view<std::int64_t, std::int64_t>(0, in.nperm);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(),
        [&](std::int64_t b) {
            std::mt19937_64 rng(detail::splitmix(
                static_cast<std::uint64_t>(in.seed) ^
                static_cast<std::uint64_t>(b)));

            detail::PermScratch sc;
            sc.resize(G, S);
            std::ranges::copy(in.labels, sc.labels_perm.begin());
            std::ranges::shuffle(sc.labels_perm, rng);

            detail::scoreAllGenes(in,
                std::span<std::int8_t const>(sc.labels_perm),
                std::span<double>(sc.stats));
            detail::argsortDesc(sc.stats, sc.order);
            detail::invertOrder(sc.order, sc.rank);
            for (std::int64_t i = 0; i < G; ++i) {
                sc.stats_sorted[static_cast<std::size_t>(i)] =
                    sc.stats[static_cast<std::size_t>(sc.order[i])];
            }

            for (std::size_t i = 0; i < kept.size(); ++i) {
                auto const& genes = in.pathway_genes[kept[i]];
                sc.positions.clear();
                sc.positions.reserve(genes.size());
                for (auto g : genes)
                    sc.positions.push_back(sc.rank[static_cast<std::size_t>(g)]);
                std::ranges::sort(sc.positions);

                auto const r = calcEs(
                    std::span<double const>(sc.stats_sorted),
                    std::span<std::int32_t const>(sc.positions),
                    in.gseaParam, in.scoreType);
                perm_es[static_cast<std::size_t>(
                    b * P + static_cast<std::int64_t>(i))] = r.es;
            }
        });
    } // end CPU permutation null branch

    // --- summarise -------------------------------------------------------
    std::vector<PathwayResult> results(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
        std::vector<double> per(static_cast<std::size_t>(in.nperm));
        for (std::int64_t b = 0; b < in.nperm; ++b) {
            per[static_cast<std::size_t>(b)] =
                perm_es[static_cast<std::size_t>(
                    b * P + static_cast<std::int64_t>(i))];
        }
        auto const summary = summarisePermutations(
            obs_es[i], per, in.scoreType);

        auto& r = results[i];
        r.pathway      = in.pathway_names[kept[i]];
        r.es           = obs_es[i];
        r.nes          = summary.nes;
        r.pval         = summary.pval;
        r.size         = static_cast<std::int64_t>(in.pathway_genes[kept[i]].size());
        r.nMoreExtreme = summary.nMoreExtreme;
        r.leadingEdge  = std::move(obs_le[i]);
    }

    // Storey q-values across kept pathways.
    {
        std::vector<double> pvals;
        pvals.reserve(results.size());
        for (auto const& r : results) pvals.push_back(r.pval);
        auto const q = fsgea::qvalue::storey(pvals);
        for (std::size_t i = 0; i < results.size(); ++i) {
            results[i].padj    = q.qvalues[i];
            results[i].pi0Used = q.pi0;
        }
    }
    return results;
}

} // namespace fsgea::phenotype
