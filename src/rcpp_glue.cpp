// rcpp_glue.cpp — converts R types to fsgea::FgseaInput, runs the dispatcher,
// returns a list-of-vectors that R wraps into a data.table.

#include <Rcpp.h>
#include "fsgea_dispatch.hpp"
#include "fsgea_multilevel.hpp"
#include "fsgea_fora.hpp"

using namespace Rcpp;

namespace {

fsgea::ScoreType parseScore(std::string const& s) {
    return fsgea::parseScoreType(s);
}

} // namespace

// [[Rcpp::export]]
List fsgea_run_cpp(
    NumericVector stats,                  // already sorted decreasing on R side
    List pathways_zero_based,             // each an IntegerVector of zero-based sorted positions
    CharacterVector pathway_names,
    int nperm,
    double gsea_param,
    std::string score_type,
    int min_size,
    int max_size,
    int seed,
    std::string device,
    double gpu_memory_gib)
{
    fsgea::FgseaInput in;
    in.stats.assign(stats.begin(), stats.end());

    in.pathwayNames.reserve(pathway_names.size());
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i)
        in.pathwayNames.emplace_back(Rcpp::as<std::string>(pathway_names[i]));

    in.pathwayPositions.reserve(pathways_zero_based.size());
    for (R_xlen_t i = 0; i < pathways_zero_based.size(); ++i) {
        IntegerVector p = pathways_zero_based[i];
        std::vector<std::int32_t> v(p.begin(), p.end());
        // Defensive: R side promises sorted ascending; assert in debug builds.
        in.pathwayPositions.emplace_back(std::move(v));
    }

    in.nperm      = nperm;
    in.gseaParam  = gsea_param;
    in.scoreType  = parseScore(score_type);
    in.minSize    = min_size;
    in.maxSize    = max_size;
    in.seed       = seed;
    in.deviceHint = device;
    in.gpuMemoryBudgetBytes = static_cast<std::int64_t>(gpu_memory_gib * (1ULL << 30));

    std::vector<fsgea::PathwayResult> res;
    try {
        res = fsgea::runFgsea(in);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fsgea: ") + e.what());
    }

    auto const P = static_cast<R_xlen_t>(res.size());
    CharacterVector pathway(P);
    NumericVector  pval(P), padj(P), es(P), nes(P);
    IntegerVector  size(P), nMoreExtreme(P);
    List leadingEdge(P);

    for (R_xlen_t i = 0; i < P; ++i) {
        pathway[i]      = res[static_cast<std::size_t>(i)].pathway;
        pval[i]         = res[static_cast<std::size_t>(i)].pval;
        padj[i]         = res[static_cast<std::size_t>(i)].padj;
        es[i]           = res[static_cast<std::size_t>(i)].es;
        nes[i]          = res[static_cast<std::size_t>(i)].nes;
        size[i]         = static_cast<int>(res[static_cast<std::size_t>(i)].size);
        nMoreExtreme[i] = static_cast<int>(res[static_cast<std::size_t>(i)].nMoreExtreme);

        auto const& le = res[static_cast<std::size_t>(i)].leadingEdge;
        IntegerVector v(le.size());
        // Return 1-based indices for R consumers
        for (std::size_t j = 0; j < le.size(); ++j) v[j] = le[j] + 1;
        leadingEdge[i]  = v;
    }

    return List::create(
        _["pathway"]      = pathway,
        _["pval"]         = pval,
        _["padj"]         = padj,
        _["ES"]           = es,
        _["NES"]          = nes,
        _["nMoreExtreme"] = nMoreExtreme,
        _["size"]         = size,
        _["leadingEdge"]  = leadingEdge);
}

// [[Rcpp::export]]
double fsgea_calc_gsea_stat_cpp(
    NumericVector stats,
    IntegerVector positions_one_based,
    double gsea_param,
    std::string score_type)
{
    std::vector<double> s(stats.begin(), stats.end());
    std::vector<std::int32_t> p;
    p.reserve(positions_one_based.size());
    for (R_xlen_t i = 0; i < positions_one_based.size(); ++i)
        p.push_back(positions_one_based[i] - 1); // to zero-based
    std::ranges::sort(p);

    auto r = fsgea::calcEs(std::span<double const>(s),
                           std::span<std::int32_t const>(p),
                           gsea_param,
                           fsgea::parseScoreType(score_type));
    return r.es;
}

// [[Rcpp::export]]
List fsgea_multilevel_cpp(
    NumericVector stats,
    List pathways_zero_based,
    CharacterVector pathway_names,
    double gsea_param,
    std::string score_type,
    int sample_size,
    double eps,
    double move_scale,
    int seed,
    int min_size,
    int max_size)
{
    std::vector<double> s(stats.begin(), stats.end());
    auto scoreType = fsgea::parseScoreType(score_type);

    // Filter pathways by size.
    auto const n = static_cast<std::int64_t>(s.size());
    std::vector<std::size_t> kept;
    std::vector<std::vector<std::int32_t>> positions;
    std::vector<std::string>               names;
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i) {
        IntegerVector p = pathways_zero_based[i];
        auto sz = static_cast<std::int64_t>(p.size());
        if (sz < min_size || sz > max_size || sz <= 0 || sz >= n) continue;
        positions.emplace_back(p.begin(), p.end());
        names.emplace_back(Rcpp::as<std::string>(pathway_names[i]));
        kept.push_back(static_cast<std::size_t>(i));
    }

    fsgea::multilevel::Config cfg{
        .sampleSize = sample_size,
        .eps        = eps,
        .moveScale  = move_scale,
        .seed       = seed
    };

    std::vector<fsgea::multilevel::PathwayMlResult> ml;
    try {
        ml = fsgea::multilevel::runMultilevel(
            std::span<double const>(s), positions,
            gsea_param, scoreType, cfg);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fsgea multilevel: ") + e.what());
    }

    // Build leading edges on the side, with the zero-ES error surfaced.
    auto const P = static_cast<R_xlen_t>(ml.size());
    CharacterVector pathway(P);
    NumericVector pval(P), padj(P), log2err(P), es(P), nes(P);
    IntegerVector size(P);
    LogicalVector floored(P);
    List leadingEdge(P);

    // Observed-EsRuler-derived mean ES under null is unavailable cheaply here,
    // so NES is computed as ES / mean(|ES| of final-level sample) on the
    // R side if a normalisation is required. We expose ES; NES defaults to NaN.
    for (R_xlen_t i = 0; i < P; ++i) {
        auto const& m = ml[static_cast<std::size_t>(i)];
        pathway[i]  = names[static_cast<std::size_t>(i)];
        pval[i]     = m.pval;
        log2err[i]  = m.log2err;
        es[i]       = m.es;
        nes[i]      = std::numeric_limits<double>::quiet_NaN();
        size[i]     = static_cast<int>(positions[static_cast<std::size_t>(i)].size());
        floored[i]  = m.floored;

        // Leading edge via the same path as the simple branch.
        try {
            auto obs = fsgea::cpu::observedEs(
                std::span<double const>(s),
                std::span<std::int32_t const>(positions[static_cast<std::size_t>(i)]),
                gsea_param, scoreType);
            IntegerVector v(obs.leadingEdge.size());
            for (std::size_t j = 0; j < obs.leadingEdge.size(); ++j)
                v[j] = obs.leadingEdge[j] + 1;
            leadingEdge[i] = v;
        } catch (std::domain_error const& e) {
            Rcpp::stop(std::string("fsgea multilevel: ") + e.what());
        }
    }

    // BH on pval.
    {
        std::vector<std::pair<double, R_xlen_t>> ord;
        ord.reserve(P);
        for (R_xlen_t i = 0; i < P; ++i) ord.emplace_back(pval[i], i);
        std::ranges::sort(ord);
        double const Md = static_cast<double>(P);
        double prev = 1.0;
        for (auto it = ord.rbegin(); it != ord.rend(); ++it) {
            auto rank = static_cast<double>(std::distance(it, ord.rend()));
            double adj = std::min(prev, it->first * Md / rank);
            padj[it->second] = adj;
            prev = adj;
        }
    }

    return List::create(
        _["pathway"]     = pathway,
        _["pval"]        = pval,
        _["padj"]        = padj,
        _["log2err"]     = log2err,
        _["ES"]          = es,
        _["NES"]         = nes,
        _["size"]        = size,
        _["leadingEdge"] = leadingEdge,
        _["floored"]     = floored);
}

// [[Rcpp::export]]
List fsgea_fora_cpp(
    int universe_size,
    int query_size,
    IntegerVector query_zero_based,
    List pathways_zero_based,
    CharacterVector pathway_names,
    int min_size,
    int max_size)
{
    fsgea::fora::Input in;
    in.universeSize = universe_size;
    in.querySize    = query_size;
    in.queryMembers.assign(query_zero_based.begin(), query_zero_based.end());
    in.pathwayNames.reserve(pathway_names.size());
    in.pathwayMembers.reserve(pathways_zero_based.size());
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i) {
        IntegerVector v = pathways_zero_based[i];
        in.pathwayMembers.emplace_back(v.begin(), v.end());
        in.pathwayNames.emplace_back(Rcpp::as<std::string>(pathway_names[i]));
    }
    in.minSize = min_size;
    in.maxSize = max_size;

    std::vector<fsgea::fora::Result> res;
    try {
        res = fsgea::fora::run(in);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fsgea fora: ") + e.what());
    }

    auto const P = static_cast<R_xlen_t>(res.size());
    CharacterVector pathway(P);
    NumericVector pval(P), padj(P), fold(P);
    IntegerVector overlap(P), size(P);
    List overlapGenes(P);

    for (R_xlen_t i = 0; i < P; ++i) {
        auto const& r = res[static_cast<std::size_t>(i)];
        pathway[i]        = r.pathway;
        pval[i]           = r.pval;
        padj[i]           = r.padj;
        fold[i]           = r.foldEnrichment;
        overlap[i]        = static_cast<int>(r.overlap);
        size[i]           = static_cast<int>(r.size);
        IntegerVector og(r.overlapGenes.size());
        for (std::size_t j = 0; j < r.overlapGenes.size(); ++j)
            og[j] = r.overlapGenes[j] + 1; // to 1-based
        overlapGenes[i]   = og;
    }
    return List::create(
        _["pathway"]        = pathway,
        _["pval"]           = pval,
        _["padj"]           = padj,
        _["foldEnrichment"] = fold,
        _["overlap"]        = overlap,
        _["size"]           = size,
        _["overlapGenes"]   = overlapGenes);
}

// [[Rcpp::export]]
List fsgea_backend_info_cpp() {
    auto const dev = fsgea::gpu::resolveDevice("auto");
    bool torch_built = fsgea::gpu::torchAvailable();
    return List::create(
        _["torch_built"]  = torch_built,
        _["device"]       = std::string(fsgea::gpu::deviceName(dev)),
        _["concurrency"]  = static_cast<int>(std::thread::hardware_concurrency()));
}
