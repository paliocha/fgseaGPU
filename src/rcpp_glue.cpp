// rcpp_glue.cpp — converts R types to fsgea::FgseaInput, runs the dispatcher,
// returns a list-of-vectors that R wraps into a data.table.

#include <Rcpp.h>
#include "fsgea_dispatch.hpp"

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
List fsgea_backend_info_cpp() {
    auto const dev = fsgea::gpu::resolveDevice("auto");
    bool torch_built = fsgea::gpu::torchAvailable();
    return List::create(
        _["torch_built"]  = torch_built,
        _["device"]       = std::string(fsgea::gpu::deviceName(dev)),
        _["concurrency"]  = static_cast<int>(std::thread::hardware_concurrency()));
}
