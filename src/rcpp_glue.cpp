// rcpp_glue.cpp — converts R types to fgsea::FgseaInput, runs the dispatcher,
// returns a list-of-vectors that R wraps into a data.table.

#include <Rcpp.h>
#include "fgsea_dispatch.h"
#include "fgsea_multilevel.h"
#include "fgsea_fora.h"
#include "fgsea_phenotype.h"
#include "fgsea_qvalue.h"
#include "fgsea_exec.h"

using namespace Rcpp;

namespace {

fgsea::ScoreType parseScore(std::string const& s) {
    return fgsea::parseScoreType(s);
}

} // namespace

// [[Rcpp::export]]
List fgsea_run_cpp(
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
    fgsea::FgseaInput in;
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

    std::vector<fgsea::PathwayResult> res;
    try {
        res = fgsea::runFgsea(in);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fgsea: ") + e.what());
    }

    auto const P = static_cast<R_xlen_t>(res.size());
    CharacterVector pathway(P);
    NumericVector  pval(P), padj(P), es(P), nes(P);
    IntegerVector  size(P), nMoreExtreme(P);
    List leadingEdge(P);
    double pi0Used = res.empty() ? 1.0 : res.front().pi0Used;

    for (R_xlen_t i = 0; i < P; ++i) {
        auto const& r = res[static_cast<std::size_t>(i)];
        pathway[i]      = r.pathway;
        pval[i]         = r.pval;
        padj[i]         = r.padj;
        es[i]           = r.es;
        nes[i]          = r.nes;
        size[i]         = static_cast<int>(r.size);
        nMoreExtreme[i] = static_cast<int>(r.nMoreExtreme);

        IntegerVector v(r.leadingEdge.size());
        for (std::size_t j = 0; j < r.leadingEdge.size(); ++j)
            v[j] = r.leadingEdge[j] + 1;
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
        _["leadingEdge"]  = leadingEdge,
        _["pi0"]          = pi0Used);
}

// [[Rcpp::export]]
double fgsea_calc_gsea_stat_cpp(
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

    auto r = fgsea::calcEs(std::span<double const>(s),
                           std::span<std::int32_t const>(p),
                           gsea_param,
                           fgsea::parseScoreType(score_type));
    return r.es;
}

// [[Rcpp::export]]
List fgsea_multilevel_cpp(
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
    std::vector<double> const s(stats.begin(), stats.end());
    auto const scoreType = fgsea::parseScoreType(score_type);
    auto const n = static_cast<std::int64_t>(s.size());

    std::vector<std::vector<std::int32_t>> positions;
    std::vector<std::string>               names;
    positions.reserve(static_cast<std::size_t>(pathway_names.size()));
    names.reserve(static_cast<std::size_t>(pathway_names.size()));
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i) {
        IntegerVector p = pathways_zero_based[i];
        auto const sz = static_cast<std::int64_t>(p.size());
        if (sz < min_size || sz > max_size || sz <= 0 || sz >= n) continue;
        positions.emplace_back(p.begin(), p.end());
        names.emplace_back(Rcpp::as<std::string>(pathway_names[i]));
    }

    fgsea::multilevel::Config cfg{
        .sampleSize = sample_size,
        .eps        = eps,
        .moveScale  = move_scale,
        .seed       = seed
    };

    std::vector<fgsea::multilevel::PathwayMlResult> ml;
    try {
        ml = fgsea::multilevel::runMultilevel(
            std::span<double const>(s), positions,
            gsea_param, scoreType, cfg);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fgsea multilevel: ") + e.what());
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
            auto obs = fgsea::cpu::observedEs(
                std::span<double const>(s),
                std::span<std::int32_t const>(positions[static_cast<std::size_t>(i)]),
                gsea_param, scoreType);
            IntegerVector v(obs.leadingEdge.size());
            for (std::size_t j = 0; j < obs.leadingEdge.size(); ++j)
                v[j] = obs.leadingEdge[j] + 1;
            leadingEdge[i] = v;
        } catch (std::domain_error const& e) {
            Rcpp::stop(std::string("fgsea multilevel: ") + e.what());
        }
    }

    // Storey q-values on the multilevel p-values.
    double pi0Used = 1.0;
    {
        std::vector<double> ps;
        ps.reserve(static_cast<std::size_t>(P));
        for (R_xlen_t i = 0; i < P; ++i) ps.push_back(pval[i]);
        auto q = fgsea::qvalue::storey(ps);
        pi0Used = q.pi0;
        for (R_xlen_t i = 0; i < P; ++i)
            padj[i] = q.qvalues[static_cast<std::size_t>(i)];
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
        _["floored"]     = floored,
        _["pi0"]         = pi0Used);
}

// [[Rcpp::export]]
List fgsea_fora_cpp(
    int universe_size,
    int query_size,
    IntegerVector query_zero_based,
    List pathways_zero_based,
    CharacterVector pathway_names,
    int min_size,
    int max_size)
{
    fgsea::fora::Input in;
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

    std::vector<fgsea::fora::Result> res;
    try {
        res = fgsea::fora::run(in);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fgsea fora: ") + e.what());
    }

    auto const P = static_cast<R_xlen_t>(res.size());
    CharacterVector pathway(P);
    NumericVector pval(P), midp(P), padj(P), fold(P);
    IntegerVector overlap(P), size(P);
    List overlapGenes(P);
    double pi0Used = res.empty() ? 1.0 : res.front().pi0Used;

    for (R_xlen_t i = 0; i < P; ++i) {
        auto const& r = res[static_cast<std::size_t>(i)];
        pathway[i]        = r.pathway;
        pval[i]           = r.pval;
        midp[i]           = r.midP;
        padj[i]           = r.qvalue;
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
        _["midP"]           = midp,
        _["padj"]           = padj,
        _["foldEnrichment"] = fold,
        _["overlap"]        = overlap,
        _["size"]           = size,
        _["overlapGenes"]   = overlapGenes,
        _["pi0"]            = pi0Used);
}

// [[Rcpp::export]]
List fgsea_phenotype_cpp(
    NumericVector exprs_row_major,     // length = n_genes * n_samples
    int n_genes,
    int n_samples,
    IntegerVector labels,              // values in {0, 1}, length = n_samples
    List pathways_zero_based,
    CharacterVector pathway_names,
    int nperm,
    std::string metric,
    double gsea_param,
    std::string score_type,
    int min_size,
    int max_size,
    int seed,
    std::string device)
{
    fgsea::phenotype::Input in;
    in.n_genes   = n_genes;
    in.n_samples = n_samples;
    in.exprs.assign(exprs_row_major.begin(), exprs_row_major.end());
    in.labels.reserve(static_cast<std::size_t>(labels.size()));
    for (R_xlen_t i = 0; i < labels.size(); ++i)
        in.labels.push_back(static_cast<std::int8_t>(labels[i]));

    in.pathway_names.reserve(static_cast<std::size_t>(pathway_names.size()));
    in.pathway_genes.reserve(static_cast<std::size_t>(pathways_zero_based.size()));
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i) {
        IntegerVector v = pathways_zero_based[i];
        in.pathway_genes.emplace_back(v.begin(), v.end());
        in.pathway_names.emplace_back(Rcpp::as<std::string>(pathway_names[i]));
    }

    in.nperm      = nperm;
    in.metric     = fgsea::phenotype::parseMetric(metric);
    in.gseaParam  = gsea_param;
    in.scoreType  = fgsea::parseScoreType(score_type);
    in.minSize    = min_size;
    in.maxSize    = max_size;
    in.seed       = seed;
    in.deviceHint = device;

    std::vector<fgsea::PathwayResult> res;
    try {
        res = fgsea::phenotype::runPhenotype(in);
    } catch (std::exception const& e) {
        Rcpp::stop(std::string("fgsea phenotype: ") + e.what());
    }

    auto const P = static_cast<R_xlen_t>(res.size());
    CharacterVector pathway(P);
    NumericVector pval(P), padj(P), es(P), nes(P);
    IntegerVector size(P), nMoreExtreme(P);
    List leadingEdge(P);
    double pi0Used = res.empty() ? 1.0 : res.front().pi0Used;

    for (R_xlen_t i = 0; i < P; ++i) {
        auto const& r = res[static_cast<std::size_t>(i)];
        pathway[i]      = r.pathway;
        pval[i]         = r.pval;
        padj[i]         = r.padj;
        es[i]           = r.es;
        nes[i]          = r.nes;
        size[i]         = static_cast<int>(r.size);
        nMoreExtreme[i] = static_cast<int>(r.nMoreExtreme);
        IntegerVector v(r.leadingEdge.size());
        for (std::size_t j = 0; j < r.leadingEdge.size(); ++j)
            v[j] = r.leadingEdge[j] + 1;        // back to 1-based
        leadingEdge[i] = v;
    }
    return List::create(
        _["pathway"]      = pathway,
        _["pval"]         = pval,
        _["padj"]         = padj,
        _["ES"]           = es,
        _["NES"]          = nes,
        _["nMoreExtreme"] = nMoreExtreme,
        _["size"]         = size,
        _["leadingEdge"]  = leadingEdge,
        _["pi0"]          = pi0Used);
}

// [[Rcpp::export]]
// Batch fgsea: run one call per element of stats_list.
// Pre-sorted stats vectors and per-HOG pathway positions are prepared on the R
// side (see fgseaBatch() in R/fgsea.R).  On CPU the HOGs are dispatched through
// fgsea::for_each — parallel (TBB) where the build picked up libtbb, sequential
// on macOS where Apple libc++ ships no parallel std::execution policies.  On
// GPU they are serialised so each call can saturate the device.
List fgsea_batch_cpp(
    List stats_list,          // H named NumericVectors, each already sorted decreasing
    List positions_list,      // H Lists of IntegerVectors (0-based sorted positions)
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
    int const H = static_cast<int>(stats_list.size());

    // ---- shared pathway name vector (built once) ----
    std::vector<std::string> pnames;
    pnames.reserve(pathway_names.size());
    for (R_xlen_t i = 0; i < pathway_names.size(); ++i)
        pnames.emplace_back(Rcpp::as<std::string>(pathway_names[i]));

    // ---- pre-convert all R inputs to C++ (single-threaded, Rcpp-safe) ----
    std::vector<fgsea::FgseaInput> inputs(static_cast<std::size_t>(H));
    for (int h = 0; h < H; ++h) {
        auto& in = inputs[static_cast<std::size_t>(h)];

        NumericVector sv = stats_list[h];
        in.stats.assign(sv.begin(), sv.end());
        in.pathwayNames = pnames;          // shared copy

        List ph = positions_list[h];
        in.pathwayPositions.reserve(static_cast<std::size_t>(ph.size()));
        for (R_xlen_t p = 0; p < ph.size(); ++p) {
            IntegerVector pv = ph[p];
            in.pathwayPositions.emplace_back(pv.begin(), pv.end());
        }

        in.nperm      = static_cast<std::int64_t>(nperm);
        in.gseaParam  = gsea_param;
        in.scoreType  = parseScore(score_type);
        in.minSize    = static_cast<std::int64_t>(min_size);
        in.maxSize    = static_cast<std::int64_t>(max_size);
        in.seed       = static_cast<std::int64_t>(seed);
        in.deviceHint = device;
        in.gpuMemoryBudgetBytes =
            static_cast<std::int64_t>(gpu_memory_gib * static_cast<double>(1ULL << 30));
    }

    // ---- dispatch ----
    std::vector<std::vector<fgsea::PathwayResult>> all_results(
        static_cast<std::size_t>(H));

    bool const useGpu =
        (fgsea::gpu::resolveDevice(device) != fgsea::gpu::Device::CPU);

    if (useGpu) {
        // GPU: serial — each call saturates the device
        for (int h = 0; h < H; ++h) {
            try {
                all_results[static_cast<std::size_t>(h)] =
                    fgsea::runFgsea(inputs[static_cast<std::size_t>(h)]);
            } catch (std::exception const& e) {
                all_results[static_cast<std::size_t>(h)] = {};
                // swallow per-HOG errors; caller can detect via missing rows
            }
        }
    } else {
        // CPU: parallel over HOGs where TBB available (Linux); falls back to
        // sequential on macOS — Apple libc++ ships no parallel std::execution
        // policy tags. Pathway-level parallelism inside each call still helps.
        std::vector<int> hog_range(static_cast<std::size_t>(H));
        std::iota(hog_range.begin(), hog_range.end(), 0);
        fgsea::for_each(
            hog_range.begin(), hog_range.end(),
            [&](int h) {
                try {
                    all_results[static_cast<std::size_t>(h)] =
                        fgsea::runFgsea(inputs[static_cast<std::size_t>(h)]);
                } catch (...) {
                    all_results[static_cast<std::size_t>(h)] = {};
                }
            });
    }

    // ---- assemble flat output ----
    std::vector<int>         hog_idx_v;
    std::vector<std::string> pathway_v;
    std::vector<double>      pval_v, padj_v, es_v, nes_v;
    std::vector<int>         size_v, nmore_v;
    List                     le_list;
    double                   pi0Used = 1.0;

    for (int h = 0; h < H; ++h) {
        auto const& res_h = all_results[static_cast<std::size_t>(h)];
        if (res_h.empty()) continue;
        pi0Used = res_h.front().pi0Used;   // last non-empty HOG's pi0
        for (auto const& r : res_h) {
            hog_idx_v.push_back(h + 1);    // 1-indexed for R
            pathway_v.push_back(r.pathway);
            pval_v.push_back(r.pval);
            padj_v.push_back(r.padj);
            es_v.push_back(r.es);
            nes_v.push_back(r.nes);
            size_v.push_back(static_cast<int>(r.size));
            nmore_v.push_back(static_cast<int>(r.nMoreExtreme));
            IntegerVector le(r.leadingEdge.size());
            for (std::size_t j = 0; j < r.leadingEdge.size(); ++j)
                le[static_cast<R_xlen_t>(j)] =
                    static_cast<int>(r.leadingEdge[j]) + 1;  // 1-indexed for R
            le_list.push_back(le);
        }
    }

    return List::create(
        _["hog_idx"]      = wrap(hog_idx_v),
        _["pathway"]      = wrap(pathway_v),
        _["pval"]         = wrap(pval_v),
        _["padj"]         = wrap(padj_v),
        _["ES"]           = wrap(es_v),
        _["NES"]          = wrap(nes_v),
        _["nMoreExtreme"] = wrap(nmore_v),
        _["size"]         = wrap(size_v),
        _["leadingEdge"]  = le_list,
        _["pi0"]          = pi0Used);
}

// [[Rcpp::export]]
List fgsea_backend_info_cpp() {
    auto const dev = fgsea::gpu::resolveDevice("auto");
    bool torch_built = fgsea::gpu::torchAvailable();
    return List::create(
        _["torch_built"]  = torch_built,
        _["device"]       = std::string(fgsea::gpu::deviceName(dev)),
        _["concurrency"]  = static_cast<int>(std::thread::hardware_concurrency()));
}
