// RcppExports.cpp — generated-style file, hand-written for clarity.
// Run `Rcpp::compileAttributes()` from R to regenerate.

#include <Rcpp.h>
using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout  = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr  = Rcpp::Rcpp_cerr_get();
#endif

// fgsea_run_cpp
List fgsea_run_cpp(NumericVector, List, CharacterVector, int, double,
                   std::string, int, int, int, std::string, double);
RcppExport SEXP _fgseaGPU_fgsea_run_cpp(
    SEXP statsSEXP, SEXP pathwaysSEXP, SEXP namesSEXP, SEXP npermSEXP,
    SEXP gseaParamSEXP, SEXP scoreTypeSEXP, SEXP minSizeSEXP,
    SEXP maxSizeSEXP, SEXP seedSEXP, SEXP deviceSEXP, SEXP gpuMemSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_run_cpp(
        as<NumericVector>(statsSEXP),
        as<List>(pathwaysSEXP),
        as<CharacterVector>(namesSEXP),
        as<int>(npermSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP),
        as<int>(minSizeSEXP),
        as<int>(maxSizeSEXP),
        as<int>(seedSEXP),
        as<std::string>(deviceSEXP),
        as<double>(gpuMemSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_calc_gsea_stat_cpp
double fgsea_calc_gsea_stat_cpp(NumericVector, IntegerVector, double, std::string);
RcppExport SEXP _fgseaGPU_fgsea_calc_gsea_stat_cpp(
    SEXP statsSEXP, SEXP posSEXP, SEXP gseaParamSEXP, SEXP scoreTypeSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_calc_gsea_stat_cpp(
        as<NumericVector>(statsSEXP),
        as<IntegerVector>(posSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_backend_info_cpp
List fgsea_backend_info_cpp();
RcppExport SEXP _fgseaGPU_fgsea_backend_info_cpp()
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_backend_info_cpp());
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_multilevel_cpp
List fgsea_multilevel_cpp(NumericVector, List, CharacterVector, double, std::string,
                          int, double, double, int, int, int);
RcppExport SEXP _fgseaGPU_fgsea_multilevel_cpp(
    SEXP statsSEXP, SEXP pathwaysSEXP, SEXP namesSEXP,
    SEXP gseaParamSEXP, SEXP scoreTypeSEXP, SEXP sampleSizeSEXP,
    SEXP epsSEXP, SEXP moveScaleSEXP, SEXP seedSEXP,
    SEXP minSizeSEXP, SEXP maxSizeSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_multilevel_cpp(
        as<NumericVector>(statsSEXP),
        as<List>(pathwaysSEXP),
        as<CharacterVector>(namesSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP),
        as<int>(sampleSizeSEXP),
        as<double>(epsSEXP),
        as<double>(moveScaleSEXP),
        as<int>(seedSEXP),
        as<int>(minSizeSEXP),
        as<int>(maxSizeSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_fora_cpp
List fgsea_fora_cpp(int, int, IntegerVector, List, CharacterVector, int, int);
RcppExport SEXP _fgseaGPU_fgsea_fora_cpp(
    SEXP universeSEXP, SEXP querySizeSEXP, SEXP querySEXP,
    SEXP pathwaysSEXP, SEXP namesSEXP, SEXP minSizeSEXP, SEXP maxSizeSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_fora_cpp(
        as<int>(universeSEXP),
        as<int>(querySizeSEXP),
        as<IntegerVector>(querySEXP),
        as<List>(pathwaysSEXP),
        as<CharacterVector>(namesSEXP),
        as<int>(minSizeSEXP),
        as<int>(maxSizeSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_batch_cpp
List fgsea_batch_cpp(List, List, CharacterVector, int, double,
                     std::string, int, int, int, std::string, double);
RcppExport SEXP _fgseaGPU_fgsea_batch_cpp(
    SEXP stats_listSEXP, SEXP positions_listSEXP, SEXP pathway_namesSEXP,
    SEXP npermSEXP, SEXP gseaParamSEXP, SEXP scoreTypeSEXP,
    SEXP minSizeSEXP, SEXP maxSizeSEXP, SEXP seedSEXP,
    SEXP deviceSEXP, SEXP gpuMemSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_batch_cpp(
        as<List>(stats_listSEXP),
        as<List>(positions_listSEXP),
        as<CharacterVector>(pathway_namesSEXP),
        as<int>(npermSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP),
        as<int>(minSizeSEXP),
        as<int>(maxSizeSEXP),
        as<int>(seedSEXP),
        as<std::string>(deviceSEXP),
        as<double>(gpuMemSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fgsea_phenotype_cpp
List fgsea_phenotype_cpp(NumericVector, int, int, IntegerVector,
                         List, CharacterVector, int, std::string,
                         double, std::string, int, int, int, std::string);
RcppExport SEXP _fgseaGPU_fgsea_phenotype_cpp(
    SEXP exprsSEXP, SEXP nGenesSEXP, SEXP nSamplesSEXP, SEXP labelsSEXP,
    SEXP pathwaysSEXP, SEXP namesSEXP, SEXP npermSEXP, SEXP metricSEXP,
    SEXP gseaParamSEXP, SEXP scoreTypeSEXP, SEXP minSizeSEXP,
    SEXP maxSizeSEXP, SEXP seedSEXP, SEXP deviceSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fgsea_phenotype_cpp(
        as<NumericVector>(exprsSEXP),
        as<int>(nGenesSEXP),
        as<int>(nSamplesSEXP),
        as<IntegerVector>(labelsSEXP),
        as<List>(pathwaysSEXP),
        as<CharacterVector>(namesSEXP),
        as<int>(npermSEXP),
        as<std::string>(metricSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP),
        as<int>(minSizeSEXP),
        as<int>(maxSizeSEXP),
        as<int>(seedSEXP),
        as<std::string>(deviceSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_fgseaGPU_fgsea_run_cpp",            (DL_FUNC) &_fgseaGPU_fgsea_run_cpp,            11},
    {"_fgseaGPU_fgsea_calc_gsea_stat_cpp", (DL_FUNC) &_fgseaGPU_fgsea_calc_gsea_stat_cpp,  4},
    {"_fgseaGPU_fgsea_backend_info_cpp",   (DL_FUNC) &_fgseaGPU_fgsea_backend_info_cpp,    0},
    {"_fgseaGPU_fgsea_multilevel_cpp",     (DL_FUNC) &_fgseaGPU_fgsea_multilevel_cpp,     11},
    {"_fgseaGPU_fgsea_fora_cpp",           (DL_FUNC) &_fgseaGPU_fgsea_fora_cpp,            7},
    {"_fgseaGPU_fgsea_batch_cpp",          (DL_FUNC) &_fgseaGPU_fgsea_batch_cpp,          11},
    {"_fgseaGPU_fgsea_phenotype_cpp",      (DL_FUNC) &_fgseaGPU_fgsea_phenotype_cpp,      14},
    {nullptr, nullptr, 0}
};

RcppExport void R_init_fgseaGPU(DllInfo* dll) {
    R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
    R_useDynamicSymbols(dll, FALSE);
}
