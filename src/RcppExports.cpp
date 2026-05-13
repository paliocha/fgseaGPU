// RcppExports.cpp — generated-style file, hand-written for clarity.
// Run `Rcpp::compileAttributes()` from R to regenerate.

#include <Rcpp.h>
using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout  = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr  = Rcpp::Rcpp_cerr_get();
#endif

// fsgea_run_cpp
List fsgea_run_cpp(NumericVector, List, CharacterVector, int, double,
                   std::string, int, int, int, std::string, double);
RcppExport SEXP _fsgeaGPU_fsgea_run_cpp(
    SEXP statsSEXP, SEXP pathwaysSEXP, SEXP namesSEXP, SEXP npermSEXP,
    SEXP gseaParamSEXP, SEXP scoreTypeSEXP, SEXP minSizeSEXP,
    SEXP maxSizeSEXP, SEXP seedSEXP, SEXP deviceSEXP, SEXP gpuMemSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fsgea_run_cpp(
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

// fsgea_calc_gsea_stat_cpp
double fsgea_calc_gsea_stat_cpp(NumericVector, IntegerVector, double, std::string);
RcppExport SEXP _fsgeaGPU_fsgea_calc_gsea_stat_cpp(
    SEXP statsSEXP, SEXP posSEXP, SEXP gseaParamSEXP, SEXP scoreTypeSEXP)
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fsgea_calc_gsea_stat_cpp(
        as<NumericVector>(statsSEXP),
        as<IntegerVector>(posSEXP),
        as<double>(gseaParamSEXP),
        as<std::string>(scoreTypeSEXP)));
    return rcpp_result_gen;
    END_RCPP
}

// fsgea_backend_info_cpp
List fsgea_backend_info_cpp();
RcppExport SEXP _fsgeaGPU_fsgea_backend_info_cpp()
{
    BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(fsgea_backend_info_cpp());
    return rcpp_result_gen;
    END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_fsgeaGPU_fsgea_run_cpp",            (DL_FUNC) &_fsgeaGPU_fsgea_run_cpp,            11},
    {"_fsgeaGPU_fsgea_calc_gsea_stat_cpp", (DL_FUNC) &_fsgeaGPU_fsgea_calc_gsea_stat_cpp,  4},
    {"_fsgeaGPU_fsgea_backend_info_cpp",   (DL_FUNC) &_fsgeaGPU_fsgea_backend_info_cpp,    0},
    {nullptr, nullptr, 0}
};

RcppExport void R_init_fsgeaGPU(DllInfo* dll) {
    R_registerRoutines(dll, nullptr, CallEntries, nullptr, nullptr);
    R_useDynamicSymbols(dll, FALSE);
}
