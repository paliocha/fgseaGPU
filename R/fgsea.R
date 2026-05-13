#' fsgeaGPU: GPU-accelerated fast gene set enrichment analysis
#'
#' Drop-in re-implementation of the upstream \pkg{fgsea} algorithm with
#' a Torch backend that runs the embarrassingly-parallel permutation null
#' on CUDA, ROCm or Apple Metal devices when available.
#'
#' @keywords internal
"_PACKAGE"

#' Run preranked gene set enrichment analysis
#'
#' @param pathways Named list of character vectors; each element is a gene set.
#' @param stats Named numeric vector of gene-level statistics. Names must
#'   match the gene identifiers in \code{pathways}.
#' @param nperm Number of permutations for the empirical null. Larger is more
#'   accurate but linearly more expensive. GPU backend handles 1e6 cheaply.
#' @param minSize,maxSize Filter gene sets by intersected size with \code{stats}.
#' @param gseaParam GSEA weighting exponent (default \code{1}; \code{0} is
#'   unweighted Kolmogorov-Smirnov).
#' @param scoreType One of \code{"std"}, \code{"pos"}, \code{"neg"}.
#' @param seed Integer seed for reproducibility.
#' @param device One of \code{"auto"}, \code{"cpu"}, \code{"cuda"}, \code{"mps"},
#'   \code{"rocm"}. Falls back to CPU if the requested device is unavailable.
#' @param gpuMemoryGiB Memory budget for the GPU permutation tensor in
#'   gibibytes. Permutations are mini-batched under this budget.
#' @return A \code{data.table} with one row per tested pathway, columns
#'   \code{pathway, pval, padj, ES, NES, nMoreExtreme, size, leadingEdge}.
#'   Column order and meaning match the upstream \code{fgsea::fgseaSimple}.
#' @export
fgsea <- function(pathways,
                  stats,
                  nperm        = 1000L,
                  minSize      = 1L,
                  maxSize      = length(stats) - 1L,
                  gseaParam    = 1,
                  scoreType    = c("std", "pos", "neg"),
                  seed         = 42L,
                  device       = c("auto", "cpu", "cuda", "mps", "rocm"),
                  gpuMemoryGiB = 2) {

    scoreType <- match.arg(scoreType)
    device    <- match.arg(device)
    stopifnot(
        is.list(pathways),
        is.numeric(stats),
        !is.null(names(stats)),
        nperm >= 1L,
        gseaParam >= 0
    )

    # Sort stats decreasing — required by the algorithm.
    ord    <- order(-stats)
    stats  <- stats[ord]

    # Map pathway gene IDs onto zero-based positions in `stats`. Use fastmatch
    # for the hot lookup. Drop NAs (genes not in stats) and sort.
    gene_index <- fastmatch::fmatch
    positions_z <- lapply(pathways, function(genes) {
        idx <- gene_index(genes, names(stats))
        idx <- idx[!is.na(idx)]
        sort.int(as.integer(idx - 1L))
    })
    names(positions_z) <- names(pathways)

    res <- .Call("_fsgeaGPU_fsgea_run_cpp",
                 as.numeric(stats),
                 positions_z,
                 as.character(names(pathways)),
                 as.integer(nperm),
                 as.numeric(gseaParam),
                 scoreType,
                 as.integer(minSize),
                 as.integer(maxSize),
                 as.integer(seed),
                 device,
                 as.numeric(gpuMemoryGiB),
                 PACKAGE = "fsgeaGPU")

    # Translate zero-based gene positions in leadingEdge into gene names.
    le_names <- lapply(res$leadingEdge, function(idx) names(stats)[idx])

    out <- data.table::data.table(
        pathway      = res$pathway,
        pval         = res$pval,
        padj         = res$padj,
        ES           = res$ES,
        NES          = res$NES,
        nMoreExtreme = res$nMoreExtreme,
        size         = res$size,
        leadingEdge  = le_names
    )
    data.table::setorder(out, pval, -ES)
    out
}

#' Calculate the enrichment score for a single gene set
#'
#' Equivalent to \code{fgsea::calcGseaStat} with the cumulative variant
#' disabled. Returns the signed ES.
#'
#' @param stats Numeric vector of gene-level statistics, sorted decreasing.
#' @param selectedStats Integer positions of the gene set in \code{stats}
#'   (1-based, as in the upstream package).
#' @param gseaParam GSEA weighting exponent.
#' @param scoreType One of \code{"std"}, \code{"pos"}, \code{"neg"}.
#' @return A single numeric: the enrichment score.
#' @export
calcGseaStat <- function(stats,
                         selectedStats,
                         gseaParam = 1,
                         scoreType = c("std", "pos", "neg")) {
    scoreType <- match.arg(scoreType)
    .Call("_fsgeaGPU_fsgea_calc_gsea_stat_cpp",
          as.numeric(stats),
          as.integer(selectedStats),
          as.numeric(gseaParam),
          scoreType,
          PACKAGE = "fsgeaGPU")
}

#' Query or set the default execution device
#'
#' @param device Optional device hint to set as the package default. If
#'   missing, returns the current default.
#' @return The resolved device string.
#' @export
fsgeaDevice <- local({
    .current <- "auto"
    function(device) {
        if (missing(device)) return(.current)
        .current <<- match.arg(device,
                               c("auto", "cpu", "cuda", "mps", "rocm"))
        invisible(.current)
    }
})

#' Report build-time backend information
#'
#' @return A list: \code{torch_built} (logical), \code{device} (resolved),
#'   \code{concurrency} (CPU thread count).
#' @export
fsgeaBackendInfo <- function() {
    .Call("_fsgeaGPU_fsgea_backend_info_cpp", PACKAGE = "fsgeaGPU")
}
