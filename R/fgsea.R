#' fgseaGPU: GPU-accelerated fast gene set enrichment analysis
#'
#' Drop-in re-implementation of the upstream \pkg{fgsea} algorithms with
#' a Torch backend that runs the embarrassingly-parallel permutation null
#' on CUDA, ROCm or Apple Metal devices when available, plus the adaptive
#' multilevel splitting algorithm for accurate small p-values, and the
#' hypergeometric over-representation test \code{fora()}.
#'
#' @keywords internal
"_PACKAGE"

# ---- helpers ----------------------------------------------------------------

.prepareStats <- function(stats) {
    stopifnot(is.numeric(stats), !is.null(names(stats)))
    ord <- order(-stats)
    stats[ord]
}

.pathwayPositions <- function(pathways, stats) {
    lapply(pathways, function(genes) {
        idx <- fastmatch::fmatch(genes, names(stats))
        idx <- idx[!is.na(idx)]
        sort.int(as.integer(idx - 1L))
    })
}

# ---- public API -------------------------------------------------------------

#' Preranked gene set enrichment analysis
#'
#' Routes to \code{\link{fgseaSimple}} when \code{nperm} is supplied
#' (matching the upstream signal that the user wants the permutation
#' implementation) and to \code{\link{fgseaMultilevel}} otherwise. The
#' returned \code{data.table} has the same column schema as upstream
#' \code{fgsea::fgsea}.
#'
#' @inheritParams fgseaSimple
#' @param ... Passed through to the chosen implementation. Use \code{nperm}
#'   to force the simple permutation path; otherwise the multilevel path
#'   is used.
#' @export
fgsea <- function(pathways, stats,
                  minSize   = 1L,
                  maxSize   = length(stats) - 1L,
                  gseaParam = 1, ...) {
    args <- list(...)
    # Route to fgseaSimple only if the user *meaningfully* set nperm
    # (numeric > 0). Passing nperm = NULL falls through to multilevel.
    use_simple <- "nperm" %in% names(args) &&
                  !is.null(args$nperm) &&
                  is.numeric(args$nperm) &&
                  args$nperm > 0
    fun <- if (use_simple) fgseaSimple else fgseaMultilevel
    fun(pathways = pathways, stats = stats,
        minSize = minSize, maxSize = maxSize,
        gseaParam = gseaParam, ...)
}

#' Simple permutation-null GSEA on CPU/GPU
#'
#' @param pathways Named list of character vectors; each element is a gene set.
#' @param stats Named numeric vector of gene-level statistics. Names must
#'   match the gene identifiers in \code{pathways}.
#' @param nperm Number of permutations for the empirical null.
#' @param minSize,maxSize Filter gene sets by intersected size with \code{stats}.
#' @param gseaParam GSEA weighting exponent.
#' @param scoreType One of \code{"std"}, \code{"pos"}, \code{"neg"}.
#' @param seed Integer seed for reproducibility.
#' @param device One of \code{"auto"}, \code{"cpu"}, \code{"cuda"}, \code{"mps"},
#'   \code{"rocm"}.
#' @param gpuMemoryGiB GPU permutation tensor memory budget.
#' @return A \code{data.table} matching \code{fgsea::fgseaSimple}'s columns.
#' @export
fgseaSimple <- function(pathways,
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
    stopifnot(is.list(pathways), nperm >= 1L, gseaParam >= 0)

    stats <- .prepareStats(stats)
    positions_z <- .pathwayPositions(pathways, stats)

    res <- .Call("_fgseaGPU_fgsea_run_cpp",
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
                 PACKAGE = "fgseaGPU")

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
    data.table::setattr(out, "pi0", res$pi0)
    data.table::setattr(out, "padj.method", "storey-tibshirani")
    data.table::setorder(out, pval, -ES)
    out
}

#' Adaptive multilevel splitting Monte Carlo GSEA
#'
#' Computes accurate small p-values via the Korotkevich et al. multilevel
#' splitting algorithm. Parallelised across pathways with
#' \code{std::execution::par_unseq}. The GPU backend doesn't speed this up
#' because each MCMC chain is inherently sequential.
#'
#' @param pathways,stats,minSize,maxSize,gseaParam,scoreType,seed See
#'   \code{\link{fgseaSimple}}.
#' @param sampleSize Number of chains per level. Must be odd and >= 3.
#'   Increase for tighter \code{log2err}.
#' @param eps P-values below this floor are returned as \code{eps} with the
#'   \code{floored} flag set. Default \code{1e-50}.
#' @param moveScale MCMC moves per level scale factor. Larger = better
#'   mixing, more work.
#' @return A \code{data.table} with columns \code{pathway, pval, padj,
#'   log2err, ES, NES, size, leadingEdge, floored}.
#' @export
fgseaMultilevel <- function(pathways,
                            stats,
                            sampleSize = 101L,
                            minSize    = 1L,
                            maxSize    = length(stats) - 1L,
                            eps        = 1e-50,
                            gseaParam  = 1,
                            scoreType  = c("std", "pos", "neg"),
                            seed       = 42L,
                            moveScale  = 1.0) {
    scoreType <- match.arg(scoreType)
    stopifnot(is.list(pathways), sampleSize >= 3, sampleSize %% 2 == 1,
              eps > 0, eps <= 1, gseaParam >= 0)

    stats <- .prepareStats(stats)
    positions_z <- .pathwayPositions(pathways, stats)

    res <- .Call("_fgseaGPU_fgsea_multilevel_cpp",
                 as.numeric(stats),
                 positions_z,
                 as.character(names(pathways)),
                 as.numeric(gseaParam),
                 scoreType,
                 as.integer(sampleSize),
                 as.numeric(eps),
                 as.numeric(moveScale),
                 as.integer(seed),
                 as.integer(minSize),
                 as.integer(maxSize),
                 PACKAGE = "fgseaGPU")

    le_names <- lapply(res$leadingEdge, function(idx) names(stats)[idx])
    out <- data.table::data.table(
        pathway     = res$pathway,
        pval        = res$pval,
        padj        = res$padj,
        log2err     = res$log2err,
        ES          = res$ES,
        NES         = res$NES,
        size        = res$size,
        leadingEdge = le_names,
        floored     = res$floored
    )
    data.table::setattr(out, "pi0", res$pi0)
    data.table::setattr(out, "padj.method", "storey-tibshirani")
    data.table::setorder(out, pval, -ES)
    out
}

#' Over-representation analysis via the hypergeometric tail
#'
#' For each pathway, computes \code{P(X >= overlap)} where \code{X} is the
#' size of the intersection between \code{genes} and the pathway under
#' uniform sampling from \code{universe}.
#'
#' @param pathways Named list of character vectors.
#' @param genes Character vector of query genes (e.g. differentially
#'   expressed).
#' @param universe Character vector of all background genes.
#' @param minSize,maxSize Size filters on pathways (post-intersection
#'   with \code{universe}).
#' @return A \code{data.table} with columns \code{pathway, pval, padj,
#'   foldEnrichment, overlap, size, overlapGenes}.
#' @export
fora <- function(pathways, genes, universe,
                 minSize = 1L,
                 maxSize = length(universe) - 1L) {
    stopifnot(is.list(pathways), is.character(genes), is.character(universe))

    if (!all(genes %in% universe)) {
        warning("Not all query genes belong to the universe; extras removed.")
    }

    universe <- unique(universe)
    queryIdx <- unique(stats::na.omit(fastmatch::fmatch(genes, universe)))
    pathwayPositions <- lapply(pathways, function(pw) {
        idx <- fastmatch::fmatch(pw, universe)
        sort.int(as.integer(stats::na.omit(idx) - 1L))
    })

    res <- .Call("_fgseaGPU_fgsea_fora_cpp",
                 as.integer(length(universe)),
                 as.integer(length(queryIdx)),
                 as.integer(queryIdx - 1L),
                 pathwayPositions,
                 as.character(names(pathways)),
                 as.integer(minSize),
                 as.integer(maxSize),
                 PACKAGE = "fgseaGPU")

    overlapGeneNames <- lapply(res$overlapGenes, function(idx) universe[idx])
    out <- data.table::data.table(
        pathway        = res$pathway,
        pval           = res$pval,
        midP           = res$midP,
        padj           = res$padj,
        foldEnrichment = res$foldEnrichment,
        overlap        = res$overlap,
        size           = res$size,
        overlapGenes   = overlapGeneNames
    )
    data.table::setattr(out, "pi0", res$pi0)
    data.table::setattr(out, "padj.method", "storey-tibshirani-on-midp")
    data.table::setorder(out, pval)
    out
}

#' Phenotype-permutation GSEA
#'
#' Classical Subramanian-style GSEA: input is a raw expression matrix plus a
#' two-class label vector. Each permutation shuffles the labels, recomputes a
#' per-gene rank metric (signal-to-noise, Welch's t, or class-mean variants),
#' re-ranks the genes, and walks the enrichment score on that fresh ranking.
#' This is the mode \pkg{fgsea} does not implement; we add it here for
#' compatibility with the original Broad GSEA workflow.
#'
#' On the GPU backend (\code{device != "cpu"} and the package was built with
#' LibTorch), the per-gene metric is computed as a batched matrix multiply
#' across permutations on the device, the ranking via argsort along the gene
#' dimension, and the ES walk via the same scatter-cumsum-reduce kernel as
#' \code{\link{fgseaSimple}} but with stats varying per column.
#'
#' @param pathways Named list of character vectors.
#' @param exprs Numeric matrix; rows are genes (with rownames giving gene
#'   IDs), columns are samples.
#' @param labels Factor, character, or integer vector of length
#'   \code{ncol(exprs)}, with exactly two distinct values. The first level
#'   (as ordered by \code{\link{factor}}) is class A; positive ES means
#'   enrichment in class A.
#' @param nperm Number of phenotype permutations.
#' @param metric Rank metric: \code{"s2n"} (signal-to-noise, default),
#'   \code{"ttest"} (Welch's), \code{"diff"}, \code{"ratio"},
#'   \code{"log2_ratio"}.
#' @param minSize,maxSize,gseaParam,scoreType,seed,device See
#'   \code{\link{fgseaSimple}}.
#' @return A \code{data.table} mirroring \code{\link{fgseaSimple}}'s columns.
#' @export
fgseaPhenotype <- function(pathways,
                           exprs,
                           labels,
                           nperm     = 1000L,
                           metric    = c("s2n", "ttest", "diff",
                                         "ratio", "log2_ratio"),
                           minSize   = 1L,
                           maxSize   = nrow(exprs) - 1L,
                           gseaParam = 1,
                           scoreType = c("std", "pos", "neg"),
                           seed      = 42L,
                           device    = c("auto", "cpu", "cuda", "mps", "rocm")) {
    metric    <- match.arg(metric)
    scoreType <- match.arg(scoreType)
    device    <- match.arg(device)

    stopifnot(
        is.list(pathways),
        is.matrix(exprs), is.numeric(exprs), !is.null(rownames(exprs)),
        length(labels) == ncol(exprs),
        nperm >= 1L, gseaParam >= 0
    )

    # Two-class encoding: first level → 0 (class A), second → 1 (class B).
    lab_fac <- if (is.factor(labels)) labels else factor(labels)
    if (nlevels(lab_fac) != 2L) {
        stop("`labels` must contain exactly two distinct values; got ",
             nlevels(lab_fac))
    }
    labels_i <- as.integer(lab_fac) - 1L          # 0/1

    gene_ids <- rownames(exprs)
    positions_z <- lapply(pathways, function(genes) {
        idx <- fastmatch::fmatch(genes, gene_ids)
        idx <- idx[!is.na(idx)]
        sort.int(as.integer(idx - 1L))
    })

    # Row-major flattening: t(exprs) is [samples, genes] column-major, which
    # is exactly the byte order we want for exprs[g, s] = flat[g * S + s].
    exprs_rm <- as.numeric(t(exprs))

    res <- .Call("_fgseaGPU_fgsea_phenotype_cpp",
                 exprs_rm,
                 as.integer(nrow(exprs)),
                 as.integer(ncol(exprs)),
                 labels_i,
                 positions_z,
                 as.character(names(pathways)),
                 as.integer(nperm),
                 metric,
                 as.numeric(gseaParam),
                 scoreType,
                 as.integer(minSize),
                 as.integer(maxSize),
                 as.integer(seed),
                 device,
                 PACKAGE = "fgseaGPU")

    # Leading-edge indices are gene row positions (1-based); map back to IDs.
    le_names <- lapply(res$leadingEdge, function(idx) gene_ids[idx])
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
    data.table::setattr(out, "pi0", res$pi0)
    data.table::setattr(out, "padj.method", "storey-tibshirani")
    data.table::setattr(out, "metric", metric)
    data.table::setattr(out, "class_levels", levels(lab_fac))
    data.table::setorder(out, pval, -ES)
    out
}

#' Calculate the enrichment score for a single gene set
#'
#' @param stats Numeric vector of gene-level statistics, sorted decreasing.
#' @param selectedStats Integer positions of the gene set in \code{stats}
#'   (1-based).
#' @param gseaParam GSEA weighting exponent.
#' @param scoreType One of \code{"std"}, \code{"pos"}, \code{"neg"}.
#' @return Numeric: the enrichment score.
#' @export
calcGseaStat <- function(stats,
                         selectedStats,
                         gseaParam = 1,
                         scoreType = c("std", "pos", "neg")) {
    scoreType <- match.arg(scoreType)
    .Call("_fgseaGPU_fgsea_calc_gsea_stat_cpp",
          as.numeric(stats),
          as.integer(selectedStats),
          as.numeric(gseaParam),
          scoreType,
          PACKAGE = "fgseaGPU")
}

#' Query or set the default execution device
#'
#' @param device Optional device hint. Returns the current default when
#'   missing.
#' @return The resolved device string.
#' @export
fgseaDevice <- local({
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
fgseaBackendInfo <- function() {
    .Call("_fgseaGPU_fgsea_backend_info_cpp", PACKAGE = "fgseaGPU")
}

#' Batch preranked GSEA over multiple stats vectors
#'
#' Runs \code{\link{fgseaSimple}} for each element of \code{stats_list} in a
#' single C++ call.  Pathway positions are precomputed on the R side, then
#' dispatched via a C++ loop — no R overhead between HOGs.  On CPU the HOGs
#' are processed in parallel via \code{std::execution::par_unseq} (TBB); on
#' GPU they are serialised so each call fully saturates the device.
#'
#' @param stats_list Named list of numeric vectors.  Each element is a
#'   pre-ranked (or raw) stats vector for one query (e.g., one focal HOG).
#'   Names of each vector must be gene identifiers matching \code{pathways}.
#' @param pathways Named list of character vectors (gene sets).
#' @param nperm,minSize,maxSize,gseaParam,scoreType,seed,device,gpuMemoryGiB
#'   See \code{\link{fgseaSimple}}.
#' @return A \code{data.table} like \code{\link{fgseaSimple}} but with an
#'   extra integer column \code{hog_idx} — the 1-based index of the
#'   corresponding element in \code{stats_list}.
#' @export
fgseaBatch <- function(stats_list,
                       pathways,
                       nperm        = 1000L,
                       minSize      = 1L,
                       maxSize      = .Machine$integer.max,
                       gseaParam    = 1,
                       scoreType    = c("std", "pos", "neg"),
                       seed         = 42L,
                       device       = c("auto", "cpu", "cuda", "mps", "rocm"),
                       gpuMemoryGiB = 2) {
    scoreType <- match.arg(scoreType)
    device    <- match.arg(device)
    stopifnot(is.list(stats_list), length(stats_list) >= 1L, is.list(pathways))

    # Precompute sorted stats + per-HOG pathway positions on R side
    sorted_list    <- lapply(stats_list, .prepareStats)
    positions_list <- lapply(sorted_list, function(s) .pathwayPositions(pathways, s))

    res <- .Call("_fgseaGPU_fgsea_batch_cpp",
                 sorted_list,
                 positions_list,
                 as.character(names(pathways)),
                 as.integer(nperm),
                 as.numeric(gseaParam),
                 scoreType,
                 as.integer(minSize),
                 as.integer(maxSize),
                 as.integer(seed),
                 device,
                 as.numeric(gpuMemoryGiB),
                 PACKAGE = "fgseaGPU")

    if (length(res$pathway) == 0L) {
        return(data.table::data.table(
            hog_idx = integer(), pathway = character(),
            pval = numeric(), padj = numeric(),
            ES = numeric(), NES = numeric(),
            nMoreExtreme = integer(), size = integer(),
            leadingEdge = list()))
    }

    # Convert 1-based leading-edge indices back to gene names
    le_names <- Map(function(idx, h) names(sorted_list[[h]])[idx],
                    res$leadingEdge, res$hog_idx)

    out <- data.table::data.table(
        hog_idx      = res$hog_idx,
        pathway      = res$pathway,
        pval         = res$pval,
        padj         = res$padj,
        ES           = res$ES,
        NES          = res$NES,
        nMoreExtreme = res$nMoreExtreme,
        size         = res$size,
        leadingEdge  = le_names
    )
    data.table::setattr(out, "pi0", res$pi0)
    data.table::setattr(out, "padj.method", "storey-tibshirani")
    data.table::setorder(out, hog_idx, pval, -ES)
    out
}
