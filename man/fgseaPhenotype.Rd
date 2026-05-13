\name{fgseaPhenotype}
\alias{fgseaPhenotype}
\title{Phenotype-permutation GSEA}
\description{
  Classical Subramanian-style GSEA on a raw expression matrix with a
  two-class label vector. Each permutation shuffles the labels,
  recomputes a per-gene rank metric (signal-to-noise, Welch's t, or a
  class-mean variant), re-ranks the genes, and walks the enrichment
  score on that fresh ranking. Parallelised across permutations with
  \code{std::execution::par_unseq} on CPU, or as a batched
  \eqn{[G\times S]\cdot[B\times S]^T} matmul on the GPU when LibTorch
  is built in.
}
\usage{
fgseaPhenotype(pathways, exprs, labels,
               nperm     = 1000L,
               metric    = c("s2n", "ttest", "diff", "ratio", "log2_ratio"),
               minSize   = 1L,
               maxSize   = nrow(exprs) - 1L,
               gseaParam = 1,
               scoreType = c("std", "pos", "neg"),
               seed      = 42L,
               device    = c("auto", "cpu", "cuda", "mps", "rocm"))
}
\arguments{
  \item{pathways}{Named list of character vectors.}
  \item{exprs}{Numeric matrix; rows are genes (with rownames giving
    gene IDs), columns are samples.}
  \item{labels}{Factor, character, or integer of length
    \code{ncol(exprs)} with exactly two distinct values. The first
    level (as ordered by \code{\link{factor}}) is class A.}
  \item{nperm}{Number of phenotype permutations.}
  \item{metric}{Per-gene rank metric. \code{"s2n"} is GSEA's default
    signal-to-noise; \code{"ttest"} is Welch's two-sample statistic.}
  \item{minSize,maxSize,gseaParam,scoreType,seed,device}{See
    \code{\link{fgseaSimple}}.}
}
\value{
  A \code{data.table} with the same columns as
  \code{\link{fgseaSimple}}'s output, plus \code{attr(., "metric")}
  recording which rank metric was used and \code{attr(., "class_levels")}
  giving the factor levels of \code{labels} (in order; first = class A).
}
\examples{
\dontrun{
set.seed(1)
G <- 200; S <- 20
exprs <- matrix(rnorm(G * S), nrow = G,
                dimnames = list(paste0("g", seq_len(G)), NULL))
labels <- rep(c("ctrl", "treat"), each = S / 2)
# Upregulate the first 30 genes in the "ctrl" group.
exprs[1:30, labels == "ctrl"] <- exprs[1:30, labels == "ctrl"] + 2

pathways <- list(top30 = rownames(exprs)[1:30],
                 random = sample(rownames(exprs), 30))
fgseaPhenotype(pathways, exprs, labels, nperm = 500)
}
}
