\name{fgseaSimple}
\alias{fgseaSimple}
\title{Simple permutation-null GSEA on CPU/GPU}
\description{
  Computes per-pathway enrichment scores and a permutation-null p-value.
  The permutation loop runs on Torch when LibTorch is built in and a GPU
  is available; otherwise it uses a C++20 \code{std::execution::par_unseq}
  CPU backend.
}
\usage{
fgseaSimple(pathways, stats, nperm = 1000L,
            minSize = 1L, maxSize = length(stats) - 1L,
            gseaParam = 1,
            scoreType = c("std", "pos", "neg"),
            seed = 42L,
            device = c("auto", "cpu", "cuda", "mps", "rocm"),
            gpuMemoryGiB = 2)
}
\arguments{
  \item{pathways}{Named list of character vectors.}
  \item{stats}{Named numeric vector of gene-level statistics.}
  \item{nperm}{Number of permutations for the null.}
  \item{minSize,maxSize}{Size filters on intersected pathways.}
  \item{gseaParam}{GSEA weighting exponent.}
  \item{scoreType}{One of \code{"std"}, \code{"pos"}, \code{"neg"}.}
  \item{seed}{Integer seed for reproducibility.}
  \item{device}{Compute device hint; falls back to CPU when unavailable.}
  \item{gpuMemoryGiB}{GPU permutation tensor memory budget.}
}
\value{
  A \code{data.table} with columns \code{pathway, pval, padj, ES, NES,
  nMoreExtreme, size, leadingEdge}. \code{attr(., "pi0")} holds the
  Storey \eqn{\pi_0} estimate used to compute \code{padj}.
}
\examples{
\dontrun{
fgseaSimple(examplePathways, exampleRanks, nperm = 1000, device = "auto")
}
}
