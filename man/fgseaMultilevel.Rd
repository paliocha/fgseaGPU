\name{fgseaMultilevel}
\alias{fgseaMultilevel}
\title{Adaptive multilevel splitting Monte Carlo GSEA}
\description{
  Computes accurate small p-values via the Korotkevich et al. multilevel
  splitting algorithm. Parallelised across pathways with
  \code{std::execution::par_unseq}; each MCMC chain is sequential, so the
  GPU backend doesn't accelerate this entry point.
}
\usage{
fgseaMultilevel(pathways, stats,
                sampleSize = 101L,
                minSize = 1L, maxSize = length(stats) - 1L,
                eps = 1e-50,
                gseaParam = 1,
                scoreType = c("std", "pos", "neg"),
                seed = 42L,
                moveScale = 1.0)
}
\arguments{
  \item{pathways}{Named list of character vectors.}
  \item{stats}{Named numeric vector of gene-level statistics.}
  \item{sampleSize}{Number of chains per level. Odd, \eqn{\geq 3}.}
  \item{minSize,maxSize}{Size filters on intersected pathways.}
  \item{eps}{P-value floor; results at the floor are flagged via
    \code{floored}.}
  \item{gseaParam}{GSEA weighting exponent.}
  \item{scoreType}{One of \code{"std"}, \code{"pos"}, \code{"neg"}.}
  \item{seed}{Integer seed for reproducibility.}
  \item{moveScale}{MCMC moves per level scale factor.}
}
\value{
  A \code{data.table} with columns \code{pathway, pval, padj, log2err,
  ES, NES, size, leadingEdge, floored}.
}
\examples{
\dontrun{
fgseaMultilevel(examplePathways, exampleRanks, sampleSize = 101)
}
}
