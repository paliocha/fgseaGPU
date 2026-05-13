\name{fgsea}
\alias{fgsea}
\title{Preranked gene set enrichment analysis}
\description{
  Routes to \code{\link{fgseaMultilevel}} by default, or to
  \code{\link{fgseaSimple}} when the user supplies a positive
  \code{nperm} — matching the upstream \code{fgsea} convention.
}
\usage{
fgsea(pathways, stats, minSize = 1L,
      maxSize = length(stats) - 1L,
      gseaParam = 1, ...)
}
\arguments{
  \item{pathways}{Named list of character vectors; each element is a gene set.}
  \item{stats}{Named numeric vector of gene-level statistics.}
  \item{minSize}{Minimal gene-set size (after intersection with \code{stats}).}
  \item{maxSize}{Maximal gene-set size.}
  \item{gseaParam}{GSEA weighting exponent (default \code{1}).}
  \item{...}{Passed to the chosen implementation.}
}
\value{
  A \code{\link[data.table]{data.table}} with enrichment results; columns
  match either \code{\link{fgseaSimple}} or \code{\link{fgseaMultilevel}}.
}
\examples{
\dontrun{
data(examplePathways, package = "fgsea")
data(exampleRanks,    package = "fgsea")
res <- fgsea(examplePathways, exampleRanks)
head(res[order(pval)])
}
}
