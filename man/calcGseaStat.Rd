\name{calcGseaStat}
\alias{calcGseaStat}
\title{Enrichment score for a single gene set}
\description{
  Computes the signed enrichment score (Subramanian et al. 2005) for a
  fixed gene set against a decreasing-sorted statistic vector.
}
\usage{
calcGseaStat(stats, selectedStats, gseaParam = 1,
             scoreType = c("std", "pos", "neg"))
}
\arguments{
  \item{stats}{Numeric vector of gene-level statistics, sorted decreasing.}
  \item{selectedStats}{Integer 1-based positions of the gene set members.}
  \item{gseaParam}{GSEA weighting exponent.}
  \item{scoreType}{One of \code{"std"}, \code{"pos"}, \code{"neg"}.}
}
\value{
  A single numeric: the enrichment score.
}
\examples{
calcGseaStat(c(5, 4, 3, 2, 1, 0, -1, -2, -3, -4),
             selectedStats = c(1L, 4L, 7L))
}
