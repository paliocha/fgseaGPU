\name{fora}
\alias{fora}
\title{Over-representation analysis via the hypergeometric tail}
\description{
  For each pathway, computes \eqn{P(X \geq overlap)} where \eqn{X} is the
  size of the intersection between \code{genes} and the pathway under
  uniform sampling from \code{universe}. Reports both the strict
  upper-tail probability and the mid-p adjusted variant (Heyse 2011),
  with q-values computed from the latter to correct the conservatism of
  one-sided discrete tests.
}
\usage{
fora(pathways, genes, universe,
     minSize = 1L,
     maxSize = length(universe) - 1L)
}
\arguments{
  \item{pathways}{Named list of character vectors.}
  \item{genes}{Character vector of query genes (e.g. differentially
    expressed).}
  \item{universe}{Character vector of all background genes.}
  \item{minSize,maxSize}{Pathway size filters after intersection with
    \code{universe}.}
}
\value{
  A \code{data.table} with columns \code{pathway, pval, midP, padj,
  foldEnrichment, overlap, size, overlapGenes}.
}
\examples{
\dontrun{
fora(examplePathways,
     genes    = tail(names(exampleRanks), 200),
     universe = names(exampleRanks))
}
}
