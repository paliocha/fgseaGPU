\name{fsgeaGPU-package}
\alias{fsgeaGPU-package}
\alias{fsgeaGPU}
\docType{package}
\title{GPU-Accelerated Fast Gene Set Enrichment Analysis}
\description{
  Drop-in re-implementation of the upstream \pkg{fgsea} algorithms with a
  Torch backend that runs the embarrassingly-parallel permutation null on
  CUDA, ROCm or Apple Metal devices when available, the adaptive multilevel
  splitting algorithm for accurate small p-values, and the hypergeometric
  over-representation test \code{\link{fora}}.
}
\section{Multiple-testing correction}{
  Every output's \code{padj} column carries a Storey-Tibshirani q-value
  rather than a BH-adjusted p-value. For \code{\link{fora}} we apply the
  Heyse 2011 / Lancaster mid-p correction before computing q-values to
  account for the discrete hypergeometric support.
}
\author{
  Martin Paliocha \email{martin.paliocha@nmbu.no}, with algorithmic credit
  to Korotkevich, Sukhov, Golikov and Sergushichev (upstream \pkg{fgsea}).
}
\seealso{
  \code{\link{fgsea}}, \code{\link{fgseaSimple}},
  \code{\link{fgseaMultilevel}}, \code{\link{fora}},
  \code{\link{calcGseaStat}}, \code{\link{fsgeaDevice}},
  \code{\link{fsgeaBackendInfo}}
}
\keyword{package}
