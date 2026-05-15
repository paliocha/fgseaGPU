\name{fgseaBackendInfo}
\alias{fgseaBackendInfo}
\title{Report build-time backend information}
\description{
  Returns whether the package was built with LibTorch support, the
  resolved compute device, and the number of CPU threads available.
}
\usage{
fgseaBackendInfo()
}
\value{
  A list with three entries: \code{torch_built} (logical),
  \code{device} (character), \code{concurrency} (integer).
}
\examples{
fgseaBackendInfo()
}
