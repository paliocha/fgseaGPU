\name{fgseaDevice}
\alias{fgseaDevice}
\title{Query or set the default execution device}
\description{
  Returns the current device hint when called without arguments. Setting
  the device persists for the duration of the R session.
}
\usage{
fgseaDevice(device)
}
\arguments{
  \item{device}{One of \code{"auto"}, \code{"cpu"}, \code{"cuda"},
    \code{"mps"}, \code{"rocm"}. Omit to query the current default.}
}
\value{
  The resolved device string (a character scalar).
}
\examples{
fgseaDevice()
}
