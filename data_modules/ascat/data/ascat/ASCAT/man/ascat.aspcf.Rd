% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ascat.aspcf.R
\name{ascat.aspcf}
\alias{ascat.aspcf}
\title{ascat.aspcf}
\usage{
ascat.aspcf(
  ASCATobj,
  selectsamples = 1:length(ASCATobj$samples),
  ascat.gg = NULL,
  penalty = 70,
  out.dir = ".",
  out.prefix = "",
  seed = as.integer(Sys.time())
)
}
\arguments{
\item{ASCATobj}{an ASCAT object}

\item{selectsamples}{a vector containing the sample number(s) to PCF. Default = all}

\item{ascat.gg}{germline genotypes (NULL if germline data is available)}

\item{penalty}{penalty of introducing an additional ASPCF breakpoint (expert parameter, don't adapt unless you know what you're doing)}

\item{out.dir}{directory in which output files will be written. Can be set to NA to not write PCFed files.}

\item{out.prefix}{prefix for output file names}

\item{seed}{A seed to be set when subsampling SNPs for X in males (optional, default=as.integer(Sys.time())).}
}
\value{
output: ascat data structure containing:\cr
1. Tumor_LogR data matrix\cr
2. Tumor_BAF data matrix\cr
3. Tumor_LogR_segmented: matrix of LogR segmented values\cr
4. Tumor_BAF_segmented: list of BAF segmented values; each element in the list is a matrix containing the segmented values for one sample (only for probes that are not germline homozygous)\cr
5. Germline_LogR data matrix\cr
6. Germline_BAF data matrix\cr
7. SNPpos: position of all SNPs\cr
8. ch: a list containing vectors with the indices for each chromosome (e.g. Tumor_LogR[ch[[13]],] will output the Tumor_LogR data of chromosome 13\cr
9. chr: a list containing vectors with the indices for each distinct part that can be segmented separately (e.g. chromosome arm, stretch of DNA between gaps in the array design)\cr
}
\description{
run ASPCF segmentation
}
\details{
This function can be easily parallelised by controlling the selectsamples parameter\cr
it saves the results in LogR_PCFed[sample]_[segment].txt and BAF_PCFed[sample]_[segment].txt
}
