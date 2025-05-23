% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ascat.prepareTargetedSeq.R
\name{ascat.prepareTargetedSeq}
\alias{ascat.prepareTargetedSeq}
\title{Method to extract a curated list of SNPs covered by a targeted sequencing experiment.}
\usage{
ascat.prepareTargetedSeq(
  Worksheet,
  Workdir,
  alleles.prefix,
  BED_file,
  allelecounter_exe,
  genomeVersion,
  nthreads = 1,
  minCounts = 10,
  chrom_names = c(1:22, "X"),
  min_base_qual = 20,
  min_map_qual = 35,
  ref.fasta = NA,
  plotQC = T
)
}
\arguments{
\item{Worksheet}{A table with the following columns: Patient_ID, Normal_ID, Normal_file and Gender. Must contain one single normal per patient. Normal_file can either be BAMs/CRAMs or paths to pre-computed (zipped) alleleCounts (e.g. "sample_alleleCounts_chr"). Gender must either be XX (females) or XY (males).}

\item{Workdir}{The folder where output should go (will be created if it doesn't exist).}

\item{alleles.prefix}{Prefix path to the allele data (e.g. "G1000_alleles_chr").}

\item{BED_file}{A BED file for only looking at SNPs within specific intervals. Must fit with the design used for targeted sequencing.}

\item{allelecounter_exe}{Path to the allele counter executable.}

\item{genomeVersion}{Genome version, either 'hg19' or 'hg38'.}

\item{nthreads}{The number of parallel processes to speed up the process (optional, default=1).}

\item{minCounts}{Minimum depth required in the normal for a SNP to be considered (optional, default=10).}

\item{chrom_names}{A vector containing the names of chromosomes to be considered (optional, default=c(1:22,'X')).}

\item{min_base_qual}{Minimum base quality required for a read to be counted (optional, default=20).}

\item{min_map_qual}{Minimum mapping quality required for a read to be counted (optional, default=35).}

\item{ref.fasta}{FASTA file used for generating CRAMs (optional, default=NA).}

\item{plotQC}{A boolean to generate QC reports as PNGs (optional, default=T).}
}
\description{
From a complete set of loci (alleles.prefix), this method will keep SNPs falling into the targeted design (based on BED_file) and check allele counts in normal samples (listed in Worksheet). The cleaned list of loci/allele files will be located under Workdir/alleleData/Cleaned/.
}
