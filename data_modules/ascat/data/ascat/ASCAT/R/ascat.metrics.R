#' Function to extract different metrics from ASCAT profiles.
#'
#' @param ASCAT_input_object R object generated by the ascat.aspcf function and given to the ascat.runAscat function.
#' @param ASCAT_output_object R object generated by the ascat.runAscat function.
#'
#' @return A dataframe (one sample per line) with the following metrics (as columns):\cr
#' 1. sex - Sex information as provided.\cr
#' 2. tumour_mapd - Median Absolute Pairwise Difference (MAPD) in tumour logR track.\cr
#' 3. normal_mapd - Median Absolute Pairwise Difference (MAPD) in normal logR track (should be NA without matched normals and 0 for sequencing data).\cr
#' 4. GC_correction_before - logR/GC correlation before correction.\cr
#' 5. GC_correction_after - logR/GC correlation after correction.\cr
#' 6. RT_correction_before - logR/RT correlation before correction.\cr
#' 7. RT_correction_after - logR/RT correlation after correction.\cr
#' 8. n_het_SNP - Number of heterozygous SNPs.\cr
#' 9. n_segs_logR - Number of segments in the logR track.\cr
#' 10. n_segs_BAF - Number of segments in the BAF track.\cr
#' 11. n_segs_logRBAF_diff - Difference between number of segments in the logR versus BAF track.\cr
#' 12. frac_homo - Fraction of homozygous (<0.1 | >0.9) probes in tumour.\cr
#' 13. purity - Purity estimate.\cr
#' 14. ploidy - Ploidy estimate.\cr
#' 15. goodness_of_fit - Goodness of fit.\cr
#' 16. n_segs - Number of copy-number segments.\cr
#' 17. segs_size - Total size of all segments.\cr
#' 18. n_segs_1kSNP - Number of segments per 1k heterozygous SNPs.\cr
#' 19. homdel_segs - Number of segments with homozygous deletion.\cr
#' 20. homdel_largest - largest segment with homozygous deletion.\cr
#' 21. homdel_size - Total size of segments with homozygous deletion.\cr
#' 22. homdel_fraction - Fraction of the genome with homozygous deletion.\cr
#' 23. LOH - Fraction of the genome with LOH (ignoring sex chromosomes).\cr
#' 24. mode_minA - Mode of the minor allele (ignoring sex chromosomes).\cr
#' 25. mode_majA - Mode of the major allele (ignoring sex chromosomes).\cr
#' 26. WGD - Whole genome doubling event (ignoring sex chromosomes).\cr
#' 27. GI - Genomic instability score (ignoring sex chromosomes).\cr
#'
#' @author tl
#' @export
ascat.metrics = function(ASCAT_input_object,ASCAT_output_object) {
  METRICS=do.call(rbind,lapply(1:length(ASCAT_input_object$samples), function(nSAMPLE) {
    SAMPLE=ASCAT_input_object$samples[nSAMPLE]
    sex=ASCAT_input_object$gender[nSAMPLE]
    tumour_mapd=round(median(abs(diff(na.omit(ASCAT_input_object$Tumor_LogR[,SAMPLE])))),4)
    if (!is.null(ASCAT_input_object$Germline_LogR) && any(SAMPLE %in% colnames(ASCAT_input_object$Germline_LogR))) {
      normal_mapd=round(median(abs(diff(na.omit(ASCAT_input_object$Germline_LogR[,SAMPLE])))),4)
    } else {
      normal_mapd=NA
    }
    if ('GC_correction_before' %in% names(ASCAT_input_object)) {GC_correction_before=ASCAT_input_object$GC_correction_before[SAMPLE]} else {GC_correction_before=NA}
    if ('GC_correction_after' %in% names(ASCAT_input_object)) {GC_correction_after=ASCAT_input_object$GC_correction_after[SAMPLE]} else {GC_correction_after=NA}
    if ('RT_correction_before' %in% names(ASCAT_input_object)) {RT_correction_before=ASCAT_input_object$RT_correction_before[SAMPLE]} else {RT_correction_before=NA}
    if ('RT_correction_after' %in% names(ASCAT_input_object)) {RT_correction_after=ASCAT_input_object$RT_correction_after[SAMPLE]} else {RT_correction_after=NA}
    if (!is.null(ASCAT_input_object$Tumor_LogR_segmented) && !is.null(ASCAT_input_object$Tumor_BAF_segmented[[nSAMPLE]])) {
      n_het_SNP=length(ASCAT_input_object$Tumor_BAF_segmented[[nSAMPLE]])
      n_segs_logR=length(rle(ASCAT_input_object$Tumor_LogR_segmented[,SAMPLE])$values)
      n_segs_BAF=length(rle(ASCAT_input_object$Tumor_BAF_segmented[[nSAMPLE]][,1])$values)
      n_segs_logRBAF_diff=abs(n_segs_logR-n_segs_BAF)
      segm_baf=ASCAT_input_object$Tumor_BAF[rownames(ASCAT_input_object$Tumor_BAF_segmented[[nSAMPLE]]),SAMPLE]
      frac_homo=round(length(which(segm_baf<0.1 | segm_baf>0.9))/length(segm_baf),4)
      rm(segm_baf)
    } else {
      n_het_SNP=NA
      n_segs_logR=NA
      n_segs_BAF=NA
      n_segs_logRBAF_diff=NA
      frac_homo=NA
    }
    if (!is.null(ASCAT_output_object$segments) && SAMPLE %in% ASCAT_output_object$segments$sample) {
      purity=round(as.numeric(ASCAT_output_object$purity[SAMPLE]),4)
      ploidy=round(as.numeric(ASCAT_output_object$ploidy[SAMPLE]),4)
      goodness_of_fit=round(ASCAT_output_object$goodnessOfFit[SAMPLE],4)
      profile=ASCAT_output_object$segments[ASCAT_output_object$segments$sample==SAMPLE,]
      profile$size=profile$endpos-profile$startpos+1
      n_segs=nrow(profile)
      segs_size=sum(profile$size)
      n_segs_1kSNP=round(n_segs/(length(ASCAT_input_object$Tumor_BAF_segmented[[nSAMPLE]])/1e3),4)
      INDEX_HD=which(profile$nMajor==0 & profile$nMinor==0)
      if (length(INDEX_HD)>0) {
        homdel_segs=length(INDEX_HD)
        homdel_largest=max(profile$size[INDEX_HD])
        homdel_size=sum(profile$size[INDEX_HD])
        homdel_fraction=round(homdel_size/sum(profile$size),4)
      } else {
        homdel_segs=homdel_largest=homdel_size=homdel_fraction=0
      }
      rm(INDEX_HD)
      profile=profile[which(profile$chr %in% setdiff(ASCAT_input_object$chrs,ASCAT_input_object$sexchromosomes)),] # do not consider sex chromosomes for the next metrics
      LOH=round(sum(profile$size[which(profile$nMinor==0)])/sum(profile$size),4)
      mode_minA=modeAllele(profile,'nMinor')
      mode_majA=modeAllele(profile,'nMajor')
      if (mode_majA==0 || !(mode_majA %in% 1:5)) {
        WGD=NA
        GI=NA
      } else {
        if (mode_majA==1) {
          WGD=0
          GI=computeGIscore(WGD,profile)
        } else if (mode_majA==2) {
          WGD=1
          GI=computeGIscore(WGD,profile)
        } else if (mode_majA %in% 3:5) {
          WGD='1+'
          GI=computeGIscore(1,profile)
        }
      }
      rm(profile)
    } else {
      purity=NA
      ploidy=NA
      goodness_of_fit=NA
      n_segs=NA
      segs_size=NA
      n_segs_1kSNP=NA
      homdel_segs=NA
      homdel_largest=NA
      homdel_size=NA
      homdel_fraction=NA
      LOH=NA
      mode_minA=NA
      mode_majA=NA
      WGD=NA
      GI=NA
    }
    OUT=data.frame(sex=sex,
                   tumour_mapd=tumour_mapd,
                   normal_mapd=normal_mapd,
                   GC_correction_before=GC_correction_before,
                   GC_correction_after=GC_correction_after,
                   RT_correction_before=RT_correction_before,
                   RT_correction_after=RT_correction_after,
                   n_het_SNP=n_het_SNP,
                   n_segs_logR=n_segs_logR,
                   n_segs_BAF=n_segs_BAF,
                   n_segs_logRBAF_diff=n_segs_logRBAF_diff,
                   frac_homo=frac_homo,
                   purity=purity,
                   ploidy=ploidy,
                   goodness_of_fit=goodness_of_fit,
                   n_segs=n_segs,
                   segs_size=segs_size,
                   n_segs_1kSNP=n_segs_1kSNP,
                   homdel_segs=homdel_segs,
                   homdel_largest=homdel_largest,
                   homdel_size=homdel_size,
                   homdel_fraction=homdel_fraction,
                   LOH=LOH,
                   mode_minA=mode_minA,
                   mode_majA=mode_majA,
                   WGD=WGD,
                   GI=GI,
                   stringsAsFactors=F)
    rownames(OUT)=SAMPLE
    return(OUT)
  }))
  return(METRICS)
}

#' Function to get mode of the allele (either minor or major)
#' @noRd
modeAllele=function(cn,col) {
  y=round(cn[,col])
  y[y>5]=5
  y=tapply(1:nrow(cn),y,function(z) sum((cn[z,'endpos']-cn[z,'startpos'])/1e6))
  ord=order(y,decreasing=T)
  y=y[ord]
  return(as.numeric(names(y)[which.max(y)]))
}

#' Function to compute GI score based on WGD information
#' @noRd
computeGIscore=function(WGD,profile) {
  stopifnot(WGD %in% 0:2)
  if (WGD==0) {
    baseline=1
  } else if (WGD==1) {
    baseline=2
  } else if (WGD==2) {
    baseline=4
  }
  return(round(1-sum(profile$size[which(profile$nMajor==baseline & profile$nMinor==baseline)])/sum(profile$size),4))
}
