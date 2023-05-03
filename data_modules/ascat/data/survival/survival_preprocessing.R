library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

library(survminer)


# Get the TCGA survival data from the RTCGA package: https://bioconnector.github.io/workshops/r-survival.html#rtcga
library(RTCGA)
library(RTCGA.clinical)
library(RTCGA.mRNA)
clin <- survivalTCGA(ACC.clinical,
                     BLCA.clinical,
                     BRCA.clinical,
                     CESC.clinical,
                     CHOL.clinical,
                     COAD.clinical,
                     COADREAD.clinical,
                     DLBC.clinical,
                     ESCA.clinical,
                     FPPP.clinical,
                     GBM.clinical,
                     GBMLGG.clinical,
                     HNSC.clinical,
                     KICH.clinical,
                     KIPAN.clinical,
                     KIRC.clinical,
                     KIRP.clinical,
                     LAML.clinical,
                     LGG.clinical,
                     LIHC.clinical,
                     LUAD.clinical,
                     LUSC.clinical,
                     MESO.clinical,
                     OV.clinical,
                     PAAD.clinical,
                     PCPG.clinical,
                     PRAD.clinical,
                     READ.clinical,
                     SARC.clinical,
                     SKCM.clinical,
                     STAD.clinical,
                     STES.clinical,
                     TGCT.clinical,
                     THCA.clinical,
                     THYM.clinical,
                     UCEC.clinical,
                     UCS.clinical,
                     UVM.clinical, #BRCA.clinical, OV.clinical, KIRC.clinical,
                     extract.cols=c("admin.disease_code","patient.gender", "patient.days_to_birth"))

# Convert to numeric and flip days to birth
clin$patient.days_to_birth <- as.numeric(clin$patient.days_to_birth)
clin$patient.days_since_birth <- -1 * as.numeric(clin$patient.days_to_birth)

# Remove patients with negative survival time
clin <- clin[clin$time>0,]

# Check this looks reasonable
hist(clin$patient.days_since_birth/365)
unique(clin$patient.gender)


clin <- na.omit(clin)
colSums(!is.na(clin))

# save to be used in the python PyTorch data module
library(feather)
write_feather(clin, "survival_data.feather")

###########################################################

# Playing around, viewing what data may be predictive
filtered_clinical <- ACC.clinical %>% select_if(~ !any(is.na(.)))


# View the CNA data from ASCAT
library(readr)
df<-readr::read_tsv("data/ascat/ReleasedData/TCGA_SNP6_hg19/summary.ascatv3TCGA.penalty70.hg19.tsv")


# # Playing around with ASCAT
# library(ASCAT)
# ascat.bc = ascat.loadData(Tumor_LogR_file = "Tumor_LogR.txt", Tumor_BAF_file = "Tumor_BAF.txt", Germline_LogR_file = "Germline_LogR.txt", Germline_BAF_file = "Germline_BAF.txt", gender = rep('XX',100), genomeVersion = "hg19") # isTargetedSeq=T for targeted sequencing data
# ascat.plotRawData(ascat.bc, img.prefix = "Before_correction_")
# ascat.bc = ascat.correctLogR(ascat.bc, GCcontentfile = "GC_example.txt", replictimingfile = "RT_example.txt")
# ascat.plotRawData(ascat.bc, img.prefix = "After_correction_")
# ascat.bc = ascat.aspcf(ascat.bc) # penalty=25 for targeted sequencing data
# ascat.plotSegmentedData(ascat.bc)
# ascat.output = ascat.runAscat(ascat.bc, write_segments=T) # gamma=1 for HTS data
# QC = ascat.metrics(ascat.bc,ascat.output)