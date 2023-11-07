library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

# Calls
# https://github.com/raphael-group/chisel-data/blob/master/patientS0/calls/sectionE/calls.tsv.gz
df_calls <- read.csv("calls.tsv", sep="\t")
write.csv(df_calls, "calls.csv")

# Clones obtained from CHISEL
#  https://github.com/raphael-group/chisel-data/tree/master/patientS0/clones/sectionE/mapping.tsv.gz
df_clones <- read.csv("mapping.tsv", sep="\t")
write.csv(df_clones, "mapping.csv")

# Merge the two files -> merged.csv