library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

# https://github.com/raphael-group/chisel-data/blob/master/patientS0/calls/sectionE/calls.tsv.gz

# Calls
df_calls <- read.csv("calls.tsv", sep="\t")
write.csv(df_calls, "calls.csv")

# Clones obtained from CHISEL
df_clones <- read.csv("mapping.tsv", sep="\t")
write.csv(df_clones, "mapping.csv")
