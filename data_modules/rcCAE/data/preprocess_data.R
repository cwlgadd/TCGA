library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

# https://drive.google.com/file/d/1TJ0KqTqG6KjIMdRwgyV0Npf8wKdnYXHo/view?usp=share_link

headers = read.csv("breast_tissue_E_2k_node_cnv_calls.bed", skip = 2, header = F, nrows = 1, as.is = T, sep="\t")
headers[1] <- sub('.', '', headers[1])

df <- read.csv("breast_tissue_E_2k_node_cnv_calls.bed", skip = 3, header=F, sep="\t")
colnames(df) <- headers

write.csv(df, "breast_tissue_E_2k_node_cnv_calls.csv")