# The Cancer Genome Atlas for PyTorch models

Package for loading [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) data into PyTorch models. Just add this project to your python path, or as a submodule. 

## Data modules

    data_modules.simulated
  - We simulate from a Markov Process, where each transition belongs to one of a pre-determined set.
  - Whole Genome Doubling (WGD) is not modelled.


    data_modules.ASCAT
- [Allele-Specific Copy number Analysis of Tumours (ASCAT)](https://www.crick.ac.uk/research/labs/peter-van-loo/software#:~:text=ASCAT%20is%20a%20tool%20to,variant%2C%20polymorphic%20in%20a%20population.).
  - ASCAT is a tool to detect somatic copy number alterations (CNAs) in cancer samples.

