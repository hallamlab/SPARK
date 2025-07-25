library(indicspecies)
library(tidyverse)
library(readr)


if (!dir.exists("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies")) {
  dir.create("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies")
}

asv_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/ASVs/ASV_final.micro.tsv"
metadata_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/metadata/metadata_updated.tsv"
asv_table <- read.table(asv_file,
                        header = TRUE,
                        row.names = 1,
                        sep = "\t",
                        check.names = FALSE,
                        comment.char = "")

metadata <- read.table(metadata_file,
                       header = TRUE,
                       row.names = 1,
                       sep = "\t",
                       check.names = FALSE)

common_samples <- intersect(colnames(asv_table), rownames(metadata))

asv_table <- asv_table[, common_samples]
metadata <- metadata[common_samples, ]

cat("ASV table dimensions:", dim(asv_table), "\n")
cat("Metadata dimensions:", dim(metadata), "\n")

group_var <- "status"
grouping <- metadata[[group_var]]
indval_results <- multipatt(
  x = t(asv_table),
  cluster = grouping,
  control = how(nperm = 999)
)
results_df <- as.data.frame(indval_results$sign)

write.table(
  results_df,
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/status_indicator_species_results.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

indval_df <- cbind(
  ASV = rownames(indval_results$sign),
  indval_results$sign,
  A = indval_results$A,
  B = indval_results$B
) %>% 
  as.data.frame()

write.table(
  indval_df, 
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/status_indicator_species_summary.tsv", 
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)
indval_results <- multipatt(
  x = t(asv_table),
  cluster = grouping,
  duleg=TRUE,
  control = how(nperm = 999)
)
results_df <- as.data.frame(indval_results$sign)

write.table(
  results_df,
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/status_indicator_species_results_DULEG.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

indval_df <- cbind(
  ASV = rownames(indval_results$sign),
  indval_results$sign,
  A = indval_results$A,
  B = indval_results$B
) %>% 
  as.data.frame()

write.table(
  indval_df, 
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/status_indicator_species_summary_DULEG.tsv", 
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)


group_var <- "type_group"
grouping <- metadata[[group_var]]

indval_results <- multipatt(
  x = t(asv_table),
  cluster = grouping,
  control = how(nperm = 999)
)
results_df <- as.data.frame(indval_results$sign)

write.table(
  results_df,
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/type_group_indicator_species_results.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

indval_df <- cbind(
  ASV = rownames(indval_results$sign),
  indval_results$sign,
  A = indval_results$A,
  B = indval_results$B
) %>% 
  as.data.frame()

write.table(
  indval_df, 
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/type_group_indicator_species_summary.tsv", 
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)

indval_results <- multipatt(
  x = t(asv_table),
  cluster = grouping,
  duleg=TRUE,
  control = how(nperm = 999)
)
results_df <- as.data.frame(indval_results$sign)

write.table(
  results_df,
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/type_group_indicator_species_results_DULEG.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

indval_df <- cbind(
  ASV = rownames(indval_results$sign),
  indval_results$sign,
  A = indval_results$A,
  B = indval_results$B
) %>% 
  as.data.frame()

write.table(
  indval_df, 
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/type_group_indicator_species_summary_DULEG.tsv", 
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)