library(indicspecies)
library(tidyverse)
library(readr)


<<<<<<< HEAD
if (!dir.exists("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies")) {
  dir.create("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies")
}

asv_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/ASVs/ASV_final.micro.tsv"
metadata_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/metadata/metadata_updated.tsv"
asv_table <- read.table(asv_file,
=======
if (!dir.exists("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies")) {
  dir.create("/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies")
}

#asv_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/ASVs/ASV_final.micro.tsv"
asv_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/cca/ASV_Table.tsv"
metadata_file <- "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/ref_db/VOC_table.tsv"
asv_table <- t(read.table(asv_file,
>>>>>>> 1dc7c5d (voc edits)
                        header = TRUE,
                        row.names = 1,
                        sep = "\t",
                        check.names = FALSE,
                        comment.char = ""))

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

<<<<<<< HEAD
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/status_indicator_species_results.tsv",
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/status_indicator_species_summary.tsv", 
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/status_indicator_species_results_DULEG.tsv",
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/status_indicator_species_summary_DULEG.tsv", 
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)


group_var <- "type_group"
=======
group_var <- "subclass2"
>>>>>>> 1dc7c5d (voc edits)
grouping <- metadata[[group_var]]

indval_results <- multipatt(
  x = t(asv_table),
  cluster = grouping,
  control = how(nperm = 999)
)
results_df <- as.data.frame(indval_results$sign)

write.table(
  results_df,
<<<<<<< HEAD
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/type_group_indicator_species_results.tsv",
=======
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/subclass2_indicator_species_results_Brush.tsv",
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/subclass2_indicator_species_summary_Brush.tsv", 
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/indicspecies/subclass2_indicator_species_results_DULEG_Brush.tsv",
>>>>>>> 1dc7c5d (voc edits)
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
<<<<<<< HEAD
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/type_group_indicator_species_summary.tsv", 
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/type_group_indicator_species_results_DULEG.tsv",
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
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/spark_old_output/indicspecies/type_group_indicator_species_summary_DULEG.tsv", 
=======
  file = "/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/cca/subclass2_indicator_lineage_summary_DULEG_Brush.tsv", 
>>>>>>> 1dc7c5d (voc edits)
  sep = "\t", 
  quote = FALSE, 
  row.names = FALSE
)