###############################################################################
# 1. Load required packages
###############################################################################
# Install packages if needed (uncomment below if not installed)
# install.packages("indicspecies")   # for multipatt
# install.packages("tidyverse")     # for data manipulation

library(indicspecies)
library(tidyverse)
library(readr)


# Create a directory for outputs if it doesn't exist
if (!dir.exists("vsearch_output/indicspecies")) {
  dir.create("vsearch_output/indicspecies")
}

###############################################################################
# 2. Read in data
###############################################################################

# Replace "asv_table.tsv" and "metadata.tsv" with your actual file paths
asv_file <- "vsearch_output/ASVs/ASV_filtered.tsv"
metadata_file <- "ref_db/spark_metadata.tsv"

# Read the ASV count table
# Assuming tab-separated, headers in the first row, row names in the first column
asv_table <- read.table(asv_file,
                        header = TRUE,
                        row.names = 1,
                        sep = "\t",
                        check.names = FALSE,
                        comment.char = "")  # check.names=FALSE keeps original column names
colnames(asv_table) <- str_replace(colnames(asv_table), "_[^_]+$", "")

# Read the metadata table
# Similarly assuming tab-separated, headers in the first row, row names in the first column
metadata <- read.table(metadata_file,
                       header = TRUE,
                       row.names = 20,
                       sep = "\t",
                       check.names = FALSE)

###############################################################################
# 3. Ensure the sample names match and are in the same order
###############################################################################

# Find the intersection of sample names between asv_table columns and metadata rows
common_samples <- intersect(colnames(asv_table), rownames(metadata))

# Subset both tables to include only the common samples
asv_table <- asv_table[, common_samples]
metadata <- metadata[common_samples, ]

# (Optional) Check dimensions
cat("ASV table dimensions:", dim(asv_table), "\n")
cat("Metadata dimensions:", dim(metadata), "\n")

###############################################################################
# 4. Run Indicator Species Analysis
###############################################################################
# For indicator analysis, you need a grouping variable (e.g., "Group", "Treatment").
# Make sure you replace "Group" below with the actual column name in your metadata
# that indicates the grouping variable of interest.

group_var <- "Case"  # <--- EDIT THIS to match the column in your metadata
grouping <- metadata[[group_var]]

# The 'multipatt' function takes the species data in rows and samples in columns.
# Our asv_table has ASVs in rows and samples in columns (which is correct).
# However, multipatt requires samples in rows and species in columns by default,
# so we transpose if necessary. Actually, 'multipatt' typically needs species in columns
# and samples in rows. Because we have ASVs as rows, we can do: t(asv_table).

# If you're unsure, you can always check ?multipatt and confirm the orientation.
indval_results <- multipatt(
  x = t(asv_table),    # transpose, so that each ASV is a column
  cluster = grouping,
  duleg=TRUE,
  control = how(nperm = 999)  # number of permutations
)

# Print a summary of the results
summary(indval_results, indvalcomp = TRUE)

###############################################################################
# 5. Interpret and export results
###############################################################################
# The 'summary()' output will list for each ASV the association index (stat), 
# p-value, and which group(s) it is indicative of.
#
# You can also extract the raw results for further manipulation or for saving:
results_df <- as.data.frame(indval_results$sign)
head(results_df)

# (Optional) Write results to a CSV or TSV for further review
write.table(
  results_df,
  file = "vsearch_output/indicspecies/Case_indicator_species_results.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)



group_var <- "Type_Group"  # <--- EDIT THIS to match the column in your metadata
grouping <- metadata[[group_var]]

# The 'multipatt' function takes the species data in rows and samples in columns.
# Our asv_table has ASVs in rows and samples in columns (which is correct).
# However, multipatt requires samples in rows and species in columns by default,
# so we transpose if necessary. Actually, 'multipatt' typically needs species in columns
# and samples in rows. Because we have ASVs as rows, we can do: t(asv_table).

# If you're unsure, you can always check ?multipatt and confirm the orientation.
indval_results <- multipatt(
  x = t(asv_table),    # transpose, so that each ASV is a column
  cluster = grouping,
  duleg=TRUE,
  control = how(nperm = 999)  # number of permutations
)

# Print a summary of the results
summary(indval_results, indvalcomp = TRUE)

###############################################################################
# 5. Interpret and export results
###############################################################################
# The 'summary()' output will list for each ASV the association index (stat), 
# p-value, and which group(s) it is indicative of.
#
# You can also extract the raw results for further manipulation or for saving:
results_df <- as.data.frame(indval_results$sign)
head(results_df)

# (Optional) Write results to a CSV or TSV for further review
write.table(
  results_df,
  file = "vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv",
  sep = "\t",
  quote = FALSE,
  col.names = NA
)