	# Load necessary libraries
library(rlang)
library(tibble)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(SpiecEasi)
library(igraph)
library(tidyverse)


.libPaths(new = .libPaths()[1])


# Create a directory for outputs if it doesn't exist
if (!dir.exists("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi")) {
  dir.create("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi")
}

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
count_data_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/ASVs/ASV_final.micro.tsv"

# Define file paths for saving intermediate objects
filtered_count_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/count_data_filtered.RDS"
spiec_easi_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/se_gl_cnt.RDS"
igraph_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/ig_gl.RDS"
layout_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/am_coord.RDS"

# Function to check if a file exists and load it, else return FALSE
load_if_exists <- function(path) {
  if (file.exists(path)) {
    readRDS(path)
  } else {
    FALSE
  }
}

# **Step 1: Load or Compute Filtered Count Data**

# Attempt to load the filtered count data
count_data_filtered <- load_if_exists(filtered_count_path)

if (identical(count_data_filtered, FALSE)) {
  # Load count data, ensuring it remains as numeric and handle non-numeric cases
  count_data <- read.table(count_data_path, row.names = 1, 
                           header = TRUE, sep = "\t",
                           comment.char = ""
                           )
  #colnames(count_data) <- str_replace(colnames(count_data), "_[^_]+$", "")

  # Transpose the data; t() returns a matrix so we convert it back to a data frame
  #df_transposed <- as.data.frame(count_data, stringsAsFactors = FALSE)  
  df_transposed <- as.data.frame(t(count_data), stringsAsFactors = FALSE)  
  count_data <- as.matrix(sapply(df_transposed, as.numeric))

  # Verify the structure of count_data
  print(dim(count_data))      # Should show samples as rows and ASVs as columns

  # Convert raw counts to relative abundances per sample (row)
  #rel_abund_matrix <- count_data / rowSums(count_data)
  # For each ASV (column), find the maximum relative abundance across all samples
  #max_ra <- apply(rel_abund_matrix, 2, max)
  # Filter to keep ASVs that reach at least 0.05% (0.0005) relative abundance in at least one sample
  count_data_filtered <- count_data #[, max_ra >= 0.0005]

  # Calculate prevalence for each ASV (column)
  #prevalence <- colSums(count_data_filtered > 0) / nrow(count_data_filtered)
  # Filter ASVs based on the prevalence threshold
  #count_data_filtered <- count_data_filtered[, prevalence >= 0.005]
  # Optionally, remove ASVs with zero variance after filtering
  #count_data_filtered <- count_data_filtered[, apply(count_data_filtered, 2, var) > 0]

  # Verify the dimensions after filtering
  print(dim(count_data_filtered))      # Check how many ASVs remain



  # **Save the filtered count data**
  saveRDS(count_data_filtered, filtered_count_path)
  print(paste("Filtered count data saved to", filtered_count_path))
} else {
  print(paste("Loaded filtered count data from", filtered_count_path))
}

# **Step 2: Load or Compute SpiecEasi Network Analysis**

# Attempt to load the SpiecEasi object
se.gl.cnt <- load_if_exists(spiec_easi_path)

# Assuming your data frame is named df
print(paste("Number of rows:", nrow(count_data_filtered)))
print(paste("Number of columns:", ncol(count_data_filtered)))

if (identical(se.gl.cnt, FALSE)) {
  # Define SpiecEasi parameters
  pargs <- list(rep.num = 50, seed = 10010, ncores = 6, thresh = 0.1)
  
  # Run SpiecEasi with the filtered count data
  se.gl.cnt <- spiec.easi(count_data_filtered, 
                          method = 'glasso', 
                          lambda.min.ratio = 1e-2,
                          nlambda = 20, 
                          pulsar.params = pargs)
  # **Save the SpiecEasi object**
  saveRDS(se.gl.cnt, spiec_easi_path)
  print(paste("SpiecEasi object saved to", spiec_easi_path))
} else {
  print(paste("Loaded SpiecEasi object from", spiec_easi_path))
}

# **Step 3: Load or Compute igraph Object**

# Attempt to load the igraph object
ig.gl <- load_if_exists(igraph_path)
am.coord <- load_if_exists(layout_path)

if (identical(ig.gl, FALSE) || identical(am.coord, FALSE)) {
  # Step 1: Extract the optimal precision matrix
  opt_index <- se.gl.cnt$select$stars$opt.index
  precision_matrix <- as.matrix(se.gl.cnt$est$icov[[opt_index]])
  precision_df <- as.data.frame(precision_matrix)
  precision_df <- tibble::rownames_to_column(precision_df, var = "ASV_ID")
  write.csv(precision_df, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/precision_matrix.csv", row.names = FALSE)

  # Step 2: Compute the partial correlation matrix
  D_inv_sqrt <- diag(1 / sqrt(diag(precision_matrix)))
  partial_cor_matrix <- -D_inv_sqrt %*% precision_matrix %*% D_inv_sqrt
  diag(partial_cor_matrix) <- 0  # Remove self-loops
  partial_cor_df <- as.data.frame(partial_cor_matrix)
  partial_cor_df <- tibble::rownames_to_column(partial_cor_df, var = "ASV_ID")
  write.csv(partial_cor_df, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/partial_correlation_matrix.csv", row.names = FALSE)

  # Step 3: Take absolute values and symmetrize
  adj_bin <- as.matrix(se.gl.cnt$refit$stars)
  adj_df <- as.data.frame(adj_bin)
  adj_df <- tibble::rownames_to_column(adj_df, var = "ASV_ID")
  write.csv(adj_df, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/adj_STARS_matrix.csv", row.names = FALSE)


  threshold <- 0.1 #.001  # Adjust based on the distribution of edge weights

  adj_weighted <- partial_cor_matrix
  adj_weighted[ abs(adj_weighted) < threshold ] <- 0
  adj_weighted[ adj_weighted < 0 ] <- 0
  write.csv(adj_df, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/adj_weighted_matrix.csv", row.names = FALSE)

  # Force symmetry on the weighted adjacency matrix
  adj_weighted <- (adj_weighted + t(adj_weighted)) / 2
  # Step 5: Create the igraph object
  ig.gl <- graph_from_adjacency_matrix(
    adj_weighted,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE)
  # Extract edge weights
  edge_weights <- E(ig.gl)$weight
  # Display summary statistics
  print(summary(edge_weights))


  all_weights <- partial_cor_matrix
  all_weights[ all_weights < 0 ] <- 0
  all_weights <- (all_weights + t(all_weights)) / 2

  ig.gl.all <- graph_from_adjacency_matrix(
    all_weights,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE)
  # Extract edge weights
  edge_weights <- E(ig.gl.all)$weight
  # Display summary statistics
  print(summary(edge_weights))



  # 1) Copy your partial correlation matrix
  signed_mat <- partial_cor_matrix

  # 2) Zero out only very small magnitudes (e.g. |r| < 0.1)
  signed_mat[ abs(signed_mat) < threshold ] <- 0

  # 3) Build the igraph
  ig_signed <- graph_from_adjacency_matrix(
    signed_mat,
    mode     = "undirected",
    weighted = TRUE,
    diag     = FALSE
  )

  # 4) Check the distribution
  summary(E(ig_signed)$weight)
  
  E(ig_signed)$color <- ifelse(E(ig_signed)$weight > 0, "red", "blue")
  
  layout_nicely_coords <- layout_nicely(ig.gl)
  layout_nicely_coords_signed <- layout_nicely(ig_signed)
  layout_nicely_coords_signed <- layout_nicely(ig.gl.all)
  V(ig.gl)$x <- layout_nicely_coords[,1]
  V(ig.gl)$y <- layout_nicely_coords[,2]
  V(ig_signed)$x <- layout_nicely_coords_signed[,1]
  V(ig_signed)$y <- layout_nicely_coords_signed[,2]
  V(ig.gl.all)$x <- layout_nicely_coords_signed[,1]
  V(ig.gl.all)$y <- layout_nicely_coords_signed[,2]

  if (!is.matrix(layout_nicely_coords)) {
    layout_nicely_coords <- matrix(layout_nicely_coords, ncol = 2)
  }

  layout_fr_coords <- layout_with_fr(ig.gl, niter = 1000)
  if (!is.matrix(layout_fr_coords)) {
    layout_fr_coords <- matrix(layout_fr_coords, ncol = 2)
  }

  layout_kk_coords <- layout_with_kk(ig.gl)
  if (!is.matrix(layout_kk_coords)) {
    layout_kk_coords <- matrix(layout_kk_coords, ncol = 2)
  }

  layout_drl_coords <- layout_with_drl(ig.gl)
  if (!is.matrix(layout_drl_coords)) {
    layout_drl_coords <- matrix(layout_drl_coords, ncol = 2)
  }


  # Adjust the vertex sizes (this is your current method)
  vsize <- rowMeans(clr(count_data_filtered, 1)) + 6
  vsize <- vsize * 0.5  # Scale down the sizes

  # Save the igraph object and one of the layouts (e.g., the layout_nicely)
  saveRDS(ig.gl, igraph_path)
  saveRDS(layout_nicely_coords, layout_path)
  print(paste("igraph object saved to", igraph_path))
  print(paste("Layout coordinates saved to", layout_path))

  # Open a multipage PDF to save all layouts in one file
  pdf("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/multipage_layouts.pdf", width = 10, height = 10)

  # Page 1: Default layout_nicely
  plot(ig.gl, layout = layout_nicely_coords, vertex.size = vsize, vertex.label = NA, 
       main = "Layout: layout_nicely")

  # Page 2: Fruchterman-Reingold layout
  plot(ig.gl, layout = layout_fr_coords, vertex.size = vsize, vertex.label = NA, 
       main = "Layout: Fruchterman-Reingold (layout_with_fr)")

  # Page 3: Kamada-Kawai layout
  plot(ig.gl, layout = layout_kk_coords, vertex.size = vsize, vertex.label = NA, 
       main = "Layout: Kamada-Kawai (layout_with_kk)")

  # Page 4: DRL layout
  plot(ig.gl, layout = layout_drl_coords, vertex.size = vsize, vertex.label = NA, 
       main = "Layout: DRL (layout_with_drl)")

  # Close the PDF device to write the file
  dev.off()
  write_graph(ig.gl, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_pos_sub.graphml", format = "graphml")
  write_graph(ig.gl, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_pos_sub.gml", format = "gml")
  write_graph(ig_signed, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_signed.graphml", format = "graphml")
  write_graph(ig_signed, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_signed.gml", format = "gml")
  write_graph(ig.gl.all, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_pos_all.graphml", format = "graphml")
  write_graph(ig.gl.all, file = "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/network_pos_all.gml", format = "gml")

} else {
  print(paste("Loaded igraph object from", igraph_path))
  print(paste("Loaded layout coordinates from", layout_path))
  
  # Ensure that vsize is computed since it's not saved
  vsize <- rowMeans(clr(count_data_filtered, 1)) + 6
  vsize <- vsize * 0.75  # Scale down the sizes

}

# Extract the edge list from the igraph object
edge_list <- as.data.frame(as_edgelist(ig.gl))
edge_list$weight <- E(ig.gl)$weight

# Assign meaningful column names
colnames(edge_list) <- c("Taxon1", "Taxon2", "Weight")

# Save the edge list as a CSV file
edge_list_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/edge_list.csv"
write.csv(edge_list, edge_list_path, row.names = FALSE)

print(paste("Edge list saved to", edge_list_path))

# **Step 5: Save Tabular Edge List with Original ASV IDs**

# Check if igraph vertices have names; if not, assign ASV IDs that match
if (is.null(V(ig.gl)$name)) {
  print("igraph vertices do not have names. Assigning matching ASV IDs.")
  filtered_asv_ids <- colnames(count_data_filtered)
  graph_asv_ids <- filtered_asv_ids[filtered_asv_ids %in% filtered_asv_ids[V(ig.gl)]]
  
  if (length(graph_asv_ids) == length(V(ig.gl))) {
    V(ig.gl)$name <- graph_asv_ids
  } else {
    stop("Mismatch between igraph vertices and ASV IDs. Please debug.")
  }
}



# **Step: Save Tabular Edge List with Original ASV IDs**

# Extract the edge list from the igraph object
edge_list <- as.data.frame(as_edgelist(ig.gl))

# Map vertex indices directly using the names from the igraph object
edge_list$Taxon1 <- V(ig.gl)$name[match(edge_list$V1, V(ig.gl)$name)]
edge_list$Taxon2 <- V(ig.gl)$name[match(edge_list$V2, V(ig.gl)$name)]

# Add edge weights to the edge list
edge_list$Weight <- E(ig.gl)$weight

# Retain only the ASV IDs and weights, renaming columns for clarity
edge_list <- edge_list[, c("Taxon1", "Taxon2", "Weight")]

# Save the edge list as a CSV file
edge_list_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/edge_list_with_asv_ids.csv"
write.csv(edge_list, edge_list_path, row.names = FALSE)

print(paste("Edge list with ASV IDs saved to", edge_list_path))

### EXPORT NODE FEATURES (CENTRALITIES)
node_degree      <- igraph::degree(ig.gl)
node_betweenness <- igraph::betweenness(ig.gl)
node_closeness   <- igraph::closeness(ig.gl)
node_eigen       <- igraph::eigen_centrality(ig.gl)$vector
node_ids <- paste0("n", seq_along(V(ig.gl)) - 1)

node_features <- data.frame(
  GraphML_ID  = node_ids,
  Taxon        = V(ig.gl)$name,
  Degree       = node_degree,
  Betweenness  = node_betweenness,
  Closeness    = node_closeness,
  EigenCentral = node_eigen
)
write.csv(node_features, "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/node_features.csv", row.names = FALSE)

# **End of Script**
