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
library(igraph)

.libPaths(new = .libPaths()[1])

# Create a directory for outputs if it doesn't exist
if (!dir.exists("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi")) {
  dir.create("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi")
}

# ── 1. Read in count table (OTUs × samples) and metadata (samples × variables)
counts_file <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/ASVs/ASV_final.micro.tsv"
counts <- read.table(counts_file,
                     header = TRUE,
                     row.names = 1,
                     sep = "\t",
                     check.names= FALSE)
metadata_file <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/metadata/metadata_updated.tsv"
metadata <- read.table(metadata_file,
                       header = TRUE,
                       row.names = 21,
                       sep = "\t",
                       check.names= FALSE)

shared_samps <- intersect(colnames(counts), rownames(metadata))
counts <- counts[, shared_samps]
metadata <- metadata[shared_samps, , drop = FALSE]

out_dir <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_networks"
# Create a directory for outputs if it doesn't exist
if (!dir.exists(out_dir)) {
  dir.create(out_dir)
}
# Check parallel capability
can_fork <- .Platform$OS.type!="windows"
actual_cores <- if (can_fork) 6 else 1
if (!can_fork && 6>1) {
  warning("Parallel forking not supported on this platform; using 1 core.")
}

process_patient <- function(pt) {
  message("→ Processing patient: ", pt)
  samp_ids <- rownames(metadata)[ metadata[['patient_code']] == pt ]
  mat      <- counts[, samp_ids, drop=FALSE]
  sub_mat  <- t(as.matrix(mat))
  storage.mode(sub_mat) <- "numeric"

  if (nrow(sub_mat) < 2) {
    warning("  Skipping ", pt, " (only ", nrow(sub_mat), " sample)")
    return(NULL)
  }
  # Single-run SparCC
  message("   • single-run SparCC …")
  sc1 <- sparcc(sub_mat, th=0.1)

  # Bootstrapped SparCC
  message("   • bootstrapped SparCC (", 5, " reps) …")
  bb <- sparccboot(
    sub_mat,
    sparcc.params = list(iter=10, inner_iter=5, th=0.1),
    R              = 5,
    ncpus          = actual_cores
  )

  # 1) call pval.sparccboot() with exactly one argument
  pval_vec <- pval.sparccboot(bb)  # only 'x', no 'sided' or 'sides' :contentReference[oaicite:0]{index=0}

  # 2) force it into a numeric vector
  pval_vec <- as.numeric(unlist(pval_vec))

  # 3) determine how many ASVs there are and how many unique pairs
  p <- ncol(sub_mat)
  num_pairs <- p*(p-1)/2

  # 4) if you got twice as many (i.e. p*(p-1)), cut it in half
  if (length(pval_vec) == 2 * num_pairs) {
    pval_vec <- pval_vec[1:num_pairs]
  } else if (length(pval_vec) != num_pairs) {
    stop("Unexpected length of pval_vec: ", length(pval_vec))
  }

  # 5) build a full p×p matrix of p-values
  asv_names <- colnames(sub_mat)
  pvals <- matrix(NA_real_, nrow=p, ncol=p,
                  dimnames=list(asv_names, asv_names))
  ut <- upper.tri(pvals, diag=FALSE)
  pvals[ut] <- pval_vec
  pvals[ t(ut) ] <- pvals[ut]   # mirror to lower triangle

  orig_cor <- sc1$Cor
  mask     <- (pvals < 0.05) & (abs(orig_cor) >= 0.5)
  cm2      <- orig_cor
  cm2[!mask] <- 0
  cm2       <- (cm2 + t(cm2)) / 2

  # after you compute cm2:
  asv_names <- colnames(sub_mat)
  rownames(cm2) <- asv_names
  colnames(cm2) <- asv_names

  G2 <- graph_from_adjacency_matrix(
    cm2,
    mode         = "undirected",
    weighted     = TRUE,
    diag         = FALSE,
    add.rownames = "ASV_ID"
  )
  out2 <- file.path(out_dir, paste0(pt, "_sparcc.graphml"))
  write_graph(G2, file=out2, format="graphml")
  message("     → wrote ", out2)

  invisible(NULL)
}

# Loop over patients
patients <- unique(metadata[['patient_code']])
lapply(patients, process_patient)
message("All done.")