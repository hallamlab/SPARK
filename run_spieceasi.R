#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(tibble)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(SpiecEasi)
  library(igraph)
})

# ----------------------------- CLI -------------------------------------------
opt <- list(
  make_option("--counts", type="character", default=NULL,
              help="Path to ASV table (TSV). Rows=ASVs, Cols=samples by default."),
  make_option("--transpose", type="logical", default=TRUE,
              help="Transpose counts so rows=samples, cols=ASVs [default %default]."),
  make_option("--strip-suffix-regex", type="character", default=NULL,
              help="Optional regex to strip from sample column names before transpose (e.g. '_[^_]+$')."),
  make_option("--outdir", type="character", default=NULL,
              help="Output directory for all results (required)."),
  make_option("--prefix", type="character", default="spieceasi",
              help="Filename prefix for outputs [default %default]."),

  # Filtering gates (set to 0 to disable)
  make_option("--min-rel-abund", type="double", default=0.0,
              help="Keep ASVs reaching at least this relative abundance in >=1 sample [default %default]."),
  make_option("--min-prevalence", type="double", default=0.0,
              help="Keep ASVs present in at least this fraction of samples [0-1] [default %default]."),
  make_option("--remove-zero-var", type="logical", default=TRUE,
              help="Drop ASVs with zero variance after filtering [default %default]."),

  # SpiecEasi params
  make_option("--method", type="character", default="glasso",
              help="SpiecEasi method: glasso | mb [default %default]."),
  make_option("--lambda-min-ratio", type="double", default=1e-2,
              help="lambda.min.ratio [default %default]."),
  make_option("--nlambda", type="integer", default=20,
              help="Number of lambda values [default %default]."),
  make_option("--rep-num", type="integer", default=50,
              help="pulsar rep.num [default %default]."),
  make_option("--thresh", type="double", default=0.1,
              help="pulsar selection threshold [default %default]."),
  make_option("--ncores", type="integer", default=4,
              help="Number of cores for pulsar [default %default]."),
  make_option("--seed", type="integer", default=10010,
              help="Random seed [default %default]."),

  # Graph construction
  make_option("--edge-threshold", type="double", default=0.1,
              help="Absolute partial correlation cutoff for thresholded graphs [default %default]."),
  make_option("--keep-negative", type="logical", default=TRUE,
              help="Also write a signed (pos/neg) thresholded network [default %default]."),

  # Layout + viz sizes
  make_option("--layout-iters", type="integer", default=1000,
              help="Fruchterman–Reingold iterations [default %default]."),
  make_option("--pdf-width", type="double", default=10,
              help="PDF width (inches) [default %default]."),
  make_option("--pdf-height", type="double", default=10,
              help="PDF height (inches) [default %default]."),
  make_option("--vsize-offset", type="double", default=6,
              help="Vertex size offset added to clr means [default %default]."),
  make_option("--vsize-scale", type="double", default=0.5,
              help="Vertex size scaling factor [default %default]."),

  # Caching / recompute
  make_option("--force-filter", type="logical", default=FALSE,
              help="Recompute filtered counts even if cache exists [default %default]."),
  make_option("--force-spieceasi", type="logical", default=FALSE,
              help="Recompute SpiecEasi even if cache exists [default %default]."),
  make_option("--force-graphs", type="logical", default=FALSE,
              help="Recompute igraph + layouts even if cache exists [default %default].")
)

opt <- parse_args(OptionParser(option_list = opt))

if (is.null(opt$counts) || is.null(opt$outdir)) {
  stop("Please provide --counts and --outdir", call.=FALSE)
}

dir.create(opt$outdir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------- Helpers ---------------------------------------
msg <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

load_if_exists <- function(path) if (file.exists(path)) readRDS(path) else FALSE
save_csv <- function(x, path) { write.csv(x, path, row.names = FALSE); msg("Wrote %s", path) }

safe_clr <- function(x, margin=1) {
  # SpiecEasi::clr assumes compositional data; add small pseudocount if zeros
  x <- as.matrix(x)
  if (any(x == 0)) x <- x + 1
  SpiecEasi::clr(x, margin)
}

get_precision_at_opt <- function(se) {
  # Attempt to extract precision matrix at selected lambda
  if (!is.null(se$est$icov)) {
    idx <- tryCatch(se$select$stars$opt.index, error=function(e) NA_integer_)
    if (is.na(idx) || idx < 1 || idx > length(se$est$icov)) {
      # fallback to last
      idx <- length(se$est$icov)
    }
    return(as.matrix(se$est$icov[[idx]]))
  }
  # Fallback: reconstruct from beta (MB) if available
  if (!is.null(se$est$path)) {
    stop("Precision matrix not found in se$est$icov; implement MB inverse-cov reconstruction if needed.")
  }
  stop("Could not find precision (icov) in SpiecEasi object.")
}

precision_to_partial_cor <- function(Theta) {
  d <- diag(1 / sqrt(diag(Theta)))
  P <- - d %*% Theta %*% d
  diag(P) <- 0
  P
}

adj_from_partial <- function(P, threshold, positive_only = TRUE) {
  A <- P
  if (positive_only) {
    A[A < 0] <- 0
    A[abs(A) < threshold] <- 0
  } else {
    A[abs(A) < threshold] <- 0
  }
  # force symmetry
  A <- (A + t(A))/2
  diag(A) <- 0
  A
}

# ----------------------------- I/O & Filtering -------------------------------
counts_path <- normalizePath(opt$counts)
prefix <- file.path(normalizePath(opt$outdir), opt$prefix)

cache_counts   <- paste0(prefix, "_count_data_filtered.rds")
cache_spiece   <- paste0(prefix, "_spiec_easi.rds")
cache_graph    <- paste0(prefix, "_igraph_main.rds")
cache_layout   <- paste0(prefix, "_layout_nicely.rds")

# Load counts
msg("Reading counts from: %s", counts_path)
raw <- suppressMessages(readr::read_tsv(counts_path, col_types = cols(.default = col_double(), .name_repair="minimal")))

if (!("X1" %in% names(raw)) && !(names(raw)[1] == "")) {
  # readr won't set rownames; first column is IDs
  ids <- colnames(raw)[1]
} else {
  ids <- "X1"
}

asv_ids <- raw[[1]]
mat <- as.matrix(raw[,-1, drop=FALSE])
colnames(mat) <- colnames(raw)[-1]
rownames(mat) <- asv_ids

if (!is.null(opt$strip_suffix_regex)) {
  colnames(mat) <- sub(opt$strip_suffix_regex, "", colnames(mat), perl=TRUE)
}

if (isTRUE(opt$transpose)) {
  mat <- t(mat)
}
# Now rows = samples, cols = ASVs
storage.mode(mat) <- "double"

# Filter (cached)
count_data_filtered <- if (!opt$force_filter) load_if_exists(cache_counts) else FALSE

if (identical(count_data_filtered, FALSE)) {
  msg("Filtering ASVs ...")
  keep <- rep(TRUE, ncol(mat))

  if (opt$min_rel_abund > 0) {
    rs <- rowSums(mat)
    rs[rs == 0] <- 1
    rel <- mat / rs
    max_ra <- apply(rel, 2, max)
    keep <- keep & (max_ra >= opt$min_rel_abund)
  }

  if (opt$min_prevalence > 0) {
    prev <- colSums(mat > 0) / nrow(mat)
    keep <- keep & (prev >= opt$min_prevalence)
  }

  mat_f <- mat[, keep, drop=FALSE]

  if (isTRUE(opt$remove_zero_var) && ncol(mat_f) > 0) {
    v <- apply(mat_f, 2, var)
    mat_f <- mat_f[, v > 0, drop=FALSE]
  }

  count_data_filtered <- mat_f
  saveRDS(count_data_filtered, cache_counts)
  msg("Saved filtered counts: %s  (samples=%d, ASVs=%d)", cache_counts, nrow(count_data_filtered), ncol(count_data_filtered))
} else {
  msg("Loaded filtered counts: %s  (samples=%d, ASVs=%d)", cache_counts, nrow(count_data_filtered), ncol(count_data_filtered))
}

stopifnot(nrow(count_data_filtered) > 1, ncol(count_data_filtered) > 1)

# ----------------------------- SpiecEasi -------------------------------------
se_obj <- if (!opt$force_spieceasi) load_if_exists(cache_spiece) else FALSE

if (identical(se_obj, FALSE)) {
  set.seed(opt$seed)
  msg("Running SpiecEasi (%s) ...", opt$method)
  pargs <- list(rep.num = opt$rep_num, seed = opt$seed, ncores = opt$ncores, thresh = opt$thresh)

  se_obj <- SpiecEasi::spiec.easi(
    count_data_filtered,
    method = opt$method,
    lambda.min.ratio = opt$lambda_min_ratio,
    nlambda = opt$nlambda,
    pulsar.params = pargs
  )
  saveRDS(se_obj, cache_spiece)
  msg("Saved SpiecEasi object: %s", cache_spiece)
} else {
  msg("Loaded SpiecEasi object: %s", cache_spiece)
}

msg("Samples: %d, ASVs: %d", nrow(count_data_filtered), ncol(count_data_filtered))

# ---------------------- Precision / Partial Cor / STARS ----------------------
msg("Extracting precision and partial correlations ...")
Theta <- get_precision_at_opt(se_obj)
stopifnot(nrow(Theta) == ncol(count_data_filtered), ncol(Theta) == ncol(count_data_filtered))

pcor <- precision_to_partial_cor(Theta)
colnames(pcor) <- colnames(count_data_filtered)
rownames(pcor) <- colnames(count_data_filtered)

# Export matrices
save_csv(tibble(ASV_ID = rownames(Theta)) %>% bind_cols(as.data.frame(Theta)), paste0(prefix, "_precision_matrix.csv"))
save_csv(tibble(ASV_ID = rownames(pcor))  %>% bind_cols(as.data.frame(pcor)),  paste0(prefix, "_partial_correlation_matrix.csv"))

# STARS adjacency (binary)
if (!is.null(se_obj$refit$stars)) {
  adj_bin <- as.matrix(se_obj$refit$stars)
  colnames(adj_bin) <- colnames(count_data_filtered)
  rownames(adj_bin) <- colnames(count_data_filtered)
  save_csv(tibble(ASV_ID = rownames(adj_bin)) %>% bind_cols(as.data.frame(adj_bin)),
           paste0(prefix, "_adj_STARS_matrix.csv"))
} else {
  adj_bin <- (abs(pcor) > 0) * 1
}

# ----------------------------- Graphs + Layouts ------------------------------
ig_main <- if (!opt$force_graphs) load_if_exists(cache_graph) else FALSE
am_coord <- if (!opt$force_graphs) load_if_exists(cache_layout) else FALSE

if (identical(ig_main, FALSE) || identical(am_coord, FALSE)) {
  msg("Building graphs ...")
  # Positive-only thresholded
  A_pos_thr <- adj_from_partial(pcor, threshold = opt$edge_threshold, positive_only = TRUE)
  ig_main <- graph_from_adjacency_matrix(A_pos_thr, mode="undirected", weighted=TRUE, diag=FALSE)

  # All positives, no threshold (for reference)
  A_pos_all <- pcor; A_pos_all[A_pos_all < 0] <- 0; A_pos_all <- (A_pos_all + t(A_pos_all))/2; diag(A_pos_all) <- 0
  ig_pos_all <- graph_from_adjacency_matrix(A_pos_all, mode="undirected", weighted=TRUE, diag=FALSE)

  # Signed thresholded (optional)
  if (isTRUE(opt$keep_negative)) {
    A_signed_thr <- adj_from_partial(pcor, threshold = opt$edge_threshold, positive_only = FALSE)
    ig_signed <- graph_from_adjacency_matrix(A_signed_thr, mode="undirected", weighted=TRUE, diag=FALSE)
    E(ig_signed)$color <- ifelse(E(ig_signed)$weight > 0, "red", "blue")
  } else {
    ig_signed <- NULL
  }

  # Vertex sizes from CLR means
  vsize <- rowMeans(safe_clr(count_data_filtered, 1)) + opt$vsize_offset
  vsize <- vsize * opt$vsize_scale
  vsize <- as.numeric(vsize)

  # Layouts
  set.seed(opt$seed)
  layout_nicely_coords <- layout_nicely(ig_main)
  if (!is.matrix(layout_nicely_coords)) layout_nicely_coords <- matrix(layout_nicely_coords, ncol = 2)

  layout_fr_coords <- layout_with_fr(ig_main, niter = opt$layout_iters)
  if (!is.matrix(layout_fr_coords)) layout_fr_coords <- matrix(layout_fr_coords, ncol = 2)

  layout_kk_coords <- layout_with_kk(ig_main)
  if (!is.matrix(layout_kk_coords)) layout_kk_coords <- matrix(layout_kk_coords, ncol = 2)

  layout_drl_coords <- layout_with_drl(ig_main)
  if (!is.matrix(layout_drl_coords)) layout_drl_coords <- matrix(layout_drl_coords, ncol = 2)

  # Cache main objects
  saveRDS(ig_main,  cache_graph);  msg("Saved igraph:  %s", cache_graph)
  saveRDS(layout_nicely_coords, cache_layout); msg("Saved layout: %s", cache_layout)

  # Multipage PDF of layouts
  pdf(file = paste0(prefix, "_multipage_layouts.pdf"), width = opt$pdf_width, height = opt$pdf_height)
  plot(ig_main, layout = layout_nicely_coords, vertex.size = vsize, vertex.label = NA,
       main = "layout_nicely")
  plot(ig_main, layout = layout_fr_coords, vertex.size = vsize, vertex.label = NA,
       main = sprintf("Fruchterman-Reingold (niter=%d)", opt$layout_iters))
  plot(ig_main, layout = layout_kk_coords, vertex.size = vsize, vertex.label = NA,
       main = "Kamada-Kawai")
  plot(ig_main, layout = layout_drl_coords, vertex.size = vsize, vertex.label = NA,
       main = "DRL")
  dev.off()
  msg("Wrote %s_multipage_layouts.pdf", prefix)

  # Write graphs (GraphML + GML)
  write_graph(ig_main,   file = paste0(prefix, "_network_pos_thr.graphml"), format = "graphml")
  write_graph(ig_main,   file = paste0(prefix, "_network_pos_thr.gml"),     format = "gml")
  write_graph(ig_pos_all,file = paste0(prefix, "_network_pos_all.graphml"), format = "graphml")
  write_graph(ig_pos_all,file = paste0(prefix, "_network_pos_all.gml"),     format = "gml")
  if (!is.null(ig_signed)) {
    write_graph(ig_signed, file = paste0(prefix, "_network_signed_thr.graphml"), format = "graphml")
    write_graph(ig_signed, file = paste0(prefix, "_network_signed_thr.gml"),     format = "gml")
  }

  # Export weighted adjacency matrices
  save_csv(tibble(ASV_ID = colnames(A_pos_thr)) %>% bind_cols(as.data.frame(A_pos_thr)),
           paste0(prefix, "_adj_weighted_pos_thr.csv"))
  save_csv(tibble(ASV_ID = colnames(A_pos_all)) %>% bind_cols(as.data.frame(A_pos_all)),
           paste0(prefix, "_adj_weighted_pos_all.csv"))
  if (!is.null(ig_signed)) {
    save_csv(tibble(ASV_ID = colnames(A_signed_thr)) %>% bind_cols(as.data.frame(A_signed_thr)),
             paste0(prefix, "_adj_weighted_signed_thr.csv"))
  }

} else {
  msg("Loaded igraph + layout from cache.")
}

# ----------------------------- Edge lists ------------------------------------
# Ensure names (graph_from_adjacency_matrix uses dimnames when present)
if (is.null(V(ig_main)$name)) {
  V(ig_main)$name <- colnames(count_data_filtered)
}

edges_df <- as_edgelist(ig_main) %>% as.data.frame()
colnames(edges_df) <- c("Taxon1", "Taxon2")
edges_df$Weight <- E(ig_main)$weight
save_csv(edges_df, paste0(prefix, "_edge_list.csv"))

# -------------------------- Node centralities --------------------------------
node_degree      <- igraph::degree(ig_main)
node_betweenness <- igraph::betweenness(ig_main)
node_closeness   <- igraph::closeness(ig_main)
node_eigen       <- igraph::eigen_centrality(ig_main)$vector
node_ids         <- paste0("n", seq_along(V(ig_main)) - 1)

node_features <- data.frame(
  GraphML_ID   = node_ids,
  Taxon        = V(ig_main)$name,
  Degree       = node_degree,
  Betweenness  = node_betweenness,
  Closeness    = node_closeness,
  EigenCentral = node_eigen
)
save_csv(node_features, paste0(prefix, "_node_features.csv"))

msg("Done. Outputs under: %s", normalizePath(opt$outdir))
