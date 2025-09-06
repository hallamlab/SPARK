#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(readr)
  library(indicspecies)
  # 'how()' comes from {permute}; indicspecies imports it but load explicitly for clarity
  library(permute)
})

# ---------- CLI ----------
opt <- OptionParser(
  usage = "%prog --asv ASV_final.micro.tsv --meta metadata.tsv --sample-col sample --group-cols status,type_group --outdir out_dir",
  option_list = list(
    make_option("--asv",        type = "character", help = "ASV count table (rows=ASVs, cols=samples). TSV."),
    make_option("--meta",       type = "character", help = "Sample metadata TSV."),
    make_option("--sample-col", type = "character", default = "sample", help = "Column in metadata matching sample IDs (default: %default)."),
    make_option("--group-cols", type = "character", default = "status,type_group", help = "Comma-separated grouping columns to analyze."),
    make_option("--perms",      type = "integer",   default = 999, help = "Permutations for multipatt (default: %default)."),
    make_option("--min-n",      type = "integer",   default = 2, help = "Minimum samples per group to keep (default: %default)."),
    make_option("--outdir",     type = "character", required = TRUE, help = "Output directory.")
  )
) |> parse_args()

stopifnot(!is.null(opt$asv), !is.null(opt$meta))

outdir <- file.path(opt$outdir, "indicspecies")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---------- IO ----------
message("Reading ASV table: ", opt$asv)
asv <- read_tsv(opt$asv, show_col_types = FALSE)
stopifnot(ncol(asv) >= 2)
# Expect first column to be ASV IDs
asv <- asv %>% rename(ASV = 1)
asv_mat <- asv %>% column_to_rownames("ASV") %>% as.matrix()
mode(asv_mat) <- "numeric"

message("Reading metadata: ", opt$meta)
meta <- read_tsv(opt$meta, show_col_types = FALSE)
if (!(opt$`sample-col` %in% names(meta))) {
  stop("Sample column '", opt$`sample-col`, "' not found in metadata. Available: ",
       paste(names(meta), collapse = ", "))
}

# Make sample IDs the rownames (dedupe safely, keep first occurrence)
meta <- meta %>%
  mutate(.row_id__ = row_number()) %>%
  distinct(.data[[opt$`sample-col`]], .keep_all = TRUE) %>%
  tibble::column_to_rownames(opt$`sample-col`) %>%
  select(-.row_id__)

# ---------- align samples ----------
common <- intersect(colnames(asv_mat), rownames(meta))
if (length(common) == 0) stop("No overlapping samples between ASV table columns and metadata rows.")
asv_mat <- asv_mat[, common, drop = FALSE]
meta    <- meta[common, , drop = FALSE]

message("ASV table dimensions: ", paste(dim(asv_mat), collapse = " x "))
message("Metadata dimensions: ", paste(dim(meta), collapse = " x "))

# ---------- helpers ----------
run_indics <- function(X_samples_by_features, grouping, perms = 999, duleg = FALSE) {
  # indicspecies::multipatt expects samples in rows, species/features in columns
  ctrl <- how(nperm = perms)
  suppressWarnings({
    multipatt(x = X_samples_by_features, cluster = grouping, duleg = duleg, control = ctrl)
  })
}

summarize_multipatt <- function(fit) {
  # Build a tidy data.frame with sign + A + B + q-values if p.value present
  sign_df <- as.data.frame(fit$sign)
  sign_df <- sign_df %>%
    rownames_to_column("ASV")

  A_df <- as.data.frame(fit$A) %>% rownames_to_column("ASV")
  B_df <- as.data.frame(fit$B) %>% rownames_to_column("ASV")

  out <- sign_df %>%
    left_join(A_df, by = "ASV", suffix = c("", ".A")) %>%
    left_join(B_df, by = "ASV", suffix = c("", ".B"))

  # If p.value present, add FDR (q) and significance flag
  if ("p.value" %in% names(out)) {
    out <- out %>%
      mutate(q.value = p.adjust(.data[["p.value"]], method = "fdr"),
             significant = q.value < 0.05)
  }
  out
}

write_tables <- function(df_sign_only, df_full, base) {
  readr::write_tsv(df_sign_only, file.path(outdir, paste0(base, "_results.tsv")))
  readr::write_tsv(df_full,      file.path(outdir, paste0(base, "_summary.tsv")))
}

# ---------- main loop over grouping columns ----------
group_cols <- strsplit(opt$`group-cols`, ",", fixed = TRUE)[[1]] |> trimws() |> discard(~ .x == "")

for (gcol in group_cols) {
  if (!(gcol %in% colnames(meta))) {
    warning("Skipping grouping column '", gcol, "' (not found in metadata).")
    next
  }
  grouping <- meta[[gcol]] |> as.factor()

  # Drop NAs and small groups
  keep_idx <- !is.na(grouping)
  grouping <- droplevels(grouping[keep_idx])
  X <- t(asv_mat[, keep_idx, drop = FALSE]) # samples x ASVs

  # enforce min-n per group
  tab <- table(grouping)
  small <- names(tab[tab < opt$`min-n`])
  if (length(small) > 0) {
    message("Dropping groups in '", gcol, "' with < ", opt$`min-n`, " samples: ",
            paste(small, collapse = ", "))
    keep_idx2 <- !(grouping %in% small)
    grouping <- droplevels(grouping[keep_idx2])
    X <- X[keep_idx2, , drop = FALSE]
  }

  if (length(unique(grouping)) < 2) {
    warning("Grouping column '", gcol, "' has <2 groups after filtering; skipping.")
    next
  }

  message("Running multipatt for '", gcol, "' (single groups, duleg=FALSE) …")
  fit1 <- run_indics(X, grouping, perms = opt$perms, duleg = FALSE)
  res1_sign <- as.data.frame(fit1$sign) %>% rownames_to_column("ASV")
  res1_full <- summarize_multipatt(fit1)
  write_tables(res1_sign, res1_full, paste0(gcol, "_indicator_species"))

  message("Running multipatt for '", gcol, "' (combos allowed, duleg=TRUE) …")
  fit2 <- run_indics(X, grouping, perms = opt$perms, duleg = TRUE)
  res2_sign <- as.data.frame(fit2$sign) %>% rownames_to_column("ASV")
  res2_full <- summarize_multipatt(fit2)
  write_tables(res2_sign, res2_full, paste0(gcol, "_indicator_species_DULEG"))
}

message("Done. Results in: ", outdir)
