#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(readr)
  library(indicspecies)
  library(permute)  # for how()
})

# ---------- CLI ----------
option_list <- list(
  make_option("--data-wide",  type="character", help="ASV count table (rows=ASVs, cols=samples). TSV."),
  make_option("--data-long",  type="character", help="Long-format data with metadata (for extracting sample metadata)."),
  # Backward-compatible aliases used by current SPARK invocations.
  make_option("--asv",        type="character", help="Alias of --data-wide (legacy)."),
  make_option("--meta",       type="character", help="Alias of --data-long (legacy)."),
  make_option("--sample-col", type="character", default="sample",
              help="Column in metadata matching sample IDs [default: %default]"),
  make_option("--patient-col", type="character", default="Participant_ID",
              help="Column for patient IDs (for blocked permutations) [default: %default]"),
  make_option("--group-cols", type="character", default="status,type_group",
              help="Comma-separated grouping columns to analyze [default: %default]"),
  make_option("--blocked-cols", type="character", default="type_group",
              help="Comma-separated grouping columns requiring blocked permutations [default: %default]"),
  make_option("--status-extra-no-contralateral", type="logical", default=TRUE,
              help="For Lung Brush status analysis, add extra run excluding contralateral cancer samples [default: %default]"),
  make_option("--contralateral-col", type="character", default="lung_status",
              help="Metadata column identifying contralateral samples [default: %default]"),
  make_option("--cancer-site-col", type="character", default="Cancer_Site",
              help="Metadata column giving cancer side for cancer patients (used to derive contralateral if needed) [default: %default]"),
  make_option("--lung-side-col", type="character", default="lung_code",
              help="Metadata column giving sample lung side (used to derive contralateral if needed) [default: %default]"),
  make_option("--contralateral-value", type="character", default="Contralateral",
              help="Value in --contralateral-col marking contralateral samples [default: %default]"),
  make_option("--cancer-label", type="character", default="Cancer",
              help="Label in status column for cancer samples [default: %default]"),
  make_option("--lung-brush-label", type="character", default="Lung Brush",
              help="Value in type_group identifying Lung Brush samples [default: %default]"),
  make_option("--perms",      type="integer",   default=999,
              help="Permutations for multipatt [default: %default]"),
  make_option("--min-n",      type="integer",   default=2,
              help="Minimum samples per group to keep [default: %default]"),
  make_option("--outdir",     type="character",
              help="Output directory (will create '<outdir>/indicspecies').")
)

parser <- OptionParser(
  usage = "%prog --data-wide ASV_wide.tsv --data-long ASV_long.tsv --sample-col sample --group-cols status,type_group --outdir out_dir",
  description = "Run indicspecies multipatt on ASV + metadata tables.",
  option_list = option_list
)

opt <- parse_args(parser)

# Support legacy flags used by existing SPARK workflows.
if (is.null(opt$`data-wide`) && !is.null(opt$asv)) {
  opt$`data-wide` <- opt$asv
}
if (is.null(opt$`data-long`) && !is.null(opt$meta)) {
  opt$`data-long` <- opt$meta
}

# Enforce required options
required <- c("data-wide", "data-long", "outdir")
missing <- required[sapply(required, function(x) is.null(opt[[x]]))]
if (length(missing)) {
  cat("Missing required option(s):", paste(missing, collapse=", "), "\n\n", file=stderr())
  print_help(parser)
  quit(status=2)
}

outdir <- file.path(opt$outdir, "indicspecies")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# ---------- IO ----------
message("Reading long format data: ", opt$`data-long`)
long_df <- read_tsv(opt$`data-long`, show_col_types = FALSE)

# Extract metadata (unique sample-level records)
required_cols <- c(opt$`sample-col`, opt$`patient-col`)
group_cols <- strsplit(opt$`group-cols`, ",", fixed = TRUE)[[1]] %>% trimws()
if ("status" %in% group_cols && !("type_group" %in% group_cols)) {
  # Needed for status stratification by sample type.
  group_cols <- c(group_cols, "type_group")
}
optional_cols <- c()
derive_contralateral_from_sides <- FALSE
if (isTRUE(opt$`status-extra-no-contralateral`) && ("status" %in% group_cols)) {
  if (opt$`contralateral-col` %in% names(long_df)) {
    optional_cols <- c(optional_cols, opt$`contralateral-col`)
  } else if (all(c(opt$`cancer-site-col`, opt$`lung-side-col`) %in% names(long_df))) {
    optional_cols <- c(optional_cols, opt$`cancer-site-col`, opt$`lung-side-col`)
    derive_contralateral_from_sides <- TRUE
    message(
      "Column '", opt$`contralateral-col`, "' not found; deriving contralateral status from ",
      opt$`cancer-site-col`, " and ", opt$`lung-side-col`, "."
    )
  } else {
    warning(
      "Optional column '", opt$`contralateral-col`, "' not found and cannot derive from ",
      opt$`cancer-site-col`, "/", opt$`lung-side-col`,
      "; extra no-contralateral status analysis will be skipped."
    )
  }
}
all_cols <- unique(c(required_cols, group_cols, optional_cols))

# Check all required columns exist
missing_cols <- setdiff(all_cols, names(long_df))
if (length(missing_cols) > 0) {
  stop("Missing columns in long data: ", paste(missing_cols, collapse = ", "))
}

meta <- long_df %>%
  select(all_of(all_cols)) %>%
  distinct() %>%
  tibble::column_to_rownames(opt$`sample-col`)

effective_contralateral_col <- opt$`contralateral-col`
if (derive_contralateral_from_sides) {
  # Derive lung status:
  # - Cancer samples: TumorSide if lung side matches Cancer_Site, else Contralateral
  # - Non-Cancer samples: Healthy
  meta <- meta %>%
    mutate(
      .derived_lung_status = case_when(
        as.character(status) == as.character(opt$`cancer-label`) &
          !is.na(.data[[opt$`cancer-site-col`]]) &
          !is.na(.data[[opt$`lung-side-col`]]) &
          toupper(substr(as.character(.data[[opt$`cancer-site-col`]]), 1, 1)) ==
            toupper(substr(as.character(.data[[opt$`lung-side-col`]]), 1, 1)) ~ "TumorSide",
        as.character(status) == as.character(opt$`cancer-label`) &
          !is.na(.data[[opt$`cancer-site-col`]]) &
          !is.na(.data[[opt$`lung-side-col`]]) ~ "Contralateral",
        TRUE ~ "Healthy"
      )
    )
  effective_contralateral_col <- ".derived_lung_status"
}

# Read wide format ASV table
message("Reading wide format ASV table: ", opt$`data-wide`)
asv <- read_tsv(opt$`data-wide`, show_col_types = FALSE)
stopifnot(ncol(asv) >= 2)
# Expect first column to be ASV IDs
asv <- asv %>% rename(ASV = 1)
asv_mat <- asv %>% column_to_rownames("ASV") %>% as.matrix()
mode(asv_mat) <- "numeric"

# ---------- align samples ----------
common <- intersect(colnames(asv_mat), rownames(meta))
if (length(common) == 0) stop("No overlapping samples between ASV table columns and metadata rows.")
asv_mat <- asv_mat[, common, drop = FALSE]
meta    <- meta[common, , drop = FALSE]

message("ASV table dimensions: ", paste(dim(asv_mat), collapse = " x "))
message("Metadata dimensions: ", paste(dim(meta), collapse = " x "))

# ---------- helpers ----------
run_indics <- function(X_samples_by_features, grouping, perms = 999, duleg = FALSE, patient_blocks = NULL) {
  # indicspecies::multipatt expects samples in rows, species/features in columns
  # If patient_blocks provided, use blocked permutations (for within-patient comparisons)
  if (!is.null(patient_blocks)) {
    message("  Using blocked permutations (patient as blocking factor)")
    ctrl <- how(nperm = perms, blocks = patient_blocks)
  } else {
    ctrl <- how(nperm = perms)
  }
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
  out_results <- file.path(outdir, paste0(base, "_results.tsv"))
  out_summary <- file.path(outdir, paste0(base, "_summary.tsv"))
  readr::write_tsv(df_sign_only, out_results)
  readr::write_tsv(df_full, out_summary)

  # Backward-compatible alias for historical DULEG naming: *_results_DULEG.tsv
  if (grepl("_DULEG$", base)) {
    base_legacy <- sub("_DULEG$", "", base)
    out_results_legacy <- file.path(outdir, paste0(base_legacy, "_results_DULEG.tsv"))
    out_summary_legacy <- file.path(outdir, paste0(base_legacy, "_summary_DULEG.tsv"))
    readr::write_tsv(df_sign_only, out_results_legacy)
    readr::write_tsv(df_full, out_summary_legacy)
  }
}

# Aggregate sample-level matrix to one row per patient (sum counts).
aggregate_to_patient <- function(X_samples_by_features, patient_ids) {
  stopifnot(nrow(X_samples_by_features) == length(patient_ids))
  df <- as.data.frame(X_samples_by_features)
  df$patient_id___ <- as.character(patient_ids)
  out <- df %>%
    group_by(patient_id___) %>%
    summarise(across(everything(), sum), .groups = "drop")
  mat <- out %>%
    tibble::column_to_rownames("patient_id___") %>%
    as.matrix()
  mode(mat) <- "numeric"
  mat
}

# ---------- main loop over grouping columns ----------
group_cols <- strsplit(opt$`group-cols`, ",", fixed = TRUE)[[1]] |> trimws() |> discard(~ .x == "")
blocked_cols <- strsplit(opt$`blocked-cols`, ",", fixed = TRUE)[[1]] |> trimws() |> discard(~ .x == "")

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
  meta_keep <- meta[keep_idx, , drop = FALSE]

  # enforce min-n per group
  tab <- table(grouping)
  small <- names(tab[tab < opt$`min-n`])
  if (length(small) > 0) {
    message("Dropping groups in '", gcol, "' with < ", opt$`min-n`, " samples: ",
            paste(small, collapse = ", "))
    keep_idx2 <- !(grouping %in% small)
    grouping <- droplevels(grouping[keep_idx2])
    X <- X[keep_idx2, , drop = FALSE]
    meta_keep <- meta_keep[keep_idx2, , drop = FALSE]
  }

  if (length(unique(grouping)) < 2) {
    warning("Grouping column '", gcol, "' has <2 groups after filtering; skipping.")
    next
  }

  # Status is analyzed within each sample type (between-patient per site).
  if (gcol == "status") {
    if (!("type_group" %in% colnames(meta_keep))) {
      warning("Column 'type_group' not found; skipping stratified status analyses.")
      next
    }
    if (!(opt$`patient-col` %in% colnames(meta_keep))) {
      warning("Patient column '", opt$`patient-col`, "' not found; cannot run status analyses.")
      next
    }

    # Collect per-site results into legacy pooled status outputs expected by SPARK readers.
    pooled_status_sign <- list()
    pooled_status_full <- list()
    pooled_status_sign_duleg <- list()
    pooled_status_full_duleg <- list()

    site_levels <- unique(as.character(meta_keep$type_group))
    site_levels <- site_levels[!is.na(site_levels)]

    for (site in site_levels) {
      site_mask <- as.character(meta_keep$type_group) == site
      X_site <- X[site_mask, , drop = FALSE]
      meta_site <- meta_keep[site_mask, , drop = FALSE]
      grouping_site <- droplevels(as.factor(meta_site[[gcol]]))

      if (length(unique(grouping_site)) < 2) {
        warning("Skipping status ISA for type_group='", site, "' (<2 status groups).")
        next
      }

      tab_site <- table(grouping_site)
      small_site <- names(tab_site[tab_site < opt$`min-n`])
      if (length(small_site) > 0) {
        keep_site <- !(grouping_site %in% small_site)
        grouping_site <- droplevels(grouping_site[keep_site])
        X_site <- X_site[keep_site, , drop = FALSE]
        meta_site <- meta_site[keep_site, , drop = FALSE]
      }

      if (length(unique(grouping_site)) < 2) {
        warning("Skipping status ISA for type_group='", site, "' after min-n filtering.")
        next
      }

      X_pat <- aggregate_to_patient(X_site, meta_site[[opt$`patient-col`]])
      status_map <- meta_site %>%
        transmute(
          patient_id___ = as.character(.data[[opt$`patient-col`]]),
          status___ = as.character(.data[[gcol]])
        ) %>%
        distinct() %>%
        group_by(patient_id___) %>%
        summarise(status___ = first(status___), .groups = "drop")
      status_vec <- status_map$status___[match(rownames(X_pat), status_map$patient_id___)]
      grouping_pat <- droplevels(factor(status_vec))

      if (length(unique(grouping_pat)) < 2) {
        warning("Skipping status ISA for type_group='", site, "' after patient aggregation.")
        next
      }

      message("Running multipatt for 'status' within type_group='", site,
              "' (single groups, duleg=FALSE) …")
      fit1 <- run_indics(X_pat, grouping_pat, perms = opt$perms, duleg = FALSE, patient_blocks = NULL)
      res1_sign <- as.data.frame(fit1$sign) %>% rownames_to_column("ASV")
      res1_full <- summarize_multipatt(fit1)
      site_slug <- gsub("[^A-Za-z0-9]+", "_", site)
      write_tables(res1_sign, res1_full, paste0("status_", site_slug, "_indicator_species"))

      pooled_status_sign[[length(pooled_status_sign) + 1]] <- res1_sign %>% mutate(type_group = site)
      pooled_status_full[[length(pooled_status_full) + 1]] <- res1_full %>% mutate(type_group = site)

      message("Running multipatt for 'status' within type_group='", site,
              "' (combos allowed, duleg=TRUE) …")
      fit2 <- run_indics(X_pat, grouping_pat, perms = opt$perms, duleg = TRUE, patient_blocks = NULL)
      res2_sign <- as.data.frame(fit2$sign) %>% rownames_to_column("ASV")
      res2_full <- summarize_multipatt(fit2)
      write_tables(res2_sign, res2_full, paste0("status_", site_slug, "_indicator_species_DULEG"))

      pooled_status_sign_duleg[[length(pooled_status_sign_duleg) + 1]] <- res2_sign %>% mutate(type_group = site)
      pooled_status_full_duleg[[length(pooled_status_full_duleg) + 1]] <- res2_full %>% mutate(type_group = site)

      # Extra status analysis for Lung Brush: remove contralateral samples from cancer patients.
      if (isTRUE(opt$`status-extra-no-contralateral`) &&
          identical(as.character(site), as.character(opt$`lung-brush-label`))) {
        if (!(effective_contralateral_col %in% colnames(meta_site))) {
          warning("Skipping extra Lung Brush no-contralateral analysis: column '",
                  effective_contralateral_col, "' not found.")
        } else {
          is_cancer <- as.character(meta_site[[gcol]]) == as.character(opt$`cancer-label`)
          is_contralateral <- as.character(meta_site[[effective_contralateral_col]]) ==
            as.character(opt$`contralateral-value`)
          keep_no_contra <- !(is_cancer & is_contralateral)

          n_removed <- sum(!keep_no_contra, na.rm = TRUE)
          message("Lung Brush extra status analysis: excluding ", n_removed,
                  " contralateral cancer sample(s) using ",
                  effective_contralateral_col, " == '", opt$`contralateral-value`, "'.")

          X_site_nc <- X_site[keep_no_contra, , drop = FALSE]
          meta_site_nc <- meta_site[keep_no_contra, , drop = FALSE]
          grouping_site_nc <- droplevels(as.factor(meta_site_nc[[gcol]]))

          if (length(unique(grouping_site_nc)) < 2) {
            warning("Skipping Lung Brush no-contralateral status ISA (<2 status groups after filtering).")
          } else {
            tab_site_nc <- table(grouping_site_nc)
            small_site_nc <- names(tab_site_nc[tab_site_nc < opt$`min-n`])
            if (length(small_site_nc) > 0) {
              keep_site_nc <- !(grouping_site_nc %in% small_site_nc)
              grouping_site_nc <- droplevels(grouping_site_nc[keep_site_nc])
              X_site_nc <- X_site_nc[keep_site_nc, , drop = FALSE]
              meta_site_nc <- meta_site_nc[keep_site_nc, , drop = FALSE]
            }

            if (length(unique(grouping_site_nc)) < 2) {
              warning("Skipping Lung Brush no-contralateral status ISA after min-n filtering.")
            } else {
              X_pat_nc <- aggregate_to_patient(X_site_nc, meta_site_nc[[opt$`patient-col`]])
              status_map_nc <- meta_site_nc %>%
                transmute(
                  patient_id___ = as.character(.data[[opt$`patient-col`]]),
                  status___ = as.character(.data[[gcol]])
                ) %>%
                distinct() %>%
                group_by(patient_id___) %>%
                summarise(status___ = first(status___), .groups = "drop")
              status_vec_nc <- status_map_nc$status___[
                match(rownames(X_pat_nc), status_map_nc$patient_id___)
              ]
              grouping_pat_nc <- droplevels(factor(status_vec_nc))

              if (length(unique(grouping_pat_nc)) < 2) {
                warning("Skipping Lung Brush no-contralateral status ISA after patient aggregation.")
              } else {
                message("Running multipatt for 'status' within type_group='", site,
                        "' excluding contralateral cancer samples (duleg=FALSE) …")
                fit1_nc <- run_indics(X_pat_nc, grouping_pat_nc, perms = opt$perms,
                                      duleg = FALSE, patient_blocks = NULL)
                res1_nc_sign <- as.data.frame(fit1_nc$sign) %>% rownames_to_column("ASV")
                res1_nc_full <- summarize_multipatt(fit1_nc)
                write_tables(
                  res1_nc_sign, res1_nc_full,
                  paste0("status_", site_slug, "_no_contralateral_indicator_species")
                )

                message("Running multipatt for 'status' within type_group='", site,
                        "' excluding contralateral cancer samples (duleg=TRUE) …")
                fit2_nc <- run_indics(X_pat_nc, grouping_pat_nc, perms = opt$perms,
                                      duleg = TRUE, patient_blocks = NULL)
                res2_nc_sign <- as.data.frame(fit2_nc$sign) %>% rownames_to_column("ASV")
                res2_nc_full <- summarize_multipatt(fit2_nc)
                write_tables(
                  res2_nc_sign, res2_nc_full,
                  paste0("status_", site_slug, "_no_contralateral_indicator_species_DULEG")
                )
              }
            }
          }
        }
      }
    }

    # Emit legacy pooled files expected by existing SPARK plotting workflows.
    if (length(pooled_status_sign) > 0) {
      write_tables(bind_rows(pooled_status_sign), bind_rows(pooled_status_full), "status_indicator_species")
    }
    if (length(pooled_status_sign_duleg) > 0) {
      write_tables(bind_rows(pooled_status_sign_duleg), bind_rows(pooled_status_full_duleg), "status_indicator_species_DULEG")
    }

    next
  }

  # Determine if this grouping requires blocked permutations (e.g., type_group)
  use_blocking <- gcol %in% blocked_cols
  patient_blocks <- NULL
  if (use_blocking) {
    if (opt$`patient-col` %in% colnames(meta_keep)) {
      patient_blocks <- droplevels(factor(meta_keep[[opt$`patient-col`]]))
      message("Grouping '", gcol, "' uses BLOCKED permutations (within-patient design)")
    } else {
      warning("Patient column '", opt$`patient-col`, "' not found; using standard permutations")
    }
  }

  message("Running multipatt for '", gcol, "' (single groups, duleg=FALSE) …")
  fit1 <- run_indics(X, grouping, perms = opt$perms, duleg = FALSE, patient_blocks = patient_blocks)
  res1_sign <- as.data.frame(fit1$sign) %>% rownames_to_column("ASV")
  res1_full <- summarize_multipatt(fit1)
  write_tables(res1_sign, res1_full, paste0(gcol, "_indicator_species"))

  message("Running multipatt for '", gcol, "' (combos allowed, duleg=TRUE) …")
  fit2 <- run_indics(X, grouping, perms = opt$perms, duleg = TRUE, patient_blocks = patient_blocks)
  res2_sign <- as.data.frame(fit2$sign) %>% rownames_to_column("ASV")
  res2_full <- summarize_multipatt(fit2)
  write_tables(res2_sign, res2_full, paste0(gcol, "_indicator_species_DULEG"))

  # Additional legacy aliases used by older SPARK scripts.
  if (gcol == "type_group") {
    write_tables(res1_sign, res1_full, "Type_Group_indicator_species")
    write_tables(res2_sign, res2_full, "Type_Group_indicator_species_DULEG")
  }
}

message("Done. Results in: ", outdir)
