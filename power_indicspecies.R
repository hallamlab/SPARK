#!/usr/bin/env Rscript
#
# power_indicspecies.R
#
# Indicator Species Analysis (ISA) power analysis for SPARK workflow.
# Matches run_indicspecies.R but adds patient-level bootstrap for power analysis.
# Properly handles hierarchical structure: multiple samples per patient.
#
# Key features:
# 1. Bootstrap patients (not samples) to preserve within-patient correlation
# 2. Assign unique bootstrap IDs to handle duplicate patients in permutations
# 3. Aggregate to patient level before running multipatt
# 4. Test multiple spike-in scenarios (Null, Weak, Moderate, Strong)
# 5. Matches run_indicspecies.R grouping approach (--group-cols)
# 6. Tests both duleg=FALSE and duleg=TRUE (like run_indicspecies.R)
#

suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(readr)
  library(indicspecies)
  library(permute)
})

# ============================================================================
# Command line options
# ============================================================================

option_list <- list(
  make_option("--data-long", type="character",
              help="Long format data file (ASV_master_long.tsv)"),
  make_option("--data-wide", type="character",
              help="Wide count matrix (ASV_master_count_wide.tsv)"),
  make_option("--sample-col", type="character", default="lmp_id",
              help="Column in long data for sample IDs [default: %default]"),
  make_option("--patient-col", type="character", default="Participant_ID",
              help="Column for patient IDs [default: %default]"),
  make_option("--group-cols", type="character", default="status,type_group",
              help="Comma-separated grouping columns to analyze [default: %default]"),
  make_option("--status-sites", type="character", default="",
              help="Optional comma-separated type_group levels for status power (default: all sites)"),
  make_option("--status-extra-no-contralateral", type="logical", default=TRUE,
              help="For Lung Brush status power, exclude contralateral cancer samples [default: %default]"),
  make_option("--contralateral-col", type="character", default="lung_status",
              help="Metadata column identifying contralateral samples [default: %default]"),
  make_option("--cancer-site-col", type="character", default="Cancer_Site",
              help="Metadata column giving cancer side for cancer patients (used to derive contralateral if needed) [default: %default]"),
  make_option("--lung-side-col", type="character", default="lung_code",
              help="Metadata column giving sample lung side (used to derive contralateral if needed) [default: %default]"),
  make_option("--contralateral-value", type="character", default="Contralateral",
              help="Value in contralateral column marking contralateral samples [default: %default]"),
  make_option("--cancer-label", type="character", default="Cancer",
              help="Label in status column for cancer samples [default: %default]"),
  make_option("--lung-brush-label", type="character", default="Lung Brush",
              help="Value in type_group identifying Lung Brush samples [default: %default]"),
  make_option("--sample-sizes", type="character", default="6,8,10,15,20,25,30",
              help="Comma-separated sample sizes to test [default: %default]"),
  make_option("--transform", type="character", default="none",
              help="Abundance transform before multipatt: none|rclr [default: %default]"),
  make_option("--n-simulations", type="integer", default=1000,
              help="Number of bootstrap simulations [default: %default]"),
  make_option("--perms", type="integer", default=199,
              help="Permutations for multipatt [default: %default]"),
  make_option("--alpha", type="double", default=0.05,
              help="Significance threshold [default: %default]"),
  make_option("--seed", type="integer", default=42,
              help="Random seed [default: %default]"),
  make_option("--test-duleg", type="logical", default=TRUE,
              help="Test both duleg=FALSE and duleg=TRUE [default: %default]"),
  make_option("--outdir", type="character",
              help="Output directory.")
)

parser <- OptionParser(
  usage = "%prog --data-long ASV_master_long.tsv --data-wide ASV_master_count_wide.tsv --group-cols status,type_group --outdir out_dir",
  description = "ISA power analysis with patient-level bootstrap (matches run_indicspecies.R).",
  option_list = option_list
)

args <- parse_args(parser)

# Enforce required options
required <- c("data-long", "data-wide", "outdir")
missing <- required[sapply(required, function(x) is.null(args[[x]]))]
if (length(missing)) {
  cat("Missing required option(s):", paste(missing, collapse=", "), "\n\n", file=stderr())
  print_help(parser)
  quit(status=2)
}

# Handle defaults for optional parameters
if (is.null(args$seed)) args$seed <- 42
if (is.null(args$`n-simulations`)) args$`n-simulations` <- 1000
if (is.null(args$perms)) args$perms <- 199
if (is.null(args$alpha)) args$alpha <- 0.05
if (is.null(args$`test-duleg`)) args$`test-duleg` <- TRUE
if (is.null(args$`sample-sizes`)) args$`sample-sizes` <- "6,8,10,15,20,25,30"
if (is.null(args$`group-cols`)) args$`group-cols` <- "status,type_group"
if (is.null(args$`status-sites`)) args$`status-sites` <- ""

if (!(args$transform %in% c("none", "rclr"))) {
  stop("--transform must be one of: none, rclr")
}

set.seed(args$seed)

cat("================================================================\n")
cat("Indicator Species Analysis (ISA) Power Analysis - SPARK\n")
cat("================================================================\n")
cat("Patient-level bootstrap preserving hierarchical structure\n")
cat("================================================================\n\n")

# ============================================================================
# Load and prepare data
# ============================================================================

cat("Loading data...\n")
cat("  Long format:", args$`data-long`, "\n")
cat("  Wide format:", args$`data-wide`, "\n\n")

# Read long format data to get metadata
message("Reading long format data: ", args$`data-long`)
long_df <- read_tsv(args$`data-long`, show_col_types = FALSE)

# Extract metadata (unique sample-level records)
required_cols <- c(args$`sample-col`, args$`patient-col`)
group_cols_vec <- strsplit(args$`group-cols`, ",", fixed = TRUE)[[1]] %>% trimws()
if ("status" %in% group_cols_vec && !("type_group" %in% group_cols_vec)) {
  # Needed for status stratification by sample type.
  group_cols_vec <- c(group_cols_vec, "type_group")
}
optional_cols <- c()
derive_contralateral_from_sides <- FALSE
if (isTRUE(args$`status-extra-no-contralateral`) && ("status" %in% group_cols_vec)) {
  if (args$`contralateral-col` %in% names(long_df)) {
    optional_cols <- c(optional_cols, args$`contralateral-col`)
  } else if (all(c(args$`cancer-site-col`, args$`lung-side-col`) %in% names(long_df))) {
    optional_cols <- c(optional_cols, args$`cancer-site-col`, args$`lung-side-col`)
    derive_contralateral_from_sides <- TRUE
    message(
      "Column '", args$`contralateral-col`, "' not found; deriving contralateral status from ",
      args$`cancer-site-col`, " and ", args$`lung-side-col`, "."
    )
  } else {
    warning(
      "Optional column '", args$`contralateral-col`, "' not found and cannot derive from ",
      args$`cancer-site-col`, "/", args$`lung-side-col`,
      "; Lung Brush no-contralateral power filter will be skipped."
    )
  }
}
all_cols <- unique(c(required_cols, group_cols_vec, optional_cols))

# Check all required columns exist
missing_cols <- setdiff(all_cols, names(long_df))
if (length(missing_cols) > 0) {
  stop("Missing columns in long data: ", paste(missing_cols, collapse = ", "))
}

meta <- long_df %>%
  select(all_of(all_cols)) %>%
  distinct() %>%
  tibble::column_to_rownames(args$`sample-col`)

effective_contralateral_col <- args$`contralateral-col`
if (derive_contralateral_from_sides) {
  # Derive lung status for filtering contralateral samples in cancer patients.
  meta <- meta %>%
    mutate(
      .derived_lung_status = case_when(
        as.character(status) == as.character(args$`cancer-label`) &
          !is.na(.data[[args$`cancer-site-col`]]) &
          !is.na(.data[[args$`lung-side-col`]]) &
          toupper(substr(as.character(.data[[args$`cancer-site-col`]]), 1, 1)) ==
            toupper(substr(as.character(.data[[args$`lung-side-col`]]), 1, 1)) ~ "TumorSide",
        as.character(status) == as.character(args$`cancer-label`) &
          !is.na(.data[[args$`cancer-site-col`]]) &
          !is.na(.data[[args$`lung-side-col`]]) ~ "Contralateral",
        TRUE ~ "Healthy"
      )
    )
  effective_contralateral_col <- ".derived_lung_status"
}

# Read wide format ASV table
message("Reading wide format ASV table: ", args$`data-wide`)
asv <- read_tsv(args$`data-wide`, show_col_types = FALSE)
stopifnot(ncol(asv) >= 2)
# Expect first column to be ASV IDs
asv <- asv %>% rename(ASV = 1)
asv_mat <- asv %>% column_to_rownames("ASV") %>% as.matrix()
mode(asv_mat) <- "numeric"

# Align samples
common <- intersect(colnames(asv_mat), rownames(meta))
if (length(common) == 0) {
  stop("No overlapping samples between ASV table and metadata.")
}

asv_mat <- asv_mat[, common, drop = FALSE]
meta <- meta[common, , drop = FALSE]

message("ASV table dimensions: ", paste(dim(asv_mat), collapse = " x "))
message("Metadata dimensions: ", paste(dim(meta), collapse = " x "))

cat("Data loaded:\n")
cat("  Samples:", ncol(asv_mat), "\n")
cat("  ASVs:", nrow(asv_mat), "\n")
cat("  Patients:", length(unique(meta[[args$`patient-col`]])), "\n\n")

# grouping columns already parsed above when extracting metadata
group_cols <- group_cols_vec
sample_sizes <- as.integer(strsplit(args$`sample-sizes`, ",")[[1]])
status_sites_requested <- strsplit(args$`status-sites`, ",", fixed = TRUE)[[1]] %>% trimws()
status_sites_requested <- status_sites_requested[status_sites_requested != ""]

cat("Grouping columns:", paste(group_cols, collapse=", "), "\n")
if (length(status_sites_requested) > 0) {
  cat("Status sites filter:", paste(status_sites_requested, collapse=", "), "\n")
} else {
  cat("Status sites filter: all\n")
}
cat("Sample sizes to test:", paste(sample_sizes, collapse=", "), "\n")
cat("Simulations:", args$`n-simulations`, "\n")
cat("Permutations per test:", args$perms, "\n")
cat("Test duleg (combos):", args$`test-duleg`, "\n\n")

# ============================================================================
# Helper functions
# ============================================================================

#' Bootstrap patients with replacement (preserves hierarchical structure)
#'
#' Generic function that works with any grouping variable.
#' For each resampled patient, include ALL their samples.
#' Assigns unique bootstrap IDs to handle duplicate patients in permutations.
#'
#' @param counts Sample × ASV count matrix
#' @param patient_ids Patient ID for each sample
#' @param grouping Grouping factor for each sample (e.g., status, type_group)
#' @param n_per_group Named vector of sample sizes per group, or single integer for balanced
#' @param seed Random seed
#' @return List with boot_counts, boot_patients, boot_grouping
bootstrap_patients_generic <- function(counts, patient_ids, grouping,
                                       n_per_group, seed = 42) {
  set.seed(seed)

  # Defensive: ensure patient_ids is character for type-safe comparisons
  patient_ids <- as.character(patient_ids)

  # Get unique groups
  unique_groups <- levels(grouping)

  # Handle n_per_group as single integer (balanced) or named vector
  if (length(n_per_group) == 1) {
    n_per_group <- setNames(rep(n_per_group, length(unique_groups)), unique_groups)
  }

  # Track draw counts for unique IDs (critical for permutation correctness)
  patient_draw_count <- list()

  boot_indices <- c()
  boot_patient_list <- c()
  boot_grouping_list <- c()

  # Process each group
  for (grp in unique_groups) {
    grp_mask <- grouping == grp
    grp_patients <- unique(patient_ids[grp_mask])

    # Sample n patients for this group
    n_sample <- n_per_group[[as.character(grp)]]

    # Skip empty groups or zero sample size
    if (length(grp_patients) == 0 || n_sample == 0) {
      next
    }

    boot_patients_grp <- sample(grp_patients, n_sample, replace = TRUE)

    for (patient in boot_patients_grp) {
      patient_key <- as.character(patient)
      if (is.null(patient_draw_count[[patient_key]])) {
        draw_idx <- 0
      } else {
        draw_idx <- patient_draw_count[[patient_key]]
      }
      patient_draw_count[[patient_key]] <- draw_idx + 1

      # CRITICAL: unique bootstrap ID prevents permutation collapse
      unique_boot_id <- paste0(patient_key, "__b", draw_idx)

      # Get all samples from this patient (CRITICAL: convert both to character for matching)
      patient_samples <- which(as.character(patient_ids) == patient_key)
      boot_indices <- c(boot_indices, patient_samples)
      boot_patient_list <- c(boot_patient_list, rep(unique_boot_id, length(patient_samples)))
      boot_grouping_list <- c(boot_grouping_list, rep(as.character(grp), length(patient_samples)))
    }
  }

  boot_counts <- counts[boot_indices, , drop = FALSE]
  boot_patients <- boot_patient_list
  boot_grouping <- factor(boot_grouping_list, levels = unique_groups)

  return(list(
    counts = boot_counts,
    patients = boot_patients,
    grouping = boot_grouping
  ))
}

#' Bootstrap patients with random label assignment (TRUE NULL)
#'
#' CRITICAL for Type I error calibration.
#' Pools ALL patients (ignoring original groups) and randomly assigns labels.
#' This breaks the link between labels and composition.
#'
#' @param counts Sample × ASV count matrix
#' @param patient_ids Patient ID for each sample
#' @param grouping Grouping factor (used only to get group levels)
#' @param n_per_group Named vector of sample sizes per group, or single integer for balanced
#' @param seed Random seed
#' @return List with boot_counts, boot_patients, boot_grouping
bootstrap_patients_true_null <- function(counts, patient_ids, grouping,
                                         n_per_group, seed = 42) {
  set.seed(seed)

  # Defensive: ensure patient_ids is character for type-safe comparisons
  patient_ids <- as.character(patient_ids)

  # Get unique groups
  unique_groups <- levels(grouping)

  # Handle n_per_group as single integer (balanced) or named vector
  if (length(n_per_group) == 1) {
    n_per_group <- setNames(rep(n_per_group, length(unique_groups)), unique_groups)
  }

  # Pool ALL patients (ignore original labels)
  all_patients <- unique(patient_ids)

  # Validate we have patients to sample
  if (length(all_patients) == 0) {
    stop("No patients available for bootstrap")
  }

  # Calculate total sample size
  n_total <- sum(n_per_group)

  # Sample n_total patients with replacement
  boot_patients_selected <- sample(all_patients, n_total, replace = TRUE)

  # Create random label assignment
  random_labels <- character(0)
  for (grp in unique_groups) {
    random_labels <- c(random_labels, rep(as.character(grp), n_per_group[[as.character(grp)]]))
  }
  random_labels <- sample(random_labels)  # Shuffle labels

  # Track draw counts for unique IDs
  patient_draw_count <- list()

  boot_indices <- c()
  boot_patient_list <- c()
  boot_grouping_list <- c()

  # Build bootstrap dataset with random labels
  for (i in seq_along(boot_patients_selected)) {
    patient <- boot_patients_selected[i]
    patient_key <- as.character(patient)

    if (is.null(patient_draw_count[[patient_key]])) {
      draw_idx <- 0
    } else {
      draw_idx <- patient_draw_count[[patient_key]]
    }
    patient_draw_count[[patient_key]] <- draw_idx + 1

    # CRITICAL: unique bootstrap ID prevents permutation collapse
    unique_boot_id <- paste0(patient_key, "__b", draw_idx)

    # Get all samples from this patient (CRITICAL: convert both to character for matching)
    patient_samples <- which(as.character(patient_ids) == patient_key)
    boot_indices <- c(boot_indices, patient_samples)
    boot_patient_list <- c(boot_patient_list, rep(unique_boot_id, length(patient_samples)))
    boot_grouping_list <- c(boot_grouping_list, rep(random_labels[i], length(patient_samples)))
  }

  boot_counts <- counts[boot_indices, , drop = FALSE]
  boot_patients <- boot_patient_list
  boot_grouping <- factor(boot_grouping_list, levels = unique_groups)

  return(list(
    counts = boot_counts,
    patients = boot_patients,
    grouping = boot_grouping
  ))
}

#' Aggregate counts to patient level (sum across samples per patient)
#'
#' @param counts Sample × ASV count matrix
#' @param patient_ids Patient ID for each sample
#' @return Patient × ASV count matrix
aggregate_to_patient_level <- function(counts, patient_ids) {
  unique_patients <- unique(patient_ids)
  patient_matrix <- matrix(0, nrow = length(unique_patients), ncol = ncol(counts))
  rownames(patient_matrix) <- unique_patients
  colnames(patient_matrix) <- colnames(counts)

  for (i in seq_along(unique_patients)) {
    patient <- unique_patients[i]
    # CRITICAL: convert both to character for type-safe matching
    patient_mask <- as.character(patient_ids) == as.character(patient)
    patient_matrix[i, ] <- colSums(counts[patient_mask, , drop = FALSE])
  }

  return(patient_matrix)
}

#' Bootstrap patients for within-patient designs (preserves sample-level labels)
#'
#' Samples patients with replacement and keeps their original per-sample grouping
#' labels (e.g., BAL/Lung Brush/Oral Rinse) intact.
#' Optionally generates a true null by shuffling labels within each bootstrap patient.
#'
#' @param counts Sample × ASV count matrix
#' @param patient_ids Patient ID for each sample
#' @param grouping Grouping factor for each sample
#' @param n_patients Number of patients to sample
#' @param true_null If TRUE, shuffle grouping labels within each bootstrap patient
#' @param seed Random seed
#' @return List with boot_counts, boot_patients, boot_grouping
bootstrap_patients_within <- function(counts, patient_ids, grouping,
                                      n_patients, true_null = FALSE, seed = 42) {
  set.seed(seed)

  patient_ids <- as.character(patient_ids)
  all_patients <- unique(patient_ids)
  if (length(all_patients) == 0) {
    stop("No patients available for bootstrap")
  }

  boot_patients_selected <- sample(all_patients, n_patients, replace = TRUE)
  patient_draw_count <- list()

  boot_indices <- c()
  boot_patient_list <- c()

  for (patient in boot_patients_selected) {
    patient_key <- as.character(patient)
    if (is.null(patient_draw_count[[patient_key]])) {
      draw_idx <- 0
    } else {
      draw_idx <- patient_draw_count[[patient_key]]
    }
    patient_draw_count[[patient_key]] <- draw_idx + 1

    unique_boot_id <- paste0(patient_key, "__b", draw_idx)
    patient_samples <- which(as.character(patient_ids) == patient_key)

    boot_indices <- c(boot_indices, patient_samples)
    boot_patient_list <- c(boot_patient_list, rep(unique_boot_id, length(patient_samples)))
  }

  boot_counts <- counts[boot_indices, , drop = FALSE]
  boot_patients <- boot_patient_list
  boot_grouping <- factor(as.character(grouping[boot_indices]), levels = levels(grouping))

  if (true_null) {
    for (p in unique(boot_patients)) {
      idx <- which(boot_patients == p)
      if (length(idx) > 1) {
        boot_grouping[idx] <- sample(boot_grouping[idx], length(idx), replace = FALSE)
      }
    }
    boot_grouping <- factor(as.character(boot_grouping), levels = levels(grouping))
  }

  return(list(
    counts = boot_counts,
    patients = boot_patients,
    grouping = boot_grouping
  ))
}

#' Apply deterministic spike-in to first group in factor
#'
#' Increases spiked ASVs by fold-change using integer reallocation.
#' Maintains library sizes and preserves zeros.
#' Spikes the first level of the grouping factor.
#'
#' @param counts Sample × ASV count matrix
#' @param grouping Grouping factor
#' @param spike_asv_idx Column indices of ASVs to spike
#' @param fold_change Multiplicative fold-change (e.g., 2.0)
#' @return Spiked count matrix
spike_in_first_group <- function(counts, grouping, spike_asv_idx, fold_change) {
  spiked_counts <- counts
  target_group <- levels(grouping)[1]
  target_mask <- grouping == target_group

  for (i in which(target_mask)) {
    count_vec <- spiked_counts[i, ]
    library_size <- sum(count_vec)

    if (library_size == 0) next

    # Convert to relative abundance
    rel_abund <- count_vec / library_size

    # Spike selected ASVs
    rel_abund[spike_asv_idx] <- rel_abund[spike_asv_idx] * fold_change

    # Renormalize
    rel_abund <- rel_abund / sum(rel_abund)

    # Resample counts (deterministic integer reconciliation)
    new_counts <- rel_abund * library_size
    new_counts_int <- floor(new_counts)
    remainders <- new_counts - new_counts_int
    deficit <- library_size - sum(new_counts_int)

    if (deficit > 0) {
      sorted_idx <- order(remainders, decreasing = TRUE)
      add_idx <- sorted_idx[1:min(deficit, length(sorted_idx))]
      new_counts_int[add_idx] <- new_counts_int[add_idx] + 1
    }

    spiked_counts[i, ] <- new_counts_int
  }

  return(spiked_counts)
}

apply_matrix_transform <- function(mat, method = "none") {
  method <- tolower(method)
  if (method == "none") {
    rs <- rowSums(mat)
    rs[rs == 0] <- 1
    rel <- sweep(mat, 1, rs, "/")
    rel[is.na(rel)] <- 0
    return(rel)
  }
  if (method != "rclr") {
    stop("Unsupported transform: ", method)
  }
  out <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
  for (i in seq_len(nrow(mat))) {
    v <- as.numeric(mat[i, ])
    pos <- !is.na(v) & v > 0
    if (any(pos)) {
      lv <- log(v[pos])
      out[i, pos] <- lv - mean(lv)
    }
  }
  out
}

#' Run ISA power simulation (generic for any grouping)
#'
#' @param counts Sample × ASV count matrix
#' @param patient_ids Patient ID for each sample
#' @param grouping Grouping factor
#' @param spike_asv_idx ASV indices to spike (NULL for null scenario)
#' @param spike_fc Fold-change for spike-in
#' @param n_size Sample size (balanced across groups)
#' @param duleg Whether to use duleg=TRUE (combinations allowed)
#' @param use_true_null If TRUE, uses pool-then-label null (for Type I error calibration)
#' @param n_simulations Number of bootstrap simulations
#' @param n_perm Permutations for multipatt
#' @param alpha Significance threshold
#' @param seed Random seed
#' @return List with power, sensitivity, fdr
run_isa_power_generic <- function(counts, patient_ids, grouping,
                                   spike_asv_idx, spike_fc, n_size,
                                   duleg = FALSE,
                                   use_true_null = FALSE,
                                   use_blocking = FALSE,
                                   n_simulations = 1000, n_perm = 199,
                                   alpha = 0.05, seed = 42,
                                   transform = "none") {

  n_total_asvs <- ncol(counts)
  n_spiked <- ifelse(is.null(spike_asv_idx), 0, length(spike_asv_idx))

  power_any <- 0
  sensitivity_sum <- 0
  fdr_sum <- 0
  fdr_conditional_sum <- 0
  n_with_discoveries <- 0

  for (i in 1:n_simulations) {
    # Bootstrap patients
    if (use_blocking) {
      # Within-patient design: sample patients and preserve sample-level labels
      boot_data <- bootstrap_patients_within(
        counts, patient_ids, grouping,
        n_patients = n_size,
        true_null = use_true_null,
        seed = seed + i
      )
    } else {
      if (use_true_null) {
        # Between-patient true null: pool patients and randomly assign labels
        boot_data <- bootstrap_patients_true_null(
          counts, patient_ids, grouping,
          n_per_group = n_size,
          seed = seed + i
        )
      } else {
        # Between-patient observed effect: resample within groups
        boot_data <- bootstrap_patients_generic(
          counts, patient_ids, grouping,
          n_per_group = n_size,
          seed = seed + i
        )
      }
    }

    # Apply spike-in if specified
    if (!is.null(spike_asv_idx) && spike_fc > 1.0) {
      boot_data$counts <- spike_in_first_group(
        boot_data$counts,
        boot_data$grouping,
        spike_asv_idx,
        spike_fc
      )
    }

    # For BETWEEN-patient: aggregate to patient level
    # For WITHIN-patient: keep sample level but use blocked permutations
    if (use_blocking) {
      # Sample-level analysis with patient blocking
      sample_rel <- apply_matrix_transform(boot_data$counts, transform)
      sample_grouping <- boot_data$grouping
      patient_blocks <- factor(boot_data$patients)

      # Run multipatt with blocked permutations
      tryCatch({
        isa_result <- suppressWarnings({
          multipatt(
            x = sample_rel,
            cluster = sample_grouping,
            duleg = duleg,
            control = how(nperm = n_perm, blocks = patient_blocks)
          )
        })
      }, error = function(e) {
        # Skip if multipatt fails (rare)
        isa_result <<- NULL
      })
    } else {
      # Patient-level analysis with standard permutations
      patient_counts <- aggregate_to_patient_level(
        boot_data$counts,
        boot_data$patients
      )

      # Map patients to grouping
      unique_patients <- unique(boot_data$patients)
      patient_grouping <- sapply(unique_patients, function(p) {
        as.character(boot_data$grouping[which(boot_data$patients == p)[1]])
      })
      patient_grouping <- factor(patient_grouping, levels = levels(grouping))

      patient_rel <- apply_matrix_transform(patient_counts, transform)

      # Run multipatt
      tryCatch({
        isa_result <- suppressWarnings({
          multipatt(
            x = patient_rel,
            cluster = patient_grouping,
            duleg = duleg,
            control = how(nperm = n_perm)
          )
        })
      }, error = function(e) {
        # Skip if multipatt fails (rare)
        isa_result <<- NULL
      })
    }

    # Process results if multipatt succeeded
    if (!is.null(isa_result)) {
      tryCatch({

      # Get significant ASVs (FDR-corrected, matching run_indicspecies.R)
      # CRITICAL: Must use FDR correction to match production analysis
      p_values <- isa_result$sign$p.value
      q_values <- p.adjust(p_values, method = "fdr")
      sig_asvs <- which(q_values < alpha)
      n_total_detected <- length(sig_asvs)

      if (n_total_detected > 0) {
        power_any <- power_any + 1
        n_with_discoveries <- n_with_discoveries + 1
      }

        if (n_spiked > 0) {
          # Sensitivity: proportion of spiked ASVs detected
          n_spiked_detected <- sum(spike_asv_idx %in% sig_asvs)
          sensitivity_sum <- sensitivity_sum + (n_spiked_detected / n_spiked)

          # FDR metrics: proportion of null ASVs falsely detected
          null_idx <- setdiff(1:n_total_asvs, spike_asv_idx)
          n_null_detected <- sum(null_idx %in% sig_asvs)

          if (n_total_detected > 0) {
            # Current FDR (diluted across all simulations)
            fdr_sum <- fdr_sum + (n_null_detected / n_total_detected)

            # Conditional FDR (only among simulations with discoveries)
            fdr_conditional_sum <- fdr_conditional_sum + (n_null_detected / n_total_detected)
          }
        }

      }, error = function(e) {
        # Skip if processing fails
      })
    }

    if (i %% 50 == 0) {
      cat(sprintf("\r      %d/%d...", i, n_simulations))
      flush.console()
    }
  }

  cat("\n")

  power <- power_any / n_simulations
  sensitivity <- ifelse(n_spiked > 0, sensitivity_sum / n_simulations, 0)
  fdr <- ifelse(n_spiked > 0, fdr_sum / n_simulations, 0)
  fdr_conditional <- ifelse(n_with_discoveries > 0, fdr_conditional_sum / n_with_discoveries, 0)

  return(list(
    power = power,
    sensitivity = sensitivity,
    fdr = fdr,
    fdr_conditional = fdr_conditional
  ))
}

# ============================================================================
# Main analysis: Loop over grouping columns (matching run_indicspecies.R)
# ============================================================================

# Transpose ASV matrix to samples × ASVs (for multipatt)
count_matrix <- t(asv_mat)

# Get patient IDs for bootstrap
patient_ids_vec <- meta[[args$`patient-col`]]

# Initialize results
all_results <- data.frame()

cat("================================================================\n")
cat("Running power analysis over grouping columns\n")
cat("================================================================\n\n")

# Loop over grouping columns (matching run_indicspecies.R approach)
for (gcol in group_cols) {
  if (!(gcol %in% colnames(meta))) {
    warning("Skipping grouping column '", gcol, "' (not found in metadata).")
    next
  }

  cat("\n--- Grouping Column:", gcol, "---\n")

  grouping_vec <- meta[[gcol]]

  # Drop NAs
  keep_idx <- !is.na(grouping_vec)
  grouping_clean <- as.factor(grouping_vec[keep_idx])
  grouping_clean <- droplevels(grouping_clean)

  # Get clean vectors
  counts_clean <- count_matrix[keep_idx, , drop = FALSE]
  patients_clean <- patient_ids_vec[keep_idx]

  if (length(unique(grouping_clean)) < 2) {
    warning("Grouping column '", gcol, "' has <2 groups after filtering; skipping.")
    next
  }

  # Status is analyzed within each sample type (between-patient per site).
  if (gcol == "status") {
    if (!("type_group" %in% colnames(meta))) {
      warning("Column 'type_group' not found; skipping status power analysis.")
      next
    }
    type_group_vec <- meta$type_group[keep_idx]
    sites <- unique(as.character(type_group_vec))
    sites <- sites[!is.na(sites)]
    if (length(status_sites_requested) > 0) {
      sites <- sites[sites %in% status_sites_requested]
      if (length(sites) == 0) {
        warning("No matching status sites found after applying --status-sites filter; skipping status power analysis.")
        next
      }
    }

    cat("  Design: BETWEEN-patient stratified by type_group\n")
    cat("  Sites:", paste(sites, collapse=", "), "\n\n")

    for (site in sites) {
      site_mask <- as.character(type_group_vec) == site
      counts_site <- counts_clean[site_mask, , drop = FALSE]
      patients_site <- patients_clean[site_mask]
      meta_site <- meta[keep_idx, , drop = FALSE][site_mask, , drop = FALSE]
      grouping_site <- droplevels(as.factor(grouping_clean[site_mask]))

      # Optional: exclude contralateral samples from cancer patients in Lung Brush.
      site_label_for_output <- site
      if (isTRUE(args$`status-extra-no-contralateral`) &&
          identical(as.character(site), as.character(args$`lung-brush-label`))) {
        if (!(effective_contralateral_col %in% colnames(meta_site))) {
          warning("Skipping no-contralateral filter for Lung Brush status power: column '",
                  effective_contralateral_col, "' not found.")
        } else {
          is_cancer <- as.character(meta_site[[gcol]]) == as.character(args$`cancer-label`)
          is_contralateral <- as.character(meta_site[[effective_contralateral_col]]) ==
            as.character(args$`contralateral-value`)
          keep_no_contra <- !(is_cancer & is_contralateral)
          n_removed <- sum(!keep_no_contra, na.rm = TRUE)
          counts_site <- counts_site[keep_no_contra, , drop = FALSE]
          patients_site <- patients_site[keep_no_contra]
          grouping_site <- droplevels(grouping_site[keep_no_contra])
          meta_site <- meta_site[keep_no_contra, , drop = FALSE]
          site_label_for_output <- paste0(site, " (no contralateral)")
          cat("    Applied no-contralateral filter:", n_removed, "sample(s) removed\n")
        }
      }

      if (length(unique(grouping_site)) < 2) {
        warning("Skipping status power for type_group='", site, "' (<2 status groups).")
        next
      }

      cat("  Site:", site, "\n")
      cat("    Status groups:", paste(levels(grouping_site), collapse=", "), "\n")
      cat("    Group sizes:", paste(table(grouping_site), collapse=", "), "\n")
      cat("    Unique patients:", length(unique(patients_site)), "\n")
      cat("    Design: BETWEEN-patient (standard permutations)\n")

      mean_abund <- colMeans(counts_site)
      top_asv_idx <- order(mean_abund, decreasing = TRUE)[1:min(10, ncol(counts_site))]

      scenarios <- list(
        list(name = "Null", asv_indices = NULL, fold_change = 1.0, use_true_null = TRUE),
        list(name = "Observed", asv_indices = NULL, fold_change = 1.0, use_true_null = FALSE),
        list(name = "Weak", asv_indices = top_asv_idx[1:min(3, length(top_asv_idx))], fold_change = 1.5, use_true_null = FALSE),
        list(name = "Moderate", asv_indices = top_asv_idx[1:min(5, length(top_asv_idx))], fold_change = 2.0, use_true_null = FALSE),
        list(name = "Strong", asv_indices = top_asv_idx, fold_change = 2.5, use_true_null = FALSE)
      )

      for (scenario in scenarios) {
        cat("\n    Scenario:", scenario$name, "(duleg=FALSE)\n")
        if (!is.null(scenario$asv_indices)) {
          cat("      Spiking", length(scenario$asv_indices), "ASVs @", scenario$fold_change, "×\n")
        }

        for (n_size in sample_sizes) {
          cat("      n =", n_size, "")
          result <- run_isa_power_generic(
            counts_site, patients_site, grouping_site,
            scenario$asv_indices, scenario$fold_change,
            n_size,
            duleg = FALSE,
            use_true_null = scenario$use_true_null,
            use_blocking = FALSE,
            n_simulations = args$`n-simulations`,
            n_perm = args$perms,
            alpha = args$alpha,
            seed = args$seed
          )
          cat(sprintf("→ Power=%.3f, Sens=%.3f, FDR=%.3f, FDRc=%.3f\n",
                      result$power, result$sensitivity, result$fdr, result$fdr_conditional))

          all_results <- rbind(all_results, data.frame(
            grouping_col = gcol,
            subgroup_col = "type_group",
            subgroup_level = site_label_for_output,
            duleg = FALSE,
            scenario = scenario$name,
            n_size = n_size,
            power = result$power,
            sensitivity = result$sensitivity,
            fdr = result$fdr,
            fdr_conditional = result$fdr_conditional,
            n_simulations = args$`n-simulations`,
            n_perm = args$perms,
            stringsAsFactors = FALSE
          ))
        }
      }

      if (args$`test-duleg`) {
        for (scenario in scenarios) {
          cat("\n    Scenario:", scenario$name, "(duleg=TRUE)\n")
          if (!is.null(scenario$asv_indices)) {
            cat("      Spiking", length(scenario$asv_indices), "ASVs @", scenario$fold_change, "×\n")
          }

          for (n_size in sample_sizes) {
            cat("      n =", n_size, "")
            result <- run_isa_power_generic(
              counts_site, patients_site, grouping_site,
              scenario$asv_indices, scenario$fold_change,
              n_size,
              duleg = TRUE,
              use_true_null = scenario$use_true_null,
              use_blocking = FALSE,
              n_simulations = args$`n-simulations`,
              n_perm = args$perms,
              alpha = args$alpha,
              seed = args$seed
            )
            cat(sprintf("→ Power=%.3f, Sens=%.3f, FDR=%.3f, FDRc=%.3f\n",
                        result$power, result$sensitivity, result$fdr, result$fdr_conditional))

            all_results <- rbind(all_results, data.frame(
              grouping_col = gcol,
              subgroup_col = "type_group",
              subgroup_level = site_label_for_output,
              duleg = TRUE,
              scenario = scenario$name,
              n_size = n_size,
              power = result$power,
              sensitivity = result$sensitivity,
              fdr = result$fdr,
              fdr_conditional = result$fdr_conditional,
              n_simulations = args$`n-simulations`,
              n_perm = args$perms,
              stringsAsFactors = FALSE
            ))
          }
        }
      }
      cat("\n")
    }
    next
  }

  # Check if this is a within-patient comparison
  patient_group_combos <- meta[keep_idx, ] %>%
    select(all_of(c(args$`patient-col`, gcol))) %>%
    distinct() %>%
    group_by(.data[[args$`patient-col`]]) %>%
    summarise(n_groups = n_distinct(.data[[gcol]]), .groups = 'drop')

  use_blocking <- any(patient_group_combos$n_groups > 1)

  cat("  Groups:", paste(levels(grouping_clean), collapse=", "), "\n")
  cat("  Group sizes:", paste(table(grouping_clean), collapse=", "), "\n")
  cat("  Unique patients:", length(unique(patients_clean)), "\n")

  if (use_blocking) {
    cat("  Design: WITHIN-patient (using blocked permutations)\n\n")
  } else {
    cat("  Design: BETWEEN-patient (standard permutations)\n\n")
  }

  # Select ASVs for spiking (top 10 most abundant overall)
  mean_abund <- colMeans(counts_clean)
  top_asv_idx <- order(mean_abund, decreasing = TRUE)[1:min(10, ncol(counts_clean))]

  # Define spike scenarios
  scenarios <- list(
    list(name = "Null", asv_indices = NULL, fold_change = 1.0, use_true_null = TRUE),
    list(name = "Observed", asv_indices = NULL, fold_change = 1.0, use_true_null = FALSE),
    list(name = "Weak", asv_indices = top_asv_idx[1:min(3, length(top_asv_idx))], fold_change = 1.5, use_true_null = FALSE),
    list(name = "Moderate", asv_indices = top_asv_idx[1:min(5, length(top_asv_idx))], fold_change = 2.0, use_true_null = FALSE),
    list(name = "Strong", asv_indices = top_asv_idx, fold_change = 2.5, use_true_null = FALSE)
  )

  # Test duleg=FALSE (single groups)
  for (scenario in scenarios) {
    cat("\n  Scenario:", scenario$name, "(duleg=FALSE)\n")
    if (!is.null(scenario$asv_indices)) {
      cat("    Spiking", length(scenario$asv_indices), "ASVs @", scenario$fold_change, "×\n")
    }

    for (n_size in sample_sizes) {
      if (use_blocking) {
        cat("    n_patients =", n_size, "")
      } else {
        cat("    n =", n_size, "")
      }

      result <- run_isa_power_generic(
        counts_clean, patients_clean, grouping_clean,
        scenario$asv_indices, scenario$fold_change,
        n_size,
        duleg = FALSE,
        use_true_null = scenario$use_true_null,
        use_blocking = use_blocking,
        n_simulations = args$`n-simulations`,
        n_perm = args$perms,
        alpha = args$alpha,
        seed = args$seed
      )

      cat(sprintf("→ Power=%.3f, Sens=%.3f, FDR=%.3f, FDRc=%.3f\n",
                  result$power, result$sensitivity, result$fdr, result$fdr_conditional))

      all_results <- rbind(all_results, data.frame(
        grouping_col = gcol,
        subgroup_col = NA_character_,
        subgroup_level = NA_character_,
        duleg = FALSE,
        scenario = scenario$name,
        n_size = n_size,
        power = result$power,
        sensitivity = result$sensitivity,
        fdr = result$fdr,
        fdr_conditional = result$fdr_conditional,
        n_simulations = args$`n-simulations`,
        n_perm = args$perms,
        stringsAsFactors = FALSE
      ))
    }
  }

  # Test duleg=TRUE (combinations allowed) if requested
  if (args$`test-duleg`) {
    for (scenario in scenarios) {
      cat("\n  Scenario:", scenario$name, "(duleg=TRUE)\n")
      if (!is.null(scenario$asv_indices)) {
        cat("    Spiking", length(scenario$asv_indices), "ASVs @", scenario$fold_change, "×\n")
      }

      for (n_size in sample_sizes) {
        if (use_blocking) {
          cat("    n_patients =", n_size, "")
        } else {
          cat("    n =", n_size, "")
        }

        result <- run_isa_power_generic(
          counts_clean, patients_clean, grouping_clean,
          scenario$asv_indices, scenario$fold_change,
          n_size,
          duleg = TRUE,
          use_true_null = scenario$use_true_null,
          use_blocking = use_blocking,
          n_simulations = args$`n-simulations`,
          n_perm = args$perms,
          alpha = args$alpha,
          seed = args$seed
        )

        cat(sprintf("→ Power=%.3f, Sens=%.3f, FDR=%.3f, FDRc=%.3f\n",
                    result$power, result$sensitivity, result$fdr, result$fdr_conditional))

        all_results <- rbind(all_results, data.frame(
          grouping_col = gcol,
          subgroup_col = NA_character_,
          subgroup_level = NA_character_,
          duleg = TRUE,
          scenario = scenario$name,
          n_size = n_size,
          power = result$power,
          sensitivity = result$sensitivity,
          fdr = result$fdr,
          fdr_conditional = result$fdr_conditional,
          n_simulations = args$`n-simulations`,
          n_perm = args$perms,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

# ============================================================================
# Save results
# ============================================================================

dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)
outfile <- file.path(args$outdir, "indicspecies_power_results.tsv")
write.table(all_results, outfile, sep = "\t", row.names = FALSE, quote = FALSE)

cat("\n================================================================\n")
cat("Power analysis complete!\n")
cat("Results saved to:", outfile, "\n")
cat("================================================================\n")
