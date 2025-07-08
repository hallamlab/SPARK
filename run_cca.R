###############################################################################
## CCA with collinearity handling + safe mapping
###############################################################################
library(vegan)
library(dendextend)
library(caret)

# Define output directory
out_dir <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
count_data_path <- "/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/ASVs/ASV_filtered.micro.tsv"
asv_counts <- t(read.delim(count_data_path, row.names = 1, check.names = FALSE))
vocdata     <- "/home/ryan/Projects/UBC/LMP/SPARK_data/ref_db/VOC_table.tsv"
voc_data    <- read.delim(vocdata, row.names = 3, check.names = FALSE)

# Clean rownames
rownames(asv_counts) <- sapply(rownames(asv_counts), function(id){
  p <- strsplit(id, "_")[[1]]
  if (grepl("^P", id)) paste(p[1:2], collapse = "_") else p[1]
})
rownames(asv_counts) <- as.character(rownames(asv_counts))
rownames(voc_data)   <- as.character(rownames(voc_data))

cols_trim <- c("VOC_1", "Undecane_144", "1-Propanol_23", "VOC_595",
               "Dimethyl sulfone_358", "1-Octanol_377", "2-Butanone_31",
               "Nonane, 3-methyl-_365", "Oxetane, 2-ethyl-3-methyl-_451",
               "Dodecane_152", "Butanal_30", "VOC_149", "1-Octanol_140",
               "Acetoin_774", "Acetone_14", "2(3H)-Furanone, dihydro-5-methyl-_441",
               "Carbamic acid, monoammonium salt_2", "3-Heptanone_347",
               "VOC_900", "Benzene_50", "Heptane, 2,4-dimethyl-_86",
               "1,3,5-Trifluorobenzene_37", "Heptane, 2,2,4,6,6-pentamethyl-_439",
               "Decane_123", "Decane, 1,1'-oxybis-_243", "Octane_81",
               "Undecane, 2-methyl-_241", "Decane, 2,6,7-trimethyl-_372",
               "Nonane, 2-methyl-_363", "1-Butanol_53", "Levomenthol_384",
               "Hexane, 2,5-dimethyl-_319", "VOC_3", "Octanoic acid_242",
               "Butanal, 3-methyl-_417", "Dodecane, 2,7,1VOC-trimethyl-_165",
               "Heptane, 2,2,4,6,6-pentamethyl-_118", "Decane, 4-methyl-_376",
               "Nonadecane_207", "1-Octene_224", "VOC_599", "Acetic acid, methyl ester_19",
               "2-Butenedioic acid (Z)-, monododecyl ester_248", "Heptane, 3-ethyl-2-methyl-_434",
               "Decane_378", "Isopropyl myristate_284", "Dodecane, 2,6,11-trimethyl-_250",
               "Decane, 2-methyl-_137", "2-Propanol_15", "Methanesulfonic anhydride_22",
               "Hexane, 2-methyl-_310", "1,2-Ethanediol, monoacetate_228",
               "Heptane_56", "Octane, 4-methyl-_93", "VOC_679", "(2-Aziridinylethyl)amine_4",
               "1,2-Benzenedicarboxylic acid, bis(2-methylpropyl) ester_968",
               "Benzene, 1-ethyl-3-methyl-_606", "Butanoic acid, 4-hydroxy-_355",
               "Methyl propionate_41", "Ethanol_12", "1-Heptene_61", "2-Pentanone_52",
               "Acetic acid_27", "Acetic acid, butyl ester_84"
               )

env_data_subset <- voc_data[, cols_trim]

# Convert to numeric
env_data_numeric <- data.frame(lapply(env_data_subset, function(x) as.numeric(as.character(x))),
                               row.names = rownames(env_data_subset))
env_data_clean <- env_data_numeric[complete.cases(env_data_numeric), ]

# Match sample IDs
common <- intersect(rownames(asv_counts), rownames(env_data_clean))
asv_counts_clean <- asv_counts[common, ]
env_data_clean   <- env_data_clean[common, ]

# ---------------------------------------------------------------------------
# Remove highly correlated variables
# ---------------------------------------------------------------------------
cor_thresh <- 0.90
cor_mat    <- cor(env_data_clean, use = "pairwise.complete.obs")
drop_vars  <- findCorrelation(cor_mat, cutoff = cor_thresh, names = TRUE)

keep_vars  <- setdiff(colnames(env_data_clean), drop_vars)
env_data_reduced <- env_data_clean[, keep_vars, drop = FALSE]

if (length(drop_vars) > 0) {
  mapping_corr <- data.frame(
    removed = drop_vars,
    representative = vapply(drop_vars, function(v) {
      keep <- keep_vars[ which.max(abs(cor_mat[v, keep_vars])) ]
      keep
    }, character(1)),
    note = "high correlation"
  )
} else {
  mapping_corr <- data.frame(
    removed = character(),
    representative = character(),
    note = character()
  )
}

# ---------------------------------------------------------------------------
# Remove aliased (linearly dependent) variables
# ---------------------------------------------------------------------------
lm_dummy <- lm(rep(1, nrow(env_data_reduced)) ~ ., data = env_data_reduced)
aliased <- alias(lm_dummy, complete = TRUE)$Complete
aliased_vars <- setdiff(rownames(aliased), "(Intercept)")
env_data_final <- env_data_reduced[, !colnames(env_data_reduced) %in% aliased_vars, drop = FALSE]

mapping_alias <- data.frame(
  removed = aliased_vars,
  representative = sapply(aliased_vars, function(v) {
    cor_vals <- abs(cor_mat[v, keep_vars])
    cor_vals <- cor_vals[names(cor_vals) != v]  # remove self if present
    if (length(cor_vals) == 0) NA else names(which.max(cor_vals))
  }),
  note = "aliased"
)

# Combine mappings
mapping_full <- rbind(mapping_corr, mapping_alias)
write.table(mapping_full,
            file = file.path(out_dir, "collinearity_mapping.tsv"),
            sep  = "\t", quote = FALSE, row.names = FALSE)

# ---------------------------------------------------------------------------
# Run CCA
# ---------------------------------------------------------------------------
asv_hellinger <- decostand(asv_counts_clean, method = "hellinger")
cca_model     <- cca(asv_hellinger, env_data_final)

# Save output
write.table(scores(cca_model, display = "sites"),
            file = file.path(out_dir, "cca_site_scores.tsv"),
            sep  = "\t", quote = FALSE, col.names = NA)

write.table(scores(cca_model, display = "species"),
            file = file.path(out_dir, "cca_species_scores.tsv"),
            sep  = "\t", quote = FALSE, col.names = NA)

write.table(scores(cca_model, display = "bp"),
            file = file.path(out_dir, "cca_env_scores.tsv"),
            sep  = "\t", quote = FALSE, col.names = NA)

## envfit -> TSV
fit_env  <- envfit(cca_model, env_data_final, perm = 9999)
env_res  <- data.frame(
  variable = rownames(fit_env$vectors$arrows),
  CCA1     = fit_env$vectors$arrows[, 1],
  CCA2     = fit_env$vectors$arrows[, 2],
  r2       = fit_env$vectors$r,
  p_value  = fit_env$vectors$pvals
)
write.table(env_res,
            file = file.path(out_dir, "cca_envfit_env.tsv"),
            sep  = "\t", quote = FALSE, row.names = FALSE)

## taxa centroids (optional)
fit_taxa <- envfit(cca_model, asv_hellinger, perm = 9999)
taxa_res <- data.frame(
  ASV      = rownames(fit_taxa$vectors$arrows),
  CCA1     = fit_taxa$vectors$arrows[, 1],
  CCA2     = fit_taxa$vectors$arrows[, 2],
  r2       = fit_taxa$vectors$r,
  p_value  = fit_taxa$vectors$pvals
)
write.table(taxa_res,
            file = file.path(out_dir, "cca_envfit_taxa.tsv"),
            sep  = "\t", quote = FALSE, row.names = FALSE)


# Prepare input matrices for cosine similarity
species_scores <- scores(cca_model, display = "species")[, 1:2]
env_scores     <- scores(cca_model, display = "bp")[, 1:2]

# species_scores  : matrix/data-frame with CCA1, CCA2 rows = taxa
# env_scores      : matrix/data-frame with CCA1, CCA2 rows = env vars
cosine <- function(a, b) sum(a*b) / (sqrt(sum(a^2))*sqrt(sum(b^2)))

# build all pairwise similarities
taxa_env_cos <- outer(
  1:nrow(species_scores),
  1:nrow(env_scores),
  Vectorize(function(i, j)
    cosine(as.numeric(species_scores[i, 1:2]),
           as.numeric(env_scores[j,    1:2])))
)

dimnames(taxa_env_cos) <- list(rownames(species_scores),
                               rownames(env_scores))

# threshold e.g. |cos| ≥ 0.7
strong_pairs <- which(abs(taxa_env_cos) >= 0.7, arr.ind = TRUE)
result <- data.frame(
  Taxon = rownames(species_scores)[strong_pairs[,1]],
  Env   = rownames(env_scores)[strong_pairs[,2]],
  cos_theta = taxa_env_cos[strong_pairs]
)
write.table(result,
            file = file.path(out_dir, "cca_taxa_env_strong.tsv"),
            sep  = "\t", quote = FALSE, row.names = FALSE)


