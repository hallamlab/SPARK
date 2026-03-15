# Beta Diversity Analysis Module

Patient-aware PERMANOVA testing for microbiome composition differences.

## Overview

Uses Bray-Curtis dissimilarity with patient-blocked permutations to test:
- Cancer vs Control differences (within sample types)
- Sample type differences (BAL vs Lung Brush vs Oral Rinse)

## Main Scripts

- **`run_bray_permanova_patient_aware.R`** - Patient-blocked PERMANOVA analysis
- **`plot_bray_permanova_patient_aware.py`** - PCoA ordination plots with statistical overlays

## Usage

Called via `run_main_analysis_pipeline.sh` or directly:

```bash
Rscript run_bray_permanova_patient_aware.R \
  --data-wide data/ASV_count_wide.tsv \
  --data-long data/ASV_master_long.tsv \
  --sample-col lmp_id \
  --patient-col Participant_ID \
  --case-col Case \
  --type-col type_group \
  --sample-types "Oral Rinse,BAL,Lung Brush" \
  --permutations 999 \
  --outdir results/tables
```

## Key Features

- **Patient blocking** - Permutations constrained within patients to handle repeated measures
- PERMDISP testing for dispersion homogeneity
- Multiple sample type comparisons
- Optional contralateral sample exclusion for lung cancer studies
- Compositional data transformation support (rCLR)

## Statistical Approach

Uses `vegan::adonis2()` with `permute::how(blocks=patient_id)` for proper handling of within-patient correlation structure.

## Outputs

- PERMANOVA results tables (R², F-statistic, p-values)
- PERMDISP results
- PCoA plots with 95% confidence ellipses
- Effect size summaries
