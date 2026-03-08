# Taxonomic Abundance Analysis Module

Patient-aware differential abundance testing at multiple taxonomic levels.

## Overview

Tests for differential abundance of taxonomic groups between:
- Cancer vs Control patients
- Different sample types (BAL, Lung Brush, Oral Rinse)

Uses Wilcoxon rank-sum tests with patient-level aggregation to avoid pseudoreplication.

## Main Scripts

- **`run_taxonomic_abundance_analysis.py`** - Cancer vs Control abundance testing
- **`run_taxonomic_sample_type_analysis.py`** - Sample type pairwise comparisons
- **`plot_taxonomic_observed_analysis.py`** - Publication-ready boxplots and significance visualizations

## Usage

Called via `run_main_analysis_pipeline.sh` or directly:

```bash
python3 run_taxonomic_abundance_analysis.py \
  --data-long data/ASV_master_long.tsv \
  --sample-col lmp_id \
  --patient-col Participant_ID \
  --case-col Case \
  --type-col type_group \
  --tax-levels "Phylum,Family" \
  --sample-types "BAL,Lung Brush,Oral Rinse" \
  --outdir results/tables
```

## Key Features

- Multiple taxonomic levels (Phylum, Family, Genus, etc.)
- Patient-level aggregation to handle repeated measures
- FDR correction for multiple testing
- Prevalence filtering to focus on common taxa
- Optional contralateral sample exclusion for lung cancer studies
- Compositional data transformation support (rCLR)

## Statistical Approach

1. Aggregate relative abundances to patient level (mean/median)
2. Wilcoxon rank-sum test for between-group comparisons
3. FDR correction across all tested taxa
4. Effect size reporting (median differences, fold changes)

## Outputs

- `*_observed.tsv` - Statistical test results with q-values
- Boxplot visualizations with significance stars
- Split panels for high vs low abundance taxa
