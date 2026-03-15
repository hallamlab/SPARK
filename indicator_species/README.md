# Indicator Species Analysis (ISA) Module

Identifies ASVs that are significantly associated with specific patient groups or sample types.

## Overview

Uses the `indicspecies` R package to find:
- Cancer-associated ASVs vs Control-associated ASVs
- Sample type-specific ASVs (BAL, Lung Brush, Oral Rinse)
- ASVs specific to combinations of groups

## Main Scripts

- **`run_indicspecies.R`** - Patient-aware ISA with blocked permutations
- **`plot_indicspecies.py`** - Significance scatter plots with taxonomic coloring
- **`plot_indicspecies_aligned.py`** - Combined main analysis + power analysis visualizations

## Usage

Called via `run_main_analysis_pipeline.sh` or directly:

```bash
Rscript run_indicspecies.R \
  --data-wide data/ASV_count_wide.tsv \
  --data-long data/ASV_master_long.tsv \
  --group-cols "status,type_group" \
  --blocked-cols "type_group" \
  --perms 999 \
  --outdir results/indicspecies
```

## Key Features

- **Patient-aware blocking** - Permutations respect within-patient sample structure
- Tests both single groups and combinations (e.g., "BAL+Lung Brush")
- FDR correction for multiple testing
- Optional contralateral sample exclusion
- Compositional data transformation support (rCLR)
- Integration with power analysis results

## Statistical Method

Uses `indicspecies::multipatt()` to compute indicator values based on:
1. **Specificity** - How exclusive is the ASV to a group?
2. **Fidelity** - How frequently does the ASV occur in that group?

Permutation tests assess statistical significance with patient-blocked permutations when appropriate.

## Outputs

- `*_indicator_species_results.tsv` - Full ISA results with indicator values and p-values
- `*_indicator_species_summary.tsv` - Filtered significant indicators
- Scatter plots showing significance vs indicator statistic
- Aligned plots combining main analysis + power curves (via `plot_indicspecies_aligned.py`)
