# Power Analysis Module

Statistical power analysis for microbiome studies across multiple analysis types.

## Overview

Simulates datasets with varying sample sizes and effect sizes to estimate statistical power for:
- PERMANOVA (beta diversity)
- Shannon diversity
- Indicator Species Analysis (ISA)
- Taxonomic abundance testing

## Main Scripts

- **`run_power_analysis_pipeline.sh`** - Master orchestrator for complete power analysis workflow
- **`estimate_effects.py`** - Estimates observed effect sizes from real data
- **`power_main.py`** - Coordinates all power analysis modules
- **`power_*.py`** - Individual power analysis modules for different tests
- **`power_indicspecies.R`** - ISA power analysis (R implementation)
- **`plot_power_curves.py`** - Generate power curve visualizations

## Usage

```bash
./run_power_analysis_pipeline.sh \
  --data-long data/ASV_master_long.tsv \
  --data-wide data/ASV_count_wide.tsv \
  --outdir power_output \
  --transform rclr \
  --scenarios "observed,null"
```

## Key Features

- Flexible scenario testing (observed, null, weak, moderate, strong effects)
- Separate sample size grids for cancer vs sample type comparisons
- Optional contralateral sample exclusion for lung cancer studies
- Compositional data transformation support (rCLR)
- Parallel power curve generation for all analyses

## Dependencies

See `power_requirements.txt` for Python dependencies. R dependencies include `vegan`, `indicspecies`, `permute`.

## Outputs

- `effect_sizes/` - Observed effect sizes and spike scenario configurations
- `results/` - Power analysis results (TSV tables)
- `plots/` - Power curve visualizations (PDF/PNG)
