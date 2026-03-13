#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Run main (non-network) lung microbiome analysis workflows and plots.

Usage:
  $(basename "$0") \
    --data-long <ASV_master_long.tsv> \
    --data-wide <ASV_count_wide.tsv> \
    --outdir <output_dir> \
    [--sample-col lmp_id] \
    [--patient-col Participant_ID] \
    [--case-col Case] \
    [--type-col type_group] \
    [--run-bal-contralateral] \
    [--skip-diversity-plots] \
    [--transform none|rclr] \
    [--keep-contralateral-in-cancer] [--contralateral-sample-types "Bronchial Brush,BAL"]

Notes:
- Networks are intentionally excluded.
- By default, this script runs patient-aware analyses plus alpha/beta diversity plots with patient-aware statistics.
- Use `--skip-diversity-plots` if you want to omit those diversity figures.
USAGE
}

DATA_LONG=""
DATA_WIDE=""
OUTDIR=""
SAMPLE_COL="lmp_id"
PATIENT_COL="Participant_ID"
CASE_COL="Case"
TYPE_COL="type_group"
RUN_BAL_CONTRA="false"
RUN_LEGACY_DIVERSITY="true"
TRANSFORM="none"
EXCLUDE_CONTRALATERAL="true"
CONTRALATERAL_SAMPLE_TYPES="Bronchial Brush,BAL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-long) DATA_LONG="$2"; shift 2 ;;
    --data-wide) DATA_WIDE="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --sample-col) SAMPLE_COL="$2"; shift 2 ;;
    --patient-col) PATIENT_COL="$2"; shift 2 ;;
    --case-col) CASE_COL="$2"; shift 2 ;;
    --type-col) TYPE_COL="$2"; shift 2 ;;
    --run-bal-contralateral) RUN_BAL_CONTRA="true"; shift ;;
    --enable-legacy-diversity) RUN_LEGACY_DIVERSITY="true"; shift ;;  # deprecated alias
    --skip-diversity-plots) RUN_LEGACY_DIVERSITY="false"; shift ;;
    --skip-legacy-diversity) RUN_LEGACY_DIVERSITY="false"; shift ;;  # deprecated, kept for backwards compatibility
    --transform) TRANSFORM="$2"; shift 2 ;;
    --keep-contralateral-in-cancer) EXCLUDE_CONTRALATERAL="false"; shift ;;
    --contralateral-sample-types) CONTRALATERAL_SAMPLE_TYPES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$DATA_LONG" ]] || { echo "ERROR: --data-long is required"; exit 2; }
[[ -n "$DATA_WIDE" ]] || { echo "ERROR: --data-wide is required"; exit 2; }
[[ -n "$OUTDIR" ]] || { echo "ERROR: --outdir is required"; exit 2; }
[[ -f "$DATA_LONG" ]] || { echo "ERROR: data-long not found: $DATA_LONG"; exit 2; }
[[ -f "$DATA_WIDE" ]] || { echo "ERROR: data-wide not found: $DATA_WIDE"; exit 2; }
[[ "$TRANSFORM" == "none" || "$TRANSFORM" == "rclr" ]] || { echo "ERROR: --transform must be one of: none, rclr"; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTDIR"

log() { printf "[%(%F %T)T] %s\n" -1 "$*"; }

log "Preparing derived metadata/taxonomy inputs"
AUX_DIR="$OUTDIR/_aux"
mkdir -p "$AUX_DIR"
DIVERSITY_META="$AUX_DIR/metadata_for_diversity.tsv"
TAXONOMY_TSV="$AUX_DIR/taxonomy_from_long.tsv"
DIVERSITY_WIDE="$AUX_DIR/asv_wide_for_diversity.tsv"

python3 - <<PY
import pandas as pd
from pathlib import Path

long_path = Path(r"$DATA_LONG")
wide_path = Path(r"$DATA_WIDE")
out_meta = Path(r"$DIVERSITY_META")
out_tax = Path(r"$TAXONOMY_TSV")
out_wide = Path(r"$DIVERSITY_WIDE")
sample_col = "$SAMPLE_COL"
case_col = "$CASE_COL"

df = pd.read_csv(long_path, sep='\t', low_memory=False)
if sample_col not in df.columns:
    raise SystemExit(f"Missing sample col: {sample_col}")

keep = [sample_col, "sample_code", "$PATIENT_COL", "$TYPE_COL", "status", case_col, "lung_code"]
keep = [c for c in keep if c in df.columns]
meta = df[keep].drop_duplicates(subset=[sample_col]).copy()
meta = meta.rename(columns={sample_col: "sample"})

if "status" not in meta.columns and case_col in meta.columns:
    meta["status"] = meta[case_col].map(lambda x: "Non-Cancer" if str(x) in {"Control", "Non-Cancer"} else "Cancer")

meta.to_csv(out_meta, sep='\t', index=False)

if {"ASV_ID", "Taxon"}.issubset(df.columns):
    tax = df[["ASV_ID", "Taxon"]].dropna().drop_duplicates()
    tax.to_csv(out_tax, sep='\t', index=False)

# Build a diversity-safe wide matrix: ASV_ID + only true sample columns
# present in long data. This avoids metadata/summary columns in the wide table
# being interpreted as samples.
wide = pd.read_csv(wide_path, sep='\t', low_memory=False)
if "ASV_ID" not in wide.columns:
    wide = wide.rename(columns={wide.columns[0]: "ASV_ID"})
sample_ids = pd.Index(df[sample_col].dropna().astype(str).unique())
keep_cols = ["ASV_ID"] + [c for c in wide.columns if c in set(sample_ids)]
wide_filtered = wide.loc[:, keep_cols].copy()
wide_filtered.to_csv(out_wide, sep='\t', index=False)
PY

################################################################################
# 1) Indicator Species Analysis + plots
################################################################################
log "Running patient-aware ISA"
ISA_DIR="$OUTDIR/indicspecies"
mkdir -p "$ISA_DIR"

Rscript "$SCRIPT_DIR/indicator_species/run_indicspecies.R" \
  --data-wide "$DATA_WIDE" \
  --data-long "$DATA_LONG" \
  --sample-col "$SAMPLE_COL" \
  --patient-col "$PATIENT_COL" \
  --group-cols "status,type_group" \
  --blocked-cols "type_group" \
  --status-exclude-contralateral "$EXCLUDE_CONTRALATERAL" \
  --status-contralateral-sites "$CONTRALATERAL_SAMPLE_TYPES" \
  --outdir "$OUTDIR" \
  --transform "$TRANSFORM"

log "Plotting ISA outputs"
TYPE_RESULTS="$OUTDIR/indicspecies/type_group_indicator_species_summary.tsv"
STATUS_RESULTS="$OUTDIR/indicspecies/status_indicator_species_summary.tsv"

PLOT_ISA_CMD=(
  python3 "$SCRIPT_DIR/indicator_species/plot_indicspecies.py"
  --type-results "$TYPE_RESULTS"
  --status-results "$STATUS_RESULTS"
  --outdir "$ISA_DIR"
  --p-thresh 0.05
  --stat-thresh 0.0
  --type-index "1=BAL,2=Bronchial Brush,3=Oral Rinse,4=BAL+Bronchial Brush,5=BAL+Oral Rinse,6=Bronchial Brush+Oral Rinse,7=Oral Rinse+BAL+Bronchial Brush"
  --status-index "1=Cancer,2=Non-Cancer,3=Cancer+Non-Cancer"
  --type-palette "Oral Rinse=#6A3D9A,BAL=#0072B2,Bronchial Brush=#009E73,BAL+Oral Rinse=#F19CBB,BAL+Bronchial Brush=#00FFFF,Bronchial Brush+Oral Rinse=#C1EAAD,Oral Rinse+BAL+Bronchial Brush=#000000"
  --status-palette "Cancer=#A50026,Non-Cancer=#FFFFFF,Cancer+Non-Cancer=#000000"
)
if [[ -f "$TAXONOMY_TSV" ]]; then
  PLOT_ISA_CMD+=(--taxonomy "$TAXONOMY_TSV")
fi
"${PLOT_ISA_CMD[@]}"

log "Generating aligned ISA summary/plot bundle"
python3 "$SCRIPT_DIR/indicator_species/plot_indicspecies_aligned.py" \
  --indicspecies-dir "$OUTDIR/indicspecies" \
  --outdir "$OUTDIR/indicspecies_aligned" \
  --alpha 0.05 \
  --min-stat 0.0 \
  --top-n 25

################################################################################
# 2) Alpha/beta diversity outputs + plots
################################################################################
if [[ "$RUN_LEGACY_DIVERSITY" == "true" ]]; then
  echo ""
  echo "ℹ️  INFO: Diversity plotting is enabled."
  echo "    Statistical tests are PATIENT-AWARE (paired Wilcoxon, blocked PERMANOVA)"
  echo "    when --patient-col and --count-table are provided (default behavior)."
  echo "    These outputs use the same patient-level aggregation as the main pipeline."
  echo ""
  log "Computing diversity metrics (Shannon/Bray/Jaccard)"
  DIV_METRICS_DIR="$OUTDIR/diversity/metrics"
  DIV_PLOTS_DIR="$OUTDIR/diversity/plots"
  mkdir -p "$DIV_METRICS_DIR" "$DIV_PLOTS_DIR" "$DIV_PLOTS_DIR/tables" "$DIV_PLOTS_DIR/plots"

  python3 "$SCRIPT_DIR/calc_div.py" \
    --micro-table "$DIVERSITY_WIDE" \
    --samples-on columns \
    --index-col 0 \
    --outdir "$DIV_METRICS_DIR"

  log "Plotting alpha/beta diversity figures"
  python3 "$SCRIPT_DIR/plot_diversity.py" \
    --metadata "$DIVERSITY_META" \
    --count-table "$DIVERSITY_WIDE" \
    --alpha "$DIV_METRICS_DIR/shannon.tsv" \
    --bray "$DIV_METRICS_DIR/bray.tsv" \
    --jacc "$DIV_METRICS_DIR/jaccard.tsv" \
    --patient-col "$PATIENT_COL" \
    --type-order "Oral Rinse,BAL,Bronchial Brush" \
    --outdir "$DIV_PLOTS_DIR"
fi

################################################################################
# 3) Patient-aware Bray PERMANOVA + plots
################################################################################
log "Running patient-aware Bray-Curtis PERMANOVA"
BRAY_DIR="$OUTDIR/bray_patient_aware"
mkdir -p "$BRAY_DIR"
Rscript "$SCRIPT_DIR/beta_diversity/run_bray_permanova_patient_aware.R" \
  --data-wide "$DATA_WIDE" \
  --data-long "$DATA_LONG" \
  --sample-col "$SAMPLE_COL" \
  --patient-col "$PATIENT_COL" \
  --case-col "$CASE_COL" \
  --type-col "$TYPE_COL" \
  --sample-types "Oral Rinse,BAL,Bronchial Brush" \
  --permutations 999 \
  --seed 42 \
  --exclude-contralateral-in-cancer "$EXCLUDE_CONTRALATERAL" \
  --contralateral-sample-types "$CONTRALATERAL_SAMPLE_TYPES" \
  --outdir "$BRAY_DIR/tables" \
  --transform "$TRANSFORM"

log "Plotting patient-aware Bray outputs"
python3 "$SCRIPT_DIR/beta_diversity/plot_bray_permanova_patient_aware.py" \
  --indir "$BRAY_DIR/tables" \
  --outdir "$BRAY_DIR/figures"

################################################################################
# 4) Patient-aware taxonomy analyses + plots
################################################################################
log "Running patient-aware taxonomy analyses"
TAX_DIR="$OUTDIR/taxonomy_patient_aware"
mkdir -p "$TAX_DIR/tables" "$TAX_DIR/figures"

python3 "$SCRIPT_DIR/taxonomic_analysis/run_taxonomic_abundance_analysis.py" \
  --data-long "$DATA_LONG" \
  --sample-col "$SAMPLE_COL" \
  --patient-col "$PATIENT_COL" \
  --case-col "$CASE_COL" \
  --type-col "$TYPE_COL" \
  --count-col count \
  --tax-levels "Phylum,Family" \
  --sample-types "BAL,Bronchial Brush,Oral Rinse" \
  --min-prevalence 0.10 \
  --contralateral-sample-types "$CONTRALATERAL_SAMPLE_TYPES" \
  $( [[ "$EXCLUDE_CONTRALATERAL" == "false" ]] && echo "--keep-contralateral-in-cancer" ) \
  --outdir "$TAX_DIR/tables" \
  --transform "$TRANSFORM"

python3 "$SCRIPT_DIR/taxonomic_analysis/run_taxonomic_sample_type_analysis.py" \
  --data-long "$DATA_LONG" \
  --sample-col "$SAMPLE_COL" \
  --patient-col "$PATIENT_COL" \
  --type-col "$TYPE_COL" \
  --count-col count \
  --tax-levels "Phylum,Family" \
  --sample-types "BAL,Oral Rinse,Bronchial Brush" \
  --min-prevalence 0.10 \
  --contralateral-sample-types "$CONTRALATERAL_SAMPLE_TYPES" \
  $( [[ "$EXCLUDE_CONTRALATERAL" == "false" ]] && echo "--keep-contralateral-in-cancer" ) \
  --outdir "$TAX_DIR/tables" \
  --transform "$TRANSFORM"

log "Plotting patient-aware taxonomy outputs"
python3 "$SCRIPT_DIR/taxonomic_analysis/plot_taxonomic_observed_analysis.py" \
  --data-long "$DATA_LONG" \
  --cancer-results "$TAX_DIR/tables/taxonomic_abundance_observed.tsv" \
  --sampletype-results "$TAX_DIR/tables/taxonomic_sample_type_observed_pairwise.tsv" \
  --sample-col "$SAMPLE_COL" \
  --patient-col "$PATIENT_COL" \
  --type-col "$TYPE_COL" \
  --case-col "$CASE_COL" \
  --count-col count \
  --outdir "$TAX_DIR/figures"

################################################################################
# 5) Contralateral workflow + plots (Bronchial Brush primary; BAL optional)
################################################################################
run_contralateral_one() {
  local sample_type="$1"
  local label="${sample_type// /_}"
  local C_DIR="$OUTDIR/contralateral/${label}"
  mkdir -p "$C_DIR/data" "$C_DIR/results" "$C_DIR/figures"

  log "Preparing contralateral data for ${sample_type}"
  python3 "$SCRIPT_DIR/prepare_lung_status_data.py" \
    --input "$DATA_LONG" \
    --sample-type "$sample_type" \
    --outdir "$C_DIR/data"

  log "Running contralateral planned contrasts for ${sample_type}"
  Rscript "$SCRIPT_DIR/run_lung_status_analysis.R" \
    "$C_DIR/data/${label}_metadata.tsv" \
    "$C_DIR/data/${label}_ASV_table.tsv" \
    "$C_DIR/results"

  log "Plotting contralateral results for ${sample_type}"
  python3 "$SCRIPT_DIR/plot_lung_status_analysis.py" \
    --metadata "$C_DIR/data/${label}_metadata.tsv" \
    --patient-level "$C_DIR/results/patient_level_metadata.tsv" \
    --distances "$C_DIR/results/patient_level_bray_distances.tsv" \
    --summary "$C_DIR/results/lung_status_contrasts_summary.tsv" \
    --pairdist-a "$C_DIR/results/contrast_A_pairwise_distances.tsv" \
    --asv-table "$C_DIR/data/${label}_ASV_table.tsv" \
    --outdir "$C_DIR/figures"
}

run_contralateral_one "Bronchial Brush"
if [[ "$RUN_BAL_CONTRA" == "true" ]]; then
  run_contralateral_one "BAL"
fi

log "Main analysis pipeline complete."
log "Output root: $OUTDIR"
