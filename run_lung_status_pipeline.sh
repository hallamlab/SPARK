#!/bin/bash
################################################################################
# Master pipeline: Lung status analysis (TumorSide vs Contralateral vs Healthy)
#
# Primary analysis: Lung Brush
# Exploratory analysis: BAL (if --run-bal flag is set)
#
# Usage:
#   ./run_lung_status_pipeline.sh [--run-bal]
################################################################################

set -euo pipefail

# Parse arguments
RUN_BAL=false
if [[ "${1:-}" == "--run-bal" ]]; then
  RUN_BAL=true
  echo "Will run exploratory BAL analysis"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/lung_status_analysis"

INPUT_LONG="${DATA_DIR}/supplementary_table_S2_ASV_master_long.tsv"

# Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found"; exit 1; }
command -v Rscript >/dev/null 2>&1 || { echo "ERROR: Rscript not found"; exit 1; }

################################################################################
# 1. LUNG BRUSH (Primary analysis)
################################################################################
echo ""
echo "=========================================="
echo "PRIMARY ANALYSIS: Lung Brush"
echo "=========================================="

BRUSH_OUT="${OUTPUT_DIR}/Lung_Brush"
mkdir -p "${BRUSH_OUT}"

echo ""
echo "[1/3] Preparing Lung Brush data..."
python3 "${SCRIPT_DIR}/prepare_lung_status_data.py" \
  --input "${INPUT_LONG}" \
  --sample-type "Lung Brush" \
  --outdir "${BRUSH_OUT}/data"

echo ""
echo "[2/3] Running planned contrasts (A/B/C)..."
Rscript "${SCRIPT_DIR}/run_lung_status_analysis.R" \
  "${BRUSH_OUT}/data/Lung_Brush_metadata.tsv" \
  "${BRUSH_OUT}/data/Lung_Brush_ASV_table.tsv" \
  "${BRUSH_OUT}/results"

echo ""
echo "[3/3] Generating figures..."
python3 "${SCRIPT_DIR}/plot_lung_status_analysis.py" \
  --metadata "${BRUSH_OUT}/data/Lung_Brush_metadata.tsv" \
  --patient-level "${BRUSH_OUT}/results/patient_level_metadata.tsv" \
  --distances "${BRUSH_OUT}/results/patient_level_bray_distances.tsv" \
  --summary "${BRUSH_OUT}/results/lung_status_contrasts_summary.tsv" \
  --pairdist-a "${BRUSH_OUT}/results/contrast_A_pairwise_distances.tsv" \
  --asv-table "${BRUSH_OUT}/data/Lung_Brush_ASV_table.tsv" \
  --outdir "${BRUSH_OUT}/figures"

echo ""
echo "Lung Brush analysis complete!"
echo "  Results: ${BRUSH_OUT}/results/"
echo "  Figures: ${BRUSH_OUT}/figures/"

################################################################################
# 2. BAL (Exploratory analysis - optional)
################################################################################
if [[ "${RUN_BAL}" == "true" ]]; then
  echo ""
  echo "=========================================="
  echo "EXPLORATORY ANALYSIS: BAL"
  echo "=========================================="
  echo "(Note: Limited paired sample size - interpret with caution)"

  BAL_OUT="${OUTPUT_DIR}/BAL"
  mkdir -p "${BAL_OUT}"

  echo ""
  echo "[1/3] Preparing BAL data..."
  python3 "${SCRIPT_DIR}/prepare_lung_status_data.py" \
    --input "${INPUT_LONG}" \
    --sample-type "BAL" \
    --outdir "${BAL_OUT}/data"

  echo ""
  echo "[2/3] Running planned contrasts (A/B/C)..."
  Rscript "${SCRIPT_DIR}/run_lung_status_analysis.R" \
    "${BAL_OUT}/data/BAL_metadata.tsv" \
    "${BAL_OUT}/data/BAL_ASV_table.tsv" \
    "${BAL_OUT}/results"

  echo ""
  echo "[3/3] Generating figures..."
  python3 "${SCRIPT_DIR}/plot_lung_status_analysis.py" \
    --metadata "${BAL_OUT}/data/BAL_metadata.tsv" \
    --patient-level "${BAL_OUT}/results/patient_level_metadata.tsv" \
    --distances "${BAL_OUT}/results/patient_level_bray_distances.tsv" \
    --summary "${BAL_OUT}/results/lung_status_contrasts_summary.tsv" \
    --pairdist-a "${BAL_OUT}/results/contrast_A_pairwise_distances.tsv" \
    --asv-table "${BAL_OUT}/data/BAL_ASV_table.tsv" \
    --outdir "${BAL_OUT}/figures"

  echo ""
  echo "BAL analysis complete!"
  echo "  Results: ${BAL_OUT}/results/"
  echo "  Figures: ${BAL_OUT}/figures/"
fi

################################################################################
# Summary
################################################################################
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Key results files:"
echo "  - Lung Brush summary: ${OUTPUT_DIR}/Lung_Brush/results/lung_status_contrasts_summary.tsv"
if [[ "${RUN_BAL}" == "true" ]]; then
  echo "  - BAL summary: ${OUTPUT_DIR}/BAL/results/lung_status_contrasts_summary.tsv"
fi
echo ""
echo "Key figures:"
echo "  - PCoA: ${OUTPUT_DIR}/Lung_Brush/figures/PCoA_lung_status.pdf"
echo "  - Alpha diversity: ${OUTPUT_DIR}/Lung_Brush/figures/Alpha_diversity_lung_status.pdf"
echo "  - R² summary: ${OUTPUT_DIR}/Lung_Brush/figures/PERMANOVA_R2_contrasts.pdf"
echo ""
