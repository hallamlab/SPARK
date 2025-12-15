#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "$0")"
CONTROL_ENV_DIR="${CONTROL_ENV_DIR:-${SCRIPT_DIR}/.controller_env}"

if [[ -z "${IN_CONTROLLER_ENV:-}" ]]; then
  if ! command -v mamba >/dev/null 2>&1; then
    echo "mamba is required to bootstrap the controller environment." >&2
    exit 1
  fi
  if [[ ! -d "$CONTROL_ENV_DIR" ]]; then
    echo "[controller] Creating mamba env at $CONTROL_ENV_DIR"
    mamba env create --yes --prefix "$CONTROL_ENV_DIR" --file "${SCRIPT_DIR}/envs/controller.yml"
  fi
  exec env IN_CONTROLLER_ENV=1 CONTROL_ENV_DIR="$CONTROL_ENV_DIR" conda run --no-capture-output -p "$CONTROL_ENV_DIR" "$SCRIPT_PATH" "$@"
fi

CONFIG_FILE=${1:-asv_pipeline_nextflow.yml}

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

WORK_DIR=$(yq -r '.paths.work_dir // empty' "$CONFIG_FILE")
OUTPUT_DIR=$(yq -r '.paths.output_dir // empty' "$CONFIG_FILE")
CONDA_CACHE_DIR=$(yq -r '.paths.conda_cache_dir // empty' "$CONFIG_FILE")

if [[ -z "$WORK_DIR" ]]; then
  echo "paths.work_dir must be set in $CONFIG_FILE" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "paths.output_dir must be set in $CONFIG_FILE" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"

if [[ -z "$CONDA_CACHE_DIR" || "$CONDA_CACHE_DIR" == "null" ]]; then
  CONDA_CACHE_DIR="${OUTPUT_DIR}/.conda_cache"
fi

mkdir -p "$CONDA_CACHE_DIR"

export NXF_WORK="$WORK_DIR"
export NXF_CONDA_CACHEDIR="$CONDA_CACHE_DIR"

nextflow run "${SCRIPT_DIR}/asv_pipeline.nf" --params-file "$CONFIG_FILE" -resume -w "$NXF_WORK"
