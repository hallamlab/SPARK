#!/usr/bin/env bash
# ================================
# ASV Processing Script (config-driven, hardened)
# ================================
set -Eeuo pipefail
IFS=$'\n\t'
shopt -s nullglob extglob

# --- mamba/conda env bootstrap (robust) ---
ENV_NAME="${ENV_NAME:-asv-py}"          # override in config if you like
ENV_YAML="${ENV_YAML:-environment.yml}" # override in config
ENV_SYNC="${ENV_SYNC:-false}"           # set true to 'update --prune' each run

have(){ command -v "$1" >/dev/null 2>&1; }

# 1) Pick a manager (prefer mamba, then micromamba, then conda)
if   have mamba;       then PM="mamba";   SHELL_TYPE="conda";     # mamba uses conda's hook
elif have micromamba;  then PM="micromamba"; SHELL_TYPE="micromamba"
elif have conda;       then PM="conda";   SHELL_TYPE="conda"
else
  echo "[ERROR] Need mamba/micromamba/conda on PATH." >&2; exit 1
fi

# 2) Load shell hook so 'activate' works in non-interactive shells
load_hook(){
  case "$SHELL_TYPE" in
    conda)
      # Try modern hook first
      if eval "$($PM shell.bash hook)" 2>/dev/null; then
        return 0
      fi
      # Fallback: source conda.sh from base
      if BASE="$($PM info --base 2>/dev/null)"; then
        # shellcheck disable=SC1091
        source "$BASE/etc/profile.d/conda.sh" || true
        return 0
      fi
      ;;
    micromamba)
      eval "$($PM shell hook -s bash)" 2>/dev/null && return 0
      # Fallback to standard init location if available
      # shellcheck disable=SC1091
      [[ -f "$HOME/.bashrc" ]] && source "$HOME/.bashrc" || true
      ;;
  esac
}
load_hook || { echo "[ERROR] Could not initialize $PM shell hooks." >&2; exit 1; }

# 3) Ensure env exists (no re-create noise)
env_exists(){
  # works for mamba/conda/micromamba without jq
  "$PM" env list 2>/dev/null | awk 'NR>2{print $1}' | sed 's/*//g' | grep -Fxq "$ENV_NAME"
}

if [[ ! -f "$ENV_YAML" ]]; then
  echo "[ERROR] Env YAML not found: $ENV_YAML" >&2; exit 1
fi

if env_exists; then
  echo "[INFO] Using existing env: $ENV_NAME"
else
  echo "[INFO] Creating env: $ENV_NAME from $ENV_YAML"
  if [[ "$PM" = "micromamba" ]]; then
    "$PM" create -y -n "$ENV_NAME" -f "$ENV_YAML" || { echo "[ERROR] create failed"; exit 1; }
  else
    "$PM" env create -n "$ENV_NAME" -f "$ENV_YAML" || { echo "[ERROR] env create failed"; exit 1; }
  fi
fi

# 4) Optional sync (off by default to avoid re-create chatter)
if [[ "$ENV_SYNC" == "true" ]]; then
  if [[ "$PM" = "micromamba" ]]; then
    "$PM" install -y -n "$ENV_NAME" -f "$ENV_YAML" || { echo "[ERROR] sync failed"; exit 1; }
  else
    "$PM" env update -n "$ENV_NAME" -f "$ENV_YAML" --prune || { echo "[ERROR] env update failed"; exit 1; }
  fi
fi

# -------- Hardened PM hook + activate --------
: "${ENV_NAME:?ENV_NAME not set}"

# Auto-detect PM if not provided: micromamba > mamba > conda
if [[ -z "${PM-}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then PM="micromamba"
  elif command -v mamba >/dev/null 2>&1; then PM="mamba"
  elif command -v conda >/dev/null 2>&1; then PM="conda"
  else
    echo "[ERROR] No micromamba/mamba/conda on PATH"; exit 1
  fi
fi

_pm_root_from_exe() {
  local exe="$1"
  [[ -n "$exe" ]] || return 1
  exe="$(command -v "$exe" 2>/dev/null || true)"
  [[ -n "$exe" ]] || return 1
  dirname "$(dirname "$exe")"
}

_source_pm_hook() {
  # 0) env hints
  for exe in "${CONDA_EXE-}" "${MAMBA_EXE-}"; do
    [[ -n "$exe" ]] && {
      local root; root="$(dirname "$(dirname "$exe")")"
      [[ -f "$root/etc/profile.d/conda.sh"  ]] && { # shellcheck disable=SC1090
        source "$root/etc/profile.d/conda.sh"; return 0; }
      [[ -f "$root/etc/profile.d/mamba.sh"  ]] && { # shellcheck disable=SC1090
        source "$root/etc/profile.d/mamba.sh"; return 0; }
    }
  done

  # 1) ask PM (works for conda/mamba)
  local base; base="$($PM info --base 2>/dev/null || true)"

  # 2) derive from executable on PATH
  local pathroot; pathroot="$(_pm_root_from_exe "$PM" || true)"

  # 3) common roots to try in order
  local roots=(
    "$base" "$pathroot"
    "$HOME/mambaforge" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"
  )

  local r
  for r in "${roots[@]}"; do
    [[ -n "$r" ]] || continue
    if [[ -f "$r/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$r/etc/profile.d/conda.sh"; return 0
    fi
    if [[ -f "$r/etc/profile.d/mamba.sh" ]]; then
      # shellcheck disable=SC1090
      source "$r/etc/profile.d/mamba.sh"; return 0
    fi
  done
  return 1
}

activate_env() {
  if [[ "$PM" == "micromamba" ]]; then
    eval "$(micromamba shell hook -s bash)" || { echo "[ERROR] micromamba hook"; exit 1; }
    micromamba activate "$ENV_NAME" || { echo "[ERROR] activate failed ($ENV_NAME)"; exit 1; }
  else
    _source_pm_hook || {
      echo "[ERROR] conda/mamba hook missing (tried \$PM info, PATH, and common roots)"; exit 1; }
    # Prefer 'conda activate'; fallback to PM activate
    conda activate "$ENV_NAME" 2>/dev/null || $PM activate "$ENV_NAME" || {
      echo "[ERROR] activate failed ($ENV_NAME)"; exit 1; }
  fi
  echo "[INFO] Activated: $ENV_NAME via $PM"
}

activate_env
# -------- end hardened block --------

# --- end bootstrap ---

# ---------- utils ----------
die(){ echo "[ERROR] $*" >&2; exit 1; }
log(){ printf "[%(%F %T)T] %s\n" -1 "$*"; }

trap 'rc=$?; echo "[FATAL] Line $LINENO exited with $rc" >&2' ERR

need() { command -v "$1" >/dev/null 2>&1 || die "Missing required tool: $1"; }

# ---------- defaults ----------
CONFIG="asv.conf"

# Skip flags
SKIP_QC=0
SKIP_MERGE=0
SKIP_FILTER=0
SKIP_CONCAT=0
SKIP_DEREP=0
SKIP_DENOISE=0
SKIP_CHIMERA=0
SKIP_SWARM=0
SKIP_COUNT=0
SKIP_TABFILT=0

usage(){
  cat <<EOF
Usage: $0 [--config asv.conf] [skip flags]

Skip flags:
  --skip-fastp   --skip-merge   --skip-filter  --skip-concat  --skip-derep
  --skip-denoise --skip-chimera --skip-swarm   --skip-count   --skip-tabfilt
  -h, --help

Notes:
- All paths/params come from the config file.
- Paired-end formats handled: *_R1/_R2, *_1/_2, Illumina *_R1_001, lanes, .fastq(.gz)/.fq(.gz).
EOF
  exit 0
}

# ---------- arg parse ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:-}"; shift 2;;
    --skip-fastp)   SKIP_QC=1; shift;;
    --skip-merge)   SKIP_MERGE=1; shift;;
    --skip-filter)  SKIP_FILTER=1; shift;;
    --skip-concat)  SKIP_CONCAT=1; shift;;
    --skip-derep)   SKIP_DEREP=1; shift;;
    --skip-denoise) SKIP_DENOISE=1; shift;;
    --skip-chimera) SKIP_CHIMERA=1; shift;;
    --skip-swarm)   SKIP_SWARM=1; shift;;
    --skip-count)   SKIP_COUNT=1; shift;;
    --skip-tabfilt) SKIP_TABFILT=1; shift;;
    -h|--help) usage;;
    *) die "Unknown option: $1";;
  esac
done

[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"
# shellcheck disable=SC1090
source "$CONFIG"

# ---------- validate deps ----------
need fastp
need vsearch
need awk
need sed
need python
need nproc || true
command -v swarm >/dev/null 2>&1 || log "Note: swarm not found (skip Step 8 or install)."

# ---------- resolve threads ----------
if [[ -z "${THREADS:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then THREADS="$(nproc)"; else THREADS=4; fi
fi

# ---------- dirs ----------
[[ -d "${INPUT_DIR:-}" ]] || die "INPUT_DIR missing or not a dir: ${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required in config}"
QC_DIR="${OUTPUT_DIR}/fastp"
MERGED_DIR="${OUTPUT_DIR}/merged"
FILTERED_DIR="${OUTPUT_DIR}/filtered"
CONCAT_DIR="${OUTPUT_DIR}/concat"
DEREP_DIR="${OUTPUT_DIR}/derep"
DEN_DIR="${OUTPUT_DIR}/denoise"
NOC_DIR="${OUTPUT_DIR}/nochimeras"
SWM_DIR="${OUTPUT_DIR}/swarm"
ASV_DIR="${OUTPUT_DIR}/ASVs"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$QC_DIR" "$MERGED_DIR" "$FILTERED_DIR" "$CONCAT_DIR" "$DEREP_DIR" \
         "$DEN_DIR" "$NOC_DIR" "$ASV_DIR" "$LOG_DIR" "$SWM_DIR"

# ---------- pairing/hardening ----------
# Build regex arrays from config
IFS='|' read -r -a R1TOK <<< "${R1_TOKENS:-R1|1}"
IFS='|' read -r -a R2TOK <<< "${R2_TOKENS:-R2|2}"
[[ ${#R1TOK[@]} -eq ${#R2TOK[@]} ]] || die "R1_TOKENS and R2_TOKENS length mismatch."

IFS='|' read -r -a EXTS <<< "${EXT_PATTERNS:-\\.fastq\\.gz$|\\.fq\\.gz$|\\.fastq$|\\.fq$}"
SAMPLE_STRIP_REGEX="${SAMPLE_STRIP_REGEX:-(_S[0-9]+)?(_L[0-9]{3})?(_R[12])?(_[12])?(_001)?$}"

is_r1_like() {
  local base="$1"
  for i in "${!R1TOK[@]}"; do
    [[ "$base" =~ (^|[_\.\-])(${R1TOK[$i]})(_|\-|\.|$) ]] && return 0
  done
  return 1
}

r2_from_r1() {
  local r1="$1"
  local cand orig
  orig="$r1"

  for i in "${!R1TOK[@]}"; do
    local r1tok="${R1TOK[$i]}"
    local r2tok="${R2TOK[$i]}"

    # start from the original each time
    cand="$orig"

    # Try a set of common paired patterns
    cand="${cand//_${r1tok}_/_${r2tok}_}"
    cand="${cand//.${r1tok}./.${r2tok}.}"
    cand="${cand//-${r1tok}-/-${r2tok}-}"
    cand="${cand//-${r1tok}\./-${r2tok}.}"
    cand="${cand//_${r1tok}\./_${r2tok}.}"
    cand="${cand//_${r1tok}$/_${r2tok}}"
    cand="${cand//${r1tok}_001/${r2tok}_001}"

    # Only accept if it actually changed AND exists
    if [[ "$cand" != "$orig" && -f "$cand" ]]; then
      echo "$cand"
      return 0
    fi
  done

  # No valid R2 found
  echo ""
  return 1
}

# Safe sample name parser (no sed)
sample_from_path() {
  local p="$1" b ext
  b="$(basename "$p")"

  # strip known extensions
  for ext in .fastq.gz .fq.gz .fastq .fq; do
    [[ "$b" == *"$ext" ]] && b="${b%$ext}" && break
  done

  # If Illumina-style suffixes exist, strip them; else fall back gently.
  # Examples handled:
  #   NAME_S1_L001_R1_001, NAME_R1_001, NAME_R1, NAME_1, NAME_L001_R2, etc.
  if [[ "$b" =~ _S[0-9]+(_L[0-9]{3})?(_R[12]|_[12])?(_001)?$ ]]; then
    b="${b%%_S[0-9]*}"
  elif [[ "$b" =~ (_L[0-9]{3})?(_R[12]|_[12])(_001)?$ ]]; then
    # strip lane then R1/R2/1/2 (plus optional _001)
    b="${b%_001}"
    b="${b%_R1}"; b="${b%_R2}"
    b="${b%_1}";  b="${b%_2}"
    b="${b%_L[0-9][0-9][0-9]}"  # literal pattern removal won’t expand; do a fallback:
    [[ "$b" =~ (.*)_L[0-9]{3}$ ]] && b="${BASH_REMATCH[1]}"
  else
    # generic: drop only a trailing token like _R1/_R2/_1/_2/_001 if present
    case "$b" in
      *_R1|*_R2|*_1|*_2) b="${b%_*}" ;;
      *_001) b="${b%_*}" ;;
    esac
  fi

  # trim trailing separators
  b="${b%%[_-.]}"
  echo "$b"
}

collect_pairs() {
  local -n _R1S=$1
  local -n _R2S=$2
  local -n _SAMPLES=$3
  local f r2 s
  _R1S=() ; _R2S=() ; _SAMPLES=()

  # candidate FASTQ files
  mapfile -t CANDS < <(find "$INPUT_DIR" -maxdepth 1 -type f \( \
    -name "*.fastq.gz" -o -name "*.fq.gz" -o -name "*.fastq" -o -name "*.fq" \) | sort)

  for f in "${CANDS[@]}"; do
    base="$(basename "$f")"
    if is_r1_like "$base"; then
      r2="$(r2_from_r1 "$f" || true)"
      if [[ -n "$r2" && -f "$r2" ]]; then
        s="$(sample_from_path "$f")"
        _R1S+=("$f")
        _R2S+=("$r2")
        _SAMPLES+=("$s")
      elif [[ "${SINGLE_END}" == "true" ]]; then
        # Treat as single-end (R1-only)
        s="$(sample_from_path "$f")"
        _R1S+=("$f")
        _R2S+=("")    # empty marks SE
        _SAMPLES+=("$s")
      else
        log "Warning: No R2 for $(basename "$f"); skipping (set SINGLE_END=true to accept)."
      fi
    fi
  done
}

# ---------- IO sets ----------
declare -a R1S R2S SAMPLES
collect_pairs R1S R2S SAMPLES
[[ ${#R1S[@]} -gt 0 ]] || die "No usable FASTQ files found in $INPUT_DIR"

log "Found ${#R1S[@]} input items (paired or SE as configured). Threads=$THREADS"

# ---------- Steps ----------
fastp_qc(){
  log "Step 1: fastp QC"
  for idx in "${!R1S[@]}"; do
    local r1="${R1S[$idx]}" r2="${R2S[$idx]}" s="${SAMPLES[$idx]}"
    if [[ -n "$r2" ]]; then
      fastp \
        -i "$r1" -I "$r2" \
        -o "${QC_DIR}/${s}_R1.fastq.gz" \
        -O "${QC_DIR}/${s}_R2.fastq.gz" \
        -f "${FASTP_TRIM_FRONT_R1}" -t "${FASTP_TRIM_TAIL_R1}" \
        -F "${FASTP_TRIM_FRONT_R2}" -T "${FASTP_TRIM_TAIL_R2}" \
        -j "${QC_DIR}/${s}.json" -h "${QC_DIR}/${s}.html" \
        -w "${THREADS}"
    else
      fastp \
        -i "$r1" \
        -o "${QC_DIR}/${s}.fastq.gz" \
        -f "${FASTP_TRIM_FRONT_R1}" -t "${FASTP_TRIM_TAIL_R1}" \
        -j "${QC_DIR}/${s}.json" -h "${QC_DIR}/${s}.html" \
        -w "${THREADS}"
    fi
  done 2>&1 | tee -a "${LOG_DIR}/fastp_log.txt"
  log "Step 1 done."
}

merge_reads(){
  log "Step 2: Merge paired-end (skip SE)"
  local allow=()
  [[ "${MERGE_ALLOW_STAGGER}" == "true" ]] && allow+=(--fastq_allowmergestagger)
  for idx in "${!R1S[@]}"; do
    local s="${SAMPLES[$idx]}"
    local q1="${QC_DIR}/${s}_R1.fastq.gz"
    local q2="${QC_DIR}/${s}_R2.fastq.gz"
    if [[ -f "$q1" && -f "$q2" ]]; then
      vsearch --fastq_mergepairs "$q1" \
              --reverse "$q2" \
              --fastqout "${MERGED_DIR}/${s}.merged.fastq" \
              --fastq_maxdiffs "${MERGE_MAXDIFFS}" \
              --fastq_minovlen "${MERGE_MINOVLEN}" \
              --fastq_truncqual "${MERGE_TRUNQUAL}" \
              "${allow[@]}" \
              --threads "${THREADS}"
    else
      log "Info: ${s} appears single-end or missing QC pairs; skipping merge for this sample."
      # For SE, just convert to merged-like FASTQ
      if [[ -f "${QC_DIR}/${s}.fastq.gz" ]]; then
        zcat "${QC_DIR}/${s}.fastq.gz" > "${MERGED_DIR}/${s}.merged.fastq"
      fi
    fi
  done 2>&1 | tee -a "${LOG_DIR}/merging_log.txt"
  log "Step 2 done."
}

filter_reads(){
  log "Step 3: Filter merged"
  local m
  for m in "${MERGED_DIR}"/*.merged.fastq; do
    [[ -f "$m" ]] || continue
    local s; s="$(basename "$m" .merged.fastq)"
    vsearch --fastx_filter "$m" \
            --fastq_maxee "${FILTER_MAXEE}" \
            --fastq_minlen "${FILTER_MINLEN}" \
            --fastq_maxlen "${FILTER_MAXLEN}" \
            --fastaout "${FILTERED_DIR}/${s}.filtered.fasta"
  done 2>&1 | tee -a "${LOG_DIR}/filtering_log.txt"
  log "Step 3 done."
}

concatenate_reads(){
  log "Step 4: Concatenate"
  : > "${CONCAT_DIR}/concat.fasta"
  local f s
  for f in "${FILTERED_DIR}"/*.filtered.fasta; do
    [[ -f "$f" ]] || continue
    # recover sample robustly from original name
    s="$(basename "$f" .filtered.fasta)"
    # if earlier stages produced NAME only, keep; if it still has Illumina bits, clean:
    s="$(sample_from_path "${s}.fastq")"
    # relabel headers
    awk -v pref="$s" '{
      if ($0 ~ /^>/) {
        sub(/^>[^:]*:/, ">" pref ":", $0)
        if ($0 !~ /^>/ pref ":") $0 = ">" pref ":" substr($0,2)
      }
      print
    }' "$f" >> "${CONCAT_DIR}/concat.fasta"
  done 2>&1 | tee -a "${LOG_DIR}/concat_log.txt"
  log "Step 4 done."
}

dereplicate_sequences(){
  log "Step 5: Dereplicate"
  vsearch --derep_fulllength "${CONCAT_DIR}/concat.fasta" \
          --output "${DEREP_DIR}/derep.fasta" \
          --sizeout \
          --threads "${THREADS}" \
          --log "${LOG_DIR}/derep_log.txt"
  log "Step 5 done."
}

denoise_asv(){
  log "Step 6: UNOISE denoise"
  vsearch --cluster_unoise "${DEREP_DIR}/derep.fasta" \
          --centroids "${DEN_DIR}/centroids.fasta" \
          --sizein --sizeout --relabel ASV \
          --minsize "${UNOISE_MINSIZE}" \
          --threads "${THREADS}" \
          --log "${LOG_DIR}/denoise_log.txt"
  log "Step 6 done."
}

chimera_check(){
  log "Step 7: Chimera (uchime3_denovo)"
  vsearch --uchime3_denovo "${DEN_DIR}/centroids.fasta" \
          --nonchimeras "${NOC_DIR}/nochimeras.fasta" \
          --sizein \
          --threads "${THREADS}" \
          --log "${LOG_DIR}/nochimera_log.txt"
  log "Step 7 done."
}

swarm_clustering(){
  if ! command -v swarm >/dev/null 2>&1; then
    log "Swarm not installed; skipping Step 8."
    return 0
  fi
  log "Step 8: Swarm clustering"
  swarm -d "${SWARM_D}" -f -t "${THREADS}" -z \
        -i "${SWM_DIR}/swarm_reps.struct" \
        -j "${SWM_DIR}/swarm_reps.network" \
        -s "${SWM_DIR}/swarm_reps.stats.txt" \
        -w "${SWM_DIR}/swarm_reps.fasta" \
        -o "${SWM_DIR}/swarm_reps.swarms" \
        "${NOC_DIR}/nochimeras.fasta" 2>&1 | tee -a "${LOG_DIR}/swarm_log.txt"
  log "Step 8 done."
}

create_count_matrix(){
  log "Step 9: Build ASV table"
  cp "${NOC_DIR}/nochimeras.fasta" "${ASV_DIR}/ASVs.fasta"
  vsearch --usearch_global "${CONCAT_DIR}/concat.fasta" \
          --db "${ASV_DIR}/ASVs.fasta" \
          --id 0.999 \
          --otutabout "${ASV_DIR}/ASV_counts.tsv" \
          --threads "${THREADS}" \
          --log "${LOG_DIR}/count_log.txt"
  log "Step 9 done."
}

filter_table(){
  log "Step 10: Filter ASV table"
  [[ -f "${TABFILT_SCRIPT}" ]] || die "TABFILT_SCRIPT not found: ${TABFILT_SCRIPT}"
  python "${TABFILT_SCRIPT}" \
      "${ASV_DIR}/ASV_counts.tsv" \
      "${ASV_DIR}/ASV_filtered.tsv" \
      "${TABFILT_MIN_SAMPLE_SUM}" "${TABFILT_MIN_ASV_SUM}" \
      "${ASV_DIR}/ASVs.fasta" \
      "${ASV_DIR}/ASVs_filtered.fasta"
  log "Step 10 done."
}

# ---------- execute ----------
[[ $SKIP_QC -eq 1 ]]      || fastp_qc
[[ $SKIP_MERGE -eq 1 ]]   || merge_reads
[[ $SKIP_FILTER -eq 1 ]]  || filter_reads
[[ $SKIP_CONCAT -eq 1 ]]  || concatenate_reads
[[ $SKIP_DEREP -eq 1 ]]   || dereplicate_sequences
[[ $SKIP_DENOISE -eq 1 ]] || denoise_asv
[[ $SKIP_CHIMERA -eq 1 ]] || chimera_check
[[ $SKIP_SWARM -eq 1 ]]   || swarm_clustering
[[ $SKIP_COUNT -eq 1 ]]   || create_count_matrix
[[ $SKIP_TABFILT -eq 1 ]] || filter_table

log "Amplicon processing completed successfully."
