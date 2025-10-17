# ASV Processing Pipeline (Config-Driven, Hardened)

A modular Bash-based amplicon sequence variant (ASV) processing pipeline designed for reproducible, fully configurable analysis of raw FASTQ data using **fastp**, **vsearch**, **swarm**, and optional **Python-based filtering**.

---

## 🧬 Overview

This pipeline performs 10 standardized steps to process paired-end (or single-end) amplicon reads into ASVs and an abundance table:

1. **Fastp QC** – Quality filter and trim reads  
2. **Merge** – Merge paired reads (vsearch)  
3. **Filter** – Remove low-quality merged reads  
4. **Concatenate** – Combine all filtered reads  
5. **Dereplicate** – Collapse identical reads  
6. **Denoise (UNOISE)** – Identify true ASVs  
7. **Chimera Check** – Remove chimeric sequences  
8. **Swarm Clustering** – Optional artifact cleanup  
9. **Count Matrix** – Build ASV abundance table  
10. **Filter Table** – Filter ASVs by abundance thresholds (Python)

---

## ⚙️ Requirements

- **Bash** ≥ 4.0  
- **fastp**
- **vsearch**
- **swarm** *(optional)*
- **Python 3** (with dependencies for your `filter_ASV_table.py`)
- **mamba**, **micromamba**, or **conda**

The script automatically checks for a compatible package manager and ensures a conda/mamba environment exists or is created from a provided YAML file.

---

## 📁 Directory Structure

```
repo
├── asv_pipeline.sh
├── asv.conf
├── environment.yml
└── <other code>
project/
└── data/
    └── *.fq.gz
```

---

## 🧩 Configuration (`asv.conf`)

All hardcoded values have been moved to a config file. Example:

```bash
# Directories
INPUT_DIR="/home/user/data"
OUTPUT_DIR="/home/user/asv_output"

# Threads
THREADS=""             # auto-detect via nproc
SINGLE_END="false"     # set true for single-end data

# fastp trimming
FASTP_TRIM_FRONT_R1=19
FASTP_TRIM_TAIL_R1=20
FASTP_TRIM_FRONT_R2=20
FASTP_TRIM_TAIL_R2=20

# vsearch merge
MERGE_MAXDIFFS=20
MERGE_MINOVLEN=5
MERGE_ALLOW_STAGGER="true"

# vsearch filter
FILTER_MAXEE=1.0
FILTER_MINLEN=245

# UNOISE
UNOISE_MINSIZE=3

# swarm
SWARM_D=1

# Table filtering
TABFILT_MIN_SAMPLE_SUM=5000
TABFILT_MIN_ASV_SUM=0
TABFILT_SCRIPT="filter_ASV_table.py"
```

---

## 🧠 Environment Management

At runtime, the script:

1. Checks for **mamba**, **micromamba**, or **conda**.  
2. Loads the proper shell hooks.  
3. Verifies the environment (default `asv-py`).  
4. Creates or activates it automatically.  

Example `environment.yml`:

```yaml
name: asv-py
channels:
  - conda-forge
  - bioconda
dependencies:
  - python>=3.10
  - fastp
  - vsearch
  - pandas
  - biopython
```

---

## 🚀 Usage

Basic run:

```bash
./asv_pipeline.sh --config asv.conf
```

Skip steps if desired:

```bash
./asv_pipeline.sh --config asv.conf --skip-fastp --skip-swarm
```

Help:

```bash
./asv_pipeline.sh -h
```

---

## 🧪 Input Format

- **Paired-end** or **single-end** reads
- Automatically recognizes:
  - `_R1/_R2`, `_1/_2`, `_R1_001/_R2_001`
  - `.fastq(.gz)` and `.fq(.gz)` extensions
  - Illumina naming formats (e.g., `Sample_S1_L001_R1_001.fastq.gz`)

---

## 📊 Outputs

| Step | Directory | Description |
|------|------------|--------------|
| 1 | `fastp/` | Trimmed and filtered reads |
| 2 | `merged/` | Merged paired-end reads |
| 3 | `filtered/` | Filtered merged reads |
| 4 | `concat/` | Combined reads across samples |
| 5 | `derep/` | Dereplicated sequences |
| 6 | `denoise/` | ASV centroids (UNOISE) |
| 7 | `nochimeras/` | Chimera-free ASVs |
| 8 | `swarm/` | Optional swarm clustering results |
| 9 | `ASVs/` | ASV FASTA and count tables |
| - | `logs/` | Log files from each step |

---

## 🧰 Example Output Files

```
ASVs/
├── ASVs.fasta
├── ASV_counts.tsv
└── ASV_filtered.tsv
```

---

## 🧩 Logging

All steps write time-stamped logs into `logs/`:
```
logs/
├── fastp_log.txt
├── merging_log.txt
├── filtering_log.txt
...
```

---

## 🛡️ Error Handling

- Exits immediately on any failed command (`set -Eeuo pipefail`)
- Each step logs warnings instead of failing on missing R2 files
- Safe sample name parsing—robust to odd Illumina or single-end naming patterns
- Auto-detects available CPUs and missing tools

---

## 💡 Tips

- Run inside **screen** or **tmux** on clusters.
- Use `SINGLE_END="true"` in `asv.conf` for single-end data.
- To re-run partially, use `--skip-*` flags to resume from later steps.

---

## 🧾 Citation

If you use this pipeline, please cite:
> McLaughlin R. *ASV Processing Pipeline (2025)* – Framework for reproducible ASV workflows.

---

## 🧩 License

MIT License © 2025 Ryan McLaughlin
