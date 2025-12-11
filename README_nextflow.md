# Nextflow ASV Pipeline

This document focuses on the Nextflow implementation (`SPARK/asv_pipeline.nf`) of the ASV processing workflow. The Bash pipeline and its README remain untouched so you can compare both approaches. Everything here assumes the accompanying YAML config (`SPARK/asv_pipeline_nextflow.yml`) as the canonical way to configure the Nextflow run.

## Pipeline Overview

The workflow mirrors the ten stages from the Bash script while embracing Nextflow’s channel-based execution and reproducibility. Each process publishes files into the same subdirectories under your configured `output_dir`.

| Stage | Process ID | Description | Key Outputs |
| --- | --- | --- | --- |
| 1 | `FASTP_QC` | Runs `fastp` on paired or single-end reads using per-sample metadata. Supports trimming controls, JSON/HTML reports, and automatic gzip handling. | `fastp/*.fastq.gz`, `fastp/*.fastp.json`, `fastp/*.fastp.html` |
| 2 | `MERGE_READS` | Uses `vsearch --fastq_mergepairs` (or passthrough for SE reads) to produce merged FASTQ files per sample. Optional staggered merging flag is available. | `merged/*.merged.fastq` |
| 3 | `FILTER_READS` | Applies `vsearch --fastx_filter` with max expected errors, min/max length cutoffs. | `filtered/*.filtered.fasta` |
| 4 | `RELABEL_FASTA` | Relabels FASTA headers so each sequence is prefixed with `sample_id:` to preserve provenance. | `filtered/*.relabeled.fasta` |
| 5 | `DEREPLICATE` | Global dereplication via `vsearch --derep_fulllength`, emitting `derep/derep.fasta` and a log. | `derep/derep.fasta`, `logs/derep.log` |
| 6 | `DENOISE` | UNOISE clustering (`vsearch --cluster_unoise`) with `--minsize` control and ASV relabeling. | `denoise/centroids.fasta`, `logs/denoise.log` |
| 7 | `CHIMERA_CHECK` | Runs `vsearch --uchime3_denovo` to remove chimeras. | `nochimeras/nochimeras.fasta`, `logs/nochimera.log` |
| 8 | `SWARM_CLUSTER` *(optional)* | Executes `swarm` for additional OTU-style clustering. Controlled by the YAML `swarm.enabled` and `steps.skip_swarm` switches. | `swarm/swarms` + reports |
| 9 | `CREATE_COUNT_MATRIX` | Copies non-chimeric centroids to `ASVs.fasta` and maps concatenated reads back with `vsearch --usearch_global` to form an ASV count table. | `ASVs/ASV_counts.tsv`, `ASVs/ASVs.fasta`, `logs/count.log` |
| 10 | `FILTER_TABLE` | Runs the Python filter script on the ASV table and FASTA to enforce abundance thresholds. | `ASVs/ASV_filtered.tsv`, `ASVs/ASVs_filtered.fasta` |

### Channel wiring at a glance
1. Input FASTQ files → `FASTP_QC` (optional) → `MERGE_READS` → `FILTER_READS` → `RELABEL_FASTA`
2. Relabeled FASTA files are concatenated using `collectFile` (Nextflow DSL2 `storeDir`) into `concat/concat.fasta`.
3. Downstream processes (`DEREPLICATE`, `DENOISE`, `CHIMERA_CHECK`, `SWARM_CLUSTER`, `CREATE_COUNT_MATRIX`, `FILTER_TABLE`) consume fan-out copies of the concatenated and filtered datasets. Skip flags short-circuit the corresponding processes while keeping channel compatibility.

## Directory Layout

Each run populates the following structure under `output_dir` (by default inherited from the Bash pipeline):

```
output_dir/
├── fastp/
├── merged/
├── filtered/
├── concat/            # concat.fasta
├── derep/
├── denoise/
├── nochimeras/
├── swarm/
├── ASVs/
└── logs/
```

Nextflow’s `work/` and `.nextflow.log` directories remain at the execution root (unless overridden with `-work-dir`).

## Configuration Reference (`asv_pipeline_nextflow.yml`)

The YAML mirrors the original `asv.conf` but is structured hierarchically:

```yaml
paths:
  input_dir: /path/to/fastq
  output_dir: /path/to/output

resources:
  threads:            # optional override, defaults to host CPUs
  single_end: false

steps:
  skip_fastp: false
  ...                 # toggles for each process

fastp:
  trim_front_r1: 19
  trim_tail_r1: 80
  trim_front_r2: 20
  trim_tail_r2: 80

merge:
  max_diffs: 20
  min_overlap: 5
  trunc_quality: 5
  allow_stagger: true
```

Additional sections include:

- `filter`: Expected errors + length cutoffs.
- `unoise`: `min_size`.
- `swarm`: `enabled` and `distance`.
- `table_filter`: thresholds and override path for the filtering script.
- `filename_patterns`: regex tokens used to identify R1/R2 files and sanitize sample names.

**Tips**
- Use absolute paths or relative paths evaluated from the YAML file’s directory.
- `steps.*` booleans let you resume from intermediate artifacts without touching the main channel definitions.
- Set `resources.threads` if you want to cap CPU usage rather than letting Nextflow use all available cores.

## Usage Tutorial

### 1. Install prerequisites

- **Nextflow** ≥ 23.x (`curl -s https://get.nextflow.io | bash`).
- **Conda/Mamba environment** matching `environment.yml` (already handled by your existing setup).
- Toolchain availability (`fastp`, `vsearch`, `swarm`, `python`) inside that environment.

### 2. Prepare the environment

```bash
cd /home/ryan/SABer_dat/SI_data/SI_ASV/SPARK
mamba activate asv-py   # or your preferred env
```

### 3. Copy & edit the YAML config

```bash
cp asv_pipeline_nextflow.yml my_run.yml
# Edit paths, skip flags, thresholds, etc.
```

Make sure `paths.input_dir` points to the folder containing FASTQ files and `paths.output_dir` is an empty (or resumable) destination.

### 4. Run a dry run (optional but recommended)

```bash
nextflow run asv_pipeline.nf --config my_run.yml -n
```

`-n` (`-dry-run`) validates the config, prints the plan, and ensures all required scripts exist.

### 5. Launch the pipeline

```bash
nextflow run asv_pipeline.nf --config my_run.yml \
    -profile standard \
    -with-report reports/asv_report.html \
    -with-trace reports/asv_trace.txt
```

- Use `-resume` to restart from cached results if the run fails or you tweak non-breaking parameters.
- Override work/output locations with `-work-dir` or custom YAML paths when needed.

### 6. Skipping or resuming stages

To reuse existing QC outputs, you can either:
1. Flip the relevant option in `my_run.yml` (`steps.skip_fastp: true`), **or**
2. Pass `--steps.skip_fastp true` on the CLI via Nextflow parameter overrides (e.g., `nextflow run ... --steps.skip_fastp true`).

Skipped stages expect their downstream inputs to exist already in the output folder—identical to the Bash pipeline’s behavior.

### 7. Inspect outputs

- `ASVs/ASV_counts.tsv`: ASV abundance table.
- `ASVs/ASV_filtered.tsv`: Post-filter version (if enabled).
- `ASVs/ASVs.fasta` / `ASVs_filtered.fasta`: Representative sequences.
- `logs/*.log`: Raw command logs for dereplication, denoising, etc.
- `nextflow.log`: Pipeline execution trace.

### 8. Troubleshooting

- **Missing tools**: Ensure the Conda environment is activated before running Nextflow.
- **No FASTQs detected**: Verify `filename_patterns` match your naming scheme (especially `R1/R2` tokens).
- **swarm not installed**: Either install it or set `steps.skip_swarm: true`.
- **Custom filter script**: Point `table_filter.script` to your Python script (relative paths are resolved from the YAML directory).

## Example Command Cheatsheet

```bash
# Default config in place
nextflow run asv_pipeline.nf

# Alternate config + limited CPUs
nextflow run asv_pipeline.nf --config configs/v4_batch.yml --resources.threads 8

# Resume partial run and skip swarm
nextflow run asv_pipeline.nf --config my_run.yml --steps.skip_swarm true -resume
```

## Why Nextflow?

- **Reproducibility**: Parameterization via YAML + immutable publish directories.
- **Portability**: Seamlessly scale from local workstation to HPC/backends.
- **Traceability**: Built-in reports (`-with-trace`, `-with-report`) complement the existing log files.
- **Modularity**: Each process can be toggled or extended without editing the main script.

Use this README as a dedicated reference for the Nextflow version while continuing to rely on `README.md` for the Bash workflow. Both pipelines share the same data-handling principles, so you can choose whichever orchestration layer suits the current task.
