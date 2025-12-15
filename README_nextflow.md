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
| 8 | `SWARM_CLUSTER` | Executes `swarm` for additional OTU-style clustering using `swarm -d <distance>` while emitting the companion stats/network outputs. | `swarm/swarms` + reports |
| 9 | `CREATE_COUNT_MATRIX` | Copies non-chimeric centroids to `ASVs.fasta` and maps concatenated reads back with `vsearch --usearch_global` to form an ASV count table. | `ASVs/ASV_counts.tsv`, `ASVs/ASVs.fasta`, `logs/count.log` |
| 10 | `FILTER_TABLE` | Runs the Python filter script on the ASV table and FASTA to enforce abundance thresholds. | `ASVs/ASV_filtered.tsv`, `ASVs/ASVs_filtered.fasta` |

### Channel wiring at a glance
1. Input FASTQ files → `FASTP_QC` (optional) → `MERGE_READS` → `FILTER_READS` → `RELABEL_FASTA`
2. Relabeled FASTA files are concatenated using `collectFile` (Nextflow DSL2 `storeDir`) into `concat/concat.fasta`.
3. Downstream processes (`DEREPLICATE`, `DENOISE`, `CHIMERA_CHECK`, `SWARM_CLUSTER`, `CREATE_COUNT_MATRIX`, `FILTER_TABLE`) consume fan-out copies of the concatenated and filtered datasets. Nextflow’s caching (`-resume`) handles restarts instead of manual skip toggles.

## Directory Layout

Each run populates the following structure under `output_dir` (by default inherited from the Bash pipeline):

```
output_dir/
├── fastp/
├── merged/
├── filtered/
├── concat/            # concat.fasta, concat_counts.fasta
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
- `swarm`: `distance` parameter for the optional OTU-style clustering radius.
- `table_filter`: thresholds and override path for the filtering script.
- `filename_patterns`: regex tokens used to identify R1/R2 files and sanitize sample names.
- `environments`: map of logical names to Conda/Mamba YAML definitions under `SPARK/envs/`.
- `config_root` *(optional when using `--params-file`)*: base directory to resolve relative paths in the inline configuration (defaults to the repository root).

**Tips**
- Use absolute paths or relative paths evaluated from the YAML file’s directory.
- Set `resources.threads` if you want to cap CPU usage rather than letting Nextflow use all available cores.

## Environment YAMLs (`envs/`)

All runtime environments used by Nextflow live under `SPARK/envs/`. The default file `envs/asv_pipeline.yml` mirrors the legacy `environment.yml`, so every process can run `fastp`, `vsearch`, `swarm`, Python tooling, etc. To introduce alternative stacks (e.g., a chimera-only env), drop a YAML file into `envs/` and reference it via the new `environments` block:

```yaml
environments:
  main: envs/asv_pipeline.yml
  chimera_freeze: envs/chimera.yml
```

Nextflow reads `environments.main` and applies it to every process via the `conda` directive, creating the env on demand (with Mamba/Micromamba/Conda) and caching it for reuse. This removes the need for manual `mamba env create` steps while keeping all future env specs centralized.

## Quick Start

1. **Install Mamba (strongly recommended):**  
   ```bash
   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
   bash Mambaforge-Linux-x86_64.sh -b -p $HOME/mambaforge
   export PATH="$HOME/mambaforge/bin:$PATH"
   ```
2. **Create a lightweight Nextflow runner env (one-time):**  
   ```bash
   mamba create -n nextflow -c conda-forge -c bioconda nextflow openjdk=17
   mamba activate nextflow
   ```
3. **Redirect the `.nextflow` cache to local storage (one-time):**  
   ```bash
   mkdir -p /home/ryan/.nextflow/cache /home/ryan/.nextflow/work
   cd /home/ryan/SABer_dat/SI_data/SI_ASV/SPARK
   rm -rf .nextflow
   ln -s /home/ryan/.nextflow .nextflow
   ```
   By symlinking the repository’s `.nextflow` folder to your home directory, Nextflow writes its cache metadata to local disk automatically—no environment variables or extra flags per run.
4. **Copy the config:** `cp asv_pipeline_nextflow.yml my_run.yml` and edit the `paths` + `environments` blocks to fit your dataset.  
5. **Dry run:** `nextflow run asv_pipeline.nf --config my_run.yml -preview` to confirm wiring (you can also use `--params-file my_run.yml` now that inline configs are supported). Thanks to the repo-level defaults, the work directory automatically points to `/home/ryan/.nextflow/work`.  
6. **Launch for real:** `nextflow run asv_pipeline.nf --config my_run.yml -profile standard`. The bundled `nextflow.config` already enables Mamba/Conda, so passing `-with-conda` is optional. Add `-resume` when re-running after tweaks to reuse cached process outputs.

### Repository defaults (`nextflow.config`)

`SPARK/nextflow.config` enforces two behaviors out of the box:

- `workDir = /home/ryan/.nextflow/work` unless you override it via `-work-dir` or `NXF_WORK`.
- `conda.enabled = true`, `conda.useMamba = true`, and `conda.mamba = bin/nxf_mamba.sh`, which mirrors running every command with `-with-conda` while routing the solver through a wrapper that translates Nextflow’s `--yes` flag for older Mamba builds.

You can still override either behavior per run (e.g., different work dir, disable Conda entirely), but the defaults allow `nextflow run asv_pipeline.nf --config my_run.yml` to “just work”.

Once you’ve done this once, the cached environment makes subsequent runs nearly instant to start.

## Usage Tutorial

### 1. Install prerequisites

- **Nextflow** ≥ 23.x (`curl -s https://get.nextflow.io | bash`).
- **Mamba** (preferred) or **Micromamba/Conda** reachable on your `PATH`.
- Optional: a base shell environment for running ancillary scripts or visualizations.

### 2. Ensure Nextflow can find your package manager and local cache/work dirs

- Make sure `mamba` (or `conda`) is on your `PATH`. If you used the Quick Start above, activating the `nextflow` environment is sufficient.
- Run the one-time symlink step from the Quick Start (Step 3). After that, the repository’s `.nextflow` directory will automatically point to your local cache, and the bundled `nextflow.config` keeps the work directory local without additional flags.

### 3. Copy & edit the YAML config

```bash
cp asv_pipeline_nextflow.yml my_run.yml
# Edit paths, thresholds, env overrides, etc.
```

Ensure `paths.input_dir` points to the FASTQ directory, `paths.output_dir` is writable (or resumable), and adjust `environments.main` if you want to test a different YAML in `envs/`. Use `--config my_run.yml` when you want the pipeline to load that file directly, or pass the YAML via `--params-file my_run.yml` if you prefer to inline the parameters—the workflow detects inline configs automatically (set `config_root` in the YAML if you need relative paths to resolve from somewhere other than the project directory).

### 4. Run a dry run (optional but recommended)

```bash
nextflow run asv_pipeline.nf --config my_run.yml \
    -preview
```

The `-preview` switch validates the config, prints the plan, and ensures all required scripts exist without launching tasks. Override `-work-dir` only when you need to place intermediates somewhere other than the default `/home/ryan/.nextflow/work`.

### 5. Launch the pipeline (with automatic env provisioning)

```bash
nextflow run asv_pipeline.nf --config my_run.yml \
    -profile standard \
    -with-report reports/asv_report.html \
    -with-trace reports/asv_trace.txt
```

- Use `-resume` to restart from cached results if the run fails or you tweak non-breaking parameters.
- Override work/output locations with `-work-dir` or custom YAML paths when needed.

### 6. Resume with caching

Nextflow automatically checkpoints every process result inside the `work/` directory. When a run stops (intentionally or due to failure), simply re-launch the same command with `-resume` and Nextflow will reuse every completed step instead of re-running the process. No manual skip toggles are needed—channel wiring remains identical regardless of how many stages restart.

### 7. Inspect outputs

- `ASVs/ASV_counts.tsv`: ASV abundance table.
- `ASVs/ASV_filtered.tsv`: Post-filter version (if enabled).
- `ASVs/ASVs.fasta` / `ASVs_filtered.fasta`: Representative sequences.
- `logs/*.log`: Raw command logs for dereplication, denoising, etc.
- `nextflow.log`: Pipeline execution trace.

### 8. Troubleshooting

- **Missing tools**: Double-check `envs/asv_pipeline.yml` includes the binaries you need and that `mamba`/`conda` is on your `PATH` (the repo defaults enable Conda automatically).
- **No FASTQs detected**: Verify `filename_patterns` match your naming scheme (especially `R1/R2` tokens).
- **swarm not installed**: Ensure the `envs/asv_pipeline.yml` (or overridden env) includes `swarm`, or install it into your active Conda/Mamba base environment.
- **Custom filter script**: Point `table_filter.script` to your Python script (relative paths are resolved from the YAML directory).

## Example Command Cheatsheet

```bash
# Default config in place
nextflow run asv_pipeline.nf

# Alternate config + limited CPUs
nextflow run asv_pipeline.nf --config configs/v4_batch.yml --resources.threads 8

# Resume partial run with cached steps
nextflow run asv_pipeline.nf --config my_run.yml -resume
```

## Why Nextflow?

- **Reproducibility**: Parameterization via YAML + immutable publish directories.
- **Portability**: Seamlessly scale from local workstation to HPC/backends.
- **Traceability**: Built-in reports (`-with-trace`, `-with-report`) complement the existing log files.
- **Modularity**: Each process remains isolated in DSL2 modules, so extending parameters or swapping tools does not require editing the downstream logic.

Use this README as a dedicated reference for the Nextflow version while continuing to rely on `README.md` for the Bash workflow. Both pipelines share the same data-handling principles, so you can choose whichever orchestration layer suits the current task.
