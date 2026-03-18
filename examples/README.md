# Nextflow Templates

This folder contains copy-ready templates for running `asv_pipeline.nf` on a new dataset.

## Files

- `my_run.min.yml`
  - Minimal config for core ASV + taxonomy only.
  - Most optional modules are disabled.

- `my_run.full.yml`
  - Config template aligned with the default metadata + mito workflow.
  - Includes required placeholders for metadata and custom BLAST DBs.

- `manifest.example.tsv`
  - Manifest format used by the pipeline.
  - **No header** in machine-readable rows (comment lines beginning with `#` are allowed).
  - Columns: `sample_id`, `fastq_r1`, `fastq_r2` (optional for single-end).

- `metadata.example.tsv`
  - Minimal metadata starter table compatible with default `filter_counts`, `metadata_plots`, and `sankey` settings.

## Quick Usage

1. Copy a config template into the SPARK root:

```bash
cd SPARK
cp examples/my_run.full.yml my_run.yml
```

2. Copy and edit input templates:

```bash
mkdir -p ref_db
cp examples/manifest.example.tsv ref_db/sample_manifest.tsv
cp examples/metadata.example.tsv ref_db/asv_cruise_metadata.tsv
```

3. Edit `my_run.yml` and replace all `/abs/path/...` placeholders.

4. Validate and run:

```bash
./run_asv_pipeline.sh --help
./run_asv_pipeline.sh my_run.yml
```

## Metadata Columns: What Is Required

For the default-enabled metadata-related modules in `my_run.full.yml`, these columns are required:

- `sampleID`: joins metadata to manifest/pipeline sample IDs.
- `Depth`: grouping variable for filtering and most plots.
- `Color`: color mapping key used by plotting scripts.

Useful optional columns for expanded analysis:

- `Cruise`, `Year`, `Month`, `Day`, `Season`, `plateID`, `Depth_anchored`.

If you enable `power_analysis.enabled`, you will also typically need:

- `Participant_ID`: patient/blocking identifier.
- `Case`: case-vs-control grouping column.
- `type_group`: sample-type grouping column.
- `lung_status` plus any side/site columns used in your dataset if you want the contralateral filtering options to be meaningful.

## Manifest Rules (Important)

- Use tab-separated values.
- Use one line per sample.
- R1 path is required; R2 is optional only when running single-end (`resources.single_end: true`).
- Paths may be absolute or relative to the manifest file location.
- The pipeline validates that each listed FASTQ path exists before execution.
