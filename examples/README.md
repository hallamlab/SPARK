# Nextflow Templates

This folder contains copy-ready templates for running `asv_pipeline.nf` on a new dataset.

## Files

- `my_run.min.yml`
  - Minimal config for core ASV + taxonomy only.
  - Most optional modules are disabled.

- `my_run.full.yml`
  - Config template aligned with the default metadata + mito workflow.
  - Includes required placeholders for metadata and custom BLAST DBs.

- `my_run.patient_aware_test.yml`
  - Minimal test template for the new patient-aware branches.
  - Enables `bray_patient_aware`, `taxonomy_patient_aware`, and `lung_status_analysis`.
  - Disables unrelated optional downstream modules to keep the test run small.

- `my_run.asv_mag_test.yml`
  - Minimal test template for the ASV-to-genome barrnap linkage branch.
  - Enables `asv_mag_link` and `master_summary`.
  - Disables unrelated optional downstream modules to keep the test run focused.

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

If you enable `bray_patient_aware.enabled`, you need the same patient-aware metadata columns as above. This branch builds a fresh master long table from the run outputs, then runs blocked PERMANOVA and PERMDISP summaries plus companion figures.

If you enable `taxonomy_patient_aware.enabled`, you also need the same patient-aware metadata columns, plus a usable count column in the generated master long table. This branch runs observed-data taxonomic case/control and sample-type comparisons, then produces summary heatmaps and boxplots.

If you enable `lung_status_analysis.enabled`, your metadata should include the lung-side fields required to assign `TumorSide`, `Contralateral`, and `Healthy`. The prep step now prefers explicit metadata columns for those labels when present, and otherwise falls back to deriving them from `Case`, `Cancer_Site`, and `lung_code`.

If you enable `indicspecies.aligned_plot_enabled`, no extra input files are required beyond a successful `indicspecies.enabled: true` run.

If you want more than two ISA groupings downstream, add extra entries to `indicspecies.group_cols` and then use matching `groupN_*` network / biochem overlay modes in the same configured order. You can also provide per-group defaults with:

- `indicspecies.group_palettes`
- `indicspecies.group_orders`
- `indicspecies.focus_labels`
- `spieceasi.isa_overlay_groups`
- `biochem_network_overlay.isa_overlay_groups`

If you enable `asv_mag_link.enabled`, the cleanest input is `asv_mag_link.genome_qc_dir` or `asv_mag_link.genome_qc_dirs`. In that mode SPARK autodetects `barrnap/`, prefers final QC MAGs from `dedupe/fasta` (falling back to `genomes_subset`), restricts the eligible references to MAGs that pass the genome-QC barrnap check, and enriches the ASV↔MAG tables with MAG metadata such as completeness, contamination, MIMAG tier, taxonomy, and source provenance. If you are not working from a genome_qc result directory, you can still provide `asv_mag_link.barrnap_dir` directly, and optionally `asv_mag_link.genome_fasta_dir` when only barrnap `*.gff`/`*.gff3` outputs are available.

## Manifest Rules (Important)

- Use tab-separated values.
- Use one line per sample.
- R1 path is required; R2 is optional only when running single-end (`resources.single_end: true`).
- Paths may be absolute or relative to the manifest file location.
- The pipeline validates that each listed FASTQ path exists before execution.
