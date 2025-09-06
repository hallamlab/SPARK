#!/usr/bin/env python3
"""
data_loss_sankey.py
Build Sankey diagrams for read/ASV flow with flexible I/O, grouping, and colors.

Two modes:
A) COMPUTE from pipeline TSVs (default)
B) MANUAL via --steps/--lmp-in/--lmp-out

Examples
--------
# A) Compute from files (defaults mirror your script paths/columns)
python data_loss_sankey.py \
  --data-dir /path/to/project \
  --sub-dir spark_combined_output \
  --metadata /path/to/project/ref_db/spark_metadata.tsv \
  --type-col type_group \
  --samp-col lmp_id \
  --keep-types "Oral Rinse,Lung Brush,BAL,Skin Brush,Scope Flush" \
  --fastq-stats stats/fastq_stats.tsv --fastq-id-suffix-underscores 4 \
  --filtered-stats stats/filtered_fastqs.tsv --filtered-id-suffix-underscores 2 \
  --asv-raw ASVs/ASV_counts.tsv --asv-id-suffix-underscores 2 \
  --asv-decon ASVs/ASV_target.decon.tsv \
  --asv-micro ASVs/ASV_target.micro.tsv \
  --palette "Scope Flush:#E69F00,Skin Brush:#CC79A7,Lung Brush:#009E73,BAL:#0072B2,Oral Rinse:#6A3D9A,Failed-QC:lightgray" \
  --title "Data Loss Flow" \
  --output-prefix metadata/data_loss_sankey --make-labeled --make-unlabeled

# B) Manual counts
python data_loss_sankey.py \
  --steps "Quality Control:123456,Error Correction:110000,Decontamination:98000,Off-Target Filtering:82000,Finished Data:76000" \
  --lmp-in "Oral Rinse:40000,Lung Brush:35000,BAL:28000,Skin Brush:12000,Scope Flush:8400" \
  --lmp-out "Oral Rinse:18000,Lung Brush:22000,BAL:24000,Skin Brush:9000,Scope Flush:5100" \
  --palette "Oral Rinse:#6A3D9A,Lung Brush:#009E73,BAL:#0072B2,Skin Brush:#CC79A7,Scope Flush:#E69F00" \
  --output-prefix out/sankey --make-labeled --make-unlabeled
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import pandas as pd
import plotly.graph_objects as go

# Optional aesthetics (kept simple; plots are Plotly HTML)
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Global aesthetics (safe no-ops if MPL not used) ----------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")


# =========================
# Utility parsers / helpers
# =========================
def parse_kv_csv(s: str, val_cast=int) -> Dict[str, object]:
    """
    Parse 'A:1,B:2' into dict. Whitespace tolerated. Empty string -> {}.
    """
    out: Dict[str, object] = {}
    if not s:
        return out
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Expected key:value pair, got '{item}'")
        k, v = item.split(':', 1)
        k = k.strip()
        v = v.strip()
        out[k] = val_cast(v) if val_cast is not None else v
    return out


def parse_steps_csv(s: str) -> Tuple[List[str], List[int]]:
    """
    Parse 'StepA:100,StepB:90,...' -> (['StepA','StepB',...],[100,90,...])
    """
    d = parse_kv_csv(s, val_cast=int)
    return list(d.keys()), list(d.values())


def extract_sample_id_from_path(path_str: str, suffix_underscores: Optional[int] = None,
                                regex: Optional[str] = None) -> str:
    """
    Extract a sample id from a file path.
    - If regex is provided: return first capturing group.
    - Else if suffix_underscores is provided: chop that many underscore-delimited tokens from end.
      e.g., name='ABC_1_2_3_4.fastq.gz', n=4 -> 'ABC'
    - Else return basename without extension(s).
    """
    import re as _re
    base = os.path.basename(path_str)
    if regex:
        m = _re.search(regex, path_str)
        if not m or not m.groups():
            raise ValueError(f"Regex did not match or capture a group: {regex} for {path_str}")
        return m.group(1)
    if suffix_underscores is not None:
        stem = base
        # Remove common extensions
        for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq', '.gz', '.tsv', '.csv', '.txt'):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        parts = stem.split('_')
        if len(parts) <= suffix_underscores:
            return parts[0]
        return '_'.join(parts[: len(parts) - suffix_underscores])
    # Fallback: strip extensions
    if '.' in base:
        return base.split('.')[0]
    return base


def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


# =========================
# I/O readers (compute mode)
# =========================
def read_metadata(path: Path, samp_col: str, type_col: str,
                  keep_types: Optional[Sequence[str]]) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=0)
    if keep_types:
        df = df[df[type_col].isin(keep_types)].copy()
    # Make sure sample ids are strings
    df[samp_col] = df[samp_col].astype(str)
    return df


def read_fastq_stats(path: Path, samp_col: str,
                     id_suffix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Expects columns: file, num_seqs
    """
    df = pd.read_csv(path, sep='\t', header=0)
    if 'file' not in df or 'num_seqs' not in df:
        raise ValueError(f"{path} must contain columns: file, num_seqs")
    df[samp_col] = df['file'].apply(lambda x: extract_sample_id_from_path(x, id_suffix_underscores, id_regex))
    out = df.groupby(samp_col, as_index=False)['num_seqs'].sum()
    return out


def read_asv_matrix(path: Path, samp_col: str,
                    id_suffix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Input: wide matrix (rows=ASVs, columns=samples), counts.
    Returns long: [ASV_ID, samp_col, count] with count>0
    """
    df = pd.read_csv(path, sep='\t', header=0, index_col=0)
    long_df = df.stack().reset_index()
    long_df.columns = ['ASV_ID', 'sample_raw', 'count']
    long_df = long_df[long_df['count'] > 0].copy()
    long_df[samp_col] = long_df['sample_raw'].apply(lambda x: extract_sample_id_from_path(x, id_suffix_underscores, id_regex))
    long_df.drop(columns=['sample_raw'], inplace=True)
    return long_df


def group_counts_by_type(long_counts: pd.DataFrame, metadata: pd.DataFrame,
                         samp_col: str, type_col: str) -> pd.DataFrame:
    merged = long_counts.merge(metadata[[samp_col, type_col]], on=samp_col, how='inner')
    grp = merged.groupby(type_col, as_index=False)['count'].sum()
    grp.rename(columns={'count': 'num_reads'}, inplace=True)
    return grp


# =========================
# Sankey construction
# =========================
def build_sankey(steps: List[str], counts: List[int],
                 lmp_in: Dict[str, int], lmp_out: Dict[str, int],
                 palette: Dict[str, str], title: str,
                 output_html: Path, labeled: bool) -> None:
    """
    Build and save a Plotly HTML sankey.
    """
    nodes: List[Dict[str, str]] = []
    links: List[Dict[str, int]] = []
    node_idx: Dict[Tuple[str, str], int] = {}  # (name, role) -> idx
    link_colors: List[str] = []

    # Input-type nodes
    for k, v in lmp_in.items():
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": palette.get(k, "black")})
        node_idx[(k, "in")] = len(nodes) - 1

    # Process nodes
    for step, cnt in zip(steps, counts):
        nodes.append({"label": f"{step} ({cnt})" if labeled else "", "color": "black"})
        node_idx[(step, "proc")] = len(nodes) - 1

    # Output-type nodes
    for k, v in lmp_out.items():
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": palette.get(k, "black")})
        node_idx[(k, "out")] = len(nodes) - 1

    # Links: input -> first step
    first_step = steps[0]
    for k, v in lmp_in.items():
        links.append({"source": node_idx[(k, "in")], "target": node_idx[(first_step, "proc")], "value": v})
        link_colors.append("grey")

    # Links: step -> next step (+ loss nodes)
    for i in range(len(steps) - 1):
        s, t = steps[i], steps[i + 1]
        links.append({"source": node_idx[(s, "proc")], "target": node_idx[(t, "proc")], "value": counts[i + 1]})
        link_colors.append("grey")

        if counts[i] > counts[i + 1]:
            loss_val = counts[i] - counts[i + 1]
            loss_label = f"Loss after {s} ({loss_val})" if labeled else ""
            nodes.append({"label": loss_label, "color": "lightgrey"})
            loss_idx = len(nodes) - 1
            links.append({"source": node_idx[(s, "proc")], "target": loss_idx, "value": loss_val})
            link_colors.append("lightgrey")

    # Links: last step -> outputs
    last_step = steps[-1]
    for k, v in lmp_out.items():
        links.append({"source": node_idx[(last_step, "proc")], "target": node_idx[(k, "out")], "value": v})
        link_colors.append("grey")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes],
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=link_colors,
        ),
    )])
    fig.update_layout(title_text=title, font_size=12)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"✔ Sankey saved: {output_html}")


# =========================
# CLI
# =========================
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Sankey diagrams for data loss/flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mode selection (manual vs compute)
    p.add_argument("--steps", default="", help="Manual mode: 'StepA:100,StepB:90,...' (order preserved)")
    p.add_argument("--lmp-in", default="", help="Manual mode: input groups 'Type:count,TypeB:count,...'")
    p.add_argument("--lmp-out", default="", help="Manual mode: output groups 'Type:count,TypeB:count,...'")

    # --- Compute mode inputs
    io = p.add_argument_group("Compute Mode Inputs")
    io.add_argument("--data-dir", type=Path, help="Project root (used to resolve defaults)")
    io.add_argument("--sub-dir", default="spark_combined_output", help="Subdir under data-dir for outputs/stats")
    io.add_argument("--metadata", type=Path, help="TSV with sample metadata")
    io.add_argument("--samp-col", default="lmp_id", help="Sample column name in metadata")
    io.add_argument("--type-col", default="type_group", help="Grouping column in metadata")
    io.add_argument("--keep-types", default="Oral Rinse,Lung Brush,BAL,Skin Brush,Scope Flush",
                    help="Comma-separated list; if empty, keep all types")

    io.add_argument("--fastq-stats", default="stats/fastq_stats.tsv",
                    help="Path (relative to sub-dir or absolute) to raw fastq stats TSV")
    io.add_argument("--fastq-id-suffix-underscores", type=int, default=4,
                    help="Chop N underscore tokens from end to form sample id (raw fastq)")
    io.add_argument("--fastq-id-regex", default="",
                    help="Regex with one capture group to extract sample id from raw fastq 'file' path")

    io.add_argument("--filtered-stats", default="stats/filtered_fastqs.tsv",
                    help="Path to filtered fastq stats TSV")
    io.add_argument("--filtered-id-suffix-underscores", type=int, default=2,
                    help="Chop N underscore tokens (filtered reads)")
    io.add_argument("--filtered-id-regex", default="", help="Regex for filtered sample id extraction")

    io.add_argument("--asv-raw", default="ASVs/ASV_counts.tsv", help="Wide ASV counts matrix")
    io.add_argument("--asv-id-suffix-underscores", type=int, default=2,
                    help="Chop N underscore tokens (ASV matrices)")
    io.add_argument("--asv-id-regex", default="", help="Regex for ASV sample id extraction")

    io.add_argument("--asv-decon", default="ASVs/ASV_target.decon.tsv", help="Wide ASV after decontamination")
    io.add_argument("--asv-micro", default="ASVs/ASV_target.micro.tsv", help="Wide ASV microbial (finished)")

    # --- Appearance / output
    out = p.add_argument_group("Output")
    out.add_argument("--palette", default="Scope Flush:#E69F00,Skin Brush:#CC79A7,Lung Brush:#009E73,BAL:#0072B2,Oral Rinse:#6A3D9A,Failed-QC:lightgray",
                     help="Comma-separated 'Group:#HEX' list")
    out.add_argument("--title", default="Data Loss Flow", help="Plot title")
    out.add_argument("--output-prefix", default="metadata/data_loss_sankey",
                     help="Output prefix ('.html' appended automatically)")
    out.add_argument("--make-labeled", action="store_true", help="Create labeled-node HTML")
    out.add_argument("--make-unlabeled", action="store_true", help="Create unlabeled-node HTML")

    # --- Misc
    p.add_argument("--verbose", action="store_true", help="Verbose logs")

    return p


def main():
    args = get_parser().parse_args()

    # Palette
    palette = parse_kv_csv(args.palette, val_cast=None) if args.palette else {}

    # If manual steps passed, run manual mode
    manual_mode = bool(args.steps.strip())
    if manual_mode:
        steps, counts = parse_steps_csv(args.steps)
        lmp_in = parse_kv_csv(args.lmp_in, val_cast=int)
        lmp_out = parse_kv_csv(args.lmp_out, val_cast=int)
        if not args.make_labeled and not args.make_unlabeled:
            args.make_labeled = True  # default to at least one output

        out_pref = Path(args.output_prefix)
        if args.make_labeled:
            build_sankey(steps, counts, lmp_in, lmp_out, palette, args.title, out_pref.with_suffix(".label.html"), True)
        if args.make_unlabeled:
            build_sankey(steps, counts, lmp_in, lmp_out, palette, args.title, out_pref.with_suffix(".html"), False)
        return

    # ---- Compute mode ----
    if not args.data_dir:
        raise SystemExit("--data-dir is required in compute mode")
    data_dir: Path = args.data_dir

    # Resolve default paths if relative
    def resolve(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return data_dir / args.sub_dir / rel_or_abs

    metadata_path = args.metadata or (data_dir / "ref_db" / "spark_metadata.tsv")
    fastq_stats_path = resolve(args.fastq_stats)
    filtered_stats_path = resolve(args.filtered_stats)
    asv_raw_path = resolve(args.asv_raw)
    asv_decon_path = resolve(args.asv_decon)
    asv_micro_path = resolve(args.asv_micro)

    keep_types = [t.strip() for t in args.keep_types.split(',')] if args.keep_types.strip() else None

    if args.verbose:
        print(f"[i] Metadata: {metadata_path}")
        print(f"[i] Raw fastq stats: {fastq_stats_path}")
        print(f"[i] Filtered stats: {filtered_stats_path}")
        print(f"[i] ASV raw: {asv_raw_path}")
        print(f"[i] ASV decon: {asv_decon_path}")
        print(f"[i] ASV micro: {asv_micro_path}")

    # Read metadata and filter by types
    meta = read_metadata(metadata_path, args.samp_col, args.type_col, keep_types)

    # Raw reads (pairs): sum num_seqs across files, then /2
    raw_df = read_fastq_stats(fastq_stats_path, args.samp_col,
                              args.fastq_id_suffix_underscores,
                              args.fastq_id_regex or None)
    raw_reads_total = int(raw_df['num_seqs'].sum() // 2)

    # Filtered reads (already single-end counts in your script)
    filt_df = read_fastq_stats(filtered_stats_path, args.samp_col,
                               args.filtered_id_suffix_underscores,
                               args.filtered_id_regex or None)
    filt_reads_total = int(filt_df['num_seqs'].sum())

    # ASV matrices -> long -> merge -> sum
    asv_raw_long = read_asv_matrix(asv_raw_path, args.samp_col,
                                   args.asv_id_suffix_underscores, args.asv_id_regex or None)
    asv_decon_long = read_asv_matrix(asv_decon_path, args.samp_col,
                                     args.asv_id_suffix_underscores, args.asv_id_regex or None)
    asv_micro_long = read_asv_matrix(asv_micro_path, args.samp_col,
                                     args.asv_id_suffix_underscores, args.asv_id_regex or None)

    # Sum by type (group)
    raw_by_type = raw_df.merge(meta[[args.samp_col, args.type_col]], on=args.samp_col, how='inner') \
                        .groupby(args.type_col, as_index=False)['num_seqs'].sum()
    raw_by_type['num_reads'] = (raw_by_type['num_seqs'] // 2).astype(int)

    filt_by_type = filt_df.merge(meta[[args.samp_col, args.type_col]], on=args.samp_col, how='inner') \
                          .groupby(args.type_col, as_index=False)['num_seqs'].sum()
    filt_by_type['num_reads'] = filt_by_type['num_seqs'].astype(int)

    asv_raw_by_type = group_counts_by_type(asv_raw_long, meta, args.samp_col, args.type_col)
    asv_decon_by_type = group_counts_by_type(asv_decon_long, meta, args.samp_col, args.type_col)
    asv_micro_by_type = group_counts_by_type(asv_micro_long, meta, args.samp_col, args.type_col)

    # Totals (match original semantics)
    asv_raw_reads = int(asv_raw_by_type['num_reads'].sum())
    asv_decon_reads = int(asv_decon_by_type['num_reads'].sum())
    asv_micro_reads = int(asv_micro_by_type['num_reads'].sum())

    # Steps & counts
    steps = ['Quality Control', 'Error Correction', 'Decontamination',
             'Off-Target Filtering', 'Finished Data']
    counts = [raw_reads_total, filt_reads_total, asv_raw_reads, asv_decon_reads, asv_micro_reads]

    # Groups to carry through (types)
    if keep_types:
        types = keep_types
    else:
        types = list(sorted(meta[args.type_col].unique()))

    # Input and output dicts for sankey ends
    lmp_in = {t: int(raw_by_type.loc[raw_by_type[args.type_col] == t, 'num_reads'].sum()) for t in types}
    lmp_out = {t: int(asv_micro_by_type.loc[asv_micro_by_type[args.type_col] == t, 'num_reads'].sum()) for t in types}

    if args.verbose:
        print("[i] Steps:")
        for s, c in zip(steps, counts):
            print(f"  - {s}: {c}")
        print("[i] Inputs by type:", lmp_in)
        print("[i] Outputs by type:", lmp_out)

    # Outputs
    out_pref = (args.data_dir / args.sub_dir / args.output_prefix) if args.data_dir else Path(args.output_prefix)
    # default: generate both if none chosen
    if not args.make_labeled and not args.make_unlabeled:
        args.make_labeled = True
        args.make_unlabeled = True

    if args.make_labeled:
        build_sankey(steps, counts, lmp_in, lmp_out, palette, args.title, out_pref.with_suffix(".label.html"), True)
    if args.make_unlabeled:
        build_sankey(steps, counts, lmp_in, lmp_out, palette, args.title, out_pref.with_suffix(".html"), False)


if __name__ == "__main__":
    main()
