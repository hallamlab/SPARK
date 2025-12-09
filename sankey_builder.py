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
  --fastq-stats stats/fastq_stats.tsv --fastq-id-prefix-underscores 2 \
  --filtered-stats stats/filtered_fastqs.tsv --filtered-id-prefix-underscores 2 \
  --asv-raw ASVs/ASV_counts.tsv --asv-id-prefix-underscores 2 \
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


def extract_sample_id_from_path(path_str: str, prefix_underscores: Optional[int] = None,
                                regex: Optional[str] = None) -> str:
    """
    Extract a sample id from a file path.
    - If regex is provided: return first capturing group.
    - Else if prefix_underscores is provided: keep the first N underscore-delimited tokens.
      e.g., name='ABC_1_2_3_4.fastq.gz', n=2 -> 'ABC_1'
    - Else return basename without extension(s).
    """
    import re as _re
    base = os.path.basename(path_str)
    if regex:
        m = _re.search(regex, path_str)
        if not m or not m.groups():
            raise ValueError(f"Regex did not match or capture a group: {regex} for {path_str}")
        return m.group(1)
    if prefix_underscores is not None:
        stem = base
        # Remove common extensions (include fasta so filtered FASTA paths behave)
        for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq',
                    '.fasta.gz', '.fasta', '.fa.gz', '.fa',
                    '.gz', '.tsv', '.csv', '.txt'):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        parts = stem.split('_')
        # If we don't have more than prefix_underscores tokens, keep the full stem
        if prefix_underscores <= 0:
            return stem
        if len(parts) <= prefix_underscores:
            return stem
        # Otherwise, keep the first prefix_underscores tokens
        stem = '_'.join(parts[: prefix_underscores]).split('-', 1)[0]
        return stem
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
                     id_prefix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Expects columns: file, num_seqs
    Collapses replicates by sample ID via groupby+sum.
    """
    df = pd.read_csv(path, sep='\t', header=0)
    if 'file' not in df or 'num_seqs' not in df:
        raise ValueError(f"{path} must contain columns: file, num_seqs")
    df[samp_col] = df['file'].apply(
        lambda x: extract_sample_id_from_path(x, id_prefix_underscores, id_regex)
    )
    out = df.groupby(samp_col, as_index=False)['num_seqs'].sum()
    return out


def read_asv_matrix(path: Path, samp_col: str,
                    id_prefix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Input: wide matrix (rows=ASVs, columns=samples), counts.
    Returns long: [ASV_ID, samp_col, count] with count>0
    """
    df = pd.read_csv(path, sep='\t', header=0, index_col=0)
    long_df = df.stack().reset_index()
    long_df.columns = ['ASV_ID', 'sample_raw', 'count']
    long_df = long_df[long_df['count'] > 0].copy()
    long_df[samp_col] = long_df['sample_raw'].apply(
        lambda x: extract_sample_id_from_path(x, id_prefix_underscores, id_regex)
    )
    long_df.drop(columns=['sample_raw'], inplace=True)
    return long_df


def group_counts_by_type(long_counts: pd.DataFrame, metadata: pd.DataFrame,
                         samp_col: str, type_col: str) -> pd.DataFrame:
    """
    Merge counts with metadata and sum by type_col.
    Replicates with the same sample ID and type are naturally summed.
    """
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
    # Helper to safely get color with black as fallback
    def get_color(key: str) -> str:
        color = palette.get(key, "black")
        return color if color else "black"

    nodes: List[Dict[str, str]] = []
    links: List[Dict[str, int]] = []
    node_idx: Dict[Tuple[str, str], int] = {}
    link_colors: List[str] = []
    node_x: List[float] = []
    node_y: List[float] = []

    # Input-type nodes (left side, x=0.01)
    n_inputs = len(lmp_in)
    for i, (k, v) in enumerate(lmp_in.items()):
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": get_color(k)})
        node_idx[(k, "in")] = len(nodes) - 1
        node_x.append(0.01)
        # Give more vertical space - use power function to spread them out
        y_pos = (i / max(n_inputs - 1, 1)) ** 0.8  # Exponent < 1 spreads them more
        node_y.append(y_pos)

    # Process nodes (middle, evenly spaced)
    n_steps = len(steps)
    for i, (step, cnt) in enumerate(zip(steps, counts)):
        nodes.append({"label": f"{step}<br>({cnt})" if labeled else "", "color": "black"})
        node_idx[(step, "proc")] = len(nodes) - 1
        node_x.append(0.2 + (i / max(n_steps - 1, 1)) * 0.6)  # Spread from 0.2 to 0.8
        node_y.append(0.5)  # Center vertically

    # Output-type nodes (right side, x=0.99)
    n_outputs = len(lmp_out)
    for i, (k, v) in enumerate(lmp_out.items()):
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": get_color(k)})
        node_idx[(k, "out")] = len(nodes) - 1
        node_x.append(0.99)
        node_y.append(i / max(n_outputs - 1, 1))  # Evenly space from 0 to 1

    # Links: input -> first step
    first_step = steps[0]
    for k, v in lmp_in.items():
        links.append({
            "source": node_idx[(k, "in")],
            "target": node_idx[(first_step, "proc")],
            "value": v
        })
        link_colors.append("grey")

    # Links: step -> next step (+ loss nodes)
    for i in range(len(steps) - 1):
        s, t = steps[i], steps[i + 1]
        # main flow to next step
        links.append({
            "source": node_idx[(s, "proc")],
            "target": node_idx[(t, "proc")],
            "value": counts[i + 1]
        })
        link_colors.append("grey")

        # loss from this step
        if counts[i] > counts[i + 1]:
            loss_val = counts[i] - counts[i + 1]
            loss_label = f"{loss_val} removed" if labeled else ""
            nodes.append({"label": loss_label, "color": "lightgrey"})
            loss_idx = len(nodes) - 1
            node_x.append(0.2 + (i / max(n_steps - 1, 1)) * 0.6 + 0.05)  # Slightly offset
            node_y.append(0.9)  # Position loss nodes at bottom
            links.append({
                "source": node_idx[(s, "proc")],
                "target": loss_idx,
                "value": loss_val
            })
            link_colors.append("lightgrey")

    # Links: last step -> outputs
    last_step = steps[-1]
    for k, v in lmp_out.items():
        links.append({
            "source": node_idx[(last_step, "proc")],
            "target": node_idx[(k, "out")],
            "value": v
        })
        link_colors.append("grey")

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes],
            x=node_x,  # Explicit x positions
            y=node_y,  # Explicit y positions
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

    # --- Compute mode inputs
    io = p.add_argument_group("Compute Mode Inputs")
    io.add_argument("--data-dir", type=Path, help="Project root (used to resolve defaults)")
    io.add_argument("--sub-dir", default="spark_combined_output", help="Subdir under data-dir for outputs/stats")
    io.add_argument("--metadata", type=Path, help="TSV with sample metadata")
    io.add_argument("--samp-col", default="lmp_id", help="Sample column name in metadata")
    io.add_argument("--type-col", default="type_group", help="Grouping column in metadata")
    io.add_argument("--color-col", default="Color", help="Color column in metadata")
    io.add_argument("--keep-types", default="",
                    help="Comma-separated list; if empty, keep all types")

    io.add_argument("--fastq-stats", default="stats/fastq_stats.tsv",
                    help="Path (relative to sub-dir or absolute) to raw fastq stats TSV")
    io.add_argument("--fastq-id-prefix-underscores", type=int,
                    help="Keep first N underscore tokens for sample id (raw fastq)")
    io.add_argument("--fastq-id-regex", default="",
                    help="Regex with one capture group to extract sample id from raw fastq 'file' path")

    io.add_argument("--filtered-stats", default="stats/filtered_fastqs.tsv",
                    help="Path to filtered fastq stats TSV")
    io.add_argument("--filtered-id-prefix-underscores", type=int,
                    help="Keep first N underscore tokens (filtered reads)")
    io.add_argument("--filtered-id-regex", default="", help="Regex for filtered sample id extraction")

    io.add_argument("--asv-raw", default="ASVs/ASV_counts.tsv", help="Wide ASV counts matrix")
    io.add_argument("--asv-id-prefix-underscores", type=int,
                    help="Keep first N underscore tokens (ASV matrices)")
    io.add_argument("--asv-id-regex", default="", help="Regex for ASV sample id extraction")

    io.add_argument("--asv-decon", default="ASVs/ASV_target.decon.tsv", help="Wide ASV after decontamination")
    io.add_argument("--asv-micro", default="ASVs/ASV_target.micro.tsv", help="Wide ASV microbial (finished)")

    # --- Appearance / output
    out = p.add_argument_group("Output")
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

    # ASV matrices -> long -> merge -> sum
    # Use consistent sample ID parsing across all ASV matrices
    asv_micro_long = read_asv_matrix(
        asv_micro_path,
        args.samp_col,
        args.asv_id_prefix_underscores or None,
        args.asv_id_regex or None,
    )

    sample_list = asv_micro_long[args.samp_col].unique().tolist()

    asv_raw_long = read_asv_matrix(
        asv_raw_path,
        args.samp_col,
        args.asv_id_prefix_underscores or None,
        args.asv_id_regex or None,
    )
    asv_raw_long = asv_raw_long[asv_raw_long[args.samp_col].isin(sample_list)].copy()

    asv_decon_long = read_asv_matrix(
        asv_decon_path,
        args.samp_col,
        args.asv_id_prefix_underscores or None,
        args.asv_id_regex or None,
    )
    asv_decon_long = asv_decon_long[asv_decon_long[args.samp_col].isin(sample_list)].copy()

    # Read metadata and filter by types, then restrict to samples with ASV micro data
    meta = read_metadata(metadata_path, args.samp_col, args.type_col, keep_types)
    meta = meta[meta[args.samp_col].isin(sample_list)].copy()

    # Raw reads (pairs): sum num_seqs across files, then /2, with replicates collapsed
    raw_df = read_fastq_stats(
        fastq_stats_path,
        args.samp_col,
        args.fastq_id_prefix_underscores or None,
        args.fastq_id_regex or None
    )

    raw_df = raw_df[raw_df[args.samp_col].isin(sample_list)].copy()

    # Filtered reads (already single-end counts in your script), replicates collapsed
    filt_df = read_fastq_stats(
        filtered_stats_path,
        args.samp_col,
        args.filtered_id_prefix_underscores or None,
        args.filtered_id_regex or None
    )
    filt_df = filt_df[filt_df[args.samp_col].isin(sample_list)].copy()

    # Build palette from metadata: type_col -> color, with string keys
    palette = {str(t): str(c) for t, c in zip(meta[args.type_col], meta[args.color_col])}

    # Sort palette deterministically (numeric if possible, else lexical)
    try:
        palette = dict(sorted(palette.items(), key=lambda x: float(x[0])))
    except (ValueError, TypeError):
        palette = dict(sorted(palette.items()))

    # Sum by type (group) — this implicitly respects keep_types and drops samples without metadata
    raw_by_type = raw_df.merge(meta[[args.samp_col, args.type_col]],
                               on=args.samp_col, how='inner') #\
                        #.groupby(args.type_col, as_index=False)['num_seqs'].sum()    
    
    raw_by_type['num_reads'] = (raw_by_type['num_seqs'] // 2).astype(int)
    print(raw_by_type['num_reads'].sum())
    
    filt_by_type = filt_df.merge(meta[[args.samp_col, args.type_col]],
                                 on=args.samp_col, how='inner') \
                          .groupby(args.type_col, as_index=False)['num_seqs'].sum()
    filt_by_type['num_reads'] = filt_by_type['num_seqs'].astype(int)
    
    # Override step totals so node labels use exactly the subset represented in the ribbons
    raw_reads_total = int(raw_by_type['num_reads'].sum())
    filt_reads_total = int(filt_by_type['num_reads'].sum())

    asv_raw_by_type = group_counts_by_type(asv_raw_long, meta, args.samp_col, args.type_col)
    asv_decon_by_type = group_counts_by_type(asv_decon_long, meta, args.samp_col, args.type_col)
    asv_micro_by_type = group_counts_by_type(asv_micro_long, meta, args.samp_col, args.type_col)

    # Totals for remaining steps
    asv_raw_reads = int(asv_raw_by_type['num_reads'].sum())
    asv_decon_reads = int(asv_decon_by_type['num_reads'].sum())
    asv_micro_reads = int(asv_micro_by_type['num_reads'].sum())

    # Steps & counts (used for node labels and loss computation)
    steps = [
        'Quality Control',
        'Error Correction',
        'Decontamination',
        'Off-Target Filtering',
        'Finished Data'
    ]
    counts = [
        raw_reads_total,
        filt_reads_total,
        asv_raw_reads,
        asv_decon_reads,
        asv_micro_reads
    ]

    if keep_types:
        types = keep_types
    else:
        unique_types = meta[args.type_col].unique()
        # Sort numerically if possible, otherwise alphabetically
        try:
            types = sorted(unique_types, key=lambda x: float(x))
        except (ValueError, TypeError):
            types = sorted(unique_types)

    # Input and output dicts for sankey ends (string keys to match palette)
    lmp_in = {
        str(t): int(raw_by_type.loc[raw_by_type[args.type_col] == t, 'num_reads'].sum())
        for t in types
    }
    lmp_out = {
        str(t): int(asv_micro_by_type.loc[asv_micro_by_type[args.type_col] == t, 'num_reads'].sum())
        for t in types
    }

    # Sort by key (numeric if possible)
    try:
        lmp_in = dict(sorted(lmp_in.items(), key=lambda x: float(x[0])))
        lmp_out = dict(sorted(lmp_out.items(), key=lambda x: float(x[0])))
    except (ValueError, TypeError):
        lmp_in = dict(sorted(lmp_in.items()))
        lmp_out = dict(sorted(lmp_out.items()))

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
        build_sankey(
            steps, counts, lmp_in, lmp_out, palette,
            args.title, out_pref.with_suffix(".label.html"), True
        )
    if args.make_unlabeled:
        build_sankey(
            steps, counts, lmp_in, lmp_out, palette,
            args.title, out_pref.with_suffix(".html"), False
        )


if __name__ == "__main__":
    main()
