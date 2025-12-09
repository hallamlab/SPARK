#!/usr/bin/env python3
"""
asv_overlap.py
-------------
Unified CLI to generate UpSet & Venn-style set intersections for ASV data.
Supports:
- Microbial and/or mitochondrial runs
- Unique-membership UpSet (counts of ASVs)
- Abundance-weighted UpSet with stacked bars by category
- Optional Venn plots (can be disabled with --skip-venn)
- Presence tables (exclusive membership lists) and exclusive-sum tables
- Fully generalized grouping via metadata column
- Color mapping from metadata Color column

Examples
--------
# Basic run with depth groups
python asv_overlap.py \
  --data-dir /home/ryan/SeqData \
  --domain micro \
  --group-col Depth \
  --color-col Color \
  --formats svg,pdf

# Skip venn plots for many groups
python asv_overlap.py \
  --data-dir /data/run1 \
  --domain both \
  --group-col SampleType \
  --color-col Color \
  --skip-venn \
  --formats png

# Run with subset of groups
python asv_overlap.py \
  --data-dir /data/run1 \
  --domain micro \
  --group-col Depth \
  --color-col Color \
  --subset-groups "10,40,60" \
  --formats pdf
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
from upsetplot import from_contents, UpSet

# Optional venn backends
_HAVE_MPL_VENN = False
_HAVE_VENN = False
try:
    from matplotlib_venn import venn2, venn3  # 2–3 sets
    _HAVE_MPL_VENN = True
except Exception:
    pass
try:
    from venn import venn
    _HAVE_VENN = True
except Exception:
    pass

# ---------- Matplotlib / seaborn defaults ----------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12, 'font.family': 'Source Sans Pro'})
sns.set_theme()
sns.set_style("white")

# ---------- Utility ----------
def split_taxa_string(taxa_str: str, delimiter: str = ';') -> Dict[str, Optional[str]]:
    levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    if pd.isna(taxa_str) or taxa_str == 'Unassigned':
        parts = ['Unassigned']
    else:
        parts = [part.strip().split('__', 1)[1] for part in str(taxa_str).split(delimiter)]
    out = {}
    for i, lvl in enumerate(levels):
        out[lvl] = parts[i] if i < len(parts) else None
    return out

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def to_bool_series(members: Mapping[str, set], elements: Iterable[str]) -> pd.DataFrame:
    """
    Build an indicator matrix (rows=element, columns=groups) telling membership.
    """
    elements = list(elements)
    cols = list(members.keys())
    data = np.zeros((len(elements), len(cols)), dtype=bool)
    for j, g in enumerate(cols):
        s = members[g]
        data[:, j] = [e in s for e in elements]
    df = pd.DataFrame(data, index=elements, columns=cols)
    return df

def exclusive_sets(name_to_set: Mapping[str, set]) -> Dict[Tuple[str, ...], List[str]]:
    """
    For each non-empty combination of groups, compute exactly-there elements:
    intersection(included) minus union(excluded).
    Returns dict with tuple of names -> sorted list of element IDs.
    """
    groups = list(name_to_set.keys())
    out: Dict[Tuple[str, ...], List[str]] = {}
    for r in range(1, len(groups) + 1):
        for combo in combinations(groups, r):
            inc = set.intersection(*(name_to_set[n] for n in combo))
            exc_names = [n for n in groups if n not in combo]
            if exc_names:
                exc = set.union(*(name_to_set[n] for n in exc_names))
                members = inc - exc
            else:
                members = inc
            if members:
                out[tuple(combo)] = sorted(members)
    return out

def sort_groups(groups: Sequence[str]) -> List[str]:
    """Sort groups - numerically if possible, otherwise alphabetically."""
    try:
        # Try to sort as integers
        return sorted(groups, key=lambda x: int(x))
    except (ValueError, TypeError):
        try:
            # Try to sort as floats
            return sorted(groups, key=lambda x: float(x))
        except (ValueError, TypeError):
            # Fall back to string sorting
            return sorted(groups, key=str)

# ---------- IO layer ----------
class Inputs:
    def __init__(
        self,
        data_dir: Path,
        subdir: str,
        domain: str,  # "micro" or "mito"
        taxonomy_path: Optional[Path] = None
    ):
        self.data_dir = Path(data_dir)
        self.subdir = subdir
        self.domain = domain
        if domain == "micro":
            self.asv_raw = self.data_dir / subdir / "ASVs" / "ASV_target.micro.tsv"
            self.asv_final = self.data_dir / subdir / "ASVs" / "ASV_final.micro.tsv"
            self.meta = self.data_dir / subdir / "metadata" / "metadata_updated_micro.tsv"
            # taxonomy shared across domains by default
            self.tax = taxonomy_path or (self.data_dir / subdir / "taxonomy" / "ASV_SILVA_tax.full-length.vsearch.tsv")
            self.out_base = self.data_dir / subdir / "metadata"
        elif domain == "mito":
            self.asv_raw = self.data_dir / subdir / "mito" / "ASVs" / "ASV_target.mito.tsv"
            self.asv_final = self.data_dir / subdir / "mito" / "ASVs" / "ASV_final.mito.tsv"
            self.meta = self.data_dir / subdir / "mito" / "metadata" / "metadata_updated_mito.tsv"
            self.tax = taxonomy_path or (self.data_dir / subdir / "taxonomy" / "ASV_SILVA_tax.full-length.vsearch.tsv")
            self.out_base = self.data_dir / subdir / "mito" / "metadata"
        else:
            raise ValueError("domain must be 'micro' or 'mito'")

    def __repr__(self) -> str:
        return f"Inputs(domain={self.domain}, raw={self.asv_raw}, final={self.asv_final}, meta={self.meta}, tax={self.tax})"

def read_asv_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Normalize sample names (drop paths and read pair endings)
    return df

def read_metadata(path: Path, group_col: str, color_col: str) -> pd.DataFrame:
    md = pd.read_csv(path, sep="\t", dtype=str)
    # Validate required columns
    for col in [group_col, color_col]:
        if col not in md.columns:
            raise ValueError(f"Metadata missing required column: '{col}'. Available: {list(md.columns)}")
    return md

def read_taxonomy(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path, sep="\t")
    if "ASV_ID" in tx.columns:
        tx["ASV_ID"] = tx["ASV_ID"].astype(str).str.split(';', 1).str[0]
        tx = tx.set_index("ASV_ID")
    elif "Feature ID" in tx.columns:
        tx['Feature ID'] = tx['Feature ID'].astype(str).str.partition(';')[0]
        tx = tx.set_index("Feature ID")
    else:
        raise ValueError("taxonomy file must have ASV_ID or Feature ID column")
    return tx

def attach_taxonomy(long_df: pd.DataFrame, tax_df: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(tax_df, how="left", left_on="ASV_ID", right_index=True)
    # expand taxon string if present
    if "Taxon" in merged.columns:
        pieces = [split_taxa_string(x) for x in merged["Taxon"]]
        tax_exp = pd.DataFrame(pieces, index=merged.index)
        merged = pd.concat([merged, tax_exp], axis=1)
    return merged

# ---------- UpSet builders ----------
def build_upset_unique(group_sets: Mapping[str, set]) -> pd.Series:
    """Series with MultiIndex booleans -> counts (unique elements)."""
    return from_contents(group_sets)  # direct counts

def build_upset_weighted_rows(
    group_sets: Mapping[str, set],
    per_group_values: Mapping[Tuple[str, str], float],
    group_order: Sequence[str]
) -> pd.DataFrame:
    """
    Build a DataFrame suitable for UpSet(..., sum_over='count') and
    .add_stacked_bars(by='group', sum_over='count').
    Rows are ASV×group (only where ASV in that group's set).
    Columns are membership indicators for ALL groups (booleans),
    and two extra columns: 'group' and 'count'.
    """
    # union of all ASVs
    all_asvs = sorted(set().union(*group_sets.values())) if group_sets else []
    indicators = to_bool_series(group_sets, all_asvs)  # rows=ASV, cols=groups
    rows = []
    for g in group_order:
        members = group_sets.get(g, set())
        for asv in members:
            row = indicators.loc[asv].to_dict()
            row['ASV_ID'] = asv
            row['group'] = g
            row['count'] = float(per_group_values.get((g, asv), 0.0))
            rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure all indicator columns present in the chosen order
    df = df[[*group_order, 'ASV_ID', 'group', 'count']]
    return df

# ---------- Plotters ----------
def savefig_multi(fig: plt.Figure, out_path_base: Path, name: str, formats: Sequence[str]) -> None:
    for ext in formats:
        fig.savefig(out_path_base.with_name(f"{out_path_base.stem}_{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)

def plot_upset_unique(
    group_sets: Mapping[str, set],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
    group_col: str
) -> None:
    rev_grp_sets = {g: s for g, s in reversed(list(group_sets.items()))}
    data = build_upset_unique(rev_grp_sets)
    upset = UpSet(
        data,
        subset_size='count',
        element_size=None,
        show_counts=True,
        sort_categories_by='input',
        min_subset_size=0
    )
    for g, c in colors.items():
        upset.style_categories([g], bar_facecolor=c, bar_edgecolor="black")
    fig = plt.figure(figsize=(12, 8))
    mpl.rcParams["font.size"] = 6
    upset.plot(fig=fig)
    fig.suptitle(title, y=0.98)
    # legend
    handles = [Patch(facecolor=colors[g], edgecolor="black", label=g) for g in group_sets.keys()]
    fig.legend(handles=handles, title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    savefig_multi(fig, out_base, "upset", formats)

def plot_upset_weighted(
    group_sets: Mapping[str, set],
    per_group_values: Mapping[Tuple[str, str], float],
    group_order: Sequence[str],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
    group_col: str
) -> None:
    rev_grp_sets = {g: s for g, s in reversed(list(group_sets.items()))}
    df = build_upset_weighted_rows(rev_grp_sets, per_group_values, group_order)
    contents = {c: set(df.loc[df[c].astype(bool), 'ASV_ID']) for c in group_order[::-1]}
    weights = df.groupby(['ASV_ID', 'group'])['count'].sum().reset_index().set_index('ASV_ID')
    # Force custom stacking order here
    weights['group'] = pd.Categorical(weights['group'],
                                      categories=list(group_order),
                                      ordered=True)
    ser = from_contents(contents, data=weights)
    upset = UpSet(
        ser, sum_over='count', subset_size='sum',
        element_size=None, show_counts=True,
        sort_categories_by='input', min_subset_size=0,
        intersection_plot_elements=0
    )
    for g, c in colors.items():
        upset.style_categories([g], bar_facecolor=c, bar_edgecolor="black")
    upset.add_stacked_bars(by="group", sum_over="count", colors=colors, title=f"Abundance by {group_col}", elements=10)
    fig = plt.figure(figsize=(12, 8))
    mpl.rcParams["font.size"] = 6
    axes = upset.plot(fig=fig)
    # fix legend ordering
    ax_extra = axes.get('extra0', None)
    if ax_extra is not None:
        handles, labels = ax_extra.get_legend_handles_labels()
        order = list(group_order)
        handles = [handles[labels.index(o)] for o in order if o in labels]
        labels = [o for o in order if o in labels]
        ax_extra.legend(handles, labels, title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.suptitle(title, y=1.02)
    savefig_multi(fig, out_base, "upset_weighted", formats)

def plot_venn(
    group_sets: Mapping[str, set],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
    weighted_labels: Optional[Mapping[Tuple[str, ...], float]] = None
) -> None:
    """
    3 sets => matplotlib-venn (if available)
    4–6 sets => venn.draw_venn (if available)
    """
    names = list(group_sets.keys())
    sets = [group_sets[n] for n in names]
    
    if len(names) == 2 and _HAVE_MPL_VENN:
        fig = plt.figure(figsize=(6, 6))
        venn2([sets[0], sets[1]], tuple(names),
              set_colors=(colors[names[0]], colors[names[1]]), alpha=0.6)
        plt.title(title)
        savefig_multi(fig, out_base, "venn", formats)
        return
    
    if len(names) == 3 and _HAVE_MPL_VENN:
        fig = plt.figure(figsize=(6, 6))
        if weighted_labels is None:
            venn3([sets[0], sets[1], sets[2]], tuple(names),
                  set_colors=(colors[names[0]], colors[names[1]], colors[names[2]]), alpha=0.6)
        else:
            # matplotlib-venn supports dict 'subsets' with binary keys like '100','010',...
            label_map = {
                (names[0],): '100',
                (names[1],): '010',
                (names[2],): '001',
                (names[0], names[1]): '110',
                (names[0], names[2]): '101',
                (names[1], names[2]): '011',
                (names[0], names[1], names[2]): '111',
            }
            subsets = {}
            for combo, val in weighted_labels.items():
                key = tuple(sorted(combo))
                if key in label_map:
                    subsets[label_map[key]] = float(val)
            venn3(subsets=subsets, set_labels=tuple(names),
                  set_colors=(colors[names[0]], colors[names[1]], colors[names[2]]), alpha=0.6)
        plt.title(title)
        savefig_multi(fig, out_base, "venn", formats)
        return
    
    if 4 <= len(names) <= 6 and _HAVE_VENN:
        labels2sets = {n: set(s) for n, s in zip(names, sets)}
        if not any(len(s) for s in labels2sets.values()):
            print("All Venn sets are empty; skipping Venn plot.")
        else:
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            ax = venn(
                labels2sets,
                ax=ax,
                cmap=[colors[n] for n in names],
                fontsize=8,
                alpha=0.45,
            )
            ax.set_title(title)
            savefig_multi(fig, out_base, "venn", formats)
            plt.close(fig)
            return
    
    # Fallback: skip venn if no backend or too many groups
    print(f"[WARN] Venn plotting not available for {len(names)} sets (matplotlib-venn or venn not installed, or >6 sets). Skipping.")

def write_presence_and_sums(
    name_to_set: Mapping[str, set],
    per_asv_total: Mapping[str, float],
    out_base: Path,
    fname_prefix: str
) -> None:
    """
    Write two TSVs:
      - <prefix>_venn_presence_table.tsv : rows = (grouping label, ASV_ID)
      - <prefix>_venn_sum_table.tsv      : rows = (grouping label, Sum_count)
    """
    exc = exclusive_sets(name_to_set)
    # Presence
    rows = [[(" + ".join(k) if len(k) > 1 else k[0]), asv] for k, ids in exc.items() for asv in ids]
    pres = pd.DataFrame(rows, columns=["grouping", "ASV_ID"])
    pres.to_csv(out_base.with_name(f"{out_base.stem}_{fname_prefix}_venn_presence_table.tsv"), sep="\t", index=False)
    # Sums
    rows2 = []
    for k, ids in exc.items():
        s = float(sum(per_asv_total.get(a, 0.0) for a in ids))
        rows2.append([(" + ".join(k) if len(k) > 1 else k[0]), s])
    sums = pd.DataFrame(rows2, columns=["grouping", "Sum_count"])
    sums.to_csv(out_base.with_name(f"{out_base.stem}_{fname_prefix}_venn_sum_table.tsv"), sep="\t", index=False)

# ---------- Pipeline per domain ----------
def run_domain(
    inp: Inputs,
    id_col: str,
    group_col: str,
    color_col: str,
    subset_groups: Optional[Sequence[str]],
    use_raw: bool,
    use_final: bool,
    skip_venn: bool,
    formats: Sequence[str],
) -> None:
    ensure_dir(inp.out_base)
    
    # Read
    raw_df = read_asv_table(inp.asv_raw) if use_raw else None
    final_df = read_asv_table(inp.asv_final) if use_final else None
    md = read_metadata(inp.meta, group_col, color_col)
    tx = read_taxonomy(inp.tax)
    
    # Build palette from metadata (group -> color mapping)
    palette_df = md[[group_col, color_col]].drop_duplicates()
    palette = dict(zip(palette_df[group_col], palette_df[color_col]))
    
    # Sort groups
    all_groups = sort_groups(list(palette.keys()))
    
    # Filter to subset if specified
    if subset_groups:
        all_groups = [g for g in all_groups if g in subset_groups]
        palette = {g: palette[g] for g in all_groups if g in palette}
    
    print(f"[{inp.domain}] Groups (in order): {all_groups}")
    print(f"[{inp.domain}] Palette: {palette}")
    
    # Long format helper
    def melt_counts(count_df: pd.DataFrame) -> pd.DataFrame:
        st = count_df.stack().reset_index()
        st.columns = ['ASV_ID', id_col, 'count']
        return st
    
    # Process raw data
    if use_raw and raw_df is not None:
        raw_long = melt_counts(raw_df)
        raw_long = raw_long.merge(md, how="left", on=id_col)
        raw_pos = raw_long[raw_long['count'] > 0].copy()
        raw_tx = attach_taxonomy(raw_pos, tx)
        
        # Per-ASV totals and per-(group, ASV) totals
        raw_asv_total = raw_tx.groupby("ASV_ID")["count"].sum().to_dict()
        raw_group_asv_total = raw_tx.groupby([group_col, "ASV_ID"])["count"].sum().to_dict()
        
        # Build sets
        groups_present = [g for g in all_groups if g in set(raw_tx[group_col].unique())]
        group_sets = {g: set(raw_tx.loc[raw_tx[group_col] == g, 'ASV_ID']) for g in groups_present}
        
        if len(group_sets) >= 2:
            base = inp.out_base / f"raw_{inp.domain}"
            
            # UpSet unique
            plot_upset_unique(group_sets, {g: palette[g] for g in groups_present},
                            f"ASV Membership by {group_col} (Raw)", base, formats, group_col)
            
            # UpSet weighted
            plot_upset_weighted(group_sets, raw_group_asv_total, groups_present,
                              {g: palette[g] for g in groups_present},
                              f"ASV Abundance by {group_col} (Raw)", base, formats, group_col)
            
            # Venn (optional)
            if not skip_venn:
                plot_venn(group_sets, {g: palette[g] for g in groups_present},
                         f"Venn: ASV Membership (Raw)", base, formats)
            
            # Tables
            write_presence_and_sums(group_sets, raw_asv_total, base, "raw")
        else:
            print(f"[{inp.domain}] Skipping raw suite (need ≥2 present groups).")
    
    # Process final data
    if use_final and final_df is not None:
        final_long = melt_counts(final_df)
        final_long = final_long.merge(md, how="left", on=id_col)
        final_pos = final_long[final_long['count'] > 0].copy()
        fin_tx = attach_taxonomy(final_pos, tx)
        
        # Per-ASV totals and per-(group, ASV) totals
        fin_asv_total = fin_tx.groupby("ASV_ID")["count"].sum().to_dict()
        fin_group_asv_total = fin_tx.groupby([group_col, "ASV_ID"])["count"].sum().to_dict()
        
        # Build sets
        groups_present = [g for g in all_groups if g in set(fin_tx[group_col].unique())]
        group_sets = {g: set(fin_tx.loc[fin_tx[group_col] == g, 'ASV_ID']) for g in groups_present}
        
        if len(group_sets) >= 2:
            base = inp.out_base / f"final_{inp.domain}"
            
            # UpSet unique
            plot_upset_unique(group_sets, {g: palette[g] for g in groups_present},
                            f"ASV Membership by {group_col} (Final)", base, formats, group_col)
            
            # UpSet weighted
            plot_upset_weighted(group_sets, fin_group_asv_total, groups_present,
                              {g: palette[g] for g in groups_present},
                              f"ASV Abundance by {group_col} (Final)", base, formats, group_col)
            
            # Venn (optional)
            if not skip_venn:
                plot_venn(group_sets, {g: palette[g] for g in groups_present},
                         f"Venn: ASV Membership (Final)", base, formats)
            
            # Tables
            write_presence_and_sums(group_sets, fin_asv_total, base, "final")
        else:
            print(f"[{inp.domain}] Skipping final suite (need ≥2 present groups).")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ASV overlap (UpSet & Venn) plotter")
    ap.add_argument("--data-dir", required=True, help="Project root directory")
    ap.add_argument("--subdir", default="spark_output", help="Root subdir with outputs")
    ap.add_argument("--domain", choices=["micro", "mito", "both"], default="micro",
                    help="Which domain(s) to run")
    ap.add_argument("--taxonomy-path", default=None,
                    help="Optional path to taxonomy, should point to <subdir>/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv")
    
    # Grouping and colors
    ap.add_argument("--sample-id-col", default="sampleID", help="Metadata column with sample IDs")
    ap.add_argument("--group-col", required=True, help="Metadata column for grouping (e.g., Depth, SampleType)")
    ap.add_argument("--color-col", default="Color", help="Metadata column with color values (hex codes)")
    ap.add_argument("--subset-groups", default=None,
                    help="Optional comma-separated list of groups to include (subset of all groups)")
    
    # Data selection
    ap.add_argument("--use-raw", action="store_true", default=True, help="Process raw ASV data")
    ap.add_argument("--use-final", action="store_true", default=True, help="Process final ASV data")
    ap.add_argument("--raw-only", action="store_true", help="Only process raw data (shortcut)")
    ap.add_argument("--final-only", action="store_true", help="Only process final data (shortcut)")
    
    # Plotting options
    ap.add_argument("--skip-venn", action="store_true",
                    help="Skip Venn diagram generation (useful for >3-5 groups)")
    ap.add_argument("--formats", default="svg,pdf",
                    help="Comma-separated figure formats: e.g., svg,pdf,png")
    
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else None
    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    
    # Parse subset groups if provided
    subset_groups = None
    if args.subset_groups:
        subset_groups = [g.strip() for g in args.subset_groups.split(",") if g.strip()]
    
    # Determine which data to process
    use_raw = args.use_raw
    use_final = args.use_final
    if args.raw_only:
        use_raw = True
        use_final = False
    elif args.final_only:
        use_raw = False
        use_final = True
    
    domains = ["micro", "mito"] if args.domain == "both" else [args.domain]
    
    for dom in domains:
        inp = Inputs(data_dir=data_dir, subdir=args.subdir, domain=dom, taxonomy_path=taxonomy_path)
        print(f"[INFO] Running {inp}")
        run_domain(
            inp=inp,
            id_col=args.sample_id_col,
            group_col=args.group_col,
            color_col=args.color_col,
            subset_groups=subset_groups,
            use_raw=use_raw,
            use_final=use_final,
            skip_venn=args.skip_venn,
            formats=formats,
        )
        print(f"[OK] Finished {dom}")

if __name__ == "__main__":
    main()