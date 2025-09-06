#!/usr/bin/env python3
"""
asv_overlap.py
-------------
Unified CLI to generate UpSet & Venn-style set intersections for ASV data.

Supports:
- Microbial and/or mitochondrial runs
- Unique-membership UpSet (counts of ASVs)
- Abundance-weighted UpSet with stacked bars by category
- Venn plots (3-set via matplotlib-venn; 4–6 sets via `venn` package if available)
- Presence tables (exclusive membership lists) and exclusive-sum tables
- Optional "Oral/Lung" composite group and kit-level plots

Examples
--------
# Microbial only, default 5-group set & 3-group set, SVG+PDF
python asv_overlap.py \
  --data-dir /home/ryan/SeqData/SeqData/UBC/LMP_priority1 \
  --domain micro \
  --formats svg,pdf

# Include mito as well, custom group lists, PNG
python asv_overlap.py \
  --data-dir /data/run1 \
  --domain both \
  --five-groups "Skin Brush,Scope Flush,Oral Rinse,BAL,Lung Brush" \
  --three-groups "Oral Rinse,BAL,Lung Brush" \
  --formats png

# Run kit analysis if 'kit' is in metadata
python asv_overlap.py --data-dir /data/run1 --domain micro --do-kit
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
    from venn import draw_venn, generate_petal_labels, generate_colors  # 4–6 sets
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


def composite_color(colors: Sequence[str], alpha: float = 0.6) -> str:
    """
    Simple alpha compositing of a sequence of colors (front-to-back, last on top).
    """
    rgba = np.array([mcolors.to_rgba(c, alpha=alpha) for c in colors])
    # start with back-most:
    acc_rgb = rgba[0, :3]
    acc_a = rgba[0, 3]
    for over in rgba[1:]:
        a_o = over[3]
        acc_rgb = over[:3] * a_o + acc_rgb * (1 - a_o)
        acc_a = a_o + acc_a * (1 - a_o)
    return mcolors.to_hex(acc_rgb, keep_alpha=False)


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
            self.meta = self.data_dir / subdir / "metadata" / "metadata_updated.tsv"
            # taxonomy shared across domains by default
            self.tax = taxonomy_path or (self.data_dir / subdir / "metadata" / "taxonomy_updated.tsv")
            self.out_base = self.data_dir / subdir / "metadata"
        elif domain == "mito":
            self.asv_raw = self.data_dir / subdir / "mito" / "ASVs" / "ASV_target.mito.tsv"
            self.asv_final = self.data_dir / subdir / "mito" / "ASVs" / "ASV_final.mito.tsv"
            self.meta = self.data_dir / subdir / "mito" / "metadata" / "metadata_updated_mito.tsv"
            self.tax = taxonomy_path or (self.data_dir / subdir / "metadata" / "taxonomy_updated.tsv")
            self.out_base = self.data_dir / subdir / "mito" / "metadata"
        else:
            raise ValueError("domain must be 'micro' or 'mito'")

    def __repr__(self) -> str:
        return f"Inputs(domain={self.domain}, raw={self.asv_raw}, final={self.asv_final}, meta={self.meta}, tax={self.tax})"


def read_asv_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Normalize sample names (drop paths and read pair endings)
    df.columns = [str(c).split('/')[-1].rsplit('_', 2)[0] for c in df.columns]
    return df


def read_metadata(path: Path) -> pd.DataFrame:
    md = pd.read_csv(path, sep="\t", dtype=str)
    return md


def read_taxonomy(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path, sep="\t")
    if "ASV_ID" in tx.columns:
        tx["ASV_ID"] = tx["ASV_ID"].astype(str).str.split(';', 1).str[0]
        tx = tx.set_index("ASV_ID")
    elif "Feature ID" in tx.columns:
        tx["Feature ID"] = tx["Feature ID"].astype(str).str.split(';', 1).str[0]
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
    formats: Sequence[str]
) -> None:
    data = build_upset_unique(group_sets)
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
    fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    savefig_multi(fig, out_base, "upset", formats)


def plot_upset_weighted(
    group_sets: Mapping[str, set],
    per_group_values: Mapping[Tuple[str, str], float],
    group_order: Sequence[str],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str]
) -> None:
    df = build_upset_weighted_rows(group_sets, per_group_values, group_order)
    upset = UpSet(
        df, sum_over='count', subset_size='sum',
        element_size=None, show_counts=True,
        sort_categories_by='input', min_subset_size=0,
        intersection_plot_elements=0
    )
    for g, c in colors.items():
        upset.style_categories([g], bar_facecolor=c, bar_edgecolor="black")
    upset.add_stacked_bars(by="group", sum_over="count", colors=colors, title="Abundance by Type", elements=10)
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
        ax_extra.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
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
            # Build mapping
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
        # use 'venn' package
        fig = plt.figure(figsize=(6, 6))
        if weighted_labels is None:
            # unweighted labels = cardinalities
            from venn import venn as venn_func  # convenience wrapper
            venn_func({n: s for n, s in zip(names, sets)},
                      cmap=[colors[n] for n in names], fontsize=8, alpha=0.45)
        else:
            # weighted: use draw_venn with petal_labels
            petal = generate_petal_labels([group_sets[n] for n in names], fmt="{size}")
            # Map 'binary' keys like '10100' to human-readable names
            def bin_to_tuple(b: str) -> Tuple[str, ...]:
                return tuple(n for n, ch in zip(names, b) if ch == '1')
            # create label dict in binary-key space
            lab = {k: weighted_labels.get(tuple(sorted(bin_to_tuple(k))), 0.0) for k in petal}
            draw_venn(petal_labels=lab,
                      dataset_labels=names,
                      colors=generate_colors(len(names), cmap="viridis", alpha=0.45),
                      figsize=(6, 6), fontsize=10, legend_loc="upper right")
        plt.title(title)
        savefig_multi(fig, out_base, "venn", formats)
        return

    # Fallback: skip venn if no backend
    print(f"[WARN] Venn plotting not available for {len(names)} sets (matplotlib-venn or venn not installed). Skipping.")


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
    five_groups: Sequence[str],
    three_groups: Sequence[str],
    do_composite_oral_lung: bool,
    do_kit: bool,
    formats: Sequence[str],
    palette_5: Mapping[str, str],
    palette_3: Mapping[str, str],
) -> None:
    ensure_dir(inp.out_base)

    # Read
    raw_df = read_asv_table(inp.asv_raw)
    final_df = read_asv_table(inp.asv_final)
    md = read_metadata(inp.meta)
    tx = read_taxonomy(inp.tax)

    # Long (raw + final) and merge
    def melt_counts(count_df: pd.DataFrame) -> pd.DataFrame:
        st = count_df.stack().reset_index()
        st.columns = ['ASV_ID', 'sample', 'count']
        return st

    raw_long = melt_counts(raw_df)
    final_long = melt_counts(final_df)

    # Merge with metadata
    raw_long = raw_long.merge(md, how="left", on="sample")
    final_long = final_long.merge(md, how="left", on="sample")

    # Filter positives
    raw_pos = raw_long[raw_long['count'] > 0].copy()
    final_pos = final_long[final_long['count'] > 0].copy()

    # Taxonomy
    raw_tx = attach_taxonomy(raw_pos, tx)
    fin_tx = attach_taxonomy(final_pos, tx)

    # Per-ASV totals and per-(group, ASV) totals
    raw_asv_total = raw_tx.groupby("ASV_ID")["count"].sum().to_dict()
    fin_asv_total = fin_tx.groupby("ASV_ID")["count"].sum().to_dict()
    raw_type_asv_total = raw_tx.groupby(["type_group", "ASV_ID"])["count"].sum().to_dict()
    fin_type_asv_total = fin_tx.groupby(["type_group", "ASV_ID"])["count"].sum().to_dict()

    # ---------------- Five-group suite (raw) ----------------
    five_present = [g for g in five_groups if g in set(raw_tx['type_group'].unique())]
    five_sets = {g: set(raw_tx.loc[raw_tx['type_group'] == g, 'ASV_ID']) for g in five_present}
    if len(five_sets) >= 2:
        base = inp.out_base / f"All_types_{inp.domain}"
        # UpSet unique
        plot_upset_unique(five_sets, {g: palette_5[g] for g in five_present},
                          "ASV Membership by Type", base, formats)
        # UpSet weighted (raw-type totals)
        plot_upset_weighted(five_sets, raw_type_asv_total, five_present,
                            {g: palette_5[g] for g in five_present},
                            "ASV Abundance by Type", base, formats)
        # Venn (cardinalities)
        plot_venn(five_sets, {g: palette_5[g] for g in five_present},
                  "Venn: ASV Membership (Top-level Types)", base, formats)
        # Tables
        write_presence_and_sums(five_sets, raw_asv_total, base, "All_types")
    else:
        print(f"[{inp.domain}] Skipping 5-group suite (need ≥2 present groups).")

    # ---------------- Three-group suite (final) ----------------
    three_present = [g for g in three_groups if g in set(fin_tx['type_group'].unique())]
    three_sets = {g: set(fin_tx.loc[fin_tx['type_group'] == g, 'ASV_ID']) for g in three_present}
    if len(three_sets) >= 2:
        base = inp.out_base / f"Three_types_{inp.domain}"
        # UpSet unique
        plot_upset_unique(three_sets, {g: palette_3[g] for g in three_present},
                          "ASV Membership by Type", base, formats)
        # UpSet weighted (final-type totals)
        plot_upset_weighted(three_sets, fin_type_asv_total, three_present,
                            {g: palette_3[g] for g in three_present},
                            "ASV Abundance by Type", base, formats)
        # Venn (cardinalities)
        plot_venn(three_sets, {g: palette_3[g] for g in three_present},
                  "Venn: ASV Membership (Three Types)", base, formats)
        # Tables
        write_presence_and_sums(three_sets, fin_asv_total, base, "Three_types")
    else:
        print(f"[{inp.domain}] Skipping 3-group suite (need ≥2 present groups).")

    # ---------------- Composite Oral/Lung vs Skin vs Scope (raw) ------------
    if do_composite_oral_lung:
        comp_map = {'Oral Rinse': 'Oral/Lung', 'BAL': 'Oral/Lung', 'Lung Brush': 'Oral/Lung'}
        comp_palette = {
            'Skin Brush': '#CC79A7',
            'Scope Flush': '#E69F00',
            'Oral/Lung': composite_color(['#6A3D9A', '#0072B2', '#009E73'], alpha=0.6),
        }
        comp_tx = raw_tx.copy()
        comp_tx['type_group'] = comp_tx['type_group'].replace(comp_map)
        comp_groups = ['Skin Brush', 'Scope Flush', 'Oral/Lung']
        comp_present = [g for g in comp_groups if g in set(comp_tx['type_group'].unique())]
        comp_sets = {g: set(comp_tx.loc[comp_tx['type_group'] == g, 'ASV_ID']) for g in comp_present}
        if len(comp_sets) >= 2:
            base = inp.out_base / f"Three_vs_controls_{inp.domain}"
            plot_upset_unique(comp_sets, {g: comp_palette[g] for g in comp_present},
                              "ASV Membership by Type", base, formats)
            # weighted with raw per-type sums *after* replace
            comp_totals = comp_tx.groupby(["type_group", "ASV_ID"])["count"].sum().to_dict()
            plot_upset_weighted(comp_sets, comp_totals, comp_present,
                                {g: comp_palette[g] for g in comp_present},
                                "ASV Abundance by Type", base, formats)
            plot_venn(comp_sets, {g: comp_palette[g] for g in comp_present},
                      "Venn: Skin vs Scope vs Oral/Lung", base, formats)
            write_presence_and_sums(comp_sets, raw_asv_total, base, "Three_vs_controls")
        else:
            print(f"[{inp.domain}] Skipping composite suite (need ≥2 present groups).")

    # ---------------- Kit analysis (final) -----------------------------------
    if do_kit and ("kit" in fin_tx.columns):
        kit_palette = {'HostZERO-DEP': 'black', 'HostZERO-NODEP': 'gray', 'SPARK-ZYMO': 'skyblue'}
        kits = ['HostZERO-DEP', 'HostZERO-NODEP', 'SPARK-ZYMO']
        kit_present = [k for k in kits if k in set(fin_tx['kit'].unique())]
        kit_sets = {k: set(fin_tx.loc[fin_tx['kit'] == k, 'ASV_ID']) for k in kit_present}
        if len(kit_sets) >= 2:
            base = inp.out_base / f"Kits_{inp.domain}"
            plot_upset_unique(kit_sets, {k: kit_palette[k] for k in kit_present},
                              "ASV Membership by Kit", base, formats)
            # weighted by per-(kit, ASV) totals (final)
            kit_totals = fin_tx.groupby(["kit", "ASV_ID"])["count"].sum().to_dict()
            plot_upset_weighted(kit_sets, kit_totals, kit_present,
                                {k: kit_palette[k] for k in kit_present},
                                "ASV Abundance by Kit", base, formats)
            plot_venn(kit_sets, {k: kit_palette[k] for k in kit_present},
                      "Venn: Kits", base, formats)
            write_presence_and_sums(kit_sets, fin_asv_total, base, "Kits")
        else:
            print(f"[{inp.domain}] Skipping kit suite (need ≥2 present kits).")


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ASV overlap (UpSet & Venn) plotter")
    ap.add_argument("--data-dir", required=True, help="Project root directory")
    ap.add_argument("--subdir", default="spark_combined_output", help="Root subdir with outputs")
    ap.add_argument("--domain", choices=["micro", "mito", "both"], default="micro",
                    help="Which domain(s) to run")
    ap.add_argument("--taxonomy-path", default=None,
                    help="Optional path to taxonomy_updated.tsv (defaults to <subdir>/metadata/taxonomy_updated.tsv)")
    ap.add_argument("--formats", default="svg,pdf",
                    help="Comma-separated figure formats: e.g., svg,pdf,png")
    ap.add_argument("--five-groups", default="Skin Brush,Scope Flush,Oral Rinse,BAL,Lung Brush",
                    help="Comma list for the 5-group suite (raw ASVs)")
    ap.add_argument("--three-groups", default="Oral Rinse,BAL,Lung Brush",
                    help="Comma list for the 3-group suite (final ASVs)")
    ap.add_argument("--do-composite-oral-lung", action="store_true",
                    help="Also make Skin vs Scope vs Oral/Lung composite (raw)")
    ap.add_argument("--do-kit", action="store_true", help="Also make kit UpSet/Venn from final ASVs")
    # Colors (defaults match your originals)
    ap.add_argument("--color-skin", default="#CC79A7")
    ap.add_argument("--color-scope", default="#E69F00")
    ap.add_argument("--color-oral", default="#6A3D9A")
    ap.add_argument("--color-bal", default="#0072B2")
    ap.add_argument("--color-lung", default="#009E73")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    taxonomy_path = Path(args.taxonomy_path) if args.taxonomy_path else None
    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]

    five_groups = [g.strip() for g in args.five_groups.split(",") if g.strip()]
    three_groups = [g.strip() for g in args.three_groups.split(",") if g.strip()]

    palette_5 = {
        'Skin Brush': args.color_skin,
        'Scope Flush': args.color_scope,
        'Oral Rinse': args.color_oral,
        'BAL': args.color_bal,
        'Lung Brush': args.color_lung,
    }
    palette_3 = {
        'Oral Rinse': args.color_oral,
        'BAL': args.color_bal,
        'Lung Brush': args.color_lung,
    }

    domains = ["micro", "mito"] if args.domain == "both" else [args.domain]
    for dom in domains:
        inp = Inputs(data_dir=data_dir, subdir=args.subdir, domain=dom, taxonomy_path=taxonomy_path)
        print(f"[INFO] Running {inp}")
        run_domain(
            inp=inp,
            five_groups=five_groups,
            three_groups=three_groups,
            do_composite_oral_lung=args.do_composite_oral_lung,
            do_kit=args.do_kit,
            formats=formats,
            palette_5=palette_5,
            palette_3=palette_3,
        )
        print(f"[OK] Finished {dom}")

if __name__ == "__main__":
    main()
