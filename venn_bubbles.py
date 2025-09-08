#!/usr/bin/env python3
"""
bubble_from_venn.py
Build per-intersection genus-level bubble plots from ASV presence tables.

Inputs
------
- ASV_meta.tsv: long table with columns including:
  ASV_ID, type_group, Family, Genus, corr_count (or count)
- Venn presence table(s): rows: grouping, ASV_ID
  e.g. Three_types_venn_presence_table.tsv

Outputs
-------
- One bubble plot per intersection label (SVG/PDF/PNG as requested)
- Combined tidy TSV with all plotted rows per mode:
  <out-prefix>_presence_tax.tsv
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Global style ----------
mpl.rcParams['pdf.fonttype'] = 42        # keep text as text
mpl.rcParams['svg.fonttype'] = 'none'    # keep text as text
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12, 'font.family': 'Source Sans Pro'})
sns.set_theme()
sns.set_style("white")


# ---------- Palettes (defaults match your originals) ----------
TYPE_PALETTE_DEFAULT = {
    'Scope Flush': '#E69F00',
    'Skin Brush':  '#CC79A7',
    'Lung Brush':  '#009E73',
    'BAL':         '#0072B2',
    'Oral Rinse':  '#6A3D9A',
    'Failed-QC':   'lightgray',
}
THREE_PALETTE_DEFAULT = {
    'Lung Brush':  '#009E73',
    'BAL':         '#0072B2',
    'Oral Rinse':  '#6A3D9A',
}

# ---------- Helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_name(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(x))

def load_asv_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # prefer corrected counts when available
    if 'corr_count' not in df.columns and 'count' in df.columns:
        df['corr_count'] = df['count']
    if 'Family' not in df.columns:
        df['Family'] = 'Unassigned'
    if 'Genus' not in df.columns:
        df['Genus'] = 'Unassigned'
    return df

def load_presence_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if not {'grouping','ASV_ID'}.issubset(df.columns):
        raise ValueError(f"Presence table {path} must contain columns: grouping, ASV_ID")
    return df

def fam_genus_col(fam, gen):
    f = 'Unassigned' if pd.isna(fam) or fam == '' else str(fam)
    g = 'Unassigned' if pd.isna(gen) or gen == '' else str(gen)
    return f"{f} {g}"

def bubble_each_intersection(pres_df: pd.DataFrame,
                             asv_meta_df: pd.DataFrame,
                             group_col: str,
                             order: list[str],
                             palette: dict[str, str],
                             out_dir: Path,
                             out_prefix: str,
                             formats: list[str]) -> pd.DataFrame:
    """
    For each 'grouping' in presence table, aggregate genus-level counts by group_col
    and draw a bubble plot.
    Returns a concatenated tidy DataFrame of all plotted rows.
    """
    plotted = []
    present_groups = set(asv_meta_df.get(group_col, pd.Series([], dtype=str)).dropna().unique())
    # restrict order/palette to what’s actually present (avoid seaborn warnings)
    order_eff  = [g for g in order if g in present_groups]
    pal_eff    = {g: palette[g] for g in order_eff if g in palette}

    for vgrp in pres_df['grouping'].dropna().unique():
        v_asvs = pres_df.loc[pres_df['grouping'] == vgrp, 'ASV_ID'].astype(str).tolist()
        sub = asv_meta_df[asv_meta_df['ASV_ID'].astype(str).isin(v_asvs)].copy()
        if sub.empty:
            # nothing to plot for this label
            continue

        agg = (sub.groupby([group_col, 'Family', 'Genus'], dropna=False)['corr_count']
                   .sum()
                   .reset_index())
        agg[group_col] = pd.Categorical(agg[group_col], categories=order_eff, ordered=True)
        agg['Family Genus'] = [fam_genus_col(f, g) for f, g in zip(agg['Family'], agg['Genus'])]
        # zero-sized bubbles are invisible; set to NaN so sizes range behaves
        agg['corr_count'] = agg['corr_count'].replace(0, np.nan)

        # keep a tidy copy for export
        tidy = agg.rename(columns={'corr_count': 'size'}).assign(grouping=vgrp)
        plotted.append(tidy)

        # figure height scales with distinct taxa (with sane bounds)
        n_taxa = agg['Family Genus'].nunique()
        fig_h  = max(4, min(0.4 * n_taxa, 15))

        fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
        sns.scatterplot(
            data=agg,
            x=group_col,
            y='Family Genus',
            hue=group_col,
            hue_order=order_eff,
            palette=pal_eff,
            size='corr_count',
            sizes=(8, 500),
            alpha=0.75,
            ax=ax
        )
        sns.despine(top=True, right=True)
        ax.margins(y=0.2)
        ax.legend(title=('Sample Type' if group_col == 'type_group' else group_col),
                  bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, frameon=False)
        plt.xticks(rotation=45)

        base = out_dir / f"{out_prefix}_{safe_name(vgrp)}_Genus_bubbleplot"
        for ext in formats:
            fig.savefig(f"{base}.{ext}", bbox_inches="tight")
        plt.close(fig)

    return pd.concat(plotted, ignore_index=True) if plotted else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Genus-level bubble plots from Venn presence tables.")
    ap.add_argument("--data-dir", default="/home/ryan/SeqData/SeqData/UBC/LMP_priority1/",
                    help="Project root directory")
    ap.add_argument("--subdir", default="spark_combined_output",
                    help="Output subdirectory root")
    ap.add_argument("--asv-meta", default=None,
                    help="Path to ASV_meta.tsv (default: <data-dir>/<subdir>/metadata/ASV_meta.tsv)")
    ap.add_argument("--presence", default=None,
                    help="Three-types presence TSV (default: metadata/Three_types_venn_presence_table.tsv)")
    ap.add_argument("--type-order", default="Oral Rinse,BAL,Lung Brush",
                    help="Comma list order for type_group axis")
    ap.add_argument("--formats", default="svg,pdf",
                    help="Comma-separated image formats (svg,pdf,png)")
    return ap.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    subdir   = args.subdir
    meta_dir = data_dir / subdir / "metadata"
    ensure_dir(meta_dir)

    asv_meta_path = Path(args.asv_meta) if args.asv_meta else (meta_dir / "ASV_meta.tsv")
    presence_path = Path(args.presence) if args.presence else (meta_dir / "Three_types_venn_presence_table.tsv")
    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    type_order = [t.strip() for t in args.type_order.split(",") if t.strip()]

    # Load ASV meta (single source for both modes)
    asv_meta_df = load_asv_meta(asv_meta_path)

    # ----- Three-types (by type_group) -----
    if presence_path.exists():
        pres_df = load_presence_table(presence_path)
        out_prefix = "Three_types"
        plotted_df = bubble_each_intersection(
            pres_df, asv_meta_df,
            group_col="type_group",
            order=type_order,
            palette=TYPE_PALETTE_DEFAULT,
            out_dir=meta_dir,
            out_prefix=out_prefix,
            formats=formats
        )
        if not plotted_df.empty:
            plotted_df.to_csv(meta_dir / f"{out_prefix}_presence_tax.tsv", sep="\t", index=False)
            print(f"[OK] Wrote {out_prefix}_presence_tax.tsv and bubble plots")
        else:
            print("[WARN] No rows to plot for three-types presence table.")
    else:
        print(f"[WARN] Presence table not found: {presence_path}")

if __name__ == "__main__":
    main()
