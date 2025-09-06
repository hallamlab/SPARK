#!/usr/bin/env python3
"""
clustermaps_cli.py
Build log-scaled abundance clustermaps (rows = taxa/ranks or ASVs, cols = samples)
with column color bars (type_group, status, optional kit), using palettes you provide.

Features
--------
- Robust CLI (no hard-coded paths).
- Ranks: Phylum/Class/Order/Family/Genus/Species/ASV_ID (configurable).
- Per-rank selection = union of:
  * top N by total abundance within each type_group (configurable per rank), and
  * any ASVs significant in an ISA table (optional; threshold configurable).
- Two heatmaps per rank:
  * `_code`   : column order preserved (no column clustering).
  * `_clustered`: columns clustered.
- Outputs: SVG + PDF figures, and the underlying pivot table (TSV).
- Optional mitochondrial ASV clustermaps.

Example
-------
python clustermaps_cli.py \
  --asv-meta /path/metadata/ASV_meta.tsv \
  --metadata /path/metadata/metadata_updated.tsv \
  --isa /path/indicspecies/Type_status_ISA_results.tsv \
  --outdir /path/diversity \
  --type-order "Oral Rinse,BAL,Lung Brush" \
  --exclude-types "Skin Brush,Scope Flush" \
  --type-palette "Oral Rinse=#6A3D9A,BAL=#0072B2,Lung Brush=#009E73" \
  --status-palette "Non-Cancer=#FFFFFF,Cancer=#A50026,methods=#D3D3D3" \
  --topN "Phylum=30,Class=30,Order=30,Family=30,Genus=30,Species=30,ASV_ID=6000" \
  --isa-min-stat 0.6 \
  --tick-values "5,50,500,5000,50000" \
  --vmax 50000

# Add mitochondrial run:
  --mito-asv /path/mito/ASVs/ASV_final.mito.tsv \
  --mito-outdir /path/mito/diversity
"""

from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ----------------------- Matplotlib/Seaborn defaults ------------------------
mpl.rcParams['pdf.fonttype'] = 42      # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")


# ------------------------------- Utilities ----------------------------------
def parse_kv_csv(s: str) -> dict:
    """
    Parse "A=#fff,B:#123,C=steelblue" or "Phylum=30,Class=30" into dict[str,str].
    Returns {} if s is falsy/empty.
    """
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
        elif ":" in item:
            k, v = item.split(":", 1)
        else:
            warnings.warn(f"Ignoring malformed mapping: {item!r}")
            continue
        out[k.strip()] = v.strip()
    return out


def ensure_cols(df: pd.DataFrame, required: list[str], where: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")


def log10_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Return log10(x+1)."""
    return np.log10(df + 1.0)


def dynamic_height(n_rows: int, per_row: float = 0.4, min_h: float = 8.0, max_h: float = 6000.0) -> float:
    return float(np.clip(per_row * max(n_rows, 1), min_h, max_h))


def build_rank_universe(
    asv_meta: pd.DataFrame,
    ranks: list[str],
    type_order: list[str],
    exclude_types: set[str],
    sig_asvs: set[str],
    topN_map: dict[str, int],
    count_col: str,
) -> dict[str, list[str]]:
    """
    For each rank, compute union across type_groups:
      top N (within group) + significant ASVs' taxa for that rank.
    Returns rank -> allowed label list.
    """
    keep = asv_meta[~asv_meta["type_group"].isin(exclude_types)].copy()
    universe: dict[str, list[str]] = {}
    for rank in ranks:
        # aggregate per group
        grp = keep.groupby(["type_group", rank], dropna=False)[count_col].sum().reset_index()
        all_labels = set()
        for g in type_order:
            gdf = grp[grp["type_group"] == g]
            if gdf.empty:
                continue
            topN = int(topN_map.get(rank, 30))
            top_labels = gdf.sort_values(count_col, ascending=False)[rank].head(topN).tolist()
            all_labels.update(top_labels)
        # add any significant ASVs (map ASV -> this rank)
        if "ASV_ID" in keep.columns and sig_asvs:
            sig_rank = keep[keep["ASV_ID"].isin(sig_asvs)][rank].unique().tolist()
            all_labels.update(sig_rank)
        # finalize
        universe[rank] = sorted({("Other" if (x is None or (isinstance(x, float) and np.isnan(x))) else x) for x in all_labels})
    return universe


def assign_plot_labels(asv_meta: pd.DataFrame, rank: str, allowed: list[str]) -> pd.Series:
    allowed_set = set(allowed)
    col = f"{rank}_plot"
    return asv_meta[rank].apply(lambda x: x if (x in allowed_set) else "Other")


def col_colors_from_meta(
    samples: list[str],
    sample_meta: pd.DataFrame,
    type_palette: dict[str, str],
    status_palette: dict[str, str],
    kit_palette: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with index = samples (columns of heatmap),
    columns = color bars (type_group, status, optionally kit), containing hex colors.
    """
    sub = sample_meta.loc[samples, ["type_group", "status"]].copy()
    sub["type_group"] = sub["type_group"].map(lambda k: type_palette.get(str(k), "#D3D3D3"))
    sub["status"] = sub["status"].map(lambda k: status_palette.get(str(k), "#D3D3D3"))
    out = pd.DataFrame({"type_group": sub["type_group"], "status": sub["status"]})
    if kit_palette is not None and "kit" in sample_meta.columns:
        out["kit"] = sample_meta.loc[samples, "kit"].map(lambda k: kit_palette.get(str(k), "#D3D3D3"))
    return out


def draw_clustermap(
    pivot: pd.DataFrame,
    col_colors_df: pd.DataFrame,
    outfile_prefix: Path,
    tick_vals_orig: list[int],
    vmax_display: int,
    figsize_w: float,
    row_height: float,
    min_fig_h: float,
    max_fig_h: float,
    method: str = "ward",
    metric: str = "euclidean",
    dendrogram_ratio=(.05, .2),
    colors_ratio=0.02,
    cbar_pos=(1.02, 0.2, 0.03, 0.4),
    alpha: float = 0.75,
):
    """
    Make two plots:
      - *_code   : column order preserved (col_cluster=False)
      - *_clustered : default clustermap with column clustering (col_cluster=True)
    """
    # Order columns for colors
    col_colors_df = col_colors_df.loc[pivot.columns]

    # Log10 transform
    pivot_log = log10_transform(pivot)

    # Greyscale cmap
    cmap = LinearSegmentedColormap.from_list("light_greyscale", ['#ffffff', '#d9d9d9', '#000000'], N=256)

    # Colorbar ticks
    tick_vals_log = [np.log10(v + 1) for v in tick_vals_orig]
    vmax_log = np.log10(vmax_display + 1)

    n_rows = pivot.shape[0]
    height = dynamic_height(n_rows, per_row=row_height, min_h=min_fig_h, max_h=max_fig_h)

    # 1) No column clustering (code)
    g = sns.clustermap(
        pivot_log,
        method=method,
        metric=metric,
        col_colors=col_colors_df,
        cmap=cmap,
        vmin=0, vmax=vmax_log,
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        figsize=(figsize_w, height),
        cbar_pos=cbar_pos,
        alpha=alpha,
        col_cluster=False
    )
    # colorbar
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_ticks(tick_vals_log)
    cbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
    cbar.set_label("ASV Count", rotation=270, labelpad=15)
    # force x tick labels visible and match columns
    g.ax_heatmap.set_xticks(g.ax_heatmap.get_xticks())
    g.ax_heatmap.set_xticklabels(pivot_log.columns, rotation=90, ha='center')
    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True, length=5)

    out_svg = outfile_prefix.with_suffix(".svg")
    out_pdf = outfile_prefix.with_suffix(".pdf")
    plt.savefig(out_svg, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    # 2) With column clustering
    g = sns.clustermap(
        pivot_log,
        method=method,
        metric=metric,
        col_colors=col_colors_df,
        cmap=cmap,
        vmin=0, vmax=vmax_log,
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        figsize=(figsize_w, height),
        cbar_pos=cbar_pos,
        alpha=alpha,
        col_cluster=True
    )
    cbar = g.ax_heatmap.collections[0].colorbar
    cbar.set_ticks(tick_vals_log)
    cbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
    cbar.set_label("ASV Count", rotation=270, labelpad=15)
    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True, length=5)

    out2_svg = outfile_prefix.with_name(outfile_prefix.stem.replace("_code", "_clustered")).with_suffix(".svg")
    out2_pdf = out2_svg.with_suffix(".pdf")
    plt.savefig(out2_svg, bbox_inches='tight')
    plt.savefig(out2_pdf, bbox_inches='tight')
    plt.close()


def read_isa_sig_asvs(isa_path: Path, min_stat: float) -> set[str]:
    """
    Read the combined ISA results and return ASV_IDs considered 'significant' for inclusion:
      (type_significance == True OR status_significance == True)
      AND (type_stat >= min_stat OR status_stat >= min_stat)
    """
    if not isa_path:
        return set()
    df = pd.read_csv(isa_path, sep="\t", header=0)
    if "ASV_ID" not in df.columns:
        df.rename(columns={df.columns[0]: "ASV_ID"}, inplace=True)
    # guard missing columns
    for c in ["type_significance", "status_significance", "type_stat", "status_stat"]:
        if c not in df.columns:
            warnings.warn(f"ISA table missing column {c!r}; falling back to top-N selection only.")
            return set()
    m = ((df["type_significance"] == True) | (df["status_significance"] == True)) & \
        ((df["type_stat"].fillna(0) >= min_stat) | (df["status_stat"].fillna(0) >= min_stat))
    return set(df.loc[m, "ASV_ID"].astype(str))


# ------------------------------- Main ---------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Clustermap pipeline (ranks/ASV + mitochondrial optional) with robust CLI."
    )
    # Core inputs
    ap.add_argument("--asv-meta", type=Path, required=True,
                    help="ASV_meta.tsv (must contain: ASV_ID, sample_code, sample, type_group, status, [kit], ranks, and count column).")
    ap.add_argument("--metadata", type=Path, required=True,
                    help="metadata_updated.tsv (must contain 'sample', 'sample_code', 'type_group', 'status', [kit]).")
    ap.add_argument("--isa", type=Path, default=None,
                    help="Optional Type_status_ISA_results.tsv to include significant ASVs into rank selection.")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Output directory for figures and pivot tables.")

    # Palettes & orders
    ap.add_argument("--type-order", type=str, default="Oral Rinse,BAL,Lung Brush",
                    help="Comma-separated ordering of type_group.")
    ap.add_argument("--exclude-types", type=str, default="Skin Brush,Scope Flush",
                    help="Comma-separated type_group values to exclude.")
    ap.add_argument("--type-palette", type=str, required=True,
                    help='e.g. "Oral Rinse=#6A3D9A,BAL=#0072B2,Lung Brush=#009E73"')
    ap.add_argument("--status-palette", type=str, required=True,
                    help='e.g. "Non-Cancer=#FFFFFF,Cancer=#A50026,methods=#D3D3D3"')
    ap.add_argument("--kit-palette", type=str, default="",
                    help='Optional: "HostZERO-DEP=#000000,HostZERO-NODEP=#808080,SPARK-ZYMO=#87CEEB"')

    # Ranks & selection
    ap.add_argument("--ranks", type=str, default="Phylum,Class,Order,Family,Genus,Species,ASV_ID",
                    help="Comma-separated ranks to plot.")
    ap.add_argument("--topN", type=str, default="Phylum=30,Class=30,Order=30,Family=30,Genus=30,Species=30,ASV_ID=6000",
                    help="Per-rank top-N selection, e.g. 'Phylum=30,...,ASV_ID=6000'.")
    ap.add_argument("--count-col", type=str, default="corr_count",
                    help="Abundance column in ASV_meta (default: corr_count).")
    ap.add_argument("--isa-min-stat", type=float, default=0.6,
                    help="Minimum stat to consider an ASV 'significant' in ISA gate (default: 0.6).")

    # Heatmap look
    ap.add_argument("--tick-values", type=str, default="5,50,500,5000,50000",
                    help="Comma-separated original scale ticks for colorbar.")
    ap.add_argument("--vmax", type=int, default=50000, help="Max display value for colorbar (on original scale).")
    ap.add_argument("--figwidth", type=float, default=32.0, help="Heatmap figure width (inches).")
    ap.add_argument("--row-height", type=float, default=0.4, help="Height per row (inches).")
    ap.add_argument("--min-height", type=float, default=8.0, help="Minimum figure height (inches).")
    ap.add_argument("--max-height", type=float, default=6000.0, help="Maximum figure height (inches).")

    # Mitochondrial (optional)
    ap.add_argument("--mito-asv", type=Path, default=None,
                    help="Optional ASV_final.mito.tsv to build mitochondrial clustermaps.")
    ap.add_argument("--mito-outdir", type=Path, default=None,
                    help="Optional output directory for mito figures/pivots (defaults to OUTDIR/'mito').")
    ap.add_argument("--mito-count-col", type=str, default="count",
                    help="Mito abundance field after stacking (default: count).")

    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Palettes
    type_palette = parse_kv_csv(args.type_palette)
    status_palette = parse_kv_csv(args.status_palette)
    kit_palette = parse_kv_csv(args.kit_palette) if args.kit_palette else None

    type_order = [x.strip() for x in args.type_order.split(",") if x.strip()]
    exclude_types = {x.strip() for x in args.exclude_types.split(",") if x.strip()}
    ranks = [x.strip() for x in args.ranks.split(",") if x.strip()]
    topN_map = {k: int(v) for k, v in parse_kv_csv(args.topN).items()}

    tick_vals_orig = [int(x.strip()) for x in args.tick_values.split(",") if x.strip()]
    vmax_display = int(args.vmax)

    # Read metadata (for column colors)
    meta = pd.read_csv(args.metadata, sep="\t", header=0)
    ensure_cols(meta, ["sample", "sample_code", "type_group", "status"], "metadata_updated.tsv")
    meta = meta.set_index("sample")

    # Read ASV meta (counts per ASV/sample with ranks)
    asv_meta = pd.read_csv(args.asv_meta, sep="\t", header=0)
    ensure_cols(asv_meta, ["ASV_ID", "sample_code", "sample", "type_group", "status", args.count_col], "ASV_meta.tsv")

    # ISA gate (optional)
    sig_asvs = read_isa_sig_asvs(args.isa, args.isa_min_stat) if args.isa else set()

    # Build rank universes
    rank_universe = build_rank_universe(
        asv_meta=asv_meta,
        ranks=ranks,
        type_order=type_order,
        exclude_types=exclude_types,
        sig_asvs=sig_asvs,
        topN_map=topN_map,
        count_col=args.count_col,
    )

    # For each rank: add *_plot labels, pivot (rows=taxa_plot, cols=sample_code), plot
    working = asv_meta.copy()
    working = working[~working["type_group"].isin(exclude_types)]

    # sample meta (for col_colors) keyed by sample_code
    smeta = meta.reset_index().drop_duplicates("sample")[["sample", "sample_code", "type_group", "status"]]
    if kit_palette is not None and "kit" in asv_meta.columns:
        smeta = pd.merge(smeta, asv_meta[["sample", "kit"]].drop_duplicates("sample"), on="sample", how="left")
    smeta = smeta.set_index("sample_code")

    for rank in ranks:
        allowed = rank_universe.get(rank, [])
        colname = f"{rank}_plot"
        working[colname] = assign_plot_labels(working, rank, allowed)

        # (rows = taxa_plot, cols = sample_code)
        pivot = (working.groupby(["sample_code", colname])[args.count_col]
                 .sum().reset_index()
                 .pivot(index=colname, columns="sample_code", values=args.count_col)
                 .fillna(0))

        # Column colors aligned to columns
        col_colors_df = col_colors_from_meta(
            samples=pivot.columns.tolist(),
            sample_meta=smeta,
            type_palette=type_palette,
            status_palette=status_palette,
            kit_palette=kit_palette,
        )

        # Legends (drawn on figure as standard legend would overlap); create patches for reference
        legend_patches = []
        for g in type_order:
            if g in type_palette:
                legend_patches.append(Patch(facecolor=type_palette[g], label=f"Type: {g}", alpha=0.75))
        for st, col in status_palette.items():
            legend_patches.append(Patch(facecolor=col, label=f"status: {st}", alpha=0.75))
        # (We rely on column color bars; patches are not explicitly added; feel free to adapt.)

        # Draw
        prefix = outdir / f"clustermap_{colname}_code"
        draw_clustermap(
            pivot=pivot,
            col_colors_df=col_colors_df,
            outfile_prefix=prefix,
            tick_vals_orig=tick_vals_orig,
            vmax_display=vmax_display,
            figsize_w=args.figwidth,
            row_height=args.row_height,
            min_fig_h=args.min_height,
            max_fig_h=args.max_height,
        )
        # Save pivot
        pivot_out = outdir / f"clustermap_{colname}.tsv"
        pivot.to_csv(pivot_out, sep="\t")

    # ------------------------- Mitochondrial (optional) -------------------------
    if args.mito_asv:
        mito_outdir = args.mito_outdir or (outdir / "mito")
        mito_outdir.mkdir(parents=True, exist_ok=True)

        mito_df = pd.read_csv(args.mito_asv, sep="\t", header=0, index_col=0)
        # columns look like samples; ensure names match metadata.sample (and we have sample_code)
        mito_df.columns = [str(c).rsplit("_", 1)[0] for c in mito_df.columns]  # mimic original cleanup
        mito_stack = mito_df.stack().reset_index()
        mito_stack.columns = ["ASV_ID", "sample", args.mito_count_col]
        mito_stack = mito_stack[mito_stack[args.mito_count_col] > 0]

        # join metadata to get sample_code/type_group/status/kit
        mito_meta = pd.merge(mito_stack, meta.reset_index(), on="sample", how="left")
        mito_meta = mito_meta[~mito_meta["type_group"].isin(exclude_types)]

        # Build pivot (rows=ASV_ID, cols=sample_code)
        pivot = (mito_meta.groupby(["sample_code", "ASV_ID"])[args.mito_count_col]
                 .sum().reset_index()
                 .pivot(index="ASV_ID", columns="sample_code", values=args.mito_count_col)
                 .fillna(0))

        # Column color bars
        smeta_m = meta.reset_index().drop_duplicates("sample")[["sample", "sample_code", "type_group", "status"]]
        if kit_palette is not None and "kit" in mito_meta.columns:
            smeta_m = pd.merge(smeta_m, mito_meta[["sample", "kit"]].drop_duplicates("sample"), on="sample", how="left")
        smeta_m = smeta_m.set_index("sample_code")

        col_colors_df = col_colors_from_meta(
            samples=pivot.columns.tolist(),
            sample_meta=smeta_m,
            type_palette=type_palette,
            status_palette=status_palette,
            kit_palette=kit_palette,
        )

        prefix = mito_outdir / "clustermap_ASV_code_mito"
        draw_clustermap(
            pivot=pivot,
            col_colors_df=col_colors_df,
            outfile_prefix=prefix,
            tick_vals_orig=tick_vals_orig,
            vmax_display=vmax_display,
            figsize_w=args.figwidth,
            row_height=args.row_height,
            min_fig_h=args.min_height,
            max_fig_h=args.max_height,
        )
        pivot.to_csv(mito_outdir / "clustermap_ASV_mito.tsv", sep="\t")

    print(f"Done. Outputs in: {outdir}")

if __name__ == "__main__":
    main()
