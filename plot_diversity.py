#!/usr/bin/env python3
"""
beta_alpha_analytics.py
Alpha (Shannon), PERMANOVA (global & pairwise), and UMAP plots from
precomputed distance matrices (Bray, Jaccard) + metadata. Optional "mito" pass.

Outputs (under --outdir and --mito-outdir):
- diversity/
  - alpha_ttest.tsv
  - permanova_global.tsv
  - permanova_pairwise.tsv
  - Beta_Heatmap_permanova.(pdf|svg)
  - UMAP_*.{pdf,svg} (several variants)
- mito/diversity/ (same set if mito inputs provided)

Examples
--------
python beta_alpha_analytics.py \
  --metadata spark_combined_output/metadata/metadata_updated.tsv \
  --master   spark_combined_output/metadata/master_table.tsv \
  --alpha    spark_combined_output/diversity/shannon.tsv \
  --bray     spark_combined_output/diversity/bray.tsv \
  --jacc     spark_combined_output/diversity/jaccard.tsv \
  --outliers-all  spark_combined_output/metadata/outliers_table.tsv \
  --outliers-type spark_combined_output/metadata/outliers_type_group.tsv \
  --exclude-types "Skin Brush,Scope Flush" \
  --type-order "Oral Rinse,BAL,Lung Brush" \
  --outdir spark_combined_output/diversity

# With mito
python beta_alpha_analytics.py \
  ... (same as above) \
  --mito-alpha spark_combined_output/mito/diversity/shannon.mito.tsv \
  --mito-bray  spark_combined_output/mito/diversity/bray.mito.tsv \
  --mito-jacc  spark_combined_output/mito/diversity/jaccard.mito.tsv \
  --mito-outdir spark_combined_output/mito/diversity
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from itertools import combinations
from statsmodels.stats.multitest import multipletests

# -------------------------- style --------------------------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
sns.set_theme()
sns.set_style("white")

PALETTE_TYPES = {
    'Scope Flush': '#E69F00',
    'Skin Brush':  '#CC79A7',
    'Lung Brush':  '#009E73',
    'BAL':         '#0072B2',
    'Oral Rinse':  '#6A3D9A',
    'Failed-QC':   'lightgray',
}
PALETTE_THREE = {k: PALETTE_TYPES[k] for k in ('Lung Brush', 'BAL', 'Oral Rinse')}
PALETTE_STATUS = {'Non-Cancer': 'white', 'Cancer': '#A50026', 'methods': 'lightgray'}

# -------------------------- utils --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_tsv(path: Path, index_col=None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", index_col=index_col)

def perform_umap_precomputed(dist_df: pd.DataFrame,
                             n_neighbors: int = 30,
                             min_dist: float = 0.01,
                             random_state: int = 42,
                             prefix: str = "") -> pd.DataFrame:
    """UMAP on a precomputed square distance matrix."""
    if dist_df.shape[0] != dist_df.shape[1]:
        raise ValueError(f"{prefix} distance matrix must be square.")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    emb = reducer.fit_transform(dist_df.values)
    umap_df = pd.DataFrame(emb, index=dist_df.index, columns=["UMAP1", "UMAP2"])
    return umap_df

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    if right.empty:
        return left
    cols = [c for c in right.columns if c != on]
    return left.merge(right[[on] + cols], how="left", on=on)

def fdr_ttests_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    res = []
    groups = [g for g in df[group_col].dropna().unique()]
    for a, b in combinations(groups, 2):
        x = df.loc[df[group_col] == a, value_col].dropna()
        y = df.loc[df[group_col] == b, value_col].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        stat = float((x.mean() - y.mean()) / (np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))))
        # Use scipy t-test? Here we use numpy fallback to keep deps simple if desired:
        from scipy.stats import ttest_ind
        _, p = ttest_ind(x, y, equal_var=False)
        res.append({"group1": a, "group2": b, "tstat": stat, "pval": p})
    out = pd.DataFrame(res)
    if not out.empty:
        out["qval"] = multipletests(out["pval"], method="fdr_bh")[1]
        out["significant"] = out["qval"] < 0.05
    return out

def global_and_pairwise_permanova(bray: pd.DataFrame, group: pd.Series, permutations: int = 999):
    # Align
    ids = bray.index.intersection(group.index)
    bray = bray.loc[ids, ids]
    grp = group.loc[ids]
    # global
    dm = DistanceMatrix(bray.values.copy(order="C"), ids=bray.index.tolist())
    glob = permanova(dm, grp, permutations=permutations).to_frame().T
    glob.insert(0, "term", "global")
    # pairwise
    res = []
    uniq = [g for g in grp.unique()]
    for a, b in combinations(uniq, 2):
        sub_ids = grp[grp.isin([a, b])].index
        counts = grp.loc[sub_ids].value_counts()
        if (counts < 2).any():
            continue
        sub_dm = DistanceMatrix(bray.loc[sub_ids, sub_ids].values.copy(order="C"), ids=sub_ids.tolist())
        pw = permanova(sub_dm, grp.loc[sub_ids], permutations=permutations)
        s = pw.copy()
        s["Group1"], s["Group2"] = a, b
        res.append(s)
    pair = pd.DataFrame(res)
    if not pair.empty:
        pair = pair.reset_index(drop=True)
        pair["q-value"] = multipletests(pair["p-value"], method="fdr_bh")[1]
    return glob, pair

def heatmap_pairwise_q(pair: pd.DataFrame, out_base: Path, title: str):
    if pair.empty:
        warnings.warn("No pairwise PERMANOVA results to plot.")
        return
    M = pair.pivot(index="Group1", columns="Group2", values="q-value")
    M = M.combine_first(M.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(M, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.1,
                cbar_kws={"label": "q-value"}, linewidths=0.5, linecolor="lightgray")
    plt.title(title)
    plt.tight_layout()
    for ext in ("svg", "pdf"):
        plt.savefig(out_base.with_suffix(f".{ext}"))
    plt.close()

def scatter_umap(df: pd.DataFrame,
                 x: str, y: str,
                 hue: str, size: str | None,
                 style: str | None,
                 palette: dict | str,
                 title: str,
                 out_base: Path,
                 size_range=(40, 400),
                 edgecolor="lightgray"):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(
        data=df, x=x, y=y, hue=hue, size=size, style=style,
        palette=palette, sizes=size_range, alpha=0.8,
        edgecolor=edgecolor, linewidth=0.5, ax=ax
    )
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1), frameon=False)
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"))
    plt.close(fig)

# -------------------------- run block --------------------------
def run_one_pass(name: str,
                 meta: pd.DataFrame,
                 master: pd.DataFrame,
                 alpha: pd.DataFrame,
                 bray: pd.DataFrame,
                 jacc: pd.DataFrame,
                 outdir: Path,
                 exclude_types: list[str],
                 type_order: list[str] | None,
                 neighbors: int,
                 min_dist: float,
                 permutations: int):
    ensure_dir(outdir)

    # Merge everything we have
    meta = meta.copy()
    if "sample" not in meta.columns:
        raise ValueError("--metadata must contain a 'sample' column.")
    df = meta.copy()

    # Alpha
    if not alpha.empty:
        alpha = alpha.rename_axis("sample").reset_index()
        df = safe_merge(df, alpha, on="sample")  # adds 'Shannon'

    # UMAPs from distances (precomputed)
    if not bray.empty:
        bray_umap = perform_umap_precomputed(
            bray, n_neighbors=neighbors, min_dist=min_dist, random_state=42, prefix=f"{name}/Bray"
        ).rename(columns={"UMAP1": "UMAP1", "UMAP2": "UMAP2"})
        df = safe_merge(df, bray_umap.reset_index(), on="sample")
    if not jacc.empty:
        jacc_umap = perform_umap_precomputed(
            jacc, n_neighbors=neighbors, min_dist=min_dist, random_state=42, prefix=f"{name}/Jacc"
        ).rename(columns={"UMAP1": "Jacc_UMAP1", "UMAP2": "Jacc_UMAP2"})
        df = safe_merge(df, jacc_umap.reset_index(), on="sample")

    # Bring counts if available from master table (used for point sizes)
    if not master.empty:
        need = {"sample", "count", "pass_filter"}
        inter = [c for c in need if c in master.columns]
        if inter:
            df = safe_merge(df, master[inter], on="sample")

    # Outlier flags if present (overall_OL / typ_grp_OL expected column names)
    # These may already exist in metadata; we leave as-is if present.

    # Filter: remove failed QC and excluded types
    if "pass_filter" in df.columns:
        df = df.loc[df["pass_filter"] != "Failed-QC"].copy()
    if exclude_types:
        df = df.loc[~df["type_group"].isin(exclude_types)].copy()

    # Establish plotting order & palettes
    if type_order:
        order = [t for t in type_order if t in df.get("type_group", pd.Series()).unique()]
    else:
        order = list(df.get("type_group", pd.Series()).dropna().unique())
    type_palette = {k: PALETTE_TYPES.get(k, PALETTE_TYPES.get("Failed-QC", "#999999")) for k in order}

    # ---------- Alpha t-tests ----------
    out_alpha = outdir / ("alpha_ttest.tsv" if name == "micro" else "alpha_ttest_mito.tsv")
    if "Shannon" in df.columns and "type_group" in df.columns:
        ttab = fdr_ttests_by_group(df, "type_group", "Shannon")
        ttab.to_csv(out_alpha, sep="\t", index=False)

        # Boxplot with stars
        g = sns.catplot(
            data=df, x="type_group", y="Shannon", hue="type_group",
            kind="box", order=order, palette=type_palette,
            saturation=1, boxprops=dict(alpha=.5), height=8, aspect=1.2
        )
        g.set_xticklabels(rotation=45)
        g.set_axis_labels("", "Shannon")
        g.fig.tight_layout()
        for ext in ("svg", "pdf"):
            g.fig.savefig(outdir / f"Alpha_type_boxplot_{name}.{ext}")
        plt.close(g.fig)

        # Faceted by status if present
        if "status" in df.columns:
            g = sns.catplot(
                data=df, x="status", y="Shannon", hue="status",
                col="type_group", col_wrap=3, sharey=True, order=["Cancer", "Non-Cancer"],
                kind="box", palette=PALETTE_STATUS, height=4
            )
            g.set_xticklabels(rotation=45)
            g.set_axis_labels("", "Shannon")
            g.fig.tight_layout()
            for ext in ("svg", "pdf"):
                g.fig.savefig(outdir / f"Alpha_status_boxplot_{name}.{ext}")
            plt.close(g.fig)

    # ---------- PERMANOVA ----------
    if not bray.empty and "type_group" in df.columns:
        # Align groups to Bray
        groups = df.set_index("sample").loc[bray.index.intersection(df["sample"]), "type_group"]
        glob, pair = global_and_pairwise_permanova(bray, groups, permutations=permutations)
        glob.to_csv(outdir / ("permanova_global.tsv" if name == "micro" else "permanova_global_mito.tsv"), sep="\t", index=False)
        pair.to_csv(outdir / ("permanova_pairwise.tsv" if name == "micro" else "permanova_pairwise_mito.tsv"), sep="\t", index=False)

        # Heatmap of pairwise q-values
        heatmap_pairwise_q(
            pair,
            outdir / f"Beta_Heatmap_permanova_{name}",
            title="Pairwise PERMANOVA (q-values)\nBlue = Not Significant, Red = Significant"
        )

    # ---------- UMAP figures (Bray) ----------
    have_umap = {"UMAP1", "UMAP2"}.issubset(df.columns)
    if have_umap and "type_group" in df.columns:
        size_col = "count" if "count" in df.columns else None

        scatter_umap(
            df, "UMAP1", "UMAP2",
            hue="type_group", size=size_col, style=None,
            palette=type_palette,
            title="UMAP colored by Sample Type",
            out_base=outdir / f"Beta_UMAP_type_{name}"
        )
        if "lung_code" in df.columns:
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="type_group", size=size_col, style="lung_code",
                palette=type_palette,
                title="UMAP by Type (style = lung_code)",
                out_base=outdir / f"Beta_UMAP_type_lung_{name}"
            )
        if "status" in df.columns:
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="type_group", size=size_col, style="status",
                palette=type_palette,
                title="UMAP by Type (style = status)",
                out_base=outdir / f"Beta_UMAP_type_status_{name}"
            )
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="status", size=size_col, style=None,
                palette=PALETTE_STATUS,
                title="UMAP colored by Cancer Status",
                out_base=outdir / f"Beta_UMAP_status_{name}"
            )
        for flag, label in (("overall_OL", "study-wide outliers"), ("typ_grp_OL", "type-wise outliers")):
            if flag in df.columns:
                scatter_umap(
                    df, "UMAP1", "UMAP2",
                    hue="type_group", size=size_col, style=flag,
                    palette=type_palette,
                    title=f"UMAP by Type (style = {label})",
                    out_base=outdir / f"Beta_UMAP_type_{flag}_{name}"
                )

# -------------------------- CLI --------------------------
def main():
    p = argparse.ArgumentParser(description="Alpha stats, PERMANOVA, and UMAP plots from precomputed distances.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--metadata", required=True, type=Path, help="Metadata TSV with at least 'sample' and 'type_group'.")
    p.add_argument("--master",   required=False, type=Path, help="Master table with counts/pass_filter (optional).")
    p.add_argument("--alpha",    required=False, type=Path, help="Shannon TSV (sample,Shannon).")
    p.add_argument("--bray",     required=False, type=Path, help="Bray-Curtis square distance TSV (index=sample).")
    p.add_argument("--jacc",     required=False, type=Path, help="Jaccard square distance TSV (index=sample).")
    p.add_argument("--outliers-all",  type=Path, help="Optional outliers table with columns: sample,is_outlier (overall_OL).")
    p.add_argument("--outliers-type", type=Path, help="Optional outliers table with columns: sample,is_outlier (typ_grp_OL).")
    p.add_argument("--exclude-types", default="", help="Comma-separated type_group values to exclude.")
    p.add_argument("--type-order",    default="", help="Comma-separated plotting order for type_group.")

    # mito inputs (optional)
    p.add_argument("--mito-alpha", type=Path, help="Mito Shannon TSV.")
    p.add_argument("--mito-bray",  type=Path, help="Mito Bray TSV.")
    p.add_argument("--mito-jacc",  type=Path, help="Mito Jaccard TSV.")

    # UMAP / PERMANOVA params
    p.add_argument("--neighbors", type=int, default=30, help="UMAP neighbors for precomputed distances.")
    p.add_argument("--min-dist",  type=float, default=0.01, help="UMAP min_dist for precomputed distances.")
    p.add_argument("--permutations", type=int, default=999, help="PERMANOVA permutations.")

    # Output dirs
    p.add_argument("--outdir", type=Path, required=True, help="Output directory (micro).")
    p.add_argument("--mito-outdir", type=Path, default=None, help="Output directory for mito pass.")

    args = p.parse_args()

    # Read inputs
    meta = read_tsv(args.metadata)
    if "sample" not in meta.columns:
        raise ValueError("metadata must contain a 'sample' column.")

    master = read_tsv(args.master) if args.master else pd.DataFrame()
    alpha  = read_tsv(args.alpha, index_col=0) if args.alpha else pd.DataFrame()
    bray   = read_tsv(args.bray,  index_col=0) if args.bray  else pd.DataFrame()
    jacc   = read_tsv(args.jacc,  index_col=0) if args.jacc  else pd.DataFrame()

    # Attach outlier flags if provided
    if args.outliers_all and Path(args.outliers_all).exists():
        o = read_tsv(args.outliers_all)
        if {"sample", "is_outlier"}.issubset(o.columns):
            o = o.rename(columns={"is_outlier": "overall_OL"})
            meta = safe_merge(meta, o, on="sample")
    if args.outliers_type and Path(args.outliers_type).exists():
        o = read_tsv(args.outliers_type)
        if {"sample", "is_outlier"}.issubset(o.columns):
            o = o.rename(columns={"is_outlier": "typ_grp_OL"})
            meta = safe_merge(meta, o, on="sample")

    exclude_types = [t.strip() for t in args.exclude_types.split(",") if t.strip()]
    type_order    = [t.strip() for t in args.type_order.split(",") if t.strip()] if args.type_order else None

    # MICRO pass
    ensure_dir(args.outdir)
    run_one_pass(
        name="micro",
        meta=meta, master=master, alpha=alpha, bray=bray, jacc=jacc,
        outdir=args.outdir,
        exclude_types=exclude_types, type_order=type_order,
        neighbors=args.neighbors, min_dist=args.min_dist,
        permutations=args.permutations
    )

    # MITO pass (optional)
    if args.mito_alpha or args.mito_bray or args.mito_jacc:
        mito_dir = args.mito_outdir if args.mito_outdir else args.outdir.parent / "mito" / "diversity"
        ensure_dir(mito_dir)
        mito_alpha = read_tsv(args.mito_alpha, index_col=0) if args.mito_alpha else pd.DataFrame()
        mito_bray  = read_tsv(args.mito_bray,  index_col=0) if args.mito_bray  else pd.DataFrame()
        mito_jacc  = read_tsv(args.mito_jacc,  index_col=0) if args.mito_jacc  else pd.DataFrame()

        run_one_pass(
            name="mito",
            meta=meta, master=master, alpha=mito_alpha, bray=mito_bray, jacc=mito_jacc,
            outdir=mito_dir,
            exclude_types=exclude_types, type_order=type_order,
            neighbors=args.neighbors, min_dist=args.min_dist,
            permutations=args.permutations
        )

if __name__ == "__main__":
    main()
