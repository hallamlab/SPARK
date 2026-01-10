#!/usr/bin/env python3
"""
scripts/env_pca_summary.py

Post-processing summary plots for env_eigengenes.py outputs.

Reads:
  <outdir>/tables/eigengenes_scores.csv
  <outdir>/tables/pca_loadings.csv
  <outdir>/tables/pca_explained_variance.csv
  <outdir>/tables/pc_keep_decision.csv (optional, if --pc-selection was run)
  <outdir>/tables/feature_clusters.csv (optional)
  <outdir>/tables/pc_stability.csv (optional)

Writes:
  <outdir>/summary_plots/*.png

USAGE:
  python scripts/env_pca_summary.py --outdir ../V4_ncbi_output/env_pca

Optional:
  --cluster-samples            # discover compartments by clustering samples in KEEP-PC space
  --cluster-method hdbscan     # hdbscan (if installed) or kmeans
  --n-clusters 6               # for kmeans
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory used by env_eigengenes.py")
    ap.add_argument("--depth-col", default="Depth", help="Depth column name in eigengenes_scores.csv (default Depth)")
    ap.add_argument("--time-col", default="date", help="Time column name in eigengenes_scores.csv (default date)")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name in eigengenes_scores.csv (default Cruise)")
    ap.add_argument("--oxygen-col", default="Oxygen", help="Oxygen column name (if present) for comparison plots")

    ap.add_argument("--cluster-samples", action="store_true", help="Cluster samples into compartments using KEEP PCs.")
    ap.add_argument("--cluster-method", choices=["hdbscan", "kmeans"], default="kmeans")
    ap.add_argument("--n-clusters", type=int, default=6, help="kmeans clusters (only used if cluster-method=kmeans)")
    ap.add_argument("--random-state", type=int, default=42)

    return ap.parse_args()


def p(path: str) -> str:
    return os.path.abspath(path)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(outpath: str) -> None:
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def read_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_pc_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("PC") and c[2:].isdigit()]


def pick_keep_pcs(pc_keep: Optional[pd.DataFrame], available_pcs: List[str]) -> List[str]:
    if pc_keep is None or "KEEP" not in pc_keep.columns:
        # fallback: first 3 available PCs if KEEP decision isn't available
        return available_pcs[: min(3, len(available_pcs))]
    keep = pc_keep.loc[pc_keep["KEEP"] == True, "PC"].astype(str).tolist()
    keep = [pc for pc in keep if pc in available_pcs]
    if len(keep) == 0:
        # fallback: PCs that passed parallel only (if present)
        if "keep_parallel" in pc_keep.columns:
            keep = pc_keep.loc[pc_keep["keep_parallel"] == True, "PC"].astype(str).tolist()
            keep = [pc for pc in keep if pc in available_pcs]
    return keep[: min(len(keep), len(available_pcs))]


def plot_keep_table(pc_keep: pd.DataFrame, outpath: str) -> None:
    cols = [c for c in ["PC", "keep_parallel", "keep_coverage", "keep_coherence", "keep_stability", "KEEP"] if c in pc_keep.columns]
    if not cols:
        return
    d = pc_keep[cols].copy()
    d = d.head(12)  # keep it readable

    # Convert bools to 0/1 for heatmap
    mat = []
    row_labels = d["PC"].tolist()
    col_labels = [c for c in cols if c != "PC"]
    for _, r in d.iterrows():
        mat.append([1 if bool(r[c]) else 0 for c in col_labels])
    mat = np.array(mat, dtype=float)

    plt.figure(figsize=(1.6 + 1.2 * len(col_labels), 1 + 0.5 * len(row_labels)))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.title("PC keep decision summary (1=True, 0=False)")
    plt.colorbar(label="Keep flag")
    save_fig(outpath)


def plot_explained_variance(ev: pd.DataFrame, outpath_scree: str, outpath_cum: str) -> None:
    if ev is None or "explained_variance_ratio" not in ev.columns:
        return
    x = np.arange(1, ev.shape[0] + 1)
    y = ev["explained_variance_ratio"].values
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree plot (explained variance ratio)")
    save_fig(outpath_scree)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, np.cumsum(y), marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("PC")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    save_fig(outpath_cum)


def plot_pc_scatter(df: pd.DataFrame, pcx: str, pcy: str, color_col: Optional[str], outpath: str) -> None:
    plt.figure(figsize=(7, 6))
    if color_col and color_col in df.columns:
        c = df[color_col]
        # If time-like, color by rank
        if np.issubdtype(c.dtype, np.datetime64):
            v = c.rank(method="first").fillna(0).to_numpy()
        else:
            # numeric or string → try numeric, else rank
            v = pd.to_numeric(c, errors="coerce")
            if v.isna().all():
                v = c.astype(str).rank(method="dense").to_numpy()
            else:
                v = v.fillna(v.median()).to_numpy()
        plt.scatter(df[pcx], df[pcy], c=v)
        plt.colorbar(label=color_col)
    else:
        plt.scatter(df[pcx], df[pcy])
    plt.xlabel(pcx)
    plt.ylabel(pcy)
    plt.title(f"{pcx} vs {pcy}")
    save_fig(outpath)


def plot_pc_vs_depth(df: pd.DataFrame, pc: str, depth_col: str, color_col: Optional[str], outpath: str) -> None:
    if depth_col not in df.columns:
        return
    plt.figure(figsize=(7, 6))
    y = pd.to_numeric(df[depth_col], errors="coerce")
    x = df[pc]
    if color_col and color_col in df.columns:
        c = df[color_col].astype(str).rank(method="dense").to_numpy()
        plt.scatter(x, y, c=c, s=14)
        plt.colorbar(label=color_col)
    else:
        plt.scatter(x, y, s=14)
    plt.gca().invert_yaxis()
    plt.xlabel(pc)
    plt.ylabel(depth_col)
    plt.title(f"{pc} vs {depth_col}")
    save_fig(outpath)


def cluster_samples(scores: pd.DataFrame, pcs: List[str], method: str, n_clusters: int, random_state: int) -> pd.Series:
    X = scores[pcs].values.astype(float)

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except Exception as e:
            raise RuntimeError("cluster-method=hdbscan requested but hdbscan is not installed.") from e
        # Conservative defaults
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=10)
        labels = clusterer.fit_predict(X)
        return pd.Series(labels, index=scores.index, name="compartment")

    # kmeans fallback
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return pd.Series(labels, index=scores.index, name="compartment")


def plot_compartment_depth_cruise(df: pd.DataFrame, depth_col: str, cruise_col: str, comp_col: str, outpath: str) -> None:
    if depth_col not in df.columns or cruise_col not in df.columns or comp_col not in df.columns:
        return

    d = df.copy()
    d[depth_col] = pd.to_numeric(d[depth_col], errors="coerce")
    d = d.dropna(subset=[depth_col, cruise_col, comp_col])

    # Bin depth a bit to make a readable raster
    depth_bins = np.unique(np.round(d[depth_col].values, 0))
    cruises = sorted(d[cruise_col].astype(str).unique().tolist())

    # Build matrix with mode compartment per (cruise, depth)
    mat = np.full((len(depth_bins), len(cruises)), np.nan)
    for j, cr in enumerate(cruises):
        sub = d[d[cruise_col].astype(str) == cr]
        for i, dep in enumerate(depth_bins):
            s = sub[np.isclose(sub[depth_col].values, dep)]
            if s.shape[0] == 0:
                continue
            # mode label
            mat[i, j] = float(pd.Series(s[comp_col]).mode().iloc[0])

    plt.figure(figsize=(max(8, 0.35 * len(cruises)), max(5, 0.18 * len(depth_bins))))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(depth_bins)), depth_bins)
    plt.xticks(np.arange(len(cruises)), cruises, rotation=90)
    plt.gca().invert_yaxis()
    plt.xlabel(cruise_col)
    plt.ylabel(f"{depth_col} (binned)")
    plt.title("Compartment map (mode label) by Cruise × Depth")
    plt.colorbar(label="Compartment label")
    save_fig(outpath)


def plot_compartment_vs_oxygen(df: pd.DataFrame, comp_col: str, oxygen_col: str, outpath: str) -> None:
    if comp_col not in df.columns or oxygen_col not in df.columns:
        return
    d = df.copy()
    d[oxygen_col] = pd.to_numeric(d[oxygen_col], errors="coerce")
    d = d.dropna(subset=[oxygen_col, comp_col])
    comps = sorted(d[comp_col].unique().tolist())

    data = [d.loc[d[comp_col] == c, oxygen_col].values for c in comps]

    plt.figure(figsize=(max(8, 1 + 0.6 * len(comps)), 5))
    plt.boxplot(data, labels=[str(c) for c in comps], showfliers=False)
    plt.xlabel("Compartment")
    plt.ylabel(oxygen_col)
    plt.title(f"{oxygen_col} distributions by compartment")
    save_fig(outpath)


def main():
    args = parse_args()

    outdir = p(args.outdir)
    tables_dir = os.path.join(outdir, "tables")
    plots_dir = ensure_dir(os.path.join(outdir, "summary_plots"))

    eig_path = os.path.join(tables_dir, "eigengenes_scores.csv")
    load_path = os.path.join(tables_dir, "pca_loadings.csv")
    ev_path = os.path.join(tables_dir, "pca_explained_variance.csv")
    keep_path = os.path.join(tables_dir, "pc_keep_decision.csv")
    stab_path = os.path.join(tables_dir, "pc_stability.csv")

    eig = read_if_exists(eig_path)
    if eig is None:
        raise FileNotFoundError(f"Missing required file: {eig_path}")

    # Coerce date if present
    if args.time_col in eig.columns:
        eig[args.time_col] = pd.to_datetime(eig[args.time_col], errors="coerce")

    loadings = read_if_exists(load_path)
    ev = read_if_exists(ev_path)
    pc_keep = read_if_exists(keep_path)
    pc_stab = read_if_exists(stab_path)

    pc_cols = get_pc_cols(eig)
    if len(pc_cols) == 0:
        raise ValueError("No PC columns found in eigengenes_scores.csv")

    keep_pcs = pick_keep_pcs(pc_keep, pc_cols)
    if len(keep_pcs) == 0:
        keep_pcs = pc_cols[: min(3, len(pc_cols))]

    # Basic PCA summary
    plot_explained_variance(
        ev,
        outpath_scree=os.path.join(plots_dir, "scree.png"),
        outpath_cum=os.path.join(plots_dir, "cumulative_variance.png"),
    )

    if pc_keep is not None:
        plot_keep_table(pc_keep, os.path.join(plots_dir, "pc_keep_decision_heatmap.png"))

    if pc_stab is not None and "PC" in pc_stab.columns:
        d = pc_stab.copy()
        if "med_abs_corr_loadings" in d.columns and "med_abs_corr_scores" in d.columns:
            plt.figure(figsize=(8, 5))
            plt.scatter(d["med_abs_corr_loadings"], d["med_abs_corr_scores"])
            for _, r in d.iterrows():
                plt.text(r["med_abs_corr_loadings"], r["med_abs_corr_scores"], str(r["PC"]), fontsize=8)
            plt.xlabel("Median |corr| loadings")
            plt.ylabel("Median |corr| scores")
            plt.title("PC stability scatter")
            save_fig(os.path.join(plots_dir, "pc_stability_scatter.png"))

    # PC-vs-depth for each KEEP PC
    for pc in keep_pcs:
        plot_pc_vs_depth(
            eig,
            pc=pc,
            depth_col=args.depth_col,
            color_col=args.cruise_col if args.cruise_col in eig.columns else None,
            outpath=os.path.join(plots_dir, f"{pc}_vs_depth.png"),
        )

    # Pairwise KEEP PC scatters
    # Color by depth if possible; else by time if possible
    color_col = args.depth_col if args.depth_col in eig.columns else (args.time_col if args.time_col in eig.columns else None)
    for i in range(len(keep_pcs)):
        for j in range(i + 1, len(keep_pcs)):
            pcx, pcy = keep_pcs[i], keep_pcs[j]
            plot_pc_scatter(
                eig,
                pcx=pcx,
                pcy=pcy,
                color_col=color_col,
                outpath=os.path.join(plots_dir, f"{pcx}_vs_{pcy}.png"),
            )

    # Optional: discover compartments in KEEP-PC space and plot maps
    if args.cluster_samples:
        labels = cluster_samples(
            scores=eig,
            pcs=keep_pcs,
            method=args.cluster_method,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )
        eig2 = eig.copy()
        eig2["compartment"] = labels.values
        eig2.to_csv(os.path.join(tables_dir, "sample_compartments.csv"), index=False)

        # Cruise × depth map
        if args.depth_col in eig2.columns and args.cruise_col in eig2.columns:
            plot_compartment_depth_cruise(
                eig2,
                depth_col=args.depth_col,
                cruise_col=args.cruise_col,
                comp_col="compartment",
                outpath=os.path.join(plots_dir, "compartment_map_cruise_by_depth.png"),
            )

        # Oxygen comparison (if oxygen present)
        if args.oxygen_col in eig2.columns:
            plot_compartment_vs_oxygen(
                eig2,
                comp_col="compartment",
                oxygen_col=args.oxygen_col,
                outpath=os.path.join(plots_dir, "oxygen_by_compartment_boxplot.png"),
            )

    print(f"[OK] Wrote summary plots to: {plots_dir}")
    print(f"[INFO] KEEP PCs used: {keep_pcs}")
    if args.cluster_samples:
        print(f"[OK] Wrote sample compartments to: {os.path.join(tables_dir, 'sample_compartments.csv')}")


if __name__ == "__main__":
    main()
