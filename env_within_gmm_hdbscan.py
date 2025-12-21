#!/usr/bin/env python3
"""
env_within_gmm_hdbscan.py

HDBSCAN-within-GMM subclustering in PC space.

Inputs (from your pipeline):
1) eigengenes_scores.csv (from env_eigengenes.py)
   Required cols: cruise_year_month_depth, PC1..PC*
2) compartments_assignments_smoothed.csv (from env_compartments_gmm.py)
   Required cols: cruise_year_month_depth, component (and optional max_prob, entropy_norm, resp_*)

What it does:
- Merge eigengenes PC coordinates with GMM assignments on cruise_year_month_depth.
- For each GMM component, run HDBSCAN in PC space to find subclusters (or noise).
- Saves:
  - within_gmm_hdbscan_assignments.csv (per-sample: component + subcluster labels)
  - within_gmm_hdbscan_summary.csv (per component: counts, noise frac, #subclusters, etc.)
  - Per-component scatter plots (PC1 vs PC2) colored by subcluster

Notes:
- This is NOT a replacement for GMM.
- It is intended to detect secondary structure (episodic/rare submodes) *within* each GMM component.

Dependencies:
- pandas, numpy, matplotlib, scikit-learn
- hdbscan  (pip/conda install: hdbscan)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

try:
    import hdbscan  # type: ignore
except Exception as e:  # pragma: no cover
    hdbscan = None
    _HDBSCAN_IMPORT_ERR = str(e)
else:
    _HDBSCAN_IMPORT_ERR = ""


# -----------------------------
# Config
# -----------------------------

@dataclass
class RunConfig:
    eigengenes: str
    assignments: str
    outdir: str
    sep: str
    id_col: str
    component_col: str

    # PC selection
    pc_cols: Optional[List[str]]
    pc_prefix: str
    pc_min: int
    pc_max: int

    # Preprocessing
    standardize_pc_space: bool

    # HDBSCAN params
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: Optional[int]
    hdbscan_metric: str
    hdbscan_cluster_selection_method: str
    hdbscan_allow_single_cluster: bool

    # Filtering
    min_rows_per_component: int
    high_conf_only: bool
    high_conf_maxprob: float

    # Safety
    strict_unique_ids: bool


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="Run HDBSCAN subclustering within each GMM component in PC space.")

    ap.add_argument("--eigengenes", required=True, help="Path to eigengenes_scores.csv (contains PC columns).")
    ap.add_argument("--assignments", required=True, help="Path to compartments_assignments_smoothed.csv (contains component).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default=",", help="Delimiter (default ','). Use $'\\t' for TSV in bash.")

    ap.add_argument("--id-col", default="cruise_year_month_depth", help="Sample ID column for merging.")
    ap.add_argument("--component-col", default="component", help="GMM component label column.")

    # PC columns: either explicit list, or infer by prefix/number range
    ap.add_argument(
        "--pc-cols",
        default=None,
        help="Comma-separated explicit PC columns to use (e.g., 'PC1,PC2,PC3'). If omitted, infer from prefix + range.",
    )
    ap.add_argument("--pc-prefix", default="PC", help="PC column prefix (default 'PC').")
    ap.add_argument("--pc-min", type=int, default=1, help="Minimum PC index to consider (default 1).")
    ap.add_argument("--pc-max", type=int, default=50, help="Maximum PC index to consider (default 50).")

    ap.add_argument(
        "--standardize-pc-space",
        action="store_true",
        help="Standardize the selected PC columns prior to HDBSCAN within each component.",
    )

    # HDBSCAN params
    ap.add_argument("--hdbscan-min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size (default 50).")
    ap.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (default None -> library default).",
    )
    ap.add_argument(
        "--hdbscan-metric",
        default="manhattan",
        help="Distance metric for HDBSCAN (default 'manhattan').",
    )
    ap.add_argument(
        "--hdbscan-cluster-selection-method",
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster_selection_method (default eom).",
    )
    ap.add_argument(
        "--hdbscan-allow-single-cluster",
        action="store_true",
        help="Allow HDBSCAN to return a single cluster (default False).",
    )

    # Filtering
    ap.add_argument(
        "--min-rows-per-component",
        type=int,
        default=100,
        help="Skip HDBSCAN for components with < this many rows (default 100).",
    )
    ap.add_argument(
        "--high-conf-only",
        action="store_true",
        help="Only run subclustering on high-confidence GMM assignments (requires max_prob column).",
    )
    ap.add_argument(
        "--high-conf-maxprob",
        type=float,
        default=0.80,
        help="Threshold for high-confidence if --high-conf-only is set (default 0.80).",
    )

    # Safety
    ap.add_argument(
        "--strict-unique-ids",
        action="store_true",
        help="If set, error out if id_col is not unique in either table (recommended).",
    )

    ns = ap.parse_args()

    pc_cols = None
    if ns.pc_cols:
        pc_cols = [c.strip() for c in ns.pc_cols.split(",") if c.strip()]

    return RunConfig(
        eigengenes=ns.eigengenes,
        assignments=ns.assignments,
        outdir=ns.outdir,
        sep=ns.sep,
        id_col=ns.id_col,
        component_col=ns.component_col,
        pc_cols=pc_cols,
        pc_prefix=ns.pc_prefix,
        pc_min=ns.pc_min,
        pc_max=ns.pc_max,
        standardize_pc_space=ns.standardize_pc_space,
        hdbscan_min_cluster_size=ns.hdbscan_min_cluster_size,
        hdbscan_min_samples=ns.hdbscan_min_samples,
        hdbscan_metric=ns.hdbscan_metric,
        hdbscan_cluster_selection_method=ns.hdbscan_cluster_selection_method,
        hdbscan_allow_single_cluster=ns.hdbscan_allow_single_cluster,
        min_rows_per_component=ns.min_rows_per_component,
        high_conf_only=ns.high_conf_only,
        high_conf_maxprob=ns.high_conf_maxprob,
        strict_unique_ids=ns.strict_unique_ids,
    )


# -----------------------------
# Helpers
# -----------------------------

def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def read_table(path: str, sep: str) -> pd.DataFrame:
    # Don't force dtype=str because PC columns are numeric.
    return pd.read_csv(path, sep=sep)


def assert_unique_id(df: pd.DataFrame, id_col: str, label: str, strict: bool) -> None:
    if id_col not in df.columns:
        raise ValueError(f"[{label}] Missing required id_col '{id_col}'")
    n = df.shape[0]
    n_uniq = df[id_col].nunique(dropna=False)
    if n_uniq != n:
        msg = f"[{label}] id_col '{id_col}' is NOT unique: rows={n}, unique={n_uniq}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"[WARN] {msg} (continuing, but merge may duplicate rows)")


def infer_pc_cols(df: pd.DataFrame, prefix: str, pc_min: int, pc_max: int) -> List[str]:
    cols = []
    for i in range(pc_min, pc_max + 1):
        c = f"{prefix}{i}"
        if c in df.columns:
            cols.append(c)
    return cols


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_component_subclusters(
    df: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    outpath: str,
    title: str,
    subcol: str = "subcluster_in_component",
) -> None:
    if pc_x not in df.columns or pc_y not in df.columns:
        return

    # noise = -1
    labels = df[subcol].astype(int).values
    x = df[pc_x].values
    y = df[pc_y].values

    plt.figure(figsize=(7.5, 6.0))

    # Plot noise first (grey), then clusters
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(x[noise_mask], y[noise_mask], s=12, alpha=0.35, label="noise (-1)")

    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        m = labels == lab
        plt.scatter(x[m], y[m], s=16, alpha=0.75, label=f"sub{lab}")

    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0, frameon=False)
    save_fig(outpath)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()

    if hdbscan is None:
        raise RuntimeError(
            "hdbscan is not installed or failed to import.\n"
            f"Import error: {_HDBSCAN_IMPORT_ERR}\n\n"
            "Install via conda-forge:\n"
            "  conda install -c conda-forge hdbscan\n"
            "or pip:\n"
            "  pip install hdbscan"
        )

    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    # Load
    df_pc = read_table(cfg.eigengenes, cfg.sep)
    df_asg = read_table(cfg.assignments, cfg.sep)

    # Validate ID uniqueness
    assert_unique_id(df_pc, cfg.id_col, "eigengenes", cfg.strict_unique_ids)
    assert_unique_id(df_asg, cfg.id_col, "assignments", cfg.strict_unique_ids)

    if cfg.component_col not in df_asg.columns:
        raise ValueError(f"assignments file missing required component col '{cfg.component_col}'")

    # Choose PCs
    if cfg.pc_cols is not None:
        pc_cols = [c for c in cfg.pc_cols if c in df_pc.columns]
        missing = [c for c in cfg.pc_cols if c not in df_pc.columns]
        if missing:
            print(f"[WARN] These requested PC columns were missing in eigengenes table and will be ignored: {missing}")
    else:
        pc_cols = infer_pc_cols(df_pc, cfg.pc_prefix, cfg.pc_min, cfg.pc_max)

    if len(pc_cols) == 0:
        raise ValueError("No PC columns found. Use --pc-cols or adjust --pc-prefix/--pc-min/--pc-max.")

    # Coerce PC cols to numeric
    df_pc = coerce_numeric(df_pc, pc_cols)

    # Merge: keep all assignment rows
    # Only bring in the PC columns + id
    df = df_asg.merge(df_pc[[cfg.id_col] + pc_cols], on=cfg.id_col, how="left", validate="one_to_one" if cfg.strict_unique_ids else None)

    # Drop rows missing PC coords
    n0 = df.shape[0]
    pc_complete_mask = df[pc_cols].notna().all(axis=1)
    df = df.loc[pc_complete_mask].copy()
    n1 = df.shape[0]
    if n1 < n0:
        print(f"[WARN] Dropped {n0 - n1} rows missing one or more selected PCs (kept {n1}).")

    # Optional high-confidence filter (only for running HDBSCAN; we still write all rows with NaNs for skipped)
    if cfg.high_conf_only:
        if "max_prob" not in df.columns:
            raise ValueError("--high-conf-only requires 'max_prob' in assignments table.")
        df["_high_conf"] = pd.to_numeric(df["max_prob"], errors="coerce") >= cfg.high_conf_maxprob
    else:
        df["_high_conf"] = True

    # Prepare outputs columns
    df["subcluster"] = -1  # global label (unique across components)
    df["subcluster_in_component"] = -1  # label within each component
    df["subcluster_prob"] = np.nan  # HDBSCAN membership strength/probability (approx)

    summary_rows = []
    global_subcluster_id = 0

    # Work per component
    components = sorted(df[cfg.component_col].dropna().unique().tolist())
    for comp in components:
        m_comp = (df[cfg.component_col] == comp)
        df_comp = df.loc[m_comp].copy()

        # apply confidence filter for fitting
        df_fit = df_comp.loc[df_comp["_high_conf"]].copy()

        n_comp = df_comp.shape[0]
        n_fit = df_fit.shape[0]

        if n_fit < cfg.min_rows_per_component:
            summary_rows.append({
                "component": comp,
                "n_total_in_component": int(n_comp),
                "n_used_for_hdbscan": int(n_fit),
                "skipped": True,
                "skip_reason": f"n_used_for_hdbscan<{cfg.min_rows_per_component}",
                "n_subclusters_found": 0,
                "n_noise": int(n_fit),
                "frac_noise": 1.0 if n_fit > 0 else np.nan,
                "min_cluster_size_param": int(cfg.hdbscan_min_cluster_size),
            })
            continue

        X = df_fit[pc_cols].values.astype(float)

        # Optional standardization within component (recommended if PC variances differ)
        if cfg.standardize_pc_space:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X = scaler.fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            metric=cfg.hdbscan_metric,
            cluster_selection_method=cfg.hdbscan_cluster_selection_method,
            allow_single_cluster=cfg.hdbscan_allow_single_cluster,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)  # -1 is noise
        probs = getattr(clusterer, "probabilities_", None)

        # Count clusters
        labs = labels.astype(int)
        uniq = sorted(set(labs.tolist()))
        clusters = [u for u in uniq if u != -1]
        n_clusters = len(clusters)
        n_noise = int(np.sum(labs == -1))
        frac_noise = float(n_noise / len(labs)) if len(labs) else np.nan

        # Map within-component labels to global unique labels
        within_to_global = {}
        for u in clusters:
            within_to_global[int(u)] = global_subcluster_id
            global_subcluster_id += 1

        # Write back into main df for just the rows we fit
        fit_idx = df_fit.index.to_numpy()
        sub_in_comp = np.array([int(u) for u in labs], dtype=int)
        sub_global = np.array([within_to_global.get(int(u), -1) for u in sub_in_comp], dtype=int)

        df.loc[fit_idx, "subcluster_in_component"] = sub_in_comp
        df.loc[fit_idx, "subcluster"] = sub_global
        if probs is not None:
            df.loc[fit_idx, "subcluster_prob"] = probs.astype(float)

        summary_rows.append({
            "component": comp,
            "n_total_in_component": int(n_comp),
            "n_used_for_hdbscan": int(n_fit),
            "skipped": False,
            "skip_reason": "",
            "n_subclusters_found": int(n_clusters),
            "n_noise": int(n_noise),
            "frac_noise": float(frac_noise),
            "min_cluster_size_param": int(cfg.hdbscan_min_cluster_size),
            "min_samples_param": int(cfg.hdbscan_min_samples) if cfg.hdbscan_min_samples is not None else None,
            "metric": cfg.hdbscan_metric,
            "cluster_selection_method": cfg.hdbscan_cluster_selection_method,
            "allow_single_cluster": bool(cfg.hdbscan_allow_single_cluster),
            "standardize_pc_space_within_component": bool(cfg.standardize_pc_space),
            "high_conf_only": bool(cfg.high_conf_only),
            "high_conf_maxprob": float(cfg.high_conf_maxprob) if cfg.high_conf_only else None,
        })

        # Plot PC1 vs PC2 if available
        if "PC1" in pc_cols and "PC2" in pc_cols:
            # Use the fitted subset for the plot (cleaner)
            plot_df = df.loc[fit_idx].copy()
            title = f"HDBSCAN within GMM component={comp} (n_fit={n_fit}, clusters={n_clusters}, noise={frac_noise:.2f})"
            plot_component_subclusters(
                plot_df,
                pc_x="PC1",
                pc_y="PC2",
                outpath=os.path.join(plots_dir, f"within_gmm_component_{comp}_PC1_PC2_subclusters.png"),
                title=title,
                subcol="subcluster_in_component",
            )

    # Save outputs
    # Drop helper col
    df_out = df.drop(columns=["_high_conf"], errors="ignore")

    # Save config + metadata
    with open(os.path.join(cfg.outdir, "within_gmm_hdbscan_run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    df_out.to_csv(os.path.join(tables_dir, "within_gmm_hdbscan_assignments.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(tables_dir, "within_gmm_hdbscan_summary.csv"), index=False)

    # Simple overall stats
    overall = {
        "n_rows_after_merge_and_pc_complete": int(df_out.shape[0]),
        "pc_cols_used": pc_cols,
        "n_components_seen": int(len(components)),
        "n_global_subclusters_assigned": int(df_out["subcluster"].astype(int).max() + 1) if df_out.shape[0] else 0,
        "frac_rows_noise_overall": float((df_out["subcluster_in_component"].astype(int) == -1).mean()) if df_out.shape[0] else np.nan,
    }
    with open(os.path.join(cfg.outdir, "within_gmm_hdbscan_qc_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")
    print(f"     PCs used: {', '.join(pc_cols)}")
    if cfg.high_conf_only:
        print(f"     High-conf only: YES (max_prob >= {cfg.high_conf_maxprob})")
    else:
        print("     High-conf only: NO")


if __name__ == "__main__":
    main()
