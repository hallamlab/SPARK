#!/usr/bin/env python3
# ==============================================================================
# File: scripts/env_eof_mode_plots.py
#
# Purpose
# -------
# Plot and interpret EOF (PCA/SVD) modes produced by:
#   scripts/env_profile_eof.py
#
# This script reads:
#   1) eof_loadings.tsv  (feature x EOFk)
#   2) optional: eof_explained_variance.tsv
#
# and produces:
#   A) One figure per EOF:
#      - heatmap: variables (rows) x depth bins (cols) showing loadings
#      - plus a compact “top contributing features” table for traceability
#
# Why this is defensible
# ----------------------
# Your EOF features are built as: "{var}@{depth}"
# where depth is on a common grid across cruises.
# Therefore each EOF loading vector can be reshaped into a variable×depth matrix.
# This directly shows the vertical profile pattern each EOF represents
# (e.g., oxycline shift, mixed profile, sulfidic bottom, nutrient accumulation).
#
# Inputs required
# ---------------
# --loadings (required)
#   Path to eof_loadings.tsv written by env_profile_eof.py
#   Format:
#     feature    EOF1   EOF2  ...
#     o2@0       ...    ...
#     o2@5       ...    ...
#     no3@0      ...    ...
#
# --outdir (required)
#   Output directory for figures.
#
# Optional inputs
# ---------------
# --explained (optional)
#   Path to eof_explained_variance.tsv for labeling plots with % variance explained.
#
# How features must be named
# --------------------------
# This script expects features follow the naming convention:
#   "<variable>@<depth>"
# where <depth> is numeric (e.g., "o2@0", "o2@5", "Temperature@100").
#
# Output
# ------
# For each EOF i:
#   - eof_mode_EOF{i}.png
#   - eof_mode_EOF{i}.pdf
#   - eof_mode_EOF{i}_top_features.tsv
#
# Dependencies
# ------------
# pip install pandas numpy matplotlib
#
# Notes
# -----
# - This script does NOT choose colors explicitly (uses default colormaps).
# - Loadings are signed: positive vs negative indicates opposite phase.
#   Interpretation should be made in conjunction with EOF scores (PC time series).
# ==============================================================================

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_RE = re.compile(r"^(?P<var>.+)@(?P<depth>-?\d+(\.\d+)?)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot EOF modes as variable×depth loading heatmaps.")
    ap.add_argument("--loadings", required=True, help="Path to eof_loadings.tsv (feature x EOF columns).")
    ap.add_argument("--outdir", required=True, help="Output directory for plots/tables.")
    ap.add_argument("--sep", default="\t", help="Delimiter (default: tab).")
    ap.add_argument(
        "--explained",
        default=None,
        help="Optional: path to eof_explained_variance.tsv (for labeling variance explained).",
    )
    ap.add_argument(
        "--eofs",
        default=None,
        help="Comma-separated list of EOF indices to plot (e.g., '1,2,3'). If omitted, plots all.",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top (absolute loading) features to write to per-EOF TSV.",
    )
    ap.add_argument(
        "--figsize",
        default="10,6",
        help="Figure size as 'W,H' inches (default 10,6).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG dpi (default 200).",
    )
    return ap.parse_args()


def parse_figsize(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--figsize must be like '10,6'")
    return float(parts[0]), float(parts[1])


def parse_eofs_arg(eofs: Optional[str], available: List[str]) -> List[str]:
    if not eofs:
        return available
    idx = [int(x.strip()) for x in eofs.split(",") if x.strip()]
    want = [f"EOF{i}" for i in idx]
    missing = [c for c in want if c not in available]
    if missing:
        raise ValueError(f"Requested EOF columns not found: {missing}")
    return want


def load_variance_table(path: Optional[str]) -> Dict[str, float]:
    """
    Returns mapping: EOF column name (EOF1..) -> explained_variance_ratio
    Accepts the file produced by env_profile_eof.py which labels components as PC1..,
    but we map PCi -> EOFi for labeling convenience.
    """
    if not path:
        return {}
    df = pd.read_csv(path, sep="\t")
    if "component" not in df.columns or "explained_variance_ratio" not in df.columns:
        return {}

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        comp = str(r["component"])
        m = re.match(r"^PC(\d+)$", comp)
        if not m:
            continue
        i = int(m.group(1))
        out[f"EOF{i}"] = float(r["explained_variance_ratio"])
    return out


def split_feature_name(feat: str) -> Tuple[str, float]:
    m = FEATURE_RE.match(feat)
    if not m:
        raise ValueError(
            f"Feature '{feat}' does not match '<var>@<depth>' naming. "
            "This plotter expects env_profile_eof.py-style features."
        )
    return m.group("var"), float(m.group("depth"))


def build_var_depth_matrix(load_df: pd.DataFrame, eof_col: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build a 2D array A with shape [n_vars, n_depths] for a single EOF column.
    Returns: (A, var_names_in_order, depth_grid_sorted)
    """
    # Parse all feature names
    vars_: List[str] = []
    depths_: List[float] = []
    for feat in load_df.index:
        v, d = split_feature_name(str(feat))
        vars_.append(v)
        depths_.append(d)

    var_set = sorted(set(vars_), key=lambda x: str(x))
    depth_grid = np.array(sorted(set(depths_)), dtype=float)

    # Map to matrix
    var_idx = {v: i for i, v in enumerate(var_set)}
    depth_idx = {d: j for j, d in enumerate(depth_grid)}

    A = np.full((len(var_set), len(depth_grid)), np.nan, dtype=float)

    for feat, loading in zip(load_df.index, load_df[eof_col].to_numpy(dtype=float)):
        v, d = split_feature_name(str(feat))
        A[var_idx[v], depth_idx[d]] = loading

    # Sanity: if any NaN remain, feature grid is incomplete; still plot, but warn via NaNs
    return A, var_set, depth_grid


def write_top_features(load_df: pd.DataFrame, eof_col: str, out_path: str, top_n: int) -> None:
    x = load_df[eof_col].to_numpy(dtype=float)
    feats = load_df.index.astype(str).to_numpy()
    order = np.argsort(np.abs(x))[::-1]
    order = order[: min(top_n, order.size)]

    out = pd.DataFrame(
        {
            "feature": feats[order],
            "loading": x[order],
            "abs_loading": np.abs(x[order]),
        }
    )
    out.to_csv(out_path, sep="\t", index=False)


def plot_heatmap(A: np.ndarray, var_names: List[str], depth_grid: np.ndarray, title: str, out_png: str, out_pdf: str,
                 figsize: Tuple[float, float], dpi: int) -> None:
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Default colormap; symmetric range around 0 for signed loadings
    vmax = np.nanmax(np.abs(A))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(
        A,
        aspect="auto",
        interpolation="nearest",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )

    ax.set_yticks(np.arange(len(var_names)))
    ax.set_yticklabels(var_names)

    # Depth ticks: don't label too many
    ndepth = len(depth_grid)
    if ndepth <= 20:
        xt = np.arange(ndepth)
    else:
        step = max(1, ndepth // 10)
        xt = np.arange(0, ndepth, step)

    ax.set_xticks(xt)
    ax.set_xticklabels([str(depth_grid[i]) for i in xt], rotation=45, ha="right")
    ax.set_xlabel("Depth (m) on common grid")
    ax.set_ylabel("Variable")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("EOF loading (signed)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figsize = parse_figsize(args.figsize)

    load_df = pd.read_csv(args.loadings, sep=args.sep)
    if load_df.shape[1] < 2:
        raise ValueError("Loadings file looks wrong. Expected 'feature' column + EOF columns.")

    # Assume first column is feature if not already index
    if load_df.columns[0] != "feature":
        # env_profile_eof.py writes "feature" as index; read_csv will load it as first col
        # We'll treat first col as feature regardless of its name.
        pass

    feature_col = load_df.columns[0]
    load_df[feature_col] = load_df[feature_col].astype(str)
    load_df = load_df.set_index(feature_col)

    eof_cols = [c for c in load_df.columns if str(c).startswith("EOF")]
    if not eof_cols:
        # If someone saved PCA loadings as PC columns, allow that too
        pc_cols = [c for c in load_df.columns if str(c).startswith("PC")]
        if pc_cols:
            # Map PC -> EOF for plotting
            rename = {c: c.replace("PC", "EOF") for c in pc_cols}
            load_df = load_df.rename(columns=rename)
            eof_cols = [c for c in load_df.columns if str(c).startswith("EOF")]

    if not eof_cols:
        raise ValueError("No EOF columns found. Expected columns named EOF1, EOF2, ...")

    eof_cols = sorted(eof_cols, key=lambda x: int(re.sub(r"\D", "", str(x)) or "0"))
    eof_cols = parse_eofs_arg(args.eofs, eof_cols)

    var_map = load_variance_table(args.explained)

    for eof in eof_cols:
        A, var_names, depth_grid = build_var_depth_matrix(load_df, eof)

        # Label with explained variance if available
        ev = var_map.get(eof, None)
        if ev is None:
            title = f"{eof} loadings (variable × depth)"
        else:
            title = f"{eof} loadings (variable × depth) — explained variance: {100.0*ev:.1f}%"

        out_png = os.path.join(args.outdir, f"eof_mode_{eof}.png")
        out_pdf = os.path.join(args.outdir, f"eof_mode_{eof}.pdf")
        plot_heatmap(
            A=A,
            var_names=var_names,
            depth_grid=depth_grid,
            title=title,
            out_png=out_png,
            out_pdf=out_pdf,
            figsize=figsize,
            dpi=args.dpi,
        )

        out_top = os.path.join(args.outdir, f"eof_mode_{eof}_top_features.tsv")
        write_top_features(load_df, eof, out_top, top_n=args.top_n)

        print(f"Wrote: {out_png}")
        print(f"Wrote: {out_pdf}")
        print(f"Wrote: {out_top}")


if __name__ == "__main__":
    main()
