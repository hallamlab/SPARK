#!/usr/bin/env python3
# ==============================================================================
# File: scripts/env_profile_eof.py
#
# Step #2 (from our plan): EOF / SVD / PCA on cruise-level vertical profiles
# ------------------------------------------------------------------------------
# What this script does
# ---------------------
# Converts a "long" biochem table (cruise × depth × variables) into a
# "wide" cruise-level feature matrix:
#
#   rows    = cruises (one row per cruise/date)
#   columns = variable × depth_bin
#
# Then it runs PCA (SVD under the hood) to compute EOFs:
#   - EOF loadings: which (variable, depth) features define each EOF mode
#   - EOF scores: per-cruise coefficients (how strongly the cruise expresses each mode)
#
# Why this is the right representation for your use-case
# ------------------------------------------------------
# Your true unit of "state" is a CRUISE (a vertical snapshot), not each individual
# depth measurement. This avoids mixing up:
#   - within-cruise vertical structure (physics/stratification)
#   - between-cruise temporal regime shifts
#
# Inputs you must provide
# -----------------------
# --input (required): a TSV/CSV in LONG format with at least these columns:
#   - cruise_id : identifier for each cruise (date or string)
#   - depth_m   : numeric depth (meters); increasing with depth
#   - plus >=1 numeric biochem columns (e.g., o2, no3, po4, temp, salinity, ...)
#
# Example rows (LONG format):
#   cruise_id   depth_m    o2    no3   po4   temp
#   2011-03-10   5       240     1.2   0.3   9.8
#   2011-03-10  10       210     2.1   0.4   9.2
#   ...
#   2011-03-24   5       260     0.9   0.2  10.1
#
# Critical requirements for defensibility
# --------------------------------------
# 1) Each cruise has multiple depths (14–17 typical). Missing depths are OK.
# 2) You MUST define a common depth grid to compare cruises. This script supports:
#    - --depth_grid: explicit grid "0,5,10,..."
#    - OR --depth_min/--depth_max/--depth_step to generate a grid
# 3) For each cruise and each variable, values are interpolated onto the grid.
#    Interpolation method:
#      - linear (default): robust + easy to defend
#      - nearest: less smooth, but avoids creating new gradients
# 4) Missing outside-range is handled by leaving NaN and then imputing later.
#
# Output files written to --outdir
# -------------------------------
# 1) cruise_feature_matrix.tsv
#    Wide matrix used for PCA (rows=cruise_id, cols=var@depth)
# 2) eof_loadings.tsv
#    EOF loadings per component (feature weights)
# 3) eof_scores.tsv
#    EOF scores per cruise (PC1..PCk)
# 4) eof_explained_variance.tsv
#    explained variance ratio per component
#
# Dependencies
# ------------
# pip install pandas numpy scikit-learn scipy
#
# Notes
# -----
# - PCA here == SVD on mean-centered data (standard).
# - You can choose scaling:
#     --scale none   (center only; good when units are comparable or you pre-scaled)
#     --scale zscore (recommended if variables have different units/magnitudes)
# - If you have strong outliers, consider robust scaling (not implemented here).
# ==============================================================================

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build cruise-level profile feature matrix and compute EOFs via PCA/SVD."
    )
    ap.add_argument("--input", required=True, help="Input LONG table (TSV/CSV).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default="\t", help="Delimiter (default: tab). Use ',' for CSV.")
    ap.add_argument("--cruise-col", default="cruise_id", help="Column name for cruise ID.")
    ap.add_argument("--depth-col", default="depth_m", help="Column name for depth (meters).")

    ap.add_argument(
        "--vars",
        default=None,
        help=(
            "Comma-separated list of biochem variable columns to use. "
            "If omitted, uses all numeric columns except cruise/depth."
        ),
    )

    # Depth grid definition
    ap.add_argument(
        "--depth-grid",
        default=None,
        help="Explicit depth grid, e.g. '0,5,10,20,50,100,150,200'. Overrides min/max/step if provided.",
    )
    ap.add_argument("--depth-min", type=float, default=None, help="Min depth for generated grid.")
    ap.add_argument("--depth-max", type=float, default=None, help="Max depth for generated grid.")
    ap.add_argument("--depth-step", type=float, default=None, help="Step size for generated grid.")

    ap.add_argument(
        "--interp",
        choices=["linear", "nearest"],
        default="linear",
        help="Interpolation method per cruise per variable onto depth grid.",
    )

    # PCA settings
    ap.add_argument("--n-components", type=int, default=10, help="Number of EOF components to keep.")
    ap.add_argument(
        "--scale",
        choices=["none", "zscore"],
        default="zscore",
        help="Feature scaling before PCA. 'zscore' recommended across mixed-unit variables.",
    )

    # Missing-data handling
    ap.add_argument(
        "--impute",
        choices=["mean", "median"],
        default="median",
        help="Imputation strategy for remaining missing grid points.",
    )

    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def parse_depth_grid(args: argparse.Namespace, depths_all: np.ndarray) -> np.ndarray:
    if args.depth_grid:
        grid = np.array([float(x) for x in args.depth_grid.split(",")], dtype=float)
        if grid.size < 2:
            raise ValueError("--depth-grid must contain at least 2 depths.")
        return np.sort(grid)

    # Generate grid
    dmin = args.depth_min if args.depth_min is not None else float(np.nanmin(depths_all))
    dmax = args.depth_max if args.depth_max is not None else float(np.nanmax(depths_all))
    step = args.depth_step if args.depth_step is not None else 5.0

    if step <= 0:
        raise ValueError("--depth-step must be > 0.")
    if dmax <= dmin:
        raise ValueError("depth_max must be > depth_min.")

    grid = np.arange(dmin, dmax + 1e-9, step, dtype=float)
    if grid.size < 2:
        raise ValueError("Generated depth grid has <2 points; adjust min/max/step.")
    return grid


def choose_vars(df: pd.DataFrame, cruise_col: str, depth_col: str, vars_arg: Optional[str]) -> List[str]:
    if vars_arg:
        cols = [c.strip() for c in vars_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"--vars contains missing columns: {missing}")
        return cols

    # Default: all numeric columns except cruise/depth
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in {cruise_col, depth_col}]
    if not cols:
        raise ValueError("No numeric biochem columns found. Provide --vars explicitly.")
    return cols


def interpolate_profile(
    depths: np.ndarray, values: np.ndarray, grid: np.ndarray, method: str
) -> np.ndarray:
    """
    Interpolate one (depth -> value) series onto the common grid.
    Leaves NaN if there are too few points or outside the observed range.
    """
    # Drop NaNs
    m = np.isfinite(depths) & np.isfinite(values)
    d = depths[m]
    v = values[m]

    if d.size < 2:
        return np.full_like(grid, np.nan, dtype=float)

    # Ensure strictly increasing depths for interp1d
    o = np.argsort(d)
    d = d[o]
    v = v[o]

    # Handle duplicate depths by averaging
    if np.unique(d).size != d.size:
        tmp = pd.DataFrame({"d": d, "v": v}).groupby("d", as_index=False).mean()
        d = tmp["d"].to_numpy()
        v = tmp["v"].to_numpy()

    f = interp1d(
        d,
        v,
        kind="linear" if method == "linear" else "nearest",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return f(grid).astype(float)


def build_cruise_feature_matrix(
    df: pd.DataFrame, cruise_col: str, depth_col: str, vars_cols: List[str], grid: np.ndarray, interp: str
) -> pd.DataFrame:
    """
    Output wide matrix:
      index = cruise_id
      columns = "{var}@{depth}"
      values = interpolated var value at that grid depth for that cruise
    """
    cruises = df[cruise_col].astype(str).unique().tolist()
    cruises.sort()

    feat_cols = [f"{v}@{int(g) if float(g).is_integer() else g}" for v in vars_cols for g in grid]
    X = np.full((len(cruises), len(feat_cols)), np.nan, dtype=float)

    col_idx = {name: j for j, name in enumerate(feat_cols)}

    for i, cid in enumerate(cruises):
        sub = df[df[cruise_col].astype(str) == cid]
        depths = sub[depth_col].to_numpy(dtype=float)

        for v in vars_cols:
            vals = sub[v].to_numpy(dtype=float)
            interp_vals = interpolate_profile(depths, vals, grid, method=interp)
            for k, g in enumerate(grid):
                cname = f"{v}@{int(g) if float(g).is_integer() else g}"
                X[i, col_idx[cname]] = interp_vals[k]

    out = pd.DataFrame(X, index=cruises, columns=feat_cols)
    out.index.name = cruise_col
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep=args.sep)
    if args.cruise_col not in df.columns:
        raise ValueError(f"Missing cruise column: {args.cruise_col}")
    if args.depth_col not in df.columns:
        raise ValueError(f"Missing depth column: {args.depth_col}")

    # Ensure depth numeric
    df[args.depth_col] = pd.to_numeric(df[args.depth_col], errors="coerce")

    vars_cols = choose_vars(df, args.cruise_col, args.depth_col, args.vars)
    grid = parse_depth_grid(args, df[args.depth_col].to_numpy(dtype=float))

    X = build_cruise_feature_matrix(
        df=df,
        cruise_col=args.cruise_col,
        depth_col=args.depth_col,
        vars_cols=vars_cols,
        grid=grid,
        interp=args.interp,
    )

    # Save the raw wide matrix (before imputation/scaling) for transparency
    X_path = os.path.join(args.outdir, "cruise_feature_matrix.tsv")
    X.to_csv(X_path, sep="\t", index=True)

    # Impute remaining missing values (common if cruises have shallower max depth, etc.)
    imputer = SimpleImputer(strategy=args.impute)
    X_imp = imputer.fit_transform(X.values)

    # Scale if requested
    if args.scale == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_proc = scaler.fit_transform(X_imp)
    else:
        # Center only (PCA will center; but we keep it explicit)
        X_proc = X_imp - X_imp.mean(axis=0, keepdims=True)

    # PCA (EOF)
    ncomp = min(args.n_components, X_proc.shape[0], X_proc.shape[1])
    pca = PCA(n_components=ncomp, random_state=args.seed)
    scores = pca.fit_transform(X_proc)  # cruise × components
    loadings = pca.components_.T        # feature × components

    # Write EOF loadings
    load_df = pd.DataFrame(
        loadings,
        index=X.columns,
        columns=[f"EOF{i+1}" for i in range(loadings.shape[1])],
    )
    load_df.index.name = "feature"
    load_df.to_csv(os.path.join(args.outdir, "eof_loadings.tsv"), sep="\t", index=True)

    # Write EOF scores (per cruise)
    score_df = pd.DataFrame(
        scores,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])],
    )
    score_df.index.name = args.cruise_col
    score_df.to_csv(os.path.join(args.outdir, "eof_scores.tsv"), sep="\t", index=True)

    # Explained variance
    ev = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance": pca.explained_variance_,
        }
    )
    ev.to_csv(os.path.join(args.outdir, "eof_explained_variance.tsv"), sep="\t", index=False)

    print(f"Wrote: {X_path}")
    print(f"Wrote: {os.path.join(args.outdir, 'eof_loadings.tsv')}")
    print(f"Wrote: {os.path.join(args.outdir, 'eof_scores.tsv')}")
    print(f"Wrote: {os.path.join(args.outdir, 'eof_explained_variance.tsv')}")


if __name__ == "__main__":
    main()
