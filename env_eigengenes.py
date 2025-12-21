#!/usr/bin/env python3
"""
env_eigengenes.py

Baseline:
- Prepare env/biochem matrix
- Handle missingness (drop sparse cols/rows + impute)
- Clamp negatives to 0 (always; counted)
- Optional log1p (after clamp)
- Standardize
- PCA
- Save key tables/plots

NEW (optional):
A) Depth anchoring (pre-flight) --anchor-depths
   DATA-DRIVEN ONLY (no “nearest” mode):
   - Choose global depth anchors from most common rounded depths
   - For each cruise (or --anchor-by-col block), compute anchor prototypes
     using ONLY observed (non-imputed) feature values
   - For each off-depth sample, compare similarity to bracketing anchors (below/above)
     within the same cruise; snap if:
       (i) enough observed overlap (>= --anchor-min-features),
       (ii) one anchor is clearly closer (<= --anchor-margin ratio),
       (iii) chosen anchor is within --anchor-tol-m meters
   - Keeps samples; avoids dropping “off-depth” samples
   Outputs:
     tables/depth_anchors.csv
     tables/depth_anchor_mapping_summary.csv
     tables/depth_anchor_decisions.csv
     tables/depth_anchor_decision_summary_by_block.csv

B) Improved imputation:
   - median/mean (baseline)
   - knn (KNNImputer)
   - iterative (IterativeImputer with BayesianRidge)
   - Optionally depth-aware: --impute-scope by_depth
     (learn imputation within each anchored depth group; fallback to global model for small groups)

C) Data-driven PC selection + feature clustering by loading similarity
   Enable with: --pc-selection

PC selection rules implemented:
1) Parallel analysis (permute columns) keep if lambda_obs > null_q
2) Coverage support on top-loading features (median_cov >= 0.70 and n_cov>=6)
3) Feature clusters (Agglomerative on cosine distance of centered loading vectors)
   PC coherence keep if >= 60% of top-loading features in one cluster
4) Block-bootstrap stability (by Cruise/Month/YearMonth) keep if:
   median abs corr(loadings) >= 0.85 AND median abs corr(scores) >= 0.80

Outputs:
- run_config.json
- qc_summary.json
- missing_expected_columns.json

Tables:
- missingness_pre_drop.csv, missingness_post_drop.csv
- dropped_rows.csv
- matrix_cleaned.csv, matrix_scaled.csv
- impute_values.csv
- (if depth anchoring) depth_anchors.csv, depth_anchor_mapping_summary.csv, depth_anchor_decisions.csv,
  depth_anchor_decision_summary_by_block.csv
- (if depth-aware impute) imputation_depth_group_audit.csv
- pca_explained_variance.csv, pca_loadings.csv, eigengenes_scores.csv

Plots:
- missingness_pre_drop.png, missingness_post_drop.png
- scree.png, cumulative_variance.png, pc1_vs_pc2.png
- top_loadings_PC{k}.png

PC selection (if enabled):
- parallel_analysis.csv + parallel_analysis.png
- feature_coverage.csv
- feature_clusters.csv
- pc_coverage_support.csv
- pc_cluster_enrichment.csv
- pc_stability.csv
- pc_keep_decision.csv
- loadings_heatmap_by_feature_cluster.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Improved imputers
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# -----------------------------
# Column configuration (YOUR DATA)
# -----------------------------

DEFAULT_ID_COL = "cruise_year_month_depth"
DERIVED_TIME_COL = "date"

YEAR_COL = "Year"
MONTH_COL = "Month"
DAY_COL = "Day"

DEFAULT_META_COLS = [
    "cruise_year_month_depth",
    "Cruise",
    "Year",
    "Month",
    "Day",
    "Depth",
]

DEFAULT_FEATURE_COLS = [
    "Pressure",
    "Temperature",
    "Conductivity",
    "Salinity",
    "Density",
    "Oxygen",
    "Phosphate",
    "Silicate",
    "Nitrogen Oxides",
    "Nitrate",
    "Nitrite",
    "Ammonium",
    "Hydrogen Sulfide",
    "Methane",
    "Nitrous Oxide",
    "Dimethyl Sulfide",
]


# -----------------------------
# CLI / config
# -----------------------------

@dataclass
class RunConfig:
    input_path: str
    outdir: str
    sep: str

    id_col: str
    time_col: str

    feature_cols: List[str]          # final features after any subsetting
    meta_cols: List[str]

    include_depth_as_feature: bool
    n_components: int
    log1p: bool

    # Imputation
    impute: str
    impute_scope: str
    impute_min_group_size: int
    knn_k: int
    iterative_max_iter: int

    dropna_row_thresh: float
    dropna_col_thresh: float
    random_state: int

    # Depth anchoring (pre-flight)
    depth_col: str
    anchor_depths: bool
    anchor_round_m: float
    anchor_tol_m: float
    anchor_top_k: int
    anchor_min_count: int
    anchored_depth_col: str

    # Data-driven anchoring params
    anchor_by_col: str
    anchor_min_features: int
    anchor_margin: float
    anchor_proto_min_n: int

    # PC selection
    pc_selection: bool
    pcsel_parallel_B: int
    pcsel_parallel_quantile: float
    pcsel_support_min_cov: float
    pcsel_support_median_cov: float
    pcsel_support_min_n: int
    pcsel_top_frac: float
    pcsel_top_min: int
    pcsel_feature_clusters: int
    pcsel_coherence_min_frac: float
    pcsel_block_col: str
    pcsel_stability_R: int
    pcsel_stability_min_load_corr: float
    pcsel_stability_min_score_corr: float
    pcsel_entropy_max: float
    pcsel_pr_max: float


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="Prep env params for PCA and extract eigengenes (+ optional PC selection).")
    ap.add_argument("--input", required=True, help="Path to input table (CSV/TSV).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default="\t", help="Delimiter (default: tab). Use ',' for CSV.")

    ap.add_argument("--n-components", type=int, default=5, help="Number of PCA components.")
    ap.add_argument("--log1p", action="store_true", help="Apply log1p to features after cleaning (negatives clamped to 0).")

    ap.add_argument(
        "--impute",
        choices=["median", "mean", "knn", "iterative"],
        default="iterative",
        help="Imputation strategy.",
    )
    ap.add_argument(
        "--impute-scope",
        choices=["global", "by_depth"],
        default="global",
        help="Imputation scope. 'by_depth' learns imputation within anchored depth groups (recommended for profiles).",
    )
    ap.add_argument(
        "--impute-min-group-size",
        type=int,
        default=30,
        help="Min rows for within-depth imputation; else fallback to global model (default 30).",
    )
    ap.add_argument("--knn-k", type=int, default=10, help="K for KNN imputation (default 10).")
    ap.add_argument("--iterative-max-iter", type=int, default=20, help="Max iterations for IterativeImputer (default 20).")

    ap.add_argument(
        "--dropna-row-thresh",
        type=float,
        default=0.4,
        help="Drop rows if missing fraction among features is > this value (default 0.4).",
    )
    ap.add_argument(
        "--dropna-col-thresh",
        type=float,
        default=0.4,
        help="Drop feature columns if missing fraction is > this value (default 0.4).",
    )
    ap.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")

    ap.add_argument(
        "--include-depth-as-feature",
        action="store_true",
        help="Include Depth as a PCA feature (default: Depth is metadata only).",
    )

    ap.add_argument(
        "--feature-cols",
        default=None,
        help=(
            "Optional comma-separated feature column override (replaces defaults). "
            "If omitted, uses DEFAULT_FEATURE_COLS."
        ),
    )

    ap.add_argument("--id-col", default=DEFAULT_ID_COL, help=f"ID column (default: {DEFAULT_ID_COL})")
    ap.add_argument("--time-col", default=DERIVED_TIME_COL, help=f"Time column (default: derived '{DERIVED_TIME_COL}')")

    # ---- Depth anchoring ----
    ap.add_argument("--depth-col", default="Depth", help="Depth column name (default Depth).")
    ap.add_argument(
        "--anchor-depths",
        action="store_true",
        help="Enable DATA-DRIVEN depth anchoring before filtering/imputation.",
    )
    ap.add_argument("--anchor-round-m", type=float, default=1.0, help="Round depths to this resolution for anchors (m; default 1).")
    ap.add_argument("--anchor-tol-m", type=float, default=15, help="Hard snap tolerance (m; default 15).")
    ap.add_argument("--anchor-top-k", type=int, default=50, help="Use at most top-K most frequent depths as anchors (default 50).")
    ap.add_argument("--anchor-min-count", type=int, default=10, help="Only depths with >= this count can be anchors (default 10).")
    ap.add_argument("--anchored-depth-col", default="Depth_anchored", help="Name of anchored depth column (default Depth_anchored).")

    # Data-driven anchoring params (no mode switch)
    ap.add_argument("--anchor-by-col", default="ALL", help="Block column for within-block prototypes (default ALL).")
    ap.add_argument("--anchor-min-features", type=int, default=5, help="Min shared observed features to decide a snap (default 8).")
    ap.add_argument("--anchor-margin", type=float, default=0.99, help="Snap only if winner distance ratio <= margin (default 0.99).")
    ap.add_argument("--anchor-proto-min-n", type=int, default=3, help="Min samples at (block, anchor) to build a prototype (default 3).")

    # ---- PC selection flags ----
    ap.add_argument("--pc-selection", action="store_true", help="Run keep-able PC selection + feature clustering.")
    ap.add_argument("--pcsel-parallel-B", type=int, default=500, help="Parallel analysis replicates (default 500).")
    ap.add_argument("--pcsel-parallel-quantile", type=float, default=0.95, help="Null quantile (default 0.95).")

    ap.add_argument("--pcsel-support-min-cov", type=float, default=0.70, help="Min per-feature coverage threshold (default 0.70).")
    ap.add_argument("--pcsel-support-median-cov", type=float, default=0.70, help="Median coverage threshold on top features (default 0.70).")
    ap.add_argument("--pcsel-support-min-n", type=int, default=5, help="Min number of well-covered top features (default 6).")

    ap.add_argument("--pcsel-top-frac", type=float, default=0.15, help="Fraction of features for top-loading set Tk (default 0.15).")
    ap.add_argument("--pcsel-top-min", type=int, default=8, help="Minimum size of top-loading set Tk (default 8).")

    ap.add_argument("--pcsel-feature-clusters", type=int, default=8, help="Number of feature clusters (default 8).")
    ap.add_argument("--pcsel-coherence-min-frac", type=float, default=0.30, help="Min dominant cluster fraction in Tk (default 0.60).")

    ap.add_argument(
        "--pcsel-block-col",
        type=str,
        default="Cruise",
        help="Block column for stability bootstrap: Cruise|Month|YearMonth (default Cruise).",
    )
    ap.add_argument(
        "--pcsel-entropy-max",
        type=float,
        default=0.85,
        help=(
            "Max normalized loading-entropy to treat a PC as 'concentrated' (0=one feature, 1=uniform). "
            "Default 0.85."
        ),
    )
    ap.add_argument(
        "--pcsel-pr-max",
        type=float,
        default=8.0,
        help=(
            "Max participation ratio to treat a PC as 'concentrated' (roughly 'effective #features'). "
            "Default 8."
        ),
    )

    ap.add_argument("--pcsel-stability-R", type=int, default=200, help="Stability bootstrap replicates (default 200).")
    ap.add_argument("--pcsel-stability-min-load-corr", type=float, default=0.85, help="Min median abs corr for loadings (default 0.85).")
    ap.add_argument("--pcsel-stability-min-score-corr", type=float, default=0.80, help="Min median abs corr for scores (default 0.80).")

    ns = ap.parse_args()

    # Start with defaults, or explicit override list
    feature_cols = DEFAULT_FEATURE_COLS.copy()
    if ns.feature_cols:
        feature_cols = [c.strip() for c in ns.feature_cols.split(",") if c.strip()]

    meta_cols = DEFAULT_META_COLS.copy()

    # Optionally treat Depth as a feature
    if ns.include_depth_as_feature:
        if "Depth" in meta_cols:
            meta_cols.remove("Depth")
        if "Depth" not in feature_cols:
            feature_cols = ["Depth"] + feature_cols

    return RunConfig(
        input_path=ns.input,
        outdir=ns.outdir,
        sep=ns.sep,
        id_col=ns.id_col,
        time_col=ns.time_col,
        feature_cols=feature_cols,
        meta_cols=meta_cols,
        include_depth_as_feature=ns.include_depth_as_feature,
        n_components=ns.n_components,
        log1p=ns.log1p,

        impute=ns.impute,
        impute_scope=ns.impute_scope,
        impute_min_group_size=ns.impute_min_group_size,
        knn_k=ns.knn_k,
        iterative_max_iter=ns.iterative_max_iter,

        dropna_row_thresh=ns.dropna_row_thresh,
        dropna_col_thresh=ns.dropna_col_thresh,
        random_state=ns.random_state,

        depth_col=ns.depth_col,
        anchor_depths=ns.anchor_depths,
        anchor_round_m=ns.anchor_round_m,
        anchor_tol_m=ns.anchor_tol_m,
        anchor_top_k=ns.anchor_top_k,
        anchor_min_count=ns.anchor_min_count,
        anchored_depth_col=ns.anchored_depth_col,

        anchor_by_col=ns.anchor_by_col,
        anchor_min_features=ns.anchor_min_features,
        anchor_margin=ns.anchor_margin,
        anchor_proto_min_n=ns.anchor_proto_min_n,

        pc_selection=ns.pc_selection,
        pcsel_parallel_B=ns.pcsel_parallel_B,
        pcsel_parallel_quantile=ns.pcsel_parallel_quantile,
        pcsel_support_min_cov=ns.pcsel_support_min_cov,
        pcsel_support_median_cov=ns.pcsel_support_median_cov,
        pcsel_support_min_n=ns.pcsel_support_min_n,
        pcsel_top_frac=ns.pcsel_top_frac,
        pcsel_top_min=ns.pcsel_top_min,
        pcsel_feature_clusters=ns.pcsel_feature_clusters,
        pcsel_coherence_min_frac=ns.pcsel_coherence_min_frac,
        pcsel_block_col=ns.pcsel_block_col,
        pcsel_stability_R=ns.pcsel_stability_R,
        pcsel_stability_min_load_corr=ns.pcsel_stability_min_load_corr,
        pcsel_stability_min_score_corr=ns.pcsel_stability_min_score_corr,
        pcsel_entropy_max=ns.pcsel_entropy_max,
        pcsel_pr_max=ns.pcsel_pr_max,
    )


def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def load_table(path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, dtype=str)


def make_date_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()

    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        return out

    missing = [c for c in [YEAR_COL, MONTH_COL, DAY_COL] if c not in out.columns]
    if missing:
        return out

    y = pd.to_numeric(out[YEAR_COL], errors="coerce")
    m = pd.to_numeric(out[MONTH_COL], errors="coerce")
    d = pd.to_numeric(out[DAY_COL], errors="coerce")
    out[time_col] = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
    return out


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# -----------------------------
# Cleaning transforms
# -----------------------------

def clamp_negatives_to_zero(X: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    X2 = X.copy()
    neg_mask = X2 < 0
    n_clamped = int(neg_mask.sum().sum())
    if n_clamped > 0:
        X2[neg_mask] = 0.0
    return X2, n_clamped


def maybe_log1p(X: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(X)


# -----------------------------
# Depth anchoring (DATA-DRIVEN)
# -----------------------------

def _choose_global_anchors(
    d: pd.Series,
    round_m: float,
    top_k: int,
    min_count: int
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    if round_m <= 0:
        round_m = 1.0
    d_round = (np.round(d / round_m) * round_m).astype(float)

    counts = d_round.value_counts(dropna=True).sort_values(ascending=False)
    counts = counts[counts >= min_count]
    if counts.empty:
        return np.array([], dtype=float), pd.DataFrame(), d_round

    anchors = counts.head(top_k)
    anchor_vals = np.sort(anchors.index.to_numpy(dtype=float))

    anchors_df = pd.DataFrame({
        "anchor_depth_m": anchors.index.astype(float).values,
        "count": anchors.values.astype(int),
    }).sort_values(["count", "anchor_depth_m"], ascending=[False, True])

    return anchor_vals, anchors_df, d_round


def _zscore_params_from_observed(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mu = X.mean(axis=0, skipna=True)
    sd = X.std(axis=0, skipna=True, ddof=0)
    sd = sd.replace(0, 1.0).fillna(1.0)
    mu = mu.fillna(0.0)
    return mu, sd


def _euclid_zdist(x: pd.Series, proto: pd.Series, mu: pd.Series, sd: pd.Series, min_features: int) -> Tuple[float, int]:
    # Use only features observed in BOTH
    m = (~x.isna()) & (~proto.isna())
    n = int(m.sum())
    if n < min_features:
        return np.inf, n

    z = (x[m] - mu[m]) / sd[m]
    zp = (proto[m] - mu[m]) / sd[m]
    diff = (z - zp).to_numpy(dtype=float)
    return float(np.sqrt(np.sum(diff * diff))), n


def anchor_depth_column_data_driven(
    df: pd.DataFrame,
    depth_col: str,
    anchored_col: str,
    by_col: str,
    feature_cols: List[str],
    round_m: float,
    tol_m: float,
    top_k: int,
    min_count: int,
    min_features: int,
    margin: float,
    proto_min_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    DATA-DRIVEN depth anchoring:
    - Global anchors chosen from most common rounded depths (whole dataset)
    - Within each block (by_col), build prototype vectors for each anchor depth using observed values only (median)
      using samples whose rounded depth == anchor depth; require proto_min_n samples
    - For each sample, find bracketing anchors (below/above). Compare similarity to those prototypes.
      Snap if:
        * enough shared observed features (>= min_features)
        * clear winner: min(dist)/max(dist) <= margin
        * chosen anchor is within tol_m meters of the sample's actual depth
    """
    out = df.copy()

    if depth_col not in out.columns:
        out[anchored_col] = np.nan
        return out, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    if by_col not in out.columns:
        # If missing grouping column, treat as one block.
        out[by_col] = "ALL"

    # Depth numeric
    d = pd.to_numeric(out[depth_col], errors="coerce")

    # Features numeric matrix for similarity (pre-impute). Clamp negatives here to avoid artifacts.
    feats_present = [c for c in feature_cols if c in out.columns]
    X = out[feats_present].apply(pd.to_numeric, errors="coerce")

    X, n_clamped = clamp_negatives_to_zero(X)

    # Z-score params from observed values (no imputation)
    mu, sd = _zscore_params_from_observed(X)

    # Global anchors
    anchor_vals, anchors_df, d_round = _choose_global_anchors(d, round_m=round_m, top_k=top_k, min_count=min_count)
    if anchor_vals.size == 0:
        out[anchored_col] = d
        return out, anchors_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), n_clamped

    # Mapping summary (rounded->anchor decision frequency later)
    mapping_df = pd.DataFrame({"depth_rounded": d_round.values})

    # Build prototypes per (block, anchor)
    blocks = out[by_col].astype(str).fillna("NA")
    proto_rows = []

    # Pre-index rows by block for speed
    block_to_idx: Dict[str, np.ndarray] = {}
    for b in blocks.unique().tolist():
        block_to_idx[b] = np.where(blocks.values == b)[0]

    # For each block + anchor, prototype from rows where d_round == anchor
    prototypes: Dict[Tuple[str, float], pd.Series] = {}
    proto_counts: Dict[Tuple[str, float], int] = {}

    d_round_arr = d_round.to_numpy(dtype=float)
    X_arr = X  # DataFrame

    for b, idxs in block_to_idx.items():
        idxs = idxs.astype(int)
        dbr = d_round_arr[idxs]
        for a in anchor_vals:
            m = np.isfinite(dbr) & (np.abs(dbr - a) < 1e-9)
            rows = idxs[m]
            n = int(len(rows))
            key = (b, float(a))
            if n >= proto_min_n:
                proto = X_arr.iloc[rows].median(axis=0, skipna=True)
                # If a feature has all-NA at this anchor, proto will be NA; that’s fine (it will be excluded in overlap).
                prototypes[key] = proto
                proto_counts[key] = n
                proto_rows.append({"block": b, "anchor_depth_m": float(a), "n_rows": n})
            else:
                proto_counts[key] = n

    # Decide anchoring per row
    anchored_vals = np.full(shape=out.shape[0], fill_value=np.nan, dtype=float)

    decision_rows = []
    for i in range(out.shape[0]):
        bi = str(blocks.iloc[i])
        di = d.iloc[i]
        if pd.isna(di):
            anchored_vals[i] = np.nan
            decision_rows.append({
                "row_index": int(i),
                "block": bi,
                "depth": np.nan,
                "anchor_low": np.nan,
                "anchor_high": np.nan,
                "chosen_anchor": np.nan,
                "snapped": False,
                "reason": "missing_depth",
                "dist_low": np.nan,
                "dist_high": np.nan,
                "ratio": np.nan,
                "n_features_used_low": 0,
                "n_features_used_high": 0,
            })
            continue

        # =========================
        # ANCHOR GUARD (NEW)
        # If this row is already at an anchor depth (by rounded depth),
        # do NOT evaluate snapping. Keep as-is.
        # =========================
        di_round = d_round_arr[i]
        if np.isfinite(di_round) and np.any(np.abs(anchor_vals - di_round) < 1e-9):
            anchored_vals[i] = float(di_round)
            decision_rows.append({
                "row_index": int(i),
                "block": bi,
                "depth": float(di),
                "anchor_low": np.nan,
                "anchor_high": np.nan,
                "chosen_anchor": float(di_round),
                "snapped": False,
                "reason": "is_anchor",
                "dist_low": np.nan,
                "dist_high": np.nan,
                "ratio": np.nan,
                "n_features_used_low": 0,
                "n_features_used_high": 0,
            })
            continue

        # find bracketing anchors
        below = anchor_vals[anchor_vals < di]
        above = anchor_vals[anchor_vals > di]
        a_low = float(below.max()) if below.size else np.nan
        a_high = float(above.min()) if above.size else np.nan

        xi = X.iloc[i]

        dist_low = np.inf
        dist_high = np.inf
        n_low = 0
        n_high = 0

        if np.isfinite(a_low):
            key = (bi, float(a_low))
            if key in prototypes:
                dist_low, n_low = _euclid_zdist(xi, prototypes[key], mu, sd, min_features=min_features)

        if np.isfinite(a_high):
            key = (bi, float(a_high))
            if key in prototypes:
                dist_high, n_high = _euclid_zdist(xi, prototypes[key], mu, sd, min_features=min_features)

        # If neither available, don't snap
        if not np.isfinite(dist_low) and not np.isfinite(dist_high):
            anchored_vals[i] = float(di)
            decision_rows.append({
                "row_index": int(i),
                "block": bi,
                "depth": float(di),
                "anchor_low": a_low,
                "anchor_high": a_high,
                "chosen_anchor": np.nan,
                "snapped": False,
                "reason": "no_prototypes_or_insufficient_overlap",
                "dist_low": dist_low if np.isfinite(dist_low) else np.nan,
                "dist_high": dist_high if np.isfinite(dist_high) else np.nan,
                "ratio": np.nan,
                "n_features_used_low": int(n_low),
                "n_features_used_high": int(n_high),
            })
            continue

        # Choose winner among finite distances
        chosen = None
        reason = None

        if np.isfinite(dist_low) and (not np.isfinite(dist_high) or dist_low <= dist_high):
            chosen = a_low
            loser = dist_high
            winner = dist_low
            n_win = n_low
            n_los = n_high
        else:
            chosen = a_high
            loser = dist_low
            winner = dist_high
            n_win = n_high
            n_los = n_low

        # If only one side finite, ratio is 0 and we still apply tol_m
        if np.isfinite(loser) and loser > 0:
            ratio = float(winner / loser)
        else:
            ratio = 0.0

        # Confidence gate
        if np.isfinite(loser) and ratio > margin:
            anchored_vals[i] = float(di)
            reason = "low_confidence_ratio"
            snapped = False
            chosen_out = np.nan
        else:
            # Depth-distance gate
            if (chosen is None) or (not np.isfinite(chosen)) or (abs(float(chosen) - float(di)) > tol_m):
                anchored_vals[i] = float(di)
                reason = "fails_depth_tolerance"
                snapped = False
                chosen_out = np.nan
            else:
                anchored_vals[i] = float(chosen)
                reason = "snapped"
                snapped = True
                chosen_out = float(chosen)

        decision_rows.append({
            "row_index": int(i),
            "block": bi,
            "depth": float(di),
            "anchor_low": a_low,
            "anchor_high": a_high,
            "chosen_anchor": chosen_out,
            "snapped": bool(snapped),
            "reason": reason,
            "dist_low": dist_low if np.isfinite(dist_low) else np.nan,
            "dist_high": dist_high if np.isfinite(dist_high) else np.nan,
            "ratio": float(ratio) if ratio is not None else np.nan,
            "n_features_used_low": int(n_low),
            "n_features_used_high": int(n_high),
        })

    out[anchored_col] = anchored_vals

    # Rounded->anchored mapping summary
    mapping_df["depth_anchored"] = out[anchored_col].values
    mapping_summary_df = (
        mapping_df.dropna(subset=["depth_rounded", "depth_anchored"])
        .groupby(["depth_rounded", "depth_anchored"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=False)
    )

    decisions_df = pd.DataFrame(decision_rows)

    # Summary by block
    summary_by_block = (
        decisions_df.groupby(["block", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["block", "n"], ascending=[True, False])
    )
    # Add snap rate per block
    snap_rate = decisions_df.groupby("block", as_index=False)["snapped"].mean().rename(columns={"snapped": "snap_rate"})
    summary_by_block = summary_by_block.merge(snap_rate, on="block", how="left")

    # Prototype counts table (audit)
    proto_counts_rows = []
    for (b, a), n in proto_counts.items():
        proto_counts_rows.append({"block": b, "anchor_depth_m": float(a), "n_rows_at_anchor": int(n), "prototype_built": (n >= proto_min_n)})
    proto_counts_df = pd.DataFrame(proto_counts_rows).sort_values(["block", "anchor_depth_m"])

    return out, anchors_df, mapping_summary_df, decisions_df, proto_counts_df, n_clamped


# -----------------------------
# Missingness + filtering
# -----------------------------

def basic_missingness_stats(df_num: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    miss = pd.DataFrame({
        "feature": feats,
        "n_missing": [int(df_num[c].isna().sum()) for c in feats],
        "frac_missing": [float(df_num[c].isna().mean()) for c in feats],
    }).sort_values(["frac_missing", "n_missing"], ascending=False)
    return miss


def drop_sparse(
    df_num: pd.DataFrame,
    feats: List[str],
    row_thresh: float,
    col_thresh: float
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    col_missing = df_num[feats].isna().mean()
    kept_feats = [c for c in feats if col_missing[c] <= col_thresh]

    row_missing = df_num[kept_feats].isna().mean(axis=1)
    keep_rows = row_missing <= row_thresh

    dropped = pd.DataFrame({
        "dropped_row_index": df_num.index[~keep_rows].astype(str),
        "row_missing_fraction": row_missing[~keep_rows].values,
    })

    return df_num.loc[keep_rows].copy(), kept_feats, dropped


# -----------------------------
# Imputation
# -----------------------------

def impute_matrix_global(
    X: pd.DataFrame,
    strategy: str,
    knn_k: int,
    iterative_max_iter: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Global imputation across all rows.
    Returns (X_imputed, summary_series).
    """
    if strategy in ("median", "mean"):
        fill = X.median(axis=0, skipna=True) if strategy == "median" else X.mean(axis=0, skipna=True)
        return X.fillna(fill), fill

    if strategy == "knn":
        imp = KNNImputer(n_neighbors=knn_k, weights="distance")
        Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
        summary = Xi.mean(axis=0)
        summary.name = "imputed_feature_mean"
        return Xi, summary

    if strategy == "iterative":
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=iterative_max_iter,
            random_state=random_state,
            sample_posterior=False,
            initial_strategy="median",
        )
        Xi = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
        summary = Xi.mean(axis=0)
        summary.name = "imputed_feature_mean"
        return Xi, summary

    raise ValueError(f"Unknown imputation strategy: {strategy}")


def impute_matrix_by_depth(
    X: pd.DataFrame,
    depth_groups: pd.Series,
    strategy: str,
    knn_k: int,
    iterative_max_iter: int,
    min_group_size: int,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Depth-aware imputation:
    - Impute within each anchored depth group.
    - If group is too small, fallback to a global fitted model (for knn/iterative),
      or fallback to global median/mean for median/mean.
    """
    X_out = X.copy()

    global_model = None
    if strategy == "knn":
        global_model = KNNImputer(n_neighbors=knn_k, weights="distance")
        global_model.fit(X)
    elif strategy == "iterative":
        global_model = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=iterative_max_iter,
            random_state=random_state,
            sample_posterior=False,
            initial_strategy="median",
        )
        global_model.fit(X)

    global_fill = X.median(axis=0, skipna=True) if strategy == "median" else X.mean(axis=0, skipna=True)

    audit_rows = []
    for g, idx in depth_groups.groupby(depth_groups).groups.items():
        rows = list(idx)
        Xg = X.loc[rows]
        n = len(rows)

        if strategy in ("median", "mean"):
            fill = Xg.median(axis=0, skipna=True) if strategy == "median" else Xg.mean(axis=0, skipna=True)
            X_out.loc[rows] = Xg.fillna(fill)
            mode = "groupwise"
        else:
            if n >= min_group_size:
                if strategy == "knn":
                    kk = min(knn_k, max(2, n - 1))
                    imp = KNNImputer(n_neighbors=kk, weights="distance")
                else:
                    imp = IterativeImputer(
                        estimator=BayesianRidge(),
                        max_iter=iterative_max_iter,
                        random_state=random_state,
                        sample_posterior=False,
                        initial_strategy="median",
                    )
                X_out.loc[rows] = imp.fit_transform(Xg)
                mode = "groupwise"
            else:
                if global_model is not None:
                    X_out.loc[rows] = global_model.transform(Xg)
                    mode = "fallback_global_model"
                else:
                    X_out.loc[rows] = Xg.fillna(global_fill)
                    mode = "fallback_global_fill"

        audit_rows.append({
            "depth_group": str(g),
            "n_rows": int(n),
            "mode": mode,
        })

    X_out = pd.DataFrame(X_out, columns=X.columns, index=X.index)
    audit_df = pd.DataFrame(audit_rows).sort_values("n_rows", ascending=False)

    summary = X_out.mean(axis=0)
    summary.name = "imputed_feature_mean"
    return X_out, audit_df, summary


# -----------------------------
# Plotting (matplotlib only)
# -----------------------------

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_missingness(miss_df: pd.DataFrame, outpath: str, top_n: int = 40) -> None:
    d = miss_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(4, 0.25 * len(d))))
    plt.barh(d["feature"], d["frac_missing"])
    plt.xlabel("Fraction missing")
    plt.ylabel("Feature")
    plt.title(f"Top {len(d)} features by missingness")
    save_fig(outpath)


def plot_scree(pca: PCA, outpath: str) -> None:
    evr = pca.explained_variance_ratio_
    x = np.arange(1, len(evr) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, evr, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("Scree plot")
    save_fig(outpath)


def plot_cumvar(pca: PCA, outpath: str) -> None:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, cum, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Principal component")
    plt.ylabel("Cumulative variance explained")
    plt.title("Cumulative variance explained")
    save_fig(outpath)


def plot_pc_scatter(scores_df: pd.DataFrame, outpath: str, time_col: str) -> None:
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return

    plt.figure(figsize=(7, 6))
    if time_col in scores_df.columns:
        t = pd.to_datetime(scores_df[time_col], errors="coerce")
        rank = t.rank(method="first").fillna(0).to_numpy()
        plt.scatter(scores_df["PC1"], scores_df["PC2"], c=rank)
        plt.colorbar(label="Time rank")
    else:
        plt.scatter(scores_df["PC1"], scores_df["PC2"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2 (eigengene space)")
    save_fig(outpath)


def plot_top_loadings(loadings: pd.DataFrame, outpath: str, pc: str, top_n: int = 25) -> None:
    if pc not in loadings.columns:
        return
    s = loadings[pc].abs().sort_values(ascending=False).head(top_n)
    d = loadings.loc[s.index, pc].sort_values()
    plt.figure(figsize=(10, max(4, 0.25 * len(d))))
    plt.barh(d.index.astype(str), d.values)
    plt.xlabel("Loading")
    plt.ylabel("Feature")
    plt.title(f"Top {len(d)} loadings for {pc}")
    save_fig(outpath)


# -----------------------------
# PC Selection + Feature clustering
# -----------------------------

def feature_coverage_from_preimpute(X_preimpute: pd.DataFrame) -> pd.Series:
    return 1.0 - X_preimpute.isna().mean(axis=0)


def top_feature_set_for_pc(loadings: pd.DataFrame, pc: str, top_min: int, top_frac: float) -> List[str]:
    nfeat = loadings.shape[0]
    top_n = max(top_min, int(np.ceil(top_frac * nfeat)))
    top_n = min(top_n, nfeat)
    return loadings[pc].abs().sort_values(ascending=False).head(top_n).index.tolist()


def parallel_analysis(
    X_scaled: np.ndarray,
    n_components: int,
    B: int,
    quantile: float,
    random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    pca_full = PCA(n_components=n_components, random_state=random_state)
    pca_full.fit(X_scaled)
    lam_obs = pca_full.explained_variance_.copy()

    null_lams = np.zeros((B, n_components), dtype=float)
    Xp = X_scaled.copy()

    for _ in range(B):
        for j in range(Xp.shape[1]):
            idx = rng.permutation(Xp.shape[0])
            Xp[:, j] = X_scaled[idx, j]
        pca_b = PCA(n_components=n_components, random_state=random_state)
        pca_b.fit(Xp)
        null_lams[_, :] = pca_b.explained_variance_

    lam_q = np.quantile(null_lams, quantile, axis=0)
    out = pd.DataFrame({
        "PC": [f"PC{i}" for i in range(1, n_components + 1)],
        "lambda_obs": lam_obs,
        f"lambda_null_q{int(quantile*100)}": lam_q,
    })
    out["keep_parallel"] = out["lambda_obs"] > out[f"lambda_null_q{int(quantile*100)}"]
    return out


def plot_parallel_analysis(par_df: pd.DataFrame, outpath: str) -> None:
    pcs = np.arange(1, par_df.shape[0] + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(pcs, par_df["lambda_obs"].values, marker="o", label="Observed eigenvalue")
    null_col = [c for c in par_df.columns if c.startswith("lambda_null_q")][0]
    plt.plot(pcs, par_df[null_col].values, marker="o", label=f"Null {null_col.replace('lambda_', '')}")
    plt.xlabel("PC")
    plt.ylabel("Eigenvalue")
    plt.title("Parallel analysis (observed vs null quantile)")
    plt.legend()
    save_fig(outpath)


def cluster_features_by_loadings(
    loadings: pd.DataFrame,
    pcs_to_use: List[str],
    n_clusters: int,
    random_state: int
) -> pd.DataFrame:
    M = loadings[pcs_to_use].values.astype(float)
    M = M - M.mean(axis=1, keepdims=True)

    row_norm = np.linalg.norm(M, axis=1)
    if np.any(row_norm == 0):
        rng = np.random.default_rng(random_state)
        M[row_norm == 0, :] += rng.normal(scale=1e-9, size=M[row_norm == 0, :].shape)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(M)

    return pd.DataFrame({
        "feature": loadings.index.tolist(),
        "feature_cluster": labels.astype(int),
    })


def plot_loadings_heatmap_by_cluster(
    loadings: pd.DataFrame,
    feature_clusters: pd.DataFrame,
    pcs_to_show: List[str],
    outpath: str
) -> None:
    if "feature" not in feature_clusters.columns or "feature_cluster" not in feature_clusters.columns:
        raise ValueError("feature_clusters must contain columns: feature, feature_cluster")

    fc = feature_clusters.copy()
    fc["feature"] = fc["feature"].astype(str)
    fc = fc.set_index("feature").reindex(loadings.index)
    fc["feature_cluster"] = fc["feature_cluster"].fillna(-1).astype(int)

    order = (
        fc.reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(["feature_cluster", "feature"])
        ["feature"]
        .tolist()
    )

    order = [f for f in order if f in loadings.index]
    M = loadings.loc[order, pcs_to_show].values

    plt.figure(figsize=(10, max(4, 0.22 * len(order))))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(order)), order, fontsize=7)
    plt.xticks(np.arange(len(pcs_to_show)), pcs_to_show, rotation=45, ha="right")
    plt.xlabel("PC")
    plt.ylabel("Feature (clustered)")
    plt.title("Loadings heatmap ordered by feature cluster")
    plt.colorbar(label="Loading")
    save_fig(outpath)


def get_block_ids(meta: pd.DataFrame, block_col: str) -> pd.Series:
    if block_col in meta.columns:
        return meta[block_col].astype(str).fillna("NA")

    if block_col == "YearMonth":
        if "Year" in meta.columns and "Month" in meta.columns:
            y = pd.to_numeric(meta["Year"], errors="coerce").astype("Int64").astype(str)
            m = pd.to_numeric(meta["Month"], errors="coerce").astype("Int64").astype(str)
            return (y + "-" + m).fillna("NA")

    if block_col == "Month" and "Month" in meta.columns:
        return meta["Month"].astype(str).fillna("NA")

    return pd.Series(["ALL"] * meta.shape[0], index=meta.index)


def align_components_by_loadings(L_full: np.ndarray, L_rep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k_full = L_full.shape[1]
    k_rep = L_rep.shape[1]
    C = np.zeros((k_full, k_rep), dtype=float)
    for i in range(k_full):
        a = L_full[:, i]
        for j in range(k_rep):
            b = L_rep[:, j]
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            C[i, j] = 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    Cabs = np.abs(C)
    best_rep = -np.ones(k_full, dtype=int)
    used = set()
    for i in range(k_full):
        order = np.argsort(-Cabs[i, :])
        j = next((int(x) for x in order if int(x) not in used), int(order[0]))
        used.add(j)
        best_rep[i] = j

    return C, best_rep


def stability_bootstrap(
    X_scaled: np.ndarray,
    meta: pd.DataFrame,
    pca_full: PCA,
    pcs_to_eval: List[str],
    block_col: str,
    R: int,
    random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    full_scores = pca_full.transform(X_scaled)
    full_loadings = pca_full.components_.T  # features x k

    pc_names_full = [f"PC{i}" for i in range(1, pca_full.n_components_ + 1)]
    pc_to_idx = {pc: i for i, pc in enumerate(pc_names_full)}

    blocks = get_block_ids(meta, block_col)
    unique_blocks = blocks.unique().tolist()
    block_to_rows: Dict[str, np.ndarray] = {b: np.where(blocks.values == b)[0] for b in unique_blocks}

    load_corrs: Dict[str, List[float]] = {pc: [] for pc in pcs_to_eval}
    score_corrs: Dict[str, List[float]] = {pc: [] for pc in pcs_to_eval}

    for _ in range(R):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_rows[b] for b in sampled_blocks], axis=0)

        Xr = X_scaled[idx, :]
        pca_r = PCA(n_components=pca_full.n_components_, random_state=random_state)
        pca_r.fit(Xr)

        Lr = pca_r.components_.T
        Sr = pca_r.transform(Xr)

        C, best_rep = align_components_by_loadings(full_loadings, Lr)

        for pc in pcs_to_eval:
            i_full = pc_to_idx[pc]
            j_rep = int(best_rep[i_full])

            corr_loading = float(C[i_full, j_rep])
            load_corrs[pc].append(abs(corr_loading))

            sign = 1.0 if corr_loading >= 0 else -1.0
            s_full = full_scores[idx, i_full]
            s_rep = sign * Sr[:, j_rep]

            if np.std(s_full) == 0 or np.std(s_rep) == 0:
                score_corrs[pc].append(0.0)
            else:
                score_corrs[pc].append(float(np.abs(np.corrcoef(s_full, s_rep)[0, 1])))

    rows = []
    for pc in pcs_to_eval:
        rows.append({
            "PC": pc,
            "med_abs_corr_loadings": float(np.median(load_corrs[pc])) if load_corrs[pc] else np.nan,
            "med_abs_corr_scores": float(np.median(score_corrs[pc])) if score_corrs[pc] else np.nan,
        })
    return pd.DataFrame(rows)


def pc_loading_concentration_metrics(loadings: pd.DataFrame) -> pd.DataFrame:
    """
    Computes concentration/interpretability metrics per PC based on loadings.

    Uses p_i ∝ loading_i^2 (energy / variance contribution proxy).
    - entropy_norm: Shannon entropy normalized to [0,1] by log(K)
        * 0   => all weight on one feature (very concentrated)
        * 1.0 => uniform across features (very diffuse)
    - effective_features: exp(H)  (a.k.a. perplexity / effective #contributors)
    - participation_ratio: (Σ e_i)^2 / Σ e_i^2 where e_i=loading_i^2
        * ~1 => one feature dominates
        * larger => more features contribute meaningfully
    """
    pcs = list(loadings.columns)
    K = loadings.shape[0]
    eps = 1e-300

    rows = []
    for pc in pcs:
        w = loadings[pc].to_numpy(dtype=float)
        e = w * w  # energy
        s = float(np.sum(e))
        if not np.isfinite(s) or s <= 0:
            rows.append({
                "PC": pc,
                "entropy_norm": np.nan,
                "effective_features": np.nan,
                "participation_ratio": np.nan,
            })
            continue

        p = e / s
        # Shannon entropy
        H = -float(np.sum(p * np.log(p + eps)))
        H_norm = H / float(np.log(K)) if K > 1 else 0.0
        eff = float(np.exp(H))
        pr = float((s * s) / float(np.sum(e * e) + eps))

        rows.append({
            "PC": pc,
            "entropy_norm": H_norm,
            "effective_features": eff,
            "participation_ratio": pr,
        })

    return pd.DataFrame(rows)


def run_pc_selection(
    cfg: RunConfig,
    tables_dir: str,
    plots_dir: str,
    X_preimpute: pd.DataFrame,
    X_scaled: np.ndarray,
    meta: pd.DataFrame,
    pca: PCA,
    loadings_df: pd.DataFrame
) -> None:
    cov = feature_coverage_from_preimpute(X_preimpute)
    cov_df = pd.DataFrame({"feature": cov.index, "coverage": cov.values}).sort_values("coverage", ascending=True)
    cov_df.to_csv(os.path.join(tables_dir, "feature_coverage.csv"), index=False)

    ncomp = pca.n_components_
    par_df = parallel_analysis(
        X_scaled=X_scaled,
        n_components=ncomp,
        B=cfg.pcsel_parallel_B,
        quantile=cfg.pcsel_parallel_quantile,
        random_state=cfg.random_state,
    )
    par_df.to_csv(os.path.join(tables_dir, "parallel_analysis.csv"), index=False)
    plot_parallel_analysis(par_df, os.path.join(plots_dir, "parallel_analysis.png"))

    pcs_all = [f"PC{i}" for i in range(1, ncomp + 1)]
    pcs_parallel = par_df.loc[par_df["keep_parallel"], "PC"].tolist()
    if len(pcs_parallel) == 0:
        keep_df = pd.DataFrame({"PC": pcs_all})
        keep_df["keep_parallel"] = par_df["keep_parallel"].values
        keep_df["KEEP"] = False
        keep_df.to_csv(os.path.join(tables_dir, "pc_keep_decision.csv"), index=False)
        return

    conc_df = pc_loading_concentration_metrics(loadings_df)
    conc_df.to_csv(os.path.join(tables_dir, "pc_loading_concentration.csv"), index=False)

    feature_clusters = cluster_features_by_loadings(
        loadings=loadings_df,
        pcs_to_use=pcs_parallel,
        n_clusters=cfg.pcsel_feature_clusters,
        random_state=cfg.random_state,
    )
    feature_clusters = feature_clusters.sort_values(["feature_cluster", "feature"])
    feature_clusters.to_csv(os.path.join(tables_dir, "feature_clusters.csv"), index=False)

    pcs_show = pcs_parallel[: min(10, len(pcs_parallel))]
    plot_loadings_heatmap_by_cluster(
        loadings=loadings_df,
        feature_clusters=feature_clusters,
        pcs_to_show=pcs_show,
        outpath=os.path.join(plots_dir, "loadings_heatmap_by_feature_cluster.png"),
    )

    fc_map = feature_clusters.set_index("feature")["feature_cluster"].to_dict()

    cov_rows = []
    coh_rows = []
    for pc in pcs_parallel:
        Tk = top_feature_set_for_pc(loadings_df, pc, cfg.pcsel_top_min, cfg.pcsel_top_frac)

        cov_Tk = cov.loc[Tk].values
        med_cov = float(np.median(cov_Tk)) if len(cov_Tk) else np.nan
        n_good = int(np.sum(cov_Tk >= cfg.pcsel_support_min_cov))
        keep_cov = (med_cov >= cfg.pcsel_support_median_cov) and (n_good >= cfg.pcsel_support_min_n)

        clusters = [fc_map.get(f, -1) for f in Tk]
        if len(clusters) == 0:
            dom_cluster = -1
            p_dom = np.nan
            keep_coh = False
        else:
            vc = pd.Series(clusters).value_counts(normalize=True)
            dom_cluster = int(vc.index[0])
            p_dom = float(vc.iloc[0])
            keep_coh = p_dom >= cfg.pcsel_coherence_min_frac

        cov_rows.append({
            "PC": pc,
            "Tk_size": int(len(Tk)),
            "median_cov_Tk": med_cov,
            f"n_cov_ge_{cfg.pcsel_support_min_cov:g}": n_good,
            "keep_coverage": bool(keep_cov),
        })
        coh_rows.append({
            "PC": pc,
            "dominant_cluster": dom_cluster,
            "p_dominant": p_dom,
            "keep_coherence": bool(keep_coh),
        })

    pc_cov_df = pd.DataFrame(cov_rows)
    pc_cov_df.to_csv(os.path.join(tables_dir, "pc_coverage_support.csv"), index=False)

    pc_coh_df = pd.DataFrame(coh_rows)
    pc_coh_df.to_csv(os.path.join(tables_dir, "pc_cluster_enrichment.csv"), index=False)

    stab_df = stability_bootstrap(
        X_scaled=X_scaled,
        meta=meta,
        pca_full=pca,
        pcs_to_eval=pcs_parallel,
        block_col=cfg.pcsel_block_col,
        R=cfg.pcsel_stability_R,
        random_state=cfg.random_state,
    )
    stab_df["keep_stability"] = (
        (stab_df["med_abs_corr_loadings"] >= cfg.pcsel_stability_min_load_corr) &
        (stab_df["med_abs_corr_scores"] >= cfg.pcsel_stability_min_score_corr)
    )
    stab_df.to_csv(os.path.join(tables_dir, "pc_stability.csv"), index=False)

    keep_df = pd.DataFrame({"PC": pcs_all}).merge(par_df[["PC", "keep_parallel"]], on="PC", how="left")
    keep_df["keep_parallel"] = keep_df["keep_parallel"].fillna(False)

    keep_df = keep_df.merge(pc_cov_df[["PC", "keep_coverage"]], on="PC", how="left")
    keep_df["keep_coverage"] = keep_df["keep_coverage"].fillna(False)

    keep_df = keep_df.merge(pc_coh_df[["PC", "keep_coherence"]], on="PC", how="left")
    keep_df["keep_coherence"] = keep_df["keep_coherence"].fillna(False)

    keep_df = keep_df.merge(stab_df[["PC", "keep_stability", "med_abs_corr_loadings", "med_abs_corr_scores"]], on="PC", how="left")
    keep_df["keep_stability"] = keep_df["keep_stability"].fillna(False)

    keep_df = keep_df.merge(conc_df, on="PC", how="left")
    keep_df["keep_entropy"] = (
        (keep_df["entropy_norm"] <= cfg.pcsel_entropy_max) |
        (keep_df["participation_ratio"] <= cfg.pcsel_pr_max)
    ).fillna(False)

    keep_df["KEEP"] = (
        keep_df["keep_parallel"] &
        keep_df["keep_coverage"] &
        (keep_df["keep_coherence"] | keep_df["keep_entropy"]) &
        keep_df["keep_stability"]
    )

    keep_df.to_csv(os.path.join(tables_dir, "pc_keep_decision.csv"), index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    df_raw = load_table(cfg.input_path, cfg.sep)
    df_raw = make_date_column(df_raw, cfg.time_col)

    missing_features = [c for c in cfg.feature_cols if c not in df_raw.columns]
    missing_meta = [c for c in cfg.meta_cols if c not in df_raw.columns]
    with open(os.path.join(cfg.outdir, "missing_expected_columns.json"), "w") as f:
        json.dump(
            {
                "missing_features": missing_features,
                "missing_meta": missing_meta,
                "configured_feature_cols_final": cfg.feature_cols,
            },
            f,
            indent=2,
        )

    # Coerce numeric for feature columns + depth columns (if present)
    numeric_cols = [c for c in cfg.feature_cols if c in df_raw.columns]
    if cfg.depth_col in df_raw.columns and cfg.depth_col not in numeric_cols:
        numeric_cols.append(cfg.depth_col)

    # Also coerce anchor-by-col if present (leave as string)
    df_num = coerce_numeric(df_raw, numeric_cols)

    # Metadata columns (plus derived time col)
    meta_cols = [c for c in cfg.meta_cols if c in df_num.columns]
    if cfg.time_col in df_num.columns and cfg.time_col not in meta_cols:
        meta_cols.append(cfg.time_col)

    feats = [c for c in cfg.feature_cols if c in df_num.columns]
    if not feats:
        raise ValueError("None of the configured feature columns were found in your input table.")

    # ---- DATA-DRIVEN depth anchoring (pre-flight) ----
    anchoring_clamped_count = 0
    if cfg.anchor_depths and (cfg.depth_col in df_num.columns):
        df_num, anchors_df, mapping_df, decisions_df, proto_counts_df, anchoring_clamped_count = anchor_depth_column_data_driven(
            df=df_num,
            depth_col=cfg.depth_col,
            anchored_col=cfg.anchored_depth_col,
            by_col=cfg.anchor_by_col,
            feature_cols=feats,
            round_m=cfg.anchor_round_m,
            tol_m=cfg.anchor_tol_m,
            top_k=cfg.anchor_top_k,
            min_count=cfg.anchor_min_count,
            min_features=cfg.anchor_min_features,
            margin=cfg.anchor_margin,
            proto_min_n=cfg.anchor_proto_min_n,
        )
        if not anchors_df.empty:
            anchors_df.to_csv(os.path.join(tables_dir, "depth_anchors.csv"), index=False)
        if not mapping_df.empty:
            mapping_df.to_csv(os.path.join(tables_dir, "depth_anchor_mapping_summary.csv"), index=False)
        if not decisions_df.empty:
            decisions_df.to_csv(os.path.join(tables_dir, "depth_anchor_decisions.csv"), index=False)
            summary_by_block = (
                decisions_df.groupby(["block", "reason"], as_index=False)
                .size()
                .rename(columns={"size": "n"})
                .sort_values(["block", "n"], ascending=[True, False])
            )
            snap_rate = decisions_df.groupby("block", as_index=False)["snapped"].mean().rename(columns={"snapped": "snap_rate"})
            summary_by_block = summary_by_block.merge(snap_rate, on="block", how="left")
            summary_by_block.to_csv(os.path.join(tables_dir, "depth_anchor_decision_summary_by_block.csv"), index=False)
        if not proto_counts_df.empty:
            proto_counts_df.to_csv(os.path.join(tables_dir, "depth_anchor_prototype_audit.csv"), index=False)
    else:
        if cfg.depth_col in df_num.columns and cfg.anchored_depth_col not in df_num.columns:
            df_num[cfg.anchored_depth_col] = pd.to_numeric(df_num[cfg.depth_col], errors="coerce")

    # Always retain anchored depth as metadata (if exists)
    if cfg.anchored_depth_col in df_num.columns and cfg.anchored_depth_col not in meta_cols:
        meta_cols.append(cfg.anchored_depth_col)

    # Missingness pre-drop (features only)
    miss0 = basic_missingness_stats(df_num, feats)
    miss0.to_csv(os.path.join(tables_dir, "missingness_pre_drop.csv"), index=False)
    plot_missingness(miss0, os.path.join(plots_dir, "missingness_pre_drop.png"))

    # Drop sparse (features only)
    df_filt, feats_kept, dropped_rows = drop_sparse(
        df_num, feats, row_thresh=cfg.dropna_row_thresh, col_thresh=cfg.dropna_col_thresh
    )
    dropped_rows.to_csv(os.path.join(tables_dir, "dropped_rows.csv"), index=False)

    miss1 = basic_missingness_stats(df_filt, feats_kept)
    miss1.to_csv(os.path.join(tables_dir, "missingness_post_drop.csv"), index=False)
    plot_missingness(miss1, os.path.join(plots_dir, "missingness_post_drop.png"))

    meta = df_filt[meta_cols].copy() if meta_cols else pd.DataFrame(index=df_filt.index)

    # X_preimpute is the coverage-truth for PC selection
    X_preimpute = df_filt[feats_kept].copy()

    # Impute
    if cfg.impute_scope == "by_depth":
        if cfg.anchored_depth_col not in meta.columns:
            X_imp, fill_values = impute_matrix_global(
                X=X_preimpute,
                strategy=cfg.impute,
                knn_k=cfg.knn_k,
                iterative_max_iter=cfg.iterative_max_iter,
                random_state=cfg.random_state,
            )
        else:
            depth_groups = meta[cfg.anchored_depth_col].astype(str).fillna("NA")
            X_imp, audit_df, fill_values = impute_matrix_by_depth(
                X=X_preimpute,
                depth_groups=depth_groups,
                strategy=cfg.impute,
                knn_k=cfg.knn_k,
                iterative_max_iter=cfg.iterative_max_iter,
                min_group_size=cfg.impute_min_group_size,
                random_state=cfg.random_state,
            )
            audit_df.to_csv(os.path.join(tables_dir, "imputation_depth_group_audit.csv"), index=False)
    else:
        X_imp, fill_values = impute_matrix_global(
            X=X_preimpute,
            strategy=cfg.impute,
            knn_k=cfg.knn_k,
            iterative_max_iter=cfg.iterative_max_iter,
            random_state=cfg.random_state,
        )

    fill_values.rename("impute_value" if cfg.impute in ("median", "mean") else "imputed_feature_mean") \
        .to_csv(os.path.join(tables_dir, "impute_values.csv"), header=True)

    # Clamp negatives to 0 (always, per your preference)
    X_imp, n_clamped = clamp_negatives_to_zero(X_imp)

    # Optional log1p
    if cfg.log1p:
        X_imp = maybe_log1p(X_imp)

    # Save cleaned matrix (post-impute / post-clamp / post-log)
    cleaned = pd.concat([meta.reset_index(drop=True), X_imp.reset_index(drop=True)], axis=1)
    cleaned.to_csv(os.path.join(tables_dir, "matrix_cleaned.csv"), index=False)

    # Scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp.values)

    scaled_df = pd.DataFrame(X_scaled, columns=feats_kept)
    scaled_out = pd.concat([meta.reset_index(drop=True), scaled_df], axis=1)
    scaled_out.to_csv(os.path.join(tables_dir, "matrix_scaled.csv"), index=False)

    # PCA
    ncomp = min(cfg.n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=ncomp, random_state=cfg.random_state)
    pca.fit(X_scaled)

    # Explained variance
    evr = pca.explained_variance_ratio_
    ev = pca.explained_variance_
    ev_tbl = pd.DataFrame({
        "PC": [f"PC{i}" for i in range(1, len(evr) + 1)],
        "explained_variance": ev,
        "explained_variance_ratio": evr,
        "cumulative_ratio": np.cumsum(evr),
    })
    ev_tbl.to_csv(os.path.join(tables_dir, "pca_explained_variance.csv"), index=False)

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feats_kept,
        columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)],
    )
    loadings.to_csv(os.path.join(tables_dir, "pca_loadings.csv"))

    # Eigengenes (scores)
    scores = pca.transform(X_scaled)
    scores_df = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)])
    eigengenes = pd.concat([meta.reset_index(drop=True), scores_df], axis=1)
    eigengenes.to_csv(os.path.join(tables_dir, "eigengenes_scores.csv"), index=False)

    # QC summary
    qc = {
        "id_col": cfg.id_col,
        "time_col": cfg.time_col,
        "meta_cols_used": meta_cols,
        "n_rows_input": int(df_raw.shape[0]),
        "n_rows_kept": int(df_filt.shape[0]),
        "configured_feature_cols_final": cfg.feature_cols,
        "n_features_found": int(len(feats)),
        "n_features_kept": int(len(feats_kept)),
        "impute_strategy": cfg.impute,
        "impute_scope": cfg.impute_scope,
        "impute_min_group_size": int(cfg.impute_min_group_size),
        "knn_k": int(cfg.knn_k),
        "iterative_max_iter": int(cfg.iterative_max_iter),
        "depth_anchoring_enabled": bool(cfg.anchor_depths),
        "depth_col": cfg.depth_col,
        "anchored_depth_col": cfg.anchored_depth_col,
        "anchor_by_col": cfg.anchor_by_col,
        "anchor_min_features": int(cfg.anchor_min_features),
        "anchor_margin": float(cfg.anchor_margin),
        "anchor_proto_min_n": int(cfg.anchor_proto_min_n),
        "log1p": bool(cfg.log1p),
        "n_negative_values_clamped_to_zero": int(n_clamped),
        "n_negative_values_clamped_to_zero_during_anchoring_similarity": int(anchoring_clamped_count),
        "n_components_fit": int(pca.n_components_),
        "pc_selection_ran": bool(cfg.pc_selection),
    }
    with open(os.path.join(cfg.outdir, "qc_summary.json"), "w") as f:
        json.dump(qc, f, indent=2)

    # Plots
    plot_scree(pca, os.path.join(plots_dir, "scree.png"))
    plot_cumvar(pca, os.path.join(plots_dir, "cumulative_variance.png"))
    plot_pc_scatter(eigengenes, os.path.join(plots_dir, "pc1_vs_pc2.png"), cfg.time_col)

    for i in range(1, min(6, pca.n_components_ + 1)):
        pc = f"PC{i}"
        plot_top_loadings(loadings, os.path.join(plots_dir, f"top_loadings_{pc}.png"), pc=pc, top_n=25)

    # Optional: PC selection & feature clustering
    if cfg.pc_selection:
        run_pc_selection(
            cfg=cfg,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            X_preimpute=X_preimpute,
            X_scaled=X_scaled,
            meta=meta.reset_index(drop=True),
            pca=pca,
            loadings_df=loadings,
        )

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"      Tables: {tables_dir}")
    print(f"      Plots : {plots_dir}")
    if missing_features:
        print(f"[WARN] Missing expected feature cols (see missing_expected_columns.json): {len(missing_features)}")
    if missing_meta:
        print(f"[WARN] Missing expected meta cols (see missing_expected_columns.json): {len(missing_meta)}")


if __name__ == "__main__":
    main()
