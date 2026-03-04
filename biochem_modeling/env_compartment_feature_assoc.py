#!/usr/bin/env python3

"""
SPARK/env_compartment_feature_assoc.py

Purpose
-------
Quantify and visualize associations between final GMM-defined environmental compartments and
individual biogeochemical features in a statistically conservative, reviewer-defensible manner.

Given:
  1) matrix_cleaned.csv
     (biogeochemical features + metadata per sample; post-QC, post-imputation)
  2) compartments_assignments_smoothed.csv
     (final GMM hard assignments + soft responsibilities resp_0..resp_{K-1})

this script computes, for each compartment × feature pair:

  - Robust effect sizes comparing the compartment to all other samples
  - Direction and magnitude of feature shifts (median-, mean-, and percent-based)
  - Continuous responsibility–feature associations
  - Cruise-blocked bootstrap confidence intervals
  - Optional depth-adjusted variants that remove linear depth trends

The goal is not hypothesis testing via p-values, but **effect-size–first characterization**
with uncertainty quantified by block resampling.

Conceptual framing
------------------
Compartments are treated as *environmental states* inferred from multivariate structure.
Feature associations answer two complementary questions:

1) Discrete contrast:
   “How different is feature X inside compartment k compared to the rest of the dataset?”

2) Continuous gradient:
   “Does increasing responsibility for compartment k covary with feature X?”

Both views are reported to avoid over-reliance on hard cluster boundaries.

Inputs
------
1) --matrix-cleaned
   Path to matrix_cleaned.csv.

   Required columns (names configurable via CLI):
     - Sample ID (default: cruise_year_month_depth)
     - Cruise identifier (default: Cruise)
     - Depth (default: Depth)
     - Anchored depth (default: Depth_anchored)
     - Oxygen (default: Oxygen; used only for O2 compartment cross-tab)
     - Optional date column (default: date)

   Feature columns:
     - If --feature-cols is provided: those columns are used verbatim.
     - Otherwise: all numeric columns in matrix_cleaned excluding:
         • metadata (ID, cruise, depth, date, etc.)
         • any PC columns
         • any resp_* columns
         • GMM diagnostic columns if present

2) --assignments
   Path to compartments_assignments_smoothed.csv.

   Required columns:
     - Sample ID (same as matrix_cleaned)
     - Cruise identifier
     - component (hard GMM label)
     - resp_0 .. resp_{K-1} (soft responsibilities)

   Optional but used if present:
     - max_prob, entropy_norm, knn_mean_dist (not required for associations themselves)

Duplicate column labels are explicitly handled by keeping the first occurrence only.

Merging behavior
----------------
Assignments are treated as authoritative for sample inclusion.

The two tables are merged as:
  assignments LEFT JOIN matrix_cleaned on id_col

This ensures:
  - All GMM-labeled samples are retained
  - Missing feature values propagate as NaN and are handled feature-wise

No assumptions are made about row order or completeness beyond the merge key.

Oxygen compartments (auxiliary)
-------------------------------
For contextual comparison only, oxygen-defined compartments are computed from the Oxygen column
(assumed to be in µM, with no unit conversion):

  - oxic    : O2 > o2_oxic_gt
  - dysoxic : o2_dysoxic_lo ≤ O2 ≤ o2_dysoxic_hi
  - suboxic : o2_suboxic_lo ≤ O2 <  o2_suboxic_hi
  - anoxic  : O2 < o2_anoxic_lt
  - NA      : Oxygen missing or non-numeric

A simple cross-tab of O2 compartment vs GMM component is written for reference only; it is not
used in the association statistics.

Association metrics (per component × feature)
----------------------------------------------
Let “in” denote samples assigned to component k, and “out” all other samples.

Computed quantities:

1) Location shifts
   - mean_in, mean_out
   - median_in, median_out
   - median_shift = median_in − median_out
   - percent_shift_vs_out = (median_in − median_out) / |median_out|

2) Effect size (primary)
   - Hedges’ g (bias-corrected standardized mean difference)
   - Computed on raw feature values
   - Returned as NaN when variance or sample size is insufficient

3) Responsibility association (continuous)
   - Slope of: resp_k ~ z(feature)
   - Equivalent to cov(z(feature), resp_k) / var(z(feature))
   - Interpretable as change in responsibility per 1 SD increase in the feature

All metrics are computed feature-wise with NaNs handled explicitly.

Cruise-blocked bootstrap (uncertainty)
--------------------------------------
To account for non-independence of samples within cruises, uncertainty is quantified using a
blocked bootstrap with Cruise as the resampling unit:

  - Cruises are sampled with replacement
  - All samples from selected cruises are included in each bootstrap replicate
  - For each replicate:
      • Hedges’ g
      • responsibility–feature slope
    are recomputed

Reported uncertainty:
  - 2.5% and 97.5% percentile confidence intervals for each metric

Bootstrap size is controlled by --bootstrap-B (default 500).

Depth-adjusted variant (optional)
---------------------------------
If --depth-adjust is set:

  - Each feature is residualized against anchored depth using a simple linear model:
        feature_resid = feature − (a + b·Depth_anchored)
  - Residualization is performed using only finite (feature, depth) pairs
  - Features with <20 valid points return NaN residuals

All association metrics and bootstraps are recomputed on the residualized feature matrix and
written as a separate result set tagged “depth_adjusted”.

Outputs
-------
Directory structure under --outdir:

tables/
  - associations_raw.csv
  - associations_depth_adjusted.csv              (if enabled)
  - associations_raw_toprank.csv
  - associations_depth_adjusted_toprank.csv      (if enabled)
  - o2_compartment_confusion_vs_gmm.csv
  - run_config.json

plots/
  - lollipop_{model}_c{k}_{component_label}_{raw|depth_adjusted}.(png|pdf|svg)
  - heatmap_effectsize_{model}_{raw|depth_adjusted}_component_labels.(png|pdf|svg)
  - heatmap_effectsize_hybrid_{raw|depth_adjusted}_component_labels_splitfiltered.(png|pdf|svg)  [when split filter is used]

Plot semantics:
  - Lollipop plots show Hedges’ g with cruise-blocked CIs for the strongest positive and negative
    features per compartment.
  - Heatmaps show Hedges’ g across compartments for the globally strongest features.

Operational notes / invariants
------------------------------
- No p-values are computed or reported.
- All uncertainty reflects cruise-level resampling, not IID assumptions.
- Hard labels (component) and soft responsibilities (resp_k) are both used, but never conflated.
- Depth adjustment is strictly linear and optional; raw associations are always preserved.
- The script makes no assumptions about feature units or scaling beyond what is present in
  matrix_cleaned.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# -----------------------------
# Constants / defaults
# -----------------------------

O2_PALETTE = {
    "oxic": "red",
    "dysoxic": "lightblue",
    "suboxic": "green",
    "anoxic": "purple",
}

O2_COMPONENT_INDEX_TO_NAME = {
    0: "oxic",
    1: "dysoxic",
    2: "suboxic",
    3: "anoxic",
}

# Low values -> blue, high values -> orange (with neutral midpoint).
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_orange",
    ["#1f5aa6", "#f7f7f7", "#f08a24"],
    N=2048,
)

DEFAULT_ID_COL = "cruise_year_month_depth"
DEFAULT_CRUISE_COL = "Cruise"
DEFAULT_DEPTH_COL = "Depth"
DEFAULT_DEPTH_ANCH_COL = "Depth_anchored"
DEFAULT_DATE_COL = "date"
DEFAULT_OXYGEN_COL = "Oxygen"


# -----------------------------
# CLI config
# -----------------------------

@dataclass
class Config:
    matrix_cleaned: str
    assignments: str | None
    assignments_gmm: str | None
    assignments_o2: str | None
    assignments_hybrid: str | None
    outdir: str
    sep_matrix: str
    sep_assign: str

    id_col: str
    cruise_col: str
    depth_col: str
    depth_anchored_col: str
    date_col: str
    oxygen_col: str

    # O2 thresholds (uM)
    o2_oxic_gt: float
    o2_dysoxic_lo: float
    o2_dysoxic_hi: float
    o2_suboxic_lo: float
    o2_suboxic_hi: float
    o2_anoxic_lt: float

    # Which biochem features to analyze
    feature_cols: List[str]

    # Bootstrap / ranking
    bootstrap_B: int
    random_state: int
    top_n_each_side: int
    min_n_comp: int
    min_n_rest: int

    # Depth adjustment
    do_depth_adjust: bool

    # Optional: restrict HYBRID heatmap compartments to those retained in split table.
    hybrid_split_table: str | None
    hybrid_split_col: str


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Compartment↔feature associations with cruise-blocked bootstrap (lollipops + tables)."
    )

    ap.add_argument("--matrix-cleaned", required=True, help="Path to PCA matrix_cleaned.csv")
    ap.add_argument(
        "--assignments",
        default=None,
        help="Legacy single assignments path. If model-specific assignment paths are provided, this is ignored.",
    )
    ap.add_argument("--assignments-gmm", default=None, help="Path to smoothed GMM assignments table.")
    ap.add_argument("--assignments-o2", default=None, help="Path to smoothed O2 assignments table.")
    ap.add_argument("--assignments-hybrid", default=None, help="Path to smoothed hybrid assignments table.")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep-matrix", default=",", help="Delimiter for matrix_cleaned (default ',')")
    ap.add_argument("--sep-assign", default=",", help="Delimiter for assignments (default ',')")

    ap.add_argument("--id-col", default=DEFAULT_ID_COL)
    ap.add_argument("--cruise-col", default=DEFAULT_CRUISE_COL)
    ap.add_argument("--depth-col", default=DEFAULT_DEPTH_COL)
    ap.add_argument("--depth-anchored-col", default=DEFAULT_DEPTH_ANCH_COL)
    ap.add_argument("--date-col", default=DEFAULT_DATE_COL)
    ap.add_argument("--oxygen-col", default=DEFAULT_OXYGEN_COL)

    ap.add_argument(
        "--feature-cols",
        default=None,
        help=(
            "Comma-separated list of feature columns to analyze. "
            "If omitted, uses all numeric non-meta columns from matrix_cleaned, excluding PCs/resp columns."
        ),
    )

    ap.add_argument("--bootstrap-B", type=int, default=500, help="Cruise-block bootstrap replicates (default 500)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-n-each-side", type=int, default=8, help="Top N positive and negative features per compartment")
    ap.add_argument("--min-n-comp", type=int, default=20, help="Min samples inside a compartment to compute stats")
    ap.add_argument("--min-n-rest", type=int, default=50, help="Min samples outside compartment to compute stats")

    ap.add_argument(
        "--depth-adjust",
        action="store_true",
        help="Also compute depth-adjusted associations (residualize features vs Depth_anchored).",
    )

    ap.add_argument(
        "--hybrid-split-table",
        default=None,
        help=(
            "Optional path to merged_o2_split_by_gmm.csv. "
            "If provided, HYBRID heatmaps keep only retained subcompartments from this table "
            "(excluding labels containing '__other')."
        ),
    )
    ap.add_argument(
        "--hybrid-split-col",
        default="o2_subcompartment_final",
        help=(
            "Column in --hybrid-split-table containing split labels like 'oxic__gmm0' "
            "(default o2_subcompartment_final)."
        ),
    )

    # O2 thresholds (uM)
    ap.add_argument("--o2-oxic-gt", type=float, default=90.0, help="oxic if O2 > this (default 90)")
    ap.add_argument("--o2-dysoxic-lo", type=float, default=20.0, help="dysoxic lower bound (default 20)")
    ap.add_argument("--o2-dysoxic-hi", type=float, default=90.0, help="dysoxic upper bound (default 90)")
    ap.add_argument("--o2-suboxic-lo", type=float, default=1.0, help="suboxic lower bound (default 1)")
    ap.add_argument("--o2-suboxic-hi", type=float, default=20.0, help="suboxic upper bound (default 20)")
    ap.add_argument("--o2-anoxic-lt", type=float, default=1.0, help="anoxic if O2 < this (default 1)")

    ns = ap.parse_args()

    feature_cols: List[str] = []
    if ns.feature_cols:
        feature_cols = [x.strip() for x in ns.feature_cols.split(",") if x.strip()]

    return Config(
        matrix_cleaned=ns.matrix_cleaned,
        assignments=ns.assignments,
        assignments_gmm=ns.assignments_gmm,
        assignments_o2=ns.assignments_o2,
        assignments_hybrid=ns.assignments_hybrid,
        outdir=ns.outdir,
        sep_matrix=ns.sep_matrix,
        sep_assign=ns.sep_assign,
        id_col=ns.id_col,
        cruise_col=ns.cruise_col,
        depth_col=ns.depth_col,
        depth_anchored_col=ns.depth_anchored_col,
        date_col=ns.date_col,
        oxygen_col=ns.oxygen_col,
        o2_oxic_gt=ns.o2_oxic_gt,
        o2_dysoxic_lo=ns.o2_dysoxic_lo,
        o2_dysoxic_hi=ns.o2_dysoxic_hi,
        o2_suboxic_lo=ns.o2_suboxic_lo,
        o2_suboxic_hi=ns.o2_suboxic_hi,
        o2_anoxic_lt=ns.o2_anoxic_lt,
        feature_cols=feature_cols,
        bootstrap_B=ns.bootstrap_B,
        random_state=ns.random_state,
        top_n_each_side=ns.top_n_each_side,
        min_n_comp=ns.min_n_comp,
        min_n_rest=ns.min_n_rest,
        do_depth_adjust=bool(ns.depth_adjust),
        hybrid_split_table=ns.hybrid_split_table,
        hybrid_split_col=ns.hybrid_split_col,
    )


# -----------------------------
# I/O helpers
# -----------------------------

def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def read_csv_keep_first_duplicate_cols(path: str, sep: str) -> pd.DataFrame:
    """
    Some upstream steps can accidentally write duplicate column labels.
    Pandas merge will error if the merge key label is duplicated.

    Fix: keep only the first occurrence of each column name.
    """
    df = pd.read_csv(path, sep=sep)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def safe_name(text: str) -> str:
    """
    Convert labels to filesystem-friendly names.
    """
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s if s else "na"


# -----------------------------
# O2 compartments
# -----------------------------

def o2_compartment(o2_uM: float, cfg: Config) -> str:
    if np.isnan(o2_uM):
        return "NA"
    if o2_uM > cfg.o2_oxic_gt:
        return "oxic"
    if cfg.o2_dysoxic_lo <= o2_uM <= cfg.o2_dysoxic_hi:
        return "dysoxic"
    if cfg.o2_suboxic_lo <= o2_uM < cfg.o2_suboxic_hi:
        return "suboxic"
    if o2_uM < cfg.o2_anoxic_lt:
        return "anoxic"
    # fallback (should rarely occur if thresholds overlap cleanly)
    return "NA"


# -----------------------------
# Stats helpers
# -----------------------------

def zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    if not np.isfinite(s) or s == 0:
        return x * 0.0
    return (x - m) / s


def hedges_g(x1: np.ndarray, x0: np.ndarray) -> float:
    """
    Hedges g (bias-corrected standardized mean difference).
    Returns NaN if not computable.
    """
    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return np.nan
    m1, m0 = float(np.mean(x1)), float(np.mean(x0))
    s1, s0 = float(np.var(x1, ddof=1)), float(np.var(x0, ddof=1))
    sp_num = (n1 - 1) * s1 + (n0 - 1) * s0
    sp_den = (n1 + n0 - 2)
    if sp_den <= 0:
        return np.nan
    sp = np.sqrt(sp_num / sp_den) if sp_num > 0 else 0.0
    if sp == 0:
        return np.nan
    d = (m1 - m0) / sp
    J = 1.0 - (3.0 / (4.0 * (n1 + n0) - 9.0))
    return float(J * d)


def resp_slope(resp: np.ndarray, feat: np.ndarray) -> float:
    """
    Slope in resp ~ z(feat) using least squares.
    Equivalent to cov(z, resp)/var(z).
    """
    mask = np.isfinite(resp) & np.isfinite(feat)
    if mask.sum() < 5:
        return np.nan
    z = zscore(feat[mask])
    y = resp[mask].astype(float)
    vz = np.var(z, ddof=1)
    if not np.isfinite(vz) or vz == 0:
        return np.nan
    return float(np.cov(z, y, ddof=1)[0, 1] / vz)


def percentile_ci(vals: List[float], lo: float = 0.025, hi: float = 0.975) -> Tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(a, lo)), float(np.quantile(a, hi)))


def percent_shift(med_in: float, med_out: float) -> float:
    """
    Percent shift relative to 'out'. Uses abs(med_out) for stability.
    """
    if not np.isfinite(med_in) or not np.isfinite(med_out):
        return np.nan
    denom = abs(med_out)
    if denom == 0:
        return np.nan
    return float((med_in - med_out) / denom * 100.0)


# -----------------------------
# Depth adjustment
# -----------------------------

def residualize_against_depth(feat: pd.Series, depth: pd.Series) -> pd.Series:
    """
    Remove a linear depth trend: feat_resid = feat - (a + b*depth).
    Uses only finite pairs. Returns residual series (NaNs preserved where feat NaN).
    """
    y = pd.to_numeric(feat, errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(depth, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return pd.Series(np.full_like(y, np.nan), index=feat.index)
    # Fit y = a + b x
    b, a = np.polyfit(x[mask], y[mask], deg=1)
    yhat = a + b * x
    resid = y - yhat
    resid[~np.isfinite(y)] = np.nan
    return pd.Series(resid, index=feat.index)


# -----------------------------
# Main association computation
# -----------------------------

def infer_resp_cols(df_assign: pd.DataFrame) -> List[str]:
    resp = [c for c in df_assign.columns if c.startswith("resp_")]
    # stable ordering: resp_0, resp_1, ...
    resp_sorted = sorted(resp, key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 10**9)
    return resp_sorted


def choose_feature_cols(cfg: Config, df_matrix: pd.DataFrame, df_assign: pd.DataFrame) -> List[str]:
    if cfg.feature_cols:
        return [c for c in cfg.feature_cols if c in df_matrix.columns]

    # Auto: numeric columns in matrix_cleaned excluding likely-meta and any PC/resp columns.
    meta_like = {
        cfg.id_col, cfg.cruise_col, "Year", "Month", "Day",
        cfg.depth_col, cfg.depth_anchored_col, cfg.date_col,
        "component", "max_prob", "entropy_norm", "knn_mean_dist",
    }
    # Exclude PCs if they exist in matrix (shouldn't, but be safe)
    pc_like = {c for c in df_matrix.columns if c.startswith("PC")}
    resp_like = {c for c in df_assign.columns if c.startswith("resp_")}

    cand = []
    for c in df_matrix.columns:
        if c in meta_like or c in pc_like or c in resp_like:
            continue
        # numeric?
        x = pd.to_numeric(df_matrix[c], errors="coerce")
        if x.notna().sum() > 0:
            cand.append(c)

    return cand


def resolve_assignment_jobs(cfg: Config) -> List[Tuple[str, str]]:
    """
    Resolve which assignment tables to analyze.
    Priority:
      1) explicit model-specific paths (--assignments-gmm/o2/hybrid)
      2) legacy --assignments as a single 'gmm' job
    """
    jobs: List[Tuple[str, str]] = []
    if cfg.assignments_gmm:
        jobs.append(("gmm", cfg.assignments_gmm))
    if cfg.assignments_o2:
        jobs.append(("o2", cfg.assignments_o2))
    if cfg.assignments_hybrid:
        jobs.append(("hybrid", cfg.assignments_hybrid))

    if jobs:
        return jobs

    if cfg.assignments:
        return [("gmm", cfg.assignments)]

    raise ValueError(
        "No assignments inputs provided. Pass at least one of: "
        "--assignments-gmm, --assignments-o2, --assignments-hybrid, or legacy --assignments."
    )


def infer_component_labels(
    df: pd.DataFrame,
    component_col: str = "component",
    label_col: str = "compartment_name",
) -> Dict[int, str]:
    """
    Build an integer component -> human-readable label map.
    If label_col does not exist, returns an empty map and callers can fall back to comp{k}.
    """
    out: Dict[int, str] = {}
    if component_col not in df.columns or label_col not in df.columns:
        return out

    comp_num = pd.to_numeric(df[component_col], errors="coerce")
    labels = df[label_col].astype("object")
    tmp = pd.DataFrame({"component": comp_num, "label": labels})
    tmp = tmp.dropna(subset=["component", "label"]).copy()
    if tmp.empty:
        return out

    tmp["component"] = tmp["component"].astype(int)
    tmp["label"] = tmp["label"].astype(str).str.strip()
    tmp = tmp[tmp["label"].str.len() > 0]
    if tmp.empty:
        return out

    for k, sub in tmp.groupby("component", sort=True):
        mode_vals = sub["label"].mode(dropna=True)
        chosen = mode_vals.iloc[0] if not mode_vals.empty else sub["label"].iloc[0]
        out[int(k)] = str(chosen)
    return out


def format_component_label(model_name: str, component: int, raw_label: str | None) -> str:
    """
    Return a readable, model-aware label for tables/plots.
    Keeps raw labels when useful, but expands hybrid cell codes.
    """
    raw = (raw_label or "").strip()

    if model_name == "hybrid":
        # Hybrid builder emits names like: hyb_C2_G4
        m = re.fullmatch(r"hyb_C(\d+)_G(\d+)", raw)
        if m:
            c_idx = int(m.group(1))
            g_idx = int(m.group(2))
            o2_name = O2_COMPONENT_INDEX_TO_NAME.get(c_idx, f"C{c_idx}")
            return f"hybrid_{o2_name}_GMM{g_idx}"
        return f"hybrid_comp{component}"

    if model_name == "o2":
        if raw:
            return raw
        return O2_COMPONENT_INDEX_TO_NAME.get(component, f"o2_comp{component}")

    if model_name == "gmm":
        if raw:
            return raw
        return f"gmm_comp{component}"

    return raw if raw else f"comp{component}"


def build_component_display_map(
    df: pd.DataFrame,
    model_name: str,
    component_col: str,
    raw_map: Dict[int, str],
    n_resp_cols: int | None = None,
) -> Dict[int, str]:
    comp_num = pd.to_numeric(df[component_col], errors="coerce").dropna().astype(int)
    comps = set(comp_num.unique().tolist())
    if n_resp_cols is not None and n_resp_cols > 0:
        comps.update(range(int(n_resp_cols)))
    comps_sorted = sorted(int(c) for c in comps)
    out: Dict[int, str] = {}
    for k in comps_sorted:
        raw_label = raw_map.get(int(k), "")
        label = format_component_label(model_name, int(k), raw_label)

        # For hybrid, some cells may have no hard assignments, so raw labels can be missing.
        # Recover readable labels directly from component index when possible.
        if model_name == "hybrid" and (not raw_label):
            if n_resp_cols is not None and n_resp_cols > 0 and n_resp_cols % 4 == 0:
                gmm_k = int(n_resp_cols // 4)
                c_idx = int(k // gmm_k)
                g_idx = int(k % gmm_k)
                if 0 <= c_idx <= 3:
                    o2_name = O2_COMPONENT_INDEX_TO_NAME.get(c_idx, f"C{c_idx}")
                    label = f"hybrid_{o2_name}_GMM{g_idx}"

        out[int(k)] = label
    return out


def load_hybrid_heatmap_allowed_labels(cfg: Config) -> set[str] | None:
    """
    Build the set of hybrid component labels to display in heatmaps from split-by-GMM output.
    Expected split labels: '<o2>__gmm<idx>' (e.g., 'oxic__gmm0').
    Rows containing '__other' are excluded.
    """
    if not cfg.hybrid_split_table:
        return None
    if not os.path.exists(cfg.hybrid_split_table):
        raise ValueError(f"--hybrid-split-table not found: {cfg.hybrid_split_table}")

    s = read_csv_keep_first_duplicate_cols(cfg.hybrid_split_table, cfg.sep_assign)
    if cfg.hybrid_split_col not in s.columns:
        raise ValueError(
            f"--hybrid-split-col '{cfg.hybrid_split_col}' not found in {cfg.hybrid_split_table}. "
            f"Available columns: {list(s.columns)}"
        )

    raw = s[cfg.hybrid_split_col].astype("object").fillna("").astype(str).str.strip()
    raw = raw[raw.str.len() > 0]
    raw = raw[~raw.str.contains("__other", case=False, regex=False)]

    out: set[str] = set()
    for v in raw.unique().tolist():
        m = re.fullmatch(r"([A-Za-z]+)__gmm(\d+)", v, flags=re.IGNORECASE)
        if not m:
            continue
        o2_name = m.group(1).lower()
        g_idx = int(m.group(2))
        out.add(f"hybrid_{o2_name}_GMM{g_idx}")
    return out if out else set()


def compute_associations(
    df: pd.DataFrame,
    resp_cols: List[str],
    feature_cols: List[str],
    cfg: Config,
    tables_dir: str,
    plots_dir: str,
    model_name: str,
    tag: str,
    component_label_map: Dict[int, str],
    component_label_raw_map: Dict[int, str],
    hybrid_heatmap_allowed_labels: set[str] | None = None,
) -> pd.DataFrame:
    """
    df must contain:
      - cfg.cruise_col
      - 'component' (hard labels)
      - resp_k columns
      - feature_cols numeric
    """
    rng = np.random.default_rng(cfg.random_state)

    # Validate blocking unit
    if cfg.cruise_col not in df.columns:
        raise ValueError(f"Missing cruise_col '{cfg.cruise_col}' in merged dataframe.")

    cruises = df[cfg.cruise_col].astype(str).fillna("NA")
    unique_cruises = cruises.unique().tolist()
    cruise_to_idx: Dict[str, np.ndarray] = {c: np.where(cruises.values == c)[0] for c in unique_cruises}

    # Convert features to numeric matrix once
    Xfeat = {}
    for f in feature_cols:
        Xfeat[f] = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)

    comp = pd.to_numeric(df["component"], errors="coerce").astype("Int64")

    rows = []
    for k in range(len(resp_cols)):
        # allow if component labels are 0..K-1 but some missing
        in_k = (comp == k).to_numpy()
        n_in = int(np.nansum(in_k))
        n_out = int(df.shape[0] - n_in)

        if n_in < cfg.min_n_comp or n_out < cfg.min_n_rest:
            # still compute, but will likely produce NaNs; keep record
            pass

        resp_k = pd.to_numeric(df[resp_cols[k]], errors="coerce").to_numpy(dtype=float)

        for f in feature_cols:
            x = Xfeat[f]
            x_in = x[in_k]
            x_out = x[~in_k]

            med_in = float(np.nanmedian(x_in)) if np.isfinite(x_in).any() else np.nan
            med_out = float(np.nanmedian(x_out)) if np.isfinite(x_out).any() else np.nan
            mean_in = float(np.nanmean(x_in)) if np.isfinite(x_in).any() else np.nan
            mean_out = float(np.nanmean(x_out)) if np.isfinite(x_out).any() else np.nan

            g = hedges_g(x_in, x_out)
            slope = resp_slope(resp_k, x)

            # Cruise-blocked bootstrap for CI
            g_boot: List[float] = []
            slope_boot: List[float] = []

            for _ in range(cfg.bootstrap_B):
                sampled = rng.choice(unique_cruises, size=len(unique_cruises), replace=True)
                idx = np.concatenate([cruise_to_idx[c] for c in sampled], axis=0)

                in_k_b = in_k[idx]
                x_b = x[idx]
                resp_b = resp_k[idx]

                # effect size
                gb = hedges_g(x_b[in_k_b], x_b[~in_k_b])
                g_boot.append(gb)

                # slope
                sb = resp_slope(resp_b, x_b)
                slope_boot.append(sb)

            g_lo, g_hi = percentile_ci(g_boot)
            s_lo, s_hi = percentile_ci(slope_boot)

            rows.append({
                "model": model_name,
                "tag": tag,
                "component": int(k),
                "component_label_raw": component_label_raw_map.get(int(k), f"comp{k}"),
                "component_label": component_label_map.get(int(k), f"comp{k}"),
                "feature": f,
                "n_in": int(n_in),
                "n_out": int(n_out),
                "mean_in": mean_in,
                "mean_out": mean_out,
                "median_in": med_in,
                "median_out": med_out,
                "median_shift": float(med_in - med_out) if np.isfinite(med_in) and np.isfinite(med_out) else np.nan,
                "percent_shift_vs_out": percent_shift(med_in, med_out),
                "hedges_g": g,
                "hedges_g_ci_lo": g_lo,
                "hedges_g_ci_hi": g_hi,
                "resp_slope_per_1SD": slope,
                "resp_slope_ci_lo": s_lo,
                "resp_slope_ci_hi": s_hi,
            })

    out = pd.DataFrame(rows)

    # Rank helper: strongest |g| with finite estimate
    out["abs_g"] = out["hedges_g"].abs()
    out["abs_slope"] = out["resp_slope_per_1SD"].abs()
    out = out.sort_values(["component", "abs_g"], ascending=[True, False])

    out.to_csv(os.path.join(tables_dir, f"associations_{model_name}_{tag}.csv"), index=False)

    # Per-compartment lollipop plots
    for k in sorted(out["component"].unique().tolist()):
        dk = out[out["component"] == k].copy()
        dk = dk[np.isfinite(dk["hedges_g"].to_numpy(dtype=float))].copy()
        if dk.empty:
            continue

        # pick top positive and negative by g
        pos = dk.sort_values("hedges_g", ascending=False).head(cfg.top_n_each_side)
        neg = dk.sort_values("hedges_g", ascending=True).head(cfg.top_n_each_side)
        dplot = pd.concat([pos, neg], axis=0).drop_duplicates(subset=["feature"]).copy()
        # order by g
        dplot = dplot.sort_values("hedges_g", ascending=True)

        plt.figure(figsize=(9, max(4, 0.35 * len(dplot))))
        y = np.arange(len(dplot))
        xg = dplot["hedges_g"].to_numpy(dtype=float)
        xlo = dplot["hedges_g_ci_lo"].to_numpy(dtype=float)
        xhi = dplot["hedges_g_ci_hi"].to_numpy(dtype=float)

        # errorbars
        plt.errorbar(
            xg, y,
            xerr=[xg - xlo, xhi - xg],
            fmt="o",
            capsize=3,
            linestyle="none",
        )
        plt.axvline(0, linewidth=1)
        plt.yticks(y, dplot["feature"].astype(str).tolist())
        plt.xlabel("Hedges g (compartment vs rest)  [cruise-blocked CI]")
        comp_label = component_label_map.get(int(k), f"comp{k}")
        plt.title(f"{model_name.upper()} compartment {k} ({comp_label}) feature associations ({tag})")

        comp_slug = safe_name(comp_label)
        for ext in ("png", "pdf", "svg"):
            save_fig(
                os.path.join(
                    plots_dir,
                    f"lollipop_{model_name}_c{k}_{comp_slug}_{tag}.{ext}",
                )
            )

    # Heatmap across compartments x features using g
    # Keep it readable: use all features if <= 30 else top 30 by global |g|
    feat_order = (
        out.groupby("feature")["abs_g"].max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if len(feat_order) > 30:
        feat_order = feat_order[:30]

    comps = sorted(out["component"].unique().tolist())
    if model_name == "hybrid" and hybrid_heatmap_allowed_labels is not None:
        comps = [
            k for k in comps
            if component_label_map.get(int(k), f"comp{k}") in hybrid_heatmap_allowed_labels
        ]
    M = np.full((len(comps), len(feat_order)), np.nan, dtype=float)
    for i, k in enumerate(comps):
        dk = out[out["component"] == k].set_index("feature")
        for j, f in enumerate(feat_order):
            if f in dk.index:
                M[i, j] = float(dk.loc[f, "hedges_g"])

    # Remove all-empty rows/columns so heatmaps only display informative entries.
    keep_rows = np.isfinite(M).any(axis=1)
    keep_cols = np.isfinite(M).any(axis=0)
    M_plot = M[np.ix_(keep_rows, keep_cols)]
    comps_plot = [comps[i] for i in range(len(comps)) if keep_rows[i]]
    feat_plot = [feat_order[j] for j in range(len(feat_order)) if keep_cols[j]]

    if M_plot.size > 0 and np.isfinite(M_plot).any():
        vals = M_plot[np.isfinite(M_plot)]
        vmax = float(np.nanquantile(np.abs(vals), 0.98))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(np.nanmax(np.abs(vals))) if vals.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        cmap = HEATMAP_CMAP.copy()
        cmap.set_bad("#e6e6e6")
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        plt.figure(figsize=(max(9, 0.45 * len(feat_plot)), max(4, 0.45 * len(comps_plot))))
        im = plt.imshow(M_plot, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    else:
        # No finite values remain after filtering; skip plotting this heatmap.
        im = None
    if im is not None:
        plt.yticks(
            np.arange(len(comps_plot)),
            [component_label_map.get(int(k), f"comp{k}") for k in comps_plot],
        )
        plt.xticks(np.arange(len(feat_plot)), feat_plot, rotation=45, ha="right")
        cbar = plt.colorbar(im, label="Hedges g")
        cbar.ax.tick_params(labelsize=8)
        plt.title(f"{model_name.upper()} effect size heatmap (top features) — {tag}")

        heatmap_suffix = "component_labels"
        if model_name == "hybrid" and hybrid_heatmap_allowed_labels is not None:
            heatmap_suffix = "component_labels_splitfiltered"
        for ext in ("png", "pdf", "svg"):
            save_fig(
                os.path.join(
                    plots_dir,
                    f"heatmap_effectsize_{model_name}_{tag}_{heatmap_suffix}.{ext}",
                )
            )

    # Convenience “toprank” table per compartment
    top_rows = []
    for k in comps:
        dk = out[out["component"] == k].copy()
        dk = dk[np.isfinite(dk["hedges_g"].to_numpy(dtype=float))].copy()
        if dk.empty:
            continue
        # keep top |g| 2*top_n_each_side
        dk = dk.sort_values("abs_g", ascending=False).head(2 * cfg.top_n_each_side)
        dk["rank_within_component_by_abs_g"] = np.arange(1, dk.shape[0] + 1)
        top_rows.append(dk)
    if top_rows:
        top_df = pd.concat(top_rows, axis=0)
        top_df.to_csv(os.path.join(tables_dir, f"associations_{model_name}_{tag}_toprank.csv"), index=False)

    return out


# -----------------------------
# O2 vs GMM quick cross-tab
# -----------------------------

def o2_vs_component_crosstab(df: pd.DataFrame, tables_dir: str, model_name: str) -> None:
    if "O2_compartment" not in df.columns or "component" not in df.columns:
        return
    tab = (
        df.dropna(subset=["O2_compartment", "component"])
        .groupby(["O2_compartment", "component"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    # normalize by O2 row
    tot = tab.groupby("O2_compartment")["n"].transform("sum")
    tab["frac_within_o2"] = tab["n"] / tot
    if "component_label" in df.columns:
        label_map = (
            df.loc[:, ["component", "component_label"]]
            .dropna(subset=["component", "component_label"])
            .drop_duplicates(subset=["component"])
            .copy()
        )
        if not label_map.empty:
            label_map["component"] = pd.to_numeric(label_map["component"], errors="coerce")
            label_map = label_map.dropna(subset=["component"])
            label_map["component"] = label_map["component"].astype(int)
            tab["component"] = pd.to_numeric(tab["component"], errors="coerce").astype("Int64")
            tab = tab.merge(label_map, on="component", how="left")
    out_path = os.path.join(tables_dir, f"o2_compartment_confusion_vs_{model_name}.csv")
    tab.to_csv(out_path, index=False)
    if model_name == "gmm":
        # Backward-compatible filename used in earlier notebooks.
        tab.to_csv(os.path.join(tables_dir, "o2_compartment_confusion_vs_gmm.csv"), index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)
    assignment_jobs = resolve_assignment_jobs(cfg)
    hybrid_heatmap_allowed_labels = load_hybrid_heatmap_allowed_labels(cfg)
    if hybrid_heatmap_allowed_labels is not None:
        pd.DataFrame(
            {"component_label_allowed": sorted(hybrid_heatmap_allowed_labels)}
        ).to_csv(
            os.path.join(tables_dir, "hybrid_heatmap_allowed_labels.csv"),
            index=False,
        )

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        payload = dict(cfg.__dict__)
        payload["assignment_jobs"] = assignment_jobs
        json.dump(payload, f, indent=2)

    # Load matrix once (shared across models)
    df_matrix = read_csv_keep_first_duplicate_cols(cfg.matrix_cleaned, cfg.sep_matrix)
    # Minimal validation on matrix
    for col in (cfg.id_col, cfg.cruise_col):
        if col not in df_matrix.columns:
            raise ValueError(f"Matrix file missing required column: {col}")

    feature_cols: List[str] | None = None
    summary_rows: List[Dict[str, object]] = []
    all_raw: List[pd.DataFrame] = []
    all_depth: List[pd.DataFrame] = []

    for model_name, assign_path in assignment_jobs:
        df_assign = read_csv_keep_first_duplicate_cols(assign_path, cfg.sep_assign)

        for col in (cfg.id_col, cfg.cruise_col):
            if col not in df_assign.columns:
                raise ValueError(f"[{model_name}] assignments file missing required column: {col}")
        if "component" not in df_assign.columns:
            raise ValueError(f"[{model_name}] assignments file must contain 'component' column (hard label).")

        resp_cols = infer_resp_cols(df_assign)
        if not resp_cols:
            raise ValueError(
                f"[{model_name}] no resp_* columns found in assignments file. "
                "Expected resp_0..resp_{K-1}."
            )

        if feature_cols is None:
            feature_cols = choose_feature_cols(cfg, df_matrix, df_assign)
            if not feature_cols:
                raise ValueError(
                    "No feature columns selected. Provide --feature-cols explicitly "
                    "or ensure matrix_cleaned has numeric feature columns."
                )

        meta_cols = [
            c
            for c in [
                cfg.cruise_col,
                "Year",
                "Month",
                "Day",
                cfg.depth_col,
                cfg.depth_anchored_col,
                cfg.date_col,
                cfg.oxygen_col,
            ]
            if c in df_matrix.columns
        ]
        matrix_keep_cols = [cfg.id_col] + [c for c in meta_cols if c != cfg.id_col] + feature_cols
        matrix_keep_cols = [c for c in matrix_keep_cols if c in df_matrix.columns]

        # Dedupe again after selection (belt+suspenders)
        df_matrix_sub = df_matrix[matrix_keep_cols].loc[:, ~pd.Index(matrix_keep_cols).duplicated()].copy()
        m = df_assign.merge(df_matrix_sub, on=cfg.id_col, how="left", suffixes=("", "_matrix"))

        # Datetime
        if cfg.date_col in m.columns:
            m[cfg.date_col] = to_datetime_safe(m[cfg.date_col])

        # Add O2 compartment for contextual crosstab
        if cfg.oxygen_col in m.columns:
            o2 = pd.to_numeric(m[cfg.oxygen_col], errors="coerce").to_numpy(dtype=float)
            m["O2_compartment"] = [o2_compartment(v, cfg) for v in o2]
        else:
            m["O2_compartment"] = "NA"

        component_label_raw_map = infer_component_labels(m, component_col="component", label_col="compartment_name")
        component_label_map = build_component_display_map(
            m,
            model_name=model_name,
            component_col="component",
            raw_map=component_label_raw_map,
            n_resp_cols=len(resp_cols),
        )
        m["component_label"] = pd.to_numeric(m["component"], errors="coerce").map(component_label_map)
        m["component_label_raw"] = pd.to_numeric(m["component"], errors="coerce").map(component_label_raw_map)
        o2_vs_component_crosstab(m, tables_dir, model_name=model_name)

        raw_df = compute_associations(
            df=m,
            resp_cols=resp_cols,
            feature_cols=feature_cols,
            cfg=cfg,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            model_name=model_name,
            tag="raw",
            component_label_map=component_label_map,
            component_label_raw_map=component_label_raw_map,
            hybrid_heatmap_allowed_labels=hybrid_heatmap_allowed_labels,
        )
        all_raw.append(raw_df)

        if cfg.do_depth_adjust:
            if cfg.depth_anchored_col not in m.columns:
                raise ValueError(
                    f"[{model_name}] --depth-adjust requested, but missing depth anchored column "
                    f"'{cfg.depth_anchored_col}' in merged data."
                )

            m_adj = m.copy()
            for f in feature_cols:
                m_adj[f] = residualize_against_depth(m_adj[f], m_adj[cfg.depth_anchored_col])

            depth_df = compute_associations(
                df=m_adj,
                resp_cols=resp_cols,
                feature_cols=feature_cols,
                cfg=cfg,
                tables_dir=tables_dir,
                plots_dir=plots_dir,
                model_name=model_name,
                tag="depth_adjusted",
                component_label_map=component_label_map,
                component_label_raw_map=component_label_raw_map,
                hybrid_heatmap_allowed_labels=hybrid_heatmap_allowed_labels,
            )
            all_depth.append(depth_df)

        summary_rows.append(
            {
                "model": model_name,
                "assignments_path": assign_path,
                "k_from_resp": len(resp_cols),
                "n_rows_assignments": int(df_assign.shape[0]),
                "n_rows_merged": int(m.shape[0]),
            }
        )

    if all_raw:
        pd.concat(all_raw, axis=0, ignore_index=True).to_csv(
            os.path.join(tables_dir, "associations_all_raw.csv"),
            index=False,
        )
    if all_depth:
        pd.concat(all_depth, axis=0, ignore_index=True).to_csv(
            os.path.join(tables_dir, "associations_all_depth_adjusted.csv"),
            index=False,
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(tables_dir, "model_run_summary.csv"),
            index=False,
        )

    print("[OK] Done.")
    print(f"     Outdir : {cfg.outdir}")
    print(f"     Tables : {tables_dir}")
    print(f"     Plots  : {plots_dir}")
    print(f"     Models : {', '.join([m for m, _ in assignment_jobs])}")
    print(f"     Features analyzed: {len(feature_cols) if feature_cols is not None else 0}")
    if cfg.do_depth_adjust:
        print("     Depth-adjusted: YES (residualize vs Depth_anchored)")
    else:
        print("     Depth-adjusted: NO")


if __name__ == "__main__":
    main()
