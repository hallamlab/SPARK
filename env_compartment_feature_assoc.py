#!/usr/bin/env python3
"""
SPARK/env_compartment_feature_assoc.py

Purpose
-------
Given:
  1) matrix_cleaned.csv (biochem + metadata per sample)
  2) compartments_assignments_smoothed.csv (GMM assignments + responsibilities)

Compute compartment↔feature associations in a reviewer-defensible way:
  - Effect size (Hedges g) per compartment vs rest
  - Median/mean shifts + percent shift
  - Responsibility association: slope of resp_k ~ z(feature)
  - Cruise-blocked bootstrap CIs for all metrics (blocking unit = Cruise)
  - Optional depth-adjusted variant (residualize features vs Depth_anchored)

Outputs (in --outdir):
  tables/
    associations_raw.csv
    associations_depth_adjusted.csv
    associations_raw_toprank.csv
    associations_depth_adjusted_toprank.csv
    o2_compartment_confusion_vs_gmm.csv
    run_config.json
  plots/
    lollipop_comp{k}_raw.(png|pdf|svg)
    lollipop_comp{k}_depthadj.(png|pdf|svg)
    heatmap_effectsize_raw.(png|pdf|svg)
    heatmap_effectsize_depthadj.(png|pdf|svg)

Expected columns
----------------
Assignments CSV (example):
  cruise_year_month_depth, Cruise, Year, Month, Day, Depth, date, Depth_anchored,
  component, max_prob, entropy_norm, knn_mean_dist, resp_0..resp_{K-1}

Matrix CSV (example):
  cruise_year_month_depth, Cruise, Year, Month, Day, Depth, date, Depth_anchored,
  Oxygen, Nitrogen Oxides, ...

Notes
-----
- Handles duplicate columns by keeping the first occurrence (prevents merge errors).
- Uses matplotlib only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Constants / defaults
# -----------------------------

O2_PALETTE = {
    "oxic": "red",
    "dysoxic": "lightblue",
    "suboxic": "green",
    "anoxic": "purple",
}

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
    assignments: str
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


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Compartment↔feature associations with cruise-blocked bootstrap (lollipops + tables)."
    )

    ap.add_argument("--matrix-cleaned", required=True, help="Path to PCA matrix_cleaned.csv")
    ap.add_argument("--assignments", required=True, help="Path to compartments_assignments_smoothed.csv")
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


def compute_associations(
    df: pd.DataFrame,
    resp_cols: List[str],
    feature_cols: List[str],
    cfg: Config,
    tables_dir: str,
    plots_dir: str,
    tag: str,
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
    k_max = int(np.nanmax(comp.to_numpy(dtype=float))) if comp.notna().any() else -1

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
                "tag": tag,
                "component": int(k),
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

    out.to_csv(os.path.join(tables_dir, f"associations_{tag}.csv"), index=False)

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
        plt.title(f"Compartment {k} feature associations ({tag})")

        for ext in ("png", "pdf", "svg"):
            save_fig(os.path.join(plots_dir, f"lollipop_comp{k}_{tag}.{ext}"))

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
    M = np.full((len(comps), len(feat_order)), np.nan, dtype=float)
    for i, k in enumerate(comps):
        dk = out[out["component"] == k].set_index("feature")
        for j, f in enumerate(feat_order):
            if f in dk.index:
                M[i, j] = float(dk.loc[f, "hedges_g"])

    plt.figure(figsize=(max(9, 0.45 * len(feat_order)), max(4, 0.45 * len(comps))))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(comps)), [f"comp{k}" for k in comps])
    plt.xticks(np.arange(len(feat_order)), feat_order, rotation=45, ha="right")
    plt.colorbar(label="Hedges g")
    plt.title(f"Effect size heatmap (top features) — {tag}")

    for ext in ("png", "pdf", "svg"):
        save_fig(os.path.join(plots_dir, f"heatmap_effectsize_{tag}.{ext}"))

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
        top_df.to_csv(os.path.join(tables_dir, f"associations_{tag}_toprank.csv"), index=False)

    return out


# -----------------------------
# O2 vs GMM quick cross-tab
# -----------------------------

def o2_vs_gmm_crosstab(df: pd.DataFrame, tables_dir: str) -> None:
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
    tab.to_csv(os.path.join(tables_dir, "o2_compartment_confusion_vs_gmm.csv"), index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Load
    df_assign = read_csv_keep_first_duplicate_cols(cfg.assignments, cfg.sep_assign)
    df_matrix = read_csv_keep_first_duplicate_cols(cfg.matrix_cleaned, cfg.sep_matrix)

    # Minimal validation
    for col in (cfg.id_col, cfg.cruise_col):
        if col not in df_assign.columns:
            raise ValueError(f"Assignments file missing required column: {col}")
        if col not in df_matrix.columns:
            raise ValueError(f"Matrix file missing required column: {col}")

    # Parse resp columns
    resp_cols = infer_resp_cols(df_assign)
    if not resp_cols:
        raise ValueError("No resp_* columns found in assignments file. Expected resp_0..resp_{K-1}.")

    # Merge (assignments is the authoritative per-sample label/responsibilities)
    # Keep matrix columns we need + selected feature cols
    feature_cols = choose_feature_cols(cfg, df_matrix, df_assign)
    if not feature_cols:
        raise ValueError(
            "No feature columns selected. Provide --feature-cols explicitly "
            "or ensure matrix_cleaned has numeric feature columns."
        )

    meta_cols = [c for c in [cfg.cruise_col, "Year", "Month", "Day", cfg.depth_col, cfg.depth_anchored_col, cfg.date_col, cfg.oxygen_col] if c in df_matrix.columns]
    matrix_keep_cols = [cfg.id_col] + [c for c in meta_cols if c != cfg.id_col] + feature_cols
    matrix_keep_cols = [c for c in matrix_keep_cols if c in df_matrix.columns]

    # Dedupe again after selection (belt+suspenders)
    df_matrix_sub = df_matrix[matrix_keep_cols].loc[:, ~pd.Index(matrix_keep_cols).duplicated()].copy()

    m = df_assign.merge(df_matrix_sub, on=cfg.id_col, how="left", suffixes=("", "_matrix"))

    # Ensure component exists
    if "component" not in m.columns:
        raise ValueError("Assignments file must contain 'component' column (hard label).")

    # Datetime
    if cfg.date_col in m.columns:
        m[cfg.date_col] = to_datetime_safe(m[cfg.date_col])

    # Add O2 compartment
    if cfg.oxygen_col in m.columns:
        o2 = pd.to_numeric(m[cfg.oxygen_col], errors="coerce").to_numpy(dtype=float)
        m["O2_compartment"] = [o2_compartment(v, cfg) for v in o2]
    else:
        m["O2_compartment"] = "NA"

    # Quick O2 vs GMM cross-tab
    o2_vs_gmm_crosstab(m, tables_dir)

    # RAW associations
    compute_associations(
        df=m,
        resp_cols=resp_cols,
        feature_cols=feature_cols,
        cfg=cfg,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        tag="raw",
    )

    # DEPTH-ADJUSTED associations (optional)
    if cfg.do_depth_adjust:
        if cfg.depth_anchored_col not in m.columns:
            raise ValueError(f"--depth-adjust requested, but missing depth anchored column '{cfg.depth_anchored_col}' in merged data.")

        m_adj = m.copy()
        for f in feature_cols:
            m_adj[f] = residualize_against_depth(m_adj[f], m_adj[cfg.depth_anchored_col])

        compute_associations(
            df=m_adj,
            resp_cols=resp_cols,
            feature_cols=feature_cols,
            cfg=cfg,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            tag="depth_adjusted",
        )

    print("[OK] Done.")
    print(f"     Outdir : {cfg.outdir}")
    print(f"     Tables : {tables_dir}")
    print(f"     Plots  : {plots_dir}")
    print(f"     K      : {len(resp_cols)} (from resp_* columns)")
    print(f"     Features analyzed: {len(feature_cols)}")
    if cfg.do_depth_adjust:
        print("     Depth-adjusted: YES (residualize vs Depth_anchored)")
    else:
        print("     Depth-adjusted: NO")


if __name__ == "__main__":
    main()
