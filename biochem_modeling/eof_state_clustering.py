#!/usr/bin/env python3
# ==============================================================================
# File: biochem_modeling/eof_state_clustering.py
#
# Purpose
# -------
# Cluster cruises in EOF PC score space to assign cruise "states" using GMM.
# Adds:
#   - soft membership probabilities (responsibilities)
#   - OPTIONAL sticky smoothing over time-ordered cruises (forward-backward)
#
# Outputs (always):
#   outdir/cruise_states_base.tsv
#   outdir/SELECTED_K.txt
#   outdir/tables/gmm_selected_assignments.tsv
#   outdir/tables/gmm_selected_component_summary.tsv
#
# Outputs (if --k auto):
#   outdir/model_selection.tsv                          (legacy path/name)
#   outdir/tables/gmm_model_selection_metrics.tsv
#   outdir/tables/gmm_k_selection_decision.tsv
#   outdir/plots/selectk_ic_bic_icl.png
#   outdir/plots/selectk_cv_loglik.png                 (if --cv-folds > 1)
#   outdir/plots/selectk_stability_ari.png
#   outdir/plots/selectk_cluster_sizes.png
#   outdir/plots/selectk_entropy.png
#
# Outputs (if --sticky-smoothing):
#   outdir/cruise_states_smoothed.tsv
#   outdir/sticky_smoothing_config.json
#
# Notes
# -----
# - This script assumes each row is a CRUISE (EOF scores by cruise).
# - Smoothing encourages temporal persistence but can hide abrupt real events.
#   Start conservative: sticky_prob ~ 0.90, apply_to=low_conf_only.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    plt = None
    _HAVE_MPL = False


# -----------------------------
# Args
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster cruises in EOF score space to assign states (GMM + optional sticky smoothing).")

    ap.add_argument("--scores", required=True, help="eof_eigenvectors_scores_by_cruise.csv (or similar).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default="\t", help="Delimiter (default: tab).")

    ap.add_argument("--pcs", default="1,2,3,4,5",
                    help="Comma-separated PC indices to use, e.g. '1,2,3,4,5'.")

    ap.add_argument("--k", default="auto",
                    help="Number of clusters. Integer like '4' or 'auto' for BIC-based selection.")

    ap.add_argument("--k-min", type=int, default=2, help="Min K for auto selection.")
    ap.add_argument("--k-max", type=int, default=10, help="Max K for auto selection.")

    ap.add_argument("--covariance-type", choices=["full", "tied", "diag", "spherical"], default="full",
                    help="GMM covariance type.")

    ap.add_argument("--seed", type=int, default=1, help="Random seed.")
    ap.add_argument("--n-init", type=int, default=10, help="GMM n_init (default 10).")
    ap.add_argument("--max-iter", type=int, default=500, help="GMM max_iter (default 500).")
    ap.add_argument("--reg-covar", type=float, default=1e-6, help="GMM reg_covar (default 1e-6).")
    ap.add_argument("--standardize-pc-space", action="store_true",
                    help="Standardize selected PC columns before GMM fits.")

    # --- SelectK-style robustness metrics (for --k auto) ---
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="K-fold CV for test log-likelihood (default 5). Set 1 to disable.")
    ap.add_argument("--stability-R", type=int, default=200,
                    help="Block-bootstrap replicates for stability ARI (default 200).")
    ap.add_argument("--stability-block-col", default="Cruise",
                    help="Column for block bootstrap stability (default Cruise).")
    ap.add_argument("--stability-oob-min", type=int, default=10,
                    help="Minimum OOB samples needed per stability replicate (default 10).")
    ap.add_argument("--stability-min-ari", type=float, default=0.70,
                    help="Median ARI threshold for feasible K (default 0.70).")
    ap.add_argument("--min-cluster-frac", type=float, default=0.04,
                    help="Minimum allowed fraction for the smallest component (default 0.04).")
    ap.add_argument("--select-by", choices=["icl", "bic", "cv"], default="icl",
                    help="Primary metric to select K among feasible values (default icl).")
    ap.add_argument("--select-delta", type=float, default=5.0,
                    help="Choose smallest K within delta of best metric (default 5).")
    ap.add_argument("--allow-no-feasible-fallback", action="store_true",
                    help="If no K passes hard constraints, fallback to unconstrained best by --select-by.")

    # --- Sticky smoothing (optional) ---
    ap.add_argument("--sticky-smoothing", action="store_true",
                    help="Apply sticky smoothing to responsibilities over time-ordered cruises.")

    ap.add_argument("--time-col", default="date",
                    help="Time column to order cruises for smoothing (default 'date'). "
                         "If missing, falls back to Year/Month/Day if available, else no smoothing order guarantee.")

    ap.add_argument("--block-col", default=None,
                    help="Optional block column for independent sequences (rare for EOF). "
                         "If provided, smoothing runs within each block separately.")

    ap.add_argument("--sticky-prob", type=float, default=0.90,
                    help="Self-transition probability for sticky smoothing (default 0.90).")

    ap.add_argument("--apply-to", choices=["all", "low_conf_only"], default="low_conf_only",
                    help="Apply smoothing to all cruises or only low-confidence cruises (default low_conf_only).")

    ap.add_argument("--lowconf-maxprob", type=float, default=0.80,
                    help="If apply-to=low_conf_only, smooth cruises where base max_prob < this (default 0.80).")

    return ap.parse_args()


def parse_pcs(pcs_str: str) -> List[str]:
    idx = [int(x.strip()) for x in pcs_str.split(",") if x.strip()]
    if not idx:
        raise ValueError("--pcs must specify at least one PC index.")
    return [f"PC{i}" for i in idx]


# -----------------------------
# Model selection
# -----------------------------

def fit_gmm(
    X: np.ndarray,
    k: int,
    cov_type: str,
    seed: int,
    n_init: int,
    max_iter: int,
    reg_covar: float,
) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type=str(cov_type),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
        reg_covar=float(reg_covar),
    )
    gmm.fit(X)
    return gmm


def responsibilities_entropy(resp: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    r = np.clip(resp, eps, 1.0)
    ent_total = -np.sum(r * np.log(r))
    ent_mean = ent_total / resp.shape[0]
    return float(ent_total), float(ent_mean)


def icl_from_bic_and_resp(bic: float, resp: np.ndarray) -> float:
    ent_total, _ = responsibilities_entropy(resp)
    return float(bic + 2.0 * ent_total)


def cv_test_loglik(
    X: np.ndarray,
    k: int,
    cov_type: str,
    seed: int,
    n_init: int,
    max_iter: int,
    reg_covar: float,
    cv_folds: int,
) -> Tuple[float, float]:
    kf = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
    vals: List[float] = []
    for train_idx, test_idx in kf.split(X):
        gmm = fit_gmm(
            X[train_idx],
            k=k,
            cov_type=cov_type,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
        )
        vals.append(float(gmm.score(X[test_idx])))
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def block_bootstrap_stability(
    X: np.ndarray,
    base_labels: np.ndarray,
    blocks: np.ndarray,
    k: int,
    cov_type: str,
    seed: int,
    n_init: int,
    max_iter: int,
    reg_covar: float,
    stability_R: int,
    stability_oob_min: int,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(blocks)
    block_to_idx: Dict[object, np.ndarray] = {b: np.where(blocks == b)[0] for b in uniq}

    aris: List[float] = []
    for _ in range(int(stability_R)):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sampled_set = set(sampled.tolist())
        oob_blocks = [b for b in uniq.tolist() if b not in sampled_set]
        if not oob_blocks:
            continue

        inbag_idx = np.concatenate([block_to_idx[b] for b in sampled], axis=0)
        oob_idx = np.concatenate([block_to_idx[b] for b in oob_blocks], axis=0)
        if oob_idx.shape[0] < int(stability_oob_min):
            continue

        gmm = fit_gmm(
            X[inbag_idx],
            k=k,
            cov_type=cov_type,
            seed=seed,
            n_init=n_init,
            max_iter=max_iter,
            reg_covar=reg_covar,
        )
        oob_labels = gmm.predict(X[oob_idx])
        aris.append(float(adjusted_rand_score(base_labels[oob_idx], oob_labels)))

    if not aris:
        return np.nan, np.nan, 0
    return float(np.median(aris)), float(np.mean(aris)), int(len(aris))


def _fail_reasons_row(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("passes_min_cluster", False)):
        reasons.append("min_cluster_frac")
    if not bool(row.get("passes_stability", False)):
        reasons.append("stability")
    if pd.isna(row.get("stability_median_ARI", np.nan)) or int(row.get("stability_n_reps", 0)) <= 0:
        if "stability" not in reasons:
            reasons.append("stability_nans_or_no_reps")
    return ",".join(reasons) if reasons else ""


def select_k(metrics: pd.DataFrame, args: argparse.Namespace) -> Tuple[int, pd.DataFrame]:
    m = metrics.copy()
    m["passes_min_cluster"] = m["min_cluster_frac"] >= float(args.min_cluster_frac)
    m["passes_stability"] = m["stability_median_ARI"] >= float(args.stability_min_ari)
    m.loc[m["stability_n_reps"].fillna(0).astype(int) <= 0, "passes_stability"] = False
    m["feasible"] = m["passes_min_cluster"] & m["passes_stability"]
    m["FAIL_REASONS"] = m.apply(_fail_reasons_row, axis=1)

    feasible = m[m["feasible"]].copy()
    if feasible.empty and not bool(args.allow_no_feasible_fallback):
        m["within_delta"] = False
        m["SELECTED"] = False
        decision_cols = [
            "K", "AIC", "BIC", "ICL", "mean_resp_entropy",
            "min_cluster_frac",
            "CV_loglik_mean", "CV_loglik_std",
            "stability_median_ARI", "stability_mean_ARI", "stability_n_reps",
            "passes_min_cluster", "passes_stability", "feasible", "within_delta",
            "FAIL_REASONS", "SELECTED",
        ]
        decision_df = m[[c for c in decision_cols if c in m.columns]].copy()
        return -1, decision_df

    pool = feasible if not feasible.empty else m.copy()
    if str(args.select_by) == "icl":
        best = float(pool.loc[pool["ICL"].idxmin(), "ICL"])
        m["within_delta"] = m["feasible"] & ((m["ICL"] - best) <= float(args.select_delta))
        if feasible.empty:
            m["within_delta"] = (m["ICL"] - best) <= float(args.select_delta)
    elif str(args.select_by) == "bic":
        best = float(pool.loc[pool["BIC"].idxmin(), "BIC"])
        m["within_delta"] = m["feasible"] & ((m["BIC"] - best) <= float(args.select_delta))
        if feasible.empty:
            m["within_delta"] = (m["BIC"] - best) <= float(args.select_delta)
    else:
        if int(args.cv_folds) <= 1:
            raise ValueError("--select-by cv requires --cv-folds > 1")
        best = float(pool["CV_loglik_mean"].max())
        m["within_delta"] = m["feasible"] & ((best - m["CV_loglik_mean"]) <= float(args.select_delta))
        if feasible.empty:
            m["within_delta"] = (best - m["CV_loglik_mean"]) <= float(args.select_delta)

    eligible = m[m["within_delta"]].copy()
    chosen = int(eligible.sort_values(["K"]).iloc[0]["K"])

    decision_cols = [
        "K", "AIC", "BIC", "ICL", "mean_resp_entropy",
        "min_cluster_frac",
        "CV_loglik_mean", "CV_loglik_std",
        "stability_median_ARI", "stability_mean_ARI", "stability_n_reps",
        "passes_min_cluster", "passes_stability", "feasible", "within_delta",
        "FAIL_REASONS",
    ]
    decision_df = m[[c for c in decision_cols if c in m.columns]].copy()
    decision_df["SELECTED"] = decision_df["K"] == chosen
    return chosen, decision_df


def save_fig(path: str) -> None:
    if not _HAVE_MPL:
        return
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_ic(metrics: pd.DataFrame, outpath: str) -> None:
    if not _HAVE_MPL:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["BIC"], marker="o", label="BIC")
    plt.plot(metrics["K"], metrics["ICL"], marker="o", label="ICL")
    plt.xlabel("K")
    plt.ylabel("Information criterion (lower is better)")
    plt.title("EOF state selection: BIC vs ICL")
    plt.legend()
    save_fig(outpath)


def plot_cv(metrics: pd.DataFrame, outpath: str) -> None:
    if not _HAVE_MPL:
        return
    if "CV_loglik_mean" not in metrics.columns:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["CV_loglik_mean"], marker="o")
    if "CV_loglik_std" in metrics.columns:
        y = metrics["CV_loglik_mean"].values
        s = metrics["CV_loglik_std"].values
        plt.fill_between(metrics["K"].values, y - s, y + s, alpha=0.2)
    plt.xlabel("K")
    plt.ylabel("CV test avg log-likelihood (higher is better)")
    plt.title("EOF state selection: CV log-likelihood vs K")
    save_fig(outpath)


def plot_stability(metrics: pd.DataFrame, outpath: str, threshold: float) -> None:
    if not _HAVE_MPL:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["stability_median_ARI"], marker="o")
    plt.axhline(y=threshold, linestyle="--", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("Median ARI (higher is better)")
    plt.title("EOF state selection: block-bootstrap stability vs K")
    save_fig(outpath)


def plot_min_cluster_frac(metrics: pd.DataFrame, outpath: str, threshold: float) -> None:
    if not _HAVE_MPL:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["min_cluster_frac"], marker="o")
    plt.axhline(y=threshold, linestyle="--", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("Minimum component fraction")
    plt.title("EOF state selection: smallest cluster fraction vs K")
    save_fig(outpath)


def plot_entropy(metrics: pd.DataFrame, outpath: str) -> None:
    if not _HAVE_MPL:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["mean_resp_entropy"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Mean responsibility entropy (lower = crisper)")
    plt.title("EOF state selection: assignment uncertainty vs K")
    save_fig(outpath)


# -----------------------------
# Audit helpers
# -----------------------------

def _coerce_sortable_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() >= max(3, int(0.5 * len(s))):
        return dt
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= max(3, int(0.5 * len(s))):
        return num
    return s.astype(str)


def _time_order_key(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().sum() > 0:
            return t

    if all(c in df.columns for c in ["Year", "Month", "Day"]):
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        d = pd.to_numeric(df["Day"], errors="coerce")
        t = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        if t.notna().sum() > 0:
            return t

    return pd.Series(np.arange(df.shape[0]), index=df.index)


def _stable_order_index(df: pd.DataFrame, time_col: str) -> np.ndarray:
    key = _time_order_key(df, time_col)
    key = _coerce_sortable_series(key)
    return key.sort_values(kind="mergesort").index.to_numpy()


def _audit_order(df: pd.DataFrame, time_col: str, block_col: Optional[str]) -> pd.DataFrame:
    work = df.copy()

    if block_col is None or block_col not in work.columns:
        work["_BLOCK_"] = "__ALL__"
        block_col_use = "_BLOCK_"
    else:
        block_col_use = block_col

    rows = []
    for lab, sub in work.groupby(block_col_use, sort=False):
        key = _time_order_key(sub, time_col)
        key2 = _coerce_sortable_series(key)

        k = key2.dropna()
        mono_inc = True
        mono_dec = True
        if len(k) > 2:
            mono_inc = bool(k.is_monotonic_increasing)
            mono_dec = bool(k.is_monotonic_decreasing)

        rows.append({
            "block_col": str(block_col_use),
            "block_label": str(lab),
            "n": int(sub.shape[0]),
            "time_col_used": str(time_col),
            "n_time_nonnull": int(pd.to_datetime(sub.get(time_col, pd.Series([pd.NaT]*len(sub))), errors="coerce").notna().sum()) if time_col in sub.columns else 0,
            "monotonic_increasing": mono_inc,
            "monotonic_decreasing": mono_dec,
            "time_min": str(k.min()) if len(k) else "",
            "time_max": str(k.max()) if len(k) else "",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["block_col", "block_label"]).reset_index(drop=True)
    return out


def _audit_transitions(
    df: pd.DataFrame,
    labels_0based: np.ndarray,
    time_col: str,
    block_col: Optional[str],
) -> pd.DataFrame:
    work = df.copy()
    work["_label_"] = labels_0based.astype(int)

    if block_col is None or block_col not in work.columns:
        work["_BLOCK_"] = "__ALL__"
        block_col_use = "_BLOCK_"
    else:
        block_col_use = block_col

    rows = []
    for lab, sub in work.groupby(block_col_use, sort=False):
        order = _stable_order_index(sub, time_col)
        y = sub.loc[order, "_label_"].to_numpy()
        if len(y) < 2:
            continue
        for a, b in zip(y[:-1], y[1:]):
            rows.append({
                "block_label": str(lab),
                "from_state": int(a) + 1,
                "to_state": int(b) + 1,
                "n": 1,
            })

    if not rows:
        return pd.DataFrame(columns=["block_label", "from_state", "to_state", "n"])

    out = (
        pd.DataFrame(rows)
        .groupby(["block_label", "from_state", "to_state"], as_index=False)["n"]
        .sum()
        .sort_values(["block_label", "from_state", "to_state"])
        .reset_index(drop=True)
    )
    return out


def _audit_smoothing_effect(
    df: pd.DataFrame,
    cruise_col: str,
    probs_base: np.ndarray,
    probs_sm: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_state = np.argmax(probs_base, axis=1) + 1
    sm_state = np.argmax(probs_sm, axis=1) + 1

    base_max = probs_base.max(axis=1)
    sm_max = probs_sm.max(axis=1)

    flipped = base_state != sm_state
    delta = sm_max - base_max

    per = pd.DataFrame({
        cruise_col: df[cruise_col].astype(str).values,
        "base_state": base_state.astype(int),
        "smoothed_state": sm_state.astype(int),
        "label_flipped": flipped.astype(bool),
        "base_max_prob": base_max.astype(float),
        "smoothed_max_prob": sm_max.astype(float),
        "delta_max_prob": delta.astype(float),
    })

    summ = pd.DataFrame([{
        "n": int(len(base_state)),
        "n_label_flipped": int(flipped.sum()),
        "frac_label_flipped": float(flipped.mean()) if len(flipped) else np.nan,
        "median_delta_max_prob": float(np.median(delta)) if len(delta) else np.nan,
        "mean_delta_max_prob": float(np.mean(delta)) if len(delta) else np.nan,
        "median_base_max_prob": float(np.median(base_max)) if len(base_max) else np.nan,
        "median_smoothed_max_prob": float(np.median(sm_max)) if len(sm_max) else np.nan,
    }])

    return per, summ


# -----------------------------
# Sticky smoothing (forward-backward over responsibilities)
# -----------------------------

def sticky_transition(K: int, sticky_prob: float) -> np.ndarray:
    sp = float(sticky_prob)
    sp = min(max(sp, 0.0), 1.0)
    if K <= 1:
        return np.ones((1, 1), dtype=float)
    off = (1.0 - sp) / (K - 1)
    T = np.full((K, K), off, dtype=float)
    np.fill_diagonal(T, sp)
    return T


def forward_backward_emission(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    E: (n, K) emission probs per time step (here: GMM responsibilities).
    Returns gamma: (n, K) smoothed posteriors.
    """
    eps = 1e-12
    E = np.clip(E, eps, 1.0)
    E = E / E.sum(axis=1, keepdims=True)

    K = E.shape[1]
    n = E.shape[0]
    pi0 = np.ones(K, dtype=float) / K

    alpha = np.zeros((n, K), dtype=float)
    c = np.zeros(n, dtype=float)

    alpha[0, :] = pi0 * E[0, :]
    c[0] = alpha[0, :].sum()
    alpha[0, :] /= (c[0] if c[0] > 0 else 1.0)

    for t in range(1, n):
        alpha[t, :] = (alpha[t - 1, :].dot(T)) * E[t, :]
        c[t] = alpha[t, :].sum()
        alpha[t, :] /= (c[t] if c[t] > 0 else 1.0)

    beta = np.zeros((n, K), dtype=float)
    beta[-1, :] = 1.0

    for t in range(n - 2, -1, -1):
        beta[t, :] = (T.dot(E[t + 1, :] * beta[t + 1, :]))
        beta[t, :] /= (c[t + 1] if c[t + 1] > 0 else 1.0)

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def selective_replace(base: np.ndarray, smoothed: np.ndarray, apply_to: str, thr: float) -> np.ndarray:
    if apply_to == "all":
        out = smoothed
    else:
        maxp = base.max(axis=1)
        mask = maxp < float(thr)
        out = base.copy()
        out[mask, :] = smoothed[mask, :]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _coerce_time_order(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Build a sortable time key.
    Preference:
      1) pd.to_datetime(df[time_col])
      2) Year/Month/Day if present
      3) index order (fallback)
    """
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        if t.notna().sum() > 0:
            return t

    # fallback: Year/Month/Day
    if all(c in df.columns for c in ["Year", "Month", "Day"]):
        y = pd.to_numeric(df["Year"], errors="coerce")
        m = pd.to_numeric(df["Month"], errors="coerce")
        d = pd.to_numeric(df["Day"], errors="coerce")
        t = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
        if t.notna().sum() > 0:
            return t

    # fallback: preserve original order
    return pd.Series(np.arange(df.shape[0]), index=df.index)


def apply_sticky_smoothing_cruises(
    df: pd.DataFrame,
    probs: np.ndarray,
    time_col: str,
    sticky_prob: float,
    apply_to: str,
    lowconf_maxprob: float,
    block_col: Optional[str] = None,
) -> np.ndarray:
    """
    Smoothing over cruises ordered by time. If block_col provided and exists, smooth per-block.
    """
    K = probs.shape[1]
    T = sticky_transition(K, sticky_prob)
    out = probs.copy()

    if block_col is None or block_col not in df.columns:
        order_key = _coerce_time_order(df, time_col)
        order = order_key.sort_values(kind="mergesort").index.to_numpy()
        gamma = forward_backward_emission(T, probs[order, :])
        out[order, :] = selective_replace(probs[order, :], gamma, apply_to, lowconf_maxprob)
        return out

    # per-block smoothing
    for _, sub in df.groupby(block_col, sort=False):
        order_key = _coerce_time_order(sub, time_col)
        order = order_key.sort_values(kind="mergesort").index.to_numpy()
        gamma = forward_backward_emission(T, probs[order, :])
        out[order, :] = selective_replace(probs[order, :], gamma, apply_to, lowconf_maxprob)

    return out


# -----------------------------
# Output helpers
# -----------------------------

def write_states(
    outpath: str,
    df_meta: pd.DataFrame,
    cruise_col: str,
    probs: np.ndarray,
    labels_0based: np.ndarray,
) -> None:
    max_prob = probs.max(axis=1)
    out = pd.DataFrame({cruise_col: df_meta[cruise_col].astype(str), "state": labels_0based.astype(int) + 1, "max_prob": max_prob})
    for j in range(probs.shape[1]):
        out[f"p_state{j+1}"] = probs[:, j]
    out.to_csv(outpath, sep="\t", index=False)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    os.makedirs(args.outdir, exist_ok=True)
    tables_dir = os.path.join(args.outdir, "tables")
    plots_dir = os.path.join(args.outdir, "plots")
    audit_dir = os.path.join(args.outdir, "audit")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)

    with open(os.path.join(args.outdir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    df = pd.read_csv(args.scores, sep=args.sep)
    if df.shape[1] < 2:
        raise ValueError("Scores file looks wrong; expected cruise_id + PC columns.")

    # Assume first column is cruise id (matches your earlier script behavior)
    cruise_col = df.columns[0]
    df[cruise_col] = df[cruise_col].astype(str)

    pcs = parse_pcs(args.pcs)
    missing = [c for c in pcs if c not in df.columns]
    if missing:
        raise ValueError(f"Requested PCs not found in scores file: {missing}")

    X_df = df[pcs].apply(pd.to_numeric, errors="coerce")
    keep_rows = ~X_df.isna().any(axis=1)
    df = df.loc[keep_rows].reset_index(drop=True)
    X = X_df.loc[keep_rows].to_numpy(dtype=float)

    if bool(args.standardize_pc_space):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if X.shape[0] < 3:
        raise ValueError("Not enough valid rows after dropping missing PC values.")

    if str(args.stability_block_col) in df.columns:
        blocks = df[str(args.stability_block_col)].astype(str).fillna("NA").to_numpy()
    else:
        blocks = np.array(["ALL"] * df.shape[0], dtype=object)

    # Fit model
    if args.k == "auto":
        rows = []
        for k in range(int(args.k_min), int(args.k_max) + 1):
            gmm_k = fit_gmm(
                X,
                k=k,
                cov_type=str(args.covariance_type),
                seed=int(args.seed),
                n_init=int(args.n_init),
                max_iter=int(args.max_iter),
                reg_covar=float(args.reg_covar),
            )
            bic = float(gmm_k.bic(X))
            aic = float(gmm_k.aic(X))
            resp = gmm_k.predict_proba(X)
            icl = icl_from_bic_and_resp(bic, resp)
            ent_total, ent_mean = responsibilities_entropy(resp)

            hard = gmm_k.predict(X)
            counts = np.bincount(hard, minlength=int(k)).astype(float)
            fracs = counts / counts.sum()
            min_frac = float(fracs.min()) if len(fracs) else np.nan

            cv_mean = np.nan
            cv_std = np.nan
            if int(args.cv_folds) > 1 and X.shape[0] >= int(args.cv_folds):
                try:
                    cv_mean, cv_std = cv_test_loglik(
                        X,
                        k=k,
                        cov_type=str(args.covariance_type),
                        seed=int(args.seed),
                        n_init=int(args.n_init),
                        max_iter=int(args.max_iter),
                        reg_covar=float(args.reg_covar),
                        cv_folds=int(args.cv_folds),
                    )
                except Exception:
                    cv_mean, cv_std = np.nan, np.nan

            med_ari, mean_ari, nrep = block_bootstrap_stability(
                X,
                base_labels=hard,
                blocks=blocks,
                k=k,
                cov_type=str(args.covariance_type),
                seed=int(args.seed),
                n_init=int(args.n_init),
                max_iter=int(args.max_iter),
                reg_covar=float(args.reg_covar),
                stability_R=int(args.stability_R),
                stability_oob_min=int(args.stability_oob_min),
            )

            rows.append(
                {
                    "K": int(k),
                    "AIC": aic,
                    "BIC": bic,
                    "ICL": icl,
                    "entropy_total": ent_total,
                    "mean_resp_entropy": ent_mean,
                    "min_cluster_frac": min_frac,
                    "CV_loglik_mean": cv_mean,
                    "CV_loglik_std": cv_std,
                    "stability_median_ARI": med_ari,
                    "stability_mean_ARI": mean_ari,
                    "stability_n_reps": int(nrep),
                }
            )

        metrics = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
        metrics.to_csv(os.path.join(tables_dir, "gmm_model_selection_metrics.tsv"), sep="\t", index=False)
        # Backward-compatible location/name:
        metrics.to_csv(os.path.join(args.outdir, "model_selection.tsv"), sep="\t", index=False)

        plot_ic(metrics, os.path.join(plots_dir, "selectk_ic_bic_icl.png"))
        if int(args.cv_folds) > 1:
            plot_cv(metrics, os.path.join(plots_dir, "selectk_cv_loglik.png"))
        plot_stability(
            metrics,
            os.path.join(plots_dir, "selectk_stability_ari.png"),
            threshold=float(args.stability_min_ari),
        )
        plot_min_cluster_frac(
            metrics,
            os.path.join(plots_dir, "selectk_cluster_sizes.png"),
            threshold=float(args.min_cluster_frac),
        )
        plot_entropy(metrics, os.path.join(plots_dir, "selectk_entropy.png"))
        if not _HAVE_MPL:
            print("[WARN] matplotlib unavailable; selectk plots were skipped.")

        k_used, decision = select_k(metrics, args)
        decision.to_csv(os.path.join(tables_dir, "gmm_k_selection_decision.tsv"), sep="\t", index=False)
        if int(k_used) < 0:
            raise SystemExit(
                "[STOP] No feasible K satisfied hard constraints; "
                "see tables/gmm_k_selection_decision.tsv for failure reasons."
            )

        gmm = fit_gmm(
            X,
            k=int(k_used),
            cov_type=str(args.covariance_type),
            seed=int(args.seed),
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
            reg_covar=float(args.reg_covar),
        )
    else:
        k_used = int(args.k)
        gmm = fit_gmm(
            X,
            k=int(k_used),
            cov_type=str(args.covariance_type),
            seed=int(args.seed),
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
            reg_covar=float(args.reg_covar),
        )

    # Base responsibilities + assignments
    probs_base = gmm.predict_proba(X)
    labels_base = gmm.predict(X)  # 0..K-1

    # Always write base
    write_states(
        outpath=os.path.join(args.outdir, "cruise_states_base.tsv"),
        df_meta=df,
        cruise_col=cruise_col,
        probs=probs_base,
        labels_0based=labels_base,
    )

    # SelectK-style assignment table + component summary
    assign = df.copy()
    assign["component"] = labels_base.astype(int)
    assign["max_prob"] = probs_base.max(axis=1).astype(float)
    assign["resp_entropy"] = (
        -np.sum(np.clip(probs_base, 1e-12, 1.0) * np.log(np.clip(probs_base, 1e-12, 1.0)), axis=1)
    ).astype(float)
    for j in range(probs_base.shape[1]):
        assign[f"resp_{j}"] = probs_base[:, j]
    assign.to_csv(os.path.join(tables_dir, "gmm_selected_assignments.tsv"), sep="\t", index=False)

    comp_rows = []
    for c in range(int(k_used)):
        idx = np.where(labels_base == c)[0]
        comp_rows.append(
            {
                "component": int(c),
                "n_samples": int(len(idx)),
                "frac_samples": float(len(idx) / len(labels_base)),
                "median_max_prob": float(np.median(assign.loc[idx, "max_prob"])) if len(idx) else np.nan,
                "median_entropy": float(np.median(assign.loc[idx, "resp_entropy"])) if len(idx) else np.nan,
            }
        )
    pd.DataFrame(comp_rows).sort_values(["n_samples"], ascending=False).to_csv(
        os.path.join(tables_dir, "gmm_selected_component_summary.tsv"),
        sep="\t",
        index=False,
    )

    with open(os.path.join(args.outdir, "SELECTED_K.txt"), "w") as f:
        f.write(f"{int(k_used)}\n")

    ord_base = _audit_order(df, time_col=str(args.time_col), block_col=args.block_col if args.block_col else None)
    ord_base.to_csv(os.path.join(audit_dir, "audit_order_base.tsv"), sep="\t", index=False)

    tr_base = _audit_transitions(df, labels_base, time_col=str(args.time_col), block_col=args.block_col if args.block_col else None)
    tr_base.to_csv(os.path.join(audit_dir, "audit_transitions_base.tsv"), sep="\t", index=False)

    # Optional sticky smoothing
    if args.sticky_smoothing:
        probs_sm = apply_sticky_smoothing_cruises(
            df=df,
            probs=probs_base,
            time_col=str(args.time_col),
            sticky_prob=float(args.sticky_prob),
            apply_to=str(args.apply_to),
            lowconf_maxprob=float(args.lowconf_maxprob),
            block_col=args.block_col if args.block_col else None,
        )
        labels_sm = np.argmax(probs_sm, axis=1)

        write_states(
            outpath=os.path.join(args.outdir, "cruise_states_smoothed.tsv"),
            df_meta=df,
            cruise_col=cruise_col,
            probs=probs_sm,
            labels_0based=labels_sm,
        )

        ord_sm = _audit_order(df, time_col=str(args.time_col), block_col=args.block_col if args.block_col else None)
        ord_sm.to_csv(os.path.join(audit_dir, "audit_order_smoothed.tsv"), sep="\t", index=False)

        tr_sm = _audit_transitions(df, labels_sm, time_col=str(args.time_col), block_col=args.block_col if args.block_col else None)
        tr_sm.to_csv(os.path.join(audit_dir, "audit_transitions_smoothed.tsv"), sep="\t", index=False)

        per, summ = _audit_smoothing_effect(df, cruise_col=cruise_col, probs_base=probs_base, probs_sm=probs_sm)
        per.to_csv(os.path.join(audit_dir, "audit_smoothing_effect.tsv"), sep="\t", index=False)
        summ.to_csv(os.path.join(audit_dir, "audit_smoothing_effect_summary.tsv"), sep="\t", index=False)

        with open(os.path.join(args.outdir, "sticky_smoothing_config.json"), "w") as f:
            json.dump(
                {
                    "time_col": args.time_col,
                    "block_col": args.block_col,
                    "sticky_prob": float(args.sticky_prob),
                    "apply_to": args.apply_to,
                    "lowconf_maxprob": float(args.lowconf_maxprob),
                    "note": "Smoothing is forward-backward over GMM responsibilities ordered by time.",
                },
                f,
                indent=2,
            )

    print(f"Used K = {k_used}")
    print(f"Wrote: {os.path.join(args.outdir, 'cruise_states_base.tsv')}")
    if args.k == "auto":
        print(f"Wrote: {os.path.join(args.outdir, 'model_selection.tsv')}")
        print(f"Wrote: {os.path.join(tables_dir, 'gmm_model_selection_metrics.tsv')}")
        print(f"Wrote: {os.path.join(tables_dir, 'gmm_k_selection_decision.tsv')}")
    if args.sticky_smoothing:
        print(f"Wrote: {os.path.join(args.outdir, 'cruise_states_smoothed.tsv')}")


if __name__ == "__main__":
    main()
