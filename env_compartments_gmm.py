# file: env_compartments_final_gmm.py
# put this in: /media/nfs/Ryan/SABer/SI_data/SI_ASV/SPARK/env_compartments_final_gmm.py
#!/usr/bin/env python3
"""
env_compartments_final_gmm.py

Purpose
-------
Given PCA eigengene scores (PC1..PCn per sample) and a PC keep/decision table,
fit a FINAL Gaussian Mixture Model (GMM) at a fixed K (default 5) and produce:

1) Reproducible model artifacts
   - model_params.json (weights/means/covariances/etc)
   - run_config.json

2) Per-sample compartment assignments with uncertainty
   - compartments_assignments_base.csv
   - responsibilities_base.csv
   (optionally)
   - compartments_assignments_smoothed.csv
   - responsibilities_smoothed.csv

3) Compartment summaries
   - compartment_summary_base.csv
   - compartment_separation_base.csv
   - covariance_diagnostics_base.csv
   - persistent_episodic_labels_base.csv
   (and *_smoothed.csv if smoothing enabled)

4) Plots (matplotlib)
   - pc_scatter_PC1_PC2_base.png
   - compartment_frequency_over_time_base.png
   - depth_time_heatmaps_component_{k}_base.png
   (and *_smoothed.png if smoothing enabled)

Optional biological inference helpers
------------------------------------
If you provide --matrix-cleaned (from env_pca/tables/matrix_cleaned.csv),
the script also computes:
   - pc_feature_correlations_spearman.csv
   - compartment_feature_shifts.csv  (simple effect size vs global mean)

Notes on “episodic smoothing”
-----------------------------
This is *second-stage temporal regularization* applied AFTER base GMM responsibilities.
It does not change K. It can be applied to:
  - only low-confidence samples (default), or
  - all samples
within each block (e.g. Cruise), ordered by provided sort cols.

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

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Config
# -----------------------------

@dataclass
class RunConfig:
    eigengenes_csv: str
    pc_keep_csv: str
    outdir: str
    sep: str

    # columns
    time_col: str
    cruise_col: str
    depth_col: str
    anchored_depth_col: str

    # PC selection
    pc_use_mode: str          # keep|all|explicit
    pc_explicit: Optional[List[str]]

    # preprocessing
    standardize_pc_space: bool
    depth_norm: str           # global
    depth_norm_col: str       # which depth col to use for depth feature
    density_space: str        # pc|pc_depthnorm

    # GMM
    K: int
    covariance_type: str
    n_init: int
    max_iter: int
    random_state: int
    reg_covar: float

    # density / diagnostics
    knn_k: int

    # second stage smoothing
    episodic_smoothing: bool
    episodic_block_col: str
    episodic_sort_cols: List[str]
    episodic_sticky_prob: float
    episodic_apply_to: str        # all|low_conf_only
    episodic_lowconf_maxprob: float

    # labeling thresholds
    persistent_min_frac_samples: float
    persistent_min_n_cruises: int
    persistent_min_span_days: int
    episodic_max_frac_samples: float
    episodic_max_span_days: int

    # optional inference
    matrix_cleaned_csv: Optional[str]


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description="Fit final fixed-K GMM compartments from PCA eigengenes + PC keep table.")

    ap.add_argument("--eigengenes", required=True, help="Path to eigengenes_scores.csv (from env_pca/tables).")
    ap.add_argument("--pc-keep", required=True, help="Path to pc_keep_decision.csv (from env_pca/tables).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default=",", help="Delimiter for eigengenes/pc_keep (default ',').")

    ap.add_argument("--time-col", default="date", help="Time column name (default date).")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name (default Cruise).")
    ap.add_argument("--depth-col", default="Depth", help="Depth column name (default Depth).")
    ap.add_argument("--anchored-depth-col", default="Depth_anchored", help="Anchored depth column name (default Depth_anchored).")

    ap.add_argument("--pc-use-mode", choices=["keep", "all", "explicit"], default="keep",
                    help="Which PCs to use: keep|all|explicit (default keep).")
    ap.add_argument("--pc-explicit", default=None,
                    help="If pc-use-mode=explicit: comma-separated list like 'PC1,PC2,PC3'.")

    ap.add_argument("--standardize-pc-space", action="store_true",
                    help="Standardize selected PC columns before GMM (recommended).")

    ap.add_argument("--depth-norm", choices=["global"], default="global",
                    help="Depth normalization mode. Currently: global zscore of depth feature (default global).")
    ap.add_argument("--depth-norm-col", default=None,
                    help="Which depth column to use for depth feature in density_space=pc_depthnorm (default anchored depth col).")
    ap.add_argument("--density-space", choices=["pc", "pc_depthnorm"], default="pc_depthnorm",
                    help="Space used for density diagnostics (default pc_depthnorm).")

    ap.add_argument("--K", type=int, default=5, help="Final number of components (default 5).")
    ap.add_argument("--covariance-type", choices=["full", "tied", "diag", "spherical"], default="full",
                    help="GMM covariance type (default full).")
    ap.add_argument("--n-init", type=int, default=30, help="GMM n_init (default 30).")
    ap.add_argument("--max-iter", type=int, default=1000, help="GMM max_iter (default 1000).")
    ap.add_argument("--random-state", type=int, default=42, help="Random state (default 42).")
    ap.add_argument("--reg-covar", type=float, default=1e-6, help="Regularization added to diagonal of covariances (default 1e-6).")

    ap.add_argument("--knn-k", type=int, default=15, help="k for kNN density diagnostics (default 15).")

    # Second-stage episodic smoothing
    ap.add_argument("--episodic-smoothing", action="store_true",
                    help="Apply sticky smoothing within each block (e.g., Cruise) to respect episodic structure.")
    ap.add_argument("--episodic-block-col", default="Cruise", help="Block col for smoothing (default Cruise).")
    ap.add_argument("--episodic-sort-cols", default="date,Depth_anchored",
                    help="Within-block sort cols, comma-separated (default 'date,Depth_anchored').")
    ap.add_argument("--episodic-sticky-prob", type=float, default=0.98,
                    help="Self-transition probability in sticky smoothing (default 0.98).")
    ap.add_argument("--episodic-apply-to", choices=["all", "low_conf_only"], default="low_conf_only",
                    help="Smooth all samples or only low-confidence ones (default low_conf_only).")
    ap.add_argument("--episodic-lowconf-maxprob", type=float, default=0.80,
                    help="If apply_to=low_conf_only, max_prob < this is low-confidence (default 0.80).")

    # persistent vs episodic labeling thresholds (applied after K fixed)
    ap.add_argument("--persistent-min-frac-samples", type=float, default=0.18,
                    help="Persistent if frac_samples >= this (default 0.18).")
    ap.add_argument("--persistent-min-n-cruises", type=int, default=50,
                    help="Persistent if n_cruises >= this (default 50).")
    ap.add_argument("--persistent-min-span-days", type=int, default=1000,
                    help="Persistent if span_days >= this (default 1000).")

    ap.add_argument("--episodic-max-frac-samples", type=float, default=0.06,
                    help="Episodic if frac_samples <= this (default 0.06).")
    ap.add_argument("--episodic-max-span-days", type=int, default=999999,
                    help="Episodic if span_days <= this (default 999999).")

    # optional inference input
    ap.add_argument("--matrix-cleaned", default=None,
                    help="Optional path to env_pca/tables/matrix_cleaned.csv for feature-space inference.")

    ns = ap.parse_args()

    pc_explicit = None
    if ns.pc_explicit:
        pc_explicit = [p.strip() for p in ns.pc_explicit.split(",") if p.strip()]

    sort_cols = [c.strip() for c in ns.episodic_sort_cols.split(",") if c.strip()]
    depth_norm_col = ns.depth_norm_col if ns.depth_norm_col else ns.anchored_depth_col

    return RunConfig(
        eigengenes_csv=ns.eigengenes,
        pc_keep_csv=ns.pc_keep,
        outdir=ns.outdir,
        sep=ns.sep,
        time_col=ns.time_col,
        cruise_col=ns.cruise_col,
        depth_col=ns.depth_col,
        anchored_depth_col=ns.anchored_depth_col,
        pc_use_mode=ns.pc_use_mode,
        pc_explicit=pc_explicit,
        standardize_pc_space=ns.standardize_pc_space,
        depth_norm=ns.depth_norm,
        depth_norm_col=depth_norm_col,
        density_space=ns.density_space,
        K=ns.K,
        covariance_type=ns.covariance_type,
        n_init=ns.n_init,
        max_iter=ns.max_iter,
        random_state=ns.random_state,
        reg_covar=ns.reg_covar,
        knn_k=ns.knn_k,
        episodic_smoothing=ns.episodic_smoothing,
        episodic_block_col=ns.episodic_block_col,
        episodic_sort_cols=sort_cols,
        episodic_sticky_prob=ns.episodic_sticky_prob,
        episodic_apply_to=ns.episodic_apply_to,
        episodic_lowconf_maxprob=ns.episodic_lowconf_maxprob,
        persistent_min_frac_samples=ns.persistent_min_frac_samples,
        persistent_min_n_cruises=ns.persistent_min_n_cruises,
        persistent_min_span_days=ns.persistent_min_span_days,
        episodic_max_frac_samples=ns.episodic_max_frac_samples,
        episodic_max_span_days=ns.episodic_max_span_days,
        matrix_cleaned_csv=ns.matrix_cleaned,
    )


def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# PC selection + data prep
# -----------------------------

def read_pc_keep(pc_keep_csv: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(pc_keep_csv, sep=sep)
    # tolerate slight variations
    if "PC" not in df.columns:
        raise ValueError("pc_keep_decision.csv must contain a 'PC' column.")
    if "KEEP" not in df.columns:
        # allow 'keep' alternative
        keep_cols = [c for c in df.columns if c.lower() == "keep"]
        if keep_cols:
            df["KEEP"] = df[keep_cols[0]]
        else:
            raise ValueError("pc_keep_decision.csv must contain a 'KEEP' column.")
    df["PC"] = df["PC"].astype(str)
    df["KEEP"] = df["KEEP"].astype(bool)
    return df


def pick_pcs(eig: pd.DataFrame, pc_keep: pd.DataFrame, mode: str, explicit: Optional[List[str]]) -> List[str]:
    pc_cols = [c for c in eig.columns if c.startswith("PC")]
    pc_cols = sorted(pc_cols, key=lambda x: int(x.replace("PC", "")) if x.replace("PC", "").isdigit() else 9999)

    if mode == "all":
        return pc_cols

    if mode == "explicit":
        if not explicit:
            raise ValueError("pc-use-mode=explicit requires --pc-explicit.")
        use = [p for p in explicit if p in eig.columns]
        if not use:
            raise ValueError("None of the requested --pc-explicit columns exist in eigengenes.")
        return use

    # keep
    keep_set = set(pc_keep.loc[pc_keep["KEEP"], "PC"].tolist())
    use = [p for p in pc_cols if p in keep_set]
    if not use:
        raise ValueError("pc-use-mode=keep produced 0 PCs. Check pc_keep_decision.csv.")
    return use


def compute_depth_feature(df: pd.DataFrame, depth_col: str) -> pd.Series:
    d = pd.to_numeric(df[depth_col], errors="coerce")
    mu = float(np.nanmean(d))
    sd = float(np.nanstd(d)) if float(np.nanstd(d)) > 0 else 1.0
    return (d - mu) / sd


def standardize_matrix(X: pd.DataFrame) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    sc = StandardScaler(with_mean=True, with_std=True)
    Xs = sc.fit_transform(X.values)
    return Xs, sc


# -----------------------------
# GMM + uncertainty
# -----------------------------

def responsibility_entropy(P: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy per row: H(p)/log(K)."""
    eps = 1e-12
    P2 = np.clip(P, eps, 1.0)
    H = -np.sum(P2 * np.log(P2), axis=1)
    K = P.shape[1]
    return H / np.log(K)


def fit_gmm(X: np.ndarray, K: int, covariance_type: str, n_init: int, max_iter: int,
            random_state: int, reg_covar: float) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=K,
        covariance_type=covariance_type,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        reg_covar=reg_covar,
    )
    gmm.fit(X)
    return gmm


# -----------------------------
# Density diagnostics (kNN)
# -----------------------------

def knn_mean_distance(Z: np.ndarray, k: int) -> np.ndarray:
    """Mean distance to k nearest neighbors (excluding self)."""
    k_eff = min(k + 1, Z.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Z)
    dists, _ = nn.kneighbors(Z, return_distance=True)
    # dists[:,0] is self=0
    if dists.shape[1] <= 1:
        return np.zeros(Z.shape[0], dtype=float)
    return np.mean(dists[:, 1:], axis=1)


# -----------------------------
# Second-stage episodic smoothing (sticky HMM on responsibilities)
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
    Forward-backward where:
      - states = components
      - emissions at t are proportional to E[t,:] (responsibilities from base GMM)
    Returns smoothed posteriors gamma[t,:].
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


def apply_sticky_smoothing(
    df: pd.DataFrame,
    resp: np.ndarray,
    block_col: str,
    sort_cols: List[str],
    sticky_prob: float,
    apply_to: str,
    lowconf_maxprob: float
) -> np.ndarray:
    """
    Apply sticky smoothing within each block, ordered by sort_cols.
    If apply_to=low_conf_only, only replace rows with max_prob < lowconf_maxprob.
    """
    out = resp.copy()
    K = resp.shape[1]
    T = sticky_transition(K, sticky_prob)

    if block_col not in df.columns:
        # treat all as one block
        order = df.sort_values(sort_cols).index.to_numpy()
        gamma = forward_backward_emission(T, resp[order, :])
        out[order, :] = _selective_replace(resp[order, :], gamma, apply_to, lowconf_maxprob)
        return out

    for b, sub in df.groupby(block_col, sort=False):
        sub_sorted = sub.sort_values(sort_cols)
        idx = sub_sorted.index.to_numpy()
        gamma = forward_backward_emission(T, resp[idx, :])
        out[idx, :] = _selective_replace(resp[idx, :], gamma, apply_to, lowconf_maxprob)

    return out


def _selective_replace(base: np.ndarray, smoothed: np.ndarray, apply_to: str, thr: float) -> np.ndarray:
    if apply_to == "all":
        return smoothed
    # low_conf_only
    maxp = base.max(axis=1)
    mask = maxp < thr
    out = base.copy()
    out[mask, :] = smoothed[mask, :]
    # renormalize (numerical safety)
    out /= out.sum(axis=1, keepdims=True)
    return out


# -----------------------------
# Summaries + labeling
# -----------------------------

def summarize_components(
    meta: pd.DataFrame,
    comp: np.ndarray,
    max_prob: np.ndarray,
    ent: np.ndarray,
    knn_md: np.ndarray,
    time_col: str,
    cruise_col: str,
    depth_group_col: str
) -> pd.DataFrame:
    df = meta.copy()
    df["_component"] = comp
    df["_max_prob"] = max_prob
    df["_entropy"] = ent
    df["_knn_md"] = knn_md

    n_total = df.shape[0]
    t = pd.to_datetime(df[time_col], errors="coerce") if time_col in df.columns else pd.Series([pd.NaT] * n_total)

    rows = []
    for k, sub in df.groupby("_component"):
        n = sub.shape[0]
        frac = n / n_total if n_total else np.nan

        n_cruises = sub[cruise_col].nunique() if cruise_col in sub.columns else np.nan
        n_depth_groups = sub[depth_group_col].nunique() if depth_group_col in sub.columns else np.nan

        tt = pd.to_datetime(sub[time_col], errors="coerce") if time_col in sub.columns else pd.Series([pd.NaT] * n)
        tmin = tt.min()
        tmax = tt.max()
        span_days = (tmax - tmin).days if pd.notna(tmin) and pd.notna(tmax) else np.nan

        rows.append({
            "component": int(k),
            "n_samples": int(n),
            "frac_samples": float(frac),
            "n_cruises": int(n_cruises) if pd.notna(n_cruises) else np.nan,
            "n_depth_groups": int(n_depth_groups) if pd.notna(n_depth_groups) else np.nan,
            "time_min": str(tmin.date()) if pd.notna(tmin) else "",
            "time_max": str(tmax.date()) if pd.notna(tmax) else "",
            "span_days": float(span_days) if pd.notna(span_days) else np.nan,
            "median_max_prob": float(np.nanmedian(sub["_max_prob"].values)),
            "median_entropy": float(np.nanmedian(sub["_entropy"].values)),
            "median_knn_mean_dist": float(np.nanmedian(sub["_knn_md"].values)),
        })

    out = pd.DataFrame(rows).sort_values(["n_samples"], ascending=False).reset_index(drop=True)
    return out


def label_persistent_episodic(
    summary: pd.DataFrame,
    persistent_min_frac: float,
    persistent_min_n_cruises: int,
    persistent_min_span_days: int,
    episodic_max_frac: float,
    episodic_max_span_days: int
) -> pd.DataFrame:
    out = summary.copy()

    def classify(r):
        frac = float(r.get("frac_samples", np.nan))
        ncr = r.get("n_cruises", np.nan)
        span = r.get("span_days", np.nan)

        is_persistent = (
            (not np.isnan(frac)) and (frac >= persistent_min_frac) and
            (not pd.isna(ncr)) and (float(ncr) >= persistent_min_n_cruises) and
            (not np.isnan(span)) and (span >= persistent_min_span_days)
        )
        is_episodic = (
            (not np.isnan(frac)) and (frac <= episodic_max_frac) and
            (not np.isnan(span)) and (span <= episodic_max_span_days)
        )

        if is_persistent:
            return "persistent"
        if is_episodic:
            return "episodic"
        return "intermediate"

    out["compartment_type"] = out.apply(classify, axis=1)
    return out


def pairwise_mean_separation(Z: np.ndarray, comp: np.ndarray, K: int) -> pd.DataFrame:
    """
    Simple separation diagnostics in the model space:
      - Euclidean distance between component means
      - (optional-ish) pooled Mahalanobis using pooled covariance (if invertible)
    """
    means = []
    covs = []
    ns = []
    for k in range(K):
        idx = np.where(comp == k)[0]
        ns.append(len(idx))
        if len(idx) == 0:
            means.append(np.full(Z.shape[1], np.nan))
            covs.append(np.eye(Z.shape[1]))
        else:
            means.append(np.mean(Z[idx, :], axis=0))
            covs.append(np.cov(Z[idx, :].T) if len(idx) > 1 else np.eye(Z.shape[1]))
    means = np.vstack(means)

    # pooled covariance
    pooled = None
    try:
        num = np.zeros((Z.shape[1], Z.shape[1]), dtype=float)
        den = 0
        for k in range(K):
            if ns[k] > 1:
                num += (ns[k] - 1) * covs[k]
                den += (ns[k] - 1)
        pooled = num / den if den > 0 else None
        inv_pooled = np.linalg.inv(pooled) if pooled is not None else None
    except Exception:
        inv_pooled = None

    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if np.any(np.isnan(means[i])) or np.any(np.isnan(means[j])):
                continue
            d_euc = float(np.linalg.norm(means[i] - means[j]))
            if inv_pooled is not None:
                v = (means[i] - means[j]).reshape(-1, 1)
                d_mah = float(np.sqrt((v.T @ inv_pooled @ v).item()))
            else:
                d_mah = np.nan
            rows.append({
                "component_i": i,
                "component_j": j,
                "mean_dist_euclidean": d_euc,
                "mean_dist_mahalanobis_pooled": d_mah,
            })
    return pd.DataFrame(rows).sort_values("mean_dist_euclidean", ascending=False)


def covariance_diagnostics(gmm: GaussianMixture) -> pd.DataFrame:
    """
    Diagnostics per component covariance:
      - trace, det, condition number (where possible)
    """
    K = gmm.n_components
    covs = gmm.covariances_

    rows = []
    for k in range(K):
        C = covs[k]
        # handle covariance_type variations
        if gmm.covariance_type == "full":
            M = C
        elif gmm.covariance_type == "tied":
            M = covs
        elif gmm.covariance_type == "diag":
            M = np.diag(C)
        else:  # spherical
            M = np.eye(gmm.means_.shape[1]) * float(C)

        try:
            tr = float(np.trace(M))
            det = float(np.linalg.det(M))
            cond = float(np.linalg.cond(M))
        except Exception:
            tr, det, cond = np.nan, np.nan, np.nan

        rows.append({
            "component": k,
            "cov_trace": tr,
            "cov_det": det,
            "cov_condition_number": cond,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Plots
# -----------------------------

def plot_pc_scatter(df: pd.DataFrame, pc1: str, pc2: str, comp_col: str, outpath: str) -> None:
    if pc1 not in df.columns or pc2 not in df.columns:
        return
    c = pd.to_numeric(df[comp_col], errors="coerce").fillna(-1).to_numpy()
    plt.figure(figsize=(7, 6))
    plt.scatter(df[pc1].values, df[pc2].values, c=c)
    plt.xlabel(pc1)
    plt.ylabel(pc2)
    plt.title(f"{pc1} vs {pc2} colored by compartment")
    plt.colorbar(label="component")
    save_fig(outpath)


def plot_compartment_frequency_over_time(df: pd.DataFrame, time_col: str, cruise_col: str, comp_col: str,
                                        K: int, outpath: str) -> None:
    if time_col not in df.columns:
        return
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col])
    if tmp.empty:
        return

    # aggregate by date (daily) across cruises; if you want cruise-resolved, you can facet later
    tmp["_date"] = tmp[time_col].dt.date.astype(str)
    g = tmp.groupby(["_date", comp_col]).size().unstack(fill_value=0)

    # ensure all components exist
    for k in range(K):
        if k not in g.columns:
            g[k] = 0
    g = g[sorted(g.columns)]

    # normalize to frequencies per date
    G = g.values.astype(float)
    denom = G.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    F = G / denom

    x = np.arange(F.shape[0])
    plt.figure(figsize=(12, 5))
    plt.stackplot(x, F.T)
    plt.xticks(x[::max(1, len(x)//12)], g.index[::max(1, len(x)//12)], rotation=45, ha="right")
    plt.ylabel("Fraction of samples")
    plt.title("Compartment frequency over time (all cruises aggregated)")
    save_fig(outpath)


def plot_depth_time_heatmaps(df: pd.DataFrame, time_col: str, depth_col: str, comp_col: str, K: int, plots_dir: str, tag: str) -> None:
    if time_col not in df.columns or depth_col not in df.columns:
        return
    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp[depth_col] = pd.to_numeric(tmp[depth_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col, depth_col])
    if tmp.empty:
        return

    # bin time to month for readability
    tmp["_month"] = tmp[time_col].dt.to_period("M").astype(str)
    tmp["_depth"] = tmp[depth_col].round(0).astype(int)

    # build frequency heatmap per component: (depth x month) = fraction of samples at that cell assigned to comp
    cell_counts = tmp.groupby(["_depth", "_month"]).size().rename("n").reset_index()
    for k in range(K):
        kk = tmp[tmp[comp_col] == k].groupby(["_depth", "_month"]).size().rename("nk").reset_index()
        merged = cell_counts.merge(kk, on=["_depth", "_month"], how="left").fillna({"nk": 0})
        merged["frac"] = merged["nk"] / merged["n"]

        depths = sorted(merged["_depth"].unique().tolist())
        months = sorted(merged["_month"].unique().tolist())
        M = np.zeros((len(depths), len(months)), dtype=float)

        di = {d: i for i, d in enumerate(depths)}
        mi = {m: i for i, m in enumerate(months)}
        for _, r in merged.iterrows():
            M[di[int(r["_depth"])], mi[str(r["_month"])]] = float(r["frac"])

        plt.figure(figsize=(12, max(4, 0.18 * len(depths))))
        plt.imshow(M, aspect="auto", interpolation="nearest")
        plt.yticks(np.arange(len(depths)), depths)
        plt.xticks(np.arange(len(months))[::max(1, len(months)//12)],
                   months[::max(1, len(months)//12)], rotation=45, ha="right")
        plt.xlabel("Month")
        plt.ylabel("Depth (m, rounded)")
        plt.title(f"Depth x time membership frequency for component {k} ({tag})")
        plt.colorbar(label="Fraction in component")
        save_fig(os.path.join(plots_dir, f"depth_time_heatmap_component_{k}_{tag}.png"))


# -----------------------------
# Optional feature-space inference
# -----------------------------

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    # minimal spearman (rank + pearson)
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def pc_feature_correlations(matrix_cleaned: pd.DataFrame, eig: pd.DataFrame, pcs: List[str],
                            meta_cols: List[str], out_csv: str) -> pd.DataFrame:
    # numeric features = columns in matrix_cleaned not in meta_cols
    feats = [c for c in matrix_cleaned.columns if c not in meta_cols]
    # coerce numeric
    mc = matrix_cleaned.copy()
    for c in feats:
        mc[c] = pd.to_numeric(mc[c], errors="coerce")

    rows = []
    for pc in pcs:
        if pc not in eig.columns:
            continue
        pcv = pd.to_numeric(eig[pc], errors="coerce").to_numpy()
        for f in feats:
            fv = mc[f].to_numpy()
            mask = ~np.isnan(pcv) & ~np.isnan(fv)
            if mask.sum() < 10:
                continue
            rho = spearman_corr(pcv[mask], fv[mask])
            rows.append({"PC": pc, "feature": f, "spearman_rho": rho})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def compartment_feature_shifts(matrix_cleaned: pd.DataFrame, comp: np.ndarray, meta_cols: List[str],
                              out_csv: str) -> pd.DataFrame:
    feats = [c for c in matrix_cleaned.columns if c not in meta_cols]
    mc = matrix_cleaned.copy()
    for c in feats:
        mc[c] = pd.to_numeric(mc[c], errors="coerce")

    X = mc[feats].to_numpy()
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0

    rows = []
    K = int(np.nanmax(comp)) + 1 if len(comp) else 0
    for k in range(K):
        idx = np.where(comp == k)[0]
        if len(idx) < 5:
            continue
        mk = np.nanmean(X[idx, :], axis=0)
        z = (mk - mu) / sd
        # report top absolute shifts
        top = np.argsort(-np.abs(z))[:20]
        for j in top:
            rows.append({
                "component": k,
                "feature": feats[j],
                "mean_component": float(mk[j]),
                "mean_global": float(mu[j]),
                "shift_z": float(z[j]),
            })

    out = pd.DataFrame(rows).sort_values(["component", "shift_z"], ascending=[True, False])
    out.to_csv(out_csv, index=False)
    return out


# -----------------------------
# Main
# -----------------------------

def run_once(tag: str,
             cfg: RunConfig,
             tables_dir: str,
             plots_dir: str,
             meta: pd.DataFrame,
             pcs_df: pd.DataFrame,
             Z_model: np.ndarray,
             Z_density: np.ndarray,
             scaler: Optional[StandardScaler],
             gmm: GaussianMixture,
             resp: np.ndarray) -> None:

    comp = np.argmax(resp, axis=1).astype(int)
    maxp = resp.max(axis=1)
    ent = responsibility_entropy(resp)
    knn_md = knn_mean_distance(Z_density, cfg.knn_k)

    # save per-sample tables
    assign = meta.copy()
    assign["component"] = comp
    assign["max_prob"] = maxp
    assign["entropy_norm"] = ent
    assign["knn_mean_dist"] = knn_md
    for k in range(resp.shape[1]):
        assign[f"resp_{k}"] = resp[:, k]
    assign.to_csv(os.path.join(tables_dir, f"compartments_assignments_{tag}.csv"), index=False)

    resp_df = pd.DataFrame(resp, columns=[f"resp_{k}" for k in range(resp.shape[1])])
    resp_df.to_csv(os.path.join(tables_dir, f"responsibilities_{tag}.csv"), index=False)

    # summaries
    depth_group_col = cfg.anchored_depth_col if cfg.anchored_depth_col in meta.columns else cfg.depth_col
    summary = summarize_components(
        meta=meta,
        comp=comp,
        max_prob=maxp,
        ent=ent,
        knn_md=knn_md,
        time_col=cfg.time_col,
        cruise_col=cfg.cruise_col,
        depth_group_col=depth_group_col,
    )
    summary.to_csv(os.path.join(tables_dir, f"compartment_summary_{tag}.csv"), index=False)

    labels = label_persistent_episodic(
        summary=summary,
        persistent_min_frac=cfg.persistent_min_frac_samples,
        persistent_min_n_cruises=cfg.persistent_min_n_cruises,
        persistent_min_span_days=cfg.persistent_min_span_days,
        episodic_max_frac=cfg.episodic_max_frac_samples,
        episodic_max_span_days=cfg.episodic_max_span_days,
    )
    labels.to_csv(os.path.join(tables_dir, f"persistent_episodic_labels_{tag}.csv"), index=False)

    sep = pairwise_mean_separation(Z_model, comp, cfg.K)
    sep.to_csv(os.path.join(tables_dir, f"compartment_separation_{tag}.csv"), index=False)

    covdiag = covariance_diagnostics(gmm)
    covdiag.to_csv(os.path.join(tables_dir, f"covariance_diagnostics_{tag}.csv"), index=False)

    # plots
    df_plot = pd.concat([meta.reset_index(drop=True), pcs_df.reset_index(drop=True)], axis=1)
    df_plot["component"] = comp
    plot_pc_scatter(df_plot, "PC1", "PC2", "component", os.path.join(plots_dir, f"pc_scatter_PC1_PC2_{tag}.png"))
    plot_compartment_frequency_over_time(df_plot, cfg.time_col, cfg.cruise_col, "component", cfg.K,
                                         os.path.join(plots_dir, f"compartment_frequency_over_time_{tag}.png"))
    plot_depth_time_heatmaps(df_plot, cfg.time_col, cfg.anchored_depth_col if cfg.anchored_depth_col in df_plot.columns else cfg.depth_col,
                             "component", cfg.K, plots_dir, tag)


def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    eig = pd.read_csv(cfg.eigengenes_csv, sep=cfg.sep)
    pc_keep = read_pc_keep(cfg.pc_keep_csv, sep=cfg.sep)
    pcs = pick_pcs(eig, pc_keep, cfg.pc_use_mode, cfg.pc_explicit)

    # metadata columns = everything that's not PCs (plus any PC columns not used)
    meta_cols = [c for c in eig.columns if not c.startswith("PC")]
    meta = eig[meta_cols].copy()

    # coerce key cols
    if cfg.time_col in meta.columns:
        meta[cfg.time_col] = pd.to_datetime(meta[cfg.time_col], errors="coerce")
    if cfg.depth_col in meta.columns:
        meta[cfg.depth_col] = pd.to_numeric(meta[cfg.depth_col], errors="coerce")
    if cfg.anchored_depth_col in meta.columns:
        meta[cfg.anchored_depth_col] = pd.to_numeric(meta[cfg.anchored_depth_col], errors="coerce")

    Xpcs = eig[pcs].copy()
    for c in pcs:
        Xpcs[c] = pd.to_numeric(Xpcs[c], errors="coerce")
    Xpcs = Xpcs.dropna(axis=0, how="any")
    kept_idx = Xpcs.index

    meta = meta.loc[kept_idx].reset_index(drop=True)
    Xpcs = Xpcs.reset_index(drop=True)

    # model space Z_model
    if cfg.standardize_pc_space:
        Z_model, scaler = standardize_matrix(Xpcs)
        pcs_df = pd.DataFrame(Z_model, columns=pcs)
    else:
        Z_model = Xpcs.values.astype(float)
        scaler = None
        pcs_df = Xpcs.copy()

    # density space Z_density
    if cfg.density_space == "pc":
        Z_density = Z_model
    else:
        dcol = cfg.depth_norm_col
        if dcol not in meta.columns:
            raise ValueError(f"density-space=pc_depthnorm requires depth column '{dcol}' in eigengenes metadata.")
        dz = compute_depth_feature(meta, dcol)
        dz = dz.to_numpy().reshape(-1, 1)
        Z_density = np.hstack([Z_model, dz])

    # Fit final GMM on Z_model (PC space), not density space
    gmm = fit_gmm(
        X=Z_model,
        K=cfg.K,
        covariance_type=cfg.covariance_type,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        reg_covar=cfg.reg_covar,
    )
    resp = gmm.predict_proba(Z_model)

    # Save model params + scaler + PC list
    model_out = {
        "K": cfg.K,
        "covariance_type": cfg.covariance_type,
        "weights": gmm.weights_.tolist(),
        "means": gmm.means_.tolist(),
        "covariances": gmm.covariances_.tolist() if hasattr(gmm.covariances_, "tolist") else gmm.covariances_,
        "precisions_cholesky": gmm.precisions_cholesky_.tolist(),
        "pcs_used": pcs,
        "standardize_pc_space": bool(cfg.standardize_pc_space),
        "scaler": {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        } if scaler is not None else None,
        "depth_feature_used_for_density": cfg.depth_norm_col if cfg.density_space == "pc_depthnorm" else None,
        "random_state": cfg.random_state,
        "n_init": cfg.n_init,
        "max_iter": cfg.max_iter,
        "reg_covar": cfg.reg_covar,
        "converged": bool(getattr(gmm, "converged_", True)),
        "n_iter": int(getattr(gmm, "n_iter_", -1)),
        "lower_bound": float(getattr(gmm, "lower_bound_", np.nan)),
    }
    with open(os.path.join(cfg.outdir, "model_params.json"), "w") as f:
        json.dump(model_out, f, indent=2)

    # base outputs
    run_once(
        tag="base",
        cfg=cfg,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        meta=meta,
        pcs_df=pcs_df,
        Z_model=Z_model,
        Z_density=Z_density,
        scaler=scaler,
        gmm=gmm,
        resp=resp,
    )

    # optional second-stage smoothing
    if cfg.episodic_smoothing:
        sort_cols = [c for c in cfg.episodic_sort_cols if c in meta.columns]
        if not sort_cols:
            raise ValueError("episodic-smoothing enabled but none of episodic-sort-cols exist in metadata.")

        # create a working df containing required cols
        work = meta.copy()
        block_col = cfg.episodic_block_col
        if block_col not in work.columns:
            # still allowed: treated as one block
            pass

        # ensure sort cols comparable
        for c in sort_cols:
            if "date" in c.lower() or "time" in c.lower():
                work[c] = pd.to_datetime(work[c], errors="coerce")
            else:
                # attempt numeric first, fallback to string
                wc = pd.to_numeric(work[c], errors="coerce")
                if wc.notna().sum() > 0:
                    work[c] = wc
                else:
                    work[c] = work[c].astype(str)

        resp_sm = apply_sticky_smoothing(
            df=work,
            resp=resp,
            block_col=block_col,
            sort_cols=sort_cols,
            sticky_prob=cfg.episodic_sticky_prob,
            apply_to=cfg.episodic_apply_to,
            lowconf_maxprob=cfg.episodic_lowconf_maxprob,
        )
        run_once(
            tag="smoothed",
            cfg=cfg,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
            meta=meta,
            pcs_df=pcs_df,
            Z_model=Z_model,
            Z_density=Z_density,
            scaler=scaler,
            gmm=gmm,   # model unchanged; smoothing is post hoc
            resp=resp_sm,
        )

    # optional feature-space inference
    if cfg.matrix_cleaned_csv:
        mc = pd.read_csv(cfg.matrix_cleaned_csv)
        # align by row order assumption: matrix_cleaned was written in same row order as eigengenes
        # We reindexed eigengenes to kept_idx above and then reset_index, so do same to matrix_cleaned:
        mc = mc.loc[kept_idx].reset_index(drop=True)

        # meta_cols for matrix_cleaned: take intersection with eigengenes meta cols
        mc_meta_cols = [c for c in mc.columns if c in meta.columns]
        pc_feature_correlations(
            matrix_cleaned=mc,
            eig=pd.concat([meta.reset_index(drop=True), pd.DataFrame(Z_model, columns=pcs)], axis=1),
            pcs=pcs,
            meta_cols=mc_meta_cols,
            out_csv=os.path.join(tables_dir, "pc_feature_correlations_spearman.csv"),
        )

        # feature shifts using base compartments
        base_assign = pd.read_csv(os.path.join(tables_dir, "compartments_assignments_base.csv"))
        comp = pd.to_numeric(base_assign["component"], errors="coerce").fillna(-1).astype(int).to_numpy()
        compartment_feature_shifts(
            matrix_cleaned=mc,
            comp=comp,
            meta_cols=mc_meta_cols,
            out_csv=os.path.join(tables_dir, "compartment_feature_shifts.csv"),
        )

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")
    print(f"     PCs used: {', '.join(pcs)}")
    if cfg.episodic_smoothing:
        print(f"     Smoothing: ON ({cfg.episodic_apply_to}, sticky_prob={cfg.episodic_sticky_prob}, lowconf<{cfg.episodic_lowconf_maxprob})")
    else:
        print("     Smoothing: OFF")


if __name__ == "__main__":
    main()
