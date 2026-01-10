#!/usr/bin/env python3
"""
env_compartments_o2_soft.py

Purpose
-------
Define legacy oxygen-based compartments (Oxic / Dysoxic / Suboxic / Anoxic) using
SOFT membership probabilities derived from oxygen thresholds, then produce:

- per-sample compartment assignments + uncertainty metrics
- per-compartment summaries and “persistent vs episodic” labels
- optional second-stage “episodic smoothing” (sticky HMM over memberships)
- simple diagnostic plots

Key idea
--------
Instead of hard thresholding on O2, we compute a probability vector over the 4
compartments using smooth logistic "gates" around each threshold.

This makes legacy compartments comparable to GMM responsibilities, and enables the
same smoothing + transition analyses downstream.

Inputs
------
Required:
- --input : CSV with metadata columns including time/cruise/depth and O2

Outputs
-------
--outdir/
  run_config.json
  tables/
    o2_compartments_assignments_base.csv
    o2_responsibilities_base.csv
    o2_compartment_summary_base.csv
    o2_persistent_episodic_labels_base.csv

    (if --episodic-smoothing)
    o2_compartments_assignments_smoothed.csv
    o2_responsibilities_smoothed.csv
    o2_compartment_summary_smoothed.csv
    o2_persistent_episodic_labels_smoothed.csv

  plots/
    o2_depth_scatter_base.png
    (if --episodic-smoothing) o2_depth_scatter_smoothed.png

Notes
-----
- This script does NOT refit anything. It only derives compartment membership from O2.
- Sticky smoothing is identical in spirit to your GMM smoothing: forward-backward on
  per-sample probability vectors, within each Cruise (or chosen block), sorted by depth.
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


# -----------------------------
# Config
# -----------------------------

@dataclass
class RunConfig:
    input_csv: str
    outdir: str
    sep: str

    # columns
    time_col: str
    cruise_col: str
    depth_col: str
    anchored_depth_col: str
    o2_col: str

    # oxygen thresholds (3 boundaries for 4 compartments)
    # convention:
    #   Oxic     : O2 >= T_oxic_dyso
    #   Dysoxic  : T_dyso_sub <= O2 < T_oxic_dyso
    #   Suboxic  : T_sub_anox <= O2 < T_dyso_sub
    #   Anoxic   : O2 < T_sub_anox
    T_oxic_dyso: float
    T_dyso_sub: float
    T_sub_anox: float

    # softness width (same units as O2)
    softness_s_oxic_dyso: float
    softness_s_dyso_sub: float
    softness_s_sub_anox: float

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


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(
        description="Compute soft legacy O2 compartments + optional sticky smoothing."
    )

    ap.add_argument("--input", required=True, help="Input CSV (e.g., eigenvectors_scores.csv or any metadata table).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default=",", help="Delimiter (default ',').")

    ap.add_argument("--time-col", default="date", help="Time column name (default date).")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name (default Cruise).")
    ap.add_argument("--depth-col", default="Depth", help="Depth column name (default Depth).")
    ap.add_argument("--anchored-depth-col", default="Depth_anchored", help="Anchored depth column name (default Depth_anchored).")
    ap.add_argument("--o2-col", default="O2", help="Oxygen column name (default O2).")

    # Thresholds: user supplies these (legacy definitions)
    ap.add_argument("--T-oxic-dyso", type=float, required=True, help="Boundary between oxic and dysoxic.")
    ap.add_argument("--T-dyso-sub", type=float, required=True, help="Boundary between dysoxic and suboxic.")
    ap.add_argument("--T-sub-anox", type=float, required=True, help="Boundary between suboxic and anoxic.")
    ap.add_argument("--softness-s-oxic-dyso", type=float, default=5.0,
                help="Softness width around oxic/dyso boundary (default 5).")
    ap.add_argument("--softness-s-dyso-sub", type=float, default=2.0,
                help="Softness width around dyso/sub boundary (default 2).")
    ap.add_argument("--softness-s-sub-anox", type=float, default=0.5,
                help="Softness width around sub/anox boundary (default 0.5).")

    # Optional sticky smoothing (same pattern as your GMM script)
    ap.add_argument("--episodic-smoothing", action="store_true",
                    help="Apply sticky smoothing within each block (e.g., Cruise) to respect episodic structure.")
    ap.add_argument("--episodic-block-col", default="Cruise", help="Block col for smoothing (default Cruise).")
    ap.add_argument("--episodic-sort-cols", default="Depth_anchored",
                    help="Within-block sort cols, comma-separated (default 'Depth_anchored').")
    ap.add_argument("--episodic-sticky-prob", type=float, default=0.85,
                    help="Self-transition probability in sticky smoothing (default 0.85).")
    ap.add_argument("--episodic-apply-to", choices=["all", "low_conf_only"], default="all",
                    help="Smooth all samples or only low-confidence ones (default all).")
    ap.add_argument("--episodic-lowconf-maxprob", type=float, default=0.80,
                    help="If apply_to=low_conf_only, max_prob < this is low-confidence (default 0.80).")

    # persistent vs episodic labeling thresholds
    ap.add_argument("--persistent-min-frac-samples", type=float, default=0.18,
                    help="Persistent if frac_samples >= this (default 0.18).")
    ap.add_argument("--persistent-min-n-cruises", type=int, default=10,
                    help="Persistent if n_cruises >= this (default 10).")
    ap.add_argument("--persistent-min-span-days", type=int, default=365,
                    help="Persistent if span_days >= this (default 365).")

    ap.add_argument("--episodic-max-frac-samples", type=float, default=0.06,
                    help="Episodic if frac_samples <= this (default 0.06).")
    ap.add_argument("--episodic-max-span-days", type=int, default=999999,
                    help="Episodic if span_days <= this (default 999999).")

    ns = ap.parse_args()

    sort_cols = [c.strip() for c in ns.episodic_sort_cols.split(",") if c.strip()]

    return RunConfig(
        input_csv=ns.input,
        outdir=ns.outdir,
        sep=ns.sep,
        time_col=ns.time_col,
        cruise_col=ns.cruise_col,
        depth_col=ns.depth_col,
        anchored_depth_col=ns.anchored_depth_col,
        o2_col=ns.o2_col,
        T_oxic_dyso=float(ns.T_oxic_dyso),
        T_dyso_sub=float(ns.T_dyso_sub),
        T_sub_anox=float(ns.T_sub_anox),
        softness_s_oxic_dyso=float(ns.softness_s_oxic_dyso),
        softness_s_dyso_sub=float(ns.softness_s_dyso_sub),
        softness_s_sub_anox=float(ns.softness_s_sub_anox),
        episodic_smoothing=bool(ns.episodic_smoothing),
        episodic_block_col=str(ns.episodic_block_col),
        episodic_sort_cols=sort_cols,
        episodic_sticky_prob=float(ns.episodic_sticky_prob),
        episodic_apply_to=str(ns.episodic_apply_to),
        episodic_lowconf_maxprob=float(ns.episodic_lowconf_maxprob),
        persistent_min_frac_samples=float(ns.persistent_min_frac_samples),
        persistent_min_n_cruises=int(ns.persistent_min_n_cruises),
        persistent_min_span_days=int(ns.persistent_min_span_days),
        episodic_max_frac_samples=float(ns.episodic_max_frac_samples),
        episodic_max_span_days=int(ns.episodic_max_span_days),
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
# Soft O2 compartments
# -----------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def soft_o2_memberships(
    o2: np.ndarray,
    T_oxic_dyso: float,
    T_dyso_sub: float,
    T_sub_anox: float,
    s_oxic_dyso: float,
    s_dyso_sub: float,
    s_sub_anox: float
) -> np.ndarray:
    """
    Returns E: (n, 4) soft memberships for [oxic, dysoxic, suboxic, anoxic]
    using three smooth gates.
    """
    s_oxic_dyso = float(s_oxic_dyso)
    s_dyso_sub = float(s_dyso_sub)
    s_sub_anox = float(s_sub_anox)
    if s_oxic_dyso <= 0:
        raise ValueError("softness_s_oxic_dyso must be > 0.")
    if s_dyso_sub <= 0:
        raise ValueError("softness_s_dyso_sub must be > 0.")
    if s_sub_anox <= 0:
        raise ValueError("softness_s_sub_anox must be > 0.")

    x = o2.astype(float)

    # Gates: g_i ~ P(O2 >= T_i) with softness s
    g1 = _sigmoid((x - T_oxic_dyso) / s_oxic_dyso)
    g2 = _sigmoid((x - T_dyso_sub) / s_dyso_sub)
    g3 = _sigmoid((x - T_sub_anox) / s_sub_anox)

    # Compartment probabilities
    p_oxic = g1
    p_dyso = (1.0 - g1) * g2
    p_sub  = (1.0 - g2) * g3
    p_anox = (1.0 - g3)

    E = np.vstack([p_oxic, p_dyso, p_sub, p_anox]).T

    # Normalize just in case of numeric edge behavior (should already sum to ~1)
    row_sums = E.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    E = E / row_sums
    return E


def responsibility_entropy(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    P2 = np.clip(P, eps, 1.0)
    H = -np.sum(P2 * np.log(P2), axis=1)
    K = P.shape[1]
    return H / np.log(K)


# -----------------------------
# Sticky smoothing (same as your GMM version)
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


def _selective_replace(base: np.ndarray, smoothed: np.ndarray, apply_to: str, thr: float) -> np.ndarray:
    if apply_to == "all":
        return smoothed
    maxp = base.max(axis=1)
    mask = maxp < thr
    out = base.copy()
    out[mask, :] = smoothed[mask, :]
    out /= out.sum(axis=1, keepdims=True)
    return out


def apply_sticky_smoothing(
    df: pd.DataFrame,
    resp: np.ndarray,
    block_col: str,
    sort_cols: List[str],
    sticky_prob: float,
    apply_to: str,
    lowconf_maxprob: float
) -> np.ndarray:
    out = resp.copy()
    K = resp.shape[1]
    T = sticky_transition(K, sticky_prob)

    if block_col not in df.columns:
        order = df.sort_values(sort_cols).index.to_numpy()
        gamma = forward_backward_emission(T, resp[order, :])
        out[order, :] = _selective_replace(resp[order, :], gamma, apply_to, lowconf_maxprob)
        return out

    for _, sub in df.groupby(block_col, sort=False):
        sub_sorted = sub.sort_values(sort_cols)
        idx = sub_sorted.index.to_numpy()
        gamma = forward_backward_emission(T, resp[idx, :])
        out[idx, :] = _selective_replace(resp[idx, :], gamma, apply_to, lowconf_maxprob)

    return out


# -----------------------------
# Summaries + labeling (mirrors your GMM pattern)
# -----------------------------

def summarize_components(
    meta: pd.DataFrame,
    comp: np.ndarray,
    max_prob: np.ndarray,
    ent: np.ndarray,
    time_col: str,
    cruise_col: str,
    depth_group_col: str
) -> pd.DataFrame:
    df = meta.copy()
    df["_component"] = comp
    df["_max_prob"] = max_prob
    df["_entropy"] = ent

    n_total = df.shape[0]
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


# -----------------------------
# Plots
# -----------------------------

def plot_o2_depth_scatter(
    df: pd.DataFrame,
    depth_col: str,
    o2_col: str,
    comp_col: str,
    outpath: str,
    title: str,
) -> None:
    if depth_col not in df.columns or o2_col not in df.columns or comp_col not in df.columns:
        return

    depth = pd.to_numeric(df[depth_col], errors="coerce")
    o2 = pd.to_numeric(df[o2_col], errors="coerce")
    labels = pd.to_numeric(df[comp_col], errors="coerce").fillna(-1).astype(int)

    msk = depth.notna() & o2.notna() & labels.notna()
    depth = depth[msk]
    o2 = o2[msk]
    labels = labels[msk]

    uniq = np.sort(labels.unique())
    if len(uniq) == 0:
        return

    # grayscale (no black/white)
    n = len(uniq)
    grays = np.linspace(0.15, 0.85, n)
    lab2c = {lab: (g, g, g) for lab, g in zip(uniq, grays)}

    plt.figure(figsize=(7, 6))
    for lab in uniq:
        m = labels == lab
        plt.scatter(
            o2[m].values,
            depth[m].values,
            s=18,
            alpha=0.85,
            edgecolors="none",
            color=lab2c[lab],
            label=str(lab),
        )

    plt.gca().invert_yaxis()
    plt.xlabel(o2_col)
    plt.ylabel(depth_col)
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title=comp_col)
    save_fig(outpath)


# -----------------------------
# Main
# -----------------------------

COMP_NAMES = ["oxic", "dysoxic", "suboxic", "anoxic"]


def run_once(
    tag: str,
    cfg: RunConfig,
    tables_dir: str,
    plots_dir: str,
    meta: pd.DataFrame,
    o2_vals: np.ndarray,
    resp: np.ndarray
) -> None:
    comp = np.argmax(resp, axis=1).astype(int)
    maxp = resp.max(axis=1)
    ent = responsibility_entropy(resp)

    out = meta.copy()
    out[cfg.o2_col] = o2_vals
    out["component"] = comp
    out["compartment_name"] = [COMP_NAMES[i] if 0 <= i < len(COMP_NAMES) else "" for i in comp]
    out["max_prob"] = maxp
    out["entropy_norm"] = ent

    # write responsibilities
    for k in range(resp.shape[1]):
        out[f"resp_{k}"] = resp[:, k]

    out.to_csv(os.path.join(tables_dir, f"o2_compartments_assignments_{tag}.csv"), index=False)

    resp_df = pd.DataFrame(resp, columns=[f"resp_{k}" for k in range(resp.shape[1])])
    resp_df.to_csv(os.path.join(tables_dir, f"o2_responsibilities_{tag}.csv"), index=False)

    depth_group_col = cfg.anchored_depth_col if cfg.anchored_depth_col in meta.columns else cfg.depth_col
    summary = summarize_components(
        meta=meta,
        comp=comp,
        max_prob=maxp,
        ent=ent,
        time_col=cfg.time_col,
        cruise_col=cfg.cruise_col,
        depth_group_col=depth_group_col,
    )
    summary.to_csv(os.path.join(tables_dir, f"o2_compartment_summary_{tag}.csv"), index=False)

    labels = label_persistent_episodic(
        summary=summary,
        persistent_min_frac=cfg.persistent_min_frac_samples,
        persistent_min_n_cruises=cfg.persistent_min_n_cruises,
        persistent_min_span_days=cfg.persistent_min_span_days,
        episodic_max_frac=cfg.episodic_max_frac_samples,
        episodic_max_span_days=cfg.episodic_max_span_days,
    )
    labels.to_csv(os.path.join(tables_dir, f"o2_persistent_episodic_labels_{tag}.csv"), index=False)

    # Plot: O2 vs depth colored by compartment
    dcol_plot = cfg.anchored_depth_col if cfg.anchored_depth_col in meta.columns else cfg.depth_col
    plot_df = out.copy()
    plot_o2_depth_scatter(
        df=plot_df,
        depth_col=dcol_plot,
        o2_col=cfg.o2_col,
        comp_col="component",
        outpath=os.path.join(plots_dir, f"o2_depth_scatter_{tag}.png"),
        title=f"O2 vs depth colored by O2-compartment ({tag})"
    )


def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    df = pd.read_csv(cfg.input_csv, sep=cfg.sep)

    # Coerce expected columns
    if cfg.time_col in df.columns:
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    if cfg.depth_col in df.columns:
        df[cfg.depth_col] = pd.to_numeric(df[cfg.depth_col], errors="coerce")
    if cfg.anchored_depth_col in df.columns:
        df[cfg.anchored_depth_col] = pd.to_numeric(df[cfg.anchored_depth_col], errors="coerce")
    if cfg.o2_col not in df.columns:
        raise ValueError(f"Missing required oxygen column: '{cfg.o2_col}'")

    df[cfg.o2_col] = pd.to_numeric(df[cfg.o2_col], errors="coerce")

    # Filter rows: need O2 and depth-sort columns (for smoothing) if enabled
    base = df.copy()

    # Keep metadata columns as-is (everything except PC columns, if present)
    meta_cols = [c for c in base.columns if not c.startswith("PC")]
    meta = base[meta_cols].copy()

    # Drop rows with missing O2 (we need it to compute memberships)
    ok = meta[cfg.o2_col].notna()
    meta = meta.loc[ok].reset_index(drop=True)

    o2_vals = meta[cfg.o2_col].to_numpy(dtype=float)

    # Compute base (unsmoothed) soft memberships
    resp = soft_o2_memberships(
        o2=o2_vals,
        T_oxic_dyso=cfg.T_oxic_dyso,
        T_dyso_sub=cfg.T_dyso_sub,
        T_sub_anox=cfg.T_sub_anox,
        s_oxic_dyso=cfg.softness_s_oxic_dyso,
        s_dyso_sub=cfg.softness_s_dyso_sub,
        s_sub_anox=cfg.softness_s_sub_anox,
    )

    # base outputs
    run_once(
        tag="base",
        cfg=cfg,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        meta=meta,
        o2_vals=o2_vals,
        resp=resp,
    )

    # optional smoothing
    if cfg.episodic_smoothing:
        sort_cols = [c for c in cfg.episodic_sort_cols if c in meta.columns]
        if not sort_cols:
            raise ValueError("episodic-smoothing enabled but none of episodic-sort-cols exist in metadata.")

        work = meta.copy()
        for c in sort_cols:
            if "date" in c.lower() or "time" in c.lower():
                work[c] = pd.to_datetime(work[c], errors="coerce")
            else:
                wc = pd.to_numeric(work[c], errors="coerce")
                if wc.notna().sum() > 0:
                    work[c] = wc
                else:
                    work[c] = work[c].astype(str)

        resp_sm = apply_sticky_smoothing(
            df=work,
            resp=resp,
            block_col=cfg.episodic_block_col,
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
            o2_vals=o2_vals,
            resp=resp_sm,
        )

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")
    print("     Compartments: 0=oxic, 1=dysoxic, 2=suboxic, 3=anoxic")
    print(f"     Thresholds: T1(oxic/dyso)={cfg.T_oxic_dyso}, T2(dyso/sub)={cfg.T_dyso_sub}, T3(sub/anox)={cfg.T_sub_anox}")
    print("     Softness:")
    print(f"       oxic/dyso = {cfg.softness_s_oxic_dyso}")
    print(f"       dyso/sub  = {cfg.softness_s_dyso_sub}")
    print(f"       sub/anox  = {cfg.softness_s_sub_anox}")
    if cfg.episodic_smoothing:
        print(f"     Smoothing: ON ({cfg.episodic_apply_to}, sticky_prob={cfg.episodic_sticky_prob}, lowconf<{cfg.episodic_lowconf_maxprob})")
    else:
        print("     Smoothing: OFF")


if __name__ == "__main__":
    main()
