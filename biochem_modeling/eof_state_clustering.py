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
#   outdir/model_selection.tsv            (if --k auto)
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
    ap.add_argument("--reg-covar", type=float, default=1e-6, help="GMM reg_covar (default 1e-6).")

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

def fit_gmm_select_k(
    X: np.ndarray,
    kmin: int,
    kmax: int,
    cov_type: str,
    seed: int,
    n_init: int,
    reg_covar: float
) -> Tuple[GaussianMixture, pd.DataFrame]:
    rows = []
    best: Optional[GaussianMixture] = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            random_state=seed,
            n_init=int(n_init),
            reg_covar=float(reg_covar),
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        rows.append({"K": k, "BIC": float(bic), "AIC": float(aic)})

        if bic < best_bic:
            best_bic = bic
            best = gmm

    assert best is not None
    return best, pd.DataFrame(rows)


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

    X = df[pcs].to_numpy(dtype=float)

    # Fit model
    if args.k == "auto":
        gmm, sel = fit_gmm_select_k(
            X=X,
            kmin=int(args.k_min),
            kmax=int(args.k_max),
            cov_type=str(args.covariance_type),
            seed=int(args.seed),
            n_init=int(args.n_init),
            reg_covar=float(args.reg_covar),
        )
        sel.to_csv(os.path.join(args.outdir, "model_selection.tsv"), sep="\t", index=False)
        k_used = int(gmm.n_components)
    else:
        k_used = int(args.k)
        gmm = GaussianMixture(
            n_components=k_used,
            covariance_type=str(args.covariance_type),
            random_state=int(args.seed),
            n_init=int(args.n_init),
            reg_covar=float(args.reg_covar),
        )
        gmm.fit(X)

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

    audit_dir = os.path.join(args.outdir, "audit")
    os.makedirs(audit_dir, exist_ok=True)

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
    if args.sticky_smoothing:
        print(f"Wrote: {os.path.join(args.outdir, 'cruise_states_smoothed.tsv')}")


if __name__ == "__main__":
    main()
