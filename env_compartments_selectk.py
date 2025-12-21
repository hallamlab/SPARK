#!/usr/bin/env python3
"""
scripts/env_compartments_gmm_selectk.py

Goal
----
Given PCA eigengenes (scores) and a PC keep-decision table, fit Gaussian Mixture Models
across a K range and choose K using reviewer-defensible criteria.

Inputs
------
1) eigengenes_scores.csv  (output of env_eigengenes.py)
   - contains metadata columns + PC1..PCn columns

2) pc_keep_decision.csv   (output of env_eigengenes.py --pc-selection)
   - contains columns: PC, KEEP (bool), plus optional other keep_* columns

Core model-selection metrics (per K)
------------------------------------
A) Information criteria:
   - BIC (standard)
   - ICL (BIC + 2 * sum_i sum_k r_ik log r_ik)  [entropy penalty; favors crisp clusters]

B) Cross-validated log-likelihood:
   - K-fold CV average test log-likelihood (optional; slower but strong)

C) Block-bootstrap stability:
   - Refit GMM on bootstrap-resampled blocks (e.g., Cruise) and compare hard labels on
     out-of-bag samples to the base model using Adjusted Rand Index (ARI).

Selection rule (bullet-proof / reviewer-defensible)
---------------------------------------------------
Hard constraints (MUST pass):
  - median bootstrap ARI >= --stability-min-ari,
  - min cluster fraction >= --min-cluster-frac

If no K passes hard constraints: STOP + explain.

Then choose the *smallest* K that is within --select-delta of the best (feasible) K under:
  - ICL (lower is better) or
  - BIC (lower is better) or
  - CV loglik (higher is better)

Outputs
-------
outdir/
  tables/
    gmm_model_selection_metrics.csv
    gmm_k_selection_decision.csv
  plots/
    selectk_ic_bic_icl.png
    selectk_cv_loglik.png            (if --cv-folds > 1)
    selectk_stability_ari.png
    selectk_cluster_sizes.png
    selectk_entropy.png

Also writes (for the selected K):
  tables/gmm_selected_assignments.csv
  tables/gmm_selected_component_summary.csv
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    eigengenes_path: str
    pc_keep_path: str
    outdir: str
    sep: str

    pc_use_mode: str            # keep | all | explicit
    pc_explicit: Optional[List[str]]

    standardize_pc_space: bool

    k_min: int
    k_max: int

    covariance_type: str        # full | diag | tied | spherical
    n_init: int
    max_iter: int
    reg_covar: float
    random_state: int

    # CV
    cv_folds: int

    # Stability (block bootstrap)
    stability_R: int
    stability_block_col: str
    stability_oob_min: int
    stability_min_ari: float

    # Anti-fragmentation constraint
    min_cluster_frac: float

    # Selection rule
    select_by: str              # icl | bic | cv
    select_delta: float         # within-delta of best, choose smallest K


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Select K for GMM compartments using BIC/ICL + CV + stability.")
    ap.add_argument("--eigengenes", required=True, help="Path to eigengenes_scores.csv")
    ap.add_argument("--pc-keep", required=True, help="Path to pc_keep_decision.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep", default=",", help="CSV separator (default ',')")

    ap.add_argument("--pc-use-mode", choices=["keep", "all", "explicit"], default="keep",
                    help="Which PCs to use: keep (from pc_keep_decision), all (all PC cols), explicit (from --pc-explicit)")
    ap.add_argument("--pc-explicit", default=None,
                    help="Comma-separated PC list for pc-use-mode explicit, e.g. 'PC1,PC2,PC3'")

    ap.add_argument("--standardize-pc-space", action="store_true",
                    help="Standardize selected PC columns before GMM (recommended).")

    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=16)

    ap.add_argument("--covariance-type", choices=["full", "diag", "tied", "spherical"], default="full")
    ap.add_argument("--n-init", type=int, default=20, help="GMM n_init (default 20)")
    ap.add_argument("--max-iter", type=int, default=500, help="GMM max_iter (default 500)")
    ap.add_argument("--reg-covar", type=float, default=1e-6, help="GMM reg_covar (default 1e-6)")
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--cv-folds", type=int, default=5,
                    help="K-fold CV for test log-likelihood. Set 1 to disable (default 5).")

    ap.add_argument("--stability-R", type=int, default=100,
                    help="Block-bootstrap replicates for stability (default 100).")
    ap.add_argument("--stability-block-col", default="Cruise",
                    help="Column for block bootstrap (default Cruise).")
    ap.add_argument("--stability-oob-min", type=int, default=200,
                    help="Min out-of-bag samples required to compute ARI for a replicate (default 200).")
    ap.add_argument("--stability-min-ari", type=float, default=0.70,
                    help="Median ARI threshold (default 0.70).")

    ap.add_argument("--min-cluster-frac", type=float, default=0.02,
                    help="Reject K if any component has < this fraction of samples (default 0.02).")

    ap.add_argument("--select-by", choices=["icl", "bic", "cv"], default="icl",
                    help="Primary selection metric (default icl).")
    ap.add_argument("--select-delta", type=float, default=10.0,
                    help="Choose smallest K within delta of best metric (default 10). "
                         "For ICL/BIC this is in IC units; for CV loglik it's absolute loglik units.")

    ns = ap.parse_args()

    pc_explicit = None
    if ns.pc_explicit:
        pc_explicit = [x.strip() for x in ns.pc_explicit.split(",") if x.strip()]

    return Config(
        eigengenes_path=ns.eigengenes,
        pc_keep_path=ns.pc_keep,
        outdir=ns.outdir,
        sep=ns.sep,
        pc_use_mode=ns.pc_use_mode,
        pc_explicit=pc_explicit,
        standardize_pc_space=ns.standardize_pc_space,
        k_min=ns.k_min,
        k_max=ns.k_max,
        covariance_type=ns.covariance_type,
        n_init=ns.n_init,
        max_iter=ns.max_iter,
        reg_covar=ns.reg_covar,
        random_state=ns.random_state,
        cv_folds=ns.cv_folds,
        stability_R=ns.stability_R,
        stability_block_col=ns.stability_block_col,
        stability_oob_min=ns.stability_oob_min,
        stability_min_ari=ns.stability_min_ari,
        min_cluster_frac=ns.min_cluster_frac,
        select_by=ns.select_by,
        select_delta=ns.select_delta,
    )


# -----------------------------
# IO helpers
# -----------------------------

def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def read_df(path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def detect_pc_columns(df: pd.DataFrame) -> List[str]:
    pcs = [c for c in df.columns if c.startswith("PC")]
    def pc_key(x: str) -> int:
        try:
            return int(x.replace("PC", ""))
        except Exception:
            return 10**9
    return sorted(pcs, key=pc_key)


def choose_pcs(df_eig: pd.DataFrame, df_keep: pd.DataFrame, mode: str, explicit: Optional[List[str]]) -> List[str]:
    pcs_all = detect_pc_columns(df_eig)
    if mode == "all":
        return pcs_all
    if mode == "explicit":
        if not explicit:
            raise ValueError("pc-use-mode explicit requires --pc-explicit")
        pcs = [pc for pc in explicit if pc in df_eig.columns]
        if not pcs:
            raise ValueError("None of the --pc-explicit columns exist in eigengenes file.")
        return pcs
    # keep
    if "PC" not in df_keep.columns or "KEEP" not in df_keep.columns:
        raise ValueError("pc_keep_decision.csv must contain columns: PC, KEEP")
    keep_pcs = df_keep.loc[df_keep["KEEP"].astype(bool), "PC"].astype(str).tolist()
    pcs = [pc for pc in keep_pcs if pc in df_eig.columns]
    if not pcs:
        raise ValueError("No KEEP PCs found in eigengenes file. Try --pc-use-mode all.")
    return pcs


# -----------------------------
# Metrics
# -----------------------------

def responsibilities_entropy(resp: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Returns:
      total_entropy = -sum_i sum_k r_ik log r_ik
      mean_entropy_per_sample = total_entropy / n
    """
    r = np.clip(resp, eps, 1.0)
    ent = -np.sum(r * np.log(r))
    return float(ent), float(ent / resp.shape[0])


def icl_from_bic_and_resp(bic: float, resp: np.ndarray) -> float:
    # ICL = BIC + 2 * entropy
    ent_total, _ = responsibilities_entropy(resp)
    return float(bic + 2.0 * ent_total)


def fit_gmm(X: np.ndarray, k: int, cfg: Config) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cfg.covariance_type,
        n_init=cfg.n_init,
        max_iter=cfg.max_iter,
        reg_covar=cfg.reg_covar,
        random_state=cfg.random_state,
    )
    gmm.fit(X)
    return gmm


def cv_test_loglik(X: np.ndarray, k: int, cfg: Config) -> Tuple[float, float]:
    """
    Returns (mean_test_loglik, std_test_loglik).
    Uses average per-sample loglik on test folds.
    """
    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scores = []
    for train_idx, test_idx in kf.split(X):
        gmm = fit_gmm(X[train_idx], k, cfg)
        scores.append(float(gmm.score(X[test_idx])))
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def block_bootstrap_stability(
    X: np.ndarray,
    base_labels: np.ndarray,
    blocks: np.ndarray,
    k: int,
    cfg: Config,
) -> Tuple[float, float, int]:
    """
    Block bootstrap:
      - sample blocks with replacement to form "in-bag"
      - out-of-bag samples = those blocks not selected
      - fit GMM on in-bag
      - predict labels for OOB samples using the fitted model
      - compute ARI vs base labels on those OOB samples

    Returns:
      (median_ari, mean_ari, n_valid_reps)
    """
    rng = np.random.default_rng(cfg.random_state)
    uniq = np.unique(blocks)
    block_to_idx: Dict[str, np.ndarray] = {b: np.where(blocks == b)[0] for b in uniq}

    aris = []
    for _ in range(cfg.stability_R):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        sampled_set = set(sampled.tolist())
        oob_blocks = [b for b in uniq.tolist() if b not in sampled_set]

        if not oob_blocks:
            continue

        inbag_idx = np.concatenate([block_to_idx[b] for b in sampled], axis=0)
        oob_idx = np.concatenate([block_to_idx[b] for b in oob_blocks], axis=0)

        if oob_idx.shape[0] < cfg.stability_oob_min:
            continue

        gmm = fit_gmm(X[inbag_idx], k, cfg)
        oob_labels = gmm.predict(X[oob_idx])

        ari = adjusted_rand_score(base_labels[oob_idx], oob_labels)
        aris.append(float(ari))

    if not aris:
        return float("nan"), float("nan"), 0

    return float(np.median(aris)), float(np.mean(aris)), int(len(aris))


# -----------------------------
# Plotting
# -----------------------------

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_ic(metrics: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["BIC"], marker="o", label="BIC")
    plt.plot(metrics["K"], metrics["ICL"], marker="o", label="ICL")
    plt.xlabel("K")
    plt.ylabel("Information criterion (lower is better)")
    plt.title("GMM model selection: BIC vs ICL")
    plt.legend()
    save_fig(outpath)


def plot_cv(metrics: pd.DataFrame, outpath: str) -> None:
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
    plt.title("K-fold CV log-likelihood vs K")
    save_fig(outpath)


def plot_stability(metrics: pd.DataFrame, outpath: str, threshold: float) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["stability_median_ARI"], marker="o")
    plt.axhline(y=threshold, linestyle="--", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("Median ARI (higher is better)")
    plt.title("Block-bootstrap stability vs K")
    save_fig(outpath)


def plot_entropy(metrics: pd.DataFrame, outpath: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["mean_resp_entropy"], marker="o")
    plt.xlabel("K")
    plt.ylabel("Mean responsibility entropy (lower = crisper)")
    plt.title("Assignment uncertainty vs K")
    save_fig(outpath)


def plot_min_cluster_frac(metrics: pd.DataFrame, outpath: str, threshold: float) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(metrics["K"], metrics["min_cluster_frac"], marker="o")
    plt.axhline(y=threshold, linestyle="--", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("Minimum component fraction")
    plt.title("Smallest cluster fraction vs K")
    save_fig(outpath)


# -----------------------------
# Selection logic (FIXED)
# -----------------------------

def _fail_reasons_row(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("passes_min_cluster", False)):
        reasons.append("min_cluster_frac")
    if not bool(row.get("passes_stability", False)):
        reasons.append("stability")
    # Optional: stability not computed properly
    if pd.isna(row.get("stability_median_ARI", np.nan)) or int(row.get("stability_n_reps", 0)) == 0:
        if "stability" not in reasons:
            reasons.append("stability_nans_or_no_reps")
    return ",".join(reasons) if reasons else ""


def select_k(metrics: pd.DataFrame, cfg: Config) -> Tuple[int, pd.DataFrame]:
    """
    Returns (selected_K, decision_df)

    Hard constraints are enforced:
      - passes_min_cluster AND passes_stability must be True.
    If no feasible K exists, raise SystemExit with an explanation.
    """
    m = metrics.copy()

    # hard constraints
    m["passes_min_cluster"] = m["min_cluster_frac"] >= cfg.min_cluster_frac
    m["passes_stability"] = m["stability_median_ARI"] >= cfg.stability_min_ari
    # Treat "no valid stability reps" as failing stability (defensible)
    m.loc[m["stability_n_reps"].fillna(0).astype(int) <= 0, "passes_stability"] = False
    m["feasible"] = m["passes_min_cluster"] & m["passes_stability"]

    feasible = m[m["feasible"]].copy()
    if feasible.empty:
        # Build decision table (no selection) and stop.
        m["within_delta"] = False
        m["SELECTED"] = False
        m["FAIL_REASONS"] = m.apply(_fail_reasons_row, axis=1)

        decision_cols = [
            "K", "BIC", "ICL",
            "mean_resp_entropy",
            "min_cluster_frac",
            "stability_median_ARI", "stability_mean_ARI", "stability_n_reps",
            "passes_min_cluster", "passes_stability", "feasible", "within_delta",
            "FAIL_REASONS", "SELECTED",
        ]
        decision_df = m[[c for c in decision_cols if c in m.columns]].copy()

        msg_lines = []
        msg_lines.append("[STOP] No feasible K satisfies the HARD constraints.")
        msg_lines.append(f"       Hard constraints:")
        msg_lines.append(f"         - min_cluster_frac >= {cfg.min_cluster_frac}")
        msg_lines.append(f"         - stability_median_ARI >= {cfg.stability_min_ari} (and stability_n_reps > 0)")
        msg_lines.append("")
        msg_lines.append("       What you can do (reviewer-defensible options):")
        msg_lines.append("         1) Relax thresholds slightly (min_cluster_frac or stability_min_ari) and rerun.")
        msg_lines.append("         2) Increase stability_oob_min only if your OOB size is too small to score ARI reliably.")
        msg_lines.append("         3) Consider changing covariance_type (e.g. diag/tied) if full is overfitting.")
        msg_lines.append("")
        msg_lines.append("       See tables/gmm_k_selection_decision.csv for per-K failure reasons.")
        raise SystemExit("\n".join(msg_lines))

    # primary metric: compute best among feasible Ks only
    if cfg.select_by == "icl":
        best = float(feasible.loc[feasible["ICL"].idxmin(), "ICL"])
        m["within_delta"] = m["feasible"] & ((m["ICL"] - best) <= cfg.select_delta)
        eligible = m[m["within_delta"]].copy()
        chosen = int(eligible.sort_values(["K"]).iloc[0]["K"])
    elif cfg.select_by == "bic":
        best = float(feasible.loc[feasible["BIC"].idxmin(), "BIC"])
        m["within_delta"] = m["feasible"] & ((m["BIC"] - best) <= cfg.select_delta)
        eligible = m[m["within_delta"]].copy()
        chosen = int(eligible.sort_values(["K"]).iloc[0]["K"])
    else:
        # cv: higher is better
        if "CV_loglik_mean" not in m.columns:
            raise ValueError("--select-by cv requires --cv-folds > 1")
        best = float(feasible["CV_loglik_mean"].max())
        m["within_delta"] = m["feasible"] & ((best - m["CV_loglik_mean"]) <= cfg.select_delta)
        eligible = m[m["within_delta"]].copy()
        chosen = int(eligible.sort_values(["K"]).iloc[0]["K"])

    decision_cols = [
        "K", "BIC", "ICL",
        "mean_resp_entropy",
        "min_cluster_frac",
        "stability_median_ARI", "stability_mean_ARI", "stability_n_reps",
        "passes_min_cluster", "passes_stability", "feasible", "within_delta",
    ]
    extra = [c for c in decision_cols if c in m.columns]
    decision_df = m[extra].copy()
    decision_df["SELECTED"] = decision_df["K"] == chosen
    return chosen, decision_df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    df_eig = read_df(cfg.eigengenes_path, cfg.sep)
    df_keep = read_df(cfg.pc_keep_path, cfg.sep)

    pcs = choose_pcs(df_eig, df_keep, cfg.pc_use_mode, cfg.pc_explicit)

    # Pull PC matrix
    X = df_eig[pcs].apply(pd.to_numeric, errors="coerce")
    # Drop rows with any missing PC values (these should be rare if upstream worked)
    keep_rows = ~X.isna().any(axis=1)
    df = df_eig.loc[keep_rows].reset_index(drop=True)
    X = X.loc[keep_rows].to_numpy(dtype=float)

    # Standardize PC space (recommended)
    scaler = None
    if cfg.standardize_pc_space:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # blocks for stability
    if cfg.stability_block_col in df.columns:
        blocks = df[cfg.stability_block_col].astype(str).fillna("NA").to_numpy()
    else:
        blocks = np.array(["ALL"] * df.shape[0], dtype=object)

    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        gmm = fit_gmm(X, k, cfg)

        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))  # not directly used for selection, but useful
        resp = gmm.predict_proba(X)
        icl = icl_from_bic_and_resp(bic, resp)

        ent_total, ent_mean = responsibilities_entropy(resp)

        hard = gmm.predict(X)
        counts = np.bincount(hard, minlength=k).astype(float)
        fracs = counts / counts.sum()
        min_frac = float(fracs.min()) if len(fracs) else float("nan")

        # CV
        cv_mean = np.nan
        cv_std = np.nan
        if cfg.cv_folds and cfg.cv_folds > 1:
            cv_mean, cv_std = cv_test_loglik(X, k, cfg)

        # Stability
        base_labels = hard
        med_ari, mean_ari, nrep = block_bootstrap_stability(X, base_labels, blocks, k, cfg)

        rows.append({
            "K": k,
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
            "stability_n_reps": nrep,
        })

    metrics = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    metrics.to_csv(os.path.join(tables_dir, "gmm_model_selection_metrics.csv"), index=False)

    # Plots
    plot_ic(metrics, os.path.join(plots_dir, "selectk_ic_bic_icl.png"))
    if cfg.cv_folds and cfg.cv_folds > 1:
        plot_cv(metrics, os.path.join(plots_dir, "selectk_cv_loglik.png"))
    plot_stability(metrics, os.path.join(plots_dir, "selectk_stability_ari.png"), threshold=cfg.stability_min_ari)
    plot_min_cluster_frac(metrics, os.path.join(plots_dir, "selectk_cluster_sizes.png"), threshold=cfg.min_cluster_frac)
    plot_entropy(metrics, os.path.join(plots_dir, "selectk_entropy.png"))

    # Selection (HARD constraints enforced here)
    selected_k, decision = select_k(metrics, cfg)
    decision.to_csv(os.path.join(tables_dir, "gmm_k_selection_decision.csv"), index=False)

    # Fit selected model and write assignments
    gmm_sel = fit_gmm(X, selected_k, cfg)
    resp_sel = gmm_sel.predict_proba(X)
    labels_sel = gmm_sel.predict(X)
    maxprob = resp_sel.max(axis=1)

    assign = df.copy()
    assign["component"] = labels_sel.astype(int)
    assign["max_prob"] = maxprob.astype(float)
    assign["resp_entropy"] = (-np.sum(np.clip(resp_sel, 1e-12, 1.0) * np.log(np.clip(resp_sel, 1e-12, 1.0)), axis=1)).astype(float)
    assign.to_csv(os.path.join(tables_dir, "gmm_selected_assignments.csv"), index=False)

    # Component summary (simple)
    comp_rows = []
    for c in range(selected_k):
        idx = np.where(labels_sel == c)[0]
        comp_rows.append({
            "component": c,
            "n_samples": int(len(idx)),
            "frac_samples": float(len(idx) / len(labels_sel)),
            "median_max_prob": float(np.median(maxprob[idx])) if len(idx) else np.nan,
            "median_entropy": float(np.median(assign.loc[idx, "resp_entropy"])) if len(idx) else np.nan,
        })
    comp = pd.DataFrame(comp_rows).sort_values(["n_samples"], ascending=False)
    comp.to_csv(os.path.join(tables_dir, "gmm_selected_component_summary.csv"), index=False)

    # Write a tiny human-readable pointer
    with open(os.path.join(cfg.outdir, "SELECTED_K.txt"), "w") as f:
        f.write(f"{selected_k}\n")

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Selected K = {selected_k}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")


if __name__ == "__main__":
    main()
