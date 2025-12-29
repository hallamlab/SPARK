#!/usr/bin/env python3

"""
scripts/env_compartments_gmm_selectk.py

Purpose
-------
Select the number of “environmental compartments” (clusters) by fitting Gaussian Mixture Models (GMMs)
to a low-dimensional environmental feature space (PCA scores) and choosing K using a reviewer-defensible,
constraint-first selection rule.

This script is intended to be run after your PCA step (env_eigenvectors.py), but note:
- The inputs are PCA *scores* over environmental / biogeochemical variables.
- Treat them as environmental eigenvectors / PCs.

Conceptual model
----------------
We represent each sample as a point in PC space (PC1..PCn). For each K in [k_min, k_max], we fit a GMM:
  p(x) = sum_{k=1..K} pi_k * Normal(x | mu_k, Sigma_k)

We then evaluate each fitted model using:
  1) Information criteria (BIC, ICL)
  2) Optional K-fold cross-validated test log-likelihood
  3) Block-bootstrap stability (Adjusted Rand Index vs a base fit)
  4) A minimum cluster size constraint to avoid tiny “artifact” components

Inputs
------
1) eigenvectors_scores.csv  (from env_eigenvectors.py)
   - Must contain PC score columns named like: PC1, PC2, ..., PCn
   - May contain metadata columns; they are preserved in output assignment tables.

2) pc_keep_decision.csv   (from env_eigenvectors.py --pc-selection)
   - Used only if --pc-use-mode keep (default).
   - Must contain columns:
       PC   (e.g., "PC1")
       KEEP (truthy/falsey; parsed with .astype(bool))

PC selection (which PC columns become the clustering space)
----------------------------------------------------------
Controlled by --pc-use-mode:

A) keep (default)
   - Uses PCs where pc_keep_decision.csv has KEEP==True
   - Only keeps those PCs that also exist as columns in eigenvectors_scores.csv
   - Fails if none remain (suggests --pc-use-mode all)

B) all
   - Uses all columns in eigenvectors_scores.csv that start with "PC" (sorted numerically)

C) explicit
   - Uses only PCs listed in --pc-explicit (comma-separated)
   - Fails if none of the requested columns exist in eigenvectors_scores.csv

Row filtering
-------------
After selecting PCs:
- PC columns are coerced to numeric with errors="coerce".
- Rows with ANY missing values across the selected PC columns are dropped before modeling.
  (If upstream PCA is clean, this should be rare.)

Optional standardization of PC space
------------------------------------
If --standardize-pc-space is set:
- A StandardScaler (mean=0, std=1) is fitted on the selected PC columns
- The scaled PC matrix is used for ALL GMM fits and ALL metrics

This is often recommended for GMMs if PCs have very different variances due to PC selection or upstream changes.

K range and GMM fit settings
----------------------------
For each K in [--k-min, --k-max], a scikit-learn GaussianMixture is fit with:
- n_components = K
- covariance_type in {full, diag, tied, spherical}
- n_init, max_iter, reg_covar, random_state as configured

Important implementation detail:
- The “base” labels used for stability are taken from the model fit on the full dataset for that K
  in the main loop (hard labels = gmm.predict(X)).

Per-K metrics computed (written to gmm_model_selection_metrics.csv)
-------------------------------------------------------------------
Let n = #samples after row filtering, d = #PC dimensions used.

1) AIC and BIC (from scikit-learn)
   - AIC = gmm.aic(X)
   - BIC = gmm.bic(X)

2) Responsibilities entropy (assignment uncertainty)
   For responsibility matrix R (n x K), r_ik = P(component=k | x_i):
     entropy_total = -sum_i sum_k r_ik * log(r_ik)
     mean_resp_entropy = entropy_total / n
   Lower mean_resp_entropy indicates “crisper” assignments.

3) ICL (Integrated Completed Likelihood proxy)
   Implemented here as:
     ICL = BIC + 2 * entropy_total
   This is a common practical form that penalizes fuzzy assignments (prefers cleaner clustering).

4) Minimum component fraction (anti-fragmentation)
   - hard labels = gmm.predict(X)
   - min_cluster_frac = min_k (count_k / n)
   Used as a hard feasibility constraint.

5) Optional K-fold CV test log-likelihood (--cv-folds > 1)
   - KFold is shuffle=True with random_state
   - For each fold:
       fit GMM on train set
       score test set with gmm.score(X_test)  (average per-sample log-likelihood)
   Reported:
       CV_loglik_mean = mean over folds
       CV_loglik_std  = std over folds

6) Block-bootstrap stability (Adjusted Rand Index)
   Controlled by:
     --stability-R
     --stability-block-col
     --stability-oob-min
     --stability-min-ari

   Block definition:
   - If --stability-block-col exists in the eigenvectors_scores.csv metadata, blocks are:
       blocks = df[block_col].astype(str).fillna("NA")
   - Otherwise, all samples are assigned to a single block "ALL" (stability becomes degenerate).

   One bootstrap replicate:
   - Sample blocks with replacement (size = number of unique blocks) to form “in-bag”
   - Out-of-bag (OOB) = blocks not sampled at least once
   - Fit GMM on in-bag samples only
   - Predict labels on OOB samples
   - Compute ARI between:
       base_labels[OOB] (from the full-data fit for that K)
       vs
       oob_labels       (from the bootstrap-fit model predictions)

   Replicates with OOB sample count < --stability-oob-min are skipped.

   Reported per K:
     stability_median_ARI
     stability_mean_ARI
     stability_n_reps     (#valid replicates contributing)

Selection logic (constraint-first, reviewer-defensible)
------------------------------------------------------
Selection is performed ONLY after all K models have been fit and metrics saved.

Hard feasibility constraints (MUST pass):
  1) min_cluster_frac >= --min-cluster-frac
  2) stability_median_ARI >= --stability-min-ari
  3) stability_n_reps > 0   (no valid bootstrap replicates is treated as failing stability)

If NO K is feasible:
- The script stops with a clear SystemExit message explaining:
  - which hard constraints failed
  - reviewer-defensible options to adjust thresholds or modeling choices
- The decision table still includes failure flags / reasons:
  tables/gmm_k_selection_decision.csv

Primary selection metric (--select-by)
--------------------------------------
Among feasible K values only:

A) icl (default)
   - Best = minimum feasible ICL
   - Eligible = feasible K with (ICL - best) <= --select-delta
   - Choose the smallest K in Eligible

B) bic
   - Best = minimum feasible BIC
   - Eligible = feasible K with (BIC - best) <= --select-delta
   - Choose the smallest K in Eligible

C) cv
   - Requires --cv-folds > 1 (else error)
   - Best = maximum feasible CV_loglik_mean
   - Eligible = feasible K with (best - CV_loglik_mean) <= --select-delta
   - Choose the smallest K in Eligible

This “smallest K within delta of best” rule is explicitly conservative and helps avoid over-partitioning.

Outputs
-------
Writes into --outdir:

Top-level
- run_config.json        (exact config used)
- SELECTED_K.txt         (one line: selected K)

tables/
- gmm_model_selection_metrics.csv
  Columns include:
    K, AIC, BIC, ICL,
    entropy_total, mean_resp_entropy,
    min_cluster_frac,
    CV_loglik_mean, CV_loglik_std,
    stability_median_ARI, stability_mean_ARI, stability_n_reps

- gmm_k_selection_decision.csv
  Contains per-K feasibility flags and within-delta eligibility, plus SELECTED.
  If no K feasible, includes FAIL_REASONS.

For the selected K (fit on full data in the chosen PC space):
- gmm_selected_assignments.csv
  A copy of the filtered eigenvectors_scores.csv rows (metadata preserved), plus:
    component    (int hard label from 0..K-1)
    max_prob     (max responsibility for the assigned component)
    resp_entropy (per-sample responsibility entropy:
                  -sum_k r_ik log r_ik)

- gmm_selected_component_summary.csv
  Per component:
    component
    n_samples
    frac_samples
    median_max_prob
    median_entropy

plots/
- selectk_ic_bic_icl.png          (BIC + ICL vs K)
- selectk_cv_loglik.png           (if --cv-folds > 1)
- selectk_stability_ari.png       (median ARI vs K with threshold line)
- selectk_cluster_sizes.png       (min_cluster_frac vs K with threshold line)
- selectk_entropy.png             (mean_resp_entropy vs K)

Notes / invariants (important)
------------------------------
- PC columns are assumed to be named starting with "PC" (e.g., PC1, PC2, ...).
- Rows with any missing selected-PC values are dropped before fitting any model.
- If --stability-block-col does not exist, all rows are treated as one block ("ALL"),
  which makes stability estimates non-informative.
- Block bootstrap compares OOB labels from bootstrap fits to labels from the full-data “base” fit for that K.
- Component labels are not post-aligned across fits; stability is measured via ARI which is label-permutation invariant.

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
    eigenvectors_path: str
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
    ap.add_argument("--eigenvectors", required=True, help="Path to eigenvectors_scores.csv")
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
        eigenvectors_path=ns.eigenvectors,
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
            raise ValueError("None of the --pc-explicit columns exist in eigenvectors file.")
        return pcs
    # keep
    if "PC" not in df_keep.columns or "KEEP" not in df_keep.columns:
        raise ValueError("pc_keep_decision.csv must contain columns: PC, KEEP")
    keep_pcs = df_keep.loc[df_keep["KEEP"].astype(bool), "PC"].astype(str).tolist()
    pcs = [pc for pc in keep_pcs if pc in df_eig.columns]
    if not pcs:
        raise ValueError("No KEEP PCs found in eigenvectors file. Try --pc-use-mode all.")
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

    df_eig = read_df(cfg.eigenvectors_path, cfg.sep)
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
