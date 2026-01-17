# Path: ./env_state_transition_analysis.py
# Purpose:
#   End-to-end state transition + persistence + coupling analysis for:
#     (1) O2 compartments (cruise-level composition vectors)
#     (2) GMM regimes (cruise-level composition vectors)
#     (3) Hybrid O2×GMM regimes (cruise-level composition vectors)
#
# Inputs (CSV):
#   - cruise_composition_o2.csv      (from env_hybrid_compartment_builder.py)
#   - cruise_composition_gmm.csv     (from env_hybrid_compartment_builder.py)
#   - cruise_composition_hybrid.csv  (from env_hybrid_compartment_builder.py)
#
# Outputs:
#   outdir/
#     run_config.json
#     tables/
#       cruise_states_o2.csv
#       cruise_states_gmm.csv
#       cruise_states_hybrid.csv
#       transition_soft_o2.csv
#       transition_soft_gmm.csv
#       transition_soft_hybrid.csv
#       transition_dominant_o2.csv
#       transition_dominant_gmm.csv
#       transition_dominant_hybrid.csv
#       persistence_o2.csv
#       persistence_gmm.csv
#       persistence_hybrid.csv
#       braycurtis_o2.csv
#       braycurtis_gmm.csv
#       braycurtis_hybrid.csv
#       change_points.csv
#       transition_agreement_summary.csv
#       transition_agreement_per_cruise.csv
#
#       (if --coupling)
#       coupling_corr_o2.csv
#       coupling_corr_gmm.csv
#       coupling_corr_hybrid.csv
#       coupling_clusters_o2.csv
#       coupling_clusters_gmm.csv
#       coupling_clusters_hybrid.csv
#       coupling_edges_o2.csv
#       coupling_edges_gmm.csv
#       coupling_edges_hybrid.csv
#
#     plots/
#       transition_soft_o2.png
#       transition_soft_gmm.png
#       transition_soft_hybrid.png
#       braycurtis_o2.png
#       braycurtis_gmm.png
#       braycurtis_hybrid.png
#       dominant_timeline_o2.png
#       dominant_timeline_gmm.png
#       dominant_timeline_hybrid.png
#
#       braycurtis_compare_three.png
#       braycurtis_compare_pairwise.png
#
#       (if --coupling)
#       coupling_heatmap_o2.png
#       coupling_heatmap_gmm.png
#       coupling_heatmap_hybrid.png
#
# Notes:
#   - Uses cruise-level composition vectors (soft memberships).
#   - Does NOT require per-sample max_prob tables.
#   - Bray–Curtis here is equivalent to 0.5 * L1 because rows are normalized to sum to 1.
#
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    o2_csv: str
    gmm_csv: str
    hybrid_csv: str
    outdir: str
    sep: str

    # identifiers / sorting
    id_cols: List[str]
    cruise_col: str
    time_col: str
    sort_by_time: bool

    # state column detection
    state_prefix: Optional[str]
    drop_prefixes: List[str]

    strat_timeseries_tsv: Optional[str]

    # change point detection
    changepoint_metric: str     # braycurtis|l1
    changepoint_threshold: float

    # plotting
    cmap: str

    # coupling
    coupling: bool
    coupling_method: str        # spearman|pearson
    coupling_min_cruises: int
    coupling_cluster_threshold: float  # cut on distance = 1-corr
    coupling_edge_threshold: float     # export edges where corr >= this

    eof_states_csv: Optional[str]
    eof_state_col: str
    eof_states_sep: Optional[str]



def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="State transition + persistence + coupling analysis for cruise-level composition matrices (O2, GMM, Hybrid)."
    )
    ap.add_argument("--o2", required=True, help="Path to cruise_composition_o2.csv")
    ap.add_argument("--gmm", required=True, help="Path to cruise_composition_gmm.csv")
    ap.add_argument("--hybrid", required=True, help="Path to cruise_composition_hybrid.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ',')")

    ap.add_argument("--id-cols", default="Cruise,date",
                    help="Comma-separated identifier columns to keep (default 'Cruise,date').")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name (default Cruise).")
    ap.add_argument("--time-col", default="date", help="Time column name (default date).")
    ap.add_argument("--sort-by-time", action="store_true", default=True,
                    help="Sort cruises by time_col if present (default True).")

    ap.add_argument("--state-prefix", default=None,
                    help="If set, only columns starting with this prefix are treated as states.")
    ap.add_argument("--drop-prefixes", default="top_,dominant_,other_",
                    help="Comma-separated prefixes to exclude from state columns (default 'top_,dominant_,other_').")

    ap.add_argument("--changepoint-metric", choices=["braycurtis", "l1"], default="braycurtis",
                    help="Metric for change-point detection on consecutive cruises (default braycurtis).")
    ap.add_argument("--changepoint-threshold", type=float, default=0.35,
                    help="Flag change points where metric >= threshold (default 0.35).")

    ap.add_argument("--cmap", default="turbo", help="Matplotlib colormap for plots (default turbo).")

    # Coupling
    ap.add_argument("--coupling", action="store_true",
                    help="Also compute compartment coupling (state-state co-variation) and cluster states into linked families.")
    ap.add_argument("--coupling-method", choices=["spearman", "pearson"], default="spearman",
                    help="Coupling method across cruises (default spearman).")
    ap.add_argument("--coupling-min-cruises", type=int, default=8,
                    help="Minimum cruises required to compute coupling (default 8).")
    ap.add_argument("--coupling-cluster-threshold", type=float, default=0.30,
                    help="Dendrogram cut threshold on distance=1-corr (lower => fewer bigger families). Default 0.30.")
    ap.add_argument("--coupling-edge-threshold", type=float, default=0.60,
                    help="Export edges for corr >= this (default 0.60).")
    ap.add_argument("--strat-timeseries", default=None,
        help="Optional TSV from stratification script (stratification_timeseries.tsv) "
        "to overlay stratification score on signed similarity compare plot.",)
    ap.add_argument("--eof-states", default=None,
        help=(
            "Optional: CSV/TSV with cruise-level EOF state assignments for "
            "background shading on braycurtis_compare_three_signed_similarity.png."),)
    ap.add_argument("--eof-state-col", default="state",
        help="Column in --eof-states containing the discrete state label "
             "(default: dominant_state).",)
    ap.add_argument("--eof-states-sep", default=None,
        help="Delimiter for --eof-states. If omitted, pandas will infer "
        "(default: None).",)

    ns = ap.parse_args()

    id_cols = [c.strip() for c in ns.id_cols.split(",") if c.strip()]
    drop_prefixes = [c.strip() for c in ns.drop_prefixes.split(",") if c.strip()]

    return Config(
        o2_csv=ns.o2,
        gmm_csv=ns.gmm,
        hybrid_csv=ns.hybrid,
        outdir=ns.outdir,
        sep=ns.sep,
        id_cols=id_cols,
        cruise_col=ns.cruise_col,
        time_col=ns.time_col,
        sort_by_time=bool(ns.sort_by_time),
        state_prefix=ns.state_prefix,
        drop_prefixes=drop_prefixes,
        changepoint_metric=ns.changepoint_metric,
        changepoint_threshold=float(ns.changepoint_threshold),
        cmap=ns.cmap,
        coupling=bool(ns.coupling),
        coupling_method=ns.coupling_method,
        coupling_min_cruises=int(ns.coupling_min_cruises),
        coupling_cluster_threshold=float(ns.coupling_cluster_threshold),
        coupling_edge_threshold=float(ns.coupling_edge_threshold),
        strat_timeseries_tsv=ns.strat_timeseries,
        eof_states_csv=ns.eof_states,
        eof_state_col=str(ns.eof_state_col),
        eof_states_sep=ns.eof_states_sep,
    )


def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


# -----------------------------
# Core helpers
# -----------------------------


def load_eof_state_overlay_for_labels(
    cfg: Config,
    labels: List[str],
) -> Tuple[Optional[np.ndarray], Optional[Dict[object, tuple]]]:
    """
    Align EOF state assignments to the x-axis timeline used in the compare plots.

    Alignment strategy:
    - Extract Cruise ID from label (prefers left of '|' if present, else full label)
    - Join on cfg.cruise_col in the --eof-states file (default Cruise)
    - Returns:
        states_aligned: array(len(labels)) of state labels (object dtype), NaN/None when missing
        color_map: dict(state_value -> RGBA tuple) for consistent shading
    """
    if not cfg.eof_states_csv:
        return None, None

    # read with either specified sep or inference
    try:
        df = pd.read_csv(cfg.eof_states_csv, sep=cfg.eof_states_sep if cfg.eof_states_sep else None, engine="python")
    except Exception:
        # fallback: try comma
        df = pd.read_csv(cfg.eof_states_csv)

    if df.empty:
        return None, None
    if cfg.cruise_col not in df.columns:
        raise ValueError(
            f"--eof-states file is missing cruise id column '{cfg.cruise_col}'. "
            f"Columns present: {list(df.columns)}"
        )
    if cfg.eof_state_col not in df.columns:
        raise ValueError(
            f"--eof-states file is missing state column '{cfg.eof_state_col}'. "
            f"Columns present: {list(df.columns)}"
        )

    df = df.copy()
    df[cfg.cruise_col] = df[cfg.cruise_col].astype(str)
    # keep last occurrence if duplicates
    df = df.dropna(subset=[cfg.cruise_col]).drop_duplicates(subset=[cfg.cruise_col], keep="last")

    cruise_to_state: Dict[str, object] = dict(zip(df[cfg.cruise_col].tolist(), df[cfg.eof_state_col].tolist()))

    def _extract_cruise_id(label: str) -> str:
        # label format in this script: "Cruise | date"
        s = "" if label is None else str(label)
        if "|" in s:
            return s.split("|")[0].strip()
        return s.strip()

    states = np.full(len(labels), None, dtype=object)
    for i, lab in enumerate(labels):
        cid = _extract_cruise_id(lab)
        states[i] = cruise_to_state.get(cid, None)

    # build a stable color map (discrete) for shading
    uniq = [u for u in pd.unique(pd.Series(states)) if u is not None and str(u) != "nan"]
    uniq_sorted = sorted(uniq, key=lambda z: str(z))
    if len(uniq_sorted) == 0:
        return states, None

    # tab20 is a good discrete palette for states
    #cm = plt.get_cmap("tab20", max(1, len(uniq_sorted)))
    color_map = {
        st: ((1.0, 1.0, 1.0, 1.0) if st == 'normal' else (0.83, 0.83, 0.83, 1.0))
        for st in uniq_sorted
    }
    return states, color_map


def _shade_state_runs(
    ax: plt.Axes,
    x: np.ndarray,
    states: np.ndarray,
    color_map: Dict[object, tuple],
    alpha: float =50,
    zorder: int = 0,
) -> None:
    """
    Shade background by contiguous runs of the same state.
    Spans TOUCH with no gaps: [i-0.5, j+0.5].
    """
    if states is None or color_map is None:
        return
    if len(states) == 0:
        return

    # x here is local panel x = 0..n-1
    n = len(states)
    s = np.asarray(states, dtype=object)

    def _is_missing(v) -> bool:
        return (v is None) or (str(v) == "nan")

    i = 0
    while i < n:
        if _is_missing(s[i]):
            i += 1
            continue
        st = s[i]
        j = i
        while (j + 1) < n and (not _is_missing(s[j + 1])) and (s[j + 1] == st):
            j += 1

        left = (i - 0.5)
        right = (j + 0.5)

        # Clamp to avoid weirdness if axis autoscale is odd
        left = max(left, -0.5)
        right = min(right, n - 0.5)

        ax.axvspan(
            left,
            right,
            facecolor=color_map.get(st, (0.9, 0.9, 0.9, 1.0)),
            alpha=alpha,
            linewidth=0.0,
            zorder=zorder,
        )
        i = j + 1
        

def _normalize_centered_pm1(values: np.ndarray) -> np.ndarray:
    """
    Map a 1D array to [-1,+1] with median at 0 using piecewise linear scaling:
      below median: [min, median] -> [-1, 0]
      above median: [median, max] -> [0, +1]
    Preserves NaNs.
    """
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return out

    xmin = float(np.nanmin(x))
    xmed = float(np.nanmedian(x))
    xmax = float(np.nanmax(x))

    # avoid divide-by-zero
    left_den = (xmed - xmin) if (xmed > xmin) else 1.0
    right_den = (xmax - xmed) if (xmax > xmed) else 1.0

    left = m & (x < xmed)
    right = m & (x >= xmed)

    out[left] = -1.0 + (x[left] - xmin) / left_den
    out[right] = (x[right] - xmed) / right_den

    out[m] = np.clip(out[m], -1.0, 1.0)
    return out


def load_strat_overlay_for_labels(
    cfg: Config,
    labels: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load cfg.strat_timeseries_tsv and return vectors aligned to `labels` (by extracted date string).

    Returns (aligned_score, aligned_is_anomaly, aligned_anomaly_type), where:
      - aligned_score uses ONLY the 'normalized_score' column (no transforms)
      - aligned_is_anomaly is a boolean mask (NaN -> False)
      - aligned_anomaly_type is an object array of strings ('' when missing)

    Returns (None, None, None) if no file or required columns are missing.
    """
    if not cfg.strat_timeseries_tsv:
        return None, None, None

    df = pd.read_csv(cfg.strat_timeseries_tsv, sep="\t")
    if df.empty or "date" not in df.columns:
        return None, None, None

    # Required columns per your new spec
    required = {"normalized_score", "is_anomaly", "anomaly_type"}
    if not required.issubset(set(df.columns)):
        return None, None, None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    score = pd.to_numeric(df["normalized_score"], errors="coerce").to_numpy(dtype=float)

    # robust bool parse (handles True/False strings, 0/1, etc.)
    is_anom = (
        df["is_anomaly"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "t", "yes", "y"])
        .to_numpy(dtype=bool)
    )

    anom_type = df["anomaly_type"].astype(str).fillna("").to_numpy(dtype=object)

    # date -> (score, is_anom, anom_type)
    date_to_row = {}
    for d, s, a, t in zip(df["date"].tolist(), score, is_anom, anom_type):
        if pd.notna(d):
            # keep row even if score is NaN so anomalies can still be marked
            date_to_row[str(d)] = (float(s) if np.isfinite(s) else np.nan, bool(a), str(t))

    import re
    def _extract_date_str(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        cand = s.split("|")[-1].strip() if "|" in s else s.strip()
        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", cand)
        if m:
            return m.group(1).replace("/", "-")
        m = re.search(r"(\d{4}[-/]\d{2})", cand)
        if m:
            return m.group(1).replace("/", "-") + "-01"
        m = re.search(r"(\d{4})", cand)
        if m:
            return m.group(1) + "-01-01"
        return ""

    aligned_score = np.full(len(labels), np.nan, dtype=float)
    aligned_is_anom = np.zeros(len(labels), dtype=bool)
    aligned_anom_type = np.full(len(labels), "", dtype=object)

    for i, lab in enumerate(labels):
        d = _extract_date_str(lab)
        if not d:
            continue
        row = date_to_row.get(d, None)
        if row is None:
            continue
        s, a, t = row
        aligned_score[i] = float(s)
        aligned_is_anom[i] = bool(a)
        aligned_anom_type[i] = str(t)

    return aligned_score, aligned_is_anom, aligned_anom_type


def _safe_row_normalize(M: np.ndarray) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def _infer_state_cols(cfg: Config, df: pd.DataFrame) -> List[str]:
    cand = [c for c in df.columns if c not in cfg.id_cols]

    if cfg.state_prefix:
        cand = [c for c in cand if c.startswith(cfg.state_prefix)]

    for pref in cfg.drop_prefixes:
        cand = [c for c in cand if not c.startswith(pref)]

    out = []
    for c in cand:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            out.append(c)

    if not out:
        raise ValueError(
            "Could not infer any state columns. "
            "Try setting --state-prefix (e.g., 'hyb_' or 'resp_') or adjust --id-cols/--drop-prefixes."
        )
    return out


def _prepare_matrix(cfg: Config, df: pd.DataFrame, state_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    out = df.copy()

    if cfg.time_col in out.columns:
        out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")

    if cfg.sort_by_time and cfg.time_col in out.columns:
        out = out.sort_values(cfg.time_col)
    elif cfg.cruise_col in out.columns:
        out = out.sort_values(cfg.cruise_col)

    for c in state_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    P = out[state_cols].to_numpy(dtype=float)
    P = _safe_row_normalize(P)
    out[state_cols] = P
    out = out.reset_index(drop=True)
    return out, P


def entropy_norm(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    P2 = np.clip(P, eps, 1.0)
    H = -np.sum(P2 * np.log(P2), axis=1)
    K = P.shape[1]
    if K <= 1:
        return np.zeros(P.shape[0], dtype=float)
    return H / np.log(K)


def bray_curtis(u: np.ndarray, v: np.ndarray) -> float:
    num = np.sum(np.abs(u - v))
    den = np.sum(u + v)
    return float(num / den) if den > 0 else np.nan


def bray_to_signed_dissimilarity_scaled(bc: np.ndarray) -> np.ndarray:
    """
    Dataset-scaled signed dissimilarity in [-1, +1] from Bray–Curtis dissimilarity.

    Preserves dissimilarity ordering (higher = more different).

      bc = min -> -1   (most similar in dataset)
      bc = mid ->  0
      bc = max -> +1   (most different in dataset)

    Use for visualization when you want centered axes but still "distance-like" meaning.
    """
    x = np.asarray(bc, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    m = np.isfinite(x)
    if m.sum() < 2:
        return out

    xmin = float(np.nanmin(x[m]))
    xmax = float(np.nanmax(x[m]))
    if xmax <= xmin:
        out[m] = 0.0
        return out

    x01 = (x - xmin) / (xmax - xmin)   # [0,1], low=similar, high=different
    out[m] = 2.0 * x01[m] - 1.0        # [-1,+1], still low=similar, high=different
    out[m] = np.clip(out[m], -1.0, 1.0)
    return out


def l1_dist(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sum(np.abs(u - v)))


def soft_transition_matrix(P: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    T = np.zeros((K, K), dtype=float)
    for t in range(P.shape[0] - 1):
        T += np.outer(P[t, :], P[t + 1, :])
    T = _safe_row_normalize(T)
    return T


def dominant_transition_counts(dom: np.ndarray, K: int) -> np.ndarray:
    C = np.zeros((K, K), dtype=float)
    for t in range(len(dom) - 1):
        i = int(dom[t])
        j = int(dom[t + 1])
        if i >= 0 and j >= 0:
            C[i, j] += 1.0
    C = _safe_row_normalize(C)
    return C


def persistence_from_dominant(dom: np.ndarray) -> Dict[int, List[int]]:
    runs: Dict[int, List[int]] = {}
    if len(dom) == 0:
        return runs
    cur = int(dom[0])
    n = 1
    for x in dom[1:]:
        x = int(x)
        if x == cur:
            n += 1
        else:
            runs.setdefault(cur, []).append(n)
            cur = x
            n = 1
    runs.setdefault(cur, []).append(n)
    return runs


def make_labels(cfg: Config, df2: pd.DataFrame) -> List[str]:
    if cfg.cruise_col in df2.columns and cfg.time_col in df2.columns:
        return (df2[cfg.cruise_col].astype(str) + " | " + df2[cfg.time_col].astype(str)).tolist()
    if cfg.cruise_col in df2.columns:
        return df2[cfg.cruise_col].astype(str).tolist()
    if cfg.time_col in df2.columns:
        return df2[cfg.time_col].astype(str).tolist()
    return [str(i) for i in range(df2.shape[0])]


# -----------------------------
# Plotting
# -----------------------------

def plot_transition_heatmap(T: np.ndarray, title: str, outpath: str, cmap: str) -> None:
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(T, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(next | current)")
    plt.xlabel("next state")
    plt.ylabel("current state")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_metric_over_time(labels: List[str], y: np.ndarray, title: str, ylabel: str, outpath: str) -> None:
    x = np.arange(len(labels))
    plt.figure(figsize=(max(10, 0.35 * len(labels)), 3.8))
    plt.plot(x, y, marker="o")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_dominant_timeline(labels: List[str], dom: np.ndarray, title: str, outpath: str, cmap: str) -> None:
    if len(dom) == 0:
        return
    x = np.arange(len(dom))
    dom = dom.astype(int)
    uniq = np.unique(dom)
    uniq = np.sort(uniq[uniq >= 0])
    if len(uniq) == 0:
        return
    cm = plt.get_cmap(cmap, len(uniq))
    color_map = {s: cm(i) for i, s in enumerate(uniq)}

    plt.figure(figsize=(max(10, 0.35 * len(labels)), 1.6))
    for i, s in enumerate(dom):
        c = color_map.get(int(s), (0.8, 0.8, 0.8, 1.0))
        plt.bar(i, 1.0, color=c, linewidth=0)
    plt.xlim(-0.5, len(labels) - 0.5)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xticks(x, labels, rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _add_panel_dangling_trend_hints(
    fig,
    panel_info: List[dict],
    delta_x: float = 0.65,        # how far beyond the panel edge (in x data units)
    pull: float = 0.35,           # how much to pull y toward the next panel (0..1)
) -> None:
    """
    Draw dashed 'trend hints' across panel breaks WITHOUT connecting.
    For each break i -> i+1:
      - Panel i: draw from (last_x, last_y) to (last_x + delta_x, y toward next_first)
      - Panel i+1: draw from (-delta_x, y toward prev_last) to (0, first_y)
    Colors/linestyles are taken from the plotted lines when available.
    """
    import numpy as np

    if len(panel_info) < 2:
        return

    for i in range(len(panel_info) - 1):
        a = panel_info[i]
        b = panel_info[i + 1]

        axA = a["ax"]
        axB = b["ax"]

        nA = int(a.get("n", 0))
        nB = int(b.get("n", 0))
        if nA < 1 or nB < 1:
            continue

        last_x = nA - 1

        # Add "extra" as a 4th series. It will work only if you stored:
        #   extra_line, extra_first, extra_last in panel_info.
        for key in ["o2", "gmm", "hyb", "extra"]:
            y_last = a.get(f"{key}_last", np.nan)
            y_first = b.get(f"{key}_first", np.nan)
            if not (np.isfinite(y_last) and np.isfinite(y_first)):
                continue

            line = a.get(f"{key}_line", None)

            # Color/linestyle from the real plotted line if present
            col = line.get_color() if line is not None else None
            ls  = line.get_linestyle() if line is not None else "--"

            # panel A "outgoing" hint (goes past the right edge)
            y_out = float(y_last + pull * (y_first - y_last))
            axA.plot(
                [last_x, last_x + delta_x],
                [y_last, y_out],
                linestyle=ls,
                linewidth=1.2,
                alpha=0.55,
                color=col,
                clip_on=False,
                zorder=10,
            )

            # panel B "incoming" hint (comes in from the left edge)
            y_in = float(y_first + pull * (y_last - y_first))
            axB.plot(
                [-delta_x, 0.0],
                [y_in, y_first],
                linestyle=ls,
                linewidth=1.2,
                alpha=0.55,
                color=col,
                clip_on=False,
                zorder=10,
            )


def plot_three_metric_compare(
    labels: List[str],
    o2: np.ndarray,
    gmm: np.ndarray,
    hyb: np.ndarray,
    cp_o2: np.ndarray,
    cp_gmm: np.ndarray,
    cp_hyb: np.ndarray,
    title: str,
    ylabel: str,
    outpath: str,
    stats_text: str,
    wrap_n: int = 30,
    row_height: float = 3.2,
    col_width: float = 14.5,
    min_per_year_panel: int = 13,
    max_years_per_panel: int = 4,
    extra: Optional[np.ndarray] = None,
    extra_label: str = "Stratification",
    extra_linestyle: str = "--",
    extra_marker: str = "s",
    extra_is_anomaly: Optional[np.ndarray] = None,
    extra_anomaly_type: Optional[np.ndarray] = None,  
    bg_states: Optional[np.ndarray] = None,
    bg_state_color_map: Optional[Dict[object, tuple]] = None,
    bg_state_alpha: float = 0.50, 
) -> None:
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    n = len(labels)
    if n == 0:
        return

    def _extract_date_str(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        cand = s.split("|")[-1].strip() if "|" in s else s.strip()

        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", cand)
        if m:
            return m.group(1)
        m = re.search(r"(\d{4}[-/]\d{2})", cand)
        if m:
            return m.group(1)
        m = re.search(r"(\d{4})", cand)
        if m:
            return m.group(1)
        return ""


    date_strs = [_extract_date_str(s) for s in labels]
    dates = pd.to_datetime(pd.Series(date_strs), errors="coerce")
    years = dates.dt.year
    have_years = years.notna().any()

    # global y-lims
    parts = [np.asarray(o2, float), np.asarray(gmm, float), np.asarray(hyb, float)]
    if extra is not None:
        parts.append(np.asarray(extra, float))
    y_all = np.concatenate(parts)
    y_finite = y_all[np.isfinite(y_all)]
    if y_finite.size == 0:
        y_min, y_max = 0.0, 1.0
    else:
        y_min, y_max = float(np.min(y_finite)), float(np.max(y_finite))
        pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        y_min -= pad
        y_max += pad

    # ----------------------------
    # MODE A: Panel by year (with merging + max years per panel)
    # ----------------------------
    if have_years:
        df = pd.DataFrame({
            "global_i": np.arange(n, dtype=int),   # <<< critical: stable index into extra arrays
            "label": labels,
            "date": dates,
            "year": years,
            "o2": np.asarray(o2, float),
            "gmm": np.asarray(gmm, float),
            "hyb": np.asarray(hyb, float),
            "cp": (np.asarray(cp_o2, bool) | np.asarray(cp_gmm, bool) | np.asarray(cp_hyb, bool)),
        })
        df = df[df["year"].notna()].copy()
        if df.shape[0] > 0:
            df = df.sort_values("date").reset_index(drop=True)

            counts = df.groupby("year").size().to_dict()
            year_list = sorted([int(y) for y in counts.keys()])

            minN = max(1, int(min_per_year_panel))
            maxY = max(1, int(max_years_per_panel))

            panels: List[List[int]] = []
            for yr in year_list:
                c = int(counts.get(yr, 0))

                if not panels:
                    panels.append([yr])
                    continue

                prev = panels[-1]
                if c < minN:
                    # short year tries to merge backward, but respect max years per panel
                    if len(prev) < maxY:
                        prev.append(yr)
                    else:
                        panels.append([yr])
                else:
                    panels.append([yr])

            # Edge case: first panel too short -> merge forward if possible (respect maxY)
            if len(panels) >= 2:
                first_cnt = sum(int(counts.get(y, 0)) for y in panels[0])
                if first_cnt < minN and (len(panels[1]) + len(panels[0]) <= maxY):
                    panels[1] = panels[0] + panels[1]
                    panels = panels[1:]

            n_panels = len(panels)
            fig, axes = plt.subplots(
                nrows=n_panels,
                ncols=1,
                figsize=(col_width, row_height * n_panels),
                sharey=True,
            )
            if n_panels == 1:
                axes = [axes]

            fig.suptitle(title, y=0.995)

            # collect per-panel info for connectors
            panel_info: List[dict] = []

            for ax, yrs in zip(axes, panels):
                sub = df[df["year"].isin(yrs)].copy().sort_values("date")
                sub = sub.reset_index(drop=True)
                x = np.arange(sub.shape[0])

                if bg_states is not None and bg_state_color_map is not None and sub.shape[0] > 0:
                    idx = sub["global_i"].to_numpy(dtype=int)
                    state_seg = np.asarray(bg_states[idx], dtype=object)
                    _shade_state_runs(
                        ax=ax,
                        x=x,
                        states=state_seg,
                        color_map=bg_state_color_map,
                        alpha=float(bg_state_alpha),
                        zorder=0,
                    )

                (o2_line,) = ax.plot(x, sub["o2"].to_numpy(dtype=float), marker="o", label="O2", color="red")
                (gmm_line,) = ax.plot(x, sub["gmm"].to_numpy(dtype=float), marker="o", label="GMM", color="orange")
                (hyb_line,) = ax.plot(x, sub["hyb"].to_numpy(dtype=float), marker="o", label="Hybrid", color="blue")

                #cp_seg = sub["cp"].to_numpy(dtype=bool)
                #for i, is_cp in enumerate(cp_seg):
                #    if is_cp:
                #        ax.axvline(i, color="0.7", alpha=1, linestyle="--", zorder=1)

                ax.set_ylim(y_min, y_max)
                ax.set_ylabel(ylabel)

                # Title like "2018" or "2018–2019" if merged
                if len(yrs) == 1:
                    ax.set_title(str(int(yrs[0])))
                else:
                    ax.set_title(f"{int(min(yrs))}–{int(max(yrs))}")

                lab_seg = sub["label"].astype(str).tolist()
                ax.set_xticks(x)
                ax.set_xticklabels(lab_seg, rotation=90, fontsize=8)

                # store endpoints for connectors
                nseg = int(sub.shape[0])
                if nseg > 0:
                    panel_info.append({
                        "ax": ax,
                        "n": nseg,
                        "o2_line": o2_line,
                        "gmm_line": gmm_line,
                        "hyb_line": hyb_line,
                        "o2_first": float(sub["o2"].iloc[0]),
                        "gmm_first": float(sub["gmm"].iloc[0]),
                        "hyb_first": float(sub["hyb"].iloc[0]),
                        "o2_last": float(sub["o2"].iloc[nseg - 1]),
                        "gmm_last": float(sub["gmm"].iloc[nseg - 1]),
                        "hyb_last": float(sub["hyb"].iloc[nseg - 1]),
                        "extra_line": None,
                        "extra_first": np.nan,
                        "extra_last": np.nan,
                    })
                else:
                    panel_info.append({
                        "ax": ax,
                        "n": 0,
                        "o2_line": o2_line,
                        "gmm_line": gmm_line,
                        "hyb_line": hyb_line,
                        "o2_first": np.nan, "gmm_first": np.nan, "hyb_first": np.nan,
                        "o2_last": np.nan,  "gmm_last": np.nan,  "hyb_last": np.nan,
                        "extra_line": None,
                        "extra_first": np.nan,
                        "extra_last": np.nan,
                    })

                # Strat overlay + anomaly markers (triangles)
                if extra is not None:
                    idx = sub["global_i"].to_numpy(dtype=int)
                    extra_seg = np.asarray(extra[idx], float)

                    anom_seg = None
                    atype_seg = None
                    if extra_is_anomaly is not None:
                        anom_seg = np.asarray(extra_is_anomaly[idx], dtype=bool)
                    if extra_anomaly_type is not None:
                        atype_seg = np.asarray(extra_anomaly_type[idx], dtype=object)

                    extra_color = "black"

                    # line (no marker)
                    (extra_line,) = ax.plot(
                        x,
                        extra_seg,
                        linestyle=extra_linestyle,
                        linewidth=1.2,
                        color=extra_color,
                        label=extra_label,
                        marker="o",
                    )

                    """
                    # normal points (marker=extra_marker)
                    normal_mask = np.isfinite(extra_seg)
                    if anom_seg is not None:
                        normal_mask = normal_mask & (~anom_seg)

                    if np.any(normal_mask):
                        ax.scatter(
                            x[normal_mask],
                            extra_seg[normal_mask],
                            marker=extra_marker,
                            s=28,
                            color=extra_color,
                            zorder=6,
                            label=None,
                        )

                    # anomalies: ^ for high_stratification, v for mixing_event
                    if anom_seg is not None and atype_seg is not None:
                        anom_mask = np.isfinite(extra_seg) & anom_seg

                        high = anom_mask & (np.char.lower(atype_seg.astype(str)) == "high_stratification")
                        mix = anom_mask & (np.char.lower(atype_seg.astype(str)) == "mixing_event")
                        
                        if np.any(high):
                            ax.scatter(
                                x[high],
                                extra_seg[high],
                                marker="^",
                                s=55,
                                color=extra_color,
                                zorder=7,
                                label=None,
                            )
                        if np.any(mix):
                            ax.scatter(
                                x[mix],
                                extra_seg[mix],
                                marker="v",
                                s=55,
                                color=extra_color,
                                zorder=7,
                                label=None,
                            )
                    """
                if extra is not None and nseg > 0:
                    panel_info[-1]["extra_line"] = extra_line
                    panel_info[-1]["extra_first"] = float(extra_seg[0])
                    panel_info[-1]["extra_last"]  = float(extra_seg[nseg - 1])

                ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--", zorder=1)

            # dashed connectors across panel breaks
            _add_panel_dangling_trend_hints(fig, panel_info, delta_x=0.65, pull=0.35)

            # Legend + stats outside to the right (robust handle pick)
            handles, leg_labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                if len(h) > 0:
                    handles, leg_labels = h, l
                    break
            if handles:
                fig.legend(
                    handles, leg_labels,
                    loc="upper right",
                    bbox_to_anchor=(0.995, 0.92),
                    frameon=False,
                )
            if stats_text:
                fig.text(
                    0.995, 0.55, stats_text,
                    va="top", ha="right",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9)
                )

            plt.tight_layout(rect=[0, 0, 0.90, 0.97])
            plt.savefig(outpath, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return

    # ----------------------------
    # MODE B: fallback chunked wrap_n
    # ----------------------------
    wrap_n = max(5, int(wrap_n))
    n_panels = int(np.ceil(n / wrap_n))

    fig, axes = plt.subplots(
        nrows=n_panels,
        ncols=1,
        figsize=(col_width, row_height * n_panels),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(title, y=0.995)

    panel_info: List[dict] = []

    for p in range(n_panels):
        ax = axes[p]
        start = p * wrap_n
        end = min(n, (p + 1) * wrap_n)

        if bg_states is not None and bg_state_color_map is not None:
            state_seg = np.asarray(bg_states[start:end], dtype=object)
            _shade_state_runs(
                ax=ax,
                x=np.arange(end - start),
                states=state_seg,
                color_map=bg_state_color_map,
                alpha=float(bg_state_alpha),
                zorder=0,
            )

        seg_labels = labels[start:end]
        o2_seg = np.asarray(o2[start:end], float)
        gmm_seg = np.asarray(gmm[start:end], float)
        hyb_seg = np.asarray(hyb[start:end], float)

        x = np.arange(end - start)

        (o2_line,) = ax.plot(x, o2_seg, marker="o", label="O2", color="red")
        (gmm_line,) = ax.plot(x, gmm_seg, marker="o", label="GMM", color="orange")
        (hyb_line,) = ax.plot(x, hyb_seg, marker="o", label="Hybrid", color="blue")

        for i_global in range(start, end):
            i_local = i_global - start
            if cp_o2[i_global] or cp_gmm[i_global] or cp_hyb[i_global]:
                ax.axvline(i_local, color="0.7", alpha=1, linestyle="--", zorder=1)

        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(seg_labels, rotation=90, fontsize=8)

        nseg = int(end - start)
        if nseg > 0:
            panel_info.append({
                "ax": ax,
                "n": nseg,
                "o2_line": o2_line,
                "gmm_line": gmm_line,
                "hyb_line": hyb_line,
                "o2_first": float(o2_seg[0]),
                "gmm_first": float(gmm_seg[0]),
                "hyb_first": float(hyb_seg[0]),
                "o2_last": float(o2_seg[-1]),
                "gmm_last": float(gmm_seg[-1]),
                "hyb_last": float(hyb_seg[-1]),
                "extra_line": None,
                "extra_first": np.nan,
                "extra_last": np.nan,
            })
        else:
            panel_info.append({
                "ax": ax,
                "n": 0,
                "o2_line": o2_line,
                "gmm_line": gmm_line,
                "hyb_line": hyb_line,
                "o2_first": np.nan, "gmm_first": np.nan, "hyb_first": np.nan,
                "o2_last": np.nan,  "gmm_last": np.nan,  "hyb_last": np.nan,
                "extra_line": None,
                "extra_first": np.nan,
                "extra_last": np.nan,
            })

        # Strat overlay + anomaly markers (triangles)
        if extra is not None:
            extra_seg = np.asarray(extra[start:end], float)

            anom_seg = None
            atype_seg = None
            if extra_is_anomaly is not None:
                anom_seg = np.asarray(extra_is_anomaly[start:end], dtype=bool)
            if extra_anomaly_type is not None:
                atype_seg = np.asarray(extra_anomaly_type[start:end], dtype=object)

            extra_color = "black"

            # line (no marker)
            (extra_line,) = ax.plot(
                x,
                extra_seg,
                linestyle=extra_linestyle,
                linewidth=1.2,
                color=extra_color,
                label=extra_label,
                marker="o",
            )

            """
            # normal points
            normal_mask = np.isfinite(extra_seg)
            if anom_seg is not None:
                normal_mask = normal_mask & (~anom_seg)

            if np.any(normal_mask):
                ax.scatter(
                    x[normal_mask],
                    extra_seg[normal_mask],
                    marker=extra_marker,
                    s=28,
                    color=extra_color,
                    zorder=6,
                    label=None,
                )

            # anomalies
            if anom_seg is not None and atype_seg is not None:
                anom_mask = np.isfinite(extra_seg) & anom_seg

                high = anom_mask & (np.char.lower(atype_seg.astype(str)) == "high_stratification")
                mix = anom_mask & (np.char.lower(atype_seg.astype(str)) == "mixing_event")

                if np.any(high):
                    ax.scatter(x[high], extra_seg[high], marker="^", s=55, color=extra_color, zorder=7, label=None)
                if np.any(mix):
                    ax.scatter(x[mix], extra_seg[mix], marker="v", s=55, color=extra_color, zorder=7, label=None)
            """

        if extra is not None and nseg > 0:
            panel_info[-1]["extra_line"] = extra_line
            panel_info[-1]["extra_first"] = float(extra_seg[0])
            panel_info[-1]["extra_last"]  = float(extra_seg[-1])

        ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--", zorder=1)


    # dashed connectors across panel breaks
    _add_panel_dangling_trend_hints(fig, panel_info, delta_x=0.65, pull=0.35)

    handles, leg_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if len(h) > 0:
            handles, leg_labels = h, l
            break
    if handles:
        fig.legend(
            handles, leg_labels,
            loc="upper left",
            bbox_to_anchor=(0.995, 0.92),
            frameon=False,
        )
    if stats_text:
        fig.text(
            0.995, 0.55, stats_text,
            va="top", ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9)
        )

    plt.tight_layout(rect=[0, 0, 0.90, 0.97])
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Coupling module (state-state co-variation)
# -----------------------------

def _rank_cols(X: np.ndarray) -> np.ndarray:
    # rank each column independently (average ranks), preserving shape
    R = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        R[:, j] = pd.Series(X[:, j]).rank(method="average").to_numpy(dtype=float)
    return R


def coupling_corr_matrix(P: np.ndarray, method: str) -> np.ndarray:
    """
    P: (n_cruises, n_states), rows sum to 1
    Computes state-state correlation across cruises.
    """
    if P.shape[0] < 2:
        return np.full((P.shape[1], P.shape[1]), np.nan, dtype=float)

    X = P.copy()
    if method == "spearman":
        X = _rank_cols(X)

    # corrcoef expects variables in rows if rowvar=True; we want columns => rowvar=False
    C = np.corrcoef(X, rowvar=False)
    # numeric safety
    C = np.clip(C, -1.0, 1.0)
    return C


def order_by_greedy_corr(C: np.ndarray) -> np.ndarray:
    """
    Lightweight ordering without SciPy:
      - start from max-mean-corr node
      - repeatedly append the node with highest mean corr to current set
    Returns indices order.
    """
    K = C.shape[0]
    if K <= 1:
        return np.arange(K)
    mean_corr = np.nanmean(C, axis=1)
    start = int(np.nanargmax(mean_corr))
    remaining = set(range(K))
    order = [start]
    remaining.remove(start)

    while remaining:
        cur_set = np.array(order, dtype=int)
        # pick node with max avg corr to current set
        best = None
        best_score = -np.inf
        for j in list(remaining):
            s = float(np.nanmean(C[j, cur_set]))
            if s > best_score:
                best_score = s
                best = j
        order.append(int(best))
        remaining.remove(int(best))

    return np.array(order, dtype=int)


def cluster_by_threshold(C: np.ndarray, thr_dist: float) -> np.ndarray:
    """
    Simple graph-based clustering with an edge if distance <= thr_dist,
    where distance = 1 - corr.
    This yields connected components as "families".

    thr_dist in [0,2]. For corr in [-1,1]:
      distance = 1 - corr
      corr >= 1 - thr_dist  <=>  distance <= thr_dist
    """
    K = C.shape[0]
    if K == 0:
        return np.array([], dtype=int)
    corr_thr = 1.0 - float(thr_dist)

    adj = [[] for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            if np.isfinite(C[i, j]) and C[i, j] >= corr_thr:
                adj[i].append(j)
                adj[j].append(i)

    labels = -np.ones(K, dtype=int)
    cur_lab = 0
    for i in range(K):
        if labels[i] != -1:
            continue
        # BFS
        stack = [i]
        labels[i] = cur_lab
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = cur_lab
                    stack.append(v)
        cur_lab += 1
    return labels


def export_coupling_edges(state_names: List[str], C: np.ndarray, edge_thr: float, out_csv: str) -> pd.DataFrame:
    rows = []
    K = len(state_names)
    for i in range(K):
        for j in range(i + 1, K):
            r = C[i, j]
            if np.isfinite(r) and r >= edge_thr:
                rows.append({"state_i": state_names[i], "state_j": state_names[j], "corr": float(r)})
    out = pd.DataFrame(rows).sort_values("corr", ascending=False) if rows else pd.DataFrame(
        columns=["state_i", "state_j", "corr"]
    )
    out.to_csv(out_csv, index=False)
    return out


def plot_coupling_heatmap(C: np.ndarray, order: np.ndarray, state_names: List[str], title: str, outpath: str, cmap: str) -> None:
    C2 = C[np.ix_(order, order)]
    names2 = [state_names[i] for i in order]

    plt.figure(figsize=(max(6, 0.25 * len(names2)), max(5, 0.25 * len(names2))))
    im = plt.imshow(C2, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="corr")
    plt.xticks(np.arange(len(names2)), names2, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(names2)), names2, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def run_coupling(cfg: Config, name: str, P: np.ndarray, state_cols: List[str], tables_dir: str, plots_dir: str) -> None:
    if P.shape[0] < cfg.coupling_min_cruises:
        # Still write a small note so runs are reproducible
        pd.DataFrame([{
            "note": f"coupling skipped: n_cruises={P.shape[0]} < coupling_min_cruises={cfg.coupling_min_cruises}"
        }]).to_csv(os.path.join(tables_dir, f"coupling_corr_{name}.csv"), index=False)
        return

    C = coupling_corr_matrix(P, method=cfg.coupling_method)
    corr_df = pd.DataFrame(C, index=state_cols, columns=state_cols)
    corr_df.to_csv(os.path.join(tables_dir, f"coupling_corr_{name}.csv"), index=True)

    order = order_by_greedy_corr(C)
    plot_coupling_heatmap(
        C=C,
        order=order,
        state_names=state_cols,
        title=f"{name.upper()} coupling ({cfg.coupling_method})",
        outpath=os.path.join(plots_dir, f"coupling_heatmap_{name}.png"),
        cmap=cfg.cmap,
    )

    # “families” via connected components at corr threshold derived from distance cut
    fam = cluster_by_threshold(C, thr_dist=cfg.coupling_cluster_threshold)
    clusters = pd.DataFrame({
        "state": state_cols,
        "family_id": fam.astype(int),
        "mean_corr_to_all": np.nanmean(C, axis=1),
    }).sort_values(["family_id", "mean_corr_to_all"], ascending=[True, False])
    clusters.to_csv(os.path.join(tables_dir, f"coupling_clusters_{name}.csv"), index=False)

    # edge list of strong links
    export_coupling_edges(
        state_names=state_cols,
        C=C,
        edge_thr=cfg.coupling_edge_threshold,
        out_csv=os.path.join(tables_dir, f"coupling_edges_{name}.csv"),
    )


# -----------------------------
# Comparative agreement module (transitions)
# -----------------------------

def _pairwise_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return {"n": float(mask.sum()), "pearson_r": np.nan, "rmse": np.nan, "mean_abs_diff": np.nan}
    aa = a[mask]
    bb = b[mask]
    r = float(np.corrcoef(aa, bb)[0, 1])
    rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
    mad = float(np.mean(np.abs(aa - bb)))
    return {"n": float(mask.sum()), "pearson_r": r, "rmse": rmse, "mean_abs_diff": mad}


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    return float(inter / union) if union > 0 else np.nan


def transition_agreement_tables(
    cfg: Config,
    labels: List[str],
    o2_metric: np.ndarray,
    gmm_metric: np.ndarray,
    hyb_metric: np.ndarray,
    cp_o2: np.ndarray,
    cp_gmm: np.ndarray,
    cp_hyb: np.ndarray,
    tables_dir: str,
    plots_dir: str,
    eof_states_aligned: Optional[np.ndarray] = None,
    eof_state_color_map: Optional[Dict[object, tuple]] = None,
) -> None:
    n = len(labels)
    o2 = o2_metric[:n]
    gmm = gmm_metric[:n]
    hyb = hyb_metric[:n]

    # per-cruise agreement table
    per = pd.DataFrame({
        "idx": np.arange(n, dtype=int),
        "label": labels[:n],
        f"o2_{cfg.changepoint_metric}_vs_local": o2,
        f"gmm_{cfg.changepoint_metric}_vs_local": gmm,
        f"hybrid_{cfg.changepoint_metric}_vs_local": hyb,
        "o2_change": cp_o2[:n].astype(bool),
        "gmm_change": cp_gmm[:n].astype(bool),
        "hybrid_change": cp_hyb[:n].astype(bool),
    })
    per["absdiff_o2_gmm"] = np.abs(per[f"o2_{cfg.changepoint_metric}_vs_local"] - per[f"gmm_{cfg.changepoint_metric}_vs_local"])
    per["absdiff_o2_hybrid"] = np.abs(per[f"o2_{cfg.changepoint_metric}_vs_local"] - per[f"hybrid_{cfg.changepoint_metric}_vs_local"])
    per["absdiff_gmm_hybrid"] = np.abs(per[f"gmm_{cfg.changepoint_metric}_vs_local"] - per[f"hybrid_{cfg.changepoint_metric}_vs_local"])
    per.to_csv(os.path.join(tables_dir, "transition_agreement_per_cruise.csv"), index=False)

    # summary stats (supports your hypothesis: hybrid matches both + higher “resolution”)
    s_o2_gmm = _pairwise_stats(o2, gmm)
    s_o2_hyb = _pairwise_stats(o2, hyb)
    s_gmm_hyb = _pairwise_stats(gmm, hyb)

    # “resolution” proxies:
    #   - variance of the metric
    #   - number of flagged change points
    res = {
        "var_o2": float(np.nanvar(o2)),
        "var_gmm": float(np.nanvar(gmm)),
        "var_hybrid": float(np.nanvar(hyb)),
        "n_change_o2": int(np.sum(cp_o2[:n])),
        "n_change_gmm": int(np.sum(cp_gmm[:n])),
        "n_change_hybrid": int(np.sum(cp_hyb[:n])),
        "jaccard_o2_gmm": _jaccard(cp_o2[:n], cp_gmm[:n]),
        "jaccard_o2_hybrid": _jaccard(cp_o2[:n], cp_hyb[:n]),
        "jaccard_gmm_hybrid": _jaccard(cp_gmm[:n], cp_hyb[:n]),
    }

    rows = [
        {"pair": "o2_vs_gmm", **s_o2_gmm},
        {"pair": "o2_vs_hybrid", **s_o2_hyb},
        {"pair": "gmm_vs_hybrid", **s_gmm_hyb},
        {"pair": "resolution", **res},
    ]
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(tables_dir, "transition_agreement_summary.csv"), index=False)

    # One-frame overlay plot with stats box
    stats_text = (
        f"Pearson r (metric):\n"
        f"  O2~GMM:   {s_o2_gmm['pearson_r']:.3f}\n"
        f"  O2~HYB:   {s_o2_hyb['pearson_r']:.3f}\n"
        f"  GMM~HYB:  {s_gmm_hyb['pearson_r']:.3f}\n"
        #f"\n"
        #f"Var(metric): O2={res['var_o2']:.4f}, GMM={res['var_gmm']:.4f}, HYB={res['var_hybrid']:.4f}\n"
        #f"Change-points: O2={res['n_change_o2']}, GMM={res['n_change_gmm']}, HYB={res['n_change_hybrid']}\n"
        #f"Jaccard(cp): O2~GMM={res['jaccard_o2_gmm']:.3f}, O2~HYB={res['jaccard_o2_hybrid']:.3f}, GMM~HYB={res['jaccard_gmm_hybrid']:.3f}"
    )

    plot_three_metric_compare(
        labels=labels[:n],
        o2=o2,
        gmm=gmm,
        hyb=hyb,
        cp_o2=cp_o2[:n],
        cp_gmm=cp_gmm[:n],
        cp_hyb=cp_hyb[:n],
        title=f"{cfg.changepoint_metric} vs local cruises (O2 vs GMM vs Hybrid)",
        ylabel=f"{cfg.changepoint_metric} vs local",
        outpath=os.path.join(plots_dir, "braycurtis_compare_three.png") if cfg.changepoint_metric == "braycurtis"
               else os.path.join(plots_dir, "metric_compare_three.png"),
        stats_text=stats_text,
    )

    if cfg.changepoint_metric == "braycurtis":
        # Signed similarity: +1 similar, -1 different
        # scale all three together so values are comparable across methods
        all_bc = np.concatenate([
            np.asarray(o2, float),
            np.asarray(gmm, float),
            np.asarray(hyb, float),
        ])

        signed_all = bray_to_signed_dissimilarity_scaled(all_bc)

        n = len(o2)
        o2_signed = signed_all[0:n]
        gmm_signed = signed_all[n:2*n]
        hyb_signed = signed_all[2*n:3*n]

        # Optional: stats box for the signed view (keeps original stats box untouched)
        s_o2_gmm_s = _pairwise_stats(o2_signed, gmm_signed)
        s_o2_hyb_s = _pairwise_stats(o2_signed, hyb_signed)
        s_gmm_hyb_s = _pairwise_stats(gmm_signed, hyb_signed)

        stats_text_signed = (
            f"Pearson r (signed sim):\n"
            f"  O2~GMM:   {s_o2_gmm_s['pearson_r']:.3f}\n"
            f"  O2~HYB:   {s_o2_hyb_s['pearson_r']:.3f}\n"
            f"  GMM~HYB:  {s_gmm_hyb_s['pearson_r']:.3f}\n"
        )

        strat_overlay, strat_is_anom, strat_anom_type = load_strat_overlay_for_labels(cfg, labels[:n])

        plot_three_metric_compare(
            labels=labels[:n],
            o2=o2_signed,
            gmm=gmm_signed,
            hyb=hyb_signed,
            cp_o2=cp_o2[:n],
            cp_gmm=cp_gmm[:n],
            cp_hyb=cp_hyb[:n],
            title="Signed similarity vs local cruises (O2 vs GMM vs Hybrid)",
            ylabel="similarity (+1 different, -1 similar)",
            outpath=os.path.join(plots_dir, "braycurtis_compare_three_signed_similarity.png"),
            stats_text=stats_text_signed,
            extra=strat_overlay,
            extra_label="Stratification",
            extra_linestyle="--",
            extra_marker="o",                 # circles for normal points
            extra_is_anomaly=strat_is_anom,   # enables triangles
            extra_anomaly_type=strat_anom_type,
            bg_states=eof_states_aligned,
            bg_state_color_map=eof_state_color_map,
            bg_state_alpha=0.50,
        )

    # Pairwise plot (absolute diffs) for “hybrid matches both best”
    plt.figure(figsize=(max(10, 0.35 * n), 3.8))
    x = np.arange(n)
    plt.plot(x, per["absdiff_o2_gmm"].values, marker="o", label="|O2 - GMM|")
    plt.plot(x, per["absdiff_o2_hybrid"].values, marker="o", label="|O2 - HYB|")
    plt.plot(x, per["absdiff_gmm_hybrid"].values, marker="o", label="|GMM - HYB|")
    plt.xticks(x, labels[:n], rotation=90)
    plt.ylabel("absolute difference")
    plt.title("Pairwise disagreement per cruise (lower = more agreement)")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "braycurtis_compare_pairwise.png") if cfg.changepoint_metric == "braycurtis"
                else os.path.join(plots_dir, "metric_compare_pairwise.png"), dpi=200)
    plt.close()

    print(
        "[signed-sim debug] braycurtis ranges:",
        f"o2 min/max={np.nanmin(o2):.3f}/{np.nanmax(o2):.3f}",
        f"gmm min/max={np.nanmin(gmm):.3f}/{np.nanmax(gmm):.3f}",
        f"hyb min/max={np.nanmin(hyb):.3f}/{np.nanmax(hyb):.3f}",
    )

# -----------------------------
# Per-method analysis
# -----------------------------

def analyze_one(
    cfg: Config,
    name: str,
    df: pd.DataFrame,
    tables_dir: str,
    plots_dir: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
    """
    Perform per-cruise state composition analysis and derive transition,
    persistence, and local change metrics for one compartment model
    (e.g., O2, GMM, or Hybrid).

    Each row of `df` represents a cruise-level composition vector over K states
    (soft memberships that sum to 1 after normalization). The analysis proceeds
    as follows:

    1) State vector preparation
       - Identify state columns based on config rules (prefix/include/exclude).
       - Coerce to numeric, fill missing with 0, and row-normalize so each cruise
         sums to 1 (probability simplex).
       - Sort cruises by time (or cruise index) to form a temporal sequence.

    2) Per-cruise summaries
       - Dominant state: argmax of the composition vector.
       - Dominant fraction: maximum state probability.
       - Normalized entropy: Shannon entropy divided by log(K), measuring how
         mixed vs. peaked the state distribution is.
       These are written to `cruise_states_<name>.csv`.

    3) Transition structure
       - Soft transition matrix: accumulated outer products P[t] ⊗ P[t+1],
         row-normalized to estimate P(next_state | current_state) using soft
         memberships rather than hard labels.
       - Dominant transition matrix: transition counts using only dominant
         states, row-normalized.
       These are written to `transition_soft_<name>.csv` and
       `transition_dominant_<name>.csv`.

    4) Persistence analysis
       - Runs of consecutive dominant states are extracted.
       - For each state, report number of runs, mean, median, and maximum run
         length (in number of cruises).
       Written to `persistence_<name>.csv`.

    5) Local change (change-point proxy)
       - For each cruise i, compute a local neighborhood change metric as the
         mean of the distance to adjacent cruises:
             metric[i] = mean(dist(P[i], P[i-1]), dist(P[i], P[i+1]))
         using Bray–Curtis or L1 distance, depending on configuration.
       - Edge cruises use the single available neighbor.
       - This produces a per-cruise measure of how different that cruise’s
         state composition is from its immediate temporal context.
       - Cruises with metric >= changepoint_threshold are flagged as
         candidate change points.
       Written to `<metric>_<name>.csv`.

    6) Visualization
       - Soft transition heatmap.
       - Local change metric vs. cruise index/time.
       - Dominant state timeline (categorical bar plot).
       Optional state–state coupling analysis is performed if enabled.

    Parameters
    ----------
    cfg : Config
        Global configuration object controlling column selection, distance
        metric, thresholds, and plotting options.
    name : str
        Label for this model (e.g., "o2", "gmm", "hybrid"), used in filenames.
    df : pandas.DataFrame
        Cruise-level composition table with identifier columns and state columns.
    tables_dir : str
        Output directory for CSV tables.
    plots_dir : str
        Output directory for plots.

    Returns
    -------
    states_out_df : pandas.DataFrame
        Per-cruise dominant state, dominant fraction, and entropy.
    metric : numpy.ndarray
        Local neighborhood change metric per cruise.
    cp_mask : numpy.ndarray (bool)
        Boolean mask indicating candidate change points.
    labels : list of str
        X-axis labels used for plots (cruise and/or date).
    state_cols : list of str
        Names of state columns used in the analysis.
    P : numpy.ndarray, shape (n_cruises, n_states)
        Row-normalized state composition matrix used for all downstream metrics.
    """

    state_cols = _infer_state_cols(cfg, df)
    df2, P = _prepare_matrix(cfg, df, state_cols)

    dom = np.argmax(P, axis=1).astype(int)
    maxp = np.max(P, axis=1)
    ent = entropy_norm(P)

    # per-cruise state table
    states_out = df2[cfg.id_cols].copy()
    states_out["dominant_state"] = dom
    states_out["dominant_frac"] = maxp
    states_out["entropy_norm"] = ent
    states_out.to_csv(os.path.join(tables_dir, f"cruise_states_{name}.csv"), index=False)

    # transitions
    Tsoft = soft_transition_matrix(P)
    pd.DataFrame(Tsoft).to_csv(os.path.join(tables_dir, f"transition_soft_{name}.csv"), index=False)

    Tdom = dominant_transition_counts(dom, P.shape[1])
    pd.DataFrame(Tdom).to_csv(os.path.join(tables_dir, f"transition_dominant_{name}.csv"), index=False)

    # persistence
    runs = persistence_from_dominant(dom)
    rows = []
    for s, lens in runs.items():
        a = np.asarray(lens, dtype=float)
        rows.append({
            "state": int(s),
            "n_runs": int(len(lens)),
            "mean_run_len_cruises": float(np.mean(a)),
            "median_run_len_cruises": float(np.median(a)),
            "max_run_len_cruises": int(np.max(a)),
        })
    pers = pd.DataFrame(rows).sort_values(["mean_run_len_cruises"], ascending=False) if rows else pd.DataFrame(
        columns=["state", "n_runs", "mean_run_len_cruises", "median_run_len_cruises", "max_run_len_cruises"]
    )
    pers.to_csv(os.path.join(tables_dir, f"persistence_{name}.csv"), index=False)

    # change metric per cruise using a sliding neighborhood:
    # metric[i] = mean(distance(P[i], P[i-1]), distance(P[i], P[i+1])) when both exist
    metric = np.full(P.shape[0], np.nan, dtype=float)

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        if cfg.changepoint_metric == "braycurtis":
            return bray_curtis(a, b)
        return l1_dist(a, b)

    for i in range(P.shape[0]):
        vals = []
        if i - 1 >= 0:
            vals.append(_dist(P[i, :], P[i - 1, :]))
        if i + 1 < P.shape[0]:
            vals.append(_dist(P[i, :], P[i + 1, :]))
        metric[i] = float(np.nanmean(vals)) if len(vals) > 0 else np.nan

    cp_mask = np.isfinite(metric) & (metric >= cfg.changepoint_threshold)

    metric_tbl = df2[cfg.id_cols].copy()
    metric_tbl[f"{cfg.changepoint_metric}_vs_local"] = metric
    metric_tbl["is_change_point"] = cp_mask.astype(bool)
    metric_tbl.to_csv(os.path.join(tables_dir, f"{cfg.changepoint_metric}_{name}.csv"), index=False)

    # plots
    labels = make_labels(cfg, df2)

    plot_transition_heatmap(
        Tsoft,
        title=f"{name.upper()} soft transition matrix",
        outpath=os.path.join(plots_dir, f"transition_soft_{name}.png"),
        cmap=cfg.cmap,
    )

    plot_metric_over_time(
        labels=labels,
        y=metric,
        title=f"{name.upper()} {cfg.changepoint_metric} vs local cruises",
        ylabel=f"{cfg.changepoint_metric} vs local",
        outpath=os.path.join(plots_dir, f"{cfg.changepoint_metric}_{name}.png"),
    )

    plot_dominant_timeline(
        labels=labels,
        dom=dom,
        title=f"{name.upper()} dominant state timeline",
        outpath=os.path.join(plots_dir, f"dominant_timeline_{name}.png"),
        cmap=cfg.cmap,
    )

    # optional coupling
    if cfg.coupling:
        run_coupling(cfg, name=name, P=P, state_cols=state_cols, tables_dir=tables_dir, plots_dir=plots_dir)

    return states_out, metric, cp_mask, labels, state_cols, P


def collect_change_points(
    cfg: Config,
    labels: List[str],
    o2_metric: np.ndarray,
    gmm_metric: np.ndarray,
    hyb_metric: np.ndarray,
    out_csv: str,
) -> pd.DataFrame:
    n = min(len(o2_metric), len(gmm_metric), len(hyb_metric), len(labels))
    rows = []
    for i in range(n):
        o2v = o2_metric[i]
        gmmv = gmm_metric[i]
        hybv = hyb_metric[i]
        rows.append({
            "idx": i,
            "label": labels[i],
            f"o2_{cfg.changepoint_metric}_vs_local": float(o2v) if np.isfinite(o2v) else np.nan,
            f"gmm_{cfg.changepoint_metric}_vs_local": float(gmmv) if np.isfinite(gmmv) else np.nan,
            f"hybrid_{cfg.changepoint_metric}_vs_local": float(hybv) if np.isfinite(hybv) else np.nan,
            "o2_change": bool(np.isfinite(o2v) and o2v >= cfg.changepoint_threshold),
            "gmm_change": bool(np.isfinite(gmmv) and gmmv >= cfg.changepoint_threshold),
            "hybrid_change": bool(np.isfinite(hybv) and hybv >= cfg.changepoint_threshold),
            "any_change": bool(
                (np.isfinite(o2v) and o2v >= cfg.changepoint_threshold) or
                (np.isfinite(gmmv) and gmmv >= cfg.changepoint_threshold) or
                (np.isfinite(hybv) and hybv >= cfg.changepoint_threshold)
            ),
            "all_change": bool(
                (np.isfinite(o2v) and o2v >= cfg.changepoint_threshold) and
                (np.isfinite(gmmv) and gmmv >= cfg.changepoint_threshold) and
                (np.isfinite(hybv) and hybv >= cfg.changepoint_threshold)
            ),
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    o2 = pd.read_csv(cfg.o2_csv, sep=cfg.sep)
    gmm = pd.read_csv(cfg.gmm_csv, sep=cfg.sep)
    hyb = pd.read_csv(cfg.hybrid_csv, sep=cfg.sep)

    o2_states, o2_metric, o2_cp, labels_o2, o2_state_cols, P_o2 = analyze_one(cfg, "o2", o2, tables_dir, plots_dir)
    gmm_states, gmm_metric, gmm_cp, labels_gmm, gmm_state_cols, P_gmm = analyze_one(cfg, "gmm", gmm, tables_dir, plots_dir)
    hyb_states, hyb_metric, hyb_cp, labels_hyb, hyb_state_cols, P_hyb = analyze_one(cfg, "hybrid", hyb, tables_dir, plots_dir)

    # Align to shortest common timeline (by index after sorting)
    n = min(len(labels_o2), len(labels_gmm), len(labels_hyb))
    labels = labels_hyb[:n]  # fine as long as all were sorted the same way (by date/cruise)

    eof_states_aligned, eof_state_color_map = load_eof_state_overlay_for_labels(cfg, labels)

    cp = collect_change_points(
        cfg=cfg,
        labels=labels,
        o2_metric=o2_metric[:n],
        gmm_metric=gmm_metric[:n],
        hyb_metric=hyb_metric[:n],
        out_csv=os.path.join(tables_dir, "change_points.csv"),
    )

    # Comparative stats + overlay plot
    transition_agreement_tables(
        cfg=cfg,
        labels=labels,
        o2_metric=o2_metric[:n],
        gmm_metric=gmm_metric[:n],
        hyb_metric=hyb_metric[:n],
        cp_o2=o2_cp[:n],
        cp_gmm=gmm_cp[:n],
        cp_hyb=hyb_cp[:n],
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        eof_states_aligned=eof_states_aligned,
        eof_state_color_map=eof_state_color_map,
    )

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")
    print(f"     Change-point threshold: {cfg.changepoint_metric} >= {cfg.changepoint_threshold}")
    print(f"     Change points flagged (any_change): {int(cp['any_change'].sum())} / {cp.shape[0]}")
    if cfg.coupling:
        print(f"     Coupling: ON ({cfg.coupling_method}), cluster_thr(dist)={cfg.coupling_cluster_threshold}, edge_thr(corr)={cfg.coupling_edge_threshold}")
    else:
        print("     Coupling: OFF")


if __name__ == "__main__":
    main()
