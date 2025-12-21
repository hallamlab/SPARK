#!/usr/bin/env python3
"""
env_compare_compartments.py

PURPOSE
- Compare data-driven GMM compartments vs contemporary O2 compartments
- Produce publication-ready depth-profile plots + UMAP embedding plots
- Compute defensible agreement + clustering-quality metrics

FIXES (already in your current version)
- Guard against duplicate column headers (e.g., duplicate 'cruise_year_month_depth')
- Use composite join keys by default: Cruise+Year+Month+Day+Depth
  (prevents collisions when multiple casts exist in the same month)

NEW ADD-ON (#4)
- Additional quality metrics in PC space and biochem space:
    silhouette, Calinski-Harabasz, Davies-Bouldin
- Biological inference helpers:
    * Per-compartment medians in biochem + PC space (GMM and O2)
    * Pairwise effect-size ranking of biochem drivers for GMM separation
- Optional PCA-stage interpretation if you provide --pca-tables-dir:
    * pc_top_loadings.csv + plots of top loadings per PC
    * merge pc_loading_concentration.csv if available (entropy/effective features)

INPUTS
- matrix_cleaned.csv (biochem + metadata; Oxygen already uM)
- eigengenes_scores.csv (PC scores + metadata)
- compartments_assignments_smoothed.csv (component/max_prob/entropy/resp_* + metadata)

OUTPUTS
- outdir/tables/*.csv
- outdir/plots/*.pdf/.svg/.png
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

try:
    import umap  # type: ignore
    _HAVE_UMAP = True
except Exception:
    umap = None
    _HAVE_UMAP = False


# ----------------------------
# Palettes (user-specified)
# ----------------------------

O2_COMPARTMENT_PALETTE = {
    "oxic": "red",
    "dysoxic": "lightblue",
    "suboxic": "green",
    "anoxic": "purple",
}

BIOCHEM_COLOR_MAP = {
    "Oxygen": "black",
    "Nitrogen Oxides": "#E7298A",
    "Nitrate": "#1B9E77",
    "Nitrite": "#66A61E",
    "Nitrous Oxide": "#0C5196",
    "Ammonium": "#7570B3",
    "Hydrogen Sulfide": "#D95F02",
    "Methane": "violet",
}


def depth_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#D8D8D8", "#2F2F2F"]  # light grey -> dark carbon
    return LinearSegmentedColormap.from_list("depth_grey_carbon", colors)


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    matrix_cleaned: str
    eigengenes: str
    assignments: str
    outdir: str
    sep_matrix: str
    sep_eig: str
    sep_assign: str

    # keying
    id_col: str
    key_mode: str            # "composite" or "id"
    key_cols: List[str]      # used if key_mode == "composite"
    key_sep: str
    derived_key_col: str     # internal merge key name

    # core cols
    depth_col: str
    depth_anchored_col: str
    date_col: str
    oxygen_col: str
    cruise_col: str

    # O2 thresholds uM
    o2_oxic_gt: float
    o2_dysoxic_hi: float
    o2_dysoxic_lo: float
    o2_suboxic_hi: float
    o2_suboxic_lo: float

    # UMAP
    umap_n_neighbors: int
    umap_min_dist: float
    umap_random_state: int
    umap_metric: str

    # bubble scaling
    bubble_q_low: float
    bubble_q_high: float
    bubble_size_min: float
    bubble_size_max: float

    # PC cols
    pc_cols: Optional[str]

    # save formats
    save_pdf: bool
    save_svg: bool
    save_png: bool
    png_dpi: int

    # #4 optional PCA-stage inputs
    pca_tables_dir: Optional[str]
    pca_loadings_path: Optional[str]
    pc_loading_concentration_path: Optional[str]

    # #4 loadings plots config
    pc_loading_top_n: int


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Compare GMM compartments vs O2 compartments; depth profiles + UMAP + stats.")

    ap.add_argument("--matrix-cleaned", required=True, help="Path to matrix_cleaned.csv")
    ap.add_argument("--eigengenes", required=True, help="Path to eigengenes_scores.csv")
    ap.add_argument("--assignments", required=True, help="Path to compartments_assignments_smoothed.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep-matrix", default=",")
    ap.add_argument("--sep-eig", default=",")
    ap.add_argument("--sep-assign", default=",")

    ap.add_argument("--id-col", default="cruise_year_month_depth", help="Legacy ID column (only used if --key-mode id).")

    # IMPORTANT: default composite key includes Day
    ap.add_argument(
        "--key-mode",
        choices=["composite", "id"],
        default="composite",
        help="Merge key mode. Default composite (Cruise+Year+Month+Day+Depth).",
    )
    ap.add_argument(
        "--key-cols",
        default="Cruise,Year,Month,Day,Depth",
        help="Comma-separated columns to build composite key (default Cruise,Year,Month,Day,Depth).",
    )
    ap.add_argument("--key-sep", default="|", help="Separator when building composite key (default '|').")

    ap.add_argument("--depth-col", default="Depth")
    ap.add_argument("--depth-anchored-col", default="Depth_anchored")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--oxygen-col", default="Oxygen")
    ap.add_argument("--cruise-col", default="Cruise")

    ap.add_argument("--o2-oxic-gt", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-hi", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-lo", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-hi", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-lo", type=float, default=1.0)

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.05)
    ap.add_argument("--umap-random-state", type=int, default=42)
    ap.add_argument("--umap-metric", default="manhattan")

    ap.add_argument("--bubble-q-low", type=float, default=0.01)
    ap.add_argument("--bubble-q-high", type=float, default=0.99)
    ap.add_argument("--bubble-size-min", type=float, default=8.0)
    ap.add_argument("--bubble-size-max", type=float, default=120.0)

    ap.add_argument("--pc-cols", default=None, help="Optional comma-separated PCs for stats (e.g. 'PC1,PC2,PC3'). If omitted, auto-detect.")

    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--png-dpi", type=int, default=300)

    # ----------------------------
    # NEW: optional PCA-stage inputs (#4)
    # ----------------------------
    ap.add_argument(
        "--pca-tables-dir",
        default=None,
        help="Optional: path to env_pca/tables/ to auto-load pca_loadings.csv and pc_loading_concentration.csv.",
    )
    ap.add_argument("--pca-loadings", default=None, help="Optional: explicit path to pca_loadings.csv (overrides --pca-tables-dir).")
    ap.add_argument("--pc-loading-concentration", default=None, help="Optional: explicit path to pc_loading_concentration.csv (overrides --pca-tables-dir).")
    ap.add_argument("--pc-loading-top-n", type=int, default=12, help="Top N features per PC for loading plots (default 12).")

    ns = ap.parse_args()
    key_cols = [c.strip() for c in ns.key_cols.split(",") if c.strip()]

    # Resolve optional PCA paths
    pca_tables_dir = ns.pca_tables_dir
    pca_loadings_path = ns.pca_loadings
    pc_loading_concentration_path = ns.pc_loading_concentration

    if pca_tables_dir:
        # Only fill defaults if not explicitly set
        if not pca_loadings_path:
            cand = os.path.join(pca_tables_dir, "pca_loadings.csv")
            if os.path.exists(cand):
                pca_loadings_path = cand
        if not pc_loading_concentration_path:
            cand = os.path.join(pca_tables_dir, "pc_loading_concentration.csv")
            if os.path.exists(cand):
                pc_loading_concentration_path = cand

    return Config(
        matrix_cleaned=ns.matrix_cleaned,
        eigengenes=ns.eigengenes,
        assignments=ns.assignments,
        outdir=ns.outdir,
        sep_matrix=ns.sep_matrix,
        sep_eig=ns.sep_eig,
        sep_assign=ns.sep_assign,

        id_col=ns.id_col,
        key_mode=ns.key_mode,
        key_cols=key_cols,
        key_sep=ns.key_sep,
        derived_key_col="__merge_key__",

        depth_col=ns.depth_col,
        depth_anchored_col=ns.depth_anchored_col,
        date_col=ns.date_col,
        oxygen_col=ns.oxygen_col,
        cruise_col=ns.cruise_col,

        o2_oxic_gt=ns.o2_oxic_gt,
        o2_dysoxic_hi=ns.o2_dysoxic_hi,
        o2_dysoxic_lo=ns.o2_dysoxic_lo,
        o2_suboxic_hi=ns.o2_suboxic_hi,
        o2_suboxic_lo=ns.o2_suboxic_lo,

        umap_n_neighbors=ns.umap_n_neighbors,
        umap_min_dist=ns.umap_min_dist,
        umap_random_state=ns.umap_random_state,
        umap_metric=ns.umap_metric,

        bubble_q_low=ns.bubble_q_low,
        bubble_q_high=ns.bubble_q_high,
        bubble_size_min=ns.bubble_size_min,
        bubble_size_max=ns.bubble_size_max,

        pc_cols=ns.pc_cols,

        save_pdf=not ns.no_pdf,
        save_svg=not ns.no_svg,
        save_png=not ns.no_png,
        png_dpi=ns.png_dpi,

        pca_tables_dir=pca_tables_dir,
        pca_loadings_path=pca_loadings_path,
        pc_loading_concentration_path=pc_loading_concentration_path,

        pc_loading_top_n=ns.pc_loading_top_n,
    )


# ----------------------------
# IO / guards
# ----------------------------

def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def read_table_dedup_cols(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # CRITICAL GUARD: remove duplicate column labels
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_merge_key(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.key_mode == "id":
        if cfg.id_col not in out.columns:
            raise ValueError(f"key-mode=id but id-col not found: {cfg.id_col}")
        out[cfg.derived_key_col] = out[cfg.id_col].astype(str)
        return out

    # composite
    missing = [c for c in cfg.key_cols if c not in out.columns]
    if missing:
        raise ValueError(f"key-mode=composite but missing columns in a table: {missing}")

    # stringify safely
    parts = []
    for c in cfg.key_cols:
        parts.append(out[c].astype(str).fillna("NA"))
    out[cfg.derived_key_col] = parts[0]
    for p in parts[1:]:
        out[cfg.derived_key_col] = out[cfg.derived_key_col] + cfg.key_sep + p
    return out


def sanitize_filename(s: str) -> str:
    s2 = str(s)
    for ch in [" ", "/", "\\", "(", ")", "[", "]", "{", "}", ":", ";", ",", "|", "<", ">", "?", "*", "\"", "'"]:
        s2 = s2.replace(ch, "_")
    while "__" in s2:
        s2 = s2.replace("__", "_")
    return s2.strip("_")


# ----------------------------
# O2 labeling
# ----------------------------

def label_o2_compartment(o2_uM: pd.Series, cfg: Config) -> pd.Series:
    x = pd.to_numeric(o2_uM, errors="coerce")
    out = pd.Series(["NA"] * len(x), index=x.index, dtype="object")
    out[x > cfg.o2_oxic_gt] = "oxic"
    out[(x <= cfg.o2_dysoxic_hi) & (x >= cfg.o2_dysoxic_lo)] = "dysoxic"
    out[(x < cfg.o2_suboxic_hi) & (x >= cfg.o2_suboxic_lo)] = "suboxic"
    out[x < cfg.o2_suboxic_lo] = "anoxic"
    return out


# ----------------------------
# Plot helpers
# ----------------------------

def save_all_formats(fig, basepath_no_ext: str, cfg: Config) -> None:
    fig.tight_layout()
    if cfg.save_pdf:
        fig.savefig(basepath_no_ext + ".pdf")
    if cfg.save_svg:
        fig.savefig(basepath_no_ext + ".svg")
    if cfg.save_png:
        fig.savefig(basepath_no_ext + ".png", dpi=cfg.png_dpi)
    plt.close(fig)


def scatter_depth_profile(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    color_col: str,
    color_map: Dict[str, str],
    title: str,
    out_base: str,
    cfg: Config,
) -> None:
    d = df[[xcol, ycol, color_col]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol])

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    classes = sorted(d[color_col].astype("object").fillna("NA").unique(), key=lambda z: str(z))
    for cls in classes:
        sub = d[d[color_col].astype("object").fillna("NA") == cls]
        ax.scatter(sub[xcol].values, sub[ycol].values, s=14, alpha=0.65, label=str(cls), c=color_map.get(cls, None))

    ax.invert_yaxis()
    ax.set_xlabel(xcol)
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    save_all_formats(fig, out_base, cfg)


def scatter_depth_profile_categorical(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    cat_col: str,
    title: str,
    out_base: str,
    cfg: Config,
) -> None:
    d = df[[xcol, ycol, cat_col]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol])

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    cats = sorted(d[cat_col].astype("object").fillna("NA").unique(), key=lambda z: str(z))
    for cat in cats:
        sub = d[d[cat_col].astype("object").fillna("NA") == cat]
        ax.scatter(sub[xcol].values, sub[ycol].values, s=14, alpha=0.65, label=str(cat))

    ax.invert_yaxis()
    ax.set_xlabel(xcol)
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    save_all_formats(fig, out_base, cfg)


def plot_umap_categorical(
    emb: np.ndarray,
    labels: pd.Series,
    title: str,
    out_base: str,
    cfg: Config,
    palette: Optional[Dict[str, str]] = None,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    lab = labels.astype("object").fillna("NA")
    classes = sorted(lab.unique(), key=lambda z: str(z))
    for cls in classes:
        mask = (lab == cls).to_numpy()
        c = palette.get(cls, None) if palette else None
        ax.scatter(emb[mask, 0], emb[mask, 1], s=14, alpha=0.70, label=str(cls), c=c)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_all_formats(fig, out_base, cfg)


def plot_umap_depth(
    emb: np.ndarray,
    depth: pd.Series,
    title: str,
    out_base: str,
    cfg: Config,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    d = pd.to_numeric(depth, errors="coerce")
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=14, alpha=0.75, c=d, cmap=depth_cmap())
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Depth (m)")
    save_all_formats(fig, out_base, cfg)


def scale_bubble_sizes(
    values: pd.Series,
    q_low: float,
    q_high: float,
    size_min: float,
    size_max: float,
) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy()
    if np.all(np.isnan(x)):
        return np.full_like(x, fill_value=size_min, dtype=float)

    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        s = np.full_like(x, fill_value=(size_min + size_max) / 2.0, dtype=float)
        s[~np.isfinite(x)] = size_min
        return s

    xc = np.clip(x, lo, hi)
    t = (xc - lo) / (hi - lo)
    s = size_min + t * (size_max - size_min)
    s[~np.isfinite(s)] = size_min
    return s.astype(float)


def plot_umap_bubble_single_color(
    emb: np.ndarray,
    values: pd.Series,
    color: str,
    title: str,
    out_base: str,
    cfg: Config,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    sizes = scale_bubble_sizes(values, cfg.bubble_q_low, cfg.bubble_q_high, cfg.bubble_size_min, cfg.bubble_size_max)
    ax.scatter(emb[:, 0], emb[:, 1], s=sizes, alpha=0.55, c=color, edgecolors="none")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    save_all_formats(fig, out_base, cfg)


# ----------------------------
# Stats
# ----------------------------

def safe_silhouette(X: np.ndarray, labels: pd.Series) -> float:
    lab = labels.astype("object").fillna("NA")
    if lab.nunique(dropna=False) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, lab.astype(str).to_numpy(), metric="euclidean"))
    except Exception:
        return float("nan")


def safe_calinski_harabasz(X: np.ndarray, labels: pd.Series) -> float:
    lab = labels.astype("object").fillna("NA")
    if lab.nunique(dropna=False) < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(X, lab.astype(str).to_numpy()))
    except Exception:
        return float("nan")


def safe_davies_bouldin(X: np.ndarray, labels: pd.Series) -> float:
    lab = labels.astype("object").fillna("NA")
    if lab.nunique(dropna=False) < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X, lab.astype(str).to_numpy()))
    except Exception:
        return float("nan")


def confusion_tables(y_true: pd.Series, y_pred: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yt = y_true.astype("object").fillna("NA")
    yp = y_pred.astype("object").fillna("NA")
    raw = pd.crosstab(yt, yp, rownames=["O2_compartment"], colnames=["GMM_component"], dropna=False)
    row_norm = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    col_norm = raw.div(raw.sum(axis=0).replace(0, np.nan), axis=1)
    return raw, row_norm, col_norm


# ----------------------------
# #4 Add-ons: interpretation + effect sizes
# ----------------------------

def compute_group_medians(df: pd.DataFrame, group_col: str, value_cols: List[str]) -> pd.DataFrame:
    rows = []
    g = df[group_col].astype("object").fillna("NA")
    for k in sorted(g.unique(), key=lambda z: str(z)):
        sub = df[g == k]
        out = {"group": str(k), "n": int(sub.shape[0])}
        for c in value_cols:
            out[c] = float(pd.to_numeric(sub[c], errors="coerce").median(skipna=True))
        rows.append(out)
    return pd.DataFrame(rows)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    return float(np.sqrt((va + vb) / 2.0))


def pairwise_effect_sizes(df: pd.DataFrame, group_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    Simple, dependency-free effect sizes:
      d = (median_i - median_j) / pooled_sd
    (uses medians to be robust-ish; pooled SD for scaling)
    """
    g = df[group_col].astype("object").fillna("NA")
    groups = sorted(g.unique(), key=lambda z: str(z))
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi = groups[i]
            gj = groups[j]
            di = df[g == gi]
            dj = df[g == gj]
            for feat in feature_cols:
                xi = pd.to_numeric(di[feat], errors="coerce").to_numpy(dtype=float)
                xj = pd.to_numeric(dj[feat], errors="coerce").to_numpy(dtype=float)
                mi = np.nanmedian(xi)
                mj = np.nanmedian(xj)
                sd = pooled_sd(xi, xj)
                dval = float("nan") if (not np.isfinite(sd) or sd == 0) else float((mi - mj) / sd)
                rows.append({
                    "group_i": str(gi),
                    "group_j": str(gj),
                    "feature": feat,
                    "median_i": float(mi) if np.isfinite(mi) else np.nan,
                    "median_j": float(mj) if np.isfinite(mj) else np.nan,
                    "pooled_sd": sd,
                    "effect_size_d": dval,
                    "abs_effect_size_d": abs(dval) if np.isfinite(dval) else np.nan,
                })
    out = pd.DataFrame(rows)
    out = out.sort_values(["abs_effect_size_d"], ascending=False)
    return out


def maybe_load_pca_tables(cfg: Config) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if cfg.pca_loadings_path and os.path.exists(cfg.pca_loadings_path):
        out["pca_loadings"] = pd.read_csv(cfg.pca_loadings_path)
    if cfg.pc_loading_concentration_path and os.path.exists(cfg.pc_loading_concentration_path):
        out["pc_loading_concentration"] = pd.read_csv(cfg.pc_loading_concentration_path)
    return out


def pca_loadings_to_long(loadings_df: pd.DataFrame) -> pd.DataFrame:
    """
    pca_loadings.csv from your PCA script is typically:
      index column with feature names + PC columns
    but sometimes saved with unnamed index col.
    We'll normalize it.
    """
    df = loadings_df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "feature"})
    if "feature" not in df.columns:
        # assume first col is feature
        df = df.rename(columns={df.columns[0]: "feature"})
    pc_cols = [c for c in df.columns if str(c).startswith("PC")]
    long = df.melt(id_vars=["feature"], value_vars=pc_cols, var_name="PC", value_name="loading")
    long["abs_loading"] = long["loading"].abs()
    return long


def plot_top_loadings_per_pc(load_long: pd.DataFrame, pc: str, top_n: int, out_base: str, cfg: Config) -> None:
    sub = load_long[load_long["PC"] == pc].sort_values("abs_loading", ascending=False).head(top_n)
    if sub.empty:
        return
    # keep signed loadings; order by signed value for plot readability
    sub2 = sub.copy().sort_values("loading")
    fig = plt.figure(figsize=(8.5, max(4, 0.30 * len(sub2))))
    ax = plt.gca()
    ax.barh(sub2["feature"].astype(str), sub2["loading"].to_numpy(dtype=float))
    ax.set_xlabel("Loading")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {len(sub2)} PCA loadings for {pc}")
    save_all_formats(fig, out_base, cfg)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    # Load (with dedup col guard)
    df_matrix = read_table_dedup_cols(cfg.matrix_cleaned, cfg.sep_matrix)
    df_eig = read_table_dedup_cols(cfg.eigengenes, cfg.sep_eig)
    df_assign = read_table_dedup_cols(cfg.assignments, cfg.sep_assign)

    # Datetimes
    df_matrix = coerce_datetime(df_matrix, cfg.date_col)
    df_eig = coerce_datetime(df_eig, cfg.date_col)
    df_assign = coerce_datetime(df_assign, cfg.date_col)

    # Build merge keys
    df_matrix = build_merge_key(df_matrix, cfg)
    df_eig = build_merge_key(df_eig, cfg)
    df_assign = build_merge_key(df_assign, cfg)

    # Detect PC columns
    if cfg.pc_cols:
        pc_cols = [c.strip() for c in cfg.pc_cols.split(",") if c.strip()]
    else:
        pc_cols = [c for c in df_eig.columns if str(c).startswith("PC")]

        def _pc_key(x: str) -> int:
            try:
                return int(str(x).replace("PC", ""))
            except Exception:
                return 9999

        pc_cols = sorted(pc_cols, key=_pc_key)

    if len(pc_cols) == 0:
        raise ValueError("No PC columns found in eigengenes (expected PC1, PC2, ...)")

    # Metadata columns expected in matrix
    meta_cols = [cfg.cruise_col, "Year", "Month", "Day", cfg.depth_col, cfg.date_col, cfg.depth_anchored_col]
    meta_cols_present = [c for c in meta_cols if c in df_matrix.columns]

    # Biochem columns = everything in matrix except merge key + meta + legacy id if present
    exclude = set([cfg.derived_key_col] + meta_cols_present + [cfg.id_col])
    biochem_cols = [c for c in df_matrix.columns if c not in exclude]

    if cfg.oxygen_col not in df_matrix.columns:
        raise ValueError(f"matrix_cleaned missing oxygen col: {cfg.oxygen_col}")
    if cfg.depth_anchored_col not in df_matrix.columns:
        raise ValueError(f"matrix_cleaned missing anchored depth col: {cfg.depth_anchored_col}")

    # Merge master
    keep_matrix = [cfg.derived_key_col] + meta_cols_present + biochem_cols
    keep_eig = [cfg.derived_key_col] + pc_cols

    m = df_assign.merge(df_matrix[keep_matrix], on=cfg.derived_key_col, how="left", suffixes=("", "_matrix"))
    m = m.merge(df_eig[keep_eig], on=cfg.derived_key_col, how="left", suffixes=("", "_eig"))

    # Label O2
    m["o2_compartment"] = label_o2_compartment(m[cfg.oxygen_col], cfg)
    m[cfg.depth_anchored_col] = pd.to_numeric(m[cfg.depth_anchored_col], errors="coerce")

    # Save merged table
    m.to_csv(os.path.join(tables_dir, "merged_for_comparison.csv"), index=False)

    # Save config
    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "umap_available": _HAVE_UMAP}, f, indent=2)

    # ----------------------------
    # A) Depth profiles: biochem
    # ----------------------------
    for feat in biochem_cols:
        x = pd.to_numeric(m[feat], errors="coerce")
        if x.notna().sum() < 3:
            continue
        scatter_depth_profile(
            df=m,
            xcol=feat,
            ycol=cfg.depth_anchored_col,
            color_col="o2_compartment",
            color_map=O2_COMPARTMENT_PALETTE,
            title=f"Depth vs {feat} (color = O2 compartment)",
            out_base=os.path.join(plots_dir, f"A1_depth_vs_{sanitize_filename(feat)}__color_o2"),
            cfg=cfg,
        )
        scatter_depth_profile_categorical(
            df=m,
            xcol=feat,
            ycol=cfg.depth_anchored_col,
            cat_col="component",
            title=f"Depth vs {feat} (color = GMM compartment)",
            out_base=os.path.join(plots_dir, f"A2_depth_vs_{sanitize_filename(feat)}__color_gmm"),
            cfg=cfg,
        )

    # ----------------------------
    # A) Depth profiles: PCs
    # ----------------------------
    for pc in pc_cols:
        x = pd.to_numeric(m[pc], errors="coerce")
        if x.notna().sum() < 3:
            continue
        scatter_depth_profile(
            df=m,
            xcol=pc,
            ycol=cfg.depth_anchored_col,
            color_col="o2_compartment",
            color_map=O2_COMPARTMENT_PALETTE,
            title=f"Depth vs {pc} (color = O2 compartment)",
            out_base=os.path.join(plots_dir, f"A3_depth_vs_{sanitize_filename(pc)}__color_o2"),
            cfg=cfg,
        )
        scatter_depth_profile_categorical(
            df=m,
            xcol=pc,
            ycol=cfg.depth_anchored_col,
            cat_col="component",
            title=f"Depth vs {pc} (color = GMM compartment)",
            out_base=os.path.join(plots_dir, f"A4_depth_vs_{sanitize_filename(pc)}__color_gmm"),
            cfg=cfg,
        )

    # ----------------------------
    # Agreement stats (existing)
    # ----------------------------
    raw, row_norm, col_norm = confusion_tables(m["o2_compartment"], m["component"])
    raw.to_csv(os.path.join(tables_dir, "confusion_o2_vs_gmm_raw.csv"))
    row_norm.to_csv(os.path.join(tables_dir, "confusion_o2_vs_gmm_row_norm.csv"))
    col_norm.to_csv(os.path.join(tables_dir, "confusion_o2_vs_gmm_col_norm.csv"))

    yt = m["o2_compartment"].astype("object").fillna("NA")
    yp = m["component"].astype("object").fillna("NA")
    ari = float(adjusted_rand_score(yt.astype(str), yp.astype(str)))
    nmi = float(normalized_mutual_info_score(yt.astype(str), yp.astype(str)))

    # PC-space matrices
    X_pc = m[pc_cols].apply(pd.to_numeric, errors="coerce")
    good_pc = X_pc.notna().all(axis=1)
    X_pc_good = X_pc.loc[good_pc].to_numpy(dtype=float)

    if X_pc_good.shape[0] >= 10:
        X_pc_scaled = StandardScaler().fit_transform(X_pc_good)
        sil_o2_pc = safe_silhouette(X_pc_scaled, yt.loc[good_pc])
        sil_gmm_pc = safe_silhouette(X_pc_scaled, yp.loc[good_pc])
        ch_o2_pc = safe_calinski_harabasz(X_pc_scaled, yt.loc[good_pc])
        ch_gmm_pc = safe_calinski_harabasz(X_pc_scaled, yp.loc[good_pc])
        db_o2_pc = safe_davies_bouldin(X_pc_scaled, yt.loc[good_pc])
        db_gmm_pc = safe_davies_bouldin(X_pc_scaled, yp.loc[good_pc])
    else:
        sil_o2_pc = float("nan")
        sil_gmm_pc = float("nan")
        ch_o2_pc = float("nan")
        ch_gmm_pc = float("nan")
        db_o2_pc = float("nan")
        db_gmm_pc = float("nan")

    pd.DataFrame([{
        "n_total": int(m.shape[0]),
        "n_pc_complete": int(X_pc_good.shape[0]),
        "n_pcs_used": int(len(pc_cols)),
        "ARI_o2_vs_gmm": ari,
        "NMI_o2_vs_gmm": nmi,
        "silhouette_PCspace_o2": sil_o2_pc,
        "silhouette_PCspace_gmm": sil_gmm_pc,
    }]).to_csv(os.path.join(tables_dir, "comparison_stats.csv"), index=False)

    # NEW (#4): richer PC-space quality metrics
    pd.DataFrame([{
        "space": "PCspace_scaled",
        "n_used": int(X_pc_good.shape[0]),
        "k_labels_o2": int(yt.loc[good_pc].nunique(dropna=False)),
        "k_labels_gmm": int(yp.loc[good_pc].nunique(dropna=False)),
        "silhouette_o2": sil_o2_pc,
        "silhouette_gmm": sil_gmm_pc,
        "calinski_harabasz_o2": ch_o2_pc,
        "calinski_harabasz_gmm": ch_gmm_pc,
        "davies_bouldin_o2": db_o2_pc,
        "davies_bouldin_gmm": db_gmm_pc,
    }]).to_csv(os.path.join(tables_dir, "quality_metrics_pcspace.csv"), index=False)

    # ----------------------------
    # NEW (#4): biochem-space quality metrics + medians + effect sizes
    # ----------------------------
    biochem_numeric = m[biochem_cols].apply(pd.to_numeric, errors="coerce")
    good_bio = biochem_numeric.notna().all(axis=1)
    m_bio = m.loc[good_bio].copy()
    X_bio = biochem_numeric.loc[good_bio].to_numpy(dtype=float)

    if X_bio.shape[0] >= 10:
        X_bio_scaled = StandardScaler().fit_transform(X_bio)
        sil_o2_bio = safe_silhouette(X_bio_scaled, m_bio["o2_compartment"])
        sil_gmm_bio = safe_silhouette(X_bio_scaled, m_bio["component"])
        ch_o2_bio = safe_calinski_harabasz(X_bio_scaled, m_bio["o2_compartment"])
        ch_gmm_bio = safe_calinski_harabasz(X_bio_scaled, m_bio["component"])
        db_o2_bio = safe_davies_bouldin(X_bio_scaled, m_bio["o2_compartment"])
        db_gmm_bio = safe_davies_bouldin(X_bio_scaled, m_bio["component"])
    else:
        sil_o2_bio = float("nan")
        sil_gmm_bio = float("nan")
        ch_o2_bio = float("nan")
        ch_gmm_bio = float("nan")
        db_o2_bio = float("nan")
        db_gmm_bio = float("nan")

    pd.DataFrame([{
        "space": "biochem_scaled_all_complete",
        "n_used": int(X_bio.shape[0]),
        "n_features": int(len(biochem_cols)),
        "silhouette_o2": sil_o2_bio,
        "silhouette_gmm": sil_gmm_bio,
        "calinski_harabasz_o2": ch_o2_bio,
        "calinski_harabasz_gmm": ch_gmm_bio,
        "davies_bouldin_o2": db_o2_bio,
        "davies_bouldin_gmm": db_gmm_bio,
    }]).to_csv(os.path.join(tables_dir, "quality_metrics_biochem_space.csv"), index=False)

    # Per-compartment medians (biochem + PC)
    value_cols_for_medians = biochem_cols + pc_cols
    med_gmm = compute_group_medians(m, "component", value_cols_for_medians)
    med_o2 = compute_group_medians(m, "o2_compartment", value_cols_for_medians)
    med_gmm.to_csv(os.path.join(tables_dir, "compartment_medians_gmm.csv"), index=False)
    med_o2.to_csv(os.path.join(tables_dir, "compartment_medians_o2.csv"), index=False)

    # Pairwise effect sizes for GMM separation (biochem only)
    eff = pairwise_effect_sizes(m, "component", biochem_cols)
    eff.to_csv(os.path.join(tables_dir, "gmm_pairwise_feature_effect_sizes.csv"), index=False)

    # ----------------------------
    # UMAP (existing)
    # ----------------------------
    if _HAVE_UMAP:
        if X_bio.shape[0] >= 10:
            reducer = umap.UMAP(
                n_neighbors=cfg.umap_n_neighbors,
                min_dist=cfg.umap_min_dist,
                metric=cfg.umap_metric,
                random_state=cfg.umap_random_state,
            )
            emb = reducer.fit_transform(StandardScaler().fit_transform(X_bio))

            umap_df = pd.DataFrame({
                cfg.derived_key_col: m_bio[cfg.derived_key_col].values,
                "UMAP1": emb[:, 0],
                "UMAP2": emb[:, 1],
                "o2_compartment": m_bio["o2_compartment"].values,
                "component": m_bio["component"].values,
                cfg.depth_anchored_col: pd.to_numeric(m_bio[cfg.depth_anchored_col], errors="coerce").values,
            })
            umap_df.to_csv(os.path.join(tables_dir, "umap_embedding.csv"), index=False)

            plot_umap_categorical(
                emb=emb,
                labels=m_bio["o2_compartment"],
                title=f"UMAP (metric={cfg.umap_metric}) colored by O2 compartments",
                out_base=os.path.join(plots_dir, "B1_umap_color_o2"),
                cfg=cfg,
                palette=O2_COMPARTMENT_PALETTE,
            )
            plot_umap_categorical(
                emb=emb,
                labels=m_bio["component"],
                title=f"UMAP (metric={cfg.umap_metric}) colored by GMM compartments",
                out_base=os.path.join(plots_dir, "B2_umap_color_gmm"),
                cfg=cfg,
                palette=None,
            )
            plot_umap_depth(
                emb=emb,
                depth=m_bio[cfg.depth_anchored_col],
                title=f"UMAP (metric={cfg.umap_metric}) colored by depth",
                out_base=os.path.join(plots_dir, "B3_umap_color_depth"),
                cfg=cfg,
            )

            for feat in biochem_cols:
                v = pd.to_numeric(m_bio[feat], errors="coerce")
                if v.notna().sum() < 3:
                    continue
                color = BIOCHEM_COLOR_MAP.get(feat, "#4A4A4A")
                plot_umap_bubble_single_color(
                    emb=emb,
                    values=v,
                    color=color,
                    title=f"UMAP bubble: {feat} (size=value; color fixed)",
                    out_base=os.path.join(plots_dir, f"B4_umap_bubble_{sanitize_filename(feat)}"),
                    cfg=cfg,
                )
        else:
            with open(os.path.join(cfg.outdir, "UMAP_SKIPPED_NOT_ENOUGH_COMPLETE_ROWS.txt"), "w") as f:
                f.write("UMAP skipped: not enough rows with complete biochem data.\n")
    else:
        with open(os.path.join(cfg.outdir, "UMAP_NOT_AVAILABLE.txt"), "w") as f:
            f.write("UMAP skipped: install umap-learn (pip install umap-learn) then rerun.\n")

    # ----------------------------
    # NEW (#4): optional PCA loadings interpretation (if available)
    # ----------------------------
    pca_tables = maybe_load_pca_tables(cfg)
    if "pca_loadings" in pca_tables:
        load_long = pca_loadings_to_long(pca_tables["pca_loadings"])

        # Keep only PCs we actually used in eigengenes (intersection)
        pcs_present = sorted(set(load_long["PC"].unique()).intersection(set(pc_cols)), key=lambda x: int(str(x).replace("PC", "")))
        top_rows = []
        for pc in pcs_present:
            sub = load_long[load_long["PC"] == pc].sort_values("abs_loading", ascending=False).head(cfg.pc_loading_top_n)
            for _, r in sub.iterrows():
                top_rows.append({
                    "PC": pc,
                    "feature": r["feature"],
                    "loading": float(r["loading"]),
                    "abs_loading": float(r["abs_loading"]),
                })
            plot_top_loadings_per_pc(
                load_long=load_long,
                pc=pc,
                top_n=cfg.pc_loading_top_n,
                out_base=os.path.join(plots_dir, f"C1_top_loadings_{sanitize_filename(pc)}"),
                cfg=cfg,
            )

        pd.DataFrame(top_rows).to_csv(os.path.join(tables_dir, "pc_top_loadings.csv"), index=False)

    if "pc_loading_concentration" in pca_tables:
        # Merge (best-effort) concentration metrics with whatever PCs we used
        conc = pca_tables["pc_loading_concentration"].copy()
        conc_cols = [c for c in conc.columns if c in ["PC", "entropy_norm", "effective_features", "participation_ratio"]]
        if "PC" in conc.columns and conc_cols:
            conc2 = conc[conc_cols].copy()
            conc2 = conc2[conc2["PC"].astype(str).isin([str(x) for x in pc_cols])]
            conc2.to_csv(os.path.join(tables_dir, "pc_loading_concentration_merged.csv"), index=False)

    # ----------------------------
    # Final print
    # ----------------------------
    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}")
    print(f"     key_mode={cfg.key_mode}  (derived key col: {cfg.derived_key_col})")
    print(f"     Agreement: ARI={ari:.3f}, NMI={nmi:.3f}")
    print(f"     Quality (PCspace): silhouette O2={sil_o2_pc:.3f}, GMM={sil_gmm_pc:.3f}")
    print(f"     Quality (Biochem): silhouette O2={sil_o2_bio:.3f}, GMM={sil_gmm_bio:.3f}")
    if cfg.pca_tables_dir:
        print(f"     PCA tables dir: {cfg.pca_tables_dir}")
        if cfg.pca_loadings_path and os.path.exists(cfg.pca_loadings_path):
            print(f"       loaded: pca_loadings.csv")
        if cfg.pc_loading_concentration_path and os.path.exists(cfg.pc_loading_concentration_path):
            print(f"       loaded: pc_loading_concentration.csv")


if __name__ == "__main__":
    main()
