# File: SPARK/env_split_o2_by_gmm.py
#
# What this does
# --------------
# 1) Compute O2 compartments from matrix_cleaned (oxic/dysoxic/suboxic/anoxic).
# 2) Define subcompartments as the intersection: O2 compartment × GMM component.
# 3) Optionally collapse tiny intersections to "<o2>__other" (reporting convenience).
# 4) Optionally reassign "borderline" samples (typically in "<o2>__other") into the nearest
#    *core* subcompartment within the SAME O2 compartment using standardized PC space centroids,
#    with a conservative radius gate derived from within-core dispersion.
#
# Inputs
# ------
# - matrix_cleaned.csv : Oxygen + metadata (for O2 compartment labeling)
# - eigenvectors_scores.csv : PC columns used for centroid distance calculations
# - compartments_assignments_smoothed.csv : GMM component labels + max_prob (optional)
#
# Outputs
# -------
# --outdir/
#   run_config.json
#   tables/
#     merged_o2_split_by_gmm.csv
#     o2_subcompartment_counts_before.csv
#     o2_subcompartment_counts_after.csv
#     o2_by_gmm_confusion_raw.csv
#     o2_by_gmm_confusion_row_norm.csv
#     o2_by_gmm_confusion_col_norm.csv
#     reassignment_qc_summary.csv
#     reassignment_centroids.csv
#     reassignment_cluster_radii.csv
#
# Notes
# -----
# - No re-fitting of clustering models. This is post-hoc hierarchical labeling only.
# - Reassignment is constrained strictly within each O2 compartment.
# - Reassignment is gated by an empirical radius: the q-quantile of within-core distances.
#
# Extensions in this script compared to env_split_o2_by_gmm.py
# Adds to the previous script:
#  1) Summary plots per O2 compartment (PC-space + biochem overlays)
#  2) Depth-profile visualizations colored by subcompartment (y-axis inverted: shallow at top)
#  3) Within-O2 silhouettes (unweighted + weighted by max_prob) computed in PC space
#
# Usage: see example command at bottom of file docstring.

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


# ----------------------------
# Palettes (user-specified)
# ----------------------------

O2_COMPARTMENT_PALETTE = {
    "oxic": "red",
    "dysoxic": "green",
    "suboxic": "lightblue",
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


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    matrix_cleaned: str
    eigenvectors: str
    umap_embedding: str
    assignments: str
    outdir: str
    sep_matrix: str
    sep_eig: str
    sep_assign: str

    # keying
    id_col: str
    key_mode: str
    key_cols: List[str]
    key_sep: str
    derived_key_col: str

    # core cols
    oxygen_col: str
    cruise_col: str
    depth_col: str
    depth_anchored_col: str
    date_col: str

    # O2 thresholds (µM)
    o2_oxic_gt: float
    o2_dysoxic_hi: float
    o2_dysoxic_lo: float
    o2_suboxic_hi: float
    o2_suboxic_lo: float

    # GMM columns
    gmm_component_col: str
    max_prob_col: str

    # PC columns
    pc_cols: List[str]

    # labeling
    sub_label_sep: str
    prefix_gmm: str

    # collapse tiny intersections
    min_subcluster_size: int

    # reassignment
    do_reassign: bool
    borderline_mode: str
    borderline_max_prob: Optional[float]
    core_min_prob: Optional[float]
    reassign_radius_quantile: float
    reassign_min_core_n: int

    # plotting
    do_plots: bool
    plot_formats: List[str]
    png_dpi: int
    point_size: float
    alpha: float


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Hierarchical labeling: O2 compartments split by GMM components with optional centroid reassignment, plots, and within-O2 silhouettes."
    )

    ap.add_argument("--matrix-cleaned", required=True)
    ap.add_argument("--eigenvectors", required=True)
    ap.add_argument("--umap-embedding", required=True)
    ap.add_argument("--assignments", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--sep-matrix", default=",")
    ap.add_argument("--sep-eig", default=",")
    ap.add_argument("--sep-assign", default=",")

    ap.add_argument("--id-col", default="cruise_year_month_depth")
    ap.add_argument("--key-mode", choices=["composite", "id"], default="composite")
    ap.add_argument("--key-cols", default="Cruise,Year,Month,Day,Depth")
    ap.add_argument("--key-sep", default="|")

    ap.add_argument("--oxygen-col", default="Oxygen")
    ap.add_argument("--cruise-col", default="Cruise")
    ap.add_argument("--depth-col", default="Depth")
    ap.add_argument("--depth-anchored-col", default="Depth_anchored")
    ap.add_argument("--date-col", default="date")

    ap.add_argument("--o2-oxic-gt", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-hi", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-lo", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-hi", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-lo", type=float, default=1.0)

    ap.add_argument("--gmm-component-col", default="component")
    ap.add_argument("--max-prob-col", default="max_prob")

    ap.add_argument("--pc-cols", default="PC1,PC2,PC3")

    ap.add_argument("--sub-label-sep", default="__")
    ap.add_argument("--prefix-gmm", default="gmm")

    ap.add_argument("--min-subcluster-size", type=int, default=10)

    ap.add_argument("--reassign", action="store_true")
    ap.add_argument(
        "--borderline-mode",
        choices=["other_only", "low_conf_only", "other_or_low_conf"],
        default="other_only",
    )
    ap.add_argument("--borderline-max-prob", type=float, default=None)
    ap.add_argument("--core-min-prob", type=float, default=None)
    ap.add_argument("--reassign-radius-quantile", type=float, default=0.90)
    ap.add_argument("--reassign-min-core-n", type=int, default=20)

    ap.add_argument("--plots", action="store_true", help="Write summary plots per O2 compartment.")
    ap.add_argument(
        "--plot-formats",
        default="png",
        help="Comma-separated list: png,pdf,svg (default png).",
    )
    ap.add_argument("--png-dpi", type=int, default=300)
    ap.add_argument("--point-size", type=float, default=18.0)
    ap.add_argument("--alpha", type=float, default=0.85)

    ns = ap.parse_args()
    key_cols = [c.strip() for c in ns.key_cols.split(",") if c.strip()]
    pc_cols = [c.strip() for c in ns.pc_cols.split(",") if c.strip()]
    plot_formats = [f.strip().lower() for f in ns.plot_formats.split(",") if f.strip()]

    return Config(
        matrix_cleaned=ns.matrix_cleaned,
        eigenvectors=ns.eigenvectors,
        assignments=ns.assignments,
        umap_embedding=ns.umap_embedding,
        outdir=ns.outdir,
        sep_matrix=ns.sep_matrix,
        sep_eig=ns.sep_eig,
        sep_assign=ns.sep_assign,
        id_col=ns.id_col,
        key_mode=ns.key_mode,
        key_cols=key_cols,
        key_sep=ns.key_sep,
        derived_key_col="__merge_key__",
        oxygen_col=ns.oxygen_col,
        cruise_col=ns.cruise_col,
        depth_col=ns.depth_col,
        depth_anchored_col=ns.depth_anchored_col,
        date_col=ns.date_col,
        o2_oxic_gt=ns.o2_oxic_gt,
        o2_dysoxic_hi=ns.o2_dysoxic_hi,
        o2_dysoxic_lo=ns.o2_dysoxic_lo,
        o2_suboxic_hi=ns.o2_suboxic_hi,
        o2_suboxic_lo=ns.o2_suboxic_lo,
        gmm_component_col=ns.gmm_component_col,
        max_prob_col=ns.max_prob_col,
        pc_cols=pc_cols,
        sub_label_sep=ns.sub_label_sep,
        prefix_gmm=ns.prefix_gmm,
        min_subcluster_size=int(ns.min_subcluster_size),
        do_reassign=bool(ns.reassign),
        borderline_mode=str(ns.borderline_mode),
        borderline_max_prob=ns.borderline_max_prob,
        core_min_prob=ns.core_min_prob,
        reassign_radius_quantile=float(ns.reassign_radius_quantile),
        reassign_min_core_n=int(ns.reassign_min_core_n),
        do_plots=bool(ns.plots),
        plot_formats=plot_formats,
        png_dpi=int(ns.png_dpi),
        point_size=float(ns.point_size),
        alpha=float(ns.alpha),
    )


# ----------------------------
# IO / helpers
# ----------------------------

def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


def read_table_dedup_cols(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def read_umap_embedding(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"__merge_key__", "UMAP1", "UMAP2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UMAP embedding missing columns: {missing}")
    return df

def _normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype("object").fillna("NA").astype(str).str.strip()

def _series_equal_tolerant(a: pd.Series, b: pd.Series, rtol=1e-6, atol=1e-9) -> bool:
    # Try numeric compare first
    an = pd.to_numeric(a, errors="coerce")
    bn = pd.to_numeric(b, errors="coerce")

    both_numeric = an.notna() & bn.notna()
    both_nan = an.isna() & bn.isna()

    # For numeric-overlap rows, compare with tolerance
    if both_numeric.any():
        if not np.allclose(an[both_numeric].to_numpy(), bn[both_numeric].to_numpy(), rtol=rtol, atol=atol):
            return False

    # For rows not both numeric, compare normalized strings
    other = ~(both_numeric | both_nan)
    if other.any():
        if not _normalize_str_series(a[other]).equals(_normalize_str_series(b[other])):
            return False

    return True

def coalesce_merge_suffix_columns(
    df: pd.DataFrame,
    suffixes=("_x", "_y"),
    prefer="x",
    rtol=1e-6,
    atol=1e-9,
    fail_on_mismatch=True,
) -> pd.DataFrame:
    """
    Coalesce columns produced by pandas merge suffixes.
    For each base col where base_x and base_y exist:
      - if identical (tolerant), keep single base col (prefer x or y)
      - else: raise (default) or keep both.
    """
    out = df.copy()
    sx, sy = suffixes

    # find candidate bases
    bases = []
    for c in out.columns:
        if c.endswith(sx):
            base = c[: -len(sx)]
            if (base + sy) in out.columns:
                bases.append(base)

    for base in bases:
        cx = base + sx
        cy = base + sy

        equal = _series_equal_tolerant(out[cx], out[cy], rtol=rtol, atol=atol)
        if not equal:
            if fail_on_mismatch:
                # show a few example mismatches to debug quickly
                ax = out[cx]
                ay = out[cy]
                # build a mismatch mask (string compare fallback)
                nx = pd.to_numeric(ax, errors="coerce")
                ny = pd.to_numeric(ay, errors="coerce")
                both_num = nx.notna() & ny.notna()
                mm = pd.Series(False, index=out.index)
                mm[both_num] = ~np.isclose(nx[both_num], ny[both_num], rtol=rtol, atol=atol)
                other = ~both_num & ~(nx.isna() & ny.isna())
                mm[other] = _normalize_str_series(ax[other]) != _normalize_str_series(ay[other])
                examples = out.loc[mm, [cx, cy]].head(10)
                raise ValueError(
                    f"Merge produced non-identical duplicate columns for base='{base}': {cx} vs {cy}\n"
                    f"Examples (first 10 mismatches):\n{examples.to_string(index=False)}"
                )
            else:
                # keep both; continue
                continue

        keep_col = cx if prefer == "x" else cy
        out[base] = out[keep_col]
        out = out.drop(columns=[cx, cy])

    return out

def build_merge_key(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    if cfg.key_mode == "id":
        if cfg.id_col not in out.columns:
            raise ValueError(f"key-mode=id but id-col not found: {cfg.id_col}")
        out[cfg.derived_key_col] = out[cfg.id_col].astype(str)
        return out

    missing = [c for c in cfg.key_cols if c not in out.columns]
    if missing:
        raise ValueError(f"key-mode=composite but missing columns: {missing}")

    parts = [out[c].astype(str).fillna("NA") for c in cfg.key_cols]
    key = parts[0]
    for p in parts[1:]:
        key = key + cfg.key_sep + p
    out[cfg.derived_key_col] = key
    return out


def label_o2_compartment(o2_uM: pd.Series, cfg: Config) -> pd.Series:
    x = pd.to_numeric(o2_uM, errors="coerce")
    out = pd.Series(["NA"] * len(x), index=x.index, dtype="object")
    out[x > cfg.o2_oxic_gt] = "oxic"
    out[(x <= cfg.o2_dysoxic_hi) & (x >= cfg.o2_dysoxic_lo)] = "dysoxic"
    out[(x < cfg.o2_suboxic_hi) & (x >= cfg.o2_suboxic_lo)] = "suboxic"
    out[x < cfg.o2_suboxic_lo] = "anoxic"
    return out


def confusion_tables(y_true: pd.Series, y_pred: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    yt = y_true.astype("object").fillna("NA")
    yp = y_pred.astype("object").fillna("NA")
    raw = pd.crosstab(yt, yp, rownames=["O2_compartment"], colnames=["GMM_component"], dropna=False)
    row_norm = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    col_norm = raw.div(raw.sum(axis=0).replace(0, np.nan), axis=1)
    return raw, row_norm, col_norm


def standardize_pc_space(df: pd.DataFrame, pc_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[pc_cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def _make_sub_label(o2: str, gmm: str, cfg: Config) -> str:
    return f"{o2}{cfg.sub_label_sep}{cfg.prefix_gmm}{gmm}"


# ----------------------------
# Color utilities
# ----------------------------

def _hex_from_rgb01(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    r = int(np.clip(round(r * 255), 0, 255))
    g = int(np.clip(round(g * 255), 0, 255))
    b = int(np.clip(round(b * 255), 0, 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb01_from_any(color: str) -> Tuple[float, float, float]:
    # Let matplotlib do the parsing (named colors + hex)
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _mix(rgb_a: Tuple[float, float, float], rgb_b: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    # linear interpolation
    t = float(np.clip(t, 0.0, 1.0))
    return (rgb_a[0] * (1 - t) + rgb_b[0] * t,
            rgb_a[1] * (1 - t) + rgb_b[1] * t,
            rgb_a[2] * (1 - t) + rgb_b[2] * t)


import colorsys

def _parse_gmm_index_from_label(label: str, cfg: Config) -> Optional[int]:
    """
    Expected labels like: '<o2>__gmm0', '<o2>__gmm12', '<o2>__other'
    Returns integer index if parseable, else None.
    """
    s = str(label)
    if cfg.sub_label_sep not in s:
        return None
    tok = s.split(cfg.sub_label_sep, 1)[1]  # e.g. 'gmm3' or 'other'
    if tok.startswith(cfg.prefix_gmm):
        tail = tok[len(cfg.prefix_gmm):]
        try:
            return int(tail)
        except Exception:
            return None
    return None

def _rgb01(color: str) -> Tuple[float, float, float]:
    import matplotlib.colors as mcolors
    return mcolors.to_rgb(color)

def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    r = int(np.clip(round(r * 255), 0, 255))
    g = int(np.clip(round(g * 255), 0, 255))
    b = int(np.clip(round(b * 255), 0, 255))
    return f"#{r:02X}{g:02X}{b:02X}"

def _variant_hls(base_color: str, t: float, sat_floor: float = 0.50) -> str:
    """
    Hue-locked variants (stays within O2 color family):
      - Hue fixed
      - Saturation fixed but floored (dysoxic can be darker and not washed out)
      - Lightness ramps from dark -> light across t in [0,1]
    """
    r, g, b = _rgb01(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    s2 = max(s, sat_floor)

    # darker + lighter bounds around base lightness
    l_dark = max(0.08, l * 0.40)  # darker (dysoxic can be darker)
    l_light = min(0.92, 1 - (1 - l) * 0.30)

    t = float(np.clip(t, 0.0, 1.0))
    l2 = (1 - t) * l_dark + t * l_light

    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s2)
    return _rgb01_to_hex((r2, g2, b2))

def build_full_subcompartment_palette(
    m: pd.DataFrame,
    cfg: Config,
    sat_floor: float = 0.50,
) -> pd.DataFrame:
    """
    Build a COMPLETE, deterministic palette for every:
      '<o2>__gmm0' ... '<o2>__gmmMax' plus '<o2>__other'

    gmm0 is always darkest; gmmMax always lightest.
    Returns a dataframe with columns: o2_compartment, label, gmm_index, color_hex
    """
    rows = []

    if "o2_compartment" not in m.columns:
        return pd.DataFrame(columns=["o2_compartment", "label", "gmm_index", "color_hex"])

    # Determine max gmm index per O2 from labels if present; fallback to gmm_component column if needed.
    for o2 in sorted(m["o2_compartment"].astype(str).unique(), key=str):
        base = O2_COMPARTMENT_PALETTE.get(o2, "gray")

        # collect gmm indices observed in final labels
        observed = (
            m.loc[m["o2_compartment"].astype(str) == o2, "o2_subcompartment_final"]
            if "o2_subcompartment_final" in m.columns
            else pd.Series([], dtype="object")
        )
        idxs = []
        for lab in observed.astype(str).unique().tolist():
            gi = _parse_gmm_index_from_label(lab, cfg)
            if gi is not None:
                idxs.append(gi)

        # fallback: if none found, try gmm_component numeric parse (optional)
        if len(idxs) == 0 and "gmm_component" in m.columns:
            tmp = m.loc[m["o2_compartment"].astype(str) == o2, "gmm_component"].astype(str).unique().tolist()
            for t in tmp:
                try:
                    idxs.append(int(t))
                except Exception:
                    pass

        max_g = int(max(idxs)) if len(idxs) > 0 else 0

        # build labels gmm0..gmmMax
        if max_g < 0:
            max_g = 0
        gmm_labels = [f"{o2}{cfg.sub_label_sep}{cfg.prefix_gmm}{k}" for k in range(0, max_g + 1)]

        # color ramp across gmm labels
        n = max(len(gmm_labels), 1)
        ts = np.linspace(0.0, 1.0, num=n).tolist()  # gmm0 darkest -> gmmMax lightest
        for lab, t in zip(gmm_labels, ts):
            rows.append(
                {
                    "o2_compartment": o2,
                    "label": lab,
                    "gmm_index": int(_parse_gmm_index_from_label(lab, cfg)),
                    "color_hex": _variant_hls(base, float(t), sat_floor=sat_floor),
                }
            )

        # add 'other' as lightest variant (t=1.0)
        other_lab = f"{o2}{cfg.sub_label_sep}other"
        rows.append(
            {
                "o2_compartment": o2,
                "label": other_lab,
                "gmm_index": np.nan,
                "color_hex": "#FFFFFF",
            }
        )

    return pd.DataFrame(rows)

def palette_df_to_dict(pal_df: pd.DataFrame) -> Dict[str, str]:
    return {str(r["label"]): str(r["color_hex"]) for _, r in pal_df.iterrows()}


# ----------------------------
# Intersection collapse + reassignment selection
# ----------------------------

def collapse_small_intersections(m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    counts = m["o2_subcompartment"].value_counts(dropna=False)
    small = set(counts[counts < cfg.min_subcluster_size].index.astype(str).tolist())
    if len(small) == 0:
        return m

    def _collapse(lbl: str) -> str:
        if lbl not in small:
            return lbl
        o2 = str(lbl).split(cfg.sub_label_sep)[0]
        return f"{o2}{cfg.sub_label_sep}other"

    out = m.copy()
    out["o2_subcompartment"] = out["o2_subcompartment"].astype(str).map(_collapse)
    return out


def determine_borderline_mask(m: pd.DataFrame, cfg: Config) -> pd.Series:
    is_other = m["o2_subcompartment"].astype(str).str.endswith(f"{cfg.sub_label_sep}other")

    if cfg.borderline_mode == "other_only":
        return is_other

    if cfg.borderline_max_prob is None:
        raise ValueError("--borderline-mode uses low_conf but --borderline-max-prob was not provided.")
    if cfg.max_prob_col not in m.columns:
        raise ValueError("--borderline-mode uses low_conf but max_prob column not present in merged table.")
    mp = pd.to_numeric(m[cfg.max_prob_col], errors="coerce")
    is_low = mp < float(cfg.borderline_max_prob)

    if cfg.borderline_mode == "low_conf_only":
        return is_low

    return is_other | is_low


def compute_core_centroids_and_radii(
    m_pc: pd.DataFrame,
    Xz: np.ndarray,
    cfg: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]]:
    sub = m_pc["o2_subcompartment"].astype(str)
    not_other = ~sub.str.endswith(f"{cfg.sub_label_sep}other")

    core_mask = not_other.copy()
    if cfg.core_min_prob is not None:
        if cfg.max_prob_col not in m_pc.columns:
            raise ValueError("--core-min-prob set but max_prob column not present.")
        mp = pd.to_numeric(m_pc[cfg.max_prob_col], errors="coerce")
        core_mask = core_mask & (mp >= float(cfg.core_min_prob))

    centroid_map: Dict[str, np.ndarray] = {}
    radius_map: Dict[str, float] = {}

    rows_centroids = []
    rows_radii = []

    for lab in sorted(sub[not_other].unique(), key=str):
        idx_all = np.where((sub == lab).to_numpy())[0]
        idx_core = np.where((sub == lab).to_numpy() & core_mask.to_numpy())[0]

        n_all = int(len(idx_all))
        n_core = int(len(idx_core))

        if n_core < cfg.reassign_min_core_n:
            continue

        c = np.nanmean(Xz[idx_core, :], axis=0)
        d = np.sqrt(np.sum((Xz[idx_core, :] - c) ** 2, axis=1))
        r = float(np.nanquantile(d, cfg.reassign_radius_quantile))

        centroid_map[str(lab)] = c
        radius_map[str(lab)] = r

        o2 = str(lab).split(cfg.sub_label_sep)[0]
        rows_radii.append(
            {
                "o2_compartment": o2,
                "o2_subcompartment": str(lab),
                "n_all": n_all,
                "n_core": n_core,
                "radius_quantile": cfg.reassign_radius_quantile,
                "radius": r,
            }
        )
        for j, pc in enumerate(cfg.pc_cols):
            rows_centroids.append(
                {
                    "o2_compartment": o2,
                    "o2_subcompartment": str(lab),
                    "pc": pc,
                    "centroid_z": float(c[j]),
                    "n_core": n_core,
                }
            )

    return pd.DataFrame(rows_centroids), pd.DataFrame(rows_radii), centroid_map, radius_map


def reassign_borderline(
    m_pc: pd.DataFrame,
    Xz: np.ndarray,
    cfg: Config,
    centroid_map: Dict[str, np.ndarray],
    radius_map: Dict[str, float],
    borderline_mask_pc: pd.Series
) -> pd.DataFrame:
    out = m_pc.copy()
    out["o2_subcompartment_before_reassign"] = out["o2_subcompartment"].astype(str)
    out["o2_subcompartment_after_reassign"] = out["o2_subcompartment"].astype(str)
    out["o2_subcompartment_final"] = out["o2_subcompartment"].astype(str)

    out["reassigned"] = False
    out["reassign_target"] = ""
    out["reassign_dist"] = np.nan
    out["reassign_radius"] = np.nan
    out["reassign_accept"] = False

    by_o2: Dict[str, List[str]] = {}
    for lab in centroid_map.keys():
        o2 = str(lab).split(cfg.sub_label_sep)[0]
        by_o2.setdefault(o2, []).append(lab)

    b_idx = np.where(borderline_mask_pc.to_numpy())[0]
    if len(b_idx) == 0:
        return out

    o2_series = out["o2_compartment"].astype(str).to_numpy()

    for i in b_idx:
        o2 = str(o2_series[i])
        candidates = by_o2.get(o2, [])
        if len(candidates) == 0:
            continue

        x = Xz[i, :]
        best_lab = None
        best_dist = None

        for lab in candidates:
            c = centroid_map[lab]
            d = float(np.sqrt(np.sum((x - c) ** 2)))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_lab = lab

        if best_lab is None or best_dist is None:
            continue

        r = float(radius_map.get(best_lab, np.nan))
        accept = bool(np.isfinite(r) and (best_dist <= r))

        out.iloc[i, out.columns.get_loc("reassign_target")] = str(best_lab)
        out.iloc[i, out.columns.get_loc("reassign_dist")] = float(best_dist)
        out.iloc[i, out.columns.get_loc("reassign_radius")] = float(r)
        out.iloc[i, out.columns.get_loc("reassign_accept")] = bool(accept)

        if accept:
            out.iloc[i, out.columns.get_loc("o2_subcompartment_after_reassign")] = str(best_lab)
            out.iloc[i, out.columns.get_loc("o2_subcompartment_final")] = str(best_lab)
            out.iloc[i, out.columns.get_loc("reassigned")] = True

    return out


# ----------------------------
# Silhouette utilities
# ----------------------------

def weighted_silhouette_precomputed(X: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """
    Robust weighted silhouette:
      - compute Euclidean distance matrix
      - silhouette_score(metric="precomputed", sample_weight=weights)

    Returns np.nan if not computable.
    """
    lab = pd.Series(labels).astype(str).fillna("NA").to_numpy()
    if len(set(lab)) < 2:
        return np.nan

    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(w).any():
        return np.nan
    w = np.where(np.isfinite(w), w, 0.0)
    if np.all(w <= 0):
        return np.nan

    try:
        D = pairwise_distances(X, metric="euclidean")
        return float(silhouette_score(D, lab, metric="precomputed", sample_weight=w))
    except Exception:
        return np.nan


def unweighted_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    lab = pd.Series(labels).astype(str).fillna("NA").to_numpy()
    if len(set(lab)) < 2:
        return np.nan
    try:
        return float(silhouette_score(X, lab, metric="euclidean"))
    except Exception:
        return np.nan


def compute_silhouette_bundle_pcspace(
    m_pc: pd.DataFrame,
    Xz: np.ndarray,
    label_col: str,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Computes unweighted + (optional) weighted silhouette for a given label column in PC space.
    Returns a dict with metrics + a short note when not computable.
    """
    if label_col not in m_pc.columns:
        return {
            "label_col": label_col,
            "n_rows_used": int(len(m_pc)),
            "n_labels": np.nan,
            "silhouette_unweighted": np.nan,
            "silhouette_weighted_max_prob": np.nan,
            "note": "label_col_missing",
        }

    lab = m_pc[label_col].astype("object").fillna("NA").astype(str).to_numpy()
    n_labels = len(set(lab))
    if len(m_pc) < 5:
        return {
            "label_col": label_col,
            "n_rows_used": int(len(m_pc)),
            "n_labels": int(n_labels),
            "silhouette_unweighted": np.nan,
            "silhouette_weighted_max_prob": np.nan,
            "note": "too_few_rows",
        }
    if n_labels < 2:
        return {
            "label_col": label_col,
            "n_rows_used": int(len(m_pc)),
            "n_labels": int(n_labels),
            "silhouette_unweighted": np.nan,
            "silhouette_weighted_max_prob": np.nan,
            "note": "only_one_label",
        }

    sil_u = unweighted_silhouette(Xz, lab)

    sil_w = np.nan
    note = ""
    if weights is not None:
        sil_w = weighted_silhouette_precomputed(Xz, lab, weights)
    else:
        note = "weights_missing"

    return {
        "label_col": label_col,
        "n_rows_used": int(len(m_pc)),
        "n_labels": int(n_labels),
        "silhouette_unweighted": float(sil_u) if np.isfinite(sil_u) else np.nan,
        "silhouette_weighted_max_prob": float(sil_w) if np.isfinite(sil_w) else np.nan,
        "note": note,
    }


# ----------------------------
# Plotting utilities
# ----------------------------

def _savefig_all(fig: plt.Figure, outbase: str, cfg: Config) -> None:
    for fmt in cfg.plot_formats:
        path = f"{outbase}.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=cfg.png_dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")


def _smooth_profile_by_depth_bins(
    depth: np.ndarray,
    x: np.ndarray,
    n_bins: int = 40,
    smooth_window: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth mean profile x(depth), per subcompartment:
      - quantile-bin depths
      - compute mean depth + mean x per bin
      - rolling mean smooth across bins

    Returns (x_smooth, depth_bin_centers)
    """
    d = pd.to_numeric(pd.Series(depth), errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(d) & np.isfinite(v)
    d = d[ok]
    v = v[ok]
    if len(d) < 10:
        return np.array([]), np.array([])

    df = pd.DataFrame({"depth": d, "x": v}).sort_values("depth")

    # robust bins even with uneven depth sampling
    try:
        q = min(n_bins, max(5, len(df) // 5))
        df["bin"] = pd.qcut(df["depth"], q=q, duplicates="drop")
    except Exception:
        return np.array([]), np.array([])

    g = df.groupby("bin", observed=True).agg(
        depth_mean=("depth", "mean"),
        x_mean=("x", "mean"),
        n=("x", "size"),
    ).reset_index(drop=True)

    if len(g) < 5:
        return np.array([]), np.array([])

    w = int(smooth_window)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    if w > len(g):
        w = max(3, (len(g) // 2) * 2 + 1)

    x_smooth = g["x_mean"].rolling(window=w, center=True, min_periods=max(3, w // 3)).mean().to_numpy()
    d_smooth = g["depth_mean"].to_numpy()
    return x_smooth, d_smooth


def plot_pc_scatter_subcompartments(
    df: pd.DataFrame,
    o2: str,
    pc_x: str,
    pc_y: str,
    sub_palette: Dict[str, str],
    plots_dir: str,
    cfg: Config,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    d = df.loc[df["o2_compartment"].astype(str) == o2].copy()
    if len(d) < 3:
        return
    if pc_x not in d.columns or pc_y not in d.columns:
        return

    x = pd.to_numeric(d[pc_x], errors="coerce")
    y = pd.to_numeric(d[pc_y], errors="coerce")
    ok = x.notna() & y.notna()
    d = d.loc[ok].copy()
    if len(d) < 3:
        return

    labs = d["o2_subcompartment_final"].astype(str).to_numpy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    # draw each subcompartment separately for a clean legend
    unique_labs = sorted(pd.unique(labs), key=str)
    for lab in unique_labs:
        msk = (labs == lab)
        col = sub_palette.get(str(lab), "gray")
        ax.scatter(
            d.loc[msk, pc_x],
            d.loc[msk, pc_y],
            s=cfg.point_size,
            alpha=cfg.alpha,
            label=str(lab),
            color=col,            # <-- REQUIRED: force palette color for points
            edgecolors="gray",
            linewidths=0.5,
        )

    # overlay reassigned points, colored by FINAL subcompartment
    if "reassigned" in d.columns:
        rr = d["reassigned"].astype(bool)
        if rr.any():
            d_rr = d.loc[rr].copy()
            rr_labs = d_rr["o2_subcompartment_final"].astype(str).to_numpy()
            for lab in sorted(pd.unique(rr_labs), key=str):
                msk = (rr_labs == lab)
                col_rr = sub_palette.get(str(lab), "gray")
                ax.scatter(
                    d_rr.loc[msk, pc_x],
                    d_rr.loc[msk, pc_y],
                    s=cfg.point_size * 1.2,
                    alpha=1.0,
                    marker="D",
                    linewidths=0.5,
                    color=col_rr,
                    edgecolors="gray",
                    label=f"{lab} (reassigned)",
                )

    ax.set_autoscale_on(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title(f"{o2}: PC scatter by subcompartment")
    present = set(d["o2_subcompartment_final"].astype(str).unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.5,
            label=str(key),
        )
        for key, color in sub_palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(right=0.75)

    outbase = os.path.join(plots_dir, f"o2_{o2}_pc_scatter_subcompartments_{pc_x}_vs_{pc_y}")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)


def plot_pc_scatter_all_subcompartments(
    df: pd.DataFrame,
    pc_x: str,
    pc_y: str,
    sub_palette: Dict[str, str],
    plots_dir: str,
    cfg: Config,
) -> None:
    if pc_x not in df.columns or pc_y not in df.columns:
        return

    d = df.copy()
    x = pd.to_numeric(d[pc_x], errors="coerce")
    y = pd.to_numeric(d[pc_y], errors="coerce")
    ok = x.notna() & y.notna()
    d = d.loc[ok].copy()
    if len(d) < 5:
        return

    labs = d["o2_subcompartment_final"].astype(str).to_numpy()
    unique_labs = sorted(pd.unique(labs), key=str)

    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111)

    for lab in unique_labs:
        msk = (labs == lab)
        col = sub_palette.get(str(lab), "gray")
        ax.scatter(
            d.loc[msk, pc_x],
            d.loc[msk, pc_y],
            s=cfg.point_size,
            alpha=cfg.alpha,
            color=col,
            label=str(lab),
            edgecolors="gray",
            linewidths=0.5,
        )

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title("All subcompartments: PC scatter (EDA)")
    present = set(d["o2_subcompartment_final"].astype(str).unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.5,
            label=str(key),
        )
        for key, color in sub_palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(right=0.70)

    outbase = os.path.join(plots_dir, f"EDA_all_subcompartments_pc_scatter_{pc_x}_vs_{pc_y}")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)


def plot_depth_profile_all_subcompartments(
    df: pd.DataFrame,
    value_col: str,
    depth_col: str,
    sub_palette: Dict[str, str],
    plots_dir: str,
    cfg: Config,
) -> None:
    if value_col not in df.columns or depth_col not in df.columns:
        return

    d = df.copy()
    x = pd.to_numeric(d[value_col], errors="coerce")
    dep = pd.to_numeric(d[depth_col], errors="coerce")
    ok = x.notna() & dep.notna()
    d = d.loc[ok].copy()
    if len(d) < 5:
        return

    labs = d["o2_subcompartment_final"].astype(str).to_numpy()
    unique_labs = sorted(pd.unique(labs), key=str)

    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111)

    for lab in unique_labs:
        msk = (labs == lab)
        col = sub_palette.get(str(lab), "gray")
        ax.scatter(
            d.loc[msk, value_col],
            d.loc[msk, depth_col],
            s=cfg.point_size,
            alpha=cfg.alpha,
            color=col,
            label=str(lab),
            edgecolors="gray",
            linewidths=0.5,
        )

        # Smooth mean "fitted" profile per subcompartment (same color as points)
        xs, ds = _smooth_profile_by_depth_bins(
            d.loc[msk, depth_col].to_numpy(),
            d.loc[msk, value_col].to_numpy(),
            n_bins=40,
            smooth_window=7,
        )
        if len(xs) > 0:
            ax.plot(xs, ds, linewidth=3.0, alpha=0.95, color=col)

    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_ylabel(depth_col)
    ax.set_title(f"All subcompartments: depth profile of {value_col} (EDA)")
    present = set(d["o2_subcompartment_final"].astype(str).unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.5,
            label=str(key),
        )
        for key, color in sub_palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(right=0.70)

    outbase = os.path.join(plots_dir, f"EDA_all_subcompartments_depth_profile_{value_col}")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)


def plot_pc_scatter_biochem_overlay(
    df: pd.DataFrame,
    o2: str,
    pc_x: str,
    pc_y: str,
    feature: str,
    plots_dir: str,
    cfg: Config,
) -> None:
    d = df.loc[df["o2_compartment"].astype(str) == o2].copy()
    if len(d) < 3:
        return
    if pc_x not in d.columns or pc_y not in d.columns or feature not in d.columns:
        return

    x = pd.to_numeric(d[pc_x], errors="coerce")
    y = pd.to_numeric(d[pc_y], errors="coerce")
    v = pd.to_numeric(d[feature], errors="coerce")
    ok = x.notna() & y.notna() & v.notna()
    d = d.loc[ok].copy()
    if len(d) < 3:
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    sc = ax.scatter(
        d[pc_x],
        d[pc_y],
        c=d[feature],
        s=cfg.point_size,
        alpha=cfg.alpha,
    )

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title(f"{o2}: {feature} over PC space")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(feature)

    outbase = os.path.join(plots_dir, f"o2_{o2}_pc_scatter_{feature}_{pc_x}_vs_{pc_y}")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)


def plot_depth_profile_subcompartments(
    df: pd.DataFrame,
    o2: str,
    value_col: str,
    depth_col: str,
    sub_palette: Dict[str, str],
    plots_dir: str,
    cfg: Config,
) -> None:
    """
    Depth profile scatter: x=value_col, y=depth (inverted so shallow at top).
    Colored by o2_subcompartment_final.
    """
    d = df.loc[df["o2_compartment"].astype(str) == o2].copy()
    if len(d) < 3:
        return
    if value_col not in d.columns or depth_col not in d.columns:
        return

    x = pd.to_numeric(d[value_col], errors="coerce")
    dep = pd.to_numeric(d[depth_col], errors="coerce")
    ok = x.notna() & dep.notna()
    d = d.loc[ok].copy()
    if len(d) < 3:
        return

    labs = d["o2_subcompartment_final"].astype(str).to_numpy()
    unique_labs = sorted(pd.unique(labs), key=str)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    for lab in unique_labs:
        msk = (labs == lab)
        col = sub_palette.get(str(lab), "gray")
        ax.scatter(
            d.loc[msk, value_col],
            d.loc[msk, depth_col],
            s=cfg.point_size,
            alpha=cfg.alpha,
            label=str(lab),
            color=col,           # <-- REQUIRED
            edgecolors="gray",
            linewidths=0.5,
        )

        # Smooth mean "fitted" profile per subcompartment (same color as points)
        xs, ds = _smooth_profile_by_depth_bins(
            d.loc[msk, depth_col].to_numpy(),
            d.loc[msk, value_col].to_numpy(),
            n_bins=40,
            smooth_window=7,
        )
        if len(xs) > 0:
            ax.plot(xs, ds, linewidth=3.0, alpha=0.95, color=col)   # <-- same palette color

    # shallow at top
    ax.invert_yaxis()

    ax.set_xlabel(value_col)
    ax.set_ylabel(depth_col)
    ax.set_title(f"{o2}: depth profile colored by subcompartment")
    present = set(d["o2_subcompartment_final"].astype(str).unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.5,
            label=str(key),
        )
        for key, color in sub_palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
    )
    fig.subplots_adjust(right=0.75)

    outbase = os.path.join(plots_dir, f"o2_{o2}_depth_profile_{value_col}_by_subcompartment")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)

def plot_umap_subcompartments(
    m: pd.DataFrame,
    umap_df: pd.DataFrame,
    sub_palette: Dict[str, str],
    plots_dir: str,
    cfg: Config,
) -> None:
    # merge UMAP with final labels
    d = umap_df.merge(
        m[["__merge_key__", "o2_subcompartment_final"]],
        on="__merge_key__",
        how="left",
    )

    d["UMAP1"] = pd.to_numeric(d["UMAP1"], errors="coerce")
    d["UMAP2"] = pd.to_numeric(d["UMAP2"], errors="coerce")
    d = d.dropna(subset=["UMAP1", "UMAP2", "o2_subcompartment_final"])

    if len(d) < 5:
        return

    labs = d["o2_subcompartment_final"].astype(str).to_numpy()
    unique_labs = sorted(pd.unique(labs), key=str)

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111)

    for lab in unique_labs:
        msk = (labs == lab)
        col = sub_palette.get(str(lab), "gray")
        ax.scatter(
            d.loc[msk, "UMAP1"],
            d.loc[msk, "UMAP2"],
            s=cfg.point_size,
            alpha=cfg.alpha,
            color=col,
            label=str(lab),
            edgecolors="gray",
            linewidths=0.4,
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP embedding colored by final O₂ × GMM subcompartments")
    present = set(d["o2_subcompartment_final"].astype(str).unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.4,
            label=str(key),
        )
        for key, color in sub_palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        frameon=False,
    )

    fig.subplots_adjust(right=0.72)

    outbase = os.path.join(plots_dir, "umap_final_subcompartments")
    _savefig_all(fig, outbase, cfg)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    df_matrix = read_table_dedup_cols(cfg.matrix_cleaned, cfg.sep_matrix)
    df_eig = read_table_dedup_cols(cfg.eigenvectors, cfg.sep_eig)
    df_assign = read_table_dedup_cols(cfg.assignments, cfg.sep_assign)

    df_matrix = build_merge_key(df_matrix, cfg)
    df_eig = build_merge_key(df_eig, cfg)
    df_assign = build_merge_key(df_assign, cfg)

    if cfg.oxygen_col not in df_matrix.columns:
        raise ValueError(f"matrix_cleaned missing oxygen col: {cfg.oxygen_col}")
    if cfg.gmm_component_col not in df_assign.columns:
        raise ValueError(f"assignments missing GMM component col: {cfg.gmm_component_col}")

    missing_pc = [c for c in cfg.pc_cols if c not in df_eig.columns]
    if missing_pc:
        raise ValueError(f"eigenvectors missing requested pc-cols: {missing_pc}")

    # merge: assignment rows authoritative
    keep_matrix_cols = [cfg.derived_key_col]
    for c in [cfg.oxygen_col, cfg.cruise_col, cfg.depth_col, cfg.depth_anchored_col, cfg.date_col]:
        if c in df_matrix.columns and c not in keep_matrix_cols:
            keep_matrix_cols.append(c)

    keep_eig_cols = [cfg.derived_key_col] + cfg.pc_cols

    m = df_assign.merge(df_matrix[keep_matrix_cols], on=cfg.derived_key_col, how="left")
    m = coalesce_merge_suffix_columns(m, prefer="x")  # keeps assignment-side cols if duplicates

    m = m.merge(df_eig[keep_eig_cols], on=cfg.derived_key_col, how="left")
    m = coalesce_merge_suffix_columns(m, prefer="x")

    # o2 compartments
    m["o2_compartment"] = label_o2_compartment(m[cfg.oxygen_col], cfg)
    m["gmm_component"] = m[cfg.gmm_component_col].astype("object").fillna("NA").astype(str)

    # intersection label
    m["o2_subcompartment"] = [
        _make_sub_label(str(o2), str(g), cfg) for o2, g in zip(m["o2_compartment"].astype(str), m["gmm_component"].astype(str))
    ]

    # collapse tiny intersections
    if cfg.min_subcluster_size > 1:
        m = collapse_small_intersections(m, cfg)

    # baseline counts
    counts_before = (
        m.groupby(["o2_compartment", "o2_subcompartment"], dropna=False)
         .size()
         .reset_index(name="n")
         .sort_values(["o2_compartment", "n"], ascending=[True, False])
    )
    counts_before.to_csv(os.path.join(tables_dir, "o2_subcompartment_counts_before.csv"), index=False)

    # confusion O2 vs GMM
    raw, row_norm, col_norm = confusion_tables(m["o2_compartment"], m["gmm_component"])
    raw.to_csv(os.path.join(tables_dir, "o2_by_gmm_confusion_raw.csv"))
    row_norm.to_csv(os.path.join(tables_dir, "o2_by_gmm_confusion_row_norm.csv"))
    col_norm.to_csv(os.path.join(tables_dir, "o2_by_gmm_confusion_col_norm.csv"))

    # coerce PCs numeric
    for c in cfg.pc_cols:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    pc_complete_mask = m[cfg.pc_cols].notna().all(axis=1)

    # default final labels (no reassignment)
    m["o2_subcompartment_before_reassign"] = m["o2_subcompartment"].astype(str)
    m["o2_subcompartment_after_reassign"] = m["o2_subcompartment"].astype(str)
    m["o2_subcompartment_final"] = m["o2_subcompartment"].astype(str)
    m["reassigned"] = False
    m["reassign_target"] = ""
    m["reassign_dist"] = np.nan
    m["reassign_radius"] = np.nan
    m["reassign_accept"] = False

    qc = {
        "pc_cols_used": cfg.pc_cols,
        "n_rows_total": int(len(m)),
        "n_rows_pc_complete": int(pc_complete_mask.sum()),
        "do_reassign": bool(cfg.do_reassign),
        "borderline_mode": cfg.borderline_mode,
        "borderline_max_prob": cfg.borderline_max_prob,
        "core_min_prob": cfg.core_min_prob,
        "reassign_radius_quantile": cfg.reassign_radius_quantile,
        "reassign_min_core_n": cfg.reassign_min_core_n,
        "n_borderline_candidates": 0,
        "n_reassignable_pc_complete": 0,
        "n_reassigned": 0,
    }

    centroids_df = pd.DataFrame([])
    radii_df = pd.DataFrame([])
    scaler_mu = None
    scaler_sd = None

    if cfg.do_reassign:
        borderline_mask = determine_borderline_mask(m, cfg)
        qc["n_borderline_candidates"] = int(borderline_mask.sum())
        qc["n_reassignable_pc_complete"] = int((borderline_mask & pc_complete_mask).sum())

        m_pc = m.loc[pc_complete_mask].copy()
        Xz, mu, sd = standardize_pc_space(m_pc, cfg.pc_cols)
        scaler_mu = mu
        scaler_sd = sd

        centroids_df, radii_df, centroid_map, radius_map = compute_core_centroids_and_radii(m_pc, Xz, cfg)

        borderline_pc = borderline_mask.loc[m_pc.index]
        m_pc_reassigned = reassign_borderline(m_pc, Xz, cfg, centroid_map, radius_map, borderline_pc)

        cols_pull = [
            "o2_subcompartment_before_reassign",
            "o2_subcompartment_after_reassign",
            "o2_subcompartment_final",
            "reassigned",
            "reassign_target",
            "reassign_dist",
            "reassign_radius",
            "reassign_accept",
        ]
        m.loc[m_pc_reassigned.index, cols_pull] = m_pc_reassigned[cols_pull]
        qc["n_reassigned"] = int(m["reassigned"].sum())

        qc["pc_scaler_mu"] = {pc: float(mu[i]) for i, pc in enumerate(cfg.pc_cols)}
        qc["pc_scaler_sd"] = {pc: float(sd[i]) for i, pc in enumerate(cfg.pc_cols)}

    # counts after reassignment
    counts_after = (
        m.groupby(["o2_compartment", "o2_subcompartment_final"], dropna=False)
         .size()
         .reset_index(name="n")
         .sort_values(["o2_compartment", "n"], ascending=[True, False])
    )
    counts_after.to_csv(os.path.join(tables_dir, "o2_subcompartment_counts_after.csv"), index=False)

    # within-O2 silhouettes in PC space (PC-complete only)
    rows_sil = []
    if pc_complete_mask.sum() >= 5:
        m_pc_all = m.loc[pc_complete_mask].copy()
        Xz_all, _, _ = standardize_pc_space(m_pc_all, cfg.pc_cols)

        weights_all = None
        if cfg.max_prob_col in m_pc_all.columns:
            weights_all = pd.to_numeric(m_pc_all[cfg.max_prob_col], errors="coerce").to_numpy(dtype=float)

        for o2 in sorted(m_pc_all["o2_compartment"].astype(str).unique(), key=str):
            idx = np.where(m_pc_all["o2_compartment"].astype(str).to_numpy() == o2)[0]
            if len(idx) < 5:
                rows_sil.append({
                    "o2_compartment": o2,
                    "n_rows_used": int(len(idx)),
                    "n_subcompartments": np.nan,
                    "silhouette_unweighted": np.nan,
                    "silhouette_weighted_max_prob": np.nan,
                    "note": "too_few_rows",
                })
                continue

            labs = m_pc_all["o2_subcompartment_final"].astype(str).to_numpy()[idx]
            n_labels = len(set(labs))
            if n_labels < 2:
                rows_sil.append({
                    "o2_compartment": o2,
                    "n_rows_used": int(len(idx)),
                    "n_subcompartments": int(n_labels),
                    "silhouette_unweighted": np.nan,
                    "silhouette_weighted_max_prob": np.nan,
                    "note": "only_one_subcompartment",
                })
                continue

            Xu = Xz_all[idx, :]
            sil_u = unweighted_silhouette(Xu, labs)

            sil_w = np.nan
            note = ""
            if weights_all is not None:
                sil_w = weighted_silhouette_precomputed(Xu, labs, weights_all[idx])
            else:
                note = "max_prob_missing"

            rows_sil.append({
                "o2_compartment": o2,
                "n_rows_used": int(len(idx)),
                "n_subcompartments": int(n_labels),
                "silhouette_unweighted": float(sil_u) if np.isfinite(sil_u) else np.nan,
                "silhouette_weighted_max_prob": float(sil_w) if np.isfinite(sil_w) else np.nan,
                "note": note,
            })

    pd.DataFrame(rows_sil).to_csv(os.path.join(tables_dir, "within_o2_silhouette_pcspace.csv"), index=False)

    # ----------------------------------------
    # GLOBAL silhouettes in PC space (PC-complete only)
    #   1) O2 compartments
    #   2) GMM components
    #   3) Combined hierarchical labels (o2_subcompartment_final)
    # ----------------------------------------
    sil_rows_global = []
    counts_rows_global = []

    if pc_complete_mask.sum() >= 5:
        m_pc_all = m.loc[pc_complete_mask].copy()
        Xz_all, _, _ = standardize_pc_space(m_pc_all, cfg.pc_cols)

        weights_all = None
        if cfg.max_prob_col in m_pc_all.columns:
            weights_all = pd.to_numeric(m_pc_all[cfg.max_prob_col], errors="coerce").to_numpy(dtype=float)

        # (A) O2 compartments
        sil_rows_global.append(
            compute_silhouette_bundle_pcspace(
                m_pc_all, Xz_all, label_col="o2_compartment", weights=weights_all
            )
        )
        counts_rows_global.append(
            m_pc_all["o2_compartment"].astype("object").fillna("NA").astype(str).value_counts().rename_axis("label").reset_index(name="n").assign(label_col="o2_compartment")
        )

        # (B) GMM components
        sil_rows_global.append(
            compute_silhouette_bundle_pcspace(
                m_pc_all, Xz_all, label_col="gmm_component", weights=weights_all
            )
        )
        counts_rows_global.append(
            m_pc_all["gmm_component"].astype("object").fillna("NA").astype(str).value_counts().rename_axis("label").reset_index(name="n").assign(label_col="gmm_component")
        )

        # (C) Combined hierarchical labels (FINAL)
        sil_rows_global.append(
            compute_silhouette_bundle_pcspace(
                m_pc_all, Xz_all, label_col="o2_subcompartment_final", weights=weights_all
            )
        )
        counts_rows_global.append(
            m_pc_all["o2_subcompartment_final"].astype("object").fillna("NA").astype(str).value_counts().rename_axis("label").reset_index(name="n").assign(label_col="o2_subcompartment_final")
        )

    # write summary + counts
    pd.DataFrame(sil_rows_global).to_csv(
        os.path.join(tables_dir, "silhouette_pcspace_global_summary.csv"),
        index=False,
    )

    if len(counts_rows_global) > 0:
        pd.concat(counts_rows_global, ignore_index=True).to_csv(
            os.path.join(tables_dir, "silhouette_pcspace_global_label_counts.csv"),
            index=False,
        )
    else:
        pd.DataFrame(columns=["label_col", "label", "n"]).to_csv(
            os.path.join(tables_dir, "silhouette_pcspace_global_label_counts.csv"),
            index=False,
        )

    # Human-friendly name (based on FINAL labels, within each O2 by abundance)
    name_rows = []
    for o2 in sorted(m["o2_compartment"].astype(str).unique(), key=str):
        sub = m.loc[m["o2_compartment"].astype(str) == o2, "o2_subcompartment_final"].astype(str)
        vc = sub.value_counts()
        for i, lab in enumerate(list(vc.index), start=1):
            name_rows.append(
                {
                    "o2_compartment": o2,
                    "o2_subcompartment_final": lab,
                    "o2_subcompartment_name": f"{o2}: subcompartment {i}",
                }
            )
    name_map = pd.DataFrame(name_rows).drop_duplicates(subset=["o2_compartment", "o2_subcompartment_final"])
    m = m.merge(name_map, on=["o2_compartment", "o2_subcompartment_final"], how="left")

    # write merged output
    m.to_csv(os.path.join(tables_dir, "merged_o2_split_by_gmm.csv"), index=False)

    # write qc + centroid/radius tables
    pd.DataFrame([qc]).to_csv(os.path.join(tables_dir, "reassignment_qc_summary.csv"), index=False)

    if len(centroids_df) > 0:
        centroids_df.to_csv(os.path.join(tables_dir, "reassignment_centroids.csv"), index=False)
    else:
        pd.DataFrame(columns=["o2_compartment", "o2_subcompartment", "pc", "centroid_z", "n_core"]).to_csv(
            os.path.join(tables_dir, "reassignment_centroids.csv"), index=False
        )

    if len(radii_df) > 0:
        radii_df.to_csv(os.path.join(tables_dir, "reassignment_cluster_radii.csv"), index=False)
    else:
        pd.DataFrame(columns=["o2_compartment", "o2_subcompartment", "n_all", "n_core", "radius_quantile", "radius"]).to_csv(
            os.path.join(tables_dir, "reassignment_cluster_radii.csv"), index=False
        )

    # ----------------------------
    # Plots (per O2 compartment)
    # ----------------------------
    if cfg.do_plots:

        # Build COMPLETE deterministic palette (covers every '<o2>__gmmK' and '<o2>__other')
        pal_df = build_full_subcompartment_palette(m, cfg, sat_floor=0.50)
        order_map = {v: i for i, v in enumerate(O2_COMPARTMENT_PALETTE.keys())}
        pal_df = pal_df.sort_values(
            by=["o2_compartment", "label"],
            key=lambda s: s.map(order_map) if s.name == "o2_compartment" else s
        )

        pal_df.to_csv(os.path.join(tables_dir, "subcompartment_palette.csv"), index=False)  # audit/stability
        sub_palette = palette_df_to_dict(pal_df)

        # Choose PC axes
        pc_x = cfg.pc_cols[0] if len(cfg.pc_cols) >= 1 else None
        pc_y = cfg.pc_cols[1] if len(cfg.pc_cols) >= 2 else None

        # Depth column to use for profiles (prefer anchored if present)
        depth_for_profile = cfg.depth_anchored_col if cfg.depth_anchored_col in m.columns else cfg.depth_col

        # EDA: all subcompartments together
        if pc_x is not None and pc_y is not None:
            plot_pc_scatter_all_subcompartments(m, pc_x, pc_y, sub_palette, plots_dir, cfg)

        depth_for_profile = cfg.depth_anchored_col if cfg.depth_anchored_col in m.columns else cfg.depth_col

        # EDA: depth profiles for each PC
        for pc in cfg.pc_cols:
            plot_depth_profile_all_subcompartments(m, pc, depth_for_profile, sub_palette, plots_dir, cfg)

        # EDA: depth profiles for each biochem feature present
        for feat in BIOCHEM_COLOR_MAP.keys():
            if feat in m.columns:
                plot_depth_profile_all_subcompartments(m, feat, depth_for_profile, sub_palette, plots_dir, cfg)

        # compute global PC limits across ALL data (or all data you want comparable)
        gx = pd.to_numeric(m[pc_x], errors="coerce")
        gy = pd.to_numeric(m[pc_y], errors="coerce")
        ok = gx.notna() & gy.notna()

        xmin, xmax = gx[ok].min(), gx[ok].max()
        ymin, ymax = gy[ok].min(), gy[ok].max()

        # optional padding
        dx = xmax - xmin
        dy = ymax - ymin
        xlim = (xmin - 0.05 * dx, xmax + 0.05 * dx)
        ylim = (ymin - 0.05 * dy, ymax + 0.05 * dy)

        for o2 in sorted(m["o2_compartment"].astype(str).unique(), key=str):
            # PC scatter by subcompartment
            if pc_x is not None and pc_y is not None:
                plot_pc_scatter_subcompartments(m, o2, pc_x, pc_y, sub_palette, plots_dir, cfg, xlim, ylim)

                # biochem overlays in PC space (only if feature exists)
                for feat in BIOCHEM_COLOR_MAP.keys():
                    if feat in m.columns:
                        plot_pc_scatter_biochem_overlay(m, o2, pc_x, pc_y, feat, plots_dir, cfg)

            # Depth profiles colored by subcompartment:
            # - for each PC used
            for pc in cfg.pc_cols:
                plot_depth_profile_subcompartments(m, o2, pc, depth_for_profile, sub_palette, plots_dir, cfg)

            # - for each biochem feature (if exists)
            for feat in BIOCHEM_COLOR_MAP.keys():
                if feat in m.columns:
                    plot_depth_profile_subcompartments(m, o2, feat, depth_for_profile, sub_palette, plots_dir, cfg)

        # ----------------------------------------
        # UMAP overlay of final subcompartments
        # ----------------------------------------
        if cfg.umap_embedding is not None and os.path.exists(cfg.umap_embedding):
            umap_df = read_umap_embedding(cfg.umap_embedding)
            plot_umap_subcompartments(
                m=m,
                umap_df=umap_df,
                sub_palette=sub_palette,
                plots_dir=plots_dir,
                cfg=cfg,
            )

    # ----------------------------
    # Save config
    # ----------------------------
    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump({"config": cfg.__dict__}, f, indent=2)

    # ----------------------------
    # Console summary
    # ----------------------------
    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    print(f"     Plots : {plots_dir}" if cfg.do_plots else "     Plots : (disabled)")
    print(f"     key_mode={cfg.key_mode}  (derived key col: {cfg.derived_key_col})")
    print(f"     O2 subcompartments: intersection (O2 × GMM), collapse < {cfg.min_subcluster_size} → '<o2>__other'")
    if cfg.do_reassign:
        print(f"     Reassign: enabled  (borderline_mode={cfg.borderline_mode})")
        print(f"       borderline_max_prob={cfg.borderline_max_prob}  core_min_prob={cfg.core_min_prob}")
        print(f"       radius_quantile={cfg.reassign_radius_quantile}  min_core_n={cfg.reassign_min_core_n}")
        print(f"       reassigned={qc['n_reassigned']} of borderline={qc['n_borderline_candidates']} (PC-complete eligible={qc['n_reassignable_pc_complete']})")
    else:
        print("     Reassign: disabled")


if __name__ == "__main__":
    main()


"""
Minimal example command (with reassignment + plots + depth profiles):

python SPARK/env_split_o2_by_gmm_reassign.py \
  --matrix-cleaned ../V4_ncbi_output/env_pca/tables/matrix_cleaned.csv \
  --eigenvectors ../V4_ncbi_output/env_pca/tables/eigenvectors_scores.csv \
  --assignments ../V4_ncbi_output/env_compartments_gmm/tables/compartments_assignments_smoothed.csv \
  --outdir ../V4_ncbi_output/env_o2_split_by_gmm \
  --sep-matrix ',' \
  --sep-eig ',' \
  --sep-assign ',' \
  --key-mode composite \
  --key-cols "Cruise,Year,Month,Day,Depth" \
  --pc-cols "PC1,PC2,PC3" \
  --min-subcluster-size 20 \
  --reassign \
  --core-min-prob 0.8 \
  --reassign-radius-quantile 0.95 \
  --reassign-min-core-n 20 \
  --plots \
  --plot-formats "png,svg"
"""
