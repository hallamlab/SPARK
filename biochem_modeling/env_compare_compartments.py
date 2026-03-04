#!/usr/bin/env python3

"""
env_compare_compartments.py

Purpose
-------
Compare (1) data-driven GMM “environmental compartments” (from your final GMM step) against
(2) contemporaneous oxygen-defined compartments (oxic/dysoxic/suboxic/anoxic) computed from the
biogeochemical matrix, and generate:

- Depth-profile plots for every biogeochemical feature and every selected PC:
    • color by O2 compartment (fixed palette)
    • color by GMM component (categorical)
- UMAP embeddings in biogeochemical space with multiple views:
    • color by O2 compartment
    • color by GMM component
    • color by depth
    • bubble plots for each biogeochemical feature (size=value, fixed color per feature)
- Agreement metrics between O2 compartments and GMM components:
    • confusion matrices (raw + row-normalized + col-normalized)
    • ARI (Adjusted Rand Index)
    • NMI (Normalized Mutual Information)
- Clustering-quality metrics in TWO spaces (scaled):
    • PC space: silhouette / Calinski–Harabasz / Davies–Bouldin for O2 labels and for GMM labels
    • Biochem space (complete rows only): silhouette / Calinski–Harabasz / Davies–Bouldin for O2 and for GMM
- Additional diagnostics and sensitivity analyses (when the required columns exist):
    • PC-space scaling audit after StandardScaler
    • cruise-block bootstrap silhouettes (CI) for O2 and GMM labels
    • within-cruise permutation test for GMM silhouette (null: labels shuffled within cruise)
    • responsibility-weighted silhouette (uses max_prob as sample_weight)
    • low-confidence sweeps vs max_prob threshold:
        - silhouette vs threshold (GMM labels)
        - ARI/NMI vs threshold (O2 vs GMM)

Key design / “be careful” behaviors (implemented in code)
---------------------------------------------------------
1) Robust merging via derived join key (default is composite)
   - The script supports two key modes:
     A) key_mode="composite" (DEFAULT)
        • Builds an internal merge key column named "__merge_key__" from:
            Cruise, Year, Month, Day, Depth
          using a configurable separator (default "|").
        • This is the default because it avoids collisions when multiple casts exist in the same month.
     B) key_mode="id"
        • Uses an existing id column (default "cruise_year_month_depth") as the merge key.
   - The derived key is constructed independently for each input table and used for left-merging.

2) Guard against duplicate column headers in input files
   - Each input is read with pandas, then any duplicated column labels are removed by keeping the
     first instance only. This specifically protects against pathological CSVs where an ID column
     appears twice (a known failure mode).

3) Strict separation of “spaces” used for different computations
   - PC-space metrics are computed on:
       X_pc_scaled = StandardScaler().fit_transform(PC matrix)
     using only rows with complete values across the selected PC columns.
   - Biochem-space metrics and UMAP are computed on:
       X_bio_scaled = StandardScaler().fit_transform(biochem feature matrix)
     using only rows that are complete across ALL biochem features (strict complete-case).
   - These spaces are not mixed: UMAP is NOT run on PC space in this script; it is run on scaled
     biochem space only.

Inputs (required)
-----------------
1) --matrix-cleaned
   Path to matrix_cleaned.csv (biochemical features + metadata)
   Required columns (names configurable via CLI flags):
     - Oxygen column (default: "Oxygen")
     - Depth anchored column (default: "Depth_anchored")  [required; used for depth profiles + UMAP depth coloring]
     - Depth column (default: "Depth")                    [used for composite key and metadata]
     - Cruise column (default: "Cruise")                  [used for composite key and block analyses]
     - Year, Month, Day                                   [used for composite key; must exist if key_mode=composite]
     - Date column (default: "date") is parsed if present but not required to exist

   Biochemical features are defined as:
     all columns in matrix_cleaned EXCEPT:
       - derived merge key "__merge_key__" (internal)
       - metadata columns present among:
           Cruise, Year, Month, Day, Depth, date, Depth_anchored
       - the legacy id column (default "cruise_year_month_depth") if present

2) --eigenvectors
   Path to eigenvectors_scores.csv (environmental eigenvectors / PCs + metadata)
   Requirements:
     - Must contain PC columns named "PC1", "PC2", ... (unless you explicitly restrict --pc-cols)
     - Must contain the columns needed to build the merge key under your chosen key_mode

   PC column selection:
     - If --pc-cols is provided, it is used verbatim (comma-separated list).
     - Otherwise, all columns starting with "PC" are auto-detected and sorted numerically by the
       integer suffix (PC1, PC2, ...).

3) --assignments
   Path to compartments_assignments_smoothed.csv (from final GMM step)
   Requirements:
     - Must contain a GMM component label column named "component" (used throughout the comparisons)
     - Must contain the columns needed to build the merge key under your chosen key_mode
     - Optional but strongly used if present:
         • "max_prob" (enables weighted silhouette + low-confidence sweeps)

All three inputs are merged into a master table (written to tables/merged_for_comparison.csv) by:
  df_assign LEFT JOIN df_matrix on __merge_key__
  then LEFT JOIN df_eig on __merge_key__
This means:
  - every assignment row is retained (unless it later fails complete-case filters for certain analyses),
  - missing matrix/eigenvectors fields become NaN and may exclude rows from specific downstream steps.

Oxygen compartments (computed here)
-----------------------------------
O2 compartments are computed from the Oxygen column (treated as µM; no unit conversion is performed here),
using inclusive threshold intervals:

- oxic    : Oxygen > o2_oxic_gt                       (default > 90)
- dysoxic : o2_dysoxic_lo <= Oxygen <= o2_dysoxic_hi   (default 20..90 inclusive)
- suboxic : o2_suboxic_lo <= Oxygen <  o2_suboxic_hi   (default 1..20, upper bound exclusive)
- anoxic  : Oxygen < o2_suboxic_lo                    (default < 1)
- "NA" is assigned when Oxygen cannot be coerced to numeric.

Note: Because dysoxic is assigned after oxic, the oxic condition dominates for Oxygen > 90; remaining
values in 20..90 become dysoxic, etc.

Outputs
-------
Directory structure under --outdir:

Top-level
- run_config.json
  JSON dump of:
    • Config (all CLI arguments)
    • umap_available flag (whether umap-learn imported successfully)

tables/
- merged_for_comparison.csv
  The merged master table used for all analyses (assignments + matrix_cleaned + eigenvectors).

- confusion_o2_vs_gmm_raw.csv
- confusion_o2_vs_gmm_row_norm.csv
- confusion_o2_vs_gmm_col_norm.csv

- comparison_stats.csv
  Contains:
    n_total, n_pc_complete, n_pcs_used,
    ARI_o2_vs_gmm, NMI_o2_vs_gmm,
    silhouette_PCspace_o2, silhouette_PCspace_gmm

- quality_metrics_pcspace.csv
  Contains (PC space, scaled, complete-PC rows only):
    silhouette_o2, silhouette_gmm,
    calinski_harabasz_o2, calinski_harabasz_gmm,
    davies_bouldin_o2, davies_bouldin_gmm,
    plus counts (n_used, k_labels_o2, k_labels_gmm)

- pc_space_scaling_audit.csv                    (only if >=10 complete-PC rows)
  Per-PC mean and std after StandardScaler.

- pc1_mean_by_cruise.csv                        (only if Cruise exists and at least PC1 exists)
  Mean of PC1_scaled by cruise; a simple cruise-offset diagnostic.

- silhouette_block_bootstrap_by_cruise.csv      (only if Cruise exists and >=10 complete-PC rows)
  Block bootstrap (resample cruises with replacement) silhouette mean and 95% CI
  for O2 labels and for GMM labels in PC space.

- within_cruise_permutation_test_gmm.csv        (only if Cruise exists and >=10 complete-PC rows)
  Within-cruise permutation test for GMM silhouette in PC space:
    observed silhouette, p_value, n_null

- responsibility_weighted_silhouette.csv        (only if "max_prob" exists and >=10 complete-PC rows)
  Responsibility-weighted silhouette for GMM labels in PC space using max_prob as sample_weight.

- low_confidence_silhouette_sweep_pcspace.csv   (only if "max_prob" exists and >=10 complete-PC rows)
  Silhouette vs max_prob threshold for GMM labels in PC space.

- low_confidence_agreement_sweep.csv            (only if "max_prob" exists and >=10 complete-PC rows)
  ARI and NMI vs max_prob threshold (O2 vs GMM) on retained high-confidence rows.

- quality_metrics_biochem_space.csv
  Clustering-quality metrics in biochem space (scaled) using STRICT complete-case rows across ALL
  biochem features:
    silhouette / Calinski–Harabasz / Davies–Bouldin for O2 and for GMM.

- compartment_medians_gmm.csv
- compartment_medians_o2.csv
  Per-group medians of (all biochem features + all PCs used) and group sample sizes.

- gmm_pairwise_feature_effect_sizes.csv
  Pairwise, biochem-only effect sizes between GMM components:
    effect_size_d = (median_i - median_j) / pooled_sd
  (medians used for robustness; pooled SD computed from sample variances)

UMAP-related tables (only if umap-learn is available AND >=10 complete biochem rows)
- umap_embedding.csv
  Contains: __merge_key__, UMAP1, UMAP2, o2_compartment, component, Depth_anchored

UMAP skip markers
- UMAP_NOT_AVAILABLE.txt                         (if umap-learn not importable)
- UMAP_SKIPPED_NOT_ENOUGH_COMPLETE_ROWS.txt      (if insufficient complete biochem rows)

Optional PCA-stage interpretation outputs (#4; only if pca tables are provided and found)
---------------------------------------------------------------------------------------
These are enabled when either:
  - --pca-tables-dir points to a directory containing pca_loadings.csv and/or pc_loading_concentration.csv, OR
  - --pca-loadings / --pc-loading-concentration are provided explicitly.

tables/
- pc_top_loadings.csv
  Long-form table of top-N absolute loadings per PC (restricted to PCs present in eigenvectors).

- pc_loading_concentration_merged.csv
  Best-effort subset of columns from pc_loading_concentration.csv (PC, entropy_norm,
  effective_features, participation_ratio) restricted to PCs present in eigenvectors.

plots/
- C1_top_loadings_PCk.{pdf,svg,png}
  Horizontal bar plots of signed loadings for the top-N features for each PC used.

Plots written (always, for complete-enough features)
----------------------------------------------------
All plots are saved in one or more formats depending on flags:
  default: pdf + svg + png (png dpi default 300)
  disable with: --no-pdf / --no-svg / --no-png

A) Depth profiles for each biochem feature (if at least 3 numeric points)
- A1_depth_vs_{feature}__color_o2
- A2_depth_vs_{feature}__color_gmm

A) Depth profiles for each PC (if at least 3 numeric points)
- A3_depth_vs_{PC}__color_o2
- A4_depth_vs_{PC}__color_gmm

B) UMAP plots (only if UMAP runs)
- B1_umap_color_o2
- B2_umap_color_gmm
- B3_umap_color_depth
- B4_umap_bubble_{feature}   (one per biochem feature; fixed color, size encodes value)

Notes / invariants (things this script does NOT do)
---------------------------------------------------
- Does not convert oxygen units; assumes the matrix_cleaned Oxygen column is already in µM.
- Does not try to reconcile differing row sets across files beyond the merge key; missing data simply
  propagates and can exclude rows from some analyses.
- Does not compute UMAP on PC space; UMAP here is biochem-space only (complete-case).
- Does not enforce uniqueness of the derived key; it relies on the composite key default to prevent
  collisions, but will merge in a pandas-default manner if duplicates exist.

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
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
pd.set_option('future.no_silent_downcasting', True)

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
    "Dimethyl Sulfide": "#E6AB02",
    "Methane": "violet",
    "Fe": "red",
    "Fluorescence": "limegreen",
    "Temperature": "gray",
    "Salinity": "darkviolet",
    "Phosphate": "brown",
    "Silicate": "peru",
    "PAR": "tan",
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
    eigenvectors: str
    assignments: str
    o2_assignments: Optional[str]
    outdir: str
    sep_matrix: str
    sep_eig: str
    sep_assign: str
    sep_o2_assign: str

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
    o2_compartment_col: str

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
    ap.add_argument("--eigenvectors", required=True, help="Path to eigenvectors_scores.csv")
    ap.add_argument("--assignments", required=True, help="Path to compartments_assignments_smoothed.csv")
    ap.add_argument(
        "--o2-assignments",
        default=None,
        help="Optional path to o2_compartments_assignments_{base|smoothed}.csv. If provided, uses these O2 labels instead of thresholding Oxygen.",
    )
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep-matrix", default=",")
    ap.add_argument("--sep-eig", default=",")
    ap.add_argument("--sep-assign", default=",")
    ap.add_argument("--sep-o2-assign", default=",")

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
    ap.add_argument(
        "--o2-compartment-col",
        default="compartment_name",
        help="Column in --o2-assignments containing O2 compartment labels (default compartment_name).",
    )

    ap.add_argument("--o2-oxic-gt", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-hi", type=float, default=90.0)
    ap.add_argument("--o2-dysoxic-lo", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-hi", type=float, default=20.0)
    ap.add_argument("--o2-suboxic-lo", type=float, default=1.0)

    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.05)
    ap.add_argument("--umap-random-state", type=int, default=42)
    ap.add_argument("--umap-metric", default="euclidean")

    ap.add_argument("--bubble-q-low", type=float, default=0.01)
    ap.add_argument("--bubble-q-high", type=float, default=0.99)
    ap.add_argument("--bubble-size-min", type=float, default=1.0)
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
        eigenvectors=ns.eigenvectors,
        assignments=ns.assignments,
        o2_assignments=ns.o2_assignments,
        outdir=ns.outdir,
        sep_matrix=ns.sep_matrix,
        sep_eig=ns.sep_eig,
        sep_assign=ns.sep_assign,
        sep_o2_assign=ns.sep_o2_assign,

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
        o2_compartment_col=ns.o2_compartment_col,

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


def normalize_o2_labels(raw: pd.Series) -> pd.Series:
    """
    Normalize O2 compartment labels to canonical names:
      oxic, dysoxic, suboxic, anoxic
    Also supports numeric-coded labels 0..3.
    """
    s = raw.copy()
    out = pd.Series(["NA"] * len(s), index=s.index, dtype="object")

    # numeric mapping (0..3)
    num = pd.to_numeric(s, errors="coerce")
    num_map = {0: "oxic", 1: "dysoxic", 2: "suboxic", 3: "anoxic"}
    for k, v in num_map.items():
        out[num == float(k)] = v

    # text mapping
    txt = s.astype("object").fillna("NA").astype(str).str.strip().str.lower()
    txt_map = {
        "oxic": "oxic",
        "dysoxic": "dysoxic",
        "suboxic": "suboxic",
        "anoxic": "anoxic",
    }
    mapped_txt = txt.map(txt_map)
    out[mapped_txt.notna()] = mapped_txt[mapped_txt.notna()]
    return out


def o2_labels_from_assignments(df_o2: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Extract and normalize O2 labels from an O2 assignment table.
    Fallback order:
      1) cfg.o2_compartment_col
      2) compartment_name
      3) o2_compartment
      4) component
    """
    candidates = [cfg.o2_compartment_col, "compartment_name", "o2_compartment", "component"]
    col = next((c for c in candidates if c in df_o2.columns), None)
    if col is None:
        raise ValueError(
            "Could not find an O2 compartment column in --o2-assignments. "
            f"Tried: {candidates}"
        )
    return normalize_o2_labels(df_o2[col])


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
        ax.scatter(sub[xcol].values,
                   sub[ycol].values,
                   s=16, alpha=0.40,
                   label=str(cls),
                   color=color_map.get(cls, None),
                   linewidths=0)

    ax.invert_yaxis()
    ax.set_xlabel(xcol)
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)

    handles = []
    for k, c in color_map.items():
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=c,
                markeredgecolor="none",
                label=str(k),
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

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

    # categorical integer labels
    labels = pd.to_numeric(df[cat_col], errors="coerce").fillna(-1).astype(int)
    uniq = np.sort(labels.unique())
    if len(uniq) == 0:
        return

    # build grayscale colors (no white, no black)
    n = len(uniq)
    grays = np.linspace(0.0, 1.0, n)  # avoid 0.0 (black) and 1.0 (white)
    label_to_color = {
        lab: (g, g, g) for lab, g in zip(uniq, grays)
    }

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    cats = sorted(d[cat_col].astype("object").fillna("NA").unique(), key=lambda z: str(z))
    for cat in cats:
        sub = d[d[cat_col].astype("object").fillna("NA") == cat]
        ax.scatter(sub[xcol].values,
                   sub[ycol].values, 
                   s=14,
                   alpha=0.40,
                   label=str(cat),
                   color=label_to_color[cat],
                   linewidths=0.25,
                   edgecolors="gray"
                   )

    ax.invert_yaxis()
    ax.set_xlabel(xcol)
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    save_all_formats(fig, out_base, cfg)

    return label_to_color


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
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=14, alpha=0.40,
                   label=str(cls), 
                   color=c,
                   linewidths=0.25,
                   edgecolors="gray"
                   )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)

    if palette is not None:
        present = set(lab.unique())

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="gray",
                markeredgewidth=0.25,
                label=str(key),
            )
            for key, color in palette.items()
            if key in present
        ]

        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    else:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

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
    sc = ax.scatter(emb[:, 0], emb[:, 1], s=14, alpha=0.40, c=d, cmap=depth_cmap())
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
    ax.scatter(emb[:, 0], emb[:, 1], s=sizes, alpha=0.40, color=color, edgecolors="none")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    save_all_formats(fig, out_base, cfg)


def build_categorical_grayscale_palette(labels: pd.Series) -> Dict[str, Tuple[float, float, float]]:
    """
    Stable categorical grayscale palette (no white, no black).
    Keys are string labels; values are RGB tuples.
    """
    lab = labels.astype("object").fillna("NA").astype(str)
    uniq = sorted(lab.unique(), key=lambda z: str(z))
    n = len(uniq)
    if n == 0:
        return {}
    grays = np.linspace(0.15, 0.85, n)  # avoid pure black/white
    return {str(u): (g, g, g) for u, g in zip(uniq, grays)}


def plot_pc1_vs_pc2_categorical(
    df: pd.DataFrame,
    pc1: str,
    pc2: str,
    label_col: str,
    palette: Dict[str, object],
    title: str,
    out_base: str,
    cfg: Config,
    s: float = 14.0,
    alpha: float = 0.70,
) -> None:
    if pc1 not in df.columns or pc2 not in df.columns or label_col not in df.columns:
        return

    d = df[[pc1, pc2, label_col]].copy()
    d[pc1] = pd.to_numeric(d[pc1], errors="coerce")
    d[pc2] = pd.to_numeric(d[pc2], errors="coerce")
    d = d.dropna(subset=[pc1, pc2])
    if d.shape[0] < 3:
        return

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = plt.gca()

    lab = d[label_col].astype("object").fillna("NA").astype(str)
    classes = sorted(lab.unique(), key=lambda z: str(z))

    for cls in classes:
        mask = (lab == cls).to_numpy()
        c = palette.get(str(cls), None)
        ax.scatter(
            d.loc[mask, pc1].to_numpy(),
            d.loc[mask, pc2].to_numpy(),
            s=s,
            alpha=alpha,
            label=str(cls),
            color=c,
            linewidths=0.25,
            edgecolors="gray",
        )

    ax.set_xlabel(pc1)
    ax.set_ylabel(pc2)
    ax.set_title(title)

    present = set(lab.unique())

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="gray",
            markeredgewidth=0.25,
            label=str(key),
        )
        for key, color in palette.items()
        if str(key) in present
    ]

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

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
# NEW (#5): Cruise-block bootstrap / low-confidence / weighted silhouette / permutation / PC audit
# ----------------------------

def block_bootstrap_silhouette(
    X: np.ndarray,
    labels: pd.Series,
    blocks: pd.Series,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Block bootstrap silhouette by resampling blocks (e.g., Cruise).
    Returns mean, CI_low, CI_high.
    """
    rng = np.random.default_rng(random_state)
    lab = labels.astype("object").fillna("NA").astype(str).to_numpy()
    blk = blocks.astype("object").fillna("NA").astype(str).to_numpy()

    uniq_blocks = np.unique(blk)
    scores: List[float] = []

    for _ in range(int(n_boot)):
        sampled_blocks = rng.choice(uniq_blocks, size=len(uniq_blocks), replace=True)
        mask = np.isin(blk, sampled_blocks)
        if np.unique(lab[mask]).size < 2 or mask.sum() < 10:
            continue
        try:
            s = float(silhouette_score(X[mask], lab[mask], metric="euclidean"))
            scores.append(s)
        except Exception:
            pass

    if len(scores) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    return {
        "mean": float(np.mean(scores)),
        "ci_low": float(np.quantile(scores, 0.025)),
        "ci_high": float(np.quantile(scores, 0.975)),
    }


def sweep_low_confidence_silhouette(
    X: np.ndarray,
    labels: pd.Series,
    max_prob: pd.Series,
    thresholds: List[float],
) -> pd.DataFrame:
    rows = []
    lab = labels.astype("object").fillna("NA").astype(str)
    mp = pd.to_numeric(max_prob, errors="coerce")
    for t in thresholds:
        keep = mp >= t
        if int(keep.sum()) < 10:
            continue
        if lab.loc[keep].nunique(dropna=False) < 2:
            continue
        try:
            s = float(silhouette_score(X[keep.to_numpy()], lab.loc[keep].to_numpy(), metric="euclidean"))
        except Exception:
            s = np.nan
        rows.append({
            "max_prob_threshold": float(t),
            "n_retained": int(keep.sum()),
            "n_labels": int(lab.loc[keep].nunique(dropna=False)),
            "silhouette": float(s),
        })
    return pd.DataFrame(rows)


def sweep_low_confidence_agreement(
    y_true: pd.Series,
    y_pred: pd.Series,
    max_prob: pd.Series,
    thresholds: List[float],
) -> pd.DataFrame:
    rows = []
    yt = y_true.astype("object").fillna("NA").astype(str)
    yp = y_pred.astype("object").fillna("NA").astype(str)
    mp = pd.to_numeric(max_prob, errors="coerce")
    for t in thresholds:
        keep = mp >= t
        if int(keep.sum()) < 10:
            continue
        ari = float(adjusted_rand_score(yt.loc[keep], yp.loc[keep]))
        nmi = float(normalized_mutual_info_score(yt.loc[keep], yp.loc[keep]))
        rows.append({
            "max_prob_threshold": float(t),
            "n_retained": int(keep.sum()),
            "ARI_o2_vs_gmm": ari,
            "NMI_o2_vs_gmm": nmi,
        })
    return pd.DataFrame(rows)


def responsibility_weighted_silhouette(X: np.ndarray, labels: pd.Series, weights: pd.Series) -> float:
    """
    Weighted silhouette for GMM labels using per-sample weights (e.g., max_prob).

    IMPORTANT: scikit-learn's silhouette_score forwards **kwds to the pairwise distance
    function for some metrics (including 'euclidean'), which can break when sample_weight
    is present. To be robust across sklearn versions, we compute distances explicitly
    and use metric='precomputed'.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import pairwise_distances

    lab = labels.astype(str).fillna("NA").to_numpy()
    if len(set(lab)) < 2:
        return np.nan

    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(w).any():
        return np.nan
    w = np.where(np.isfinite(w), w, 0.0)
    if np.all(w <= 0):
        return np.nan

    # Precompute Euclidean distances to avoid passing sample_weight into distance functions
    D = pairwise_distances(X, metric="euclidean")

    try:
        return float(silhouette_score(D, lab, metric="precomputed", sample_weight=w))
    except Exception:
        return np.nan


def within_block_permutation_test(
    X: np.ndarray,
    labels: pd.Series,
    blocks: pd.Series,
    n_perm: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Permute labels ONLY within each block (e.g., Cruise) and compare silhouette.
    p-value is one-sided: P(null >= observed)
    """
    rng = np.random.default_rng(random_state)
    lab = labels.astype("object").fillna("NA").astype(str).to_numpy()
    blk = blocks.astype("object").fillna("NA").astype(str).to_numpy()

    if np.unique(lab).size < 2 or X.shape[0] < 10:
        return {"observed": np.nan, "p_value": np.nan, "n_null": 0}

    try:
        obs = float(silhouette_score(X, lab, metric="euclidean"))
    except Exception:
        return {"observed": np.nan, "p_value": np.nan, "n_null": 0}

    null_scores: List[float] = []
    uniq_blocks = np.unique(blk)

    for _ in range(int(n_perm)):
        perm = lab.copy()
        for b in uniq_blocks:
            idx = np.where(blk == b)[0]
            if idx.size >= 2:
                rng.shuffle(perm[idx])
        try:
            null_scores.append(float(silhouette_score(X, perm, metric="euclidean")))
        except Exception:
            pass

    if len(null_scores) == 0:
        return {"observed": obs, "p_value": np.nan, "n_null": 0}

    null_arr = np.array(null_scores, dtype=float)
    p = float(np.mean(null_arr >= obs))
    return {"observed": obs, "p_value": p, "n_null": int(null_arr.size)}


def pc_space_audit(X_scaled: np.ndarray, pc_cols: List[str]) -> pd.DataFrame:
    """
    Quick scaling audit: means and stds of each PC after StandardScaler.
    """
    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0, ddof=0)
    return pd.DataFrame({
        "PC": pc_cols,
        "mean_scaled": np.round(means, 8),
        "std_scaled": np.round(stds, 8),
    })


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
    df_eig = read_table_dedup_cols(cfg.eigenvectors, cfg.sep_eig)
    df_assign = read_table_dedup_cols(cfg.assignments, cfg.sep_assign)
    df_o2_assign = (
        read_table_dedup_cols(cfg.o2_assignments, cfg.sep_o2_assign)
        if cfg.o2_assignments
        else None
    )

    # Datetimes
    df_matrix = coerce_datetime(df_matrix, cfg.date_col)
    df_eig = coerce_datetime(df_eig, cfg.date_col)
    df_assign = coerce_datetime(df_assign, cfg.date_col)
    if df_o2_assign is not None:
        df_o2_assign = coerce_datetime(df_o2_assign, cfg.date_col)

    # Build merge keys
    df_matrix = build_merge_key(df_matrix, cfg)
    df_eig = build_merge_key(df_eig, cfg)
    df_assign = build_merge_key(df_assign, cfg)
    if df_o2_assign is not None:
        df_o2_assign = build_merge_key(df_o2_assign, cfg)

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
        raise ValueError("No PC columns found in eigenvectors (expected PC1, PC2, ...)")

    # Metadata columns expected in matrix
    meta_cols = [cfg.cruise_col, "Year", "Month", "Day", cfg.depth_col, cfg.date_col, cfg.depth_anchored_col]
    meta_cols_present = [c for c in meta_cols if c in df_matrix.columns]

    # Biochem columns = everything in matrix except merge key + meta + legacy id if present
    exclude = set([cfg.derived_key_col] + meta_cols_present + [cfg.id_col])
    biochem_all_cols = [c for c in df_matrix.columns if c not in exclude]
    biochem_cols = [c for c in biochem_all_cols if pd.to_numeric(df_matrix[c], errors="coerce").notna().all()]

    if (df_o2_assign is None) and (cfg.oxygen_col not in df_matrix.columns):
        raise ValueError(f"matrix_cleaned missing oxygen col: {cfg.oxygen_col}")
    if cfg.depth_anchored_col not in df_matrix.columns:
        raise ValueError(f"matrix_cleaned missing anchored depth col: {cfg.depth_anchored_col}")

    # Merge master
    keep_matrix = [cfg.derived_key_col] + meta_cols_present + biochem_cols
    keep_eig = [cfg.derived_key_col] + pc_cols
    keep_sparce = [cfg.derived_key_col] + meta_cols_present + biochem_all_cols

    m = df_assign.merge(df_matrix[keep_matrix], on=cfg.derived_key_col, how="left", suffixes=("", "_matrix"))
    m = m.merge(df_eig[keep_eig], on=cfg.derived_key_col, how="left", suffixes=("", "_eig"))

    m_sp = df_assign.merge(df_matrix[keep_sparce], on=cfg.derived_key_col, how="left", suffixes=("", "_matrix"))

    # Label O2 (prefer external soft/smoothed assignments when provided)
    if df_o2_assign is not None:
        o2_tbl = df_o2_assign[[cfg.derived_key_col]].copy()
        o2_tbl["o2_compartment"] = o2_labels_from_assignments(df_o2_assign, cfg)
        o2_tbl = o2_tbl.dropna(subset=[cfg.derived_key_col]).drop_duplicates(subset=[cfg.derived_key_col], keep="first")
        o2_lookup = o2_tbl.set_index(cfg.derived_key_col)["o2_compartment"]
        m["o2_compartment"] = m[cfg.derived_key_col].map(o2_lookup).fillna("NA").astype("object")
        m_sp["o2_compartment"] = m_sp[cfg.derived_key_col].map(o2_lookup).fillna("NA").astype("object")
    else:
        m["o2_compartment"] = label_o2_compartment(m[cfg.oxygen_col], cfg)
        m_sp["o2_compartment"] = label_o2_compartment(m_sp[cfg.oxygen_col], cfg)

    m[cfg.depth_anchored_col] = pd.to_numeric(m[cfg.depth_anchored_col], errors="coerce")
    m_sp[cfg.depth_anchored_col] = pd.to_numeric(m_sp[cfg.depth_anchored_col], errors="coerce")

    # Save merged table
    m.to_csv(os.path.join(tables_dir, "merged_for_comparison.csv"), index=False)

    # Save config
    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump({"config": cfg.__dict__, "umap_available": _HAVE_UMAP}, f, indent=2)

    # ----------------------------
    # A) Depth profiles: biochem
    # ----------------------------
    for feat in biochem_all_cols:
        x = pd.to_numeric(m_sp[feat], errors="coerce")
        if x.notna().sum() < 3:
            continue
        scatter_depth_profile(
            df=m_sp,
            xcol=feat,
            ycol=cfg.depth_anchored_col,
            color_col="o2_compartment",
            color_map=O2_COMPARTMENT_PALETTE,
            title=f"Depth vs {feat} (color = O2 compartment)",
            out_base=os.path.join(plots_dir, f"A1_depth_vs_{sanitize_filename(feat)}__color_o2"),
            cfg=cfg,
        )
        GMM_COMPARTMENT_PALETTE = scatter_depth_profile_categorical(
            df=m_sp,
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
        X_pc_scaled = np.empty((0, len(pc_cols)), dtype=float)
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
    # NEW (#5): PC-space audit + bootstrap/permutation + weighted silhouette + low-confidence sweep
    # ----------------------------
    if X_pc_good.shape[0] >= 10:
        # audit scaling/standardization
        pc_audit_df = pc_space_audit(X_pc_scaled, pc_cols)
        pc_audit_df.to_csv(os.path.join(tables_dir, "pc_space_scaling_audit.csv"), index=False)

        # cruise structure helper table (simple mean PC1 per cruise, helps spot cruise offsets)
        if cfg.cruise_col in m.columns and X_pc_scaled.shape[1] >= 1:
            pd.DataFrame({
                cfg.cruise_col: m.loc[good_pc, cfg.cruise_col].astype("object").fillna("NA").values,
                "PC1_scaled": X_pc_scaled[:, 0],
            }).groupby(cfg.cruise_col, dropna=False).mean(numeric_only=True).to_csv(
                os.path.join(tables_dir, "pc1_mean_by_cruise.csv")
            )

        # block bootstrap silhouettes by Cruise
        if cfg.cruise_col in m.columns:
            boot_o2 = block_bootstrap_silhouette(
                X_pc_scaled,
                yt.loc[good_pc],
                m.loc[good_pc, cfg.cruise_col],
            )
            boot_gmm = block_bootstrap_silhouette(
                X_pc_scaled,
                yp.loc[good_pc],
                m.loc[good_pc, cfg.cruise_col],
            )
            pd.DataFrame([{
                "label": "O2",
                "silhouette_mean": boot_o2["mean"],
                "silhouette_ci_low": boot_o2["ci_low"],
                "silhouette_ci_high": boot_o2["ci_high"],
            }, {
                "label": "GMM",
                "silhouette_mean": boot_gmm["mean"],
                "silhouette_ci_low": boot_gmm["ci_low"],
                "silhouette_ci_high": boot_gmm["ci_high"],
            }]).to_csv(
                os.path.join(tables_dir, "silhouette_block_bootstrap_by_cruise.csv"),
                index=False,
            )

            # within-cruise permutation test (GMM labels)
            perm_gmm = within_block_permutation_test(
                X_pc_scaled,
                yp.loc[good_pc],
                m.loc[good_pc, cfg.cruise_col],
            )
            pd.DataFrame([{
                "label": "GMM",
                "space": "PCspace_scaled",
                **perm_gmm,
            }]).to_csv(
                os.path.join(tables_dir, "within_cruise_permutation_test_gmm.csv"),
                index=False,
            )

        # responsibility-weighted silhouette (uses max_prob as weight if present)
        sil_gmm_weighted = np.nan
        if "max_prob" in m.columns:
            sil_gmm_weighted = responsibility_weighted_silhouette(
                X_pc_scaled,
                yp.loc[good_pc],
                m.loc[good_pc, "max_prob"],
            )
        pd.DataFrame([{
            "space": "PCspace_scaled",
            "silhouette_gmm_weighted": sil_gmm_weighted,
        }]).to_csv(
            os.path.join(tables_dir, "responsibility_weighted_silhouette.csv"),
            index=False,
        )

        # low-confidence sweep: silhouette vs threshold (+ agreement vs O2)
        if "max_prob" in m.columns:
            thresholds = np.linspace(0.0, 0.95, 20).tolist()
            sweep_sil = sweep_low_confidence_silhouette(
                X_pc_scaled,
                yp.loc[good_pc],
                m.loc[good_pc, "max_prob"],
                thresholds,
            )
            sweep_sil.to_csv(
                os.path.join(tables_dir, "low_confidence_silhouette_sweep_pcspace.csv"),
                index=False,
            )

            sweep_agree = sweep_low_confidence_agreement(
                yt.loc[good_pc],
                yp.loc[good_pc],
                m.loc[good_pc, "max_prob"],
                thresholds,
            )
            sweep_agree.to_csv(
                os.path.join(tables_dir, "low_confidence_agreement_sweep.csv"),
                index=False,
            )

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
                palette=GMM_COMPARTMENT_PALETTE,
            )
            plot_umap_depth(
                emb=emb,
                depth=m_bio[cfg.depth_anchored_col],
                title=f"UMAP (metric={cfg.umap_metric}) colored by depth",
                out_base=os.path.join(plots_dir, "B3_umap_color_depth"),
                cfg=cfg,
            )

            for feat in biochem_all_cols:
                v = pd.to_numeric(m_sp[feat], errors="coerce")
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

        # Keep only PCs we actually used in eigenvectors (intersection)
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
    # NEW: PC1 vs PC2 scatter overlays
    # ----------------------------
    if "PC1" in m.columns and "PC2" in m.columns:
        # 1) PC1 vs PC2 colored by O2 compartments (fixed palette)
        plot_pc1_vs_pc2_categorical(
            df=m,
            pc1="PC1",
            pc2="PC2",
            label_col="o2_compartment",
            palette=O2_COMPARTMENT_PALETTE,
            title="PC1 vs PC2 (color = O2 compartment)",
            out_base=os.path.join(plots_dir, "D1_pc1_vs_pc2__color_o2"),
            cfg=cfg,
            s=14.0,
            alpha=0.40,
        )

        # 2) PC1 vs PC2 colored by GMM compartments (categorical grayscale)
        # Use existing GMM palette if already created; otherwise build deterministically from labels
        if "GMM_COMPARTMENT_PALETTE" in locals() and isinstance(GMM_COMPARTMENT_PALETTE, dict) and len(GMM_COMPARTMENT_PALETTE) > 0:
            gmm_pal = {str(k): v for k, v in GMM_COMPARTMENT_PALETTE.items()}
        else:
            gmm_pal = build_categorical_grayscale_palette(m["component"])

        plot_pc1_vs_pc2_categorical(
            df=m,
            pc1="PC1",
            pc2="PC2",
            label_col="component",
            palette=gmm_pal,
            title="PC1 vs PC2 (color = GMM compartment)",
            out_base=os.path.join(plots_dir, "D2_pc1_vs_pc2__color_gmm"),
            cfg=cfg,
            s=14.0,
            alpha=0.40,
        )

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
    if "max_prob" in m.columns and X_pc_good.shape[0] >= 10:
        print(f"     Quality (PCspace): weighted silhouette GMM={float(pd.read_csv(os.path.join(tables_dir, 'responsibility_weighted_silhouette.csv'))['silhouette_gmm_weighted'].iloc[0]):.3f}")
    if cfg.pca_tables_dir:
        print(f"     PCA tables dir: {cfg.pca_tables_dir}")
        if cfg.pca_loadings_path and os.path.exists(cfg.pca_loadings_path):
            print(f"       loaded: pca_loadings.csv")
        if cfg.pc_loading_concentration_path and os.path.exists(cfg.pc_loading_concentration_path):
            print(f"       loaded: pc_loading_concentration.csv")


if __name__ == "__main__":
    main()
