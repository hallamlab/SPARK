# /media/nfs/Ryan/SABer/SI_data/SI_ASV/SPARK/aligned_biochem_visuals.py
#
# Cruise-sliced UMAP over biochem features, aligned by (fuzzy) depth anchors,
# with cruise-aware interpolation imputation and Oxygen-compartment trajectory visualization.
#
# UI requirements implemented:
# 1) TRUE checkboxes for compartments + depths (HTML sidebar; no Dash server).
# 2) If a depth is EVER in any selected compartment, the ENTIRE depth track is eligible to show.
#    (Depth checkbox then turns that eligible track on/off.)
# 3) Trajectories only (no points).
# 4) Cruise-step scaling removed. Use --aspect-* to stretch plot geometry instead.
#
# NEW (requested; added without removing existing behavior):
# - Adds HDBSCAN labeling (in addition to O2-based compartments), with THREE labeling modes in the HTML:
#     (A) Legacy O2-based compartments
#     (B) HDBSCAN on UMAP space (umap1/umap2)
#     (C) HDBSCAN on standardized feature space
#   and you can switch between them in the HTML with a radio toggle.
# - HDBSCAN params can be set independently for (B) vs (C).
#
# NEW (requested; added without removing existing behavior):
# - Robust scaling for feature-space HDBSCAN (median/IQR), replacing z-score.
# - Transition entropy per depth (per labeling mode) to quantify switching.
# - Outputs transition-entropy tables and barplots.
# - Trajectory link width scales by transition entropy for the ACTIVE mode.
#
# Example:
# python aligned_biochem_visuals.py \
#   --input ../ref_db/all_cruise_metadata.tsv \
#   --outdir ../V4_ncbi_output/aligned_biochem_cruise \
#   --required "Temperature,Oxygen,Nitrogen Oxides,Nitrate,Nitrite,Ammonium,Hydrogen Sulfide,Phosphate,Silicate" \
#   --delimiter $'\t' \
#   --metric cosine --n-neighbors 20 --n-epochs 200 \
#   --depth-tol-m 5 --anchor-min-count 3 --anchor-min-coverage 0.25 --max-nan-feature 0.4 --max-gap 2 \
#   --aspect-x 1 --aspect-y 12 --aspect-z 1 --trace-width 6 \
#   --hdbscan \
#   --hdbscan-umap-min-cluster-size 20 --hdbscan-umap-min-samples 10 \
#   --hdbscan-feat-min-cluster-size 25 --hdbscan-feat-min-samples 10

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import umap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


DEFAULT_COMPARTMENTS = [
    ("oxic", 90.0, np.inf, "red"),
    ("dysoxic", 20.0, 90.0, "lightskyblue"),
    ("suboxic", 1.0, 20.0, "green"),
    ("anoxic", -np.inf, 1.0, "purple"),
]
DEFAULT_COMPARTMENT_ORDER = [x[0] for x in DEFAULT_COMPARTMENTS] + ["unknown"]

# Deterministic cluster colors (cycled)
CLUSTER_COLOR_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


@dataclass(frozen=True)
class Args:
    input: str
    outdir: str
    delimiter: str

    cruise_col: str
    depth_col: str
    oxygen_col: str

    required: List[str]

    max_nan_feature: float
    anchor_min_count: int
    anchor_top_k: int
    anchor_min_coverage: float
    depth_tol_m: float
    drop_unassigned: bool

    impute_method: str
    max_gap: int
    impute_fallback: str

    metric: str
    n_neighbors: int
    n_epochs: int
    random_state: int

    umap_components: int  # 2 or 3

    plot_every: int
    point_size: float

    trace_width: float

    aspect_x: float
    aspect_y: float
    aspect_z: float
    debug_anchors: bool

    # HDBSCAN (added)
    hdbscan: bool
    hdbscan_umap_min_cluster_size: int
    hdbscan_umap_min_samples: int
    hdbscan_feat_min_cluster_size: int
    hdbscan_feat_min_samples: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="UMAP for cruise-sliced biochem data with shared depth anchors."
    )

    p.add_argument("--input", required=True, help="Input table (TSV/CSV).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--delimiter", default="\t", help=r"Delimiter for input. Default: TAB. Use ',' for CSV.")

    p.add_argument("--cruise-col", default="Cruise")
    p.add_argument("--depth-col", default="Depth")
    p.add_argument("--oxygen-col", default="Oxygen")

    p.add_argument(
        "--required",
        required=True,
        help="Comma-separated list of feature columns to include in the model.",
    )

    p.add_argument(
        "--max-nan-feature",
        type=float,
        default=0.40,
        help="ERROR if required feature has missing fraction > this. Default 0.40",
    )

    p.add_argument("--anchor-min-count", type=int, default=3)
    p.add_argument("--anchor-top-k", type=int, default=0)
    p.add_argument(
        "--anchor-min-coverage",
        type=float,
        default=0.25,
        help="Fraction of total rows an anchor depth must cover to be included (0-1). Default 0.25.",
    )

    p.add_argument("--depth-tol-m", type=float, default=5.0)
    p.add_argument("--drop-unassigned", action="store_true")

    p.add_argument("--impute-method", choices=["time_linear"], default="time_linear")
    p.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Max consecutive missing cruises to fill (within existing cruises only). Default 2",
    )
    p.add_argument(
        "--impute-fallback",
        choices=["depth_median", "global_median", "none"],
        default="depth_median",
    )

    p.add_argument("--metric", default="cosine")
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument(
        "--umap-components",
        type=int,
        choices=[2, 3],
        default=2,
        help="UMAP embedding dims. Default 2. (Trajectories use first 2 dims either way.)",
    )

    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--point-size", type=float, default=120.0)

    p.add_argument(
        "--trace-width",
        type=float,
        default=6.0,
        help="Trajectory line width in Plotly. Default 6.0",
    )

    # aspect ratio flags
    p.add_argument("--aspect-x", type=float, default=1.0, help="3D scene aspect ratio X. Default 1.0")
    p.add_argument("--aspect-y", type=float, default=10.0, help="3D scene aspect ratio Y (cruise axis). Default 10.0")
    p.add_argument("--aspect-z", type=float, default=1.0, help="3D scene aspect ratio Z. Default 1.0")

    p.add_argument("--debug-anchors", action="store_true", help="Print depth anchor selection stats in `choose_depth_anchors`.")

    # HDBSCAN flags (added; does not alter existing O2 behavior)
    p.add_argument("--hdbscan", action="store_true", help="Compute HDBSCAN clusters and enable cluster modes in HTML.")
    p.add_argument("--hdbscan-umap-min-cluster-size", type=int, default=20, help="HDBSCAN (UMAP space) min_cluster_size. Default 20.")
    p.add_argument("--hdbscan-umap-min-samples", type=int, default=10, help="HDBSCAN (UMAP space) min_samples. Default 10.")
    p.add_argument("--hdbscan-feat-min-cluster-size", type=int, default=20, help="HDBSCAN (feature space) min_cluster_size. Default 20.")
    p.add_argument("--hdbscan-feat-min-samples", type=int, default=10, help="HDBSCAN (feature space) min_samples. Default 10.")

    a = p.parse_args()

    required = [x.strip() for x in a.required.split(",") if x.strip()]
    if not required:
        raise SystemExit("ERROR: --required must contain at least one feature column name.")

    if not (0.0 <= a.anchor_min_coverage <= 1.0):
        raise SystemExit("ERROR: --anchor-min-coverage must be between 0 and 1 (inclusive).")

    if a.hdbscan_umap_min_cluster_size < 2 or a.hdbscan_feat_min_cluster_size < 2:
        raise SystemExit("ERROR: HDBSCAN min-cluster-size must be >= 2.")
    if a.hdbscan_umap_min_samples < 1 or a.hdbscan_feat_min_samples < 1:
        raise SystemExit("ERROR: HDBSCAN min-samples must be >= 1.")

    return Args(
        input=a.input,
        outdir=a.outdir,
        delimiter=a.delimiter,
        cruise_col=a.cruise_col,
        depth_col=a.depth_col,
        oxygen_col=a.oxygen_col,
        required=required,
        max_nan_feature=a.max_nan_feature,
        anchor_min_count=a.anchor_min_count,
        anchor_top_k=a.anchor_top_k,
        anchor_min_coverage=a.anchor_min_coverage,
        depth_tol_m=a.depth_tol_m,
        drop_unassigned=bool(a.drop_unassigned),
        impute_method=a.impute_method,
        max_gap=a.max_gap,
        impute_fallback=a.impute_fallback,
        metric=a.metric,
        n_neighbors=a.n_neighbors,
        n_epochs=a.n_epochs,
        random_state=a.random_state,
        umap_components=a.umap_components,
        plot_every=a.plot_every,
        point_size=a.point_size,
        trace_width=a.trace_width,
        aspect_x=a.aspect_x,
        aspect_y=a.aspect_y,
        aspect_z=a.aspect_z,
        debug_anchors=bool(a.debug_anchors),
        hdbscan=bool(a.hdbscan),
        hdbscan_umap_min_cluster_size=a.hdbscan_umap_min_cluster_size,
        hdbscan_umap_min_samples=a.hdbscan_umap_min_samples,
        hdbscan_feat_min_cluster_size=a.hdbscan_feat_min_cluster_size,
        hdbscan_feat_min_samples=a.hdbscan_feat_min_samples,
    )


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(path, "tables"), exist_ok=True)


def load_table(args: Args) -> pd.DataFrame:
    df = pd.read_csv(args.input, sep=args.delimiter, low_memory=False)
    df = df.replace({"NAN": np.nan, "NaN": np.nan, "nan": np.nan, "": np.nan, "None": np.nan})

    required_cols = [args.cruise_col, args.depth_col, args.oxygen_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns: {missing}")

    df[args.cruise_col] = pd.to_numeric(df[args.cruise_col], errors="coerce").astype("Int64")
    df[args.depth_col] = pd.to_numeric(df[args.depth_col], errors="coerce")
    df[args.oxygen_col] = pd.to_numeric(df[args.oxygen_col], errors="coerce")

    df = df.dropna(subset=[args.cruise_col, args.depth_col]).copy()

    df["cruise_order"] = df[args.cruise_col].astype(int)
    df["slice_id"] = df["cruise_order"].map(lambda x: f"cruise_{x:04d}")

    df = df.sort_values(["cruise_order", args.depth_col]).reset_index(drop=True)
    return df


def validate_and_select_features(df: pd.DataFrame, args: Args) -> List[str]:
    missing = [c for c in args.required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: Some --required columns are missing from the table: {missing}")

    for c in args.required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nan_frac = df[args.required].isna().mean(axis=0).sort_values(ascending=False)
    bad = nan_frac[nan_frac > args.max_nan_feature].index.tolist()
    if bad:
        raise SystemExit(
            "ERROR: Required feature(s) exceed --max-nan-feature threshold.\n"
            f"  max_nan_feature={args.max_nan_feature}\n"
            "  Offending features (nan_fraction):\n"
            + "\n".join([f"    - {c}: {nan_frac[c]:.3f}" for c in bad])
        )

    return args.required


def choose_depth_anchors(df: pd.DataFrame, args: Args) -> np.ndarray:
    debug = args.debug_anchors

    def log(msg: str, *, force: bool = False) -> None:
        if debug or force:
            print(msg)

    total_rows = len(df)
    total_slices = df["slice_id"].nunique()
    log(
        f"[anchor_debug] total_rows={total_rows}, total_slices={total_slices}, "
        f"anchor_min_count={args.anchor_min_count}, anchor_top_k={args.anchor_top_k}, anchor_min_coverage={args.anchor_min_coverage}"
    )

    rounded = df[args.depth_col].round(3)
    depth_slice = pd.DataFrame({"rounded_depth": rounded, "slice_id": df["slice_id"]})
    counts = depth_slice["rounded_depth"].value_counts().sort_values(ascending=False)
    slice_counts = (
        depth_slice.drop_duplicates()
        .groupby("rounded_depth")["slice_id"]
        .nunique()
        .reindex(counts.index)
        .fillna(0)
        .astype(int)
    )

    log(f"[anchor_debug] depth counts sample (top 10):\n{counts.head(10)}")
    if debug and total_rows > 0 and total_slices > 0:
        coverage_lines = "\n".join(
            f"    {float(depth):.3f}m — {int(cnt)} rows ({cnt/total_rows*100:.2f}% rows, "
            f"{slice_counts[depth]}/{total_slices} slices ({slice_counts[depth]/total_slices*100:.2f}% coverage))"
            for depth, cnt in counts.items()
        )
        log(f"[anchor_debug] coverage per depth (rows + cruise slices):\n{coverage_lines}")

    if args.anchor_top_k and args.anchor_top_k > 0:
        selected = counts.head(args.anchor_top_k)
    else:
        selected = counts[counts >= args.anchor_min_count]

    log(f"[anchor_debug] selected depths after count filter (n={len(selected)}): {selected.index.tolist()[:10]}")

    if args.anchor_min_coverage > 0 and total_slices > 0:
        coverage_threshold = max(1, int(np.ceil(total_slices * args.anchor_min_coverage)))
        slice_counts_for_selected = slice_counts.reindex(selected.index).fillna(0).astype(int)
        mask = slice_counts_for_selected >= coverage_threshold
        selected = selected[mask]
        log(
            f"[anchor_debug] applying coverage filter threshold={coverage_threshold} (per cruise slices), "
            f"depths remaining={len(selected)}"
        )

    anchors = selected.index.to_numpy(dtype=float)
    log(f"[anchor_debug] final anchors (n={len(anchors)}): {anchors.tolist()}")

    if anchors.size == 0:
        log(f"[anchor_debug] drained all anchors - final selected counts:\n{selected.head(10)}", force=True)
        raise SystemExit(
            "ERROR: No depth anchors selected. Try lowering --anchor-min-count, reducing --anchor-min-coverage, or using --anchor-top-k."
        )

    return np.sort(anchors.astype(float))


def nearest_anchor(depth: float, anchors: np.ndarray) -> Tuple[float, float]:
    idx = int(np.argmin(np.abs(anchors - depth)))
    a = float(anchors[idx])
    return a, float(abs(a - depth))


def assign_anchor_depths(df: pd.DataFrame, anchors: np.ndarray, args: Args) -> pd.DataFrame:
    df = df.copy()

    slice_table = (
        df[["slice_id", "cruise_order"]]
        .drop_duplicates()
        .sort_values("cruise_order")
    )
    slices = slice_table["slice_id"].tolist()
    slice_index = {s: i for i, s in enumerate(slices)}

    slice_to_depths = {
        s: df.loc[df["slice_id"] == s, args.depth_col].to_numpy(dtype=float)
        for s in slices
    }

    def snap_if_close(d: float) -> Optional[float]:
        a, dist = nearest_anchor(d, anchors)
        return a if dist <= args.depth_tol_m else None

    slice_to_present_anchors: Dict[str, np.ndarray] = {}
    for s in slices:
        depths = slice_to_depths[s]
        snapped = [snap_if_close(float(d)) for d in depths]
        present = [x for x in snapped if x is not None]
        slice_to_present_anchors[s] = (
            np.array(sorted(set(map(float, present))), dtype=float)
            if present
            else np.array([], dtype=float)
        )

    assigned = []
    for _, row in df.iterrows():
        s = row["slice_id"]
        d = float(row[args.depth_col])

        a0 = snap_if_close(d)
        if a0 is not None:
            assigned.append(a0)
            continue

        i = slice_index[s]
        prev_s = slices[i - 1] if i - 1 >= 0 else None
        next_s = slices[i + 1] if i + 1 < len(slices) else None

        candidates: List[Tuple[float, float]] = []

        if prev_s is not None:
            prev_anchors = slice_to_present_anchors[prev_s]
            if prev_anchors.size:
                ap, distp = nearest_anchor(d, prev_anchors)
                if distp <= args.depth_tol_m:
                    candidates.append((float(ap), float(distp)))

        if next_s is not None:
            next_anchors = slice_to_present_anchors[next_s]
            if next_anchors.size:
                an, distn = nearest_anchor(d, next_anchors)
                if distn <= args.depth_tol_m:
                    candidates.append((float(an), float(distn)))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            assigned.append(float(candidates[0][0]))
        else:
            assigned.append(np.nan)

    df["anchor_depth"] = pd.to_numeric(assigned, errors="coerce")
    if args.drop_unassigned:
        df = df.dropna(subset=["anchor_depth"]).copy()
    return df


def aggregate_pseudo_reps(df: pd.DataFrame, feature_cols: List[str], args: Args) -> pd.DataFrame:
    group_cols = ["slice_id", "anchor_depth"]

    agg_cols = list(feature_cols)
    if args.oxygen_col not in agg_cols:
        agg_cols.append(args.oxygen_col)

    out = (
        df[group_cols + agg_cols]
        .groupby(group_cols, dropna=False)
        .agg(["mean", "count"])
    )
    out.columns = [f"{c}__{stat}" for (c, stat) in out.columns]
    out = out.reset_index()

    cruise_map = df[["slice_id", "cruise_order"]].drop_duplicates()
    out = out.merge(cruise_map, on="slice_id", how="left")

    for c in agg_cols:
        out[c] = out[f"{c}__mean"]

    out["n_obs_in_cell"] = out[f"{args.oxygen_col}__count"].fillna(0).astype(int)

    keep = ["slice_id", "cruise_order", "anchor_depth", "n_obs_in_cell"] + agg_cols + [f"{c}__count" for c in agg_cols]
    return out[keep].copy()


def time_impute(df_agg: pd.DataFrame, feature_cols: List[str], args: Args) -> pd.DataFrame:
    cols_to_impute = list(feature_cols)
    if args.oxygen_col not in cols_to_impute:
        cols_to_impute.append(args.oxygen_col)

    out_parts = []
    for depth, sub in df_agg.groupby("anchor_depth", dropna=False):
        sub = sub.copy().sort_values("cruise_order").set_index("cruise_order")

        for c in cols_to_impute:
            if c not in sub.columns:
                continue
            s = sub[c].astype(float)
            sub[c] = s.interpolate(method="linear", limit=args.max_gap, limit_area="inside")

        if args.impute_fallback != "none":
            if args.impute_fallback == "depth_median":
                for c in cols_to_impute:
                    if c not in sub.columns:
                        continue
                    arr = sub[c].to_numpy(dtype=float)
                    if np.isnan(arr).all():
                        continue
                    med = np.nanmedian(arr)
                    if np.isfinite(med):
                        sub[c] = sub[c].fillna(med)
            elif args.impute_fallback == "global_median":
                pass

        sub = sub.reset_index()
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True)

    if args.impute_fallback == "global_median":
        for c in cols_to_impute:
            if c in out.columns:
                arr = out[c].to_numpy(dtype=float)
                if np.isnan(arr).all():
                    continue
                med = np.nanmedian(arr)
                if np.isfinite(med):
                    out[c] = out[c].fillna(med)

    return out


def oxygen_compartment(o2_um: float) -> Tuple[str, str]:
    for name, lo, hi, color in DEFAULT_COMPARTMENTS:
        if (o2_um > lo) and (o2_um <= hi):
            return name, color
    return "unknown", "gray"


def run_umap_embedding(df: pd.DataFrame, feature_cols: List[str], args: Args) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value .*overridden to 1 by setting random_state.*",
        )
        mapper = umap.UMAP(
            n_components=args.umap_components,
            metric=args.metric,
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            random_state=args.random_state,
        )
        return mapper.fit_transform(df[feature_cols].to_numpy(dtype=float))


def plot_global_overlays(
    emb_df: pd.DataFrame,
    args: Args,
    hdbscan_labels: Optional[np.ndarray] = None,
) -> None:
    plots_dir = os.path.join(args.outdir, "plots")
    if emb_df.empty:
        return

    x = emb_df["umap1"].to_numpy(dtype=float)
    y = emb_df["umap2"].to_numpy(dtype=float)
    size = max(8.0, args.point_size / 4.0)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 0.1
    y_pad = (y_max - y_min) * 0.02 if y_max > y_min else 0.1
    common_xlim = (x_min - x_pad, x_max + x_pad)
    common_ylim = (y_min - y_pad, y_max + y_pad)

    o2 = emb_df[args.oxygen_col].to_numpy(dtype=float)
    valid_o2 = np.isfinite(o2)

    cruise_order = emb_df["cruise_order"].to_numpy(dtype=float)
    valid_cruise = np.isfinite(cruise_order)

    comp_names = []
    comp_colors = []
    for v in o2:
        if np.isfinite(v):
            name, color = oxygen_compartment(float(v))
        else:
            name, color = "unknown", "gray"
        comp_names.append(name)
        comp_colors.append(color)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()
    def apply_common_limits(ax):
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

    # Panel 1: Oxygen scalar overlay
    ax = axes[0]
    scatter = None
    if valid_o2.any():
        scatter = ax.scatter(x[valid_o2], y[valid_o2], c=o2[valid_o2], s=size,
                             cmap="viridis", alpha=0.9, edgecolors="none")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(args.oxygen_col)
        if (~valid_o2).any():
            ax.scatter(x[~valid_o2], y[~valid_o2], color="gray", s=size, alpha=0.4, edgecolors="none",
                       label="missing O2")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_title("UMAP — O2 overlay")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    apply_common_limits(ax)

    # Panel 2: Compartments
    ax = axes[1]
    comp_df = pd.DataFrame(
        {"name": comp_names, "color": comp_colors, "x": x, "y": y}
    )
    for comp in (DEFAULT_COMPARTMENT_ORDER + ["unknown"]):
        comp_mask = comp_df["name"] == comp
        if not comp_mask.any():
            continue
        subset = comp_df.loc[comp_mask]
        ax.scatter(subset["x"], subset["y"], color=subset["color"].iloc[0], s=size,
                   alpha=0.85, edgecolors="none", label=comp)
    ax.set_title("UMAP — compartments")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    apply_common_limits(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
              markerscale=1.2, fontsize="small")

    # Panel 3: Cruise order gradient
    ax = axes[2]
    if valid_cruise.any():
        cruise_norm = (cruise_order[valid_cruise] - np.nanmin(cruise_order[valid_cruise]))
        denom = np.nanmax(cruise_norm) if np.nanmax(cruise_norm) and np.isfinite(np.nanmax(cruise_norm)) else 1.0
        cruise_norm = cruise_norm / denom if denom != 0 else cruise_norm
        scatter = ax.scatter(x[valid_cruise], y[valid_cruise], c=cruise_norm, s=size,
                             cmap="Greys", alpha=0.9, edgecolors="none")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("cruise (order normalized)")
        if (~valid_cruise).any():
            ax.scatter(x[~valid_cruise], y[~valid_cruise], color="gray", s=size, alpha=0.4,
                       edgecolors="none", label="missing cruise")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_title("UMAP — cruise order")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    apply_common_limits(ax)
    for ax in axes[:3]:
        ax.set_aspect("equal", adjustable="datalim")

    # Panel 4: HDBSCAN from UMAP coords
    ax = axes[3]
    if hdbscan_labels is not None:
        labels = hdbscan_labels.astype(int)
        scatter_artists = []
        for lbl in sorted(np.unique(labels)):
            mask = labels == lbl
            if not mask.any():
                continue
            color = _hdbscan_color_for_label(int(lbl))
            label = f"cluster_{lbl}" if lbl >= 0 else "noise"
            scatter_artists.append(
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=color,
                    s=size,
                    alpha=0.85,
                    edgecolors="none",
                    label=label,
                )
            )
        ax.set_title("UMAP — HDBSCAN (UMAP coords)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        apply_common_limits(ax)
        if scatter_artists:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
                      markerscale=1.2, fontsize="small")
    else:
        ax.text(0.5, 0.5, "HDBSCAN not available", ha="center", va="center")
        ax.set_title("UMAP — HDBSCAN (UMAP coords)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        apply_common_limits(ax)
    ax.set_aspect("equal", adjustable="datalim")

    fig.subplots_adjust(wspace=0.5, hspace=0.4, right=0.85)
    outpath = os.path.join(plots_dir, "umap_global_overlays.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_embeddings(emb_df: pd.DataFrame, slice_ids: List[str], args: Args) -> None:
    plots_dir = os.path.join(args.outdir, "plots")

    for i, s in enumerate(slice_ids):
        if args.plot_every > 1 and (i % args.plot_every != 0):
            continue

        sub = (
            emb_df.loc[emb_df["slice_id"] == s]
            .sort_values("anchor_depth")
            .reset_index(drop=True)
        )
        if sub.empty:
            continue

        o2 = sub[args.oxygen_col].to_numpy(dtype=float)
        colors = []
        for v in o2:
            if np.isfinite(v):
                _, col = oxygen_compartment(float(v))
            else:
                col = "gray"
            colors.append(col)

        plt.figure()
        plt.scatter(sub["umap1"].to_numpy(dtype=float), sub["umap2"].to_numpy(dtype=float),
                    s=args.point_size, c=colors, alpha=0.85, edgecolors="none")
        plt.title(f"UMAP — {s} (O2 compartments)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        outpath = os.path.join(plots_dir, f"umap_{s}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()


def _hdbscan_color_for_label(label: int) -> str:
    if label == -1:
        return "#9e9e9e"  # noise
    return CLUSTER_COLOR_CYCLE[int(label) % len(CLUSTER_COLOR_CYCLE)]


def _robust_scale_matrix(X: np.ndarray) -> np.ndarray:
    """
    Robust scaling: (x - median) / IQR, column-wise.
    - Ignores NaNs.
    - Guards IQR==0 by setting denom=1.
    """
    X = X.astype(float, copy=True)
    med = np.nanmedian(X, axis=0)
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[~np.isfinite(iqr)] = 1.0
    iqr[iqr == 0] = 1.0
    med[~np.isfinite(med)] = 0.0
    return (X - med) / iqr


def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    s = counts.sum()
    if not np.isfinite(s) or s <= 0:
        return float("nan")
    p = counts / s
    p = p[(p > 0) & np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def _compute_transition_entropy_per_depth(
    emb_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """
    For each depth:
      - order by cruise_order
      - split blocks where cruise gap > 1 (no transition across missing cruises)
      - within each block, compute transitions between consecutive labels
      - entropy over transition TYPES (a->b), normalized by log(K) where K=#unique transition types
      - also compute switch_rate = (#label-changes)/(#adjacent-steps)
    Returns DF with: anchor_depth, n_steps, n_transitions, switch_rate, trans_entropy, trans_entropy_norm
    """
    rows = []
    for depth, sub in emb_df.groupby("anchor_depth"):
        sub = sub.sort_values("cruise_order").reset_index(drop=True)
        if len(sub) < 2:
            continue

        t = sub["cruise_order"].to_numpy(dtype=float)
        splits = np.where(np.diff(t) > 1)[0] + 1
        blocks = np.split(np.arange(len(sub)), splits)

        trans_counts: Dict[str, int] = {}
        n_adj = 0
        n_change = 0

        for idxs in blocks:
            if len(idxs) < 2:
                continue
            sb = sub.loc[idxs].reset_index(drop=True)
            lab = sb[label_col].astype(str).to_numpy(dtype=object)

            # adjacent steps inside the block
            for i in range(len(lab) - 1):
                a = str(lab[i])
                b = str(lab[i + 1])
                n_adj += 1
                if a != b:
                    n_change += 1
                key = f"{a}→{b}"
                trans_counts[key] = trans_counts.get(key, 0) + 1

        if n_adj == 0:
            continue

        counts = np.array(list(trans_counts.values()), dtype=float)
        H = _shannon_entropy_from_counts(counts)
        K = len(trans_counts)
        Hn = float("nan")
        if np.isfinite(H) and K > 1:
            Hn = float(H / np.log(K))
        elif np.isfinite(H) and K == 1:
            Hn = 0.0

        rows.append(
            dict(
                anchor_depth=float(depth),
                n_steps=int(len(sub)),
                n_adjacent=int(n_adj),
                n_label_changes=int(n_change),
                switch_rate=float(n_change / n_adj) if n_adj > 0 else float("nan"),
                trans_entropy=float(H),
                trans_entropy_norm=float(Hn),
            )
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # stable sort
    out = out.sort_values("anchor_depth").reset_index(drop=True)
    return out


def _save_transition_entropy_barplot(df_ent: pd.DataFrame, outpath: str, title: str) -> None:
    if df_ent is None or df_ent.empty:
        return

    x = df_ent["anchor_depth"].astype(float).to_numpy()
    y = df_ent["trans_entropy_norm"].astype(float).to_numpy()

    # Replace NaN with 0 for plotting (still keep NaN in table)
    y_plot = np.where(np.isfinite(y), y, 0.0)

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(x)), y_plot)
    plt.xticks(np.arange(len(x)), [f"{v:g}" for v in x], rotation=90)
    plt.ylabel("Transition entropy (normalized)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def compute_hdbscan_labels(
    emb_df: pd.DataFrame,
    feature_cols: List[str],
    args: Args,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns: (labels_umap, labels_feat) where each is aligned with emb_df rows.
    If args.hdbscan is False or hdbscan isn't installed, returns (None, None).

    NOTE: Feature-space clustering uses ROBUST scaling (median/IQR) now.
    """
    if not args.hdbscan:
        return None, None

    try:
        import hdbscan  # type: ignore
    except Exception as e:
        print("WARNING: --hdbscan requested but `hdbscan` is not importable. Install via `pip install hdbscan`.")
        print("  Error:", e)
        return None, None

    # HDBSCAN on UMAP coords (2D)
    X_umap = emb_df[["umap1", "umap2"]].to_numpy(dtype=float)
    clu_umap = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdbscan_umap_min_cluster_size),
        min_samples=int(args.hdbscan_umap_min_samples),
    )
    labels_umap = clu_umap.fit_predict(X_umap)

    # HDBSCAN on robust-scaled feature space
    X_feat = emb_df[feature_cols].to_numpy(dtype=float)
    X_feat_rs = _robust_scale_matrix(X_feat)

    clu_feat = hdbscan.HDBSCAN(
        min_cluster_size=int(args.hdbscan_feat_min_cluster_size),
        min_samples=int(args.hdbscan_feat_min_samples),
    )
    labels_feat = clu_feat.fit_predict(X_feat_rs)

    return labels_umap, labels_feat


def plot_3d_checkbox_html_with_modes(
    emb_df: pd.DataFrame,
    args: Args,
    has_hdbscan: bool,
    entropy_maps: Optional[Dict[str, Dict[str, float]]] = None,  # mode -> depth_str -> norm_entropy
) -> None:
    """
    Standalone HTML with:
      - Mode radio: O2 vs HDBSCAN-UMAP vs HDBSCAN-FEATURES (if available)
      - Group checkboxes: compartments OR clusters (depending on mode)
      - Depth checkboxes
      - Eligibility rule: if a depth EVER appears in ANY selected group, ENTIRE depth track eligible to show.

    ADDED:
      - Trajectory link width scales by transition entropy for the trace's depth, per mode.
        width = args.trace_width * (0.5 + 1.5 * entropy_norm) where entropy_norm in [0,1].
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("WARNING: plotly not installed; skipping 3D trajectory HTML. Error:", e)
        return

    plots_dir = os.path.join(args.outdir, "plots")
    out_html = os.path.join(plots_dir, "umap_3d_trajectories_checkboxes.html")

    # Build label columns (always keep O2)
    o2_comp = []
    o2_color = []
    for v in emb_df[args.oxygen_col].to_numpy(dtype=float):
        if np.isfinite(v):
            name, col = oxygen_compartment(float(v))
        else:
            name, col = "unknown", "gray"
        o2_comp.append(name)
        o2_color.append(col)
    emb_df = emb_df.copy()
    emb_df["label_o2"] = o2_comp
    emb_df["color_o2"] = o2_color

    # HDBSCAN labels if present
    if has_hdbscan and ("hdbscan_umap_label" in emb_df.columns) and ("hdbscan_feat_label" in emb_df.columns):
        emb_df["label_hdb_umap"] = emb_df["hdbscan_umap_label"].astype(int).map(lambda x: f"cluster_{x}" if x >= 0 else "noise")
        emb_df["label_hdb_feat"] = emb_df["hdbscan_feat_label"].astype(int).map(lambda x: f"cluster_{x}" if x >= 0 else "noise")
        emb_df["color_hdb_umap"] = emb_df["hdbscan_umap_label"].astype(int).map(_hdbscan_color_for_label)
        emb_df["color_hdb_feat"] = emb_df["hdbscan_feat_label"].astype(int).map(_hdbscan_color_for_label)
    else:
        emb_df["label_hdb_umap"] = ""
        emb_df["label_hdb_feat"] = ""
        emb_df["color_hdb_umap"] = "#9e9e9e"
        emb_df["color_hdb_feat"] = "#9e9e9e"

    depth_values = np.sort(emb_df["anchor_depth"].dropna().unique().astype(float))

    modes = ["o2"]
    if has_hdbscan and ("hdbscan_umap_label" in emb_df.columns) and ("hdbscan_feat_label" in emb_df.columns):
        modes += ["hdb_umap", "hdb_feat"]

    mode_label_cols = {"o2": "label_o2", "hdb_umap": "label_hdb_umap", "hdb_feat": "label_hdb_feat"}
    mode_color_cols = {"o2": "color_o2", "hdb_umap": "color_hdb_umap", "hdb_feat": "color_hdb_feat"}
    mode_titles = {"o2": "Legacy O2 compartments", "hdb_umap": "HDBSCAN clusters (UMAP space)", "hdb_feat": "HDBSCAN clusters (feature space)"}

    # Width scaling maps (mode -> depth_str -> width)
    # Default: constant width
    width_by_mode_depth: Dict[str, Dict[str, float]] = {m: {} for m in modes}
    if entropy_maps is None:
        entropy_maps = {}

    for m in modes:
        em = entropy_maps.get(m, {})
        for d in depth_values:
            ds = f"{d:g}"
            en = float(em.get(ds, 0.0))
            if not np.isfinite(en):
                en = 0.0
            # clamp
            en = max(0.0, min(1.0, en))
            width_by_mode_depth[m][ds] = float(args.trace_width) * (0.5 + 1.5 * en)

    fig = go.Figure()

    # For JS: trace metadata
    trace_meta = []  # list of dicts parallel to fig.data

    # For eligibility: for each mode, for each depth, set of groups encountered
    depth_groups_by_mode: Dict[str, Dict[str, List[str]]] = {}  # mode -> depth_str -> sorted groups
    all_groups_by_mode: Dict[str, List[str]] = {}

    for mode in modes:
        depth_groups_by_mode[mode] = {}

        # Precompute groups encountered per depth
        for depth in depth_values:
            sub = emb_df.loc[emb_df["anchor_depth"] == depth].sort_values("cruise_order").reset_index(drop=True)
            if len(sub) == 0:
                continue
            groups = sorted(set(sub[mode_label_cols[mode]].astype(str).tolist()))
            groups = [g for g in groups if g and g != "nan"]
            depth_groups_by_mode[mode][f"{depth:g}"] = groups

        # Global groups for mode
        gset = set()
        for _d, gs in depth_groups_by_mode[mode].items():
            gset.update(gs)
        all_groups_by_mode[mode] = sorted(gset)

    # Add traces (segments)
    for mode in modes:
        for depth in depth_values:
            sub = emb_df.loc[emb_df["anchor_depth"] == depth].sort_values("cruise_order").reset_index(drop=True)
            if len(sub) < 2:
                continue

            t = sub["cruise_order"].to_numpy(dtype=float)
            # split where cruise gap > 1
            splits = np.where(np.diff(t) > 1)[0] + 1
            blocks = np.split(np.arange(len(sub)), splits)

            depth_str = f"{depth:g}"
            line_w = width_by_mode_depth.get(mode, {}).get(depth_str, float(args.trace_width))

            for idxs in blocks:
                if len(idxs) < 2:
                    continue
                sb = sub.loc[idxs].reset_index(drop=True)
                for i in range(len(sb) - 1):
                    x0, z0 = float(sb.loc[i, "umap1"]), float(sb.loc[i, "umap2"])
                    x1, z1 = float(sb.loc[i + 1, "umap1"]), float(sb.loc[i + 1, "umap2"])
                    y0 = float(sb.loc[i, "cruise_order"])
                    y1 = float(sb.loc[i + 1, "cruise_order"])

                    seg_group = str(sb.loc[i, mode_label_cols[mode]])
                    seg_color = str(sb.loc[i, mode_color_cols[mode]])

                    # hover
                    hover = (
                        f"Mode={mode_titles[mode]}<br>"
                        f"Depth={depth:g}m<br>"
                        f"Cruise={int(sb.loc[i,'cruise_order'])} → {int(sb.loc[i+1,'cruise_order'])}<br>"
                        f"Group={seg_group}<br>"
                        f"Width={line_w:.2f}"
                    )

                    # Only show O2 mode by default; others hidden until selected
                    visible_default = True if mode == "o2" else False

                    fig.add_trace(
                        go.Scatter3d(
                            x=[x0, x1],
                            y=[y0, y1],
                            z=[z0, z1],
                            mode="lines",
                            line=dict(width=float(line_w), color=seg_color),
                            showlegend=False,
                            hovertext=hover,
                            hoverinfo="text",
                            visible=visible_default,
                        )
                    )
                    trace_meta.append(
                        dict(
                            mode=mode,
                            depth=depth_str,
                            group=seg_group,
                        )
                    )

    if len(fig.data) == 0:
        raise SystemExit("ERROR: No trajectory segments were generated. Check anchors / depth assignment / data sparsity.")

    fig.update_layout(
        title="UMAP trajectories (checkbox UI; mode switch: O2 vs HDBSCAN clusters)",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="Cruise (order)",
            zaxis_title="UMAP-2",
            aspectmode="manual",
            aspectratio=dict(x=float(args.aspect_x), y=float(args.aspect_y), z=float(args.aspect_z)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # Serialize data for embedded JS
    trace_meta_json = json.dumps(trace_meta)
    depth_values_json = json.dumps([f"{d:g}" for d in depth_values.tolist()])
    all_groups_by_mode_json = json.dumps(all_groups_by_mode)
    depth_groups_by_mode_json = json.dumps(depth_groups_by_mode)
    modes_json = json.dumps(modes)
    mode_titles_json = json.dumps(mode_titles)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>UMAP Trajectories</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; }}
    #wrap {{ display:flex; height: 100vh; }}
    #side {{
      width: 360px; padding: 12px 12px 16px 12px; overflow:auto;
      border-right: 1px solid #ddd; background: #fafafa;
    }}
    #plot {{ flex:1; }}
    .section-title {{ font-weight: 700; margin: 10px 0 6px 0; }}
    .hint {{ color:#555; font-size: 12px; line-height: 1.25em; margin-bottom: 8px; }}
    .row {{ margin: 4px 0; }}
    .swatch {{ display:inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; border: 1px solid #999; vertical-align: middle; }}
    .btn {{ margin-right: 6px; margin-top: 6px; }}
    .small {{ font-size: 12px; color:#444; }}
    .divider {{ height:1px; background:#ddd; margin:10px 0; }}
    label {{ cursor:pointer; }}
  </style>
</head>
<body>
<div id="wrap">
  <div id="side">
    <div class="section-title">Mode</div>
    <div class="hint">
      Choose how trajectories are colored + how the "Group" checkbox list is interpreted.
      (Line widths scale by transition entropy for the chosen mode.)
    </div>
    <div id="modeRadios"></div>

    <div class="divider"></div>

    <div class="section-title">Groups</div>
    <div class="hint">
      Eligibility rule: a depth track is eligible if it EVER appears in ANY selected group (in the chosen mode).
      Depth checkboxes then toggle eligible tracks on/off.
    </div>
    <div>
      <button class="btn" id="groupsAll">All</button>
      <button class="btn" id="groupsNone">None</button>
    </div>
    <div id="groupChecks" style="margin-top:6px;"></div>

    <div class="divider"></div>

    <div class="section-title">Depths</div>
    <div class="hint">These toggle entire tracks (once eligible).</div>
    <div>
      <button class="btn" id="depthsAll">All</button>
      <button class="btn" id="depthsNone">None</button>
    </div>
    <div id="depthChecks" style="margin-top:6px;"></div>

    <div class="divider"></div>
    <div class="small">
      Tip: Use --aspect-y to make the cruise axis "longer" without changing data.
      Transition entropy outputs: see plots/transition_entropy_*.png and tables/transition_entropy_*.tsv
    </div>
  </div>

  <div id="plot"></div>
</div>

<script>
  const fig = {json.dumps(fig.to_plotly_json())};

  const traceMeta = {trace_meta_json};
  const depthList = {depth_values_json};
  const modes = {modes_json};
  const modeTitles = {mode_titles_json};
  const allGroupsByMode = {all_groups_by_mode_json};
  const depthGroupsByMode = {depth_groups_by_mode_json};

  // Current UI state
  let currentMode = "o2";
  let selectedGroups = new Set(allGroupsByMode[currentMode] || []);
  let selectedDepths = new Set(depthList);

  // Render plot
  Plotly.newPlot("plot", fig.data, fig.layout, {{responsive: true}});

  function buildModeRadios() {{
    const host = document.getElementById("modeRadios");
    host.innerHTML = "";
    modes.forEach((m) => {{
      const id = "mode_" + m;
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `
        <label>
          <input type="radio" name="mode" id="${{id}}" value="${{m}}" ${{m===currentMode ? "checked": ""}}>
          ${{modeTitles[m] || m}}
        </label>
      `;
      host.appendChild(row);
      row.querySelector("input").addEventListener("change", (ev) => {{
        currentMode = ev.target.value;
        // reset group selection to "all" in the new mode (safe default)
        selectedGroups = new Set(allGroupsByMode[currentMode] || []);
        renderGroupChecks();
        applyVisibility();
      }});
    }});
  }}

  function renderGroupChecks() {{
    const host = document.getElementById("groupChecks");
    host.innerHTML = "";
    const groups = allGroupsByMode[currentMode] || [];
    if (!groups.length) {{
      const p = document.createElement("div");
      p.className = "small";
      p.textContent = (currentMode === "o2") ? "No groups?? (unexpected)" : "No clusters available (run with --hdbscan).";
      host.appendChild(p);
      return;
    }}

    // For O2 mode, show compartment swatches. For clusters, show cluster color.
    groups.forEach((g) => {{
      const id = "grp_" + currentMode + "_" + g.replace(/[^a-zA-Z0-9_-]/g, "_");
      const row = document.createElement("div");
      row.className = "row";

      // swatch color:
      let color = "#999";
      if (currentMode === "o2") {{
        const cmap = {{
          "oxic":"red",
          "dysoxic":"lightskyblue",
          "suboxic":"green",
          "anoxic":"purple",
          "unknown":"gray"
        }};
        color = cmap[g] || "#999";
      }} else {{
        if (g === "noise") color = "#9e9e9e";
        else {{
          const m = g.match(/cluster_(-?\\d+)/);
          if (m) {{
            const k = parseInt(m[1], 10);
            const cycle = {json.dumps(CLUSTER_COLOR_CYCLE)};
            color = cycle[((k % cycle.length) + cycle.length) % cycle.length];
          }}
        }}
      }}

      const checked = selectedGroups.has(g) ? "checked" : "";
      row.innerHTML = `
        <label>
          <input type="checkbox" id="${{id}}" value="${{g}}" ${{checked}}>
          <span class="swatch" style="background:${{color}}"></span>
          ${{g}}
        </label>
      `;
      host.appendChild(row);
      row.querySelector("input").addEventListener("change", (ev) => {{
        const val = ev.target.value;
        if (ev.target.checked) selectedGroups.add(val);
        else selectedGroups.delete(val);
        applyVisibility();
      }});
    }});
  }}

  function renderDepthChecks() {{
    const host = document.getElementById("depthChecks");
    host.innerHTML = "";
    depthList.forEach((d) => {{
      const id = "dep_" + d.replace(/[^a-zA-Z0-9_-]/g, "_");
      const row = document.createElement("div");
      row.className = "row";
      const checked = selectedDepths.has(d) ? "checked" : "";
      row.innerHTML = `
        <label>
          <input type="checkbox" id="${{id}}" value="${{d}}" ${{checked}}>
          ${{d}} m
        </label>
      `;
      host.appendChild(row);
      row.querySelector("input").addEventListener("change", (ev) => {{
        const val = ev.target.value;
        if (ev.target.checked) selectedDepths.add(val);
        else selectedDepths.delete(val);
        applyVisibility();
      }});
    }});
  }}

  function applyVisibility() {{
    // Eligibility: depth is eligible if its set-of-groups intersects selectedGroups (for current mode)
    const eligibleDepth = new Set();
    const perDepth = depthGroupsByMode[currentMode] || {{}};
    Object.keys(perDepth).forEach((d) => {{
      const gs = perDepth[d] || [];
      for (const g of gs) {{
        if (selectedGroups.has(g)) {{
          eligibleDepth.add(d);
          break;
        }}
      }}
    }});

    // Build new visibility array
    const vis = new Array(traceMeta.length).fill(false);
    for (let i = 0; i < traceMeta.length; i++) {{
      const tm = traceMeta[i];
      if (tm.mode !== currentMode) continue;
      if (selectedDepths.has(tm.depth) && eligibleDepth.has(tm.depth)) {{
        vis[i] = true;
      }}
    }}

    Plotly.restyle("plot", {{visible: vis}});
  }}

  // Buttons
  document.getElementById("groupsAll").addEventListener("click", () => {{
    selectedGroups = new Set(allGroupsByMode[currentMode] || []);
    renderGroupChecks();
    applyVisibility();
  }});
  document.getElementById("groupsNone").addEventListener("click", () => {{
    selectedGroups = new Set();
    renderGroupChecks();
    applyVisibility();
  }});
  document.getElementById("depthsAll").addEventListener("click", () => {{
    selectedDepths = new Set(depthList);
    renderDepthChecks();
    applyVisibility();
  }});
  document.getElementById("depthsNone").addEventListener("click", () => {{
    selectedDepths = new Set();
    renderDepthChecks();
    applyVisibility();
  }});

  // Init
  buildModeRadios();
  renderGroupChecks();
  renderDepthChecks();
  applyVisibility();
</script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote checkbox trajectories HTML: {out_html}")
    if not has_hdbscan:
        print("NOTE: HDBSCAN modes unavailable (run with --hdbscan and ensure `hdbscan` is installed).")


def main() -> None:
    args = parse_args()
    ensure_outdir(args.outdir)

    df = load_table(args)
    feature_cols = validate_and_select_features(df, args)

    anchors = choose_depth_anchors(df, args)
    df_a = assign_anchor_depths(df, anchors, args)

    df_agg = aggregate_pseudo_reps(df_a, feature_cols, args)
    df_agg.to_csv(os.path.join(args.outdir, "tables", "agg_time_anchor_depth.tsv"), sep="\t", index=False)

    df_imp = time_impute(df_agg, feature_cols, args)

    # Final NaN cleanup for UMAP features
    global_medians = df_imp[feature_cols].median(axis=0, skipna=True)
    df_imp[feature_cols] = df_imp[feature_cols].fillna(global_medians)

    all_nan_features = [c for c in feature_cols if df_imp[c].isna().all()]
    if all_nan_features:
        print("WARNING: Dropping all-NaN features after imputation:", all_nan_features)
        feature_cols = [c for c in feature_cols if c not in all_nan_features]
        if not feature_cols:
            raise SystemExit("ERROR: After dropping all-NaN features, no features remain for UMAP.")

    before = len(df_imp)
    df_imp = df_imp.dropna(subset=feature_cols)
    after = len(df_imp)
    if after < before:
        print(f"WARNING: Dropped {before - after} rows that still had NaNs in features after cleanup.")

    df_imp.to_csv(os.path.join(args.outdir, "tables", "imputed_time_anchor_depth.tsv"), sep="\t", index=False)

    embedding_arr = run_umap_embedding(df_imp, feature_cols, args)
    emb_df = df_imp.copy()
    for dim_idx in range(args.umap_components):
        emb_df[f"umap{dim_idx + 1}"] = embedding_arr[:, dim_idx]

    slice_ids = (
        emb_df[["slice_id", "cruise_order"]]
        .drop_duplicates()
        .sort_values("cruise_order")["slice_id"]
        .tolist()
    )

    # HDBSCAN labels (added)
    labels_umap, labels_feat = compute_hdbscan_labels(emb_df, feature_cols, args)
    has_hdbscan = (labels_umap is not None) and (labels_feat is not None)

    if has_hdbscan:
        emb_df["hdbscan_umap_label"] = labels_umap.astype(int)
        emb_df["hdbscan_feat_label"] = labels_feat.astype(int)

    plot_global_overlays(
        emb_df,
        args,
        hdbscan_labels=labels_umap if has_hdbscan else None,
    )

    # ---- Transition entropy outputs (added) ----
    # We compute per-depth entropy for each available labeling mode:
    #   - O2 mode (always)
    #   - HDBSCAN UMAP clusters (if available)
    #   - HDBSCAN feature clusters (if available)
    # and write:
    #   tables/transition_entropy_<mode>.tsv
    #   plots/transition_entropy_<mode>.png
    plots_dir = os.path.join(args.outdir, "plots")
    tables_dir = os.path.join(args.outdir, "tables")

    # Build label columns for entropy computation (match HTML behavior)
    o2_labels = []
    for v in emb_df[args.oxygen_col].to_numpy(dtype=float):
        if np.isfinite(v):
            nm, _ = oxygen_compartment(float(v))
        else:
            nm = "unknown"
        o2_labels.append(nm)
    emb_df["label_o2"] = o2_labels

    entropy_maps: Dict[str, Dict[str, float]] = {}

    ent_o2 = _compute_transition_entropy_per_depth(emb_df, "label_o2")
    if not ent_o2.empty:
        ent_o2.to_csv(os.path.join(tables_dir, "transition_entropy_o2.tsv"), sep="\t", index=False)
        _save_transition_entropy_barplot(
            ent_o2,
            os.path.join(plots_dir, "transition_entropy_o2.png"),
            "Transition entropy per depth — O2 compartments",
        )
        entropy_maps["o2"] = {
            f"{float(r.anchor_depth):g}": float(r.trans_entropy_norm) if np.isfinite(r.trans_entropy_norm) else 0.0
            for r in ent_o2.itertuples(index=False)
        }
    else:
        entropy_maps["o2"] = {}

    if has_hdbscan:
        emb_df["label_hdb_umap"] = emb_df["hdbscan_umap_label"].astype(int).map(lambda x: f"cluster_{x}" if x >= 0 else "noise")
        emb_df["label_hdb_feat"] = emb_df["hdbscan_feat_label"].astype(int).map(lambda x: f"cluster_{x}" if x >= 0 else "noise")

        ent_hu = _compute_transition_entropy_per_depth(emb_df, "label_hdb_umap")
        if not ent_hu.empty:
            ent_hu.to_csv(os.path.join(tables_dir, "transition_entropy_hdb_umap.tsv"), sep="\t", index=False)
            _save_transition_entropy_barplot(
                ent_hu,
                os.path.join(plots_dir, "transition_entropy_hdb_umap.png"),
                "Transition entropy per depth — HDBSCAN clusters (UMAP space)",
            )
            entropy_maps["hdb_umap"] = {
                f"{float(r.anchor_depth):g}": float(r.trans_entropy_norm) if np.isfinite(r.trans_entropy_norm) else 0.0
                for r in ent_hu.itertuples(index=False)
            }
        else:
            entropy_maps["hdb_umap"] = {}

        ent_hf = _compute_transition_entropy_per_depth(emb_df, "label_hdb_feat")
        if not ent_hf.empty:
            ent_hf.to_csv(os.path.join(tables_dir, "transition_entropy_hdb_feat.tsv"), sep="\t", index=False)
            _save_transition_entropy_barplot(
                ent_hf,
                os.path.join(plots_dir, "transition_entropy_hdb_feat.png"),
                "Transition entropy per depth — HDBSCAN clusters (feature space)",
            )
            entropy_maps["hdb_feat"] = {
                f"{float(r.anchor_depth):g}": float(r.trans_entropy_norm) if np.isfinite(r.trans_entropy_norm) else 0.0
                for r in ent_hf.itertuples(index=False)
            }
        else:
            entropy_maps["hdb_feat"] = {}

    # Save embeddings (with optional hdbscan cols) after entropy label columns are added
    emb_df.to_csv(os.path.join(args.outdir, "tables", "umap_embeddings.tsv"), sep="\t", index=False)

    # Keep PNGs for sanity checks (optional)
    plot_embeddings(emb_df, slice_ids, args)

    # Main requested viz: checkbox UI + modes + entropy-weighted link widths
    plot_3d_checkbox_html_with_modes(emb_df, args, has_hdbscan=has_hdbscan, entropy_maps=entropy_maps)

    print("DONE")
    print(f"  Output dir: {args.outdir}")
    print(f"  Slices (cruises): {len(slice_ids)}")
    print(f"  Anchors: {len(anchors)}")
    print("  Key outputs:")
    print("    - tables/agg_time_anchor_depth.tsv")
    print("    - tables/imputed_time_anchor_depth.tsv")
    print("    - tables/umap_embeddings.tsv")
    print("    - plots/umap_<slice>.png")
    print("    - plots/umap_global_overlays.png")
    print("    - plots/umap_3d_trajectories_checkboxes.html")
    print("    - tables/transition_entropy_o2.tsv (+ hdb_* if --hdbscan)")
    print("    - plots/transition_entropy_o2.png (+ hdb_* if --hdbscan)")


if __name__ == "__main__":
    main()
