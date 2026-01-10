# Path: ./env_hybrid_compartment_builder.py
# Purpose: Build SAMPLE-weighted (each bottle equal) hybrid cruise-level composition vectors
# Inputs:
#   1) GMM assignments table (from env_compartments_final_gmm.py): compartments_assignments_{base|smoothed}.csv
#   2) O2 soft-compartment assignments table (from env_compartments_o2_soft.py): o2_compartments_assignments_{base|smoothed}.csv
#
# Outputs:
#   outdir/
#     run_config.json
#     tables/
#       hybrid_per_sample_joined.csv
#       cruise_composition_o2.csv
#       cruise_composition_gmm.csv
#       cruise_composition_hybrid.csv
#       cruise_composition_hybrid_long.csv
#       cruise_composition_hybrid_topN.csv
#       cruise_composition_hybrid_dominant.csv
#       cruise_braycurtis_hybrid.csv
#     plots/
#       cruise_composition_hybrid_stacked.png
#       cruise_composition_hybrid_stacked_topN.png
#       cruise_braycurtis_hybrid.png

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------

@dataclass
class RunConfig:
    gmm_assign_csv: str
    o2_assign_csv: str
    outdir: str
    sep: str

    join_key: str

    cruise_col: str
    time_col: str
    depth_col: str

    gmm_resp_prefix: str
    o2_resp_prefix: str

    # output options
    make_plot: bool
    topn: int
    cmap: str


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(
        description="Build sample-weighted hybrid cruise-level composition vectors from O2 memberships and GMM responsibilities."
    )
    ap.add_argument("--gmm-assign", required=True, help="Path to compartments_assignments_{base|smoothed}.csv from GMM script.")
    ap.add_argument("--o2-assign", required=True, help="Path to o2_compartments_assignments_{base|smoothed}.csv from O2 script.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ',').")

    ap.add_argument("--join-key", default="cruise_year_month_depth",
                    help="Column to join on (default cruise_year_month_depth).")

    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name (default Cruise).")
    ap.add_argument("--time-col", default="date", help="Time column name (default date).")
    ap.add_argument("--depth-col", default="Depth_anchored", help="Depth column name (default Depth_anchored).")

    ap.add_argument("--gmm-resp-prefix", default="resp_", help="Prefix for GMM responsibility columns (default resp_).")
    ap.add_argument("--o2-resp-prefix", default="resp_", help="Prefix for O2 membership columns (default resp_).")

    ap.add_argument("--make-plot", action="store_true", help="Write plots (stacked bars + Bray-Curtis change plot).")
    ap.add_argument("--topn", type=int, default=12,
                    help="For readability: also write a Top-N hybrid stacked bar (default 12).")
    ap.add_argument("--cmap", default="rainbow",
                    help="Colormap for stacked bars (default 'rainbow' ~ red->violet). Good alternatives: 'turbo', 'plasma'.")

    ns = ap.parse_args()

    return RunConfig(
        gmm_assign_csv=ns.gmm_assign,
        o2_assign_csv=ns.o2_assign,
        outdir=ns.outdir,
        sep=ns.sep,
        join_key=ns.join_key,
        cruise_col=ns.cruise_col,
        time_col=ns.time_col,
        depth_col=ns.depth_col,
        gmm_resp_prefix=ns.gmm_resp_prefix,
        o2_resp_prefix=ns.o2_resp_prefix,
        make_plot=bool(ns.make_plot),
        topn=int(ns.topn),
        cmap=str(ns.cmap),
    )


def ensure_dirs(outdir: str) -> Tuple[str, str]:
    tables = os.path.join(outdir, "tables")
    plots = os.path.join(outdir, "plots")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return tables, plots


# -----------------------------
# Helpers
# -----------------------------

def _find_resp_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    def keyfn(x: str) -> int:
        suf = x.replace(prefix, "")
        return int(suf) if suf.isdigit() else 10**9
    return sorted(cols, key=keyfn)


def _safe_row_normalize(M: np.ndarray) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def _cruise_group_cols(cfg: RunConfig, df: pd.DataFrame) -> List[str]:
    cols = []
    if cfg.cruise_col in df.columns:
        cols.append(cfg.cruise_col)
    if cfg.time_col in df.columns:
        cols.append(cfg.time_col)
    return cols


def _coalesce_joined_cols(joined: pd.DataFrame, base: str) -> pd.Series:
    """
    Pick a canonical column after merge. Prefer unsuffixed, else _o2, else _gmm.
    """
    if base in joined.columns:
        return joined[base]
    if f"{base}_o2" in joined.columns:
        return joined[f"{base}_o2"]
    if f"{base}_gmm" in joined.columns:
        return joined[f"{base}_gmm"]
    return pd.Series([np.nan] * joined.shape[0])


def _labels_for_plot(cfg: RunConfig, cruise_hyb: pd.DataFrame) -> pd.Series:
    # prefer cfg names first
    if cfg.cruise_col in cruise_hyb.columns and cfg.time_col in cruise_hyb.columns:
        return cruise_hyb[cfg.cruise_col].astype(str) + " | " + cruise_hyb[cfg.time_col].astype(str)
    if cfg.cruise_col in cruise_hyb.columns:
        return cruise_hyb[cfg.cruise_col].astype(str)
    if cfg.time_col in cruise_hyb.columns:
        return cruise_hyb[cfg.time_col].astype(str)
    return pd.Series([str(i) for i in range(cruise_hyb.shape[0])])


def _make_gradient_colors(n: int, cmap_name: str) -> List[Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]


def _bray_curtis(u: np.ndarray, v: np.ndarray) -> float:
    num = np.sum(np.abs(u - v))
    den = np.sum(u + v)
    return float(num / den) if den > 0 else np.nan


def responsibility_entropy(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    P2 = np.clip(P, eps, 1.0)
    H = -np.sum(P2 * np.log(P2), axis=1)
    K = P.shape[1]
    return H / np.log(K) if K > 1 else np.zeros(P.shape[0], dtype=float)


def write_hybrid_assignments_tables(
    cfg: RunConfig,
    joined: pd.DataFrame,
    hyb_cols: List[str],
    tables_dir: str,
) -> None:
    """
    Write tables that mirror the style of your existing inputs:
      - compartments_assignments_hybrid.csv: metadata + component + max_prob + entropy_norm + resp_*
      - responsibilities_hybrid.csv: resp_* only
    """
    # Build hybrid responsibility matrix and rename to resp_0..resp_{H-1}
    H = len(hyb_cols)
    Hyb = joined[hyb_cols].to_numpy(dtype=float)
    Hyb = _safe_row_normalize(Hyb)

    resp_cols = [f"resp_{i}" for i in range(H)]
    resp_df = pd.DataFrame(Hyb, columns=resp_cols)

    component = np.argmax(Hyb, axis=1).astype(int)
    max_prob = Hyb.max(axis=1)
    entropy = responsibility_entropy(Hyb)

    # component_name (optional but useful)
    # hyb index order is C-major then G: hyb_C{c}_G{g}
    comp_name = [hyb_cols[i] for i in component]

    # Choose a consistent metadata set (match your input vibe)
    meta_keep = []
    for c in [
        cfg.join_key, cfg.cruise_col, "Year", "Month", "Day",
        "Depth", cfg.time_col, "Season", cfg.depth_col
    ]:
        if c in joined.columns and c not in meta_keep:
            meta_keep.append(c)

    meta = joined[meta_keep].copy()

    out = pd.concat([meta.reset_index(drop=True), resp_df.reset_index(drop=True)], axis=1)
    out.insert(len(meta.columns), "component", component)
    out.insert(len(meta.columns) + 1, "compartment_name", comp_name)
    out.insert(len(meta.columns) + 2, "max_prob", max_prob)
    out.insert(len(meta.columns) + 3, "entropy_norm", entropy)

    out.to_csv(os.path.join(tables_dir, "compartments_assignments_hybrid.csv"), index=False)
    resp_df.to_csv(os.path.join(tables_dir, "responsibilities_hybrid.csv"), index=False)

# -----------------------------
# Core logic
# -----------------------------

def build_hybrid_per_sample(
    cfg: RunConfig,
    gmm: pd.DataFrame,
    o2: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    if cfg.join_key not in gmm.columns:
        raise ValueError(f"GMM table missing join key column: {cfg.join_key}")
    if cfg.join_key not in o2.columns:
        raise ValueError(f"O2 table missing join key column: {cfg.join_key}")

    gmm_resp_cols = _find_resp_cols(gmm, cfg.gmm_resp_prefix)
    o2_resp_cols = _find_resp_cols(o2, cfg.o2_resp_prefix)

    if len(gmm_resp_cols) == 0:
        raise ValueError(f"No GMM responsibility columns found with prefix '{cfg.gmm_resp_prefix}'")
    if len(o2_resp_cols) == 0:
        raise ValueError(f"No O2 membership columns found with prefix '{cfg.o2_resp_prefix}'")
    if len(o2_resp_cols) != 4:
        raise ValueError(f"Expected 4 O2 membership columns, found {len(o2_resp_cols)}: {o2_resp_cols}")

    gmm2 = gmm.copy()
    o22 = o2.copy()

    gmm_map = {c: f"gmm_{c}" for c in gmm_resp_cols}
    o2_map = {c: f"o2_{c}" for c in o2_resp_cols}

    gmm2 = gmm2.rename(columns=gmm_map)
    o22 = o22.rename(columns=o2_map)

    gmm_resp_cols_mapped = [gmm_map[c] for c in gmm_resp_cols]
    o2_resp_cols_mapped = [o2_map[c] for c in o2_resp_cols]

    joined = pd.merge(
        o22,
        gmm2,
        on=cfg.join_key,
        how="inner",
        suffixes=("_o2", "_gmm"),
        validate="one_to_one",
    )

    # Canonicalize grouping metadata
    for base in [cfg.cruise_col, cfg.time_col, cfg.depth_col]:
        if base not in joined.columns:
            joined[base] = _coalesce_joined_cols(joined, base)

    if cfg.time_col in joined.columns:
        joined[cfg.time_col] = pd.to_datetime(joined[cfg.time_col], errors="coerce")

    # hybrid per-sample: outer product of (O2 4) x (GMM K)
    O = _safe_row_normalize(joined[o2_resp_cols_mapped].to_numpy(dtype=float))
    G = _safe_row_normalize(joined[gmm_resp_cols_mapped].to_numpy(dtype=float))

    n = O.shape[0]
    C = O.shape[1]   # 4
    K = G.shape[1]

    hyb_cols = [f"hyb_C{c}_G{g}" for c in range(C) for g in range(K)]

    hyb = np.zeros((n, C * K), dtype=float)
    col = 0
    for c in range(C):
        for g in range(K):
            hyb[:, col] = O[:, c] * G[:, g]
            col += 1

    hyb = _safe_row_normalize(hyb)

    for j, name in enumerate(hyb_cols):
        joined[name] = hyb[:, j]

    joined["hyb_component"] = np.argmax(hyb, axis=1).astype(int)
    joined["hyb_max_prob"] = hyb.max(axis=1)

    return joined, o2_resp_cols_mapped, gmm_resp_cols_mapped, hyb_cols


def cruise_composition_sample_weighted(
    cfg: RunConfig,
    joined: pd.DataFrame,
    cols: List[str],
    out_csv: str,
) -> pd.DataFrame:
    group_cols = _cruise_group_cols(cfg, joined)
    if len(group_cols) == 0:
        raise ValueError(
            f"No cruise/time columns found for grouping. I looked for: "
            f"{cfg.cruise_col!r} and/or {cfg.time_col!r}. "
            f"Available columns include: {list(joined.columns)[:30]} ..."
        )

    out = joined.copy()
    if cfg.time_col in out.columns:
        out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")

    agg = out.groupby(group_cols, dropna=False)[cols].mean().reset_index()
    agg.to_csv(out_csv, index=False)
    return agg


def to_long(df: pd.DataFrame, id_cols: List[str], value_cols: List[str],
            var_name: str, val_name: str) -> pd.DataFrame:
    return df.melt(id_vars=id_cols, value_vars=value_cols, var_name=var_name, value_name=val_name)


def cruise_dominant_state(
    cfg: RunConfig,
    cruise_hyb: pd.DataFrame,
    hyb_cols: List[str],
    out_csv: str,
) -> pd.DataFrame:
    out = cruise_hyb.copy()
    X = _safe_row_normalize(out[hyb_cols].to_numpy(dtype=float))
    out["dominant_hybrid_cell"] = [hyb_cols[i] for i in np.argmax(X, axis=1)]
    out["dominant_fraction"] = np.max(X, axis=1)
    out.to_csv(out_csv, index=False)
    return out


def cruise_topN_table(
    cfg: RunConfig,
    cruise_hyb: pd.DataFrame,
    hyb_cols: List[str],
    topn: int,
    out_csv: str,
) -> pd.DataFrame:
    """
    For each cruise: keep only top-N hybrid cells; remainder pooled into 'other'.
    """
    out = cruise_hyb.copy()
    X = _safe_row_normalize(out[hyb_cols].to_numpy(dtype=float))

    n = X.shape[0]
    keep = min(int(topn), X.shape[1])
    top_idx = np.argsort(-X, axis=1)[:, :keep]

    # Build new matrix with keep cols + other
    new_cols = [f"top_{i+1}" for i in range(keep)] + ["other"]
    Y = np.zeros((n, keep + 1), dtype=float)

    labels = []
    for i in range(n):
        idx = top_idx[i]
        Y[i, :keep] = X[i, idx]
        Y[i, keep] = 1.0 - Y[i, :keep].sum()
        labels.append([hyb_cols[j] for j in idx])

    meta_cols = _cruise_group_cols(cfg, out)
    meta = out[meta_cols].copy()
    for j in range(keep):
        meta[f"top_{j+1}_cell"] = [lab[j] for lab in labels]
        meta[f"top_{j+1}_fraction"] = Y[:, j]
    meta["other_fraction"] = Y[:, keep]
    meta.to_csv(out_csv, index=False)
    return meta


# -----------------------------
# Plots
# -----------------------------

def plot_hybrid_stacked_gradient(
    cfg: RunConfig,
    cruise_hyb: pd.DataFrame,
    hyb_cols: List[str],
    outpath: str,
) -> None:
    if cruise_hyb.shape[0] == 0:
        return

    labels = _labels_for_plot(cfg, cruise_hyb)
    X = _safe_row_normalize(cruise_hyb[hyb_cols].to_numpy(dtype=float))

    # Sort columns by (C then G) already encoded in hyb_C{c}_G{g}
    # (they are created in that order), but enforce it anyway:
    hyb_cols_sorted = list(hyb_cols)
    X = _safe_row_normalize(cruise_hyb[hyb_cols_sorted].to_numpy(dtype=float))

    colors = _make_gradient_colors(len(hyb_cols_sorted), cfg.cmap)

    plt.figure(figsize=(max(10, 0.35 * X.shape[0]), 5))
    bottom = np.zeros(X.shape[0], dtype=float)
    x = np.arange(X.shape[0])

    for j, col in enumerate(hyb_cols_sorted):
        plt.bar(x, X[:, j], bottom=bottom, label=col, linewidth=0, color=colors[j])
        bottom += X[:, j]

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("composition (sample-weighted)")
    plt.title(f"Hybrid cruise composition (stacked; colormap={cfg.cmap})")
    # legend is usually unreadable for 4*K; keep it off by default
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_hybrid_stacked_topN(
    cfg: RunConfig,
    cruise_hyb: pd.DataFrame,
    hyb_cols: List[str],
    topn: int,
    outpath: str,
) -> None:
    """
    Much more readable stacked bar:
      For each cruise, plot top-N hybrid cells + other.
    Colors follow the same red->violet gradient for the ranked slots (top_1..top_N).
    """
    if cruise_hyb.shape[0] == 0:
        return

    labels = _labels_for_plot(cfg, cruise_hyb)
    X = _safe_row_normalize(cruise_hyb[hyb_cols].to_numpy(dtype=float))

    keep = min(int(topn), X.shape[1])
    top_idx = np.argsort(-X, axis=1)[:, :keep]

    Y = np.zeros((X.shape[0], keep + 1), dtype=float)
    for i in range(X.shape[0]):
        idx = top_idx[i]
        Y[i, :keep] = X[i, idx]
        Y[i, keep] = 1.0 - Y[i, :keep].sum()

    colors = _make_gradient_colors(keep, cfg.cmap)
    other_color = (0.8, 0.8, 0.8, 1.0)

    plt.figure(figsize=(max(10, 0.35 * X.shape[0]), 5))
    bottom = np.zeros(Y.shape[0], dtype=float)
    x = np.arange(Y.shape[0])

    for j in range(keep):
        plt.bar(x, Y[:, j], bottom=bottom, linewidth=0, color=colors[j], label=f"top_{j+1}")
        bottom += Y[:, j]
    plt.bar(x, Y[:, keep], bottom=bottom, linewidth=0, color=other_color, label="other")

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("composition (sample-weighted)")
    plt.title(f"Hybrid cruise composition (Top-{keep} + other; colormap={cfg.cmap})")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bray_curtis_changes(
    cfg: RunConfig,
    cruise_hyb: pd.DataFrame,
    hyb_cols: List[str],
    out_csv: str,
    out_png: str,
) -> pd.DataFrame:
    """
    Better “regime shift” plot than stacked bars:
    compute Bray–Curtis dissimilarity between consecutive cruises.
    High spikes = big composition change (candidate transitions).
    """
    out = cruise_hyb.copy()
    if cfg.time_col in out.columns:
        out = out.sort_values(cfg.time_col)
    else:
        out = out.sort_values(cfg.cruise_col) if cfg.cruise_col in out.columns else out

    X = _safe_row_normalize(out[hyb_cols].to_numpy(dtype=float))
    n = X.shape[0]
    bc = np.full(n, np.nan, dtype=float)
    for i in range(1, n):
        bc[i] = _bray_curtis(X[i - 1, :], X[i, :])

    # table
    tbl = out[_cruise_group_cols(cfg, out)].copy()
    tbl["bray_curtis_vs_prev"] = bc
    tbl.to_csv(out_csv, index=False)

    # plot
    labels = _labels_for_plot(cfg, out)
    x = np.arange(n)

    plt.figure(figsize=(max(10, 0.35 * n), 3.8))
    plt.plot(x, bc, marker="o")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Bray–Curtis vs previous cruise")
    plt.title("Hybrid regime-change signal (spikes = big shift)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return tbl


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    gmm = pd.read_csv(cfg.gmm_assign_csv, sep=cfg.sep)
    o2 = pd.read_csv(cfg.o2_assign_csv, sep=cfg.sep)

    joined, o2_resp_cols, gmm_resp_cols, hyb_cols = build_hybrid_per_sample(cfg, gmm=gmm, o2=o2)

    # per-sample joined table
    joined_out = os.path.join(tables_dir, "hybrid_per_sample_joined.csv")
    joined.to_csv(joined_out, index=False)

    # cruise-level compositions (sample-weighted)
    o2_out = os.path.join(tables_dir, "cruise_composition_o2.csv")
    gmm_out = os.path.join(tables_dir, "cruise_composition_gmm.csv")
    hyb_out = os.path.join(tables_dir, "cruise_composition_hybrid.csv")

    cruise_o2 = cruise_composition_sample_weighted(cfg, joined, o2_resp_cols, o2_out)
    cruise_gmm = cruise_composition_sample_weighted(cfg, joined, gmm_resp_cols, gmm_out)
    cruise_hyb = cruise_composition_sample_weighted(cfg, joined, hyb_cols, hyb_out)

    # long format hybrid
    id_cols = _cruise_group_cols(cfg, joined)
    hyb_long = to_long(cruise_hyb, id_cols=id_cols, value_cols=hyb_cols, var_name="hybrid_cell", val_name="fraction")
    hyb_long.to_csv(os.path.join(tables_dir, "cruise_composition_hybrid_long.csv"), index=False)

    # dominant state per cruise
    dom = cruise_dominant_state(cfg, cruise_hyb, hyb_cols, os.path.join(tables_dir, "cruise_composition_hybrid_dominant.csv"))

    # topN table (for quick inspection / labeling)
    topn_tbl = cruise_topN_table(cfg, cruise_hyb, hyb_cols, cfg.topn, os.path.join(tables_dir, "cruise_composition_hybrid_topN.csv"))

    write_hybrid_assignments_tables(
        cfg=cfg,
        joined=joined,
        hyb_cols=hyb_cols,
        tables_dir=tables_dir,
    )

    # plots
    if cfg.make_plot:
        plot_hybrid_stacked_gradient(
            cfg=cfg,
            cruise_hyb=cruise_hyb,
            hyb_cols=hyb_cols,
            outpath=os.path.join(plots_dir, "cruise_composition_hybrid_stacked.png"),
        )

        plot_hybrid_stacked_topN(
            cfg=cfg,
            cruise_hyb=cruise_hyb,
            hyb_cols=hyb_cols,
            topn=cfg.topn,
            outpath=os.path.join(plots_dir, "cruise_composition_hybrid_stacked_topN.png"),
        )

        plot_bray_curtis_changes(
            cfg=cfg,
            cruise_hyb=cruise_hyb,
            hyb_cols=hyb_cols,
            out_csv=os.path.join(tables_dir, "cruise_braycurtis_hybrid.csv"),
            out_png=os.path.join(plots_dir, "cruise_braycurtis_hybrid.png"),
        )

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Per-sample join: {joined_out}")
    print(f"     Cruise O2 comp : {o2_out}")
    print(f"     Cruise GMM comp: {gmm_out}")
    print(f"     Cruise HYB comp: {hyb_out}")
    print(f"     Hybrid cells   : {len(hyb_cols)} (= 4 * {len(gmm_resp_cols)})")
    if cfg.make_plot:
        print(f"     Plot (all cells): {os.path.join(plots_dir, 'cruise_composition_hybrid_stacked.png')}")
        print(f"     Plot (TopN)     : {os.path.join(plots_dir, 'cruise_composition_hybrid_stacked_topN.png')}")
        print(f"     Plot (BC)       : {os.path.join(plots_dir, 'cruise_braycurtis_hybrid.png')}")
        print(f"     Table (BC)      : {os.path.join(tables_dir, 'cruise_braycurtis_hybrid.csv')}")


if __name__ == "__main__":
    main()
