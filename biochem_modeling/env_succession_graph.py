# Path: ./env_succession_graph.py
# Purpose:
#   Build soft (responsibility-weighted) state succession graphs from cruise-level
#   composition vectors (O2 / GMM / Hybrid), producing edge-list tables and simple plots.
#
# Inputs (CSV):
#   - cruise_composition_o2.csv
#   - cruise_composition_gmm.csv
#   - cruise_composition_hybrid.csv
#
# Outputs:
#   outdir/
#     run_config.json
#     tables/
#       succession_matrix_mass_<name>.csv
#       succession_matrix_prob_<name>.csv
#       succession_edges_soft_<name>.csv
#       succession_top_successors_soft_<name>.csv
#       succession_transition_uncertainty_<name>.csv
#     plots/
#       succession_top_successor_bars_<name>.png
#       succession_composition_timeseries_<name>.png
#       succession_transition_uncertainty_<name>.png
#
# Notes:
#   - Soft transitions use: M_ij = sum_t P_t(i) * P_{t+1}(j)
#     Then row-normalize to get conditional probs T_ij = P(next=j | current=i).
#   - Filtering (to keep graphs readable):
#       Always keep self-loop i->i
#       Keep top-N outgoing edges per state (excluding self), and optionally min-prob threshold.
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
    outdir: str
    sep: str

    # inputs (optional; you can pass any subset)
    o2_csv: Optional[str]
    gmm_csv: Optional[str]
    hybrid_csv: Optional[str]

    # identifiers / sorting
    id_cols: List[str]          # columns to keep as identifiers (e.g., Cruise,date)
    cruise_col: str
    time_col: str
    sort_by_time: bool

    # state column detection
    state_prefix: Optional[str]     # if set, only columns starting with this prefix are treated as states
    drop_prefixes: List[str]        # columns with these prefixes are excluded from states

    # edge filtering
    top_n: int                      # keep top N outgoing edges per node (excluding self)
    min_prob: float                 # additionally require prob >= min_prob (excluding self)
    keep_self: bool                 # always keep i->i

    # plotting
    cmap: str
    make_plots: bool
    timeseries_top_k: int           # keep top-K mean-abundance states in stackplot; remainder grouped as other_states


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Build soft succession graphs from cruise-level composition vectors (O2/GMM/Hybrid)."
    )
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default ',').")

    ap.add_argument("--o2", default=None, help="Path to cruise_composition_o2.csv (optional).")
    ap.add_argument("--gmm", default=None, help="Path to cruise_composition_gmm.csv (optional).")
    ap.add_argument("--hybrid", default=None, help="Path to cruise_composition_hybrid.csv (optional).")

    ap.add_argument("--id-cols", default="Cruise,date",
                    help="Comma-separated identifier columns to keep (default 'Cruise,date').")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name (default Cruise).")
    ap.add_argument("--time-col", default="date", help="Time column name (default date).")
    ap.add_argument("--sort-by-time", action="store_true", default=True,
                    help="Sort cruises by time_col if present (default True).")

    ap.add_argument("--state-prefix", default=None,
                    help="If set, only columns starting with this prefix are treated as states (e.g., 'hyb_').")
    ap.add_argument("--drop-prefixes", default="top_,dominant_,other_",
                    help="Comma-separated prefixes to exclude from state columns (default 'top_,dominant_,other_').")

    ap.add_argument("--top-n", type=int, default=2,
                    help="Keep top N outgoing edges per state (excluding self) (default 2).")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="Additionally require prob >= this for kept edges (excluding self) (default 0.0).")
    ap.add_argument("--no-keep-self", action="store_true",
                    help="If set, do NOT force-keep self-loops (default keep self-loops).")

    ap.add_argument("--make-plots", action="store_true",
                    help="Write successor bars + composition timeseries + transition uncertainty plots (default off).")
    ap.add_argument("--cmap", default="turbo", help="Matplotlib colormap (default turbo).")
    ap.add_argument(
        "--timeseries-top-k",
        type=int,
        default=10,
        help="In composition timeseries, keep top-K states by mean abundance and sum the rest as other_states (default 10).",
    )

    ns = ap.parse_args()

    id_cols = [c.strip() for c in ns.id_cols.split(",") if c.strip()]
    drop_prefixes = [c.strip() for c in ns.drop_prefixes.split(",") if c.strip()]

    return Config(
        outdir=ns.outdir,
        sep=ns.sep,
        o2_csv=ns.o2,
        gmm_csv=ns.gmm,
        hybrid_csv=ns.hybrid,
        id_cols=id_cols,
        cruise_col=ns.cruise_col,
        time_col=ns.time_col,
        sort_by_time=bool(ns.sort_by_time),
        state_prefix=ns.state_prefix,
        drop_prefixes=drop_prefixes,
        top_n=int(ns.top_n),
        min_prob=float(ns.min_prob),
        keep_self=(not bool(ns.no_keep_self)),
        cmap=ns.cmap,
        make_plots=bool(ns.make_plots),
        timeseries_top_k=int(ns.timeseries_top_k),
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

def _safe_row_normalize(M: np.ndarray) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def _infer_state_cols(cfg: Config, df: pd.DataFrame) -> List[str]:
    # start with candidates that are not id cols
    cand = [c for c in df.columns if c not in cfg.id_cols]

    # optionally restrict by prefix
    if cfg.state_prefix:
        cand = [c for c in cand if c.startswith(cfg.state_prefix)]

    # drop known non-state prefixes
    for pref in cfg.drop_prefixes:
        cand = [c for c in cand if not c.startswith(pref)]

    # numeric-only
    out = []
    for c in cand:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            out.append(c)

    if not out:
        raise ValueError(
            "Could not infer any state columns. "
            "Try setting --state-prefix (e.g., 'hyb_') or adjust --id-cols / --drop-prefixes."
        )

    return out


def _prepare(cfg: Config, df: pd.DataFrame, state_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    out = df.copy()

    # parse time if present
    if cfg.time_col in out.columns:
        out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce")

    # sort
    if cfg.sort_by_time and cfg.time_col in out.columns:
        out = out.sort_values(cfg.time_col)
    elif cfg.cruise_col in out.columns:
        out = out.sort_values(cfg.cruise_col)

    # coerce / fill state cols
    for c in state_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    P = out[state_cols].to_numpy(dtype=float)
    P = _safe_row_normalize(P)
    out[state_cols] = P
    out = out.reset_index(drop=True)
    return out, P


def soft_transition_mass(P: np.ndarray) -> np.ndarray:
    """
    M_ij = sum_t P[t,i] * P[t+1,j]
    """
    K = P.shape[1]
    M = np.zeros((K, K), dtype=float)
    for t in range(P.shape[0] - 1):
        M += np.outer(P[t, :], P[t + 1, :])
    return M


def soft_transition_prob(M: np.ndarray) -> np.ndarray:
    """
    Row-normalize mass matrix into conditional transition probabilities.
    """
    return _safe_row_normalize(M)


def build_edge_list(
    cfg: Config,
    state_names: List[str],
    M: np.ndarray,
    T: np.ndarray,
) -> pd.DataFrame:
    """
    Build edge list with filtering:
      - Always keep self loops (if cfg.keep_self)
      - Keep top-N outgoing edges per node (excluding self)
      - Apply min_prob to non-self edges
    """
    K = T.shape[0]
    rows = []

    # precompute ranks per row
    for i in range(K):
        probs = T[i, :].copy()
        # rank outgoing (excluding self)
        order = np.argsort(-probs)  # descending
        # build set of kept j's
        kept = set()

        # keep self
        if cfg.keep_self:
            kept.add(i)

        # add top-N excluding self, respecting min_prob
        n_added = 0
        for j in order:
            if j == i:
                continue
            if probs[j] < cfg.min_prob:
                continue
            kept.add(int(j))
            n_added += 1
            if n_added >= max(0, cfg.top_n):
                break

        # write rows for kept edges
        # also compute rank_outgoing among non-self edges
        # rank = 1 is the highest-prob non-self successor
        nonself_order = [int(j) for j in order if int(j) != i]
        rank_map = {j: (r + 1) for r, j in enumerate(nonself_order)}

        for j in sorted(list(kept)):
            rows.append({
                "from_state": int(i),
                "to_state": int(j),
                "from_name": state_names[i],
                "to_name": state_names[j],
                "p_next_given_current": float(T[i, j]),
                "mass": float(M[i, j]),
                "is_self": bool(i == j),
                "rank_outgoing": int(rank_map.get(int(j), 0)) if i != j else 0,
            })

    edges = pd.DataFrame(rows)
    # nice sort: by from_state then prob desc
    edges = edges.sort_values(["from_state", "is_self", "p_next_given_current"], ascending=[True, False, False])
    return edges.reset_index(drop=True)


def top_successors_table(edges: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Compact per-state summary: self-loop + top_n successors (already filtered in edges, but keep compact).
    """
    rows = []
    for i, sub in edges.groupby("from_state", sort=True):
        sub = sub.sort_values(["is_self", "p_next_given_current"], ascending=[False, False])

        # force include self if present
        self_row = sub[sub["is_self"]].head(1)
        other = sub[~sub["is_self"]].head(max(0, int(top_n)))

        use = pd.concat([self_row, other], axis=0)

        # pack into a wide-ish format that reads well
        # self
        self_p = float(use.loc[use["is_self"], "p_next_given_current"].iloc[0]) if (use["is_self"].any()) else np.nan

        # successors
        succs = use[~use["is_self"]]
        succ_list = []
        for _, r in succs.iterrows():
            succ_list.append(f"{int(r['to_state'])}:{float(r['p_next_given_current']):.3f}")

        rows.append({
            "from_state": int(i),
            "from_name": str(use["from_name"].iloc[0]) if use.shape[0] else str(i),
            "p_self": self_p,
            "top_successors": "; ".join(succ_list),
        })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting (optional)
# -----------------------------

def _select_successors_for_plot(
    row_probs: np.ndarray,
    i: int,
    top_n: int,
    min_prob: float,
) -> List[int]:
    keep: List[int] = [int(i)]
    order = np.argsort(-row_probs)
    added = 0
    for j in order:
        jj = int(j)
        if jj == int(i):
            continue
        if float(row_probs[jj]) < float(min_prob):
            continue
        keep.append(jj)
        added += 1
        if added >= max(0, int(top_n)):
            break
    return keep


def _short_label(s: str, max_len: int = 24) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def plot_top_successor_bars(
    T: np.ndarray,
    state_names: List[str],
    cfg: Config,
    title: str,
    outpath: str,
) -> None:
    K = int(T.shape[0])
    if K == 0:
        return

    ncols = 4 if K > 12 else 3
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 2.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i in range(K):
        ax = axes_flat[i]
        keep = _select_successors_for_plot(T[i, :], i=i, top_n=cfg.top_n, min_prob=cfg.min_prob)
        labels = [_short_label(state_names[j]) for j in keep]
        vals = [float(T[i, j]) for j in keep]
        y = np.arange(len(keep))
        colors = ["#2a9d8f" if j == i else "#577590" for j in keep]

        ax.barh(y, vals, color=colors, alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.set_title(f"from {i}: {_short_label(state_names[i], max_len=18)}", fontsize=9)

    for j in range(K, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.supxlabel("P(next state | current state)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_composition_timeseries(
    df2: pd.DataFrame,
    state_cols: List[str],
    cfg: Config,
    title: str,
    outpath: str,
) -> None:
    if len(state_cols) == 0 or df2.shape[0] == 0:
        return

    # Reduce over-dense legends: keep top-K states by mean abundance, pool remainder.
    mean_abund = df2[state_cols].mean(axis=0).sort_values(ascending=False)
    if cfg.timeseries_top_k > 0 and len(state_cols) > cfg.timeseries_top_k:
        keep = list(mean_abund.index[: cfg.timeseries_top_k])
        plot_df = df2[keep].copy()
        remainder = (1.0 - plot_df.sum(axis=1)).clip(lower=0.0)
        plot_df["other_states"] = remainder
    else:
        plot_df = df2[state_cols].copy()

    if cfg.time_col in df2.columns and pd.to_datetime(df2[cfg.time_col], errors="coerce").notna().sum() >= 2:
        x = pd.to_datetime(df2[cfg.time_col], errors="coerce")
        x_label = cfg.time_col
    else:
        x = np.arange(df2.shape[0], dtype=int)
        x_label = "ordered_cruise_index"

    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    cmap = plt.get_cmap(cfg.cmap)
    colors = cmap(np.linspace(0.08, 0.92, plot_df.shape[1]))
    layers = [plot_df[c].to_numpy(dtype=float) for c in plot_df.columns]
    ax.stackplot(x, layers, labels=list(plot_df.columns), colors=colors, alpha=0.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cruise composition fraction")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def transition_uncertainty_table(
    T: np.ndarray,
    state_names: List[str],
) -> pd.DataFrame:
    K = int(T.shape[0])
    if K == 0:
        return pd.DataFrame(
            columns=[
                "from_state",
                "from_name",
                "p_self",
                "entropy",
                "entropy_norm",
                "effective_next_states",
                "dominant_next_state",
                "dominant_next_name",
                "p_dominant_next",
            ]
        )

    eps = 1e-12
    T_clip = np.clip(T, eps, 1.0)
    H = -np.sum(T * np.log(T_clip), axis=1)
    H_norm = H / np.log(float(K)) if K > 1 else np.zeros(K, dtype=float)
    eff = np.exp(H)
    dom_idx = np.argmax(T, axis=1)
    dom_p = T[np.arange(K), dom_idx]
    p_self = np.diag(T)

    return pd.DataFrame(
        {
            "from_state": np.arange(K, dtype=int),
            "from_name": state_names,
            "p_self": p_self.astype(float),
            "entropy": H.astype(float),
            "entropy_norm": H_norm.astype(float),
            "effective_next_states": eff.astype(float),
            "dominant_next_state": dom_idx.astype(int),
            "dominant_next_name": [state_names[int(j)] for j in dom_idx],
            "p_dominant_next": dom_p.astype(float),
        }
    )


def plot_transition_uncertainty(
    summary: pd.DataFrame,
    title: str,
    outpath: str,
) -> None:
    if summary.shape[0] == 0:
        return

    s = summary.sort_values("entropy_norm", ascending=False).reset_index(drop=True)
    y = np.arange(s.shape[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, max(4.0, 0.42 * s.shape[0])),
        sharey=True,
    )

    axes[0].barh(y, s["entropy_norm"].to_numpy(dtype=float), color="#577590", alpha=0.9)
    axes[0].set_xlabel("Outgoing transition entropy (normalized)")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].grid(axis="x", alpha=0.25, linewidth=0.6)

    axes[1].barh(y, s["p_self"].to_numpy(dtype=float), color="#2a9d8f", alpha=0.9)
    axes[1].set_xlabel("Self-transition probability")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].grid(axis="x", alpha=0.25, linewidth=0.6)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_short_label(v, max_len=28) for v in s["from_name"].astype(str)], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("State")
    axes[0].set_title("Uncertainty")
    axes[1].set_title("Persistence")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# -----------------------------
# One dataset runner
# -----------------------------

def run_one(
    cfg: Config,
    name: str,
    df: pd.DataFrame,
    tables_dir: str,
    plots_dir: str,
) -> None:
    state_cols = _infer_state_cols(cfg, df)
    df2, P = _prepare(cfg, df, state_cols)

    M = soft_transition_mass(P)
    T = soft_transition_prob(M)

    # state names = column names (keeps mapping stable)
    state_names = state_cols

    # write matrices
    pd.DataFrame(M, columns=state_names, index=state_names).to_csv(
        os.path.join(tables_dir, f"succession_matrix_mass_{name}.csv")
    )
    pd.DataFrame(T, columns=state_names, index=state_names).to_csv(
        os.path.join(tables_dir, f"succession_matrix_prob_{name}.csv")
    )

    # edges + top successors
    edges = build_edge_list(cfg, state_names=state_names, M=M, T=T)
    edges.to_csv(os.path.join(tables_dir, f"succession_edges_soft_{name}.csv"), index=False)

    top = top_successors_table(edges, top_n=cfg.top_n)
    top.to_csv(os.path.join(tables_dir, f"succession_top_successors_soft_{name}.csv"), index=False)

    uncertainty = transition_uncertainty_table(T=T, state_names=state_names)
    uncertainty.to_csv(os.path.join(tables_dir, f"succession_transition_uncertainty_{name}.csv"), index=False)

    # plots (optional)
    if cfg.make_plots:
        plot_top_successor_bars(
            T=T,
            state_names=state_names,
            cfg=cfg,
            title=f"{name.upper()} top successors by source state",
            outpath=os.path.join(plots_dir, f"succession_top_successor_bars_{name}.png"),
        )
        plot_composition_timeseries(
            df2=df2,
            state_cols=state_cols,
            cfg=cfg,
            title=f"{name.upper()} cruise composition timeseries",
            outpath=os.path.join(plots_dir, f"succession_composition_timeseries_{name}.png"),
        )
        plot_transition_uncertainty(
            summary=uncertainty,
            title=f"{name.upper()} transition uncertainty and persistence",
            outpath=os.path.join(plots_dir, f"succession_transition_uncertainty_{name}.png"),
        )

    print(f"[OK] {name}: K={P.shape[1]} states, cruises={P.shape[0]}")
    print(f"     state cols: {len(state_cols)}")
    print(f"     edges kept: {edges.shape[0]} (top_n={cfg.top_n}, min_prob={cfg.min_prob}, keep_self={cfg.keep_self})")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    cfg = parse_args()
    tables_dir, plots_dir = ensure_dirs(cfg.outdir)

    with open(os.path.join(cfg.outdir, "run_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    any_input = False

    if cfg.o2_csv:
        any_input = True
        o2 = pd.read_csv(cfg.o2_csv, sep=cfg.sep)
        run_one(cfg, "o2", o2, tables_dir, plots_dir)

    if cfg.gmm_csv:
        any_input = True
        gmm = pd.read_csv(cfg.gmm_csv, sep=cfg.sep)
        run_one(cfg, "gmm", gmm, tables_dir, plots_dir)

    if cfg.hybrid_csv:
        any_input = True
        hyb = pd.read_csv(cfg.hybrid_csv, sep=cfg.sep)
        run_one(cfg, "hybrid", hyb, tables_dir, plots_dir)

    if not any_input:
        raise ValueError("No inputs provided. Pass at least one of: --o2, --gmm, --hybrid")

    print(f"[OK] Wrote outputs to: {cfg.outdir}")
    print(f"     Tables: {tables_dir}")
    if cfg.make_plots:
        print(f"     Plots : {plots_dir}")


if __name__ == "__main__":
    main()
