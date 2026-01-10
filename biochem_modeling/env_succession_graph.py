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
#     plots/
#       succession_heatmap_prob_<name>.png
#       succession_network_topN_<name>.png
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
    network_max_edges: int          # cap edges drawn in network plot (after filtering)


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
                    help="Write heatmap + simple network plot (default off).")
    ap.add_argument("--cmap", default="turbo", help="Matplotlib colormap (default turbo).")
    ap.add_argument("--network-max-edges", type=int, default=200,
                    help="Cap number of edges drawn in network plot (default 200).")

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
        network_max_edges=int(ns.network_max_edges),
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

def plot_heatmap(T: np.ndarray, title: str, outpath: str, cmap: str) -> None:
    plt.figure(figsize=(6.4, 5.4))
    im = plt.imshow(T, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="P(next | current)")
    plt.xlabel("to_state")
    plt.ylabel("from_state")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_network_simple(
    edges: pd.DataFrame,
    K: int,
    title: str,
    outpath: str,
    max_edges: int = 200,
) -> None:
    """
    Minimal network plot with a circular layout (no external deps).
    - Node positions on a circle
    - Edge width scales with probability
    - Only draws up to max_edges edges (highest prob first)
    """
    if edges.shape[0] == 0:
        return

    # take top edges globally by prob (already filtered, but hybrid can still be large)
    e = edges.sort_values("p_next_given_current", ascending=False).head(int(max_edges)).copy()

    # circular layout
    theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
    xs = np.cos(theta)
    ys = np.sin(theta)

    plt.figure(figsize=(7.2, 7.2))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # nodes
    ax.scatter(xs, ys, s=120, alpha=0.9)
    for i in range(K):
        ax.text(xs[i] * 1.07, ys[i] * 1.07, str(i), ha="center", va="center", fontsize=10)

    # edges
    # scale widths
    w = e["p_next_given_current"].to_numpy(dtype=float)
    w = np.clip(w, 0.0, 1.0)
    widths = 0.6 + 6.0 * w

    for (idx, r), lw in zip(e.iterrows(), widths):
        i = int(r["from_state"])
        j = int(r["to_state"])
        if i == j:
            # self-loop as small arc
            x, y = xs[i], ys[i]
            ax.annotate(
                "",
                xy=(x + 0.10, y + 0.10),
                xytext=(x + 0.02, y + 0.18),
                arrowprops=dict(arrowstyle="->", linewidth=lw * 0.35, alpha=0.6),
            )
        else:
            ax.annotate(
                "",
                xy=(xs[j], ys[j]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", linewidth=lw * 0.25, alpha=0.55),
            )

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


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

    # plots (optional)
    if cfg.make_plots:
        plot_heatmap(
            T,
            title=f"{name.upper()} soft succession (P[next|current])",
            outpath=os.path.join(plots_dir, f"succession_heatmap_prob_{name}.png"),
            cmap=cfg.cmap,
        )
        plot_network_simple(
            edges=edges,
            K=T.shape[0],
            title=f"{name.upper()} succession graph (top edges)",
            outpath=os.path.join(plots_dir, f"succession_network_topN_{name}.png"),
            max_edges=cfg.network_max_edges,
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
