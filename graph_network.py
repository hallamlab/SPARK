#!/usr/bin/env python3
"""
Network visualization CLI for SPIEC-EASI GraphML outputs + ISA/metadata overlays.

Features
- Robust CLI (argparse)
- All paths & style values are configurable
- Safe I/O + schema checks
- Reusable spring-layout (cache to JSON)
- Multiple plot modes (combined degree+edgeweight, abundance, type ISA, status ISA, venn, phylum×{abundance,ISA}, labeled variants)
- Consistent aesthetics with your rcParams
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Tuple, Optional, List, Set, Union

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import math
from collections import Counter

# Optional: labels w/ collision-avoidance
try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except Exception:
    _HAS_ADJUSTTEXT = False

# ---------------------------- Global style -----------------------------------
mpl.rcParams['pdf.fonttype'] = 42                 # text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'             # text as text in SVG
plt.rcParams.update({'font.size': 12})
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")


# ---------------------------- Helpers ----------------------------------------
def die(msg: str, code: int = 2):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def ok(msg: str):
    print(f"[+] {msg}")


def load_table(path: str, sep: str = None, index_col: Optional[Union[int, str]] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        die(f"Missing file: {path}")
    if sep is None:
        # guess by extension
        sep = '\t' if path.endswith(('.tsv', '.tab')) else ','
    try:
        return pd.read_csv(path, sep=sep, header=0, index_col=index_col)
    except Exception as e:
        die(f"Failed to read {path}: {e}")


def ensure_cols(df: pd.DataFrame, cols: Iterable[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"{name} missing required columns: {missing}")


def _safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        return float(x)
    except Exception:
        return default

def preflight_node_attr_report(
    G: nx.Graph,
    name: str,
    color_attr: str,
    size_attr: str = "AxB",
    palette_values: Optional[Set] = None
) -> None:
    """Print helpful counts before plotting."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[check:{name}] nodes={n_nodes} edges={n_edges}")

    if n_nodes == 0:
        print(f"[check:{name}] graph has no nodes")
        return

    colors = [G.nodes[n].get(color_attr, None) for n in G.nodes()]
    sizes  = [G.nodes[n].get(size_attr, None) for n in G.nodes()]

    n_missing_color = sum(c is None for c in colors)
    n_missing_size  = sum(s is None for s in sizes)
    n_nonzero_size  = sum(_safe_float(s, 0.0) > 0 for s in sizes)

    # Compact distribution of color values
    color_counts = Counter(c if c is not None else "<missing>" for c in colors)
    top_colors = ", ".join([f"{k}:{v}" for k, v in color_counts.most_common(8)])
    print(f"[check:{name}] color_attr='{color_attr}': missing={n_missing_color}, top={top_colors}")

    if palette_values is not None:
        n_outside = sum((c not in palette_values) for c in colors if c is not None)
        print(f"[check:{name}] colors not in palette: {n_outside}")

    # Basic size stats
    vals = np.array([_safe_float(s, 0.0) for s in sizes], dtype=float)
    if len(vals):
        print(f"[check:{name}] size_attr='{size_attr}': >0={n_nonzero_size}, "
              f"min={vals.min():.3g}, median={np.median(vals):.3g}, max={vals.max():.3g}")
    else:
        print(f"[check:{name}] size_attr='{size_attr}': no values found")

def save_figure(figpath: str) -> None:
    """Safe save; avoid crashing on empty collections."""
    try:
        # If nothing was added (no artists), add an invisible dot so renderer has something.
        ax = plt.gca()
        if not (ax.collections or ax.patches or ax.lines):
            ax.plot([0], [0], alpha=0)  # invisible fallback
        plt.savefig(figpath, bbox_inches='tight')
        if figpath.endswith(".svg"):
            plt.savefig(figpath.replace(".svg", ".pdf"), bbox_inches='tight')
        print(f"[+] Saved: {figpath}")
    except Exception as e:
        print(f"[!] save_figure failed for {figpath}: {e}")
    finally:
        plt.close()

def split_taxa_string(taxa_str: str, delimiter: str = ';') -> Dict[str, Optional[str]]:
    """Split a SILVA/Greengenes-like lineage into 7 standard levels."""
    levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    if pd.isna(taxa_str) or taxa_str == 'Unassigned':
        return {lv: ('Unassigned' if lv == 'Domain' else None) for lv in levels}
    parts = [p.strip() for p in taxa_str.split(delimiter)]
    parts = [p.split('__', 1)[1] if '__' in p else p for p in parts]
    out = {}
    for i, lv in enumerate(levels):
        out[lv] = parts[i] if i < len(parts) else None
    return out


def write_layout_json(path: str, pos: Dict[str, Tuple[float, float]]):
    try:
        with open(path, 'w') as f:
            json.dump(pos, f)
        ok(f"Cached layout: {path}")
    except Exception as e:
        print(f"[WARN] Could not write layout JSON: {e}", file=sys.stderr)


def read_layout_json(path: str) -> Optional[Dict[str, Tuple[float, float]]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            d = json.load(f)
        # keys must be strings; coords list->tuple
        return {str(k): tuple(v) for k, v in d.items()}
    except Exception as e:
        print(f"[WARN] Failed loading layout JSON ({path}): {e}", file=sys.stderr)
        return None


def spring_layout_cached(G: nx.Graph, seed: int, scale_xy: float,
                         layout_json: Optional[str]) -> Dict[str, Tuple[float, float]]:
    pos = read_layout_json(layout_json)
    if pos is None:
        ok("Computing spring_layout (NetworkX) ...")
        pos = nx.spring_layout(G, seed=seed)
        if scale_xy != 1.0:
            pos = {n: (x * scale_xy, y * scale_xy) for n, (x, y) in pos.items()}
        if layout_json:
            write_layout_json(layout_json, {k: list(v) for k, v in pos.items()})
    else:
        ok(f"Loaded layout from: {layout_json}")
        if scale_xy != 1.0:
            pos = {n: (x * scale_xy, y * scale_xy) for n, (x, y) in pos.items()}
    return pos


def add_node_attrs_from_df(G: nx.Graph, attrs_df: pd.DataFrame, keep_cols: Iterable[str]):
    """Copy selected columns from attrs_df (index must match node IDs) into G."""
    found = list(set(G.nodes()).intersection(set(attrs_df.index)))
    if not found:
        print("[WARN] No overlapping node IDs between graph and attribute table.")
        return
    for col in keep_cols:
        if col in attrs_df.columns:
            nx.set_node_attributes(G, values=attrs_df.loc[found, col].to_dict(), name=col)


def build_type_palette() -> Dict[str, str]:
    return {
        'Oral Rinse': '#6A3D9A',
        'BAL+Oral Rinse': '#F19CBB',
        'BAL': '#0072B2',
        'BAL+Bronchial Brush': '#00FFFF',
        'Bronchial Brush': '#009E73',
        'Bronchial Brush+Oral Rinse': '#C1EAAD',
        'BAL+Bronchial Brush+Oral Rinse': 'lightgray'
    }


def build_status_palette() -> Dict[str, str]:
    return {'Non-Cancer': 'white', 'Cancer': '#A50026', 'Cancer+Non-Cancer': 'lightgray'}


def venn_type_map() -> Dict[str, str]:
    return {
        'Oral Rinse': 'Oral Rinse',
        'BAL': 'BAL',
        'Bronchial Brush': 'Bronchial Brush',
        'Oral Rinse + BAL': 'BAL+Oral Rinse',
        'Oral Rinse + Bronchial Brush': 'Bronchial Brush+Oral Rinse',
        'BAL + Bronchial Brush': 'BAL+Bronchial Brush',
        'Oral Rinse + BAL + Bronchial Brush': 'BAL+Bronchial Brush+Oral Rinse'
    }


def size_legend_handles(vals: Iterable[float], label_prefix: str, scale: float,
                        edge='black', face='gray') -> List[plt.Line2D]:
    return [plt.scatter([], [], s=v * scale, edgecolors=edge, facecolors=face,
                        alpha=1, label=f"{label_prefix} {v}")
            for v in vals]


def draw_edges_light(G: nx.Graph, pos: Dict, alpha: float = 1.0,
                     edge_widths: Optional[List[float]] = None):
    if edge_widths is None:
        nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.2',
                               edge_color='lightgray', alpha=alpha)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()),
                               width=edge_widths, edge_color='lightgray', alpha=alpha)


def draw_nodes_one_by_one(G: nx.Graph, pos: Dict, color_fn, size_fn, alpha_fn=None,
                          lw_fn=None):
    for n in G.nodes():
        color = color_fn(n)
        size = size_fn(n)
        alpha = alpha_fn(n) if alpha_fn else 1.0
        lw = lw_fn(n) if lw_fn else 0.25
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n],
            node_color=[color], node_size=[size],
            edgecolors='black', linewidths=lw, alpha=alpha
        )


def label_selected(G: nx.Graph, pos: Dict, select_nodes: List[str], text_attr: str = 'Taxon'):
    if not _HAS_ADJUSTTEXT:
        print("[WARN] adjustText not installed; skipping labels.")
        return
    texts = []
    for n in select_nodes:
        x, y = pos[n]
        lbl = str(G.nodes[n].get(text_attr, ""))
        texts.append(plt.text(x, y, lbl, fontsize=9, weight='bold', ha='center', va='center'))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                expand_text=(1.2, 1.2), force_text=0.5, force_points=0.2)


# ---------------------------- ISA summaries ----------------------------------
import pandas as pd

def reshape_indicspecies_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    # 1) standardize IDs
    df = summary_df.rename(columns={'ASV': 'ASV_ID'}).copy()

    # 2) infer group names from the *.B columns
    b_cols = [c for c in df.columns if c.endswith('.B')]
    groups = [c[:-2] for c in b_cols]  # strip ".B"

    # 3) build rename map to create A.<group> and B.<group>
    rename_map = {}
    # A columns are the same group names w/o ".B"
    for g in groups:
        if g in df.columns:
            rename_map[g] = f"A.{g}"
        rename_map[f"{g}.B"] = f"B.{g}"

    df = df.rename(columns=rename_map)

    # 4) reshape to long; keep both A and B plus summary stats like stat/p/q
    extra_cols = [
        c for c in df.columns
        if c not in {'ASV_ID', 'index'} and not c.startswith('A.') and not c.startswith('B.')
    ]
    out = pd.wide_to_long(
        df,
        stubnames=['A', 'B'],
        i=['ASV_ID', 'index'],
        j='Group',
        sep='.',
        suffix='.+',          # group names can include spaces/plus signs
    ).reset_index()
    ordered_cols = ['ASV_ID', 'index', 'Group', 'A', 'B'] + [c for c in extra_cols if c in out.columns]
    out = out[ordered_cols].sort_values(['ASV_ID', 'index', 'Group'], ignore_index=True)
    return out


def apply_indicator_overlay(
    df: pd.DataFrame,
    *,
    label_col: str,
    color_col: str,
    stat_col: str = "stat",
    metric_col: str = "q.value",
    stat_thresh: float = 0.25,
    metric_thresh: float = 0.05,
    default_color: str = "lightgray",
    allowed_labels: Optional[set[str]] = None,
) -> pd.DataFrame:
    out = df.copy()
    stat_vals = pd.to_numeric(out.get(stat_col), errors="coerce")
    metric_vals = pd.to_numeric(out.get(metric_col), errors="coerce")
    sig_mask = (stat_vals >= stat_thresh) & (metric_vals <= metric_thresh)
    sig_mask = sig_mask.fillna(False)
    if allowed_labels is not None:
        sig_mask = sig_mask & out[label_col].isin(allowed_labels)
    out[f"{label_col}_sig"] = sig_mask
    out.loc[~sig_mask, label_col] = np.nan
    out.loc[~sig_mask, color_col] = default_color
    return out


def resolve_indicator_metric(summary_df: pd.DataFrame) -> str:
    if "q.value" in summary_df.columns:
        return "q.value"
    if "p.value" in summary_df.columns:
        return "p.value"
    die("Summary table must contain either 'q.value' or 'p.value'.")


def long_AB_for_group(summary_df: pd.DataFrame, index_map: Dict[int, str]) -> pd.DataFrame:
    """
    Convert indicspecies *_summary.tsv into long-form with aligned group rows.
    Keeps rows where Group matches the mapped 'index'.
    """
    df = reshape_indicspecies_summary(summary_df)
    df['Group_mapped'] = df['index'].map(index_map)
    df = df.loc[df['Group'] == df['Group_mapped']].drop(columns=['Group_mapped'])
    df['AxB'] = (df['A'] * df['B']).fillna(0)
    return df


def _simple_undirected_graph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        H.add_edge(u, v, **data)
    return H


def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    H = _simple_undirected_graph(G)
    n_nodes = H.number_of_nodes()
    n_edges = H.number_of_edges()
    abs_weights = []
    for _, _, data in H.edges(data=True):
        try:
            abs_weights.append(abs(float(data.get("weight", 0.0))))
        except Exception:
            abs_weights.append(0.0)
    weights = np.asarray(abs_weights, dtype=float) if abs_weights else np.asarray([], dtype=float)

    metrics = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": float(nx.density(H)) if n_nodes > 1 else np.nan,
        "average_degree": float((2.0 * n_edges) / n_nodes) if n_nodes else np.nan,
        "average_clustering": float(nx.average_clustering(H)) if n_nodes else np.nan,
        "average_clustering_weighted": float(nx.average_clustering(H, weight="weight")) if n_nodes and n_edges else np.nan,
        "transitivity": float(nx.transitivity(H)) if n_nodes and n_edges else np.nan,
        "connected_components": int(nx.number_connected_components(H)) if n_nodes else 0,
        "largest_component_nodes": 0,
        "largest_component_fraction": np.nan,
        "second_largest_component_nodes": 0,
        "mean_component_size": np.nan,
        "median_component_size": np.nan,
        "min_component_size": np.nan,
        "max_component_size": np.nan,
        "singleton_components": 0,
        "non_singleton_components": 0,
        "avg_shortest_path_lcc": np.nan,
        "mean_abs_edge_weight": float(np.mean(weights)) if weights.size else np.nan,
        "median_abs_edge_weight": float(np.median(weights)) if weights.size else np.nan,
        "max_abs_edge_weight": float(np.max(weights)) if weights.size else np.nan,
    }

    if n_nodes:
        components = list(nx.connected_components(H))
        if components:
            component_sizes = sorted((len(c) for c in components), reverse=True)
            largest_nodes = max(components, key=len)
            largest_size = len(largest_nodes)
            metrics["largest_component_nodes"] = int(largest_size)
            metrics["largest_component_fraction"] = float(largest_size / n_nodes)
            metrics["second_largest_component_nodes"] = int(component_sizes[1]) if len(component_sizes) > 1 else 0
            metrics["mean_component_size"] = float(np.mean(component_sizes))
            metrics["median_component_size"] = float(np.median(component_sizes))
            metrics["min_component_size"] = int(np.min(component_sizes))
            metrics["max_component_size"] = int(np.max(component_sizes))
            metrics["singleton_components"] = int(sum(size == 1 for size in component_sizes))
            metrics["non_singleton_components"] = int(sum(size > 1 for size in component_sizes))
            if largest_size > 1:
                largest_sub = H.subgraph(largest_nodes).copy()
                metrics["avg_shortest_path_lcc"] = float(nx.average_shortest_path_length(largest_sub))
    return metrics


def randomize_graph_degree_preserving(
    G: nx.Graph,
    *,
    seed: int,
    nswap_multiplier: int = 10,
) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    n_edges = H.number_of_edges()
    if n_edges < 2:
        return H

    nswap = max(1, int(n_edges * nswap_multiplier))
    max_tries = max(100, nswap * 20)
    nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=seed)
    return H


def summarize_null_model(
    G: nx.Graph,
    *,
    network_id: str,
    replicates: int,
    seed: int,
    nswap_multiplier: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(replicates):
        H = randomize_graph_degree_preserving(
            G,
            seed=int(rng.integers(0, 2**31 - 1)),
            nswap_multiplier=nswap_multiplier,
        )
        metrics = compute_graph_metrics(H)
        metrics["network_id"] = network_id
        metrics["replicate"] = rep + 1
        rows.append(metrics)

    null_df = pd.DataFrame(rows)
    summary = {
        "null_model": "degree_preserving_edge_swap",
        "null_replicates": int(len(null_df)),
    }
    for metric in ["average_clustering", "transitivity", "avg_shortest_path_lcc"]:
        vals = pd.to_numeric(null_df.get(metric), errors="coerce").dropna()
        if vals.empty:
            summary[f"null_{metric}_mean"] = np.nan
            summary[f"null_{metric}_sd"] = np.nan
            summary[f"null_{metric}_p2_5"] = np.nan
            summary[f"null_{metric}_p97_5"] = np.nan
        else:
            summary[f"null_{metric}_mean"] = float(vals.mean())
            summary[f"null_{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary[f"null_{metric}_p2_5"] = float(vals.quantile(0.025))
            summary[f"null_{metric}_p97_5"] = float(vals.quantile(0.975))
    return null_df, summary


def build_component_membership_table(
    G: nx.Graph,
    *,
    network_id: str,
) -> pd.DataFrame:
    H = _simple_undirected_graph(G)
    rows = []
    components = sorted(nx.connected_components(H), key=len, reverse=True)
    for comp_idx, nodes in enumerate(components, start=1):
        component_nodes = sorted(nodes, key=str)
        component_size = len(component_nodes)
        is_largest = comp_idx == 1
        for node_id in component_nodes:
            node_attrs = G.nodes[node_id]
            rows.append({
                "network_id": network_id,
                "component_id": comp_idx,
                "component_size": component_size,
                "is_largest_component": is_largest,
                "GraphML_ID": node_id,
                "ASV_ID": node_attrs.get("Taxon", node_id),
            })
    return pd.DataFrame(rows)


# ---------------------------- Plot modes -------------------------------------
def plot_degree(G: nx.Graph, pos: Dict, out_svg: str, degree_scale: float, edge_width_scale: float):
    plt.figure(figsize=(18, 18))
    # edges scaled linearly by |weight| into a visible width range
    edgelist = list(G.edges())
    abs_w = []
    for e in edgelist:
        try:
            abs_w.append(abs(float(G.edges[e].get('weight', 0.0))))
        except Exception:
            abs_w.append(0.0)

    edge_min_w = 0.25
    edge_max_w = max(edge_min_w + 0.25, float(edge_width_scale))

    if abs_w:
        w_arr = np.asarray(abs_w, dtype=float)
        lo = float(np.nanmin(w_arr))
        hi = float(np.nanmax(w_arr))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = lo
        if hi <= lo:
            edge_widths = [0.5 * (edge_min_w + edge_max_w)] * len(edgelist)
            legend_weights = [lo]
        else:
            def map_weight_to_width(weight: float) -> float:
                t = (float(weight) - lo) / (hi - lo)
                return edge_min_w + t * (edge_max_w - edge_min_w)

            edge_widths = [map_weight_to_width(w) for w in w_arr]
            legend_weights = sorted(set([
                float(np.quantile(w_arr, 0.1)),
                float(np.quantile(w_arr, 0.5)),
                float(np.quantile(w_arr, 0.9)),
            ]))
        draw_edges_light(G, pos, alpha=0.6, edge_widths=edge_widths)
    else:
        legend_weights = []
        draw_edges_light(G, pos, alpha=0.6, edge_widths=None)

    def size_fn(n):
        deg = _safe_float(G.nodes[n].get('Degree', 0), 0.0)
        if deg <= 0:
            return max(2.0, degree_scale * 0.05)   # tiny but visible
        return (deg + 1.0) * degree_scale
    def color_fn(_): return 'black'
    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=lambda n: 0.5)
 
    # legend: sizes must match size_fn exactly
    svals = [0, 1, 3, 5, 10]

    def legend_size(deg):
        if deg <= 0:
            return max(2.0, degree_scale * 0.05)   # must match size_fn
        return (deg + 1.0) * degree_scale

    degree_handles = [
        plt.scatter([], [], s=legend_size(s),
                    edgecolors='black', facecolors='gray', alpha=1,
                    label=f'Degree: {s}')
        for s in svals
    ]
    edge_handles = []
    if legend_weights:
        if len(legend_weights) == 1:
            weight_vals = legend_weights
        else:
            weight_vals = [legend_weights[0], legend_weights[len(legend_weights) // 2], legend_weights[-1]]
        # Preserve order while removing near-duplicates after rounding.
        seen = set()
        for wt in weight_vals:
            key = round(float(wt), 4)
            if key in seen:
                continue
            seen.add(key)
            if len(legend_weights) == 1 or hi <= lo:
                line_w = 0.5 * (edge_min_w + edge_max_w)
            else:
                t = (float(wt) - lo) / (hi - lo)
                line_w = edge_min_w + t * (edge_max_w - edge_min_w)
            edge_handles.append(
                Line2D([0], [0], color="darkgray", linewidth=line_w, alpha=0.8, label=f"|weight|: {float(wt):.3g}")
            )

    legend_handles = degree_handles + edge_handles
    legend_labels = [h.get_label() for h in legend_handles]
    legend_title = "Node Degree / |weight|"
    plt.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        title=legend_title,
        frameon=False,
        scatterpoints=1,
        handlelength=3.0,
        labelspacing=1.5,
    )

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    plt.title("SPIEC-EASI Network\nNode size: Degree | Edges scaled by |weight|")
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


# FILE: network_viz_cli.py
# LOCATION: Ctrl+F -> "def plot_edgeweight_equalnodes(" and REPLACE THE WHOLE FUNCTION with this version
#
# What this does:
# - TRUE LINEAR mapping from a fixed weight range [scale_w_min, scale_w_max]
# - Legend uses exactly the weights you specify (e.g. 0.1, 0.2, 0.3)
# - No quantile clipping, no gamma, no log. Linear means linear.

from matplotlib.lines import Line2D

def plot_edgeweight_equalnodes(
    G: nx.Graph,
    pos: Dict,
    out_svg: str,
    *,
    node_size: float = 80.0,
    node_color: str = "black",
    edge_alpha: float = 1.0,
    edge_min_w: float = 0.25,
    edge_max_w: float = 15.0,

    # NEW: force a linear scale range (THIS is what fixes your problem)
    scale_w_min: float = 0.15,
    scale_w_max: float = 0.5,

    # NEW: legend values you want to show (exact)
    legend_weights: list[float] = None,
):
    """
    Equal-size nodes, neutral node color, and edges scaled LINEARLY by |weight|.

    Linear mapping:
      width = edge_min_w + ((clip(|w|, scale_w_min, scale_w_max) - scale_w_min)
                            / (scale_w_max - scale_w_min)) * (edge_max_w - edge_min_w)

    If legend_weights is provided, legend uses those exact values.
    """

    if legend_weights is None:
        legend_weights = [0.1, 0.25, 0.5]

    plt.figure(figsize=(18, 18))

    # Collect abs weights + edgelist in the same order
    edgelist = []
    abs_w = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        try:
            w = abs(float(w))
        except Exception:
            w = 0.0
        edgelist.append((u, v))
        abs_w.append(w)

    # Always draw nodes (even if no edges)
    if len(edgelist) == 0:
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_size,
            node_color=node_color,
            edgecolors="none",
            alpha=1.0,
        )
        plt.axis("equal")
        plt.axis("off")
        plt.title("SPIEC-EASI POS_SUB Network\nNodes: uniform | Edges: width = |weight| (linear)")
        save_figure(out_svg)
        plt.close()
        return

    # Linear mapper with a fixed range
    lo = float(scale_w_min)
    hi = float(scale_w_max)
    if hi <= lo:
        hi = lo + 1e-12
    denom = (hi - lo)

    def _map_w_to_width(w: float) -> float:
        wc = min(max(float(w), lo), hi)  # clip to [lo, hi]
        t = (wc - lo) / denom            # linear 0..1
        return edge_min_w + t * (edge_max_w - edge_min_w)

    # Map weights -> widths
    w_arr = np.asarray(abs_w, dtype=float)
    edge_widths = [_map_w_to_width(w) for w in w_arr]

    # Draw edges and nodes
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edgelist,
        width=edge_widths,
        edge_color="darkgray",
        alpha=edge_alpha,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors="none",
        alpha=1.0,
    )

    # Legend: EXACT values you want (linear mapping matches them)
    leg_handles = [
        Line2D(
            [0], [0],
            color="darkgray",
            alpha=edge_alpha,
            linewidth=_map_w_to_width(wt),
            label=f"{wt:.1f}",
        )
        for wt in legend_weights
    ]

    plt.legend(
        handles=leg_handles,
        title=f"|weight| → edge width (linear)",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
        handlelength=3.2,
        labelspacing=1.0,
    )

    plt.axis("equal")
    plt.axis("off")
    plt.title("SPIEC-EASI POS_SUB Network\nNodes: uniform | Edges: width = |weight| (linear)")
    save_figure(out_svg)
    plt.close()


# FILE: network_viz_cli.py
# LOCATION 1: Ctrl+F -> "def plot_abundance(" and REPLACE THE WHOLE FUNCTION with this version

def plot_abundance(
    G: nx.Graph,
    pos: Dict,
    out_svg: str,
    *,
    size_attr: str = "mean_all",                 # <- choose mean_all / mean_nonzero / median_all / median_nonzero
    size_label: str = "Abundance",
    size_scale: float = 1.0,
    size_floor: float = 1.0,                      # keep 0s barely visible
    legend_vals: list = None,
    node_color: str = "black",
):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    # quick sanity check (prints)
    preflight_node_attr_report(G, name=f"abundance_{size_attr}", color_attr="Taxon", size_attr=size_attr)

    if legend_vals is None:
        legend_vals = [1, 10, 100, 500, 1000]

    def size_fn(n):
        v = _safe_float(G.nodes[n].get(size_attr, 0.0), 0.0)
        s = v * size_scale
        return max(size_floor, s)

    def color_fn(_):
        return node_color

    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=lambda n: 0.5)

    handles = size_legend_handles(legend_vals, f"{size_label}:", size_scale)
    plt.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Node Attributes",
        frameon=False,
        scatterpoints=1,
        labelspacing=1.5,
    )

    plt.axis("equal"); plt.xlim(auto=False); plt.ylim(auto=False)
    plt.title(f"SPIEC-EASI Network\nNode size: {size_label} ({size_attr})")
    plt.axis("off")
    save_figure(out_svg)
    plt.close()


def plot_type_isa(
    G,
    pos,
    out_svg,
    type_palette,
    isa_scale=500,
    label=False,
    title=None,
    color_attr="type_color",
    label_attr=None,
):
    """
    Robust type-ISA plot:
    - Defaults missing colors to 'lightgray' and missing AxB to 0
    - Ensures node list is non-empty
    - Skips category draws when no nodes in that category
    """
    nodes = list(G.nodes())
    if not nodes:
        print("[!] plot_type_isa: graph has no nodes; skipping.")
        return

    # Build per-node attributes with safe defaults
    node_colors = []
    node_sizes  = []
    for n in nodes:
        d = G.nodes[n]
        c = d.get(color_attr, "lightgray")
        # if value is not one of your palette values, fall back to gray
        if c not in set(type_palette.values()):
            c = "lightgray"
        node_colors.append(c)

        s = _safe_float(d.get("AxB", 0.0), 0.0) * isa_scale
        # floor size so matplotlib doesn't choke on all-zeros
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        node_sizes.append(s)

    fig = plt.figure(figsize=(18, 18))
    ax = plt.gca()

    # Edges: don't use connectionstyle for LineCollection (silences the warning)
    nx.draw_networkx_edges(
        G, pos,
        edge_color="lightgray", alpha=0.8, ax=ax
    )

    # Nodes: single call with complete nodelist (never empty)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black", linewidths=0.25, alpha=0.9, ax=ax
    )

    # Optional labels for non-gray nodes
    if label:
        if label_attr is None:
            label_attr = color_attr
        for n in nodes:
            if G.nodes[n].get(label_attr, "lightgray") != "lightgray":
                x, y = pos[n]
                ax.text(x, y, G.nodes[n].get("Taxon", n),
                        fontsize=9, fontweight="bold",
                        ha="center", va="center")

    # Legend (types) + size legend
    type_handles = [mpatches.Patch(color=c, label=t) for t, c in type_palette.items()]
    size_legend_vals = [0.1, 0.25, 0.5, 0.75, 1.0]
    size_handles = [plt.scatter([], [], s=max(1.0, v*isa_scale),
                                edgecolors="black", facecolors="gray", alpha=1,
                                label=f"ISA: {v:g}") for v in size_legend_vals]

    ax.legend(type_handles + size_handles, [h.get_label() for h in type_handles + size_handles],
              loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title="Node Attributes",
              scatterpoints=1, labelspacing=1.5)

    # Keep proportions & avoid autoscaling surprises
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale(enable=True)

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()
    plt.savefig(out_svg, bbox_inches="tight")
    # optional PDF sibling
    if out_svg.endswith(".svg"):
        plt.savefig(out_svg.replace(".svg", ".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_type_venn(G, pos, out_svg, type_palette, isa_scale=500, label=False, title=None):
    """
    Robust Venn-colored plot:
    - Colors from node['venn_color'] (fallback 'lightgray')
    - Sizes from node['AxB'] * isa_scale (floored to 1)
    - Always draws a non-empty node list in one call
    """
    nodes = list(G.nodes())
    if not nodes:
        print("[!] plot_type_venn: graph has no nodes; skipping.")
        return

    # Preflight report (prints to stdout)
    preflight_node_attr_report(
        G, name="type_venn",
        color_attr="venn_color",
        size_attr="AxB",
        palette_values=set(type_palette.values())
    )

    # Build per-node color/size with safe defaults
    node_colors, node_sizes = [], []
    palette_vals = set(type_palette.values())
    for n in nodes:
        d = G.nodes[n]
        c = d.get("venn_color", "lightgray")
        if c not in palette_vals:
            c = "lightgray"
        node_colors.append(c)

        s = _safe_float(d.get("AxB", 0.0), 0.0) * isa_scale
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        node_sizes.append(s)

    fig = plt.figure(figsize=(18, 18))
    ax = plt.gca()

    # Edges (no connectionstyle for LineCollection)
    nx.draw_networkx_edges(G, pos, edge_color="lightgray", alpha=0.8, ax=ax)

    # Nodes (single call; never an empty list)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black", linewidths=0.25, alpha=0.9, ax=ax
    )

    # Optional labels for colored nodes
    if label:
        for n in nodes:
            if G.nodes[n].get("venn_color", "lightgray") != "lightgray":
                x, y = pos[n]
                ax.text(x, y, G.nodes[n].get("Taxon", n),
                        fontsize=9, fontweight="bold",
                        ha="center", va="center")

    # Legend (types) + size legend
    type_handles = [mpatches.Patch(color=c, label=t) for t, c in type_palette.items()]
    size_vals = [0.1, 0.25, 0.5, 0.75, 1.0]
    size_handles = [plt.scatter([], [], s=max(1.0, v * isa_scale),
                                edgecolors="black", facecolors="gray", alpha=1,
                                label=f"ISA: {v:g}") for v in size_vals]

    ax.legend(type_handles + size_handles,
              [h.get_label() for h in type_handles + size_handles],
              loc="upper left", bbox_to_anchor=(1, 1),
              frameon=False, title="Node Attributes",
              scatterpoints=1, labelspacing=1.5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale(enable=True)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(out_svg)


def plot_status_isa(G: nx.Graph, pos: Dict, out_svg: str, status_palette: Dict[str, str],
                    isa_scale: float, label: bool = False):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    def color_fn(n): return G.nodes[n].get('status_color', 'lightgray')
    def size_fn(n): return float(G.nodes[n].get('AxB', 0.0)) * isa_scale
    def alpha_fn(n): return 0.5 if color_fn(n) == 'lightgray' else 1.0
    def lw_fn(n): return 1.0 if color_fn(n) == 'white' else 0.25  # thicker edge for Non-Cancer

    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=alpha_fn, lw_fn=lw_fn)

    # Legend: thicker box for Non-Cancer
    handles = []
    for stat, col in status_palette.items():
        lw = 1.0 if stat == "Non-Cancer" else 0.25
        edge = "black" if stat == "Non-Cancer" else col
        handles.append(mpatches.Patch(facecolor=col, edgecolor=edge, linewidth=lw, label=stat))
    size_handles = size_legend_handles([0.1, 0.25, 0.5, 0.75, 1.0], "ISA:", isa_scale)
    plt.legend(handles=handles + size_handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Attributes", frameon=False, scatterpoints=1, labelspacing=1.5)

    if label:
        to_label = [n for n in G.nodes() if color_fn(n) != 'lightgray']
        label_selected(G, pos, to_label, text_attr='Taxon')

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    ttl = "SPIEC-EASI Network\nNode color: Cancer Status | Node size: Indicator Species Strength"
    if label: ttl += " (Labeled)"
    plt.title(ttl)
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


def plot_phylum(
    G: nx.Graph,
    pos: Dict,
    out_svg: str,
    phylum_palette: Dict[str, str],
    size_attr: str,
    size_label: str,
    size_scale: float,
):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    # DEBUG/preflight: confirm the attribute is actually on nodes
    preflight_node_attr_report(
        G, name=f"phylum_{size_attr}",
        color_attr="Phylum",
        size_attr=size_attr
    )

    def color_fn(n):
        p = G.nodes[n].get("Phylum")
        return phylum_palette.get(p, "lightgray")

    def size_fn(n):
        # robust against missing/None/str/nan
        v = _safe_float(G.nodes[n].get(size_attr, 0.0), 0.0)
        s = v * size_scale
        # tiny floor so nodes remain barely visible even if 0
        return max(1.0, s)

    def alpha_fn(n):
        return 0.5 if color_fn(n) == "lightgray" else 1.0

    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=alpha_fn)

    # Legends
    phyla_vals = {color_fn(n): G.nodes[n].get("Phylum") for n in G.nodes() if color_fn(n) != "lightgray"}
    color_patches = [
        mpatches.Patch(color=c, label=l)
        for c, l in sorted(phyla_vals.items(), key=lambda x: (x[1] or ""))
    ]

    # Use abundance-like legend when size_attr is an abundance metric
    is_abund = size_attr in ("mean_all", "mean_nonzero", "median_all", "median_nonzero")

    if is_abund:
        # Pick values that match your abundance scale; adjust if needed
        size_vals = [1, 10, 100, 500, 1000]
        titletail = f"{size_attr.replace('_', ' ').title()} ASV Abundance"
    else:
        size_vals = [0.1, 0.25, 0.5, 0.75, 1.0]
        titletail = "Indicator Species Strength"

    size_handles = size_legend_handles(size_vals, size_label + ":", size_scale)

    plt.legend(
        handles=color_patches + size_handles,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Node Attributes",
        frameon=False,
        scatterpoints=1,
        labelspacing=1.5,
    )

    plt.axis("equal")
    plt.xlim(auto=False)
    plt.ylim(auto=False)
    plt.title(f"SPIEC-EASI Network\nNode color: Phylum | Node size: {titletail}")
    plt.axis("off")
    save_figure(out_svg)
    plt.close()



# ---------------------------- Main pipeline ----------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Render SPIEC-EASI GraphML networks with ISA/metadata overlays."
    )
    # Base dir + defaults
    p.add_argument("--data-dir", default="/home/ryan/SeqData/SeqData/UBC/LMP_priority1",
                   help="Base data directory (used to resolve default paths).")
    p.add_argument("--outdir", required=True, help="Output directory for figures.")

    # Inputs (override if paths differ)
    p.add_argument("--graph-pos-all", default=None,
                   help="GraphML with all positive edges (default: <data-dir>/spark_combined_output/spieceasi/network_pos_all.graphml)")
    p.add_argument("--graph-pos-sub", default=None,
                   help="GraphML with thresholded positive edges (default: <data-dir>/spark_combined_output/spieceasi/network_pos_thr.graphml OR _pos_sub.graphml)")
    p.add_argument("--node-features", default=None,
                   help="Node features CSV (default: <data-dir>/spark_combined_output/spieceasi/node_features.csv)")
    p.add_argument("--asv-counts", default=None,
                   help="ASV count table (tsv; default: <data-dir>/spark_combined_output/ASVs/ASV_final.micro.tsv)")
    p.add_argument("--taxonomy", default=None,
                   help="taxonomy_updated.tsv (default: <data-dir>/spark_combined_output/metadata/taxonomy_updated.tsv)")
    p.add_argument("--venn", default=None,
                   help="Three_types_venn_presence_table.tsv (default: <data-dir>/spark_combined_output/metadata/Three_types_venn_presence_table.tsv)")
    p.add_argument("--type-summary", default=None,
                   help="type_group_indicator_species_summary.tsv")
    p.add_argument("--status-summary", default=None,
                   help="status_indicator_species_summary.tsv")
    p.add_argument("--type-q-thresh", type=float, default=0.05,
                   help="Maximum q.value to keep a type indicator color overlay (default: 0.05).")
    p.add_argument("--type-stat-thresh", type=float, default=0.25,
                   help="Minimum stat to keep a type indicator color overlay (default: 0.25).")
    p.add_argument("--status-q-thresh", type=float, default=0.05,
                   help="Maximum q.value to keep a status indicator color overlay (default: 0.05).")
    p.add_argument("--status-stat-thresh", type=float, default=0.25,
                   help="Minimum stat to keep a status indicator color overlay (default: 0.25).")

    # Layout options
    p.add_argument("--layout-json-all", default=None, help="Cache/Load layout JSON for graph-pos-all.")
    p.add_argument("--layout-json-sub", default=None, help="Cache/Load layout JSON for graph-pos-sub.")
    p.add_argument("--layout-seed", type=int, default=42, help="Seed for spring_layout.")
    p.add_argument("--layout-scale", type=float, default=3.0, help="Scale factor applied to layout coordinates.")

    # Visual scales
    p.add_argument("--degree-scale", type=float, default=80.0, help="Base size multiplier for degree plots.")
    p.add_argument("--edge-width-scale", type=float, default=5.0, help="Edge width multiplier for |weight|.")
    p.add_argument("--isa-scale", type=float, default=500.0, help="Node size multiplier for ISA (AxB).")
    p.add_argument("--null-replicates", type=int, default=100,
                   help="Number of degree-preserving random graphs for topology summary (default: 100).")
    p.add_argument("--null-nswap-multiplier", type=int, default=10,
                   help="Edge swaps per observed edge in each null graph (default: 10).")

    # Which plots to render
    p.add_argument("--modes", nargs="+", default=["all"],
                   choices=[
                       "degree_all", "degree_sub",
                       "abundance_sub", "abundance_median_sub",
                       "edgeweight_sub",
                       "type_isa", "type_isa_labeled",
                       "type_isa_all", "type_isa_all_labeled",
                       "type_isa_all_lung_labels",
                       "type_venn", "type_venn_labeled",
                       "status_isa", "status_isa_labeled",
                       "phylum_abund", "phylum_isa",
                       "all"
                   ],
                   help="Which figure(s) to render.")

    args = p.parse_args()

    data_dir = args.data_dir
    os.makedirs(args.outdir, exist_ok=True)
    tables_outdir = os.path.join(args.outdir, "tables")
    os.makedirs(tables_outdir, exist_ok=True)

    # Resolve defaults
    graph_all = args.graph_pos_all or os.path.join(data_dir, "spark_combined_output/spieceasi/network_pos_all.graphml")
    graph_sub = args.graph_pos_sub or os.path.join(data_dir, "spark_combined_output/spieceasi/network_pos_thr.graphml")
    node_features_path = args.node_features or os.path.join(data_dir, "spark_combined_output/spieceasi/node_features.csv")
    asv_counts_path = args.asv_counts or os.path.join(data_dir, "spark_combined_output/ASVs/ASV_final.micro.tsv")
    taxonomy_path = args.taxonomy or os.path.join(data_dir, "spark_combined_output/metadata/taxonomy_updated.tsv")
    venn_path = args.venn or os.path.join(data_dir, "spark_combined_output/metadata/Three_types_venn_presence_table.tsv")
    type_summary_path = args.type_summary or os.path.join(data_dir, "spark_combined_output/indicspecies/type_group_indicator_species_summary.tsv")
    status_summary_path = args.status_summary or os.path.join(data_dir, "spark_combined_output/indicspecies/status_indicator_species_summary.tsv")

    # Load inputs
    nf = load_table(node_features_path, sep=',', index_col=0)  # index = GraphML_ID
    ensure_cols(nf, ["Taxon", "Degree", "Betweenness", "Closeness", "EigenCentral"], "node_features")

    asv = load_table(asv_counts_path, sep='\t', index_col=0)
    asv_stack = asv.stack().reset_index()
    asv_stack.columns = ['ASV_ID', 'sample', 'count']

    # --- Abundance statistics per ASV ---
    abund_stats = (
        asv_stack
        .groupby("ASV_ID")["count"]
        .agg(
            mean_all="mean",                          # includes zeros
            mean_nonzero=lambda s: s[s > 0].mean(),   # only when present
            median_all="median",                      # includes zeros
            median_nonzero=lambda s: s[s > 0].median()
        )
        .reset_index()
    )

    # Keep plotting-friendly numeric types
    abund_stats["mean_all"] = np.ceil(abund_stats["mean_all"]).astype(float)
    abund_stats["mean_nonzero"] = abund_stats["mean_nonzero"].astype(float).fillna(0.0)
    abund_stats["median_all"] = abund_stats["median_all"].astype(float)
    abund_stats["median_nonzero"] = abund_stats["median_nonzero"].astype(float).fillna(0.0)

    abund_stats = abund_stats[
        ["ASV_ID", "mean_all", "mean_nonzero", "median_all", "median_nonzero"]
    ]

    tax = load_table(taxonomy_path, sep='\t')
    if "Feature ID" in tax.columns:
        tax['ASV_ID'] = tax['Feature ID'].astype(str).str.partition(';')[0]
    if 'ASV_ID' not in tax.columns or 'Taxon' not in tax.columns:
        # Handle case where ASV_ID is embedded; keep your original logic
        if 'ASV_ID' in tax.columns:
            pass
        else:
            die("taxonomy_updated.tsv must include 'ASV_ID' and 'Taxon' columns.")
    # normalize ASV_ID if it contains suffix like ';...' at end
    tax['ASV_ID'] = [x.rsplit(';', 1)[0] if isinstance(x, str) and ';' in x else x for x in tax['ASV_ID']]
    # expand taxonomy levels
    tdf = pd.DataFrame([split_taxa_string(x) for x in tax['Taxon']])
    tax = pd.concat([tax[['ASV_ID']], tdf], axis=1).set_index('ASV_ID', drop=True)

    venn = load_table(venn_path, sep='\t')
    ensure_cols(venn, ['ASV_ID', 'grouping'], "venn_presence_table")
    venn = venn.set_index('ASV_ID')

    # ISA summaries (A,B, AxB)
    # maps from integer index to group labels:
    status_index = {1: 'Cancer', 2: 'Non-Cancer', 3: 'Cancer+Non-Cancer'}
    type_index = {
        1: 'BAL', 2: 'Bronchial Brush', 3: 'Oral Rinse',
        4: 'BAL+Bronchial Brush', 5: 'BAL+Oral Rinse', 6: 'Bronchial Brush+Oral Rinse',
        7: 'BAL+Bronchial Brush+Oral Rinse'
    }
    type_palette = build_type_palette()
    status_palette = build_status_palette()

    type_sum = load_table(type_summary_path, sep='\t')
    status_sum = load_table(status_summary_path, sep='\t')

    ensure_cols(type_sum, ['ASV', 'index', 'stat', 'p.value'], "type_summary")
    ensure_cols(status_sum, ['ASV', 'index', 'stat', 'p.value'], "status_summary")
    type_metric_col = resolve_indicator_metric(type_sum)
    status_metric_col = resolve_indicator_metric(status_sum)
    ok(f"type ISA overlay metric: {type_metric_col}")
    ok(f"status ISA overlay metric: {status_metric_col}")
    type_lung_labels = {
        "Bronchial Brush",
        "BAL+Bronchial Brush",
        "Bronchial Brush+Oral Rinse",
        "BAL+Bronchial Brush+Oral Rinse",
    }

    type_long = long_AB_for_group(type_sum.copy(), type_index)
    status_long = long_AB_for_group(status_sum.copy(), status_index)

    # ----- Build augmented node attribute tables -----
    # nfeat_type_df: join nf (GraphML_ID index) with type ISA via Taxon (ASV)
    nfeat_type = nf.reset_index().merge(
        type_long.set_index('ASV_ID'), left_on='Taxon', right_index=True, how='left'
    ).set_index('GraphML_ID')
    nfeat_type['type_stat'] = pd.to_numeric(nfeat_type.get('stat'), errors='coerce')
    nfeat_type['type_metric_value'] = pd.to_numeric(nfeat_type.get(type_metric_col), errors='coerce')

    # color per type group index
    nfeat_type['type_name'] = nfeat_type['index'].map(type_index)
    nfeat_type['type_color_all'] = nfeat_type['type_name'].map(type_palette).fillna('lightgray')
    nfeat_type = apply_indicator_overlay(
        nfeat_type,
        label_col="type_name",
        color_col="type_color_all",
        stat_col="type_stat",
        metric_col="type_metric_value",
        stat_thresh=args.type_stat_thresh,
        metric_thresh=args.type_q_thresh,
    )
    nfeat_type['type_name_all'] = nfeat_type['type_name']
    nfeat_type['type_color_lung'] = nfeat_type['type_name'].map(type_palette).fillna('lightgray')
    nfeat_type = apply_indicator_overlay(
        nfeat_type,
        label_col="type_name",
        color_col="type_color_lung",
        stat_col="type_stat",
        metric_col="type_metric_value",
        stat_thresh=args.type_stat_thresh,
        metric_thresh=args.type_q_thresh,
        allowed_labels=type_lung_labels,
    )
    nfeat_type['type_name_lung'] = nfeat_type['type_name']
    nfeat_type['type_color'] = nfeat_type['type_color_lung']
    ok(f"type ISA all-significant colored nodes: {int((nfeat_type['type_color_all'] != 'lightgray').sum())}")
    ok(f"type ISA lung-associated colored nodes: {int((nfeat_type['type_color_lung'] != 'lightgray').sum())}")

    # venn color
    venn_map = venn_type_map()
    tmp = nf.reset_index().merge(venn, left_on='Taxon', right_index=True, how='left')
    tmp['venn_color'] = tmp['grouping'].map(lambda g: type_palette.get(venn_map.get(g, ""), 'lightgray'))
    nfeat_type['venn_color'] = tmp.set_index('GraphML_ID')['venn_color']

    # add taxonomy columns to type table
    nfeat_type = nfeat_type.reset_index().merge(
        tax.reset_index(), left_on='Taxon', right_on='ASV_ID', how='left'
    ).set_index('GraphML_ID')

    # nfeat_status_df: join nf with status ISA via Taxon (ASV)
    nfeat_status = nf.reset_index().merge(
        status_long.set_index('ASV_ID'), left_on='Taxon', right_index=True, how='left'
    ).set_index('GraphML_ID')
    nfeat_status['status_stat'] = pd.to_numeric(nfeat_status.get('stat'), errors='coerce')
    nfeat_status['status_metric_value'] = pd.to_numeric(nfeat_status.get(status_metric_col), errors='coerce')
    nfeat_status['status_name'] = nfeat_status['index'].map(status_index)
    nfeat_status['status_color'] = nfeat_status['status_name'].map(status_palette).fillna('lightgray')
    nfeat_status = apply_indicator_overlay(
        nfeat_status,
        label_col="status_name",
        color_col="status_color",
        stat_col="status_stat",
        metric_col="status_metric_value",
        stat_thresh=args.status_stat_thresh,
        metric_thresh=args.status_q_thresh,
    )
    ok(f"status ISA colored nodes: {int((nfeat_status['status_color'] != 'lightgray').sum())}")
    # add taxonomy to status table (for consistency)
    nfeat_status = nfeat_status.reset_index().merge(
        tax.reset_index(), left_on='Taxon', right_on='ASV_ID', how='left'
    ).set_index('GraphML_ID')

    # abundance table with taxonomy
    nfeat_abund = nf.reset_index().merge(
        abund_stats, left_on="Taxon", right_on="ASV_ID", how="left"
    ).set_index("GraphML_ID")
    nfeat_abund = nfeat_abund.reset_index().merge(
        tax.reset_index(), left_on='Taxon', right_on='ASV_ID', how='left'
    ).set_index('GraphML_ID')

    # Build Phylum palette from type table (prefer real categories)
    phyla = pd.concat([nfeat_type['Phylum'], nfeat_abund['Phylum']]).dropna().unique().tolist()
    phylum_palette = {p: c for p, c in zip(phyla, sns.color_palette('tab20', len(phyla)))}

    keep_cols = [
        'Taxon', 'Degree', 'Betweenness', 'Closeness', 'EigenCentral',
        'A', 'B', 'AxB',
        'type_name', 'type_name_all', 'type_name_lung',
        'type_color', 'type_color_all', 'type_color_lung', 'type_name_sig', 'type_stat', 'type_metric_value',
        'status_name', 'status_color', 'status_name_sig', 'status_stat', 'status_metric_value',
        'venn_color', 'Phylum',
        'mean_all', 'mean_nonzero', 'median_all', 'median_nonzero',
    ]

    # -------------------- Load graphs + positions -----------------------------
    def load_graph(path: str) -> nx.Graph:
        if not os.path.exists(path):
            die(f"GraphML not found: {path}")
        G = nx.read_graphml(path)
        # GraphML may load numeric node IDs as strings; good for attribute joins
        return G

    G_all = load_graph(graph_all)
    G_sub = load_graph(graph_sub)

    # Topology summaries for manuscript-ready reporting
    topology_rows = []
    null_rows = []
    for network_id, graph_obj, seed_offset in [
        ("POS_ALL", G_all, 0),
        ("POS_SUB", G_sub, 1),
    ]:
        observed = compute_graph_metrics(graph_obj)
        observed["network_id"] = network_id
        null_df, null_summary = summarize_null_model(
            graph_obj,
            network_id=network_id,
            replicates=args.null_replicates,
            seed=args.layout_seed + seed_offset,
            nswap_multiplier=args.null_nswap_multiplier,
        )
        obs_clust = observed.get("average_clustering", np.nan)
        null_clust = pd.to_numeric(null_df.get("average_clustering"), errors="coerce").dropna()
        observed["empirical_p_null_clustering_ge_observed"] = (
            float((null_clust >= obs_clust).mean()) if len(null_clust) else np.nan
        )
        observed["clustering_ratio_vs_null_mean"] = (
            float(obs_clust / null_summary["null_average_clustering_mean"])
            if pd.notna(obs_clust) and pd.notna(null_summary["null_average_clustering_mean"]) and null_summary["null_average_clustering_mean"] not in (0, 0.0)
            else np.nan
        )
        observed.update(null_summary)
        topology_rows.append(observed)
        null_rows.append(null_df)

    topology_summary_df = pd.DataFrame(topology_rows)
    null_replicates_df = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    topology_summary_path = os.path.join(tables_outdir, "network_topology_summary.tsv")
    null_replicates_path = os.path.join(tables_outdir, "network_topology_null_replicates.tsv")
    topology_summary_df.to_csv(topology_summary_path, sep="\t", index=False)
    null_replicates_df.to_csv(null_replicates_path, sep="\t", index=False)
    ok(f"wrote topology summary: {topology_summary_path}")
    ok(f"wrote topology null replicates: {null_replicates_path}")

    # Attach attributes (each plot function can use what it needs)
    add_node_attrs_from_df(G_all, nfeat_type, keep_cols)
    add_node_attrs_from_df(G_all, nfeat_status, keep_cols)
    add_node_attrs_from_df(G_all, nfeat_abund, keep_cols)

    add_node_attrs_from_df(G_sub, nfeat_type, keep_cols)
    add_node_attrs_from_df(G_sub, nfeat_status, keep_cols)
    add_node_attrs_from_df(G_sub, nfeat_abund, keep_cols)

    component_membership_all = build_component_membership_table(G_all, network_id="POS_ALL")
    component_membership_sub = build_component_membership_table(G_sub, network_id="POS_SUB")
    component_membership_all_path = os.path.join(tables_outdir, "network_component_membership_POS_ALL.tsv")
    component_membership_sub_path = os.path.join(tables_outdir, "network_component_membership_POS_SUB.tsv")
    component_membership_all.to_csv(component_membership_all_path, sep="\t", index=False)
    component_membership_sub.to_csv(component_membership_sub_path, sep="\t", index=False)
    ok(f"wrote component membership: {component_membership_all_path}")
    ok(f"wrote component membership: {component_membership_sub_path}")

    # Positions (cached)
    pos_all = spring_layout_cached(G_all, seed=args.layout_seed,
                                   scale_xy=args.layout_scale,
                                   layout_json=args.layout_json_all)
    pos_sub = spring_layout_cached(G_sub, seed=args.layout_seed,
                                   scale_xy=args.layout_scale,
                                   layout_json=args.layout_json_sub)

    # -------------------- Choose and render plots -----------------------------
    modes = set(args.modes)
    if "all" in modes:
        modes = {
            "degree_all", "degree_sub",
            "abundance_sub", "abundance_median_sub",
            "type_isa", "type_isa_labeled",
            "type_isa_all", "type_isa_all_labeled",
            "type_isa_all_lung_labels",
            "type_venn", "type_venn_labeled",
            "status_isa", "status_isa_labeled",
            "phylum_abund", "phylum_isa"
        }

    # Example for the Venn plot on the thresholded graph
    preflight_node_attr_report(
        G_sub, name="pre-venn-sub",
        color_attr="venn_color", size_attr="AxB",
        palette_values=set(type_palette.values())
    )

    # Combined degree + edgeweight (all edges)
    if "degree_all" in modes:
        out = os.path.join(args.outdir, "network_degree_edgeweight_combined_POS_ALL.svg")
        plot_degree(G_all, pos_all, out, degree_scale=args.degree_scale,
                    edge_width_scale=args.edge_width_scale)

    # Combined degree + edgeweight (thresholded subgraph)
    render_degree_sub = ("degree_sub" in modes) or ("edgeweight_sub" in modes)
    if render_degree_sub:
        out = os.path.join(args.outdir, "network_degree_edgeweight_combined_POS_SUB.svg")
        plot_degree(G_sub, pos_sub, out, degree_scale=args.degree_scale,
                    edge_width_scale=args.edge_width_scale)

    # Abundance (thresholded subgraph) — multiple variants
    if "abundance_sub" in modes:
        out = os.path.join(args.outdir, "network_ABUND_mean_all_POS_SUB.svg")
        plot_abundance(
            G_sub, pos_sub, out,
            size_attr="mean_all",
            size_label="Mean abundance (all samples)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_mean_nonzero_POS_SUB.svg")
        plot_abundance(
            G_sub, pos_sub, out,
            size_attr="mean_nonzero",
            size_label="Mean abundance (nonzero only)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_median_all_POS_SUB.svg")
        plot_abundance(
            G_sub, pos_sub, out,
            size_attr="median_all",
            size_label="Median abundance (all samples)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_median_nonzero_POS_SUB.svg")
        plot_abundance(
            G_sub, pos_sub, out,
            size_attr="median_nonzero",
            size_label="Median abundance (nonzero only)",
            size_scale=1.0,
        )

    # Type ISA (lung-associated significant colors only; size by AxB)
    if "type_isa" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_lung_associated.svg")
        plot_type_isa(
            G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=False,
            title="SPIEC-EASI Network\nNode color: Type Group Indicators (Lung-associated) | Node size: Indicator Species Strength",
            color_attr="type_color_lung",
        )
    if "type_isa_labeled" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_lung_associated_LABELED.svg")
        plot_type_isa(
            G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=True,
            title="SPIEC-EASI Network\nNode color: Type Group Indicators (Lung-associated) | Node size: Indicator Species Strength",
            color_attr="type_color_lung",
        )
    if "type_isa_all" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_all_significant.svg")
        plot_type_isa(
            G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=False,
            title="SPIEC-EASI Network\nNode color: Type Group Indicators (All significant) | Node size: Indicator Species Strength",
            color_attr="type_color_all",
        )
    if "type_isa_all_labeled" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_all_significant_LABELED.svg")
        plot_type_isa(
            G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=True,
            title="SPIEC-EASI Network\nNode color: Type Group Indicators (All significant) | Node size: Indicator Species Strength",
            color_attr="type_color_all",
        )
    if "type_isa_all_lung_labels" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_all_significant_lung_labels.svg")
        plot_type_isa(
            G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=True,
            title="SPIEC-EASI Network\nNode color: Type Group Indicators (All significant) | Labels: Lung-associated only | Node size: Indicator Species Strength",
            color_attr="type_color_all",
            label_attr="type_color_lung",
        )

    # Type Venn
    if "type_venn" in modes:
        out = os.path.join(args.outdir, "network_type_VENN.svg")
        plot_type_venn(G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=False)
    if "type_venn_labeled" in modes:
        out = os.path.join(args.outdir, "network_type_VENN_LABELED.svg")
        plot_type_venn(G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=True)

    # Status ISA
    if "status_isa" in modes:
        out = os.path.join(args.outdir, "network_status_ISA.svg")
        plot_status_isa(G_sub, pos_sub, out, status_palette, isa_scale=args.isa_scale, label=False)
    if "status_isa_labeled" in modes:
        out = os.path.join(args.outdir, "network_status_ISA_LABELED.svg")
        plot_status_isa(G_sub, pos_sub, out, status_palette, isa_scale=args.isa_scale, label=True)

    # Phylum × {abundance, ISA}
    if "phylum_abund" in modes:
        out = os.path.join(args.outdir, "network_ABUND_mean_all_Phylum.svg")
        plot_phylum(
            G_sub, pos_sub, out,
            phylum_palette,
            size_attr="mean_all",
            size_label="Mean abundance (all samples)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_median_all_Phylum.svg")
        plot_phylum(
            G_sub, pos_sub, out,
            phylum_palette,
            size_attr="median_all",
            size_label="Median abundance (all samples)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_mean_nonzero_Phylum.svg")
        plot_phylum(
            G_sub, pos_sub, out,
            phylum_palette,
            size_attr="mean_nonzero",
            size_label="Mean abundance (nonzero only)",
            size_scale=1.0,
        )

        out = os.path.join(args.outdir, "network_ABUND_median_nonzero_Phylum.svg")
        plot_phylum(
            G_sub, pos_sub, out,
            phylum_palette,
            size_attr="median_nonzero",
            size_label="Median abundance (nonzero only)",
            size_scale=1.0,
        )

    if "phylum_isa" in modes:
        out = os.path.join(args.outdir, "network_phylum_ISA.svg")
        plot_phylum(
            G_sub, pos_sub, out,
            phylum_palette,
            size_attr="AxB",
            size_label="ISA",
            size_scale=args.isa_scale,
        )

    ok("All done.")


if __name__ == "__main__":
    main()
