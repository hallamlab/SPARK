#!/usr/bin/env python3
"""
Network visualization CLI for SPIEC-EASI GraphML outputs + ISA/metadata overlays.

Features
- Robust CLI (argparse)
- All paths & style values are configurable
- Safe I/O + schema checks
- Reusable spring-layout (cache to JSON)
- Multiple plot “modes” (degree, abundance, type ISA, status ISA, venn, phylum×{abundance,ISA}, labeled variants)
- Consistent aesthetics with your rcParams
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Tuple, Optional, List

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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


def load_table(path: str, sep: str = None, index_col: Optional[int | str] = None) -> pd.DataFrame:
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


def save_figure(figpath: str):
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches='tight')
    root, ext = os.path.splitext(figpath)
    if ext.lower() == ".svg":
        plt.savefig(root + ".pdf", bbox_inches='tight')
    ok(f"Saved: {figpath}")


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
    found = set(G.nodes()).intersection(set(attrs_df.index))
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
        'BAL+Lung Brush': '#00FFFF',
        'Lung Brush': '#009E73',
        'Lung Brush+Oral Rinse': '#C1EAAD',
        'BAL+Lung Brush+Oral Rinse': 'lightgray'
    }


def build_status_palette() -> Dict[str, str]:
    return {'Non-Cancer': 'white', 'Cancer': '#A50026', 'Cancer+Non-Cancer': 'lightgray'}


def venn_type_map() -> Dict[str, str]:
    return {
        'Oral Rinse': 'Oral Rinse',
        'BAL': 'BAL',
        'Lung Brush': 'Lung Brush',
        'Oral Rinse + BAL': 'BAL+Oral Rinse',
        'Oral Rinse + Lung Brush': 'Lung Brush+Oral Rinse',
        'BAL + Lung Brush': 'BAL+Lung Brush',
        'Oral Rinse + BAL + Lung Brush': 'BAL+Lung Brush+Oral Rinse'
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
def long_AB_for_group(summary_df: pd.DataFrame, index_map: Dict[int, str]) -> pd.DataFrame:
    """
    Convert indicspecies *_summary.tsv into long-form with aligned group rows.
    Keeps rows where Group matches the mapped 'index'.
    """
    df = pd.wide_to_long(
        summary_df.rename(columns={'ASV': 'ASV_ID'}),
        stubnames=['A', 'B'],
        i=['ASV_ID', 'index'],
        j='Group',
        sep='.',
        suffix='.*'
    ).reset_index()[['ASV_ID', 'index', 'Group', 'A', 'B']]
    df['Group_mapped'] = df['index'].map(index_map)
    df = df.loc[df['Group'] == df['Group_mapped']].drop(columns=['Group_mapped'])
    df['AxB'] = (df['A'] * df['B']).fillna(0)
    return df


# ---------------------------- Plot modes -------------------------------------
def plot_degree(G: nx.Graph, pos: Dict, out_svg: str, degree_scale: float, edge_width_scale: float):
    plt.figure(figsize=(18, 18))
    # edges weighted by |weight|
    e_w = [abs(G.edges[e].get('weight', 0)) * edge_width_scale for e in G.edges()]
    draw_edges_light(G, pos, alpha=0.6, edge_widths=e_w)

    def size_fn(n): return (G.nodes[n].get('Degree', 0) + 1) * degree_scale
    def color_fn(_): return 'black'
    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=lambda n: 0.5)

    # legend
    svals = [0, 1, 3, 5, 10]
    handles = [plt.scatter([], [], s=(s + 1) * degree_scale, edgecolors='black',
                           facecolors='gray', alpha=1, label=f'{s}') for s in svals]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Degree", frameon=False, scatterpoints=1, labelspacing=1.5)

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    plt.title("SPIEC-EASI Network\nNode size: Degree | Edges scaled by |weight|")
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


def plot_abundance(G: nx.Graph, pos: Dict, out_svg: str):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    def size_fn(n): return float(G.nodes[n].get('mean', 1))
    def color_fn(_): return 'black'
    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=lambda n: 0.5)

    handles = size_legend_handles([1, 10, 100, 500, 1000], "Abundance:", 1.0)
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Attributes", frameon=False, scatterpoints=1, labelspacing=1.5)

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    plt.title("SPIEC-EASI Network\nNode size: Mean ASV Abundance")
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


def plot_type_isa(G: nx.Graph, pos: Dict, out_svg: str, type_palette: Dict[str, str],
                  isa_scale: float, label: bool = False):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    # Color is per node 'color' (already mapped); Size is AxB * scale
    def color_fn(n): return G.nodes[n].get('color', 'lightgray')
    def size_fn(n): return float(G.nodes[n].get('AxB', 0.0)) * isa_scale
    def alpha_fn(n): return 0.5 if color_fn(n) == 'lightgray' else 1.0
    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=alpha_fn)

    # legends: type colors + ISA size
    color_handles = [mpatches.Patch(color=c, label=k) for k, c in type_palette.items()]
    size_handles = size_legend_handles([0.1, 0.25, 0.5, 0.75, 1.0], "ISA:", isa_scale)
    plt.legend(handles=color_handles + size_handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Attributes", frameon=False, scatterpoints=1, labelspacing=1.5)

    if label:
        to_label = [n for n in G.nodes() if color_fn(n) != 'lightgray']
        label_selected(G, pos, to_label, text_attr='Taxon')

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    ttl = "SPIEC-EASI Network\nNode color: Sample Type | Node size: Indicator Species Strength"
    if label: ttl += " (Labeled)"
    plt.title(ttl)
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


def plot_type_venn(G: nx.Graph, pos: Dict, out_svg: str, type_palette: Dict[str, str],
                   isa_scale: float, label: bool = False):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    def color_fn(n): return G.nodes[n].get('venn_color', 'lightgray')
    def size_fn(n): return float(G.nodes[n].get('AxB', 0.0)) * isa_scale
    def alpha_fn(n): return 0.5 if color_fn(n) == 'lightgray' else 1.0
    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=alpha_fn)

    color_handles = [mpatches.Patch(color=c, label=k) for k, c in type_palette.items()]
    size_handles = size_legend_handles([0.1, 0.25, 0.5, 0.75, 1.0], "ISA:", isa_scale)
    plt.legend(handles=color_handles + size_handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Attributes", frameon=False, scatterpoints=1, labelspacing=1.5)

    if label:
        to_label = [n for n in G.nodes() if color_fn(n) != 'lightgray']
        label_selected(G, pos, to_label, text_attr='Taxon')

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    ttl = "SPIEC-EASI Network\nNode color: Venn Group | Node size: Indicator Species Strength"
    if label: ttl += " (Labeled)"
    plt.title(ttl)
    plt.axis('off')
    save_figure(out_svg)
    plt.close()


def plot_status_isa(G: nx.Graph, pos: Dict, out_svg: str, status_palette: Dict[str, str],
                    isa_scale: float, label: bool = False):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    def color_fn(n): return G.nodes[n].get('color', 'lightgray')
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


def plot_phylum(G: nx.Graph, pos: Dict, out_svg: str, phylum_palette: Dict[str, str],
                size_attr: str, size_label: str, size_scale: float):
    plt.figure(figsize=(18, 18))
    draw_edges_light(G, pos, alpha=1.0)

    def color_fn(n):
        p = G.nodes[n].get('Phylum')
        return phylum_palette.get(p, 'lightgray')

    def size_fn(n):
        return float(G.nodes[n].get(size_attr, 0.0)) * size_scale

    def alpha_fn(n): return 0.5 if color_fn(n) == 'lightgray' else 1.0

    draw_nodes_one_by_one(G, pos, color_fn, size_fn, alpha_fn=alpha_fn)

    # Legends
    # Unique phyla/colors present in the graph
    phyla_vals = {color_fn(n): G.nodes[n].get('Phylum') for n in G.nodes() if color_fn(n) != 'lightgray'}
    color_patches = [mpatches.Patch(color=c, label=l) for c, l in sorted(phyla_vals.items(), key=lambda x: (x[1] or ""))]
    size_vals = [1, 10, 100, 500, 1000] if size_attr == 'mean' else [0.1, 0.25, 0.5, 0.75, 1.0]
    size_handles = size_legend_handles(size_vals, size_label + ":", size_scale)
    plt.legend(handles=color_patches + size_handles, loc='upper left', bbox_to_anchor=(1, 1),
               title="Node Attributes", frameon=False, scatterpoints=1, labelspacing=1.5)

    plt.axis('equal'); plt.xlim(auto=False); plt.ylim(auto=False)
    titletail = "Mean ASV Abundance" if size_attr == 'mean' else "Indicator Species Strength"
    plt.title(f"SPIEC-EASI Network\nNode color: Phylum | Node size: {titletail}")
    plt.axis('off')
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

    # Layout options
    p.add_argument("--layout-json-all", default=None, help="Cache/Load layout JSON for graph-pos-all.")
    p.add_argument("--layout-json-sub", default=None, help="Cache/Load layout JSON for graph-pos-sub.")
    p.add_argument("--layout-seed", type=int, default=42, help="Seed for spring_layout.")
    p.add_argument("--layout-scale", type=float, default=3.0, help="Scale factor applied to layout coordinates.")

    # Visual scales
    p.add_argument("--degree-scale", type=float, default=80.0, help="Base size multiplier for degree plots.")
    p.add_argument("--edge-width-scale", type=float, default=5.0, help="Edge width multiplier for |weight|.")
    p.add_argument("--isa-scale", type=float, default=500.0, help="Node size multiplier for ISA (AxB).")

    # Which plots to render
    p.add_argument("--modes", nargs="+", default=["all"],
                   choices=[
                       "degree_all", "degree_sub",
                       "abundance_sub",
                       "type_isa", "type_isa_labeled",
                       "type_venn", "type_venn_labeled",
                       "status_isa", "status_isa_labeled",
                       "phylum_abund", "phylum_isa",
                       "all"
                   ],
                   help="Which figure(s) to render.")

    args = p.parse_args()

    data_dir = args.data_dir
    os.makedirs(args.outdir, exist_ok=True)

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
    mean_abund = asv_stack.groupby('ASV_ID')['count'].mean().reset_index()
    mean_abund['mean'] = np.ceil(mean_abund['count']).astype(float)
    mean_abund = mean_abund[['ASV_ID', 'mean']]

    tax = load_table(taxonomy_path, sep='\t')
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
        1: 'BAL', 2: 'Lung Brush', 3: 'Oral Rinse',
        4: 'BAL+Lung Brush', 5: 'BAL+Oral Rinse', 6: 'Lung Brush+Oral Rinse',
        7: 'BAL+Lung Brush+Oral Rinse'
    }
    type_palette = build_type_palette()
    status_palette = build_status_palette()

    type_sum = load_table(type_summary_path, sep='\t')
    status_sum = load_table(status_summary_path, sep='\t')
    ensure_cols(type_sum, ['ASV', 'index'], "type_summary")
    ensure_cols(status_sum, ['ASV', 'index'], "status_summary")

    type_long = long_AB_for_group(type_sum.copy(), type_index)
    status_long = long_AB_for_group(status_sum.copy(), status_index)

    # ----- Build augmented node attribute tables -----
    # nfeat_type_df: join nf (GraphML_ID index) with type ISA via Taxon (ASV)
    nfeat_type = nf.reset_index().merge(
        type_long.set_index('ASV_ID'), left_on='Taxon', right_index=True, how='left'
    ).set_index('GraphML_ID')
    # color per type group index
    nfeat_type['type_name'] = nfeat_type['index'].map(type_index)
    nfeat_type['color'] = nfeat_type['type_name'].map(type_palette).fillna('lightgray')
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
    nfeat_status['status_name'] = nfeat_status['index'].map(status_index)
    nfeat_status['color'] = nfeat_status['status_name'].map(status_palette).fillna('lightgray')
    # add taxonomy to status table (for consistency)
    nfeat_status = nfeat_status.reset_index().merge(
        tax.reset_index(), left_on='Taxon', right_on='ASV_ID', how='left'
    ).set_index('GraphML_ID')

    # abundance table with taxonomy
    nfeat_abund = nf.reset_index().merge(mean_abund, left_on='Taxon', right_on='ASV_ID', how='left').set_index('GraphML_ID')
    nfeat_abund = nfeat_abund.reset_index().merge(
        tax.reset_index(), left_on='Taxon', right_on='ASV_ID', how='left'
    ).set_index('GraphML_ID')

    # Build Phylum palette from type table (prefer real categories)
    phyla = pd.concat([nfeat_type['Phylum'], nfeat_abund['Phylum']]).dropna().unique().tolist()
    phylum_palette = {p: c for p, c in zip(phyla, sns.color_palette('tab20', len(phyla)))}

    keep_cols = [
        'Taxon', 'Degree', 'Betweenness', 'Closeness', 'EigenCentral',
        'A', 'B', 'AxB', 'color', 'venn_color', 'Phylum', 'mean'
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

    # Attach attributes (each plot function can use what it needs)
    add_node_attrs_from_df(G_all, nfeat_type, keep_cols)
    add_node_attrs_from_df(G_all, nfeat_status, keep_cols)
    add_node_attrs_from_df(G_all, nfeat_abund, keep_cols)

    add_node_attrs_from_df(G_sub, nfeat_type, keep_cols)
    add_node_attrs_from_df(G_sub, nfeat_status, keep_cols)
    add_node_attrs_from_df(G_sub, nfeat_abund, keep_cols)

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
            "abundance_sub",
            "type_isa", "type_isa_labeled",
            "type_venn", "type_venn_labeled",
            "status_isa", "status_isa_labeled",
            "phylum_abund", "phylum_isa"
        }

    # Degree (all edges; weighted widths)
    if "degree_all" in modes:
        out = os.path.join(args.outdir, "network_degree_POS_ALL.svg")
        plot_degree(G_all, pos_all, out, degree_scale=args.degree_scale,
                    edge_width_scale=args.edge_width_scale)

    # Degree (thresholded subgraph)
    if "degree_sub" in modes:
        out = os.path.join(args.outdir, "network_degree_POS_SUB.svg")
        plot_degree(G_sub, pos_sub, out, degree_scale=args.degree_scale,
                    edge_width_scale=args.edge_width_scale)

    # Abundance (thresholded subgraph)
    if "abundance_sub" in modes:
        out = os.path.join(args.outdir, "network_abundance.svg")
        plot_abundance(G_sub, pos_sub, out)

    # Type ISA (color by type ISA, size by AxB)
    if "type_isa" in modes:
        out = os.path.join(args.outdir, "network_type_ISA.svg")
        plot_type_isa(G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=False)
    if "type_isa_labeled" in modes:
        out = os.path.join(args.outdir, "network_type_ISA_LABELED.svg")
        plot_type_isa(G_sub, pos_sub, out, type_palette, isa_scale=args.isa_scale, label=True)

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
        out = os.path.join(args.outdir, "network_phylum_ABUND.svg")
        plot_phylum(G_sub, pos_sub, out, phylum_palette, size_attr='mean', size_label='Abundance', size_scale=1.0)

    if "phylum_isa" in modes:
        out = os.path.join(args.outdir, "network_phylum_ISA.svg")
        plot_phylum(G_sub, pos_sub, out, phylum_palette, size_attr='AxB', size_label='ISA', size_scale=args.isa_scale)

    ok("All done.")


if __name__ == "__main__":
    main()
