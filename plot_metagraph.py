#!/usr/bin/env python3
"""
Combine sample–ASV membership and per-sample co-occurrence networks into one meta-graph.

Nodes:
  - Samples (“sample”): sized by a global metric (e.g., mean_degree)
  - ASVs    (“asv”):    sized by a node-level metric (e.g., pagerank)

Edges:
  - membership     (sample ↔ ASV): weight = raw count
  - cooccurrence   (ASV ↔ ASV):    weight = summed Spiec-Easi weight

Writes a GraphML of the combined network.
"""
import os
import argparse
import pandas as pd
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import colorsys
from matplotlib import font_manager as fm, rcParams
from adjustText import adjust_text


# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams.update({'font.size': 12})  # Set your desired size
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback
pd.set_option('display.max_columns', None)
# Set font globally
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()  # re-applies style with updated rcParams
sns.set_style("white")


def split_taxa_string(taxa_str, delimiter=';'):
    tax_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    if taxa_str != 'Unassigned':
        parts = [part.strip().split('__', 1)[1] for part in taxa_str.split(delimiter)]
    else:
        parts = ['Unassigned']
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--counts",        required=True,
                   help="TSV: ASVs x samples counts (rows=ASVs, cols=samples)")
    p.add_argument("--global-stats",  required=True,
                   help="TSV: global network stats, index=sample (e.g. mean_degree column)")
    p.add_argument("--node-stats",    required=True,
                   help="TSV: node-level stats with a 'node' column (e.g. pagerank column)")
    p.add_argument("--networks-dir",  required=True,
                   help="Directory of per-sample .graphml co-occurrence networks (ASV–ASV)")
    p.add_argument("--output",        required=True,
                   help="Path to write combined GraphML")
    args = p.parse_args()

    data_dir = "/home/ryan/Projects/UBC/LMP/SPARK_data/"

    node_features_file = os.path.join(data_dir, "vsearch_output/spieceasi/node_features.csv")
    nfeatures_df = pd.read_csv(node_features_file, header=0, sep=',', index_col=0)
    isa_type_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv")
    isatype_df = pd.read_csv(isa_type_file, header=0, sep='\t', index_col=0).reset_index()
    isatype_df.rename(columns={'level_0': 'ASV_ID'}, inplace=True)
    isa_status_file = os.path.join(data_dir, "vsearch_output/indicspecies/status_indicator_species_results.tsv")
    isastatus_df = pd.read_csv(isa_status_file, header=0, sep='\t', index_col=0).reset_index()
    isastatus_df.rename(columns={'level_0': 'ASV_ID'}, inplace=True)
    type_summary_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_summary.tsv")
    type_summary_df = pd.read_csv(type_summary_file, header=0, sep='\t')
    type_summary_df.rename(columns={'ASV': 'ASV_ID'}, inplace=True)
    status_summary_file = os.path.join(data_dir, "vsearch_output/indicspecies/status_indicator_species_summary.tsv")
    status_type_summary_df = pd.read_csv(status_summary_file, header=0, sep='\t')
    status_type_summary_df.rename(columns={'ASV': 'ASV_ID'}, inplace=True)
    venn_df = pd.read_csv(os.path.join(data_dir, "vsearch_output/metadata/venn3_presence_table.tsv"), sep="\t", header=0)

    status_index = {1: 'Cancer',
                    2: 'Non-Cancer',
                    3: 'Cancer+Non-Cancer'
                    }
    convert_status = {'Cancer': 'Cancer',
                    'Control': 'Non-Cancer',
                     }
    status_palette = {'Non-Cancer':'white',
                    'Cancer':'#A50026',
                    'Cancer+Non-Cancer': 'lightgray'
                    }
    type_index = {1: 'BAL',
                2: 'Lung Brush',
                3: 'Oral Rinse',
                4: 'BAL+Lung Brush',
                5: 'BAL+Oral Rinse',
                6: 'Lung Brush+Oral Rinse',
                7: 'BAL+Lung Brush+Oral Rinse'
                }
    type_palette = {'Oral Rinse': '#6A3D9A',
                    'BAL+Oral Rinse': '#F19CBB',
                    'BAL': '#0072B2',
                    'BAL+Lung Brush': '#00FFFF',
                    'Lung Brush': '#009E73',
                    'Lung Brush+Oral Rinse': '#C1EAAD',
                    'BAL+Lung Brush+Oral Rinse': 'lightgray'
                    }
    venn_type = {'Only Oral Rinse': 'Oral Rinse',
                'Only BAL': 'BAL',
                'Only Lung Brush': 'Lung Brush',
                'Oral + BAL': 'BAL+Oral Rinse',
                'Oral + Lung': 'Lung Brush+Oral Rinse',
                'BAL + Lung': 'BAL+Lung Brush',
                'All Three': 'BAL+Lung Brush+Oral Rinse'
                }

    ts_long_df = pd.wide_to_long(
        type_summary_df,
        stubnames=['A','B'],
        i=['ASV_ID', 'index'],
        j='Group',
        sep='.',
        suffix='.*'
    ).reset_index(
        )[['ASV_ID', 'index', 'Group', 'A', 'B']]

    ts_long_df['tmp_grp'] = [type_index[x] for x in ts_long_df['index']]
    ts_long_df = ts_long_df.loc[ts_long_df['Group'] == ts_long_df['tmp_grp']]
    ts_long_df.drop(columns=['tmp_grp'], inplace=True)

    ss_long_df = pd.wide_to_long(
        status_type_summary_df,
        stubnames=['A','B'],
        i=['ASV_ID', 'index'],
        j='Group',
        sep='.',
        suffix='.*'
    ).reset_index(
        )[['ASV_ID', 'index', 'Group', 'A', 'B']]

    ss_long_df['tmp_grp'] = [status_index[x] for x in ss_long_df['index']]
    ss_long_df = ss_long_df.loc[ss_long_df['Group'] == ss_long_df['tmp_grp']]
    ss_long_df.drop(columns=['tmp_grp'], inplace=True)

    asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_final.micro.tsv')
    asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
    asv_stack_df = asv_df.stack().reset_index()
    asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
    mean_stack_df = asv_stack_df.groupby(['ASV_ID'])['count'].mean().reset_index()
    mean_stack_df.columns = ['ASV_ID', 'mean']
    mean_stack_df['mean'] = np.ceil(mean_stack_df['mean'])

    metadata_table_path = os.path.join(data_dir, 'vsearch_output/metadata/metadata_updated.tsv')
    metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
    metadata_df.set_index('sample', inplace=True)
    metadata_df['status'] = ['Non-Cancer' if x == 'Non-Cancer' else x for x in metadata_df['Case']]
    patient_dict = {x: y for x, y in zip(metadata_df['patient_code'], metadata_df['status'])}
    isastatus_df = isastatus_df
    isastatus_df['color'] = [status_palette[status_index[x]] if x in status_index else 'lightgray' for x in isastatus_df['index']]
    isastatus_df = isastatus_df.merge(ss_long_df, how='left', on=['ASV_ID', 'index']).set_index('ASV_ID')
    isastatus_df['AxB'] = isastatus_df['A'] * isastatus_df['B']
    isastatus_df['AxB'] = isastatus_df['AxB'].fillna(0)
    isatype_df = isatype_df
    isatype_df['color'] = [type_palette[type_index[x]] if x in type_index else 'lightgray' for x in isatype_df['index']]
    isatype_df = isatype_df.merge(ts_long_df, how='left', on=['ASV_ID', 'index']).set_index('ASV_ID')
    isatype_df['AxB'] = isatype_df['A'] * isatype_df['B']
    isatype_df['AxB'] = isatype_df['AxB'].fillna(0)

    venn_df = venn_df.set_index('ASV_ID')
    isatype_df = isatype_df.join(venn_df, how='left')
    venn_colors = []
    for g in isatype_df['grouping']:
        if g in venn_type:
            g_t = venn_type[g]
            c = type_palette[g_t]
        else:
            c = 'lightgray'
        venn_colors.append(c)
    isatype_df['venn_color'] = venn_colors

    taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
    tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
    tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
    tax_df.set_index('Feature ID', inplace=True)
    taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                    'Order': [], 'Family': [], 'Genus': [],
                    'Species': []
                    }
    for t in tax_df['Taxon']:
        lineage = split_taxa_string(t)
        for l in lineage:
            v = lineage[l]
            taxonomy_dict[l].append(v)
    for t in taxonomy_dict:
        tax_df[t] = taxonomy_dict[t]

    isatype_df = isatype_df.merge(tax_df, left_index=True, right_on="Feature ID")
    isastatus_df = isastatus_df.merge(tax_df, left_index=True, right_on="Feature ID")

    # List of unique Phyla in your data
    phyla = isatype_df['Phylum'].unique()
    # Generate color palette (qualitative)
    palette = sns.color_palette('tab20', len(phyla))
    # Map phylum to color
    phylum_color_dict = dict(zip(phyla, palette))

    # ── Load inputs ───────────────────────────────────────────────────────────────
    counts       = pd.read_csv(args.counts, sep="\t", index_col=0).T
    global_stats = pd.read_csv(args.global_stats, sep="\t", index_col=0)
    node_stats   = pd.read_csv(args.node_stats, sep="\t")

    G = nx.Graph()

    # ── Add sample nodes ─────────────────────────────────────────────────────────
    for sample, stats in global_stats.iterrows():
        size  = float(stats.get("mean_degree", 1.0)) * 200
        color = stats.get("color", "red")
        G.add_node(sample,
                   node_type="patient",
                   size=size,
                   color=color)

    # ── Add ASV nodes ────────────────────────────────────────────────────────────
    for _, row in node_stats.iterrows():
        asv   = str(row["ASV_ID"])
        size  = float(row.get("pagerank", 1.0)) * 10000
        color = row.get("color", "skyblue")
        G.add_node(asv,
                   node_type="ASV",
                   size=size,
                   color=color)

    # ── Add membership edges (sample ↔ ASV) ──────────────────────────────────────
    for sample in counts.index:
        if sample not in G:
            continue
        for asv, cnt in counts.loc[sample].items():
            if cnt > 0 and asv in G:
                G.add_edge(sample, asv,
                           edge_type="membership",
                           weight=float(cnt))

    # ── Merge in co-occurrence edges from each per-sample GraphML ───────────────
    for fname in os.listdir(args.networks_dir):
        if not fname.lower().endswith(".graphml"):
            continue
        sample = os.path.splitext(fname)[0]
        path   = os.path.join(args.networks_dir, fname)
        H      = nx.read_graphml(path)
        # H should contain only ASV–ASV edges
        for u, v, data in H.edges(data=True):
            w = float(data.get("weight", 1.0))
            if G.has_edge(u, v):
                # sum weights if edge already exists
                G[u][v]["weight"]    = G[u][v].get("weight", 0.0) + w
                G[u][v]["edge_type"] = "cooccurrence"
            else:
                G.add_edge(u, v,
                           edge_type="cooccurrence",
                           weight=w)

    # ── Write combined GraphML ───────────────────────────────────────────────────
    nx.write_graphml(G, args.output)
    print(f"Wrote combined graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges → {args.output}")
    
    G = nx.read_graphml(args.output)
    # === Visualization ===
    # Separate node lists
    patient_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'patient']
    asv_nodes     = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'ASV']

    # Compute layout once (you can tweak the graph passed here)
    pos = nx.spring_layout(G, seed=42)

    # Node sizes & colors
    patient_sizes = [G.nodes[n]['size'] / 200 for n in patient_nodes]
    asv_sizes     = [G.nodes[n]['size'] * 0.1 for n in asv_nodes]  # 10× smaller

    patient_colors = [status_palette[convert_status[patient_dict[n.split('_', 1)[0]]]]
                      for n in patient_nodes]
    asv_colors     = [G.nodes[n]['color'] for n in asv_nodes]

    # Only keep edges where both ends are patients
    patient_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if G.nodes[u].get('node_type') == 'patient'
        and G.nodes[v].get('node_type') == 'patient'
    ]

    plt.figure(figsize=(12, 12))
    # Draw ASV nodes first (so they sit under patients)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=asv_nodes,
        node_size=asv_sizes,
        node_color=asv_colors,
        alpha=0.6
    )
    # Draw patient nodes on top
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=patient_nodes,
        node_size=patient_sizes,
        node_color=patient_colors,
        edgecolors='black',
        linewidths=1.0
    )
    # Draw only patient–patient edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=patient_edges,
        width=2.0,
        edge_color='gray',
        alpha=0.7
    )
    # Label just the patients
    labels = {n: n.split('_', 1)[0] for n in patient_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=6,
        font_weight='bold'
    )

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_metagraph.svg", bbox_inches='tight')
    plt.savefig("/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi_multi/patient_metagraph.pdf", bbox_inches='tight')



if __name__ == "__main__":
    main()

