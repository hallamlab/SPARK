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



def compute_relative_abundance(count_df):
    """
    Computes relative abundance per sample.

    Args:
        count_df (pd.DataFrame): ASV count data.

    Returns:
        pd.DataFrame: Relative abundance data.
    """
    try:
        relative_abundance = count_df.div(count_df.sum(axis=1), axis=0)
        print("Computed relative abundance per sample.")
        return relative_abundance
    except Exception as e:
        print(f"Error computing relative abundance: {e}")
        sys.exit(1)

def clr_transform(rel_abundance_df):
    """
    Applies Center-Log Ratio (CLR) transformation to the relative abundance data.

    Args:
        rel_abundance_df (pd.DataFrame): ASV relative abundance data.

    Returns:
        pd.DataFrame: CLR-transformed data.
    """
    # Replace zeros with a small pseudocount to avoid log(0)
    pseudocount = 1e-6
    rel_abundance_nonzero = rel_abundance_df.replace(0, pseudocount)
    
    # Calculate geometric mean for each sample
    geom_mean = rel_abundance_nonzero.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
    
    # Apply CLR: log(x / geometric mean)
    clr_df = np.log(rel_abundance_nonzero.div(geom_mean, axis=0))
    
    print("Applied CLR transformation to relative abundance data.")
    return clr_df

def split_taxa_string(taxa_str, delimiter=';'):
    """
    Split a taxonomic string into the 7 standard levels.
    
    Parameters:
        taxa_str (str): The taxonomic string, e.g.
            "k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Enterobacterales; f__Enterobacteriaceae; g__Escherichia; s__coli"
        delimiter (str): The delimiter used in the string (default is semicolon).
    
    Returns:
        dict: A dictionary with keys 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'
              mapping to their respective values.
    """
    # Define the taxonomic levels in order
    tax_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    
    # Split the string by the delimiter and strip whitespace
    if taxa_str != 'Unassigned':
        parts = [part.strip().split('__', 1)[1] for part in taxa_str.split(delimiter)]
    else:
        parts = ['Unassigned']
    # In status there are missing levels, fill them with None
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict


data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'

node_features_file = os.path.join(data_dir, "spark_old_output/brush/spieceasi/node_features.csv")
nfeatures_df = pd.read_csv(node_features_file, header=0, sep=',', index_col=0)
isa_type_file = os.path.join(data_dir, "spark_old_output/brush/indicspecies/subclass2_indicator_species_results.tsv")
isatype_df = pd.read_csv(isa_type_file, header=0, sep='\t', index_col=0).reset_index()
isatype_df.rename(columns={'level_0': 'ASV_ID'}, inplace=True)
isa_status_file = os.path.join(data_dir, "spark_old_output/brush/indicspecies/status_indicator_species_results.tsv")
isastatus_df = pd.read_csv(isa_status_file, header=0, sep='\t', index_col=0).reset_index()
isastatus_df.rename(columns={'level_0': 'ASV_ID'}, inplace=True)
type_summary_file = os.path.join(data_dir, "spark_old_output/brush/indicspecies/subclass2_indicator_species_summary.tsv")
type_summary_df = pd.read_csv(type_summary_file, header=0, sep='\t')
type_summary_df.rename(columns={'ASV': 'ASV_ID'}, inplace=True)
status_summary_file = os.path.join(data_dir, "spark_old_output/brush/indicspecies/status_indicator_species_summary.tsv")
status_type_summary_df = pd.read_csv(status_summary_file, header=0, sep='\t')
status_type_summary_df.rename(columns={'ASV': 'ASV_ID'}, inplace=True)
venn_df = pd.read_csv(os.path.join(data_dir, "spark_old_output/brush/metadata/Three_types_venn_presence_table.tsv"), sep="\t", header=0)

status_index = {1: 'Cancer',
                2: 'Non-Cancer',
                3: 'Cancer+Non-Cancer'
                }
status_palette = {'Non-Cancer':'white',
                  'Cancer':'#A50026',
                  'Cancer+Non-Cancer': 'lightgray'
                  }
type_index = {1: 'ca-contra',
              2: 'ca-lung',
              3: 'ctrl-brush',
              4: 'ca-contra+ca-lung',
              5: 'ca-contra+ctrl-brush',
              6: 'ca-lung+ctrl-brush',
              7: 'ca-contra+ca-lung+ctrl-brush'
              }

type_palette = {'ctrl-brush': '#6A3D9A',
                'ca-contra+ctrl-brush': '#F19CBB',
                'ca-contra': '#0072B2',
                'ca-contra+ca-lung': '#00FFFF',
                'ca-lung': '#009E73',
                'ca-lung+ctrl-brush': '#C1EAAD',
                'ca-contra+ca-lung+ctrl-brush': 'lightgray'
                }

venn_type = {'ca-contra + ca-lung': 'ca-contra+ca-lung',
             'ctrl-brush + ca-contra': 'ca-contra+ctrl-brush',
             'ctrl-brush + ca-lung': 'ca-lung+ctrl-brush',
             'Only ca-contra': 'ca-contra',
             'Only ca-lung': 'ca-lung',
             'Only ctrl-brush': 'ctrl-brush',
             'All Three': 'ca-contra+ca-lung+ctrl-brush'
             }

ts_long_df = pd.wide_to_long(
    type_summary_df.loc[type_summary_df['index'].notna()],
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
    status_type_summary_df.loc[status_type_summary_df['index'].notna()],
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

asv_path = os.path.join(data_dir, 'spark_old_output/brush/ASVs/ASV_final.micro.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
mean_stack_df = asv_stack_df.groupby(['ASV_ID'])['count'].mean().reset_index()
mean_stack_df.columns = ['ASV_ID', 'mean']
mean_stack_df['mean'] = np.ceil(mean_stack_df['mean'])

metadata_table_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/metadata_updated.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df.set_index('sample', inplace=True)
metadata_df['status'] = ['Non-Cancer' if x == 'Non-Cancer' else x for x in metadata_df['Case']]

#p_thresh = 0.05
#stat_thresh = 0.0

isastatus_df = isastatus_df #.loc[isastatus_df['index'].isin(status_index.keys())]
# Compute log-transformed p-values
#isastatus_df['log_p'] = -np.log10(isastatus_df['p.value'])
# Define colors based on thresholds
#isastatus_df['significance'] = False  # Default color for non-significant
#isastatus_df.loc[((isastatus_df['p.value'] < p_thresh) & (isastatus_df['stat'] > stat_thresh)), 'significance'] = True 
#status_colors = []
#for i,s in zip(isastatus_df['index'], isastatus_df['significance']):
#    #i_v = status_index[i]
#    if s == True: #) & (i_v != 'All')):
#        c = status_palette[status_index[i]]
#    #elif i_v == 'All':
#    #    c = 'lightgray'
#    else:
#        c = 'lightgray'
#    status_colors.append(c)
#isastatus_df['color'] = status_colors

isastatus_df['color'] = [status_palette[status_index[x]] if x in status_index else 'lightgray' for x in isastatus_df['index']]
isastatus_df = isastatus_df.merge(ss_long_df, how='left', on=['ASV_ID', 'index']).set_index('ASV_ID')
isastatus_df['IndVal'] = np.sqrt(isastatus_df['A'] * isastatus_df['B'])
isastatus_df['IndVal'] = isastatus_df['IndVal'].fillna(0)
isatype_df = isatype_df #.loc[isatype_df['index'].isin(type_index.keys())]
# Compute log-transformed p-values
#isatype_df['log_p'] = -np.log10(isatype_df['p.value'])
# Define colors based on thresholds
#isatype_df['significance'] = False  # Default color for non-significant
#isatype_df.loc[((isatype_df['p.value'] < p_thresh) & (isatype_df['stat'] > stat_thresh)), 'significance'] = True 
#type_colors = []
#for i,s in zip(isatype_df['index'], isatype_df['significance']):
#    #i_v = type_index[i]
#    if s == True: #) & (i_v != 'All')):
#        c = type_palette[type_index[i]]
#    #elif i_v == 'All':
#    #    c = '#999999'
#    else:
#        c = 'lightgray'
#    type_colors.append(c)
#isatype_df['color'] = type_colors

isatype_df['color'] = [type_palette[type_index[x]] if x in type_index else 'lightgray' for x in isatype_df['index']]
isatype_df = isatype_df.merge(ts_long_df, how='left', on=['ASV_ID', 'index']).set_index('ASV_ID')
isatype_df['IndVal'] = np.sqrt(isatype_df['A'] * isatype_df['B'])
isatype_df['IndVal'] = isatype_df['IndVal'].fillna(0)

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





taxonomy_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/taxonomy_updated.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['ASV_ID'] = [x.rsplit(';', 1)[0] for x in tax_df['ASV_ID']]
tax_df.set_index('ASV_ID', inplace=True)
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







isatype_df = isatype_df.merge(tax_df, left_index=True, right_on="ASV_ID")
isastatus_df = isastatus_df.merge(tax_df, left_index=True, right_on="ASV_ID")

'''
sub_status_df = isastatus_df[['index', 'significance']]
sub_status_df.columns = ['status_index', 'status_sig']
isatype_df = isatype_df.merge(sub_status_df, left_index=True, right_on="ASV_ID")
isatype_df['status_sig_color'] = [x if y else 'lightgray'
                                  for x,y in zip(isatype_df['color'],
                                                 isatype_df['status_sig']
                                                 )]
isatype_df['cancer_color'] = [x if y == 1
                              else 'lightgray' for x,y in
                              zip(isatype_df['status_sig_color'],
                                  isatype_df['status_index']
                                  )]
isatype_df['non-cancer_color'] = [x if y == 2
                                  else 'lightgray' for x,y in
                              zip(isatype_df['status_sig_color'],
                                  isatype_df['status_index']
                                  )]
isatype_df['all_status_color'] = [x if y == 3
                                  else 'lightgray' for x,y in
                              zip(isatype_df['color'],
                                  isatype_df['status_index']
                                  )]
'''

# List of unique Phyla in your data
phyla = isatype_df['Phylum'].unique()
# Generate color palette (qualitative)
palette = sns.color_palette('tab20', len(phyla))
# Map phylum to color
phylum_color_dict = dict(zip(phyla, palette))

nfeat_type_df = nfeatures_df.merge(isatype_df, left_on='Taxon', right_index=True)
nfeat_status_df = nfeatures_df.merge(isastatus_df, left_on='Taxon', right_index=True)

nfeat_type_df.to_csv(os.path.join(data_dir, "spark_old_output/brush/spieceasi/node_features.type.tsv"),
                     sep='\t'
                     )

nfeat_status_df.to_csv(os.path.join(data_dir, "spark_old_output/brush/spieceasi/node_features.status.tsv"),
                     sep='\t'
                     )

nfeat_abund_df = nfeatures_df.reset_index().merge(mean_stack_df[['ASV_ID', 'mean']],
                                        left_on='Taxon', right_on='ASV_ID',
                                        how='left'
                                        ).set_index('GraphML_ID')
nfeat_abund_df = nfeat_abund_df.reset_index().merge(tax_df, left_on='Taxon',
                                                    right_on="ASV_ID", how='left'
                                                    ).set_index('GraphML_ID')



keep_cols = ['Taxon', 'Degree',
             'Betweenness', 'Closeness',
             'EigenCentral','stat',
             'p.value', 'log_p',
             'significance', 'color',
             'subclass2',
             'mean',
             'IndVal', 'status_color',
             'type_color', 'Phylum',
             'cancer_color',
             'status_sig_color',
             'non-cancer_color',
             'all_status_color',
             'venn_color'
             ]


network_file = os.path.join(data_dir, "spark_old_output/brush/spieceasi/network_pos_all.graphml")
G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

# — separate edges by sign and collect absolute weights for width scaling —
pos_edges, pos_w = [], []
neg_edges, neg_w = [], []
for u, v, data in G.edges(data=True):
    w = data.get("weight", 0)
    if w > 0:
        pos_edges.append((u, v));    pos_w.append(w)
    elif w < 0:
        neg_edges.append((u, v));    neg_w.append(abs(w))

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = 'black'
    size = (G.nodes[node].get('Degree', 1) + 1) * 80
    alpha = 0.5

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       edgelist=pos_edges,
                       width=[w * 5 for w in pos_w],
                       edge_color="lightgray",
                       alpha=0.6)

# Build size legend
size_legend = [0, 1, 3, 5, 10]
size_handles = [plt.scatter([], [], s=(s + 1) * 80, edgecolors='black',
                            facecolors='gray', alpha=1, label=f'{s}')
                for s in size_legend]

plt.legend(
    handles=size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Degree",
    frameon=False,
    scatterpoints=1,
    labelspacing=1.5
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode size based on Degree\nEdge are all positive correlations")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_degree_plot_POS_ALL.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_degree_plot_POS_ALL.pdf"), bbox_inches='tight')

network_file = os.path.join(data_dir, "spark_old_output/brush/spieceasi/network_pos_sub.graphml")
G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

# — separate edges by sign and collect absolute weights for width scaling —
pos_edges, pos_w = [], []
neg_edges, neg_w = [], []
for u, v, data in G.edges(data=True):
    w = data.get("weight", 0)
    if w > 0:
        pos_edges.append((u, v));    pos_w.append(w)
    elif w < 0:
        neg_edges.append((u, v));    neg_w.append(abs(w))

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = 'black'
    size = (G.nodes[node].get('Degree', 1) + 1) * 80
    alpha = 0.5

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       edgelist=pos_edges,
                       width=[w * 5 for w in pos_w],
                       edge_color="lightgray",
                       alpha=0.6)

# Build size legend
size_legend = [0, 1, 3, 5, 10]
size_handles = [plt.scatter([], [], s=(s + 1) * 80, edgecolors='black',
                            facecolors='gray', alpha=1, label=f'{s}')
                for s in size_legend]

plt.legend(
    handles=size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Degree",
    frameon=False,
    scatterpoints=1,
    labelspacing=1.5
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode size based on Degree\nEdge are positive correlations >= 0.1")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_degree_plot_POS_SUB.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_degree_plot_POS_SUB.pdf"), bbox_inches='tight')

G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_abund_df.index:
        for col in nfeat_abund_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_abund_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = 'black'
    size = G.nodes[node].get('mean', 1)
    alpha = 0.5

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

# Build size legend
size_legend = [1, 10, 100, 500, 1000]
size_handles = [plt.scatter([], [], s=s, edgecolors='black',
                            facecolors='gray', alpha=1, label=f'{s}')
                for s in size_legend]

plt.legend(
    handles=size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="ASV Mean Abundance",
    frameon=False,
    scatterpoints=1,
    labelspacing=1.5
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode size based on ASV Mean Abundance")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_abundance_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_abundance_plot.pdf"), bbox_inches='tight')

G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = G.nodes[node].get('color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
					   connectionstyle='arc3,rad=0.2',
					   edge_color='lightgray',
					   alpha=1)

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=type)
    for type, color in type_palette.items()
    ]

# Build size legend
#size_legend = [0.1, 0.25, 0.50, 0.75, 1.0]
#size_handles = [plt.scatter([], [], s=s * 500, edgecolors='black',
#                            facecolors='gray', alpha=1, label=f'IndVal: {s}')
#                for s in size_legend]

size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]


plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Sample Type\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot.pdf"), bbox_inches='tight')

'''
G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
gets_label = []
for node in G.nodes:
    color = G.nodes[node].get('color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0)  ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    if color != 'lightgray':
        gets_label.append(node)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
					   connectionstyle='arc3,rad=0.2',
					   edge_color='lightgray',
					   alpha=1)

# add labels
texts = []
for n in gets_label:
    x, y = pos[n]
    label = G.nodes[n].get('Taxon', "")
    texts.append(
        plt.text(
            x, y, label,
            fontsize=9,
            weight='bold',
            ha='center', va='center'
        )
    )
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    expand_text=(1.2, 1.2),
    force_text=0.5,
    force_points=0.2
)

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=type)
    for type, color in type_palette.items()
    ]

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Sample Type\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_LABELED.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_LABELED.pdf"), bbox_inches='tight')
'''
'''
G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
gets_label = []
for node in G.nodes:
    color = G.nodes[node].get('venn_color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    if color != 'lightgray':
        gets_label.append(node)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

# add labels
texts = []
for n in gets_label:
    x, y = pos[n]
    label = G.nodes[n].get('Taxon', "")
    texts.append(
        plt.text(
            x, y, label,
            fontsize=9,
            weight='bold',
            ha='center', va='center'
        )
    )
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    expand_text=(1.2, 1.2),
    force_text=0.5,
    force_points=0.2
)

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=type)
    for type, color in type_palette.items()
    ]

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Venn Diagram Grouping\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_venn_plot_LABELED.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_venn_plot_LABELED.pdf"), bbox_inches='tight')
'''

G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# === Visualization ===
pos = nx.spring_layout(G, seed=42)

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = G.nodes[node].get('venn_color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=type)
    for type, color in type_palette.items()
    ]

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Venn Diagram Grouping\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_venn_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_venn_plot.pdf"), bbox_inches='tight')

'''
G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_status_df.index:
        for col in nfeat_status_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_status_df.loc[node, col]

edgecolors = ['white' if c == 'lightgray' else c for c in nfeat_type_df['color']]

plt.figure(figsize=(18, 18))
gets_label = []
for node in G.nodes:
    color = G.nodes[node].get('color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    edgecolor = 'white' if color == 'lightgray' else 'lightgray'
    edgecolor = 'black' if color == 'white' else 'lightgray'
    lw = 1 if color == 'white' else 0.25
    if color != 'lightgray':
        gets_label.append(node)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=lw,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

legend_handles = []
for status, color in status_palette.items():
    if status == "Non-Cancer":
        # thicker edge on this one
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=1,      # <-- bold
            label=status
        )
    else:
        # normal edge on the others
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor=color,
            linewidth=0.25,      # <-- default
            label=status
        )
    legend_handles.append(patch)

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

# add labels
texts = []
for n in gets_label:
    x, y = pos[n]
    label = G.nodes[n].get('Taxon', "")
    texts.append(
        plt.text(
            x, y, label,
            fontsize=9,
            weight='bold',
            ha='center', va='center'
        )
    )
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
    expand_text=(1.2, 1.2),
    force_text=0.5,
    force_points=0.2
)
plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on ISA for Cancer Status\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_status_plot_LABELED.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_status_plot_LABELED.pdf"), bbox_inches='tight')
'''

G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_status_df.index:
        for col in nfeat_status_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_status_df.loc[node, col]

edgecolors = ['white' if c == 'lightgray' else c for c in nfeat_type_df['color']]

plt.figure(figsize=(18, 18))
for node in G.nodes:
    color = G.nodes[node].get('color', 'lightgray')
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0
    edgecolor = 'white' if color == 'lightgray' else 'lightgray'
    edgecolor = 'black' if color == 'white' else 'lightgray'
    lw = 1 if color == 'white' else 0.25
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=lw,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

legend_handles = []
for status, color in status_palette.items():
    if status == "Non-Cancer":
        # thicker edge on this one
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=1,      # <-- bold
            label=status
        )
    else:
        # normal edge on the others
        patch = mpatches.Patch(
            facecolor=color,
            edgecolor=color,
            linewidth=0.25,      # <-- default
            label=status
        )
    legend_handles.append(patch)

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

plt.legend(
    handles=legend_handles + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on ISA for Cancer Status\nNode size based on Indicator Species Strength")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_status_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_status_plot.pdf"), bbox_inches='tight')


G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_abund_df.index:
        for col in nfeat_abund_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_abund_df.loc[node, col]

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
for node in G.nodes:
    p = G.nodes[node].get('Phylum')
    if p in phylum_color_dict:
        color = phylum_color_dict[p]
    else:
        color = 'lightgray'
    size = G.nodes[node].get('mean', 1)
    alpha = 0.5 if color == 'lightgray' else 1.0

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

# Build color legend
type_family = nx.get_node_attributes(G, 'Phylum')

# Unique colors + labels
unique_colors = {}
for node, phylum in type_family.items():
    color = phylum_color_dict[phylum]
    label = type_family[node]
    unique_colors[color] = label

color_patches = [mpatches.Patch(color=c, label=l) for c, l in unique_colors.items()]

# Build size legend
size_legend = [1, 10, 100, 500, 1000]
size_handles = [plt.scatter([], [], s=s, edgecolors='black',
                            facecolors='gray', alpha=1, label=f'Abundance: {s}')
                for s in size_legend]

plt.legend(
    handles=color_patches + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Phylum \nNode size based on Mean ASV Abundance")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_Phylum_ABUND.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_Phylum_ABUND.pdf"), bbox_inches='tight')



G = nx.read_graphml(network_file)
# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_type_df.index:
        for col in nfeat_type_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_type_df.loc[node, col]

# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
for node in G.nodes:
    p = G.nodes[node].get('Phylum')
    if p in phylum_color_dict:
        color = phylum_color_dict[p]
    else:
        color = 'lightgray'
    size = (G.nodes[node].get('IndVal', 0) ** 2) * 500
    alpha = 0.5 if color == 'lightgray' else 1.0

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=[color],
        node_size=[size],
        edgecolors='black',
        linewidths=0.25,
        alpha=alpha
    )

nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=1)

# Build color legend
type_family = nx.get_node_attributes(G, 'Phylum')

# Unique colors + labels
unique_colors = {}
for node, phylum in type_family.items():
    color = phylum_color_dict[phylum]
    label = type_family[node]
    unique_colors[color] = label

color_patches = [mpatches.Patch(color=c, label=l) for c, l in unique_colors.items()]

# Build size legend
size_legend = [0.49, 0.69, 1.0]
size_handles = [plt.scatter([], [], s=(s ** 2) * 500, edgecolors='black',
                            facecolors='gray', alpha=1, label=l)
                for s,l in zip(size_legend, ['Weak (<0.5)', 'Moderate (0.5-0.7)', 'Strong (>0.7)'])]

plt.legend(
    handles=color_patches + size_handles,
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title="Node Attributes",
    frameon=False,
    scatterpoints=1,     # Don't stack points
    labelspacing=1.5     # Increase vertical space between entries
)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based on Phylum \nNode size based on Mean ASV Abundance")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_Phylum_ISA.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/spieceasi/network_type_plot_Phylum_ISA.pdf"), bbox_inches='tight')