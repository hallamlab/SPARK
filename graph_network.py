import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib as mpl

# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback


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


data_dir = "/home/ryan/Projects/UBC/LMP/SPARK_data/"

node_features_file = os.path.join(data_dir, "vsearch_output/spieceasi/node_features.csv")
nfeatures_df = pd.read_csv(node_features_file, header=0, sep=',', index_col=0)
isa_type_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv")
isatype_df = pd.read_csv(isa_type_file, header=0, sep='\t', index_col=0)
isa_status_file = os.path.join(data_dir, "vsearch_output/indicspecies/status_indicator_species_results.tsv")
isastatus_df = pd.read_csv(isa_status_file, header=0, sep='\t', index_col=0)

asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [x.rsplit('_', 1)[0] for x in asv_df.columns]

metadata_table_path = os.path.join(data_dir, 'ref_db/spark_metadata.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df.set_index('sample', inplace=True)
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]

# Compute relative abundance
df_rel = compute_relative_abundance(asv_df)
df_group_mean = df_rel.groupby(metadata_df['Type_Group'], axis=1).mean()
mean_stack_df = df_group_mean.stack().reset_index()
mean_stack_df.columns = ['ASV_ID', 'type', 'mean']
mean_stack_df = mean_stack_df.loc[~mean_stack_df['type'].isin(['Skin Brush', 'Scope Flush'])]

# CLR transform on group means
df_group_clr = np.log(df_group_mean + 1e-6).sub(
    np.log(df_group_mean + 1e-6).mean(axis=0), axis=1
    )
clr_stack_df = df_group_clr.stack().reset_index()
clr_stack_df.columns = ['ASV_ID', 'type', 'clr']
clr_stack_df = clr_stack_df.loc[~clr_stack_df['type'].isin(['Skin Brush', 'Scope Flush'])]

status_index = {1: 'Cancer', 2: 'Non-Cancer'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026'}
p_thresh = 0.05
stat_thresh = 0

isastatus_df = isastatus_df.loc[isastatus_df['index'].isin(status_index.keys())]
# Compute log-transformed p-values
isastatus_df['log_p'] = -np.log10(isastatus_df['p.value'])
# Define colors based on thresholds
isastatus_df['significance'] = False  # Default color for non-significant
isastatus_df.loc[((isastatus_df['p.value'] < p_thresh) & (isastatus_df['stat'] > stat_thresh)), 'significance'] = True 
isastatus_df['color'] = [status_palette[status_index[i]] if s else 'lightgray'
                       for i,s in zip(isastatus_df['index'],
                       isastatus_df['significance'])
                       ]

type_index = {1: 'BAL', 2: 'Lung Brush', 3: 'Oral Rinse'}
type_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A'
           }

isatype_df = isatype_df.loc[isatype_df['index'].isin(type_index.keys())]
# Compute log-transformed p-values
isatype_df['log_p'] = -np.log10(isatype_df['p.value'])
# Define colors based on thresholds
isatype_df['significance'] = False  # Default color for non-significant
isatype_df.loc[((isatype_df['p.value'] < p_thresh) & (isatype_df['stat'] > stat_thresh)), 'significance'] = True 
isatype_df['color'] = [type_palette[type_index[i]] if s else 'lightgray'
                       for i,s in zip(isatype_df['index'],
                       isatype_df['significance'])
                       ]

nfeat_type_df = nfeatures_df.merge(isatype_df, left_on='Taxon', right_index=True)
nfeat_status_df = nfeatures_df.merge(isastatus_df, left_on='Taxon', right_index=True)

nfeat_groups_df = nfeatures_df.reset_index().merge(mean_stack_df[['ASV_ID', 'type', 'mean']],
                                                   left_on='Taxon', right_on='ASV_ID'
                                                   )
nfeat_groups_df = nfeat_groups_df.merge(clr_stack_df[['ASV_ID', 'type', 'clr']],
                                        left_on=['ASV_ID', 'type'],
                                        right_on=['ASV_ID', 'type']
                                        ).set_index('GraphML_ID')

keep_cols = ['Taxon', 'Degree',
             'Betweenness', 'Closeness',
             'EigenCentral','stat',
             'p.value', 'log_p',
             'significance', 'color',
             'Type_Group',
             'mean', 'clr' 
             #'count',
             #'log_count', 'norm_count',
             #'arcsinh_count'
             ]


network_file = os.path.join(data_dir, "vsearch_output/spieceasi/network.graphml")
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

# Map group to color
node_colors = nx.get_node_attributes(G, 'color').values()

# Choose node size by abundance (optional)
stat = list(nx.get_node_attributes(G, 'Degree').values())
node_sizes = [x * 20 for x in stat]
plt.figure(figsize=(18, 18))

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       #node_size=75,
                       edgecolors='black',
                       linewidths=0.25,
                       alpha=0.5)
nx.draw_networkx_edges(G, pos,
					   connectionstyle='arc3,rad=0.2',
					   edge_color='lightgray',
					   alpha=0.5)
#nx.draw_networkx_labels(G, pos, font_size=6)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based in ISA for Sample Type")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_type_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_type_plot.pdf"), bbox_inches='tight')

# Add metadata to graph nodes
for node in G.nodes:
    if node in nfeat_status_df.index:
        for col in nfeat_status_df.columns:
            if col in keep_cols:
                G.nodes[node][col] = nfeat_status_df.loc[node, col]

node_colors = nx.get_node_attributes(G, 'color').values()
edgecolors = ['white' if c == 'lightgray' else c for c in nfeat_type_df['color']]

# Choose node size by abundance (optional)
stat = list(nx.get_node_attributes(G, 'Degree').values())
node_sizes = [x * 20 for x in stat]
plt.figure(figsize=(18, 18))

nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       node_size=node_sizes,
                       #node_size=75,
                       edgecolors=edgecolors,
                       linewidths=3,
                       alpha=0.5)
nx.draw_networkx_edges(G, pos,
                       connectionstyle='arc3,rad=0.2',
                       edge_color='lightgray',
                       alpha=0.5)
#nx.draw_networkx_labels(G, pos, font_size=6)

# >>> STOP matplotlib from rescaling everything <<<
plt.axis('equal')         # Keep proportions
plt.xlim(auto=False)      # Freeze x-axis scaling
plt.ylim(auto=False)      # Freeze y-axis scaling

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based in ISA for status Status")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_status_plot.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_status_plot.pdf"), bbox_inches='tight')







for t in type_palette.keys():
    t_str = t.replace(' ', '_')
    nfeat_sub_df = nfeat_groups_df.loc[nfeat_groups_df['type'] == t]
    # Add metadata to graph nodes
    for node in G.nodes:
        if node in nfeat_sub_df.index:
            for col in nfeat_sub_df.columns:
                if col in keep_cols:
                    G.nodes[node][col] = nfeat_sub_df.loc[node, col]


    # Choose your main color
    base_color = mcolors.to_rgb(type_palette[t])
    node_colors = []
    norm_cnts = nx.get_node_attributes(G, 'mean')

    for node in G.nodes:
        if node in norm_cnts:
            norm = norm_cnts[node] * 50
            blended = np.clip(
                np.array(base_color) * norm + np.array([1, 1, 1]) * (1 - norm),
                0, 1
            )
            node_colors.append(blended)
        else:
            node_colors.append('white')


    node_sizes = []
    for node in G.nodes:
        if node in norm_cnts:
            relabund = G.nodes[node].get('mean')
            node_sizes.append(relabund * 5e4)
        else:
            node_sizes.append(0)

    plt.figure(figsize=(18, 18))

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors='black',
                           linewidths=0.25,
                           alpha=1)
    nx.draw_networkx_edges(G, pos,
                           connectionstyle='arc3,rad=0.2',
                           edge_color='lightgray',
                           alpha=0.5)

    # >>> STOP matplotlib from rescaling everything <<<
    plt.axis('equal')         # Keep proportions
    plt.xlim(auto=False)      # Freeze x-axis scaling
    plt.ylim(auto=False)      # Freeze y-axis scaling

    plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based in ISA for {t}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_{t_str}_plot.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_{t_str}_plot.pdf"), bbox_inches='tight')



