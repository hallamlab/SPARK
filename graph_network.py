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


# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams.update({'font.size': 12})  # Set your desired size
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback
pd.set_option('display.max_columns', None)
font_path = '/home/ryan/.fonts/MYRIADPRO-REGULAR.OTF'  # update to your path
myriad_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = myriad_font.get_name()
sns.set_theme()  # re-applies style with updated rcParams



Family_palette = {'UBA1547': (0.07294117647058829, 0.2799999999999998, 0.4235294117647058),
                  'Thalassarchaeaceae': (0.21126696832579195, 0.4409049773755658, 0.7440271493212669),
                  'Veillonellaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Bacteroidaceae': (0.8823529411764707, 0.43790849673202614, 0.0),
                  'Streptococcaceae': (0.10352941176470587, 0.3764705882352941, 0.10352941176470587),
                  'Neisseriaceae_563222': (0.262234504540071, 0.666987761547572, 0.1824240031583102),
                  'Porphyromonadaceae': (0.5035294117647059, 0.09176470588235297, 0.0941176470588235),
                  'Fusobacteriaceae_993521': (0.9529411764705883, 0.018151260504201617, 0.0),
                  'Fusobacteriaceae': (0.9529411764705883, 0.018151260504201617, 0.0),
                  'Pasteurellaceae': (0.3498327037236912, 0.20800863464651914, 0.4790501888828926),
                  'Gemellaceae': (0.47655809431210483, 0.3177053962080699, 0.5975887214389888),
                  'UBA5272': (0.32941176470588246, 0.2023529411764706, 0.1764705882352941),
                  'Aerococcaceae': (0.5217292700212616, 0.32669029057406096, 0.2876824946846208),
                  'Leptotrichiaceae': (0.675121951219512, 0.13899569583931148, 0.5113055954088948),
                  'Actinomycetaceae': (0.902164124909223, 0.10724763979665919, 0.4496732026143783),
                  'Micrococcaceae': (0.2988235294117647, 0.2988235294117647, 0.2988235294117647),
                  'Megasphaeraceae': (0.468235294117647, 0.468235294117647, 0.468235294117647),
                  'Flavobacteriaceae': (0.44235294117647067, 0.4447058823529411, 0.08000000000000002),
                  'Atopobiaceae': (0.6437647058823528, 0.6437647058823529, 0.20329411764705874),
                  'Peptoniphilaceae': (0.05411764705882355, 0.4470588235294116, 0.4870588235294117),
                  'Campylobacteraceae': (0.19248206599713058, 0.6366714490674316, 0.718106169296987),
                  'Lachnospiraceae': (0.14588235294117657, 0.5599999999999996, 0.8470588235294116),
                  'Selenomonadaceae_42771': (0.9303619909502263, 0.9518552036199095, 0.9802262443438914),
                  'CAG-508': (0.9999999999999999, 0.6100951916036124, 0.26588235294117657),
                  'Tannerellaceae': (0.9999999999999999, 0.8814814814814815, 0.7647058823529415),
                  'Weeksellaceae': (0.20705882352941174, 0.7529411764705882, 0.20705882352941174),
                  'Burkholderiaceae_A_595425': (0.7918041847611526, 0.9353178049743387, 0.7635057244374257),
                  'Paludibacteraceae': (0.8752290165077887, 0.315359218786329, 0.3185584747733087),
                  'Peptostreptococcaceae_256921': (1.0, 0.9076750700280113, 0.9058823529411766),
                  'Mycobacteriaceae': (0.6928008634646518, 0.5636049649217484, 0.8105126821370751),
                  'Treponemataceae': (0.9187943607194943, 0.8893923189110353, 0.9411959163830821),
                  'Anaerovoracaceae': (0.6552667578659371, 0.4070588235294119, 0.356497948016416),
                  'Dialisteraceae': (0.8645216158752658, 0.7726718639262935, 0.754301913536499),
                  'Enterococcaceae': (0.9365279770444763, 0.6917073170731707, 0.861721664275466),
                  'Pseudomonadaceae': (1.0, 1.0, 1.0),
                  'Cardiobacteriaceae': (0.5976470588235294, 0.5976470588235294, 0.5976470588235294),
                  'Staskawiczbacteraceae': (0.936470588235294, 0.936470588235294, 0.936470588235294),
                  'Bifidobacteriaceae': (0.8508045370614616, 0.8550672645739912, 0.19434450013189108),
                  'Aminobacteriaceae': (0.9265882352941175, 0.9265882352941175, 0.7675294117647058),
                  'Tenuifilaceae': (0.1741176470588235, 0.8404092071611251, 0.908235294117647),
                  'Sphingomonadaceae': (0.8589765662362504, 0.946207556193209, 0.9621999043519848),
                  'Erysipelotrichaceae': (0.07294117647058829, 0.2799999999999998, 0.4235294117647058),
                  'Nanosynbacteraceae': (0.21126696832579195, 0.4409049773755658, 0.7440271493212669),
                  'Propionibacteriaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Ignavibacteriaceae_785640': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Coprothermobacteraceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Petrotogaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Absconditicoccaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Nanopelagicaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'UBA660': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Fastidiosipilaceae': (0.6329411764705882, 0.2967732487185746, 0.0)
                  }



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


data_dir = "/home/ryan/Projects/UBC/LMP/SPARK_data/"

node_features_file = os.path.join(data_dir, "vsearch_output/spieceasi/node_features.csv")
nfeatures_df = pd.read_csv(node_features_file, header=0, sep=',', index_col=0)
isa_type_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv")
isatype_df = pd.read_csv(isa_type_file, header=0, sep='\t', index_col=0)
isa_status_file = os.path.join(data_dir, "vsearch_output/indicspecies/status_indicator_species_results.tsv")
isastatus_df = pd.read_csv(isa_status_file, header=0, sep='\t', index_col=0)
isa_summary_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_summary.tsv")
isasummary_df = pd.read_csv(isa_summary_file, header=0, sep='\t', index_col=0)

taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)
#mito = "d__Bacteria; p__Pseudomonadota; c__Alphaproteobacteria; o__Rickettsiales; f__Mitochondria; g__; s__"
#tax_df = tax_df.loc[((tax_df['Confidence'] >= 0.7) & (tax_df['Taxonomy'] != mito))]
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

summary_stack_df = isasummary_df.stack().reset_index()
summary_stack_df.columns = ['ASV_ID', 'metric', 'value']

stat_list = ['A.BAL', 'A.Lung Brush', 'A.Oral Rinse', 'B.BAL', 'B.Lung Brush', 'B.Oral Rinse']
stat_dfs = []
for s in stat_list:
    s0 = s.split('.', 1)[0]
    s1 = s.split('.', 1)[1]
    sub_df = summary_stack_df.loc[summary_stack_df['metric'] == s]
    sub_df['stat'] = [x.split('.', 1)[0] for x in sub_df['metric']]
    sub_df['type'] = [x.split('.', 1)[1] for x in sub_df['metric']]
    sub_df = sub_df[['ASV_ID', 'stat', 'type', 'value']]
    sub_df.columns = ['ASV_ID', 'metric', 'type', 'value']
    stat_dfs.append(sub_df)
stat_df = pd.concat(stat_dfs)
stat_piv_df = stat_df.pivot_table(index=['ASV_ID', 'type'], columns='metric', values='value').reset_index()

asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_final.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)

metadata_table_path = os.path.join(data_dir, 'ref_db/spark_metadata.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df.set_index('sample', inplace=True)
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]

# Compute relative abundance
df_rel = compute_relative_abundance(asv_df)
df_rel.to_csv(os.path.join(data_dir, f"vsearch_output/indicspecies/rel_abundance.tsv"), sep='\t')

df_group_mean = df_rel.groupby(metadata_df['Type_Group'], axis=1).mean()
mean_stack_df = df_group_mean.stack().reset_index()
mean_stack_df.columns = ['ASV_ID', 'type', 'mean']
mean_stack_df = mean_stack_df.loc[~mean_stack_df['type'].isin(['Skin Brush', 'Scope Flush'])]
stat_mean_df = mean_stack_df.merge(stat_piv_df, left_on=['ASV_ID', 'type'], right_on=['ASV_ID', 'type'])
stat_mean_df['AxB'] = stat_mean_df['A'] * stat_mean_df['B'] 

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
stat_thresh = 0.0

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

isatype_df = isatype_df.merge(tax_df, left_index=True, right_on="Feature ID")
isastatus_df = isastatus_df.merge(tax_df, left_index=True, right_on="Feature ID")
#isastatus_df['status_color'] = [Family_palette[i] if s else 'lightgray' for i,s in zip(isastatus_df['Family'], isastatus_df['significance'])]
#isatype_df['type_color'] = [Family_palette[i] if s else 'lightgray' for i,s in zip(isatype_df['Family'], isatype_df['significance'])]
asv_type_sig = {a:(type_index[i] if s else None) for a,s,i in 
                zip(isatype_df.index.values, isatype_df['significance'], isatype_df['index'])
                }

nfeat_type_df = nfeatures_df.merge(isatype_df, left_on='Taxon', right_index=True)
nfeat_status_df = nfeatures_df.merge(isastatus_df, left_on='Taxon', right_index=True)
nfeat_groups_df = nfeatures_df.reset_index().merge(stat_mean_df[['ASV_ID', 'type', 'mean', 'AxB']],
                                                   left_on='Taxon', right_on='ASV_ID'
                                                   )
nfeat_groups_df = nfeat_groups_df.merge(clr_stack_df[['ASV_ID', 'type', 'clr']],
                                        left_on=['ASV_ID', 'type'],
                                        right_on=['ASV_ID', 'type']
                                        ).set_index('GraphML_ID')
grp_colors = []
for a,t in zip(nfeat_groups_df['ASV_ID'], nfeat_groups_df['type']):
    if a in asv_type_sig:
        if t == asv_type_sig[a]:
            grp_colors.append(type_palette[t])
        else:
            grp_colors.append('white')
    else:
        grp_colors.append('white')
nfeat_groups_df['color'] = grp_colors


keep_cols = ['Taxon', 'Degree',
             'Betweenness', 'Closeness',
             'EigenCentral','stat',
             'p.value', 'log_p',
             'significance', 'color',
             'Type_Group',
             'mean', 'clr',
             'AxB', 'status_color',
             'type_color', 'Family'
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

plt.figure(figsize=(18, 18))
# Loop through nodes to apply custom alpha
for node in G.nodes:
    color = G.nodes[node].get('color', 'lightgray')
    size = G.nodes[node].get('Degree', 1) * 10
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








'''
# Stretch the layout (e.g., 2x wider)
scale = 3.0
pos = {node: (x * scale, y * scale) for node, (x, y) in pos.items()}

plt.figure(figsize=(18, 18))
for node in G.nodes:
    color = G.nodes[node].get('type_color', 'lightgray')
    size = G.nodes[node].get('Degree', 1) * 20
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
#nx.draw_networkx_labels(G, pos, font_size=6)

# Build color legend
type_color = nx.get_node_attributes(G, 'type_color')
type_family = nx.get_node_attributes(G, 'Family')  # Or whatever attribute holds family name

# Unique colors + labels
unique_colors = {}
for node, color in type_color.items():
    label = type_family[node]
    unique_colors[color] = label

color_patches = [mpatches.Patch(color=c, label=l) for c, l in unique_colors.items()]

# Build size legend
size_legend = [1, 5, 10, 25, 50]  # Example degree values
size_handles = [plt.scatter([], [], s=s * 20, edgecolors='black',
                            facecolors='gray', alpha=1, label=f'Degree: {s}')
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

plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based in ISA for Sample Type")
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_type_plot_Family.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_type_plot_Familys.pdf"), bbox_inches='tight')
'''





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
    size = G.nodes[node].get('Degree', 1) * 10
    alpha = 0.5 if color == 'lightgray' else 1.0
    edgecolor = 'white' if color == 'lightgray' else 'lightgray'

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
    node_colors = {}
    norm_cnts = nx.get_node_attributes(G, 'AxB')

    for node in G.nodes:
        if node in norm_cnts:
            norm = norm_cnts[node]
            blended = np.clip(
                np.array(base_color) * norm + np.array([1, 1, 1]) * (1 - norm),
                0, 1
            )
            node_colors[node] = blended
        else:
            node_colors[node] = 'white'


    node_sizes = {}
    for node in G.nodes:
        if node in norm_cnts:
            relabund = G.nodes[node].get('AxB')
            node_sizes[node] = relabund * 5e2
        else:
            node_sizes[node] = 0

    plt.figure(figsize=(18, 18))

    for node in G.nodes:
        #color = node_colors[node]
        color = G.nodes[node].get('color', 'white')
        size = node_sizes[node]

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color=[color],
            node_size=[size],
            edgecolors='black',
            linewidths=0.25,
            alpha=1.0
        )
    nx.draw_networkx_edges(G, pos,
                           connectionstyle='arc3,rad=0.2',
                           edge_color='lightgray',
                           alpha=1)

    # >>> STOP matplotlib from rescaling everything <<<
    plt.axis('equal')         # Keep proportions
    plt.xlim(auto=False)      # Freeze x-axis scaling
    plt.ylim(auto=False)      # Freeze y-axis scaling

    plt.title("SPIEC-EASI Co-Occurrence Network\nNode color based in ISA for {t}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_{t_str}_plot.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"vsearch_output/spieceasi/network_{t_str}_plot.pdf"), bbox_inches='tight')



