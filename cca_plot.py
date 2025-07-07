# cca_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.stats import spearmanr
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from skbio.diversity.alpha import shannon
import umap
import umap.plot
import scanpy as sc
from scipy.stats import kruskal
from sklearn.preprocessing import LabelEncoder
from matplotlib_venn import venn2, venn2_circles

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

def indicator_analysis(data, metadata, group_col):
    features = data.columns
    samples = data.index

    groups = metadata.loc[samples, group_col]
    grouped = groups.groupby(groups).groups  # dict: group -> sample indices

    results = []
    for feature in features:
        vals = [data.loc[grouped[g], feature].dropna() for g in grouped]
        if any(len(v) < 2 for v in vals):  # skip poorly sampled features
            continue
        try:
            stat, p = kruskal(*vals)
        except ValueError:
            continue
        means = [v.mean() for v in vals]
        max_group = list(grouped.keys())[np.argmax(means)]
        results.append({
            "feature": feature,
            "stat": stat,
            "p_value": p,
            "max_group": max_group,
            "mean_diff": np.ptp(means)
        })

    return pd.DataFrame(results).sort_values("p_value")

def gini(x):
    x = np.abs(x)
    if np.sum(x) == 0:
        return 0
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * x)) / (n * np.sum(x)) - (n + 1) / n

def hill_number(x):
    x = np.abs(x)
    if np.sum(x) == 0:
        return 0
    p = x / x.sum()
    return 1 / np.sum(p ** 2)

def compute_metrics(df):
    results = pd.DataFrame(index=df.index)
    results["gini"] = df.apply(gini, axis=1)
    results["variance"] = df.var(axis=1)
    results["hill_number_q2"] = df.apply(hill_number, axis=1)
    return results

def sanitize_columns(df, chars_to_replace):
    pattern = f"[{re.escape(''.join(chars_to_replace))}]"
    df.columns = df.columns.str.replace(pattern, "_", regex=True)
    df.columns = df.columns.str.replace("__", "_")
    df.columns = df.columns.str.replace("__", "_")
    
    return df

def save_umap_plots(reducer, output_base_path):
    """
    Generates and saves UMAP plots as static images using matplotlib.

    Args:
        reducer (umap.UMAP): Fitted UMAP reducer.
        output_base_path (str): Base file path for saving plots (without extension).
    """
    # Original UMAP scatter plot
    plt.figure(figsize=(12, 10))
    umap.plot.points(reducer)
    plt.title("UMAP Scatter Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_scatter.png", dpi=600)
    plt.close()
    print(f"Saved UMAP scatter plot to {output_base_path}_scatter.png")

    # Connectivity plot
    plt.figure(figsize=(12, 10))
    umap.plot.connectivity(reducer, show_points=False)
    plt.title("UMAP Connectivity Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity.png", dpi=600)
    plt.close()
    print(f"Saved UMAP connectivity plot to {output_base_path}_connectivity.png")



    plt.figure(figsize=(12, 10))
    # Generate the connectivity plot
    ax = umap.plot.connectivity(reducer, edge_bundling='hammer', show_points=False, edge_cmap='gray_r')

    # Get current limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Dynamically calculate new limits with padding
    x_padding = (x_max - x_min) * 0.05  # 5% padding
    y_padding = (y_max - y_min) * 0.05  # 5% padding
    new_x_min = x_min - x_padding
    new_x_max = x_max + x_padding
    new_y_min = y_min - y_padding
    new_y_max = y_max + y_padding

    # Update the axis limits
    ax.set_xlim(new_x_min, new_x_max)
    ax.set_ylim(new_y_min, new_y_max)

    # Add title and save the plot
    plt.title("UMAP Connectivity Plot with Edge Bundling and Points")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity_bundled.png", dpi=600)
    plt.close()
    print(f"Saved bundled connectivity plot with points to {output_base_path}_connectivity_bundled.png")

def perform_umap(
    data: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    random_state: int = 42,
    precomputed: bool = False
):
    """
    Performs UMAP dimensionality reduction on the data or on a precomputed distance matrix.

    Args:
        data (pd.DataFrame): 
            - If precomputed=False, rows are samples × features.
            - If precomputed=True, must be a square (samples × samples) distance matrix.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        metric (str): Distance metric to use (ignored if precomputed=True).
        random_state (int): Random state for reproducibility.
        precomputed (bool): If True, treat `data` as a distance matrix and set metric='precomputed'.

    Returns:
        umap.UMAP: Fitted UMAP reducer.
        pd.DataFrame: DataFrame with UMAP embeddings (UMAP1, UMAP2).
    """
    umap_metric = 'precomputed' if precomputed else metric

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=random_state
    )

    # For precomputed, pass the matrix values directly
    input_array = data.values if precomputed else data

    embedding = reducer.fit_transform(input_array)
    umap_df = pd.DataFrame(
        embedding,
        index=data.index,
        columns=["UMAP1", "UMAP2"]
    )
    print(f"Performed UMAP (precomputed={precomputed}, metric='{umap_metric}'). "
          f"Embedding shape: {umap_df.shape}")
    return reducer, umap_df

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

def plot_cca_biplot(species_scores, env_scores, sub_species_scores, sub_env_scores,
                    output_path, meta, label=False, sub_plot=False, dpi_setting=600
                    ):
    """
    Generates and saves a CCA biplot using Seaborn, coloring ASVs by their cluster.
    Cluster '-1' is colored grey to represent 'Other'.
    
    Args:
        species_scores (pd.DataFrame): DataFrame containing 'CCA1', 'CCA2', and 'Cluster' columns for ASVs.
        env_scores (pd.DataFrame): DataFrame containing environmental variables with 'CCA1' and 'CCA2' as columns.
        output_path (str, optional): Path to save the plot. Defaults to 'vsearch_new_output/cca/cca_biplot.png'.
        dpi_setting (int, optional): Resolution of the saved plot. Defaults to 300.
    """
    species_scale = 1  # increase to spread further
    arrow_scale = 1.25 # increase to make arrows longer
    if sub_plot:
        ss_df = sub_species_scores.copy()
        es_df = sub_env_scores.copy()
    else:
        ss_df = species_scores.copy()
        es_df = env_scores.copy()

    ss_df['CCA1'] *= species_scale
    ss_df['CCA2'] *= species_scale

    # Ensure 'Cluster' column is of type string for consistent handling
    if meta == 'Cluster':
        species_scores[meta] = species_scores[meta].astype(int)
    
    # Step 1: Define Unique Clusters
    unique_clusters = species_scores[meta].unique()
    
    # Step 2: Create Color Palette
    # Exclude '-1' from the list of clusters to assign unique colors
    clusters_except_minus1 = [cluster for cluster in unique_clusters if cluster != -1]
    
    # Number of clusters excluding '-1'
    num_clusters = len(clusters_except_minus1)
    
    # Generate a color palette for the clusters excluding '-1'
    # You can choose other palettes like "tab20", "Set2", etc., based on the number of clusters
    palette = sns.color_palette("hsv", num_clusters)
    
    # Create a palette dictionary
    palette_dict = {cluster: color for cluster, color in zip(clusters_except_minus1, palette)}
    
    # Assign grey color to cluster '-1'
    palette_dict[-1] = (0.9, 0.9, 0.9)  # RGB tuple for grey
    
    # Step 3: Initialize the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Step 4: Create Seaborn Scatter Plot
    sns.scatterplot(
        data=ss_df,
        x='CCA1',
        y='CCA2',
        hue=meta,
        palette=palette_dict,
        legend='full',
        alpha=0.7,
        edgecolor='gray',
        linewidth=0.5,
        s=50  # Adjust the size as needed
    )
    
    # Step 5: Plot Environmental Variables as Vectors
    ax = plt.gca()  # Get current axes

    for var in es_df.index:
        c1 = es_df.loc[var, 'CCA1'] * arrow_scale
        c2 = es_df.loc[var, 'CCA2'] * arrow_scale
        print(var)
        print(c1)
        print(c2)
        ax.arrow(
            0, 0, c1, c2,
            color='red',
            width=0.005,
            head_width=0.005,
            length_includes_head=True,
            alpha=0.5
        )
        if label:
            ax.text(
                c1 * 1.1, c2 * 1.1,
                var,
                color='red',
                ha='right',
                va='center',
                fontsize=8,
                fontweight='regular'
            )
    
    # Step 6: Customize Plot
    plt.xlabel('CCA1', fontsize=14)
    plt.ylabel('CCA2', fontsize=14)
    plt.title('CCA Biplot of ASVs and Environmental Variables', fontsize=16)
    
    # Show legend outside the plot
    plt.legend(
        title=meta,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
        fontsize=12
    )
    if label:
        plt.legend().remove()  # Remove legend for environmental variables
    
    # Show grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Step 7: Adjust Plot Limits to Accommodate Arrows and Labels
    #all_x = np.concatenate([species_scores['CCA1'], env_scores['CCA1']])
    #all_y = np.concatenate([species_scores['CCA2'], env_scores['CCA2']])
    #buffer_x = (all_x.max() - all_x.min()) * 0.3
    #buffer_y = (all_y.max() - all_y.min()) * 0.3
    #plt.xlim(all_x.min() - buffer_x, all_x.max() + buffer_x)
    #plt.ylim(all_y.min() - buffer_y, all_y.max() + buffer_y)
    plt.xlim(-1.1, 5)
    # Optimize layout
    plt.tight_layout()
    
    # Step 8: Save the Plot
    plt.savefig(output_path, dpi=dpi_setting)
    plt.clf()
    plt.close()
    
    print(f"Saved CCA biplot to {output_path}")


# Load the data
data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
output_dir = os.path.join(data_dir, "vsearch_output")

site_scores = pd.read_csv(os.path.join(output_dir, 'cca/cca_site_scores.tsv'),
                          sep='\t', index_col=0)
species_scores = pd.read_csv(os.path.join(output_dir, 'cca/cca_species_scores.tsv'),
                             sep='\t', index_col=0)
env_scores = pd.read_csv(os.path.join(output_dir, 'cca/cca_env_scores.tsv'),
                         sep='\t', index_col=0)
envfit_taxa = pd.read_csv(os.path.join(output_dir, 'cca/cca_envfit_taxa.tsv'),
                         sep='\t', index_col=0)
envfit_env = pd.read_csv(os.path.join(output_dir, 'cca/cca_envfit_env.tsv'),
                         sep='\t', index_col=0)
taxa_env_str = pd.read_csv(os.path.join(output_dir, 'cca/cca_taxa_env_strong.tsv'),
                         sep='\t', index_col=0)
#conlin_map = pd.read_csv(os.path.join(output_dir, 'cca/collinearity_mapping.tsv'),
#                         sep='\t', index_col=0)

env_scores.index = [x.replace('.', '_').replace('__', '_').replace('__', '_')
                    for x in list(env_scores.index)
                    ]
envfit_env.index = [x.replace('.', '_').replace('__', '_').replace('__', '_')
                    for x in list(envfit_env.index)
                    ]

tax_df = pd.read_csv(os.path.join(output_dir, 'taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv'),
                            sep='\t', index_col=0)
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

species_scores = species_scores.join(tax_df)

metadata_path = os.path.join(data_dir, "ref_db/spark_metadata.tsv")
metadata_df = pd.read_csv(metadata_path, sep="\t", header=0)
asv_filter_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/metadata/ASV_meta.tsv'), sep="\t", header=0)

# Not subset the matrices based on the r^2 and p-value thresholds
envfit_taxa = envfit_taxa.loc[(round(envfit_taxa['r2'], 2) >= 0.05) & (round(envfit_taxa['p_value'], 2) <= 0.05)]
envfit_env = envfit_env.loc[(round(envfit_env['r2'], 2) >= 0.05) & (round(envfit_env['p_value'], 2) <= 0.05)]
taxa_env_str = taxa_env_str.loc[round(taxa_env_str['cos_theta'], 2).abs() >= 0.70]
sub_species_scores = species_scores.loc[species_scores.index.isin(envfit_taxa.index)]
sub_env_scores = env_scores.loc[env_scores.index.isin(envfit_env.index)]

plot_cca_biplot(species_scores, env_scores, sub_species_scores, sub_env_scores,
                os.path.join(output_dir, 'cca/cca_biplot_Phylum.png'),
                'Phylum')
plot_cca_biplot(species_scores, env_scores, sub_species_scores, sub_env_scores,
                os.path.join(output_dir, 'cca/cca_biplot_Phylum_filtered.png'),
                'Phylum', sub_plot=True)
plot_cca_biplot(species_scores, env_scores, sub_species_scores, sub_env_scores,
                os.path.join(output_dir, 'cca/cca_biplot_Phylum_filtered_labelled.png'),
                'Phylum', label=True, sub_plot=True)


sub_species_scores.to_csv(os.path.join(output_dir, 'cca/cca_species_scores_filtered.tsv'),
                         sep='\t')
sub_env_scores.to_csv(os.path.join(output_dir, 'cca/cca_env_scores_filtered.tsv'),
                      sep='\t')




# run more basic correlation analysis and HCA

# Load your tables
asv_path = os.path.join(output_dir, "ASVs/ASV_filtered.decon.tsv")
asv_df = pd.read_csv(asv_path, sep="\t", index_col=0)
voc_path = "/home/ryan/Projects/UBC/LMP/SPARK_data/ref_db/VOC_table.tsv"
voc_df = pd.read_csv(voc_path, sep="\t", index_col=2)
#voc_nonorm_path = "/home/ryan/Projects/UBC/LMP/SPARK_data/ref_db/VOC_nonorm_table.tsv"
#voc_nonorm_df = pd.read_csv(voc_nonorm_path, sep="\t", index_col=2)

network_path = os.path.join(output_dir, "spieceasi/spieceasi_asv_transformed_distance.tsv")
network_df = pd.read_csv(network_path, sep="\t", index_col=0)

asv_df.columns = [x.rsplit('_', 2)[0] for x in asv_df.columns]
#asv_df = asv_df[asv_df.index.isin(network_df.index)]
asv_df = asv_df[asv_df.index.isin(asv_filter_df['ASV_ID'])]

asv_df = asv_df.T

cols_trim = ["VOC_1", "Undecane_144", "1-Propanol_23", "VOC_595",
            "Dimethyl sulfone_358", "1-Octanol_377", "2-Butanone_31",
            "Nonane, 3-methyl-_365", "Oxetane, 2-ethyl-3-methyl-_451",
            "Dodecane_152", "Butanal_30", "VOC_149", "1-Octanol_140",
            "Acetoin_774", "Acetone_14", "2(3H)-Furanone, dihydro-5-methyl-_441",
            "Carbamic acid, monoammonium salt_2", "3-Heptanone_347",
            "VOC_900", "Benzene_50", "Heptane, 2,4-dimethyl-_86",
            "1,3,5-Trifluorobenzene_37", "Heptane, 2,2,4,6,6-pentamethyl-_439",
            "Decane_123", "Decane, 1,1'-oxybis-_243", "Octane_81",
            "Undecane, 2-methyl-_241", "Decane, 2,6,7-trimethyl-_372",
            "Nonane, 2-methyl-_363", "1-Butanol_53", "Levomenthol_384",
            "Hexane, 2,5-dimethyl-_319", "VOC_3", "Octanoic acid_242",
            "Butanal, 3-methyl-_417", "Dodecane, 2,7,1VOC-trimethyl-_165",
            "Heptane, 2,2,4,6,6-pentamethyl-_118", "Decane, 4-methyl-_376",
            "Nonadecane_207", "1-Octene_224", "VOC_599", "Acetic acid, methyl ester_19",
            "2-Butenedioic acid (Z)-, monododecyl ester_248", "Heptane, 3-ethyl-2-methyl-_434",
            "Decane_378", "Isopropyl myristate_284", "Dodecane, 2,6,11-trimethyl-_250",
            "Decane, 2-methyl-_137", "2-Propanol_15", "Methanesulfonic anhydride_22",
            "Hexane, 2-methyl-_310", "1,2-Ethanediol, monoacetate_228",
            "Heptane_56", "Octane, 4-methyl-_93", "VOC_679", "(2-Aziridinylethyl)amine_4",
            "1,2-Benzenedicarboxylic acid, bis(2-methylpropyl) ester_968",
            "Benzene, 1-ethyl-3-methyl-_606", "Butanoic acid, 4-hydroxy-_355",
            "Methyl propionate_41", "Ethanol_12", "1-Heptene_61", "2-Pentanone_52",
            "Acetic acid_27", "Acetic acid, butyl ester_84"
            ]
voc_tmp_df = voc_df.copy()
voc_df = voc_df[cols_trim]
voc_df = sanitize_columns(voc_df, chars_to_replace=["-", ".", " ", ",", "(", ")"])
'''
voc_batch = voc_nonorm_df['sample_family']
voc_nonorm_df = voc_nonorm_df[[x for x in voc_nonorm_df.columns if x in cols_trim]]
voc_nonorm_df = sanitize_columns(voc_nonorm_df, chars_to_replace=["-", ".", " ", ",", "(", ")"])
voc_nonorm_df = voc_nonorm_df.loc[voc_batch.index]

# Create AnnData object
adata = sc.AnnData(voc_nonorm_df)
adata.obs['batch'] = voc_batch
# Run ComBat
sc.pp.combat(adata, key='batch')
# Get corrected data as DataFrame
corrected = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

pos_corr = corrected + abs(corrected.min())
'''

status_index = {1: 'Cancer',
                2: 'Non-Cancer',
                3: 'Cancer+Non-Cancer'
                }
status_palette = {'Non-Cancer':'white',
                  'Cancer':'#A50026',
                  'Cancer+Non-Cancer': '#d27f85cc'
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

status_summary_file = os.path.join(data_dir, "vsearch_output/cca/status_indicator_lineage_summary.tsv")
status_type_summary_df = pd.read_csv(status_summary_file, header=0, sep='\t')
status_type_summary_df.rename(columns={'ASV': 'lineage'}, inplace=True)
isa_status_file = os.path.join(data_dir, "vsearch_output/cca/status_indicator_lineage_results.tsv")
isastatus_df = pd.read_csv(isa_status_file, header=0, sep='\t', index_col=0).reset_index()
isastatus_df.rename(columns={'level_0': 'lineage'}, inplace=True)

'''
isa_type_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv")
isatype_df = pd.read_csv(isa_type_file, header=0, sep='\t', index_col=0).reset_index()
isatype_df.rename(columns={'level_0': 'ASV_ID'}, inplace=True)
type_summary_file = os.path.join(data_dir, "vsearch_output/indicspecies/Type_Group_indicator_species_summary.tsv")
type_summary_df = pd.read_csv(type_summary_file, header=0, sep='\t')
type_summary_df.rename(columns={'ASV': 'ASV_ID'}, inplace=True)
venn_df = pd.read_csv(os.path.join(data_dir, "vsearch_output/metadata/venn3_presence_table.tsv"), sep="\t", header=0)
'''

ss_long_df = pd.wide_to_long(
    status_type_summary_df,
    stubnames=['A','B'],
    i=['lineage', 'index'],
    j='Group',
    sep='.',
    suffix='.*'
).reset_index(
    )[['lineage', 'index', 'Group', 'A', 'B']]
ss_long_df = ss_long_df[ss_long_df['index'].notna()]
ss_long_df['tmp_grp'] = [status_index[x] for x in ss_long_df['index']]
ss_long_df = ss_long_df.loc[ss_long_df['Group'] == ss_long_df['tmp_grp']]
ss_long_df.drop(columns=['tmp_grp'], inplace=True)

isastatus_df['color'] = [status_palette[status_index[x]] if x in status_index else 'lightgray' for x in isastatus_df['index']]
isastatus_df = isastatus_df.merge(ss_long_df, how='left', on=['lineage', 'index']).set_index('lineage')
isastatus_df['AxB'] = isastatus_df['A'] * isastatus_df['B']
isastatus_df['AxB'] = isastatus_df['AxB'].fillna(0)
'''
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
'''

# Align on common samples
common_samples = asv_df.index.intersection(voc_df.index)
asv_df = asv_df.loc[common_samples]
asv_df.reset_index().to_csv(os.path.join(output_dir, "cca/Brush_ASV_Table.tsv"), sep='\t', index=False)
flurp
voc_df = voc_df.loc[common_samples]
#voc_nonorm_df = voc_nonorm_df.loc[common_samples]

# Align common ASVs
#common_asvs = asv_df.columns.intersection(isastatus_df.index)
#asv_df = asv_df[common_asvs]
#isastatus_df = isastatus_df.loc[isastatus_df.index.isin(common_asvs)]

# Align common ASVs
#common_asvs = asv_df.columns.intersection(isatype_df.index)
#asv_df = asv_df[common_asvs]
#isatype_df = isatype_df.loc[isatype_df.index.isin(common_asvs)]

print(f"ASV matrix shape: {asv_df.shape}")
print(f"VOC matrix shape: {voc_df.shape}")
print(f"ISA status matrix shape: {isastatus_df.shape}")
#print(f"ISA type matrix shape: {isatype_df.shape}")

asv_stack_df = asv_df.unstack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
asv_tax_df = asv_stack_df.merge(tax_df, left_on = 'ASV_ID', right_index=True, how='left')

order_df = asv_tax_df.groupby(['Phylum', 'Class', 'Order', 'Family', 'Genus', 'sample']
                              )['count'].sum().reset_index()

order_df['lineage'] = order_df['Order'] + '_' + \
                       order_df['Family'] + '_' + \
                       order_df['Genus']

ord_us_df = order_df.pivot(index='sample', columns='lineage', values='count')
ord_us_df.T.reset_index().to_csv(os.path.join(output_dir, "cca/Lineage_ASV_Table.tsv"), sep='\t', index=False)



asv_stack_df = order_df[['lineage', 'sample', 'count']]
asv_stack_df.columns = ['LIN_ID', 'sample', 'count']


asv_status_df = asv_stack_df.merge(metadata_df[['Sample_renamed', 'Case']], left_on='sample',
                                   right_on='Sample_renamed', how='left'
                                   )
asv_grp_df = asv_status_df.groupby(['LIN_ID', 'Case'])['count'].sum().reset_index()
asv_unstack_df = asv_grp_df.pivot(index='LIN_ID', columns='Case', values=['count']).reset_index()
asv_unstack_df.columns = ['LIN_ID', 'Cancer', 'Control']
# Boolean presence
print(asv_unstack_df.head())

case1_present = asv_unstack_df["Cancer"] > 0
case2_present = asv_unstack_df["Control"] > 0

# Counts
only_case1 = ((case1_present) & (~case2_present)).sum()
only_case2 = ((case2_present) & (~case1_present)).sum()
shared      = ((case1_present) & (case2_present)).sum()

print(f"Only in Cancer: {only_case1}")
print(f"Only in Control: {only_case2}")
print(f"Shared: {shared}")

cancer_set = set(asv_grp_df.loc[(asv_grp_df['Case'] == 'Cancer') & (asv_grp_df['count'] > 0)]['LIN_ID'])
control_set = set(asv_grp_df.loc[(asv_grp_df['Case'] == 'Control') & (asv_grp_df['count'] > 0)]['LIN_ID'])

cancer_sub = len(set(cancer_set - control_set))
control_sub = len(set(control_set - cancer_set))
shared_sub = len(cancer_set.intersection(control_set))

subsets=(cancer_sub, control_sub, shared_sub)

plt.figure(figsize=(6, 6))

v2 = venn2(
    subsets=subsets,
    set_labels=("Cancer", "Non-Cancer"),
    set_colors=(status_palette['Cancer'], status_palette['Non-Cancer']),
    alpha=0.6
)
# Get the circle objects
circles = venn2_circles(subsets=subsets ,linewidth=1, color='grey')

plt.savefig(os.path.join(data_dir, "vsearch_output/cca/venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/cca/venn_diagram.pdf"), format="pdf", bbox_inches="tight")

# Result matrix: ASVs x Chem
corr_matrix = pd.DataFrame(index=ord_us_df.columns, columns=voc_df.columns, dtype=float)
for asv in ord_us_df.columns:
    if ord_us_df[asv].nunique() <= 1:
        continue
    for voc in voc_df.columns:
        if voc_df[voc].nunique() <= 1:
            continue
        r, _ = spearmanr(ord_us_df[asv], voc_df[voc])
        corr_matrix.loc[asv, voc] = r

# Drop rows/cols with all NaNs
corr_matrix = corr_matrix.dropna(how="all").dropna(axis=1, how="all")

# Fill remaining NaNs with 0 (or any neutral value)
corr_matrix = corr_matrix.fillna(0.0)

'''
print(isastatus_df.head())
print(isatype_df.head())
'''
# Each column is a category with color-mapped values
row_colors = isastatus_df.loc[corr_matrix.index, ["Group"]].copy()
row_colors["ISA_Cancer_Status"] = row_colors["Group"].map(status_palette)

'''
row_colors["ISA_Sample_Type"] = isatype_df.loc[corr_matrix.index, "Group"].map(type_palette)
row_colors["Venn_Sample_MAP"] = isatype_df.loc[corr_matrix.index, "grouping"].map(venn_type)
row_colors["Venn_Sample_Type"] = row_colors["Venn_Sample_MAP"].map(type_palette)
row_colors.drop(columns=['Group', "Venn_Sample_MAP"], inplace=True)
'''

row_colors.drop(columns=['Group'], inplace=True)


# Blue-gray-orange diverging (colorblind-friendly)
cmap_cblind = mcolors.LinearSegmentedColormap.from_list(
    "cblind_diverging", ["#0072B2", "#F0F0F0", "#D55E00"]
)

output_path = os.path.join(output_dir, "cca/ISA_spearman_correlation.pdf")
g = sns.clustermap(
    corr_matrix.astype(float),
    cmap=cmap_cblind,
    center=0,
    metric="correlation",
    method="average",
    figsize=(30, 22),
    dendrogram_ratio=(0.1, 0.02),  # (rows, columns)
    cbar_pos=None,  # remove default colorbar
    colors_ratio=(0.01, 0.01),
    row_colors=row_colors,
    
    )

# Create separate colorbar on the right
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap_cblind, norm=norm)
sm.set_array([])

# Add colorbar manually
cbar_ax = g.fig.add_axes([0.8, 0.4, 0.05, 0.3])  # [x, y, width, height]
g.fig.colorbar(sm, cax=cbar_ax, label="Spearman correlation")

# Build legend handles
legend_handles = []

# Add ISA status legend
unique_values = isastatus_df['Group'].unique()
for value in unique_values:
    if value in status_palette:
        color = status_palette[value]
        legend_handles.append(Patch(color=color, label=f'ISA Cancer Status: {value}'))  
'''
# Add ISA type legend
unique_values = isatype_df['Group'].unique()
for value in unique_values:
    if value in type_palette:
        color = type_palette[value]
        legend_handles.append(Patch(color=color, label=f'ISA Type Status: {value}'))  
# Add Venn type legend
unique_values = isatype_df['grouping'].unique()
for value in unique_values:
    if value in venn_type:
        color = type_palette[venn_type[value]]
        legend_handles.append(Patch(color=color, label=f'Venn Type Status: {value}'))  
'''
# Add legend to right of the heatmap
g.ax_heatmap.legend(
    handles=legend_handles,
    loc='center',
    bbox_to_anchor=(1.6, 0.8),
    frameon=False,
    #prop={'size': 24},
)


plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", dpi=600)
plt.clf()
plt.close()


col_order = g.data2d.columns

# Result matrix: VOC x VOC
corr_matrix = pd.DataFrame(index=voc_df.columns, columns=voc_df.columns, dtype=float)
for v1 in voc_df.columns:
    if voc_df[v1].nunique() <= 1:
        continue
    for v2 in voc_df.columns:
        if voc_df[v2].nunique() <= 1:
            continue
        r, _ = spearmanr(voc_df[v1], voc_df[v2])
        corr_matrix.loc[v1, v2] = r

# Drop rows/cols with all NaNs
corr_matrix = corr_matrix.dropna(how="all").dropna(axis=1, how="all")

# Fill remaining NaNs with 0 (or any neutral value)
corr_matrix = corr_matrix.fillna(0.0)

# Blue-gray-orange diverging (colorblind-friendly)
cmap_cblind = mcolors.LinearSegmentedColormap.from_list(
    "cblind_diverging", ["#0072B2", "#F0F0F0", "#D55E00"]
)

corr_matrix = corr_matrix[col_order]

output_path = os.path.join(output_dir, "cca/voc_spearman_correlation.pdf")
g = sns.clustermap(
    corr_matrix.astype(float),
    cmap=cmap_cblind,
    center=0,
    metric="correlation",
    method="average",
    figsize=(24, 24),
    dendrogram_ratio=(0.1, 0.02),  # (rows, columns)
    cbar_pos=None,  # remove default colorbar
    colors_ratio=(0.01, 0.01),
    col_cluster=False    
    )

# Create separate colorbar on the right
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cmap_cblind, norm=norm)
sm.set_array([])

# Add colorbar manually
cbar_ax = g.fig.add_axes([1.05, 0.5, 0.05, 0.3])  # [x, y, width, height]
g.fig.colorbar(sm, cax=cbar_ax, label="Spearman correlation")

plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight", dpi=600)
plt.clf()
plt.close()

flurp

# Compute pairwise Bray-Curtis distances between samples
# Here, each row in asv_df is assumed to be a sample
voc_array = voc_df.values

distances = pdist(voc_array, metric="braycurtis")
bray_curtis_matrix = squareform(distances)
bray_df = pd.DataFrame(bray_curtis_matrix, index=voc_df.index, columns=voc_df.index)
bray_df.index.name = "sample"
output_bray = os.path.join(data_dir, 'vsearch_output/cca/voc_bray.tsv')
bray_df.to_csv(output_bray, sep="\t")
print(f"Bray-Curtis beta diversity saved to {output_bray}")

bray_df = pd.read_csv(output_bray, header=0, sep='\t', index_col=0)
bray_reducer, bray_umap = perform_umap(data=bray_df,
                                       random_state=42,
                                       precomputed=True
                                       )

# Beta diversity
plt.figure(figsize=(12, 10))
sns.scatterplot(data=bray_umap, x="UMAP1", y="UMAP2", s=50,
                alpha=0.7, edgecolor='gray', linewidth=0.5
                )

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC.pdf"))
plt.close()

sample_plot_path = os.path.join(data_dir, f"vsearch_output/cca/VOC_UMAP_")
save_umap_plots(bray_reducer, sample_plot_path)

# Merge metadata with UMAP results
bray_umap = bray_umap.merge(metadata_df, left_index=True, right_on='Sample_renamed', how='left')
# Beta diversity
plt.figure(figsize=(12, 10))
sns.scatterplot(data=bray_umap, x="UMAP1", y="UMAP2", s=50,
                hue="Case",
                alpha=0.7, edgecolor='gray', linewidth=0.5
                )

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_case.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_case.pdf"))
plt.close()

# Beta diversity
plt.figure(figsize=(12, 10))
sns.scatterplot(data=bray_umap, x="UMAP1", y="UMAP2", s=50,
                hue="Set",
                alpha=0.7, edgecolor='gray', linewidth=0.5
                )

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_set.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_set.pdf"))
plt.close()

# Beta diversity
plt.figure(figsize=(12, 10))
sns.scatterplot(data=bray_umap, x="UMAP1", y="UMAP2", s=50,
                hue="Cancer_Site",
                alpha=0.7, edgecolor='gray', linewidth=0.5
                )

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_lung.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/Beta_VOC_lung.pdf"))
plt.close()

voc_metrics = compute_metrics(voc_df)

voc_metrics = voc_metrics.join(voc_tmp_df)
voc_metrics = voc_metrics.merge(metadata_df, left_index=True, right_on='Sample_renamed', how='left')

print(voc_metrics.head())
print(voc_metrics.shape)

# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="gini", hue="Case", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/gini_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/gini_boxplot.pdf"))
plt.close()

# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="variance", hue="Case", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/variance_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/variance_boxplot.pdf"))
plt.close()

# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="hill_number_q2", hue="Case", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/hill_number_q2_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/hill_number_q2_boxplot.pdf"))
plt.close()


# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="gini", hue="subclass2", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_gini_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_gini_boxplot.pdf"))
plt.close()

# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="variance", hue="subclass2", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_variance_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_variance_boxplot.pdf"))
plt.close()

# Plot
plt.figure(figsize=(10, 10))
g = sns.catplot(data=voc_metrics,
            x="Type_Group", y="hill_number_q2", hue="subclass2", kind="box",
	        boxprops=dict(alpha=.5)
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_hill_number_q2_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/cca/subclass2_hill_number_q2_boxplot.pdf"))
plt.close()





metadata_df.set_index('Sample_renamed', inplace=True)
result_df = indicator_analysis(voc_df, metadata_df, "Case")
result_df.to_csv(os.path.join(data_dir, f"vsearch_output/cca/VOC.Case.Kruskal–Wallis.tsv"), sep="\t", index=False)