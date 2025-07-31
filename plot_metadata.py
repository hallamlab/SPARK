import pandas as pd
import numpy as np
import sys
import umap
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import PowerNorm
from matplotlib import gridspec
from matplotlib import font_manager as fm, rcParams
from matplotlib.patches import Patch
import matplotlib as mpl
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import math
from itertools import cycle
import colorsys
import matplotlib.colors as mcolors


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
            - If precomputed=False, rows are lmp_ids × features.
            - If precomputed=True, must be a square (lmp_ids × lmp_ids) distance matrix.
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


# Create output directory if it doesn't exist
data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'
output_dir = os.path.join(data_dir, "methods_output/metadata")
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

metadata_table_path = os.path.join(data_dir, 'ref_db/methods_metadata.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df['lmp_id'] = metadata_df.copy()['sample']
metadata_df['lmp_id'].astype(str)
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]

keep_types = ['Scope Flush',
              'Skin Brush',
              'Oral Rinse',
              'BAL',
              'Lung Brush'
              ]

all_type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           'Failed-QC': 'lightgray'
           }

type_palette = {'Lung Brush': '#009E73',
                'BAL': '#0072B2',
                'Oral Rinse': '#6A3D9A'
                }
status_palette = {'Non-Cancer':'white',
                  'Cancer':'#A50026',
                  'methods':'lightgray'
                  }

kit_pallete = {'HostZERO-DEP': 'black',
               'HostZERO-NODEP': 'gray',
               'SPARK-ZYMO': 'skyblue',
               }

metadata_df = metadata_df.loc[metadata_df['type_group'].isin(keep_types)]

fastq_stats_path = os.path.join(data_dir, 'methods_output/stats/fastq_stats.tsv')
fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
fstats_df['lmp_id'] = [str(x.split('/')[-1].rsplit('_')[0]) for x in fstats_df['file']]
reads_df = fstats_df.groupby(['lmp_id'])['num_seqs'].sum().reset_index()

taxonomy_path = os.path.join(data_dir, 'methods_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)

asv_raw_path = os.path.join(data_dir, 'methods_output/ASVs/ASV_target.micro.tsv')
asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
asv_raw_df.columns = [str(x.split('/')[-1].split('-')[-1].split('_')[0]) for x in asv_raw_df.columns]
asv_raw_stack_df = asv_raw_df.stack().reset_index()
asv_raw_stack_df.columns = ['ASV_ID', 'lmp_id', 'raw_count']
asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['raw_count'] > 0]
asv_raw_stack_df.set_index('ASV_ID', inplace=True)
asv_raw_stack_df['lmp_id'] = asv_raw_stack_df['lmp_id'].astype(str)
metadata_df['lmp_id'] = metadata_df['lmp_id'].astype(str)
asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='lmp_id')
asv_raw_cnt_df = asv_raw_meta_df.groupby(['lmp_id'])['raw_count'].sum().reset_index()

asv_path = os.path.join(data_dir, 'methods_output/ASVs/ASV_target.micro.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [str(x.split('/')[-1].split('-')[-1].split('_')[0]) for x in asv_df.columns]
asv_df = asv_df.loc[[a for a in asv_df.index.values if a in list(tax_df.index.values)]]

asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
asv_stack_df = asv_stack_df.loc[asv_stack_df['count'] > 0]
asv_stack_df.set_index('ASV_ID', inplace=True)
asv_tax_df = asv_stack_df.merge(tax_df, how='left', left_index=True, right_index=True)
taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

asv_meta_df = asv_tax_df.reset_index().merge(metadata_df, on='lmp_id', how='inner')
cnt_df = asv_meta_df.groupby(['lmp_id'])['count'].sum().reset_index()

metastat_df = metadata_df.merge(reads_df, how='left', on='lmp_id')
metastat_df = metastat_df.merge(cnt_df, how='left', on='lmp_id')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='lmp_id')
metastat_df['pass_filter'] = [s if s in list(asv_meta_df['lmp_id']) else 'Failed-QC'
                              for s in  metastat_df['lmp_id']
                              ]

long_df = metastat_df.groupby(['type_group', 'pass_filter', 'lmp_id'])['raw_count'].sum().reset_index()
long_df = long_df.loc[long_df['raw_count'] > 0] # remove empty values

# Plot
plt.figure(figsize=(10, 10))
ax = sns.boxplot(
    x='type_group', y='raw_count', data=long_df,
    color='lightgray',  # box color
    fliersize=0,        # hide default outliers
    linewidth=1,        # box edge width
    showcaps=True,
    order=keep_types
    )

# Overlay with swarm plot
sns.stripplot(data=long_df, x='type_group', y='raw_count',
              hue='type_group', alpha=0.75, ax=ax, legend=False,
              jitter=0.25, palette=all_type_palette
              )

# Dashed line at 5k
plt.axhline(y=1000, linestyle='--', color='black', linewidth=1)

plt.title("Sample Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "methods_output/metadata/type_group_swarmplot.svg"))
plt.savefig(os.path.join(data_dir, "methods_output/metadata/type_group_swarmplot.pdf"))
plt.close()

sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
scope_lmp_ids = list(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['lmp_id'])
scope_asvs = asv_meta_df.loc[((asv_meta_df['lmp_id'].isin(scope_lmp_ids)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
skin_lmp_ids = list(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['lmp_id'])
skin_asvs = asv_meta_df.loc[((asv_meta_df['lmp_id'].isin(skin_lmp_ids)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
offtarg_asvs = list(set(scope_asvs + skin_asvs))
keep_cols = ['ASV_ID', 'lmp_id', 'count']
skin_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(skin_asvs)) &
    (asv_meta_df['type_group'].isin(['Skin Brush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
skin_df.columns = ['ASV_ID', 'offtarg_mean']
keep_cols = ['ASV_ID', 'lmp_id', 'count']
scope_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(scope_asvs)) &
    (asv_meta_df['type_group'].isin(['Scope Flush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
scope_df.columns = ['ASV_ID', 'nctrl_mean']
asv_meta_df = asv_meta_df.merge(skin_df, how='left', on=['ASV_ID'])
asv_meta_df['offtarg_mean'] = asv_meta_df['offtarg_mean'].fillna(0)
asv_meta_df = asv_meta_df.merge(scope_df, how='left', on='ASV_ID')
asv_meta_df['nctrl_mean'] = asv_meta_df['nctrl_mean'].fillna(0)
asv_meta_df['count_sub_scope'] = asv_meta_df['count'] - asv_meta_df['nctrl_mean']
asv_meta_df['count_sub_skin'] = asv_meta_df['count_sub_scope'] - asv_meta_df['offtarg_mean']
asv_meta_df['corr_count'] = [int(x) if x > 0 else int(0) for x in asv_meta_df['count_sub_skin']]
asv_meta_df = asv_meta_df.loc[~asv_meta_df['type_group'].isin(['Scope Flush', 'Skin Brush'])]
cleaned_asv_df = asv_meta_df.pivot_table(index='ASV_ID', columns='lmp_id',
                              values='corr_count', aggfunc='sum', fill_value=0
                              )
asv_keep_list = list(asv_meta_df.loc[asv_meta_df['Domain'] != 'Unassigned']['ASV_ID'].unique())
final_asv_df = cleaned_asv_df[[x for x in cleaned_asv_df.columns if x in list(sub_df['lmp_id'])]]
final_asv_df = final_asv_df.loc[asv_keep_list]
final_asv_df = final_asv_df.loc[~(final_asv_df == 0).all(axis=1)]
asv_meta_df.to_csv(os.path.join(data_dir, 'methods_output/metadata/ASV_meta.tsv'), sep='\t', index=False)
asv_path = os.path.join(data_dir, 'methods_output/ASVs/ASV_final.micro.tsv')
final_asv_df.to_csv(asv_path, sep='\t', index=True)
metastat_df.to_csv(os.path.join(data_dir, 'methods_output/metadata/master_table.tsv'), sep='\t', index=False)
metadata_df.to_csv(os.path.join(data_dir, 'methods_output/metadata/metadata_updated.tsv'), sep='\t', index=False)
print(final_asv_df.shape)

# Map to colors
m_df = metastat_df.loc[metastat_df['lmp_id'].isin(final_asv_df.columns)].set_index('lmp_id')
filtered_asv_df = final_asv_df[m_df.index.tolist()]
#invert_bray_df = 1 - bray_df.loc[m_df.index.tolist(), m_df.index.tolist()]

col_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    'kit': m_df['kit'].map(kit_pallete)
}, index=m_df.index)
row_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    'kit': m_df['kit'].map(kit_pallete)
}, index=m_df.index)

# Create a new colormap with white at the beginning
viridis = plt.cm.get_cmap('viridis', 256)
colors = viridis(np.linspace(0, 1, 256))
colors[0] = [1, 1, 1, 1]  # Replace the first color (low end) with white
# Create new colormap
viridis_white = mcolors.ListedColormap(colors)

# Faster shift from white to gray
colors = [
    (0.0, '#ffffff'),  # white at 0.0
    (0.2, '#d9d9d9'),  # light gray quickly after
    (1.0, '#000000')   # black at 1.0
]
cmap = LinearSegmentedColormap.from_list("light_greyscale", colors, N=256)

presence_absence = (filtered_asv_df > 0).astype(int)
shared = presence_absence.T.dot(presence_absence)
n = presence_absence.sum()
n_array = n.to_numpy()  # convert to NumPy
shared_percent = shared.div(n_array[:, None] + n_array[None, :] - shared.to_numpy()) * 100
shared_percent = pd.DataFrame(shared_percent, index=shared.index, columns=shared.columns).fillna(0)

g = sns.clustermap(
    shared_percent,
    method='ward',
    metric='euclidean',
    col_colors=col_colors_df,
    row_colors=row_colors_df,
    cmap=cmap,
    vmin=0,
    vmax=100,
    linewidths=0,
    xticklabels=False,
    yticklabels=False,
    dendrogram_ratio=(0.05, 0.05),
    colors_ratio=(0.02, 0.02),
    figsize=(32, 32),
    cbar_pos=(1.02, 0.2, 0.03, 0.4),
    alpha=1.0,
    #col_cluster=False
    )
# Create legend entries
handles = []
for group in ['Oral Rinse', 'BAL', 'Lung Brush']:
    color = type_palette[group]
    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
for group in ['Non-Cancer', 'Cancer', 'methods']:
    color = status_palette[group]
    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
for group in ['HostZERO-DEP', 'HostZERO-NODEP', 'SPARK-ZYMO']:
    color = kit_pallete[group]
    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
# Add legend outside the clustermap
plt.legend(
    handles=handles,
    bbox_to_anchor=(1, 1),
    bbox_transform=plt.gcf().transFigure,
    loc='upper left',
    title="Sample Type / Kit",
    frameon=False
)
# Format colorbar
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_label("% Shared ASVs", rotation=270, labelpad=15)
g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
g.ax_heatmap.tick_params(axis='x', which='both', length=5)
plt.savefig(os.path.join(data_dir, f"methods_output/metadata/clustermap_ASVpercent.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"methods_output/metadata/clustermap_ASVpercent.pdf"), bbox_inches='tight')
plt.close()

'''
g = sns.clustermap(
    invert_bray_df,
    method='ward',
    metric='euclidean',
    col_colors=col_colors_df,
    row_colors=row_colors_df,
    cmap=cmap,
    vmin=0,
    vmax=1,
    linewidths=0,
    xticklabels=False,
    yticklabels=False,
    dendrogram_ratio=(0.05, 0.05),
    colors_ratio=(0.02, 0.02),
    figsize=(32, 32),
    cbar_pos=(1.02, 0.2, 0.03, 0.4),
    alpha=1.0,
    #col_cluster=False
    )
# Create legend entries
handles = []
# For Type_Group
for group in ['Oral Rinse', 'BAL', 'Lung Brush']:
    color = type_palette[group]
    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
# For Type_Group
for group in ['Non-Cancer', 'Cancer']:
    color = status_palette[group]
    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
# Add legend outside the clustermap
plt.legend(
    handles=handles,
    bbox_to_anchor=(1, 1),
    bbox_transform=plt.gcf().transFigure,
    loc='upper left',
    title="Sample Type / Cancer Status",
    frameon=False
)
# Format colorbar
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_label("Bray Curtis", rotation=270, labelpad=15)
g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
g.ax_heatmap.tick_params(axis='x', which='both', length=5)
plt.savefig(os.path.join(data_dir, f"methods_output/diversity/clustermap_braycurtis.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"methods_output/diversity/clustermap_braycurtis.pdf"), bbox_inches='tight')
plt.close()
'''

