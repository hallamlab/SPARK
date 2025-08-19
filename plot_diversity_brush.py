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
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from itertools import combinations
from statsmodels.stats.multitest import multipletests


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




data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'
# Load ASV metadata
metastat_df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/metadata/master_table.tsv'), sep='\t')
asv_meta_df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/metadata/ASV_meta.tsv'), sep='\t', header=0)
metadata_table_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/metadata_updated.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')

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

brush_palette = {'ca-lung': '#009E73',
           'ca-contra': '#0072B2',
           'ctrl-brush': '#6A3D9A'
           }

filter_types = ['Skin Brush', 'Scope Flush', 'No VOCs']

seqtype_list = ['ctrl-brush', 'ca-contra', 'ca-lung']

###########################################################################################################
alpha_path = os.path.join(data_dir, 'spark_old_output/brush/diversity/shannon.tsv')
alpha_df = pd.read_csv(alpha_path, header=0, sep='\t')
bray_path = os.path.join(data_dir, 'spark_old_output/brush/diversity/bray.tsv')
bray_df = pd.read_csv(bray_path, header=0, sep='\t', index_col=0)
bray_reducer, bray_umap = perform_umap(data=bray_df,
                                       n_neighbors=30,
                                       min_dist=0.01,
                                       random_state=42,
                                       precomputed=True
                                       )
jacc_path = os.path.join(data_dir, 'spark_old_output/brush/diversity/jaccard.tsv')
jacc_df = pd.read_csv(jacc_path, header=0, sep='\t', index_col=0)
jacc_reducer, jacc_umap = perform_umap(data=jacc_df,
                                       n_neighbors=30,
                                       min_dist=0.01,
                                       random_state=42,
                                       precomputed=True
                                       )
jacc_umap.columns = ['Jacc_UMAP1', 'Jacc_UMAP2']

olall_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/outliers_table.tsv')
olall_df = pd.read_csv(olall_path, header=0, sep='\t')

oltype_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/outliers_subclass2.tsv')
oltype_df = pd.read_csv(oltype_path, header=0, sep='\t')


metastat_df = metastat_df.merge(olall_df[['sample', 'is_outlier']], how='left', left_on='sample', right_on='sample')
metastat_df.rename(columns={'is_outlier': 'overall_OL'}, inplace=True)
metastat_df = metastat_df.merge(oltype_df[['sample', 'is_outlier']], how='left', left_on='sample', right_on='sample')
metastat_df.rename(columns={'is_outlier': 'typ_grp_OL'}, inplace=True)

metastat_df = metastat_df.merge(alpha_df, how='left', on='sample')
metastat_df = metastat_df.merge(bray_umap.reset_index(), how='left', on='sample')
metastat_df = metastat_df.merge(jacc_umap.reset_index(), how='left', on='sample')

############################################################################################################

# Define comparisons to annotate:
sub_df = metastat_df.loc[((metastat_df['pass_filter'] != 'Failed-QC') & (~metastat_df['subclass2'].isin(filter_types)))]
order = seqtype_list

sub_brush_palette = {k: brush_palette[k] for k in brush_palette if k in sub_df['subclass2'].unique()}
keep_order = seqtype_list
comparisons = list(combinations(keep_order, 2))

results = []
for a, b in combinations(sub_df['subclass2'].unique(), 2):
    group1 = sub_df[sub_df['subclass2'] == a]['Shannon'].dropna()
    group2 = sub_df[sub_df['subclass2'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv(os.path.join(data_dir, 'spark_old_output/brush/diversity/alpha_sample_ttest.tsv'), sep='\t', index=False)

# Load Bray-Curtis matrix
bray_path = os.path.join(data_dir, 'spark_old_output/brush/diversity/bray.tsv')
bray_df = pd.read_csv(bray_path, header=0, sep='\t', index_col=0)
m_df = metadata_df.copy().set_index('sample')
sample_ids = m_df.index
valid_ids = bray_df.index.intersection(sample_ids)
bray_df = bray_df.loc[valid_ids, valid_ids]
m_df = m_df.loc[valid_ids]
dm = DistanceMatrix(bray_df.values.copy(order='C'), ids=bray_df.index)

# Load sample metadata with group labels
grouping = m_df.loc[bray_df.index, 'subclass2']  # align order

# Run PERMANOVA
result = permanova(dm, grouping, permutations=999)
print(result)

# Inputs: full Bray-Curtis matrix (bray_df), metadata with 'Group'
groups = m_df['subclass2']
unique_groups = groups.unique()
results = []
for g1, g2 in combinations(unique_groups, 2):
    subset_ids = groups[groups.isin([g1, g2])].index.tolist()
    
    # Must have at least 2 samples per group
    sub_groups = groups.loc[subset_ids]
    counts = sub_groups.value_counts()
    if (counts < 2).any():
        continue  # Skip this pair

    sub_dm = DistanceMatrix(
        bray_df.loc[subset_ids, subset_ids].values.copy(order='C'),
        ids=subset_ids
    )
    
    res = permanova(sub_dm, sub_groups, permutations=999)
    res['Group1'] = g1
    res['Group2'] = g2
    results.append(res)

# Collect results
pairwise_df = pd.DataFrame(results)

pairwise_df = pairwise_df[['Group1', 'Group2', 'test statistic', 'p-value']]
pvals = pairwise_df['p-value']
pairwise_df['q-value'] = multipletests(pvals, method='fdr_bh')[1]

print(pairwise_df)
sample_type_ttests.to_csv(os.path.join(data_dir, 'spark_old_output/brush/diversity/beta_subclass2_permanova.tsv'), sep='\t', index=False)

# Pivot to symmetric matrix
heatmap_df = pairwise_df.pivot(index='Group1', columns='Group2', values='q-value')
heatmap_df_full = heatmap_df.combine_first(heatmap_df.T)

# Plot: low q = red, high q = white/blue (non-significant = cool/neutral)
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_df_full,
    annot=True,
    cmap='coolwarm_r',        # reversed so high q = blue
    vmin=0,
    vmax=0.1,
    cbar_kws={'label': 'q-value'},
    linewidths=0.5,
    linecolor='lightgray'
)
plt.title('Pairwise PERMANOVA (q-values)\nBlue = Not Significant, Red = Significant')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_Heatmap_permanova.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_Heatmap_permanova.pdf"))
plt.close()


alpha_y_bounds = (0, max(sub_df['Shannon'].astype(int)) + 1)
plt.figure(figsize=(10, 10))
g = sns.catplot(data=sub_df,
            x="subclass2", y="Shannon", hue="subclass2", kind="box",
	        palette=sub_brush_palette, saturation=1, boxprops=dict(alpha=.5),
            order=order
            )

# Add annotations:
ax = g.ax  # Extract axis from FacetGrid
annotator = Annotator(ax, comparisons, data=sub_df, x="subclass2", y="Shannon", order=keep_order)
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()
plt.xticks(rotation=45)
plt.ylim(alpha_y_bounds)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Alpha_subclass2_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Alpha_subclass2_boxplot.pdf"))
plt.close()



plt.figure(figsize=(10, 10))
g = sns.catplot(data=sub_df,
            x="status", y="Shannon", hue="status", kind="box",
	        palette=status_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Cancer', 'Non-Cancer'], col='subclass2', col_wrap=3, sharey=True,
            col_order=seqtype_list
            )

plt.xticks(rotation=45)
plt.ylim(alpha_y_bounds)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Alpha_status_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Alpha_status_boxplot.pdf"))
plt.close()








fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="subclass2",
                size='count', sizes=(40, 400), palette=brush_palette,
                alpha=0.75
                )
ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Sample Type"
)
plt.title("Sample Type UMAP")
fig.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2.pdf"))
plt.close()



fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="subclass2",
                size='count', sizes=(40, 400), palette=brush_palette,
                alpha=0.75, style='lung_code'
                )
ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Sample Type"
)
plt.title("Sample Type UMAP")
fig.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2_lung.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2_lung.pdf"))
plt.close()





fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="subclass2",
                size='count', sizes=(40, 400), palette=brush_palette,
                alpha=0.75, style='status'
                )
ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Sample Type"
)
plt.title("Sample Type UMAP")
fig.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2_status.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_subclass2_status.pdf"))
plt.close()







fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="status",
                size='count', sizes=(40, 400), palette=status_palette,
                alpha=0.75, edgecolor='lightgray', linewidth=0.5
                )

ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Cancer Status"
)
plt.title("Cancer Status UMAP")
fig.tight_layout()

plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_status.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_status.pdf"))
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="subclass2",
                size='count', style='overall_OL', sizes=(40, 400),
                palette=brush_palette, alpha=0.75, edgecolor='lightgray',
                linewidth=0.5
                )

ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Sample Type"
)
plt.title("Sample Type UMAP with study-wide outliers annotated")
fig.tight_layout()

plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_type_olall.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_type_olall.pdf"))
plt.close()

fig, ax = plt.subplots(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="subclass2",
                size='count', style='typ_grp_OL', sizes=(40, 400),
                palette=brush_palette, alpha=0.75, edgecolor='lightgray',
                linewidth=0.5
                )

ax = plt.gca()  # Or whatever axis you're plotting on
# Get current legend entries
handles, labels = ax.get_legend_handles_labels()
# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
size_start = next(i for i, l in enumerate(labels) if l.isdigit())
# Split them:
hue_handles = handles[:size_start]
hue_labels = labels[:size_start]
hue_handles = [
               plt.Line2D([], [], marker='o', linestyle='None',
               markersize=8,  # or whatever size you want for hue markers
               color=h.get_color())
               for h in hue_handles
               ]
# Manual size legend values + labels (replace with yours)
size_values = [5000, 25000, 50000, 100000, 200000, 300000]
size_labels = [f"{v:,}" for v in size_values]
# Match seaborn scaling
min_size, max_size = 40, 400  # from sizes=(40, 400)
def scale_size(val):
    return min_size + (val - min(size_values)) / (max(size_values) - min(size_values)) * (max_size - min_size)
size_handles = [
    plt.Line2D([], [], marker='o', linestyle='None',
               markersize=np.sqrt(scale_size(v)),  # sqrt because scatter uses area
               color='gray', alpha=0.75)
    for v in size_values
]
style_handles = handles[size_start+5:]
style_labels = labels[size_start+5:]
# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.25, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Sample Type"
)
plt.title("Sample Type UMAP with type-wise outliers annotated")
fig.tight_layout()

plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_type_oltype.svg"))
plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/Beta_UMAP_type_oltype.pdf"))
plt.close()