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


def perform_umap(data, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Performs UMAP dimensionality reduction on the data.

    Args:
        data (pd.DataFrame): Input data.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        random_state (int): Random state for reproducibility.

    Returns:
        umap.UMAP: Fitted UMAP reducer.
        pd.DataFrame: DataFrame with UMAP embeddings (X and Y).
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(data)
    umap_df = pd.DataFrame(embedding, index=data.index, columns=["UMAP1", "UMAP2"])
    print(f"Performed UMAP dimensionality reduction. Embedding shape: {umap_df.shape}")
    return reducer, umap_df




data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
# Load ASV metadata
metastat_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/metadata/master_table.tsv'), sep='\t')
asv_meta_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/metadata/ASV_meta.tsv'), sep='\t', header=0)

metadata_table_path = os.path.join(data_dir, 'vsearch_output/metadata/metadata_updated.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')

all_type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           'Failed-QC': 'lightgray'
           }

three_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A'
           }

status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026'}


# Define comparisons to annotate:
sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
order = ['Oral Rinse', 'BAL', 'Lung Brush']
sub_df = sub_df.loc[sub_df['Type_Group'].isin(order)]
sub_df['Type_Group'] = pd.Categorical(sub_df['Type_Group'], order)

keep_order = [t for t in order if t in list(sub_df['pass_filter'])]
comparisons = list(combinations(keep_order, 2))
sub_type_palette = {k: all_type_palette[k] for k in all_type_palette if k in sub_df['Type_Group'].unique()}

results = []
for a, b in combinations(sub_df['Type_Group'].unique(), 2):
    group1 = sub_df[sub_df['Type_Group'] == a]['Shannon'].dropna()
    group2 = sub_df[sub_df['Type_Group'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv(os.path.join(data_dir, 'vsearch_output/diversity/alpha_sample_ttest.tsv'), sep='\t', index=False)

results = []
for a, b in combinations(sub_df['status'].unique(), 2):
    group1 = sub_df[sub_df['status'] == a]['Shannon'].dropna()
    group2 = sub_df[sub_df['status'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv(os.path.join(data_dir, 'vsearch_output/diversity/alpha_status_ttest.tsv'), sep='\t', index=False)

plt.figure(figsize=(10, 10))
g = sns.catplot(data=sub_df,
            x="Type_Group", y="Shannon", hue="Type_Group", kind="box",
	        palette=sub_type_palette, saturation=1, boxprops=dict(alpha=.5),
            order=order
            )

# Add annotations:
ax = g.ax  # Extract axis from FacetGrid
annotator = Annotator(ax, comparisons, data=sub_df, x="Type_Group", y="Shannon", order=keep_order)
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_combined_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_combined_boxplot.pdf"))
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=sub_df,
            x="status", y="Shannon", hue="status", kind="box",
            palette=status_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Non-Cancer', 'Cancer']
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_status_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_status_boxplot.pdf"))
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=sub_df,
            x="status", y="Shannon", hue="status", col='Type_Group', kind="box",
            palette=status_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Non-Cancer', 'Cancer'],
            col_order=keep_order
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_boxplot.pdf"))
plt.close()


# Beta diversity
plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2",
                hue="Type_Group", hue_order=keep_order,
                size='count', sizes=(40, 400),
                palette=sub_type_palette, style="status",
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
size_values = [5000, 10000, 20000, 50000, 100000]
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
    title="Type"
)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status.pdf"))
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="Jacc_UMAP1", y="Jacc_UMAP2",
                hue="Type_Group", hue_order=keep_order,
                size='count', sizes=(40, 400),
                palette=sub_type_palette, style="status",
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
size_values = [5000, 10000, 20000, 50000, 100000]
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
    title="Type"
)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_jaccard.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_jaccard.pdf"))
plt.close()


plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=sub_type_palette, alpha=0.75, style='lung_code', markers=['o', 's', '^'],
                edgecolor='grey', linewidth=0.5
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
size_values = [5000, 10000, 20000, 50000, 100000]
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
    title="Type"
)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_LungCode.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_LungCode.pdf"))
plt.close()


plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=sub_type_palette, alpha=0.75, style='Set',
                edgecolor='grey', linewidth=0.5
                )

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False
)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Set.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Set.pdf"))
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=sub_type_palette, alpha=0.75, style='patient_code',
                edgecolor='grey', linewidth=0.5
                )

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False
)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Patient.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Patient.pdf"))
plt.close()


plt.figure(figsize=(10, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group",
                size='count', sizes=(40, 400), palette=sub_type_palette,
                alpha=0.75
                )

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_UMAP_NoControls.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_UMAP_NoControls.pdf"))
plt.close()










# MITO diversity
alpha_path = os.path.join(data_dir, 'vsearch_output/diversity/shannon.mito.tsv')
alpha_df = pd.read_csv(alpha_path, header=0, sep='\t')

bray_path = os.path.join(data_dir, 'vsearch_output/diversity/bray.mito.tsv')
bray_df = pd.read_csv(bray_path, header=0, sep='\t', index_col=0)
bray_reducer, bray_umap = perform_umap(bray_df, random_state=42)

jacc_path = os.path.join(data_dir, 'vsearch_output/diversity/jaccard.mito.tsv')
jacc_df = pd.read_csv(jacc_path, header=0, sep='\t', index_col=0)
jacc_reducer, jacc_umap = perform_umap(jacc_df, random_state=42)
jacc_umap.columns = ['Jacc_UMAP1', 'Jacc_UMAP2']

mito_asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.mito.tsv')
mito_asv_df = pd.read_csv(mito_asv_path, header=0, sep='\t', index_col=0)
mito_asv_df.columns = [x.rsplit('_', 1)[0] for x in mito_asv_df.columns]
mito_asv_stack_df = mito_asv_df.stack().reset_index()
mito_asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
mito_asv_stack_df = mito_asv_stack_df.loc[mito_asv_stack_df['count'] > 0]
mito_asv_stack_df.set_index('ASV_ID', inplace=True)
cnt_df = mito_asv_stack_df.groupby(['sample'])['count'].sum().reset_index()

mito_meta_df = metadata_df.merge(alpha_df, how='left', on='sample')
mito_meta_df = mito_meta_df.merge(bray_umap.reset_index(), how='left', on='sample')
mito_meta_df = mito_meta_df.merge(jacc_umap.reset_index(), how='left', on='sample')
mito_meta_df = mito_meta_df.merge(cnt_df, how='left', on='sample')
mito_meta_df['pass_filter'] = [t if s in list(asv_meta_df['sample']) else 'Failed-QC'
                              for s,t in  zip(mito_meta_df['sample'], mito_meta_df['Type_Group'])
                              ]
asv_mito_meta_df = mito_asv_stack_df.reset_index().merge(metadata_df, how='left', on='sample')

mito_meta_df.to_csv(os.path.join(data_dir, 'vsearch_output/metadata/master_table_mito.tsv'), sep='\t', index=False)
mito_meta_df = mito_meta_df.loc[((~mito_meta_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])) & (mito_meta_df['pass_filter'] != 'Failed-QC'))]
mito_meta_df = mito_meta_df.loc[mito_meta_df['pass_filter'] != 'Failed-QC']
type_order = ['Oral Rinse', 'BAL', 'Lung Brush']
mito_meta_df['Type_Group'] = pd.Categorical(mito_meta_df['Type_Group'], [t for t in type_order if t in list(mito_meta_df['Type_Group'])])

results = []
for a, b in combinations(mito_meta_df['Type_Group'].unique(), 2):
    group1 = mito_meta_df[mito_meta_df['Type_Group'] == a]['Shannon'].dropna()
    group2 = mito_meta_df[mito_meta_df['Type_Group'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv(os.path.join(data_dir, 'vsearch_output/diversity/alpha_sample_ttest_mitochondrial.tsv'), sep='\t', index=False)

results = []
for a, b in combinations(mito_meta_df['status'].unique(), 2):
    group1 = mito_meta_df[mito_meta_df['status'] == a]['Shannon'].dropna()
    group2 = mito_meta_df[mito_meta_df['status'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv(os.path.join(data_dir, 'vsearch_output/diversity/alpha_status_ttest_mitochondrial.tsv'), sep='\t', index=False)

plt.figure(figsize=(10, 10))
g = sns.catplot(data=mito_meta_df,
            x="Type_Group", y="Shannon", hue="Type_Group", kind="box",
            palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5),
            order=type_order
            )

comparisons = list(combinations(type_order, 2))

# Add annotations:
ax = g.ax  # Extract axis from FacetGrid
annotator = Annotator(ax, comparisons, data=mito_meta_df, x="Type_Group", y="Shannon", order=type_order)
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
annotator.apply_and_annotate()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_type_mitochondrial_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_type_mitochondrial_boxplot.pdf"))
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=mito_meta_df,
            x="status", y="Shannon", hue="status", kind="box",
            palette=status_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Non-Cancer', 'Cancer']
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_status_mitochondrial_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_status_mitochondrial_boxplot.pdf"))
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=mito_meta_df,
            x="status", y="Shannon", hue="status", col='Type_Group', kind="box",
            palette=status_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Non-Cancer', 'Cancer'],
            col_order=type_order
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_mitochondrial_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_mitochondrial_boxplot.pdf"))
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=mito_meta_df, x="UMAP1", y="UMAP2",
                hue="Type_Group", hue_order=type_order,
                size='count', sizes=(40, 400),
                palette=all_type_palette, style="status",
                alpha=0.75
                )

ax = plt.gca()  # Or whatever axis you're plotting on

# Get current legend entries
handles, labels = ax.get_legend_handles_labels()

# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
numeric_labels = [i for i, l in enumerate(labels) if l.isdigit()]
if numeric_labels:
    size_start = numeric_labels[0]
    size_end = numeric_labels[-1] + 1

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
size_values = [1000, 5000, 10000, 20000, 50000]
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

style_handles = handles[size_end:]
style_labels = labels[size_end:]

# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Type"
)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_mitochondrial.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_mitochondrial.pdf"))
plt.close()


plt.figure(figsize=(12, 10))
sns.scatterplot(data=mito_meta_df, x="Jacc_UMAP1", y="Jacc_UMAP2",
                hue="Type_Group", hue_order=type_order,
                size='count', sizes=(40, 400),
                palette=all_type_palette, style="status",
                alpha=0.75
                )

ax = plt.gca()  # Or whatever axis you're plotting on

# Get current legend entries
handles, labels = ax.get_legend_handles_labels()

# Seaborn usually orders: hue first, then size, then style
# Find where the size section starts:
numeric_labels = [i for i, l in enumerate(labels) if l.isdigit()]
if numeric_labels:
    size_start = numeric_labels[0]
    size_end = numeric_labels[-1] + 1

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
size_values = [1000, 5000, 10000, 20000, 50000]
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

style_handles = handles[size_end:]
style_labels = labels[size_end:]

# Rebuild the legend
ax.legend(
    hue_handles + size_handles + style_handles,
    hue_labels + size_labels + style_labels,
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False,
    title="Type"
)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_mitochondrial_jaccard.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_Sample_status_mitochondrial_jaccard.pdf"))
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=mito_meta_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=all_type_palette, alpha=0.75, style='lung_code', markers=['o', 's', '^'],
                edgecolor='grey', linewidth=0.5, hue_order=type_order,
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
size_values = [5000, 10000, 20000, 50000, 100000]
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
    title="Type"
)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_LungCode_mitochondrial.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_LungCode_mitochondrial.pdf"))
plt.close()

