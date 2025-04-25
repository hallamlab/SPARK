import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import umap
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import PowerNorm
from scipy.stats import ttest_ind
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Patch
from statannotations.Annotator import Annotator
import matplotlib as mpl
import math
from itertools import cycle
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
data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
output_dir = os.path.join(data_dir, "vsearch_output/metadata")
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

metadata_table_path = os.path.join(data_dir, 'ref_db/spark_metadata.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]
patient_set = sorted(list(set(metadata_df['Participant_ID'])))
patient_dict = {x:i for i,x in enumerate(patient_set)}
metadata_df['patient_code'] = ['P' + str(patient_dict[p]) for p in metadata_df['Participant_ID']]
metadata_df['patient_int'] = [patient_dict[p] for p in metadata_df['Participant_ID']]
metadata_df['type_code'] = [t[0:2] for t in metadata_df['Type_Group']]
metadata_df['lung_code'] = [l[0] if l[0] in ['R', 'L'] else '' for l in metadata_df['Type']]
# Define desired order
patient_order = sorted(list(metadata_df['patient_int'].unique()))
type_order = ['Sk', 'Sc', 'Or', 'BA', 'Lu']
lung_order = ['R', 'L', '']
# Convert columns to categorical with specified order
metadata_df['type_code'] = pd.Categorical(metadata_df['type_code'], categories=type_order, ordered=True)
metadata_df['lung_code'] = pd.Categorical(metadata_df['lung_code'], categories=lung_order, ordered=True)
# Sort the dataframe
metadata_df = metadata_df.sort_values(['patient_int', 'type_code', 'lung_code'])
# Create unique sample code
metadata_df['sample_code'] = [f"{i+1:03d}" for i in range(len(metadata_df['sample']))]
col = 'sample_code'
metadata_df = metadata_df[[col] + [c for c in metadata_df.columns if c != col]]

fastq_stats_path = os.path.join(data_dir, 'vsearch_output/stats/fastq_stats.tsv')
fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
fstats_df['sample'] = [x.split('/')[-1].split('_L001_R')[0] for x in fstats_df['file']]
reads_df = fstats_df.groupby(['sample'])[['num_seqs', 'sum_len']].sum().reset_index()

alpha_path = os.path.join(data_dir, 'vsearch_output/diversity/shannon.tsv')
alpha_df = pd.read_csv(alpha_path, header=0, sep='\t')

bray_path = os.path.join(data_dir, 'vsearch_output/diversity/bray.tsv')
bray_df = pd.read_csv(bray_path, header=0, sep='\t', index_col=0)
bray_reducer, bray_umap = perform_umap(bray_df, random_state=42)

taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)
#mito = "d__Bacteria; p__Pseudomonadota; c__Alphaproteobacteria; o__Rickettsiales; f__Mitochondria; g__; s__"
#tax_df = tax_df.loc[((tax_df['Confidence'] >= 0.7) & (tax_df['Taxonomy'] != mito))]

asv_raw_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.micro.tsv')
asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
asv_raw_df.columns = [x.rsplit('_', 1)[0] for x in asv_raw_df.columns]
asv_raw_stack_df = asv_raw_df.stack().reset_index()
asv_raw_stack_df.columns = ['ASV_ID', 'sample', 'raw_count']
asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['raw_count'] > 0]
asv_raw_stack_df.set_index('ASV_ID', inplace=True)
asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='sample', how='left')
asv_raw_cnt_df = asv_raw_meta_df.groupby(['sample'])['raw_count'].sum().reset_index()

asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.micro.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [x.rsplit('_', 1)[0] for x in asv_df.columns]
asv_df = asv_df.loc[[a for a in asv_df.index.values if a in list(tax_df.index.values)]]

asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
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

asv_meta_df = asv_tax_df.reset_index().merge(metadata_df, on='sample', how='left')
cnt_df = asv_meta_df.groupby(['sample'])['count'].sum().reset_index()

metastat_df = metadata_df.merge(reads_df, how='left', on='sample')
metastat_df = metastat_df.merge(alpha_df, how='left', on='sample')
metastat_df = metastat_df.merge(bray_umap.reset_index(), how='left', on='sample')
metastat_df = metastat_df.merge(cnt_df, how='left', on='sample')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='sample')
metastat_df['pass_filter'] = [t if s in list(asv_meta_df['sample']) else 'Failed-QC'
                              for s,t in  zip(metastat_df['sample'], metastat_df['Type_Group'])
                              ]

metastat_df.to_csv(os.path.join(data_dir, 'vsearch_output/metadata/master_table.tsv'), sep='\t', index=False)
#sub_df = metastat_df.loc[((~metastat_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])) & (metastat_df['pass_filter'] != 'Failed-QC'))]
sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
type_order = ['Lung Brush', 'BAL', 'Oral Rinse', 'Scope Flush', 'Skin Brush']
sub_df['Type_Group'] = pd.Categorical(sub_df['Type_Group'], [t for t in type_order if t in list(sub_df['Type_Group'])])

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


ordered_type = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']

scope_samples = list(sub_df.loc[sub_df['Type_Group'] == 'Scope Flush']['sample'])
scope_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(scope_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
scope_spp_df = asv_meta_df.loc[asv_meta_df['ASV_ID'].isin(scope_asvs)
                               ].groupby(['Type_Group', 'Family', 'Genus', 'ASV_ID'])['count'].sum().reset_index(
                                )
scope_spp_df['Type_Group'] = pd.Categorical(scope_spp_df['Type_Group'], ordered_type)
scope_spp_df['Family Genus (ASV)'] = [f'{x} {y} ({z})' for x,y,z in zip(scope_spp_df['Family'], scope_spp_df['Genus'], scope_spp_df['ASV_ID'])]
fig, ax = plt.subplots(figsize=(12, 12))
sns.scatterplot(data=scope_spp_df, x='Type_Group', y='Family Genus (ASV)',
                size='count', sizes=(5, 500), palette=all_type_palette,
                hue_order=ordered_type, hue='Type_Group', alpha=0.75
                )
# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0
)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Scope_ASVs_bubbleplot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Scope_ASVs_bubbleplot.pdf"))
plt.close()

skin_samples = list(sub_df.loc[sub_df['Type_Group'] == 'Skin Brush']['sample'])
skin_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(skin_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
skin_spp_df = asv_meta_df.loc[asv_meta_df['ASV_ID'].isin(skin_asvs)
                               ].groupby(['Type_Group', 'Family', 'Genus', 'ASV_ID'])['count'].sum().reset_index(
                                )
skin_spp_df['Type_Group'] = pd.Categorical(skin_spp_df['Type_Group'], ordered_type)
skin_spp_df['Family Genus (ASV)'] = [f'{x} {y} ({z})' for x,y,z in zip(skin_spp_df['Family'], skin_spp_df['Genus'], skin_spp_df['ASV_ID'])]
fig, ax = plt.subplots(figsize=(12, 18))
sns.scatterplot(data=skin_spp_df, x='Type_Group', y='Family Genus (ASV)',
                size='count', sizes=(5, 500), palette=all_type_palette,
                hue_order=ordered_type, hue='Type_Group', alpha=0.75
                )
# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0
)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Skin_ASVs_bubbleplot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Skin_ASVs_bubbleplot.pdf"))
plt.close()

metastat_df['Type_Group'] = pd.Categorical(metastat_df['Type_Group'], ordered_type)
ms_grp_df = metastat_df.groupby(['Type_Group', 'pass_filter'])['sample'].size().reset_index()

pivot_df = ms_grp_df.pivot_table(index='Type_Group', columns='pass_filter',
                              values='sample', aggfunc='sum', fill_value=0
                              )

col_order = [x for x in ordered_type if x in pivot_df.columns]
pivot_df = pivot_df[col_order[::-1]].loc[pivot_df.index != 'Failed-QC']

long_df = metastat_df.groupby(['Type_Group', 'pass_filter', 'sample'])['raw_count'].sum().reset_index()
long_df = long_df.loc[long_df['raw_count'] > 0] # remove empty values

# Plot
plt.figure(figsize=(10, 10))
ax = sns.boxplot(
    x='Type_Group', y='raw_count', data=long_df,
    color='lightgray',  # box color
    fliersize=0,        # hide default outliers
    linewidth=1,        # box edge width
    showcaps=True
)

# Overlay with swarm plot
sns.stripplot(data=long_df, x='Type_Group', y='raw_count',
              hue='pass_filter', palette=all_type_palette,
              alpha=0.75, ax=ax, legend=False,
              jitter=0.25
              )

# Dashed line at 5k
plt.axhline(y=1000, linestyle='--', color='black', linewidth=1)

plt.title("Sample Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Type_Group_swarmplot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/Type_Group_swarmplot.pdf"))
plt.close()

# Plot using the inverted data
ax = pivot_df.plot(
                    kind='bar',
                    stacked=True,
                    figsize=(10, 10),
                    color=[all_type_palette[col] for col in pivot_df.columns],
                    edgecolor='gray',
                    linewidth=1,
                    alpha=0.75
                    )

handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(label)] for label in col_order]
ordered_labels = [label for label in col_order]
ax.legend(ordered_handles, ordered_labels)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Type_Group_histogram.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Type_Group_histogram.pdf"))
plt.close()


metastat_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed']
ordered_type = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']
metastat_df['Type_Group'] = pd.Categorical(metastat_df['Type_Group'], ordered_type)

print(metastat_df.head())
print(metastat_df.shape)

plt.figure(figsize=(10, 10))
sns.boxplot(data=metastat_df, x="Type_Group", y="num_seqs", hue="Type_Group",
			palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5), order=ordered_type
            )
plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Read_count_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Read_count_boxplot.pdf"))
plt.close()

plt.figure(figsize=(10, 10))
sns.boxplot(data=metastat_df, x="Type_Group", y="sum_len", hue="Type_Group",
	        palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5),
            order=ordered_type
            )
plt.title("Sample Type")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Basepair_sum_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Basepair_sum_boxplot.pdf"))
plt.close()


# Define comparisons to annotate:
order = ['Oral Rinse', 'BAL', 'Lung Brush']
sub_df = sub_df.loc[sub_df['Type_Group'].isin(order)]
sub_df['Type_Group'] = pd.Categorical(sub_df['Type_Group'], order)

keep_order = [t for t in order if t in list(sub_df['pass_filter'])]
comparisons = list(combinations(keep_order, 2))
sub_type_palette = {k: all_type_palette[k] for k in all_type_palette if k in sub_df['Type_Group'].unique()}

asv_keep_list = list(asv_meta_df.loc[asv_meta_df['Domain'] != 'Unassigned']['ASV_ID'].unique())

final_asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_final.tsv')
final_asv_df = asv_df[sub_df['sample']] # remove control samples and low quality before performing any downstream
final_asv_df = final_asv_df.loc[asv_keep_list] # remove ASVs with an 'Unassigned' taxonomy, as they are likely artifacts
final_asv_df = final_asv_df.loc[~(final_asv_df == 0).all(axis=1)]
final_asv_df.to_csv(final_asv_path, sep='\t')

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
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="status", size='count', sizes=(40, 400),
                palette=status_palette, alpha=0.75,
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
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_status.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Beta_status.pdf"))
plt.close()


plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=sub_type_palette, alpha=0.75, style='lung_code', markers=['^', 's'],
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
mito_meta_df = mito_meta_df.merge(cnt_df, how='left', on='sample')

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
            order=ordered_type
            )

comparisons = list(combinations(ordered_type, 2))

# Add annotations:
ax = g.ax  # Extract axis from FacetGrid
annotator = Annotator(ax, comparisons, data=mito_meta_df, x="Type_Group", y="Shannon", order=ordered_type)
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
            col_order=ordered_type
            )
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_mitochondrial_boxplot.svg"))
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/Alpha_sample_status_mitochondrial_boxplot.pdf"))
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=mito_meta_df, x="UMAP1", y="UMAP2",
                hue="Type_Group", hue_order=ordered_type,
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






# HCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from matplotlib import gridspec


isa_path = os.path.join(data_dir, 'vsearch_output/indicspecies/Type_status_ISA_results.tsv')
isa_df = pd.read_csv(isa_path, sep='\t')
sig_isa_df = isa_df.loc[((isa_df['type_significance'] == True) | (isa_df['status_significance'] == True)) &
                        ((isa_df['type_stat'] >= 0.6) | (isa_df['status_stat'] >= 0.6))
                        ]
sig_isa_asvs = list(sig_isa_df['ASV_ID'])

asv_meta_df = asv_meta_df.loc[~asv_meta_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])]

rank_type_dict = {}
rank_dict = {}
for rank in ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
    asv_rank_df = asv_meta_df.groupby(['Type_Group', rank, 'sample', 'Sample'])['count'].sum().reset_index()
    rank_type_dict[rank] = {}
    rank_dict[rank] = []
    for group in asv_meta_df['Type_Group'].unique():
        df_group = asv_rank_df[asv_rank_df['Type_Group'] == group]
        total_rank = df_group.groupby(rank)['count'].sum()
        N = 25
        topN = total_rank.sort_values(ascending=False).head(N).index.tolist()
        sig = asv_meta_df[asv_meta_df['ASV_ID'].isin(sig_isa_asvs)][rank].unique().tolist()
        All_r = list(set(topN + sig))
        rank_type_dict[rank][group] = All_r
        rank_dict[rank] = list(set(rank_dict[rank] + All_r))

for rank in ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']:
    rank_list = rank_dict[rank]
    plot_col = f"{rank}_plot"
    asv_meta_df[plot_col] = asv_meta_df[rank].apply(lambda x: x if x in rank_list else "Other")

for t in ['Phylum_plot', 'Class_plot', 'Order_plot', 'Family_plot', 'Genus_plot', 'Species_plot']:
    bubble_df = asv_meta_df.groupby(['sample_code', t, 'Type_Group', 'status'])['count'].sum().reset_index()
    pivot_df = bubble_df.pivot(index='sample_code', columns=t, values='count').fillna(0)

    # Map sample_code to sample_type
    col_meta = bubble_df.drop_duplicates('sample_code')[['sample_code', 'Type_Group', 'status']].set_index('sample_code')

    # Map to colors
    col_colors_df = pd.DataFrame({
        'Type_Group': col_meta['Type_Group'].map(all_type_palette),
        'status': col_meta['status'].map(status_palette)
        })

    pivot_df = asv_meta_df.groupby(['sample_code', t])['count'
                                    ].sum().reset_index().pivot(index=t,
                                                                columns='sample_code',
                                                                values='count'
                                                                ).fillna(0)

    # Log transform (add 1 to avoid log(0))
    pivot_log = np.log10(pivot_df + 1)

    # Create a custom colormap: white → light gray → black
    colors = ['#ffffff', '#d9d9d9', '#000000']  # white → light gray → black
    cmap = LinearSegmentedColormap.from_list("light_greyscale", colors, N=256)

    # Custom figure height based on number of taxa
    base_h = 8
    n_taxa = len(pivot_df.index.values)
    h_scaler = math.ceil(n_taxa / 25)
    height = base_h * h_scaler

    # Define your desired tick values (original scale)
    tick_vals_orig = [5, 50, 500, 5000, 50000]
    # Convert to log scale used in heatmap (log10(count + 1))
    tick_vals_log = [np.log10(v + 1) for v in tick_vals_orig]

    # Define your max display value
    vmax_display = 50000
    # Compute log-transformed vmax
    vmax_log = np.log10(vmax_display + 1)

    g = sns.clustermap(
        pivot_log,
        method='ward',
        metric='euclidean',
        col_colors=col_colors_df,
        cmap=cmap,
        vmin=0,
        vmax=vmax_log,
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(.05, .2),
        colors_ratio=0.02,
        figsize=(32, height),
        cbar_pos=(1.02, 0.2, 0.03, 0.4),
        alpha=0.75,
        col_cluster=False
        )

    # Create legend entries
    handles = []

    # For Type_Group
    for group, color in all_type_palette.items():
        handles.append(Patch(facecolor=color, label=f"Type: {group}", alpha=0.75))

    # For status
    for status, color in status_palette.items():
        handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

    # Add legend outside the clustermap
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper left',
        title="Legend",
        frameon=False
    )

    # Format colorbar
    colorbar = g.ax_heatmap.collections[0].colorbar
    colorbar.set_ticks(tick_vals_log)
    colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
    colorbar.set_label("ASV Count", rotation=270, labelpad=15)

    # Force x-axis ticks and labels
    g.ax_heatmap.set_xticks(g.ax_heatmap.get_xticks())
    g.ax_heatmap.set_xticklabels(pivot_log.columns, rotation=90, ha='center')
    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
    g.ax_heatmap.tick_params(axis='x', which='both', length=5)  # <-- this restores the tick *marks*

    plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_{t}_code.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_{t}_code.pdf"), bbox_inches='tight')
    plt.close()

    g = sns.clustermap(
        pivot_log,
        method='ward',
        metric='euclidean',
        col_colors=col_colors_df,
        cmap=cmap,
        vmin=0,
        vmax=vmax_log,
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(.05, .2),
        colors_ratio=0.02,
        figsize=(32, height),
        cbar_pos=(1.02, 0.2, 0.03, 0.4),
        alpha=0.75
        )

    # Create legend entries
    handles = []

    # For Type_Group
    for group, color in all_type_palette.items():
        handles.append(Patch(facecolor=color, label=f"Type: {group}", alpha=0.75))

    # For status
    for status, color in status_palette.items():
        handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

    # Add legend outside the clustermap
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper left',
        title="Legend",
        frameon=False
    )

    # Format colorbar
    colorbar = g.ax_heatmap.collections[0].colorbar
    colorbar.set_ticks(tick_vals_log)
    colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
    colorbar.set_label("ASV Count", rotation=270, labelpad=15)

    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
    g.ax_heatmap.tick_params(axis='x', which='both', length=5)  # <-- this restores the tick *marks*
    g.ax_heatmap.invert_xaxis()

    plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_{t}_clustered.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_{t}_clustered.pdf"), bbox_inches='tight')
    plt.close()

    pivot_df.to_csv(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_{t}.tsv"), sep='\t')


flurp


fam_asv_df = asv_meta_df[['ASV_ID', 'Family_plot']].loc[asv_meta_df['ASV_ID'].isin(sig_isa_asvs)].drop_duplicates()
fam_asv_df['ASV_int'] = [int(x.replace('ASV', '')) for x in fam_asv_df['ASV_ID']]
fam_asv_df = fam_asv_df.sort_values(['ASV_int'])

# Get unique families
families = fam_asv_df['Family_plot'].unique()

# Function to darken or mute an RGB color
def adjust_color(color, factor):
    h, l, s = colorsys.rgb_to_hls(*color)
    l = max(0, min(1, l * factor))  # ensure within bounds
    return colorsys.hls_to_rgb(h, l, s)

# Build combined palette (as from earlier)
base_colors = sns.color_palette('tab20')
darker_colors = [adjust_color(c, 0.6) for c in base_colors]
muted_colors = [adjust_color(c, 1.2) for c in base_colors]
combined_palette = darker_colors + muted_colors

# Assign colors by cycling through the palette
palette = {
    f: combined_palette[i % len(combined_palette)]
    for i, f in enumerate(families)
}

fig, ax = plt.subplots(figsize=(8, len(palette) * 0.4))
for i, (key, color) in enumerate(palette.items()):
    ax.barh(i, 1, color=color)
    ax.text(1.05, i, key, va='center', ha='left', fontsize=24)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 1.5)
ax.set_frame_on(False)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Family_palette.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/metadata/Family_palette.pdf"), bbox_inches='tight')
plt.close()

print(palette)














# HCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from matplotlib import gridspec

asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
asv_stack_df = asv_stack_df.loc[((asv_stack_df['count'] > 0) & (asv_stack_df['ASV_ID'].isin(sig_isa_asvs)))]
asv_tax_df = asv_stack_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)

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
metadata_df.reset_index(inplace=True)
asv_meta_df = asv_tax_df.merge(metadata_df, on='sample', how='left')

#asv_meta_df = asv_meta_df.loc[~asv_meta_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])]
asv_meta_df['Family_plot'] = asv_meta_df["Family"].apply(lambda x: x if x in top10_fam else "Other")

bubble_df = asv_meta_df.groupby(['sample_code', 'Family_plot', 'Type_Group', 'status'])['count'].sum().reset_index()
pivot_df = bubble_df.pivot(index='sample_code', columns='Family_plot', values='count').fillna(0)

Z = linkage(pivot_df, method='ward')
leaves = leaves_list(Z)
ordered_samples = pivot_df.index[leaves].tolist()

# 2. Apply sample_code order to long dataframe
bubble_df['sample_code'] = pd.Categorical(bubble_df['sample_code'], categories=ordered_samples, ordered=True)

# Plot and save dendrogram
plt.figure(figsize=(24, 8))
dendrogram(Z, labels=ordered_samples, color_threshold=0)
plt.xticks(rotation=90)
plt.tight_layout()
#plt.savefig(os.path.join(data_dir, "vsearch_output/diversity/dendrogram_ISA.svg"))
#plt.savefig(os.path.join(data_dir, "vsearch_output/diversity/dendrogram_ISA.pdf"))
plt.close()


fig, ax = plt.subplots(figsize=(24, 6))
sns.scatterplot(
    data=bubble_df,
    x='sample_code',
    y='Family_plot',
    size='count',
    sizes=(20, 400),
    hue='Type_Group',
    palette=all_type_palette,
    edgecolor='black',
    alpha=0.7,
    ax=ax
    )

# No internal margin at ends of x-axis
ax.margins(x=0.01)

# Rotate x-tick labels
plt.setp(ax.get_xticklabels(), rotation=90)

# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0
)

plt.xticks(rotation=90)
plt.tight_layout()
#plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/bubbleplot_ISA.svg"))
#plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/bubbleplot_ISA.pdf"))
plt.close()

# Map sample_code to sample_type
col_meta = bubble_df.drop_duplicates('sample_code')[['sample_code', 'Type_Group', 'status']].set_index('sample_code')

# Map to colors
col_colors_df = pd.DataFrame({
    'Type_Group': col_meta['Type_Group'].map(all_type_palette),
    'status': col_meta['status'].map(status_palette)
    })

pivot_df = asv_meta_df.groupby(['sample_code', 'Family_plot'])['count'
                                ].sum().reset_index().pivot(index='Family_plot',
                                                            columns='sample_code',
                                                            values='count'
                                                            ).fillna(0)
# Log transform (add 1 to avoid log(0))
pivot_log = np.log10(pivot_df + 1)

# Create a custom colormap: white → light gray → black
colors = ['#ffffff', '#d9d9d9', '#000000']  # white → light gray → black
cmap = LinearSegmentedColormap.from_list("light_greyscale", colors, N=256)

# Define your desired tick values (original scale)
tick_vals_orig = [5, 50, 500, 5000, 50000]
# Convert to log scale used in heatmap (log10(count + 1))
tick_vals_log = [np.log10(v + 1) for v in tick_vals_orig]

# Define your max display value
vmax_display = 50000
# Compute log-transformed vmax
vmax_log = np.log10(vmax_display + 1)

g = sns.clustermap(
    pivot_log,
    method='ward',
    metric='euclidean',
    col_colors=col_colors_df,
    cmap=cmap,
    vmin=0,
    vmax=vmax_log,
    linewidths=0.5,
    xticklabels=True,
    yticklabels=True,
    dendrogram_ratio=(.05, .2),
    colors_ratio=0.05,
    figsize=(24, 8),
    cbar_pos=(1.02, 0.2, 0.03, 0.4),
    alpha=0.75
    )

# Create legend entries
handles = []

# For Type_Group
for group, color in all_type_palette.items():
    handles.append(Patch(facecolor=color, label=f"Type: {group}", alpha=0.75))

# For status
for status, color in status_palette.items():
    handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

# Add legend outside the clustermap
plt.legend(
    handles=handles,
    bbox_to_anchor=(1, 1),
    bbox_transform=plt.gcf().transFigure,
    loc='upper left',
    title="Legend",
    frameon=False
)

# Format colorbar
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_ticks(tick_vals_log)
colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
colorbar.set_label("ASV Count", rotation=270, labelpad=15)

plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_ISA_Family.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_ISA_Family.pdf"), bbox_inches='tight')
plt.close()














# HCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from matplotlib import gridspec

asv_stack_df = asv_raw_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
asv_stack_df = asv_stack_df.loc[asv_stack_df['count'] > 0]
asv_tax_df = asv_stack_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)

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
asv_meta_df = asv_tax_df.merge(metadata_df, on='sample', how='left')

#asv_meta_df = asv_meta_df #.loc[~asv_meta_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])]
asv_meta_df['Family_plot'] = asv_meta_df["Family"].apply(lambda x: x if x in top10_fam else "Other")

bubble_df = asv_meta_df.groupby(['sample_code', 'Family_plot'])['count'].sum().reset_index()
pivot_df = bubble_df.pivot(index='sample_code', columns='Family_plot', values='count').fillna(0)

Z = linkage(pivot_df, method='ward')
leaves = leaves_list(Z)
ordered_samples = pivot_df.index[leaves].tolist()

# 2. Apply sample_code order to long dataframe
bubble_df = asv_meta_df.groupby(['sample_code', 'Family_plot', 'Type_Group', 'status'])['count'].sum().reset_index()
bubble_df['sample_code'] = pd.Categorical(bubble_df['sample_code'], categories=ordered_samples, ordered=True)

# Plot and save dendrogram
plt.figure(figsize=(24, 8))
dendrogram(Z, labels=ordered_samples, color_threshold=0)
plt.xticks(rotation=90)
plt.tight_layout()
#plt.savefig(os.path.join(data_dir, "vsearch_output/diversity/dendrogram_wControls.svg"))
#plt.savefig(os.path.join(data_dir, "vsearch_output/diversity/dendrogram_wControls.pdf"))
plt.close()


fig, ax = plt.subplots(figsize=(24, 6))
sns.scatterplot(
    data=bubble_df,
    x='sample_code',
    y='Family_plot',
    size='count',
    sizes=(20, 400),
    hue='Type_Group',
    palette=all_type_palette,
    edgecolor='black',
    alpha=0.7,
    ax=ax
    )

# No internal margin at ends of x-axis
ax.margins(x=0.01)

# Rotate x-tick labels
plt.setp(ax.get_xticklabels(), rotation=90)

# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0
)

plt.xticks(rotation=90)
plt.tight_layout()
#plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/bubbleplot_wControls.svg"))
#plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/bubbleplot_wControls.pdf"))
plt.close()

# Map sample_code to sample_type
col_meta = bubble_df.drop_duplicates('sample_code')[['sample_code', 'Type_Group', 'status']].set_index('sample_code')

# Map to colors
col_colors_df = pd.DataFrame({
    'Type_Group': col_meta['Type_Group'].map(all_type_palette),
    'status': col_meta['status'].map(status_palette)
    })

pivot_df = asv_meta_df.groupby(['sample_code', 'Family_plot'])['count'
                                ].sum().reset_index().pivot(index='Family_plot',
                                                            columns='sample_code',
                                                            values='count'
                                                            ).fillna(0)
# Log transform (add 1 to avoid log(0))
pivot_log = np.log10(pivot_df + 1)

# Create a custom colormap: white → light gray → black
colors = ['#ffffff', '#d9d9d9', '#000000']  # white → light gray → black
cmap = LinearSegmentedColormap.from_list("light_greyscale", colors, N=256)

# Define your desired tick values (original scale)
tick_vals_orig = [5, 50, 500, 5000, 50000]
# Convert to log scale used in heatmap (log10(count + 1))
tick_vals_log = [np.log10(v + 1) for v in tick_vals_orig]

# Define your max display value
vmax_display = 50000
# Compute log-transformed vmax
vmax_log = np.log10(vmax_display + 1)

g = sns.clustermap(
    pivot_log,
    method='ward',
    metric='euclidean',
    col_colors=col_colors_df,
    cmap=cmap,
    vmin=0,
    vmax=vmax_log,
    linewidths=0.5,
    xticklabels=True,
    yticklabels=True,
    dendrogram_ratio=(.05, .2),
    colors_ratio=0.05,
    figsize=(36, 8),
    cbar_pos=(1.02, 0.2, 0.03, 0.4),
    alpha=0.75
    )

# Create legend entries
handles = []

# For Type_Group
for group in ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']:
    color = all_type_palette[group]
    if group != 'Failed-QC':
        handles.append(Patch(facecolor=color, label=f"Type: {group}", alpha=0.75))

# For status
for status, color in status_palette.items():
    handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

# Add legend outside the clustermap
plt.legend(
    handles=handles,
    bbox_to_anchor=(1, 1),
    bbox_transform=plt.gcf().transFigure,
    loc='upper left',
    title="Legend",
    frameon=False
)

# Format colorbar
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_ticks(tick_vals_log)
colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
colorbar.set_label("ASV Count", rotation=270, labelpad=15)

plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_wControls_Family.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/diversity/clustermap_wControls_Family.pdf"), bbox_inches='tight')
plt.close()







#Collectors curve for taxonomy
# Taxonomic ranks in order
ranks = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

# Count ASVs with *any* non-null, non-Unclassified value at each rank
counts = {
    rank: asv_tax_df[rank].replace(['Unclassified', 'NA', None, ''], pd.NA).dropna().shape[0]
    for rank in ranks
}

# Format for plotting
counts_df = pd.DataFrame({
    'Rank': ranks,
    'ASVs_Assigned': [counts[r] for r in ranks]
})

# Optional: % of total ASVs
counts_df['Percent'] = counts_df['ASVs_Assigned'] / counts_df['ASVs_Assigned'].iloc[0] * 100

# Plot
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=counts_df,
    x='Rank',
    y='Percent',
    marker='o',
    linewidth=2,
    color='teal'
)

plt.fill_between(counts_df['Rank'], counts_df['Percent'], alpha=0.2, color='teal')

plt.ylabel('% ASVs Assigned')
plt.xlabel('Taxonomic Rank')
plt.title('Taxonomic Assignment Drop-off Curve')
plt.ylim(0, 105)
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"vsearch_output/taxonomy/rank_curve.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"vsearch_output/taxonomy/rank_curve.pdf"), bbox_inches='tight')
plt.close()
