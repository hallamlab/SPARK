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
metadata_df['lung_code'] = [l[0] if l[0] in ['R', 'L'] else 'N' for l in metadata_df['Type']]
# Define desired order
patient_order = sorted(list(metadata_df['patient_int'].unique()))
type_order = ['Sk', 'Sc', 'Or', 'BA', 'Lu']
lung_order = ['R', 'L', 'N']
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
bray_reducer, bray_umap = perform_umap(data=bray_df,
                                       n_neighbors=30,
                                       min_dist=0.01,
                                       random_state=42,
                                       precomputed=True
                                       )

jacc_path = os.path.join(data_dir, 'vsearch_output/diversity/jaccard.tsv')
jacc_df = pd.read_csv(jacc_path, header=0, sep='\t', index_col=0)
jacc_reducer, jacc_umap = perform_umap(data=jacc_df,
                                       n_neighbors=30,
                                       min_dist=0.01,
                                       random_state=42,
                                       precomputed=True
                                       )
jacc_umap.columns = ['Jacc_UMAP1', 'Jacc_UMAP2']

taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)

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
metastat_df = metastat_df.merge(jacc_umap.reset_index(), how='left', on='sample')
metastat_df = metastat_df.merge(cnt_df, how='left', on='sample')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='sample')
metastat_df['pass_filter'] = [t if s in list(asv_meta_df['sample']) else 'Failed-QC'
                              for s,t in  zip(metastat_df['sample'], metastat_df['Type_Group'])
                              ]

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
scope_spp_df.replace(0, np.nan, inplace=True)
fig, ax = plt.subplots(figsize=(12, 12))
sns.scatterplot(data=scope_spp_df, x='Type_Group', y='Family Genus (ASV)',
                size='count', sizes=(5, 500), palette=all_type_palette,
                hue_order=ordered_type, hue='Type_Group', alpha=0.75
                )
sns.despine(top=True, right=True)

ax = plt.gca()
ax.margins(y=0.1)          # remove top/bottom padding
plt.tight_layout()

# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=False
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
skin_spp_df.replace(0, np.nan, inplace=True)

fig, ax = plt.subplots(figsize=(12, 18))
sns.scatterplot(data=skin_spp_df, x='Type_Group', y='Family Genus (ASV)',
                size='count', sizes=(5, 500), palette=all_type_palette,
                hue_order=ordered_type, hue='Type_Group', alpha=0.75
                )
sns.despine(top=True, right=True)

ax = plt.gca()
ax.margins(y=0.1)          # remove top/bottom padding
plt.tight_layout()

# Move legend outside
ax.legend(
    title='Sample Type',
    bbox_to_anchor=(1.01, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=False
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

offtarg_asvs = list(set(scope_asvs + skin_asvs))
keep_cols = ['ASV_ID', 'sample', 'patient_code', 'count']
skin_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(skin_asvs)) &
    (asv_meta_df['Type_Group'].isin(['Skin Brush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
skin_df.columns = ['ASV_ID', 'offtarg_mean']

keep_cols = ['ASV_ID', 'sample', 'count']
scope_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(scope_asvs)) &
    (asv_meta_df['Type_Group'].isin(['Scope Flush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
scope_df.columns = ['ASV_ID', 'nctrl_mean']

asv_meta_df = asv_meta_df.merge(skin_df, how='left', on=['ASV_ID'])
asv_meta_df['offtarg_mean'] = asv_meta_df['offtarg_mean'].fillna(0)

asv_meta_df = asv_meta_df.merge(scope_df, how='left', on='ASV_ID')
asv_meta_df['nctrl_mean'] = asv_meta_df['nctrl_mean'].fillna(0)

asv_meta_df['count_sub_scope'] = asv_meta_df['count'] - asv_meta_df['nctrl_mean']
asv_meta_df['count_sub_skin'] = asv_meta_df['count_sub_scope'] - asv_meta_df['offtarg_mean']
asv_meta_df['corr_count'] = [int(x) if x > 0 else int(0) for x in asv_meta_df['count_sub_skin']]
asv_meta_df = asv_meta_df.loc[~asv_meta_df['Type_Group'].isin(['Scope Flush', 'Skin Brush'])]
cleaned_asv_df = asv_meta_df.pivot_table(index='ASV_ID', columns='sample',
                              values='corr_count', aggfunc='sum', fill_value=0
                              )
asv_keep_list = list(asv_meta_df.loc[asv_meta_df['Domain'] != 'Unassigned']['ASV_ID'].unique())
final_asv_df = cleaned_asv_df[[x for x in cleaned_asv_df.columns if x in list(sub_df['sample'])]]
final_asv_df = final_asv_df.loc[asv_keep_list]
final_asv_df = final_asv_df.loc[~(final_asv_df == 0).all(axis=1)]

asv_meta_df.to_csv(os.path.join(data_dir, 'vsearch_output/metadata/ASV_meta.tsv'), sep='\t', index=False)
asv_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_final.micro.tsv')
final_asv_df.to_csv(asv_path, sep='\t', index=True)
metastat_df.to_csv(os.path.join(data_dir, 'vsearch_output/metadata/master_table.tsv'), sep='\t', index=False)
metadata_df.to_csv(os.path.join(data_dir, 'vsearch_output/metadata/metadata_updated.tsv'), sep='\t', index=False)
print(final_asv_df.shape)