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
from itertools import combinations, combinations_with_replacement
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import math
from itertools import cycle
import colorsys
import matplotlib.colors as mcolors
import re
import warnings
from matplotlib.colors import to_rgba


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


def pad_ids(ids, prefix, pad_width=None):
    # Extract numeric part
    numbers = [int(re.search(r'\d+', i).group()) for i in ids]
    
    # Auto-determine padding if not provided
    if pad_width is None:
        pad_width = len(str(max(numbers)))
    
    # Return padded ASV IDs
    return [f"{prefix}{num:0{pad_width}d}" for num in numbers]


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
output_dir = os.path.join(data_dir, "spark_old_output/metadata")
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

metadata_table_path = os.path.join(data_dir, 'ref_db/spark_metadata.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]

voc_metadata_path = os.path.join(data_dir, 'ref_db/VOC_table.tsv')
voc_metadata_df = pd.read_csv(voc_metadata_path, sep='\t')

brush_meta_df = voc_metadata_df.merge(metadata_df, left_on='ASV_sample_id', right_on='Sample_renamed', how='left')
brush_meta_df['sample'] = brush_meta_df['Sample_renamed']
brush_meta_df = brush_meta_df.drop(columns=['Type_y'])
brush_meta_df = brush_meta_df.rename(columns={'Type_x': 'Type'})

patient_set = sorted(list(set(brush_meta_df['Participant_ID'])))
patient_dict = {x:i for i,x in enumerate(patient_set)}
metadata_df['patient_code'] = ['P' + str(patient_dict[p]) for p in metadata_df['Participant_ID']]
metadata_df['patient_int'] = [patient_dict[p] for p in metadata_df['Participant_ID']]
metadata_df['type_code'] = [t[0:2] for t in metadata_df['type_group']]
metadata_df['lung_code'] = [l[0] if l[0] in ['R', 'L'] else 'N' for l in metadata_df['Type']]
# Define desired order
patient_order = sorted(list(brush_meta_df['patient_int'].unique()))
type_order = ['Sk', 'Sc', 'Or', 'BA', 'Lu']
lung_order = ['R', 'L', 'N']

# Convert columns to categorical with specified order
brush_meta_df['type_code'] = pd.Categorical(brush_meta_df['type_code'], categories=type_order, ordered=True)
brush_meta_df['lung_code'] = pd.Categorical(brush_meta_df['lung_code'], categories=lung_order, ordered=True)

# Sort the dataframe
brush_meta_df = brush_meta_df.sort_values(['patient_int', 'type_code', 'lung_code'])

# Create unique sample code
metadata_df['sample_code'] = [str(f"S{i+1:03d}") for i in range(len(metadata_df['sample']))]
col = 'sample_code'
metadata_df = metadata_df[[col] + [c for c in metadata_df.columns if c != col]]
metadata_df.drop_duplicates(subset=['sample'], inplace=True)

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

fastq_stats_path = os.path.join(data_dir, 'spark_old_output/stats/fastq_stats.tsv')
fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
fstats_df['sample'] = [str(x.split('/')[-1].rsplit('_', 4)[0]) for x in fstats_df['file']]
reads_df = fstats_df.groupby(['sample'])['num_seqs'].sum().reset_index()

taxonomy_path = os.path.join(data_dir, 'spark_old_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.split(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)

asv_raw_path = os.path.join(data_dir, 'spark_old_output/ASVs/ASV_target.micro.tsv')
asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
asv_raw_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in asv_raw_df.columns]
asv_raw_stack_df = asv_raw_df.stack().reset_index()
asv_raw_stack_df.columns = ['sample', 'ASV_ID', 'raw_count']
asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['raw_count'] > 0]
asv_raw_stack_df.set_index('ASV_ID', inplace=True)
asv_raw_stack_df['sample'] = asv_raw_stack_df['sample'].astype(str)
metadata_df['sample'] = metadata_df['sample'].astype(str)
asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='sample')
asv_raw_cnt_df = asv_raw_meta_df.groupby(['sample'])['raw_count'].sum().reset_index()

asv_path = os.path.join(data_dir, 'spark_old_output/ASVs/ASV_target.micro.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in asv_df.columns]
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

asv_meta_df = asv_tax_df.reset_index().merge(metadata_df, on='sample', how='inner')

cnt_df = asv_meta_df.groupby(['sample'])['count'].sum().reset_index()

metastat_df = metadata_df.merge(reads_df, how='left', on='sample')
metastat_df = metastat_df.merge(cnt_df, how='left', on='sample')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='sample')
metastat_df['pass_filter'] = [s if s in list(asv_meta_df['sample']) else 'Failed-QC'
                              for s in  metastat_df['sample']
                              ]

long_df = metastat_df.groupby(['type_group', 'pass_filter', 'sample'])['raw_count'].sum().reset_index()
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
plt.savefig(os.path.join(data_dir, "spark_old_output/metadata/type_group_swarmplot.svg"))
plt.savefig(os.path.join(data_dir, "spark_old_output/metadata/type_group_swarmplot.pdf"))
plt.close()

sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
scope_samples = list(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['sample'])
scope_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(scope_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
skin_samples = list(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['sample'])
skin_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(skin_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
offtarg_asvs = list(set(scope_asvs + skin_asvs))
keep_cols = ['ASV_ID', 'sample', 'count']
skin_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(skin_asvs)) &
    (asv_meta_df['type_group'].isin(['Skin Brush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
skin_df.columns = ['ASV_ID', 'offtarg_mean']
keep_cols = ['ASV_ID', 'sample', 'count']
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

cleaned_asv_df = asv_meta_df.pivot_table(index='ASV_ID', columns='sample',
                              values='corr_count', aggfunc='sum', fill_value=0
                              )
asv_keep_list = list(asv_meta_df.loc[asv_meta_df['Domain'] != 'Unassigned']['ASV_ID'].unique())
final_asv_df = cleaned_asv_df[[x for x in cleaned_asv_df.columns if x in list(sub_df['sample'])]]
final_asv_df = final_asv_df.loc[asv_keep_list]
final_asv_df = final_asv_df.loc[~(final_asv_df == 0).all(axis=1)]

# Extract numeric parts and find max
max_val = max(int(re.search(r'\d+', asv).group()) for asv in tax_df.index.tolist())
# Get pad width
pad_width = len(str(max_val))

new_asv_meta_df = asv_meta_df.copy()
#new_asv_meta_df['ASV_ID'] = pad_asv_ids(new_asv_meta_df['ASV_ID'].tolist(), pad_width=pad_width) 
#final_asv_df.index = pad_asv_ids(final_asv_df.index.tolist(), pad_width=pad_width) 

new_tax_df = tax_df.copy()
#new_tax_df.index = pad_asv_ids(new_tax_df.index.tolist(), pad_width=pad_width) 
new_tax_df = new_tax_df.reset_index()
new_tax_df.columns = ['ASV_ID', 'Taxon', 'Concensus']

new_tax_df.to_csv(os.path.join(data_dir, 'spark_old_output/metadata/taxonomy_updated.tsv'), sep='\t', index=False)
new_asv_meta_df.to_csv(os.path.join(data_dir, 'spark_old_output/metadata/ASV_meta.tsv'), sep='\t', index=False)
final_asv_df.to_csv(os.path.join(data_dir, 'spark_old_output/ASVs/ASV_final.micro.tsv'), sep='\t', index=True)
metastat_df.to_csv(os.path.join(data_dir, 'spark_old_output/metadata/master_table.tsv'), sep='\t', index=False)
metadata_df.to_csv(os.path.join(data_dir, 'spark_old_output/metadata/metadata_updated.tsv'), sep='\t', index=False)

# Map to colors
m_df = metastat_df.loc[metastat_df['sample'].isin(final_asv_df.columns)].set_index('sample')
filtered_asv_df = final_asv_df[m_df.index.tolist()]
#invert_bray_df = 1 - bray_df.loc[m_df.index.tolist(), m_df.index.tolist()]

col_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    #'kit': m_df['kit'].map(kit_pallete)
}, index=m_df.index)
row_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    #'kit': m_df['kit'].map(kit_pallete)
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
#for group in ['HostZERO-DEP', 'HostZERO-NODEP', 'SPARK-ZYMO']:
#    color = kit_pallete[group]
#    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
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
plt.savefig(os.path.join(data_dir, f"spark_old_output/metadata/clustermap_ASVpercent.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/metadata/clustermap_ASVpercent.pdf"), bbox_inches='tight')
plt.close()

# Violins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import combinations, combinations_with_replacement
from matplotlib.colors import to_rgba

def plot_grouppair_violins_sns(
    shared_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
    group_col: str = "sample_type",
    include_within: bool = True,
    group_order: list | None = None,
    group_colors: dict | None = None,   # {"GroupA":"#1b9e77", "GroupB":"#d95f02", ...}
    title: str = "Group pair % shared",
    ylabel: str = "% shared",
    inner: str = "quartile",            # seaborn violin inner: "box", "quartile", "point", None
    cut: float = 0,                      # 0 = trim to data range
    bw: str | float = "scott",
    scale: str = "width",
    ax=None,
):
    """
    Seaborn violins of %shared for non-redundant group pairs.
    - shared_df: square DataFrame (samples x samples) of float percent values.
    - meta_df:   DataFrame with sample->group mapping.
    - group_colors: dict group -> color. Missing groups auto-colored; invalid colors warn.
    - Only samples present in BOTH the square matrix (index & columns) and metadata are used.
      Duplicate samples in metadata are dropped (first occurrence kept).
    Returns (ax, tidy_df).
    """

    # -------- align & sanitize --------
    # (1) ensure square overlap between index and columns
    matrix_samples = [s for s in shared_df.index if s in shared_df.columns]
    if not matrix_samples:
        raise ValueError("Matrix has no overlapping index/column sample names.")

    # (2) clean metadata & drop duplicate sample rows
    meta = meta_df.copy()
    meta[sample_id_col] = meta[sample_id_col].astype(str).str.strip()
    meta[group_col] = meta[group_col].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[sample_id_col], keep="first")

    # (3) intersect with metadata samples
    meta = meta[meta[sample_id_col].isin(matrix_samples)]
    if meta.empty:
        raise ValueError("No overlapping samples between matrix and metadata.")

    # (4) final ordered sample list: preserve matrix row order, ensure in meta
    samples = [s for s in matrix_samples if s in set(meta[sample_id_col])]
    shared = shared_df.loc[samples, samples]

    # (5) reorder meta to match 'samples'
    meta = meta.set_index(sample_id_col).loc[samples].reset_index()

    # -------- groups & ordering --------
    # appearance order from filtered metadata
    seen = set(); groups_in_use = []
    for g in meta[group_col]:
        if g not in seen:
            seen.add(g); groups_in_use.append(g)

    if group_order:
        specified = [str(g).strip() for g in group_order if str(g).strip()]
        unknown = [g for g in specified if g not in groups_in_use]
        if unknown:
            warnings.warn(f"Unknown groups in group_order {unknown}; ignored.")
        groups = [g for g in specified if g in groups_in_use] + [g for g in groups_in_use if g not in specified]
    else:
        groups = groups_in_use

    # map group -> samples (order respects 'samples')
    group_to_samples = {
        g: [s for s in samples if meta.loc[meta[sample_id_col]==s, group_col].iloc[0] == g]
        for g in groups
    }

    # -------- color resolution (by group) --------
    def _rgba_or_none(c, g):
        try:
            return to_rgba(c)
        except ValueError:
            warnings.warn(f"Ignoring invalid color '{c}' for group '{g}'.")
            return None

    resolved_group_colors = {}
    user_map = group_colors or {}

    # warn if user passes colors for groups not present
    extras = [g for g in user_map if g not in groups]
    if extras:
        warnings.warn(f"group_colors provided for unknown groups {extras}; ignored.")

    # take valid user colors
    for g in groups:
        c = user_map.get(g, None)
        if c is not None:
            rgba = _rgba_or_none(c, g)
            if rgba is not None:
                resolved_group_colors[g] = rgba

    # auto-assign for missing groups
    cmap = plt.get_cmap("tab20")
    auto_i = 0
    for g in groups:
        if g not in resolved_group_colors:
            resolved_group_colors[g] = cmap(auto_i % cmap.N)
            auto_i += 1
            warnings.warn(f"No color for group '{g}'; using auto color from 'tab20'.")

    def _blend(a, b):
        a = np.array(a); b = np.array(b)
        mix = (a + b) / 2.0
        mix[3] = max(a[3], b[3])
        return tuple(mix)

    # -------- collect values per group-pair --------
    gpairs = combinations_with_replacement(groups, 2) if include_within else combinations(groups, 2)

    rows = []        # build tidy df rows
    pair_labels = [] # to preserve order for plotting
    pair_colors = {} # palette for seaborn keyed by pair label

    for g1, g2 in gpairs:
        s1 = group_to_samples.get(g1, [])
        s2 = group_to_samples.get(g2, [])
        if not s1 or not s2:
            continue

        if g1 == g2:
            if len(s1) < 2:
                continue
            sub = shared.loc[s1, s1].to_numpy()
            iu = np.triu_indices(len(s1), k=1)
            vals = sub[iu]
            col = resolved_group_colors[g1]
        else:
            vals = shared.loc[s1, s2].to_numpy().ravel()
            col = _blend(resolved_group_colors[g1], resolved_group_colors[g2])

        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        label = f"{g1} × {g2}"
        pair_labels.append(label)
        pair_colors[label] = col
        rows.append(pd.DataFrame({"pair": label, "value": vals}))

    if not rows:
        raise ValueError("No pairwise values to plot (after alignment/dedup).")

    tidy = pd.concat(rows, ignore_index=True)

    # -------- plot with seaborn --------
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.3*len(pair_labels)), 4.5), dpi=150)

    sns.violinplot(
        data=tidy,
        x="pair",
        y="value",
        order=pair_labels,              # keep deterministic pair order
        palette=pair_colors,            # per-pair colors (within=group color, between=blend)
        cut=cut,
        bw=bw,
        scale=scale,
        inner=inner,
        ax=ax,
    )

    ax.set_xlabel("")                  # x labels already categorical
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.5)

    return ax, tidy

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
ax, tidy = plot_grouppair_violins_sns(
    shared_percent, metadata_df,
    sample_id_col="sample",
    group_col="type_group",
    include_within=True,
    group_order=["Oral Rinse","BAL","Lung Brush"],
    group_colors=type_palette,   # used for within; between are blended
    title="ASVs Shared by Sample Type",
    ylabel="ASVs Shared (%)",
    inner="quartile",            # or "box", "point", None
    cut=0,
    ax=ax,
)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, f"spark_old_output/metadata/violin_ASVpercent.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/metadata/violin_ASVpercent.pdf"), bbox_inches='tight')
plt.close()

flurp







# MITOCHONDRIAL
asv_raw_path = os.path.join(data_dir, 'spark_old_output/mito/ASVs/ASV_target.mito.tsv')
asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
asv_raw_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in asv_raw_df.columns]
asv_raw_stack_df = asv_raw_df.stack().reset_index()
asv_raw_stack_df.columns = ['ASV_ID', 'sample', 'raw_count']
asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['raw_count'] > 0]
asv_raw_stack_df.set_index('ASV_ID', inplace=True)
asv_raw_stack_df['sample'] = asv_raw_stack_df['sample'].astype(str)
metadata_df['sample'] = metadata_df['sample'].astype(str)
asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='sample')
asv_raw_cnt_df = asv_raw_meta_df.groupby(['sample'])['raw_count'].sum().reset_index()

asv_path = os.path.join(data_dir, 'spark_old_output/mito/ASVs/ASV_target.mito.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in asv_df.columns]
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

asv_meta_df = asv_tax_df.reset_index().merge(metadata_df, on='sample', how='inner')

cnt_df = asv_meta_df.groupby(['sample'])['count'].sum().reset_index()
metastat_df = metadata_df.merge(reads_df, how='left', on='sample')
metastat_df = metastat_df.merge(cnt_df, how='left', on='sample')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='sample')
metastat_df['pass_filter'] = [s if s in list(asv_meta_df['sample']) else 'Failed-QC'
                              for s in  metastat_df['sample']
                              ]

long_df = metastat_df.groupby(['type_group', 'pass_filter', 'sample'])['raw_count'].sum().reset_index()
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
plt.savefig(os.path.join(data_dir, "spark_old_output/mito/metadata/type_group_swarmplot_mito.svg"))
plt.savefig(os.path.join(data_dir, "spark_old_output/mito/metadata/type_group_swarmplot_mito.pdf"))
plt.close()

sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
scope_samples = list(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['sample'])
scope_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(scope_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
skin_samples = list(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['sample'])
skin_asvs = asv_meta_df.loc[((asv_meta_df['sample'].isin(skin_samples)) & (asv_meta_df['count'] > 0))]['ASV_ID'].tolist()
offtarg_asvs = list(set(scope_asvs + skin_asvs))
keep_cols = ['ASV_ID', 'sample', 'count']
skin_df = asv_meta_df.loc[(
    (asv_meta_df['ASV_ID'].isin(skin_asvs)) &
    (asv_meta_df['type_group'].isin(['Skin Brush']))
    )].copy()[keep_cols].groupby(['ASV_ID'])['count'].mean().reset_index().fillna(0)
skin_df.columns = ['ASV_ID', 'offtarg_mean']
keep_cols = ['ASV_ID', 'sample', 'count']
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
cleaned_asv_df = asv_meta_df.pivot_table(index='ASV_ID', columns='sample',
                              values='count', aggfunc='sum', fill_value=0
                              )
asv_keep_list = list(asv_meta_df.loc[asv_meta_df['Domain'] != 'Unassigned']['ASV_ID'].unique())
final_asv_df = cleaned_asv_df[[x for x in cleaned_asv_df.columns if x in list(sub_df['sample'])]]
final_asv_df = final_asv_df.loc[asv_keep_list]
final_asv_df = final_asv_df.loc[~(final_asv_df == 0).all(axis=1)]

#asv_meta_df['ASV_ID'] = pad_asv_ids(asv_meta_df['ASV_ID'].tolist(), pad_width=pad_width)  
#final_asv_df.index = pad_asv_ids(final_asv_df.index.tolist(), pad_width=pad_width) 

asv_meta_df.to_csv(os.path.join(data_dir, 'spark_old_output/mito/metadata/ASV_meta_mito.tsv'), sep='\t', index=False)
final_asv_df.to_csv(os.path.join(data_dir, 'spark_old_output/mito/ASVs/ASV_final.mito.tsv'), sep='\t', index=True)
metastat_df.to_csv(os.path.join(data_dir, 'spark_old_output/mito/metadata/master_table_mito.tsv'), sep='\t', index=False)
metadata_df.to_csv(os.path.join(data_dir, 'spark_old_output/mito/metadata/metadata_updated_mito.tsv'), sep='\t', index=False)
print(final_asv_df.shape)

# Map to colors
m_df = metastat_df.loc[metastat_df['sample'].isin(final_asv_df.columns)].set_index('sample')
filtered_asv_df = final_asv_df[m_df.index.tolist()]
#invert_bray_df = 1 - bray_df.loc[m_df.index.tolist(), m_df.index.tolist()]

col_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    #'kit': m_df['kit'].map(kit_pallete)
}, index=m_df.index)
row_colors_df = pd.DataFrame({
    'sample_type': m_df['type_group'].map(all_type_palette),
    'status': m_df['status'].map(status_palette),
    #'kit': m_df['kit'].map(kit_pallete)
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
#for group in ['HostZERO-DEP', 'HostZERO-NODEP', 'SPARK-ZYMO']:
#    color = kit_pallete[group]
#    handles.append(Patch(facecolor=color, label=f"{group}", alpha=0.75))
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
plt.savefig(os.path.join(data_dir, f"spark_old_output/mito/metadata/clustermap_ASVpercent_mito.svg"), bbox_inches='tight')
plt.savefig(os.path.join(data_dir, f"spark_old_output/mito/metadata/clustermap_ASVpercent_mito.pdf"), bbox_inches='tight')
plt.close()