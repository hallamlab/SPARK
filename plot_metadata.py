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


pd.set_option('display.max_columns', None)


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
    parts = [part.strip().split('__', 1)[1] for part in taxa_str.split(delimiter)]
    
    # In case there are missing levels, fill them with None
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict


# Create output directory if it doesn't exist
output_dir = "vsearch_output/metadata"
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

metadata_table_path = 'ref_db/spark_metadata.tsv'
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t', index_col=0)

fastq_stats_path = 'vsearch_output/stats/fastq_stats.tsv'
fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
fstats_df['sample'] = [x.split('/')[1].split('_L001_R')[0] for x in fstats_df['file']]
reads_df = fstats_df.groupby(['sample'])[['num_seqs', 'sum_len']].sum().reset_index()

alpha_path = 'vsearch_output/diversity/shannon.tsv'
alpha_df = pd.read_csv(alpha_path, header=0, sep='\t')

bray_path = 'vsearch_output/diversity/bray.tsv'
bray_df = pd.read_csv(bray_path, header=0, sep='\t', index_col=0)
bray_reducer, bray_umap = perform_umap(bray_df, random_state=42)

taxonomy_path = 'vsearch_output/taxonomy/ASV_GG2_tax.tsv'
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Sequence_ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Sequence_ID']]
tax_df.set_index('Sequence_ID', inplace=True)

asv_path = 'vsearch_output/ASVs/ASV_filtered.tsv'
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0)
asv_df.columns = [x.rsplit('_', 1)[0] for x in asv_df.columns]

asv_raw_path = 'vsearch_output/ASVs/ASV_counts.tsv'
asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
asv_raw_df.columns = [x.rsplit('_', 1)[0] for x in asv_raw_df.columns]
asv_raw_stack_df = asv_raw_df.stack().reset_index()
asv_raw_stack_df.columns = ['ASV_ID', 'sample', 'raw_count']
asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['raw_count'] > 0]
asv_raw_stack_df.set_index('ASV_ID', inplace=True)
asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='sample', how='left')
asv_raw_cnt_df = asv_raw_meta_df.groupby(['sample'])['raw_count'].sum().reset_index()


asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
asv_stack_df = asv_stack_df.loc[asv_stack_df['count'] > 0]
asv_stack_df.set_index('ASV_ID', inplace=True)
asv_tax_df = asv_stack_df.merge(tax_df, how='left', left_index=True, right_index=True)
taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in asv_tax_df['Taxonomy']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

asv_meta_df = asv_tax_df.merge(metadata_df, on='sample', how='left')
asv_grp_df = asv_meta_df.groupby(['Type_Group', 'Phylum', 'sample', 'Sample'])['count'].sum().reset_index()
cnt_df = asv_meta_df.groupby(['sample'])['count'].sum().reset_index()

# Compute total abundance for each phylum
total_abundance = asv_grp_df.groupby('Phylum')['count'].sum()
# Get the top 10 phyla (by abundance)
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_grp_df['Phylum_plot'] = asv_grp_df["Phylum"].apply(lambda x: x if x in top10 else "Other")


metastat_df = metadata_df.merge(reads_df, how='left', on='sample')
metastat_df = metastat_df.merge(alpha_df, how='left', on='sample')
metastat_df = metastat_df.merge(bray_umap.reset_index(), how='left', on='sample')
metastat_df = metastat_df.merge(cnt_df, how='left', on='sample')
metastat_df = metastat_df.merge(asv_raw_cnt_df, how='left', on='sample')
metastat_df['pass_filter'] = [t if s in list(asv_grp_df['sample']) else 'Failed-QC'
                              for s,t in  zip(metastat_df['sample'], metastat_df['Type_Group'])
                              ]

metastat_df.to_csv('vsearch_output/metadata/master_table.tsv', sep='\t', index=False)
sub_df = metastat_df.loc[~metastat_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])]
sub_df['Type_Group'] = pd.Categorical(sub_df['Type_Group'], ['Lung Brush', 'BAL', 'Oral Rinse'])


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
sample_type_ttests.to_csv('vsearch_output/diversity/alpha_sample_ttest.tsv', sep='\t', index=False)

results = []
for a, b in combinations(sub_df['Case'].unique(), 2):
    group1 = sub_df[sub_df['Case'] == a]['Shannon'].dropna()
    group2 = sub_df[sub_df['Case'] == b]['Shannon'].dropna()
    stat, pval = ttest_ind(group1, group2, equal_var=False)
    results.append({'group1': a, 'group2': b, 'pval': pval, 'tstat': stat})
sample_type_ttests = pd.DataFrame(results)
_, pvals_corrected, _, _ = multipletests(sample_type_ttests['pval'], method='fdr_bh')
sample_type_ttests['pval_adj'] = pvals_corrected
sample_type_ttests['significant'] = sample_type_ttests['pval_adj'] < 0.05  # Boolean
print(sample_type_ttests)
sample_type_ttests.to_csv('vsearch_output/diversity/alpha_case_ttest.tsv', sep='\t', index=False)

all_type_palette = {'Scope Flush': '#6A3D9A',
		   'Skin Brush': '#CC79A7',
		   'Lung Brush': '#009E73',
		   'BAL': '#0072B2',
		   'Oral Rinse': '#E69F00',
           'Failed-QC': 'lightgray'
           }

three_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#E69F00'
           }

case_palette = {'Control':'white', 'Cancer':'#A50026'}


ordered_type = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']
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
sns.swarmplot(
    x='Type_Group', y='raw_count', data=long_df,
    hue='Type_Group', palette=all_type_palette,
    alpha=0.5, dodge=False, ax=ax,
    legend=False
)

# Dashed line at 5k
plt.axhline(y=5000, linestyle='--', color='black', linewidth=1)



plt.title("Sample Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("vsearch_output/metadata/Type_Group_swarmplot.svg")
plt.close()

# Plot using the inverted data
ax = pivot_df.plot(
                    kind='bar',
                    stacked=True,
                    figsize=(10, 10),
                    color=[all_type_palette[col] for col in pivot_df.columns],
                    edgecolor='gray',
                    linewidth=1,
                    alpha=0.5
                    )

handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(label)] for label in col_order]
ordered_labels = [label for label in col_order]
ax.legend(ordered_handles, ordered_labels)

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(f"vsearch_output/metadata/Type_Group_histogram.svg")
plt.close()


metastat_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed']
ordered_type = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']
metastat_df['Type_Group'] = pd.Categorical(metastat_df['Type_Group'], ordered_type)

plt.figure(figsize=(10, 10))
sns.boxplot(data=metastat_df, x="Type_Group", y="num_seqs", hue="Type_Group",
			palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5), order=ordered_type
            )
plt.title("Sample Type")
plt.tight_layout()
plt.savefig(f"vsearch_output/metadata/Read_count_boxplot.svg")
plt.close()

plt.figure(figsize=(10, 10))
sns.boxplot(data=metastat_df, x="Type_Group", y="sum_len", hue="Type_Group",
	        palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5),
            order=ordered_type
            )
plt.title("Sample Type")
plt.tight_layout()
plt.savefig(f"vsearch_output/metadata/Basepair_sum_boxplot.svg")
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=sub_df,
            x="Type_Group", y="Shannon", hue="Type_Group", kind="box",
	        palette=all_type_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Oral Rinse', 'BAL', 'Lung Brush']
            )
plt.tight_layout()
plt.savefig(f"vsearch_output/diversity/Alpha_combined_boxplot.svg")
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=sub_df,
            x="Case", y="Shannon", hue="Case", kind="box",
            palette=case_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Control', 'Cancer']
            )
plt.tight_layout()
plt.savefig(f"vsearch_output/diversity/Alpha_case_boxplot.svg")
plt.close()

plt.figure(figsize=(10, 10))
sns.catplot(data=sub_df,
            x="Case", y="Shannon", hue="Case", col='Type_Group', kind="box",
            palette=case_palette, saturation=1, boxprops=dict(alpha=.5),
            order=['Control', 'Cancer'],
            col_order=['Oral Rinse', 'BAL', 'Lung Brush']
            )
plt.tight_layout()
plt.savefig(f"vsearch_output/diversity/Alpha_sample_case_boxplot.svg")
plt.close()

plt.figure(figsize=(12, 10))

sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group", size='count', sizes=(40, 400),
                palette=three_palette, style="Case",
                alpha=0.5
                )

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.2, 1),
    borderaxespad=0,
    labelspacing=1.25,
    frameon=False
)

plt.tight_layout()
plt.savefig(f"vsearch_output/diversity/Beta_Sample_Case.svg")
plt.close()

plt.figure(figsize=(12, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Case", size='count', sizes=(40, 400),
                palette=case_palette, alpha=0.5,
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
plt.savefig(f"vsearch_output/diversity/Beta_Case.svg")
plt.close()

plt.figure(figsize=(10, 10))
sns.scatterplot(data=sub_df, x="UMAP1", y="UMAP2", hue="Type_Group",
                size='count', sizes=(40, 400), palette=three_palette,
                alpha=0.5
                )

plt.title("Sample Type")
plt.tight_layout()
plt.savefig(f"vsearch_output/diversity/Beta_UMAP_NoControls.svg")
plt.close()


type_groups = asv_grp_df['Type_Group'].unique()

color_palette = {"Blue": "#377eb8",
                 "Orange": "#ff7f00",
                 "Green": "#4daf4a",
                 "Purple": "#984ea3",
                 "Pink": "#e41a1c",
                 "Yellow": "#fdbf6f",
                 "Cyan": "#a6cee3",
                 "Red": "#e31a1c",
                 "Brown": "#8b4513",
                 "Teal": "#008080",
                 "Magenta": "#d62728",
                 "Olive": "#808000",
                 "Turquoise": "#40E0D0",
                 "Lime": "#32CD32",
                 "Gray": "#808080",
                 "Black": "#000000"
                 }

color_map = {'Bacteroidota': 'Blue',
             'Patescibacteria': 'Orange',
             'Firmicutes_D': 'Green',
             'Proteobacteria': 'Red',
             'Fusobacteriota': 'Purple',
             'Firmicutes_C': 'Brown',
             'Thermoplasmatota': 'Pink',
             'Actinobacteriota': 'Gray',
             'Firmicutes_A': 'Olive',
             'Campylobacterota': 'Turquoise',
             'Acidobacteriota': 'Yellow',
             'Planctomycetota': 'Magenta',
             'Verrucomicrobiota': 'Lime',
             'Spirochaetota': 'Teal',
             'Other': 'Black'
             }

for t in type_groups:
    sub_df = asv_grp_df.loc[asv_grp_df['Type_Group'] == t]
    # Pivot the DataFrame: rows are samples, columns are phyla, values are abundances
    pivot_df = sub_df.pivot_table(index='Sample', columns='Phylum_plot',
                                  values='count', aggfunc='sum', fill_value=0
                                  )

    # Reorder the columns: top10 phyla first (in order of overall abundance), then "Other" if it exists
    ordered_cols = [p for p in top10 if p in pivot_df.columns]
    if "Other" in pivot_df.columns:
        ordered_cols.append("Other")
    pivot_df = pivot_df[ordered_cols]

    # Plot using the inverted data
    ax = pivot_df.plot(
        kind='bar',
        stacked=True,
        figsize=(16, 10),
        color=[color_palette[color_map[col]] for col in pivot_df.columns]
    )

    # Fix the y-axis labels to show positive values
    ax.set_yticks(ax.get_yticks())  # Keep the ticks unchanged
    ax.set_yticklabels([str(abs(tick)) for tick in ax.get_yticks()])  # Make labels positive
    
    ax.set_xlabel("Sample")
    ax.set_ylabel("Abundance")
    ax.set_title("Stacked Bar Chart of Top 10 Phyla vs. Other")
    plt.legend(title="Phylum")

    
    # Move the legend outside the plot
    ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()
    plt.savefig(f"vsearch_output/diversity/{t}_count_data.svg")
    plt.close()







# HCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from matplotlib import gridspec

asv_stack_df = asv_df.stack().reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
asv_stack_df = asv_stack_df.loc[asv_stack_df['count'] > 0]
asv_tax_df = asv_stack_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)

taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in asv_tax_df['Taxonomy']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

asv_meta_df = asv_tax_df.merge(metadata_df, on='sample', how='left').reset_index()
asv_meta_df = asv_meta_df.loc[~asv_meta_df['Type_Group'].isin(['Skin Brush', 'Scope Flush'])]
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_meta_df['Phylum_plot'] = asv_meta_df["Phylum"].apply(lambda x: x if x in top10 else "Other")

bubble_df = asv_meta_df.groupby(['sample', 'Phylum_plot', 'Type_Group', 'Case'])['count'].sum().reset_index()
pivot_df = bubble_df.pivot(index='sample', columns='Phylum_plot', values='count').fillna(0)

Z = linkage(pivot_df, method='ward')
leaves = leaves_list(Z)
ordered_samples = pivot_df.index[leaves].tolist()

# 2. Apply sample order to long dataframe
bubble_df['sample'] = pd.Categorical(bubble_df['sample'], categories=ordered_samples, ordered=True)

# Plot and save dendrogram
plt.figure(figsize=(24, 8))
dendrogram(Z, labels=ordered_samples, color_threshold=0)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("vsearch_output/diversity/dendrogram.svg")
plt.close()


fig, ax = plt.subplots(figsize=(24, 6))
sns.scatterplot(
    data=bubble_df,
    x='sample',
    y='Phylum_plot',
    size='count',
    sizes=(20, 400),
    hue='Type_Group',
    palette=three_palette,
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
plt.savefig(f"vsearch_output/diversity/bubbleplot.svg")
plt.close()

# Map sample to sample_type
col_meta = bubble_df.drop_duplicates('sample')[['sample', 'Type_Group', 'Case']].set_index('sample')

# Map to colors
col_colors_df = pd.DataFrame({
    'Type_Group': col_meta['Type_Group'].map(three_palette),
    'Case': col_meta['Case'].map(case_palette)
    })

pivot_df = asv_meta_df.groupby(['sample', 'Phylum_plot'])['count'
                                ].sum().reset_index().pivot(index='Phylum_plot',
                                                            columns='sample',
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
    figsize=(24, 6),
    cbar_pos=(1.02, 0.2, 0.03, 0.4)
    )

# Format colorbar
colorbar = g.ax_heatmap.collections[0].colorbar
colorbar.set_ticks(tick_vals_log)
colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
colorbar.set_label("ASV Count", rotation=270, labelpad=15)

plt.savefig(f"vsearch_output/diversity/clustermap.svg", bbox_inches='tight')
plt.close()