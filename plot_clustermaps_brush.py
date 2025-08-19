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



data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'

brush_palette = {'ca-lung': '#009E73',
           'ca-contra': '#0072B2',
           'ctrl-brush': '#6A3D9A'
           }

status_palette = {'Non-Cancer':'white',
                  'Cancer':'#A50026',
                  'methods':'lightgray'
                  }

kit_pallete = {'HostZERO-DEP': 'black',
               'HostZERO-NODEP': 'gray',
               'SPARK-ZYMO': 'skyblue',
               }

ordered_type = ['ctrl-brush', 'ca-contra', 'ca-lung']

metadata_table_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/metadata_updated.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')

asv_meta_df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/metadata/ASV_meta.tsv'), sep='\t', header=0)

# clustermaps
isa_path = os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Type_status_ISA_results.tsv')
isa_df = pd.read_csv(isa_path, sep='\t')
sig_isa_df = isa_df.loc[((isa_df['type_significance'] == True) | (isa_df['status_significance'] == True)) &
                        ((isa_df['type_stat'] >= 0.6) | (isa_df['status_stat'] >= 0.6))
                        ]
sig_isa_asvs = list(sig_isa_df['ASV_ID'])
asv_meta_df = asv_meta_df.loc[~asv_meta_df['subclass2'].isin(['No VOCs'])]

rank_type_dict = {}
rank_dict = {}
for rank in ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'ASV_ID']:
    asv_rank_df = asv_meta_df.groupby(['subclass2', rank, 'sample_code', 'sample'])['corr_count'].sum().reset_index()
    rank_type_dict[rank] = {}
    rank_dict[rank] = []
    if rank == 'ASV_ID':
        N = 30
    else:
        N = 30
    for group in asv_meta_df['subclass2'].unique():
        df_group = asv_rank_df[asv_rank_df['subclass2'] == group]
        total_rank = df_group.groupby(rank)['corr_count'].sum()
        topN = total_rank.sort_values(ascending=False).head(N).index.tolist()
        sig = asv_meta_df[asv_meta_df['ASV_ID'].isin(sig_isa_asvs)][rank].unique().tolist()
        All_r = list(set(topN + sig))
        rank_type_dict[rank][group] = All_r
        rank_dict[rank] = list(set(rank_dict[rank] + All_r))

for rank in ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'ASV_ID']:
    rank_list = rank_dict[rank]
    plot_col = f"{rank}_plot"
    asv_meta_df[plot_col] = asv_meta_df[rank].apply(lambda x: x if x in rank_list else "Other")

for t in ['Phylum_plot', 'Class_plot', 'Order_plot', 'Family_plot',
          'Genus_plot', 'Species_plot', 'ASV_ID_plot'
          ]:
    bubble_df = asv_meta_df.groupby(['sample_code', t, 'subclass2', 'status', 'kit'])['corr_count'].sum().reset_index()
    pivot_df = bubble_df.pivot(index='sample_code', columns=t, values='corr_count').fillna(0)

    # Map sample_code to sample_type
    col_meta = bubble_df.drop_duplicates('sample_code')[['sample_code', 'subclass2', 'status', 'kit']].set_index('sample_code')

    # Map to colors
    col_colors_df = pd.DataFrame({
        'subclass2': col_meta['subclass2'].map(brush_palette),
        'status': col_meta['status'].map(status_palette),
        #'kit': col_meta['kit'].map(kit_pallete)
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
    num_rows = len(pivot_df.index.values)
    print(f"Number of taxa for {t}: {num_rows}")
    #if t == 'ASV_ID_plot':
    #    height = 120
    #else:
    height = max(8, min(0.4 * num_rows, 50))  # auto-scale with sane bounds
    
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
        figsize=(24, height),
        cbar_pos=(1.02, 0.2, 0.03, 0.4),
        alpha=0.75,
        col_cluster=False
        )

    # Create legend entries
    handles = []

    for group in ordered_type:
        color = brush_palette[group]
        handles.append(Patch(facecolor=color, label=f"Type: {group}", alpha=0.75))

    # For status
    for status, color in status_palette.items():
        handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

    # For kit
    #for kit, color in kit_pallete.items():
    #    handles.append(Patch(facecolor=color, label=f"kit: {kit}", alpha=0.75))

    # Add legend outside the clustermap
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper left',
        title="Sample Subclass2 / Cancer Status",
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

    plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/clustermap_{t}_code.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/clustermap_{t}_code.pdf"), bbox_inches='tight')
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
        figsize=(24, height),
        cbar_pos=(1.02, 0.2, 0.03, 0.4),
        alpha=0.75
        )

    # Create legend entries
    handles = []

    # For subclass2
    for group in ordered_type:
        color = brush_palette[group]
        handles.append(Patch(facecolor=color, label=f"subclass2: {group}", alpha=0.75))

    # For status
    for status, color in status_palette.items():
        handles.append(Patch(facecolor=color, label=f"status: {status}", alpha=0.75))

    # For kit
    #for kit, color in kit_pallete.items():
    #    handles.append(Patch(facecolor=color, label=f"kit: {kit}", alpha=0.75))

    # Add legend outside the clustermap
    plt.legend(
        handles=handles,
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure,
        loc='upper left',
        title="Sample Subclass2 / Cancer Status",
        frameon=False
    )

    # Format colorbar
    colorbar = g.ax_heatmap.collections[0].colorbar
    colorbar.set_ticks(tick_vals_log)
    colorbar.set_ticklabels([f"{v:,}" for v in tick_vals_orig])
    colorbar.set_label("ASV Count", rotation=270, labelpad=15)

    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
    g.ax_heatmap.tick_params(axis='x', which='both', length=5)

    plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/clustermap_{t}_clustered.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(data_dir, f"spark_old_output/brush/diversity/clustermap_{t}_clustered.pdf"), bbox_inches='tight')
    plt.close()

    pivot_df.to_csv(os.path.join(data_dir, f"spark_old_output/brush/diversity/clustermap_{t}.tsv"), sep='\t')