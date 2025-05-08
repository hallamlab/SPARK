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


data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
# Load ASV metadata
metastat_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/metadata/master_table.tsv'), sep='\t')
asv_meta_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/metadata/ASV_meta.tsv'), sep='\t', header=0)
# Load venn diagram data
venn_df = pd.read_csv(os.path.join(data_dir, "vsearch_output/metadata/venn3_presence_table.tsv"), sep="\t", header=0)
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
# Bubbles for the Venn Groups
sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']
height = {'Only Oral Rinse': 4,
          'Only BAL': 4,
          'Only Lung Brush': 4,
          'Oral + BAL': 18,
          'Oral + Lung': 4,
          'BAL + Lung': 4,
          'All Three': 72
          }
ordered_type = ['Oral Rinse', 'BAL', 'Lung Brush']

for vgrp in venn_df['grouping'].unique():
    vgrp_str = vgrp.replace(' ', '_')
    v_asvs = venn_df.loc[venn_df['grouping'] == vgrp]['ASV_ID'].tolist()
    v_spp_df = asv_meta_df.loc[asv_meta_df['ASV_ID'].isin(v_asvs)
                                   ].groupby(['Type_Group', 'Family', 'Genus', 'ASV_ID'])['corr_count'].sum().reset_index(
                                    )
    v_spp_df['Type_Group'] = pd.Categorical(v_spp_df['Type_Group'], ordered_type)
    v_spp_df['Family Genus (ASV)'] = [f'{x} {y} ({z})' for x,y,z in zip(v_spp_df['Family'], v_spp_df['Genus'], v_spp_df['ASV_ID'])]
    v_spp_df.replace(0, np.nan, inplace=True)

    fig, ax = plt.subplots(figsize=(12, height[vgrp]))
    sns.scatterplot(data=v_spp_df, x='Type_Group', y='Family Genus (ASV)',
                    size='corr_count', sizes=(5, 500), palette=all_type_palette,
                    hue_order=ordered_type, hue='Type_Group', alpha=0.75
                    )
    sns.despine(top=True, right=True)

    ax = plt.gca()
    ax.margins(y=0.2)          # remove top/bottom padding
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
    plt.savefig(os.path.join(data_dir, f'vsearch_output/metadata/{vgrp_str}_ASVs_bubbleplot.svg'))
    plt.savefig(os.path.join(data_dir, f'vsearch_output/metadata/{vgrp_str}_ASVs_bubbleplot.pdf'))
    plt.close()








