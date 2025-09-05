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
# Load ASV metadata
metastat_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/master_table_TYPE.tsv'), sep='\t')
asv_meta_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/ASV_meta_TYPE.tsv'), sep='\t', header=0)
# Load venn diagram data
venn_df = pd.read_csv(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_presence_table_TYPE.tsv"), sep="\t", header=0)
#venn_kit_df = pd.read_csv(os.path.join(data_dir, "spark_methods_output_tester/metadata/venn3_presence_table_kit.tsv"), sep="\t", header=0)
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

kit_pallete = {'HostZERO-DEP': 'black',
               'HostZERO-NODEP': 'gray',
               'SPARK-ZYMO': 'skyblue',
               }
keep_types = [
                  'Oral Rinse',
                  'BAL',
                  'Lung Brush'
                  ]
               

status_palette = {'Non-Cancer':'white',
                  'Cancer':'#A50026',
                  'methods':'lightgray'
                  }
                  
# Bubbles for the Venn Groups
sub_df = metastat_df.loc[metastat_df['pass_filter'] != 'Failed-QC']

ordered_type = keep_types
venn_tax_dfs = []
for vgrp in venn_df['grouping'].unique():
    print(f"Processing Venn group: {vgrp}")
    vgrp_str = vgrp.replace(' ', '_')
    v_asvs = venn_df.loc[venn_df['grouping'] == vgrp]['ASV_ID'].tolist()
    v_spp_df = asv_meta_df.loc[asv_meta_df['ASV_ID'].isin(v_asvs)
                                   ].groupby(['type_group', 'Family', 'Genus'], dropna=False)['corr_count'].sum().reset_index(
                                    )

    v_spp_df['type_group'] = pd.Categorical(v_spp_df['type_group'], ordered_type)
    v_spp_df['Family Genus'] = [f'{x} {y}' for x,y in 
             zip(v_spp_df['Family'], v_spp_df['Genus'])
             ]
    v_spp_df.replace(0, np.nan, inplace=True)
    venn_tax_dfs.append(v_spp_df)

    num_rows = len(v_spp_df['Family Genus'].unique())
    fig_height = max(4, min(0.4 * num_rows, 15))  # auto-scale with sane bounds
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    sns.scatterplot(data=v_spp_df, x='type_group', y='Family Genus',
                    size='corr_count', sizes=(5, 500), palette=three_palette,
                    hue_order=ordered_type, hue='type_group', alpha=0.75
                    )
    sns.despine(top=True, right=True)

    ax = plt.gca()
    ax.margins(y=0.2)          # remove top/bottom padding
    
    # Move legend outside
    ax.legend(
        title='Sample Type',
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=False
    )
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(data_dir, f'spark_methods_output_tester/metadata/{vgrp_str}_Genus_bubbleplot_TYPE.svg'))
    plt.savefig(os.path.join(data_dir, f'spark_methods_output_tester/metadata/{vgrp_str}_Genus_bubbleplot_TYPE.pdf'))
    plt.close()
venn_tax_df = pd.concat(venn_tax_dfs)
venn_tax_df.to_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/Three_types_venn_presence_tax_TYPE.tsv'), sep='\t')







flurp
ordered_type = ['HostZERO-DEP', 'HostZERO-NODEP', 'SPARK-ZYMO']
venn_tax_dfs = []
for vgrp in venn_kit_df['grouping'].unique():
    vgrp_str = vgrp.replace(' ', '_')
    v_asvs = venn_kit_df.loc[venn_kit_df['grouping'] == vgrp]['ASV_ID'].tolist()
    v_spp_df = asv_meta_df.loc[asv_meta_df['ASV_ID'].isin(v_asvs)
                                   ].groupby(['type_group', 'Family', 'Genus'])['corr_count'].sum().reset_index(
                                    )
    v_spp_df['type_group'] = pd.Categorical(v_spp_df['type_group'], ordered_type)
    v_spp_df['Family Genus'] = [f'{x} {y}' for x,y in
                                zip(v_spp_df['Family'], v_spp_df['Genus'])]
    v_spp_df.replace(0, np.nan, inplace=True)
    venn_tax_dfs.append(v_spp_df)

    num_rows = len(v_spp_df['Family Genus'].unique())
    fig_height = max(4, min(0.4 * num_rows, 15))  # auto-scale with sane bounds
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    sns.scatterplot(data=v_spp_df, x='type_group', y='Family Genus',
                    size='corr_count', sizes=(5, 500), palette=three_palette,
                    hue_order=ordered_type, hue='type_group', alpha=0.75
                    )
    sns.despine(top=True, right=True)

    ax = plt.gca()
    ax.margins(y=0.2)          # remove top/bottom padding
    
    # Move legend outside
    ax.legend(
        title='Sample Type',
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=False
    )
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(data_dir, f'spark_methods_output_tester/metadata/{vgrp_str}_Genus_bubbleplot_type_group.svg'))
    plt.savefig(os.path.join(data_dir, f'spark_methods_output_tester/metadata/{vgrp_str}_Genus_bubbleplot_type_group.pdf'))
    plt.close()
venn_tax_df = pd.concat(venn_tax_dfs)
venn_tax_df.to_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/venn3_presence_table_type_group_tax.tsv'), sep='\t')




