import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import seaborn as sns
import matplotlib.patches as mpatches
import colorsys
from matplotlib import font_manager as fm, rcParams


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


def plot_volcano(df, ind, cmap, p_thresh=0.05, stat_thresh=0.0,
                 output_file='volcano_plot.svg', no_sig=False
                 ):
    # Load data
    
    # Compute log-transformed p-values
    df['log_p'] = -np.log10(df['p.value']).round(1)
    
    # Define colors based on thresholds
    if no_sig:
        df['significance'] = True
    else:
        df['significance'] = False  # Default color for non-significant
        df.loc[((df['p.value'] < p_thresh) & (df['stat'] > stat_thresh)), 'significance'] = True 
    
    df['color'] = [cmap[ind[i]] if s else 'lightgrey' for i,s in zip(df['index'], df['significance'])]
    cmap['not_indicator'] = 'lightgrey'
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot non-red points first
    #non_sig = df[df['significance'] == False]
    #sns.swarmplot(data=non_sig, x='stat', y='log_p', color='gray', orient="h",
    #            dodge=True, ax=ax, alpha=0.75, s=1, legend=False
    #            )
    #plt.scatter(non_sig['stat'], non_sig['log_p'], c=non_sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25,
    #            s=10
    #            )
    
    # Then plot red points on top
    sig = df #[df['significance'] == True]
    palette = dict(zip(sig['color'], sig['color']))

    ax= sns.stripplot(data=sig, x='stat', y='log_p', hue='color', orient="h",
                  dodge=True, ax=ax, alpha=0.75, legend=False, palette=palette,
                  jitter=True, size=5, linewidth=0.25, edgecolor='gray'
                  )
    # Set edgecolor for all swarm points
#    for collection in ax.collections:
#        collection.set_edgecolor('gray')
#        collection.set_linewidth(0.5)

    #plt.scatter(sig['stat'], sig['log_p'], c=sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25,
    #            s=75
    #            )
    
    # Add reference lines
    #plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    # Create legend handles with light grey borders
    legend_handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='lightgrey', linewidth=0.5, label=type)
        for type, color in cmap.items()
    ]

    # Add legend outside plot
    plt.legend(
        handles=legend_handles,
        title='ISA type',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")
    
    # Round limits for nice ticks
    xmin = 0
    xmax = np.ceil(df['stat'].max() * 10) / 10  # e.g., 0.87 → 0.9
    # Generate ticks every 0.1
    xticks = np.arange(xmin, xmax + 0.01, 0.1)  # add 0.01 to ensure inclusion
    # Set ticks and limits
    ax.set_xticks(xticks)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelsize=10, bottom=True)
    ax.invert_yaxis()
    # Get current ticks and keep every other one
    current_ticks = plt.yticks()[0]
    plt.yticks(current_ticks[::2])
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', which='both', length=4, width=1, color='black', left=True, right=False)

    # Optional: fix layout
    fig.subplots_adjust(bottom=0.15)

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()

    return df


def plot_type_taxa(df, ind, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):

    # List of unique Phyla in your data
    phyla = df['Phylum'].unique()

    # Generate color palette (qualitative)
    palette = sns.color_palette('tab20', len(phyla))  # or 'Set3', 'Paired', etc.

    # Map phylum to color
    phylum_color_dict = dict(zip(phyla, palette))
    phylum_color_dict['not_indicator'] = "lightgrey"

    # Define colors based on thresholds
    df['type_significance'] = False  # Default color for non-significant
    df.loc[((df['type_p_value'] < p_thresh) & (df['type_stat'] > stat_thresh)), 'type_significance'] = True 
    sig_phyla = df.loc[df['type_significance'], 'Phylum'].unique()
    df['type_color'] = [i if s else 'not_indicator' for i,s in zip(df['Phylum'], df['type_significance'])]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot non-red points first
        # Then plot red points on top
    sig = df #[df['significance'] == True]

    ax= sns.stripplot(data=sig, x='type_stat', y='type_log_p', hue='type_color', orient="h",
                  dodge=True, ax=ax, alpha=0.75, legend=False, palette=phylum_color_dict,
                  jitter=True, size=5, linewidth=0.25, edgecolor='gray'
                  )

    '''
    non_sig = df[df['type_significance'] == False]
    plt.scatter(non_sig['type_stat'], non_sig['type_log_p'], c=non_sig['type_color'], alpha=1,
                edgecolors='gray', linewidths=0.25,
                s=10
                )
    
    # Then plot red points on top
    sig = df[df['type_significance'] == True]
    plt.scatter(sig['type_stat'], sig['type_log_p'], c=sig['type_color'], alpha=1, edgecolors='gray', linewidths=0.25,
                s=75
                )
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')
    '''
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=color, label=phylum)
        for phylum, color in phylum_color_dict.items()
        if phylum in sig_phyla
    ]
    

    # Add legend outside plot
    plt.legend(
        handles=legend_handles,
        title='Phylum',
        bbox_to_anchor=(1.05, 1),  # Right side
        loc='upper left',
        borderaxespad=0.
    )

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Round limits for nice ticks
    xmin = 0
    xmax = np.ceil(df['type_stat'].max() * 10) / 10  # e.g., 0.87 → 0.9
    # Generate ticks every 0.1
    xticks = np.arange(xmin, xmax + 0.01, 0.1)  # add 0.01 to ensure inclusion
    # Set ticks and limits
    ax.set_xticks(xticks)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelsize=10, bottom=True)
    ax.invert_yaxis()
    # Get current ticks and keep every other one
    current_ticks = plt.yticks()[0]
    plt.yticks(current_ticks[::2])
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', which='both', length=4, width=1, color='black', left=True, right=False)

    # Optional: fix layout
    fig.subplots_adjust(bottom=0.15)

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()


def plot_combined(df, output_file, type_palette, no_sig=False):
    
    p_thresh=0.05
    if no_sig:
        df['status_significance'] = True
    else:
        df['type_color'] = [x if y == True else 'lightgrey' for x,y in zip(df['type_color'], df['status_significance'])]
        df['type_color'] = ['lightgrey' if ((y == True) & (x == 'lightgrey')) else x
                            for x,y in zip(df['type_color'], df['status_significance']
                            )]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot non-red points first
        # Then plot red points on top
    sig = df #[df['significance'] == True]
    palette = dict(zip(sig['type_color'], sig['type_color']))

    ax= sns.stripplot(data=sig, x='status_stat', y='status_log_p', hue='type_color', orient="h",
                  dodge=True, ax=ax, alpha=0.75, legend=False, palette=palette,
                  jitter=True, size=5, linewidth=0.25, edgecolor='gray'
                  )

    '''
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['type_color'],
                s=10, alpha=0.75, edgecolors='gray', linewidths=0.25
                )
    
    marker_dict = {1.0: 'X', 2.0: 'o'}
    
    for g in marker_dict.keys():
        sig = df[((df['status_significance'] == True) & (df['status_index'] == g))]
        print(sig.shape)
        plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['type_color'],
                    s=75, marker=marker_dict[g], alpha=0.75, edgecolors='gray',
                    linewidths=0.25
                    )

    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')
    '''
    # Build color legend (hue)
    color_handles = [mpatches.Patch(color=type_palette[k], label=k) for k in ['ca-contra', 'ca-lung', 'ctrl-brush',
                                                                              'ca-contra+ca-lung', 'ca-contra+ctrl-brush',
                                                                              'ca-lung+ctrl-brush']]

    # Build marker legend (shape)
    status_dict = {'Non-Cancer': 'o', 'Cancer': 'X'}
    marker_handles = [mlines.Line2D([], [], color='gray', marker=status_dict[k], linestyle='None',
                                     markersize=8, label=f'{k}')
                      for k in status_dict]

    # Add legends
    legend1 = plt.legend(handles=color_handles, title='Type', loc='upper right', bbox_to_anchor=(1.5, 1))
    legend2 = plt.legend(handles=marker_handles, title='status', loc='upper right', bbox_to_anchor=(1.5, 0.6))

    plt.gca().add_artist(legend1)  # Keep both legends

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Round limits for nice ticks
    xmin = 0
    xmax = np.ceil(df['status_stat'].max() * 10) / 10  # e.g., 0.87 → 0.9
    # Generate ticks every 0.1
    xticks = np.arange(xmin, xmax + 0.01, 0.1)  # add 0.01 to ensure inclusion
    # Set ticks and limits
    ax.set_xticks(xticks)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelsize=10, bottom=True)
    ax.invert_yaxis()
    # Get current ticks and keep every other one
    current_ticks = plt.yticks()[0]
    plt.yticks(current_ticks[::2])
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', which='both', length=4, width=1, color='black', left=True, right=False)
    # Optional: fix layout
    fig.subplots_adjust(bottom=0.15)

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()

    return df


def plot_comb_taxa(df, ind, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):

    # List of unique Phyla in your data
    phyla = df['Phylum'].unique()
    # Generate color palette (qualitative)
    palette = sns.color_palette('tab20', len(phyla))  # or 'Set3', 'Paired', etc.
    # Map phylum to color
    phylum_color_dict = dict(zip(phyla, palette))
    # Define colors based on thresholds
    df['status_significance'] = False  # Default color for non-significant
    df.loc[((df['status_p_value'] < p_thresh) & (df['status_stat'] > stat_thresh)), 'status_significance'] = True 
    sig_phyla = df.loc[df['status_significance'], 'Phylum'].unique()

    
    df['status_color'] = [phylum_color_dict[i] if s else 'lightgrey' for i,s in zip(df['Phylum'], df['status_significance'])]
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sig = df #[df['significance'] == True]
    palette = dict(zip(sig['status_color'], sig['status_color']))

    ax= sns.stripplot(data=sig, x='status_stat', y='status_log_p', hue='status_color', orient="h",
                  dodge=True, ax=ax, alpha=0.75, legend=False, palette=palette,
                  jitter=True, size=5, linewidth=0.25, edgecolor='gray'
                  )

    '''
    # Plot non-red points first
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['status_color'], alpha=1,
                edgecolors='gray', linewidths=0.25,
                s=10
                )
    
    # Then plot red points on top
    sig = df[df['status_significance'] == True]
    plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['status_color'], alpha=1, edgecolors='gray',
        linewidths=0.25,
                s=75
                )
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')
    '''
    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=color, label=phylum)
        for phylum, color in phylum_color_dict.items()
        if phylum in sig_phyla
    ]

    # Add legend outside plot
    plt.legend(
        handles=legend_handles,
        title='Phylum',
        bbox_to_anchor=(1.05, 1),  # Right side
        loc='upper left',
        borderaxespad=0.
    )

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Round limits for nice ticks
    xmin = 0
    xmax = np.ceil(df['status_stat'].max() * 10) / 10  # e.g., 0.87 → 0.9
    # Generate ticks every 0.1
    xticks = np.arange(xmin, xmax + 0.01, 0.1)  # add 0.01 to ensure inclusion
    # Set ticks and limits
    ax.set_xticks(xticks)
    ax.set_xlim(xmin, xmax)
    ax.tick_params(axis='x', labelsize=10, bottom=True)
    ax.invert_yaxis()
    # Get current ticks and keep every other one
    current_ticks = plt.yticks()[0]
    plt.yticks(current_ticks[::2])
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', which='both', length=4, width=1, color='black', left=True, right=False)
   # Optional: fix layout
    fig.subplots_adjust(bottom=0.15)

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')
    plt.savefig(output_file.replace('.svg', '.pdf'), bbox_inches='tight')
    plt.close()


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


data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'
taxonomy_path = os.path.join(data_dir, 'spark_old_output/metadata/taxonomy_updated.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['ASV_ID'] = [x.rsplit(';', 1)[0] for x in tax_df['ASV_ID']]
tax_df.set_index('ASV_ID', inplace=True)
taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    tax_df[t] = taxonomy_dict[t]

df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/indicspecies/type_group_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

v_df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/metadata/Three_types_venn_presence_table.tsv'), sep='\t')
venn_dict = {a:g for a,g in zip(v_df['ASV_ID'], v_df['grouping'])}

type_index = {1: 'BAL',
              2: 'Lung Brush',
              3: 'Oral Rinse',
              4: 'BAL+Lung Brush',
              5: 'BAL+Oral Rinse',
              6: 'Lung Brush+Oral Rinse',
              7: 'not_indicator'
              }

type2_ind = {v: k for k, v in type_index.items()}

type_palette = {'Oral Rinse': '#6A3D9A',
                'BAL+Oral Rinse': '#F19CBB',
                'BAL': '#0072B2',
                'BAL+Lung Brush': '#00FFFF',
                'Lung Brush': '#009E73',
                'Lung Brush+Oral Rinse': '#C1EAAD',
                'not_indicator': 'lightgray'
                }

venn2palette = {'Oral Rinse':'Oral Rinse',
                'BAL':'BAL',
                'Lung Brush':'Lung Brush',
                'Oral Rinse + BAL':'BAL+Oral Rinse',
                'Oral Rinse + Lung Brush':'Lung Brush+Oral Rinse',
                'BAL + Lung Brush':'BAL+Lung Brush',
                'Oral Rinse + BAL + Lung Brush': 'not_indicator'
                }

sub_df = df.loc[df['index'].isin(type_index.keys())]
venn_sub_df = sub_df.copy()
venn_sub_df['index'] = [type2_ind[venn2palette[venn_dict[x]]] for x in venn_sub_df['ASV_ID']]

type_isa_df = plot_volcano(sub_df, type_index, type_palette,
                           output_file=os.path.join(data_dir, 'spark_old_output/indicspecies/type_group_ISA_plot.svg')
                           )
type_venn_df = plot_volcano(venn_sub_df, type_index, type_palette,
                           output_file=os.path.join(data_dir, 'spark_old_output/indicspecies/type_group_Venn_plot.svg'),
                           no_sig=True
                           )

type_isa_df.columns = ['ASV_ID', 'BAL', 'Lung Brush', 'Oral Rinse',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color'
                      ]
type_venn_df.columns = ['ASV_ID', 'BAL', 'Lung Brush', 'Oral Rinse',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color'
                      ]

sub_tax_df = sub_df.merge(tax_df, on='ASV_ID')

plot_type_taxa(sub_tax_df, type_index, output_file=os.path.join(data_dir,
               'spark_old_output/indicspecies/type_group_ISA_plot_Phylum.svg')
)

df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/indicspecies/status_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

index_dict = {1: 'Cancer', 2: 'Non-Cancer'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026'}

status_isa_df = plot_volcano(df, index_dict, status_palette, output_file=os.path.join(data_dir, 'spark_old_output/indicspecies/status_Cancer_ISA_plot.svg'))
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat', 'status_p_value', 'status_log_p', 'status_significance', 'status_color']
type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/indicspecies/Type_status_ISA_results.tsv'), sep='\t')
plot_combined(type_status_df, os.path.join(data_dir, 'spark_old_output/indicspecies/Combined_ISA_plot.svg'), type_palette)
plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'spark_old_output/indicspecies/Combined_noNoType_ISA_plot.svg'), type_palette)
TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/indicspecies/Combined_ISA_plot_Phylum.svg')
               )

type_status_df = pd.merge(type_venn_df, status_isa_df, on='ASV_ID')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/indicspecies/Type_status_Venn_results.tsv'), sep='\t')
plot_combined(type_status_df, os.path.join(data_dir, 'spark_old_output/indicspecies/Combined_Venn_plot.svg'),
              type_palette, no_sig=True
              )
plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'spark_old_output/indicspecies/Combined_noNoType_Venn_plot.svg'),
              type_palette, no_sig=True
              )
TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/indicspecies/Combined_Venn_plot_Phylum.svg')
               )





