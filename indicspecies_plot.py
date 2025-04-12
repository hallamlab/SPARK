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


# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback


def plot_volcano(df, ind, cmap, p_thresh=0.05, stat_thresh=0, output_file='volcano_plot.svg'):
    # Load data
    
    # Compute log-transformed p-values
    df['log_p'] = -np.log10(df['p.value'])
    
    # Define colors based on thresholds
    df['significance'] = False  # Default color for non-significant
    df.loc[((df['p.value'] < p_thresh) & (df['stat'] > stat_thresh)), 'significance'] = True 
    
    df['color'] = [cmap[ind[i]] if s else 'lightgrey' for i,s in zip(df['index'], df['significance'])]
    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['significance'] == False]
    plt.scatter(non_sig['stat'], non_sig['log_p'], c=non_sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['significance'] == True]
    plt.scatter(sig['stat'], sig['log_p'], c=sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')

    return df


def plot_type_taxa(df, ind, p_thresh=0.05, stat_thresh=0, output_file='volcano_plot.svg'):

    # List of unique Phyla in your data
    phyla = df['Phylum'].unique()

    # Generate color palette (qualitative)
    palette = sns.color_palette('tab20', len(phyla))  # or 'Set3', 'Paired', etc.

    # Map phylum to color
    phylum_color_dict = dict(zip(phyla, palette))
    
    # Define colors based on thresholds
    df['type_significance'] = False  # Default color for non-significant
    df.loc[((df['type_p_value'] < p_thresh) & (df['type_stat'] > stat_thresh)), 'type_significance'] = True 
    sig_phyla = df.loc[df['type_significance'], 'Phylum'].unique()
    
    df['type_color'] = [phylum_color_dict[i] if s else 'lightgrey' for i,s in zip(df['Phylum'], df['type_significance'])]
    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['type_significance'] == False]
    plt.scatter(non_sig['type_stat'], non_sig['type_log_p'], c=non_sig['type_color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['type_significance'] == True]
    plt.scatter(sig['type_stat'], sig['type_log_p'], c=sig['type_color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

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

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')


def plot_status_taxa(df, ind, p_thresh=0.05, stat_thresh=0, output_file='volcano_plot.svg'):

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
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['status_color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['status_significance'] == True]
    plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['status_color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

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

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')


def plot_combined(df, output_file, three_palette):
    
    p_thresh=0.05
    
    df['type_color'] = [x if y == True else 'lightgrey' for x,y in zip(df['type_color'], df['status_significance'])]
    df['type_color'] = ['lightgrey' if ((y == True) & (x == 'lightgrey')) else x
                        for x,y in zip(df['type_color'], df['status_significance']
                            )]

    
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['type_color'],
                s=10, alpha=0.75, edgecolors='gray', linewidths=0.25
                )
    
    marker_dict = {1.0: 'X', 2.0: 'o'}
    
    for g in marker_dict.keys():
        sig = df[((df['status_significance'] == True) & (df['status_index'] == g))]
        print(sig.shape)
        plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['type_color'],
                    s=100, marker=marker_dict[g], alpha=0.75, edgecolors='gray',
                    linewidths=0.25
                    )

    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    # Build color legend (hue)
    color_handles = [mpatches.Patch(color=three_palette[k], label=k) for k in ['Oral Rinse', 'BAL', 'Lung Brush']]

    # Build marker legend (shape)
    status_dict = {'Non-Cancer': 'o', 'Cancer': 'X'}
    marker_handles = [mlines.Line2D([], [], color='gray', marker=status_dict[k], linestyle='None',
                                     markersize=8, label=f'{k}')
                      for k in status_dict]

    # Add legends
    legend1 = plt.legend(handles=color_handles, title='Type', loc='upper right', bbox_to_anchor=(1.25, 1))
    legend2 = plt.legend(handles=marker_handles, title='status', loc='upper right', bbox_to_anchor=(1.25, 0.6))

    plt.gca().add_artist(legend1)  # Keep both legends

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Save and show
    plt.savefig(output_file, bbox_inches='tight')

    return df

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
    
    # In status there are missing levels, fill them with None
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict



data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_GG2_tax.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Sequence_ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Sequence_ID']]
tax_df.set_index('Sequence_ID', inplace=True)
taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in tax_df['Taxonomy']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    tax_df[t] = taxonomy_dict[t]

print(tax_df.head())


df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv'), sep='\t')
index_dict = {1: 'BAL', 2: 'Lung Brush', 3: 'Oral Rinse'}
three_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           'No Type': 'gray'
           }
sub_df = df.loc[df['index'].isin(index_dict.keys())]
type_isa_df = plot_volcano(sub_df, index_dict, three_palette,
                           output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/Type_Group_ISA_plot.svg')
                           )
type_isa_df.columns = ['ASV_ID', 'BAL', 'Lung Brush', 'Oral Rinse',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color'
                      ]

sub_tax_df = sub_df.merge(tax_df, left_on='ASV_ID', right_index=True)
print(sub_tax_df.head())
plot_type_taxa(sub_tax_df, index_dict, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/Type_Group_ISA_plot_Phylum.svg'))


df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/indicspecies/status_indicator_species_results.tsv'), sep='\t')
index_dict = {1: 'Cancer', 2: 'Non-Cancer'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026'}
status_isa_df = plot_volcano(df, index_dict, status_palette, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/status_Cancer_ISA_plot.svg'))
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat', 'status_p_value', 'status_log_p', 'status_significance', 'status_color']

sub_tax_df = status_isa_df.merge(tax_df, left_on='ASV_ID', right_index=True)
print(sub_tax_df.head())
plot_status_taxa(sub_tax_df, index_dict, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/status_Cancer_ISA_plot_Phylum.svg'))



type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID')
type_status_df.to_csv(os.path.join(data_dir, 'vsearch_output/indicspecies/Type_status_ISA_results.tsv'), sep='\t')

plot_combined(type_status_df, os.path.join(data_dir, 'vsearch_output/indicspecies/Combined_ISA_plot.svg'), three_palette)

plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'vsearch_output/indicspecies/Combined_noNoType_ISA_plot.svg'), three_palette)


