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
font_path = '/home/ryan/.fonts/MYRIADPRO-REGULAR.OTF'  # update to your path
myriad_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = myriad_font.get_name()
sns.set_theme()  # re-applies style with updated rcParams



Family_palette = {'UBA1547': (0.07294117647058829, 0.2799999999999998, 0.4235294117647058),
                  'Thalassarchaeaceae': (0.21126696832579195, 0.4409049773755658, 0.7440271493212669),
                  'Veillonellaceae': (0.6329411764705882, 0.2967732487185746, 0.0),
                  'Bacteroidaceae': (0.8823529411764707, 0.43790849673202614, 0.0),
                  'Streptococcaceae': (0.10352941176470587, 0.3764705882352941, 0.10352941176470587),
                  'Neisseriaceae_563222': (0.262234504540071, 0.666987761547572, 0.1824240031583102),
                  'Porphyromonadaceae': (0.5035294117647059, 0.09176470588235297, 0.0941176470588235),
                  'Fusobacteriaceae_993521': (0.9529411764705883, 0.018151260504201617, 0.0),
                  'Pasteurellaceae': (0.3498327037236912, 0.20800863464651914, 0.4790501888828926),
                  'Gemellaceae': (0.47655809431210483, 0.3177053962080699, 0.5975887214389888),
                  'UBA5272': (0.32941176470588246, 0.2023529411764706, 0.1764705882352941),
                  'Aerococcaceae': (0.5217292700212616, 0.32669029057406096, 0.2876824946846208),
                  'Leptotrichiaceae': (0.675121951219512, 0.13899569583931148, 0.5113055954088948),
                  'Actinomycetaceae': (0.902164124909223, 0.10724763979665919, 0.4496732026143783),
                  'Micrococcaceae': (0.2988235294117647, 0.2988235294117647, 0.2988235294117647),
                  'Megasphaeraceae': (0.468235294117647, 0.468235294117647, 0.468235294117647),
                  'Flavobacteriaceae': (0.44235294117647067, 0.4447058823529411, 0.08000000000000002),
                  'Atopobiaceae': (0.6437647058823528, 0.6437647058823529, 0.20329411764705874),
                  'Peptoniphilaceae': (0.05411764705882355, 0.4470588235294116, 0.4870588235294117),
                  'Campylobacteraceae': (0.19248206599713058, 0.6366714490674316, 0.718106169296987),
                  'Lachnospiraceae': (0.14588235294117657, 0.5599999999999996, 0.8470588235294116),
                  'Selenomonadaceae_42771': (0.9303619909502263, 0.9518552036199095, 0.9802262443438914),
                  'CAG-508': (0.9999999999999999, 0.6100951916036124, 0.26588235294117657),
                  'Tannerellaceae': (0.9999999999999999, 0.8814814814814815, 0.7647058823529415),
                  'Weeksellaceae': (0.20705882352941174, 0.7529411764705882, 0.20705882352941174),
                  'Burkholderiaceae_A_595425': (0.7918041847611526, 0.9353178049743387, 0.7635057244374257),
                  'Paludibacteraceae': (0.8752290165077887, 0.315359218786329, 0.3185584747733087),
                  'Peptostreptococcaceae_256921': (1.0, 0.9076750700280113, 0.9058823529411766),
                  'Mycobacteriaceae': (0.6928008634646518, 0.5636049649217484, 0.8105126821370751),
                  'Treponemataceae': (0.9187943607194943, 0.8893923189110353, 0.9411959163830821),
                  'Anaerovoracaceae': (0.6552667578659371, 0.4070588235294119, 0.356497948016416),
                  'Dialisteraceae': (0.8645216158752658, 0.7726718639262935, 0.754301913536499),
                  'Enterococcaceae': (0.9365279770444763, 0.6917073170731707, 0.861721664275466),
                  'Pseudomonadaceae': (1.0, 1.0, 1.0),
                  'Cardiobacteriaceae': (0.5976470588235294, 0.5976470588235294, 0.5976470588235294),
                  'Staskawiczbacteraceae': (0.936470588235294, 0.936470588235294, 0.936470588235294),
                  'Bifidobacteriaceae': (0.8508045370614616, 0.8550672645739912, 0.19434450013189108),
                  'Aminobacteriaceae': (0.9265882352941175, 0.9265882352941175, 0.7675294117647058),
                  'Tenuifilaceae': (0.1741176470588235, 0.8404092071611251, 0.908235294117647),
                  'Sphingomonadaceae': (0.8589765662362504, 0.946207556193209, 0.9621999043519848),
                  'Erysipelotrichaceae': (0.07294117647058829, 0.2799999999999998, 0.4235294117647058),
                  'Nanosynbacteraceae': (0.21126696832579195, 0.4409049773755658, 0.7440271493212669),
                  'Propionibacteriaceae': (0.6329411764705882, 0.2967732487185746, 0.0)
                  }




def plot_volcano(df, ind, cmap, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):
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


def plot_type_taxa(df, ind, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):

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
    plt.scatter(non_sig['type_stat'], non_sig['type_log_p'], c=non_sig['type_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['type_significance'] == True]
    plt.scatter(sig['type_stat'], sig['type_log_p'], c=sig['type_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
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


def plot_type_family(df, ind, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):

    df['type_color'] = [Family_palette[i] if s else 'lightgrey' for i,s in zip(df['Family'], df['type_significance'])]
    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['type_significance'] == False]
    plt.scatter(non_sig['type_stat'], non_sig['type_log_p'], c=non_sig['type_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['type_significance'] == True]
    plt.scatter(sig['type_stat'], sig['type_log_p'], c=sig['type_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=color, label=family)
        for family, color in Family_palette.items()
        if family in Family_palette.keys()
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


def plot_status_taxa(df, ind, p_thresh=0.05, stat_thresh=0.0, output_file='volcano_plot.svg'):

    df['status_color'] = [Family_palette[i] if s else 'lightgrey' for i,s in zip(df['Family'], df['status_significance'])]
    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['status_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['status_significance'] == True]
    plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['status_color'], alpha=1, edgecolors='gray', linewidths=0.25)
    
    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    # Create legend handles
    legend_handles = [
        mpatches.Patch(color=color, label=family)
        for family, color in Family_palette.items()
        if family in Family_palette.keys()
    ]

    # Add legend outside plot
    plt.legend(
        handles=legend_handles,
        title='Family',
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
    if taxa_str != 'Unassigned':
        parts = [part.strip().split('__', 1)[1] for part in taxa_str.split(delimiter)]
    else:
        parts = ['Unassigned']
    # In status there are missing levels, fill them with None
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict

def plot_combined_taxa(df, output_file, three_palette):
    
    p_thresh=0.05
    stat_thresh=0.0
    
    df['type_color'] = [Family_palette[x] if y == True else 'lightgrey' for x,y in zip(df['Family'], df['status_significance'])]
    df['type_color'] = ['lightgrey' if ((y == True) & (x == 'lightgrey')) else x
                        for x,y in zip(df['type_color'], df['status_significance']
                            )]

    
    fig = plt.figure(figsize=(12, 6))
    # Plot non-red points first
    non_sig = df[df['status_significance'] == False]
    plt.scatter(non_sig['status_stat'], non_sig['status_log_p'], c=non_sig['type_color'],
                s=10, alpha=1, edgecolors='gray', linewidths=0.25
                )
    
    marker_dict = {1.0: 'X', 2.0: 'o'}
    
    for g in marker_dict.keys():
        sig = df[((df['status_significance'] == True) & (df['status_index'] == g))]
        print(sig.shape)
        plt.scatter(sig['status_stat'], sig['status_log_p'], c=sig['type_color'],
                    s=100, marker=marker_dict[g], alpha=1, edgecolors='gray',
                    linewidths=0.25
                    )

    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')

    '''
    # Create legend handles
    color_handles = [
        mpatches.Patch(color=color, label=family)
        for family, color in Family_palette.items()
        if family in Family_palette.keys()
    ]

    # Build marker legend (shape)
    status_dict = {'Non-Cancer': 'o', 'Cancer': 'X'}
    marker_handles = [mlines.Line2D([], [], color='gray', marker=status_dict[k], linestyle='None',
                                     markersize=8, label=f'{k}')
                      for k in status_dict]

    # Add legends
    legend1 = plt.legend(handles=color_handles, title='Family', loc='upper left', bbox_to_anchor=(1.05, 1))
    legend2 = plt.legend(handles=marker_handles, title='status', loc='upper left', bbox_to_anchor=(1.05, 0.01))

    plt.gca().add_artist(legend1)  # Keep both legends
    '''
    
    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    plt.tight_layout(rect=[0, 0, 0.75, 1])  # Reserve space on right
    plt.savefig(output_file, dpi=300)


    return df



data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
#mito = "d__Bacteria; p__Pseudomonadota; c__Alphaproteobacteria; o__Rickettsiales; f__Mitochondria; g__; s__"
#tax_df = tax_df.loc[((tax_df['Confidence'] >= 0.7) & (tax_df['Taxonomy'] != mito))]
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)
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
#plot_type_family(sub_tax_df, index_dict, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/Type_Group_ISA_plot_Family.svg'))

df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/indicspecies/status_indicator_species_results.tsv'), sep='\t')
index_dict = {1: 'Cancer', 2: 'Non-Cancer'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026'}
status_isa_df = plot_volcano(df, index_dict, status_palette, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/status_Cancer_ISA_plot.svg'))
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat', 'status_p_value', 'status_log_p', 'status_significance', 'status_color']

sub_tax_df = status_isa_df.merge(tax_df, left_on='ASV_ID', right_index=True)
print(sub_tax_df.head())
#plot_status_taxa(sub_tax_df, index_dict, output_file=os.path.join(data_dir, 'vsearch_output/indicspecies/status_Cancer_ISA_plot_Family.svg'))



type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID')
type_status_df.to_csv(os.path.join(data_dir, 'vsearch_output/indicspecies/Type_status_ISA_results.tsv'), sep='\t')

plot_combined(type_status_df, os.path.join(data_dir, 'vsearch_output/indicspecies/Combined_ISA_plot.svg'), three_palette)

plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'vsearch_output/indicspecies/Combined_noNoType_ISA_plot.svg'), three_palette)

type_status_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
#plot_combined_taxa(type_status_tax_df.loc[type_status_tax_df['type_significance'] == True],
#                   os.path.join(data_dir, 'vsearch_output/indicspecies/Combined_noNoType_ISA_plot_Family.svg'), three_palette)

