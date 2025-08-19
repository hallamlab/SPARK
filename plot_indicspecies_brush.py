import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
import seaborn as sns
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
    
    df['color'] = [cmap[ind[i]] if s else 'lightgray' for i,s in zip(df['index'], df['significance'])]
    cmap['not_indicator'] = 'lightgray'
    
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
        mpatches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='lightgray', linewidth=0.5, label=type)
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
    phylum_color_dict['not_indicator'] = "lightgray"

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


def plot_combined(df, output_file, type_palette, marker_dict, no_sig=False):
    """
    Stripplot-based viz with fixed 0.1 y-bins and stable ordering.
    - df must have columns: ['status_stat', 'status_log_p', 'type_color', 'status_color', 'status_significance']
    - type_palette: mapping of type name -> color (for legend)
    - marker_dict: mapping of status_color -> marker (e.g., {'Non-Cancer':'o','Cancer':'X'})
    """

    p_thresh = 0.05

    df = df.copy()

    # -------- significance coloring (preserve 'lightgray' for non-significant) --------
    if no_sig:
        df['status_significance'] = True
    else:
        # Keep color if significant, else lightgray
        df['type_color'] = [
            x if bool(y) is True else 'lightgray'
            for x, y in zip(df['type_color'], df['status_significance'])
        ]
        # Guard: if color was already lightgray and significant, keep it lightgray
        df['type_color'] = [
            'lightgray' if (bool(y) is True and x == 'lightgray') else x
            for x, y in zip(df['type_color'], df['status_significance'])
        ]

    # ------------------------ bin y at 0.1 and lock order ------------------------
    y = pd.to_numeric(df['status_log_p'], errors='coerce').replace([np.inf, -np.inf], np.nan)

    if np.isfinite(np.nanmin(y)):
        ymin = float(np.floor(np.nanmin(y) * 10) / 10)
    else:
        ymin = 0.0
    if np.isfinite(np.nanmax(y)):
        ymax = float(np.ceil(np.nanmax(y) * 10) / 10)
    else:
        ymax = 1.0
    if ymin == ymax:
        ymin -= 0.05
        ymax += 0.05

    edges = np.round(np.arange(ymin, ymax + 0.1001, 0.1), 1)
    cats = pd.cut(y, bins=edges, include_lowest=True)  # Interval categories (unique)
    df['status_log_p_bin'] = pd.Categorical(cats, categories=cats.cat.categories, ordered=True)
    y_order = list(df['status_log_p_bin'].cat.categories)  # stable global order

    # ------------------------------- plotting -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Identity palette for per-point colors already encoded in df['type_color']
    point_palette = {c: c for c in df['type_color'].dropna().unique()}

    # Plot each marker group using the SAME y order so axis stays fixed
    for k, sub in df.groupby("status_color", dropna=False):
        sns.stripplot(
            data=sub,
            x='status_stat',
            y='status_log_p_bin',
            order=y_order,
            hue='type_color',
            orient="h",
            dodge=True,
            jitter=0.15,                    # consistent jitter
            size=5,
            linewidth=0.25,
            edgecolor='gray',
            marker=marker_dict.get(k, 'o'), # fallback if missing
            legend=False,
            palette=point_palette,
            ax=ax,
        )

    # ------------------------------- legends ------------------------------------
    # Color legend (types): use types present if available, otherwise all in type_palette
    present_types = []
    if 'type' in df.columns:
        present_types = [t for t in df['type'].dropna().unique() if t in type_palette]
    if not present_types:
        present_types = [t for t in type_palette]

    color_handles = [mpatches.Patch(color=type_palette[t], label=t) for t in present_types]

    status_dict = (
        {k: marker_dict.get(k, 'o') for k in df['status_color'].dropna().unique()}
        or {'Non-Cancer': 'o', 'Cancer': 'X'}
    )

    marker_handles = [
        mlines.Line2D([], [], color='gray', marker=mk, linestyle='None', markersize=8, label=lab)
        for lab, mk in {'Non-Cancer': 'D', 'Cancer': 'X', 'not_indicator': 'o'}.items()
    ]

    leg1 = ax.legend(handles=color_handles, title='Type', loc='upper right', bbox_to_anchor=(1.5, 1))
    leg2 = ax.legend(handles=marker_handles, title='status', loc='upper right', bbox_to_anchor=(1.5, 0.4))
    ax.add_artist(leg1)

    # ------------------------------ axes & labels -------------------------------
    ax.set_xlabel('Effect Size (stat)')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # X axis ticks/limits
    xnum = pd.to_numeric(df['status_stat'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    xmax = float(np.ceil(np.nanmax(xnum) * 10) / 10) if np.isfinite(np.nanmax(xnum)) else 1.0
    ax.set_xticks(np.arange(0, xmax + 0.01, 0.1))
    ax.set_xlim(0, xmax)

    # Y axis: EVERY 0.1 tick label
    tick_positions = np.arange(len(y_order))
    tick_labels = [f"{iv.right:.1f}" for iv in y_order]  # label by bin right edge
    ax.set_yticks(tick_positions, tick_labels)
    ax.invert_yaxis()  # larger -log10(p) at top
    ax.tick_params(axis='y', labelsize=8)
    # Get current ticks and keep every other one
    current_ticks = plt.yticks()[0]
    plt.yticks(current_ticks[::2])
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='y', which='both', length=4, width=1, color='black', left=True, right=False)

    # Styling
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', which='both', length=4, width=1, color='black')
    fig.subplots_adjust(bottom=0.15)

    # ------------------------------- save ---------------------------------------
    plt.savefig(output_file, bbox_inches='tight')
    if output_file.endswith('.svg'):
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

    
    df['status_color'] = [phylum_color_dict[i] if s else 'lightgray' for i,s in zip(df['Phylum'], df['status_significance'])]
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
taxonomy_path = os.path.join(data_dir, 'spark_old_output/brush/metadata/taxonomy_updated.tsv')
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

df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/subclass2_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

v_df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/metadata/Three_types_venn_presence_table.tsv'), sep='\t')
venn_dict = {a:g for a,g in zip(v_df['ASV_ID'], v_df['grouping'])}

type_index = {1: 'ca-contra',
              2: 'ca-lung',
              3: 'ctrl-brush',
              4: 'ca-contra+ca-lung',
              5: 'ca-contra+ctrl-brush',
              6: 'ca-lung+ctrl-brush',
              7: 'not_indicator'
              }

type2_ind = {v: k for k, v in type_index.items()}

type_palette = {'ctrl-brush': '#6A3D9A',
                'ca-contra+ctrl-brush': '#F19CBB',
                'ca-contra': '#0072B2',
                'ca-contra+ca-lung': '#00FFFF',
                'ca-lung': '#009E73',
                'ca-lung+ctrl-brush': '#C1EAAD',
                'not_indicator': 'lightgray'
                }

venn2palette = {'ctrl-brush':'ctrl-brush',
                'ca-contra':'ca-contra',
                'ca-lung':'ca-lung',
                'ctrl-brush + ca-contra':'ca-contra+ctrl-brush',
                'ctrl-brush + ca-lung':'ca-lung+ctrl-brush',
                'ca-contra + ca-lung':'ca-contra+ca-lung',
                'ctrl-brush + ca-contra + ca-lung': 'not_indicator'
                }

sub_df = df.loc[df['index'].isin(type_index.keys())]
venn_sub_df = sub_df.copy()
venn_sub_df['index'] = [type2_ind[venn2palette[venn_dict[x]]] for x in venn_sub_df['ASV_ID']]

type_isa_df = plot_volcano(sub_df, type_index, type_palette,
                           output_file=os.path.join(data_dir, 'spark_old_output/brush/indicspecies/subclass2_ISA_plot.svg')
                           )
type_venn_df = plot_volcano(venn_sub_df, type_index, type_palette,
                           output_file=os.path.join(data_dir, 'spark_old_output/brush/indicspecies/subclass2_Venn_plot.svg'),
                           no_sig=True
                           )

type_isa_df.columns = ['ASV_ID', 'ca-contra', 'ca-lung', 'ctrl-brush',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color'
                      ]
type_venn_df.columns = ['ASV_ID', 'ca-contra', 'ca-lung', 'ctrl-brush',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color'
                      ]

sub_tax_df = sub_df.merge(tax_df, on='ASV_ID')

plot_type_taxa(sub_tax_df, type_index, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/subclass2_ISA_plot_Phylum.svg')
)

df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/status_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

index_dict = {1: 'Cancer', 2: 'Non-Cancer', 3: 'not_indicator'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026', 'not_indicator': 'lightgray'}
marker_dict = {'white':'D', '#A50026':'X', 'lightgray':'o'}
status_isa_df = plot_volcano(df, index_dict, status_palette, output_file=os.path.join(data_dir, 'spark_old_output/brush/indicspecies/status_Cancer_ISA_plot.svg'))
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat', 'status_p_value', 'status_log_p', 'status_significance', 'status_color']

type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Type_status_ISA_results.tsv'), sep='\t')
plot_combined(type_status_df,
              os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_ISA_plot.svg'),
              type_palette, marker_dict
              )
plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_noNoType_ISA_plot.svg'),
              type_palette, marker_dict
              )
TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/Combined_ISA_plot_Phylum.svg')
               )

type_status_df = pd.merge(type_venn_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Type_status_Venn_results.tsv'), sep='\t')
plot_combined(type_status_df,
              os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_Venn_plot.svg'),
              type_palette, marker_dict, no_sig=True
              )
plot_combined(type_status_df.loc[type_status_df['type_significance'] == True],
              os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_noNoType_Venn_plot.svg'),
              type_palette, marker_dict, no_sig=True
              )
TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/Combined_Venn_plot_Phylum.svg')
               )





