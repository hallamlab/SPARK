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
import matplotlib.colors as mcolors


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
    df['label'] = [ind[i] if s else 'not_indicator' for i,s in zip(df['index'], df['significance'])]
    cmap['not_indicator'] = 'lightgray'

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Then plot red points on top
    sig = df #[df['significance'] == True]
    palette = dict(zip(sig['color'], sig['color']))

    ax= sns.stripplot(data=sig, x='stat', y='log_p', hue='color', orient="h",
                  dodge=True, ax=ax, alpha=0.75, legend=False, palette=palette,
                  jitter=True, size=5, linewidth=0.25, edgecolor='gray'
                  )

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

def plot_p_vs_stat_no_overlap(
    df,
    output_file,
    type_palette=None,      # optional: {category_name -> color}; legend will show ALL keys here
    marker_dict=None,       # optional: {category_name -> marker}; legend will show ALL keys here
    x_col="status_stat",
    y_col="-log10(p)",      # name you want on the y-axis label
    hue_col=None,           # column used for color groups; can be names OR actual color strings
    style_col=None,         # column used for marker groups (categories)
    # --- axis-wise jitter controls (normalized [0..1]) ---
    min_dist_x=0.02,
    min_dist_y=0.03,
    step_x=0.35,
    step_y=0.35,
    anchor=0.05,
    iters=200,
    invert_y=False,
    point_size=50,
    alpha=0.85,
    add_random_eps=(0.0, 0.0),  # tiny extra jitter (x,y) in norm units
    show_legend=True,
    legend_color_title="Type",
    legend_marker_title="Status",
):
    """
    Scatter of x_col vs y_col with guaranteed non-overlap via axis-wise repulsive jitter.
    - If hue_col values are *color strings*, but you also provide `type_palette={name->color}`,
      the plot will use the NAMES (keys) for hue so the legend shows names, not colors.
    - Legends:
        * Color legend lists ALL entries in `type_palette` (even if not present in data).
        * Marker legend lists ALL entries in `marker_dict` (even if not present in data).

    Returns: jittered DataFrame with columns '_x_' and '_y_'.
    """

    # ---------- prep ----------
    dd = df.copy()
    # Try common actual columns if user kept old names:
    x_src = x_col if x_col in dd.columns else ("status_stat" if "status_stat" in dd.columns else x_col)
    y_src = y_col if y_col in dd.columns else ("status_log_p" if "status_log_p" in dd.columns else y_col)

    dd[x_src] = pd.to_numeric(dd[x_src], errors="coerce")
    dd[y_src] = pd.to_numeric(dd[y_src], errors="coerce")
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_src, y_src])
    if dd.empty:
        raise ValueError("No data to plot after dropping NaN/inf in x/y.")

    x = dd[x_src].to_numpy()
    y = dd[y_src].to_numpy()

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if xmin == xmax: xmin -= 0.05; xmax += 0.05
    if ymin == ymax: ymin -= 0.05; ymax += 0.05

    # ---------- normalize & repulse (axis-wise rectangle) ----------
    nx = (x - xmin) / (xmax - xmin)
    ny = (y - ymin) / (ymax - ymin)
    pos = np.stack([nx, ny], axis=1).astype(float)
    orig = pos.copy()

    n = len(pos)
    eye_mask = ~np.eye(n, dtype=bool)
    for _ in range(iters):
        dx = pos[:, None, 0] - pos[None, :, 0]  # (n,n)
        dy = pos[:, None, 1] - pos[None, :, 1]  # (n,n)
        mask = eye_mask & (np.abs(dx) < min_dist_x) & (np.abs(dy) < min_dist_y)
        if not mask.any():
            break

        # directions (avoid zero)
        sign_x = np.sign(dx); sign_y = np.sign(dy)
        if np.any(sign_x == 0):
            sign_x[sign_x == 0] = np.random.choice([-1.0, 1.0], size=(sign_x == 0).sum())
        if np.any(sign_y == 0):
            sign_y[sign_y == 0] = np.random.choice([-1.0, 1.0], size=(sign_y == 0).sum())

        # push toward boundary of the no-overlap rectangle
        force_x = np.zeros_like(dx); force_y = np.zeros_like(dy)
        force_x[mask] = (min_dist_x - np.abs(dx[mask])) * sign_x[mask]
        force_y[mask] = (min_dist_y - np.abs(dy[mask])) * sign_y[mask]

        delta_x = force_x.sum(axis=1)
        delta_y = force_y.sum(axis=1)

        pos[:, 0] += step_x * delta_x - anchor * (pos[:, 0] - orig[:, 0])
        pos[:, 1] += step_y * delta_y - anchor * (pos[:, 1] - orig[:, 1])

        np.clip(pos, 0.0, 1.0, out=pos)

    if add_random_eps != (0.0, 0.0):
        rng = np.random.default_rng(0)
        pos[:, 0] = np.clip(pos[:, 0] + rng.normal(0, add_random_eps[0], n), 0, 1)
        pos[:, 1] = np.clip(pos[:, 1] + rng.normal(0, add_random_eps[1], n), 0, 1)

    dd["_x_"] = pos[:, 0] * (xmax - xmin) + xmin
    dd["_y_"] = pos[:, 1] * (ymax - ymin) + ymin

    # ---------- resolve hue (color) semantics ----------
    hue_used = None
    plot_palette = None

    if hue_col is not None:
        # Build a plotting hue column that prefers *names* when a mapping is given
        dd["__hue_raw__"] = dd[hue_col].astype(str)

        if type_palette:
            # reverse map {color -> name}
            rev = {v: k for k, v in type_palette.items()}
            def to_name(v):
                # if the cell already holds a name key, keep it; else try color->name
                if v in type_palette:
                    return v
                if mcolors.is_color_like(v) and v in rev:
                    return rev[v]
                return v  # fallback: leave as-is (but legend will still show palette keys)
            dd["__hue__"] = dd["__hue_raw__"].map(to_name)
            hue_used = "__hue__"
            # palette for plotting only needs present labels that are in the provided palette
            present = [h for h in dd["__hue__"].dropna().unique().tolist() if h in type_palette]
            plot_palette = {k: type_palette[k] for k in present}
        else:
            # No palette provided: if values are actual colors, use identity palette
            levels = dd["__hue_raw__"].dropna().unique().tolist()
            hue_used = "__hue_raw__"
            if all(mcolors.is_color_like(v) for v in levels):
                plot_palette = {v: v for v in levels}  # identity
            else:
                plot_palette = None  # seaborn default palette
    # else: no hue

    # ---------- resolve style (marker) semantics ----------
    style_used = None
    markers_for_plot = True
    if style_col is not None:
        style_used = style_col
        cats = dd[style_used].dropna().unique().tolist()
        if marker_dict:
            markers_for_plot = {c: marker_dict.get(c, "o") for c in cats}
        else:
            default_markers = ["o", "s", "D", "X", "^", "v", "P", "*", "h", "H", "8", "p", "<", ">"]
            markers_for_plot = {c: default_markers[i % len(default_markers)] for i, c in enumerate(cats)}

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_kwargs = dict(
        data=dd, x="_x_", y="_y_",
        s=point_size, alpha=alpha,
        linewidth=0.5, edgecolor="black",
        ax=ax, legend=False  # we'll build explicit legends
    )
    if hue_used is not None:
        plot_kwargs["hue"] = hue_used
        if plot_palette is not None:
            plot_kwargs["palette"] = plot_palette
    if style_used is not None:
        plot_kwargs["style"] = style_used
        plot_kwargs["markers"] = markers_for_plot

    sns.scatterplot(**plot_kwargs)

    ax.set_xlabel(x_col if x_col in df.columns else x_src)
    ax.set_ylabel(y_col if y_col in df.columns else y_src)
    ax.set_xlim(0.1, xmax + 0.1)
    ax.set_ylim(-0.1, ymax + 0.1)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(axis="both", which="both", length=4, width=1)

    # ---------- legends (never clip) ----------
    extra_artists = []

    if show_legend:
        # Color legend: show ALL keys from type_palette if provided,
        # otherwise show present hue levels (with colors if available).
        color_handles = []
        if type_palette:
            for name, color in type_palette.items():
                color_handles.append(
                    mlines.Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor=color, markeredgecolor="black",
                                  markersize=8, label=str(name), markeredgewidth=0.5)
                )
        elif hue_used is not None:
            present_levels = dd[hue_used].dropna().unique().tolist()
            if plot_palette:
                for lab in present_levels:
                    col = plot_palette.get(lab, "lightgray")
                    color_handles.append(
                        mlines.Line2D([], [], marker="o", linestyle="None",
                                      markerfacecolor=col, markeredgecolor="black",
                                      markersize=8, label=str(lab), markeredgewidth=0.5)
                    )
            else:
                for lab in present_levels:
                    color_handles.append(
                        mlines.Line2D([], [], marker="o", linestyle="None",
                                      markerfacecolor="lightgray", markeredgecolor="black",
                                      markersize=8, label=str(lab), markeredgewidth=0.5)
                    )

        if color_handles:
            leg1 = ax.legend(handles=color_handles, title=legend_color_title,
                             loc="upper left", bbox_to_anchor=(1.01, 1.0),
                             borderaxespad=0.0, frameon=True)
            ax.add_artist(leg1)
            extra_artists.append(leg1)

        # Marker legend: ALL keys from marker_dict if provided,
        # else only present styles.
        marker_handles = []
        if style_used is not None:
            if marker_dict:
                for name, mk in marker_dict.items():
                    marker_handles.append(
                        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                                      markersize=8, label=str(name), markeredgewidth=0.5)
                    )
            elif isinstance(markers_for_plot, dict):
                for name, mk in markers_for_plot.items():
                    marker_handles.append(
                        mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                                      markersize=8, label=str(name), markeredgewidth=0.5)
                    )
        if marker_handles:
            leg2 = ax.legend(handles=marker_handles, title=legend_marker_title,
                             loc="upper left", bbox_to_anchor=(1.01, 0.55),
                             borderaxespad=0.0, frameon=True)
            extra_artists.append(leg2)

    # ---------- save (expand bbox to include legends) ----------
    plt.tight_layout()
    if extra_artists:
        plt.savefig(output_file, bbox_inches="tight", bbox_extra_artists=extra_artists)
        if output_file.endswith(".svg"):
            plt.savefig(output_file.replace(".svg", ".pdf"),
                        bbox_inches="tight", bbox_extra_artists=extra_artists)
    else:
        plt.savefig(output_file, bbox_inches="tight")
        if output_file.endswith(".svg"):
            plt.savefig(output_file.replace(".svg", ".pdf"), bbox_inches="tight")
    plt.close()

    return dd

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
                      'type_color', 'type_label'
                      ]
type_venn_df.columns = ['ASV_ID', 'ca-contra', 'ca-lung', 'ctrl-brush',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color', 'type_label'
                      ]

sub_tax_df = sub_df.merge(tax_df, on='ASV_ID')

plot_type_taxa(sub_tax_df, type_index, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/subclass2_ISA_plot_Phylum.svg')
)

df = pd.read_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/status_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

index_dict = {1: 'Cancer', 2: 'Non-Cancer', 3: 'not_indicator'}
status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026', 'not_indicator': 'lightgray'}
marker_dict = {'Non-Cancer':'D', 'Cancer':'X', 'not_indicator':'o'}
status_isa_df = plot_volcano(df, index_dict, status_palette, output_file=os.path.join(data_dir, 'spark_old_output/brush/indicspecies/status_Cancer_ISA_plot.svg'))
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat',
                         'status_p_value', 'status_log_p', 'status_significance', 'status_color',
                         'status_label'
                         ]

type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Type_status_ISA_results.tsv'), sep='\t')
plot_p_vs_stat_no_overlap(type_status_df,
                          os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_ISA_plot.svg'),
                          type_palette=type_palette,
                          marker_dict=marker_dict,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="type_color",
                          style_col="status_label"
                          )

TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/Combined_ISA_plot_Phylum.svg')
               )

type_status_df = pd.merge(type_venn_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Type_status_Venn_results.tsv'), sep='\t')
plot_p_vs_stat_no_overlap(type_status_df,
                          os.path.join(data_dir, 'spark_old_output/brush/indicspecies/Combined_Venn_plot.svg'),
                          type_palette=type_palette,
                          marker_dict=marker_dict,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="type_color",
                          style_col="status_color"
                          )

TS_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_comb_taxa(TS_tax_df, index_dict, output_file=os.path.join(data_dir,
               'spark_old_output/brush/indicspecies/Combined_Venn_plot_Phylum.svg')
               )





