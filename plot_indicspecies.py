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


def sig_table(df, ind, cmap, p_thresh=0.05, stat_thresh=0.0, non_sig=False):
    # Compute log-transformed p-values
    df['log_p'] = -np.log10(df['p.value']).round(1)
    if non_sig:
        df['significance'] = True
    else:
        df['significance'] = False  # Default color for non-significant
        df.loc[((df['p.value'] < p_thresh) & (df['stat'] > stat_thresh)), 'significance'] = True 
    df['color'] = [cmap[ind[i]] if s else 'lightgray' for i,s in zip(df['index'], df['significance'])]
    df['label'] = [ind[i] if s else 'not_indicator' for i,s in zip(df['index'], df['significance'])]
    cmap['not_indicator'] = 'lightgray'

    return df

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
    # ---------- NEW (only what’s needed to lock plot size & add outer pad) ----------
    plot_size_in=(8.0, 6.0),      # fixed size of the *data rectangle* (inside spines)
    axes_pad_in=(0.8, 0.6, 0.3, 0.2),  # (left, bottom, right, top) space for tick/axis labels, in inches
    figure_edge_pad_in=0.25,      # uniform pad around the whole figure (plot + legends), in inches
    legend_pad_in=0.45,           # gap between plot pane and legend pane (inches)
    legend_vgap_in=0.25,          # vertical gap between color & marker legends (inches)
    legend_fontsize=10,
):
    """
    Scatter of x_col vs y_col with guaranteed non-overlap via axis-wise repulsive jitter.
    The *data area* (inside the spines) is locked to `plot_size_in` no matter how big legends are.
    An outer pad is applied so nothing is clipped in the saved image.
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
        dd["__hue_raw__"] = dd[hue_col].astype(str)

        if type_palette:
            rev = {v: k for k, v in type_palette.items()}
            def to_name(v):
                if v in type_palette:
                    return v
                if mcolors.is_color_like(v) and v in rev:
                    return rev[v]
                return v  # fallback: leave as-is
            dd["__hue__"] = dd["__hue_raw__"].map(to_name)
            hue_used = "__hue__"
            present = [h for h in dd["__hue__"].dropna().unique().tolist() if h in type_palette]
            plot_palette = {k: type_palette[k] for k in present}
        else:
            levels = dd["__hue_raw__"].dropna().unique().tolist()
            hue_used = "__hue_raw__"
            if all(mcolors.is_color_like(v) for v in levels):
                plot_palette = {v: v for v in levels}
            else:
                plot_palette = None  # seaborn default

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

    # ---------- legends (handles for measurement) ----------
    color_handles = []
    if show_legend:
        if type_palette:
            for name, color in type_palette.items():
                color_handles.append(
                    mlines.Line2D([], [], marker="o", linestyle="None",
                                  markerfacecolor=color, markeredgecolor="black",
                                  markeredgewidth=0.5, markersize=8, label=str(name))
                )
        elif hue_used is not None:
            present_levels = dd[hue_used].dropna().unique().tolist()
            if plot_palette:
                for lab in present_levels:
                    col = plot_palette.get(lab, "lightgray")
                    color_handles.append(
                        mlines.Line2D([], [], marker="o", linestyle="None",
                                      markerfacecolor=col, markeredgecolor="black",
                                      markeredgewidth=0.5, markersize=8, label=str(lab))
                    )
            else:
                for lab in present_levels:
                    color_handles.append(
                        mlines.Line2D([], [], marker="o", linestyle="None",
                                      markerfacecolor="lightgray", markeredgecolor="black",
                                      markeredgewidth=0.5, markersize=8, label=str(lab))
                    )

    marker_handles = []
    if show_legend and style_used is not None:
        if marker_dict:
            for name, mk in marker_dict.items():
                marker_handles.append(
                    mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                                  markeredgewidth=0.5, markersize=8, label=str(name))
                )
        elif isinstance(markers_for_plot, dict):
            for name, mk in markers_for_plot.items():
                marker_handles.append(
                    mlines.Line2D([], [], color="gray", marker=mk, linestyle="None",
                                  markeredgewidth=0.5, markersize=8, label=str(name))
                )

    # ---------- measure legend sizes (inches) ----------
    def _legend_size_in(handles, title, fontsize):
        if not handles:
            return (0.0, 0.0)
        ftmp, axtmp = plt.subplots(figsize=(2, 2), dpi=100)
        leg = axtmp.legend(handles=handles, title=title, frameon=True, loc="upper left",
                           fontsize=fontsize, title_fontsize=fontsize)
        ftmp.canvas.draw()
        bbox = leg.get_window_extent(ftmp.canvas.get_renderer())
        w_in = bbox.width / ftmp.dpi
        h_in = bbox.height / ftmp.dpi
        plt.close(ftmp)
        return (w_in, h_in)

    color_w, color_h = _legend_size_in(color_handles, legend_color_title, legend_fontsize)
    marker_w, marker_h = _legend_size_in(marker_handles, legend_marker_title, legend_fontsize)

    legend_w_in = 0.0
    legend_h_in = 0.0
    if show_legend:
        legend_w_in = max(color_w, marker_w)
        legend_h_in = (color_h if color_h else 0.0) + (marker_h if marker_h else 0.0)
        if color_h and marker_h:
            legend_h_in += legend_vgap_in

    # ---------- figure layout with fixed data area & outer pad ----------
    plot_w_in, plot_h_in = plot_size_in                  # size of data rectangle
    padL, padB, padR, padT = axes_pad_in                 # label/tick space around data rectangle
    pane_w_in = plot_w_in + padL + padR                  # total plot pane width
    pane_h_in = plot_h_in + padT + padB                  # total plot pane height

    fig_w_in = (figure_edge_pad_in + pane_w_in +               # left edge + plot pane
                ((legend_pad_in + legend_w_in) if (show_legend and legend_w_in > 0) else 0.0) +
                figure_edge_pad_in)                            # right edge
    fig_h_in = figure_edge_pad_in + max(pane_h_in, legend_h_in if show_legend else pane_h_in) + figure_edge_pad_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)

    # Main axes placed so the *data rectangle* is exactly plot_size_in
    ax_left = (figure_edge_pad_in + padL) / fig_w_in
    ax_bottom = (figure_edge_pad_in + padB) / fig_h_in
    ax_w = plot_w_in / fig_w_in
    ax_h = plot_h_in / fig_h_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])

    # Legend pane to the right (off axes)
    leg_ax = None
    if show_legend and legend_w_in > 0:
        leg_left = (figure_edge_pad_in + pane_w_in + legend_pad_in) / fig_w_in
        leg_w = legend_w_in / fig_w_in
        leg_ax = fig.add_axes([leg_left, figure_edge_pad_in / fig_h_in, leg_w,
                               1.0 - 2 * figure_edge_pad_in / fig_h_in])
        leg_ax.axis("off")

    # ---------- plot ----------
    plot_kwargs = dict(
        data=dd, x="_x_", y="_y_",
        s=point_size, alpha=alpha,
        linewidth=0.5, edgecolor="black",
        ax=ax, legend=False
    )
    if hue_used is not None:
        plot_kwargs["hue"] = hue_used
        if plot_palette is not None:
            plot_kwargs["palette"] = plot_palette
    if style_used is not None:
        plot_kwargs["style"] = style_used
        plot_kwargs["markers"] = markers_for_plot

    sns.scatterplot(**plot_kwargs)

    # labels, ticks, limits (ticks/labels now have room due to axes_pad_in)
    ax.set_xlabel(x_col if x_col in df.columns else x_src)
    ax.set_ylabel(y_col if y_col in df.columns else y_src)
    ax.set_xlim(0.1, xmax + 0.1)
    ax.set_ylim(-0.1, ymax + 0.1)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(axis="both", which="both", length=4, width=1)

    # ---------- place legends inside legend pane ----------
    if show_legend and leg_ax is not None:
        y_cursor = 1.0
        if color_handles:
            leg1 = leg_ax.legend(handles=color_handles, title=legend_color_title,
                                 loc="upper left", bbox_to_anchor=(0.0, y_cursor),
                                 frameon=True, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
            leg_ax.add_artist(leg1)
            y_cursor -= ((color_h if color_h else 0.0) + legend_vgap_in) / fig_h_in
        if marker_handles:
            leg_ax.legend(handles=marker_handles, title=legend_marker_title,
                          loc="upper left", bbox_to_anchor=(0.0, y_cursor),
                          frameon=True, fontsize=legend_fontsize, title_fontsize=legend_fontsize)

    # ---------- save (no tight bbox so the data area stays fixed) ----------
    fig.savefig(output_file)
    if output_file.endswith(".svg"):
        fig.savefig(output_file.replace(".svg", ".pdf"))
    plt.close(fig)

    return dd

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
taxonomy_path = os.path.join(data_dir, 'spark_methods_output_tester/metadata/taxonomy_updated.tsv')
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

df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/type_group_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

v_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/Three_types_venn_presence_table_TYPE.tsv'), sep='\t')
venn_dict = {a:g for a,g in zip(v_df['ASV_ID'], v_df['grouping'])}

type_index = {1: 'HostZERO-DEP',
              2: 'HostZERO-NODEP',
              3: 'SPARK-ZYMO',
              4: 'HostZERO-DEP+HostZERO-NODEP',
              5: 'HostZERO-DEP+SPARK-ZYMO',
              6: 'HostZERO-NODEP+SPARK-ZYMO',
              7: 'HostZERO-DEP+HostZERO-NODEP+SPARK-ZYMO'
              }

type_index = {1: 'BAL',
              2: 'Lung Brush',
              3: 'Oral Rinse',
              4: 'BAL+Lung Brush',
              5: 'BAL+Oral Rinse',
              6: 'Lung Brush+Oral Rinse',
              7: 'BAL+Lung Brush+Oral Rinse'
              }

type2_ind = {v: k for k, v in type_index.items()}

type_palette = {'SPARK-ZYMO': 'skyblue',
                'HostZERO-DEP+SPARK-ZYMO': '#F19CBB',
                'HostZERO-DEP': 'gray',
                'HostZERO-DEP+HostZERO-NODEP': '#00FFFF',
                'HostZERO-NODEP': 'black',
                'HostZERO-NODEP+SPARK-ZYMO': '#C1EAAD',
                'HostZERO-DEP+HostZERO-NODEP+SPARK-ZYMO': '#009E73'
                }

venn2palette = {'SPARK-ZYMO':'SPARK-ZYMO',
                'HostZERO-DEP':'HostZERO-DEP',
                'HostZERO-NODEP':'HostZERO-NODEP',
                'HostZERO-DEP + SPARK-ZYMO':'HostZERO-DEP+SPARK-ZYMO',
                'HostZERO-NODEP + SPARK-ZYMO':'HostZERO-NODEP+SPARK-ZYMO',
                'HostZERO-DEP + HostZERO-NODEP':'HostZERO-DEP+HostZERO-NODEP',
                'HostZERO-DEP + HostZERO-NODEP + SPARK-ZYMO': 'HostZERO-DEP+HostZERO-NODEP+SPARK-ZYMO'
                }

type_palette = {'Oral Rinse': '#6A3D9A',
                'BAL+Oral Rinse': '#F19CBB',
                'BAL': '#0072B2',
                'BAL+Lung Brush': '#00FFFF',
                'Lung Brush': '#009E73',
                'Lung Brush+Oral Rinse': '#C1EAAD',
                'BAL+Lung Brush+Oral Rinse': 'black'
                }

venn2palette = {'Oral Rinse':'Oral Rinse',
                'BAL':'BAL',
                'Lung Brush':'Lung Brush',
                'Oral Rinse + BAL':'BAL+Oral Rinse',
                'Oral Rinse + Lung Brush':'Lung Brush+Oral Rinse',
                'BAL + Lung Brush':'BAL+Lung Brush',
                'Oral Rinse + BAL + Lung Brush': 'BAL+Lung Brush+Oral Rinse'
                }

sub_df = df.loc[df['index'].isin(type_index.keys())]
venn_sub_df = sub_df.copy()


print(set(venn_dict.values()))
print(venn2palette.keys())

venn_sub_df['index'] = [type2_ind[venn2palette[venn_dict[x]]] for x in venn_sub_df['ASV_ID']]

type_isa_df = sig_table(sub_df, type_index, type_palette)
type_isa_df.columns = ['ASV_ID', 'ca-contra', 'ca-lung', 'ctrl-brush',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color', 'type_label'
                      ]
plot_p_vs_stat_no_overlap(type_isa_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/type_group_ISA_plot.svg'),
                          type_palette=type_palette,
                          x_col="type_stat",
                          y_col="type_log_p",
                          hue_col="type_label",
                          )
                           
type_venn_df = sig_table(venn_sub_df, type_index, type_palette, non_sig=True)
type_venn_df.columns = ['ASV_ID', 'ca-contra', 'ca-lung', 'ctrl-brush',
                      'type_index', 'type_stat', 'type_p_value', 'type_log_p', 'type_significance',
                      'type_color', 'type_label'
                      ]
plot_p_vs_stat_no_overlap(type_venn_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/type_group_Venn_plot.svg'),
                          type_palette=type_palette,
                          x_col="type_stat",
                          y_col="type_log_p",
                          hue_col="type_label",
                          )

sub_tax_df = sub_df.merge(tax_df, on='ASV_ID')
# List of unique Phyla in your data
phyla = sub_tax_df['Phylum'].unique()
# Generate color palette (qualitative)
palette = sns.color_palette('tab20', len(phyla))  # or 'Set3', 'Paired', etc.
# Map phylum to color
phylum_color_dict = dict(zip(phyla, palette))
plot_p_vs_stat_no_overlap(sub_tax_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/type_group_ISA_plot_Phylum.svg'),
                          type_palette=phylum_color_dict,
                          x_col="type_stat",
                          y_col="type_log_p",
                          hue_col="Phylum",
                          )
type_venn_df.to_csv(os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Type_status_ISA_results.tsv'), sep='\t')

'''
df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/status_indicator_species_results.tsv'), sep='\t')
df.rename(columns={df.columns[0]: 'ASV_ID'}, inplace=True)

index_dict = {1: 'Cancer', 2: 'Non-Cancer', 3: 'Cancer+Non-Cancer'}

status_palette = {'Non-Cancer':'white', 'Cancer':'#A50026', 'Cancer+Non-Cancer': 'black'}
marker_dict = {'Non-Cancer':'D', 'Cancer':'X', 'Cancer+Non-Cancer':'o'}
status_isa_df = sig_table(df, index_dict, status_palette)
status_isa_df.columns = ['ASV_ID', 'Cancer', 'Non-Cancer', 'status_index', 'status_stat',
                         'status_p_value', 'status_log_p', 'status_significance', 'status_color',
                         'status_label'
                         ]
plot_p_vs_stat_no_overlap(status_isa_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/status_ISA_plot.svg'),
                          type_palette=status_palette,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="status_label",
                          )

type_status_df = pd.merge(type_isa_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Type_status_ISA_results.tsv'), sep='\t')
plot_p_vs_stat_no_overlap(type_status_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Combined_ISA_plot.svg'),
                          type_palette=type_palette,
                          marker_dict=marker_dict,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="type_color",
                          style_col="status_label"
                          )

ts_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_p_vs_stat_no_overlap(ts_tax_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Combined_ISA_plot_Phylum.svg'),
                          type_palette=phylum_color_dict,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="Phylum",
                          )

type_status_df = pd.merge(type_venn_df, status_isa_df, on='ASV_ID', how='right')
type_status_df.to_csv(os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Type_status_Venn_results.tsv'), sep='\t')
plot_p_vs_stat_no_overlap(type_status_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Combined_Venn_plot.svg'),
                          type_palette=type_palette,
                          marker_dict=marker_dict,
                          x_col="status_stat",
                          y_col="status_log_p",
                          hue_col="type_color",
                          style_col="status_color"
                          )

ts_tax_df = type_status_df.merge(tax_df, left_on='ASV_ID', right_index=True)
plot_p_vs_stat_no_overlap(ts_tax_df,
                          os.path.join(data_dir, 'spark_methods_output_tester/indicspecies/Combined_Venn_plot_Phylum.svg'),
                          type_palette=phylum_color_dict,
                          x_col="type_stat",
                          y_col="type_log_p",
                          hue_col="Phylum",
                          )
'''