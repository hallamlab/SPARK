import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_contents
from upsetplot import UpSet
import matplotlib
from matplotlib_venn import venn3, venn2
import os
import matplotlib as mpl
from matplotlib import font_manager as fm, rcParams
import seaborn as sns
from itertools import zip_longest
from venn import venn, generate_petal_labels, draw_venn
from venn._venn import is_valid_dataset_dict, generate_colors
from itertools import combinations
from functools import partial
from matplotlib import patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


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


def custom_venn_dispatch(data, func, fmt="{size}", hint_hidden=False,
                         dataset_labels=None, petal_labels=None, cmap="viridis",
                         alpha=.4, figsize=(8, 8), fontsize=14, legend_loc="upper right",
                         ax=None):
    """Check input, generate petal labels, draw venn or pseudovenn diagram"""
    if not is_valid_dataset_dict(data):
        raise TypeError("Only dictionaries of sets are understood")
    if hint_hidden and (func == draw_pseudovenn6) and (fmt != "{size}"):
        error_message = "To use fmt='{}', set hint_hidden=False".format(fmt)
        raise NotImplementedError(error_message)
    n_sets = len(data)
    if petal_labels is None:
        petal_labels = generate_petal_labels(data.values(), fmt=fmt)
    if dataset_labels is None:
        dataset_labels = data.keys()
    return func(
        petal_labels=petal_labels, dataset_labels=dataset_labels, hint_hidden=hint_hidden,
        colors=generate_colors(n_colors=n_sets, cmap=cmap, alpha=alpha),
        figsize=figsize, fontsize=fontsize, legend_loc=legend_loc, ax=ax
    )

def map_binary_keys_to_labels(data_dict, sample_types):
    mapping = {}
    for key in data_dict.keys():
        active = [sample_types[i] for i, bit in enumerate(key) if bit == "1"]
        mapping[key] = " + ".join(active) if active else "None"
    return mapping

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
    if ((taxa_str != 'Unassigned') & (str(taxa_str) != 'nan')):
        parts = [part.strip().split('__', 1)[1] for part in taxa_str.split(delimiter)]
    else:
        parts = ['Unassigned']
    # In status there are missing levels, fill them with None
    tax_dict = {}
    for i, level in enumerate(tax_levels):
        tax_dict[level] = parts[i] if i < len(parts) else None
    
    return tax_dict

data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'

# Replace 'your_file.tsv' with the path to your TSV
raw_asv_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/ASVs/ASV_target.micro.tsv'), sep='\t', index_col=0)
raw_asv_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in raw_asv_df.columns]

asv_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/ASVs/ASV_final.micro.tsv'), sep='\t', index_col=0)
metadata_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/metadata/metadata_updated_TYPE.tsv'), sep='\t')

raw_asv_stack_df = raw_asv_df.stack(future_stack=True).reset_index()
raw_asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
'''
raw_asv_stack_df['lmp_id'] = raw_asv_stack_df['sample'].copy()
raw_asv_stack_df['sample'] = raw_asv_stack_df['sample'].map(
    metadata_df.drop_duplicates('lmp_id').set_index('lmp_id')['sample']
).fillna(raw_asv_stack_df['sample'])
'''
raw_asv_stack_df = raw_asv_stack_df.loc[raw_asv_stack_df['lmp_id'].isin(metadata_df['lmp_id'])]
metadata_df = metadata_df.loc[metadata_df['lmp_id'].isin(list(raw_asv_stack_df['lmp_id']))]
raw_merge_df = raw_asv_stack_df.merge(metadata_df, how='left', on='lmp_id')

asv_stack_df = asv_df.stack(future_stack=True).reset_index()
asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
merge_df = asv_stack_df.merge(metadata_df, how='left', on='lmp_id')

raw_filter_df = raw_merge_df.loc[raw_merge_df['count'] > 0]
filter_df = merge_df.loc[merge_df['count'] > 0]

taxonomy_path = os.path.join(data_dir, 'spark_methods_output_tester/metadata/taxonomy_updated_TYPE.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['ASV_ID'] = [x.split(';', 1)[0] for x in tax_df['ASV_ID']]
tax_df.set_index('ASV_ID', inplace=True)

raw_asv_tax_df = raw_filter_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)
asv_tax_df = filter_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)

raw_taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }

taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }

for t in raw_asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        raw_taxonomy_dict[l].append(v)
for t in raw_taxonomy_dict:
    raw_asv_tax_df[t] = raw_taxonomy_dict[t]

for t in asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

raw_asv_sum_dict = {
    k: v for k, v in raw_asv_tax_df.groupby("ASV_ID")["count"].sum().to_dict().items()
    if v != 0
}
raw_type_asv_sum_dict = {
    k: v
    for k, v in raw_asv_tax_df.groupby(['type_group','ASV_ID'])['count'].sum().to_dict().items()
    if v != 0
}

asv_sum_dict = {
    k: v for k, v in asv_tax_df.groupby("ASV_ID")["count"].sum().to_dict().items()
    if v != 0
}
type_asv_sum_dict = {
    k: v
    for k, v in asv_tax_df.groupby(['type_group','ASV_ID'])['count'].sum().to_dict().items()
    if v != 0
}

type_group_asv_sum_dict = asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()

raw_total_abundance = raw_asv_tax_df.groupby('Phylum')['count'].sum()
raw_top10 = raw_total_abundance.sort_values(ascending=False).head(10).index.tolist()
raw_asv_tax_df['Phylum_plot'] = raw_asv_tax_df["Phylum"].apply(lambda x: x if x in raw_top10 else "Other")
raw_asv_phy_dict = {x:y for x,y in zip(raw_asv_tax_df['ASV_ID'], raw_asv_tax_df['Phylum_plot'])}

total_abundance = asv_tax_df.groupby('Phylum')['count'].sum()
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_tax_df['Phylum_plot'] = asv_tax_df["Phylum"].apply(lambda x: x if x in top10 else "Other")
asv_phy_dict = {x:y for x,y in zip(asv_tax_df['ASV_ID'], asv_tax_df['Phylum_plot'])}

all_type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           }
three_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A'
           }

#three_palette = {'HostZERO-DEP': 'black',
#               'HostZERO-NODEP': 'gray',
#               'SPARK-ZYMO': 'skyblue',
#               }
colors = ['#6A3D9A', '#0072B2', '#009E73']

keep_types = [
                  'Oral Rinse',
                  'BAL',
                  'Lung Brush'
                  ]

alpha = 0.6
'''
# Convert to RGBA arrays
rgba = np.array([mcolors.to_rgba(c, alpha=alpha) for c in colors])

# Simple alpha compositing in order
final = rgba[0]
for over in rgba[1:]:
    final = over[:3]*over[3] + final[:3]*(1 - over[3])
    final_alpha = over[3] + final_alpha*(1 - over[3]) if 'final_alpha' in locals() else over[3]

# Convert back to hex
overlap_color = mcolors.to_hex(final)

LMP_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Oral/Lung': overlap_color
           }

# Five Groups
# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
five_list = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']
five_list = [x for x in five_list if x in list(metadata_df['type_group'])]
sub_df = raw_asv_tax_df.loc[raw_asv_tax_df['type_group'].isin(five_list)]
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)  # Series with MultiIndex of booleans, values = unique counts

# 2) Plot UpSet (no stacked bars)
upset = UpSet(
    upset_data,
    subset_size='count',       # show unique counts
    element_size=None,
    show_counts=True,
    sort_categories_by='input',
    min_subset_size=0
)

# 3) Color LEFT (category) bars
for name in five_list:
    upset.style_categories([name], bar_facecolor=all_type_palette[name], bar_edgecolor="black")

# 4) Render and legend
fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

handles = [mpatches.Patch(facecolor=all_type_palette[t], edgecolor="black", label=t) for t in five_list]
fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by type_group", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_upset_plot.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_upset_plot.pdf"),
            format="pdf", bbox_inches="tight")


sub_df = raw_asv_tax_df
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []

for i,row in upset_data.iterrows():
    for t in five_list:
        if (t, row['index']) in raw_type_asv_sum_dict:
            val = raw_type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(five_list, inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(five_list[::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], five_list)

# Create and plot the UpSet plot.
upset = UpSet(upset_sum_df, sum_over='count', subset_size='sum',
              element_size=None, show_counts=True,
              sort_categories_by='input', min_subset_size=0,
              intersection_plot_elements=0
              )

for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="type", sum_over='count',
    colors=all_type_palette,
    title="Count by Type", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
order = five_list
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_upset_plot_sum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_upset_plot_sum.pdf"), format="pdf", bbox_inches="tight")

# Venns
venn = partial(custom_venn_dispatch, func=draw_venn, hint_hidden=False)
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['type_group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['type_group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['type_group'] == 'Lung Brush']['ASV_ID'])
skin_set = set(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['ASV_ID'])
scope_set = set(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['ASV_ID'])

name_to_set = {
    "Skin Brush": skin_set,
    "Scope Flush": scope_set,
    "Oral Rinse": oral_set,
    "BAL": bal_set,
    "Lung Brush": lung_set
    }

plt.figure(figsize=(6,6))

ordered_dict = {k: name_to_set[k] for k in five_list}
v = venn(ordered_dict,
         cmap=[all_type_palette[k] for k in five_list],
         fontsize=8,
         alpha=0.45
         )

plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_diagram.pdf"), format="pdf", bbox_inches="tight")

all_names = five_list
columns = {}

for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        # IDs present in all included sets
        included = set.intersection(*(name_to_set[n] for n in combo))
        # Subtract anything present in any excluded set
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included

        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Flatten to rows
venn_list = [[label, asv] for label, ids in columns.items() for asv in ids]

# Make DataFrame and save
venn_table = pd.DataFrame(venn_list, columns=["grouping", "ASV_ID"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_presence_table.tsv"),
    sep="\t",
    index=False
)

# Fresh Venn from saved/known sets
plt.figure(figsize=(6,6))
ordered_dict = {k: name_to_set[k] for k in five_list}
petal_labels = generate_petal_labels(ordered_dict.values(), fmt="{size}")
bin2lab_map = map_binary_keys_to_labels(petal_labels, five_list)
lab2bin_map = {v: k for k, v in bin2lab_map.items()}

# Compute exclusive ASV_ID lists and sums
all_names = five_list
columns = {}
for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        included = set.intersection(*(name_to_set[n] for n in combo))
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included
        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Save exclusive sums table
pedal_sums = {
    lab2bin_map[label]: int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))
    for label, ids in columns.items()
}

venn_list = [[label, int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))]
             for label, ids in columns.items()
             ]
venn_table = pd.DataFrame(venn_list, columns=["grouping", "Sum_count"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_sum_table.tsv"),
    sep="\t",
    index=False
)

v = venn(ordered_dict,
         petal_labels=pedal_sums,
         dataset_labels=five_list,
         cmap=[all_type_palette[k] for k in five_list],
         fontsize=8,
         alpha=0.45)

# Save diagram with sums
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_sum_diagram.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/All_types_venn_sum_diagram.pdf"),
            format="pdf", bbox_inches="tight")
'''
# Three Groups
# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
sub_list = ['Oral Rinse', 'BAL', 'Lung Brush']
sub_df = asv_tax_df.loc[asv_tax_df['type_group'].isin(sub_list)]
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)  # Series with MultiIndex of booleans, values = unique counts
upset_data = upset_data.reorder_levels(sub_list[::-1])

# 2) Plot UpSet (no stacked bars)
upset = UpSet(
    upset_data,
    subset_size='count',       # show unique counts
    element_size=None,
    show_counts=True,
    sort_categories_by='input',
    min_subset_size=0
)

# 3) Color LEFT (category) bars
for name in sub_list:
    upset.style_categories([name], bar_facecolor=three_palette[name], bar_edgecolor="black")

# 4) Render and legend
fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

handles = [mpatches.Patch(facecolor=three_palette[t], edgecolor="black", label=t) for t in sub_list]
fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by type_group", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_upset_plot_TYPE.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_upset_plot_TYPE.pdf"),
            format="pdf", bbox_inches="tight")

sub_df = asv_tax_df.loc[asv_tax_df['type_group'].isin(sub_list)]
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in sub_list:
        if (t, row['index']) in type_asv_sum_dict:
            val = type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(sub_list, inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(sub_list[::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], sub_list)

# Create and plot the UpSet plot.
upset = UpSet(upset_sum_df, sum_over='count', subset_size='sum',
              element_size=None, show_counts=True,
              sort_categories_by='input', min_subset_size=0,
              intersection_plot_elements=0
              )

for t in three_palette.keys():
    upset.style_categories([t], bar_facecolor=three_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="type", sum_over='count',
    colors=three_palette,
    title="Count by Type", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
order = sub_list
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_upset_plot_sum_TYPE.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_upset_plot_sum_TYPE.pdf"), format="pdf", bbox_inches="tight")

# Venns
venn = partial(custom_venn_dispatch, func=draw_venn, hint_hidden=False)
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['type_group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['type_group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['type_group'] == 'Lung Brush']['ASV_ID'])

name_to_set = {
    "Oral Rinse": oral_set,
    "BAL": bal_set,
    "Lung Brush": lung_set
    }

plt.figure(figsize=(6,6))

venn3([oral_set, bal_set, lung_set], ('Oral Rinse', 'BAL', 'Lung Brush'),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )
#ordered_dict = {k: name_to_set[k] for k in sub_list}
#v = venn(ordered_dict,
#         cmap=[three_palette[k] for k in sub_list],
#         fontsize=8,
#         alpha=0.45
#         )

plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_diagram_TYPE.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_diagram_TYPE.pdf"), format="pdf", bbox_inches="tight")

all_names = sub_list
columns = {}

for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        # IDs present in all included sets
        included = set.intersection(*(name_to_set[n] for n in combo))

        # Subtract anything present in any excluded set
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included

        if not group_ids:
            continue

        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Flatten to rows
venn_list = [[label, asv] for label, ids in columns.items() for asv in ids]

# Make DataFrame and save
venn_table = pd.DataFrame(venn_list, columns=["grouping", "ASV_ID"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_presence_table_TYPE.tsv"),
    sep="\t",
    index=False
)

# Fresh Venn from saved/known sets
plt.figure(figsize=(6,6))
ordered_dict = {k: name_to_set[k] for k in sub_list}
petal_labels = generate_petal_labels(ordered_dict.values(), fmt="{size}")
bin2lab_map = map_binary_keys_to_labels(petal_labels, sub_list)
lab2bin_map = {v: k for k, v in bin2lab_map.items()}

# Compute exclusive ASV_ID lists and sums
all_names = sub_list
columns = {}
for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        included = set.intersection(*(name_to_set[n] for n in combo))
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included
        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Save exclusive sums table
pedal_sums = {
    lab2bin_map[label]: int(sum(asv_sum_dict.get(asv, 0) for asv in ids))
    for label, ids in columns.items()
}

venn_list = [[label, int(sum(asv_sum_dict.get(asv, 0) for asv in ids))]
             for label, ids in columns.items()
             ]
venn_table = pd.DataFrame(venn_list, columns=["grouping", "Sum_count"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_sum_table_TYPE.tsv"),
    sep="\t",
    index=False
)
venn3(subsets=pedal_sums, set_labels=('Oral Rinse', 'BAL', 'Lung Brush'),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )
#v = venn(ordered_dict,
#         petal_labels=pedal_sums,
#         dataset_labels=sub_list,
#         cmap=[all_type_palette[k] for k in sub_list],
#         fontsize=8,
#         alpha=0.45)

# Save diagram with sums
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_sum_diagram_TYPE.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_types_venn_sum_diagram_TYPE.pdf"),
            format="pdf", bbox_inches="tight")










'''
# Combined Three, Skin and Scope Groups
# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
grp_list = ['Skin Brush', 'Oral/Lung']
sub_df = raw_asv_tax_df
sub_df['type_group'] = sub_df['type_group'].replace({'Oral Rinse': 'Oral/Lung',
                                                     'BAL': 'Oral/Lung',
                                                     'Lung Brush': 'Oral/Lung'})
raw_grp_asv_sum_dict = sub_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)  # Series with MultiIndex of booleans, values = unique counts
# 2) Plot UpSet (no stacked bars)
upset = UpSet(
    upset_data,
    subset_size='count',       # show unique counts
    element_size=None,
    show_counts=True,
    sort_categories_by='input',
    min_subset_size=0
)

# 3) Color LEFT (category) bars
for name in grp_list:
    upset.style_categories([name], bar_facecolor=LMP_palette[name], bar_edgecolor="black")

# 4) Render and legend
fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

handles = [mpatches.Patch(facecolor=LMP_palette[t], edgecolor="black", label=t) for t in grp_list]
fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by type_group", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_upset_plot.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_upset_plot.pdf"),
            format="pdf", bbox_inches="tight")


# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in grp_list:
        if (t, row['index']) in raw_grp_asv_sum_dict:
            val = raw_grp_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(grp_list, inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(grp_list[::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], grp_list)

# Create and plot the UpSet plot.
upset = UpSet(upset_sum_df, sum_over='count', subset_size='sum',
              element_size=None, show_counts=True,
              sort_categories_by='input', min_subset_size=0,
              intersection_plot_elements=0
              )

for t in LMP_palette.keys():
    upset.style_categories([t], bar_facecolor=LMP_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="type", sum_over='count',
    colors=LMP_palette,
    title="Count by Type", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
order = grp_list
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_upset_plot_sum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_upset_plot_sum.pdf"), format="pdf", bbox_inches="tight")

# Venns
venn = partial(custom_venn_dispatch, func=draw_venn, hint_hidden=False)
# Create venn3 with custom colors
skin_set = set(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['ASV_ID'])
scope_set = set(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['ASV_ID'])
olung_set = set(sub_df.loc[sub_df['type_group'] == 'Oral/Lung']['ASV_ID'])

name_to_set = {
    "Skin Brush": skin_set,
    "Scope Flush": scope_set,
    "Oral/Lung": olung_set
    }

plt.figure(figsize=(6,6))
if len(scope_set) == 0:
    venn2([skin_set, olung_set], ("Skin Brush", "Oral/Lung"),
        set_colors=(LMP_palette['Skin Brush'], LMP_palette['Oral/Lung']),
        alpha=0.6
        )
else:
    venn3([skin_set, scope_set, olung_set], ("Skin Brush", "Scope Flush", "Oral/Lung"),
        set_colors=(LMP_palette['Skin Brush'], LMP_palette['Scope Flush'], LMP_palette['Oral/Lung']),
        alpha=0.6
        )

plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_diagram.pdf"), format="pdf", bbox_inches="tight")

all_names = grp_list
columns = {}

for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        # IDs present in all included sets
        included = set.intersection(*(name_to_set[n] for n in combo))

        # Subtract anything present in any excluded set
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included

        if not group_ids:
            continue

        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Flatten to rows
venn_list = [[label, asv] for label, ids in columns.items() for asv in ids]

# Make DataFrame and save
venn_table = pd.DataFrame(venn_list, columns=["grouping", "ASV_ID"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_presence_table.tsv"),
    sep="\t",
    index=False
)

# Fresh Venn from saved/known sets
plt.figure(figsize=(6,6))
ordered_dict = {k: name_to_set[k] for k in grp_list}
petal_labels = generate_petal_labels(ordered_dict.values(), fmt="{size}")
bin2lab_map = map_binary_keys_to_labels(petal_labels, grp_list)
lab2bin_map = {v: k for k, v in bin2lab_map.items()}

# Compute exclusive ASV_ID lists and sums
all_names = grp_list
columns = {}
for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        included = set.intersection(*(name_to_set[n] for n in combo))
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included
        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Save exclusive sums table
pedal_sums = {
    lab2bin_map[label]: int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))
    for label, ids in columns.items()
}

venn_list = [[label, int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))]
             for label, ids in columns.items()
             ]
venn_table = pd.DataFrame(venn_list, columns=["grouping", "Sum_count"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_sum_table.tsv"),
    sep="\t",
    index=False
)

if len(scope_set) == 0:
    venn2(subsets=pedal_sums, set_labels=("Skin Brush", "Oral/Lung"),
      set_colors=(LMP_palette['Skin Brush'], LMP_palette['Oral/Lung']),
      alpha=0.6
      )
else:
    venn3(subsets=pedal_sums, set_labels=("Skin Brush", "Scope Flush", "Oral/Lung"),
      set_colors=(LMP_palette['Skin Brush'], LMP_palette['Scope Flush'], LMP_palette['Oral/Lung']),
      alpha=0.6
      )

# Save diagram with sums
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_sum_diagram.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/metadata/Three_vs_controls_venn_sum_diagram.pdf"),
            format="pdf", bbox_inches="tight")
'''





# MITOCHONDRIA
raw_asv_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/mito/ASVs/ASV_target.mito.tsv'), sep='\t', index_col=0)
raw_asv_df.columns = [str(x.split('/')[-1].rsplit('_', 2)[0]) for x in raw_asv_df.columns]
asv_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/mito/ASVs/ASV_final.mito_TYPE.tsv'), sep='\t', index_col=0)
metadata_df = pd.read_csv(os.path.join(data_dir, 'spark_methods_output_tester/mito/metadata/metadata_updated_mito_TYPE.tsv'), sep='\t')

raw_asv_stack_df = raw_asv_df.stack(future_stack=True).reset_index()
raw_asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
'''
raw_asv_stack_df['lmp_id'] = raw_asv_stack_df['sample'].copy()
raw_asv_stack_df['sample'] = raw_asv_stack_df['sample'].map(
    metadata_df.drop_duplicates('lmp_id').set_index('lmp_id')['sample']
).fillna(raw_asv_stack_df['sample'])
'''
raw_asv_stack_df = raw_asv_stack_df.loc[raw_asv_stack_df['lmp_id'].isin(metadata_df['lmp_id'])]
metadata_df = metadata_df.loc[metadata_df['lmp_id'].isin(list(raw_asv_stack_df['lmp_id']))]
raw_merge_df = raw_asv_stack_df.merge(metadata_df, how='left', on='lmp_id')

asv_stack_df = asv_df.stack(future_stack=True).reset_index()
asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
merge_df = asv_stack_df.merge(metadata_df, how='left', on='lmp_id')

raw_filter_df = raw_merge_df.loc[raw_merge_df['count'] > 0]
filter_df = merge_df.loc[merge_df['count'] > 0]

taxonomy_path = os.path.join(data_dir, 'spark_methods_output_tester/metadata/taxonomy_updated_TYPE.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['ASV_ID'] = [x.split(';', 1)[0] for x in tax_df['ASV_ID']]
tax_df.set_index('ASV_ID', inplace=True)

raw_asv_tax_df = raw_filter_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)
asv_tax_df = filter_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)

raw_taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }

taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }

for t in raw_asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        raw_taxonomy_dict[l].append(v)
for t in raw_taxonomy_dict:
    raw_asv_tax_df[t] = raw_taxonomy_dict[t]

for t in asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

raw_asv_sum_dict = raw_asv_tax_df.groupby("ASV_ID")["count"].sum().to_dict()
raw_type_asv_sum_dict = raw_asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()
raw_type_group_asv_sum_dict = raw_asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()

asv_sum_dict = asv_tax_df.groupby("ASV_ID")["count"].sum().to_dict()
type_asv_sum_dict = asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()
type_group_asv_sum_dict = asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()

raw_total_abundance = raw_asv_tax_df.groupby('Phylum')['count'].sum()
raw_top10 = raw_total_abundance.sort_values(ascending=False).head(10).index.tolist()
raw_asv_tax_df['Phylum_plot'] = raw_asv_tax_df["Phylum"].apply(lambda x: x if x in raw_top10 else "Other")
raw_asv_phy_dict = {x:y for x,y in zip(raw_asv_tax_df['ASV_ID'], raw_asv_tax_df['Phylum_plot'])}

total_abundance = asv_tax_df.groupby('Phylum')['count'].sum()
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_tax_df['Phylum_plot'] = asv_tax_df["Phylum"].apply(lambda x: x if x in top10 else "Other")
asv_phy_dict = {x:y for x,y in zip(asv_tax_df['ASV_ID'], asv_tax_df['Phylum_plot'])}

# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
raw_group_dict = raw_asv_tax_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
group_dict = asv_tax_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()

'''
# Five Groups
# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
five_list = ['Skin Brush', 'Scope Flush', 'Lung Brush', 'BAL', 'Oral Rinse']
five_list = [x for x in five_list if x in list(metadata_df['type_group'])]
sub_df = raw_asv_tax_df.loc[raw_asv_tax_df['type_group'].isin(five_list)]
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)  # Series with MultiIndex of booleans, values = unique counts

# 2) Plot UpSet (no stacked bars)
upset = UpSet(
    upset_data,
    subset_size='count',       # show unique counts
    element_size=None,
    show_counts=True,
    sort_categories_by='input',
    min_subset_size=0
)

# 3) Color LEFT (category) bars
for name in five_list:
    upset.style_categories([name], bar_facecolor=all_type_palette[name], bar_edgecolor="black")

# 4) Render and legend
fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

handles = [mpatches.Patch(facecolor=all_type_palette[t], edgecolor="black", label=t) for t in five_list]
fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by type_group", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_upset_plot.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_upset_plot.pdf"),
            format="pdf", bbox_inches="tight")

sub_df = raw_asv_tax_df
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in five_list:
        if (t, row['index']) in raw_type_asv_sum_dict:
            val = raw_type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(five_list, inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(five_list[::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], five_list)

# Create and plot the UpSet plot.
upset = UpSet(upset_sum_df, sum_over='count', subset_size='sum',
              element_size=None, show_counts=True,
              sort_categories_by='input', min_subset_size=0,
              intersection_plot_elements=0
              )

for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="type", sum_over='count',
    colors=all_type_palette,
    title="Count by Type", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
order = five_list
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_upset_plot_sum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_upset_plot_sum.pdf"), format="pdf", bbox_inches="tight")

# Venns
venn = partial(custom_venn_dispatch, func=draw_venn, hint_hidden=False)
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['type_group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['type_group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['type_group'] == 'Lung Brush']['ASV_ID'])
skin_set = set(sub_df.loc[sub_df['type_group'] == 'Skin Brush']['ASV_ID'])
scope_set = set(sub_df.loc[sub_df['type_group'] == 'Scope Flush']['ASV_ID'])

name_to_set = {
    "Skin Brush": skin_set,
    "Scope Flush": scope_set,
    "Oral Rinse": oral_set,
    "BAL": bal_set,
    "Lung Brush": lung_set
    }

plt.figure(figsize=(6,6))

ordered_dict = {k: name_to_set[k] for k in five_list}
v = venn(ordered_dict,
         cmap=[all_type_palette[k] for k in five_list],
         fontsize=8,
         alpha=0.45
         )

plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_diagram.pdf"), format="pdf", bbox_inches="tight")

all_names = five_list
columns = {}

for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        # IDs present in all included sets
        included = set.intersection(*(name_to_set[n] for n in combo))

        # Subtract anything present in any excluded set
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included

        if not group_ids:
            continue

        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Flatten to rows
venn_list = [[label, asv] for label, ids in columns.items() for asv in ids]

# Make DataFrame and save
venn_table = pd.DataFrame(venn_list, columns=["grouping", "ASV_ID"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_presence_table.tsv"),
    sep="\t",
    index=False
)

# Fresh Venn from saved/known sets
plt.figure(figsize=(6,6))
ordered_dict = {k: name_to_set[k] for k in five_list}
petal_labels = generate_petal_labels(ordered_dict.values(), fmt="{size}")
bin2lab_map = map_binary_keys_to_labels(petal_labels, five_list)
lab2bin_map = {v: k for k, v in bin2lab_map.items()}

# Compute exclusive ASV_ID lists and sums
all_names = five_list
columns = {}
for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        included = set.intersection(*(name_to_set[n] for n in combo))
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included
        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Save exclusive sums table
pedal_sums = {
    lab2bin_map[label]: int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))
    for label, ids in columns.items()
}

venn_list = [[label, int(sum(raw_asv_sum_dict.get(asv, 0) for asv in ids))]
             for label, ids in columns.items()
             ]
venn_table = pd.DataFrame(venn_list, columns=["grouping", "Sum_count"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_sum_table.tsv"),
    sep="\t",
    index=False
)

v = venn(ordered_dict,
         petal_labels=pedal_sums,
         dataset_labels=five_list,
         cmap=[all_type_palette[k] for k in five_list],
         fontsize=8,
         alpha=0.45)

# Save diagram with sums
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_sum_diagram.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/All_types_venn_sum_diagram.pdf"),
            format="pdf", bbox_inches="tight")
'''

# Three Groups
# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
sub_list = ['Oral Rinse', 'BAL', 'Lung Brush']
sub_df = asv_tax_df.loc[asv_tax_df['type_group'].isin(sub_list)]
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)  # Series with MultiIndex of booleans, values = unique counts

# 2) Plot UpSet (no stacked bars)
upset = UpSet(
    upset_data,
    subset_size='count',       # show unique counts
    element_size=None,
    show_counts=True,
    sort_categories_by='input',
    min_subset_size=0
)

# 3) Color LEFT (category) bars
for name in sub_list:
    upset.style_categories([name], bar_facecolor=three_palette[name], bar_edgecolor="black")

# 4) Render and legend
fig = plt.figure(figsize=(12, 8))
plt.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

handles = [mpatches.Patch(facecolor=three_palette[t], edgecolor="black", label=t) for t in sub_list]
fig.legend(handles=handles, title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by type_group", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_upset_plot_TYPE.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_upset_plot_TYPE.pdf"),
            format="pdf", bbox_inches="tight")

sub_df = asv_tax_df.loc[asv_tax_df['type_group'].isin(sub_list)]
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in sub_list:
        if (t, row['index']) in type_asv_sum_dict:
            val = type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(sub_list, inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(sub_list[::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], sub_list)

# Create and plot the UpSet plot.
upset = UpSet(upset_sum_df, sum_over='count', subset_size='sum',
              element_size=None, show_counts=True,
              sort_categories_by='input', min_subset_size=0,
              intersection_plot_elements=0
              )

for t in three_palette.keys():
    upset.style_categories([t], bar_facecolor=three_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="type", sum_over='count',
    colors=three_palette,
    title="Count by Type", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
order = sub_list
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type", y=1.05)
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_upset_plot_sum_TYPE.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_upset_plot_sum_TYPE.pdf"), format="pdf", bbox_inches="tight")

# Venns
venn = partial(custom_venn_dispatch, func=draw_venn, hint_hidden=False)
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['type_group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['type_group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['type_group'] == 'Lung Brush']['ASV_ID'])

name_to_set = {
    'Oral Rinse': oral_set,
    'BAL': bal_set,
    'Lung Brush': lung_set
    }

plt.figure(figsize=(6,6))

venn3([oral_set, bal_set, lung_set], ('Oral Rinse', 'BAL', 'Lung Brush'),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )
#ordered_dict = {k: name_to_set[k] for k in sub_list}
#v = venn(ordered_dict,
#         cmap=[three_palette[k] for k in sub_list],
#         fontsize=8,
#         alpha=0.45
#         )

plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_diagram_TYPE.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_diagram_TYPE.pdf"), format="pdf", bbox_inches="tight")

all_names = sub_list
columns = {}

for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        # IDs present in all included sets
        included = set.intersection(*(name_to_set[n] for n in combo))

        # Subtract anything present in any excluded set
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included

        if not group_ids:
            continue

        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Flatten to rows
venn_list = [[label, asv] for label, ids in columns.items() for asv in ids]

# Make DataFrame and save
venn_table = pd.DataFrame(venn_list, columns=["grouping", "ASV_ID"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_presence_table_TYPE.tsv"),
    sep="\t",
    index=False
)

# Fresh Venn from saved/known sets
plt.figure(figsize=(6,6))
ordered_dict = {k: name_to_set[k] for k in sub_list}
petal_labels = generate_petal_labels(ordered_dict.values(), fmt="{size}")
bin2lab_map = map_binary_keys_to_labels(petal_labels, sub_list)
lab2bin_map = {v: k for k, v in bin2lab_map.items()}

# Compute exclusive ASV_ID lists and sums
all_names = sub_list
columns = {}
for r in range(1, len(all_names) + 1):
    for combo in combinations(all_names, r):
        included = set.intersection(*(name_to_set[n] for n in combo))
        excluded_names = [n for n in all_names if n not in combo]
        if excluded_names:
            excluded_union = set.union(*(name_to_set[n] for n in excluded_names))
            group_ids = included - excluded_union
        else:
            group_ids = included
        if not group_ids:
            continue
        label = f"{combo[0]}" if len(combo) == 1 else " + ".join(combo)
        columns[label] = sorted(group_ids)

# Save exclusive sums table
pedal_sums = {
    lab2bin_map[label]: int(sum(asv_sum_dict.get(asv, 0) for asv in ids))
    for label, ids in columns.items()
}

venn_list = [[label, int(sum(asv_sum_dict.get(asv, 0) for asv in ids))]
             for label, ids in columns.items()
             ]
venn_table = pd.DataFrame(venn_list, columns=["grouping", "Sum_count"])
venn_table.to_csv(
    os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_sum_table_TYPE.tsv"),
    sep="\t",
    index=False
)
venn3(subsets=pedal_sums, set_labels=('Oral Rinse', 'BAL', 'Lung Brush'),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )
#v = venn(ordered_dict,
#         petal_labels=pedal_sums,
#         dataset_labels=sub_list,
#         cmap=[all_type_palette[k] for k in sub_list],
#         fontsize=8,
#         alpha=0.45)

# Save diagram with sums
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_sum_diagram_TYPE.svg"),
            format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "spark_methods_output_tester/mito/metadata/Three_types_venn_sum_diagram_TYPE.pdf"),
            format="pdf", bbox_inches="tight")