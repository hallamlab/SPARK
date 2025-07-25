import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_contents
from upsetplot import UpSet
import matplotlib
from matplotlib_venn import venn3
import os
import matplotlib as mpl
from matplotlib import font_manager as fm, rcParams
import seaborn as sns
from itertools import zip_longest


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

# Replace 'your_file.tsv' with the path to your TSV
asv_df = pd.read_csv(os.path.join(data_dir, 'final_output/ASVs/ASV_final.micro.tsv'), sep='\t', index_col=0)
metadata_df = pd.read_csv(os.path.join(data_dir, 'final_output/metadata/metadata_updated.tsv'), sep='\t')
asv_stack_df = asv_df.stack(future_stack=True).reset_index()
asv_stack_df.columns = ['ASV_ID', 'lmp_id', 'count']
merge_df = asv_stack_df.merge(metadata_df, how='left', on='lmp_id')

filter_df = merge_df.loc[merge_df['count'] > 0]
taxonomy_path = os.path.join(data_dir, 'final_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Feature ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Feature ID']]
tax_df.set_index('Feature ID', inplace=True)
asv_tax_df = filter_df.merge(tax_df, how='left', left_on='ASV_ID', right_index=True)
taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }

for t in asv_tax_df['Taxon']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    asv_tax_df[t] = taxonomy_dict[t]

asv_sum_dict = asv_tax_df.groupby("ASV_ID")["count"].sum().to_dict()
type_asv_sum_dict = asv_tax_df.groupby(["type_group","ASV_ID"])["count"].sum().to_dict()

total_abundance = asv_tax_df.groupby('Phylum')['count'].sum()
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_tax_df['Phylum_plot'] = asv_tax_df["Phylum"].apply(lambda x: x if x in top10 else "Other")
asv_phy_dict = {x:y for x,y in zip(asv_tax_df['ASV_ID'], asv_tax_df['Phylum_plot'])}

# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
group_dict = asv_tax_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()

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

# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
sub_df = asv_tax_df
group_dict = sub_df.groupby("type_group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]
upset_data = upset_data.reorder_levels(["Oral Rinse", "BAL", "Lung Brush"][::-1])

# Create and plot the UpSet plot.
upset = UpSet(upset_data, subset_size='count', element_size=None,
			  sort_categories_by='input', show_counts=True
			  )
for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

plt.title("ASV Membership by type_group")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot.pdf"), format="pdf", bbox_inches="tight")

# Create and plot the UpSet plot.
fig = plt.figure(figsize=(12, 8))
upset = UpSet(upset_data, sum_over='value', subset_size='sum', element_size=None,
			  sort_categories_by='input', show_counts=True
              )
for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

plt.title("ASV Abundance by type_group")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sum.pdf"), format="pdf", bbox_inches="tight")

# Venns
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['type_group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['type_group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['type_group'] == 'Lung Brush']['ASV_ID'])
plt.figure(figsize=(6,6))
venn3([oral_set, bal_set, lung_set], ("Oral Rinse", "BAL", "Lung Brush"),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )

plt.savefig(os.path.join(data_dir, "final_output/metadata/venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "final_output/metadata/venn_diagram.pdf"), format="pdf", bbox_inches="tight")


# Get all possible combinations
only_oral = oral_set - bal_set - lung_set
only_bal = bal_set - oral_set - lung_set
only_lung = lung_set - oral_set - bal_set

oral_bal = (oral_set & bal_set) - lung_set
oral_lung = (oral_set & lung_set) - bal_set
bal_lung = (bal_set & lung_set) - oral_set

all_three = oral_set & bal_set & lung_set

# Convert sets to sorted lists
columns = {
    "Only Oral Rinse": sorted(only_oral),
    "Only BAL": sorted(only_bal),
    "Only Lung Brush": sorted(only_lung),
    "Oral + BAL": sorted(oral_bal),
    "Oral + Lung": sorted(oral_lung),
    "BAL + Lung": sorted(bal_lung),
    "All Three": sorted(all_three),
}

venn_list = []
for k in columns:
    v = columns[k]
    for a in v:
        venn_list.append([k, a])

# Create the DataFrame
venn_table = pd.DataFrame(venn_list, columns=['grouping', 'ASV_ID'])

# Save as TSV
venn_table.to_csv(os.path.join(data_dir, "final_output/metadata/venn3_presence_table.tsv"), sep="\t", index=False)

# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
sub_df = asv_tax_df
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in ["Skin Brush", "Scope Flush", 'Lung Brush', 'BAL', 'Oral Rinse']:
        if (t, row['index']) in type_asv_sum_dict:
            val = type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(["Oral Rinse", "BAL", "Lung Brush"], inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(["Oral Rinse", "BAL", "Lung Brush"][::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], ["Oral Rinse", "BAL", "Lung Brush"])

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
order = ["Oral Rinse", "BAL", "Lung Brush"]
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sum_type_group.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sum_type_group.pdf"), format="pdf", bbox_inches="tight")

# Create a dictionary mapping each type_group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
sub_df = asv_tax_df
group_dict = sub_df.groupby(['type_group'])["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data.reset_index(inplace=True)
upset_sum = []
for i,row in upset_data.iterrows():
    for t in ['Oral Rinse', 'BAL', 'Lung Brush']:
        if (t, row['index']) in type_asv_sum_dict:
            val = type_asv_sum_dict[(t, row['index'])]
            row['count'] = val
            row['type'] = t
            upset_sum.append(list(row))

upset_sum_df = pd.DataFrame(upset_sum, columns=list(upset_data.columns) + ['count', 'type'])
upset_sum_df.set_index(["Oral Rinse", "BAL", "Lung Brush"], inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(["Oral Rinse", "BAL", "Lung Brush"][::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], ["Oral Rinse", "BAL", "Lung Brush"])

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
order = ["Oral Rinse", "BAL", "Lung Brush"]
handles, labels = ax.get_legend_handles_labels()
handles = [handles[labels.index(o)] for o in order]
labels = order
ax.legend(handles, labels, title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Type")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sub_sum_3_Group.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "final_output/metadata/upset_plot_sub_sum_3_Group.pdf"), format="pdf", bbox_inches="tight")