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

color_palette = {"Blue": "#377eb8",
                 "Orange": "#ff7f00",
                 "Green": "#4daf4a",
                 "Purple": "#984ea3",
                 "Pink": "#e41a1c",
                 "Yellow": "#fdbf6f",
                 "Cyan": "#a6cee3",
                 "Red": "#e31a1c",
                 "Brown": "#8b4513",
                 "Teal": "#008080",
                 "Magenta": "#d62728",
                 "Olive": "#808000",
                 "Turquoise": "#40E0D0",
                 "Lime": "#32CD32",
                 "Gray": "#808080",
                 "Black": "#000000"
                 }

color_map = {'Bacteroidota': 'Blue',
             'Patescibacteria': 'Orange',
             'Firmicutes_D': 'Green',
             'Proteobacteria': 'Red',
             'Fusobacteriota': 'Purple',
             'Firmicutes_C': 'Brown',
             'Thermoplasmatota': 'Pink',
             'Actinobacteriota': 'Gray',
             'Firmicutes_A': 'Olive',
             'Campylobacterota': 'Turquoise',
             'Acidobacteriota': 'Yellow',
             'Planctomycetota': 'Magenta',
             'Verrucomicrobiota': 'Lime',
             'Spirochaetota': 'Teal',
             'Other': 'Black'
             }

data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'

# Replace 'your_file.tsv' with the path to your TSV
asv_df = pd.read_csv(os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.micro.tsv'), sep='\t', index_col=0)
asv_df.columns = [x.rsplit('_', 1)[0] for x in asv_df.columns]
metadata_df = pd.read_csv(os.path.join(data_dir, 'ref_db/spark_metadata.tsv'), sep='\t')
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]
asv_stack_df = asv_df.stack(future_stack=True).reset_index()
asv_stack_df.columns = ['ASV_ID', 'sample', 'count']
merge_df = asv_stack_df.merge(metadata_df, how='left', on='sample')

filter_df = merge_df.loc[merge_df['count'] > 0]
taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv')
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
type_asv_sum_dict = asv_tax_df.groupby(["Type_Group","ASV_ID"])["count"].sum().to_dict()

total_abundance = asv_tax_df.groupby('Phylum')['count'].sum()
top10 = total_abundance.sort_values(ascending=False).head(10).index.tolist()
asv_tax_df['Phylum_plot'] = asv_tax_df["Phylum"].apply(lambda x: x if x in top10 else "Other")
asv_phy_dict = {x:y for x,y in zip(asv_tax_df['ASV_ID'], asv_tax_df['Phylum_plot'])}

# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
group_dict = asv_tax_df.groupby("Type_Group")["ASV_ID"].apply(set).to_dict()

all_type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           #'Failed-QC': 'lightgray'
           }

three_palette = {'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A'
           }

# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df#.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby("Type_Group")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]
upset_data = upset_data.reorder_levels(["Skin Brush", "Scope Flush", "Oral Rinse", "BAL", "Lung Brush"][::-1])

# Create and plot the UpSet plot.
upset = UpSet(upset_data, subset_size='count', element_size=None,
			  sort_categories_by='input', show_counts=True
			  )
for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

plt.title("ASV Membership by Type_Group")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub.pdf"), format="pdf", bbox_inches="tight")

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

plt.title("ASV Abundance by Type_Group")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum.pdf"), format="pdf", bbox_inches="tight")


'''
# Create and plot the UpSet plot.
upset = UpSet(upset_data, sum_over='value', subset_size='sum', element_size=None,
			  show_counts=True, sort_categories_by='input', intersection_plot_elements=0
			  )
for t in all_type_palette.keys():
    upset.style_categories([t], bar_facecolor=all_type_palette[t], bar_edgecolor="black")

upset.add_stacked_bars(
    by="Phylum_plot", colors=color_map,
    title="Count by Phylum", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Abundance by Type_Group, stack by top 10 Phyla")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_Phylum.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_Phylum.pdf"), format="pdf", bbox_inches="tight")
'''
# Venns
# Create venn3 with custom colors
oral_set = set(sub_df.loc[sub_df['Type_Group'] == 'Oral Rinse']['ASV_ID'])
bal_set = set(sub_df.loc[sub_df['Type_Group'] == 'BAL']['ASV_ID'])
lung_set = set(sub_df.loc[sub_df['Type_Group'] == 'Lung Brush']['ASV_ID'])
plt.figure(figsize=(6,6))
venn3([oral_set, bal_set, lung_set], ("Oral Rinse", "BAL", "Lung Brush"),
      set_colors=(three_palette['Oral Rinse'], three_palette['BAL'], three_palette['Lung Brush']),
      alpha=0.6
      )

plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/venn_diagram.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/venn_diagram.pdf"), format="pdf", bbox_inches="tight")


'''
# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby("Participant_ID")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]

# Create and plot the UpSet plot.
upset = UpSet(upset_data, subset_size='count',
			  element_size=40, show_counts=True,
			  sort_categories_by='input', min_subset_size=0,
  			  intersection_plot_elements=0
			  )
upset.add_stacked_bars(
    by="Phylum_plot", colors=color_map,
    title="Count by Phylum", elements=10
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 24
axes = upset.plot(fig=fig)

ax = axes['extra0']
ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Patient")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_Patient.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_Patient.pdf"), format="pdf", bbox_inches="tight")

# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby("status")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]

# Create and plot the UpSet plot.
upset = UpSet(upset_data, subset_size='count',
			  element_size=None, show_counts=True,
			  sort_categories_by='input', min_subset_size=0,
			  intersection_plot_elements=0
			  )
upset.add_stacked_bars(
    by="Phylum_plot", colors=color_map,
    title="Count by Phylum", elements=3
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra0']
ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Patient")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_status.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_status.pdf"), format="pdf", bbox_inches="tight")
'''



'''
# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby("Participant_ID")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]

# Create and plot the UpSet plot.
upset = UpSet(upset_data, sum_over='value', subset_size='sum',
			  element_size=40, show_counts=True,
			  sort_categories_by='input', min_subset_size=0,
  			  #intersection_plot_elements=0
			  )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 12
axes = upset.plot(fig=fig)

#ax = axes['extra0']
#ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Patient")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_Patient.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_Patient.pdf"), format="pdf", bbox_inches="tight")

# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby("status")["ASV_ID"].apply(set).to_dict()
# Now create the upset data from the dictionary.
upset_data = from_contents(group_dict)
upset_data.columns = ['index']
upset_data['value'] = [asv_sum_dict[x] for x in upset_data['index']]
upset_data['Phylum_plot'] = [asv_phy_dict[x] for x in upset_data['index']]

# Create and plot the UpSet plot.
upset = UpSet(upset_data, sum_over='value', subset_size='sum',
			  element_size=None, show_counts=True,
			  sort_categories_by='input', min_subset_size=0,
			  #intersection_plot_elements=0
			  )
upset.add_stacked_bars(
    by="Phylum_plot", colors=color_map,
    title="Count by Phylum", elements=3
    )

fig = plt.figure(figsize=(12, 8))
matplotlib.rcParams["font.size"] = 6
axes = upset.plot(fig=fig)

ax = axes['extra1']
ax.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title("ASV Membership by Patient")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_status.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_status.pdf"), format="pdf", bbox_inches="tight")
'''













# Create a dictionary mapping each Type_Group to a set of ASV_IDs that are present.
sub_list = ['Lung Brush', 'BAL', 'Oral Rinse']
ex_list = ['Skin Brush', 'Scope Flush']
sub_df = asv_tax_df#.loc[~asv_tax_df['Type_Group'].isin(ex_list)]
group_dict = sub_df.groupby(['Type_Group'])["ASV_ID"].apply(set).to_dict()
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
upset_sum_df.set_index(["Skin Brush", "Scope Flush", "Oral Rinse", "BAL", "Lung Brush"], inplace=True)
upset_sum_df = upset_sum_df.reorder_levels(["Skin Brush", "Scope Flush", "Oral Rinse", "BAL", "Lung Brush"][::-1])
upset_sum_df['type'] = pd.Categorical(upset_sum_df['type'], ["Skin Brush", "Scope Flush", "Oral Rinse", "BAL", "Lung Brush"])

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
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_Type_Group.svg"), format="svg", bbox_inches="tight")
plt.savefig(os.path.join(data_dir, "vsearch_output/metadata/upset_plot_sub_sum_Type_Group.pdf"), format="pdf", bbox_inches="tight")






