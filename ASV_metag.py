
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

# Define column names
cols = [
    "ASV_ID", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore"
]
# Directory containing TSVs
tsv_dir = Path(data_dir + "/fastq_easy-search_results")

# Load and concat
m8_df = pd.concat(
    [
        pd.read_csv(f, sep='\t', names=cols).assign(source_file=f.name)
        for f in tsv_dir.glob("*.m8")
    ],
    ignore_index=True
)

m8_df['sample'] = [x.split('.', 1)[0].rsplit('_', 1)[0] for x in m8_df['source_file']]

taxonomy_path = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_GG2_tax.tsv')
tax_df = pd.read_csv(taxonomy_path, header=0, sep='\t')
tax_df['Sequence_ID'] = [x.rsplit(';', 1)[0] for x in tax_df['Sequence_ID']]
tax_df.set_index('Sequence_ID', inplace=True)

isa_path = os.path.join(data_dir, 'vsearch_output/indicspecies/Type_status_ISA_results.tsv')
isa_df = pd.read_csv(isa_path, sep='\t')
sig_isa_df = isa_df.loc[((isa_df['type_significance'] == True) | (isa_df['status_significance'] == True)) &
                        ((isa_df['type_stat'] >= 0.6) | (isa_df['status_stat'] >= 0.6))
                        ]
sig_isa_asvs = list(sig_isa_df['ASV_ID'])

m8_tax_df = m8_df.merge(tax_df, left_on='ASV_ID', right_on='Sequence_ID')

taxonomy_dict = {'Domain': [], 'Phylum': [], 'Class': [],
                 'Order': [], 'Family': [], 'Genus': [],
                 'Species': []
                 }
for t in m8_tax_df['Taxonomy']:
    lineage = split_taxa_string(t)
    for l in lineage:
        v = lineage[l]
        taxonomy_dict[l].append(v)
for t in taxonomy_dict:
    m8_tax_df[t] = taxonomy_dict[t]

m8_filt_df = m8_tax_df.loc[((m8_tax_df['length'] >= 100) & (m8_tax_df['pident'] >= 0.999))]
m8_filt_df['indicator'] = [True if a in sig_isa_asvs else False for a in m8_filt_df['ASV_ID']]
m8_filt_df.to_csv(os.path.join(data_dir, "ASV_fastq_mapping.tsv"), sep='\t', index=False)

flurp

m8_grp_df = m8_tax_df.groupby(['sample', 'Phylum'])['ASV_ID'].size().reset_index()

m8_phy_df = m8_grp_df.groupby(['Phylum'])['ASV_ID'].size().reset_index()
#m8_fam_df = m8_grp_df.groupby(['Family'])['ASV_ID'].size().reset_index()

# Compute total abundance for each phylum
total_phylum = m8_phy_df.groupby('Phylum')['ASV_ID'].sum()
#total_family = m8_fam_df.groupby('Family')['ASV_ID'].sum()
top10_phy = total_phylum.sort_values(ascending=False).head(25).index.tolist()
#top10_fam = total_family.sort_values(ascending=False).head(25).index.tolist()
m8_grp_df['Phylum_plot'] = m8_grp_df["Phylum"].apply(lambda x: x if x in top10_phy else "Other")
#m8_grp_df['Family_plot'] = m8_grp_df["Family"].apply(lambda x: x if x in top10_fam else "Other")

# Plot
plt.figure(figsize=(24, 10))
ax = sns.boxplot(
    x='Phylum_plot', y='ASV_ID', data=m8_grp_df,
    color='lightgray',  # box color
    fliersize=0,        # hide default outliers
    linewidth=1,        # box edge width
    showcaps=True
)

# Overlay with swarm plot
sns.swarmplot(
    x='Phylum_plot', y='ASV_ID', data=m8_grp_df,
    hue='Phylum_plot',
    alpha=0.75, dodge=False, ax=ax,
    legend=False
)

plt.title("")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "ASV_metag_swarmplot.svg"))
plt.savefig(os.path.join(data_dir, "ASV_metag_swarmplot.pdf"))
plt.close()

pivot_df = m8_grp_df.pivot_table(index='sample', columns='Phylum_plot',
                              values='ASV_ID', aggfunc='sum', fill_value=0
                              )

# Plot using the inverted data
ax = pivot_df.plot(
                    kind='bar',
                    stacked=True,
                    figsize=(10, 10),
                    edgecolor='gray',
                    linewidth=1,
                    alpha=0.75
                    )

plt.title("")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "ASV_metag_histogram.svg"))
plt.savefig(os.path.join(data_dir, "ASV_metag_histogram.pdf"))
plt.close()