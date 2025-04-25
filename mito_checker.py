#!/usr/bin/env python3
import os
import time
import requests
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
mitomaster_full_file = os.path.join(data_dir, 'vsearch_output/mitomap/Mitomaster_all.csv')
mito_blast_file = os.path.join(data_dir, 'vsearch_output/mitomap/mito_ncbi.blast6.tsv')
gg2_full_file = os.path.join(data_dir, 'vsearch_output/taxonomy/ASV_GG2_tax.full-length.tsv')
biof_file = os.path.join(data_dir, 'vsearch_output/mitomap/ssu_pipeline_contaminants.blast6.tsv')

mmfl_df = pd.read_csv(mitomaster_full_file, header=0, sep=',')
mibl_df = pd.read_csv(mito_blast_file, header=0, sep='\t')
mibl_df['percov'] = (mibl_df['length'] / mibl_df['qlen']) * 100
mibl_df = mibl_df.loc[((mibl_df['pident'] >= 97) & (mibl_df['percov'] >= 51.0))]
ggfl_df = pd.read_csv(gg2_full_file, header=0, sep='\t')
ggfl_df = ggfl_df.set_index('Sequence_ID').reset_index()
biof_df = pd.read_csv(biof_file, header=0, sep='\t')
biof_df['percov'] = (biof_df['length'] / biof_df['qlen']) * 100
biof_df = biof_df.loc[((biof_df['pident'] >= 97) & (biof_df['percov'] >= 51.0))]

master_df = ggfl_df[['Sequence_ID']].copy()
master_df['BioFactorial'] = [0 if x in list(biof_df['qseqid']) else 1 for x in master_df['Sequence_ID']]
master_df['Qiime_NB_FULL'] = [0 if 'mitochondria' in t.lower() else 1 for x,t in zip(ggfl_df['Sequence_ID'], ggfl_df['Taxonomy'])]
master_df['MITOMASTER'] = [0 if x in list(mmfl_df['Sequence']) else 1 for x in master_df['Sequence_ID']]
master_df['BLAST_mito'] = [0 if x in list(mibl_df['qseqid']) else 1 for x in master_df['Sequence_ID']]

sub_biof_df = biof_df[['qseqid', 'sseqid', 'pident']]
sub_biof_df.columns = ['Sequence_ID', 'BF_ID', 'BF_pid']
ggfl_df.columns = ['Sequence_ID', 'FL_Taxonomy', 'FL_Confidence']
sub_mibl_df = mibl_df[['qseqid', 'sseqid', 'pident']]
sub_mibl_df.columns = ['Sequence_ID', 'MI_Accession', 'MI_pid']

master_df = master_df.merge(sub_biof_df, on='Sequence_ID', how='left')
master_df = master_df.merge(ggfl_df, on='Sequence_ID', how='left')
master_df = master_df.merge(sub_mibl_df, on='Sequence_ID', how='left')
master_df.drop_duplicates(subset='Sequence_ID', inplace=True)
master_df.to_csv(os.path.join(data_dir, 'vsearch_output/mitomap/nontarget.master.tsv'), sep='\t', index=False)

# Step order
steps = ['BioFactorial', 'Qiime_NB_FULL', 'MITOMASTER', 'BLAST_mito'] #, 'BLAST_human']
# Track cumulative contaminant flags
contaminant_flags = pd.Series(False, index=master_df.index)
# Store results
results = []
for step in steps:
    new_contaminants = (master_df[step] == 0) & (~contaminant_flags)
    contaminant_flags |= new_contaminants
    num_contaminants = contaminant_flags.sum()
    num_microbes = len(master_df) - num_contaminants
    results.append({
        'Method': step,
        'Non-Target': num_contaminants,
        'Microbes': num_microbes
    })
# Prepare DataFrame
plot_df = pd.DataFrame(results)
plot_df_melted = plot_df.melt(id_vars='Method', value_vars=['Non-Target', 'Microbes'],
                              var_name='Classification', value_name='Count')
# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
ax = sns.lineplot(
    data=plot_df_melted,
    x='Method', y='Count', hue='Classification',
    marker='o', linewidth=2, markersize=8
)
# Add text labels at each point
for i, row in plot_df_melted.iterrows():
    ax.text(x=i % len(steps), y=row['Count'] + 10, s=str(row['Count']),
            ha='center', va='bottom', fontsize=9)
# Legend outside
plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.title("Cumulative ASV Classification: Microbes vs Non-Target")
plt.ylabel("Number of ASVs")
plt.xlabel("Method")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/non-target_plot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/non-target_plot.pdf"))
plt.close()


# Step order
steps = ['BioFactorial', 'Qiime_NB_FULL', 'MITOMASTER', 'BLAST_mito'] #, 'BLAST_human']
# Track cumulative contaminant flags
contaminant_flags = pd.Series(False, index=master_df.index)
# Store results
results = []
for step in steps:
    new_contaminants = (master_df[step] == 0) & (~contaminant_flags)
    contaminant_flags |= new_contaminants
    num_contaminants = contaminant_flags.sum()
    num_microbes = len(master_df) - num_contaminants
    if step == 'BioFactorial':
        biof_val = num_contaminants
    else:
        num_contaminants = num_contaminants - biof_val
    results.append({
        'Method': step,
        'Host': num_contaminants,
        'Microbial': num_microbes
    })
# Prepare DataFrame
plot_df = pd.DataFrame(results)
plot_df_melted = plot_df.melt(id_vars='Method', value_vars=['Host', 'Microbial'],
                              var_name='Classification', value_name='Count')
plot_df_melted = plot_df_melted.loc[plot_df_melted['Method'] != 'BioFactorial']
# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
ax = sns.lineplot(
    data=plot_df_melted,
    x='Method', y='Count', hue='Classification',
    marker='o', linewidth=2, markersize=8
)
# Add text labels at each point
for i, row in plot_df_melted.iterrows():
    ax.text(x=i % len(steps)-1, y=row['Count'] + 10, s=str(row['Count']),
            ha='center', va='bottom', fontsize=9)
# Legend outside
plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.title("Cumulative ASV Classification: Microbial vs Host")
plt.ylabel("Number of ASVs")
plt.xlabel("Method")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/Microbial_Host_plot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/Microbial_Host_plot.pdf"))
plt.close()









# Step order
steps = ['BioFactorial', 'Qiime_NB_FULL', 'MITOMASTER', 'BLAST_mito'] #, 'BLAST_human']
# Track cumulative contaminant flags
contaminant_flags = pd.Series(False, index=master_df.index)
# Store results
results = []
for step in steps:
    contaminants = master_df[step] == 0
    num_contaminants = contaminants.sum()
    num_microbes = len(master_df) - num_contaminants
    results.append({
        'Method': step,
        'Non-Target': num_contaminants,
        'Microbes': num_microbes
    })
# Prepare DataFrame
plot_df = pd.DataFrame(results)
plot_df_melted = plot_df.melt(id_vars='Method', value_vars=['Non-Target', 'Microbes'],
                              var_name='Classification', value_name='Count')
plot_df_melted = plot_df_melted.loc[plot_df_melted['Classification'] == 'Non-Target']

# Plot
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
ax = sns.catplot(
    data=plot_df_melted, kind="bar",
    x='Method', y='Count', # hue='Classification',
    legend=False
)

# Legend outside
#plt.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.title("Non-Target")
plt.ylabel("Number of ASVs")
plt.xlabel("Method")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/Non-Target_barplot.svg"))
plt.savefig(os.path.join(data_dir, "vsearch_output/mitomap/Non-Target_barplot.pdf"))
plt.close()