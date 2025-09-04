import pandas as pd
import sys
import os

count_table = sys.argv[1]
nontarget_table = sys.argv[2]
filtered_table = sys.argv[3]
tax_table = sys.argv[4]
asv_abund_threshold = float(sys.argv[5])

count_df = pd.read_csv(count_table, sep='\t', header=0, index_col=0)
nontarget_df = pd.read_csv(nontarget_table, header=0, sep='\t', index_col=0)
nontarget_df.index = [i.split(';', 1)[0] for i in nontarget_df.index.values]
tax_df = pd.read_csv(tax_table, header=0, sep='\t', index_col=0)
tax_df.index = [i.split(';', 1)[0] for i in tax_df.index.values]
qual_tax_df = tax_df.loc[((tax_df['Taxon'] != 'Unassigned'))] # & (tax_df['Consensus'] >= 0.7))]

# subset nontarget table
decon_df = nontarget_df.loc[nontarget_df['BioFactorial'] == 1]

decon_df.to_csv(filtered_table.replace('.tsv', '.decon.tsv'), sep='\t', header=True, index=True)

micro_df = decon_df.loc[~((decon_df['MITOMASTER'] == 0) |
						 (decon_df['BLAST_mito'] == 0)
						 )]
mito_df = decon_df.drop(index=micro_df.index.values)

decon_cnt_df = count_df.loc[count_df.index.isin(decon_df.index.values)]
decon_cnt_df.to_csv(filtered_table.replace('.tsv', '.decon.tsv'), sep='\t', header=True, index=True)

micro_indices = list(set(micro_df.index.values))
mito_indices = list(set(mito_df.index.values))
micro_cnt_df = decon_cnt_df.loc[decon_cnt_df.index.isin(micro_indices)]
mito_cnt_df = decon_cnt_df.loc[decon_cnt_df.index.isin(mito_indices)]
mito_output_file = os.path.join(os.path.dirname(os.path.dirname(filtered_table)),
												'mito/ASVs',
												os.path.basename(filtered_table).replace('.tsv', '.mito.tsv')
												)
mito_cnt_df.to_csv(mito_output_file, sep='\t', header=True, index=True)

abund_filter_df = micro_cnt_df.loc[(micro_cnt_df.sum(axis=1)/micro_cnt_df.sum(axis=1).sum())*100 >= asv_abund_threshold]
tax_filter_df = abund_filter_df.loc[abund_filter_df.index.isin(qual_tax_df.index)]
tax_filter_df.to_csv(filtered_table.replace('.tsv', '.micro.tsv'), sep='\t', header=True, index=True)

print('Filtered ASV Table...')