import pandas as pd
import sys

count_table = sys.argv[1]
nontarget_table = sys.argv[2]
filtered_table = sys.argv[3]
count_threshold = int(sys.argv[4])
asv_abund_threshold = float(sys.argv[5])

count_df = pd.read_csv(count_table, sep='\t', header=0, index_col=0)
nontarget_df = pd.read_csv(nontarget_table, header=0, sep='\t', index_col=0)
nontarget_df.index = [i.split(';', 1)[0] for i in nontarget_df.index.values]
# subset nontarget table
decon_df = nontarget_df.loc[nontarget_df['BioFactorial'] == 1]

decon_df.to_csv(filtered_table.replace('.tsv', '.decon.tsv'), sep='\t', header=True, index=True)

micro_df = decon_df.loc[~((decon_df['MITOMASTER'] == 0) |
						 (decon_df['BLAST_mito'] == 0)
						 )]
mito_df = decon_df.drop(index=micro_df.index.values)

decon_cnt_df = count_df.loc[decon_df.index.values]
decon_cnt_df.to_csv(filtered_table.replace('.tsv', '.decon.tsv'), sep='\t', header=True, index=True)

micro_indices = list(set(micro_df.index.values))
mito_indices = list(set(mito_df.index.values))
micro_cnt_df = decon_cnt_df.loc[micro_indices]
mito_cnt_df = decon_cnt_df.loc[mito_indices]
mito_cnt_df.to_csv(filtered_table.replace('.tsv', '.mito.tsv'), sep='\t', header=True, index=True)

abund_filter_df = micro_cnt_df.loc[(micro_cnt_df.sum(axis=1)/micro_cnt_df.sum(axis=1).sum())*100 >= asv_abund_threshold]
low_filter_df = abund_filter_df.loc[:, abund_filter_df.sum() >= count_threshold]
low_filter_df.to_csv(filtered_table.replace('.tsv', '.micro.tsv'), sep='\t', header=True, index=True)

print('Filtered ASV Table...')