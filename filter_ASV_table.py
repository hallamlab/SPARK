import pandas as pd
import sys

count_table = sys.argv[1]
metadata_table = sys.argv[2]
filtered_table = sys.argv[3]
count_threshold = int(sys.argv[4])
asv_abund_threshold = float(sys.argv[5])

count_df = pd.read_csv(count_table, sep='\t', header=0, index_col=0)
metadata_df = pd.read_csv(metadata_table, header=0, sep='\t', index_col=0)

low_filter_df = count_df.loc[:, count_df.sum() >= count_threshold]
abund_filter_df = low_filter_df.loc[(low_filter_df.sum(axis=1)/low_filter_df.sum(axis=1).sum())*100 >= asv_abund_threshold]
abund_filter_df.to_csv(filtered_table, sep='\t', header=True, index=True)

#exclude_sample_list = list(metadata_df.loc[metadata_df['Type_Group'].isin(['Scope Flush', 'Skin Brush'])]['sample_asv'])
#exclude_asv_df = abund_filter_df.loc[abund_filter_df.index.isin(exclude_sample_list)]
#exclude_asv_df = exclude_asv_df.loc[:, ~(exclude_asv_df == 0).all(axis=0)]

#filter_df = abund_filter_df[[a for a in abund_filter_df.columns if a not in exclude_asv_df.columns]]
#filter_df = filter_df.loc[~filter_df.index.isin(exclude_sample_list)]

#merge_df = pd.concat([filter_df, exclude_asv_df]).fillna(0)
#merge_df = merge_df[~(merge_df == 0).all(axis=1)].T
#merge_df.to_csv(filtered_table, sep='\t', header=True, index=True)
print('Filtered ASV Table...')

#asv_trimmed_df = count_df.loc[count_df.index.isin(merge_df.index)]
#asv_trimmed_df = asv_trimmed_df[~(asv_trimmed_df == 0).all(axis=1)].T
#asv_trimmed_df.to_csv(filtered_table, sep='\t', header=True, index=True)