import pandas as pd
import sys
from Bio import SeqIO

# Inputs
count_table = sys.argv[1]
filtered_table = sys.argv[2]
count_threshold = int(sys.argv[3])
asv_abund_threshold = float(sys.argv[4])
input_fasta = sys.argv[5]
filtered_fasta = sys.argv[6]

# Load count table
count_df = pd.read_csv(count_table, sep='\t', header=0, index_col=0)

# Filter ASVs by relative abundance
abund_filter_df = count_df.loc[
    (count_df.sum(axis=1) / count_df.values.sum()) * 100 >= asv_abund_threshold
]

# Filter samples by total count
low_filter_df = abund_filter_df.loc[:, abund_filter_df.sum() >= count_threshold]

# Save filtered count table
low_filter_df.to_csv(filtered_table, sep='\t', header=True, index=True)

# Subset FASTA
filtered_asvs = set(low_filter_df.index)

with open(filtered_fasta, "w") as out_fa:
    for record in SeqIO.parse(input_fasta, "fasta"):
        if record.id.split(';')[0] in filtered_asvs:
            SeqIO.write(record, out_fa, "fasta")
