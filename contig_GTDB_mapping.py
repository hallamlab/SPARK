
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns



data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'

cols = ["contig_id", "taxid", "rank", "name", "frag_num",
		"frag_retained", "frag_assigned", "frag_agreement"
		]
# Directory containing TSVs
tsv_dir = Path(data_dir + "/asm_taxonomy_results")
# Load and concat
maps_df = pd.concat([
			 pd.read_csv(f, sep='\t', names=cols).assign(source_file=f.name)
			 for f in tsv_dir.glob("*.tsv")],ignore_index=True
			 )
maps_df['sample'] = [x.split('.', 1)[0].rsplit('_', 1)[0] for x in maps_df['source_file']]
maps_df['taxid'] = maps_df['taxid'].astype(int)

cols = ["total_percent", "total_counts", "rank_counts", "rank", "taxid", "name"]
# Directory containing TSVs
tsv_dir = Path(data_dir + "/asm_taxonomy_results")
# Load and concat
tax_df = pd.concat([
			 pd.read_csv(f, sep='\t', names=cols).assign(source_file=f.name)
			 for f in tsv_dir.glob("*_report")],ignore_index=True
			 )
tax_df.columns = tax_df.columns.str.strip()  # Trim column names
tax_df = tax_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
tax_df['sample'] = [x.split('.', 1)[0].rsplit('_', 1)[0] for x in tax_df['source_file']]
tax_df['taxid'] = tax_df['taxid'].astype(int)

merge_df = maps_df.merge(tax_df[['taxid', 'name']], on='taxid')

