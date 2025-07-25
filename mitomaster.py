#!/usr/bin/env python3
import os
import time
import requests
from pathlib import Path

data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/ASVs/chunks'
output_file = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/final_output/mitomap/mitomaster_combined.tsv'

# List all .fasta files
fasta_files = sorted(Path(data_dir).glob("*.fasta"))

# Collect results
all_results = []

for i, fasta in enumerate(fasta_files, 1):
    print(f"Processing {fasta.name} ({i}/{len(fasta_files)})")

    with open(fasta, 'rb') as f:
        files = {
            "file": f,
            "fileType": ('', 'sequences'),
            "output": ('', 'hsd')
        }
        try:
            response = requests.post("https://mitomap.org/mitomaster/websrvc.cgi", files=files)
            response.raise_for_status()
            output = response.text

            # Keep header only once
            if i == 1:
                all_results.append(output)
            else:
                all_results.append('\n'.join(output.splitlines()[1:]))

        except Exception as e:
            print(f"Error processing {fasta.name}: {e}")

    #time.sleep(10)

# Write combined output
with open(output_file, 'w') as f:
    f.write('\n'.join(all_results))

print(f"All results saved to {output_file}")






'''
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("your_table.tsv", sep="\t", index_col=0)

def confusion_counts(df, pred_col):
    TP = ((df['Contaminant'] == 1) & (df[pred_col] == 1)).sum()
    FP = ((df['Contaminant'] == 0) & (df[pred_col] == 1)).sum()
    TN = ((df['Contaminant'] == 0) & (df[pred_col] == 0)).sum()
    FN = ((df['Contaminant'] == 1) & (df[pred_col] == 0)).sum()
    return pd.Series({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})

# Calculate confusion components for both databases
db1_results = confusion_counts(df, 'DB1_Called_Contam')
db2_results = confusion_counts(df, 'DB2_Called_Contam')

# Combine into a single dataframe for plotting
conf_matrix_df = pd.DataFrame({'DB1': db1_results, 'DB2': db2_results})

# Plot
conf_matrix_df.T.plot(kind='bar', stacked=False, figsize=(8, 5), colormap='Set2')
plt.ylabel("Count")
plt.title("Confusion Matrix Breakdown by Database")
plt.xticks(rotation=0)
plt.legend(title="Outcome")
plt.tight_layout()
plt.show()
'''