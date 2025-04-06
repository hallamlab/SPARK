#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import argparse
import os


# Try to import the Shannon diversity function from scikit-bio; if not available, define our own.
try:
    from skbio.diversity.alpha import shannon
    def calc_shannon(counts):
        return shannon(counts)
except ImportError:
    def calc_shannon(counts):
        counts = np.array(counts)
        total = counts.sum()
        if total == 0:
            return 0
        proportions = counts / total
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log(proportions))

def main():

    # Create output directory if it doesn't exist
    output_dir = "vsearch_output/diversity"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    asv_table = 'vsearch_output/ASVs/ASV_filtered.tsv'
    taxonomy_table = 'vsearch_output/taxonomy/ASV_GG2_tax.tsv'

    output_shannon = 'vsearch_output/diversity/shannon.tsv'
    output_bray = 'vsearch_output/diversity/bray.tsv'

    # Load input tables
    asv_df = pd.read_csv(asv_table, sep="\t", index_col=0).T
    asv_df = asv_df[~(asv_df == 0).all(axis=1)]
    asv_df.index = [x.rsplit('_', 1)[0] for x in asv_df.index]
    tax_df = pd.read_csv(taxonomy_table, sep="\t", index_col=0)

    # Compute Shannon diversity for each sample from the ASV counts
    shannon_results = {}
    for sample in asv_df.index:
        counts = asv_df.loc[sample].values
        shannon_results[sample] = calc_shannon(counts)
    shannon_df = pd.DataFrame.from_dict(shannon_results, orient="index", columns=["Shannon"])
    shannon_df.index.name = "sample"
    shannon_df.to_csv(output_shannon, sep="\t")
    print(f"Shannon diversity saved to {output_shannon}")

    # Compute pairwise Bray-Curtis distances between samples
    # Here, each row in asv_df is assumed to be a sample
    asv_array = asv_df.values
    distances = pdist(asv_array, metric="braycurtis")
    bray_curtis_matrix = squareform(distances)
    bray_df = pd.DataFrame(bray_curtis_matrix, index=asv_df.index, columns=asv_df.index)
    bray_df.index.name = "sample"
    bray_df.to_csv(output_bray, sep="\t")
    print(f"Bray-Curtis beta diversity saved to {output_bray}")

if __name__ == "__main__":
    main()
