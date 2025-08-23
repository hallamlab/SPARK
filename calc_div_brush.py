#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import argparse
import os
from skbio.diversity.alpha import shannon


def calc_shannon(counts):
    return shannon(counts)

def main():
    data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'

    # Create output directory if it doesn't exist
    output_dir = os.path.join(data_dir, "spark_combined_output/brush/diversity")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    asv_table = os.path.join(data_dir, 'spark_combined_output/brush/ASVs/ASV_final.micro.tsv')

    output_shannon = os.path.join(data_dir, 'spark_combined_output/brush/diversity/shannon.tsv')
    output_bray = os.path.join(data_dir, 'spark_combined_output/brush/diversity/bray.tsv')

    # Load input tables
    asv_df = pd.read_csv(asv_table, sep="\t", index_col=0).T
    asv_df = asv_df[~(asv_df == 0).all(axis=1)]

    # Compute Shannon diversity for each sample from the ASV counts
    shannon_results = {}
    for sample in asv_df.index:
        counts = asv_df.loc[sample].values
        shannon_results[sample] = calc_shannon(counts)
    shannon_df = pd.DataFrame.from_dict(shannon_results, orient="index", columns=["Shannon"])
    shannon_df.index.name = "sample"
    shannon_df.to_csv(output_shannon, sep="\t")

    print(f"Shannon diversity saved to {output_shannon}")

    # Compute pairwise Bray-Curtis distances between sample
    # Here, each row in asv_df is assumed to be a sample
    asv_array = asv_df.values
    distances = pdist(asv_array, metric="braycurtis")
    bray_curtis_matrix = squareform(distances)
    bray_df = pd.DataFrame(bray_curtis_matrix, index=asv_df.index, columns=asv_df.index)
    bray_df.index.name = "sample"
    bray_df.to_csv(output_bray, sep="\t")

    print(f"Bray-Curtis beta diversity saved to {output_bray}")

    # Compute pairwise Jaccard distances between sample
    jaccard_output = os.path.join(data_dir, 'spark_combined_output/brush/diversity/jaccard.tsv')

    asv_binary = (asv_df > 0).astype(int).values  # Convert counts to presence/absence
    distances = pdist(asv_binary, metric="jaccard")
    jaccard_matrix = squareform(distances)
    jaccard_df = pd.DataFrame(jaccard_matrix, index=asv_df.index, columns=asv_df.index)
    jaccard_df.index.name = "sample"
    jaccard_df.to_csv(jaccard_output, sep="\t")

    print(f"Jaccard beta diversity saved to {jaccard_output}")


if __name__ == "__main__":
    main()
