#!/usr/bin/env python

import qiime2
import argparse
import os
import time
import pandas as pd
from Bio import SeqIO
from qiime2.plugins.feature_classifier.pipelines import classify_consensus_vsearch
import tempfile
from qiime2 import Artifact


def load_sequences_to_artifact(fasta_path):
    print(f"Importing sequences from '{fasta_path}'...")
    temp_path = tempfile.NamedTemporaryFile(suffix=".qza", delete=False).name
    seqs_artifact = qiime2.Artifact.import_data('FeatureData[Sequence]', fasta_path)
    return seqs_artifact

def classify_with_vsearch(query_seqs, ref_seqs, ref_taxonomy):
    print("Running classify-consensus-vsearch...")
    start = time.time()
    result = classify_consensus_vsearch(
        query=query_seqs,
        reference_reads=ref_seqs,
        reference_taxonomy=ref_taxonomy,
        #perc_identity=0.99,
        #maxaccepts=10,
        #top_hits_only=True,
        threads=2

    )
    elapsed = time.time() - start
    print(f"Classification completed in {elapsed:.2f} seconds.")
    return result.classification.view(pd.DataFrame)

def save_classifications(df, output_path):
    print(f"Saving classifications to {output_path}...")
    df.to_csv(output_path, sep='\t', index=True)

def calculate_statistics(df, output_path):
    print("Calculating statistics...")
    stats = df['Taxon'].value_counts()
    stats.to_csv(output_path, sep='\t', header=['Count'])

def parse_args():
    parser = argparse.ArgumentParser(description="QIIME2 VSEARCH Classifier")
    parser.add_argument('--input-fasta', '-i', required=True)
    parser.add_argument('--ref-taxonomy', '-t', required=True)   # .qza Taxonomy
    parser.add_argument('--ref-seqs', '-r', required=True)       # .qza Reference Sequences
    parser.add_argument('--output-tsv', '-o', default='classifications.tsv')
    parser.add_argument('--stats-output', '-s', default='classification_stats.tsv')
    return parser.parse_args()

def main():
    args = parse_args()
    query_seqs = load_sequences_to_artifact(args.input_fasta)
    ref_tax = qiime2.Artifact.load(args.ref_taxonomy)
    ref_seqs = qiime2.Artifact.load(args.ref_seqs)

    classifications = classify_with_vsearch(query_seqs, ref_seqs, ref_tax)
    save_classifications(classifications, args.output_tsv)
    calculate_statistics(classifications, args.stats_output)

if __name__ == '__main__':
    main()
