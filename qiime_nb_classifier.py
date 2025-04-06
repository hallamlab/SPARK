#!/usr/bin/env python

import qiime2
import sklearn
from pprint import pprint
import sys
import argparse
import os
import time
from Bio import SeqIO
import pandas as pd

# Import necessary Qiime2 plugins and custom modules
from qiime2.plugins import feature_classifier
import q2_feature_classifier
from q2_feature_classifier.custom import LowMemoryMultinomialNB

# Ensure that the custom classes are recognized during unpickling
sys.modules['q2_feature_classifier.custom'] = q2_feature_classifier.custom

def load_classifier(classifier_path):
    """Load the Qiime2 classifier artifact and return the scikit-learn pipeline."""
    print(f"Loading classifier from '{classifier_path}'...")
    start_time = time.time()
    classifier_artifact = qiime2.Artifact.load(classifier_path)
    pipeline = classifier_artifact.view(sklearn.pipeline.Pipeline)
    elapsed_time = time.time() - start_time
    print(f"Classifier loaded in {elapsed_time:.2f} seconds.")
    return pipeline

def load_sequences(fasta_path):
    """Load sequences from a FASTA file."""
    print(f"Loading sequences from '{fasta_path}'...")
    start_time = time.time()
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        seq_ids.append(record.id)
    elapsed_time = time.time() - start_time
    print(f"{len(sequences)} sequences loaded in {elapsed_time:.2f} seconds.")
    return sequences, seq_ids

def classify_sequences(pipeline, sequences):
    """Classify sequences using the provided pipeline."""
    print("Classifying sequences...")
    start_time = time.time()
    try:
        predictions = pipeline.predict(sequences)
        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba(sequences)
        else:
            probabilities = None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        sys.exit(1)
    elapsed_time = time.time() - start_time
    print(f"Sequences classified in {elapsed_time:.2f} seconds.")
    return predictions, probabilities

def save_classifications(output_path, seq_ids, predictions, probabilities=None):
    """Save the classifications and confidence scores to a TSV file."""
    print(f"Saving classifications to '{output_path}'...")
    start_time = time.time()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = {
        'Sequence_ID': seq_ids,
        'Taxonomy': predictions
    }
    if probabilities is not None:
        # If probabilities are available, include them
        # This assumes binary classification; modify as needed for multi-class
        # For multi-class, you might want to store probabilities for each class
        # Here, we'll store the max probability as confidence
        max_probs = probabilities.max(axis=1)
        data['Confidence'] = max_probs
    df = pd.DataFrame(data)
    df.to_csv(output_path, sep='\t', index=False)
    elapsed_time = time.time() - start_time
    print(f"Classifications saved in {elapsed_time:.2f} seconds.")

def calculate_statistics(predictions, stats_output_path):
    """Calculate statistics from the classifications and save them."""
    print(f"Calculating statistics...")
    start_time = time.time()
    # Example: Count of each taxonomic classification
    taxonomy_counts = pd.Series(predictions).value_counts()
    # Save statistics to a TSV file
    taxonomy_counts.to_csv(stats_output_path, sep='\t', header=['Count'])
    elapsed_time = time.time() - start_time
    print(f"Statistics calculated and saved to '{stats_output_path}' in {elapsed_time:.2f} seconds.")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Classify sequences using a Qiime2 classifier.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input-fasta', required=True, help='Path to input FASTA file containing query sequences.')
    parser.add_argument('-c', '--classifier', required=True, help='Path to Qiime2 classifier artifact (.qza).')
    parser.add_argument('-o', '--output-tsv', default='classifications.tsv', help='Path to output TSV file for classifications.')
    parser.add_argument('-s', '--stats-output', default='classification_stats.tsv', help='Path to output TSV file for statistics.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Set verbose mode
    if args.verbose:
        print("Verbose mode enabled.")
        print(f"Arguments received: {args}")

    # Load the classifier pipeline
    pipeline = load_classifier(args.classifier)

    # Load the input sequences
    sequences, seq_ids = load_sequences(args.input_fasta)

    # Classify the sequences
    predictions, probabilities = classify_sequences(pipeline, sequences)

    # Save the classifications to a TSV file
    save_classifications(args.output_tsv, seq_ids, predictions, probabilities)

    # Calculate and save statistics
    calculate_statistics(predictions, args.stats_output)

    print("Process completed successfully.")

if __name__ == '__main__':
    main()
