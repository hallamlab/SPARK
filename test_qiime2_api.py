#!/usr/bin/env python3

import sys
import os
import qiime2
from qiime2.plugins.feature_classifier.pipelines import classify_consensus_vsearch
from pathlib import Path

# ---- EDIT THESE PATHS IF NEEDED ----
REF_TAX = "/cephfs/rjm_work/si_asv/NF_V4_ncbi_output/taxonomy_reference/silva-138_2-ssu-nr99-tax.qza"
REF_SEQS = "/cephfs/rjm_work/si_asv/NF_V4_ncbi_output/taxonomy_reference/silva-138_2-ssu-nr99-seqs-DNA.qza"
# -----------------------------------

TEST_FASTA = "test_seqs.fasta"
TEST_QZA = "test_seqs.qza"
OUT_QZA = "test_result.qza"

def write_test_fasta():
    with open(TEST_FASTA, "w") as f:
        f.write(
            ">seq1\n"
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n"
            ">seq2\n"
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGA\n"
        )

def main():
    print("Python:", sys.executable)
    print("QIIME2:", qiime2.__version__)

    print("Writing test FASTA...")
    write_test_fasta()

    print("Importing FASTA as QIIME2 artifact...")
    seqs = qiime2.Artifact.import_data("FeatureData[Sequence]", TEST_FASTA)
    seqs.save(TEST_QZA)

    print("Loading reference artifacts...")
    ref_tax = qiime2.Artifact.load(REF_TAX)
    ref_seqs = qiime2.Artifact.load(REF_SEQS)

    print("Running classify_consensus_vsearch...")
    res = classify_consensus_vsearch(
        query=seqs,
        reference_reads=ref_seqs,
        reference_taxonomy=ref_tax,
        threads=2
    )

    print("Saving result...")
    res.classification.save(OUT_QZA)

    print("Viewing results:")
    df = res.classification.view(pd.DataFrame) if False else None
    print("SUCCESS — QIIME2 API + feature-classifier working")

if __name__ == "__main__":
    main()

