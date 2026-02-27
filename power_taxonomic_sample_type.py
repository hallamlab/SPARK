#!/usr/bin/env python3
"""
power_taxonomic_sample_type.py

Power analysis for taxonomic abundance differences BETWEEN sample types.
Tests: "Is Phylum A more abundant in BAL vs Oral Rinse vs Lung Brush?"
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations

warnings.filterwarnings('ignore')


def aggregate_to_taxonomy(long_df, tax_level='Phylum', min_prevalence=0.1):
    """Aggregate ASV counts to taxonomic level."""
    agg = long_df.groupby(['lmp_id', tax_level])['count'].sum().reset_index()
    wide = agg.pivot(index='lmp_id', columns=tax_level, values='count').fillna(0)

    # Filter by prevalence
    prevalence = (wide > 0).sum(axis=0) / wide.shape[0]
    taxa_keep = prevalence[prevalence >= min_prevalence].index

    return wide[taxa_keep]


def relative_abundance(count_matrix):
    """Convert counts to relative abundance."""
    totals = count_matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    return count_matrix / totals


def bootstrap_patients_sample_types(count_matrix, patient_ids, sample_types,
                                     n_patients, seed=42):
    """Bootstrap resample patients, keeping all their sample types."""
    np.random.seed(seed)

    unique_patients = np.unique(patient_ids)
    boot_patients_selected = np.random.choice(unique_patients, size=n_patients, replace=True)

    patient_draw_count = {}
    boot_indices = []
    boot_patient_list = []
    boot_sample_type_list = []

    for patient in boot_patients_selected:
        draw_idx = patient_draw_count.get(patient, 0)
        patient_draw_count[patient] = draw_idx + 1

        unique_boot_id = f"{patient}__b{draw_idx}"

        patient_samples = np.where(patient_ids == patient)[0]
        boot_indices.extend(patient_samples)
        boot_patient_list.extend([unique_boot_id] * len(patient_samples))
        boot_sample_type_list.extend(sample_types[patient_samples])

    boot_counts = count_matrix[boot_indices, :]
    boot_patient_ids = np.array(boot_patient_list)
    boot_sample_types = np.array(boot_sample_type_list)

    return boot_counts, boot_patient_ids, boot_sample_types


def patient_level_abundance_by_type(count_matrix, patient_ids, sample_types):
    """
    Aggregate to patient level WITHIN each sample type.
    Returns dict: {sample_type: {patient: abundance_vector}}
    """
    unique_stypes = np.unique(sample_types)
    patient_abundances = {}

    for stype in unique_stypes:
        stype_mask = sample_types == stype
        stype_counts = count_matrix[stype_mask, :]
        stype_patients = patient_ids[stype_mask]

        unique_patients = np.unique(stype_patients)
        patient_matrix = []
        patient_list = []

        for patient in unique_patients:
            patient_mask = stype_patients == patient
            # Sum counts across samples of this type from this patient
            patient_counts = stype_counts[patient_mask, :].sum(axis=0)
            patient_matrix.append(patient_counts)
            patient_list.append(patient)

        # Convert to relative abundance
        patient_matrix = np.array(patient_matrix)
        patient_rel = relative_abundance(patient_matrix)

        # Store as dict
        patient_abundances[stype] = {
            patient_list[i]: patient_rel[i, :]
            for i in range(len(patient_list))
        }

    return patient_abundances, unique_stypes


def run_power_simulation(count_matrix, patient_ids, sample_types, taxa_names,
                         n_patients, n_simulations=1000, alpha=0.05, seed=42):
    """
    Power to detect sample type differences in taxonomic abundance.

    For each taxon, test: Does abundance differ between sample types?
    Uses pairwise t-tests with FDR correction.

    Returns power to detect at least one significant difference.
    """
    n_taxa = len(taxa_names)
    power_any = 0

    for i in range(n_simulations):
        # Bootstrap patients
        boot_counts, boot_patients, boot_stypes = bootstrap_patients_sample_types(
            count_matrix, patient_ids, sample_types,
            n_patients, seed=seed+i
        )

        # Get patient-level abundances by sample type
        patient_abund, sample_type_list = patient_level_abundance_by_type(
            boot_counts, boot_patients, boot_stypes
        )

        # For each taxon, test sample type differences
        all_p_values = []

        for taxon_idx in range(n_taxa):
            # Pairwise comparisons between sample types
            pairwise_p = []

            for stype1, stype2 in combinations(sample_type_list, 2):
                # Get patients who have both sample types (paired data)
                patients_both = set(patient_abund[stype1].keys()) & \
                               set(patient_abund[stype2].keys())

                if len(patients_both) >= 3:
                    vals1 = np.array([patient_abund[stype1][p][taxon_idx]
                                     for p in patients_both])
                    vals2 = np.array([patient_abund[stype2][p][taxon_idx]
                                     for p in patients_both])

                    # Paired t-test (same patients, different sites)
                    _, p = stats.ttest_rel(vals1, vals2)
                    pairwise_p.append(p)

            if len(pairwise_p) > 0:
                # Take minimum p-value across pairwise tests for this taxon
                all_p_values.append(min(pairwise_p))

        # FDR correction across all taxa
        if len(all_p_values) > 0:
            reject, _, _, _ = multipletests(all_p_values, alpha=alpha, method='fdr_bh')
            if np.any(reject):
                power_any += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_simulations}...", end='\r')

    print()
    power = power_any / n_simulations
    return power


def main():
    parser = argparse.ArgumentParser(
        description="Taxonomic abundance power - Sample Type Comparisons"
    )
    parser.add_argument("--data-long", required=True)
    parser.add_argument("--sample-sizes", default="10,15,20,25,30,40,50",
                       help="Comma-separated sample sizes (n patients)")
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient-col", default="Participant_ID")
    parser.add_argument("--type-col", default="type_group")
    parser.add_argument("--sample-col", default="lmp_id")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample_sizes = [int(x) for x in args.sample_sizes.split(",")]

    print("="*60)
    print("Taxonomic Abundance - Sample Type Comparisons")
    print("="*60)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Simulations: {args.n_simulations}")
    print()

    # Load data
    print("Loading data...")
    long_df = pd.read_csv(args.data_long, sep='\t')

    all_results = []

    for tax_level in ['Phylum', 'Family']:
        print(f"\n{'='*60}")
        print(f"Processing {tax_level} level")
        print("="*60)

        # Aggregate to taxonomy
        tax_wide = aggregate_to_taxonomy(long_df, tax_level=tax_level, min_prevalence=0.1)

        # Get metadata
        metadata = long_df[[args.sample_col, args.patient_col, args.type_col]].drop_duplicates()
        metadata = metadata.set_index(args.sample_col)

        sample_ids = metadata.index.intersection(tax_wide.index)
        metadata = metadata.loc[sample_ids]
        tax_wide = tax_wide.loc[sample_ids]

        count_matrix = tax_wide.values.astype(float)
        taxa_names = tax_wide.columns.tolist()
        patient_ids = metadata[args.patient_col].values
        sample_types = metadata[args.type_col].values

        print(f"Samples: {count_matrix.shape[0]}, Taxa: {count_matrix.shape[1]}")
        print(f"Patients: {len(np.unique(patient_ids))}")
        print(f"Sample types: {list(np.unique(sample_types))}")

        for n in sample_sizes:
            print(f"\n  n={n} patients", end=' ')

            power = run_power_simulation(
                count_matrix, patient_ids, sample_types, taxa_names,
                n_patients=n,
                n_simulations=args.n_simulations,
                alpha=args.alpha,
                seed=args.seed
            )

            print(f"→ Power={power:.3f}")

            all_results.append({
                'tax_level': tax_level,
                'comparison': 'Sample_Type',
                'n_patients': n,
                'power': power,
                'n_simulations': args.n_simulations
            })

    # Save results
    results_df = pd.DataFrame(all_results)
    outfile = outdir / 'taxonomic_sample_type_power.tsv'
    results_df.to_csv(outfile, sep='\t', index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {outfile}")
    print("="*60)


if __name__ == '__main__':
    main()
