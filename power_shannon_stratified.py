#!/usr/bin/env python3
"""
power_shannon_stratified.py

Shannon diversity power analysis stratified by sample type.
Uses patient-level aggregation and t-tests.
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from simulate_data import bootstrap_patients, filter_by_sample_type

warnings.filterwarnings('ignore')


def shannon_diversity(count_matrix):
    """
    Compute Shannon diversity for each sample.

    Parameters
    ----------
    count_matrix : np.ndarray
        Samples × ASVs count matrix

    Returns
    -------
    shannon : np.ndarray
        Shannon diversity for each sample
    """
    # Convert to relative abundance
    rel_abund = count_matrix / count_matrix.sum(axis=1, keepdims=True)

    # Shannon: -sum(p * log(p)) for p > 0
    shannon = np.zeros(count_matrix.shape[0])
    for i in range(count_matrix.shape[0]):
        p = rel_abund[i, :]
        p = p[p > 0]  # Remove zeros
        shannon[i] = -np.sum(p * np.log(p))

    return shannon


def aggregate_shannon_to_patient(count_matrix, patient_ids):
    """
    Compute mean Shannon diversity per patient.

    Parameters
    ----------
    count_matrix : np.ndarray
        Samples × ASVs
    patient_ids : np.ndarray
        Patient ID for each sample

    Returns
    -------
    patient_shannon : dict
        {patient_id: mean_shannon}
    """
    shannon = shannon_diversity(count_matrix)

    patient_shannon = {}
    unique_patients = np.unique(patient_ids)

    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_shannon[patient] = shannon[patient_mask].mean()

    return patient_shannon


def bootstrap_patients_true_null(count_matrix, patient_ids, case_status,
                                  n_cancer, n_control, seed=42):
    """
    TRUE NULL: Pool all patients and randomly assign cancer/control labels.

    This breaks the link between labels and compositions.
    Type I error should be ≈ 0.05 if test is calibrated.
    """
    np.random.seed(seed)

    # Pool ALL patients (ignore original labels)
    unique_patients = np.unique(patient_ids)

    # Sample n_cancer + n_control patients with replacement
    n_total = n_cancer + n_control
    boot_patients_selected = np.random.choice(unique_patients, size=n_total, replace=True)

    # Randomly assign labels
    random_labels = np.random.permutation(
        ['Cancer'] * n_cancer + ['Control'] * n_control
    )

    # Build bootstrap dataset with random labels
    patient_draw_count = {}
    boot_indices = []
    boot_patient_list = []
    boot_case_list = []

    for i, patient in enumerate(boot_patients_selected):
        draw_idx = patient_draw_count.get(patient, 0)
        patient_draw_count[patient] = draw_idx + 1

        unique_boot_id = f"{patient}__b{draw_idx}"

        patient_samples = np.where(patient_ids == patient)[0]
        boot_indices.extend(patient_samples)
        boot_patient_list.extend([unique_boot_id] * len(patient_samples))
        boot_case_list.extend([random_labels[i]] * len(patient_samples))

    boot_counts = count_matrix[boot_indices, :]
    boot_patient_ids = np.array(boot_patient_list)
    boot_case = np.array(boot_case_list)

    return boot_counts, boot_patient_ids, boot_case


def run_power_simulation(count_matrix, patient_ids, case_status,
                         n_cancer, n_control,
                         n_simulations=1000, alpha=0.05, seed=42,
                         use_true_null=False):
    """
    Run power simulation for Shannon diversity t-test.

    Parameters
    ----------
    use_true_null : bool
        If True, uses pool-then-label null (for Type I error calibration)
    """
    significant_count = 0
    cohens_d_values = []

    for i in range(n_simulations):
        if use_true_null:
            # True null: pool patients and randomly assign labels
            boot_counts, boot_patients, boot_case = bootstrap_patients_true_null(
                count_matrix, patient_ids, case_status,
                n_cancer, n_control, seed=seed+i
            )
        else:
            # Standard: resample within groups (preserves observed effect)
            boot_counts, boot_patients, boot_case = bootstrap_patients(
                count_matrix, patient_ids, case_status,
                n_cancer, n_control, seed=seed+i
            )

        # Aggregate to patient level
        patient_shannon = aggregate_shannon_to_patient(boot_counts, boot_patients)

        # Split by group
        cancer_shannon = []
        control_shannon = []

        for patient, shannon_val in patient_shannon.items():
            # Get group label for this patient
            patient_mask = boot_patients == patient
            group = boot_case[patient_mask][0]

            if group == 'Cancer':
                cancer_shannon.append(shannon_val)
            else:
                control_shannon.append(shannon_val)

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(cancer_shannon, control_shannon)

        # Cohen's d
        pooled_std = np.sqrt(
            (np.var(cancer_shannon, ddof=1) + np.var(control_shannon, ddof=1)) / 2
        )
        if pooled_std > 0:
            d = (np.mean(cancer_shannon) - np.mean(control_shannon)) / pooled_std
        else:
            d = 0.0

        cohens_d_values.append(d)

        if p_value < alpha:
            significant_count += 1

    power = significant_count / n_simulations
    mean_d = np.mean(cohens_d_values)

    return power, mean_d, cohens_d_values


def main():
    parser = argparse.ArgumentParser(description="Stratified Shannon diversity power analysis")
    parser.add_argument("--data-wide", required=True)
    parser.add_argument("--data-long", required=True)
    parser.add_argument("--sample-sizes", default="6,8,10,15,20,25,30")
    parser.add_argument("--n-control", type=int, default=25)
    parser.add_argument("--n-simulations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    wide_df = pd.read_csv(args.data_wide, sep='\t', index_col=0)
    long_df = pd.read_csv(args.data_long, sep='\t')

    metadata = long_df[['lmp_id', 'Participant_ID', 'Case', 'type_group']].drop_duplicates().set_index('lmp_id')
    sample_ids = metadata.index.intersection(wide_df.columns)

    count_matrix = wide_df[sample_ids].T.values
    patient_ids = metadata.loc[sample_ids, 'Participant_ID'].values
    case_status = metadata.loc[sample_ids, 'Case'].values
    sample_types = metadata.loc[sample_ids, 'type_group'].values

    # Define scenarios: True_Null and Observed
    scenarios = [
        {'name': 'True_Null', 'use_true_null': True},
        {'name': 'Observed', 'use_true_null': False}
    ]

    cancer_sizes = [int(x) for x in args.sample_sizes.split(',')]

    all_results = []

    for sample_type in np.unique(sample_types):
        print(f"\n{'='*60}")
        print(f"Sample Type: {sample_type}")
        print(f"{'='*60}")

        st_counts, st_patients, st_case = filter_by_sample_type(
            count_matrix, patient_ids, case_status, sample_types, sample_type
        )

        print(f"Data: {st_counts.shape[0]} samples, {len(np.unique(st_patients))} patients")

        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")

            for n_cancer in cancer_sizes:
                print(f"  n={n_cancer}...", end=' ', flush=True)

                power, mean_d, d_vals = run_power_simulation(
                    st_counts, st_patients, st_case,
                    n_cancer, args.n_control,
                    args.n_simulations, args.alpha, args.seed,
                    use_true_null=scenario['use_true_null']
                )

                print(f"Power={power:.3f}, d={mean_d:.3f}")

                all_results.append({
                    'Sample_type': sample_type,
                    'Scenario': scenario['name'],
                    'n_cancer': n_cancer,
                    'n_control': args.n_control,
                    'Power': power,
                    'Mean_Cohens_d': mean_d,
                    'd_CI_lower': np.percentile(d_vals, 2.5),
                    'd_CI_upper': np.percentile(d_vals, 97.5)
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / 'shannon_power_stratified.tsv', sep='\t', index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {outdir}/shannon_power_stratified.tsv")
    print("\n✓ Shannon diversity power analysis complete!")


if __name__ == '__main__':
    main()
