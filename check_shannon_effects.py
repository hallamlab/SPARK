#!/usr/bin/env python3
"""Quick check of actual Shannon diversity effect sizes."""

import numpy as np
import pandas as pd
from scipy import stats

# Shannon diversity function
def shannon_diversity(count_matrix):
    rel_abund = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    shannon = np.zeros(count_matrix.shape[0])
    for i in range(count_matrix.shape[0]):
        p = rel_abund[i, :]
        p = p[p > 0]
        shannon[i] = -np.sum(p * np.log(p))
    return shannon

# Load data
wide_df = pd.read_csv('data/wide_ASV_table_filtered.tsv', sep='\t', index_col=0)
long_df = pd.read_csv('data/long_ASV_table_filtered.tsv', sep='\t')

metadata = long_df[['lmp_id', 'Participant_ID', 'Case', 'type_group']].drop_duplicates().set_index('lmp_id')
sample_ids = metadata.index.intersection(wide_df.columns)

count_matrix = wide_df[sample_ids].T.values
patient_ids = metadata.loc[sample_ids, 'Participant_ID'].values
case_status = metadata.loc[sample_ids, 'Case'].values
sample_types = metadata.loc[sample_ids, 'type_group'].values

# Calculate Shannon
shannon = shannon_diversity(count_matrix)

# Aggregate to patient level by sample type
for stype in ['BAL', 'Lung Brush', 'Oral Rinse']:
    print(f"\n{'='*60}")
    print(f"{stype}")
    print('='*60)

    # Filter by sample type
    stype_mask = sample_types == stype
    stype_shannon = shannon[stype_mask]
    stype_patients = patient_ids[stype_mask]
    stype_case = case_status[stype_mask]

    # Aggregate to patient level (mean)
    patient_shannon = {}
    patient_case = {}
    for i, patient in enumerate(stype_patients):
        if patient not in patient_shannon:
            patient_shannon[patient] = []
            patient_case[patient] = stype_case[i]
        patient_shannon[patient].append(stype_shannon[i])

    # Get mean per patient
    patient_means = {p: np.mean(vals) for p, vals in patient_shannon.items()}

    # Separate by case status
    cancer_vals = [v for p, v in patient_means.items() if patient_case[p] == 'Cancer']
    control_vals = [v for p, v in patient_means.items() if patient_case[p] == 'Control']

    print(f"Cancer: n={len(cancer_vals)}, mean={np.mean(cancer_vals):.3f}, std={np.std(cancer_vals):.3f}")
    print(f"Control: n={len(control_vals)}, mean={np.mean(control_vals):.3f}, std={np.std(control_vals):.3f}")

    # Calculate effect size
    pooled_std = np.sqrt(((len(cancer_vals)-1)*np.var(cancer_vals) + (len(control_vals)-1)*np.var(control_vals)) /
                         (len(cancer_vals) + len(control_vals) - 2))
    cohens_d = (np.mean(cancer_vals) - np.mean(control_vals)) / pooled_std

    print(f"Difference: {np.mean(cancer_vals) - np.mean(control_vals):.3f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    # T-test
    t_stat, p_val = stats.ttest_ind(cancer_vals, control_vals)
    print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")
