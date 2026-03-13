#!/usr/bin/env python3
"""
simulate_data.py

Patient-level bootstrap simulator for microbiome power analysis.
Resamples patients with replacement to preserve within-patient correlation,
sparsity, library sizes, and all real data characteristics.

CRITICAL FIX: Assigns unique bootstrap IDs to duplicate patients to avoid
collapsing permutations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def bootstrap_patients(count_matrix: np.ndarray,
                       patient_ids: np.ndarray,
                       case_status: np.ndarray,
                       n_cancer: int,
                       n_control: int,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap resample patients with replacement within groups.

    For each resampled patient, include ALL their observed samples.
    Assigns UNIQUE bootstrap IDs to handle patient duplication correctly.

    Parameters
    ----------
    count_matrix : np.ndarray
        ASV count matrix (samples × ASVs)
    patient_ids : np.ndarray
        Patient ID for each sample
    case_status : np.ndarray
        Case status for each sample ('Cancer' or 'Control')
    n_cancer : int
        Number of cancer patients to resample
    n_control : int
        Number of control patients to resample
    seed : int
        Random seed

    Returns
    -------
    boot_counts : np.ndarray
        Bootstrapped count matrix
    boot_patients : np.ndarray
        Patient IDs for bootstrapped samples (with unique bootstrap IDs)
    boot_case : np.ndarray
        Case status for bootstrapped samples
    """
    np.random.seed(seed)

    # Get unique patients in each group
    cancer_mask = (case_status == 'Cancer')
    control_mask = (case_status == 'Control') | (case_status == 'Non-Cancer')

    cancer_patients = np.unique(patient_ids[cancer_mask])
    control_patients = np.unique(patient_ids[control_mask])

    # Resample patients with replacement
    boot_cancer_patients = np.random.choice(cancer_patients, size=n_cancer, replace=True)
    boot_control_patients = np.random.choice(control_patients, size=n_control, replace=True)

    # Track how many times each patient has been drawn to create unique IDs
    patient_draw_count = {}

    # Collect all samples from resampled patients
    boot_indices = []
    boot_patient_list = []
    boot_case_list = []

    # Process cancer patients
    for patient in boot_cancer_patients:
        # Get draw count for this patient (for unique ID)
        draw_idx = patient_draw_count.get(patient, 0)
        patient_draw_count[patient] = draw_idx + 1

        # Create unique bootstrap ID
        unique_boot_id = f"{patient}__b{draw_idx}"

        # Get all samples from this patient
        patient_samples = np.where(patient_ids == patient)[0]
        boot_indices.extend(patient_samples)
        boot_patient_list.extend([unique_boot_id] * len(patient_samples))
        boot_case_list.extend(['Cancer'] * len(patient_samples))

    # Process control patients
    for patient in boot_control_patients:
        draw_idx = patient_draw_count.get(patient, 0)
        patient_draw_count[patient] = draw_idx + 1

        unique_boot_id = f"{patient}__b{draw_idx}"

        patient_samples = np.where(patient_ids == patient)[0]
        boot_indices.extend(patient_samples)
        boot_patient_list.extend([unique_boot_id] * len(patient_samples))
        boot_case_list.extend(['Control'] * len(patient_samples))

    boot_counts = count_matrix[boot_indices, :]
    boot_patients = np.array(boot_patient_list)
    boot_case = np.array(boot_case_list)

    return boot_counts, boot_patients, boot_case


def spike_in_deterministic(count_matrix: np.ndarray,
                           case_status: np.ndarray,
                           asv_indices: List[int],
                           fold_change: float,
                           treatment_group: str = 'Cancer') -> np.ndarray:
    """
    Apply deterministic fold-change spike-in WITHOUT resampling.

    Preserves zeros, overdispersion, and all sample quirks by NOT drawing new counts.
    Uses integer reallocation to maintain library sizes.

    Parameters
    ----------
    count_matrix : np.ndarray
        ASV count matrix (samples × ASVs)
    case_status : np.ndarray
        Case status for each sample
    asv_indices : List[int]
        Column indices of ASVs to spike
    fold_change : float
        Multiplicative fold-change (e.g., 2.0 = double counts)
    treatment_group : str
        Which group to spike ('Cancer' or 'Control')

    Returns
    -------
    spiked_counts : np.ndarray
        Count matrix with spike-in applied (deterministic, integer counts)
    """
    spiked_counts = count_matrix.copy()
    treatment_mask = (case_status == treatment_group)

    for i in np.where(treatment_mask)[0]:
        counts = spiked_counts[i, :].astype(float)
        library_size = counts.sum()

        if library_size == 0:
            continue

        # Increase spiked ASVs by fold-change in count space
        increased_counts = counts.copy()
        increased_counts[asv_indices] *= fold_change

        # Rescale non-spiked ASVs to maintain library size
        # Total increase in spiked ASVs
        spike_increase = increased_counts[asv_indices].sum() - counts[asv_indices].sum()

        # Non-spiked ASV mask
        non_spike_mask = np.ones(len(counts), dtype=bool)
        non_spike_mask[asv_indices] = False

        # Proportionally reduce non-spiked ASVs
        non_spike_total = counts[non_spike_mask].sum()
        if non_spike_total > spike_increase:
            scale_factor = (non_spike_total - spike_increase) / non_spike_total
            increased_counts[non_spike_mask] *= scale_factor

        # Integer reconciliation using largest remainder method
        increased_counts_int = np.floor(increased_counts).astype(int)
        remainders = increased_counts - increased_counts_int

        # Add back remaining counts to maintain library size
        deficit = int(library_size) - increased_counts_int.sum()
        if deficit > 0:
            # Sort by remainder, largest first
            sorted_idx = np.argsort(-remainders)
            for j in range(min(deficit, len(sorted_idx))):
                increased_counts_int[sorted_idx[j]] += 1

        spiked_counts[i, :] = increased_counts_int

    return spiked_counts


def spike_in_fold_change(count_matrix: np.ndarray,
                         case_status: np.ndarray,
                         asv_indices: List[int],
                         fold_change: float,
                         treatment_group: str = 'Cancer') -> np.ndarray:
    """
    Wrapper that calls deterministic spike-in (backward compatibility).
    """
    return spike_in_deterministic(count_matrix, case_status, asv_indices,
                                  fold_change, treatment_group)


def filter_by_sample_type(count_matrix: np.ndarray,
                          patient_ids: np.ndarray,
                          case_status: np.ndarray,
                          sample_types: np.ndarray,
                          target_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data to a specific sample type.

    Parameters
    ----------
    count_matrix : np.ndarray
        Full count matrix
    patient_ids : np.ndarray
        Patient IDs
    case_status : np.ndarray
        Case status
    sample_types : np.ndarray
        Sample type labels
    target_type : str
        Sample type to keep

    Returns
    -------
    filtered_counts, filtered_patients, filtered_case : Tuple[np.ndarray]
        Filtered data for the target sample type
    """
    mask = sample_types == target_type
    return count_matrix[mask, :], patient_ids[mask], case_status[mask]


def aggregate_to_patient_level(count_matrix: np.ndarray,
                               patient_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate counts to patient level (sum across samples per patient).

    Used for making patient the experimental unit.

    Parameters
    ----------
    count_matrix : np.ndarray
        Sample × ASV count matrix
    patient_ids : np.ndarray
        Patient ID for each sample

    Returns
    -------
    patient_counts : np.ndarray
        Patient × ASV count matrix
    unique_patients : np.ndarray
        Ordered list of patient IDs
    """
    unique_patients = np.unique(patient_ids)
    n_asvs = count_matrix.shape[1]
    patient_counts = np.zeros((len(unique_patients), n_asvs))

    for i, patient in enumerate(unique_patients):
        patient_mask = patient_ids == patient
        patient_counts[i, :] = count_matrix[patient_mask, :].sum(axis=0)

    return patient_counts, unique_patients
