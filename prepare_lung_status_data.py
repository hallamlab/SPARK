#!/usr/bin/env python3
"""
Prepare data for cancerous vs. contralateral vs. healthy lung analysis.

Creates lung_status grouping:
- TumorSide: sample side matches Cancer_Site (cancer patients only)
- Contralateral: opposite side in cancer patients
- Healthy: control patients

For control patients, averages left/right samples to create one profile per patient.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def assign_lung_status(row: pd.Series) -> str:
    """
    Assign lung status based on Case, Cancer_Site, and lung_code.

    Returns:
        - "TumorSide": cancer patient, sample matches tumor side
        - "Contralateral": cancer patient, sample opposite tumor side
        - "Healthy": control patient
    """
    case = row.get("Case", "")
    cancer_site = row.get("Cancer_Site", "")
    lung_code = row.get("lung_code", "")

    if case == "Cancer":
        # Map Cancer_Site to lung_code
        # Cancer_Site can be "Left" or "Right"
        # lung_code is "L" or "R" for lung samples
        if cancer_site == "Left" and lung_code == "L":
            return "TumorSide"
        elif cancer_site == "Right" and lung_code == "R":
            return "TumorSide"
        elif cancer_site in ["Left", "Right"] and lung_code in ["L", "R"]:
            return "Contralateral"
        else:
            # Non-lung samples (Oral Rinse) or missing lung_code
            return "Other"
    elif case in ["Control", "Non-Cancer"]:
        return "Healthy"
    else:
        return "Unknown"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare lung status groupings for cancer vs. contralateral vs. healthy analysis."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input ASV table (long format TSV with metadata)",
    )
    p.add_argument(
        "--sample-type",
        type=str,
        default="Bronchial Brush",
        help="Sample type to filter for (default: 'Bronchial Brush')",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for prepared data",
    )
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input, sep="\t", low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df['sample'].nunique()} samples")

    # Filter for specified sample type - use type_group which has clean values
    if "type_group" in df.columns:
        type_col = "type_group"
    elif "Type" in df.columns:
        type_col = "Type"
        print(f"  WARNING: Using 'Type' column; values may include laterality (e.g., 'Left Brush')")
    else:
        print("ERROR: No sample type column found (expected 'type_group' or 'Type')")
        sys.exit(1)

    print(f"  Available sample types in '{type_col}': {sorted(df[type_col].dropna().unique())}")

    df_filt = df[df[type_col] == args.sample_type].copy()
    print(f"  Filtered to {args.sample_type}: {len(df_filt):,} rows, {df_filt['sample'].nunique()} samples")

    if df_filt.empty:
        print(f"ERROR: No samples found for sample type '{args.sample_type}'")
        print(f"  Available types: {sorted(df[type_col].dropna().unique())}")
        sys.exit(1)

    # Assign lung status
    print("\nAssigning lung_status...")
    df_filt["lung_status"] = df_filt.apply(assign_lung_status, axis=1)

    # Remove "Other" and "Unknown" categories
    df_filt = df_filt[df_filt["lung_status"].isin(["TumorSide", "Contralateral", "Healthy"])].copy()
    print(f"  After assignment: {df_filt['sample'].nunique()} samples")

    if df_filt.empty:
        print("ERROR: No samples remaining after lung_status assignment")
        print("  Check that your data has proper Case, Cancer_Site, and lung_code columns")
        sys.exit(1)

    # Summary by lung_status
    summary = df_filt.groupby("lung_status")["sample"].nunique().to_dict()
    print("\nSample counts by lung_status:")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")

    # Summary by patient and lung_status
    patient_col = "Participant_ID" if "Participant_ID" in df_filt.columns else "patient_code"
    print(f"\nPatient counts by lung_status:")
    patient_summary = df_filt.groupby("lung_status")[patient_col].nunique().to_dict()
    for status, count in sorted(patient_summary.items()):
        print(f"  {status}: {count} patients")

    # Check paired structure for cancer patients
    cancer_df = df_filt[df_filt["Case"] == "Cancer"]
    if not cancer_df.empty:
        paired_counts = (
            cancer_df.groupby(patient_col)["lung_status"]
            .apply(lambda x: set(x))
            .value_counts()
        )
        print("\nPairing structure for cancer patients:")
        for pair_set, count in paired_counts.items():
            print(f"  {pair_set}: {count} patients")

    # Save full filtered data with lung_status
    out_full = args.outdir / f"{args.sample_type.replace(' ', '_')}_with_lung_status.tsv"
    df_filt.to_csv(out_full, sep="\t", index=False)
    print(f"\nSaved full data to: {out_full}")

    # Create wide-format ASV table for easier downstream analysis
    if "ASV_ID" in df_filt.columns and "count" in df_filt.columns:
        asv_wide = df_filt.pivot_table(
            index="sample",
            columns="ASV_ID",
            values="count",
            fill_value=0,
        )
        out_asv = args.outdir / f"{args.sample_type.replace(' ', '_')}_ASV_table.tsv"
        asv_wide.to_csv(out_asv, sep="\t")
        print(f"Saved ASV table to: {out_asv}")

        # Save metadata
        meta_cols = [
            "sample",
            patient_col,
            "Case",
            "Cancer_Site",
            type_col,
            "lung_code",
            "lung_status",
        ]
        # Add any other relevant metadata columns
        available_cols = [c for c in meta_cols if c in df_filt.columns]

        metadata = df_filt[available_cols].drop_duplicates(subset=["sample"]).copy()
        out_meta = args.outdir / f"{args.sample_type.replace(' ', '_')}_metadata.tsv"
        metadata.to_csv(out_meta, sep="\t", index=False)
        print(f"Saved metadata to: {out_meta}")

    print("\nData preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Run paired analysis (TumorSide vs Contralateral)")
    print(f"2. Average control samples and run between-patient analyses")
    print(f"3. Visualize results")


if __name__ == "__main__":
    main()
