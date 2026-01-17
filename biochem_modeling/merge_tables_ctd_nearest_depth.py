#!/usr/bin/env python3
"""
scripts/merge_tableA_tableB_ctd_nearest_depth_and_corrections.py

Why this script exists
----------------------
You (or someone years later) should be able to re-run this script and understand, from the script alone,
exactly how Table A and Table B were merged, how CTD oxygen was corrected to match Winkler titrations,
and how a year-based batch correction was applied across numeric features.

Inputs
------
Table A (master; discrete depths; more cruises):
  Latitude Longitude Cruise Year Month Day Depth PO4 SI NOx_SeaTech NOx_New NO2 NO3 NH4 H2S Cells_per_ml N2 O2 CO2 N2O CH4

Table B (CTD; many depths; sometimes float depths):
  Latitude Longitude Cruise Year Month Day Depth Temperature Conductivity Fluorescence Beam_Transmission PAR
  Oxygen, SBE Oxygen Salinity Density

Joining keys (conceptual identity)
---------------------------------
  Latitude Longitude Cruise Year Month Day Depth

Key intricacies handled here
----------------------------
(1) Table A is master; Table B is “aligned” to A, not the other way around.
    - Rows present in B but absent in A are ignored (dropped implicitly).

(2) CTD depth mismatch handling:
    - When CTD Depth is float and A Depth is integer (or otherwise not exactly matching),
      we select from B the depth(s) closest to that A.Depth within the same station+date+cruise group
      (Latitude, Longitude, Cruise, Year, Month, Day).
    - If multiple CTD rows are equally close (tie), we average CTD numeric values.

Additions requested
-------------------
(A) Oxygen correction curve:
    - Table A column "O2" is Winkler titration (treated as ground truth).
    - Table B column "Oxygen" is CTD oxygen (to be corrected).
    - We fit a correction curve using rows where both are available after alignment.
    - Default model: simple linear regression mapping CTD Oxygen -> predicted Winkler O2:
          O2_hat = intercept + slope * Oxygen
      Rationale: transparent, easy to audit, robust enough for many CTD-vs-Winkler use cases.
    - The correction is then applied to *all* available CTD "Oxygen" values.

(B) Year-based batch correction (annual):
    - Because documentation for true batch sources is missing, we treat "Year" as the batch label.
    - We apply a simple additive correction for each numeric feature:
          x_corrected = x - mean(x in that Year) + mean(x overall)
      Rationale:
        - Removes year-specific mean shifts (“batch effects”) while preserving global scale.
        - Fully deterministic and easy to audit.
    - This is applied after oxygen correction, so we do not distort the calibration step.

Audit trail outputs (all written to --outdir)
---------------------------------------------
01_merged_nearest_depth.tsv
    - Table A plus CTD columns aligned via nearest-depth logic.
    - Includes CTD_Depth_Used to show which CTD depth was selected per A row.

02_oxygen_calibration_pairs.tsv
    - Only rows used to fit the oxygen correction (non-null O2 and CTD Oxygen).

02_oxygen_model_summary.txt
    - Human-readable model details: slope, intercept, R^2, N, etc.

03_oxygen_corrected.tsv
    - Same as (01) plus:
        Oxygen_CTD_raw           (copy of CTD Oxygen)
        Oxygen_CTD_to_Winkler    (corrected CTD oxygen; predicted Winkler-equivalent)

04_batch_offsets_by_year.tsv
    - For each Year and each numeric feature: year_mean, global_mean, offset_applied.

05_batch_corrected.tsv
    - Final table after annual batch correction across numeric features.

Plots (PNG; minimal styling; no hard-coded colors)
-------------------------------------------------
plot_oxygen_calibration_scatter.png
plot_oxygen_calibration_residuals.png
plot_samples_per_year.png
plot_batch_offset_heatmap.png   (overview; default colormap)

Usage
-----
python scripts/merge_tableA_tableB_ctd_nearest_depth_and_corrections.py \
  --table-a tableA.tsv \
  --table-b tableB.tsv \
  --outdir merge_out \
  --sep-a $'\\t' --sep-b $'\\t'

Notes on assumptions (explicit)
-------------------------------
- Latitude/Longitude are treated as exact numeric keys after parsing. If these values can differ
  by tiny rounding (e.g., 49.1234 vs 49.1233999), you should add rounding/binning BEFORE merge.
- "Cruise" is treated as a string key (whitespace stripped).
- Depth units are assumed consistent between A and B.
- The oxygen correction is linear; if later you want non-linear (e.g., spline), we can extend it,
  but linear is intentionally the most auditable first pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib: required only for audit plots.
# We deliberately avoid seaborn and do not set custom styles/colors.
import matplotlib.pyplot as plt


# -----------------------------
# Configuration: key columns
# -----------------------------
KEY_COLS_NO_DEPTH = ["Latitude", "Longitude", "Cruise", "Year", "Month", "Day"]
KEY_COLS_WITH_DEPTH = KEY_COLS_NO_DEPTH + ["Depth"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge Table A (master) with Table B (CTD) using nearest-depth matching, then apply oxygen calibration and annual batch correction."
    )
    ap.add_argument("--table-a", required=True, help="Path to Table A (CSV/TSV).")
    ap.add_argument("--table-b", required=True, help="Path to Table B (CSV/TSV).")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created if missing).")

    ap.add_argument(
        "--sep-a",
        default=None,
        help="Delimiter for Table A. Default: auto-detect (tab if .tsv/.tab, else comma).",
    )
    ap.add_argument(
        "--sep-b",
        default=None,
        help="Delimiter for Table B. Default: auto-detect (tab if .tsv/.tab, else comma).",
    )
    ap.add_argument(
        "--sep-out",
        default="\t",
        help="Delimiter for output tables (default: tab).",
    )

    # Nearest-depth matching safety valve:
    ap.add_argument(
        "--max-depth-diff",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed absolute depth difference. "
            "If set, matches with |B.Depth - A.Depth| > this are treated as missing CTD for that row."
        ),
    )

    # Oxygen calibration behavior:
    ap.add_argument(
        "--min-calibration-pairs",
        type=int,
        default=10,
        help=(
            "Minimum number of (O2 Winkler, CTD Oxygen) paired points required to fit a correction model "
            "(default: 10). If fewer, oxygen correction is skipped and documented."
        ),
    )

    ap.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")
    return ap.parse_args()


def guess_sep(path: str, user_sep: Optional[str]) -> str:
    if user_sep is not None:
        return user_sep
    lower = path.lower()
    if lower.endswith(".tsv") or lower.endswith(".tab"):
        return "\t"
    return ","


def read_table(path: str, sep: str) -> pd.DataFrame:
    # Load everything as string first so we can explicitly coerce the key columns.
    return pd.read_csv(path, sep=sep, header=0, dtype=str, na_values=["", "NA", "NaN"])


def coerce_key_types(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Enforce key presence and consistent types across A and B.

    Decisions:
    - Cruise: string key, whitespace stripped.
    - Latitude/Longitude/Depth: numeric floats (Depth may be float in CTD).
    - Year/Month/Day: nullable integers (Int64).
    - Rows missing any join key are dropped because their identity is ambiguous for joining.

    This is intentionally strict: bad keys lead to bad merges.
    """
    missing = [c for c in KEY_COLS_WITH_DEPTH if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required key columns: {missing}")

    out = df.copy()

    out["Cruise"] = out["Cruise"].astype(str).str.strip()

    for c in ["Latitude", "Longitude", "Depth"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["Year", "Month", "Day"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    before = len(out)
    out = out.dropna(subset=KEY_COLS_WITH_DEPTH)
    dropped = before - len(out)

    return out


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_tsv(df: pd.DataFrame, path: str, sep: str = "\t") -> None:
    df.to_csv(path, sep=sep, index=False)


def aggregate_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Collapse exact duplicate keys.

    Used for CTD (Table B) because exports can contain duplicated key rows.

    Aggregation:
      - Numeric columns: mean
      - Non-numeric columns: first non-null

    This step is audited implicitly: it is deterministic and applied before nearest-depth matching.
    """
    non_group = [c for c in df.columns if c not in group_cols]

    # Attempt numeric conversion for non-group columns.
    numeric_set = set()
    tmp = df.copy()
    for c in non_group:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            numeric_set.add(c)
        else:
            coerced = pd.to_numeric(tmp[c], errors="coerce")
            if coerced.notna().any():
                numeric_set.add(c)
                tmp[c] = coerced

    agg: Dict[str, object] = {}
    for c in non_group:
        if c in numeric_set:
            agg[c] = "mean"
        else:
            agg[c] = lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan

    return tmp.groupby(group_cols, dropna=False, as_index=False).agg(agg)


def match_ctd_nearest_depth(
    a: pd.DataFrame,
    b: pd.DataFrame,
    max_depth_diff: Optional[float],
    verbose: bool,
) -> pd.DataFrame:
    """
    Align CTD values (B) onto Table A rows via nearest-depth selection within each station/date/cruise group.

    Output is one CTD-aligned row per A row (same index as A), plus CTD_Depth_Used.

    Why index-aligned output matters:
      - It prevents accidental row multiplication.
      - It makes the merge step unambiguous (concat by row order after reset_index).

    Tie-handling:
      - If multiple CTD depths are equally close to the target depth, average CTD numeric values.
    """
    ctd_cols = [c for c in b.columns if c not in KEY_COLS_WITH_DEPTH]

    matched = pd.DataFrame(index=a.index, columns=ctd_cols, dtype=float)
    matched_depth_used = pd.Series(index=a.index, dtype=float)

    b_groups = {k: sub for k, sub in b.groupby(KEY_COLS_NO_DEPTH, dropna=False)}

    n_no_b = 0
    n_capped = 0
    n_ties = 0

    for gkey, a_sub in a.groupby(KEY_COLS_NO_DEPTH, dropna=False):
        b_sub = b_groups.get(gkey, None)
        if b_sub is None or b_sub.empty:
            n_no_b += len(a_sub)
            continue

        b_sub = b_sub.sort_values("Depth").reset_index(drop=True)
        b_depths = b_sub["Depth"].to_numpy(dtype=float)

        for idx, arow in a_sub.iterrows():
            target = float(arow["Depth"])
            diffs = np.abs(b_depths - target)
            if diffs.size == 0 or np.all(np.isnan(diffs)):
                n_no_b += 1
                continue

            min_diff = np.nanmin(diffs)
            if max_depth_diff is not None and min_diff > max_depth_diff:
                n_capped += 1
                continue

            tie_mask = np.isclose(diffs, min_diff, rtol=0.0, atol=0.0)
            tie_rows = b_sub.loc[tie_mask]
            if len(tie_rows) > 1:
                n_ties += 1

            vals = tie_rows[ctd_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0, skipna=True)
            matched.loc[idx, ctd_cols] = vals.values
            matched_depth_used.loc[idx] = tie_rows["Depth"].mean()

    if verbose:
        print(f"[merge] A rows: {len(a)}", file=sys.stderr)
        print(f"[merge] A rows with no CTD group: {n_no_b}", file=sys.stderr)
        if max_depth_diff is not None:
            print(f"[merge] A rows capped by max_depth_diff: {n_capped}", file=sys.stderr)
        print(f"[merge] nearest-depth ties (averaged): {n_ties}", file=sys.stderr)

    matched = matched.copy()
    matched.insert(0, "CTD_Depth_Used", matched_depth_used.values)
    return matched


@dataclass
class OxygenModel:
    slope: float
    intercept: float
    r2: float
    n: int


def fit_oxygen_correction(df: pd.DataFrame, verbose: bool, min_pairs: int) -> Tuple[Optional[OxygenModel], pd.DataFrame]:
    """
    Fit a linear correction mapping CTD Oxygen -> Winkler O2.

    Model form:
        O2 = intercept + slope * Oxygen

    Data used:
        Rows where both:
          - df["O2"] is present (Winkler)
          - df["Oxygen"] is present (CTD)
        Note: 'Oxygen' here refers specifically to Table B's "Oxygen" column (per your request).

    Returns:
      - OxygenModel or None if insufficient data
      - calibration_pairs dataframe used in fit (for audit trail)
    """
    if "O2" not in df.columns:
        raise ValueError("Expected Table A column 'O2' (Winkler) to exist after merge.")
    if "Oxygen" not in df.columns:
        raise ValueError("Expected Table B column 'Oxygen' (CTD) to exist after merge.")

    pairs = df[KEY_COLS_WITH_DEPTH + ["O2", "Oxygen"]].copy()
    pairs["O2"] = pd.to_numeric(pairs["O2"], errors="coerce")
    pairs["Oxygen"] = pd.to_numeric(pairs["Oxygen"], errors="coerce")
    pairs = pairs.dropna(subset=["O2", "Oxygen"])

    n = len(pairs)
    if verbose:
        print(f"[o2-cal] paired points available: {n}", file=sys.stderr)

    if n < min_pairs:
        # Decision: do not fit a model with too few points.
        # Rationale: year-to-year or cruise-to-cruise variability can otherwise dominate the fit.
        return None, pairs

    x = pairs["Oxygen"].to_numpy(dtype=float)
    y = pairs["O2"].to_numpy(dtype=float)

    # Fit y = a + b*x using least squares.
    slope, intercept = np.polyfit(x, y, deg=1)

    # Compute R^2 for auditability.
    yhat = intercept + slope * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else np.nan)

    return OxygenModel(slope=slope, intercept=intercept, r2=r2, n=n), pairs


def apply_oxygen_correction(df: pd.DataFrame, model: Optional[OxygenModel]) -> pd.DataFrame:
    """
    Create a single oxygen column on the Winkler (Table A 'O2') scale, using the best available source.

    Key logic (explicit audit intent)
    ---------------------------------
    - Table A 'O2' is Winkler titration and is treated as the preferred ground-truth measurement.
    - Table B 'Oxygen' is CTD oxygen and may be missing for some Table A rows after alignment.

    We construct:

      Oxygen_raw_CTD
        - numeric copy of CTD 'Oxygen' (may be NaN)

      Oxygen_WinklerScale
        - If Winkler O2 exists for that row, use it directly (highest-trust source).
        - Else, if CTD oxygen exists and a calibration model exists, use calibrated CTD value.
        - Else, if CTD oxygen exists and no model exists, leave as NaN (we refuse to guess).
        - Else, NaN.

    Rationale:
    - This preserves the "best available" oxygen on a single comparable scale, while never
      overwriting original measurements.
    - It also fixes the specific case you described:
        If A has Winkler O2 but CTD Oxygen is missing, we still populate the final oxygen column.
    """
    out = df.copy()

    # Preserve originals in explicit audit columns.
    out["Oxygen_raw_CTD"] = pd.to_numeric(out["Oxygen"], errors="coerce")
    out["O2_Winkler"] = pd.to_numeric(out["O2"], errors="coerce")

    # Start with Winkler where available (preferred).
    out["Oxygen_WinklerScale"] = out["O2_Winkler"].copy()

    # If Winkler missing, optionally fill from calibrated CTD.
    winkler_missing = out["Oxygen_WinklerScale"].isna()
    if model is not None:
        ctd_calibrated = model.intercept + model.slope * out["Oxygen_raw_CTD"]
        out.loc[winkler_missing, "Oxygen_WinklerScale"] = ctd_calibrated.loc[winkler_missing]

    # If model is None, we do NOT backfill from raw CTD because it would be on an uncalibrated scale.
    return out


def plot_oxygen_calibration(pairs: pd.DataFrame, model: Optional[OxygenModel], outdir: str) -> None:
    """
    Create two audit plots:
      - scatter of CTD Oxygen vs Winkler O2, with fitted line if available
      - residuals vs CTD Oxygen if available
    """
    if len(pairs) == 0:
        return

    x = pairs["Oxygen"].to_numpy(dtype=float)
    y = pairs["O2"].to_numpy(dtype=float)

    # Scatter + fit line
    plt.figure(figsize=(7.5, 6.0))
    plt.scatter(x, y, s=18, alpha=0.6)
    plt.xlabel("CTD Oxygen (raw; column 'Oxygen')")
    plt.ylabel("Winkler O2 (column 'O2')")
    plt.title("Oxygen calibration: CTD Oxygen -> Winkler O2")

    if model is not None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        xs = np.linspace(xmin, xmax, 200)
        ys = model.intercept + model.slope * xs
        plt.plot(xs, ys)  # default color
        plt.text(
            0.02,
            0.98,
            f"Model: O2 = {model.intercept:.4g} + {model.slope:.4g} * Oxygen\nR²={model.r2:.4g}, N={model.n}",
            transform=plt.gca().transAxes,
            va="top",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_oxygen_calibration_scatter.png"), dpi=200)
    plt.close()

    # Residuals
    if model is not None:
        yhat = model.intercept + model.slope * x
        resid = y - yhat

        plt.figure(figsize=(7.5, 6.0))
        plt.scatter(x, resid, s=18, alpha=0.6)
        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("CTD Oxygen (raw; column 'Oxygen')")
        plt.ylabel("Residual (Winkler O2 - predicted)")
        plt.title("Oxygen calibration residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plot_oxygen_calibration_residuals.png"), dpi=200)
        plt.close()


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def infer_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Decide which columns are eligible for year-based batch correction.

    Explicit exclusions:
      - Join keys: Latitude, Longitude, Cruise, Year, Month, Day, Depth
      - Obviously non-numeric columns (kept untouched)

    Inclusions:
      - Columns that can be coerced to numeric and have at least one non-null numeric value.

    Important note:
      - We keep BOTH raw and derived oxygen columns; batch correction will also act on them
        (unless you later decide to exempt some features).
    """
    exclude = set(KEY_COLS_WITH_DEPTH)
    candidates = [c for c in df.columns if c not in exclude]

    numeric_cols: List[str] = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any():
                numeric_cols.append(c)
    return numeric_cols


def annual_batch_correct(
    df: pd.DataFrame,
    feature_cols: List[str],
    year_col: str = "Year",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply mean-centering by Year for each numeric feature:
        x_corrected = x - mean_year + mean_global

    Outputs:
      - corrected_df: same shape as df; corrected numeric features only (others unchanged)
      - offsets_df: audit table with (Year, Feature, YearMean, GlobalMean, OffsetApplied)

    OffsetApplied is defined as (GlobalMean - YearMean) so:
      corrected = x + OffsetApplied
    """
    out = df.copy()

    # Coerce features to numeric for computation (store numeric arrays, but keep original columns overwritten).
    numeric = {}
    for c in feature_cols:
        numeric[c] = pd.to_numeric(out[c], errors="coerce")

    years = out[year_col].astype("Int64")
    unique_years = sorted([int(y) for y in years.dropna().unique().tolist()])

    offsets_records = []

    # Precompute global means for each feature (over all years).
    global_means = {c: numeric[c].mean(skipna=True) for c in feature_cols}

    # Apply year offsets
    for y in unique_years:
        mask = (years == y).to_numpy()
        for c in feature_cols:
            yr_mean = numeric[c][mask].mean(skipna=True)
            gl_mean = global_means[c]

            # If a feature has no data in this year, we skip (no correction possible).
            if pd.isna(yr_mean) or pd.isna(gl_mean):
                offsets_records.append(
                    {"Year": y, "Feature": c, "YearMean": yr_mean, "GlobalMean": gl_mean, "OffsetApplied": np.nan}
                )
                continue

            offset = float(gl_mean - yr_mean)
            offsets_records.append(
                {"Year": y, "Feature": c, "YearMean": float(yr_mean), "GlobalMean": float(gl_mean), "OffsetApplied": offset}
            )

            # Apply correction: x + offset (only where x is not NaN)
            x = numeric[c].to_numpy()
            x_corr = x.copy()
            idxs = np.where(mask)[0]
            for i in idxs:
                if not np.isnan(x_corr[i]):
                    x_corr[i] = x_corr[i] + offset
            out[c] = x_corr

    offsets_df = pd.DataFrame(offsets_records)
    return out, offsets_df


def plot_samples_per_year(df: pd.DataFrame, outdir: str) -> None:
    years = df["Year"].astype("Int64")
    counts = years.value_counts(dropna=True).sort_index()
    if len(counts) == 0:
        return

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(counts.index.astype(int), counts.values)
    plt.xlabel("Year")
    plt.ylabel("Number of samples (rows)")
    plt.title("Samples per year (post-merge)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_samples_per_year.png"), dpi=200)
    plt.close()


def plot_batch_offset_heatmap(offsets_df: pd.DataFrame, outdir: str) -> None:
    """
    A compact overview: heatmap of OffsetApplied (Year x Feature).

    We do not pick a custom colormap; matplotlib default is used for neutrality.
    """
    if offsets_df.empty:
        return

    piv = offsets_df.pivot(index="Year", columns="Feature", values="OffsetApplied")
    if piv.shape[0] == 0 or piv.shape[1] == 0:
        return

    data = piv.to_numpy(dtype=float)

    plt.figure(figsize=(min(16, 0.35 * piv.shape[1] + 4), min(10, 0.35 * piv.shape[0] + 4)))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(label="OffsetApplied (GlobalMean - YearMean)")
    plt.yticks(range(piv.shape[0]), [str(y) for y in piv.index.tolist()])
    # If lots of features, x labels will be dense; keep readable by rotating.
    plt.xticks(range(piv.shape[1]), piv.columns.tolist(), rotation=90)
    plt.title("Annual batch offsets by feature")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_batch_offset_heatmap.png"), dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    ensure_outdir(args.outdir)

    sep_a = guess_sep(args.table_a, args.sep_a)
    sep_b = guess_sep(args.table_b, args.sep_b)

    # -----------------------------
    # Step 0: Load tables
    # -----------------------------
    a_raw = read_table(args.table_a, sep_a)
    b_raw = read_table(args.table_b, sep_b)

    a = coerce_key_types(a_raw, "Table A")
    b = coerce_key_types(b_raw, "Table B")

    if args.verbose:
        print(f"[load] Table A rows (after dropping bad keys): {len(a)}", file=sys.stderr)
        print(f"[load] Table B rows (after dropping bad keys): {len(b)}", file=sys.stderr)

    # -----------------------------
    # Step 1: Pre-aggregate CTD exact duplicates on full keys
    # -----------------------------
    b = aggregate_rows(b, KEY_COLS_WITH_DEPTH)

    # -----------------------------
    # Step 2: Align CTD to Table A via nearest-depth matching
    # -----------------------------
    ctd_aligned = match_ctd_nearest_depth(a=a, b=b, max_depth_diff=args.max_depth_diff, verbose=args.verbose)

    merged = pd.concat([a.reset_index(drop=True), ctd_aligned.reset_index(drop=True)], axis=1)

    out_01 = os.path.join(args.outdir, "01_merged_nearest_depth.tsv")
    write_tsv(merged, out_01, sep=args.sep_out)

    # Basic audit plots on merged data
    plot_samples_per_year(merged, args.outdir)

    # -----------------------------
    # Step 3: Oxygen correction curve (CTD Oxygen -> Winkler O2)
    # -----------------------------
    model, pairs = fit_oxygen_correction(merged, verbose=args.verbose, min_pairs=args.min_calibration_pairs)

    out_02_pairs = os.path.join(args.outdir, "02_oxygen_calibration_pairs.tsv")
    write_tsv(pairs, out_02_pairs, sep=args.sep_out)

    out_02_summary = os.path.join(args.outdir, "02_oxygen_model_summary.txt")
    with open(out_02_summary, "w", encoding="utf-8") as f:
        f.write("Oxygen correction summary\n")
        f.write("=========================\n\n")
        f.write("Intent:\n")
        f.write("  Fit a correction mapping CTD Oxygen (Table B column 'Oxygen') to Winkler O2 (Table A column 'O2').\n\n")
        f.write("Model form:\n")
        f.write("  O2_hat = intercept + slope * Oxygen\n\n")
        f.write(f"Paired points available (after merge): {len(pairs)}\n")
        f.write(f"Minimum required to fit: {args.min_calibration_pairs}\n\n")

        if model is None:
            f.write("Result: NO MODEL FIT\n")
            f.write("Reason: insufficient calibration pairs.\n\n")
            f.write("Action taken:\n")
            f.write("  - Oxygen_raw_CTD will be created (numeric copy of CTD 'Oxygen').\n")
            f.write("  - O2_Winkler will be created (numeric copy of Table A 'O2').\n")
            f.write("  - Oxygen_WinklerScale will be created as:\n")
            f.write("      * Winkler O2 where available\n")
            f.write("      * otherwise NaN (no calibration model available)\n")

        else:
            f.write("Result: MODEL FIT\n\n")
            f.write(f"slope:     {model.slope:.10g}\n")
            f.write(f"intercept: {model.intercept:.10g}\n")
            f.write(f"R^2:       {model.r2:.10g}\n")
            f.write(f"N:         {model.n}\n\n")
            f.write("Interpretation:\n")
            f.write("  - Oxygen_WinklerScale is constructed as:\n")
            f.write("      * Winkler O2 where available (preferred)\n")
            f.write("      * otherwise, calibrated CTD oxygen: intercept + slope * Oxygen_raw_CTD\n")
            f.write("  - This yields a single oxygen value per row on the Winkler scale, with transparent provenance.\n")

    plot_oxygen_calibration(pairs, model, args.outdir)

    merged_o2 = apply_oxygen_correction(merged, model)

    out_03 = os.path.join(args.outdir, "03_oxygen_corrected.tsv")
    write_tsv(merged_o2, out_03, sep=args.sep_out)

    # -----------------------------
    # Step 4: Annual batch correction (batch = Year)
    # -----------------------------
    # Decide numeric features eligible for correction
    numeric_features = infer_numeric_feature_columns(merged_o2)

    # Write a small “what will be corrected” manifest for audit
    manifest_path = os.path.join(args.outdir, "04_batch_correction_features.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "excluded_columns": KEY_COLS_WITH_DEPTH,
                "numeric_features_corrected": numeric_features,
                "method": "x_corrected = x - mean_year + mean_global",
                "batch_column": "Year",
                "note": "All numeric-like columns except join keys are corrected. Non-numeric columns are unchanged.",
            },
            f,
            indent=2,
        )

    batch_corrected, offsets_df = annual_batch_correct(merged_o2, feature_cols=numeric_features, year_col="Year")

    out_04_offsets = os.path.join(args.outdir, "04_batch_offsets_by_year.tsv")
    write_tsv(offsets_df, out_04_offsets, sep=args.sep_out)

    plot_batch_offset_heatmap(offsets_df, args.outdir)

    out_05 = os.path.join(args.outdir, "05_batch_corrected.tsv")
    write_tsv(batch_corrected, out_05, sep=args.sep_out)

    if args.verbose:
        print(f"[out] {out_01}", file=sys.stderr)
        print(f"[out] {out_02_pairs}", file=sys.stderr)
        print(f"[out] {out_02_summary}", file=sys.stderr)
        print(f"[out] {out_03}", file=sys.stderr)
        print(f"[out] {out_04_offsets}", file=sys.stderr)
        print(f"[out] {out_05}", file=sys.stderr)
        print(f"[out] plots written to {args.outdir}/plot_*.png", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
