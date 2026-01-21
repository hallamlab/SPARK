# Path: ./scripts/merge_tableA_tableB_ctd_nearest_depth_simple.py
# Purpose: Merge Table A (master) with Table B (CTD) using nearest-depth matching,
#          then create a single oxygen column using:
#            - Table B "Oxygen" when present
#            - else Table A "O2" when Oxygen is missing
#
# Outputs (written to --outdir):
#   01_merged_nearest_depth.tsv   (A + CTD columns aligned; includes CTD_Depth_Used)
#   02_oxygen_best_available.tsv  (same as 01 + Oxygen_best_available)
#
# Notes:
# - No oxygen calibration/correction.
# - No batch correction.
# - Nearest-depth matching behavior matches your original script:
#     within each (Latitude, Longitude, Cruise, Year, Month, Day) group,
#     pick CTD row(s) with closest Depth to A.Depth; ties are averaged.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

KEY_COLS_NO_DEPTH = ["Latitude", "Longitude", "Cruise", "Year", "Month", "Day"]
KEY_COLS_WITH_DEPTH = KEY_COLS_NO_DEPTH + ["Depth"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge Table A (master) with Table B (CTD) using nearest-depth matching, then fill oxygen from CTD Oxygen else Winkler O2."
    )
    ap.add_argument("--table-a", required=True, help="Path to Table A (CSV/TSV).")
    ap.add_argument("--table-b", required=True, help="Path to Table B (CSV/TSV).")
    ap.add_argument("--outdir", required=True, help="Output directory (created if missing).")

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

    ap.add_argument(
        "--max-depth-diff",
        type=float,
        default=None,
        help=(
            "Optional maximum allowed absolute depth difference. "
            "If set, matches with |B.Depth - A.Depth| > this are treated as missing CTD for that row."
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
    # Load as string first, then explicitly coerce keys
    return pd.read_csv(path, sep=sep, header=0, dtype=str, na_values=["", "NA", "NaN"])


def coerce_key_types(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
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

    if dropped > 0:
        # Strict: rows with missing key identity get dropped
        pass

    return out


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_table(df: pd.DataFrame, path: str, sep: str) -> None:
    df.to_csv(path, sep=sep, index=False)


def aggregate_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Collapse exact duplicate keys (commonly in CTD exports).

    Aggregation:
      - Numeric columns: mean
      - Non-numeric columns: first non-null
    """
    non_group = [c for c in df.columns if c not in group_cols]

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
    Tie-handling: average numeric CTD values across tied depths.
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


def add_best_available_oxygen(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create Oxygen_best_available:
      - prefer CTD column 'Oxygen' when present
      - else fall back to Table A 'O2' when Oxygen is missing
    """
    out = merged.copy()

    if "Oxygen" not in out.columns:
        # If CTD doesn't have Oxygen at all, we still produce the column from O2 (if present)
        out["Oxygen"] = np.nan

    if "O2" not in out.columns:
        raise ValueError("Expected Table A column 'O2' to exist in merged output (for fallback).")

    oxy_ctd = pd.to_numeric(out["Oxygen"], errors="coerce")
    oxy_wink = pd.to_numeric(out["O2"], errors="coerce")

    out["Oxygen_best_available"] = oxy_ctd
    miss = out["Oxygen_best_available"].isna()
    out.loc[miss, "Oxygen_best_available"] = oxy_wink.loc[miss]

    return out


def main() -> int:
    args = parse_args()
    ensure_outdir(args.outdir)

    sep_a = guess_sep(args.table_a, args.sep_a)
    sep_b = guess_sep(args.table_b, args.sep_b)

    # Step 0: Load + key coercion
    a_raw = read_table(args.table_a, sep_a)
    b_raw = read_table(args.table_b, sep_b)

    a = coerce_key_types(a_raw, "Table A")
    b = coerce_key_types(b_raw, "Table B")

    if args.verbose:
        print(f"[load] Table A rows (after dropping bad keys): {len(a)}", file=sys.stderr)
        print(f"[load] Table B rows (after dropping bad keys): {len(b)}", file=sys.stderr)

    # Step 1: Pre-aggregate CTD exact duplicates on full keys
    b = aggregate_rows(b, KEY_COLS_WITH_DEPTH)

    # Step 2: Align CTD to Table A via nearest-depth matching
    ctd_aligned = match_ctd_nearest_depth(a=a, b=b, max_depth_diff=args.max_depth_diff, verbose=args.verbose)
    merged = pd.concat([a.reset_index(drop=True), ctd_aligned.reset_index(drop=True)], axis=1)

    out_01 = os.path.join(args.outdir, "01_merged_nearest_depth.tsv")
    write_table(merged, out_01, sep=args.sep_out)

    # Step 3: Oxygen selection (CTD Oxygen if present else Winkler O2)
    merged2 = add_best_available_oxygen(merged)
    out_02 = os.path.join(args.outdir, "02_oxygen_best_available.tsv")
    write_table(merged2, out_02, sep=args.sep_out)

    if args.verbose:
        print(f"[out] {out_01}", file=sys.stderr)
        print(f"[out] {out_02}", file=sys.stderr)
        n_ctd = pd.to_numeric(merged2["Oxygen"], errors="coerce").notna().sum()
        n_wink = pd.to_numeric(merged2["O2"], errors="coerce").notna().sum()
        n_best = pd.to_numeric(merged2["Oxygen_best_available"], errors="coerce").notna().sum()
        print(f"[oxygen] non-null counts: CTD Oxygen={n_ctd}, Winkler O2={n_wink}, best_available={n_best}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
