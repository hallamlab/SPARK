#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_rename_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"Invalid rename entry '{piece}'. Expected old:new format.")
        old, new = piece.split(":", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise ValueError(f"Invalid rename entry '{piece}'. Empty old/new value.")
        mapping[old] = new
    return mapping


def _id_token(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text:
        return "NA"
    num = pd.to_numeric(text, errors="coerce")
    if pd.notna(num):
        val = float(num)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    return text


def _ensure_default_id_col(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    id_col = "cruise_year_month_depth"
    id_parts = ["Cruise", "Year", "Month", "Depth"]
    if id_col in df.columns or id_col not in keep_cols:
        return df

    missing = [c for c in id_parts if c not in df.columns]
    if missing:
        print(f"[warn] Could not create '{id_col}'; missing source columns: {missing}")
        return df

    out = df.copy()
    out[id_col] = out[id_parts].apply(lambda r: "_".join(_id_token(v) for v in r.values), axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Programmatic replacement for notebook custom biochem cleaning step."
    )
    ap.add_argument("--input", required=True, type=Path, help="Input table (TSV/CSV).")
    ap.add_argument("--output", required=True, type=Path, help="Output table path.")
    ap.add_argument("--in-sep", default="\t", help="Input delimiter (default: tab).")
    ap.add_argument("--out-sep", default="\t", help="Output delimiter (default: tab).")
    ap.add_argument("--keep-cols", default="", help="Comma-separated columns to keep.")
    ap.add_argument("--drop-cols", default="", help="Comma-separated columns to drop.")
    ap.add_argument("--rename-map", default="", help="Comma-separated old:new mappings.")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=args.in_sep)
    before_cols = list(df.columns)

    rename_map = _parse_rename_map(args.rename_map)
    if rename_map:
        for old, new in rename_map.items():
            if old not in df.columns:
                continue
            # If destination exists already, force replacement with source values.
            if new in df.columns and new != old:
                df[new] = df[old]
                df = df.drop(columns=[old])
            elif new != old:
                df = df.rename(columns={old: new})

    keep_cols = _parse_list(args.keep_cols)
    if keep_cols:
        df = _ensure_default_id_col(df, keep_cols)
        present_keep = [c for c in keep_cols if c in df.columns]
        missing_keep = [c for c in keep_cols if c not in df.columns]
        if missing_keep:
            print(f"[warn] keep-cols not found and skipped: {missing_keep}")
        if present_keep:
            df = df.loc[:, present_keep]

    drop_cols = _parse_list(args.drop_cols)
    if drop_cols:
        present_drop = [c for c in drop_cols if c in df.columns]
        if present_drop:
            df = df.drop(columns=present_drop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep=args.out_sep, index=False)

    print(f"[ok] Input rows/cols: {df.shape[0]} / {len(before_cols)}")
    print(f"[ok] Output rows/cols: {df.shape[0]} / {df.shape[1]}")
    print(f"[ok] Wrote cleaned table: {args.output}")


if __name__ == "__main__":
    main()
