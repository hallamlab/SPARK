#!/usr/bin/env python3
"""
env_calc_density.py

Compute in-situ seawater density from a TSV using TEOS-10 (gsw).

Requires:
  - salinity (SP)
  - temperature (in-situ, deg C)
  - pressure (dbar) OR depth (m) + latitude (deg)

Optional:
  - --sigma0 to add potential density anomaly at 0 dbar (pressure effects removed).

Example:
  python env_calc_density.py \
    --input data.tsv \
    --salinity-col Salinity \
    --temperature-col Temperature \
    --depth-col Depth \
    --latitude 49.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gsw  # TEOS-10
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'gsw'. Install with: conda install -c conda-forge gsw"
    ) from exc


def _load_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in input.")
    return pd.to_numeric(df[col], errors="coerce").to_numpy()


def _resolve_vector(
    df: pd.DataFrame,
    const_value: float | None,
    col: str | None,
    name: str,
    n: int,
) -> np.ndarray | None:
    if const_value is not None and col:
        raise ValueError(f"Provide either --{name} or --{name}-col, not both.")
    if col:
        return _load_numeric(df, col)
    if const_value is not None:
        return np.full(n, float(const_value))
    return None


def _compute_pressure(
    df: pd.DataFrame,
    pressure_col: str | None,
    depth_col: str | None,
    latitude: np.ndarray | None,
) -> np.ndarray:
    if pressure_col:
        return _load_numeric(df, pressure_col)
    if not depth_col:
        raise ValueError("Provide --pressure-col or --depth-col.")
    if latitude is None:
        raise ValueError("Latitude is required when computing pressure from depth.")
    depth = _load_numeric(df, depth_col)
    return gsw.p_from_z(-depth, latitude)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute in-situ density from salinity, temperature, and pressure/depth."
    )
    ap.add_argument("--input", required=True, help="Input TSV file.")
    ap.add_argument(
        "--output",
        default=None,
        help="Output TSV file (default: <input>_density.tsv).",
    )
    ap.add_argument("--sep", default="\t", help="Input/output separator (default: tab).")

    ap.add_argument("--salinity-col", required=True, help="Salinity column (SP).")
    ap.add_argument("--temperature-col", required=True, help="Temperature column (deg C).")
    ap.add_argument("--pressure-col", default=None, help="Pressure column (dbar).")
    ap.add_argument("--depth-col", default=None, help="Depth column (m, positive down).")

    ap.add_argument("--latitude", type=float, default=None, help="Latitude (deg).")
    ap.add_argument("--latitude-col", default=None, help="Latitude column (deg).")
    ap.add_argument("--longitude", type=float, default=None, help="Longitude (deg).")
    ap.add_argument("--longitude-col", default=None, help="Longitude column (deg).")

    ap.add_argument(
        "--density-col",
        default="density_kg_m3",
        help="Output density column name (default: density_kg_m3).",
    )
    ap.add_argument(
        "--sigma0",
        action="store_true",
        help="Add potential density anomaly (sigma0) alongside in-situ density.",
    )
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    in_path = Path(ns.input)
    out_path = Path(ns.output) if ns.output else in_path.with_name(f"{in_path.stem}_density.tsv")

    df = pd.read_csv(in_path, sep=ns.sep)
    n = len(df)

    sal = _load_numeric(df, ns.salinity_col)
    temp = _load_numeric(df, ns.temperature_col)

    lat = _resolve_vector(df, ns.latitude, ns.latitude_col, "latitude", n)
    lon = _resolve_vector(df, ns.longitude, ns.longitude_col, "longitude", n)
    if lon is None:
        lon = np.zeros(n)

    pressure = _compute_pressure(df, ns.pressure_col, ns.depth_col, lat)
    if lat is None:
        lat = np.zeros(n)
        print("[i] Latitude not provided; using 0 deg for SA conversion.")

    SA = gsw.SA_from_SP(sal, pressure, lon, lat)
    CT = gsw.CT_from_t(SA, temp, pressure)

    rho = gsw.rho(SA, CT, pressure)
    df[ns.density_col] = rho

    if ns.sigma0:
        # sigma0 is useful for stratification because it removes pressure effects.
        sigma0 = gsw.sigma0(SA, CT)
        df["sigma0_kg_m3"] = sigma0
    df.to_csv(out_path, sep=ns.sep, index=False)
    print(f"[i] Wrote {out_path}")


if __name__ == "__main__":
    main()
