#!/usr/bin/env python3
"""
env_stratification_metrics.py

Compute standard physical-oceanography stratification metrics from CTD data:
  - In-situ density (rho) and potential density anomaly (sigma0)
  - Brunt-Vaisala frequency (N^2)
  - Mixed Layer Depth (MLD) by density threshold
  - Potential Energy Anomaly (PEA, Simpson-Hunter)
  - Pycnocline depth (max N^2) and layer-specific metrics
  - Optional global layer split via mld125

Inputs: salinity, temperature, depth, and either pressure or latitude (for depth->pressure).
Outputs: density profiles, N^2 profiles, per-profile summary metrics, upper/lower feature traces, and review plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import gsw  # TEOS-10
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'gsw'. Install with: conda install -c conda-forge gsw"
    ) from exc


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _split_float_csv(s: str | None) -> list[float]:
    parts = _split_csv(s)
    if not parts:
        return []
    return [float(p) for p in parts]


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _infer_plot_feature_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if str(col).startswith("ASV"):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(vals).any():
            cols.append(col)
    return cols


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()


def _resolve_vector(
    df: pd.DataFrame,
    col: str | None,
    value: float | None,
    name: str,
    n: int,
) -> np.ndarray | None:
    if col and value is not None:
        raise ValueError(f"Provide either --{name} or --{name}-col, not both.")
    if col:
        return _numeric(df[col])
    if value is not None:
        return np.full(n, float(value))
    return None


def _aggregate_by_depth(
    df: pd.DataFrame,
    depth_col: str,
    cols: list[str],
) -> pd.DataFrame:
    grouped = df.groupby(depth_col, sort=True)
    out = grouped[cols].mean().reset_index()
    out["__n_samples"] = grouped.size().to_numpy()
    return out


def _compute_pressure(
    depth: np.ndarray,
    pressure: np.ndarray | None,
    lat: np.ndarray | None,
) -> np.ndarray:
    if pressure is not None:
        return pressure
    if lat is None:
        raise ValueError("Latitude is required to compute pressure from depth.")
    return gsw.p_from_z(-depth, lat)


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapezoid was added in newer NumPy; use trapz for compatibility.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _compute_pea(depth: np.ndarray, sigma0: np.ndarray, lat: float | None) -> float:
    depth_rel = depth - np.nanmin(depth)
    h = np.nanmax(depth_rel)
    if not np.isfinite(h) or h <= 0:
        return np.nan
    rho_bar = _integrate_trapezoid(sigma0, depth_rel) / h
    g = float(gsw.grav(lat, 0)) if lat is not None else 9.81
    pea = g / h * _integrate_trapezoid((sigma0 - rho_bar) * depth_rel, depth_rel)
    return float(pea)


def _compute_mld(
    depth: np.ndarray,
    sigma0: np.ndarray,
    ref_depth: float,
    delta_rho: float,
) -> tuple[float, float, float]:
    if len(depth) == 0:
        return (np.nan, np.nan, np.nan)
    idx_ref = int(np.nanargmin(np.abs(depth - ref_depth)))
    rho_ref = sigma0[idx_ref]
    valid = np.isfinite(sigma0) & np.isfinite(depth)
    depth_v = depth[valid]
    sigma_v = sigma0[valid]
    if len(depth_v) == 0:
        return (np.nan, np.nan, np.nan)
    candidates = depth_v[(sigma_v - rho_ref) <= delta_rho]
    if len(candidates) == 0:
        return (np.nan, rho_ref, np.nan)
    mld = float(np.nanmax(candidates))
    deeper_mask = depth_v > mld
    if not deeper_mask.any():
        delta_below = np.nan
    else:
        idx_below = int(np.nanargmin(depth_v[deeper_mask]))
        sigma_below = sigma_v[deeper_mask][idx_below]
        delta_below = float(sigma_below - rho_ref)
    return (mld, rho_ref, delta_below)


def _format_threshold_tag(value: float) -> str:
    s = f"{value:g}"
    return s.replace(".", "p")


def _adaptive_delta_rho(
    depth: np.ndarray,
    sigma0: np.ndarray,
    layer_max_depth: float,
    percentile: float,
) -> float:
    if depth.size == 0:
        return np.nan
    min_depth = float(np.nanmin(depth))
    surface_mask = depth <= (min_depth + layer_max_depth)
    surface_sigma = sigma0[surface_mask & np.isfinite(sigma0)]
    if surface_sigma.size < 2:
        return np.nan
    low = float(np.nanmin(surface_sigma))
    high = float(np.nanpercentile(surface_sigma, percentile))
    delta = high - low
    return float(delta) if np.isfinite(delta) and delta > 0 else np.nan


def _split_layers(
    depth: np.ndarray,
    split_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(split_depth):
        return (np.array([], dtype=bool), np.array([], dtype=bool))
    upper = depth <= split_depth
    lower = depth > split_depth
    return upper, lower


def _compute_global_pycnocline_depth(
    df: pd.DataFrame,
    profile_cols: list[str],
    sal_col: str,
    temp_col: str,
    depth_col: str,
    pressure_col: str | None,
    latitude_col: str | None,
    longitude_col: str | None,
    latitude: float | None,
    longitude: float | None,
) -> float:
    max_depth = np.nan
    warned_lat_zero = False

    for _, group in df.groupby(profile_cols):
        g = group.copy()
        g["__sal"] = _numeric(g[sal_col])
        g["__temp"] = _numeric(g[temp_col])
        g["__depth"] = _numeric(g[depth_col])
        g["__press"] = _numeric(g[pressure_col]) if pressure_col else np.nan
        lat_vec = _resolve_vector(g, latitude_col, latitude, "latitude", len(g))
        lon_vec = _resolve_vector(g, longitude_col, longitude, "longitude", len(g))
        g["__lat"] = lat_vec if lat_vec is not None else np.nan
        g["__lon"] = lon_vec if lon_vec is not None else np.nan

        req_mask = np.isfinite(g["__sal"]) & np.isfinite(g["__temp"]) & np.isfinite(g["__depth"])
        g = g.loc[req_mask]
        if g.empty:
            continue

        depth_df = _aggregate_by_depth(
            g,
            depth_col=depth_col,
            cols=["__sal", "__temp", "__press", "__lat", "__lon"],
        )
        depth = _numeric(depth_df[depth_col])
        sal = _numeric(depth_df["__sal"])
        temp = _numeric(depth_df["__temp"])
        lat = _numeric(depth_df["__lat"]) if "__lat" in depth_df else None
        lon = _numeric(depth_df["__lon"]) if "__lon" in depth_df else None
        if lat is not None and np.isnan(lat).all():
            lat = None
        if lon is not None and np.isnan(lon).all():
            lon = None
        pressure = _numeric(depth_df["__press"]) if pressure_col else None

        if lat is None:
            if pressure_col:
                lat = np.zeros(len(depth))
                if not warned_lat_zero:
                    print("[i] Latitude not provided; using 0 deg for SA conversion.")
                    warned_lat_zero = True
            else:
                raise ValueError("Latitude is required when pressure is not provided.")
        if lon is None:
            lon = np.zeros(len(depth))

        pressure = _compute_pressure(depth, pressure, lat)
        if len(depth) < 2:
            continue

        SA = gsw.SA_from_SP(sal, pressure, lon, lat)
        CT = gsw.CT_from_t(SA, temp, pressure)
        n2, p_mid = gsw.Nsquared(SA, CT, pressure, lat)
        if n2.size == 0:
            continue
        lat_mid = float(np.nanmean(lat)) if np.isfinite(np.nanmean(lat)) else 0.0
        depth_mid = -gsw.z_from_p(p_mid, lat_mid)
        idx_max = int(np.nanargmax(n2))
        depth_max = float(depth_mid[idx_max]) if depth_mid.size else np.nan
        if np.isfinite(depth_max):
            max_depth = depth_max if not np.isfinite(max_depth) else max(max_depth, depth_max)

    return max_depth


def _compute_global_mld_depth(
    df: pd.DataFrame,
    profile_cols: list[str],
    sal_col: str,
    temp_col: str,
    depth_col: str,
    pressure_col: str | None,
    latitude_col: str | None,
    longitude_col: str | None,
    latitude: float | None,
    longitude: float | None,
    ref_depth: float,
    delta_rho: float,
    stat: str,
) -> float:
    mld_vals: list[float] = []
    warned_lat_zero = False

    for _, group in df.groupby(profile_cols):
        g = group.copy()
        g["__sal"] = _numeric(g[sal_col])
        g["__temp"] = _numeric(g[temp_col])
        g["__depth"] = _numeric(g[depth_col])
        g["__press"] = _numeric(g[pressure_col]) if pressure_col else np.nan
        lat_vec = _resolve_vector(g, latitude_col, latitude, "latitude", len(g))
        lon_vec = _resolve_vector(g, longitude_col, longitude, "longitude", len(g))
        g["__lat"] = lat_vec if lat_vec is not None else np.nan
        g["__lon"] = lon_vec if lon_vec is not None else np.nan

        req_mask = np.isfinite(g["__sal"]) & np.isfinite(g["__temp"]) & np.isfinite(g["__depth"])
        g = g.loc[req_mask]
        if g.empty:
            continue

        depth_df = _aggregate_by_depth(
            g,
            depth_col=depth_col,
            cols=["__sal", "__temp", "__press", "__lat", "__lon"],
        )
        depth = _numeric(depth_df[depth_col])
        sal = _numeric(depth_df["__sal"])
        temp = _numeric(depth_df["__temp"])
        lat = _numeric(depth_df["__lat"]) if "__lat" in depth_df else None
        lon = _numeric(depth_df["__lon"]) if "__lon" in depth_df else None
        if lat is not None and np.isnan(lat).all():
            lat = None
        if lon is not None and np.isnan(lon).all():
            lon = None
        pressure = _numeric(depth_df["__press"]) if pressure_col else None

        if lat is None:
            if pressure_col:
                lat = np.zeros(len(depth))
                if not warned_lat_zero:
                    print("[i] Latitude not provided; using 0 deg for SA conversion.")
                    warned_lat_zero = True
            else:
                raise ValueError("Latitude is required when pressure is not provided.")
        if lon is None:
            lon = np.zeros(len(depth))

        pressure = _compute_pressure(depth, pressure, lat)
        if len(depth) < 2:
            continue

        SA = gsw.SA_from_SP(sal, pressure, lon, lat)
        CT = gsw.CT_from_t(SA, temp, pressure)
        sigma0 = gsw.sigma0(SA, CT)
        mld_depth, _, _ = _compute_mld(
            depth=depth,
            sigma0=sigma0,
            ref_depth=ref_depth,
            delta_rho=delta_rho,
        )
        if np.isfinite(mld_depth):
            mld_vals.append(float(mld_depth))

    if not mld_vals:
        return np.nan
    arr = np.array(mld_vals, dtype=float)
    if stat == "max":
        return float(np.nanmax(arr))
    if stat == "p90":
        return float(np.nanpercentile(arr, 90))
    if stat == "median":
        return float(np.nanmedian(arr))
    raise ValueError(f"Unknown --layer-split-stat '{stat}'.")


def _profile_label(keys: dict[str, object]) -> str:
    return "|".join([str(v) for v in keys.values()])


def _classify_pea(
    summary_df: pd.DataFrame,
    mode: str,
    low_q: float,
    high_q: float,
    month_col: str | None,
    date_col: str | None,
) -> pd.DataFrame:
    if "pea_J_m3" not in summary_df.columns:
        return summary_df

    df = summary_df.copy()
    df["pea_class"] = "unknown"
    df["pea_threshold_low"] = np.nan
    df["pea_threshold_high"] = np.nan

    if mode == "global":
        values = df["pea_J_m3"].to_numpy()
        valid = np.isfinite(values)
        if not valid.any():
            return df
        low = float(np.nanquantile(values[valid], low_q))
        high = float(np.nanquantile(values[valid], high_q))
        df["pea_threshold_low"] = low
        df["pea_threshold_high"] = high
        df.loc[valid & (values <= low), "pea_class"] = "mixed"
        df.loc[valid & (values >= high), "pea_class"] = "stratified"
        df.loc[valid & (values > low) & (values < high), "pea_class"] = "transition"
        return df

    if mode == "monthly":
        if month_col and month_col in df.columns:
            df["__pea_month"] = pd.to_numeric(df[month_col], errors="coerce")
        elif date_col and date_col in df.columns:
            df["__pea_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.month
        else:
            raise ValueError("--pea-class-mode monthly requires --month-col or --date-col.")

        for month, sub_idx in df.groupby("__pea_month").groups.items():
            if not np.isfinite(month):
                continue
            sub_vals = df.loc[sub_idx, "pea_J_m3"].to_numpy()
            valid = np.isfinite(sub_vals)
            if not valid.any():
                continue
            low = float(np.nanquantile(sub_vals[valid], low_q))
            high = float(np.nanquantile(sub_vals[valid], high_q))
            df.loc[sub_idx, "pea_threshold_low"] = low
            df.loc[sub_idx, "pea_threshold_high"] = high
            df.loc[sub_idx, "pea_class"] = "transition"
            df.loc[sub_idx, "pea_class"] = np.where(
                valid & (sub_vals <= low),
                "mixed",
                df.loc[sub_idx, "pea_class"],
            )
            df.loc[sub_idx, "pea_class"] = np.where(
                valid & (sub_vals >= high),
                "stratified",
                df.loc[sub_idx, "pea_class"],
            )
        df = df.drop(columns=["__pea_month"])
        return df

    raise ValueError(f"Unknown --pea-class-mode '{mode}'.")


def _classify_deep_intrusion(
    summary_df: pd.DataFrame,
    mode: str,
    quantile: float,
    value_col: str,
) -> pd.DataFrame:
    if value_col not in summary_df.columns:
        return summary_df

    df = summary_df.copy()
    df["deep_intrusion_class"] = "unknown"
    df["deep_intrusion_threshold"] = np.nan
    df["deep_intrusion_score"] = np.nan

    if mode == "none":
        return df
    if mode != "global":
        raise ValueError(f"Unknown --deep-intrusion-mode '{mode}'.")

    values = df[value_col].to_numpy()
    valid = np.isfinite(values)
    if not valid.any():
        return df
    thr = float(np.nanquantile(values[valid], quantile))
    mean = float(np.nanmean(values[valid]))
    std = float(np.nanstd(values[valid]))
    if std > 0:
        df.loc[valid, "deep_intrusion_score"] = (values[valid] - mean) / std
    df["deep_intrusion_threshold"] = thr
    df.loc[valid & (values >= thr), "deep_intrusion_class"] = "intrusion"
    df.loc[valid & (values < thr), "deep_intrusion_class"] = "baseline"
    return df


def _plot_review(
    summary_df: pd.DataFrame,
    mld_long_df: pd.DataFrame,
    feature_df: pd.DataFrame | None,
    feature_cols: list[str] | None,
    out_path: Path,
    date_col: str | None,
    layer_split_depth: float | None,
) -> None:
    if summary_df.empty:
        print("[i] No summary rows to plot.")
        return

    plot_df = summary_df.copy()
    use_dates = False
    if date_col and "profile_date" in plot_df.columns:
        plot_df["plot_x"] = pd.to_datetime(plot_df["profile_date"], errors="coerce")
        use_dates = plot_df["plot_x"].notna().sum() >= 2
    if not use_dates:
        plot_df["plot_x"] = np.arange(len(plot_df))

    plot_df = plot_df.sort_values("plot_x")
    x_map = dict(zip(plot_df["profile_label"], plot_df["plot_x"]))

    has_pea_class = "pea_class" in plot_df.columns
    has_intrusion = "deep_intrusion_class" in plot_df.columns
    plot_features = feature_cols or []
    if feature_df is None:
        plot_features = []
    else:
        kept = []
        for feature in plot_features:
            upper_col = f"{feature}_upper"
            lower_col = f"{feature}_lower"
            if upper_col in feature_df.columns or lower_col in feature_df.columns:
                kept.append(feature)
        dropped = [f for f in plot_features if f not in kept]
        plot_features = kept
        if dropped:
            print(f"  [i] Dropping features with no upper/lower traces: {dropped}")
    if not plot_features:
        print("  [i] No feature trace panels to plot (empty feature set).")
    nrows = 4 + int(has_intrusion) + int(has_pea_class) + len(plot_features)
    fig, axes = plt.subplots(nrows, 1, figsize=(20, 3.0 * nrows), sharex=True)
    axes = np.atleast_1d(axes)

    def _place_legend(ax: plt.Axes) -> None:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # MLD + pycnocline panel
    ax = axes[0]
    if not mld_long_df.empty:
        for label, sub in mld_long_df.groupby("threshold_label"):
            sub = sub.copy()
            sub["plot_x"] = sub["profile_label"].map(x_map)
            sub = sub.dropna(subset=["plot_x", "mld_depth_m"])
            if sub.empty:
                continue
            sub = sub.sort_values("plot_x")
            ax.plot(sub["plot_x"], sub["mld_depth_m"], marker="o", label=label)
        ax.invert_yaxis()
        ax.set_ylabel("Depth (m)")
    if np.isfinite(layer_split_depth or np.nan):
        ax.plot(
            plot_df["plot_x"],
            np.full(len(plot_df), layer_split_depth),
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="layer split depth",
        )
    if "pycnocline_depth_m" in plot_df:
        ax.plot(
            plot_df["plot_x"],
            plot_df["pycnocline_depth_m"],
            color="black",
            linestyle="--",
            marker="o",
            label="pycnocline",
        )
    if ax.get_legend_handles_labels()[1]:
        _place_legend(ax)
    ax.set_title("Mixed Layer Depth + Pycnocline")

    # Adaptive delta panel
    ax = axes[1]
    adaptive = mld_long_df[mld_long_df["threshold_type"] == "adaptive"].copy()
    if not adaptive.empty:
        adaptive["plot_x"] = adaptive["profile_label"].map(x_map)
        adaptive = adaptive.dropna(subset=["plot_x", "delta_rho_kg_m3"])
        adaptive = adaptive.sort_values("plot_x")
        ax.plot(adaptive["plot_x"], adaptive["delta_rho_kg_m3"], marker="o", color="tab:orange")
    ax.set_ylabel("Adaptive delta_rho (kg/m3)")
    ax.set_title("Adaptive threshold used")

    # N2 panel
    ax = axes[2]
    ax.plot(plot_df["plot_x"], plot_df["n2_mean_s-2"], marker="o", label="N2 mean")
    if "n2_mean_upper_s-2" in plot_df:
        ax.plot(plot_df["plot_x"], plot_df["n2_mean_upper_s-2"], marker="o", label="N2 mean upper")
    if "n2_mean_lower_s-2" in plot_df:
        ax.plot(plot_df["plot_x"], plot_df["n2_mean_lower_s-2"], marker="o", label="N2 mean lower")
    ax.plot(plot_df["plot_x"], plot_df["n2_max_s-2"], marker="o", label="N2 max")
    ax.set_ylabel("N2 (s^-2)")
    ax.set_title("Brunt-Vaisala frequency")
    _place_legend(ax)

    # PEA panel
    ax = axes[3]
    ax.plot(plot_df["plot_x"], plot_df["pea_J_m3"], marker="o", color="tab:green", label="PEA total")
    if "pea_upper_J_m3" in plot_df:
        ax.plot(plot_df["plot_x"], plot_df["pea_upper_J_m3"], marker="o", label="PEA upper")
    if "pea_lower_J_m3" in plot_df:
        ax.plot(plot_df["plot_x"], plot_df["pea_lower_J_m3"], marker="o", label="PEA lower")
    ax.set_ylabel("PEA (J/m3)")
    ax.set_title("Potential Energy Anomaly")
    _place_legend(ax)

    panel_idx = 4
    if has_intrusion:
        ax = axes[panel_idx]
        colors = {
            "intrusion": "tab:red",
            "baseline": "tab:blue",
            "unknown": "tab:gray",
        }
        for cls in ["intrusion", "baseline", "unknown"]:
            sub = plot_df[plot_df["deep_intrusion_class"] == cls]
            if sub.empty:
                continue
            ax.scatter(
                sub["plot_x"],
                sub["sigma0_lower_mean_kg_m3"],
                label=cls,
                color=colors[cls],
                s=30,
            )
        if "deep_intrusion_threshold" in plot_df:
            ax.plot(
                plot_df["plot_x"],
                plot_df["deep_intrusion_threshold"],
                linestyle="--",
                color="black",
                label="intrusion threshold",
            )
        ax.set_ylabel("Sigma0 lower mean (kg/m3)")
        ax.set_title("Deep intrusion indicator")
        _place_legend(ax)
        panel_idx += 1

    # PEA classification panel
    if has_pea_class:
        ax = axes[panel_idx]
        colors = {
            "mixed": "tab:blue",
            "transition": "tab:gray",
            "stratified": "tab:red",
            "unknown": "tab:purple",
        }
        for cls in ["mixed", "transition", "stratified", "unknown"]:
            sub = plot_df[plot_df["pea_class"] == cls]
            if sub.empty:
                continue
            ax.scatter(sub["plot_x"], sub["pea_J_m3"], label=cls, color=colors[cls], s=30)
        if "pea_threshold_low" in plot_df and "pea_threshold_high" in plot_df:
            ax.plot(plot_df["plot_x"], plot_df["pea_threshold_low"], linestyle="--", color="black", label="PEA low")
            ax.plot(plot_df["plot_x"], plot_df["pea_threshold_high"], linestyle="--", color="black", label="PEA high")
        ax.set_ylabel("PEA (J/m3)")
        ax.set_title("PEA classification")
        _place_legend(ax)
        panel_idx += 1

    if plot_features:
        feature_plot = feature_df.copy()
        feature_plot["plot_x"] = feature_plot["profile_label"].map(x_map)
        for feature in plot_features:
            ax = axes[panel_idx]
            upper_col = f"{feature}_upper"
            lower_col = f"{feature}_lower"
            upper_n = 0
            lower_n = 0
            if upper_col in feature_plot:
                sub = feature_plot[["plot_x", upper_col]].dropna()
                if not sub.empty:
                    sub = sub.sort_values("plot_x")
                    ax.scatter(sub["plot_x"], sub[upper_col], color="tab:blue", s=25, label="upper")
                    upper_n = len(sub)
            if lower_col in feature_plot:
                sub = feature_plot[["plot_x", lower_col]].dropna()
                if not sub.empty:
                    sub = sub.sort_values("plot_x")
                    ax.scatter(sub["plot_x"], sub[lower_col], color="tab:orange", s=25, label="lower")
                    lower_n = len(sub)
            if upper_n == 0 and lower_n == 0:
                print(f"  [i] Feature '{feature}' has no upper/lower points to plot.")
            ax.set_ylabel(feature)
            ax.set_title(feature)
            if ax.get_legend_handles_labels()[1]:
                _place_legend(ax)
            panel_idx += 1

    if use_dates:
        axes[-1].set_xlabel("Date")
    else:
        axes[-1].set_xlabel("Profile index")

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute stratification metrics (density, N^2, MLD, PEA) from CTD data."
    )
    ap.add_argument("--input", required=True, help="Input TSV file.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--sep", default="\t", help="Input/output separator (default: tab).")

    ap.add_argument("--salinity-col", required=True, help="Salinity column (SP).")
    ap.add_argument("--temperature-col", required=True, help="Temperature column (deg C).")
    ap.add_argument("--depth-col", required=True, help="Depth column (m, positive down).")
    ap.add_argument("--pressure-col", default=None, help="Pressure column (dbar).")

    ap.add_argument("--latitude", type=float, default=None, help="Latitude (deg).")
    ap.add_argument("--latitude-col", default=None, help="Latitude column (deg).")
    ap.add_argument("--longitude", type=float, default=None, help="Longitude (deg).")
    ap.add_argument("--longitude-col", default=None, help="Longitude column (deg).")

    ap.add_argument(
        "--profile-cols",
        default=None,
        help="Comma-separated columns defining each profile (default: treat all rows as one profile).",
    )

    ap.add_argument("--depth-min", type=float, default=None, help="Minimum depth to include.")
    ap.add_argument("--depth-max", type=float, default=None, help="Maximum depth to include.")

    ap.add_argument(
        "--mld-delta-rho",
        default="0.03,0.125",
        help="Comma-separated MLD density thresholds (kg/m^3, default 0.03,0.125).",
    )
    ap.add_argument(
        "--mld-reference-depth",
        type=float,
        default=10.0,
        help="Reference depth for MLD (m, default 10).",
    )
    ap.add_argument(
        "--adaptive-layer-max-depth",
        type=float,
        default=10.0,
        help="Surface layer depth for adaptive MLD threshold (m, default 10).",
    )
    ap.add_argument(
        "--adaptive-percentile",
        type=float,
        default=90.0,
        help="Percentile for adaptive delta_rho (default 90).",
    )
    ap.add_argument(
        "--layer-split-mode",
        choices=["global_pycnocline", "mld125", "explicit"],
        default="global_pycnocline",
        help="Layer split depth mode (default global_pycnocline).",
    )
    ap.add_argument(
        "--layer-split-stat",
        choices=["max", "p90", "median"],
        default="max",
        help="Statistic for mld125 layer split depth (default max).",
    )
    ap.add_argument(
        "--layer-split-depth",
        type=float,
        default=None,
        help="Explicit depth for layer split (m). Use with --layer-split-mode explicit.",
    )
    ap.add_argument("--date-col", default=None, help="Optional date column for plotting.")
    ap.add_argument("--month-col", default=None, help="Optional month column for PEA classification.")
    ap.add_argument(
        "--pea-class-mode",
        choices=["global", "monthly", "none"],
        default="global",
        help="PEA classification mode (default global).",
    )
    ap.add_argument(
        "--pea-low-quantile",
        type=float,
        default=0.30,
        help="Lower quantile for PEA class split (default 0.30).",
    )
    ap.add_argument(
        "--pea-high-quantile",
        type=float,
        default=0.70,
        help="Upper quantile for PEA class split (default 0.70).",
    )
    ap.add_argument(
        "--plot-feature-cols",
        default=None,
        help="Comma-separated feature columns to plot as traces (default: infer non-metadata numeric columns).",
    )
    ap.add_argument(
        "--deep-intrusion-mode",
        choices=["global", "none"],
        default="global",
        help="Deep intrusion classification mode (default global).",
    )
    ap.add_argument(
        "--deep-intrusion-quantile",
        type=float,
        default=0.85,
        help="Quantile for deep intrusion threshold (default 0.85).",
    )
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STRATIFICATION METRICS (TEOS-10)")
    print("=" * 70)

    print("\n[1/9] Loading input table...")
    df = pd.read_csv(args.input, sep=args.sep)

    required = [
        args.salinity_col,
        args.temperature_col,
        args.depth_col,
    ]
    if args.pressure_col:
        required.append(args.pressure_col)
    if args.latitude_col:
        required.append(args.latitude_col)
    if args.longitude_col:
        required.append(args.longitude_col)
    if args.date_col:
        required.append(args.date_col)
    if args.month_col:
        required.append(args.month_col)
    _require_cols(df, required)

    profile_cols = _split_csv(args.profile_cols)
    if not profile_cols:
        df = df.copy()
        df["profile_id"] = "all"
        profile_cols = ["profile_id"]
    _require_cols(df, profile_cols)

    if args.depth_min is not None:
        df = df[pd.to_numeric(df[args.depth_col], errors="coerce") >= args.depth_min]
    if args.depth_max is not None:
        df = df[pd.to_numeric(df[args.depth_col], errors="coerce") <= args.depth_max]

    print(f"  Profiles: {df[profile_cols].drop_duplicates().shape[0]}")

    exclude_cols = {
        args.salinity_col,
        args.temperature_col,
        args.depth_col,
        args.pressure_col,
        args.latitude_col,
        args.longitude_col,
        args.date_col,
        args.month_col,
    }
    exclude_cols.update(profile_cols)
    exclude_cols = {c for c in exclude_cols if c}

    if args.plot_feature_cols:
        plot_feature_cols = _split_csv(args.plot_feature_cols)
        missing = [c for c in plot_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"--plot-feature-cols includes columns not found in input: {missing}")
    else:
        plot_feature_cols = _infer_plot_feature_cols(df, exclude_cols)
    plot_feature_cols = [c for c in plot_feature_cols if c not in profile_cols]
    if plot_feature_cols:
        print(f"  [i] Plotting {len(plot_feature_cols)} feature traces")
    else:
        print("  [i] No feature traces selected for plotting")

    if args.layer_split_mode == "explicit":
        if args.layer_split_depth is None:
            raise ValueError("--layer-split-depth is required when --layer-split-mode explicit.")
        layer_split_depth = float(args.layer_split_depth)
        layer_split_method = "explicit"
        print(f"\n[2/9] Using explicit layer split depth: {layer_split_depth:.2f} m")
    else:
        if args.layer_split_depth is not None:
            raise ValueError("Use --layer-split-mode explicit when providing --layer-split-depth.")
        if args.layer_split_mode == "global_pycnocline":
            print("\n[2/9] Computing global pycnocline split depth...")
            layer_split_depth = _compute_global_pycnocline_depth(
                df=df,
                profile_cols=profile_cols,
                sal_col=args.salinity_col,
                temp_col=args.temperature_col,
                depth_col=args.depth_col,
                pressure_col=args.pressure_col,
                latitude_col=args.latitude_col,
                longitude_col=args.longitude_col,
                latitude=args.latitude,
                longitude=args.longitude,
            )
            if np.isfinite(layer_split_depth):
                layer_split_method = "global_pycnocline_max"
                print(f"  [i] Using global pycnocline max depth for layer split: {layer_split_depth:.2f} m")
            else:
                layer_split_method = "none"
                print("  [i] No valid pycnocline depths; layer metrics will be NaN.")
        elif args.layer_split_mode == "mld125":
            print("\n[2/9] Computing global mld125 split depth...")
            layer_split_depth = _compute_global_mld_depth(
                df=df,
                profile_cols=profile_cols,
                sal_col=args.salinity_col,
                temp_col=args.temperature_col,
                depth_col=args.depth_col,
                pressure_col=args.pressure_col,
                latitude_col=args.latitude_col,
                longitude_col=args.longitude_col,
                latitude=args.latitude,
                longitude=args.longitude,
                ref_depth=args.mld_reference_depth,
                delta_rho=0.125,
                stat=args.layer_split_stat,
            )
            if np.isfinite(layer_split_depth):
                layer_split_method = f"mld125_{args.layer_split_stat}"
                print(
                    f"  [i] Using global mld125 ({args.layer_split_stat}) depth for layer split: "
                    f"{layer_split_depth:.2f} m"
                )
            else:
                layer_split_method = "none"
                print("  [i] No valid mld125 depths; layer metrics will be NaN.")
        else:
            raise ValueError(f"Unknown --layer-split-mode '{args.layer_split_mode}'.")

    density_rows = []
    n2_rows = []
    summary_rows = []
    mld_rows = []
    warned_lat_zero = False
    fixed_thresholds = _split_float_csv(args.mld_delta_rho)
    fixed_thresholds = [t for t in fixed_thresholds if t > 0]
    if not fixed_thresholds:
        raise ValueError("--mld-delta-rho must include at least one positive threshold.")

    print("\n[3/9] Computing profiles...")
    for keys, group in df.groupby(profile_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(profile_cols, keys))
        profile_label = _profile_label(key_map)

        g = group.copy()
        g["__sal"] = _numeric(g[args.salinity_col])
        g["__temp"] = _numeric(g[args.temperature_col])
        g["__depth"] = _numeric(g[args.depth_col])
        g["__press"] = _numeric(g[args.pressure_col]) if args.pressure_col else np.nan
        lat_vec = _resolve_vector(g, args.latitude_col, args.latitude, "latitude", len(g))
        lon_vec = _resolve_vector(g, args.longitude_col, args.longitude, "longitude", len(g))
        g["__lat"] = lat_vec if lat_vec is not None else np.nan
        g["__lon"] = lon_vec if lon_vec is not None else np.nan

        req_mask = np.isfinite(g["__sal"]) & np.isfinite(g["__temp"]) & np.isfinite(g["__depth"])
        g = g.loc[req_mask]
        if g.empty:
            continue

        depth_df = _aggregate_by_depth(
            g,
            depth_col=args.depth_col,
            cols=["__sal", "__temp", "__press", "__lat", "__lon"],
        )

        depth = _numeric(depth_df[args.depth_col])
        sal = _numeric(depth_df["__sal"])
        temp = _numeric(depth_df["__temp"])
        lat = _numeric(depth_df["__lat"]) if "__lat" in depth_df else None
        lon = _numeric(depth_df["__lon"]) if "__lon" in depth_df else None
        if lat is not None and np.isnan(lat).all():
            lat = None
        if lon is not None and np.isnan(lon).all():
            lon = None
        pressure = _numeric(depth_df["__press"]) if args.pressure_col else None

        if lat is None:
            if args.pressure_col:
                lat = np.zeros(len(depth))
                if not warned_lat_zero:
                    print("[i] Latitude not provided; using 0 deg for SA conversion.")
                    warned_lat_zero = True
            else:
                raise ValueError("Latitude is required when pressure is not provided.")
        if lon is None:
            lon = np.zeros(len(depth))

        pressure = _compute_pressure(depth, pressure, lat)

        SA = gsw.SA_from_SP(sal, pressure, lon, lat)
        CT = gsw.CT_from_t(SA, temp, pressure)
        rho = gsw.rho(SA, CT, pressure)
        sigma0 = gsw.sigma0(SA, CT)

        profile_date = None
        if args.date_col:
            date_vals = g[args.date_col].dropna()
            if not date_vals.empty:
                profile_date = date_vals.iloc[0]

        n2 = np.array([])
        p_mid = np.array([])
        depth_mid = np.array([])
        if len(depth) >= 2:
            n2, p_mid = gsw.Nsquared(SA, CT, pressure, lat)
            lat_mid = float(np.nanmean(lat)) if np.isfinite(np.nanmean(lat)) else 0.0
            depth_mid = -gsw.z_from_p(p_mid, lat_mid)

        mld_summary = {}
        mld_rho_ref = np.nan
        mld_entries = []
        for delta_rho in fixed_thresholds:
            mld_depth, rho_ref, delta_below = _compute_mld(
                depth=depth,
                sigma0=sigma0,
                ref_depth=args.mld_reference_depth,
                delta_rho=delta_rho,
            )
            if np.isnan(mld_rho_ref):
                mld_rho_ref = rho_ref
            tag = _format_threshold_tag(delta_rho)
            mld_summary[f"mld_depth_m_dr{tag}"] = mld_depth
            mld_summary[f"mld_delta_rho_below_kg_m3_dr{tag}"] = delta_below
            mld_entries.append(
                {
                    **key_map,
                    "profile_label": profile_label,
                    "profile_date": profile_date,
                    "threshold_type": "fixed",
                    "threshold_label": f"dr{tag}",
                    "delta_rho_kg_m3": float(delta_rho),
                    "mld_depth_m": mld_depth,
                    "mld_rho_ref_kg_m3": rho_ref,
                    "mld_delta_rho_below_kg_m3": delta_below,
                }
            )

        adaptive_delta = _adaptive_delta_rho(
            depth=depth,
            sigma0=sigma0,
            layer_max_depth=args.adaptive_layer_max_depth,
            percentile=args.adaptive_percentile,
        )
        if np.isfinite(adaptive_delta):
            mld_depth, rho_ref, delta_below = _compute_mld(
                depth=depth,
                sigma0=sigma0,
                ref_depth=args.mld_reference_depth,
                delta_rho=adaptive_delta,
            )
        else:
            mld_depth, rho_ref, delta_below = (np.nan, np.nan, np.nan)
        mld_entries.append(
            {
                **key_map,
                "profile_label": profile_label,
                "profile_date": profile_date,
                "threshold_type": "adaptive",
                "threshold_label": "adaptive",
                "delta_rho_kg_m3": adaptive_delta,
                "mld_depth_m": mld_depth,
                "mld_rho_ref_kg_m3": rho_ref,
                "mld_delta_rho_below_kg_m3": delta_below,
            }
        )
        lat_mean = float(np.nanmean(lat)) if np.isfinite(np.nanmean(lat)) else np.nan
        pea = _compute_pea(depth=depth, sigma0=sigma0, lat=lat_mean if np.isfinite(lat_mean) else None)
        n2_mean = float(np.nanmean(n2)) if n2.size else np.nan
        n2_max = float(np.nanmax(n2)) if n2.size else np.nan
        if n2.size:
            idx_max = int(np.nanargmax(n2))
            n2_max_depth = float(depth_mid[idx_max]) if depth_mid.size else np.nan
        else:
            n2_max_depth = np.nan
        pycnocline_depth = n2_max_depth

        if n2.size:
            if np.isfinite(layer_split_depth):
                n2_layer = np.where(depth_mid <= layer_split_depth, "upper", "lower")
            else:
                n2_layer = np.array(["unknown"] * len(n2))
            for i in range(len(n2)):
                row = {
                    **key_map,
                    "profile_label": profile_label,
                    "profile_date": profile_date,
                    "pressure_mid_dbar": p_mid[i],
                    "depth_mid_m": depth_mid[i],
                    "N2_s-2": n2[i],
                    "pycnocline_depth_m": pycnocline_depth,
                    "layer_split_depth_m": layer_split_depth,
                    "layer_split_method": layer_split_method,
                    "layer": n2_layer[i],
                }
                n2_rows.append(row)

        if np.isfinite(layer_split_depth):
            density_layer = np.where(depth <= layer_split_depth, "upper", "lower")
        else:
            density_layer = np.array(["unknown"] * len(depth))

        for i in range(len(depth)):
            row = {
                **key_map,
                "profile_label": profile_label,
                "profile_date": profile_date,
                args.depth_col: depth[i],
                "pycnocline_depth_m": pycnocline_depth,
                "layer_split_depth_m": layer_split_depth,
                "layer_split_method": layer_split_method,
                "layer": density_layer[i],
                "pressure_dbar": pressure[i],
                "salinity_sp": sal[i],
                "temperature_c": temp[i],
                "SA": SA[i],
                "CT": CT[i],
                "rho_kg_m3": rho[i],
                "sigma0_kg_m3": sigma0[i],
                "n_samples_at_depth": int(depth_df["__n_samples"].iloc[i]),
            }
            density_rows.append(row)

        upper_mask_depth, lower_mask_depth = _split_layers(depth, layer_split_depth)
        upper_mask_n2, lower_mask_n2 = _split_layers(depth_mid, layer_split_depth)

        sigma0_upper_mean = float(np.nanmean(sigma0[upper_mask_depth])) if upper_mask_depth.any() else np.nan
        sigma0_lower_mean = float(np.nanmean(sigma0[lower_mask_depth])) if lower_mask_depth.any() else np.nan
        if np.isfinite(sigma0_upper_mean) and np.isfinite(sigma0_lower_mean):
            sigma0_upper_lower_diff = sigma0_lower_mean - sigma0_upper_mean
        else:
            sigma0_upper_lower_diff = np.nan

        pea_upper = (
            _compute_pea(depth[upper_mask_depth], sigma0[upper_mask_depth], lat_mean)
            if upper_mask_depth.any()
            else np.nan
        )
        pea_lower = (
            _compute_pea(depth[lower_mask_depth], sigma0[lower_mask_depth], lat_mean)
            if lower_mask_depth.any()
            else np.nan
        )

        n2_mean_upper = float(np.nanmean(n2[upper_mask_n2])) if upper_mask_n2.any() else np.nan
        n2_mean_lower = float(np.nanmean(n2[lower_mask_n2])) if lower_mask_n2.any() else np.nan
        n2_max_upper = float(np.nanmax(n2[upper_mask_n2])) if upper_mask_n2.any() else np.nan
        n2_max_lower = float(np.nanmax(n2[lower_mask_n2])) if lower_mask_n2.any() else np.nan

        mld_extra = {
            "pycnocline_depth_m": pycnocline_depth,
            "layer_split_depth_m": layer_split_depth,
            "layer_split_method": layer_split_method,
            "sigma0_upper_mean_kg_m3": sigma0_upper_mean,
            "sigma0_lower_mean_kg_m3": sigma0_lower_mean,
            "sigma0_upper_lower_diff_kg_m3": sigma0_upper_lower_diff,
            "pea_J_m3": pea,
            "pea_upper_J_m3": pea_upper,
            "pea_lower_J_m3": pea_lower,
            "n2_mean_s-2": n2_mean,
            "n2_max_s-2": n2_max,
            "n2_mean_upper_s-2": n2_mean_upper,
            "n2_mean_lower_s-2": n2_mean_lower,
            "n2_max_upper_s-2": n2_max_upper,
            "n2_max_lower_s-2": n2_max_lower,
        }
        for entry in mld_entries:
            entry.update(mld_extra)
        mld_rows.extend(mld_entries)

        summary = {
            **key_map,
            "profile_label": profile_label,
            "profile_date": profile_date,
            "n_depths": int(len(depth)),
            "depth_min_m": float(np.nanmin(depth)),
            "depth_max_m": float(np.nanmax(depth)),
            "pycnocline_depth_m": pycnocline_depth,
            "layer_split_depth_m": layer_split_depth,
            "layer_split_method": layer_split_method,
            "sigma0_upper_mean_kg_m3": sigma0_upper_mean,
            "sigma0_lower_mean_kg_m3": sigma0_lower_mean,
            "sigma0_upper_lower_diff_kg_m3": sigma0_upper_lower_diff,
            "mld_reference_depth_m": float(args.mld_reference_depth),
            "mld_delta_rho_fixed_list_kg_m3": ",".join([str(t) for t in fixed_thresholds]),
            "mld_delta_rho_adaptive_kg_m3": adaptive_delta,
            "mld_rho_ref_kg_m3": mld_rho_ref,
            "adaptive_layer_max_depth_m": float(args.adaptive_layer_max_depth),
            "adaptive_percentile": float(args.adaptive_percentile),
            "pea_J_m3": pea,
            "pea_upper_J_m3": pea_upper,
            "pea_lower_J_m3": pea_lower,
            "n2_mean_s-2": n2_mean,
            "n2_max_s-2": n2_max,
            "n2_max_depth_m": n2_max_depth,
            "n2_mean_upper_s-2": n2_mean_upper,
            "n2_mean_lower_s-2": n2_mean_lower,
            "n2_max_upper_s-2": n2_max_upper,
            "n2_max_lower_s-2": n2_max_lower,
        }
        summary.update(mld_summary)
        summary_rows.append(summary)

    density_df = pd.DataFrame(density_rows)
    n2_df = pd.DataFrame(n2_rows)
    summary_df = pd.DataFrame(summary_rows)
    mld_long_df = pd.DataFrame(mld_rows)

    print("\n[4/9] Preparing feature traces...")
    feature_trace_df = None
    if plot_feature_cols:
        if not np.isfinite(layer_split_depth):
            print("  [i] Skipping feature traces (no valid layer split depth).")
        else:
            feature_trace_df = df[profile_cols + [args.depth_col] + plot_feature_cols].copy()
            for col in plot_feature_cols + [args.depth_col]:
                feature_trace_df[col] = pd.to_numeric(feature_trace_df[col], errors="coerce")
            depth_vals = feature_trace_df[args.depth_col]
            upper_df = feature_trace_df[depth_vals <= layer_split_depth]
            lower_df = feature_trace_df[depth_vals > layer_split_depth]
            print(
                "  [i] Feature trace split counts: "
                f"upper_rows={len(upper_df)} lower_rows={len(lower_df)}"
            )
            upper_means = upper_df.groupby(profile_cols, sort=False)[plot_feature_cols].mean()
            lower_means = lower_df.groupby(profile_cols, sort=False)[plot_feature_cols].mean()
            feature_trace_df = upper_means.add_suffix("_upper").join(
                lower_means.add_suffix("_lower"), how="outer"
            ).reset_index()
            feature_trace_df["profile_label"] = (
                feature_trace_df[profile_cols].astype(str).agg("|".join, axis=1)
            )
            if feature_trace_df.empty:
                print("  [i] Feature trace table is empty after aggregation.")
            else:
                nonnull_counts = feature_trace_df.notna().sum().to_dict()
                total_profiles = len(feature_trace_df)
                print(f"  [i] Feature trace table rows: {total_profiles}")
                for col in plot_feature_cols[:5]:
                    upper_col = f"{col}_upper"
                    lower_col = f"{col}_lower"
                    upper_n = nonnull_counts.get(upper_col, 0)
                    lower_n = nonnull_counts.get(lower_col, 0)
                    print(f"  [i] {col}: upper_nonnull={upper_n} lower_nonnull={lower_n}")

    print("\n[5/9] Classifying PEA...")
    if args.pea_class_mode != "none":
        summary_df = _classify_pea(
            summary_df=summary_df,
            mode=args.pea_class_mode,
            low_q=args.pea_low_quantile,
            high_q=args.pea_high_quantile,
            month_col=args.month_col,
            date_col="profile_date" if args.date_col else None,
        )
        if not mld_long_df.empty:
            mld_long_df = mld_long_df.merge(
                summary_df[
                    [
                        "profile_label",
                        "pea_class",
                        "pea_threshold_low",
                        "pea_threshold_high",
                    ]
                ],
                on="profile_label",
                how="left",
            )

    print("\n[6/9] Classifying deep intrusion...")
    if args.deep_intrusion_mode != "none":
        summary_df = _classify_deep_intrusion(
            summary_df=summary_df,
            mode=args.deep_intrusion_mode,
            quantile=args.deep_intrusion_quantile,
            value_col="sigma0_lower_mean_kg_m3",
        )
        if not mld_long_df.empty:
            mld_long_df = mld_long_df.merge(
                summary_df[
                    [
                        "profile_label",
                        "deep_intrusion_class",
                        "deep_intrusion_threshold",
                        "deep_intrusion_score",
                    ]
                ],
                on="profile_label",
                how="left",
            )

    print("\n[7/9] Writing outputs...")
    density_df.to_csv(out_dir / "stratification_density_profiles.tsv", sep="\t", index=False)
    n2_df.to_csv(out_dir / "stratification_n2_profiles.tsv", sep="\t", index=False)
    summary_df.to_csv(out_dir / "stratification_summary.tsv", sep="\t", index=False)
    mld_long_df.to_csv(out_dir / "stratification_mld_timeseries.tsv", sep="\t", index=False)
    if feature_trace_df is not None and not feature_trace_df.empty:
        feature_trace_df.to_csv(out_dir / "stratification_feature_traces.tsv", sep="\t", index=False)

    if not args.no_plots:
        print("\n[8/9] Plotting review figure...")
        _plot_review(
            summary_df=summary_df,
            mld_long_df=mld_long_df,
            feature_df=feature_trace_df,
            feature_cols=plot_feature_cols,
            out_path=out_dir / "stratification_review.pdf",
            date_col=args.date_col,
            layer_split_depth=layer_split_depth,
        )

    print("\n[9/9] Done.")
    print(f"Outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
