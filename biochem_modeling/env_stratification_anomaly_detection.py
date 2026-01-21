# Path: stratification_timeseries_anomaly.py
# Full replacement script (single-input biochem/metadata table)
# Outputs (exact same filenames as before):
#   - stratification_timeseries.tsv
#   - annual_extremes.tsv
#   - stratification_monthly_profile.pdf

#!/usr/bin/env python3
"""
stratification_timeseries_anomaly.py

Single-table stratification time-series analysis using a biochem/physical table
that also contains all metadata required for plotting.

Produces ONLY:
  - stratification_timeseries.tsv
  - annual_extremes.tsv
  - stratification_monthly_profile.pdf

Feature selection:
  - If --features is provided: use that comma-separated list.
  - Else: use all columns AFTER --features-after-col (default: Depth_anchored).
  - Then apply coverage filtering with COVERAGE_THRESHOLD (same behavior as before).

Example:
  python stratification_timeseries_anomaly.py \
    --input biochem_table.tsv \
    --sample-id-col cruise_year_month_depth \
    --date-col date \
    --month-col Month \
    --year-col Year \
    --depth-col Depth \
    --output-dir strat_monthly_out
"""

import argparse
import warnings
from pathlib import Path
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
    }
)
sns.set_style("white")

COVERAGE_THRESHOLD = 0.51


# ============================================================================
# Helpers
# ============================================================================

def _safe_euclidean(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return None
    diff = a[mask] - b[mask]
    return np.linalg.norm(diff)


def _split_features_arg(s: str) -> List[str]:
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


# ============================================================================
# Feature selection
# ============================================================================

def select_feature_columns(
    df: pd.DataFrame,
    features_csv: str | None,
    features_after_col: str,
    coverage_threshold: float,
) -> List[str]:
    """
    Select candidate feature columns, then filter by coverage threshold.
    """
    if features_csv:
        candidates = _split_features_arg(features_csv)
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            raise ValueError(f"--features includes columns not found in input: {missing}")
    else:
        if features_after_col not in df.columns:
            raise ValueError(
                f"--features-after-col '{features_after_col}' not found in input columns."
            )
        idx = list(df.columns).index(features_after_col)
        candidates = list(df.columns)[idx + 1 :]
        if not candidates:
            raise ValueError(
                f"No columns found after --features-after-col '{features_after_col}'."
            )

    # Drop ASV-like cols if present (keeps prior behavior)
    candidates = [c for c in candidates if not str(c).startswith("ASV")]

    # Coverage filter (same concept as before: fraction of rows with non-null)
    cov = df[candidates].notna().sum() / max(len(df), 1)
    keep = [c for c in candidates if cov.get(c, 0.0) >= coverage_threshold]
    drop = [c for c in candidates if c not in keep]

    print(f"  [i] Keeping {len(keep)} feature columns for stratification: {keep}")
    if drop:
        print(f"  [i] Dropping undersampled columns (<{coverage_threshold*100:.0f}% coverage): {drop}")
    if not keep:
        raise ValueError("No feature columns meet the coverage requirement.")
    return keep


# ============================================================================
# Core computations
# ============================================================================

def calculate_stratification_score_timeseries(
    integrated_data: pd.DataFrame,
    metadata: pd.DataFrame,
    cruise_col: str,
    date_col: str,
    depth_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    For each unique date:
      - compute mean pairwise distance between depth centroids (features)
    """
    print("  [i] Calculating stratification score time series...")

    results = []
    unique_dates = sorted(metadata[date_col].unique())

    for date in unique_dates:
        date_mask = metadata[date_col] == date
        date_meta = metadata.loc[date_mask]
        date_data = integrated_data.loc[date_mask, feature_cols]

        cruise = date_meta[cruise_col].unique()

        if date_data.empty or date_meta.empty:
            continue

        depths = sorted(date_meta[depth_col].unique())
        depth_centroids = {}

        for depth in depths:
            depth_mask = date_meta[depth_col] == depth
            depth_samples = date_data.loc[depth_mask]
            if depth_samples.empty:
                continue
            centroid = depth_samples.mean(axis=0, skipna=True)
            if centroid.notna().sum() == 0:
                continue
            depth_centroids[depth] = centroid.values

        distances = []
        for depth1, depth2 in combinations(depth_centroids.keys(), 2):
            dist = _safe_euclidean(depth_centroids[depth1], depth_centroids[depth2])
            if dist is not None:
                distances.append(dist)

        if not distances:
            continue

        mean_dist = np.mean(distances)
        total_cells = date_data.size
        non_na = np.isfinite(date_data.values).sum()
        coverage = non_na / total_cells if total_cells > 0 else 0.0

        results.append(
            {
                'Cruise': cruise[0] if len(cruise) == 1 else "multiple",
                "date": date,
                "stratification_score": mean_dist,
                "n_depths": len(depths),
                "depths_present": ",".join(map(str, depths)),
                "n_samples": len(date_data),
                "coverage": coverage,
            }
        )

    timeseries_df = pd.DataFrame(results)
    print(f"      Computed {len(timeseries_df)} time points")
    return timeseries_df


def normalize_to_centered_scale(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize stratification scores to [-1, 1] centered at 0 (median at 0).
    """
    print("  [i] Normalizing to centered scale...")

    min_score = timeseries_df["stratification_score"].min()
    max_score = timeseries_df["stratification_score"].max()
    median_score = timeseries_df["stratification_score"].median()

    timeseries_df = timeseries_df.copy()
    normalized_scores = []
    for score in timeseries_df["stratification_score"]:
        if score < median_score:
            norm = -1 + (score - min_score) / (median_score - min_score)
        else:
            norm = (score - median_score) / (max_score - median_score)
        normalized_scores.append(norm)

    timeseries_df["normalized_score"] = normalized_scores
    print(
        f"      Normalized: min={min(normalized_scores):.3f}, median=0, max={max(normalized_scores):.3f}"
    )
    return timeseries_df


def detect_anomalies_consensus(
    timeseries_df: pd.DataFrame,
    consensus_threshold: int = 2,
    date_col: str = "date",
    window_months: int = 12,
    min_points_in_window: int = 5,
) -> pd.DataFrame:
    """
    Detect anomalies using a sliding N-month window around each point.

    Requires:
      - timeseries_df['normalized_score']
      - timeseries_df[date_col] parseable as datetime

    For each point i:
      - Build a local window of points with dates in [date_i - window_months, date_i + window_months]
      - If window has < min_points_in_window points (including i): no anomaly (n_votes=0)
      - Run 4 methods on the WINDOW distribution, then evaluate whether point i is an outlier
        relative to that window.
      - Output schema preserved:
          is_anomaly (bool), anomaly_type (str), n_votes (int)

    Notes:
      - Z-score + IQR are computed from the window stats.
      - IsolationForest + LOF are fit on the window, then we use their predictions for point i.
      - This yields a *local* notion of anomaly (good for regime shifts / nonstationarity).
    """
    print(
        f"  [i] Detecting anomalies (±{window_months} month sliding window, consensus ≥{consensus_threshold})..."
    )

    if "normalized_score" not in timeseries_df.columns:
        raise ValueError("timeseries_df missing required column: normalized_score")
    if date_col not in timeseries_df.columns:
        raise ValueError(f"timeseries_df missing required date column: {date_col}")

    out = timeseries_df.copy()

    # Parse dates and keep original order stable
    dates = pd.to_datetime(out[date_col], errors="coerce")
    scores = pd.to_numeric(out["normalized_score"], errors="coerce").to_numpy(dtype=float)

    out["is_anomaly"] = False
    out["anomaly_type"] = "normal"
    out["n_votes"] = 0

    finite = np.isfinite(scores) & dates.notna().to_numpy()

    if finite.sum() < 3:
        return out

    # Work in date order for predictable windows
    order = np.argsort(dates.to_numpy())
    dates_s = dates.to_numpy()[order]
    scores_s = scores[order]
    finite_s = finite[order]

    # Precompute month window as a DateOffset
    halfwin = pd.DateOffset(months=int(window_months))

    # Map back to original row indices
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))

    for pos in range(len(scores_s)):
        if not finite_s[pos]:
            continue

        center_date = pd.Timestamp(dates_s[pos])

        left = center_date - halfwin
        right = center_date + halfwin

        in_window = (dates_s >= np.datetime64(left)) & (dates_s <= np.datetime64(right)) & finite_s
        w_idx = np.where(in_window)[0]
        n = len(w_idx)

        if n < int(min_points_in_window):
            continue

        y = scores_s[w_idx]
        # Identify where the center point sits inside the window
        # (pos is in sorted space)
        center_in_window = np.where(w_idx == pos)[0]
        if center_in_window.size != 1:
            # Shouldn't happen, but be defensive
            continue
        j = int(center_in_window[0])
        x0 = float(y[j])

        votes = 0

        # Method 1: Z-score (window)
        mu = float(np.nanmean(y))
        sd = float(np.nanstd(y, ddof=0))
        if sd > 0:
            z = abs((x0 - mu) / sd)
            votes += int(z > 1.5)

        # Method 2: IQR (window)
        Q1 = float(np.nanquantile(y, 0.25))
        Q3 = float(np.nanquantile(y, 0.75))
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            votes += int((x0 < lower) or (x0 > upper))

        # Method 3: Isolation Forest (fit on window, predict all window points)
        # (contamination is still a fixed fraction; keep as-is for now)
        try:
            iso = IsolationForest(contamination=0.15, random_state=42)
            iso_pred = iso.fit_predict(y.reshape(-1, 1))
            votes += int(iso_pred[j] == -1)
        except Exception:
            pass

        # Method 4: LOF (fit on window, predict all window points)
        try:
            if n >= 3:
                n_neighbors = min(10, n - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.15)
                lof_pred = lof.fit_predict(y.reshape(-1, 1))
                votes += int(lof_pred[j] == -1)
        except Exception:
            pass

        is_anom = votes >= int(consensus_threshold)

        # Write to *original* row index
        orig_i = int(order[pos])
        out.at[orig_i, "n_votes"] = int(votes)
        out.at[orig_i, "is_anomaly"] = bool(is_anom)

        if is_anom:
            out.at[orig_i, "anomaly_type"] = "high_stratification" if x0 > 0 else "mixing_event"
        else:
            out.at[orig_i, "anomaly_type"] = "normal"

    n_anomalies = int(out["is_anomaly"].sum())
    n_mixing = int((out["anomaly_type"] == "mixing_event").sum())
    n_high_strat = int((out["anomaly_type"] == "high_stratification").sum())

    print(f"      Total anomalies: {n_anomalies}/{len(out)} time points")
    print(f"      Mixing events: {n_mixing}")
    print(f"      High stratification: {n_high_strat}")

    return out


def identify_annual_extremes(
    timeseries_df: pd.DataFrame,
    metadata: pd.DataFrame,
    year_col: str,
) -> pd.DataFrame:
    """
    Identify max stratification and max mixing events per year.
    Kept identical to the earlier (quirky) behavior to preserve outputs.
    """
    print("  [i] Identifying annual extremes...")

    timeseries_df = timeseries_df.copy()

    # This mapping logic is intentionally preserved
    date_to_year = metadata.set_index(metadata.columns[0])[year_col].to_dict()

    years = []
    for date in timeseries_df["date"]:
        year_found = False
        for sample, meta_year in date_to_year.items():
            sample_meta = metadata[metadata.index == sample]
            if len(sample_meta) > 0:
                sample_date = sample_meta[sample_meta.columns[0]].iloc[0]
                if sample_date == date:
                    years.append(meta_year)
                    year_found = True
                    break

        if not year_found:
            try:
                if isinstance(date, str):
                    year = int(date.split("-")[0]) if "-" in date else int(date[:4])
                else:
                    year = date.year
                years.append(year)
            except Exception:
                years.append(None)

    timeseries_df["year"] = years

    extremes = []
    for year in sorted(timeseries_df["year"].dropna().unique()):
        year_data = timeseries_df[timeseries_df["year"] == year]

        max_strat_idx = year_data["normalized_score"].idxmax()
        max_strat_row = year_data.loc[max_strat_idx]
        extremes.append(
            {
                "date": max_strat_row["date"],
                "year": year,
                "normalized_score": max_strat_row["normalized_score"],
                "extreme_type": "max_stratification",
            }
        )

        min_mix_idx = year_data["normalized_score"].idxmin()
        min_mix_row = year_data.loc[min_mix_idx]
        extremes.append(
            {
                "date": min_mix_row["date"],
                "year": year,
                "normalized_score": min_mix_row["normalized_score"],
                "extreme_type": "max_mixing",
            }
        )

    extremes_df = pd.DataFrame(extremes)
    print(
        f"      Found {len(extremes_df)} extremes across {len(extremes_df['year'].unique())} years"
    )
    return extremes_df


# ============================================================================
# PEA comparison plot
# ============================================================================

def _load_pea_timeseries(pea_path: Path, date_col: str) -> pd.DataFrame:
    pea_df = pd.read_csv(pea_path, sep="\t")
    if date_col not in pea_df.columns:
        raise ValueError(f"PEA metrics file missing date column '{date_col}'")
    pea_df = pea_df.copy()
    pea_df["date"] = pd.to_datetime(pea_df[date_col], errors="coerce")
    pea_df = pea_df.dropna(subset=["date"])

    for col in ["pea_J_m3", "pea_upper_J_m3", "pea_lower_J_m3"]:
        if col in pea_df.columns:
            pea_df[col] = pd.to_numeric(pea_df[col], errors="coerce")

    keep_cols = [c for c in ["pea_J_m3", "pea_upper_J_m3", "pea_lower_J_m3"] if c in pea_df.columns]
    if not keep_cols:
        raise ValueError("PEA metrics file missing PEA columns (pea_J_m3/pea_upper_J_m3/pea_lower_J_m3)")

    pea_ts = pea_df.groupby("date", sort=True)[keep_cols].mean().reset_index()
    return pea_ts


def plot_stratification_vs_pea_timeseries(
    timeseries_df: pd.DataFrame,
    pea_df: pd.DataFrame,
    output_path: Path,
) -> None:
    print("  [i] Creating stratification vs PEA time-series plot...")
    if timeseries_df.empty or pea_df.empty:
        print("  [i] Skipping PEA comparison plot (no data).")
        return

    strat_df = timeseries_df.copy()
    strat_df["date"] = pd.to_datetime(strat_df["date"], errors="coerce")
    strat_df = strat_df.dropna(subset=["date"]).sort_values("date")
    pea_df = pea_df.sort_values("date")

    def _plot_with_gaps(ax, x, y, color):
        sub = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
        if sub.empty:
            return
        xs = sub["x"].to_numpy()
        ys = sub["y"].to_numpy()
        if len(xs) == 1:
            ax.plot(xs, ys, color=color, label="_nolegend_")
            return

        deltas = np.diff(xs)
        median_delta = np.median(deltas)
        if not np.isfinite(median_delta) or median_delta == 0:
            ax.plot(xs, ys, color=color, label="_nolegend_")
            return

        gap_thresh = median_delta * 2
        seg_start = 0
        for i in range(1, len(xs)):
            if (xs[i] - xs[i - 1]) > gap_thresh:
                ax.plot(xs[seg_start:i], ys[seg_start:i], color=color, label="_nolegend_")
                ax.plot(
                    xs[i - 1:i + 1],
                    ys[i - 1:i + 1],
                    color=color,
                    linestyle=":",
                    label="_nolegend_",
                )
                seg_start = i
        ax.plot(xs[seg_start:], ys[seg_start:], color=color, label="_nolegend_")

    fig, ax_left = plt.subplots(1, 1, figsize=(22, 6))
    ax_right = ax_left.twinx()

    ax_left.scatter(
        strat_df["date"],
        strat_df["stratification_score"],
        color="black",
        s=30,
        label="depth_centroid_distance",
    )
    _plot_with_gaps(
        ax_left,
        strat_df["date"],
        strat_df["stratification_score"],
        color="black",
    )
    ax_left.set_ylabel("Depth centroid distance (stability index)")
    ax_left.grid(axis="y", linestyle="--", alpha=0.35)

    if "pea_J_m3" in pea_df.columns:
        ax_right.scatter(pea_df["date"], pea_df["pea_J_m3"], color="tab:green", s=30, label="PEA total")
        _plot_with_gaps(ax_right, pea_df["date"], pea_df["pea_J_m3"], color="tab:green")
    if "pea_upper_J_m3" in pea_df.columns:
        ax_right.scatter(pea_df["date"], pea_df["pea_upper_J_m3"], color="tab:blue", s=30, label="PEA upper")
        _plot_with_gaps(ax_right, pea_df["date"], pea_df["pea_upper_J_m3"], color="tab:blue")
    if "pea_lower_J_m3" in pea_df.columns:
        ax_right.scatter(pea_df["date"], pea_df["pea_lower_J_m3"], color="tab:orange", s=30, label="PEA lower")
        _plot_with_gaps(ax_right, pea_df["date"], pea_df["pea_lower_J_m3"], color="tab:orange")
    ax_right.set_ylabel("PEA (J/m3)")

    ax_left.set_title("Stability index vs PEA")
    ax_left.set_xlabel("Date")

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [✓] Saved stratification vs PEA time-series plot")


# ============================================================================
# Plot (monthly profile)
# ============================================================================

def plot_stratification_monthly_profile(
    timeseries_df: pd.DataFrame,
    extremes_df: pd.DataFrame,
    metadata: pd.DataFrame,
    date_col: str,
    month_col: str,
    year_col: str,
    output_path: Path,
) -> None:
    """
    Monthly profile plot (kept byte-for-byte logic from prior pruned version).
    """
    print("  [i] Creating enriched monthly stratification profile...")
    plot_df = timeseries_df.copy()

    meta_reset = metadata.reset_index()
    if date_col not in meta_reset.columns:
        raise ValueError(f"Metadata missing date column '{date_col}' for monthly profile")
    if month_col not in meta_reset.columns or year_col not in meta_reset.columns:
        raise ValueError("Metadata must supply month/year columns for monthly profile")

    date_to_month = meta_reset.groupby(date_col)[month_col].first().to_dict()
    date_to_year = meta_reset.groupby(date_col)[year_col].first().to_dict()

    plot_df["month"] = plot_df["date"].map(date_to_month)
    plot_df["year"] = plot_df["date"].map(date_to_year)
    plot_df = plot_df.dropna(subset=["month", "year"]).copy()
    plot_df["coverage"] = plot_df["coverage"].fillna(0.0)
    plot_df["month"] = plot_df["month"].astype(int)
    plot_df["year"] = plot_df["year"].astype(int)
    
    pivot = (
        plot_df.groupby(["month", "year"])["normalized_score"]
        .mean()
        .unstack(level=1)
        .reindex(range(1, 13))
    )
    pivot = pivot.apply(pd.to_numeric, errors="coerce")
    coverage_pivot = (
        plot_df.groupby(["month", "year"])["coverage"]
        .mean()
        .unstack(level=1)
        .reindex(range(1, 13))
        .apply(pd.to_numeric, errors="coerce")
    )

    plot_range = np.linspace(0.8, 12.2, 300)

    month_means = pivot.mean(axis=1, skipna=True)
    mean_values = (
        month_means.interpolate(limit_direction="both")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )
    mean_smoothed = gaussian_filter1d(mean_values.values, sigma=1.1)
    mean_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed)

    month_std = pivot.std(axis=1, ddof=0).fillna(0.0)
    std_smoothed = gaussian_filter1d(month_std.values, sigma=1.1)
    lower_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed - std_smoothed)
    upper_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed + std_smoothed)

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.fill_between(plot_range, lower_curve, upper_curve, color="lightgrey", alpha=0.6, zorder=1)
    ax.plot(plot_range, mean_curve, color="black", linewidth=3, zorder=2)
    ax.axhline(0, color="black", linewidth=2, alpha=0.6, zorder=4)

    extremes_df = extremes_df.copy()
    extremes_df["month"] = extremes_df["date"].map(date_to_month)
    strat_points = extremes_df[extremes_df["extreme_type"] == "max_stratification"]
    mix_points = extremes_df[extremes_df["extreme_type"] == "max_mixing"]

    coverage_counts = {}
    for year in coverage_pivot.columns:
        coverage_counts[year] = (coverage_pivot[year] >= COVERAGE_THRESHOLD).sum()
    min_months_needed = 7
    eligible_years = {year for year, count in coverage_counts.items() if count >= min_months_needed}

    for _, row in strat_points.iterrows():
        if pd.isna(row["month"]) or row.get("year", None) not in eligible_years:
            continue
        month_val = row["month"]
        ax.scatter(
            month_val,
            row["normalized_score"],
            marker="^",
            color="black",
            s=200,
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )

    for _, row in mix_points.iterrows():
        if pd.isna(row["month"]) or row.get("year", None) not in eligible_years:
            continue
        month_val = row["month"]
        ax.scatter(
            month_val,
            row["normalized_score"],
            marker="v",
            color="black",
            s=200,
            edgecolor="black",
            linewidth=2.0,
            zorder=5,
        )

    ax.set_xlim(0.9, 12.1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=12,
    )

    score_min = timeseries_df["normalized_score"].min()
    score_max = timeseries_df["normalized_score"].max()
    y_pad = (score_max - score_min) * 0.1 if score_max != score_min else 0.5
    ax.set_ylim(score_min - y_pad, score_max + y_pad)

    ax.set_xlabel("Month", fontsize=14, fontweight="bold")
    ax.set_ylabel(
        "Stratification Index\n(−1=Mixed, 0=Intermediate, +1=Stratified)",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(-0.05, 0.5, "", transform=ax.transAxes, fontsize=12, fontweight="bold", rotation=90, va="center")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(
        ["Max\nMixed", "Mixed", "Intermediate", "Stratified", "Max\nStratified"],
        fontsize=11,
        fontweight="bold",
    )
    ax.tick_params(axis="y", which="major", pad=8)
    ax.set_title("Monthly Stratification Profile", fontsize=16, fontweight="bold")

    point_df = pivot.stack().reset_index(name="normalized_score")
    coverage_df = coverage_pivot.stack().reset_index(name="coverage")
    point_df = point_df.merge(coverage_df, on=["month", "year"], how="left")

    for _, row in point_df.iterrows():
        if pd.isna(row["normalized_score"]) or pd.isna(row["coverage"]):
            continue
        if row["coverage"] < COVERAGE_THRESHOLD:
            continue
        color = "black" #"royalblue" if row["normalized_score"] >= 0 else "darkorange"
        ax.scatter(
            row["month"],
            row["normalized_score"],
            color=color,
            s=48,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=5,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("  [✓] Saved monthly stratification profile")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Stratification monthly profile from a single biochem/metadata TSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", type=Path, required=True, help="Single input TSV containing metadata + biochem features.")
    ap.add_argument("--sep", default=",", help="Delimiter (default: comma). Use '\t' for TSV.")
    ap.add_argument("--sample-id-col", default="cruise_year_month_depth", help="Unique sample ID column.")
    ap.add_argument("--cruise-col", default="Cruise", help="Cruise column name.")
    ap.add_argument("--date-col", default="date", help="Date column name.")
    ap.add_argument("--month-col", default="Month", help="Month column name.")
    ap.add_argument("--year-col", default="Year", help="Year column name.")
    ap.add_argument("--depth-col", default="Depth_anchored", help="Depth column name.")
    ap.add_argument("--depth-min", type=float, default=None, help="Minimum depth to include.")
    ap.add_argument("--depth-max", type=float, default=None, help="Maximum depth to include.")
    ap.add_argument(
        "--pea-metrics",
        type=Path,
        default=None,
        help="Path to stratification_summary.tsv from env_stratification_metrics.py (for PEA comparison plot).",
    )
    ap.add_argument(
        "--pea-date-col",
        default="profile_date",
        help="Date column in PEA metrics file (default profile_date).",
    )

    ap.add_argument(
        "--features",
        default=None,
        help="Comma-separated list of feature columns to use (overrides --features-after-col).",
    )
    ap.add_argument(
        "--features-after-col",
        default="Depth_anchored",
        help="If --features is not provided, use all columns after this column as features.",
    )

    ap.add_argument("--consensus-threshold", type=int, default=1, help="Minimum methods for anomaly consensus [1].")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STRATIFICATION MONTHLY PROFILE (SINGLE INPUT)")
    print("=" * 70)

    print("\n[1/5] Loading input table...")
    df = pd.read_csv(args.input, sep=args.sep)

    # Validate required columns
    required_cols = [args.sample_id_col, args.date_col, args.month_col, args.year_col, args.depth_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input missing required columns: {missing_required}")

    # Select features
    feature_cols = select_feature_columns(
        df=df,
        features_csv=args.features,
        features_after_col=args.features_after_col,
        coverage_threshold=COVERAGE_THRESHOLD,
    )

    # Build "integrated_data" and "metadata" to match existing functions
    print("\n[2/5] Preparing matrices...")
    df_dedup = df.drop_duplicates(subset=[args.sample_id_col]).copy()

    # Metadata indexed by sample id
    metadata = df_dedup.set_index(args.sample_id_col)

    # Feature matrix indexed by sample id
    integrated_data = df_dedup.set_index(args.sample_id_col)[feature_cols].copy()
    integrated_data = integrated_data.apply(pd.to_numeric, errors="coerce")

    # Align indices (defensive)
    common = integrated_data.index.intersection(metadata.index)
    integrated_data = integrated_data.loc[common]
    metadata = metadata.loc[common]

    # Optional depth filtering
    if args.depth_min is not None or args.depth_max is not None:
        depth_vals = pd.to_numeric(metadata[args.depth_col], errors="coerce")
        depth_mask = np.isfinite(depth_vals)
        if args.depth_min is not None:
            depth_mask &= depth_vals >= args.depth_min
        if args.depth_max is not None:
            depth_mask &= depth_vals <= args.depth_max
        integrated_data = integrated_data.loc[depth_mask]
        metadata = metadata.loc[depth_mask]
        print(f"  [i] Depth filter kept {len(metadata)} samples")

    print(f"  Samples: {len(common)}")
    print(f"  Features used: {len(feature_cols)}")
    print(f"  Time points: {len(metadata[args.date_col].unique())}")

    print("\n[3/5] Calculating stratification time series...")
    timeseries_df = calculate_stratification_score_timeseries(
        integrated_data=integrated_data,
        metadata=metadata,
        cruise_col=args.cruise_col,
        date_col=args.date_col,
        depth_col=args.depth_col,
        feature_cols=feature_cols,
    )

    print("\n[4/5] Normalizing + anomaly detection...")
    timeseries_df = normalize_to_centered_scale(timeseries_df)
    timeseries_df = detect_anomalies_consensus(timeseries_df, consensus_threshold=args.consensus_threshold)

    # Global thresholds based on anomaly values (apply to all rows)
    mix_scores = timeseries_df.loc[timeseries_df["anomaly_type"] == "mixing_event", "normalized_score"]
    strat_scores = timeseries_df.loc[timeseries_df["anomaly_type"] == "high_stratification", "normalized_score"]

    mix_thresh = float(mix_scores.max()) if not mix_scores.empty else np.nan
    strat_thresh = float(strat_scores.min()) if not strat_scores.empty else np.nan

    def _global_event_type(score: float) -> str:
        if np.isfinite(strat_thresh) and score >= strat_thresh:
            return "high_stratification"
        if np.isfinite(mix_thresh) and score <= mix_thresh:
            return "mixing_event"
        return "normal"

    timeseries_df = timeseries_df.copy()
    timeseries_df["global_event_type"] = timeseries_df["normalized_score"].apply(_global_event_type)
    timeseries_df["global_mixing_threshold"] = mix_thresh
    timeseries_df["global_strat_threshold"] = strat_thresh

    print("\n[5/5] Annual extremes + monthly profile...")
    extremes_df = identify_annual_extremes(timeseries_df, metadata, args.year_col)

    # Write the 2 TSVs (rename columns for output)
    rename_map = {
        "stratification_score": "depth_centroid_distance",
        "normalized_score": "normalized_depth_centroid_distance",
    }
    timeseries_out = timeseries_df.rename(columns=rename_map)
    extremes_out = extremes_df.rename(columns=rename_map)
    timeseries_out.to_csv(out_dir / "stratification_timeseries.tsv", sep="\t", index=False)
    extremes_out.to_csv(out_dir / "annual_extremes.tsv", sep="\t", index=False)

    # Write the PDF
    plot_stratification_monthly_profile(
        timeseries_df=timeseries_df,
        extremes_df=extremes_df,
        metadata=metadata,
        date_col=args.date_col,
        month_col=args.month_col,
        year_col=args.year_col,
        output_path=out_dir / "stratification_monthly_profile.pdf",
    )

    # Optional PEA comparison plot
    if args.pea_metrics is not None:
        pea_ts = _load_pea_timeseries(args.pea_metrics, args.pea_date_col)
        plot_stratification_vs_pea_timeseries(
            timeseries_df=timeseries_df,
            pea_df=pea_ts,
            output_path=out_dir / "stratification_vs_pea_timeseries.pdf",
        )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
