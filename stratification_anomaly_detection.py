#!/usr/bin/env python3
"""
stratification_timeseries_anomaly.py

Continuous time-series analysis of stratification dynamics using integrated data.
Identifies flux between mixed and stratified states with anomaly detection.

Usage:
  python stratification_timeseries_anomaly.py \
    --integrated-data compartment_analysis/integrated_data.tsv \
    --metadata spark_combined_output/metadata/metadata_updated.tsv \
    --date-col Date \
    --month-col Month \
    --year-col Year \
    --depth-col Depth \
    --output-dir stratification_timeseries_analysis
"""

import argparse
import warnings
from pathlib import Path
from itertools import combinations
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, pdist
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})
sns.set_style("white")

SAMPLE_ID_COL = 'sampleid'
COVERAGE_THRESHOLD = 0.51
BIOCHEM_COLOR_MAP = {
    'Oxygen': 'black',
    'Nitrogen Oxides': "#E7298A",
    'Nitrate': "#1B9E77",
    'Nitrite': "#66A61E",
    'Nitrous Oxide': "#0C5196",
    'Ammonium': "#7570B3",
    'Hydrogen Sulfide': "#D95F02",
    'Methane': "violet"
}


# ============================================================================
# Core Analysis Functions
# ============================================================================

def calculate_stratification_score_timeseries(
    integrated_data: pd.DataFrame,
    metadata: pd.DataFrame,
    date_col: str,
    depth_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Calculate stratification score for each time point.
    
    For each unique date:
    - Calculate mean pairwise distance between depth centroids
    - Higher distance = more stratified
    
    Returns
    -------
    DataFrame with columns: date, stratification_score, n_depths, depths_present
    """
    print("  [i] Calculating stratification score time series...")
    
    results = []
    
    # Get unique dates
    unique_dates = sorted(metadata[date_col].unique())
    
    for date in unique_dates:
        date_mask = metadata[date_col] == date
        date_meta = metadata.loc[date_mask]
        date_data = integrated_data.loc[date_mask, feature_cols]

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

        results.append({
            'date': date,
            'stratification_score': mean_dist,
            'n_depths': len(depths),
            'depths_present': ','.join(map(str, depths)),
            'n_samples': len(date_data),
            'coverage': coverage,
        })
    
    timeseries_df = pd.DataFrame(results)
    
    print(f"      Computed {len(timeseries_df)} time points")
    
    return timeseries_df


def _dominant_category(values: pd.Series):
    """
    Determine the most frequent non-null value, breaking ties by first appearance.
    """
    values = values.dropna()
    if values.empty:
        return np.nan
    counts = values.value_counts()
    top_freq = counts.max()
    top_values = counts[counts == top_freq].index.tolist()
    for val in values:
        if val in top_values:
            return val
    return top_values[0]


def _safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_euclidean(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return None
    diff = a[mask] - b[mask]
    return np.linalg.norm(diff)


def integrate_biochem_only_samples(
    integrated_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    args,
    sample_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Append biogeochemistry-only samples (year/month/depth + features) to the
    integrated data matrix and metadata.
    """
    print("  [i] Augmenting with biochem-only data...")
    biochem_df = pd.read_csv(args.biochem_only_data, sep="\t")
    required = {
        args.biochem_cruise_col,
        args.biochem_year_col,
        args.biochem_month_col,
        args.biochem_depth_col,
    }
    if args.biochem_sample_id_col:
        required.add(args.biochem_sample_id_col)
    missing = [col for col in required if col not in biochem_df.columns]
    if missing:
        raise ValueError(f"Biochem-only data missing required columns: {missing}")

    feature_cols = [c for c in biochem_df.columns if c in integrated_data.columns]
    if not feature_cols:
        raise ValueError(
            "Biochem-only data does not share any feature columns with the integrated data."
        )

    sample_ids = []
    metadata_rows = []
    feature_rows = []

    for _, row in biochem_df.iterrows():
        year = _safe_int(row[args.biochem_year_col])
        month = _safe_int(row[args.biochem_month_col])
        if year is None or month is None:
            continue
        depth_val = row[args.biochem_depth_col]
        cruise_val = row[args.biochem_cruise_col]

        if args.biochem_sample_id_col:
            sample_id = str(row[args.biochem_sample_id_col])
        else:
            depth_str = (
                str(int(depth_val)) if pd.notna(depth_val) and float(depth_val).is_integer()
                else str(depth_val).replace(".", "p")
            )
            cruise_sanitized = (
                str(cruise_val).strip().replace(" ", "_") if pd.notna(cruise_val) else "cruise"
            )
            sample_id = f"{cruise_sanitized}_{year}_{month:02d}_{depth_str}_{args.biochem_sample_suffix}"

        sample_id = sample_id.strip()
        try:
            sample_date = pd.Timestamp(year=year, month=month, day=args.biochem_date_day)
        except ValueError:
            sample_date = pd.Timestamp(year=year, month=month, day=1)

        metadata_rows.append({
            sample_col: sample_id,
            args.date_col: sample_date.strftime("%Y-%m-%d"),
            args.year_col: year,
            args.month_col: month,
            args.depth_col: depth_val,
        })

        feature_rows.append({
            "_biochem_sampleid": sample_id,
            **{col: row[col] for col in feature_cols},
        })
        sample_ids.append(sample_id)

    if not sample_ids:
        print("  [i] No valid biochem-only samples found, skipping augmentation.")
        return integrated_data, metadata_df

    if not feature_rows:
        raise ValueError(
            "Biochem-only data does not share any feature columns with the integrated data."
        )

    features_df = pd.DataFrame(feature_rows).set_index("_biochem_sampleid")
    features_df = features_df.apply(pd.to_numeric, errors='coerce')

    aligned = pd.DataFrame(
        np.nan,
        index=sample_ids,
        columns=integrated_data.columns,
    )
    for col in feature_cols:
        if col in features_df.columns:
            aligned[col] = features_df[col]

    metadata_extra = pd.DataFrame(metadata_rows)
    if metadata_df.columns is not None:
        for col in metadata_df.columns:
            if col not in metadata_extra.columns:
                metadata_extra[col] = np.nan

    integrated_data = pd.concat([integrated_data, aligned], axis=0)
    metadata_df = pd.concat([metadata_df, metadata_extra], ignore_index=True, sort=False)

    print(f"  [i] Appended {len(sample_ids)} biochem-only samples")
    return integrated_data, metadata_df


def select_biochem_features(
    integrated_data: pd.DataFrame,
    metadata_df: pd.DataFrame,
    threshold: float,
) -> List[str]:
    """
    Identify biogeochemistry columns that pass the coverage threshold.
    """
    candidates = [
        col for col in integrated_data.columns
        if not col.startswith('ASV') and col in metadata_df.columns
    ]
    if not candidates:
        raise ValueError('No biochem columns could be matched between the data and metadata.')
    coverage = integrated_data[candidates].notna().sum() / len(integrated_data)
    keep = [col for col in candidates if coverage.get(col, 0.0) >= threshold]
    drop = [col for col in candidates if col not in keep]
    print(f"  [i] Keeping {len(keep)} biochem columns for stratification: {keep}")
    if drop:
        print(f"  [i] Dropping undersampled columns (<{threshold*100:.0f}% coverage): {drop}")
    if not keep:
        raise ValueError('No biochem columns meet the coverage requirement.')
    return keep


def select_metadata_biochem_columns(
    metadata_df: pd.DataFrame,
) -> List[str]:
    """
    Pick numeric metadata columns (non-ASV) for vertical plotting.
    """
    exclude = {
        SAMPLE_ID_COL, 'sample_code', 'longID', 'sampleID', 'plateID',
        'Cruise', 'Date', 'Year', 'Month', 'Day', 'Season', 'Depth',
        'Color', 'Month_Color', 'Month_Marker', 'Index'
    }
    candidates = [
        col for col in metadata_df.columns
        if col not in exclude and not col.startswith('ASV') and col in BIOCHEM_COLOR_MAP
    ]
    if not candidates:
        return []
    available = []
    for col in candidates:
        series = pd.to_numeric(metadata_df[col], errors='coerce')
        if series.notna().any():
            available.append(col)
    print(f"  [i] Vertical profile biochem variables (any data present): {available}")
    return available


def plot_biochem_vertical_profiles(
    metadata: pd.DataFrame,
    depth_col: str,
    variables: List[str],
    output_path: Path,
) -> None:
    """
    Recreate the assign_compartments vertical-depth plots for biochem vars.
    """
    available = [var for var in variables if var in metadata.columns]
    if not available:
        print("  [!] Skipping vertical biochem plot (no variables available).")
        return

    n_vars = len(available)
    ncols = 2 if n_vars > 1 else 1
    nrows = int(np.ceil(n_vars / ncols))
    fig_height = max(4.8 * nrows, 7)
    fig_width = max(5.0 * ncols, 5.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, var in zip(axes, available):
        data = metadata[[depth_col, var]].copy()
        data[var] = pd.to_numeric(data[var], errors='coerce')
        data = data.dropna()
        data = data[data[var] >= 0]
        if data.empty:
            ax.set_visible(False)
            continue
        color = BIOCHEM_COLOR_MAP.get(var, '#1f77b4')
        ax.scatter(data[var], data[depth_col], s=8, color='black', alpha=0.45, label='Samples')

        grouped = data.groupby(depth_col)[var]
        mean_vals = grouped.mean().sort_index()
        std_vals = grouped.std().fillna(0).reindex(mean_vals.index).fillna(0)
        depths_sorted = mean_vals.index.values
        ax.plot(mean_vals.values, depths_sorted, color=color, linewidth=2.5, label='Mean')
        ax.fill_betweenx(
            depths_sorted,
            mean_vals.values - std_vals.values,
            mean_vals.values + std_vals.values,
            color=color,
            alpha=0.2,
            label='±1 SD'
        )
        ax.set_title(var, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel(f"{var} (a.u.)", fontsize=11)
        overall_mean = data[var].mean()
        overall_std = data[var].std()
        if not np.isnan(overall_mean):
            ax.axvline(overall_mean, color=color, linestyle='--', linewidth=1.4, alpha=0.65)
            if not np.isnan(overall_std):
                ax.axvspan(
                    overall_mean - overall_std,
                    overall_mean + overall_std,
                    color=color,
                    alpha=0.08,
                )
        ax.invert_yaxis()
        ax.grid(alpha=0.2, linestyle='--')

    # Hide unused axes
    for ax in axes[len(available):]:
        ax.set_visible(False)

    # Ensure uniform inverted depth axis for all panels
    if not metadata[depth_col].dropna().empty:
        depth_min = metadata[depth_col].min()
        depth_max = metadata[depth_col].max()
        for ax in axes[:len(available)]:
            ax.set_ylim(depth_max, depth_min)

    fig.suptitle(
        '',
        fontsize=12,
        fontweight='bold',
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  [✓] Saved vertical biochem profiles: {output_path.name}")
    plt.close()

def determine_dominant_group_per_date(
    metadata: pd.DataFrame,
    date_col: str,
    group_col: str,
) -> pd.Series:
    """
    Map each sampling date to the dominant (most common) group value.
    """
    meta_reset = metadata.reset_index()
    if group_col not in meta_reset.columns:
        raise ValueError(f"Metadata column '{group_col}' required for trajectory context")
    if date_col not in meta_reset.columns:
        raise ValueError(f"Metadata column '{date_col}' required for trajectory context")

    dominant_series = (
        meta_reset.dropna(subset=[date_col])
        .groupby(date_col)[group_col]
        .apply(_dominant_category)
    )
    return dominant_series


def load_trajectory_summary(summary_path: Path) -> pd.DataFrame:
    """
    Load trajectory summary TSV and validate required columns.
    """
    summary_df = pd.read_csv(summary_path, sep='\t')
    if summary_df.empty:
        raise ValueError(f"Trajectory summary {summary_path} is empty")
    if 'group' not in summary_df.columns:
        raise ValueError(f"Trajectory summary {summary_path} missing 'group' column")
    return summary_df


def attach_trajectory_context(
    timeseries_df: pd.DataFrame,
    extremes_df: pd.DataFrame,
    metadata: pd.DataFrame,
    date_col: str,
    summary_df: pd.DataFrame,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge trajectory context (group assignments, colors, metrics) into outputs.
    """
    print("  [i] Integrating trajectory context from seasonal analysis...")
    date_group_map = determine_dominant_group_per_date(metadata, date_col, group_col)
    timeseries_df = timeseries_df.copy()
    timeseries_df['trajectory_group'] = timeseries_df['date'].map(date_group_map)
    missing_dates = timeseries_df['trajectory_group'].isna().sum()
    if missing_dates:
        print(f"      Warning: {missing_dates} time points lack trajectory group assignments")

    summary_pref = summary_df.rename(columns={'group': 'trajectory_group'}).copy()
    summary_pref = summary_pref.rename(columns={
        col: f"trajectory_{col}"
        for col in summary_pref.columns
        if col != 'trajectory_group'
    })

    timeseries_df = timeseries_df.merge(summary_pref, on='trajectory_group', how='left')

    if extremes_df is not None and not extremes_df.empty:
        context_cols = ['date', 'trajectory_group'] + [
            col for col in timeseries_df.columns
            if col.startswith('trajectory_')
        ]
        context_df = timeseries_df[context_cols].drop_duplicates(subset=['date'])
        extremes_df = extremes_df.merge(context_df, on='date', how='left')

    return timeseries_df, extremes_df


def normalize_to_centered_scale(
    timeseries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize stratification scores to [-1, 1] scale centered on zero.
    
    -1 = maximally mixed
     0 = intermediate
    +1 = maximally stratified
    """
    print("  [i] Normalizing to centered scale...")
    
    # Get min/max
    min_score = timeseries_df['stratification_score'].min()
    max_score = timeseries_df['stratification_score'].max()
    median_score = timeseries_df['stratification_score'].median()
    
    # Normalize to [-1, 1] with median at 0
    timeseries_df = timeseries_df.copy()
    
    normalized_scores = []
    for score in timeseries_df['stratification_score']:
        if score < median_score:
            # Mixed side: map [min, median] to [-1, 0]
            norm = -1 + (score - min_score) / (median_score - min_score)
        else:
            # Stratified side: map [median, max] to [0, 1]
            norm = (score - median_score) / (max_score - median_score)
        normalized_scores.append(norm)
    
    timeseries_df['normalized_score'] = normalized_scores
    
    print(f"      Normalized: min={min(normalized_scores):.3f}, "
          f"median=0, max={max(normalized_scores):.3f}")
    
    return timeseries_df


def detect_anomalies_consensus(
    timeseries_df: pd.DataFrame,
    consensus_threshold: int = 2,
) -> pd.DataFrame:
    """
    Detect anomalies using consensus of multiple methods.
    """
    print(f"  [i] Detecting anomalies (consensus threshold ≥{consensus_threshold})...")
    
    scores = timeseries_df['normalized_score'].values.reshape(-1, 1)
    
    anomaly_votes = np.zeros((len(timeseries_df), 4))
    
    # Method 1: Z-score
    z_scores = np.abs(zscore(timeseries_df['normalized_score']))
    anomaly_votes[:, 0] = (z_scores > 1.5).astype(int)
    
    # Method 2: IQR
    Q1 = timeseries_df['normalized_score'].quantile(0.25)
    Q3 = timeseries_df['normalized_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomaly_votes[:, 1] = ((timeseries_df['normalized_score'] < lower_bound) | 
                           (timeseries_df['normalized_score'] > upper_bound)).astype(int)
    
    # Method 3: Isolation Forest
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_predictions = iso_forest.fit_predict(scores)
    anomaly_votes[:, 2] = (iso_predictions == -1).astype(int)
    
    # Method 4: Local Outlier Factor
    n_neighbors = min(10, len(scores) - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.15)
    lof_predictions = lof.fit_predict(scores)
    anomaly_votes[:, 3] = (lof_predictions == -1).astype(int)
    
    # Consensus
    total_votes = anomaly_votes.sum(axis=1)
    is_anomaly = total_votes >= consensus_threshold
    
    # Classify type
    anomaly_type = []
    for is_anom, score in zip(is_anomaly, timeseries_df['normalized_score']):
        if is_anom:
            if score > 0:
                anomaly_type.append('high_stratification')
            else:
                anomaly_type.append('mixing_event')
        else:
            anomaly_type.append('normal')
    
    timeseries_df = timeseries_df.copy()
    timeseries_df['is_anomaly'] = is_anomaly
    timeseries_df['anomaly_type'] = anomaly_type
    timeseries_df['n_votes'] = total_votes
    
    n_anomalies = is_anomaly.sum()
    n_mixing = (timeseries_df['anomaly_type'] == 'mixing_event').sum()
    n_high_strat = (timeseries_df['anomaly_type'] == 'high_stratification').sum()
    
    print(f"      Total anomalies: {n_anomalies}/{len(timeseries_df)} time points")
    print(f"      Mixing events: {n_mixing}")
    print(f"      High stratification: {n_high_strat}")
    
    return timeseries_df


def identify_annual_extremes(
    timeseries_df: pd.DataFrame,
    metadata: pd.DataFrame,
    year_col: str,
) -> pd.DataFrame:
    """
    Identify max stratification and max mixing events per year.
    """
    print("  [i] Identifying annual extremes...")
    
    # Add year to timeseries
    timeseries_df = timeseries_df.copy()
    
    # Get year for each date
    date_to_year = metadata.set_index(metadata.columns[0])[year_col].to_dict()
    
    # Map dates to years (handle both string and datetime)
    years = []
    for date in timeseries_df['date']:
        # Try to find year from metadata
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
            # Fallback: extract year from date if possible
            try:
                if isinstance(date, str):
                    year = int(date.split('-')[0]) if '-' in date else int(date[:4])
                else:
                    year = date.year
                years.append(year)
            except:
                years.append(None)
    
    timeseries_df['year'] = years
    
    # Find max/min per year
    extremes = []
    
    for year in sorted(timeseries_df['year'].dropna().unique()):
        year_data = timeseries_df[timeseries_df['year'] == year]
        
        # Max stratification (highest positive score)
        max_strat_idx = year_data['normalized_score'].idxmax()
        max_strat_row = year_data.loc[max_strat_idx]
        
        extremes.append({
            'date': max_strat_row['date'],
            'year': year,
            'normalized_score': max_strat_row['normalized_score'],
            'extreme_type': 'max_stratification',
        })
        
        # Max mixing (lowest negative score)
        min_mix_idx = year_data['normalized_score'].idxmin()
        min_mix_row = year_data.loc[min_mix_idx]
        
        extremes.append({
            'date': min_mix_row['date'],
            'year': year,
            'normalized_score': min_mix_row['normalized_score'],
            'extreme_type': 'max_mixing',
        })
    
    extremes_df = pd.DataFrame(extremes)
    
    print(f"      Found {len(extremes_df)} extremes across {len(extremes_df['year'].unique())} years")
    
    return extremes_df


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_stratification_timeseries(
    timeseries_df: pd.DataFrame,
    extremes_df: pd.DataFrame,
    metadata: pd.DataFrame,
    date_col: str,
    month_col: str,
    output_path: Path,
) -> None:
    """
    Continuous time-series plot with year/month axis and sized extreme markers.
    """
    print("  [i] Creating stratification time series plot...")
    
    # Sort by date
    timeseries_df = timeseries_df.sort_values('date').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(24, 8))
    
    x_pos = np.arange(len(timeseries_df))
    
    # Plot line
    ax.plot(x_pos, timeseries_df['normalized_score'],
           '-', linewidth=2.5, color='steelblue', alpha=0.9, zorder=2)
    
    # Fill areas
    ax.fill_between(x_pos, 0, timeseries_df['normalized_score'],
                    where=timeseries_df['normalized_score'] >= 0,
                    color='blue', alpha=0.3, label='Stratified', zorder=1)
    
    ax.fill_between(x_pos, 0, timeseries_df['normalized_score'],
                    where=timeseries_df['normalized_score'] < 0,
                    color='orange', alpha=0.3, label='Mixed', zorder=1)
    
    # Zero line
    ax.axhline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=3)
    
    # ========================================================================
    # Plot extremes with sizes proportional to intensity
    # ========================================================================
    
    # Get global max/min for sizing
    global_max_strat = timeseries_df['normalized_score'].max()
    global_min_mix = timeseries_df['normalized_score'].min()
    
    date_to_idx = {date: idx for idx, date in enumerate(timeseries_df['date'])}
    
    for _, row in extremes_df.iterrows():
        if row['date'] in date_to_idx:
            x_idx = date_to_idx[row['date']]
            
            if row['extreme_type'] == 'max_stratification':
                # Size by how close to global max (0 to 1 scale)
                intensity = row['normalized_score'] / global_max_strat if global_max_strat > 0 else 1
                marker_size = 200 + (intensity * 600)  # Range: 200-800
                
                ax.scatter(x_idx, row['normalized_score'],
                          s=marker_size, marker='^', color='darkblue',
                          edgecolors='black', linewidth=2.5, zorder=10,
                          alpha=0.85)
            
            else:  # max_mixing
                # Size by how close to global min (0 to 1 scale)
                intensity = abs(row['normalized_score'] / global_min_mix) if global_min_mix < 0 else 1
                marker_size = 200 + (intensity * 600)  # Range: 200-800
                
                ax.scatter(x_idx, row['normalized_score'],
                          s=marker_size, marker='v', color='darkorange',
                          edgecolors='black', linewidth=2.5, zorder=10,
                          alpha=0.85)
    
    # ========================================================================
    # X-axis: Major ticks = Years, Minor ticks = Months
    # ========================================================================
    
    # Get year and month for each date from timeseries (already has year from identify_annual_extremes)
    if 'year' in timeseries_df.columns:
        years = timeseries_df['year'].values
    else:
        # Fallback: extract from metadata
        date_to_year = {row[date_col]: row['Year'] 
                       for _, row in metadata.iterrows() 
                       if 'Year' in metadata.columns}
        years = [date_to_year.get(date, None) for date in timeseries_df['date']]
    
    # Get months from metadata
    date_to_month = {row[date_col]: row[month_col] 
                    for _, row in metadata.iterrows() 
                    if month_col in metadata.columns}
    months = [date_to_month.get(date, None) for date in timeseries_df['date']]
    
    print(f"      Years range: {[y for y in years if y is not None][:5]} ... {[y for y in years if y is not None][-5:]}")
    print(f"      Unique years: {sorted(set([y for y in years if y is not None]))}")
    
    # Find year boundaries for major ticks
    # Get first occurrence of each year
    year_positions = {}
    year_order = []
    
    for i, year in enumerate(years):
        if year is not None and not pd.isna(year):
            year = int(year)
            if year not in year_positions:
                year_positions[year] = i
                year_order.append(year)
    
    print(f"      Year positions: {year_positions}")
    
    # Major ticks: years
    if len(year_positions) > 0:
        major_tick_positions = [year_positions[y] for y in sorted(year_order)]
        major_tick_labels = sorted(year_order)
        
        ax.set_xticks(major_tick_positions)
        ax.set_xticklabels([int(y) for y in major_tick_labels], fontsize=13, fontweight='bold')
        
        print(f"      Set {len(major_tick_positions)} year ticks")
    else:
        print("      WARNING: No year positions found!")
    
    # Minor ticks: months (first letter)
    month_names_short = ['', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    
    # Create minor ticks at regular intervals to avoid overcrowding
    n_points = len(timeseries_df)
    
    # Sample points for minor ticks - aim for ~60 ticks across the timeline
    target_minor_ticks = min(60, n_points)
    step = max(1, n_points // target_minor_ticks)
    
    minor_tick_positions = list(range(0, n_points, step))
    minor_tick_labels = []
    
    for pos in minor_tick_positions:
        month_num = months[pos]
        if month_num is not None and not pd.isna(month_num) and 1 <= month_num <= 12:
            minor_tick_labels.append(month_names_short[int(month_num)])
        else:
            minor_tick_labels.append('')
    
    # Set minor ticks
    ax.set_xticks(minor_tick_positions, minor=True)
    ax.set_xticklabels(minor_tick_labels, minor=True, fontsize=9, alpha=0.7)
    
    print(f"      Set {len(minor_tick_positions)} month ticks")
    
    # Adjust tick appearance
    ax.tick_params(axis='x', which='major', length=10, width=2, labelsize=13, pad=10)
    ax.tick_params(axis='x', which='minor', length=5, width=1, labelsize=9, pad=5)
    
    
    # ========================================================================
    # Labels and styling
    # ========================================================================
    
    ax.set_xlabel('Time (Years with Monthly Resolution)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stratification Index\n(−1=Mixed, 0=Intermediate, +1=Stratified)',
                 fontsize=14, fontweight='bold')
    ax.set_title('Water Column Stratification Dynamics\n' +
                'Continuous Time Series with Annual Extremes (Triangle Size = Intensity)',
                fontsize=15, fontweight='bold', pad=20)
    
    # Y-axis
    ax.set_ylim([-1.2, 1.2])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(['Max\nMixed', 'Mixed', 'Intermediate', 'Stratified', 'Max\nStratified'],
                       fontsize=11, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='blue', alpha=0.5, edgecolor='black', linewidth=2,
             label='Stratified Period'),
        Patch(facecolor='orange', alpha=0.5, edgecolor='black', linewidth=2,
             label='Mixed Period'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='darkblue',
              markersize=16, markeredgecolor='black', markeredgewidth=2,
              linestyle='None', label='Annual Max Stratification'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='darkorange',
              markersize=16, markeredgecolor='black', markeredgewidth=2,
              linestyle='None', label='Annual Max Mixing'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
             frameon=True, fancybox=True, ncol=2, framealpha=0.95)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
    
    # Ensure x-axis labels don't overlap
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved stratification time series")


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
    Monthly index plot: grey tracelines per year plus smoothed average and CI,
    with strat/mix extremes annotated. Matches the y-axis styling of the main
    time series.
    """
    print("  [i] Creating enriched monthly stratification profile...")
    plot_df = timeseries_df.copy()

    meta_reset = metadata.reset_index()
    if date_col not in meta_reset.columns:
        raise ValueError(f"Metadata missing date column '{date_col}' for monthly profile")
    if month_col not in meta_reset.columns or year_col not in meta_reset.columns:
        raise ValueError("Metadata must supply month/year columns for monthly profile")

    date_to_month = (
        meta_reset.groupby(date_col)[month_col]
        .first()
        .to_dict()
    )
    date_to_year = (
        meta_reset.groupby(date_col)[year_col]
        .first()
        .to_dict()
    )

    plot_df['month'] = plot_df['date'].map(date_to_month)
    plot_df['year'] = plot_df['date'].map(date_to_year)
    plot_df = plot_df.dropna(subset=['month', 'year']).copy()
    plot_df['coverage'] = plot_df['coverage'].fillna(0.0)
    plot_df['month'] = plot_df['month'].astype(int)
    plot_df['year'] = plot_df['year'].astype(int)

    pivot = (
        plot_df.groupby(['month', 'year'])['normalized_score']
        .mean()
        .unstack(level=1)
        .reindex(range(1, 13))
    )
    pivot = pivot.apply(pd.to_numeric, errors='coerce')
    coverage_pivot = (
        plot_df.groupby(['month', 'year'])['coverage']
        .mean()
        .unstack(level=1)
        .reindex(range(1, 13))
        .apply(pd.to_numeric, errors='coerce')
    )

    plot_range = np.linspace(0.8, 12.2, 300)

    month_means = pivot.mean(axis=1, skipna=True)
    mean_values = month_means.interpolate(limit_direction='both').fillna(method='ffill').fillna(method='bfill')
    mean_smoothed = gaussian_filter1d(mean_values.values, sigma=1.1)
    mean_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed)

    month_std = pivot.std(axis=1, ddof=0).fillna(0.0)
    std_smoothed = gaussian_filter1d(month_std.values, sigma=1.1)
    lower_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed - std_smoothed)
    upper_curve = np.interp(plot_range, np.arange(1, 13), mean_smoothed + std_smoothed)

    fig, ax = plt.subplots(figsize=(20, 6))

    ax.fill_between(plot_range, lower_curve, upper_curve,
                    color='lightgrey', alpha=0.6, zorder=1)
    ax.plot(plot_range, mean_curve, color='black', linewidth=3, zorder=2)
    ax.axhline(0, color='black', linewidth=2, alpha=0.6, zorder=4)

    extremes_df = extremes_df.copy()
    extremes_df['month'] = extremes_df['date'].map(date_to_month)
    strat_points = extremes_df[extremes_df['extreme_type'] == 'max_stratification']
    mix_points = extremes_df[extremes_df['extreme_type'] == 'max_mixing']

    coverage_counts = {}
    for year in coverage_pivot.columns:
        coverage_counts[year] = (coverage_pivot[year] >= COVERAGE_THRESHOLD).sum()
    min_months_needed = 7
    eligible_years = {year for year, count in coverage_counts.items() if count >= min_months_needed}

    for _, row in strat_points.iterrows():
        if pd.isna(row['month']) or row.get('year', None) not in eligible_years:
            continue
        month_val = row['month']
        ax.scatter(month_val, row['normalized_score'],
                   marker='^', color='royalblue', s=200, edgecolor='black', linewidth=2.0, zorder=5)

    for _, row in mix_points.iterrows():
        if pd.isna(row['month']) or row.get('year', None) not in eligible_years:
            continue
        month_val = row['month']
        ax.scatter(month_val, row['normalized_score'],
                   marker='v', color='darkorange', s=200, edgecolor='black', linewidth=2.0, zorder=5)

    ax.set_xlim(0.9, 12.1)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)

    score_min = timeseries_df['normalized_score'].min()
    score_max = timeseries_df['normalized_score'].max()
    y_pad = (score_max - score_min) * 0.1 if score_max != score_min else 0.5
    ax.set_ylim(score_min - y_pad, score_max + y_pad)

    ax.set_xlabel('Month', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stratification Index\n(−1=Mixed, 0=Intermediate, +1=Stratified)',
                  fontsize=14, fontweight='bold')
    ax.text(-0.05, 0.5, '', transform=ax.transAxes,
            fontsize=12, fontweight='bold', rotation=90, va='center')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(['Max\nMixed', 'Mixed', 'Intermediate', 'Stratified', 'Max\nStratified'],
                       fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', which='major', pad=8)
    ax.set_title('Monthly Stratification Profile', fontsize=16, fontweight='bold')

    point_df = pivot.stack().reset_index(name='normalized_score')
    coverage_df = coverage_pivot.stack().reset_index(name='coverage')
    point_df = point_df.merge(coverage_df, on=['month', 'year'], how='left')

    for _, row in point_df.iterrows():
        if pd.isna(row['normalized_score']) or pd.isna(row['coverage']):
            continue
        if row['coverage'] < COVERAGE_THRESHOLD:
            continue
        color = 'royalblue' if row['normalized_score'] >= 0 else 'darkorange'
        ax.scatter(row['month'], row['normalized_score'],
                   color=color, s=48, edgecolor='black', linewidth=0.6, alpha=0.9, zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved monthly stratification profile")



def plot_anomaly_heatmap(
    timeseries_df: pd.DataFrame,
    metadata: pd.DataFrame,
    month_col: str,
    year_col: str,
    output_path: Path,
) -> None:
    """
    Heatmap showing stratification state by month and year.
    """
    print("  [i] Creating anomaly heatmap...")
    
    # Add month and year
    timeseries_df = timeseries_df.copy()
    
    date_to_info = {}
    for idx, row in metadata.iterrows():
        date_val = row[metadata.columns[0]]
        if month_col in metadata.columns and year_col in metadata.columns:
            date_to_info[date_val] = (row[month_col], row[year_col])
    
    months = []
    years = []
    for date in timeseries_df['date']:
        if date in date_to_info:
            month, year = date_to_info[date]
            months.append(month)
            years.append(year)
        else:
            months.append(None)
            years.append(None)
    
    timeseries_df['month'] = months
    timeseries_df['year'] = years
    
    # Drop rows without month/year
    timeseries_df = timeseries_df.dropna(subset=['month', 'year'])
    
    # Pivot to matrix
    pivot_data = timeseries_df.pivot_table(
        index='month',
        columns='year',
        values='normalized_score',
        aggfunc='mean'
    )
    
    # Sort by month
    pivot_data = pivot_data.sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Heatmap
    sns.heatmap(
        pivot_data,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Stratification Index\n(Blue=Mixed, Red=Stratified)'},
        linewidths=1,
        linecolor='white',
        square=False,
        ax=ax,
        annot=False,
    )
    
    # Labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels(month_names[:len(pivot_data)], rotation=0, fontsize=11)
    
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Month', fontsize=13, fontweight='bold')
    ax.set_title('Stratification Dynamics: Monthly × Yearly Heatmap',
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved anomaly heatmap")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stratification time-series analysis with integrated data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--integrated-data", type=Path, required=True,
                       help="Integrated data TSV (samples × features)")
    parser.add_argument("--metadata", type=Path, required=True,
                       help="Metadata TSV file")
    parser.add_argument("--date-col", required=True,
                       help="Column name for date/time")
    parser.add_argument("--month-col", required=True,
                       help="Column name for month")
    parser.add_argument("--year-col", required=True,
                       help="Column name for year")
    parser.add_argument("--depth-col", required=True,
                       help="Column name for depth")
    parser.add_argument("--sample-id-col", default=SAMPLE_ID_COL,
                       help="Column containing unique sample IDs (default: sampleid)")
    parser.add_argument("--trajectory-summary", type=Path,
                       help="Optional trajectory summary TSV (from trajectory_analysis.py) to add contextual metrics")
    parser.add_argument("--trajectory-group-col", default=None,
                       help="Metadata column that matches the 'group' column in the trajectory summary (e.g., cluster)")
    parser.add_argument("--consensus-threshold", type=int, default=2,
                       help="Minimum methods for anomaly consensus")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--biochem-only-data", type=Path,
                       help="Optional biogeochemistry-only matrix (year/month/depth + biochem features)")
    parser.add_argument("--biochem-sample-id-col", default=None,
                       help="Column to use as sample identifier in the biochem-only file (defaults to generated IDs)")
    parser.add_argument("--biochem-cruise-col", default="Cruise",
                       help="Column name for cruise identifier in the biochem-only file")
    parser.add_argument("--biochem-year-col", default="Year",
                       help="Column name for year in the biochem-only file")
    parser.add_argument("--biochem-month-col", default="Month",
                       help="Column name for month in the biochem-only file")
    parser.add_argument("--biochem-depth-col", default="Depth",
                       help="Column name for depth in the biochem-only file")
    parser.add_argument("--biochem-date-day", type=int, default=1,
                       help="Day of month to assign when only year/month are available")
    parser.add_argument("--biochem-sample-suffix", default="biochem",
                       help="Suffix to append to generated sample IDs for biochem-only rows")
    parser.add_argument("--vertical-profile-output", type=Path,
                       help="Optional path to save vertical biochem depth profiles (PDF)")
    
    args = parser.parse_args()
    meta_sample_col = args.sample_id_col
    
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STRATIFICATION TIME-SERIES ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    integrated_data = pd.read_csv(args.integrated_data, sep="\t", index_col=0)
    metadata_df = pd.read_csv(args.metadata, sep="\t")
    if meta_sample_col not in metadata_df.columns:
        raise ValueError(f"Metadata column '{meta_sample_col}' not found in {args.metadata}")
    if args.biochem_only_data:
        integrated_data, metadata_df = integrate_biochem_only_samples(
            integrated_data,
            metadata_df,
            args,
            meta_sample_col,
        )
    feature_cols = select_biochem_features(
        integrated_data,
        metadata_df,
        COVERAGE_THRESHOLD,
    )
    integrated_data[feature_cols] = integrated_data[feature_cols].apply(pd.to_numeric, errors='coerce')
    metadata = metadata_df.drop_duplicates(subset=[meta_sample_col]).set_index(meta_sample_col)
    vertical_vars = select_metadata_biochem_columns(metadata_df)
    trajectory_summary_df = None
    trajectory_group_col = args.trajectory_group_col
    if args.trajectory_summary:
        if not args.trajectory_summary.exists():
            raise FileNotFoundError(f"Trajectory summary not found: {args.trajectory_summary}")
        trajectory_summary_df = load_trajectory_summary(args.trajectory_summary)
        inferred_group = None
        if 'group_col' in trajectory_summary_df.columns:
            inferred_vals = trajectory_summary_df['group_col'].dropna()
            if not inferred_vals.empty:
                inferred_group = str(inferred_vals.iloc[0])
        if trajectory_group_col is None:
            trajectory_group_col = inferred_group
        if not trajectory_group_col:
            raise ValueError("Trajectory group column is required when providing a trajectory summary")
        if trajectory_group_col not in metadata_df.columns:
            raise ValueError(f"Metadata missing trajectory group column '{trajectory_group_col}'")
        print(f"  [i] Trajectory context: using column '{trajectory_group_col}' with summary {args.trajectory_summary.name}")

    common = integrated_data.index.intersection(metadata.index)
    integrated_data = integrated_data.loc[common]
    metadata = metadata.loc[common]

    print(f"  Samples: {len(common)}")
    print(f"  Features: {integrated_data.shape[1]}")
    print(f"  Time points: {len(metadata[args.date_col].unique())}")
    
    # Calculate stratification time series
    print("\n[2/6] Calculating stratification time series...")
    timeseries_df = calculate_stratification_score_timeseries(
        integrated_data,
        metadata,
        args.date_col,
        args.depth_col,
        feature_cols,
    )
    
    # Normalize to centered scale
    print("\n[3/6] Normalizing to centered scale...")
    timeseries_df = normalize_to_centered_scale(timeseries_df)
    
    # Anomaly detection
    print("\n[4/6] Running consensus anomaly detection...")
    timeseries_df = detect_anomalies_consensus(
        timeseries_df,
        consensus_threshold=args.consensus_threshold,
    )
    
    # Identify annual extremes
    print("\n[5/6] Identifying annual extremes...")
    extremes_df = identify_annual_extremes(
        timeseries_df,
        metadata,
        args.year_col,
    )

    if trajectory_summary_df is not None:
        timeseries_df, extremes_df = attach_trajectory_context(
            timeseries_df,
            extremes_df,
            metadata,
            args.date_col,
            trajectory_summary_df,
            trajectory_group_col,
        )
    
    # Save results
    timeseries_df.to_csv(out_dir / "stratification_timeseries.tsv", sep='\t', index=False)
    extremes_df.to_csv(out_dir / "annual_extremes.tsv", sep='\t', index=False)
    
    # Visualizations
    print("\n[6/6] Creating visualizations...")
    
    plot_stratification_timeseries(
        timeseries_df,
        extremes_df,
        metadata,
        args.date_col,
        args.month_col,
        out_dir / "stratification_timeseries.pdf"
    )
    
    plot_stratification_monthly_profile(
        timeseries_df,
        extremes_df,
        metadata,
        args.date_col,
        args.month_col,
        args.year_col,
        out_dir / "stratification_monthly_profile.pdf"
    )

    vertical_output = args.vertical_profile_output or (out_dir / "biochem_vertical_profiles.pdf")
    plot_biochem_vertical_profiles(
        metadata_df,
        args.depth_col,
        vertical_vars,
        vertical_output,
    )
    
    plot_anomaly_heatmap(
        timeseries_df,
        metadata,
        args.month_col,
        args.year_col,
        out_dir / "stratification_heatmap.pdf"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTime series: {len(timeseries_df)} time points")
    print(f"Score range: [{timeseries_df['normalized_score'].min():.3f}, "
          f"{timeseries_df['normalized_score'].max():.3f}]")
    
    print("\nAnnual Extremes:")
    for _, row in extremes_df.iterrows():
        print(f"  {row['extreme_type']:20s} - Year {int(row['year'])}: "
              f"score={row['normalized_score']:.3f}")
    
    print(f"\n{len(timeseries_df[timeseries_df['is_anomaly']])} anomalous time points detected")
    
    print("\n" + "="*70)
    print(f"\nOutputs saved to: {out_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
