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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, pdist
from scipy.stats import zscore
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


# ============================================================================
# Core Analysis Functions
# ============================================================================

def calculate_stratification_score_timeseries(
    integrated_data: pd.DataFrame,
    metadata: pd.DataFrame,
    date_col: str,
    depth_col: str,
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
        date_data = integrated_data.loc[date_mask]
        date_meta = metadata.loc[date_mask]
        
        depths = sorted(date_meta[depth_col].unique())
        
        if len(depths) < 2:
            # Can't calculate stratification with only one depth
            continue
        
        # Calculate centroid for each depth at this time point
        depth_centroids = {}
        
        for depth in depths:
            depth_mask = date_meta[depth_col] == depth
            depth_samples = date_data.loc[depth_mask]
            
            if len(depth_samples) > 0:
                depth_centroids[depth] = depth_samples.mean().values
        
        # Calculate mean pairwise distance
        if len(depth_centroids) >= 2:
            distances = []
            for depth1, depth2 in combinations(depth_centroids.keys(), 2):
                dist = euclidean(depth_centroids[depth1], depth_centroids[depth2])
                distances.append(dist)
            
            mean_dist = np.mean(distances)
            
            results.append({
                'date': date,
                'stratification_score': mean_dist,
                'n_depths': len(depths),
                'depths_present': ','.join(map(str, depths)),
                'n_samples': len(date_data),
            })
    
    timeseries_df = pd.DataFrame(results)
    
    print(f"      Computed {len(timeseries_df)} time points")
    
    return timeseries_df


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
    parser.add_argument("--consensus-threshold", type=int, default=2,
                       help="Minimum methods for anomaly consensus")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory")
    
    args = parser.parse_args()
    
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STRATIFICATION TIME-SERIES ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    integrated_data = pd.read_csv(args.integrated_data, sep="\t", index_col=0)
    metadata_df = pd.read_csv(args.metadata, sep="\t")
    if SAMPLE_ID_COL not in metadata_df.columns:
        raise ValueError(f"Metadata column '{SAMPLE_ID_COL}' not found in {args.metadata}")
    metadata = metadata_df.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(SAMPLE_ID_COL)
    
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
