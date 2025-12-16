#!/usr/bin/env python3
"""
trajectory_analysis_seasonal.py

Seasonal trajectory analysis of microbial communities.
Compares hypothesis-driven (depth) vs data-driven (clustering) groupings.
Uses months as temporal axis with years as pseudo-replicates.

Usage:
  python trajectory_analysis_seasonal.py \
    --umap-results spark_combined_output/batch_correction/umap_hdbscan_results.tsv \
    --asv-data spark_combined_output/batch_correction/asv_clr_after_correction.tsv \
    --metadata spark_combined_output/metadata/metadata_updated.tsv \
    --month-col Month \
    --group-cols Depth,cluster_after \
    --output-dir spark_combined_output/trajectory_analysis \
    --top-taxa 20 \
    --verbose
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.stats import spearmanr, pearsonr, circmean, circstd
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
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']  # retained for reference


def format_temporal_label(value) -> str:
    """
    Format temporal values (which we keep as ints) for plotting.
    """
    if pd.isna(value):
        return ''
    return str(int(value))


def standardize_temporal_column(metadata: pd.DataFrame, month_col: str) -> Tuple[List, Dict]:
    """
    Coerce the temporal column to integers and return ordering.
    """
    series = pd.to_numeric(metadata[month_col], errors='coerce')
    if series.isna().any():
        raise ValueError(f"Month column '{month_col}' must be numeric; found non-numeric entries.")
    series = series.astype(int)
    metadata[month_col] = series
    ordered = sorted(series.dropna().unique())
    rank_map = {val: idx + 1 for idx, val in enumerate(ordered)}
    return ordered, rank_map


def normalize_group_labels(series: pd.Series) -> pd.Series:
    """
    Normalize grouping labels (Depth, clusters) to consistent numeric values when possible.
    """
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype(int)
    return series.astype(str).str.strip()


# ============================================================================
# Data Processing
# ============================================================================

def aggregate_monthly_data(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    month_col: str,
    group_col: str,
    temporal_order: Optional[List] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate data by group and month, treating years as replicates.
    
    Parameters
    ----------
    data : DataFrame (samples x features)
        Data to aggregate (e.g., UMAP coords or ASV abundances)
    metadata : DataFrame
        Sample metadata
    month_col : str
        Month column name
    group_col : str
        Grouping column name (e.g., 'Depth' or 'cluster_after')
    
    Returns
    -------
    mean_data : DataFrame
        Mean values per group-month
    std_data : DataFrame
        Standard deviation per group-month
    count_data : DataFrame
        Sample counts per group-month
    """
    # Combine data with metadata
    combined = data.copy()
    if temporal_order:
        combined['month'] = pd.Categorical(metadata[month_col].values,
                                           categories=temporal_order,
                                           ordered=True)
    else:
        combined['month'] = metadata[month_col].values
    combined['group'] = metadata[group_col].values
    
    # Group by group and month
    grouped = combined.groupby(['group', 'month'], observed=True)
    
    # Calculate statistics
    mean_data = grouped.mean().reset_index()
    std_data = grouped.std().reset_index()
    count_data = grouped.size().reset_index(name='n_samples')
    
    return mean_data, std_data, count_data


def calculate_seasonal_trajectory_metrics(
    coords_mean: pd.DataFrame,
    coords_std: pd.DataFrame,
    month_rank_map: Dict,
    group_col: str = 'group',
) -> pd.DataFrame:
    """
    Calculate trajectory metrics for seasonal patterns.
    
    Parameters
    ----------
    coords_mean : DataFrame
        Mean coordinates per group-month
    coords_std : DataFrame
        Std coordinates per group-month
    group_col : str
        Name of grouping column
    
    Returns
    -------
    metrics_df : DataFrame
        Trajectory metrics per group
    """
    print("  [i] Calculating seasonal trajectory metrics...")
    
    metrics = []
    
    for grp in coords_mean[group_col].unique():
        grp_data = coords_mean[coords_mean[group_col] == grp].copy()
        grp_std = coords_std[coords_std[group_col] == grp].copy()
        if month_rank_map:
            grp_data['month_order'] = grp_data['month'].map(month_rank_map)
            grp_std['month_order'] = grp_std['month'].map(month_rank_map)
            grp_data = grp_data.sort_values('month_order')
            grp_std = grp_std.sort_values('month_order')
        else:
            grp_data = grp_data.sort_values('month')
            grp_std = grp_std.sort_values('month')
        
        if len(grp_data) < 2:
            print(f"    [!] Skipping group '{grp}': insufficient months (n={len(grp_data)})")
            continue
        
        # Extract coordinates (assuming UMAP1, UMAP2 or similar)
        coord_cols = [c for c in grp_data.columns if c not in [group_col, 'month', 'n_samples']]
        coords = grp_data[coord_cols].values
        
        # Calculate trajectory length (cumulative distance through months)
        segment_dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
        total_length = np.sum(segment_dists)
        
        # Calculate circular trajectory (month 12 → month 1)
        if len(coords) >= 12 or (grp_data['month'].min() == 1 and grp_data['month'].max() == 12):
            # Distance from last month back to first month (seasonal cycle)
            cycle_closure = np.sqrt(np.sum((coords[-1] - coords[0])**2))
        else:
            cycle_closure = np.nan
        
        # Net displacement (start to end of year)
        net_displacement = np.sqrt(np.sum((coords[-1] - coords[0])**2))
        
        # Directionality
        directionality = net_displacement / total_length if total_length > 0 else 0
        
        # Mean within-month variance (stability)
        std_coords = grp_std[coord_cols].values
        mean_variance = np.mean(np.sum(std_coords**2, axis=1))
        
        # Number of months with data
        n_months = len(grp_data)
        
        metrics.append({
            'group': grp,
            'n_months': n_months,
            'trajectory_length': total_length,
            'net_displacement': net_displacement,
            'cycle_closure': cycle_closure,
            'directionality': directionality,
            'mean_variance': mean_variance,
            'mean_segment_dist': np.mean(segment_dists) if len(segment_dists) > 0 else 0,
        })
    
    return pd.DataFrame(metrics)


def identify_seasonal_marker_taxa(
    asv_data: pd.DataFrame,
    metadata: pd.DataFrame,
    month_col: str,
    group_col: str,
    top_n: int = 20,
    temporal_order: Optional[List] = None,
    month_rank_map: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Identify taxa that show seasonal patterns within each group.
    
    Parameters
    ----------
    asv_data : DataFrame (samples x ASVs)
        CLR-transformed ASV abundances
    metadata : DataFrame
        Sample metadata
    month_col : str
        Month column name
    group_col : str
        Group column name
    top_n : int
        Number of top marker taxa per group
    
    Returns
    -------
    marker_taxa : dict
        Dictionary with group as key and DataFrame of marker taxa as value
    """
    print("  [i] Identifying seasonal marker taxa...")
    
    marker_taxa = {}
    
    for grp in metadata[group_col].unique():
        mask = metadata[group_col] == grp
        
        if mask.sum() < 3:
            print(f"    [!] Skipping group '{grp}': insufficient samples (n={mask.sum()})")
            continue
        
        # Get data for this group
        grp_asv = asv_data.loc[mask]
        grp_month_raw = metadata.loc[mask, month_col]
        if month_rank_map:
            grp_month = pd.to_numeric(grp_month_raw.map(month_rank_map), errors='coerce')
            period_divisor = max(len(month_rank_map), 1)
        else:
            grp_month = pd.to_numeric(grp_month_raw, errors='coerce')
            period_divisor = 12
        
        # Calculate correlation with month for each ASV
        correlations = []
        for asv in grp_asv.columns:
            # Spearman correlation with month
            rho, pval = spearmanr(grp_month, grp_asv[asv])
            
            # Also calculate circular correlation (treating months as circular)
            # Convert months to radians
            months_rad = grp_month * 2 * np.pi / period_divisor
            
            correlations.append({
                'asv': asv,
                'correlation': rho,
                'p_value': pval,
                'abs_correlation': abs(rho),
                'mean_abundance': grp_asv[asv].mean(),
                'max_abundance': grp_asv[asv].max(),
                'min_abundance': grp_asv[asv].min(),
                'range': grp_asv[asv].max() - grp_asv[asv].min(),
                'direction': 'increase' if rho > 0 else 'decrease',
            })
        
        df_corr = pd.DataFrame(correlations)
        
        # Filter significant and rank by absolute correlation
        df_sig = df_corr[df_corr['p_value'] < 0.05].copy()
        df_sig = df_sig.sort_values('abs_correlation', ascending=False)
        
        # Get top markers
        top_markers = df_sig.head(top_n)
        
        marker_taxa[grp] = top_markers
        
        print(f"    Group {grp}: {len(df_sig)} significant markers, showing top {min(top_n, len(top_markers))}")
    
    return marker_taxa


def compare_grouping_methods(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    month_col: str,
    group_cols: List[str],
    temporal_order: Optional[List] = None,
) -> pd.DataFrame:
    """
    Compare how well different grouping methods capture seasonal variation.
    
    Uses ratio of between-group to within-group variance.
    """
    print("  [i] Comparing grouping methods...")
    
    results = []
    
    for group_col in group_cols:
        # Calculate between-group variance
        months = temporal_order if temporal_order else metadata[month_col].unique()
        
        between_var = 0
        within_var = 0
        
        for month in months:
            month_mask = metadata[month_col] == month
            month_coords = coords[month_mask]
            month_groups = metadata.loc[month_mask, group_col]
            
            # Grand centroid for this month
            grand_centroid = month_coords.mean(axis=0)
            
            for grp in month_groups.unique():
                grp_mask = month_groups == grp
                grp_coords = month_coords[grp_mask]
                
                if len(grp_coords) < 2:
                    continue
                
                # Group centroid
                grp_centroid = grp_coords.mean(axis=0)
                
                # Between-group variance
                between_var += len(grp_coords) * np.sum((grp_centroid - grand_centroid)**2)
                
                # Within-group variance
                within_var += np.sum(np.sum((grp_coords - grp_centroid)**2, axis=1))
        
        # F-ratio (higher = better separation)
        f_ratio = (between_var / (len(metadata[group_col].unique()) - 1)) / \
                  (within_var / (len(coords) - len(metadata[group_col].unique())))
        
        results.append({
            'grouping_method': group_col,
            'n_groups': len(metadata[group_col].unique()),
            'between_group_var': between_var,
            'within_group_var': within_var,
            'f_ratio': f_ratio,
            'variance_explained': between_var / (between_var + within_var) * 100,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_seasonal_trajectories(
    coords_mean: pd.DataFrame,
    coords_std: pd.DataFrame,
    group_colors: Dict,
    group_col: str,
    output_path: Path,
    title: str = "Seasonal Community Trajectories",
    month_rank_map: Optional[Dict] = None,
) -> None:
    """
    Plot seasonal trajectories in facets, one per group.
    Each point shows the month number, with arrows showing progression.
    """
    groups = sorted(coords_mean[group_col].unique())
    n_groups = len(groups)
    
    # Determine grid layout
    n_cols = min(3, n_groups)  # Max 3 columns
    n_rows = int(np.ceil(n_groups / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    
    # Flatten axes for easy iteration
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_groups > 1 else axes
    
    # Coordinate column names
    coord_cols = [c for c in coords_mean.columns if c not in [group_col, 'month', 'n_samples']]
    x_col, y_col = coord_cols[0], coord_cols[1]
    
    # Find global axis limits for consistent scaling
    all_x = coords_mean[x_col]
    all_y = coords_mean[y_col]
    x_margin = (all_x.max() - all_x.min()) * 0.15
    y_margin = (all_y.max() - all_y.min()) * 0.15
    x_lim = [all_x.min() - x_margin, all_x.max() + x_margin]
    y_lim = [all_y.min() - y_margin, all_y.max() + y_margin]
    
    for idx, grp in enumerate(groups):
        ax = axes[idx]
        
        grp_mean = coords_mean[coords_mean[group_col] == grp].copy()
        if month_rank_map:
            grp_mean['month_order'] = grp_mean['month'].map(month_rank_map)
            grp_mean = grp_mean.sort_values('month_order')
        else:
            grp_mean = grp_mean.sort_values('month')
        
        if len(grp_mean) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(grp_mean)})', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{grp}', fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
        
        color = group_colors.get(grp, '#808080')
        
        # Plot trajectory line
        ax.plot(
            grp_mean[x_col],
            grp_mean[y_col],
            color=color,
            linewidth=3,
            alpha=0.7,
            zorder=2,
        )
        
        # Add arrows between consecutive months
        for i in range(len(grp_mean) - 1):
            # Calculate arrow direction
            dx = grp_mean[x_col].iloc[i+1] - grp_mean[x_col].iloc[i]
            dy = grp_mean[y_col].iloc[i+1] - grp_mean[y_col].iloc[i]
            
            # Place arrow at midpoint of segment
            mid_x = grp_mean[x_col].iloc[i] + dx * 0.5
            mid_y = grp_mean[y_col].iloc[i] + dy * 0.5
            
            # Arrow length (slightly shorter than full segment)
            arrow_length = np.sqrt(dx**2 + dy**2) * 0.35
            
            ax.annotate(
                '',
                xy=(mid_x + dx * 0.3, mid_y + dy * 0.3),
                xytext=(mid_x - dx * 0.2, mid_y - dy * 0.2),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=2.5,
                    alpha=0.8,
                    mutation_scale=25,
                ),
                zorder=3,
            )
        
        # Plot points with month/season labels inside
        for idx_pt, row in grp_mean.iterrows():
            month_value = row['month']
            month_label = format_temporal_label(month_value)
            x_pos = row[x_col]
            y_pos = row[y_col]
            
            # Outer circle (larger, colored)
            circle = plt.Circle(
                (x_pos, y_pos),
                radius=(x_lim[1] - x_lim[0]) * 0.025,  # Size scales with plot
                facecolor='white',
                edgecolor=color,
                linewidth=3,
                alpha=0.95,
                zorder=5,
            )
            ax.add_patch(circle)
            
            # Month number text
            ax.text(
                x_pos,
                y_pos,
                month_label,
                ha='center',
                va='center',
                fontsize=13,
                fontweight='bold',
                color=color,
                zorder=6,
            )
        
        # Styling
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(f'{x_col}', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{y_col}', fontsize=11, fontweight='bold')
        ax.set_title(f'{grp}', fontsize=13, fontweight='bold', 
                    color=color, pad=10)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_aspect('equal', adjustable='box')
        
        # Add subtle box around plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Remove extra subplots if any
    for idx in range(n_groups, len(axes)):
        fig.delaxes(axes[idx])
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved seasonal trajectory plot (faceted)")


def plot_circular_trajectories(
    coords_mean: pd.DataFrame,
    group_colors: Dict,
    group_col: str,
    output_path: Path,
    temporal_order: Optional[List] = None,
    month_rank_map: Optional[Dict] = None,
) -> None:
    """
    Plot trajectories in circular/polar coordinates to emphasize seasonality.
    """
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection='polar')
    
    coord_cols = [c for c in coords_mean.columns if c not in [group_col, 'month', 'n_samples']]
    x_col = coord_cols[0]  # Use first UMAP dimension as radius
    
    if temporal_order and month_rank_map and len(temporal_order) > 0:
        period_divisor = len(temporal_order)
        xtick_positions = np.linspace(0, 2*np.pi, period_divisor, endpoint=False)
        xtick_labels = [format_temporal_label(val) for val in temporal_order]
    else:
        period_divisor = 12
        xtick_positions = np.linspace(0, 2*np.pi, 12, endpoint=False)
        xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for grp in coords_mean[group_col].unique():
        grp_mean = coords_mean[coords_mean[group_col] == grp].sort_values('month')
        
        if len(grp_mean) < 2:
            continue
        
        color = group_colors.get(grp, '#808080')
        
        # Convert months to angles (radians)
        month_values = grp_mean['month']
        if month_rank_map:
            numeric_months = pd.to_numeric(month_values.map(month_rank_map), errors='coerce')
            divisor = max(period_divisor, 1)
            theta = numeric_months * 2 * np.pi / divisor - np.pi/2
        else:
            month_numeric = pd.to_numeric(month_values, errors='coerce')
            theta = month_numeric * 2 * np.pi / period_divisor - np.pi/2  # Start at top
        
        # Use UMAP coordinate as radius (normalized)
        r = grp_mean[x_col]
        r_normalized = (r - r.min()) / (r.max() - r.min() + 1e-10)
        
        # Plot
        ax.plot(theta, r_normalized, 'o-', color=color, linewidth=3, 
               markersize=10, alpha=0.7, label=f"{grp}")
        
        # Close the circle if we have all 12 months
        if len(grp_mean) >= 10:
            ax.plot([theta.iloc[-1], theta.iloc[0]], 
                   [r_normalized.iloc[-1], r_normalized.iloc[0]], 
                   '--', color=color, linewidth=2, alpha=0.5)
    
    # Set month labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_zero_location('N')  # Start at top
    
    ax.set_ylabel('Normalized UMAP 1', fontsize=11)
    ax.set_title('Seasonal Cycle (Circular Representation)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(title=group_col, bbox_to_anchor=(1.3, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved circular trajectory plot")


def plot_linear_trajectories(
    coords_mean: pd.DataFrame,
    group_colors: Dict,
    group_col: str,
    output_path: Path,
    temporal_order: Optional[List] = None,
    month_rank_map: Optional[Dict] = None,
) -> None:
    """
    Plot normalized UMAP position over the ordered temporal axis.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    coord_cols = [c for c in coords_mean.columns if c not in [group_col, 'month', 'n_samples']]
    x_col = coord_cols[0]
    groups = sorted(coords_mean[group_col].unique())
    for grp in groups:
        grp_mean = coords_mean[coords_mean[group_col] == grp].copy()
        if month_rank_map:
            grp_mean['order'] = grp_mean['month'].map(month_rank_map)
        else:
            grp_mean['order'] = grp_mean['month']
        grp_mean = grp_mean.dropna(subset=['order']).sort_values('order')
        if grp_mean.empty:
            continue
        values = grp_mean[x_col]
        norm_vals = (values - values.min()) / (values.max() - values.min() + 1e-10)
        color = group_colors.get(grp, '#808080')
        ax.plot(grp_mean['order'], norm_vals, marker='o', linewidth=2.5,
                markersize=7, alpha=0.8, color=color, label=str(grp))
    if month_rank_map and temporal_order:
        tick_pos = [month_rank_map.get(val) for val in temporal_order if val in month_rank_map]
        tick_labels = [format_temporal_label(val) for val in temporal_order if val in month_rank_map]
    else:
        tick_pos = list(range(1, 13))
        tick_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(min(tick_pos) - 0.5, max(tick_pos) + 0.5)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized UMAP 1', fontsize=12, fontweight='bold')
    ax.set_title('Seasonal Trajectory Profile', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title=group_col)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [✓] Saved linear seasonal trajectory plot")


def plot_trajectory_heatmap(
    coords_mean: pd.DataFrame,
    group_col: str,
    output_path: Path,
    temporal_order: Optional[List] = None,
    month_rank_map: Optional[Dict] = None,
) -> None:
    """
    Heatmap of normalized UMAP coordinates across time for each group.
    """
    coord_cols = [c for c in coords_mean.columns if c not in [group_col, 'month', 'n_samples']]
    x_col = coord_cols[0]
    df = coords_mean[[group_col, 'month', x_col]].copy()
    if month_rank_map:
        df['order'] = df['month'].map(month_rank_map)
    else:
        df['order'] = df['month']
    df = df.dropna(subset=['order'])
    df['value'] = df.groupby(group_col)[x_col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
    )
    pivot = df.pivot_table(
        index=group_col,
        columns='order',
        values='value',
        aggfunc='mean',
    )
    if temporal_order and month_rank_map:
        ordered_cols = [month_rank_map.get(val) for val in temporal_order if val in month_rank_map]
        pivot = pivot.reindex(columns=ordered_cols)
        col_labels = [format_temporal_label(val) for val in temporal_order if val in month_rank_map]
    else:
        pivot = pivot.sort_index(axis=1)
        col_labels = [format_temporal_label(col) for col in pivot.columns]
    plt.figure(figsize=(14, max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        cmap='viridis',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Normalized UMAP 1'},
    )
    plt.xlabel('Time', fontsize=12, fontweight='bold')
    plt.ylabel(group_col, fontsize=12, fontweight='bold')
    plt.xticks(ticks=np.arange(len(col_labels)) + 0.5, labels=col_labels, rotation=45, ha='right')
    plt.title('Seasonal Trajectory Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [✓] Saved seasonal trajectory heatmap")


def plot_comparison_between_groupings(
    coords_mean_dict: Dict[str, pd.DataFrame],
    metrics_dict: Dict[str, pd.DataFrame],
    comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare trajectory metrics between different grouping methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Trajectory lengths
    ax = axes[0, 0]
    data_for_plot = []
    for method, metrics in metrics_dict.items():
        for _, row in metrics.iterrows():
            data_for_plot.append({
                'Method': method,
                'Group': row['group'],
                'Value': row['trajectory_length'],
            })
    df_plot = pd.DataFrame(data_for_plot)
    
    sns.boxplot(data=df_plot, x='Method', y='Value', ax=ax, palette='Set2')
    sns.swarmplot(data=df_plot, x='Method', y='Value', ax=ax, 
                 color='black', alpha=0.5, size=6)
    ax.set_ylabel('Trajectory Length', fontweight='bold')
    ax.set_title('Trajectory Length by Grouping Method', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Directionality
    ax = axes[0, 1]
    data_for_plot = []
    for method, metrics in metrics_dict.items():
        for _, row in metrics.iterrows():
            data_for_plot.append({
                'Method': method,
                'Group': row['group'],
                'Value': row['directionality'],
            })
    df_plot = pd.DataFrame(data_for_plot)
    
    sns.boxplot(data=df_plot, x='Method', y='Value', ax=ax, palette='Set2')
    sns.swarmplot(data=df_plot, x='Method', y='Value', ax=ax,
                 color='black', alpha=0.5, size=6)
    ax.set_ylabel('Directionality', fontweight='bold')
    ax.set_title('Directionality by Grouping Method', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Variance explained
    ax = axes[1, 0]
    ax.bar(comparison_df['grouping_method'], 
          comparison_df['variance_explained'],
          color=['#66c2a5', '#fc8d62'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Variance Explained (%)', fontweight='bold')
    ax.set_xlabel('Grouping Method', fontweight='bold')
    ax.set_title('Variance Explained by Grouping Method', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(comparison_df['variance_explained']):
        ax.text(i, val, f'{val:.1f}%', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    # Plot 4: F-ratio
    ax = axes[1, 1]
    ax.bar(comparison_df['grouping_method'],
          comparison_df['f_ratio'],
          color=['#66c2a5', '#fc8d62'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('F-ratio (Between/Within Variance)', fontweight='bold')
    ax.set_xlabel('Grouping Method', fontweight='bold')
    ax.set_title('Group Separation Quality', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, val in enumerate(comparison_df['f_ratio']):
        ax.text(i, val, f'{val:.2f}', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved grouping comparison plot")


def plot_seasonal_marker_taxa(
    marker_taxa: Dict[str, pd.DataFrame],
    asv_data: pd.DataFrame,
    metadata: pd.DataFrame,
    month_col: str,
    group_col: str,
    group_colors: Dict,
    output_path: Path,
    top_n: int = 5,
    temporal_order: Optional[List] = None,
    month_rank_map: Optional[Dict] = None,
) -> None:
    """
    Plot seasonal patterns of marker taxa, averaging across years.
    """
    n_groups = len(marker_taxa)
    if n_groups == 0:
        print("  [!] No marker taxa available for plotting.")
        return
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, 5 * n_groups))
    
    if n_groups == 1:
        axes = [axes]
    
    for ax, (grp, markers) in zip(axes, marker_taxa.items()):
        top_markers = markers.head(top_n)
        
        # Get samples for this group
        mask = metadata[group_col] == grp
        grp_meta = metadata.loc[mask].copy()
        grp_asv = asv_data.loc[mask]
        
        # Aggregate by month (average across years)
        for idx, row in top_markers.iterrows():
            asv = row['asv']
            direction = row['direction']
            corr = row['correlation']
            
            # Calculate monthly means
            monthly_data = []
            periods = temporal_order if temporal_order else sorted(metadata[month_col].dropna().unique())
            for period in periods:
                month_mask = grp_meta[month_col] == period
                if month_mask.sum() > 0:
                    order_value = month_rank_map.get(period, np.nan) if month_rank_map else period
                    if pd.isna(order_value):
                        continue
                    mean_val = grp_asv.loc[month_mask, asv].mean()
                    std_val = grp_asv.loc[month_mask, asv].std()
                    monthly_data.append({
                        'period_order': order_value,
                        'label': period,
                        'mean': mean_val,
                        'std': std_val,
                    })
            
            df_monthly = pd.DataFrame(monthly_data)
            if 'period_order' in df_monthly.columns:
                df_monthly = df_monthly.sort_values('period_order')
            
            linestyle = '-' if direction == 'increase' else '--'
            label = f"{asv[:30]}... ({direction}, ρ={corr:.2f})"
            
            ax.plot(df_monthly['period_order'], df_monthly['mean'],
                   marker='o', linestyle=linestyle, linewidth=2,
                   markersize=8, alpha=0.7, label=label)
            
            # Add error bars
            ax.fill_between(df_monthly['period_order'],
                           df_monthly['mean'] - df_monthly['std'],
                           df_monthly['mean'] + df_monthly['std'],
                           alpha=0.2)
        
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('CLR Abundance (mean ± SD)', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Seasonal Marker Taxa - Group: {grp}',
                    fontsize=12, fontweight='bold', pad=10)
        if temporal_order and month_rank_map:
            tick_positions = [month_rank_map.get(val) for val in temporal_order]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([format_temporal_label(val) for val in temporal_order])
        else:
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved seasonal marker taxa plot")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seasonal trajectory analysis comparing grouping methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input files
    parser.add_argument("--umap-results", type=Path, required=True,
                        help="Path to umap_hdbscan_results.tsv")
    parser.add_argument("--asv-data", type=Path, required=True,
                        help="Path to CLR-transformed ASV data")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to metadata TSV")
    parser.add_argument("--cluster-color-table", type=Path,
                        help="Optional TSV (e.g., compartment_umap_clusters.tsv) containing columns sampleid, cluster, cluster_color, etc. "
                             "Values will override matching metadata columns.")
    
    # Column specifications
    parser.add_argument("--month-col", required=True,
                        help="Month column name (1-12)")
    parser.add_argument("--group-cols", required=True,
                        help="Comma-separated grouping columns (e.g., 'Depth,cluster_after'). "
                             "Note: cluster_before/cluster_after will be loaded from UMAP results.")
    parser.add_argument("--color-col", default="Color",
                        help="Column with color codes (for first grouping)")
    
    # Analysis parameters
    parser.add_argument("--top-taxa", type=int, default=20,
                        help="Number of top marker taxa per group")
    
    # Output
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    group_cols = [c.strip() for c in args.group_cols.split(",")]
    
    if args.verbose:
        print("\n" + "="*70)
        print("SEASONAL TRAJECTORY ANALYSIS")
        print("="*70)
        print(f"Grouping methods: {group_cols}")
        print(f"Temporal axis: {args.month_col}")
        print(f"Years treated as: Pseudo-replicates")
        print("="*70 + "\n")
    
    # ========================================================================
    # Load data and merge cluster info
    # ========================================================================
    
    print("[1/8] Loading data...")
    umap_df = pd.read_csv(args.umap_results, sep="\t", index_col=0)
    asv_data = pd.read_csv(args.asv_data, sep="\t", index_col=0)
    metadata_df = pd.read_csv(args.metadata, sep="\t")
    if SAMPLE_ID_COL not in metadata_df.columns:
        raise ValueError(f"Metadata column '{SAMPLE_ID_COL}' not found in {args.metadata}")
    metadata = metadata_df.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(SAMPLE_ID_COL)

    # Optional: merge in cluster color/assignment overrides
    if args.cluster_color_table:
        cluster_table_path = args.cluster_color_table
        cluster_df = pd.read_csv(cluster_table_path, sep="\t")
        if SAMPLE_ID_COL not in cluster_df.columns:
            raise ValueError(f"Cluster color table {cluster_table_path} must contain '{SAMPLE_ID_COL}'")
        cluster_df = cluster_df.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(SAMPLE_ID_COL)
        overlap_idx = metadata.index.intersection(cluster_df.index)
        if overlap_idx.empty:
            raise ValueError("No overlapping samples found between metadata and cluster color table.")
        for col in cluster_df.columns:
            metadata.loc[overlap_idx, col] = cluster_df.loc[overlap_idx, col]
    
    print(f"  UMAP data: {umap_df.shape}")
    print(f"  ASV data: {asv_data.shape}")
    print(f"  Metadata: {metadata.shape}")
    
    # Merge cluster information from UMAP results into metadata
    print("\n  [i] Merging cluster information from UMAP results...")
    cluster_cols = ['compartment_cluster']  # columns that might be in UMAP results
    available_cols = [col for col in cluster_cols if col in umap_df.columns]
    
    if available_cols:
        # Add columns to metadata (only if not already there)
        for col in available_cols:
            if col not in metadata.columns:
                metadata[col] = umap_df[col]
                print(f"      ✓ Added '{col}' to metadata")
            else:
                print(f"      - '{col}' already in metadata, skipping")
    
    # Align all data
    common_samples = umap_df.index.intersection(asv_data.index).intersection(metadata.index)
    umap_df = umap_df.loc[common_samples]
    asv_data = asv_data.loc[common_samples]
    metadata = metadata.loc[common_samples]
    
    print(f"\n  Common samples: {len(common_samples)}")
    
    # Verify all grouping columns exist
    print(f"\n  [i] Verifying grouping columns...")
    for group_col in group_cols:
        if group_col not in metadata.columns:
            print(f"\n  [!] ERROR: Grouping column '{group_col}' not found!")
            print(f"  Available columns in metadata: {list(metadata.columns)}")
            print(f"  Available columns in UMAP results: {list(umap_df.columns)}")
            raise ValueError(f"Grouping column '{group_col}' not found")
        
        metadata[group_col] = normalize_group_labels(metadata[group_col])
        unique_vals = sorted([val for val in metadata[group_col].dropna().unique()])
        n_groups = len(unique_vals)
        print(f"      ✓ {group_col}: {n_groups} unique groups {unique_vals}")
    
    # Check month column and standardize ordering
    if args.month_col not in metadata.columns:
        print(f"\n  [!] ERROR: Month column '{args.month_col}' not found!")
        print(f"  Available columns: {list(metadata.columns)}")
        raise ValueError(f"Month column '{args.month_col}' not found")
    temporal_order, month_rank_map = standardize_temporal_column(metadata, args.month_col)
    months = [int(m) for m in (temporal_order if temporal_order else sorted(metadata[args.month_col].dropna().unique()))]
    print(f"\n  ✓ Temporal order: {months} ({len(months)} total)")
    
    # Get UMAP coordinates (after correction)
    coords = umap_df[['umap1', 'umap2']]
    coords_df = coords.copy()
    coords_df.columns = ['UMAP1', 'UMAP2']
        
    # Process each grouping method
    print("[2/8] Aggregating data by month...")
    
    coords_mean_dict = {}
    coords_std_dict = {}
    count_dict = {}
    metrics_dict = {}
    marker_taxa_dict = {}
    
    for group_col in group_cols:
        print(f"\n  Processing grouping: {group_col}")
        
        # Aggregate coordinates
        mean_data, std_data, count_data = aggregate_monthly_data(
            coords_df,
            metadata,
            args.month_col,
            group_col,
            temporal_order=temporal_order,
        )
        
        coords_mean_dict[group_col] = mean_data
        coords_std_dict[group_col] = std_data
        count_dict[group_col] = count_data
        
        print(f"    Groups: {mean_data['group'].unique()}")
        covered = [val for val in temporal_order if val in mean_data['month'].unique()] if temporal_order else sorted(mean_data['month'].unique())
        print(f"    Months covered: {covered}")
    
    # Calculate metrics for each grouping
    print("[3/8] Calculating trajectory metrics...")
    
    for group_col in group_cols:
        metrics = calculate_seasonal_trajectory_metrics(
            coords_mean_dict[group_col],
            coords_std_dict[group_col],
            month_rank_map,
            group_col='group',
        )
        metrics_dict[group_col] = metrics
        
        # Save
        metrics.to_csv(out_dir / f"trajectory_metrics_{group_col}.tsv", 
                      sep='\t', index=False)
    
    # Compare grouping methods
    print("[4/8] Comparing grouping methods...")
    
    comparison_df = compare_grouping_methods(
        coords.values,
        metadata,
        args.month_col,
        group_cols,
        temporal_order,
    )
    comparison_df.to_csv(out_dir / "grouping_comparison.tsv", sep='\t', index=False)
    
    # Identify marker taxa for each grouping
    print("[5/8] Identifying seasonal marker taxa...")
    
    for group_col in group_cols:
        markers = identify_seasonal_marker_taxa(
            asv_data,
            metadata,
            args.month_col,
            group_col,
            top_n=args.top_taxa,
            temporal_order=temporal_order,
            month_rank_map=month_rank_map,
        )
        marker_taxa_dict[group_col] = markers
        
        # Save
        for grp, marker_df in markers.items():
            filename = f"marker_taxa_{group_col}_{grp}.tsv"
            marker_df.to_csv(out_dir / filename, sep='\t', index=False)
    
    # Get colors
    print("[6/8] Preparing colors...")
    
    color_dict = {}
    for group_col in group_cols:
        groups = metadata[group_col].unique()

        # Determine explicit color column priority:
        # 1. Column specific to group (e.g., cluster -> cluster_color)
        # 2. User-specified args.color_col (only applied to first grouping as legacy behavior)
        # 3. Auto-generated palette fallback
        specific_color_col = f"{group_col}_color"
        if specific_color_col in metadata.columns:
            group_colors = (
                metadata[[group_col, specific_color_col]]
                .dropna()
                .drop_duplicates(subset=[group_col])
                .set_index(group_col)[specific_color_col]
                .to_dict()
            )
            # Ensure all groups covered
            missing_groups = set(groups) - set(group_colors.keys())
            if missing_groups:
                palette = sns.color_palette("tab10", len(missing_groups))
                for grp, color in zip(sorted(missing_groups), palette):
                    group_colors[grp] = plt.matplotlib.colors.rgb2hex(color)
        elif group_col == group_cols[0] and args.color_col in metadata.columns:
            group_colors = dict(zip(metadata[group_col], metadata[args.color_col]))
        else:
            # Auto-generate
            palette = sns.color_palette("tab10", len(groups))
            group_colors = dict(zip(sorted(groups), 
                                  [plt.matplotlib.colors.rgb2hex(c) for c in palette]))
        
        color_dict[group_col] = group_colors
    
    # Consolidate summary for each grouping (primary also saved as trajectory_summary.tsv)
    summary_written = False
    for idx, group_col in enumerate(group_cols):
        if group_col not in metrics_dict:
            continue
        summary_df = metrics_dict[group_col].copy()
        summary_df['group_col'] = group_col
        if group_col in color_dict:
            summary_df['group_color'] = summary_df['group'].map(color_dict[group_col]).fillna('#808080')
        else:
            summary_df['group_color'] = '#808080'
        if group_col in marker_taxa_dict:
            marker_tables = marker_taxa_dict[group_col]
            def summarize_markers(group_value, tables=marker_tables) -> str:
                if group_value in tables and not tables[group_value].empty:
                    return ','.join(tables[group_value]['asv'].astype(str).head(5))
                # Fallback if group_value stored as string or numeric mismatch
                str_key = str(group_value)
                for key in tables:
                    if str(key) == str_key and not tables[key].empty:
                        return ','.join(tables[key]['asv'].astype(str).head(5))
                return ''
            summary_df['top_markers'] = summary_df['group'].apply(summarize_markers)
        else:
            summary_df['top_markers'] = ''
        summary_path = out_dir / f"trajectory_summary_{group_col}.tsv"
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"[i] Saved trajectory summary: {summary_path.name}")
        if idx == 0:
            summary_generic = out_dir / "trajectory_summary.tsv"
            summary_df.to_csv(summary_generic, sep='\t', index=False)
            print(f"[i] Updated primary summary copy: trajectory_summary.tsv")
            summary_written = True
    if not summary_written:
        print("[!] No trajectory summary tables written (missing metrics).")

    # Create visualizations
    print("[7/8] Creating visualizations...")
    
    for group_col in group_cols:
        print(f"\n  Plotting {group_col}...")
        
        # Seasonal trajectories
        plot_seasonal_trajectories(
            coords_mean_dict[group_col],
            coords_std_dict[group_col],
            color_dict[group_col],
            'group',
            out_dir / f"seasonal_trajectories_{group_col}.pdf",
            title=f"Seasonal Trajectories - {group_col}",
            month_rank_map=month_rank_map,
        )
        
        # Circular plot
        plot_circular_trajectories(
            coords_mean_dict[group_col],
            color_dict[group_col],
            'group',
            out_dir / f"circular_trajectories_{group_col}.pdf",
            temporal_order=temporal_order,
            month_rank_map=month_rank_map,
        )
        
        # Linear profile
        plot_linear_trajectories(
            coords_mean_dict[group_col],
            color_dict[group_col],
            'group',
            out_dir / f"linear_trajectories_{group_col}.pdf",
            temporal_order=temporal_order,
            month_rank_map=month_rank_map,
        )
        
        # Heatmap summary
        plot_trajectory_heatmap(
            coords_mean_dict[group_col],
            'group',
            out_dir / f"trajectory_heatmap_{group_col}.pdf",
            temporal_order=temporal_order,
            month_rank_map=month_rank_map,
        )
        
        # Marker taxa
        plot_seasonal_marker_taxa(
            marker_taxa_dict[group_col],
            asv_data,
            metadata,
            args.month_col,
            group_col,
            color_dict[group_col],
            out_dir / f"seasonal_marker_taxa_{group_col}.pdf",
            top_n=5,
            temporal_order=temporal_order,
            month_rank_map=month_rank_map,
        )
    
    # Comparison plot
    plot_comparison_between_groupings(
        coords_mean_dict,
        metrics_dict,
        comparison_df,
        out_dir / "grouping_comparison.pdf",
    )
    
    # Summary report
    print("[8/8] Generating summary...")
    
    summary = []
    summary.append("="*70)
    summary.append("SEASONAL TRAJECTORY ANALYSIS SUMMARY")
    summary.append("="*70)
    summary.append(f"\nAnalyzed {len(common_samples)} samples")
    summary.append(f"Temporal order: {months}")
    summary.append(f"\nGrouping methods compared: {group_cols}")
    
    summary.append("\n\nGROUPING METHOD COMPARISON:")
    summary.append(comparison_df.to_string(index=False))
    
    for group_col in group_cols:
        summary.append(f"\n\nTRAJECTORY METRICS - {group_col}:")
        summary.append(metrics_dict[group_col].to_string(index=False))
        
        summary.append(f"\n\nMARKER TAXA SUMMARY - {group_col}:")
        for grp, markers in marker_taxa_dict[group_col].items():
            summary.append(f"\n  Group {grp}: {len(markers)} significant markers")
    
    summary.append("\n" + "="*70)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open(out_dir / "seasonal_analysis_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput: {out_dir}")
    print("\nKey insight: Check grouping_comparison.pdf to see which method")
    print("(Depth vs Compartment) better captures seasonal variation!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
