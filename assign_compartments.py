#!/usr/bin/env python3
"""
figure1_depth_stratification.py

Recreate Figure 1-style depth stratification analysis:
- Hierarchical clustering dendrogram with environmental heatmap
- UMAP ordinations colored by depth and cluster
- ASV richness by depth
- GC content calculation from ASV sequences

Usage:
  python figure1_depth_stratification.py \
    --asv-clr spark_combined_output/batch_correction/asv_clr_after_correction.tsv \
    --asv-counts spark_combined_output/combined_asv_table.tsv \
    --asv-fasta spark_combined_output/combined_asv_seqs.fasta \
    --metadata spark_combined_output/metadata/metadata_updated.tsv \
    --depth-col Depth \
    --month-col Month \
    --biochem-cols "Temperature,Oxygen,Phosphate,Silicate,Nitrogen Oxides,Nitrate,Nitrite,Ammonium,Hydrogen Sulfide,Iron,Methane,Nitrous Oxide,Dimethyl Sulfide,Fluorescence,PAR/Irradiance" \
    --output-dir spark_combined_output/figure1_stratification \
    --verbose
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import umap
from Bio import SeqIO
from matplotlib.colors import to_hex, ListedColormap, BoundaryNorm
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 150,
})
sns.set_style("white")

SAMPLE_ID_COL = 'sampleid'
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
BIOCHEM_ALLOWED_VARS = list(BIOCHEM_COLOR_MAP.keys())


# ============================================================================
# GC Content Calculation
# ============================================================================

def calculate_gc_content_from_fasta(
    fasta_path: Path,
) -> pd.Series:
    """
    Calculate GC content (%) for each ASV sequence.
    
    Returns
    -------
    pd.Series with ASV IDs as index and GC% as values
    """
    print("  [i] Calculating GC content from FASTA...")
    
    gc_content = {}
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        
        # Count G and C
        gc_count = seq.count('G') + seq.count('C')
        total = len(seq)
        
        if total > 0:
            gc_pct = (gc_count / total) * 100
            gc_content[record.id.split(';', 1)[0]] = gc_pct
    
    gc_series = pd.Series(gc_content)
    
    print(f"      Calculated GC% for {len(gc_series)} ASVs")
    print(f"      Mean GC%: {gc_series.mean():.2f}%")
    print(f"      Range: {gc_series.min():.2f}% - {gc_series.max():.2f}%")
    
    return gc_series


def calculate_sample_weighted_gc(
    asv_clr: pd.DataFrame,
    gc_content: pd.Series,
) -> pd.Series:
    """
    Calculate community-weighted mean GC content for each sample.
    
    Parameters
    ----------
    asv_clr : DataFrame
        CLR-transformed ASV abundances (samples x ASVs)
    gc_content : Series
        GC% for each ASV
    
    Returns
    -------
    Series with sample-level weighted mean GC%
    """
    print("  [i] Calculating sample-weighted mean GC content...")
    
    # Get common ASVs
    common_asvs = asv_clr.columns.intersection(gc_content.index)
    
    print(f"      ASVs in abundance table: {len(asv_clr.columns)}")
    print(f"      ASVs with GC data: {len(gc_content)}")
    print(f"      Common ASVs: {len(common_asvs)}")
    
    # Subset to common ASVs
    asv_clr_subset = asv_clr[common_asvs]
    gc_subset = gc_content[common_asvs]
    
    # Convert CLR to relative abundances (softmax-like transformation)
    # CLR is log-ratio, so exp() then normalize
    asv_exp = np.exp(asv_clr_subset)
    asv_rel = asv_exp.div(asv_exp.sum(axis=1), axis=0)
    
    # Calculate weighted mean GC for each sample
    sample_gc = (asv_rel * gc_subset).sum(axis=1)
    
    print(f"      Sample GC% range: {sample_gc.min():.2f}% - {sample_gc.max():.2f}%")
    
    return sample_gc


# ============================================================================
# Hierarchical Clustering and Cluster Assignment
# ============================================================================


def determine_optimal_clusters(
    asv_clr: pd.DataFrame,
    k_range: range,
    output_path: Optional[Path] = None,
) -> int:
    """
    Determine optimal number of clusters using multiple metrics.
    
    Metrics used:
    - Silhouette score (higher = better separation)
    - Calinski-Harabasz score (higher = better defined clusters)
    - Davies-Bouldin score (lower = better separation)
    - Dendrogram elbow (visual inspection)
    
    Returns
    -------
    optimal_k : int
        Recommended number of clusters
    """
    print(f"  [i] Evaluating cluster quality for k={k_range.start} to {k_range.stop-1}...")
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Calculate linkage matrix once
    linkage_matrix = linkage(asv_clr, method='ward', metric='euclidean')
    
    # Evaluate different k values
    results = []
    
    for k in k_range:
        # Get cluster labels
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        
        # Calculate metrics
        silhouette = silhouette_score(asv_clr, labels)
        calinski = calinski_harabasz_score(asv_clr, labels)
        davies_bouldin = davies_bouldin_score(asv_clr, labels)
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin,
        })
        
        print(f"      k={k}: Silhouette={silhouette:.3f}, "
              f"Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # ========================================================================
    # Determine optimal k using consensus ranking
    # ========================================================================
    
    # Rank each metric (lower rank = better)
    results_df['silhouette_rank'] = results_df['silhouette'].rank(ascending=False)
    results_df['calinski_rank'] = results_df['calinski_harabasz'].rank(ascending=False)
    results_df['davies_bouldin_rank'] = results_df['davies_bouldin'].rank(ascending=True)  # Lower is better
    
    # Mean rank (lower = better)
    results_df['mean_rank'] = results_df[['silhouette_rank', 'calinski_rank', 'davies_bouldin_rank']].mean(axis=1)
    
    # Best k by mean rank
    best_k = results_df.loc[results_df['mean_rank'].idxmin(), 'k']
    
    print(f"\n  [✓] Optimal k={int(best_k)} (by consensus ranking)")
    
    # ========================================================================
    # Create diagnostic plot
    # ========================================================================
    
    if output_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Silhouette score
        ax = axes[0, 0]
        ax.plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2, markersize=8)
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax.set_title('Silhouette Score (higher = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 2: Calinski-Harabasz score
        ax = axes[0, 1]
        ax.plot(results_df['k'], results_df['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='green')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
        ax.set_title('Calinski-Harabasz Score (higher = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 3: Davies-Bouldin score
        ax = axes[1, 0]
        ax.plot(results_df['k'], results_df['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Score', fontsize=11, fontweight='bold')
        ax.set_title('Davies-Bouldin Score (lower = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 4: Mean rank (consensus)
        ax = axes[1, 1]
        ax.plot(results_df['k'], results_df['mean_rank'], 'o-', linewidth=2, markersize=8, color='purple')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Rank', fontsize=11, fontweight='bold')
        ax.set_title('Consensus Ranking (lower = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  [✓] Saved cluster evaluation plot")
    
    return int(best_k), results_df, linkage_matrix


def perform_hierarchical_clustering(
    asv_clr: pd.DataFrame,
    n_clusters: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering on ASV data.
    
    Returns
    -------
    linkage_matrix : array
        Hierarchical clustering linkage matrix
    cluster_labels : array
        Cluster assignment for each sample
    """
    print(f"  [i] Performing hierarchical clustering (k={n_clusters})...")
    
    # Calculate linkage matrix
    linkage_matrix = linkage(asv_clr, method='ward', metric='euclidean')
    
    # Cut tree to get clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Make labels 0-indexed
    cluster_labels = cluster_labels - 1
    
    print(f"      Created {n_clusters} clusters")
    for i in range(n_clusters):
        n = (cluster_labels == i).sum()
        print(f"        Cluster {i}: {n} samples")
    
    return linkage_matrix, cluster_labels


# ============================================================================
# Distance and Richness Calculations
# ============================================================================

def calculate_between_depth_distances(
    asv_clr: pd.DataFrame,
    metadata: pd.DataFrame,
    depth_col: str,
) -> pd.DataFrame:
    """
    Calculate Euclidean distances between adjacent depth levels.
    """
    print("  [i] Calculating between-depth distances...")
    
    depths = sorted(metadata[depth_col].unique())
    
    results = []
    
    for i in range(len(depths) - 1):
        depth1 = depths[i]
        depth2 = depths[i + 1]
        
        # Get samples at each depth
        samples1 = metadata[metadata[depth_col] == depth1].index
        samples2 = metadata[metadata[depth_col] == depth2].index
        
        # Get ASV data
        data1 = asv_clr.loc[samples1]
        data2 = asv_clr.loc[samples2]
        
        # Calculate pairwise distances
        for s1 in samples1:
            for s2 in samples2:
                dist = np.linalg.norm(asv_clr.loc[s1] - asv_clr.loc[s2])
                results.append({
                    'depth1': depth1,
                    'depth2': depth2,
                    'interval': f"{depth1}-{depth2}m",
                    'distance': dist,
                })
    
    return pd.DataFrame(results)


def calculate_richness(
    asv_counts: pd.DataFrame,
) -> pd.Series:
    """
    Calculate ASV richness (number of non-zero ASVs) per sample.
    """
    print("  [i] Calculating ASV richness...")
    
    richness = (asv_counts > 0).sum(axis=1)
    
    print(f"      Richness range: {richness.min()} - {richness.max()} ASVs")
    
    return richness


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_umap_by_depth(
    asv_clr: pd.DataFrame,
    metadata: pd.DataFrame,
    depth_col: str,
    output_path: Path,
) -> np.ndarray:
    """
    UMAP ordination colored by depth (Panel c).
    
    Returns UMAP coordinates for reuse.
    """
    print("  [i] Creating UMAP colored by depth...")
    
    # Perform UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(asv_clr)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by depth
    if 'Color' in metadata.columns:
        depth_color_map = dict(zip(metadata[depth_col], metadata['Color']))
    else:
        depths = sorted(metadata[depth_col].unique())
        palette = sns.color_palette("tab10", len(depths))
        depth_color_map = dict(zip(depths, palette))
    
    for depth in sorted(metadata[depth_col].unique()):
        mask = metadata[depth_col] == depth
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[depth_color_map[depth]],
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"{depth}m",
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Community Composition by Depth', fontsize=13, fontweight='bold', pad=15)
    ax.legend(title=depth_col, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved UMAP by depth")
    
    return coords


def plot_umap_by_cluster(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_color_map: Dict[int, str],
    output_path: Path,
) -> None:
    """
    UMAP ordination colored by cluster (Panel d).
    """
    print("  [i] Creating UMAP colored by cluster...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cluster_color_map.get(cluster_id, '#808080')],
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"Cluster {cluster_id}",
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Community Composition by Depth Cluster', fontsize=13, fontweight='bold', pad=15)
    ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved UMAP by cluster")


def plot_richness_by_depth(
    richness: pd.Series,
    metadata: pd.DataFrame,
    depth_col: str,
    output_path: Path,
) -> None:
    """
    ASV richness boxplot by depth (Panel e).
    """
    print("  [i] Creating richness plot by depth...")
    
    # Combine richness with metadata
    plot_data = pd.DataFrame({
        'richness': richness,
        'depth': metadata.loc[richness.index, depth_col],
    })
    
    fig, ax = plt.subplots(figsize=(6, 10))
    
    # Color by depth
    if 'Color' in metadata.columns:
        depth_color_map = dict(zip(metadata[depth_col], metadata['Color']))
        palette = [depth_color_map[d] for d in sorted(plot_data['depth'].unique())]
    else:
        palette = sns.color_palette("tab10", len(plot_data['depth'].unique()))
    
    # Boxplot
    sns.boxplot(
        data=plot_data,
        x='richness',
        y='depth',
        palette=palette,
        ax=ax,
        linewidth=1.5,
        orient='h'
    )
    
    ax.set_xlabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ASV Richness', fontsize=12, fontweight='bold')
    ax.set_title('ASV Richness by Depth', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved richness plot")


def plot_between_depth_distances(
    distance_df: pd.DataFrame,
    metadata: pd.DataFrame,
    depth_col: str,
    output_path: Path,
) -> None:
    """
    Between-depth distance boxplot (vertical orientation).
    
    - Y-axis: Depth intervals (shallow to deep, top to bottom)
    - Box border color: deeper depth
    - Box fill color: shallower depth
    """
    print("  [i] Creating between-depth distance plot...")
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Get depth color mapping
    if 'Color' in metadata.columns:
        depth_color_map = dict(zip(metadata[depth_col], metadata['Color']))
    else:
        depths = sorted(metadata[depth_col].unique())
        palette = sns.color_palette("tab10", len(depths))
        depth_color_map = dict(zip(depths, [plt.matplotlib.colors.rgb2hex(c) for c in palette]))
    
    # Get unique intervals in order (shallow to deep)
    intervals = distance_df['interval'].unique()
    
    # Sort intervals by first depth
    interval_depths = []
    for interval in intervals:
        depth1 = distance_df[distance_df['interval'] == interval]['depth1'].iloc[0]
        interval_depths.append((interval, depth1))
    
    interval_depths.sort(key=lambda x: x[1])
    sorted_intervals = [x[0] for x in interval_depths]
    
    # Create horizontal boxplot data
    box_data = []
    fill_colors = []
    edge_colors = []
    
    for interval in sorted_intervals:
        data = distance_df[distance_df['interval'] == interval]['distance'].values
        box_data.append(data)
        
        # Get depths for this interval
        depth1 = distance_df[distance_df['interval'] == interval]['depth1'].iloc[0]
        depth2 = distance_df[distance_df['interval'] == interval]['depth2'].iloc[0]
        
        # Fill = shallower depth, Edge = deeper depth
        fill_colors.append(depth_color_map[depth1])
        edge_colors.append(depth_color_map[depth2])
    
    # Create horizontal boxplot
    bp = ax.boxplot(
        box_data,
        vert=False,  # Horizontal orientation
        labels=sorted_intervals,
        patch_artist=True,
        widths=0.6,
    )
    
    # Color boxes
    for patch, fill_color, edge_color in zip(bp['boxes'], fill_colors, edge_colors):
        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(2)
        patch.set_alpha(0.7)
    
    # Color other elements to match edge (deeper depth)
    for whisker, cap, median, flier, edge_color in zip(
        bp['whiskers'][::2], bp['caps'][::2], bp['medians'], bp['fliers'], edge_colors
    ):
        whisker.set_color(edge_color)
        whisker.set_linewidth(1.5)
        bp['whiskers'][bp['whiskers'].index(whisker) + 1].set_color(edge_color)
        bp['whiskers'][bp['whiskers'].index(whisker) + 1].set_linewidth(1.5)
        
        cap.set_color(edge_color)
        cap.set_linewidth(1.5)
        bp['caps'][bp['caps'].index(cap) + 1].set_color(edge_color)
        bp['caps'][bp['caps'].index(cap) + 1].set_linewidth(1.5)
        
        median.set_color('black')
        median.set_linewidth(2)
        
        flier.set_markeredgecolor(edge_color)
        flier.set_markerfacecolor(edge_color)
        flier.set_alpha(0.5)
    
    ax.set_xlabel('Euclidean Distance (CLR-transformed ASVs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth Interval (m)', fontsize=12, fontweight='bold')
    ax.set_title('Between-Depth Community Distances\n(Fill=Shallow, Border=Deep)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Invert y-axis so shallow is on top
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved between-depth distances plot")


def plot_compartment_depth_time_section(
    metadata: pd.DataFrame,
    depth_col: str,
    month_col: str,
    cluster_labels: np.ndarray,
    cluster_color_map: Dict[int, str],
    output_path: Path,
) -> None:
    """
    Create a depth vs. time contour-style plot of compartment assignments.
    """
    print("  [i] Creating depth-time compartment section...")

    section_df = metadata[[depth_col, month_col]].copy()
    section_df['cluster'] = cluster_labels
    section_df = section_df.dropna(subset=[depth_col, month_col, 'cluster'])

    section_df[depth_col] = pd.to_numeric(section_df[depth_col], errors='coerce')
    section_df[month_col] = pd.to_numeric(section_df[month_col], errors='coerce')
    section_df = section_df.dropna(subset=[depth_col, month_col])

    if section_df.empty:
        print("      [!] No valid depth-time data available for contour plot.")
        return

    depths = section_df[depth_col].values.astype(float)
    times = section_df[month_col].values.astype(float)

    unique_clusters = sorted(section_df['cluster'].unique())
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    idx_to_cluster = {i: c for c, i in cluster_to_idx.items()}

    mapped_values = section_df['cluster'].map(cluster_to_idx).values.astype(float)

    grid_time = np.linspace(times.min(), times.max(), max(len(np.unique(times)) * 5, 100))
    grid_depth = np.linspace(depths.min(), depths.max(), 300)
    grid_T, grid_D = np.meshgrid(grid_time, grid_depth)

    grid_clusters = griddata(
        np.column_stack([times, depths]),
        mapped_values,
        (grid_T, grid_D),
        method='nearest',
    )

    cmap_colors = [cluster_color_map.get(idx_to_cluster[i], '#808080') for i in range(len(unique_clusters))]
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(np.arange(len(unique_clusters) + 1) - 0.5, len(unique_clusters))

    fig, ax = plt.subplots(figsize=(14, 6))
    pcm = ax.pcolormesh(
        grid_time,
        grid_depth,
        grid_clusters,
        cmap=cmap,
        norm=norm,
        shading='auto',
        alpha=0.9,
    )

    scatter_colors = [cluster_color_map.get(cl, '#808080') for cl in section_df['cluster']]
    ax.scatter(times, depths, c=scatter_colors, s=20, edgecolor='k', linewidth=0.2, alpha=0.8)

    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title('Compartment Assignments Across Depth and Time', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.2, linestyle='--')

    cbar = fig.colorbar(pcm, ax=ax, ticks=np.arange(len(unique_clusters)))
    cbar.ax.set_yticklabels([str(idx_to_cluster[i]) for i in range(len(unique_clusters))])
    cbar.set_label('Compartment Cluster', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [✓] Saved compartment depth-time section: {output_path.name}")


def plot_biochem_depth_profiles(
    metadata: pd.DataFrame,
    depth_col: str,
    variables: List[str],
    output_path: Path,
) -> None:
    """
    Plot depth profiles for specified biochemical variables with variance shading.
    """
    available_vars = [var for var in variables if var in metadata.columns]
    if not available_vars:
        print("  [!] Skipping biochem depth profiles (no variables available).")
        return

    n_vars = len(available_vars)
    ncols = 2 if n_vars > 1 else 1
    nrows = int(np.ceil(n_vars / ncols))
    fig_height = max(4.8 * nrows, 7)
    fig_width = max(5.0 * ncols, 5.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, var in zip(axes, available_vars):
        data = metadata[[depth_col, var]].dropna()
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
        ax.grid(alpha=0.25, linestyle='--')
        if not data[depth_col].empty:
            ax.set_ylim(data[depth_col].max(), data[depth_col].min())
    for ax in axes[n_vars:]:
        ax.set_visible(False)
    axes[0].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] Saved biochem depth profiles: {output_path.name}")


def run_biochem_only_outputs(
    biochem_scaled: pd.DataFrame,
    kept_biochem_cols: List[str],
    metadata: pd.DataFrame,
    sample_gc: pd.Series,
    args,
    out_dir: Path,
) -> None:
    """
    Generate full set of outputs using only biochemical variables.
    """
    if biochem_scaled.empty or not kept_biochem_cols:
        print("\n[!] Skipping biochem-only analysis (no valid biochemical data).")
        return
    print("\n[9/9] Running biochem-only compartment analysis...")
    bio_dir = out_dir / "biochem_only"
    bio_dir.mkdir(exist_ok=True)

    optimal_k, cluster_eval_df, linkage_matrix = determine_optimal_clusters_euclidean(
        biochem_scaled,
        metadata,
        args.depth_col,
        k_range=range(2, 11),
        linkage_method='average',
        output_path=bio_dir / "cluster_evaluation_biochem.png",
    )
    cluster_eval_df.to_csv(bio_dir / "cluster_evaluation_biochem.tsv", sep='\t', index=False)

    cluster_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1
    unique_clusters = sorted(np.unique(cluster_labels))
    palette = sns.color_palette("Set3", len(unique_clusters))
    cluster_color_map = {cluster: to_hex(palette[i]) for i, cluster in enumerate(unique_clusters)}

    plot_dendrogram_with_heatmap_euclidean(
        biochem_scaled,
        metadata,
        linkage_matrix,
        cluster_labels,
        sample_gc,
        kept_biochem_cols,
        args.depth_col,
        bio_dir / "figure1_panel_ab_dendrogram_heatmap_biochem.png",
    )

    umap_coords = plot_umap_by_depth_euclidean(
        biochem_scaled,
        metadata,
        args.depth_col,
        bio_dir / "figure1_panel_c_umap_by_depth_biochem.png",
    )

    plot_umap_by_cluster_euclidean(
        umap_coords,
        cluster_labels,
        cluster_color_map,
        bio_dir / "figure1_panel_d_umap_by_cluster_biochem.png",
    )

    biochem_cluster_table = pd.DataFrame({
        SAMPLE_ID_COL: biochem_scaled.index,
        'compartment_cluster': cluster_labels,
        args.depth_col: metadata.loc[biochem_scaled.index, args.depth_col].values,
        'compartment_color': [cluster_color_map.get(lbl, '#808080') for lbl in cluster_labels],
        'umap1': umap_coords[:, 0],
        'umap2': umap_coords[:, 1],
    })
    biochem_cluster_table.to_csv(bio_dir / "compartment_umap_clusters_biochem.tsv", sep='\t', index=False)

    plot_compartment_depth_time_section(
        metadata,
        args.depth_col,
        args.month_col,
        cluster_labels,
        cluster_color_map,
        bio_dir / "compartment_depth_time_section_biochem.png",
    )

    plot_biochem_depth_profiles(
        metadata,
        args.depth_col,
        [var for var in BIOCHEM_ALLOWED_VARS if var in kept_biochem_cols],
        bio_dir / "biochem_depth_profiles_biochem_only.png",
    )

    depth_cluster_table = pd.DataFrame({
        SAMPLE_ID_COL: biochem_scaled.index,
        'compartment_cluster_biochem': cluster_labels,
        args.depth_col: metadata.loc[biochem_scaled.index, args.depth_col].values,
    })
    depth_cluster_table.to_csv(bio_dir / "depth_clusters_biochem.tsv", sep='\t', index=False)

    summary_lines = []
    summary_lines.append("="*70)
    summary_lines.append("BIOCHEM-ONLY COMPARTMENT ANALYSIS")
    summary_lines.append("="*70)
    summary_lines.append(f"Samples analyzed: {len(biochem_scaled)}")
    summary_lines.append(f"Biochem features: {len(biochem_scaled.columns)}")
    summary_lines.append(f"Optimal clusters: {optimal_k}")
    summary_lines.append("\nCluster sizes:")
    for cluster in unique_clusters:
        n = (cluster_labels == cluster).sum()
        pct = n / len(cluster_labels) * 100
        summary_lines.append(f"  Cluster {cluster}: {n} samples ({pct:.1f}%)")
    summary_lines.append("\nBiochemical variables retained:")
    for col in kept_biochem_cols:
        summary_lines.append(f"  - {col}")
    summary_lines.append("\n" + "="*70)
    summary_text = "\n".join(summary_lines)
    with open(bio_dir / "biochem_only_summary.txt", 'w') as fh:
        fh.write(summary_text)
    print(summary_text)


def reorder_dendrogram_by_depth_enhanced(
    linkage_matrix, 
    depth_values, 
    shallow_on_top=True,
    max_tree_depth=None,  # NEW: Only rotate branches up to this depth
    min_depth_diff=0,      # NEW: Only rotate if depth difference > this
    verbose=False          # NEW: Print rotation decisions
):
    """
    Rotate dendrogram branches to order samples by depth.
    
    Parameters
    ----------
    max_tree_depth : int, optional
        Only rotate branches up to this depth in the tree (None = all levels)
        Tree depth 0 = root, increases towards leaves
    min_depth_diff : float
        Only rotate if mean depth difference between children > this value
    verbose : bool
        Print rotation decisions at each node
    """
    from scipy.cluster.hierarchy import to_tree
    
    print(f"      Reordering dendrogram by depth...")
    if max_tree_depth is not None:
        print(f"        Only rotating branches at tree depth ≤ {max_tree_depth}")
    if min_depth_diff > 0:
        print(f"        Only rotating if depth difference > {min_depth_diff}m")
    
    n = len(depth_values)
    linkage_matrix = linkage_matrix.copy()
    tree = to_tree(linkage_matrix, rd=False)
    
    rotation_count = 0
    
    # Get mean depth for a subtree
    def get_subtree_mean_depth(node):
        if node.is_leaf():
            return depth_values[node.id]
        else:
            left_mean = get_subtree_mean_depth(node.left)
            right_mean = get_subtree_mean_depth(node.right)
            # Weighted by number of samples
            return (left_mean * node.left.count + right_mean * node.right.count) / node.count
    
    # Rotate branches with level tracking
    def rotate_branches(node, linkage_idx, tree_depth=0):
        nonlocal rotation_count
        
        if node.is_leaf():
            return
        
        # Check if we should process this level
        if max_tree_depth is not None and tree_depth > max_tree_depth:
            if verbose:
                print(f"          [Depth {tree_depth}] Node {linkage_idx}: Skipped (beyond max_tree_depth)")
            return
        
        # Calculate mean depths
        left_mean = get_subtree_mean_depth(node.left)
        right_mean = get_subtree_mean_depth(node.right)
        depth_diff = abs(left_mean - right_mean)
        
        # Check if difference is large enough to rotate
        if depth_diff < min_depth_diff:
            if verbose:
                print(f"          [Depth {tree_depth}] Node {linkage_idx}: "
                      f"No rotation (diff={depth_diff:.1f}m < {min_depth_diff}m)")
        else:
            # Decide if we should swap
            # For orientation='left', right child appears ABOVE left child
            should_swap = False
            if shallow_on_top:
                should_swap = left_mean < right_mean  # Put shallower in right (top)
            else:
                should_swap = left_mean > right_mean  # Put deeper in right (top)
            
            if should_swap:
                # Swap in linkage matrix
                linkage_matrix[linkage_idx, 0], linkage_matrix[linkage_idx, 1] = \
                    linkage_matrix[linkage_idx, 1], linkage_matrix[linkage_idx, 0]
                node.left, node.right = node.right, node.left
                rotation_count += 1
                
                if verbose:
                    print(f"          [Depth {tree_depth}] Node {linkage_idx}: "
                          f"ROTATED (left={left_mean:.1f}m, right={right_mean:.1f}m, diff={depth_diff:.1f}m)")
            else:
                if verbose:
                    print(f"          [Depth {tree_depth}] Node {linkage_idx}: "
                          f"No rotation needed (left={left_mean:.1f}m, right={right_mean:.1f}m)")
        
        # RECURSE to children (go deeper into tree)
        if not node.left.is_leaf():
            rotate_branches(node.left, node.left.id - n, tree_depth + 1)
        if not node.right.is_leaf():
            rotate_branches(node.right, node.right.id - n, tree_depth + 1)
    
    # Start from root (tree_depth=0)
    rotate_branches(tree, len(linkage_matrix) - 1, tree_depth=0)
    
    print(f"      ✓ Rotated {rotation_count} internal branches")
    
    return linkage_matrix


def reorder_dendrogram_multipass(
    linkage_matrix,
    depth_values,
    shallow_on_top=True,
    n_passes=10,  # Number of optimization passes
):
    """
    Multiple passes of rotation for better optimization.
    """
    print(f"      Reordering dendrogram with {n_passes} passes...")
    
    for pass_num in range(n_passes):
        print(f"        Pass {pass_num + 1}/{n_passes}...")
        linkage_matrix = reorder_dendrogram_by_depth_enhanced(
            linkage_matrix,
            depth_values,
            shallow_on_top=shallow_on_top,
            verbose=(pass_num == 0)  # Only verbose on first pass
        )
    
    return linkage_matrix


def get_nodes_by_distance(linkage_matrix, min_dist=None, max_dist=None):
    """
    Get list of internal node IDs within a distance range.
    
    Parameters
    ----------
    linkage_matrix : array
        Linkage matrix
    min_dist : float, optional
        Minimum distance (default: 0)
    max_dist : float, optional  
        Maximum distance (default: max distance in tree)
    
    Returns
    -------
    node_info : list of dict
        List of nodes with their info: {'node_id', 'distance', 'left_child', 'right_child'}
    """
    if min_dist is None:
        min_dist = 0
    if max_dist is None:
        max_dist = linkage_matrix[:, 2].max()
    
    nodes = []
    for i, row in enumerate(linkage_matrix):
        dist = row[2]
        if min_dist <= dist <= max_dist:
            nodes.append({
                'node_id': i,
                'distance': dist,
                'left_child': int(row[0]),
                'right_child': int(row[1]),
                'n_samples': int(row[3]),
            })
    
    # Sort by distance (largest first = closer to root)
    nodes.sort(key=lambda x: x['distance'], reverse=True)
    
    return nodes


def rotate_specific_nodes(linkage_matrix, node_ids):
    """
    Rotate specific nodes. THAT'S IT!
    
    Parameters
    ----------
    node_ids : list of int
        Node IDs to rotate (linkage matrix row numbers)
    """
    linkage_matrix = linkage_matrix.copy()
    
    for node_id in node_ids:
        # Swap left and right
        linkage_matrix[node_id, 0], linkage_matrix[node_id, 1] = \
            linkage_matrix[node_id, 1], linkage_matrix[node_id, 0]
    
    return linkage_matrix


def calculate_cluster_stability(
    data: np.ndarray,
    linkage_matrix: np.ndarray,
    k_range: range,
    n_iterations: int = 20,
    subsample_frac: float = 0.8,
    random_state: int = 42,
) -> Tuple[int, pd.DataFrame]:
    """
    Test cluster stability by subsampling.
    More stable k = better clusters.
    """
    from sklearn.metrics import adjusted_rand_score
    
    print(f"      Testing cluster stability with {n_iterations} subsamples...")
    
    np.random.seed(random_state)
    n_samples = len(data)
    subsample_size = int(n_samples * subsample_frac)
    
    stability_scores = []
    
    for k in k_range:
        # Get full clustering
        labels_full = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        
        ari_scores = []
        
        for iteration in range(n_iterations):
            # Random subsample
            subsample_idx = np.random.choice(n_samples, subsample_size, replace=False)
            subsample_data = data[subsample_idx]
            
            # Cluster subsample
            subsample_distances = pdist(subsample_data, metric='euclidean')
            subsample_linkage = linkage(subsample_distances, method='average')
            labels_subsample = fcluster(subsample_linkage, k, criterion='maxclust') - 1
            
            # Compare to full clustering (on subsample indices)
            labels_full_subsample = labels_full[subsample_idx]
            
            # Calculate agreement (ARI)
            ari = adjusted_rand_score(labels_full_subsample, labels_subsample)
            ari_scores.append(ari)
        
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        
        stability_scores.append({
            'k': k,
            'stability': mean_ari,
            'stability_std': std_ari,
        })
        
        print(f"        k={k}: stability={mean_ari:.3f} ± {std_ari:.3f}")
    
    stability_df = pd.DataFrame(stability_scores)
    
    # Optimal k = highest stability
    optimal_k = int(stability_df.loc[stability_df['stability'].idxmax(), 'k'])
    print(f"      → Stability suggests k={optimal_k}")
    
    return optimal_k, stability_df


def find_elbow_knee(
    data: np.ndarray,
    linkage_matrix: np.ndarray,
    k_range: range,
) -> Tuple[int, np.ndarray]:
    """
    Find the 'elbow' or 'knee' in within-cluster sum of squares.
    """
    print(f"      Finding elbow using within-cluster dispersion...")
    
    wcss = []
    
    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        
        # Within-cluster sum of squares
        wc_sum = 0
        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id]
            if len(cluster_data) > 1:
                centroid = cluster_data.mean(axis=0)
                wc_sum += np.sum((cluster_data - centroid) ** 2)
        
        wcss.append(wc_sum)
    
    # Find elbow using maximum distance from line
    wcss = np.array(wcss)
    
    # Normalize to [0, 1]
    k_normalized = (np.array(list(k_range)) - k_range.start) / (k_range.stop - 1 - k_range.start)
    wcss_normalized = (wcss - wcss.min()) / (wcss.max() - wcss.min())
    
    # Find point with maximum distance from line connecting first and last points
    line_start = np.array([k_normalized[0], wcss_normalized[0]])
    line_end = np.array([k_normalized[-1], wcss_normalized[-1]])
    
    distances = []
    for i in range(len(k_range)):
        point = np.array([k_normalized[i], wcss_normalized[i]])
        distance = np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        distances.append(distance)
    
    elbow_idx = np.argmax(distances)
    optimal_k = list(k_range)[elbow_idx]
    
    print(f"      → Elbow suggests k={optimal_k}")
    
    return optimal_k, wcss


def evaluate_standard_metrics(
    data: np.ndarray,
    linkage_matrix: np.ndarray,
    k_range: range,
) -> Tuple[int, pd.DataFrame]:
    """
    Standard clustering metrics (silhouette, calinski, davies-bouldin).
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    print(f"      Evaluating standard metrics...")
    
    results = []
    
    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        
        silhouette = silhouette_score(data, labels, metric='euclidean')
        calinski = calinski_harabasz_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin,
        })
        
        print(f"        k={k}: Silhouette={silhouette:.3f}, "
              f"Calinski={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Consensus ranking
    results_df['silhouette_rank'] = results_df['silhouette'].rank(ascending=False)
    results_df['calinski_rank'] = results_df['calinski_harabasz'].rank(ascending=False)
    results_df['davies_bouldin_rank'] = results_df['davies_bouldin'].rank(ascending=True)
    results_df['mean_rank'] = results_df[['silhouette_rank', 'calinski_rank', 'davies_bouldin_rank']].mean(axis=1)
    
    optimal_k = int(results_df.loc[results_df['mean_rank'].idxmin(), 'k'])
    print(f"      → Standard metrics suggest k={optimal_k}")
    
    return optimal_k, results_df


def determine_optimal_clusters_integrated(
    asv_clr: pd.DataFrame,
    biochem_scaled: pd.DataFrame,
    metadata: pd.DataFrame,
    depth_col: str,
    k_range: range = range(2, 11),
    linkage_method: str = 'average',
    auto_weight: str = 'log',  # 'log', 'sqrt', 'capped', or None
    weight_microbiome: float = 0.5,
    weight_biochem: float = 0.5,
    optimization_method: str = 'consensus',  # 'consensus', 'stability', 'standard'
    n_stability_iterations: int = 20,
    output_path: Optional[Path] = None,
    output_table: Optional[Path] = None,
) -> Tuple[int, pd.DataFrame, np.ndarray]:
    """
    Cluster on INTEGRATED microbiome + biochemical data with robust optimization.
    
    Parameters
    ----------
    optimization_method : str
        'consensus' - use multiple methods and vote (recommended)
        'stability' - use stability analysis only (most robust)
        'standard' - use standard metrics only (fastest)
    n_stability_iterations : int
        Number of subsampling iterations for stability analysis
    
    Returns
    -------
    optimal_k, results_df, linkage_matrix, integrated_data, kept_biochem_cols
    """
    print(f"  [i] Clustering on integrated microbiome + biochem data...")
    print(f"      Using Euclidean distance with {linkage_method} linkage")
    print(f"      Optimization method: {optimization_method}")
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Integrate data
    integrated_data = integrate_microbiome_and_biochem(
        asv_clr,
        biochem_scaled,
        auto_weight=auto_weight,
        weight_microbiome=weight_microbiome,
        weight_biochem=weight_biochem,
    )

    # Save integrated table
    save_table = integrated_data.copy()
    integrated_data.to_csv(output_table, sep='\t')
    print(f"      ✓ Saved integrated data table: {output_table.name}")
    
    # Calculate distances on integrated data
    print(f"\n      Computing distances on integrated data...")
    distances = pdist(integrated_data.values, metric='euclidean')
    linkage_matrix = linkage(distances, method=linkage_method)
    
    # Reorder by depth
    depth_values = metadata.loc[integrated_data.index, depth_col].values
    linkage_matrix = reorder_dendrogram_by_depth_enhanced(
        linkage_matrix,
        depth_values,
        shallow_on_top=True,
    )
    
    # ========================================================================
    # ROBUST CLUSTER OPTIMIZATION
    # ========================================================================
    
    print(f"\n      Determining optimal k using {optimization_method} method...")
    
    optimal_ks = {}
    all_results = {}
    
    if optimization_method == 'consensus':
        # Use multiple methods
        print(f"\n      [1/3] Standard metrics:")
        optimal_k_standard, results_standard = evaluate_standard_metrics(
            integrated_data.values,
            linkage_matrix,
            k_range
        )
        optimal_ks['standard'] = optimal_k_standard
        all_results['standard'] = results_standard
        
        print(f"\n      [2/3] Stability analysis:")
        optimal_k_stability, results_stability = calculate_cluster_stability(
            integrated_data.values,
            linkage_matrix,
            k_range,
            n_iterations=n_stability_iterations,
        )
        optimal_ks['stability'] = optimal_k_stability
        all_results['stability'] = results_stability
        
        print(f"\n      [3/3] Elbow detection:")
        optimal_k_elbow, wcss = find_elbow_knee(
            integrated_data.values,
            linkage_matrix,
            k_range
        )
        optimal_ks['elbow'] = optimal_k_elbow
        all_results['elbow'] = wcss
        
        # Consensus
        print(f"\n      Results summary:")
        for method, k in optimal_ks.items():
            print(f"        {method:12s}: k={k}")
        
        # Most common k
        k_values = list(optimal_ks.values())
        vote_counts = {k: k_values.count(k) for k in set(k_values)}
        optimal_k_consensus = max(vote_counts, key=vote_counts.get)
        n_votes = vote_counts[optimal_k_consensus]
        
        print(f"\n  [✓] Consensus k={optimal_k_consensus} ({n_votes}/{len(optimal_ks)} methods agree)")
        
        # If no clear consensus, use median
        if n_votes == 1 and len(set(k_values)) > 2:
            optimal_k_consensus = int(np.median(k_values))
            print(f"      No consensus - using median k={optimal_k_consensus}")
        
        best_k = optimal_k_consensus
        
        # Combine results for output
        results_df = results_standard.copy()
        results_df['stability'] = results_stability['stability'].values
        results_df['stability_std'] = results_stability['stability_std'].values
        
    elif optimization_method == 'stability':
        # Stability only
        optimal_k_stability, results_stability = calculate_cluster_stability(
            integrated_data.values,
            linkage_matrix,
            k_range,
            n_iterations=n_stability_iterations,
        )
        best_k = optimal_k_stability
        optimal_ks['stability'] = optimal_k_stability
        results_df = results_stability
        all_results['stability'] = results_stability
        
        print(f"\n  [✓] Optimal k={best_k} (stability analysis)")
        
    else:  # standard
        # Standard metrics only
        optimal_k_standard, results_standard = evaluate_standard_metrics(
            integrated_data.values,
            linkage_matrix,
            k_range
        )
        best_k = optimal_k_standard
        optimal_ks['standard'] = optimal_k_standard
        results_df = results_standard
        all_results['standard'] = results_standard
        
        print(f"\n  [✓] Optimal k={best_k} (standard metrics)")
    
    # ========================================================================
    # SAVE EVALUATION PLOTS
    # ========================================================================
    
    if output_path and optimization_method == 'consensus':
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Standard metrics
        ax = axes[0, 0]
        ax.plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2.5, markersize=8)
        if 'standard' in optimal_ks:
            ax.axvline(optimal_ks['standard'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(best_k, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Consensus k={best_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax.set_title('Silhouette Score', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        ax = axes[0, 1]
        ax.plot(results_df['k'], results_df['calinski_harabasz'], 'o-', linewidth=2.5, markersize=8, color='green')
        if 'standard' in optimal_ks:
            ax.axvline(optimal_ks['standard'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(best_k, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Consensus k={best_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
        ax.set_title('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        ax = axes[0, 2]
        ax.plot(results_df['k'], results_df['davies_bouldin'], 'o-', linewidth=2.5, markersize=8, color='orange')
        if 'standard' in optimal_ks:
            ax.axvline(optimal_ks['standard'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(best_k, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Consensus k={best_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Score', fontsize=11, fontweight='bold')
        ax.set_title('Davies-Bouldin Score', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Stability
        ax = axes[1, 0]
        ax.errorbar(results_df['k'], results_df['stability'], yerr=results_df['stability_std'],
                   fmt='o-', linewidth=2.5, markersize=8, color='purple', capsize=5)
        if 'stability' in optimal_ks:
            ax.axvline(optimal_ks['stability'], color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(best_k, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Consensus k={best_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Stability (ARI)', fontsize=11, fontweight='bold')
        ax.set_title('Cluster Stability', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Elbow
        ax = axes[1, 1]
        if 'elbow' in all_results:
            ax.plot(list(k_range), all_results['elbow'], 'o-', linewidth=2.5, markersize=8, color='brown')
            if 'elbow' in optimal_ks:
                ax.axvline(optimal_ks['elbow'], color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(best_k, color='green', linestyle='-', linewidth=2, alpha=0.7, label=f'Consensus k={best_k}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Within-Cluster SS', fontsize=11, fontweight='bold')
        ax.set_title('Elbow Method (WCSS)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Consensus summary
        ax = axes[1, 2]
        methods = list(optimal_ks.keys())
        ks = [optimal_ks[m] for m in methods]
        colors = ['blue', 'purple', 'brown'][:len(methods)]
        ax.barh(methods, ks, color=colors, alpha=0.7)
        ax.axvline(best_k, color='green', linestyle='-', linewidth=3, alpha=0.9, label=f'Consensus k={best_k}')
        ax.set_xlabel('Optimal k', fontsize=11, fontweight='bold')
        ax.set_ylabel('Method', fontsize=11, fontweight='bold')
        ax.set_title('Method Comparison', fontsize=12, fontweight='bold')
        ax.set_xlim([k_range.start - 0.5, k_range.stop - 0.5])
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  [✓] Saved cluster evaluation plot")
        
    elif output_path:
        # Single method plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if optimization_method == 'stability':
            ax = axes[0, 0]
            ax.errorbar(results_df['k'], results_df['stability'], yerr=results_df['stability_std'],
                       fmt='o-', linewidth=2.5, markersize=8, color='purple', capsize=5)
            ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={best_k}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Stability (ARI)', fontsize=11, fontweight='bold')
            ax.set_title('Cluster Stability', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            # Hide other subplots
            for i in range(1, 4):
                axes.flat[i].axis('off')
        else:
            # Standard metrics
            ax = axes[0, 0]
            ax.plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2.5, markersize=8)
            ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={best_k}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
            ax.set_title('Silhouette Score', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            ax = axes[0, 1]
            ax.plot(results_df['k'], results_df['calinski_harabasz'], 'o-', linewidth=2.5, markersize=8, color='green')
            ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={best_k}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
            ax.set_title('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            ax = axes[1, 0]
            ax.plot(results_df['k'], results_df['davies_bouldin'], 'o-', linewidth=2.5, markersize=8, color='orange')
            ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={best_k}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Davies-Bouldin Score', fontsize=11, fontweight='bold')
            ax.set_title('Davies-Bouldin Score', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
            
            ax = axes[1, 1]
            ax.plot(results_df['k'], results_df['mean_rank'], 'o-', linewidth=2.5, markersize=8, color='purple')
            ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={best_k}')
            ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Rank', fontsize=11, fontweight='bold')
            ax.set_title('Consensus Ranking', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  [✓] Saved cluster evaluation plot")
    
    return int(best_k), results_df, linkage_matrix


def determine_optimal_clusters_euclidean(
    asv_clr: pd.DataFrame,
    metadata: pd.DataFrame,  # ADD THIS
    depth_col: str,           # ADD THIS
    k_range: range = range(2, 11),
    linkage_method: str = 'average',
    output_path: Optional[Path] = None,
) -> Tuple[int, pd.DataFrame, np.ndarray]:
    """
    Determine optimal number of clusters using hierarchical clustering with Euclidean distance.
    
    Uses Euclidean distance on CLR-transformed data.
    
    Parameters
    ----------
    linkage_method : str
        'average', 'complete', or 'single' (Ward requires Euclidean)
    
    Returns
    -------
    optimal_k : int
    results_df : DataFrame
    linkage_matrix : array
    """
    print(f"  [i] Evaluating cluster quality for k={k_range.start} to {k_range.stop-1}...")
    print(f"      Using Euclidean distance (on CLR data) with {linkage_method} linkage")
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from scipy.spatial.distance import euclidean
    
    # Calculate distances
    distances = pdist(asv_clr, metric='euclidean')
    linkage_matrix = linkage(distances, method=linkage_method)
    
    # REORDER by depth (shallow on top)
    depth_values = metadata.loc[asv_clr.index, depth_col].values
    linkage_matrix = reorder_dendrogram_multipass(
        linkage_matrix, 
        depth_values, 
        shallow_on_top=True
    )

    # STEP 1: Find nodes in a distance range
    print("\n  [i] Finding nodes by distance range...")
    print(f"      Total distance range: {linkage_matrix[:, 2].min():.4f} - {linkage_matrix[:, 2].max():.4f}")

    # Get nodes in different distance bands
    high_nodes = get_nodes_by_distance(linkage_matrix, min_dist=0.14, max_dist=None)
    mid_nodes = get_nodes_by_distance(linkage_matrix, min_dist=0.10, max_dist=0.14)
    low_nodes = get_nodes_by_distance(linkage_matrix, min_dist=0.05, max_dist=0.10)

    print(f"\n      High-level nodes (dist > 0.14): {len(high_nodes)} nodes")
    for node in high_nodes[:10]:  # Show first 10
        print(f"        Node {node['node_id']:3d}: dist={node['distance']:.4f}, "
            f"left={node['left_child']:3d}, right={node['right_child']:3d}, n={node['n_samples']}")

    print(f"\n      Mid-level nodes (0.10 < dist < 0.14): {len(mid_nodes)} nodes")
    for node in mid_nodes[:10]:
        print(f"        Node {node['node_id']:3d}: dist={node['distance']:.4f}, "
            f"left={node['left_child']:3d}, right={node['right_child']:3d}, n={node['n_samples']}")

    # Calculate Euclidean distance matrix on CLR data
    print("      Computing Euclidean distances on CLR-transformed data...")
    
    # Evaluate different k values
    results = []
    
    for k in k_range:
        # Get cluster labels
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        
        # Calculate metrics (use Euclidean for silhouette)
        silhouette = silhouette_score(asv_clr, labels, metric='euclidean')
        calinski = calinski_harabasz_score(asv_clr, labels)
        davies_bouldin = davies_bouldin_score(asv_clr, labels)
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin,
        })
        
        print(f"      k={k}: Silhouette={silhouette:.3f}, "
              f"Calinski-Harabasz={calinski:.1f}, Davies-Bouldin={davies_bouldin:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # ========================================================================
    # Determine optimal k using consensus ranking
    # ========================================================================
    
    # Rank each metric (lower rank = better)
    results_df['silhouette_rank'] = results_df['silhouette'].rank(ascending=False)
    results_df['calinski_rank'] = results_df['calinski_harabasz'].rank(ascending=False)
    results_df['davies_bouldin_rank'] = results_df['davies_bouldin'].rank(ascending=True)  # Lower is better
    
    # Mean rank (lower = better)
    results_df['mean_rank'] = results_df[['silhouette_rank', 'calinski_rank', 'davies_bouldin_rank']].mean(axis=1)
    
    # Best k by mean rank
    best_k = results_df.loc[results_df['mean_rank'].idxmin(), 'k']
    
    print(f"\n  [✓] Optimal k={int(best_k)} (by consensus ranking)")
    
    # ========================================================================
    # Create diagnostic plot
    # ========================================================================
    
    if output_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Silhouette score
        ax = axes[0, 0]
        ax.plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2.5, markersize=8)
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax.set_title('Silhouette Score (higher = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 2: Calinski-Harabasz score
        ax = axes[0, 1]
        ax.plot(results_df['k'], results_df['calinski_harabasz'], 'o-', linewidth=2.5, markersize=8, color='green')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
        ax.set_title('Calinski-Harabasz Score (higher = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 3: Davies-Bouldin score
        ax = axes[1, 0]
        ax.plot(results_df['k'], results_df['davies_bouldin'], 'o-', linewidth=2.5, markersize=8, color='orange')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Score', fontsize=11, fontweight='bold')
        ax.set_title('Davies-Bouldin Score (lower = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Plot 4: Mean rank (consensus)
        ax = axes[1, 1]
        ax.plot(results_df['k'], results_df['mean_rank'], 'o-', linewidth=2.5, markersize=8, color='purple')
        ax.axvline(best_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Rank', fontsize=11, fontweight='bold')
        ax.set_title('Consensus Ranking (lower = better)', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  [✓] Saved cluster evaluation plot")
    
    return int(best_k), results_df, linkage_matrix


def plot_dendrogram_with_heatmap_euclidean(
    asv_clr: pd.DataFrame,
    metadata: pd.DataFrame,
    linkage_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    sample_gc: pd.Series,
    biochem_cols: List[str],
    depth_col: str,
    output_path: Path,
) -> None:
    """
    Create combined vertical dendrogram + environmental heatmap.
    Dendrogram uses Euclidean distance on CLR-transformed data.
    Each variable normalized independently, grayscale (light=low, dark=high).
    Excludes variables with <50% valid values.
    """
    print("  [i] Creating dendrogram with environmental heatmap (Euclidean distance)...")
    
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    
    # ========================================================================
    # Get dendrogram order and VERIFY alignment
    # ========================================================================
    
    print("\n      [ALIGNMENT CHECK] Getting dendrogram leaf order...")
    dend = dendrogram(linkage_matrix, no_plot=True, orientation='left')
    sample_order = dend['leaves']
    
    print(f"      Dendrogram has {len(sample_order)} leaves")
    print(f"      ASV CLR table has {len(asv_clr)} samples")
    print(f"      Metadata has {len(metadata)} samples")
    
    # Verify all indices are valid
    assert len(sample_order) == len(asv_clr), "Dendrogram leaf count doesn't match ASV table!"
    assert all(i < len(asv_clr) for i in sample_order), "Invalid indices in dendrogram!"
    
    print(f"      ✓ Dendrogram indices are valid")
    
    # Get sample names in dendrogram order
    sample_names_ordered = asv_clr.index[sample_order].tolist()
    
    print(f"\n      First 5 samples in dendrogram order:")
    for i, (idx, sample) in enumerate(zip(sample_order[:5], sample_names_ordered[:5])):
        print(f"        Position {i}: index={idx}, sample={sample}")
    
    print(f"      Last 5 samples in dendrogram order:")
    for i, (idx, sample) in enumerate(zip(sample_order[-5:], sample_names_ordered[-5:]), start=len(sample_order)-5):
        print(f"        Position {i}: index={idx}, sample={sample}")
    
    # ========================================================================
    # Reorder data using the dendrogram order
    # ========================================================================
    
    print("\n      [ALIGNMENT CHECK] Reordering data to match dendrogram...")
    
    # Use .iloc for integer-based indexing (matches dendrogram leaf indices)
    metadata_ordered = metadata.iloc[sample_order].copy()
    cluster_labels_ordered = cluster_labels[sample_order]
    
    # Verify the order is correct
    assert list(metadata_ordered.index) == sample_names_ordered, "Metadata order doesn't match dendrogram!"
    print(f"      ✓ Metadata order matches dendrogram")
    
    # ========================================================================
    # Create ALIGNMENT VERIFICATION TABLE
    # ========================================================================
    
    print("\n      Creating alignment verification table...")
    
    alignment_check = pd.DataFrame({
        'dendrogram_position': range(len(sample_order)),
        'array_index': sample_order,
        'sample_id': sample_names_ordered,
        'depth': metadata_ordered[depth_col].values,
        'cluster': cluster_labels_ordered,
    })
    
    # Add first few biochem values for spot-checking
    for col in biochem_cols[:3]:  # First 3 biochem variables
        if col in metadata_ordered.columns:
            alignment_check[f'{col}_value'] = metadata_ordered[col].values
    
    # Save alignment table
    alignment_file = output_path.parent / f"{output_path.stem}_alignment_check.tsv"
    alignment_check.to_csv(alignment_file, sep='\t', index=False)
    print(f"      ✓ Saved alignment verification table: {alignment_file.name}")
    
    # ========================================================================
    # Prepare environmental data and FILTER out variables with <50% valid
    # ========================================================================
    
    env_data = metadata_ordered[biochem_cols].copy()
    
    # Add GC content
    env_data['GC Content (%)'] = sample_gc.loc[metadata_ordered.index]
    
    # CRITICAL: Verify env_data order matches metadata_ordered
    assert list(env_data.index) == list(metadata_ordered.index), "env_data index doesn't match metadata_ordered!"
    print(f"      ✓ Environmental data order matches metadata")
    
    env_data_numeric = env_data.copy()
    
    # Calculate % valid for each variable
    valid_pct = env_data_numeric.notna().sum() / len(env_data_numeric) * 100
    
    print(f"\n      Filtering biochemical variables (must have ≥50% valid values):")
    vars_to_keep = []
    vars_excluded = []
    
    for var in env_data_numeric.columns:
        pct = valid_pct[var]
        if pct >= 50:
            vars_to_keep.append(var)
            print(f"        ✓ KEEP: {var:30} ({pct:.1f}% valid)")
        else:
            vars_excluded.append(var)
            print(f"        ✗ EXCLUDE: {var:30} ({pct:.1f}% valid)")
    
    if vars_excluded:
        print(f"\n      Excluded {len(vars_excluded)} variables with <50% valid values")
    
    # Filter to kept variables
    env_data_numeric = env_data_numeric[vars_to_keep]
    
    if len(vars_to_keep) == 0:
        print("      [!] ERROR: No variables remain after filtering!")
        return
    
    print(f"      Final variables for heatmap: {len(vars_to_keep)}")
    
    # ========================================================================
    # Normalize EACH variable INDEPENDENTLY to [0, 1]
    # ========================================================================
    
    env_normalized = pd.DataFrame(index=env_data_numeric.index, columns=env_data_numeric.columns)
    
    print(f"\n      Normalizing each variable independently:")
    for col in env_data_numeric.columns:
        col_data = env_data_numeric[col]
        valid_data = col_data.dropna()
        
        if len(valid_data) > 0:
            col_min = valid_data.min()
            col_max = valid_data.max()
            
            print(f"        {col:30} range: [{col_min:.3f}, {col_max:.3f}]")
            
            if col_max > col_min:
                env_normalized[col] = (col_data - col_min) / (col_max - col_min)
            else:
                env_normalized[col] = 0.5
        else:
            env_normalized[col] = np.nan
    
    env_normalized = env_normalized.astype(float)
    
    # FINAL ALIGNMENT CHECK
    assert list(env_normalized.index) == sample_names_ordered, "Final env_normalized order doesn't match dendrogram!"
    print(f"\n      ✓✓✓ FINAL ALIGNMENT VERIFIED ✓✓✓")
    print(f"          Dendrogram leaves: {len(sample_order)}")
    print(f"          Heatmap rows: {len(env_normalized)}")
    print(f"          Orders match: {list(env_normalized.index) == sample_names_ordered}")
    
    # ========================================================================
    # Save heatmap data for external verification
    # ========================================================================
    
    heatmap_data_file = output_path.parent / f"{output_path.stem}_heatmap_data.tsv"
    heatmap_export = env_normalized.copy()
    heatmap_export.insert(0, 'dendrogram_position', range(len(env_normalized)))
    heatmap_export.insert(1, 'depth', metadata_ordered[depth_col].values)
    heatmap_export.insert(2, 'cluster', cluster_labels_ordered)
    heatmap_export.to_csv(heatmap_data_file, sep='\t')
    print(f"      ✓ Saved heatmap data for verification: {heatmap_data_file.name}")
    
    # ========================================================================
    # Color mappings
    # ========================================================================
    
    # Depth colors
    if 'Color' in metadata.columns:
        depth_color_map = dict(zip(metadata[depth_col], metadata['Color']))
    else:
        depths = sorted(metadata[depth_col].unique())
        palette_depth = sns.color_palette("tab10", len(depths))
        depth_color_map = dict(zip(depths, [plt.matplotlib.colors.rgb2hex(c) for c in palette_depth]))
    
    depth_colors = [depth_color_map[d] for d in metadata_ordered[depth_col]]
    
    # Cluster colors
    n_clusters = len(np.unique(cluster_labels))
    palette_cluster = sns.color_palette("Set2", n_clusters)
    cluster_color_map = dict(zip(range(n_clusters), [plt.matplotlib.colors.rgb2hex(c) for c in palette_cluster]))
    cluster_colors = [cluster_color_map[c] for c in cluster_labels_ordered]
    
    # VERIFY color bar alignment
    assert len(depth_colors) == len(env_normalized), "Depth color bar length doesn't match heatmap!"
    assert len(cluster_colors) == len(env_normalized), "Cluster color bar length doesn't match heatmap!"
    print(f"      ✓ Color bars aligned with heatmap")
    
    # ========================================================================
    # Create figure with GridSpec
    # ========================================================================
    
    fig = plt.figure(figsize=(16, 14))
    
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1, 0.05, 0.05, 3],
        wspace=0.02,
        left=0.08,
        right=0.75,
        top=0.80,
        bottom=0.08,
    )
    
    n_samples = len(sample_order)
    
    # ========================================================================
    # Plot vertical dendrogram (Euclidean)
    # ========================================================================
    
    ax_dend = fig.add_subplot(gs[0, 0])
    dend_plot = dendrogram(
        linkage_matrix,
        ax=ax_dend,
        orientation='left',
        no_labels=True,
        color_threshold=0,
        above_threshold_color='gray',
    )
    
    # VERIFY the plotted dendrogram has same order as our sample_order
    assert dend_plot['leaves'] == sample_order, "Plotted dendrogram order changed!"
    
    ax_dend.set_ylim([0, n_samples * 10])
    ax_dend.set_xlabel('Distance\n(Euclidean)', fontsize=10, fontweight='bold')
    ax_dend.set_ylabel('')
    ax_dend.set_yticks([])
    ax_dend.spines['top'].set_visible(False)
    ax_dend.spines['right'].set_visible(False)
    ax_dend.spines['left'].set_visible(False)
    
    # ========================================================================
    # Depth color bar (vertical) - LABEL ROTATED
    # ========================================================================
    
    ax_depth = fig.add_subplot(gs[0, 1])
    for i, color in enumerate(depth_colors):
        rect = Rectangle((0, i), 1, 1, facecolor=color, edgecolor='none')
        ax_depth.add_patch(rect)
    ax_depth.set_xlim(0, 1)
    ax_depth.set_ylim(0, n_samples)
    ax_depth.set_xticks([0.5])
    ax_depth.set_xticklabels([depth_col], fontsize=9, fontweight='bold', rotation=90, ha='center', va='bottom')
    ax_depth.set_yticks([])
    ax_depth.spines['top'].set_visible(False)
    ax_depth.spines['right'].set_visible(False)
    ax_depth.spines['bottom'].set_visible(False)
    ax_depth.spines['left'].set_visible(False)
    ax_depth.xaxis.set_ticks_position('top')
    
    # ========================================================================
    # Cluster color bar (vertical) - LABEL ROTATED
    # ========================================================================
    
    ax_cluster = fig.add_subplot(gs[0, 2])
    for i, color in enumerate(cluster_colors):
        rect = Rectangle((0, i), 1, 1, facecolor=color, edgecolor='none')
        ax_cluster.add_patch(rect)
    ax_cluster.set_xlim(0, 1)
    ax_cluster.set_ylim(0, n_samples)
    ax_cluster.set_xticks([0.5])
    ax_cluster.set_xticklabels(['Cluster'], fontsize=9, fontweight='bold', rotation=90, ha='center', va='bottom')
    ax_cluster.set_yticks([])
    ax_cluster.spines['top'].set_visible(False)
    ax_cluster.spines['right'].set_visible(False)
    ax_cluster.spines['bottom'].set_visible(False)
    ax_cluster.spines['left'].set_visible(False)
    ax_cluster.xaxis.set_ticks_position('top')
    
    # ========================================================================
    # Environmental heatmap
    # ========================================================================
    
    ax_heatmap = fig.add_subplot(gs[0, 3])
    
    # Create custom colormap: purple -> light blue -> light green -> red
    from matplotlib.colors import LinearSegmentedColormap

    colors_list = [
        "#F18D09",
        "#1512E6",
    ]

    cmap_custom = LinearSegmentedColormap.from_list('purple_green_red', colors_list)
    cmap_custom.set_bad(color='white')  # NaN values = white

    heatmap_array = env_normalized.values
    print(f"\n      [PLOTTING] Heatmap array shape: {heatmap_array.shape}")
    print(f"                 (rows=samples={heatmap_array.shape[0]}, cols=variables={heatmap_array.shape[1]})")
    
    im = ax_heatmap.imshow(
        heatmap_array,
        aspect='auto',
        cmap=cmap_custom,
        interpolation='none',
        vmin=0,
        vmax=1,
        origin='lower',
    )
    
    # Vertical biochem labels
    ax_heatmap.set_xticks(range(len(env_normalized.columns)))
    ax_heatmap.set_xticklabels(
        env_normalized.columns, 
        fontsize=9, 
        rotation=90,
        ha='center',
        va='bottom',
    )
    ax_heatmap.xaxis.set_ticks_position('top')
    ax_heatmap.xaxis.set_label_position('top')
    ax_heatmap.set_ylabel('', fontsize=11, fontweight='bold')
    ax_heatmap.set_yticks([])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.03, pad=0.02, location='right')
    cbar.set_label('Normalized Value\n(per variable)', fontsize=9, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    
    # ========================================================================
    # Legends (TOP RIGHT)
    # ========================================================================
    
    depth_patches = [plt.matplotlib.patches.Patch(color=color, label=f"{depth}m") 
                    for depth, color in sorted(depth_color_map.items())]
    legend_depth = fig.legend(
        handles=depth_patches,
        title=depth_col,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
    )
    legend_depth.set_title(depth_col, prop={'weight': 'bold', 'size': 9})
    
    cluster_patches = [plt.matplotlib.patches.Patch(color=cluster_color_map[c], label=f"Cluster {c}") 
                      for c in sorted(cluster_color_map.keys())]
    legend_cluster = fig.legend(
        handles=cluster_patches,
        title='Depth Cluster',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98 - 0.02 * len(depth_patches) - 0.05),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
    )
    legend_cluster.set_title('Depth Cluster', prop={'weight': 'bold', 'size': 9})
    
    fig.add_artist(legend_depth)
    
    # Title
    fig.suptitle(
        'Hierarchical Clustering with Environmental Profiles\n(Euclidean distance on CLR data, average linkage)',
        fontsize=13,
        fontweight='bold',
        y=0.96,
    )
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n  [✓] Saved dendrogram with heatmap (Euclidean)")
    print(f"      Variables in heatmap: {len(vars_to_keep)}")
    if vars_excluded:
        print(f"      Variables excluded (<50% valid): {', '.join(vars_excluded)}")
    print(f"\n  [✓] Verification files created:")
    print(f"      - {alignment_file.name} (check dendrogram order)")
    print(f"      - {heatmap_data_file.name} (check heatmap values)")

    # ========================================================================
    # Create COMPREHENSIVE VERIFICATION TABLE with colors
    # ========================================================================

    print("\n      Creating comprehensive verification table with color assignments...")

    verification_table = pd.DataFrame({
        'plot_row': range(len(sample_order)),  # 0 = top of plot
        'array_index': sample_order,
        'sample_id': sample_names_ordered,
        'depth': metadata_ordered[depth_col].values,
        'depth_color': depth_colors,
        'cluster': cluster_labels_ordered,
        'cluster_color': cluster_colors,
    })

    # Add the raw (non-normalized) biochem values for first 5 variables
    for i, col in enumerate(biochem_cols[:5]):
        if col in metadata_ordered.columns:
            verification_table[f'raw_{col}'] = metadata_ordered[col].values

    # Add normalized biochem values for first 5 variables (after filtering)
    for i, col in enumerate(vars_to_keep[:5]):
        if col in env_normalized.columns:
            verification_table[f'normalized_{col}'] = env_normalized[col].values

    # Add dendrogram leaf order for verification
    # The dendrogram y-coordinates go from 5, 15, 25, 35... (increments of 10)
    # Position 0 (top) corresponds to y=5, position 1 to y=15, etc.
    verification_table['dendro_y_coordinate'] = [(i * 10) + 5 for i in range(len(sample_order))]

    # Save verification table
    verification_file = output_path.parent / f"{output_path.stem}_VERIFICATION.tsv"
    verification_table.to_csv(verification_file, sep='\t', index=False)

    print(f"      ✓ Saved VERIFICATION table: {verification_file.name}")
    print(f"\n      VERIFICATION TABLE PREVIEW:")
    print(f"      {'Row':<5} {'Sample':<15} {'Depth':<8} {'DepthColor':<12} {'Cluster':<8} {'ClusterColor':<12}")
    print(f"      {'-'*70}")

    # Show first 10 rows
    for i in range(min(10, len(verification_table))):
        row = verification_table.iloc[i]
        print(f"      {row['plot_row']:<5} {row['sample_id']:<15} {row['depth']:<8} {row['depth_color']:<12} "
            f"{row['cluster']:<8} {row['cluster_color']:<12}")

    print(f"      ...")

    # Show last 5 rows
    for i in range(max(0, len(verification_table) - 5), len(verification_table)):
        row = verification_table.iloc[i]
        print(f"      {row['plot_row']:<5} {row['sample_id']:<15} {row['depth']:<8} {row['depth_color']:<12} "
            f"{row['cluster']:<8} {row['cluster_color']:<12}")

    # ========================================================================
    # Create DEPTH SUMMARY to check if similar depths cluster together
    # ========================================================================

    print(f"\n      DEPTH CLUSTERING CHECK:")
    depth_runs = []
    current_depth = None
    run_start = 0
    run_length = 0

    for i, depth in enumerate(verification_table['depth']):
        if depth != current_depth:
            if current_depth is not None:
                depth_runs.append({
                    'depth': current_depth,
                    'start_row': run_start,
                    'end_row': i - 1,
                    'n_samples': run_length,
                })
            current_depth = depth
            run_start = i
            run_length = 1
        else:
            run_length += 1

    # Add last run
    if current_depth is not None:
        depth_runs.append({
            'depth': current_depth,
            'start_row': run_start,
            'end_row': len(verification_table) - 1,
            'n_samples': run_length,
        })

    depth_runs_df = pd.DataFrame(depth_runs)
    depth_runs_file = output_path.parent / f"{output_path.stem}_depth_runs.tsv"
    depth_runs_df.to_csv(depth_runs_file, sep='\t', index=False)

    print(f"      Contiguous depth blocks in dendrogram order:")
    for _, run in depth_runs_df.iterrows():
        print(f"        Rows {run['start_row']:3d}-{run['end_row']:3d}: Depth {run['depth']:3.0f}m ({run['n_samples']} samples)")

    print(f"      ✓ Saved depth runs table: {depth_runs_file.name}")

    # ========================================================================
    # Create CLUSTER SUMMARY
    # ========================================================================

    print(f"\n      CLUSTER RUNS CHECK:")
    cluster_runs = []
    current_cluster = None
    run_start = 0
    run_length = 0

    for i, cluster in enumerate(verification_table['cluster']):
        if cluster != current_cluster:
            if current_cluster is not None:
                cluster_runs.append({
                    'cluster': current_cluster,
                    'start_row': run_start,
                    'end_row': i - 1,
                    'n_samples': run_length,
                })
            current_cluster = cluster
            run_start = i
            run_length = 1
        else:
            run_length += 1

    # Add last run
    if current_cluster is not None:
        cluster_runs.append({
            'cluster': current_cluster,
            'start_row': run_start,
            'end_row': len(verification_table) - 1,
            'n_samples': run_length,
        })

    cluster_runs_df = pd.DataFrame(cluster_runs)
    cluster_runs_file = output_path.parent / f"{output_path.stem}_cluster_runs.tsv"
    cluster_runs_df.to_csv(cluster_runs_file, sep='\t', index=False)

    print(f"      Contiguous cluster blocks in dendrogram order:")
    for _, run in cluster_runs_df.iterrows():
        print(f"        Rows {run['start_row']:3d}-{run['end_row']:3d}: Cluster {run['cluster']} ({run['n_samples']} samples)")

    print(f"      ✓ Saved cluster runs table: {cluster_runs_file.name}")


def plot_umap_by_depth_euclidean(
    asv_clr: pd.DataFrame,
    metadata: pd.DataFrame,
    depth_col: str,
    output_path: Path,
) -> np.ndarray:
    """
    UMAP ordination with Euclidean distance, colored by depth.
    Legend positioned outside plot area.
    """
    print("  [i] Creating UMAP with Euclidean distance (colored by depth)...")
    
    reducer = umap.UMAP(
        n_components=2, 
        metric='euclidean',
        random_state=42, 
        n_neighbors=15, 
        min_dist=0.1
    )
    coords = reducer.fit_transform(asv_clr)
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    # Color by depth
    if 'Color' in metadata.columns:
        depth_color_map = dict(zip(metadata[depth_col], metadata['Color']))
    else:
        depths = sorted(metadata[depth_col].unique())
        palette = sns.color_palette("tab10", len(depths))
        depth_color_map = dict(zip(depths, palette))
    
    for depth in sorted(metadata[depth_col].unique()):
        mask = metadata[depth_col] == depth
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[depth_color_map[depth]],
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"{depth}m",
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Community Composition by Depth (Euclidean distance)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(title=depth_col, bbox_to_anchor=(1.02, 1), loc='upper left', 
             frameon=True, fontsize=9, title_fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved UMAP by depth (Euclidean)")
    
    return coords


def plot_umap_by_cluster_euclidean(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_color_map: Dict[int, str],
    output_path: Path,
) -> None:
    """
    UMAP ordination colored by cluster.
    Legend positioned outside plot area.
    """
    print("  [i] Creating UMAP colored by cluster...")
    
    fig, ax = plt.subplots(figsize=(11, 8))
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cluster_color_map.get(cluster_id, '#808080')],
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=f"Cluster {cluster_id}",
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title('Community Composition by Depth Cluster (Euclidean distance)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', 
             frameon=True, fontsize=9, title_fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  [✓] Saved UMAP by cluster")


def calculate_log_weights(n_microbiome, n_biochem):
    """
    Weight inversely proportional to log of feature count.
    Most balanced approach - prevents extreme weighting.
    """
    # Add 1 to avoid log(0), use log10 for interpretability
    weight_microbiome = 1.0 / (1 + np.log10(n_microbiome))
    weight_biochem = 1.0 / (1 + np.log10(n_biochem))
    
    # Normalize to sum to 1
    total = weight_microbiome + weight_biochem
    weight_microbiome /= total
    weight_biochem /= total
    
    print(f"      Log-balanced weights:")
    print(f"        Features: microbiome={n_microbiome}, biochem={n_biochem}")
    print(f"        Weight microbiome: {weight_microbiome:.3f}")
    print(f"        Weight biochem: {weight_biochem:.3f}")
    
    return weight_microbiome, weight_biochem


def prepare_biochem_dataset(
    metadata: pd.DataFrame,
    biochem_cols: List[str],
    min_valid_pct: float = 50.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare numeric, scaled biochemical dataset aligned to metadata.
    """
    print("\n  [i] Preparing biochemical dataset...")
    if biochem_cols:
        requested_cols = [col for col in biochem_cols if col in BIOCHEM_ALLOWED_VARS]
    else:
        requested_cols = BIOCHEM_ALLOWED_VARS.copy()
    existing_cols = [col for col in requested_cols if col in metadata.columns]
    missing_cols = [col for col in requested_cols if col not in metadata.columns]
    if missing_cols:
        print(f"      [!] Missing biochem columns (skipping): {missing_cols}")
    if not existing_cols:
        print("      [!] No requested biochemical columns found in metadata.")
        return pd.DataFrame(index=metadata.index), []
    biochem_data = metadata[existing_cols].copy()
    for col in biochem_data.columns:
        biochem_data[col] = pd.to_numeric(biochem_data[col], errors='coerce')
    valid_pct = biochem_data.notna().sum() / len(biochem_data) * 100
    kept_cols = valid_pct[valid_pct >= min_valid_pct].index.tolist()
    excluded_cols = valid_pct[valid_pct < min_valid_pct].index.tolist()
    print(f"      Biochemical variables requested: {len(biochem_cols)}")
    print(f"      Kept (≥{min_valid_pct}% valid): {len(kept_cols)}")
    if excluded_cols:
        print(f"      Excluded due to low coverage ({len(excluded_cols)}): {', '.join(excluded_cols)}")
    if not kept_cols:
        print("      [!] No biochemical variables passed filtering.")
        return pd.DataFrame(index=metadata.index), []
    biochem_filtered = biochem_data[kept_cols].copy()
    for col in biochem_filtered.columns:
        if biochem_filtered[col].isna().any():
            median_val = biochem_filtered[col].median()
            if np.isnan(median_val):
                median_val = 0
            biochem_filtered[col].fillna(median_val, inplace=True)
    scaler = StandardScaler()
    biochem_scaled = pd.DataFrame(
        scaler.fit_transform(biochem_filtered),
        index=biochem_filtered.index,
        columns=biochem_filtered.columns,
    )
    return biochem_scaled, kept_cols


def integrate_microbiome_and_biochem(
    asv_clr: pd.DataFrame,
    biochem_scaled: pd.DataFrame,
    auto_weight: bool = True,
    weight_microbiome: float = 0.5,
    weight_biochem: float = 0.5,
) -> pd.DataFrame:
    """
    Integrate CLR-transformed microbiome and scaled biochemical data.
    """
    from sklearn.preprocessing import StandardScaler
    
    print(f"\n  [i] Integrating microbiome and biochemical data...")
    asv_clr_aligned = asv_clr.copy()
    biochem_scaled_aligned = biochem_scaled.loc[asv_clr_aligned.index]
    if auto_weight and biochem_scaled_aligned.shape[1] > 0:
        n_microbiome = asv_clr_aligned.shape[1]
        n_biochem = biochem_scaled_aligned.shape[1]
        weight_microbiome, weight_biochem = calculate_log_weights(n_microbiome, n_biochem)
    elif auto_weight and biochem_scaled_aligned.shape[1] == 0:
        weight_microbiome, weight_biochem = 1.0, 0.0
    else:
        print(f"      Using user-specified weights:")
        print(f"      Microbiome weight: {weight_microbiome}")
        print(f"      Biochem weight: {weight_biochem}")
    
    scaler_microbiome = StandardScaler()
    
    microbiome_scaled = pd.DataFrame(
        scaler_microbiome.fit_transform(asv_clr_aligned),
        index=asv_clr_aligned.index,
        columns=asv_clr_aligned.columns
    )
    
    if biochem_scaled_aligned.shape[1] > 0:
        biochem_weighted = biochem_scaled_aligned * weight_biochem
        microbiome_weighted = microbiome_scaled * weight_microbiome
        integrated_frames = [microbiome_weighted, biochem_weighted]
    else:
        microbiome_weighted = microbiome_scaled
        integrated_frames = [microbiome_weighted]
        print("      [!] No biochemical variables available; using microbiome data only.")
    
    integrated_data = pd.concat(integrated_frames, axis=1)
    
    print(f"\n      Integrated data shape: {integrated_data.shape}")
    print(f"        Microbiome features: {len(microbiome_weighted.columns)}")
    print(f"        Biochemical features: {biochem_scaled_aligned.shape[1]}")
    print(f"        Total features: {integrated_data.shape[1]}")
    
    return integrated_data


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Figure 1: Depth stratification analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input files
    parser.add_argument("--asv-clr", type=Path, required=True,
                       help="CLR-transformed ASV data (from batch correction)")
    parser.add_argument("--asv-counts", type=Path, required=True,
                       help="Raw ASV count table (for richness)")
    parser.add_argument("--asv-fasta", type=Path, required=True,
                       help="ASV sequences in FASTA format (for GC content)")
    parser.add_argument("--metadata", type=Path, required=True,
                       help="Sample metadata")
    
    # Metadata columns
    parser.add_argument("--depth-col", default="Depth",
                       help="Depth column name")
    parser.add_argument("--month-col", default="Month",
                       help="Month column name")
    parser.add_argument("--biochem-cols", required=True,
                       help="Comma-separated list of biochemical variables")
    
    # Clustering
    parser.add_argument("--use-integrated", action="store_true",
                   help="Cluster on integrated microbiome + biochem data (default: microbiome only)")
    parser.add_argument("--auto-weight", action="store_true", default=True,
                   help="Weight integrated microbiome + biochem features (default: True)")
    parser.add_argument("--weight-microbiome", type=float, default=0.5,
                    help="Weight for microbiome data in integration (0-1)")
    parser.add_argument("--weight-biochem", type=float, default=0.5,
                    help="Weight for biochemical data in integration (0-1)")
    parser.add_argument("--k-range", default="2,10",
                   help="Range of k values to test (format: min,max)")
    
    # Output
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Setup
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    biochem_cols = [c.strip() for c in args.biochem_cols.split(",")]
    
    print("\n" + "="*70)
    print("FIGURE 1: DEPTH STRATIFICATION ANALYSIS")
    print("="*70)
    
    # Load data
    print("\n[1/9] Loading data...")
    asv_clr = pd.read_csv(args.asv_clr, sep="\t", index_col=0)
    asv_counts = pd.read_csv(args.asv_counts, sep="\t", index_col=0).T
    metadata_df = pd.read_csv(args.metadata, sep="\t")
    if SAMPLE_ID_COL not in metadata_df.columns:
        raise ValueError(f"Metadata column '{SAMPLE_ID_COL}' not found in {args.metadata}")
    metadata = metadata_df.drop_duplicates(subset=[SAMPLE_ID_COL]).set_index(SAMPLE_ID_COL)
    
    # CLEAN BIOCHEMICAL DATA TYPES
    print("  Cleaning biochemical data types...")
    biochem_cols_to_clean = biochem_cols + ['PAR/Irradiance', 'Fluorescence', args.depth_col]

    for col in biochem_cols_to_clean:
        if col in metadata.columns:
            # Convert to numeric
            original_valid = metadata[col].notna().sum()
            metadata[col] = pd.to_numeric(metadata[col], errors='coerce')
            new_valid = metadata[col].notna().sum()
            
            if new_valid < original_valid:
                print(f"    {col}: {original_valid} → {new_valid} valid values (converted {original_valid - new_valid} non-numeric to NaN)")
            else:
                print(f"    {col}: {new_valid} valid values")

    # ========================================================================
    # Filter biochemical variables (≥50% valid values required)
    # ========================================================================

    print("\n  Filtering biochemical variables (≥50% valid values)...")

    valid_pct = {}
    for col in biochem_cols:
        if col in metadata.columns:
            valid_pct[col] = metadata[col].notna().sum() / len(metadata) * 100

    biochem_cols_filtered = []
    biochem_cols_excluded = []

    for col in biochem_cols:
        if col in valid_pct:
            pct = valid_pct[col]
            if pct >= 50:
                biochem_cols_filtered.append(col)
                print(f"    ✓ {col:30} ({pct:.1f}% valid)")
            else:
                biochem_cols_excluded.append(col)
                print(f"    ✗ {col:30} ({pct:.1f}% valid) - EXCLUDED")

    if biochem_cols_excluded:
        print(f"\n  [!] Excluded {len(biochem_cols_excluded)} biochemical variables with <50% valid values:")
        for col in biochem_cols_excluded:
            print(f"      - {col}")

    print(f"\n  [✓] Using {len(biochem_cols_filtered)} biochemical variables for analysis")

    # Use filtered list for all subsequent analyses
    biochem_cols = biochem_cols_filtered

    # Align samples
    common_samples = asv_clr.index.intersection(metadata.index)
    asv_clr = asv_clr.loc[common_samples]
    asv_counts = asv_counts.loc[common_samples]
    metadata = metadata.loc[common_samples]
    
    print(f"  Samples: {len(common_samples)}")
    print(f"  ASVs: {asv_clr.shape[1]}")
    print(f"  Depths: {sorted(metadata[args.depth_col].unique())}")
    
    biochem_scaled, kept_biochem_cols = prepare_biochem_dataset(metadata, biochem_cols)
    
    # Calculate GC content
    print("\n[2/9] Calculating GC content...")
    gc_content = calculate_gc_content_from_fasta(args.asv_fasta)
    sample_gc = calculate_sample_weighted_gc(asv_clr, gc_content)

    # [4/8] Determine optimal number of clusters
    if args.use_integrated:
        print(f"\n[4/8] Clustering on INTEGRATED microbiome + biochem data...")
        optimal_k, cluster_eval_df, linkage_matrix = determine_optimal_clusters_integrated(
                asv_clr,
                biochem_scaled,
                metadata,
                args.depth_col,
                k_range=range(2, 11),
                linkage_method='average',
                auto_weight='log',
                optimization_method='stability',
                n_stability_iterations=20,
                output_path=out_dir / "cluster_evaluation_integrated.png",
                output_table=out_dir / "data_integrated.tsv",
            )
            
    else:
        print(f"\n[4/8] Clustering on MICROBIOME data only...")
        optimal_k, cluster_eval_df, linkage_matrix = determine_optimal_clusters_euclidean(
            asv_clr,
            metadata,
            args.depth_col,
            k_range=range(2, 11),
            linkage_method='average',
            output_path=out_dir / "cluster_evaluation_euclidean.png",
        )
        
    # Use just ASV data for UMAP
    use_for_umap = asv_clr

    # Save evaluation results
    cluster_eval_df.to_csv(
        out_dir / f"cluster_evaluation_{'integrated' if args.use_integrated else 'microbiome'}.tsv",
        sep='\t', index=False
    )

    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1

    # ============================================================================
    # MANUAL ROTATIONS - Edit this list to rotate specific nodes
    # ============================================================================

    print("\n[MANUAL ROTATIONS]")

    # HARDCODE YOUR NODES HERE - just change these numbers!
    NODES_TO_ROTATE = []

    if NODES_TO_ROTATE:
        print(f"  Rotating {len(NODES_TO_ROTATE)} nodes: {NODES_TO_ROTATE}")
        linkage_matrix = rotate_specific_nodes(linkage_matrix, NODES_TO_ROTATE)
        print(f"  ✓ Rotations applied")
    else:
        print(f"  No manual rotations (NODES_TO_ROTATE is empty)")

    # ============================================================================

    # Get cluster labels
    print(f"\n[5/9] Assigning samples to clusters (k={optimal_k})...")
    cluster_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust') - 1
    unique_clusters = sorted(np.unique(cluster_labels))
    cluster_palette = sns.color_palette("Set2", len(unique_clusters))
    cluster_color_map = {cluster: to_hex(cluster_palette[i]) for i, cluster in enumerate(unique_clusters)}

    print(f"      Created {optimal_k} clusters:")
    for i in range(optimal_k):
        n = (cluster_labels == i).sum()
        print(f"        Cluster {i}: {n} samples")

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'sample': asv_clr.index,
        'cluster': cluster_labels,
        'depth': metadata.loc[asv_clr.index, args.depth_col],
        'cluster_color': [cluster_color_map.get(lbl, '#808080') for lbl in cluster_labels],
    })
    cluster_df.to_csv(out_dir / "depth_clusters.tsv", sep='\t', index=False)
    
    # Calculate distances and richness
    print("\n[6/9] Calculating distances and richness...")
    distance_df = calculate_between_depth_distances(
        asv_clr, metadata, args.depth_col
    )
    distance_df.to_csv(out_dir / "between_depth_distances.tsv", sep='\t', index=False)
    
    richness = calculate_richness(asv_counts)
    richness_df = pd.DataFrame({
        'sample': richness.index,
        'richness': richness.values,
        'depth': metadata.loc[richness.index, args.depth_col],
    })
    richness_df.to_csv(out_dir / "asv_richness.tsv", sep='\t', index=False)
    
    # Create visualizations
    print("\n[7/9] Creating Figure 1 panels...")
    
    # Panel a+b: Dendrogram + heatmap (Euclidean)
    plot_dendrogram_with_heatmap_euclidean(
        asv_clr,
        metadata,
        linkage_matrix,
        cluster_labels,
        sample_gc,
        biochem_cols,
        args.depth_col,
        out_dir / "figure1_panel_ab_dendrogram_heatmap.png",
    )

    umap_coords = plot_umap_by_depth_euclidean(
        use_for_umap,  # Use integrated data if --use-integrated
        metadata,
        args.depth_col,
        out_dir / "figure1_panel_c_umap_by_depth.png",
    )
    color_values = [cluster_color_map.get(lbl, '#808080') for lbl in cluster_labels]
    umap_cluster_table = pd.DataFrame({
        SAMPLE_ID_COL: use_for_umap.index,
        'compartment_cluster': cluster_labels,
        args.depth_col: metadata.loc[use_for_umap.index, args.depth_col].values,
        'compartment_color': color_values,
        'umap1': umap_coords[:, 0],
        'umap2': umap_coords[:, 1],
    })
    umap_cluster_table.to_csv(out_dir / "compartment_umap_clusters.tsv", sep='\t', index=False)
    print(f"  [✓] Saved UMAP coordinates table: compartment_umap_clusters.tsv")
    
    # Panel d: UMAP by cluster (euclidean)
    plot_umap_by_cluster_euclidean(
        umap_coords,
        cluster_labels,
        cluster_color_map,
        out_dir / "figure1_panel_d_umap_by_cluster.png",
    )
    
    # Panel e: Richness by depth
    plot_richness_by_depth(
        richness,
        metadata,
        args.depth_col,
        out_dir / "figure1_panel_e_richness_by_depth.png",
    )
    
    # Bonus: Between-depth distances (pass metadata for colors)
    plot_between_depth_distances(
        distance_df,
        metadata,
        args.depth_col,
        out_dir / "figure1_bonus_between_depth_distances.png",
    )
    
    plot_compartment_depth_time_section(
        metadata,
        args.depth_col,
        args.month_col,
        cluster_labels,
        cluster_color_map,
        out_dir / "compartment_depth_time_section.png",
    )

    plot_biochem_depth_profiles(
        metadata,
        args.depth_col,
        kept_biochem_cols,
        out_dir / "biochem_depth_profiles.png",
    )
    
    
    # Summary
    print("\n[8/9] Generating summary...")
    
    summary = []
    summary.append("="*70)
    summary.append("FIGURE 1: DEPTH STRATIFICATION SUMMARY")
    summary.append("="*70)
    summary.append(f"\nSamples analyzed: {len(asv_clr)}")
    summary.append(f"ASVs: {asv_clr.shape[1]}")
    summary.append(f"Depths sampled: {sorted(metadata[args.depth_col].unique())}")
    summary.append(f"\nHierarchical clusters identified: {optimal_k}")
    summary.append(f"\nCluster sizes:")
    for i in range(optimal_k):
        n = (cluster_labels == i).sum()
        pct = n / len(cluster_labels) * 100
        depths_in_cluster = metadata.loc[cluster_labels == i, args.depth_col].unique()
        summary.append(f"  Cluster {i}: {n} samples ({pct:.1f}%) - Depths: {sorted(depths_in_cluster)}")
    
    summary.append(f"\nGC content:")
    summary.append(f"  Sample mean GC%: {sample_gc.mean():.2f}%")
    summary.append(f"  Range: {sample_gc.min():.2f}% - {sample_gc.max():.2f}%")
    
    summary.append(f"\nRichness:")
    summary.append(f"  Mean: {richness.mean():.0f} ASVs")
    summary.append(f"  Range: {richness.min()} - {richness.max()} ASVs")
    
    summary.append("\n" + "="*70)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open(out_dir / "figure1_summary.txt", 'w') as f:
        f.write(summary_text)
    
    run_biochem_only_outputs(
        biochem_scaled.loc[metadata.index],
        kept_biochem_cols,
        metadata,
        sample_gc,
        args,
        out_dir,
    )
    
    print("\n[9/9] Complete!")
    print("="*70)
    print(f"\nOutput directory: {out_dir}")
    print("\nFigure 1 panels:")
    print("  - figure1_panel_ab_dendrogram_heatmap.png")
    print("  - figure1_panel_c_umap_by_depth.png")
    print("  - figure1_panel_d_umap_by_cluster.png")
    print("  - figure1_panel_e_richness_by_depth.png")
    print("  - figure1_bonus_between_depth_distances.png")
    print("\nData files:")
    print("  - depth_clusters.tsv")
    print("  - between_depth_distances.tsv")
    print("  - asv_richness.tsv")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
