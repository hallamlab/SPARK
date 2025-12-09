#!/usr/bin/env python3
"""
asv_umap_clustering.py
---------------------
Perform UMAP dimensionality reduction and HDBSCAN clustering on ASV count data.
Generates scatter plots colored by depth, month, and cluster assignments.

Features:
- UMAP dimensionality reduction of ASV count matrix
- HDBSCAN clustering
- Multiple visualization options
- Rich metadata integration
- Configurable parameters

Example:
--------
python asv_umap_clustering.py \
  --input asv_data.tsv \
  --asv-col ASV_ID \
  --sample-col sampleID \
  --count-col count \
  --depth-col Depth \
  --color-col Color \
  --month-col Month \
  --output-prefix umap_clustering \
  --formats pdf,png
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# UMAP and HDBSCAN
try:
    import umap
except ImportError:
    print("[ERROR] umap-learn not installed. Install with: pip install umap-learn")
    sys.exit(1)

try:
    import hdbscan
except ImportError:
    print("[ERROR] hdbscan not installed. Install with: pip install hdbscan")
    sys.exit(1)

from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_pivot_data(
    input_file: Path,
    asv_col: str,
    sample_col: str,
    count_col: str,
    metadata_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ASV data and pivot to sample × ASV matrix.
    
    Args:
        input_file: Path to input TSV
        asv_col: ASV ID column
        sample_col: Sample ID column
        count_col: Count column
        metadata_cols: Metadata columns to preserve
    
    Returns:
        Tuple of (count_matrix, metadata_df)
        - count_matrix: samples as rows, ASVs as columns
        - metadata_df: metadata for each sample
    """
    print(f"[INFO] Reading {input_file}...")
    df = pd.read_csv(input_file, sep='\t')
    
    print(f"[INFO] Loaded {len(df)} rows")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Validate required columns
    required = [asv_col, sample_col, count_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract metadata (one row per sample)
    available_metadata = [col for col in metadata_cols if col in df.columns]
    if available_metadata:
        metadata_df = df[[sample_col] + available_metadata].drop_duplicates(subset=[sample_col])
        metadata_df = metadata_df.set_index(sample_col)
        print(f"[INFO] Preserved metadata columns: {available_metadata}")
    else:
        metadata_df = pd.DataFrame(index=df[sample_col].unique())
        print("[WARN] No metadata columns found")
    
    # Pivot to sample × ASV matrix
    print("[INFO] Pivoting to sample × ASV matrix...")
    count_matrix = df.pivot_table(
        index=sample_col,
        columns=asv_col,
        values=count_col,
        aggfunc='sum',
        fill_value=0
    )
    
    print(f"[INFO] Count matrix shape: {count_matrix.shape} (samples × ASVs)")
    print(f"[INFO] Metadata shape: {metadata_df.shape}")
    
    # Align metadata with count matrix
    metadata_df = metadata_df.reindex(count_matrix.index)
    
    return count_matrix, metadata_df


def preprocess_counts(
    count_matrix: pd.DataFrame,
    normalize: str = 'total',
    transform: str = 'log1p',
    scale: bool = True
) -> np.ndarray:
    """
    Preprocess count matrix for UMAP.
    
    Args:
        count_matrix: Sample × ASV count matrix
        normalize: Normalization method ('total', 'clr', 'none')
        transform: Transformation method ('log1p', 'sqrt', 'none')
        scale: Whether to standardize features (Z-score)
    
    Returns:
        Preprocessed matrix as numpy array
    """
    print("[INFO] Preprocessing count matrix...")
    X = count_matrix.values.copy().astype(float)
    
    print(f"[INFO] Input shape: {X.shape}")
    print(f"[INFO] Input range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"[INFO] Zeros: {(X == 0).sum()} / {X.size} ({100*(X == 0).sum()/X.size:.1f}%)")
    
    # Normalization
    if normalize == 'total':
        # Total-sum normalization (relative abundance)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        X = X / row_sums
        print("[INFO] Applied total-sum normalization")
    elif normalize == 'clr':
        # Centered log-ratio transformation
        # Add pseudocount to avoid log(0)
        pseudocount = 1
        X = X + pseudocount
        
        # Calculate geometric mean per sample (row)
        # Use log space to avoid overflow
        log_X = np.log(X)
        geometric_mean = np.exp(log_X.mean(axis=1, keepdims=True))
        
        # Avoid division by zero
        geometric_mean[geometric_mean == 0] = 1
        
        X = np.log(X / geometric_mean)
        
        # Check for NaN/inf
        if np.any(~np.isfinite(X)):
            print("[WARN] CLR produced non-finite values, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("[INFO] Applied CLR transformation")
    
    # Transformation
    if transform == 'log1p':
        # Ensure non-negative before log1p
        if X.min() < 0:
            print("[WARN] Negative values detected, shifting to non-negative")
            X = X - X.min()
        X = np.log1p(X)
        print("[INFO] Applied log1p transformation")
    elif transform == 'sqrt':
        # Ensure non-negative before sqrt
        if X.min() < 0:
            print("[WARN] Negative values detected, shifting to non-negative")
            X = X - X.min()
        X = np.sqrt(X)
        print("[INFO] Applied sqrt transformation")
    
    # Final check for NaN/inf
    if np.any(~np.isfinite(X)):
        print("[WARN] Non-finite values detected after transformation, replacing with zeros")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Remove zero-variance features (constant columns)
    col_std = X.std(axis=0)
    zero_var_cols = col_std == 0
    if zero_var_cols.any():
        n_zero_var = zero_var_cols.sum()
        print(f"[INFO] Removing {n_zero_var} zero-variance features")
        X = X[:, ~zero_var_cols]
    
    # Scaling
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Final check after scaling
        if np.any(~np.isfinite(X)):
            print("[WARN] Non-finite values after scaling, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("[INFO] Applied standard scaling (Z-score)")
    
    print(f"[INFO] Final shape: {X.shape}")
    print(f"[INFO] Final range: [{X.min():.2f}, {X.max():.2f}]")
    
    return X


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = 'euclidean',
    random_state: int = 42
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction.
    
    Args:
        X: Preprocessed count matrix
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of UMAP components
        metric: Distance metric
        random_state: Random seed
    
    Returns:
        UMAP embeddings (n_samples × n_components)
    """
    print(f"[INFO] Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False
    )
    
    embedding = reducer.fit_transform(X)
    
    print(f"[INFO] UMAP embedding shape: {embedding.shape}")
    
    return embedding


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Run HDBSCAN clustering on UMAP embeddings.
    
    Args:
        embedding: UMAP embeddings
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core point
        metric: Distance metric
    
    Returns:
        Cluster labels (noise points = -1)
    """
    print(f"[INFO] Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom'
    )
    
    labels = clusterer.fit_predict(embedding)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"[INFO] Found {n_clusters} clusters")
    print(f"[INFO] Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
    
    return labels


def plot_umap_scatter(
    embedding: np.ndarray,
    colors: pd.Series,
    color_map: Optional[dict],
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: str,
    output_file: Path,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
    alpha: float = 0.7,
    point_size: float = 50
) -> None:
    """
    Create UMAP scatter plot with custom coloring.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different color types
    if color_map is not None:
        # Use provided color map - sort values numerically if possible
        unique_vals = colors.unique()
        
        # Try to sort numerically
        try:
            unique_vals_sorted = sorted(unique_vals, key=lambda x: float(x))
        except (ValueError, TypeError):
            unique_vals_sorted = sorted(unique_vals, key=str)
        
        for val in unique_vals_sorted:
            mask = colors == val
            color = color_map.get(val, 'gray')
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=color,
                label=str(val),
                alpha=alpha,
                s=point_size,
                edgecolors='black',
                linewidth=0.5
            )
    else:
        # Auto-assign categorical colors
        unique_vals = colors.unique()
        
        # Sort with -1 (noise) at the end
        if -1 in unique_vals:
            other_vals = sorted([v for v in unique_vals if v != -1])
            unique_vals_sorted = other_vals + [-1]
        else:
            try:
                unique_vals_sorted = sorted(unique_vals, key=lambda x: float(x) if x != -1 else float('inf'))
            except (ValueError, TypeError):
                unique_vals_sorted = sorted(unique_vals, key=str)
        
        palette = sns.color_palette('tab20', n_colors=len(unique_vals_sorted))
        color_dict = dict(zip(unique_vals_sorted, palette))
        
        for val in unique_vals_sorted:
            mask = colors == val
            color = color_dict[val]
            if val == -1:
                label = "Noise"
            else:
                label = str(val)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                label=label,
                alpha=alpha if val != -1 else 0.3,
                s=point_size,
                edgecolors='black',
                linewidth=0.5
            )
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Legend
    legend = ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        fontsize=10
    )
    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if output_file.suffix == '.png':
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(output_file, bbox_inches='tight')
    
    print(f"[OK] Saved: {output_file}")
    plt.close()


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="UMAP dimensionality reduction and HDBSCAN clustering of ASV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output
    io = ap.add_argument_group("Input/Output")
    io.add_argument("--input", "-i", required=True, type=Path,
                    help="Input TSV file with ASV counts")
    io.add_argument("--output-prefix", "-o", default="umap_clustering",
                    help="Output file prefix")
    io.add_argument("--formats", default="pdf,png",
                    help="Comma-separated output formats (pdf,png,svg)")
    io.add_argument("--dpi", type=int, default=300,
                    help="DPI for raster outputs")
    
    # Column names
    cols = ap.add_argument_group("Column Names")
    cols.add_argument("--asv-col", default="ASV_ID",
                      help="ASV ID column")
    cols.add_argument("--sample-col", default="sampleID",
                      help="Sample ID column")
    cols.add_argument("--count-col", default="count",
                      help="Count column")
    cols.add_argument("--depth-col", default="Depth",
                      help="Depth column for coloring")
    cols.add_argument("--month-col", default="Month",
                      help="Month column for coloring")
    cols.add_argument("--color-col", default="Color",
                      help="Color mapping column for depths")
    
    # Preprocessing
    prep = ap.add_argument_group("Preprocessing")
    prep.add_argument("--normalize", choices=['total', 'clr', 'none'], default='total',
                      help="Normalization method")
    prep.add_argument("--transform", choices=['log1p', 'sqrt', 'none'], default='log1p',
                      help="Transformation method")
    prep.add_argument("--no-scale", action="store_true",
                      help="Skip standard scaling")
    
    # UMAP parameters
    umap_group = ap.add_argument_group("UMAP Parameters")
    umap_group.add_argument("--n-neighbors", type=int, default=15,
                            help="UMAP n_neighbors parameter")
    umap_group.add_argument("--min-dist", type=float, default=0.1,
                            help="UMAP min_dist parameter")
    umap_group.add_argument("--umap-metric", default="euclidean",
                            help="UMAP distance metric")
    umap_group.add_argument("--n-components", type=int, default=2,
                            help="Number of UMAP dimensions")
    
    # HDBSCAN parameters
    hdb = ap.add_argument_group("HDBSCAN Parameters")
    hdb.add_argument("--min-cluster-size", type=int, default=5,
                     help="Minimum cluster size")
    hdb.add_argument("--min-samples", type=int, default=3,
                     help="Minimum samples for core point")
    hdb.add_argument("--hdbscan-metric", default="euclidean",
                     help="HDBSCAN distance metric")
    
    # Plotting
    plot = ap.add_argument_group("Plotting")
    plot.add_argument("--figsize", default="10,8",
                      help="Figure size as 'width,height' in inches")
    plot.add_argument("--point-size", type=float, default=50,
                      help="Scatter point size")
    plot.add_argument("--alpha", type=float, default=0.7,
                      help="Point transparency (0-1)")
    
    # Other
    ap.add_argument("--random-state", type=int, default=42,
                    help="Random seed for reproducibility")
    
    return ap.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)
    
    # Parse parameters
    formats = [f.strip().lstrip('.') for f in args.formats.split(',')]
    figsize = tuple(map(float, args.figsize.split(',')))
    
    print("="*60)
    print("ASV UMAP Clustering Pipeline")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output prefix: {args.output_prefix}")
    print("="*60)
    
    # Metadata columns to preserve
    metadata_cols = [
        args.depth_col, args.month_col, args.color_col,
        'Year', 'Cruise', 'Temperature (deg C)', 'Salinity (PSU)',
        'Oxygen SBE (mL/L)', 'sample_code', 'longID'
    ]
    
    # Load and pivot data
    count_matrix, metadata_df = load_and_pivot_data(
        input_file=args.input,
        asv_col=args.asv_col,
        sample_col=args.sample_col,
        count_col=args.count_col,
        metadata_cols=metadata_cols
    )
    
    # Preprocess
    X = preprocess_counts(
        count_matrix=count_matrix,
        normalize=args.normalize,
        transform=args.transform,
        scale=not args.no_scale
    )
    
    # Run UMAP
    embedding = run_umap(
        X=X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.umap_metric,
        random_state=args.random_state
    )
    
    # Run HDBSCAN
    cluster_labels = run_hdbscan(
        embedding=embedding,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.hdbscan_metric
    )
    
    # Create output directory
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("\n[INFO] Saving results...")
    results_df = pd.DataFrame({
        'sample': count_matrix.index,
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'cluster': cluster_labels
    })
    
    # Add metadata
    for col in metadata_df.columns:
        results_df[col] = metadata_df[col].values
    
    results_file = output_prefix.parent / f"{output_prefix.stem}_results.tsv"
    results_df.to_csv(results_file, sep='\t', index=False)
    print(f"[OK] Saved results: {results_file}")
    
    # Generate plots
    print("\n" + "="*60)
    print("[INFO] Generating plots...")
    print("="*60)
    
    # Plot 1: Colored by Depth (using Color column)
    if args.depth_col in metadata_df.columns and args.color_col in metadata_df.columns:
        print(f"\n[INFO] Creating UMAP plot colored by {args.depth_col}...")
        
        # Build color map from metadata
        depth_color_df = metadata_df[[args.depth_col, args.color_col]].dropna().drop_duplicates()
        color_map = dict(zip(depth_color_df[args.depth_col], depth_color_df[args.color_col]))
        
        for fmt in formats:
            output_file = output_prefix.parent / f"{output_prefix.stem}_depth.{fmt}"
            plot_umap_scatter(
                embedding=embedding,
                colors=metadata_df[args.depth_col],
                color_map=color_map,
                title=f"UMAP Projection Colored by {args.depth_col}",
                xlabel="UMAP1",
                ylabel="UMAP2",
                legend_title=args.depth_col,
                output_file=output_file,
                figsize=figsize,
                dpi=args.dpi,
                alpha=args.alpha,
                point_size=args.point_size
            )
    else:
        print(f"[WARN] Skipping depth plot: {args.depth_col} or {args.color_col} not in metadata")
    
    # Plot 2: Colored by Month
    if args.month_col in metadata_df.columns:
        print(f"\n[INFO] Creating UMAP plot colored by {args.month_col}...")
        
        for fmt in formats:
            output_file = output_prefix.parent / f"{output_prefix.stem}_month.{fmt}"
            plot_umap_scatter(
                embedding=embedding,
                colors=metadata_df[args.month_col],
                color_map=None,  # Auto-assign colors
                title=f"UMAP Projection Colored by {args.month_col}",
                xlabel="UMAP1",
                ylabel="UMAP2",
                legend_title=args.month_col,
                output_file=output_file,
                figsize=figsize,
                dpi=args.dpi,
                alpha=args.alpha,
                point_size=args.point_size
            )
    else:
        print(f"[WARN] Skipping month plot: {args.month_col} not in metadata")
    
    # Plot 3: Colored by HDBSCAN clusters
    print("\n[INFO] Creating UMAP plot colored by HDBSCAN clusters...")
    
    cluster_series = pd.Series(cluster_labels, index=count_matrix.index)
    
    for fmt in formats:
        output_file = output_prefix.parent / f"{output_prefix.stem}_clusters.{fmt}"
        plot_umap_scatter(
            embedding=embedding,
            colors=cluster_series,
            color_map=None,  # Auto-assign colors
            title="UMAP Projection with HDBSCAN Clusters",
            xlabel="UMAP1",
            ylabel="UMAP2",
            legend_title="Cluster",
            output_file=output_file,
            figsize=figsize,
            dpi=args.dpi,
            alpha=args.alpha,
            point_size=args.point_size
        )
    
    print("\n" + "="*60)
    print("[OK] Pipeline complete!")
    print("="*60)
    print(f"Results saved to: {results_file}")
    print(f"Plots saved with prefix: {output_prefix}")


if __name__ == "__main__":
    main()