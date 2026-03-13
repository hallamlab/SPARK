#!/usr/bin/env python3
"""
ASV Analysis Script

This script performs Center-Log Ratio (CLR) transformation on ASV count data,
applies UMAP for dimensionality reduction, and uses HDBSCAN for clustering
both samples and ASVs.

Usage:
    python asv_analysis.py \
        --count_table path/to/count_table.csv \
        --taxonomy path/to/taxonomy.csv \
        --fasta path/to/sequences.fasta \
        --output_prefix results/output

Author: Your Name
Date: YYYY-MM-DD
"""

import argparse
import pandas as pd
import numpy as np
import umap
import umap.plot
import bokeh.io
from bokeh.plotting import output_file, save
import hdbscan
import os
import sys
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import networkx as nx
from scipy.stats import spearmanr
from tqdm import tqdm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Perform CLR transformation, UMAP dimensionality reduction, and HDBSCAN clustering on ASV data."
    )
    parser.add_argument(
        "--count_table",
        required=True,
        help="Path to the ASV count table CSV file. Columns should be ASVs and rows should be samples.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to the sample metadata CSV file.",
    )
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to the ASV taxonomy CSV file.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the ASV FASTA file containing sequences.",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        help="Prefix for all output files.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum count threshold for ASV filtering. ASVs must have at least one count >= min_count in any sample to be retained. Default is 1.",
    )
    return parser.parse_args()

def load_count_table(count_table_path):
    """
    Loads the ASV count table from a CSV file.

    Args:
        count_table_path (str): Path to the count table CSV.

    Returns:
        pd.DataFrame: ASV count data with samples as rows and ASVs as columns.
    """
    try:
        count_df = pd.read_csv(count_table_path, header=0, sep='\t', index_col=0).T
        print(f"Loaded count table with shape {count_df.shape}")
        return count_df
    except Exception as e:
        print(f"Error loading count table: {e}")
        sys.exit(1)

def load_wgs_table(wgs_table):
    """
    Loads the ASV count table from a CSV file.

    Args:
        count_table_path (str): Path to the count table CSV.

    Returns:
        pd.DataFrame: ASV count data with samples as rows and ASVs as columns.
    """
    try:
        wgs_df = pd.read_csv(wgs_table, header=0, sep='\t')
        print(f"Loaded WGS table with shape {wgs_df.shape}")
        return wgs_df
    except Exception as e:
        print(f"Error loading count table: {e}")
        sys.exit(1)

def load_metadata_table(metadata_table_path):
    """
    Loads the sample metadata table from a CSV file.

    Args:
        metadata_table_path (str): Path to the count table CSV.

    Returns:
        pd.DataFrame: ASV count data with samples as rows and ASVs as columns.
    """
    #try:
    metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t', index_col=0)
    metadata_df['Batch'] = metadata_df['DNA_plate']
    
    print(f"Loaded metadata table with shape {metadata_df.shape}")
    return metadata_df
    #except Exception as e:
    #    print(f"Error loading metadata table: {e}")
    #    sys.exit(1)

def filter_asvs(count_df, metadata_df, min_count):
    """
    Filters out ASVs that do not have at least one count >= min_count across all samples.

    Args:
        count_df (pd.DataFrame): ASV count data.
        min_count (int): Minimum count threshold.

    Returns:
        pd.DataFrame: Filtered ASV count data.
    """
    #try:
    # Identify ASVs with at least one count >= min_count

    count_df_filtered = count_df.copy()
    num_asvs_before = count_df_filtered.shape[1]
    #count_df_filtered = count_df_filtered.loc[:, count_df_filtered.mean(axis=0) >= 1]
    count_df_filtered = count_df_filtered.loc[:, (count_df_filtered >= min_count).sum() >= 1]
    count_df_filtered['total_counts'] = count_df_filtered.sum(axis=1)
    count_df_filtered = count_df_filtered.sort_values('total_counts', ascending=False)
    count_df_filtered['sample'] = [x.replace('_L001', '') for x in count_df_filtered.index]
    count_df_filtered = count_df_filtered.drop_duplicates(subset=['sample'], keep='first')
    num_asvs_after = count_df_filtered.shape[1] - 2
    num_asvs_removed = num_asvs_before - num_asvs_after

    metadata_df = metadata_df.reset_index()
    count_df_filtered = count_df_filtered.reset_index()
    common_samples = list(set(count_df_filtered['sample']).intersection(set(metadata_df['sample'])))
    count_df_filtered = count_df_filtered.loc[count_df_filtered['sample'].isin(common_samples)]
    metadata_df = metadata_df.loc[metadata_df['sample'].isin(common_samples)]

    count_df_filtered.set_index('sample', inplace=True)
    count_df_filtered.drop(columns=['total_counts', 'index'], inplace=True)
    metadata_df.set_index('sample', inplace=True)

    # Reorder metadata to match count table
    metadata_df = metadata_df.loc[count_df_filtered.index]

    print(f"Filtered ASVs based on min_count={min_count}: Removed {num_asvs_removed} ASVs, retained {num_asvs_after} ASVs.")
    return count_df_filtered, metadata_df
    #except Exception as e:
    #    print(f"Error during ASV filtering: {e}")
    #    sys.exit(1)

def parse_taxonomy(taxonomy_df):
    """
    Parses the Taxonomy column to extract taxonomic levels, 
    particularly Phylum and other levels. If a level is unknown,
    it is assigned as 'unknown_<last_known>' where '<last_known>'
    is the last known taxon name from a higher rank.

    Args:
        taxonomy_df (pd.DataFrame): DataFrame containing 'Sequence_ID' and 'Taxonomy' columns.

    Returns:
        pd.DataFrame: Updated taxonomy DataFrame with separate columns for each taxonomic level.
    """
    # Define the taxonomic levels in order
    tax_levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Initialize columns for each taxonomic level
    for level in tax_levels:
        taxonomy_df[level] = 'Unknown'  # Default value

    # Iterate over each taxonomy string to extract taxonomic levels
    for idx, row in taxonomy_df.iterrows():
        taxonomy_str = row['Taxonomy']
        taxa = taxonomy_str.split(';')
        
        # Keep track of the last known taxon name
        last_known = None
        
        for taxon in taxa:
            taxon = taxon.strip()  # Remove leading/trailing whitespace
            if taxon:
                if '__' in taxon:
                    prefix, name = taxon.split('__', 1)
                    prefix = prefix.lower().strip()
                    name = name.strip()

                    # Determine which column to fill based on prefix
                    level_col = None
                    if prefix.startswith('d'):
                        level_col = 'Domain'
                    elif prefix.startswith('p'):
                        level_col = 'Phylum'
                    elif prefix.startswith('c'):
                        level_col = 'Class'
                    elif prefix.startswith('o'):
                        level_col = 'Order'
                    elif prefix.startswith('f'):
                        level_col = 'Family'
                    elif prefix.startswith('g'):
                        level_col = 'Genus'
                    elif prefix.startswith('s'):
                        level_col = 'Species'

                    if level_col:
                        if name:
                            # We have a known name for this rank
                            taxonomy_df.at[idx, level_col] = name
                            last_known = name
                        else:
                            # Unknown name for this rank
                            if last_known:
                                taxonomy_df.at[idx, level_col] = f'unknown_{last_known}'
                            else:
                                # If no last known is available yet, just 'unknown'
                                taxonomy_df.at[idx, level_col] = 'unknown'
    print("Parsed taxonomy and extracted taxonomic levels with unknown_<last_known> labeling.")
    return taxonomy_df

def load_taxonomy(taxonomy_path):
    """
    Loads and parses ASV taxonomy from a CSV file.

    Args:
        taxonomy_path (str): Path to the taxonomy CSV.

    Returns:
        pd.DataFrame: Parsed ASV taxonomy data with separate columns for taxonomic levels.
    """
    try:
        taxonomy_df = pd.read_csv(taxonomy_path, sep='\t')  # Assuming tab-separated based on provided sample
        if 'Sequence_ID' not in taxonomy_df.columns or 'Taxonomy' not in taxonomy_df.columns:
            print("Error: Taxonomy file must contain 'Sequence_ID' and 'Taxonomy' columns.")
            sys.exit(1)
        taxonomy_df.set_index('Sequence_ID', inplace=True)
        print(f"Loaded taxonomy data with shape {taxonomy_df.shape}")
        # Parse taxonomy to extract taxonomic levels
        taxonomy_df = parse_taxonomy(taxonomy_df)
        return taxonomy_df
    except Exception as e:
        print(f"Error loading taxonomy file: {e}")
        sys.exit(1)

def load_fasta(fasta_path):
    """
    Loads ASV sequences from a FASTA file.

    Args:
        fasta_path (str): Path to the FASTA file.

    Returns:
        dict: Dictionary mapping ASV IDs to sequences.
    """
    try:
        fasta_dict = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
        print(f"Loaded {len(fasta_dict)} sequences from FASTA")
        return fasta_dict
    except Exception as e:
        print(f"Error loading FASTA file: {e}")
        sys.exit(1)

def save_fasta(asv_list, fasta_dict, output_file):
    """
    Save a dictionary of sequences to a FASTA file.

    Parameters:
        fasta_dict (dict): Dictionary where keys are sequence identifiers (headers) and values are sequences.
        output_file (str): Path to the output FASTA file.

    Returns:
        None
    """
    with open(output_file, 'w') as f:
        for header, sequence in fasta_dict.items():
            if header in asv_list:
                f.write(f">{header}\n")
                # Split the sequence into lines of 80 characters for readability
                f.write("\n".join(sequence[i:i+80] for i in range(0, len(sequence), 80)))
                f.write("\n")

def compute_relative_abundance(count_df):
    """
    Computes relative abundance per sample.

    Args:
        count_df (pd.DataFrame): ASV count data.

    Returns:
        pd.DataFrame: Relative abundance data.
    """
    try:
        relative_abundance = count_df.div(count_df.sum(axis=1), axis=0)
        print("Computed relative abundance per sample.")
        return relative_abundance
    except Exception as e:
        print(f"Error computing relative abundance: {e}")
        sys.exit(1)

def clr_transform(rel_abundance_df):
    """
    Applies Center-Log Ratio (CLR) transformation to the relative abundance data.

    Args:
        rel_abundance_df (pd.DataFrame): ASV relative abundance data.

    Returns:
        pd.DataFrame: CLR-transformed data.
    """
    # Replace zeros with a small pseudocount to avoid log(0)
    pseudocount = 1e-6
    rel_abundance_nonzero = rel_abundance_df.replace(0, pseudocount)
    
    # Calculate geometric mean for each sample
    geom_mean = rel_abundance_nonzero.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
    
    # Apply CLR: log(x / geometric mean)
    clr_df = np.log(rel_abundance_nonzero.div(geom_mean, axis=0))
    
    print("Applied CLR transformation to relative abundance data.")
    return clr_df

def batch_correct(data_df, metadata_df, batch_col='Plate'):
    """
    Applies batch correction to the data using ComBat from Scanpy.

    Args:
        data_df (pd.DataFrame): Data to be batch corrected (rows as samples).
        metadata_df (pd.DataFrame): Metadata containing batch information.
        batch_col (str): Column name in metadata indicating batch.

    Returns:
        pd.DataFrame: Batch-corrected data.
    """
    try:
        # Ensure that the samples in data_df and metadata_df match
        if not data_df.index.equals(metadata_df.index):
            print("Error: Sample IDs in count table and metadata do not match.")
            sys.exit(1)
        
        # Create an AnnData object as required by Scanpy's batch correction
        adata = sc.AnnData(X=data_df.values, obs=metadata_df[[batch_col]].copy(), var=pd.DataFrame(index=data_df.columns))
        
        # Apply ComBat batch correction
        sc.pp.combat(adata, key=batch_col)
        
        # Create a DataFrame from the corrected data
        corrected_df = pd.DataFrame(adata.X, index=data_df.index, columns=data_df.columns)
        
        print("Applied batch correction using ComBat.")
        return corrected_df
    except Exception as e:
        print(f"Error during batch correction: {e}")
        sys.exit(1)

def perform_umap(data, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Performs UMAP dimensionality reduction on the data.

    Args:
        data (pd.DataFrame): Input data.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.
        random_state (int): Random state for reproducibility.

    Returns:
        umap.UMAP: Fitted UMAP reducer.
        pd.DataFrame: DataFrame with UMAP embeddings (X and Y).
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(data)
    umap_df = pd.DataFrame(embedding, index=data.index, columns=["UMAP1", "UMAP2"])
    print(f"Performed UMAP dimensionality reduction. Embedding shape: {umap_df.shape}")
    return reducer, umap_df

def save_umap_plots(reducer, metadata_df, output_base_path):
    """
    Generates and saves UMAP plots as static images using matplotlib.

    Args:
        reducer (umap.UMAP): Fitted UMAP reducer.
        output_base_path (str): Base file path for saving plots (without extension).
    """
    # Original UMAP scatter plot
    plt.figure(figsize=(10, 10))
    umap.plot.points(reducer ) #, labels=metadata_df['Cluster'])
    plt.title("UMAP Scatter Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_scatter.png", dpi=600)
    plt.close()
    print(f"Saved UMAP scatter plot to {output_base_path}_scatter.png")

    # Connectivity plot
    plt.figure(figsize=(10, 10))
    umap.plot.connectivity(reducer, show_points=False, labels=metadata_df['Cluster'])
    plt.title("UMAP Connectivity Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity.png", dpi=600)
    plt.close()
    print(f"Saved UMAP connectivity plot to {output_base_path}_connectivity.png")

    plt.figure(figsize=(10, 10))
    # Generate the connectivity plot with edge bundling and points
    umap.plot.connectivity(
        reducer,
        edge_bundling='hammer',
        show_points=False,
        #labels=metadata_df['Cluster'],  # Pass cluster labels
        #cmap='black',  # Colormap for points
        edge_cmap='gray_r',  # Colormap for edges
    )
    # Add title and save the plot
    plt.title("UMAP Connectivity Plot with Edge Bundling and Points")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity_bundled.png", dpi=600)
    plt.close()
    print(f"Saved bundled connectivity plot with points to {output_base_path}_connectivity_bundled.png")

    # Diagnostic plot
    plt.figure(figsize=(10, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='pca')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_pca.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_pca.png")
    
    # Diagnostic plot
    plt.figure(figsize=(10, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='vq')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_vq.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_vq.png")

    # Diagnostic plot
    plt.figure(figsize=(10, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='local_dim')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_local_dim.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_local_dim.png")

    # Diagnostic plot
    plt.figure(figsize=(10, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_neighborhood.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_neighborhood.png")

def perform_hdbscan(umap_df, min_cluster_size=5, min_samples=None):
    """
    Performs HDBSCAN clustering on UMAP embeddings.

    Args:
        umap_df (pd.DataFrame): UMAP embeddings.
        min_cluster_size (int): Minimum size of clusters.
        min_samples (int or None): Minimum number of samples in a neighborhood for a point to be considered a core point.

    Returns:
        pd.Series: Cluster labels for each data point.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(umap_df)
    print(f"Performed HDBSCAN clustering. Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    return pd.Series(cluster_labels, index=umap_df.index, name="Cluster")

def save_dataframe(df, filepath):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame or pd.Series): Data to save.
        filepath (str): Path to the output CSV file.
    """
    try:
        df.to_csv(filepath, sep='\t')
        print(f"Saved data to {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")
        sys.exit(1)

def plot_sample_clusters(sample_umap, sample_clusters, metadata_df, output_path, meta, m_color="black"):
    """
    Generates and saves a UMAP scatter plot for samples colored by cluster or a continuous variable.

    Args:
        sample_umap (pd.DataFrame): UMAP embeddings for samples.
        sample_clusters (pd.Series): Cluster assignments for samples.
        metadata_df (pd.DataFrame): Metadata for samples.
        output_path (str): Path to save the plot.
        meta (str): Column name for coloring. Can be categorical or continuous.
    """
    plot_df = sample_umap.join(sample_clusters).join(metadata_df)
    # Handle NaNs by dropping
    plot_df = plot_df.dropna(subset=[meta, 'UMAP1', 'UMAP2'])

    plt.figure(figsize=(10, 8))
    
    if ((pd.api.types.is_numeric_dtype(plot_df[meta])) & (meta not in ['Depth', 'Month'])):
        # Define custom colormap from white to #D95F02
        #custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', m_color])
        # Normalize the meta for sizing
        size_min, size_max = 20, 200  # Adjust these values as needed
        meta_norm = (plot_df[meta] - plot_df[meta].min()) / (plot_df[meta].max() - plot_df[meta].min())
        sizes = size_min + meta_norm * (size_max - size_min)
       
        scatter = plt.scatter(
            plot_df["UMAP1"],
            plot_df["UMAP2"],
            c=m_color, #plot_df[meta],
            s=sizes,
            #cmap=custom_cmap,
            alpha=0.7,
            edgecolors='gray',
            linewidths=0.5
        )
        # Create a legend for sizes
        # Define some example sizes
        for size in [size_min, (size_min + size_max)/2, size_max]:
            plt.scatter([], [], c='gray', alpha=0.7, s=size, label=f'{(size - size_min)/(size_max - size_min)*(plot_df[meta].max() - plot_df[meta].min()) + plot_df[meta].min():.2f}')
        
        plt.legend(title=meta, scatterpoints=1, labelspacing=1, title_fontsize='13', fontsize='11', loc='upper right', bbox_to_anchor=(1.3, 1))
        
        #plt.colorbar(scatter, label=meta)
    else:
        num_categories = plot_df[meta].nunique()
        #palette = sns.color_palette("hsv", num_categories)[::-1]
        sns.scatterplot(
            data=plot_df,
            x="UMAP1",
            y="UMAP2",
            hue=meta,
            palette=m_color,
            legend="full",
            alpha=0.7,
            edgecolors='gray',
            linewidths=0.5
        )
        plt.legend(title=meta, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title(f"UMAP Projection {'Sized by' if pd.api.types.is_numeric_dtype(plot_df[meta]) and meta not in ['Depth', 'Month'] else 'Colored by'} {meta}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")

    # Invert axes
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved sample cluster plot to {output_path}")

def plot_depth_profiles(metadata_df, metrics, m_colors, output_dir):
    metadata_df['Depth'] = pd.to_numeric(metadata_df['Depth'], errors='coerce')

    num_metrics = len(metrics)
    num_cols = 3
    num_rows = int(np.ceil(num_metrics / num_cols))



    # Create a subplot figure
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        shared_yaxes=True,
        subplot_titles=[]#f'Depth Profile of {metric}' for metric in metrics]
    )

    for idx, (metric, m_col) in enumerate(zip(metrics, m_colors)):
        metadata_df[metric] = pd.to_numeric(metadata_df[metric], errors='coerce')
        df = metadata_df.dropna(subset=['Depth', metric])

        # Group data by Depth
        grouped = df.groupby('Depth')
        mean_values = grouped[metric].mean()
        std_values = grouped[metric].std()
        depth_values = mean_values.index.astype(float)
        lower_bound = mean_values - std_values
        upper_bound = mean_values + std_values

        # Sort the data by depth
        sorted_indices = np.argsort(depth_values)
        depth_values = depth_values[sorted_indices]
        mean_values = mean_values.values[sorted_indices]
        lower_bound = lower_bound.values[sorted_indices]
        upper_bound = upper_bound.values[sorted_indices]

        row = int(idx / num_cols) + 1
        col = int(idx % num_cols) + 1

        # Add the mean line
        fig.add_trace(
            go.Scatter(
                x=mean_values,
                y=depth_values,
                mode='lines',
                line=dict(color=m_col),
                name=metric,
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add the shaded area for variance
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lower_bound, upper_bound[::-1]]),
                y=np.concatenate([depth_values, depth_values[::-1]]),
                fill='toself',
                fillcolor=m_col,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add data points as small black dots
        fig.add_trace(
            go.Scatter(
                x=df[metric],
                y=df['Depth'],
                mode='markers',
                marker=dict(color='black', size=4),
                name='Data Points',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Invert the y-axis
        fig.update_yaxes(autorange='reversed', row=row, col=col)
        # Set x-axis title
        fig.update_xaxes(title_text=metric, row=row, col=col)
        # Set y-axis title only for the first column
        if col == 1:
            fig.update_yaxes(title_text='Depth (m)', row=row, col=col)
    
    fig.update_layout(
        plot_bgcolor='white',   # Removes the gray background in the plotting area
        paper_bgcolor='white',  # Removes any surrounding background
    )

    # Remove gridlines from all axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    # Update layout for presentation:
    # - Decrease total height for thinner plots.
    # - Increase font sizes for better visibility.
    fig.update_layout(
        height=800 * num_rows,     # Reduce height for thinner plots
        width=400 * num_cols,
        title_text="",
        showlegend=False,
        font=dict(
            family="Arial",       # choose a readable font
            size=20,              # increase overall font size
        ),
        title_font_size=20         # Increase title font size
    )

    # Further customize axes font sizes if needed
    # For example, to increase tick label fonts:
    fig.update_xaxes(tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20))

    # Save to PNG and PDF
    plot_filename_png = os.path.join(f"{output_dir}_depth_profiles.png")
    plot_filename_pdf = os.path.join(f"{output_dir}_depth_profiles.pdf")
    fig.write_image(plot_filename_png, scale=6.25)
    fig.write_image(plot_filename_pdf)
    print(f"Saved depth profile plots to {plot_filename_png} and {plot_filename_pdf}")


def plot_asv_depth_profiles(asv_df, asvs, m_colors, tax_df, output_dir):
    num_asvs = len(asvs)
    num_cols = 2
    num_rows = int(np.ceil(num_asvs / num_cols))

    # Create a subplot figure
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[f'Depth Profile of {asv}' for asv in asvs]
    )

    for idx, (asv, m_col) in enumerate(zip(asvs, m_colors)):
        asv_sub_df = asv_df.loc[asv_df['ASV'] ==    asv]
        df = asv_sub_df.dropna(subset=['Depth', 'relative_abundance'])

        # Group data by Depth
        grouped = df.groupby('Depth')
        mean_values = grouped['relative_abundance'].mean()
        std_values = grouped['relative_abundance'].std()
        depth_values = mean_values.index.astype(float)
        lower_bound = mean_values - std_values
        upper_bound = mean_values + std_values

        # Sort the data by depth
        sorted_indices = np.argsort(depth_values)
        depth_values = depth_values[sorted_indices]
        mean_values = mean_values.values[sorted_indices]
        lower_bound = lower_bound.values[sorted_indices]
        upper_bound = upper_bound.values[sorted_indices]

        row = int(idx / num_cols) + 1
        col = int(idx % num_cols) + 1

        # Add the mean line
        fig.add_trace(
            go.Scatter(
                x=mean_values,
                y=depth_values,
                mode='lines',
                line=dict(color=m_col),
                name='relative_abundance',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add the shaded area for variance
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lower_bound, upper_bound[::-1]]),
                y=np.concatenate([depth_values, depth_values[::-1]]),
                fill='toself',
                fillcolor=m_col,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add data points as small black dots
        fig.add_trace(
            go.Scatter(
                x=df['relative_abundance'],
                y=df['Depth'],
                mode='markers',
                marker=dict(color='black', size=4),
                name='Data Points',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Invert the y-axis
        fig.update_yaxes(autorange='reversed', row=row, col=col)
        # Set x-axis title
        fig.update_xaxes(title_text='relative_abundance', row=row, col=col)
        # Set y-axis title only for the first column
        if col == 1:
            fig.update_yaxes(title_text='Depth', row=row, col=col)
    
    # Make all x-axes match each other
    fig.update_xaxes(matches='x')

    # Make all y-axes match each other and reverse their direction
    fig.update_yaxes(matches='y', autorange='reversed')

    fig.update_layout(
        height=1200 * num_rows,
        width=800 * num_cols,
        title_text="Depth Profiles",
        showlegend=False
    )
    # Save to PNG and PDF
    plot_filename_png = os.path.join(f"{output_dir}_ASV_depth_profiles.png")
    plot_filename_pdf = os.path.join(f"{output_dir}_ASV_depth_profiles.pdf")
    fig.write_image(plot_filename_png)
    fig.write_image(plot_filename_pdf)
    print(f"Saved depth profile plots to {plot_filename_png} and {plot_filename_pdf}")

def convert_rgb(x):
    r, g, b = x
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    color_str = f"rgb({r_int},{g_int},{b_int})"

    return color_str

def plot_metag_depth_profiles(metag_df, asvs, m_colors, tax_df, output_dir):
    num_asvs = len(asvs)
    num_cols = 3
    num_rows = int(np.ceil(num_asvs / num_cols))

    # Create a subplot figure
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[f'Depth Profile of {asv}' for asv in asvs]
    )

    grp_list = ['mg_id', 'Cruise', 'Depth']
    for idx, (asv, m_col) in enumerate(zip(asvs, m_colors)):
        if asv == 'SUP/Thio_clade':
            asv_sub_df = metag_df.loc[metag_df['genus'].isin(asvs)]
            asv_sub_df = asv_sub_df.groupby(grp_list)['uTPM'].sum().reset_index()
        else:
            asv_sub_df = metag_df.loc[metag_df['genus'] ==  asv]
        df = asv_sub_df.dropna(subset=['Depth', 'uTPM'])

        # Group data by Depth
        grouped = df.groupby('Depth')
        mean_values = grouped['uTPM'].mean()
        std_values = grouped['uTPM'].std()
        depth_values = mean_values.index.astype(float)
        lower_bound = mean_values - std_values
        upper_bound = mean_values + std_values

        # Sort the data by depth
        sorted_indices = np.argsort(depth_values)
        depth_values = depth_values[sorted_indices]
        mean_values = mean_values.values[sorted_indices]
        lower_bound = lower_bound.values[sorted_indices]
        upper_bound = upper_bound.values[sorted_indices]

        row = int(idx / num_cols) + 1
        col = int(idx % num_cols) + 1

        # Add the mean line
        fig.add_trace(
            go.Scatter(
                x=mean_values,
                y=depth_values,
                mode='lines',
                line=dict(color=m_col),
                name='uTPM',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add the shaded area for variance
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lower_bound, upper_bound[::-1]]),
                y=np.concatenate([depth_values, depth_values[::-1]]),
                fill='toself',
                fillcolor=m_col,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add data points as small black dots
        fig.add_trace(
            go.Scatter(
                x=df['uTPM'],
                y=df['Depth'],
                mode='markers',
                marker=dict(color='black', size=4),
                name='Data Points',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Invert the y-axis
        fig.update_yaxes(autorange='reversed', row=row, col=col)
        # Set x-axis title
        fig.update_xaxes(title_text='uTPM', row=row, col=col)
        # Set y-axis title only for the first column
        if col == 1:
            fig.update_yaxes(title_text='Depth', row=row, col=col)
    
    # Make all x-axes match each other
    fig.update_xaxes(matches='x')

    # Make all y-axes match each other and reverse their direction
    fig.update_yaxes(matches='y', autorange='reversed')

    fig.update_layout(
        height=1200 * num_rows,
        width=800 * num_cols,
        title_text="Depth Profiles",
        showlegend=False,
        font=dict(
            family="Arial",       # choose a readable font
            size=20,
        )
    )
    # Further customize axes font sizes if needed
    # For example, to increase tick label fonts:
    fig.update_xaxes(tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20))

    # Save to PNG and PDF
    plot_filename_png = os.path.join(f"{output_dir}_METAG_depth_profiles.png")
    plot_filename_pdf = os.path.join(f"{output_dir}_METAG_depth_profiles.pdf")
    fig.write_image(plot_filename_png)
    fig.write_image(plot_filename_pdf)
    print(f"Saved depth profile plots to {plot_filename_png} and {plot_filename_pdf}")

def plot_metat_depth_profiles(metat_df, asvs, m_colors, tax_df, output_dir):
    num_asvs = len(asvs)
    num_cols = 3
    num_rows = int(np.ceil(num_asvs / num_cols))

    # Create a subplot figure
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[f'Depth Profile of {asv}' for asv in asvs]
    )

    grp_list = ['mg_id', 'Cruise', 'Depth']
    for idx, (asv, m_col) in enumerate(zip(asvs, m_colors)):
        if asv == 'SUP/Thio_clade':
            asv_sub_df = metat_df.loc[metat_df['genus'].isin(asvs)]
            asv_sub_df = asv_sub_df.groupby(grp_list)['uTPM'].sum().reset_index()
        else:
            asv_sub_df = metat_df.loc[metat_df['genus'] ==  asv]
        df = asv_sub_df.dropna(subset=['Depth', 'uTPM'])

        # Group data by Depth
        grouped = df.groupby('Depth')
        mean_values = grouped['uTPM'].mean()
        std_values = grouped['uTPM'].std()
        depth_values = mean_values.index.astype(float)
        lower_bound = mean_values - std_values
        upper_bound = mean_values + std_values

        # Sort the data by depth
        sorted_indices = np.argsort(depth_values)
        depth_values = depth_values[sorted_indices]
        mean_values = mean_values.values[sorted_indices]
        lower_bound = lower_bound.values[sorted_indices]
        upper_bound = upper_bound.values[sorted_indices]

        row = int(idx / num_cols) + 1
        col = int(idx % num_cols) + 1

        # Add the mean line
        fig.add_trace(
            go.Scatter(
                x=mean_values,
                y=depth_values,
                mode='lines',
                line=dict(color=m_col),
                name='uTPM',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add the shaded area for variance
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([lower_bound, upper_bound[::-1]]),
                y=np.concatenate([depth_values, depth_values[::-1]]),
                fill='toself',
                fillcolor=m_col,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Add data points as small black dots
        fig.add_trace(
            go.Scatter(
                x=df['uTPM'],
                y=df['Depth'],
                mode='markers',
                marker=dict(color='black', size=4),
                name='Data Points',
                showlegend=False
            ),
            row=row,
            col=col
        )

        # Invert the y-axis
        fig.update_yaxes(autorange='reversed', row=row, col=col)
        # Set x-axis title
        fig.update_xaxes(title_text='uTPM', row=row, col=col)
        # Set y-axis title only for the first column
        if col == 1:
            fig.update_yaxes(title_text='Depth', row=row, col=col)
    
    # Make all x-axes match each other
    fig.update_xaxes(matches='x')

    # Make all y-axes match each other and reverse their direction
    fig.update_yaxes(matches='y', autorange='reversed')

    fig.update_layout(
        height=1200 * num_rows,
        width=800 * num_cols,
        title_text="Depth Profiles",
        showlegend=False,
        font=dict(
            family="Arial",       # choose a readable font
            size=20,
        )
    )
    # Further customize axes font sizes if needed
    # For example, to increase tick label fonts:
    fig.update_xaxes(tickfont=dict(size=20))
    fig.update_yaxes(tickfont=dict(size=20))

    # Save to PNG and PDF
    plot_filename_png = os.path.join(f"{output_dir}_METAT_depth_profiles.png")
    plot_filename_pdf = os.path.join(f"{output_dir}_METAT_depth_profiles.pdf")
    fig.write_image(plot_filename_png)
    fig.write_image(plot_filename_pdf)
    print(f"Saved depth profile plots to {plot_filename_png} and {plot_filename_pdf}")


def plot_asv_phylum(asv_umap, taxonomy_df, output_path):
    """
    Generates and saves a UMAP scatter plot for ASVs colored by phylum.

    Args:
        asv_umap (pd.DataFrame): UMAP embeddings for ASVs.
        taxonomy_df (pd.DataFrame): ASV taxonomy data.
        output_path (str): Path to save the plot.
    """
    # Ensure that taxonomy_df has a 'Phylum' column
    if 'Phylum' not in taxonomy_df.columns:
        print("Error: 'Phylum' column not found in taxonomy data.")
        sys.exit(1)

    # Merge UMAP embeddings with taxonomy data
    plot_df = asv_umap.copy()
    #plot_df['Phylum'] = [x.split('; ')[1].replace('p__', '') for x in plot_df.index]
    plot_df = plot_df.join(taxonomy_df, how='left')

    # Handle ASVs with missing phylum information
    plot_df['Phylum'] = plot_df['Phylum'].fillna('Unknown')

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Define a color palette. Limit to top 20 phyla and group others as 'Other' to avoid too many colors
    top_phyla = plot_df['Phylum'].value_counts().nlargest(20).index
    plot_df['Phylum_Categorized'] = plot_df['Phylum'].apply(lambda x: x if x in top_phyla else 'Other')
    num_phyla = plot_df['Phylum_Categorized'].nunique()
    palette = sns.color_palette("muted", num_phyla)

    # Create scatter plot
    sns.scatterplot(
        data=plot_df,
        x="UMAP1",
        y="UMAP2",
        hue="Phylum_Categorized",
        palette=palette,
        legend="full",
        alpha=0.6,
        edgecolor="gray",
        linewidth=0.5
    )

    plt.title("UMAP Projection of ASVs Colored by Phylum")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Phylum", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ASV phylum plot to {output_path}")

def plot_asv_clusters(asv_umap, output_path):
    """
    Generates and saves a UMAP scatter plot for ASVs colored by phylum.

    Args:
        asv_umap (pd.DataFrame): UMAP embeddings for ASVs.
        taxonomy_df (pd.DataFrame): ASV taxonomy data.
        output_path (str): Path to save the plot.
    """

    # Merge UMAP embeddings with taxonomy data
    plot_df = asv_umap.copy()

    # Handle ASVs with missing phylum information
    plot_df['Cluster'] = plot_df['Cluster'].fillna('Unknown')

    # Initialize the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Define a color palette. Limit to top 20 phyla and group others as 'Other' to avoid too many colors
    top_phyla = plot_df['Cluster'].value_counts().nlargest(20).index
    plot_df['Cluster_Categorized'] = plot_df['Cluster'].apply(lambda x: x if x in top_phyla else 'Other')
    num_phyla = plot_df['Cluster_Categorized'].nunique()
    palette = sns.color_palette("muted", num_phyla)

    # Create scatter plot
    sns.scatterplot(
        data=plot_df,
        x="UMAP1",
        y="UMAP2",
        hue="Cluster_Categorized",
        palette=palette,
        legend="full",
        alpha=0.6,
        edgecolor="gray",
        linewidth=0.5
    )

    plt.title("UMAP Projection of ASVs Colored by Cluster")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ASV Cluster plot to {output_path}")

def plot_bubbleplot(tax_df, count_df, meta_df, output_path):
    asv_list = list(count_df.loc[count_df['count'] >= 0]['ASV'].unique())
    count_df['ASV_ID'] = [int(x.replace('ASV', '')) for x in count_df['ASV']]

    # Merge tax and count info by ASV
    tax_df = tax_df.reset_index()
    meta_df = meta_df.reset_index()

    merged_df = pd.merge(count_df, tax_df, left_on='ASV', right_on='Sequence_ID', how='inner')
    merged_df = pd.merge(merged_df, meta_df, left_on='sample', right_on='sample', how='inner')
    merged_df['Depth'] = pd.to_numeric(merged_df['Depth'], errors='coerce')

    # Group by Phylum and depth and get mean relative abundance
    phylum_counts = merged_df.groupby(['Phylum', 'sample', 'Cruise', 'Depth'])['count'].sum().reset_index()

    # Next, to calculate relative abundance, you need a denominator. Typically, relative abundance is calculated within a subset.
    # For example, if you want the relative abundance of each 'col3' category within each (col1, col2) pair:
    group_sums = phylum_counts.groupby(['sample', 'Cruise', 'Depth'])['count'].transform('sum')

    # Compute relative abundance as (count for col3) / (total count for that col1,col2) * 100
    phylum_counts['relabund'] = (phylum_counts['count'] / group_sums) * 100
    phyla_top_list = phylum_counts.loc[phylum_counts['relabund'] >= 10]['Phylum'].unique()
    phylum_counts['Top_Phyla'] = [x if x in phyla_top_list else 'Other' for x in phylum_counts['Phylum']]
    phylum_counts = phylum_counts.groupby(['Top_Phyla', 'sample', 'Cruise', 'Depth'])['relabund'].sum().reset_index()
    phylum_counts = phylum_counts.replace(0, np.nan)

    phyla_list = list(phylum_counts['Top_Phyla'].unique())
    phyla_list.sort()
    color_map = {'Actinobacteriota': "black",
                 'Bacteroidota': "#1B9E77", 
                 'Campylobacterota': "#D95F02", 
                 'Cyanobacteria': 'green', 
                 'Desulfobacterota_I': "#0C5196", 
                 'Marinisomatota': "#7570B3", 
                 'Planctomycetota': "#66A61E", 
                 'Proteobacteria': "violet", 
                 'SAR324': 'skyblue', 
                 'Thermoplasmatota': "#E7298A", 
                 'Thermoproteota': 'tan',
                 'Other': 'lightgray'
                 }
    #palette = sns.color_palette("muted", num_phyla)
    #color_map = {phylum: color for phylum, color in zip(phyla_list, palette)}
    phylum_counts = phylum_counts.sort_values('Depth', ascending=False)
    
    g = sns.relplot(
        data=phylum_counts,
        x='Cruise',
        y='Depth',
        hue='Top_Phyla',              # Color points by Phylum
        size='relabund', # Map relative abundance to point size
        col='Top_Phyla',              # Facet by Phylum to separate subplots
        kind='scatter',
        palette=color_map,
        col_order=color_map.keys(),
        sizes=(5, 200),
        alpha=0.7,
        height=4,                  # Adjust height as needed
        aspect=1,
        col_wrap=4
    )
    
    # Get the min and max depth
    depth_min = 0 #merged_df['depth'].min()
    depth_max = 210 #merged_df['depth'].max()

    # Set the y-limits so that the larger values are at the bottom and smaller at the top
    # This effectively "inverts" the y-axis
    g.set(ylim=(depth_max, depth_min))
    
    # Move the legend outside the figure to the right
    g._legend.set_bbox_to_anchor((1.0, 0.5))
    g.fig.subplots_adjust(right=0.8) # Make room on the right side for legend

    #plt.tight_layout()

    # Save the plot
    png_file = f"{output_path}_asv_bubbleplot.png"
    pdf_file = f"{output_path}_asv_bubbleplot.pdf"
    plt.savefig(png_file, dpi=300)
    plt.savefig(pdf_file, dpi=300)
    plt.close()
    print(f"Saved ASV phylum barplot to {output_path}")

    return color_map

def plot_horizontal_bar_by_phylum(tax_df, count_df, meta_df, output_path, color_map):
    asv_list = list(count_df.loc[count_df['count'] >= 0]['ASV'].unique())
    count_df['ASV_ID'] = [int(x.replace('ASV', '')) for x in count_df['ASV']]

    # Merge tax and count info by ASV
    tax_df = tax_df.reset_index()
    meta_df = meta_df.reset_index()

    merged_df = pd.merge(count_df, tax_df, left_on='ASV', right_on='Sequence_ID', how='inner')
    merged_df = pd.merge(merged_df, meta_df, left_on='sample', right_on='sample', how='inner')
    merged_df['Depth'] = pd.to_numeric(merged_df['Depth'], errors='coerce')

    # Define top phyla and others
    top_phyla = merged_df.loc[merged_df['ASV'].isin(asv_list)]['Phylum'].unique()
    merged_df['Top_Phyla'] = merged_df['Phylum'].apply(lambda x: x if x in top_phyla else 'Other')

    # Group by Phylum and depth and get mean relative abundance
    phylum_counts = merged_df.groupby(['Top_Phyla', 'Depth'])['count'].sum().reset_index()

    # Pivot to get wide format: rows = depth, columns = phyla, values = relative_abundance
    pivot_df = phylum_counts.pivot(index='Depth', columns='Top_Phyla', values='count').fillna(0)

    # Sort by depth so that the shallowest depth is at the top (if desired)
    # If you want the deepest at the bottom and shallowest at the top, just ensure the order you want:
    # By default, barh will have the first row at the bottom. To invert it visually,
    # we can sort ascending and invert the axis after plotting.
    pivot_df = pivot_df.sort_index(ascending=True)
    # Normalize each row to sum to 100% (i.e., convert to percentages)
    row_sums = pivot_df.sum(axis=1)
    pivot_df = pivot_df.div(row_sums, axis=0) * 100
    
    # Suppose you've already identified the top_phyla as described:
    top_phyla = [x for x in pivot_df.columns if x in color_map.keys()]

    # Create a list of non-top phyla
    non_top_phyla = [col for col in pivot_df.columns if col not in top_phyla]
    print(top_phyla)
    print(non_top_phyla)

    # Sum all non-top phyla into one "Other" column
    pivot_df['Other'] = pivot_df[non_top_phyla].sum(axis=1)

    # Now select only the top_phyla and the new "Other" column
    final_columns = top_phyla + ['Other']
    pivot_df = pivot_df[final_columns]

    num_phyla = len(pivot_df.columns)
    palette = sns.color_palette("muted", num_phyla)

    # Plot the stacked horizontal bar plot
    ax = pivot_df.plot(
        kind='barh',
        stacked=True,
        figsize=(12, 10),
        color=[color_map[p] for p in pivot_df.columns],
        alpha=0.7
    )

    # Invert y-axis if you want deeper depths at the bottom (optional)
    # If your smallest depth number is supposed to be at the top,
    # you can invert the y-axis:
    ax.invert_yaxis()

    # Move the legend outside the figure
    leg = ax.legend(title='Phylum', bbox_to_anchor=(1.0, 0.5), loc='center left')
    plt.subplots_adjust(right=0.8)

    # Set labels
    ax.set_xlabel('Relative Abundance')
    ax.set_ylabel('Depth')

    # Save the plot
    png_file = f"{output_path}_phylum_horizontal_bar.png"
    pdf_file = f"{output_path}_phylum_horizontal_bar.pdf"
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ASV phylum barplot to {output_path}")

def plot_proteo_bubbleplot(tax_df, count_df, meta_df, output_path):
    asv_list = list(count_df.loc[count_df['count'] >= 0]['ASV'].unique())
    count_df['ASV_ID'] = [int(x.replace('ASV', '')) for x in count_df['ASV']]

    # Merge tax and count info by ASV
    tax_df = tax_df.reset_index()
    meta_df = meta_df.reset_index()

    merged_df = pd.merge(count_df, tax_df, left_on='ASV', right_on='Sequence_ID', how='inner')
    merged_df = pd.merge(merged_df, meta_df, left_on='sample', right_on='sample', how='inner')
    merged_df['Depth'] = pd.to_numeric(merged_df['Depth'], errors='coerce')

    proteo_df = merged_df.loc[merged_df['Phylum'] == 'Proteobacteria']

    # Group by Phylum and depth and get mean relative abundance
    phylum_counts = proteo_df.groupby(['Genus', 'sample', 'Cruise', 'Depth'])['count'].sum().reset_index()

    # Next, to calculate relative abundance, you need a denominator. Typically, relative abundance is calculated within a subset.
    # For example, if you want the relative abundance of each 'col3' category within each (col1, col2) pair:
    group_sums = phylum_counts.groupby(['sample', 'Cruise', 'Depth'])['count'].transform('sum')

    # Compute relative abundance as (count for col3) / (total count for that col1,col2) * 100
    phylum_counts['relabund'] = (phylum_counts['count'] / group_sums) * 100
    phyla_top_list = phylum_counts.loc[phylum_counts['relabund'] >= 25]['Genus'].unique()
    phylum_counts['Top_Genera'] = [x if x in phyla_top_list else 'Other' for x in phylum_counts['Genus']]
    phylum_counts = phylum_counts.groupby(['Top_Genera', 'sample', 'Cruise', 'Depth'])['relabund'].sum().reset_index()
    phylum_counts = phylum_counts.replace(0, np.nan)

    phyla_list = list(phylum_counts['Top_Genera'].unique())
    phyla_list.sort()
    color_map = {'Pseudothioglobus': (0.0, 0.4562523625023627, 1.0),
                 'Thiodubiliella': (0.14595412205706287, 0.0, 1.0),
                 'unknown_Nitrococcales': (0.724998818748819, 0.0, 1.0)
                 }
    phylum_counts = phylum_counts.loc[phylum_counts['Top_Genera'].isin(color_map.keys())]

    phylum_counts = phylum_counts.sort_values('Depth', ascending=False)
    g = sns.relplot(
        data=phylum_counts,
        x='Cruise',
        y='Depth',
        hue='Top_Genera',              # Color points by Phylum
        size='relabund', # Map relative abundance to point size
        col='Top_Genera',              # Facet by Phylum to separate subplots
        kind='scatter',
        palette=color_map,
        col_order=color_map.keys(),
        sizes=(5, 200),
        alpha=0.7,
        height=4,                  # Adjust height as needed
        aspect=1,
        col_wrap=3
    )
    
    # Get the min and max depth
    depth_min = 0 #merged_df['depth'].min()
    depth_max = 210 #merged_df['depth'].max()

    # Set the y-limits so that the larger values are at the bottom and smaller at the top
    # This effectively "inverts" the y-axis
    g.set(ylim=(depth_max, depth_min))
    
    # Move the legend outside the figure to the right
    g._legend.set_bbox_to_anchor((1.0, 0.5))
    g.fig.subplots_adjust(right=0.8) # Make room on the right side for legend

    #plt.tight_layout()

    # Save the plot
    png_file = f"{output_path}_Family_bubbleplot.png"
    pdf_file = f"{output_path}_Family_bubbleplot.pdf"
    plt.savefig(png_file, dpi=300)
    plt.savefig(pdf_file, dpi=300)
    plt.close()
    print(f"Saved ASV phylum barplot to {output_path}")

def plot_asv_bubbleplot(tax_df, count_df, meta_df, output_path):
    count_df['ASV_ID'] = [int(x.replace('ASV', '')) for x in count_df['ASV']]

    # Merge tax and count info by ASV
    tax_df = tax_df.reset_index()
    meta_df = meta_df.reset_index()

    merged_df = pd.merge(count_df, tax_df, left_on='ASV', right_on='Sequence_ID', how='inner')
    merged_df = pd.merge(merged_df, meta_df, left_on='sample', right_on='sample', how='inner')
    merged_df['Depth'] = pd.to_numeric(merged_df['Depth'], errors='coerce')

    # Group by Phylum and depth and get mean relative abundance
    phylum_counts = merged_df.groupby(['ASV_ID', 'sample'])['count'].sum().reset_index()

    # Next, to calculate relative abundance, you need a denominator. Typically, relative abundance is calculated within a subset.
    # For example, if you want the relative abundance of each 'col3' category within each (col1, col2) pair:
    group_sums = phylum_counts.groupby(['sample'])['count'].transform('sum')

    # Compute relative abundance as (count for col3) / (total count for that col1,col2) * 100
    phylum_counts['relabund'] = (phylum_counts['count'] / group_sums) * 100
    phylum_counts = phylum_counts.replace(0, np.nan)
    phylum_counts = phylum_counts.sort_values('ASV_ID', ascending=False)

    g = sns.relplot(
        data=phylum_counts,
        x='ASV_ID',
        y='sample',
        size='relabund', # Map relative abundance to point size
        kind='scatter',
        color='black',
        sizes=(5, 200),
        alpha=0.7,
        height=15,                  # Adjust height as needed
        aspect=1.9,
    )
    
    # Move the legend outside the figure to the right
    g._legend.set_bbox_to_anchor((1.0, 0.5))
    g.fig.subplots_adjust(right=0.8) # Make room on the right side for legend

    #plt.tight_layout()

    # Save the plot
    png_file = f"{output_path}_ASV_SAMPLE_bubbleplot.png"
    pdf_file = f"{output_path}_ASV_SAMPLE_bubbleplot.pdf"
    plt.savefig(png_file, dpi=600)
    plt.savefig(pdf_file, dpi=600)
    plt.close()
    print(f"Saved ASV phylum barplot to {output_path}")


def plot_cca_biplot(species_scores, env_scores, output_path='/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot.png', meta='Cluster', color_map=None, dpi_setting=600):
    """
    Generates and saves a CCA biplot using Seaborn, coloring ASVs by their cluster.
    Cluster '-1' is colored grey to represent 'Other'.
    
    Args:
        species_scores (pd.DataFrame): DataFrame containing 'CCA1', 'CCA2', and 'Cluster' columns for ASVs.
        env_scores (pd.DataFrame): DataFrame containing environmental variables with 'CCA1' and 'CCA2' as columns.
        output_path (str, optional): Path to save the plot. Defaults to '/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot.png'.
        dpi_setting (int, optional): Resolution of the saved plot. Defaults to 300.
    """
    
    # Ensure 'Cluster' column is of type string for consistent handling
    if meta == 'Cluster':
        species_scores[meta] = species_scores[meta].astype(int)

        unique_clusters = species_scores[meta].unique()
        clusters_except_minus1 = [cluster for cluster in unique_clusters if cluster != -1]
        num_clusters = len(clusters_except_minus1)
        palette = sns.color_palette("muted", num_clusters)
        color_map = {cluster: color for cluster, color in zip(clusters_except_minus1, palette)}
        color_map[-1] = (0.9, 0.9, 0.9)  # RGB tuple for grey
    
    if meta == 'Phylum':
        species_scores[meta] = [x if x in color_map.keys() else 'Other' for x in species_scores[meta]]

    if meta == 'Genus':
        species_scores.loc[species_scores['Order'] == 'Nitrococcales', meta] = 'Unknown_Nitrococcales'
        color_map = {'Pseudothioglobus': (0.0, 0.4562523625023627, 1.0),
                 'Thiodubiliella': (0.14595412205706287, 0.0, 1.0),
                 'Unknown_Nitrococcales': (0.724998818748819, 0.0, 1.0),
                 'Other': 'lightgray'
                 }
        species_scores[meta] = [x if x in color_map.keys() else 'Other' for x in species_scores[meta]]
    if meta == 'MAG_pair':
        color_map = {'True': 'black',
                     'False': 'lightgray'
                     }
    
    # Step 3: Initialize the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Step 4: Create Seaborn Scatter Plot
    sns.scatterplot(
        data=species_scores,
        x='CCA1',
        y='CCA2',
        hue=meta,
        palette=color_map,
        legend='full',
        alpha=0.7,
        edgecolor='gray',
        linewidth=0.5,
        s=50  # Adjust the size as needed
    )
    
    # Step 5: Plot Environmental Variables as Vectors
    ax = plt.gca()  # Get current axes
    
    for var in env_scores.index:
        c1 = env_scores.loc[var, 'CCA1']
        c2 = env_scores.loc[var, 'CCA2']
        ax.arrow(
            0, 0, c1, c2,
            color='red',
            width=0.005,
            head_width=0.05,
            length_includes_head=True,
            alpha=0.8
        )
        ax.text(
            c1 * 1.15, c2 * 1.15,
            var,
            color='red',
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold'
        )
    
    # Step 6: Customize Plot
    plt.xlabel('CCA1', fontsize=14)
    plt.ylabel('CCA2', fontsize=14)
    plt.title('CCA Biplot of ASVs and Environmental Variables', fontsize=16)
    
    # Show legend outside the plot
    plt.legend(
        title=meta,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
        fontsize=12
    )
    
    # Show grid
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Step 7: Adjust Plot Limits to Accommodate Arrows and Labels
    all_x = np.concatenate([species_scores['CCA1'], env_scores['CCA1']])
    all_y = np.concatenate([species_scores['CCA2'], env_scores['CCA2']])
    max_x = all_x.max() * 1.5
    min_x = all_x.min() * 1.1
    max_y = all_y.max() * 1.5
    min_y = all_y.min() * 1.1
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # Optimize layout
    plt.tight_layout()
    
    # Step 8: Save the Plot
    plt.savefig(output_path, dpi=dpi_setting)
    plt.close()
    
    print(f"Saved CCA biplot to {output_path}")
    return color_map


def main():
    """
    Main function to execute the ASV analysis workflow.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Load input data
    count_df = load_count_table(args.count_table)
    taxonomy_df = load_taxonomy(args.taxonomy)
    metadata_df = load_metadata_table(args.metadata)
    
    # Load the data
    #site_scores = pd.read_csv('/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_site_scores.tsv', sep='\t', index_col=0)
    #species_scores = pd.read_csv('/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_species_scores.tsv', sep='\t', index_col=0)
    #env_scores = pd.read_csv('/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_env_scores.tsv', sep='\t', index_col=0)
    #spiec_easi_df = pd.read_csv('/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/spieceasi/spieceasi_asv_transformed_distance.csv', sep='\t', index_col=0)
    #asv_mag_df = pd.read_csv('/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/lr_mags/ASV2MAG.tsv', sep='\t', header=0)
    #asv_mag_list = list(asv_mag_df['qseqid'].unique())
    #env_scores.index = [x.split('.')[0] for x in list(env_scores.index)]

    #species_scores = species_scores.join(spiec_easi_df)
    #species_scores['MAG_pair'] = ['True' if x in asv_mag_list else 'False' for x in species_scores.index.values]
    #species_scores['Cluster'] = species_scores['Cluster'].fillna(-1).astype(int)

    # Filter ASVs based on min_count
    count_df_filtered, metadata_df = filter_asvs(count_df, metadata_df, args.min_count)
    # only use samples that have metadata
    save_dataframe(count_df_filtered, f"{args.output_prefix}_filtered_count_table.csv")
    save_dataframe(metadata_df, f"{args.output_prefix}_metadata.csv")

    cnt_abund_df = count_df_filtered.reset_index().rename(columns={'index': 'sample'})
    cnt_long_df = cnt_abund_df.melt(id_vars='sample', var_name='ASV', value_name='count')

    # Save a FASTA of the filtered ASVs
    fasta_dict = load_fasta(args.fasta)
    save_fasta(list(count_df_filtered.columns), fasta_dict, f"{args.output_prefix}_ASV_representatives.fasta")

    # Compute relative abundance
    rel_abundance_df = compute_relative_abundance(count_df_filtered)
    save_dataframe(rel_abundance_df, f"{args.output_prefix}_relative_abundance.csv")

    # Apply CLR transformation
    clr_df = clr_transform(rel_abundance_df)
    save_dataframe(clr_df, f"{args.output_prefix}_clr_transformed.csv")

    # Apply batch correction using 'Batch' variable
    clr_batch_corrected_df = batch_correct(clr_df, metadata_df, batch_col='Batch')
    save_dataframe(clr_batch_corrected_df, f"{args.output_prefix}_clr_batch_corrected.csv")
    
    # Perform UMAP on samples
    sample_reducer, sample_umap = perform_umap(clr_df, n_neighbors=15, min_dist=0.01, metric='cosine', random_state=42)
    save_dataframe(sample_umap, f"{args.output_prefix}_sample_umap.csv")

    # Perform HDBSCAN clustering on samples
    sample_clusters = perform_hdbscan(sample_umap)
    save_dataframe(sample_clusters, f"{args.output_prefix}_sample_clusters.csv")

    # Perform UMAP on ASVs (transpose CLR batch-corrected data)
    asv_reducer, asv_umap = perform_umap(clr_df.transpose(), n_neighbors=15, min_dist=0.01, metric='cosine', random_state=42)
    tax_asv_umap = asv_umap.join(taxonomy_df, how='left')
    save_dataframe(tax_asv_umap, f"{args.output_prefix}_asv_umap.csv")

    # Perform HDBSCAN clustering on ASVs
    asv_clusters = perform_hdbscan(asv_umap)
    tax_asv_umap = asv_umap.join(asv_clusters).join(taxonomy_df)
    save_dataframe(tax_asv_umap, f"{args.output_prefix}_asv_clusters.csv")
    
    meta = 'Type_Group'
    sample_plot_path = f"{args.output_prefix}_sample_{meta}_plot.pdf"
    palette = {'Scope Flush': '#0072B2',
           'Skin Brush': '#009E73',
           'Bronchial Brush': '#E69F00',
           'BAL': '#CC79A7',
           'Oral Rinse': '#D55E00'
           }
    plot_sample_clusters(sample_umap, sample_clusters, metadata_df, sample_plot_path, meta, m_color=palette)
    
    flurp


    meta = 'Type_Group'
    sample_plot_path = f"{args.output_prefix}_sample_{meta}_subset_plot.png"
    sub_list = ['Oral Rinse', 'BAL', 'Bronchial Brush']
    plot_sample_clusters(sample_umap, sample_clusters,
                         metadata_df.loc[metadata_df['Type_Group'].isin(sub_list)],
                         sample_plot_path, meta, m_color=False)
    
    meta = 'Set'
    sample_plot_path = f"{args.output_prefix}_sample_{meta}_plot.png"
    plot_sample_clusters(sample_umap, sample_clusters, metadata_df, sample_plot_path, meta, m_color=False)

    meta = 'Case'
    sample_plot_path = f"{args.output_prefix}_sample_{meta}_plot.png"
    plot_sample_clusters(sample_umap, sample_clusters, metadata_df, sample_plot_path, meta, m_color=False)


    ## Generate depth profiles for metadata
    #metrics = ["O2 (uM)", "NOx (uM)", "NO3- (uM)", "NO2- (uM)",  "N2O (uM)", "NH4+ (uM)", "H2S (uM)", "CH4"]
    #m_colors = ["black", "#E7298A", "#1B9E77", "#66A61E", "#0C5196", "#7570B3", "#D95F02", "violet"]
    #plot_depth_profiles(metadata_df, metrics, m_colors, args.output_prefix)
    
    #m_colors = ["#1B9E77", "#E7298A" , "#0C5196", "#D95F02"]
    #genera_list = ['SUP05', 'Thioglobus', 'SUP/Thio_clade']
    #genera_colors = [convert_rgb(x) for x in [(0.14595412205706287, 0.0, 1.0), (0.46, 0.4, 0.8)]] + ["#1B9E77"]

    #plot_asv_depth_profiles(ra_long_df, asv_list, m_colors, taxonomy_df, args.output_prefix)
    #plot_metag_depth_profiles(metag_df, genera_list, genera_colors, taxonomy_df, args.output_prefix)
    #plot_metat_depth_profiles(metat_df, genera_list, genera_colors, taxonomy_df, args.output_prefix)

    #color_map = plot_bubbleplot(taxonomy_df, cnt_long_df, metadata_df, args.output_prefix)
    
    #plot_horizontal_bar_by_phylum(taxonomy_df, cnt_long_df, metadata_df, args.output_prefix, color_map)
    
    #plot_proteo_bubbleplot(taxonomy_df, cnt_long_df, metadata_df, args.output_prefix)
    #plot_asv_bubbleplot(taxonomy_df, cnt_long_df, metadata_df, args.output_prefix)
    
    # Generate and save ASV phylum plot
    asv_plot_path = f"{args.output_prefix}_asv_phylum_plot.png"
    plot_asv_phylum(asv_umap, taxonomy_df, asv_plot_path)

    asv_plot_path = f"{args.output_prefix}_asv_cluster_plot.png"
    plot_asv_clusters(tax_asv_umap, asv_plot_path)

    sample_plot_path = f"{args.output_prefix}_sample"
    metadata_df['Cluster'] = sample_clusters
    save_umap_plots(sample_reducer, metadata_df, sample_plot_path)
    sample_plot_path = f"{args.output_prefix}_asv"
    save_umap_plots(asv_reducer, tax_asv_umap, sample_plot_path)
    
    #plot_cca_biplot(species_scores, env_scores, '/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot_Cluster.png', 'Cluster')
    #plot_cca_biplot(species_scores, env_scores, '/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot_MAG_pair.png', 'MAG_pair')
    #plot_cca_biplot(species_scores, env_scores, '/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot_Phylum.png', 'Phylum', color_map)
    #plot_cca_biplot(species_scores, env_scores, '/home/ryan/Projects/UBC/LMP/SPARK_data/vsearch_output/cca/cca_biplot_Genus.png', 'Genus', color_map)

    print("Clustering analysis and visualization complete.")
    
if __name__ == "__main__":
    main()
