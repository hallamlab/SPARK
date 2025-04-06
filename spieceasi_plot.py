# cca_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import hdbscan
import umap.plot


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

def parse_taxonomy(taxonomy_df):
    """
    Parses the Taxonomy column to extract taxonomic levels, particularly Phylum.

    Args:
        taxonomy_df (pd.DataFrame): DataFrame containing 'Sequence_ID' and 'Taxonomy' columns.

    Returns:
        pd.DataFrame: Updated taxonomy DataFrame with separate columns for each taxonomic level.
    """
    # Define the taxonomic levels we're interested in
    tax_levels = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # Initialize columns for each taxonomic level
    for level in tax_levels:
        taxonomy_df[level] = 'Unknown'  # Default value

    # Iterate over each taxonomy string to extract taxonomic levels
    for idx, row in taxonomy_df.iterrows():
        taxonomy_str = row['Taxonomy']
        # Split the taxonomy string by ';'
        taxa = taxonomy_str.split(';')
        for taxon in taxa:
            taxon = taxon.strip()  # Remove leading/trailing whitespace
            if taxon:
                # Split by '__' to separate the prefix from the taxon name
                if '__' in taxon:
                    prefix, name = taxon.split('__', 1)
                    prefix = prefix.lower()
                    name = name.strip()
                    # Map prefix to taxonomic level
                    if prefix.startswith('d'):
                        taxonomy_df.at[idx, 'Domain'] = name if name else 'Unknown'
                    elif prefix.startswith('p'):
                        taxonomy_df.at[idx, 'Phylum'] = name if name else 'Unknown'
                    elif prefix.startswith('c'):
                        taxonomy_df.at[idx, 'Class'] = name if name else 'Unknown'
                    elif prefix.startswith('o'):
                        taxonomy_df.at[idx, 'Order'] = name if name else 'Unknown'
                    elif prefix.startswith('f'):
                        taxonomy_df.at[idx, 'Family'] = name if name else 'Unknown'
                    elif prefix.startswith('g'):
                        taxonomy_df.at[idx, 'Genus'] = name if name else 'Unknown'
                    elif prefix.startswith('s'):
                        taxonomy_df.at[idx, 'Species'] = name if name else 'Unknown'
    print("Parsed taxonomy and extracted taxonomic levels.")
    return taxonomy_df

def save_umap_plots(reducer, metadata_df, output_base_path):
    """
    Generates and saves UMAP plots as static images using matplotlib.

    Args:
        reducer (umap.UMAP): Fitted UMAP reducer.
        output_base_path (str): Base file path for saving plots (without extension).
    """
    # Original UMAP scatter plot
    plt.figure(figsize=(12, 10))
    umap.plot.points(reducer, labels=metadata_df['Cluster'])
    plt.title("UMAP Scatter Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_scatter.png", dpi=600)
    plt.close()
    print(f"Saved UMAP scatter plot to {output_base_path}_scatter.png")

    # Connectivity plot
    plt.figure(figsize=(12, 10))
    umap.plot.connectivity(reducer, show_points=False, labels=metadata_df['Cluster'])
    plt.title("UMAP Connectivity Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity.png", dpi=600)
    plt.close()
    print(f"Saved UMAP connectivity plot to {output_base_path}_connectivity.png")



    plt.figure(figsize=(12, 10))
    # Generate the connectivity plot
    ax = umap.plot.connectivity(reducer, edge_bundling='hammer', show_points=False, edge_cmap='gray_r')

    # Get current limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Dynamically calculate new limits with padding
    x_padding = (x_max - x_min) * 0.05  # 5% padding
    y_padding = (y_max - y_min) * 0.05  # 5% padding
    new_x_min = x_min - x_padding
    new_x_max = x_max + x_padding
    new_y_min = y_min - y_padding
    new_y_max = y_max + y_padding

    # Update the axis limits
    ax.set_xlim(new_x_min, new_x_max)
    ax.set_ylim(new_y_min, new_y_max)

    # Add title and save the plot
    plt.title("UMAP Connectivity Plot with Edge Bundling and Points")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_connectivity_bundled.png", dpi=600)
    plt.close()
    print(f"Saved bundled connectivity plot with points to {output_base_path}_connectivity_bundled.png")



    # Diagnostic plot
    plt.figure(figsize=(12, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='pca')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_pca.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_pca.png")
    
    # Diagnostic plot
    plt.figure(figsize=(12, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='vq')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_vq.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_vq.png")

    # Diagnostic plot
    plt.figure(figsize=(12, 10))
    umap.plot.diagnostic(reducer, diagnostic_type='local_dim')
    plt.title("UMAP Diagnostic Plot")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_diagnostic_local_dim.png", dpi=600)
    plt.close()
    print(f"Saved UMAP diagnostic plot to {output_base_path}_diagnostic_local_dim.png")

    # Diagnostic plot
    plt.figure(figsize=(12, 10))
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


# Load the data
spiec_easi_df = pd.read_csv('vsearch_new_output/spieceasi/edge_list_with_asv_ids.csv')
spiec_easi_df.loc[spiec_easi_df['Weight'] < 0.03, 'Weight'] = 0
spiec_easi_df['Distance'] = 1 - spiec_easi_df['Weight']
spiec_easi_df['TransformedWeight'] = (spiec_easi_df['Weight'] - spiec_easi_df['Weight'].min()) / (spiec_easi_df['Weight'].max() - spiec_easi_df['Weight'].min())
spiec_easi_df['TransformedDistance'] = 1 - spiec_easi_df['TransformedWeight']


#piv_df = spiec_easi_df.pivot(index='Taxon1', columns='Taxon2', values='Weight').fillna(0)
# Extract all unique taxa from both columns
all_taxa = sorted(set(spiec_easi_df["Taxon1"]).union(set(spiec_easi_df["Taxon2"])))

# Create a square matrix with taxa as indices and columns, filled with zeros
distance_matrix = pd.DataFrame(1, index=all_taxa, columns=all_taxa, dtype=float)

# Populate the matrix with the provided weights
for _, row in spiec_easi_df.iterrows():
    if row["Taxon1"] == row["Taxon2"]:
        distance_matrix.loc[row["Taxon1"], row["Taxon2"]] = 0
        distance_matrix.loc[row["Taxon2"], row["Taxon1"]] = 0
    else:
        distance_matrix.loc[row["Taxon1"], row["Taxon2"]] = row["TransformedDistance"]
        distance_matrix.loc[row["Taxon2"], row["Taxon1"]] = row["TransformedDistance"]  # Ensure symmetry
np.fill_diagonal(distance_matrix.values, 0)

# Display the resulting distance matrix
print(distance_matrix.head())

sample_reducer, sample_umap = perform_umap(distance_matrix, n_neighbors=5, min_dist=0.0, metric='precomputed', random_state=42)

taxonomy_df = load_taxonomy("vsearch_new_output/taxonomy/ASV_GG2_tax.tsv")

plot_df = sample_umap.join(taxonomy_df, how='left')

# Handle ASVs with missing phylum information
plot_df['Phylum'] = plot_df['Phylum'].fillna('Unknown')

# Perform HDBSCAN clustering on ASVs
asv_clusters = perform_hdbscan(sample_umap, min_cluster_size=5)
tax_asv_umap = plot_df.join(asv_clusters)

# Initialize the matplotlib figure
plt.figure(figsize=(12, 10))

# Define a color palette. Limit to top 20 phyla and group others as 'Other' to avoid too many colors
top_phyla = tax_asv_umap['Phylum'].value_counts().nlargest(20).index
tax_asv_umap['Phylum_Categorized'] = tax_asv_umap['Phylum'].apply(lambda x: x if x in top_phyla else 'Other')
num_phyla = tax_asv_umap['Phylum_Categorized'].nunique()
palette = sns.color_palette("muted", num_phyla)
palette = {x[0]: x[1] for x in zip(tax_asv_umap['Phylum_Categorized'].unique(), palette)}
palette['Other'] = (0.9, 0.9, 0.9)

# Create scatter plot
sns.scatterplot(
    data=tax_asv_umap,
    x="UMAP1",
    y="UMAP2",
    hue="Phylum_Categorized",
    palette=palette,
    legend="full",
    alpha=0.6,
    edgecolor="gray",
    linewidth=0.5
)

plt.title("UMAP Projection of ASVs SPIEC-EASI Colored by Phylum")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="Phylum", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()

# Save the plot
plt.savefig("vsearch_new_output/spieceasi/spieceasi_umap_phylum_plot.png", dpi=600)
plt.close()


# Initialize the matplotlib figure
plt.figure(figsize=(12, 10))

# Define a color palette. Limit to top 20 phyla and group others as 'Other' to avoid too many colors
top_phyla = tax_asv_umap['Cluster'].value_counts().nlargest(20).index
tax_asv_umap['Cluster_Categorized'] = tax_asv_umap['Cluster'].apply(lambda x: x if x in top_phyla else 'Other')
num_phyla = tax_asv_umap['Cluster_Categorized'].nunique()
palette = sns.color_palette("muted", num_phyla)
palette = {x[0]: x[1] for x in zip(tax_asv_umap['Cluster_Categorized'].unique(), palette)}
palette[-1] = (0.9, 0.9, 0.9)

# Create scatter plot
sns.scatterplot(
    data=tax_asv_umap,
    x="UMAP1",
    y="UMAP2",
    hue="Cluster_Categorized",
    palette=palette,
    legend="full",
    alpha=0.6,
    edgecolor="gray",
    linewidth=0.5
)

plt.title("UMAP Projection of ASVs SPIEC-EASI Colored by Cluster")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="Phylum", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()

# Save the plot
plt.savefig("vsearch_new_output/spieceasi/spieceasi_umap_cluster_plot.png", dpi=600)
plt.close()


save_dataframe(tax_asv_umap, f"vsearch_new_output/spieceasi/spieceasi_asv_transformed_distance.csv")

sample_plot_path = f"vsearch_new_output/spieceasi/spieceasi_asv"
save_umap_plots(sample_reducer, tax_asv_umap, sample_plot_path)


