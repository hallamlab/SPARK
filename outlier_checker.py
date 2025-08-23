import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import hdbscan
from hdbscan import approximate_predict
import os
from skbio.stats.composition import clr, multiplicative_replacement

def ensemble_outlier_detection(asv_table, sample_metadata, group_col):
    results = []
    if not group_col:
        sample_metadata['no_group'] = True
        group_col = 'no_group'
    for group in sample_metadata[group_col].unique():
        # Subset samples for this group
        group_samples = list(sample_metadata[sample_metadata[group_col] == group].index)
        spark_samples = list(sample_metadata[((sample_metadata[group_col] == group) &
                                              (sample_metadata['kit'] == 'SPARK-ZYMO'))
                                              ].index)
        methods_samples = list(sample_metadata[((sample_metadata[group_col] == group) &
                                                (sample_metadata['kit'] != 'SPARK-ZYMO'))
                                                ].index)
        train_samples = asv_table.loc[group_samples]
        test_samples = asv_table.loc[group_samples]
        #train_samples = asv_table.loc[spark_samples]
        #test_samples = asv_table.loc[methods_samples]
        
        X_train = train_samples.values
        X_test = test_samples.values

        # Isolation Forest
        iso = IsolationForest(contamination='auto', random_state=42)
        iso_out = iso.fit(X_train)
        iso_out = iso_out.predict(X_test)

        # One-Class SVM
        svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        svm_out = svm.fit(X_train)
        svm_out = svm_out.predict(X_test)

        # HDBSCAN (fit on train, predict on test)
        hdb = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
        hdb.fit(X_train)
        hdb_out, strengths = approximate_predict(hdb, X_test)
        hdb_out = np.where(hdb_out == -1, -1, 1)  # convert to -1 for outliers

        # Combine results
        df = pd.DataFrame({
            'sample': test_samples.index,  # assuming it's a DataFrame
            'group': group,
            'IsolationForest': iso_out,
            'OneClassSVM': svm_out,
            'HDBSCAN': hdb_out
        }).set_index('sample')

        # Consensus voting
        df['outlier_votes'] = (df == -1).sum(axis=1)
        df['is_outlier'] = df['outlier_votes'] == 3

        results.append(df)

    return pd.concat(results)

# Create output directory if it doesn't exist
data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'
output_dir = os.path.join(data_dir, "spark_combined_output/metadata")
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

metadata_table_path = os.path.join(data_dir, 'spark_combined_output/metadata/metadata_updated.tsv')
metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')
metadata_df.set_index('sample', inplace=True)
metadata_df['status'] = ['Non-Cancer' if x == 'Control' else x for x in metadata_df['Case']]

asv_path = os.path.join(data_dir, 'spark_combined_output/ASVs/ASV_final.micro.tsv')
asv_df = pd.read_csv(asv_path, header=0, sep='\t', index_col=0).T

# Subset and align both tables
shared_samples = asv_df.index.intersection(metadata_df.index)
asv_table = asv_df.loc[shared_samples]
asv_table = asv_table.loc[:, (asv_table != 0).any(axis=0)]
asv_table_nonzero = asv_table[(asv_table != 0).any(axis=1)]
shared_samples = asv_table_nonzero.index.intersection(metadata_df.index)

# Add pseudocounts via multiplicative replacement
asv_array = multiplicative_replacement(asv_table_nonzero.values)
# Apply centered log-ratio
clr_transformed = clr(asv_array)
# Optional: convert back to DataFrame
clr_df = pd.DataFrame(clr_transformed, index=asv_table_nonzero.index, columns=asv_table_nonzero.columns)

sample_metadata = metadata_df.loc[shared_samples]

outliers_df = ensemble_outlier_detection(clr_df, sample_metadata, group_col=None).reset_index()
outliers_df.to_csv(os.path.join(data_dir, 'spark_combined_output/metadata/outliers_table.tsv'), sep='\t', index=False)

outliers_df = ensemble_outlier_detection(clr_df, sample_metadata, group_col='type_group').reset_index()
outliers_df.to_csv(os.path.join(data_dir, 'spark_combined_output/metadata/outliers_type_group.tsv'), sep='\t', index=False)
