import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

def plot_volcano(df, ind, cmap, p_thresh=0.05, stat_thresh=0, output_file='volcano_plot.pdf'):
    # Load data
    
    # Compute log-transformed p-values
    df['log_p'] = -np.log10(df['p.value'])
    
    # Define colors based on thresholds
    df['significance'] = False  # Default color for non-significant
    df.loc[((df['p.value'] < p_thresh) & (df['stat'] > stat_thresh)), 'significance'] = True 
    
    df['color'] = [cmap[ind[i]] if s else 'white' for i,s in zip(df['index'], df['significance'])]
    # Create plot
    plt.figure(figsize=(8, 6))
    # Plot non-red points first
    non_sig = df[df['significance'] == False]
    plt.scatter(non_sig['stat'], non_sig['log_p'], c=non_sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
    
    # Then plot red points on top
    sig = df[df['significance'] == True]
    plt.scatter(sig['stat'], sig['log_p'], c=sig['color'], alpha=0.75, edgecolors='gray', linewidths=0.25)
     
    
    '''
    # Add legend
    import matplotlib.patches as mpatches
    cancer_patch = mpatches.Patch(color='red', label='Feature', alpha=0.75)
    control_patch = mpatches.Patch(color='lightgray', label='Other', alpha=0.75)
    na_patch = mpatches.Patch(color='white', label='N/A')
    plt.legend(handles=[cancer_patch, control_patch, na_patch], loc='best', title='Significance')
    '''

    # Add reference lines
    plt.axhline(-np.log10(p_thresh), linestyle='--', color='gray', linewidth=1, label=f'p={p_thresh}')
    #plt.axvline(stat_thresh, linestyle='--', color='gray', linewidth=1, label=f'stat={stat_thresh}')

    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('-log10(p-value)')
    plt.title(f"Indicator Species Analysis (pval <= {p_thresh})")

    # Save and show
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

def plot_horizontal_bar(input_file, p_thresh=0.05, stat_thresh=0.3, output_file='bar_plot.png'):
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # Filter significant ASVs for only one status
    sig_df = df[(df['p.value'] < p_thresh) & (abs(df['stat']) > stat_thresh)]
    
    # Assign colors and make control values negative
    sig_df['color'] = sig_df.apply(lambda x: 'red' if x['index'] == 1 else 'blue', axis=1)
    sig_df.loc[sig_df['index'] == 2, 'stat'] *= -1
    
    # Sort by stat value
    sig_df = sig_df.sort_values(by='stat', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=(8, 6))
    plt.barh(sig_df['OTU_ID'], sig_df['stat'], color=sig_df['color'])
    
    # Labels and title
    plt.xlabel('Effect Size (stat)')
    plt.ylabel('OTU ID')
    plt.title('Significant ASVs by Effect Size')
    
    # Save and show
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


df = pd.read_csv('vsearch_output/indicspecies/Case_indicator_species_results.tsv', sep='\t')
index_dict = {1: 'Cancer', 2: 'Control'}
cmap = {'Cancer': 'Grey', 'Control': 'white'}

plot_volcano(df, index_dict, cmap, output_file=f"vsearch_output/indicspecies/Case_Cancer_ISA_plot.pdf")

df = pd.read_csv('vsearch_output/indicspecies/Type_Group_indicator_species_results.tsv', sep='\t')

print(df.head())
index_dict = {1: 'BAL', 2: 'Lung Brush', 3: 'Oral Rinse'}
cmap = {'Lung Brush': '#E69F00', 'BAL': '#CC79A7', 'Oral Rinse': '#D55E00'}
sub_df = df.loc[df['index'].isin(index_dict.keys())]
plot_volcano(sub_df, index_dict, cmap, output_file=f"vsearch_output/indicspecies/Type_Group_ISA_plot.pdf")

