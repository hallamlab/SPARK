#!/usr/bin/env python3

import argparse
import sys
import plotly.graph_objects as go
import re
import pandas as pd
import os
import matplotlib as mpl
from matplotlib import font_manager as fm, rcParams
import seaborn as sns
import matplotlib.pyplot as plt


# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams.update({'font.size': 12})  # Set your desired size
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback
pd.set_option('display.max_columns', None)
font_path = '/home/ryan/.fonts/MYRIADPRO-REGULAR.OTF'  # update to your path
myriad_font = fm.FontProperties(fname=font_path)
rcParams['font.family'] = myriad_font.get_name()
sns.set_theme()  # re-applies style with updated rcParams


def parse_steps(unknown_args):
    steps = {}
    samples = {}
    colors = {}
    args = unknown_args
    i = 0
    while i < len(args):
        arg = args[i]
        match_step = re.match(r"-(\d+)$", arg)
        match_sample = re.match(r"--samples", arg)
        match_colors = re.match(r"--colors", arg)

        if match_step:
            step_order = int(match_step.group(1))
            i += 1
            if i >= len(args):
                print(f"Expected value after {arg}")
                sys.exit(1)
            step_value = args[i]
            steps[step_order] = step_value
            i += 1
        elif match_sample:
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                sample_data = args[i]
                if ':' not in sample_data:
                    print(f"Invalid sample format: {sample_data}")
                    sys.exit(1)
                sample_name, count_str = sample_data.split(':', 1)
                try:
                    samples[sample_name.strip()] = int(count_str)
                except ValueError:
                    print(f"Invalid count for sample {sample_name}: {count_str}")
                    sys.exit(1)
                i += 1
        elif match_colors:
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                color_data = args[i]
                if ':' not in color_data:
                    print(f"Invalid color format: {color_data}")
                    sys.exit(1)
                sample_name, color_code = color_data.split(':', 1)
                colors[sample_name.strip()] = color_code.strip()
                i += 1
        else:
            i += 1
    return steps, samples, colors

def process_steps(steps_dict):
    steps_list = []
    counts_list = []
    for step_order in sorted(steps_dict.keys()):
        step_value = steps_dict[step_order]
        if ':' not in step_value:
            print(f"Invalid step format for step {step_order}: {step_value}")
            sys.exit(1)
        step_name, count_str = step_value.split(':', 1)
        try:
            count = int(count_str)
        except ValueError:
            print(f"Invalid count for step {step_order}: {count_str}")
            sys.exit(1)
        steps_list.append(step_name.strip())
        counts_list.append(count)
    return steps_list, counts_list

def build_sankey(steps_list, counts_list, samples_dict, output_dict, colors_dict, output_base):
    nodes = []
    links = []
    node_indices = {}
    link_colors = []

    # Add sample nodes
    for sample_name, count in samples_dict.items():
        label = '' #f"{sample_name} ({count})"
        color = colors_dict.get(sample_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[sample_name, 'input'] = len(nodes) - 1

    # Add main process nodes
    for idx, (step_name, count) in enumerate(zip(steps_list, counts_list)):
        label = '' #f"{step_name} ({count})"
        nodes.append({"label": label, "color": "black"})
        node_indices[step_name] = len(nodes) - 1

    # Add sample nodes
    for sample_name, count in output_dict.items():
        label = '' #f"{sample_name} ({count})"
        color = colors_dict.get(sample_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[sample_name, 'output'] = len(nodes) - 1

    # Create links from samples to the first step
    first_step_name = steps_list[0]
    first_step_idx = node_indices[first_step_name]

    for sample_name, count in samples_dict.items():
        sample_idx = node_indices[sample_name, 'input']
        color = colors_dict.get(sample_name, "black")
        link_colors.append(f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": sample_idx,
            "target": first_step_idx,
            "value": count
        })

    # Build links between steps
    for i in range(len(steps_list) - 1):
        source_idx = node_indices[steps_list[i]]
        target_idx = node_indices[steps_list[i + 1]]
        value = counts_list[i + 1]
        links.append({
            "source": source_idx,
            "target": target_idx,
            "value": value
        })
        link_colors.append("grey")

        if counts_list[i] > counts_list[i+1]:
            loss_value = counts_list[i] - counts_list[i+1]
            loss_label = '' #f"Loss after {steps_list[i]} ({loss_value})"
            nodes.append({"label": loss_label, "color": "lightgrey"})
            loss_node_idx = len(nodes) - 1
            links.append({
                "source": source_idx,
                "target": loss_node_idx,
                "value": loss_value
            })
            link_colors.append("lightgrey")

    # Create links from last step to samples
    last_step_name = steps_list[-1]
    last_step_idx = node_indices[last_step_name]

    for sample_name, count in output_dict.items():
        sample_idx = node_indices[sample_name, 'output']
        color = colors_dict.get(sample_name, "black")
        link_colors.append(f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": last_step_idx,
            "target": sample_idx,
            "value": count
        })

    fig = go.Figure(data=[go.Sankey(
        node={
            "pad": 15,
            "thickness": 20,
            "line": {"color": "black", "width": 0.5},
            "label": [node['label'] for node in nodes],
            "color": [node.get('color', 'black') for node in nodes]
        },
        link={
            "source": [link['source'] for link in links],
            "target": [link['target'] for link in links],
            "value": [link['value'] for link in links],
            "color": link_colors
        }
    )])

    fig.update_layout(title_text="Data Loss Flow", font_size=12)
    fig.write_html(f"{output_base}")
    print(f"Sankey diagram saved as {output_base}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate a Sankey diagram with samples and custom colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--output", type=str, default="sankey_diagram.html", help="Output HTML file")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show help message and exit.')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def main():  

    all_type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           'Failed-QC': 'lightgray'
           }

    data_dir = '/home/ryan/Projects/UBC/LMP/SPARK_data/'
       
    output = os.path.join(data_dir, 'vsearch_output/metadata/data_loss_sankey.html')

    metadata_table_path = os.path.join(data_dir, 'ref_db/spark_metadata.tsv')
    metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t', index_col=0)

    fastq_stats_path = os.path.join(data_dir, 'vsearch_output/stats/fastq_stats.tsv')
    fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
    fstats_df['sample'] = [x.split('/')[-1].split('_L001_R')[0] for x in fstats_df['file']]
    raw_reads_df = fstats_df.groupby(['sample'])[['num_seqs', 'sum_len']].sum().reset_index()
    read_meta_df = raw_reads_df.merge(metadata_df, on='sample')

    filter_stats_path = os.path.join(data_dir, 'vsearch_output/stats/filtered_fastqs.tsv')
    filter_stats_df = pd.read_csv(filter_stats_path, header=0, sep='\t')
    filter_stats_df['sample'] = [x.split('/')[-1].split('_L001')[0] for x in filter_stats_df['file']]
    filter_reads_df = filter_stats_df.groupby(['sample'])[['num_seqs', 'sum_len']].sum().reset_index()
    filter_meta_df = filter_reads_df.merge(metadata_df, on='sample')
    
    asv_raw_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_counts.tsv')
    asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
    asv_raw_df.columns = [x.rsplit('_', 1)[0] for x in asv_raw_df.columns]
    asv_raw_stack_df = asv_raw_df.stack().reset_index()
    asv_raw_stack_df.columns = ['ASV_ID', 'sample', 'count']
    asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['count'] > 0]
    asv_raw_stack_df.set_index('ASV_ID', inplace=True)
    asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on='sample', how='left')
    asv_raw_cnt_df = asv_raw_meta_df.groupby(['Type_Group', 'sample'])['count'].sum().reset_index()

    asv_decon_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.decon.tsv')
    asv_decon_df = pd.read_csv(asv_decon_path, header=0, sep='\t', index_col=0)
    asv_decon_df.columns = [x.rsplit('_', 1)[0] for x in asv_decon_df.columns]
    asv_decon_stack_df = asv_decon_df.stack().reset_index()
    asv_decon_stack_df.columns = ['ASV_ID', 'sample', 'count']
    asv_decon_stack_df = asv_decon_stack_df.loc[asv_decon_stack_df['count'] > 0]
    asv_decon_stack_df.set_index('ASV_ID', inplace=True)
    asv_decon_meta_df = asv_decon_stack_df.merge(metadata_df, on='sample', how='left')
    asv_decon_cnt_df = asv_decon_meta_df.groupby(['Type_Group', 'sample'])['count'].sum().reset_index()

    asv_micro_path = os.path.join(data_dir, 'vsearch_output/ASVs/ASV_filtered.micro.tsv')
    asv_micro_df = pd.read_csv(asv_micro_path, header=0, sep='\t', index_col=0)
    asv_micro_df.columns = [x.rsplit('_', 1)[0] for x in asv_micro_df.columns]
    asv_micro_stack_df = asv_micro_df.stack().reset_index()
    asv_micro_stack_df.columns = ['ASV_ID', 'sample', 'count']
    asv_micro_stack_df = asv_micro_stack_df.loc[asv_micro_stack_df['count'] > 0]
    asv_micro_stack_df.set_index('ASV_ID', inplace=True)
    asv_micro_meta_df = asv_micro_stack_df.merge(metadata_df, on='sample', how='left')
    asv_micro_cnt_df = asv_micro_meta_df.groupby(['Type_Group', 'sample'])['count'].sum().reset_index()

    read_grp_df = read_meta_df.groupby(['Type_Group'])['num_seqs'].sum().reset_index()
    read_grp_df['num_reads'] = read_grp_df['num_seqs'] / 2

    filter_grp_df = filter_meta_df.groupby(['Type_Group'])['num_seqs'].sum().reset_index()
    filter_grp_df['num_reads'] = filter_grp_df['num_seqs']

    asv_raw_grp_df = asv_raw_cnt_df.groupby(['Type_Group'])['count'].sum().reset_index()
    asv_raw_grp_df['num_reads'] = asv_raw_grp_df['count']

    asv_decon_grp_df = asv_decon_cnt_df.groupby(['Type_Group'])['count'].sum().reset_index()
    asv_decon_grp_df['num_reads'] = asv_decon_grp_df['count']

    asv_micro_grp_df = asv_micro_cnt_df.groupby(['Type_Group'])['count'].sum().reset_index()
    asv_micro_grp_df['num_reads'] = asv_micro_grp_df['count']

    raw_reads = int(read_grp_df['num_reads'].sum())
    filter_reads = int(filter_grp_df['num_reads'].sum())
    asv_raw_reads = int(asv_raw_grp_df['num_reads'].sum())
    asv_decon_reads = int(asv_decon_grp_df['num_reads'].sum())
    asv_micro_reads = int(asv_micro_grp_df['num_reads'].sum())
    
    steps_list = ['Quality Control', 'Error Correction', 'Decontamination', 'Off-Target Filtering', 'Finished Data']
    counts_list = [raw_reads, filter_reads, asv_raw_reads, asv_decon_reads, asv_micro_reads]

    type_list = ['Skin Brush', 'Scope Flush', 'Oral Rinse', 'BAL', 'Lung Brush']
    input_samples_dict = {x: int(read_grp_df.loc[read_grp_df['Type_Group'] == x]['num_reads']) for x in type_list}
    output_samples_dict = {x: int(asv_micro_grp_df.loc[asv_micro_grp_df['Type_Group'] == x]['num_reads']) for x in type_list}

    print("Parsed steps:")
    for step_name, count in zip(steps_list, counts_list):
        print(f"{step_name}: {count}")
    print("Parsed samples:")
    for sample_name, count in input_samples_dict.items():
        print(f"{sample_name}: {count}")

    print(input_samples_dict)

    build_sankey(steps_list, counts_list, input_samples_dict, output_samples_dict, all_type_palette, output)

if __name__ == "__main__":
    main()
