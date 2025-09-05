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
import matplotlib.colors as mcolors


# Global settings — at the top of script or notebook cell
mpl.rcParams['pdf.fonttype'] = 42   # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
plt.rcParams.update({'font.size': 12})  # Set your desired size
mpl.rcParams['savefig.dpi'] = 600   # Optional — affects raster fallback
pd.set_option('display.max_columns', None)
# Set font globally
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()  # re-applies style with updated rcParams
sns.set_style("white")

def parse_steps(unknown_args):
    steps = {}
    lmp_ids = {}
    colors = {}
    args = unknown_args
    i = 0
    while i < len(args):
        arg = args[i]
        match_step = re.match(r"-(\d+)$", arg)
        match_lmp_id = re.match(r"--lmp_ids", arg)
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
        elif match_lmp_id:
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                lmp_id_data = args[i]
                if ':' not in lmp_id_data:
                    print(f"Invalid lmp_id format: {lmp_id_data}")
                    sys.exit(1)
                lmp_id_name, count_str = lmp_id_data.split(':', 1)
                try:
                    lmp_ids[lmp_id_name.strip()] = int(count_str)
                except ValueError:
                    print(f"Invalid count for lmp_id {lmp_id_name}: {count_str}")
                    sys.exit(1)
                i += 1
        elif match_colors:
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                color_data = args[i]
                if ':' not in color_data:
                    print(f"Invalid color format: {color_data}")
                    sys.exit(1)
                lmp_id_name, color_code = color_data.split(':', 1)
                colors[lmp_id_name.strip()] = color_code.strip()
                i += 1
        else:
            i += 1
    return steps, lmp_ids, colors

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

def build_sankey(steps_list, counts_list, lmp_ids_dict, output_dict, colors_dict, output_base):
    nodes = []
    links = []
    node_indices = {}
    link_colors = []

    # Add lmp_id nodes
    for lmp_id_name, count in lmp_ids_dict.items():
        label = f"{lmp_id_name} ({count})"
        color = colors_dict.get(lmp_id_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[lmp_id_name, 'input'] = len(nodes) - 1

    # Add main process nodes
    for idx, (step_name, count) in enumerate(zip(steps_list, counts_list)):
        label = f"{step_name} ({count})"
        nodes.append({"label": label, "color": "black"})
        node_indices[step_name] = len(nodes) - 1

    # Add lmp_id nodes
    for lmp_id_name, count in output_dict.items():
        label = f"{lmp_id_name} ({count})"
        color = colors_dict.get(lmp_id_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[lmp_id_name, 'output'] = len(nodes) - 1

    # Create links from lmp_ids to the first step
    first_step_name = steps_list[0]
    first_step_idx = node_indices[first_step_name]

    for lmp_id_name, count in lmp_ids_dict.items():
        lmp_id_idx = node_indices[lmp_id_name, 'input']
        color = colors_dict.get(lmp_id_name, "black")
        link_colors.append("grey") #f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": lmp_id_idx,
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
            loss_label = f"Loss after {steps_list[i]} ({loss_value})"
            nodes.append({"label": loss_label, "color": "lightgrey"})
            loss_node_idx = len(nodes) - 1
            links.append({
                "source": source_idx,
                "target": loss_node_idx,
                "value": loss_value
            })
            link_colors.append("lightgrey")

    # Create links from last step to lmp_ids
    last_step_name = steps_list[-1]
    last_step_idx = node_indices[last_step_name]

    for lmp_id_name, count in output_dict.items():
        lmp_id_idx = node_indices[lmp_id_name, 'output']
        color = colors_dict.get(lmp_id_name, "black")
        link_colors.append("grey") #f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": last_step_idx,
            "target": lmp_id_idx,
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
        description="Generate a Sankey diagram with lmp_ids and custom colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--output", type=str, default="sankey_diagram.html", help="Output HTML file")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show help message and exit.')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def build_sankey_nolabels(steps_list, counts_list, lmp_ids_dict, output_dict, colors_dict, output_base):
    nodes = []
    links = []
    node_indices = {}
    link_colors = []

    # Add lmp_id nodes
    for lmp_id_name, count in lmp_ids_dict.items():
        label = '' #f"{lmp_id_name} ({count})"
        color = colors_dict.get(lmp_id_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[lmp_id_name, 'input'] = len(nodes) - 1

    # Add main process nodes
    for idx, (step_name, count) in enumerate(zip(steps_list, counts_list)):
        label = '' #f"{step_name} ({count})"
        nodes.append({"label": label, "color": "black"})
        node_indices[step_name] = len(nodes) - 1

    # Add lmp_id nodes
    for lmp_id_name, count in output_dict.items():
        label = '' #f"{lmp_id_name} ({count})"
        color = colors_dict.get(lmp_id_name, "black")
        nodes.append({"label": label, "color": color})
        node_indices[lmp_id_name, 'output'] = len(nodes) - 1

    # Create links from lmp_ids to the first step
    first_step_name = steps_list[0]
    first_step_idx = node_indices[first_step_name]

    for lmp_id_name, count in lmp_ids_dict.items():
        lmp_id_idx = node_indices[lmp_id_name, 'input']
        color = colors_dict.get(lmp_id_name, "black")
        link_colors.append("grey") #f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": lmp_id_idx,
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

    # Create links from last step to lmp_ids
    last_step_name = steps_list[-1]
    last_step_idx = node_indices[last_step_name]

    for lmp_id_name, count in output_dict.items():
        lmp_id_idx = node_indices[lmp_id_name, 'output']
        color = colors_dict.get(lmp_id_name, "black")
        link_colors.append("grey") #f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)")
        links.append({
            "source": last_step_idx,
            "target": lmp_id_idx,
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
        description="Generate a Sankey diagram with lmp_ids and custom colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    parser.add_argument("--output", type=str, default="sankey_diagram.html", help="Output HTML file")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show help message and exit.')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def main():  
    
    ### MAGIC VALUES ###    
    data_dir = '/home/ryan/SeqData/SeqData/UBC/LMP_priority1/'
    sub_dir = "spark_methods_output_tester"
    samp_col = "lmp_id"
    ###  END  MAGIC  ###
    
    metadata_table_path = os.path.join(data_dir, 'ref_db/methods_metadata.tsv')
    metadata_df = pd.read_csv(metadata_table_path, header=0, sep='\t')

    keep_types = [
                'Oral Rinse',
                'Lung Brush',
                'BAL',
                'Skin Brush',
                'Scope Flush'
                ]
    #keep_types = [
    #              'HostZERO-DEP',
    #              'HostZERO-NODEP',
    #              'SPARK-ZYMO'
    #              ]

    kit_pallete = {'HostZERO-DEP': 'black',
               'HostZERO-NODEP': 'gray',
               'SPARK-ZYMO': 'skyblue',
               }

    type_palette = {'Scope Flush': '#E69F00',
           'Skin Brush': '#CC79A7',
           'Lung Brush': '#009E73',
           'BAL': '#0072B2',
           'Oral Rinse': '#6A3D9A',
           'Failed-QC': 'lightgray'
           }

    metadata_df = metadata_df.loc[metadata_df['type_group'].isin(keep_types)]
   
    fastq_stats_path = os.path.join(data_dir, f"{sub_dir}/stats/fastq_stats.tsv")
    fstats_df = pd.read_csv(fastq_stats_path, header=0, sep='\t')
    fstats_df[samp_col] = [str(x.split('/')[-1].split('_', 1)[0]) for x in fstats_df['file']]
    raw_reads_df = fstats_df.groupby([samp_col])['num_seqs'].sum().reset_index()
    read_meta_df = raw_reads_df.merge(metadata_df, on=samp_col)
    
    filter_stats_path = os.path.join(data_dir, f"{sub_dir}/stats/filtered_fastqs.tsv")
    filter_stats_df = pd.read_csv(filter_stats_path, header=0, sep='\t')
    filter_stats_df[samp_col] = [str(x.split('/')[-1].split('.', 1)[0]) for x in filter_stats_df['file']]
    filter_reads_df = filter_stats_df.groupby([samp_col])['num_seqs'].sum().reset_index()
    filter_meta_df = filter_reads_df.merge(metadata_df, on=samp_col)

    asv_raw_path = os.path.join(data_dir, f"{sub_dir}/ASVs/ASV_counts.tsv")
    asv_raw_df = pd.read_csv(asv_raw_path, header=0, sep='\t', index_col=0)
    asv_raw_stack_df = asv_raw_df.stack().reset_index()
    asv_raw_stack_df.columns = ['ASV_ID', samp_col, 'count']
    asv_raw_stack_df[samp_col] = [str(x.split('/')[-1].split('_', 1)[0]) for x in asv_raw_stack_df[samp_col]]
    asv_raw_stack_df = asv_raw_stack_df.loc[asv_raw_stack_df['count'] > 0]
    asv_raw_stack_df.set_index('ASV_ID', inplace=True)
    asv_raw_meta_df = asv_raw_stack_df.merge(metadata_df, on=samp_col)
    asv_raw_cnt_df = asv_raw_meta_df.groupby(['type_group', samp_col])['count'].sum().reset_index()

    asv_decon_path = os.path.join(data_dir, f"{sub_dir}/ASVs/ASV_target.decon.tsv")
    asv_decon_df = pd.read_csv(asv_decon_path, header=0, sep='\t', index_col=0)
    asv_decon_stack_df = asv_decon_df.stack().reset_index()
    asv_decon_stack_df.columns = ['ASV_ID', samp_col, 'count']
    asv_decon_stack_df[samp_col] = [str(x.split('/')[-1].split('_', 1)[0]) for x in asv_decon_stack_df[samp_col]]
    asv_decon_stack_df = asv_decon_stack_df.loc[asv_decon_stack_df['count'] > 0]
    asv_decon_stack_df.set_index('ASV_ID', inplace=True)
    asv_decon_meta_df = asv_decon_stack_df.merge(metadata_df, on=samp_col)
    asv_decon_cnt_df = asv_decon_meta_df.groupby(['type_group', samp_col])['count'].sum().reset_index()

    asv_micro_path = os.path.join(data_dir, f"{sub_dir}/ASVs/ASV_target.micro.tsv")
    asv_micro_df = pd.read_csv(asv_micro_path, header=0, sep='\t', index_col=0)
    asv_micro_stack_df = asv_micro_df.stack().reset_index()
    asv_micro_stack_df.columns = ['ASV_ID', samp_col, 'count']
    asv_micro_stack_df[samp_col] = [str(x.split('/')[-1].split('_', 1)[0]) for x in asv_micro_stack_df[samp_col]]
    asv_micro_stack_df = asv_micro_stack_df.loc[asv_micro_stack_df['count'] > 0]
    asv_micro_stack_df.set_index('ASV_ID', inplace=True)
    asv_micro_meta_df = asv_micro_stack_df.merge(metadata_df, on=samp_col)
    asv_micro_cnt_df = asv_micro_meta_df.groupby(['type_group', samp_col])['count'].sum().reset_index()

    read_grp_df = read_meta_df.groupby(['type_group'])['num_seqs'].sum().reset_index()
    read_grp_df['num_reads'] = read_grp_df['num_seqs'] / 2
    
    filter_grp_df = filter_meta_df.groupby(['type_group'])['num_seqs'].sum().reset_index()
    filter_grp_df['num_reads'] = filter_grp_df['num_seqs']

    asv_raw_grp_df = asv_raw_cnt_df.groupby(['type_group'])['count'].sum().reset_index()
    asv_raw_grp_df['num_reads'] = asv_raw_grp_df['count']

    asv_decon_grp_df = asv_decon_cnt_df.groupby(['type_group'])['count'].sum().reset_index()
    asv_decon_grp_df['num_reads'] = asv_decon_grp_df['count']

    asv_micro_grp_df = asv_micro_cnt_df.groupby(['type_group'])['count'].sum().reset_index()
    asv_micro_grp_df['num_reads'] = asv_micro_grp_df['count']

    raw_reads = int(read_grp_df['num_reads'].sum())
    filter_reads = int(filter_grp_df['num_reads'].sum())
    asv_raw_reads = int(asv_raw_grp_df['num_reads'].sum())
    asv_decon_reads = int(asv_decon_grp_df['num_reads'].sum())
    asv_micro_reads = int(asv_micro_grp_df['num_reads'].sum())
    
    metadata_df = metadata_df.loc[metadata_df[samp_col].isin(list(asv_raw_cnt_df[samp_col]))]

    steps_list = ['Quality Control', 'Error Correction', 'Decontamination',
                  'Off-Target Filtering', 'Finished Data'
                  ]
    counts_list = [raw_reads, filter_reads, asv_raw_reads, asv_decon_reads, asv_micro_reads]
    seqtype_list = keep_types

    input_lmp_ids_dict = {x: int(read_grp_df.loc[read_grp_df['type_group'] == x]['num_reads'].values) for x in seqtype_list}
    output_lmp_ids_dict = {x: int(asv_micro_grp_df.loc[asv_micro_grp_df['type_group'] == x]['num_reads'].values)
                           if x in list(asv_micro_grp_df['type_group'])
                           else 0 for x in seqtype_list
                           }

    print("Parsed steps:")
    for step_name, count in zip(steps_list, counts_list):
        print(f"{step_name}: {count}")
    print("Parsed sample IDs:")
    for lmp_id_name, count in input_lmp_ids_dict.items():
        print(f"{lmp_id_name}: {count}")

    output = os.path.join(data_dir, f"{sub_dir}/metadata/data_loss_sankey_label_TYPE.html")
    build_sankey(steps_list, counts_list, input_lmp_ids_dict, output_lmp_ids_dict, type_palette, output)
    output = os.path.join(data_dir, f"{sub_dir}/metadata/data_loss_sankey_TYPE.html")
    build_sankey_nolabels(steps_list, counts_list, input_lmp_ids_dict, output_lmp_ids_dict, type_palette, output)

if __name__ == "__main__":
    main()
