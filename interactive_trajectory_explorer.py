#!/usr/bin/env python3
"""
interactive_trajectory_explorer.py

Create an interactive Plotly explorer for a single trajectory grouping.
Points show the aggregated UMAP coordinates per group/month, with a slider
to advance through the months while lines show the full trajectory per group.

Usage:
  python interactive_trajectory_explorer.py \
    --umap-results ../V4_ncbi_output/compartments/compartment_umap_clusters.tsv \
    --metadata ../V4_ncbi_output/metadata/metadata_updated_micro.tsv \
    --month-col Month \
    --group-col Depth \
    --color-col Color \
    --output-dir ../V4_ncbi_output/trajectory_analysis \
    --output-file trajectory_explorer.html
"""

import argparse
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import plotly.colors as plc
import plotly.graph_objects as go
import plotly.io as pio


def load_data(umap_path: Path, metadata_path: Path, sample_col: str = 'sampleid') -> pd.DataFrame:
    umap_df = pd.read_csv(umap_path, sep='\t')
    meta_df = pd.read_csv(metadata_path, sep='\t')
    if sample_col not in umap_df.columns:
        raise ValueError(f"{sample_col} missing from {umap_path}")
    if sample_col not in meta_df.columns:
        raise ValueError(f"{sample_col} missing from {metadata_path}")
    merged = umap_df.merge(meta_df, on=sample_col, how='inner', suffixes=('_umap', ''))
    if merged.empty:
        raise ValueError("No overlapping samples between UMAP and metadata")
    return merged


def aggregate_coords(
    data: pd.DataFrame,
    month_col: str,
    group_col: str,
) -> pd.DataFrame:
    if month_col not in data.columns:
        raise ValueError(f"{month_col} missing from merged data")
    if group_col not in data.columns:
        raise ValueError(f"{group_col} missing from merged data")

    data['month_numeric'] = pd.to_numeric(data[month_col], errors='coerce')
    if data['month_numeric'].isna().any():
        raise ValueError(f"Month column {month_col} must be numeric")
    data['month_numeric'] = data['month_numeric'].astype(int)

    columns = ['umap1', 'umap2']
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"{col} missing from UMAP results")

    agg = (
        data.groupby([group_col, 'month_numeric'], observed=True)[columns]
        .mean()
        .reset_index()
    )
    agg = agg.sort_values(['month_numeric', group_col])
    return agg


def build_color_map(data: pd.DataFrame, group_col: str, color_col: str = 'Color') -> dict:
    if color_col in data.columns:
        mapping = dict(zip(data[group_col], data[color_col]))
        return mapping
    unique_groups = sorted(data[group_col].unique())
    palette = plc.qualitative.Plotly
    extended = palette * ((len(unique_groups) // len(palette)) + 1)
    return {grp: extended[idx] for idx, grp in enumerate(unique_groups)}


def build_interactive_figure(
    agg_df: pd.DataFrame,
    group_col: str,
    color_map: dict,
    output_path: Path,
    monthly_plot: Optional[Path] = None,
) -> None:
    months = sorted(agg_df['month_numeric'].unique())
    if not months:
        raise ValueError("No month data to plot")

    fig = go.Figure()
    # add lines for groups
    group_order = sorted(agg_df[group_col].dropna().unique())
    line_count = 0
    for group in group_order:
        grp_df = agg_df[agg_df[group_col] == group]
        fig.add_trace(go.Scatter(
            x=grp_df['umap1'],
            y=grp_df['umap2'],
            mode='lines',
            line=dict(color=color_map.get(group, '#888888'), width=2),
            name=f"{group} path",
            hoverinfo='none',
            showlegend=False,
            opacity=0.35,
            line_shape='spline',
        ))
        line_count += 1

    marker_month_map: dict[int, list[int]] = {month: [] for month in months}
    for month in months:
        month_df = agg_df[agg_df['month_numeric'] == month]
        for group in group_order:
            row = month_df[month_df[group_col] == group]
            if row.empty:
                continue
            x_val = float(row['umap1'].iloc[0])
            y_val = float(row['umap2'].iloc[0])
            trace = go.Scatter(
                x=[x_val],
                y=[y_val],
                mode='markers',
                marker=dict(size=12, color=color_map.get(group, '#444444')),
                name=str(group),
                hovertemplate=(
                    f"{group_col}: {group}<br>UMAP1: {{x:.3f}}<br>UMAP2: {{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
                visible=(month == months[0]),
            )
            fig.add_trace(trace)
            marker_month_map[month].append(len(fig.data) - 1)

    slider_steps = []
    total_traces = len(fig.data)
    for month in months:
        visible = [True] * line_count + [False] * (total_traces - line_count)
        for idx in marker_month_map.get(month, []):
            visible[idx] = True
        slider_steps.append(dict(
            method='restyle',
            label=str(month),
            args=[{'visible': visible}],
        ))

    x_min = agg_df['umap1'].min()
    x_max = agg_df['umap1'].max()
    y_min = agg_df['umap2'].min()
    y_max = agg_df['umap2'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    margin_x = x_span * 0.03 if x_span > 0 else 0.1
    margin_y = y_span * 0.03 if y_span > 0 else 0.1

    fig.update_layout(
        title='Interactive Seasonal Trajectories',
        xaxis=dict(
            title='UMAP1',
            range=[x_min - margin_x, x_max + margin_x],
            fixedrange=True,
        ),
        yaxis=dict(
            title='UMAP2',
            range=[y_min - margin_y, y_max + margin_y],
            fixedrange=True,
        ),
        margin=dict(l=80, r=30, t=70, b=60),
        sliders=[dict(
            active=0,
            steps=slider_steps,
            xanchor='left',
            y=-0.1,
            len=0.9,
            pad=dict(t=60),
            currentvalue=dict(prefix='Month: ', visible=True),
        )],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
    monthly_section = ""
    if monthly_plot:
        if not monthly_plot.exists():
            raise FileNotFoundError(f"Monthly plot '{monthly_plot}' does not exist")
        dest = output_path.parent / monthly_plot.name
        if dest.resolve() != monthly_plot.resolve():
            shutil.copy(monthly_plot, dest)
        tag: str
        if monthly_plot.suffix.lower() == '.pdf':
            tag = f'<object data="{monthly_plot.name}" type="application/pdf" width="100%" height="600px"></object>'
        else:
            tag = f'<img src="{monthly_plot.name}" alt="Monthly stratification profile" loading="lazy">'
        monthly_section = f"""
        <div class="monthly-container">
          <h3>Monthly stratification profile</h3>
          {tag}
        </div>
        """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Interactive Seasonal Trajectories</title>
<style>
  body {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #f9f9f9;
    color: #2a2a2a;
  }}
  .content-wrapper {{
    width: 90%;
    margin: 0 auto;
    padding: 1rem 0 3rem;
  }}
  .monthly-container {{
    margin-bottom: 1.5rem;
    text-align: center;
  }}
  .monthly-container h3 {{
    margin-bottom: 0.5rem;
    font-size: 1.25rem;
    letter-spacing: 0.01em;
  }}
  .monthly-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid #dcdcdc;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
  }}
  .plotly-wrapper {{
    margin-bottom: 2rem;
  }}
  .plotly-wrapper .plotly-graph-div {{
    min-height: 500px;
  }}
</style>
</head>
<body>
<div class="content-wrapper">
{monthly_section}
  <div class="plotly-wrapper">
{fig_html}
  </div>
</div>
</body>
</html>"""
    output_path.write_text(html_content)
    print(f"[✓] Saved interactive trajectory explorer: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive trajectory explorer (UMAP)")
    parser.add_argument("--umap-results", type=Path, required=True,
                        help="UMAP results file with columns sampleid, umap1, umap2")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Metadata TSV (must contain month and grouping columns)")
    parser.add_argument("--group-col", required=True,
                        help="Grouping column to show in the explorer (e.g., Depth)")
    parser.add_argument("--month-col", required=True,
                        help="Temporal column (numeric month)")
    parser.add_argument("--color-col", default='Color',
                        help="Metadata column for color mapping (optional)")
    parser.add_argument("--sample-col", default='sampleid',
                        help="Sample identifier column shared between files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where the interactive HTML should be saved")
    parser.add_argument("--output-file", type=str, default="interactive_trajectory.html",
                        help="Output HTML filename")
    parser.add_argument("--monthly-plot", type=Path,
                        help="Optional monthly strat plot (PNG/PDF) to embed above the slider")
    args = parser.parse_args()

    combined = load_data(args.umap_results, args.metadata, sample_col=args.sample_col)
    agg = aggregate_coords(combined, args.month_col, args.group_col)
    color_map = build_color_map(combined, args.group_col, color_col=args.color_col)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_file
    build_interactive_figure(
        agg,
        args.group_col,
        color_map,
        output_path,
        monthly_plot=args.monthly_plot,
    )


if __name__ == "__main__":
    main()
