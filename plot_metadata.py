#!/usr/bin/env python3
"""
asv_summary_and_plots.py
Project-independent CLI to build ASV master tables and plots (microbial & mitochondrial),
with configurable paths, palettes, and plot toggles.

Quickstart (mirrors your current layout):
  python asv_summary_and_plots.py \
    --data-dir /home/ryan/SeqData/SeqData/UBC/LMP_priority1 \
    --sub-dir spark_combined_output \
    --metadata ref_db/spark_metadata.tsv \
    --keep-types "Skin Brush,Scope Flush,Oral Rinse,BAL,Lung Brush" \
    --fastq-stats stats/fastq_stats.tsv \
    --asv-micro ASVs/ASV_target.micro.tsv --asv-mito mito/ASVs/ASV_target.mito.tsv \
    --taxonomy taxonomy/ASV_SILVA_tax.full-length.vsearch.tsv \
    --make-micro --make-mito

Notes:
- Outputs go to <data-dir>/<sub-dir>/{metadata,mito/metadata} and <data-dir>/<sub-dir>/{ASVs,mito/ASVs}.
- Sample IDs are assumed to match those defined in the manifest/metadata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import warnings
from itertools import combinations, combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Patch

# ---------- Global aesthetics ----------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")

SAMPLE_ID_COL = 'sampleid'


# ========= Utilities =========
def parse_kv_csv(s: str, cast: Optional[type] = None) -> Dict[str, object]:
    """Parse 'A:1,B:2' -> dict; tolerate whitespace."""
    out: Dict[str, object] = {}
    if not s:
        return out
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Expected key:value, got '{item}'")
        k, v = item.split(':', 1)
        k = k.strip()
        v = v.strip()
        out[k] = cast(v) if cast else v
    return out


def parse_list_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(',') if x.strip()] if s else []


def normalize_taxon_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text.lower()


def parse_rank_filters(items: Sequence[str]) -> Dict[str, set[str]]:
    """Parse LEVEL:Name entries and normalize taxonomy level casing."""
    valid_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    lookup = {lvl.lower(): lvl for lvl in valid_levels}
    filters: Dict[str, set[str]] = {}
    for raw in items or []:
        if not raw:
            continue
        if ':' not in raw:
            raise ValueError(f"--include-rank expects LEVEL:Name, got '{raw}'")
        level, name = raw.split(':', 1)
        level_key = lookup.get(level.strip().lower())
        if not level_key:
            raise ValueError(f"Unknown taxonomy level '{level}'. Expected one of: {', '.join(valid_levels)}")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError(f"Missing taxon name in include-rank entry '{raw}'")
        filters.setdefault(level_key, set()).add(clean_name.lower())
    return filters


def extract_sample_id_from_path(path_str: str) -> str:
    """
    Remove common sequencing extensions/suffixes to recover the sample ID.
    """
    base = os.path.basename(path_str)
    stem = base
    for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq',
                '.fasta.gz', '.fasta', '.fa.gz', '.fa',
                '.tsv', '.csv', '.txt', '.gz'):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    stem = re.sub(r'(\.filtered|\.merged|\.trimmed)$', '', stem)
    stem = re.sub(r'(_R[12]|_[12])?(_001)?$', '', stem)
    return stem


def split_taxa_string(taxa_str: str, delimiter=';') -> Dict[str, Optional[str]]:
    levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    if taxa_str != 'Unassigned':
        parts = [part.strip().split('__', 1)[1] if '__' in part else part.strip()
                 for part in taxa_str.split(delimiter)]
    else:
        parts = ['Unassigned']
    return {lvl: (parts[i] if i < len(parts) else None) for i, lvl in enumerate(levels)}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep='\t', index=False)


def save_mat(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep='\t', index=True)


# ========= Core helpers =========
def read_metadata(meta_path: Path, meta_sample_col: str, keep_types: Optional[Sequence[str]]) -> pd.DataFrame:
    df = pd.read_csv(meta_path, sep='\t', header=0)
    # Derive status if present
    if 'Case' in df.columns:
        df['status'] = np.where(df['Case'] == 'Control', 'Non-Cancer', df['Case'])
    if 'Participant_ID' in df.columns:
        patient_set = sorted(df['Participant_ID'].astype(str).unique())
        pid_map = {p: i for i, p in enumerate(patient_set)}
        df['patient_int'] = df['Participant_ID'].astype(str).map(pid_map)
        df['patient_code'] = df['patient_int'].apply(lambda i: f'P{i}')
    if 'type_group' in df.columns:
        df['type_code'] = df['type_group'].astype(str).str[:2]
    if 'Type' in df.columns:
        df['lung_code'] = df['Type'].astype(str).str[0].where(lambda s: s.isin(['R', 'L']), other='N')
    if keep_types is not None and 'type_group' in df.columns:
        df = df[df['type_group'].isin(keep_types)].copy()
    # Create sample_code if meta_sample_col exists
    if meta_sample_col in df.columns:
        df = df.drop_duplicates(subset=[meta_sample_col])
        df = df.copy()
        df['sample_code'] = [f"S{i+1:03d}" for i in range(len(df))]
        # Move sample_code first
        cols = ['sample_code'] + [c for c in df.columns if c != 'sample_code']
        df = df[cols]
    # Ensure meta_sample_col exists
    if meta_sample_col not in df.columns:
        raise ValueError(f"Metadata column '{meta_sample_col}' not found; expected column matching manifest sample IDs.")
    return df


def load_sample_manifest(path: Path) -> Dict[str, str]:
    """
    Build a lookup from FASTQ file path (or basename) to sample ID.
    Manifest columns: sample_id, fastq_r1, fastq_r2 (no header).
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['sample_id', 'r1', 'r2'])
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        sample_id = str(row['sample_id']).strip()
        if not sample_id:
            continue
        for col in ('r1', 'r2'):
            fastq_path = str(row[col]).strip()
            if not fastq_path or fastq_path.lower() == 'nan':
                continue
            candidates = {
                fastq_path,
                os.path.basename(fastq_path),
            }
            try:
                candidates.add(str(Path(fastq_path).resolve()))
            except Exception:
                pass
            for cand in candidates:
                if cand in mapping and mapping[cand] != sample_id:
                    raise ValueError(
                        f"FASTQ '{cand}' maps to multiple sample IDs ({mapping[cand]} vs {sample_id})"
                    )
                mapping[cand] = sample_id
    if not mapping:
        raise ValueError(f"No FASTQ entries were parsed from manifest: {path}")
    return mapping


def read_fastq_stats(path: Path, samp_col: str,
                     manifest_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Expects columns: file, num_seqs
    Collapses replicates by sample ID via groupby+sum.
    """
    df = pd.read_csv(path, sep='\t', header=0)
    if 'file' not in df or 'num_seqs' not in df:
        raise ValueError(f"{path} must contain columns: file, num_seqs")
    stats_dir = path.parent

    def lookup_sample(file_path: str) -> str:
        if manifest_map:
            candidates = [
                file_path,
                os.path.basename(file_path),
                extract_sample_id_from_path(file_path),
            ]
            rel_path = (stats_dir / file_path)
            candidates.append(str(rel_path))
            candidates.append(os.path.basename(rel_path))
            try:
                candidates.append(str(rel_path.resolve()))
            except Exception:
                pass
            for cand in candidates:
                if cand in manifest_map:
                    return manifest_map[cand]
            print(candidates)
            raise ValueError(f"File '{file_path}' not found in manifest")
        return extract_sample_id_from_path(file_path)

    df[samp_col] = df['file'].apply(lookup_sample)
    out = df.groupby(samp_col, as_index=False)['num_seqs'].sum()
    return out


def read_taxonomy_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=0)
    if 'Feature ID' not in df or 'Taxon' not in df:
        raise ValueError(f"{path} must have columns 'Feature ID' and 'Taxon'")
    df['Feature ID'] = df['Feature ID'].astype(str).str.partition(';')[0]
    return df.set_index('Feature ID')


def read_asv_wide_to_long(path: Path, meta_sample_col: str,
                          tax_index: Optional[pd.Index] = None) -> pd.DataFrame:
    wide = pd.read_csv(path, sep='\t', header=0, index_col=0)
    if tax_index is not None:
        wide = wide.loc[[a for a in wide.index if a in set(tax_index)]]
    wide.columns = [c.strip() for c in wide.columns]
    long = wide.stack().reset_index()
    long.columns = ['ASV_ID', meta_sample_col, 'count']
    long = long[long['count'] > 0].copy()
    long.set_index('ASV_ID', inplace=True)
    return long


def add_taxonomy(long_asv: pd.DataFrame, tax_df: pd.DataFrame) -> pd.DataFrame:
    merged = long_asv.merge(tax_df, left_index=True, right_index=True, how='left')
    # Expand taxonomy levels
    tax_map = {lvl: [] for lvl in ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]}
    for t in merged['Taxon'].fillna('Unassigned'):
        parts = split_taxa_string(t)
        for lvl in tax_map:
            tax_map[lvl].append(parts[lvl])
    for lvl, vals in tax_map.items():
        merged[lvl] = vals
    return merged.reset_index()


def correct_counts_against_controls(asv_meta: pd.DataFrame, meta: pd.DataFrame,
                                    meta_sample_col: str, type_col: str,
                                    scope_label='Scope Flush', skin_label='Skin Brush') -> pd.DataFrame:
    """Subtract per-ASV means from control groups (scope, skin)."""
    # Determine control ASV sets
    ctrl = meta[[meta_sample_col, type_col]].copy()
    df = asv_meta.merge(ctrl, left_on=meta_sample_col, right_on=meta_sample_col, how='left', suffixes=('', '_meta'))

    # Compute per-ASV mean in control types
    keep_cols = ['ASV_ID', meta_sample_col, 'count']
    scope_mean = df[df[type_col] == scope_label][keep_cols].groupby('ASV_ID')['count'].mean().reset_index().fillna(0)
    scope_mean.columns = ['ASV_ID', 'nctrl_mean']
    skin_mean = df[df[type_col] == skin_label][keep_cols].groupby('ASV_ID')['count'].mean().reset_index().fillna(0)
    skin_mean.columns = ['ASV_ID', 'offtarg_mean']

    out = df.merge(scope_mean, on='ASV_ID', how='left').merge(skin_mean, on='ASV_ID', how='left')
    out['nctrl_mean'] = out['nctrl_mean'].fillna(0)
    out['offtarg_mean'] = out['offtarg_mean'].fillna(0)
    out['corr_count'] = (out['count'] - out['nctrl_mean'] - out['offtarg_mean']).clip(lower=0).astype(int)

    # Remove control types from downstream matrix
    out = out[~out[type_col].isin([scope_label, skin_label])].copy()
    return out


def presence_shared_percent(count_mat: pd.DataFrame) -> pd.DataFrame:
    """Presence/absence Jaccard * 100 from count matrix (ASVs x samples)."""
    pa = (count_mat > 0).astype(int)
    shared = pa.T.dot(pa)
    n = pa.sum()
    n_arr = n.to_numpy()
    pct = shared.div(n_arr[:, None] + n_arr[None, :] - shared.to_numpy()) * 100
    return pd.DataFrame(pct, index=shared.index, columns=shared.columns).fillna(0)


def build_greys_cmap() -> LinearSegmentedColormap:
    colors = [(0.0, '#ffffff'), (0.2, '#d9d9d9'), (1.0, '#000000')]
    return LinearSegmentedColormap.from_list("light_greyscale", colors, N=256)


def clustermap_shared_percent(
    shared_pct: pd.DataFrame,
    col_legend_df: pd.DataFrame,
    row_legend_df: pd.DataFrame,
    col_palette: Dict[int, str],
    out_svg: Path,
    out_pdf: Path,
    cmap=None,
) -> None:
    cmap = cmap or build_greys_cmap()
    g = sns.clustermap(
        shared_pct, method='ward', metric='euclidean',
        col_colors=col_legend_df, row_colors=row_legend_df,
        cmap=cmap, vmin=0, vmax=100, linewidths=0,
        xticklabels=False, yticklabels=False,
        dendrogram_ratio=(0.05, 0.05), colors_ratio=(0.02, 0.02),
        figsize=(32, 32), cbar_pos=(1.02, 0.2, 0.03, 0.4), alpha=1.0,
    )
    
    colorbar = g.ax_heatmap.collections[0].colorbar
    colorbar.set_label("% Shared ASVs", rotation=270, labelpad=15)
    g.ax_heatmap.tick_params(axis='x', bottom=True, labelbottom=True)
    g.ax_heatmap.tick_params(axis='x', which='both', length=5)
    
    # Create legend with Depth values as labels
    handles = [Patch(facecolor=color, edgecolor='black', label=str(depth)) 
               for depth, color in sorted(col_palette.items())]
    
    g.ax_heatmap.legend(handles=handles, title='Depth', 
                        bbox_to_anchor=(1.05, 1), loc='upper left',
                        frameon=True, fontsize=10)
    
    g.fig.savefig(out_svg, bbox_inches='tight')
    g.fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(g.fig)


# ========= Violin plotting (from your earlier function, tweaked to be standalone) =========
def plot_grouppair_violins_sns(
    shared_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_id_col: str = "sample",
    group_col: str = "type_group",
    include_within: bool = True,
    group_order: list | None = None,
    group_colors: dict | None = None,
    title: str = "Group pair % shared",
    ylabel: str = "% shared",
    inner: str = "quartile",
    cut: float = 0,
    bw: str | float = "scott",
    scale: str = "width",
    ax=None,
):
    # Align samples
    matrix_samples = [s for s in shared_df.index if s in shared_df.columns]
    if not matrix_samples:
        raise ValueError("Matrix has no overlapping index/column names.")
    meta = meta_df.copy()
    meta[sample_id_col] = meta[sample_id_col].astype(str).str.strip()
    meta[group_col] = meta[group_col].astype(str).str.strip()
    meta = meta.drop_duplicates(subset=[sample_id_col], keep="first")
    meta = meta[meta[sample_id_col].isin(matrix_samples)]
    if meta.empty:
        raise ValueError("No overlapping samples between matrix and metadata.")
    samples = [s for s in matrix_samples if s in set(meta[sample_id_col])]
    shared = shared_df.loc[samples, samples]
    meta = meta.set_index(sample_id_col).loc[samples].reset_index()

    # Groups & order
    seen = set(); groups_in_use = []
    for g in meta[group_col]:
        if g not in seen:
            seen.add(g); groups_in_use.append(g)
    if group_order:
        specified = [str(g).strip() for g in group_order if str(g).strip()]
        groups = [g for g in specified if g in groups_in_use] + [g for g in groups_in_use if g not in specified]
    else:
        groups = groups_in_use

    group_to_samples = {g: [s for s in samples if meta.loc[meta[sample_id_col] == s, group_col].iloc[0] == g] for g in groups}

    def _rgba(c, g):
        try:
            return to_rgba(c)
        except ValueError:
            warnings.warn(f"Ignoring invalid color '{c}' for group '{g}'.")
            return None

    resolved = {}
    user_map = group_colors or {}
    for g in groups:
        if g in user_map:
            rgba = _rgba(user_map[g], g)
            if rgba is not None:
                resolved[g] = rgba
    cmap = plt.get_cmap("tab20"); auto_i = 0
    for g in groups:
        if g not in resolved:
            resolved[g] = cmap(auto_i % cmap.N); auto_i += 1

    def _blend(a, b):
        a = np.array(a); b = np.array(b)
        m = (a + b) / 2.0
        m[3] = max(a[3], b[3])
        return tuple(m)

    gpairs = combinations_with_replacement(groups, 2) if include_within else combinations(groups, 2)
    rows, labels, pal = [], [], {}
    for g1, g2 in gpairs:
        s1, s2 = group_to_samples.get(g1, []), group_to_samples.get(g2, [])
        if not s1 or not s2:
            continue
        if g1 == g2:
            if len(s1) < 2: continue
            sub = shared.loc[s1, s1].to_numpy()
            iu = np.triu_indices(len(s1), k=1); vals = sub[iu]
            col = resolved[g1]
        else:
            vals = shared.loc[s1, s2].to_numpy().ravel()
            col = _blend(resolved[g1], resolved[g2])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0: continue
        lab = f"{g1} × {g2}"
        labels.append(lab); pal[lab] = col
        rows.append(pd.DataFrame({"pair": lab, "value": vals}))
    if not rows:
        raise ValueError("No pairwise values to plot.")
    tidy = pd.concat(rows, ignore_index=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels)), 4.5), dpi=150)
    sns.violinplot(data=tidy, x="pair", y="value", order=labels, palette=pal, cut=cut, bw=bw, scale=scale, inner=inner, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(ylabel); ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.5)
    return ax, tidy


# ========= Pipeline pieces =========
def compute_and_save_block(
    mode_name: str,                      # "micro" or "mito"
    asv_path: Path,
    out_root: Path,                      # e.g., <data>/<sub>/metadata or <data>/<sub>/mito/metadata
    asv_out_root: Path,                  # e.g., <data>/<sub>/ASVs or <data>/<sub>/mito/ASVs
    meta: pd.DataFrame,
    tax_df: pd.DataFrame,
    meta_sample_col: str,
    type_col: str,
    type_palette: Dict[str, str],
    keep_types: Sequence[str],
    fastq_stats_df: pd.DataFrame,
    dashed_line_y: Optional[float] = None,   # Only used in mito box+swarm
    include_rank_filters: Optional[Dict[str, set[str]]] = None,
) -> None:
    ensure_dir(out_root)
    ensure_dir(asv_out_root)

    # Long ASV
    long_asv = read_asv_wide_to_long(asv_path, meta_sample_col, tax_df.index)
    print(f"[i] {mode_name} ASV long shape: {long_asv.shape}")
    if long_asv.empty:
        print(f"No ASV counts found in {asv_path}, moving on...")
        return
    asv_tax = add_taxonomy(long_asv, tax_df)
    if include_rank_filters:
        pre_filter = len(asv_tax)
        mask = pd.Series(False, index=asv_tax.index, dtype=bool)
        for level, names in include_rank_filters.items():
            if level not in asv_tax.columns:
                continue
            norm_col = asv_tax[level].apply(normalize_taxon_value)
            mask |= norm_col.isin(names)
        asv_tax = asv_tax[mask].copy()
        post_filter = len(asv_tax)
        print(f"[i] include-rank filters kept {post_filter} / {pre_filter} ASV rows in {mode_name} block")
        if asv_tax.empty:
            print(f"[w] No ASVs matched include-rank filters {include_rank_filters}; skipping {mode_name} block.")
            return
    sample_list = asv_tax[meta_sample_col].unique().tolist()
    meta = meta[meta[meta_sample_col].isin(sample_list)].copy()

    if not keep_types:
        meta[type_col] = meta[type_col].astype(int)
        keep_types = sorted(meta[type_col].unique().tolist())

    # Merge with metadata
    asv_meta = asv_tax.merge(meta, on=meta_sample_col, how='inner')
    
    # Stats per sample for raw reads
    reads_df = fastq_stats_df.copy()
    reads_df = reads_df.rename(columns={'num_seqs': 'num_reads_total'})
    reads_df['raw_count'] = (reads_df['num_reads_total'] / 2.0)

    # Build metastat table
    cnt_df = asv_meta.groupby([meta_sample_col])['count'].sum().reset_index()
    metastat = meta.merge(reads_df[[meta_sample_col, 'raw_count']], on=meta_sample_col, how='left') \
                   .merge(cnt_df, on=meta_sample_col, how='left')
    metastat['pass_filter'] = [t if s in set(asv_meta[meta_sample_col]) else 'Failed-QC'
                               for s, t in zip(metastat[meta_sample_col], metastat[type_col])]
    long_df = metastat.groupby([type_col, 'pass_filter', meta_sample_col])['raw_count'].sum().reset_index()
    long_df = long_df[long_df['raw_count'] > 0]

    # Box + swarm
    plt.figure(figsize=(10, 10))
    ax = sns.boxplot(x=type_col, y='raw_count', data=long_df, color='white', fliersize=0, linewidth=1, showcaps=True,
                     order=list(keep_types))
    sns.stripplot(data=long_df, x=type_col, y='raw_count', hue='pass_filter', alpha=0.75, ax=ax, legend=False,
                  jitter=0.25, palette=type_palette)
    if dashed_line_y is not None:
        plt.axhline(y=dashed_line_y, linestyle='--', color='black', linewidth=1)
    plt.title("Sample Type"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(out_root / f"type_group_swarmplot_{mode_name}.svg")
    plt.savefig(out_root / f"type_group_swarmplot_{mode_name}.png")
    plt.close()

    # Control subtraction (scope+skin), pivot to ASV x sample corrected counts
    corr_meta = correct_counts_against_controls(asv_meta, meta, meta_sample_col, type_col)
    cleaned = corr_meta.pivot_table(index='ASV_ID', columns=meta_sample_col, values='corr_count', aggfunc='sum', fill_value=0)

    # Keep only assigned Domain and only kept samples
    keep_asvs = corr_meta[corr_meta['Domain'] != 'Unassigned']['ASV_ID'].unique()
    kept_samples = metastat[metastat['pass_filter'] != 'Failed-QC'][meta_sample_col].unique().tolist()
    final_mat = cleaned.reindex(index=keep_asvs).dropna(how='all')
    final_mat = final_mat[[c for c in final_mat.columns if c in kept_samples]].fillna(0).astype(int)

    # Write outputs
    save_df(corr_meta, out_root / f"ASV_meta_{mode_name}.tsv")
    save_mat(final_mat, asv_out_root / f"ASV_final.{mode_name}.tsv")
    save_df(metastat, out_root / f"master_table_{mode_name}.tsv")
    save_df(meta, out_root / f"metadata_updated_{mode_name}.tsv")

    # Legends (sample colors)
    m_df = metastat[metastat[meta_sample_col].isin(final_mat.columns)].set_index(meta_sample_col)
    filtered = final_mat[m_df.index.tolist()]
    col_colors_df = pd.DataFrame({
        type_col: m_df[type_col].map(type_palette),
    }, index=m_df.index)
    row_colors_df = col_colors_df.copy()
    
    # Shared % matrix and clustermap
    shared_pct = presence_shared_percent(filtered)
    sub_palette = {x:type_palette[x] for x in type_palette if x in keep_types}
    clustermap_shared_percent(
        shared_pct,
        col_legend_df=col_colors_df,
        row_legend_df=row_colors_df,
        col_palette=sub_palette,
        out_svg=out_root / f"clustermap_ASVpercent_{mode_name}.svg",
        out_pdf=out_root / f"clustermap_ASVpercent_{mode_name}.png",
    )

    # Violin pairs (only for selected groups present)
    vg = [g for g in set(meta[type_col])]
    if vg:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        _, tidy = plot_grouppair_violins_sns(
            shared_pct, meta,
            sample_id_col=meta_sample_col, group_col=type_col,
            include_within=True,
            group_order=vg,
            group_colors=type_palette,
            title="ASVs Shared by Sample Type",
            ylabel="ASVs Shared (%)",
            inner="quartile",
            cut=0,
            ax=ax,
        )
        plt.tight_layout()
        plt.savefig(out_root / f"violin_ASVpercent_{mode_name}.svg", bbox_inches='tight')
        plt.savefig(out_root / f"violin_ASVpercent_{mode_name}.png", bbox_inches='tight')
        plt.close()


# ========= CLI =========
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ASV summary tables and plots (microbial & mitochondrial).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("Project I/O")
    io.add_argument("--data-dir", type=Path, required=True, help="Project root")
    io.add_argument("--sub-dir", default="spark_combined_output", help="Subdirectory under data-dir")
    io.add_argument("--metadata", type=Path, default=None, help="Metadata TSV path (default: <data>/ref_db/spark_metadata.tsv)")
    io.add_argument("--sample-manifest", type=Path,
                    help="TSV with columns: sample_id, fastq_r1, fastq_r2")
    io.add_argument("--taxonomy", type=Path, required=True, help="SILVA taxonomy TSV (Feature ID, Taxon)")

    cols = p.add_argument_group("Columns / Groups")
    cols.add_argument("--group1-col", default="group1", help="Primary grouping column in metadata")
    cols.add_argument("--color-col", default="Color", help="Color column in metadata")
    cols.add_argument("--sample-id-col", default=SAMPLE_ID_COL,
                      help="Column containing unique sample IDs (must match manifest/ASV headers)")
    cols.add_argument("--keep-types", default="",
                      help="Comma-separated list of types to keep (order honored)")

    reads = p.add_argument_group("Read Stats")
    reads.add_argument("--fastq-stats", default="stats/fastq_stats.tsv", help="TSV with columns: file, num_seqs")

    asv = p.add_argument_group("ASV Matrices")
    asv.add_argument("--asv-micro", type=Path, required=True, help="ASV_target.micro.tsv")
    asv.add_argument("--asv-mito", type=Path, required=True, help="ASV_target.mito.tsv")

    tax = p.add_argument_group("Taxonomy Filters")
    tax.add_argument(
        "--include-rank",
        action="append",
        default=[],
        metavar="LEVEL:NAME",
        help="Only keep ASVs whose taxonomy at LEVEL (Domain/Phylum/Class/Order/Family/Genus/Species) matches NAME."
             " Provide multiple flags to OR filters.",
    )

    vis = p.add_argument_group("Visual")
    vis.add_argument("--mito-threshold-line", type=float, default=1000.0, help="Dashed line Y on mito swarm plot (set negative to disable)")

    mode = p.add_argument_group("Modes / Toggles")
    mode.add_argument("--make-micro", action="store_true", help="Run microbial block")
    mode.add_argument("--make-mito", action="store_true", help="Run mitochondrial block")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--verbose", action="store_true", help="Verbose logging")

    return p


def main():
    args = get_parser().parse_args()

    data_dir = args.data_dir
    sub_dir = args.sub_dir
    meta_path = args.metadata
    keep_types = parse_list_csv(args.keep_types)
    include_rank_filters = parse_rank_filters(args.include_rank)
    
    # Resolve canonical paths
    def resolve(rel_or_abs: str | Path) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (data_dir / sub_dir / p)

    fastq_stats_path = resolve(args.fastq_stats)
    asv_micro_path = resolve(args.asv_micro)
    asv_mito_path = resolve(args.asv_mito)
    taxonomy_path = resolve(args.taxonomy)

    if args.verbose:
        print(f"[i] Metadata: {meta_path}")
        print(f"[i] Taxonomy: {taxonomy_path}")
        print(f"[i] Fastq stats: {fastq_stats_path}")
        print(f"[i] ASV micro: {asv_micro_path}")
        print(f"[i] ASV mito : {asv_mito_path}")

    # Read data
    meta_sample_col = args.sample_id_col
    meta = read_metadata(meta_path, meta_sample_col, keep_types)

    manifest_path = args.sample_manifest
    manifest_map = load_sample_manifest(manifest_path)

    # set palette
    palette = {k[0]: k[1] for k in zip(meta[args.group1_col], meta[args.color_col])}
    palette = dict(sorted(palette.items()))
    palette['Failed-QC'] = '#d3d3d3'  # light grey for failed QC
    
    tax_df = read_taxonomy_table(taxonomy_path)
    fastq_df = read_fastq_stats(
        fastq_stats_path,
        meta_sample_col,
        manifest_map,
    )

    # Output roots
    meta_root = data_dir / sub_dir / "metadata"
    mito_meta_root = data_dir / sub_dir / "mito" / "metadata"
    asv_root = data_dir / sub_dir / "ASVs"
    mito_asv_root = data_dir / sub_dir / "mito" / "ASVs"

    # If neither toggle provided, run both
    run_micro = args.make_micro or (not args.make_micro and not args.make_mito)
    run_mito = args.make_mito or (not args.make_micro and not args.make_mito)
    
    # MICRO
    if run_micro:
        if args.verbose: print("[i] Running microbial block …")
        compute_and_save_block(
            mode_name="micro",
            asv_path=asv_micro_path,
            out_root=meta_root,
            asv_out_root=asv_root,
            meta=meta.copy(),
            tax_df=tax_df,
            meta_sample_col=meta_sample_col,
            type_col=args.group1_col,
            type_palette=palette,
            keep_types=keep_types,
            fastq_stats_df=fastq_df.copy(),
            dashed_line_y=None,
            include_rank_filters=include_rank_filters,
        )

    # MITO
    if run_mito:
        if args.verbose: print("[i] Running mitochondrial block …")
        compute_and_save_block(
            mode_name="mito",
            asv_path=asv_mito_path,
            out_root=mito_meta_root,
            asv_out_root=mito_asv_root,
            meta=meta.copy(),
            tax_df=tax_df,
            meta_sample_col=meta_sample_col,
            type_col=args.group1_col,
            type_palette=palette,
            keep_types=keep_types,
            fastq_stats_df=fastq_df.copy(),
            dashed_line_y=(args.mito_threshold_line if args.mito_threshold_line >= 0 else None),
            include_rank_filters=include_rank_filters,
        )

    if args.verbose:
        print("✔ Done.")


if __name__ == "__main__":
    main()
