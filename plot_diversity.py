#!/usr/bin/env python3
"""
beta_alpha_analytics.py
Alpha (Shannon), PERMANOVA (global & pairwise), and UMAP plots from
precomputed distance matrices (Bray, Jaccard) + metadata. Optional "mito" pass.

Outputs (under --outdir and --mito-outdir):
- diversity/
  - alpha_wilcoxon.tsv
  - permanova_global.tsv
  - permanova_pairwise.tsv
  - Beta_Heatmap_permanova.(pdf|svg)
  - UMAP_*.{pdf,svg} (several variants)
- mito/diversity/ (same set if mito inputs provided)

Examples
--------
python beta_alpha_analytics.py \
  --metadata spark_combined_output/metadata/metadata_updated.tsv \
  --master   spark_combined_output/metadata/master_table.tsv \
  --alpha    spark_combined_output/diversity/shannon.tsv \
  --bray     spark_combined_output/diversity/bray.tsv \
  --jacc     spark_combined_output/diversity/jaccard.tsv \
  --outliers-all  spark_combined_output/metadata/outliers_table.tsv \
  --outliers-type spark_combined_output/metadata/outliers_type_group.tsv \
  --exclude-types "Skin Brush,Scope Flush" \
  --type-order "Oral Rinse,BAL,Bronchial Brush" \
  --outdir spark_combined_output/diversity

# With mito
python beta_alpha_analytics.py \
  ... (same as above) \
  --mito-alpha spark_combined_output/mito/diversity/shannon.mito.tsv \
  --mito-bray  spark_combined_output/mito/diversity/bray.mito.tsv \
  --mito-jacc  spark_combined_output/mito/diversity/jaccard.mito.tsv \
  --mito-outdir spark_combined_output/mito/diversity
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import umap
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon
from scipy.spatial.distance import pdist, squareform


# -------------------------- style --------------------------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
sns.set_theme()
sns.set_style("white")

PALETTE_TYPES = {
    'Scope Flush': '#E69F00',
    'Skin Brush':  '#CC79A7',
    'Bronchial Brush':  '#009E73',
    'BAL':         '#0072B2',
    'Oral Rinse':  '#6A3D9A',
    'Failed-QC':   'lightgray',
}
PALETTE_THREE = {k: PALETTE_TYPES[k] for k in ('Bronchial Brush', 'BAL', 'Oral Rinse')}
PALETTE_STATUS = {'Non-Cancer': 'white', 'Cancer': '#A50026', 'methods': 'lightgray'}

# -------------------------- utils --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_tsv(path: Path, index_col=None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", index_col=index_col)

def read_count_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if "ASV_ID" in df.columns:
        df = df.set_index("ASV_ID")
    else:
        df = df.set_index(df.columns[0])
    return df

def perform_umap_precomputed(dist_df: pd.DataFrame,
                             n_neighbors: int = 30,
                             min_dist: float = 0.01,
                             random_state: int = 42,
                             prefix: str = "") -> pd.DataFrame:
    """UMAP on a precomputed square distance matrix."""
    if dist_df.shape[0] != dist_df.shape[1]:
        raise ValueError(f"{prefix} distance matrix must be square.")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    emb = reducer.fit_transform(dist_df.values)
    umap_df = pd.DataFrame(emb, index=dist_df.index, columns=["UMAP1", "UMAP2"])
    return umap_df

def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    if right.empty:
        return left
    cols = [c for c in right.columns if c != on]
    return left.merge(right[[on] + cols], how="left", on=on)

def patient_level_pairwise_wilcoxon(df: pd.DataFrame, patient_col: str, group_col: str, value_col: str) -> pd.DataFrame:
    work = df[[patient_col, group_col, value_col]].dropna().copy()
    patient_group = work.groupby([patient_col, group_col], as_index=False)[value_col].mean()
    res = []
    for a, b in combinations([g for g in patient_group[group_col].dropna().unique()], 2):
        wide = (
            patient_group[patient_group[group_col].isin([a, b])]
            .pivot(index=patient_col, columns=group_col, values=value_col)
            .dropna(subset=[a, b])
        )
        if len(wide) < 3:
            continue
        try:
            stat, p = wilcoxon(
                wide[a].values,
                wide[b].values,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                mode="auto",
            )
        except ValueError:
            stat, p = np.nan, np.nan
        res.append(
            {
                "group1": a,
                "group2": b,
                "n_paired_patients": int(len(wide)),
                "wilcoxon_w": float(stat) if not np.isnan(stat) else np.nan,
                "pval": float(p) if not np.isnan(p) else np.nan,
            }
        )
    out = pd.DataFrame(res)
    if not out.empty:
        valid = out["pval"].notna()
        out["qval"] = np.nan
        out.loc[valid, "qval"] = multipletests(out.loc[valid, "pval"], method="fdr_bh")[1]
        out["significant"] = out["qval"] < 0.05
    return out

def aggregate_mean_relative(counts: pd.DataFrame, group_ids: pd.Series) -> pd.DataFrame:
    sample_rel = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    tmp = sample_rel.copy()
    tmp["_group"] = group_ids.values
    return tmp.groupby("_group").mean()

def permanova_r2(dist_matrix: np.ndarray, group_labels: np.ndarray) -> float:
    n = dist_matrix.shape[0]
    if n < 2:
        return np.nan
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ (dist_matrix ** 2) @ H
    ss_total = np.trace(G)
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for group in np.unique(group_labels):
        mask = group_labels == group
        n_g = mask.sum()
        if n_g > 0:
            ss_between += np.sum(G[np.ix_(mask, mask)]) / n_g
    return ss_between / ss_total

def permanova_stat(dist_matrix: np.ndarray, group_labels: np.ndarray) -> tuple[float, float]:
    r2 = permanova_r2(dist_matrix, group_labels)
    groups = np.unique(group_labels)
    n = len(group_labels)
    g = len(groups)
    if g < 2 or n <= g:
        return r2, np.nan
    f = (r2 / (g - 1)) / ((1 - r2) / (n - g)) if r2 < 1 else np.inf
    return r2, f

def blocked_permanova(counts: pd.DataFrame, meta: pd.DataFrame, patient_col: str, group_col: str, permutations: int = 999) -> pd.DataFrame:
    counts = counts.loc[meta["sample"]]
    group_ids = meta[patient_col].astype(str) + "||" + meta[group_col].astype(str)
    agg = aggregate_mean_relative(counts, group_ids)
    agg_meta = (
        meta.assign(_group=group_ids)
        .drop_duplicates("_group")
        .loc[:, ["_group", patient_col, group_col]]
        .set_index("_group")
        .loc[agg.index]
    )
    patient_counts = agg_meta.groupby(patient_col)[group_col].nunique()
    keep_patients = patient_counts[patient_counts >= 2].index
    agg_meta = agg_meta[agg_meta[patient_col].isin(keep_patients)].copy()
    agg = agg.loc[agg_meta.index]
    if agg.empty or agg_meta[group_col].nunique() < 2:
        raise ValueError("Blocked PERMANOVA requires at least two sample types across patients with >= 2 observed types.")

    dist_array = squareform(pdist(agg.values, metric="braycurtis"))
    dist = DistanceMatrix(dist_array.copy(order="C"), ids=agg.index.tolist())
    labels = agg_meta[group_col].astype(str).values
    patients = agg_meta[patient_col].astype(str).values
    obs_r2, obs_f = permanova_stat(dist.data, labels)

    perm_stats = []
    rng = np.random.default_rng(42)
    for _ in range(permutations):
        perm_labels = labels.copy()
        for patient in np.unique(patients):
            mask = patients == patient
            perm_labels[mask] = rng.permutation(perm_labels[mask])
        _, perm_f = permanova_stat(dist.data, perm_labels)
        perm_stats.append(perm_f)

    p = (1 + np.sum(np.array(perm_stats) >= obs_f)) / (permutations + 1)
    return pd.DataFrame([{
        "term": "sample_type_within_patient",
        "test statistic": obs_f,
        "p-value": p,
        "R2": obs_r2,
        "n_patient_type_profiles": int(len(labels)),
        "n_patients": int(pd.Series(patients).nunique()),
    }])

def blocked_pairwise_permanova(counts: pd.DataFrame, meta: pd.DataFrame, patient_col: str, group_col: str, permutations: int = 999) -> pd.DataFrame:
    rows = []
    for a, b in combinations(sorted(meta[group_col].dropna().unique()), 2):
        sub = meta[meta[group_col].isin([a, b])].copy()
        patient_counts = sub.groupby(patient_col)[group_col].nunique()
        keep_patients = patient_counts[patient_counts == 2].index
        sub = sub[sub[patient_col].isin(keep_patients)].copy()
        if sub.empty:
            continue
        res = blocked_permanova(counts, sub, patient_col, group_col, permutations)
        if res.empty:
            continue
        rows.append({
            "Group1": a,
            "Group2": b,
            "test statistic": float(res.loc[0, "test statistic"]),
            "p-value": float(res.loc[0, "p-value"]),
            "R2": float(res.loc[0, "R2"]),
            "n_patients": int(len(keep_patients)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q-value"] = multipletests(out["p-value"], method="fdr_bh")[1]
    return out

def global_and_pairwise_permanova(bray: pd.DataFrame, group: pd.Series, permutations: int = 999):
    # Align
    ids = bray.index.intersection(group.index)
    bray = bray.loc[ids, ids]
    grp = group.loc[ids]
    # global
    dm = DistanceMatrix(bray.values.copy(order="C"), ids=bray.index.tolist())
    glob = permanova(dm, grp, permutations=permutations).to_frame().T
    glob.insert(0, "term", "global")
    # pairwise
    res = []
    uniq = [g for g in grp.unique()]
    for a, b in combinations(uniq, 2):
        sub_ids = grp[grp.isin([a, b])].index
        counts = grp.loc[sub_ids].value_counts()
        if (counts < 2).any():
            continue
        sub_dm = DistanceMatrix(bray.loc[sub_ids, sub_ids].values.copy(order="C"), ids=sub_ids.tolist())
        pw = permanova(sub_dm, grp.loc[sub_ids], permutations=permutations)
        s = pw.copy()
        s["Group1"], s["Group2"] = a, b
        res.append(s)
    pair = pd.DataFrame(res)
    if not pair.empty:
        pair = pair.reset_index(drop=True)
        pair["q-value"] = multipletests(pair["p-value"], method="fdr_bh")[1]
    return glob, pair

def heatmap_pairwise_q(pair: pd.DataFrame, out_base: Path, title: str):
    if pair.empty:
        warnings.warn("No pairwise PERMANOVA results to plot.")
        return
    M = pair.pivot(index="Group1", columns="Group2", values="q-value")
    M = M.combine_first(M.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(M, annot=True, cmap="coolwarm_r", vmin=0, vmax=0.1,
                cbar_kws={"label": "q-value"}, linewidths=0.5, linecolor="lightgray")
    plt.title(title)
    plt.tight_layout()
    for ext in ("svg", "pdf"):
        plt.savefig(out_base.with_suffix(f".{ext}"))
    plt.close()

def scatter_umap(df: pd.DataFrame,
                 x: str, y: str,
                 hue: str, size: str | None,
                 style: str | None,
                 palette: dict | str,
                 title: str,
                 out_base: Path,
                 label: str | None = None,
                 size_range=(40, 400),
                 edgecolor="lightgray"):

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.scatterplot(
        data=df, x=x, y=y,
        hue=hue, size=size, style=style,
        palette=palette, sizes=size_range,
        alpha=0.8,
        edgecolor=edgecolor, linewidth=0.5,
        ax=ax,
        hue_order=palette.keys() if isinstance(palette, dict) else None
    )

    # --- ADD LABELS IF PROVIDED ---
    if label is not None:
        for _, r in df.iterrows():
            ax.annotate(
                text=str(r[label]),
                xy=(r[x], r[y]),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=6,
                alpha=0.85,
                ha="left",
                va="center",
            )

    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1), frameon=False)

    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"))

    plt.close(fig)


# -------------------------- run block --------------------------
def run_one_pass(name: str,
                 meta: pd.DataFrame,
                 master: pd.DataFrame,
                 count_table: pd.DataFrame,
                 alpha: pd.DataFrame,
                 bray: pd.DataFrame,
                 jacc: pd.DataFrame,
                 outdir: Path,
                 exclude_types: list[str],
                 type_order: list[str] | None,
                 neighbors: int,
                 min_dist: float,
                 permutations: int,
                 patient_col: str | None):
    ensure_dir(outdir)

    # Merge everything we have
    meta = meta.copy()
    if "sample" not in meta.columns:
        raise ValueError("--metadata must contain a 'sample' column.")
    df = meta.copy()

    # Alpha
    if not alpha.empty:
        alpha = alpha.rename_axis("sample").reset_index()
        df = safe_merge(df, alpha, on="sample")  # adds 'Shannon'

    # UMAPs from distances (precomputed)
    if not bray.empty:
        bray_umap = perform_umap_precomputed(
            bray, n_neighbors=neighbors, min_dist=min_dist, random_state=42, prefix=f"{name}/Bray"
        ).rename(columns={"UMAP1": "UMAP1", "UMAP2": "UMAP2"})
        df = safe_merge(df, bray_umap.reset_index(), on="sample")
    if not jacc.empty:
        jacc_umap = perform_umap_precomputed(
            jacc, n_neighbors=neighbors, min_dist=min_dist, random_state=42, prefix=f"{name}/Jacc"
        ).rename(columns={"UMAP1": "Jacc_UMAP1", "UMAP2": "Jacc_UMAP2"})
        df = safe_merge(df, jacc_umap.reset_index(), on="sample")

    # Bring counts if available from master table (used for point sizes)
    if not master.empty:
        need = {"sample", "count", "pass_filter"}
        inter = [c for c in need if c in master.columns]
        if inter:
            df = safe_merge(df, master[inter], on="sample")

    # Outlier flags if present (overall_OL / typ_grp_OL expected column names)
    # These may already exist in metadata; we leave as-is if present.

    # Filter: remove failed QC and excluded types
    if "pass_filter" in df.columns:
        df = df.loc[df["pass_filter"] != "Failed-QC"].copy()
    if exclude_types:
        df = df.loc[~df["type_group"].isin(exclude_types)].copy()

    # Establish plotting order & palettes
    if type_order:
        order = [t for t in type_order if t in df.get("type_group", pd.Series()).unique()]
    else:
        order = list(df.get("type_group", pd.Series()).dropna().unique())
    type_palette = {k: PALETTE_TYPES.get(k, PALETTE_TYPES.get("Failed-QC", "#999999")) for k in order}

    if alpha.empty and bray.empty:
        warnings.warn("No alpha or beta inputs were provided; only descriptive plots without inferential tables can be generated.")

    if not alpha.empty:
        if patient_col is None:
            raise ValueError("--patient-col is required for patient-aware alpha-diversity inference.")
        if patient_col not in df.columns:
            raise ValueError(f"Patient column '{patient_col}' is missing from metadata.")

    if not bray.empty:
        if patient_col is None:
            raise ValueError("--patient-col is required for patient-aware beta-diversity inference.")
        if patient_col not in df.columns:
            raise ValueError(f"Patient column '{patient_col}' is missing from metadata.")
        if count_table.empty:
            raise ValueError("--count-table is required for patient-aware beta-diversity inference.")

    # ---------- Alpha Wilcoxon tests + boxplot with visible alpha ----------
    out_alpha = outdir / ("tables/alpha_wilcoxon.tsv" if name == "micro" else "tables/alpha_wilcoxon_mito.tsv")
    if {"Shannon", "type_group"} <= set(df.columns):
        ttab = patient_level_pairwise_wilcoxon(df, patient_col, "type_group", "Shannon")
        ttab.to_csv(out_alpha, sep="\t", index=False)

        # single-axis boxplot (no hue), colors contain alpha
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(
            data=df, x="type_group", y="Shannon",
            order=order, palette=type_palette, linewidth=0.8
        )
        # also fade lines (whiskers/medians) a bit
        for line in ax.lines:
            line.set_alpha(0.6)

        # Add stats
        pairs = [(r["group1"], r["group2"]) for _, r in ttab.iterrows() if r["group1"] in order and r["group2"] in order]
        if pairs:
            annot = Annotator(ax, pairs, data=df, x="type_group", y="Shannon", order=order)
            annot.configure(test=None, text_format='star', loc='inside', verbose=0)
            annot.set_pvalues_and_annotate(ttab.set_index(["group1", "group2"]).loc[pairs, "qval"].tolist())

        ax.set_xlabel("")
        ax.set_ylabel("Shannon")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis='x', rotation=0)
        sns.despine(ax=ax, top=True, right=True)
        fig.tight_layout()
        for ext in ("svg", "pdf"):
            fig.savefig(outdir / f"plots/Alpha_type_boxplot_{name}.{ext}")
        plt.close(fig)

        # Faceted by status (boxed with alpha’d palette)
        if "status" in df.columns:
            TYPE_ORDER = ["Oral Rinse", "BAL", "Bronchial Brush"]

            g = sns.FacetGrid(
                df,
                col="type_group",
                col_order=TYPE_ORDER,
                sharey=True,
                height=5,
                aspect=0.2,
                col_wrap=3,
            )

            # draw each facet with a single color taken from that facet's type_group
            def facet_boxplot(data, **kwargs):
                tg = str(data["type_group"].iloc[0])
                facet_color = PALETTE_TYPES.get(tg, None)

                sns.boxplot(
                    data=data,
                    x="status",
                    y="Shannon",
                    order=["Cancer", "Non-Cancer"],
                    color=facet_color,        # <-- one color per facet
                    linewidth=0.8,
                    width=0.85,
                    gap=0.15,
                    showfliers=True,
                    boxprops={"alpha": 0.5},
                    medianprops={"alpha": 1},
                    whiskerprops={"alpha": 1},
                    capprops={"alpha": 1},
                )

            g.map_dataframe(facet_boxplot)

            for ax in g.axes.flat:
                sns.despine(ax=ax, left=False, bottom=False)
                ax.set_ylim(0, 5)
                ax.set_xlabel("")
                ax.set_ylabel("Shannon")
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.tick_params(axis="x", rotation=0)

            g.figure.set_size_inches(12, 5)
            plt.tight_layout()
            for ext in ("svg", "pdf"):
                g.figure.savefig(outdir / f"plots/Alpha_status_boxplot_{name}.{ext}")
            plt.close(g.figure)

    # ---------- PERMANOVA ----------
    if not bray.empty and "type_group" in df.columns:
        meta_beta = (
            df[["sample", patient_col, "type_group"]]
            .dropna()
            .drop_duplicates("sample")
        )
        sample_ids = [s for s in meta_beta["sample"] if s in count_table.columns]
        meta_beta = meta_beta[meta_beta["sample"].isin(sample_ids)].copy()
        counts_beta = count_table[sample_ids].T
        glob = blocked_permanova(counts_beta, meta_beta, patient_col, "type_group", permutations=permutations)
        pair = blocked_pairwise_permanova(counts_beta, meta_beta, patient_col, "type_group", permutations=permutations)
        glob.to_csv(outdir / ("tables/permanova_global.tsv" if name == "micro" else "tables/permanova_global_mito.tsv"), sep="\t", index=False)
        pair.to_csv(outdir / ("tables/permanova_pairwise.tsv" if name == "micro" else "tables/permanova_pairwise_mito.tsv"), sep="\t", index=False)

        # Heatmap of pairwise q-values
        #heatmap_pairwise_q(
        #    pair,
        #    outdir / f"Beta_Heatmap_permanova_{name}",
        #    title="Pairwise PERMANOVA (q-values)\nBlue = Not Significant, Red = Significant"
        #)

    # ---------- UMAP figures (Bray) ----------
    have_umap = {"UMAP1", "UMAP2"}.issubset(df.columns)
    if have_umap and "type_group" in df.columns:
        size_col = "count" if "count" in df.columns else None

        scatter_umap(
            df, "UMAP1", "UMAP2",
            hue="type_group", size=size_col, style=None,
            palette=type_palette,
            title="UMAP colored by Sample Type",
            out_base=outdir / f"plots/Beta_UMAP_type_{name}"
        )
        scatter_umap(
            df, "UMAP1", "UMAP2",
            hue="type_group", size=size_col, style=None,
            palette=type_palette,
            label="sample_code",
            title="UMAP colored by Sample Type",
            out_base=outdir / f"plots/Beta_UMAP_type_labelled_{name}"
        )
        if "lung_code" in df.columns:
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="type_group", size=size_col, style="lung_code",
                palette=type_palette,
                title="UMAP by Type (style = lung_code)",
                out_base=outdir / f"plots/Beta_UMAP_type_lung_{name}"
            )
        if "status" in df.columns:
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="type_group", size=size_col, style="status",
                palette=type_palette,
                title="UMAP by Type (style = status)",
                out_base=outdir / f"plots/Beta_UMAP_type_status_{name}"
            )
            scatter_umap(
                df, "UMAP1", "UMAP2",
                hue="status", size=size_col, style=None,
                palette=PALETTE_STATUS,
                title="UMAP colored by Cancer Status",
                out_base=outdir / f"plots/Beta_UMAP_status_{name}"
            )
        for flag, label in (("overall_OL", "study-wide outliers"), ("typ_grp_OL", "type-wise outliers")):
            if flag in df.columns:
                scatter_umap(
                    df, "UMAP1", "UMAP2",
                    hue="type_group", size=size_col, style=flag,
                    palette=type_palette,
                    title=f"UMAP by Type (style = {label})",
                    out_base=outdir / f"plots/Beta_UMAP_type_{flag}_{name}"
                )

# -------------------------- CLI --------------------------
def main():
    p = argparse.ArgumentParser(description="Alpha stats, PERMANOVA, and UMAP plots from precomputed distances.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--metadata", required=True, type=Path, help="Metadata TSV with at least 'sample' and 'type_group'.")
    p.add_argument("--master",   required=False, type=Path, help="Master table with counts/pass_filter (optional).")
    p.add_argument("--count-table", required=False, type=Path, help="Wide ASV count table used for patient-aware beta stats.")
    p.add_argument("--alpha",    required=False, type=Path, help="Shannon TSV (sample,Shannon).")
    p.add_argument("--bray",     required=False, type=Path, help="Bray-Curtis square distance TSV (index=sample).")
    p.add_argument("--jacc",     required=False, type=Path, help="Jaccard square distance TSV (index=sample).")
    p.add_argument("--outliers-all",  type=Path, help="Optional outliers table with columns: sample,is_outlier (overall_OL).")
    p.add_argument("--outliers-type", type=Path, help="Optional outliers table with columns: sample,is_outlier (typ_grp_OL).")
    p.add_argument("--exclude-types", default="", help="Comma-separated type_group values to exclude.")
    p.add_argument("--type-order",    default="", help="Comma-separated plotting order for type_group.")

    # mito inputs (optional)
    p.add_argument("--mito-alpha", type=Path, help="Mito Shannon TSV.")
    p.add_argument("--mito-bray",  type=Path, help="Mito Bray TSV.")
    p.add_argument("--mito-jacc",  type=Path, help="Mito Jaccard TSV.")

    # UMAP / PERMANOVA params
    p.add_argument("--neighbors", type=int, default=30, help="UMAP neighbors for precomputed distances.")
    p.add_argument("--min-dist",  type=float, default=0.01, help="UMAP min_dist for precomputed distances.")
    p.add_argument("--permutations", type=int, default=999, help="PERMANOVA permutations.")
    p.add_argument("--patient-col", default=None, help="Patient ID column in metadata for patient-aware stats.")

    # Output dirs
    p.add_argument("--outdir", type=Path, required=True, help="Output directory (micro).")
    p.add_argument("--mito-outdir", type=Path, default=None, help="Output directory for mito pass.")

    args = p.parse_args()

    # Read inputs
    meta = read_tsv(args.metadata)
    if "sample" not in meta.columns:
        raise ValueError("metadata must contain a 'sample' column.")

    master = read_tsv(args.master) if args.master else pd.DataFrame()
    count_table = read_count_table(args.count_table) if args.count_table else pd.DataFrame()
    alpha  = read_tsv(args.alpha, index_col=0) if args.alpha else pd.DataFrame()
    bray   = read_tsv(args.bray,  index_col=0) if args.bray  else pd.DataFrame()
    jacc   = read_tsv(args.jacc,  index_col=0) if args.jacc  else pd.DataFrame()

    # Attach outlier flags if provided
    if args.outliers_all and Path(args.outliers_all).exists():
        o = read_tsv(args.outliers_all)
        if {"sample", "is_outlier"}.issubset(o.columns):
            o = o.rename(columns={"is_outlier": "overall_OL"})
            meta = safe_merge(meta, o, on="sample")
    if args.outliers_type and Path(args.outliers_type).exists():
        o = read_tsv(args.outliers_type)
        if {"sample", "is_outlier"}.issubset(o.columns):
            o = o.rename(columns={"is_outlier": "typ_grp_OL"})
            meta = safe_merge(meta, o, on="sample")

    exclude_types = [t.strip() for t in args.exclude_types.split(",") if t.strip()]
    type_order    = [t.strip() for t in args.type_order.split(",") if t.strip()] if args.type_order else None

    # MICRO pass
    ensure_dir(args.outdir)
    run_one_pass(
        name="micro",
        meta=meta, master=master, count_table=count_table, alpha=alpha, bray=bray, jacc=jacc,
        outdir=args.outdir,
        exclude_types=exclude_types, type_order=type_order,
        neighbors=args.neighbors, min_dist=args.min_dist,
        permutations=args.permutations, patient_col=args.patient_col
    )

    # MITO pass (optional)
    if args.mito_alpha or args.mito_bray or args.mito_jacc:
        mito_dir = args.mito_outdir if args.mito_outdir else args.outdir.parent / "M6_downstream_analysis" / "diversity" / "mitochondrial"
        ensure_dir(mito_dir)
        mito_alpha = read_tsv(args.mito_alpha, index_col=0) if args.mito_alpha else pd.DataFrame()
        mito_bray  = read_tsv(args.mito_bray,  index_col=0) if args.mito_bray  else pd.DataFrame()
        mito_jacc  = read_tsv(args.mito_jacc,  index_col=0) if args.mito_jacc  else pd.DataFrame()

        run_one_pass(
            name="mito",
            meta=meta, master=master, count_table=count_table, alpha=mito_alpha, bray=mito_bray, jacc=mito_jacc,
            outdir=mito_dir,
            exclude_types=exclude_types, type_order=type_order,
            neighbors=args.neighbors, min_dist=args.min_dist,
            permutations=args.permutations, patient_col=args.patient_col
        )

if __name__ == "__main__":
    main()
