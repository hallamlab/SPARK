#!/usr/bin/env python3
"""
Plot observed taxonomic analysis results.

Inputs:
- data_long: original long ASV table (for abundance boxplots)
- cancer_results: output from run_taxonomic_abundance_analysis.py
- sampletype_results: output from run_taxonomic_sample_type_analysis.py (pairwise)

Outputs (SVG + PDF):
- cancer_vs_control_boxplots_*.{svg,pdf}
- sample_type_three_group_boxplots_*.{svg,pdf}
- effect_heatmap_cancer_*.{svg,pdf}
- effect_heatmap_sample_type_*.{svg,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# Match prior plot style
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.dpi"] = 600
plt.rcParams.update({"font.size": 12})
sns.set_theme()
sns.set_style("white")

PALETTE_STATUS = {"Control": "#bdbdbd", "Cancer": "#A50026"}
PALETTE_TYPES = {"BAL": "#0072B2", "Lung Brush": "#009E73", "Oral Rinse": "#6A3D9A"}


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def canonical_case(x: str) -> str:
    return "Control" if str(x) in {"Control", "Non-Cancer"} else "Cancer"


def patient_type_taxon_rel(
    long_df: pd.DataFrame,
    tax_level: str,
    sample_col: str,
    patient_col: str,
    type_col: str,
    case_col: str,
    count_col: str,
) -> pd.DataFrame:
    agg = long_df.groupby([sample_col, tax_level], as_index=False)[count_col].sum()
    sample_tax = agg.pivot(index=sample_col, columns=tax_level, values=count_col).fillna(0)

    meta = long_df[[sample_col, patient_col, type_col, case_col]].drop_duplicates().set_index(sample_col)
    common = sample_tax.index.intersection(meta.index)
    sample_tax = sample_tax.loc[common]
    meta = meta.loc[common]

    tmp = sample_tax.copy()
    tmp[patient_col] = meta[patient_col].values
    tmp[type_col] = meta[type_col].values
    tmp[case_col] = meta[case_col].map(canonical_case).values

    taxa_cols = [c for c in tmp.columns if c not in {patient_col, type_col, case_col}]
    pt = tmp.groupby([patient_col, type_col, case_col], as_index=False)[taxa_cols].sum()

    totals = pt[taxa_cols].sum(axis=1).replace(0, 1)
    pt[taxa_cols] = pt[taxa_cols].div(totals, axis=0)

    out = pt.melt(
        id_vars=[patient_col, type_col, case_col],
        value_vars=taxa_cols,
        var_name="taxon",
        value_name="rel_abundance",
    )
    return out


def top_taxa(df: pd.DataFrame, n: int, q_col: str = "q_value") -> list[str]:
    d = df.copy()
    if q_col in d.columns:
        d = d.sort_values(q_col)
    return d["taxon"].dropna().astype(str).head(n).tolist()


def plot_cancer_boxplots(
    rel_df: pd.DataFrame,
    cancer_res: pd.DataFrame,
    tax_level: str,
    outdir: Path,
    type_col: str,
    case_col: str,
    alpha: float,
    top_n: int,
) -> None:
    sub = cancer_res[(cancer_res["tax_level"] == tax_level) & (cancer_res["q_value"].notna())]
    for st in sorted(sub["sample_type"].unique()):
        s = sub[sub["sample_type"] == st]
        sig = s[s["q_value"] <= alpha]
        chosen = top_taxa(sig if not sig.empty else s, top_n)
        if not chosen:
            continue

        d = rel_df[(rel_df[type_col] == st) & (rel_df["taxon"].isin(chosen))].copy()
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(chosen) * 0.8), 5.8))
        sns.boxplot(
            data=d,
            x="taxon",
            y="rel_abundance",
            hue=case_col,
            order=chosen,
            hue_order=["Control", "Cancer"],
            palette=PALETTE_STATUS,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=d,
            x="taxon",
            y="rel_abundance",
            hue=case_col,
            order=chosen,
            hue_order=["Control", "Cancer"],
            dodge=True,
            color="black",
            size=2.2,
            alpha=0.30,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], frameon=False, title="Status")
        ax.set_xlabel(f"{tax_level}")
        ax.set_ylabel("Relative abundance")
        ax.set_title(f"Cancer vs Control ({st}) - {tax_level}")
        ax.tick_params(axis="x", rotation=45)
        sns.despine(ax=ax)
        save_fig(fig, outdir / f"cancer_vs_control_boxplots_{tax_level}_{st.replace(' ', '_')}")


def plot_sample_type_three_group_boxplots(
    rel_df: pd.DataFrame,
    pair_res: pd.DataFrame,
    tax_level: str,
    outdir: Path,
    patient_col: str,
    type_col: str,
    alpha: float,
    top_n: int,
) -> None:
    sub = pair_res[(pair_res["tax_level"] == tax_level) & (pair_res["q_value"].notna())].copy()
    if sub.empty:
        return

    # Choose top taxa by best q across all pairwise contrasts at this level
    ranked = sub.groupby("taxon", as_index=False)["q_value"].min().sort_values("q_value")
    sig_ranked = ranked[ranked["q_value"] <= alpha]
    chosen = sig_ranked["taxon"].head(top_n).tolist()
    if not chosen:
        chosen = ranked["taxon"].head(top_n).tolist()
    if not chosen:
        return

    d = rel_df[rel_df["taxon"].isin(chosen)].copy()
    if d.empty:
        return

    # Keep only patients with all three sample types for clean paired comparability
    needed_types = {"BAL", "Oral Rinse", "Lung Brush"}
    patient_types = d.groupby(patient_col)[type_col].apply(lambda x: set(x.dropna().unique()))
    keep_patients = patient_types[patient_types.apply(lambda s: needed_types.issubset(s))].index
    d = d[d[patient_col].isin(keep_patients)]
    if d.empty:
        return

    hue_order = [x for x in ["BAL", "Oral Rinse", "Lung Brush"] if x in d[type_col].unique()]
    fig, ax = plt.subplots(figsize=(max(9, len(chosen) * 0.85), 6.0))
    sns.boxplot(
        data=d,
        x="taxon",
        y="rel_abundance",
        hue=type_col,
        order=chosen,
        hue_order=hue_order,
        palette=PALETTE_TYPES,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=d,
        x="taxon",
        y="rel_abundance",
        hue=type_col,
        order=chosen,
        hue_order=hue_order,
        dodge=True,
        color="black",
        size=2.2,
        alpha=0.25,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order)
    ax.legend(handles[:n], labels[:n], frameon=False, title="Sample type")
    ax.set_xlabel(f"{tax_level}")
    ax.set_ylabel("Relative abundance")
    ax.set_title(f"Sample Type Comparison (All 3 Types) - {tax_level}")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)
    save_fig(fig, outdir / f"sample_type_three_group_boxplots_{tax_level}")


def effect_heatmap(df: pd.DataFrame, tax_level: str, col_name: str, value_name: str, out_base: Path, top_n: int) -> None:
    sub = df[df["tax_level"] == tax_level].copy()
    if sub.empty:
        return

    # pick taxa with best min q across comparisons
    ranked = (
        sub.groupby("taxon", as_index=False)["q_value"].min().sort_values("q_value").head(top_n)["taxon"].tolist()
    )
    m = sub[sub["taxon"].isin(ranked)].pivot(index="taxon", columns=col_name, values=value_name)
    m = m.reindex(ranked)
    if m.empty:
        return

    vmax = np.nanmax(np.abs(m.values))
    vmax = 0.01 if (not np.isfinite(vmax) or vmax == 0) else vmax

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * m.shape[1] + 2), max(5, 0.35 * m.shape[0] + 2)))
    sns.heatmap(
        m,
        cmap="coolwarm",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": value_name},
        ax=ax,
    )
    ax.set_title(f"Effect Heatmap ({tax_level})")
    ax.set_xlabel(col_name)
    ax.set_ylabel(tax_level)
    save_fig(fig, out_base)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot observed taxonomic analysis results")
    p.add_argument("--data-long", required=True)
    p.add_argument("--cancer-results", required=True)
    p.add_argument("--sampletype-results", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--sample-col", default="lmp_id")
    p.add_argument("--patient-col", default="Participant_ID")
    p.add_argument("--type-col", default="type_group")
    p.add_argument("--case-col", default="Case")
    p.add_argument("--count-col", default="count")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(args.data_long, sep="\t", low_memory=False)
    cancer_res = pd.read_csv(args.cancer_results, sep="\t")
    pair_res = pd.read_csv(args.sampletype_results, sep="\t")

    for tax_level in ["Phylum", "Family"]:
        rel_df = patient_type_taxon_rel(
            long_df,
            tax_level=tax_level,
            sample_col=args.sample_col,
            patient_col=args.patient_col,
            type_col=args.type_col,
            case_col=args.case_col,
            count_col=args.count_col,
        )

        plot_cancer_boxplots(
            rel_df,
            cancer_res,
            tax_level=tax_level,
            outdir=outdir,
            type_col=args.type_col,
            case_col=args.case_col,
            alpha=args.alpha,
            top_n=args.top_n,
        )

        plot_sample_type_three_group_boxplots(
            rel_df,
            pair_res,
            tax_level=tax_level,
            outdir=outdir,
            patient_col=args.patient_col,
            type_col=args.type_col,
            alpha=args.alpha,
            top_n=args.top_n,
        )

        effect_heatmap(
            cancer_res,
            tax_level=tax_level,
            col_name="sample_type",
            value_name="delta_median",
            out_base=outdir / f"effect_heatmap_cancer_{tax_level}",
            top_n=args.top_n,
        )
        effect_heatmap(
            pair_res,
            tax_level=tax_level,
            col_name="contrast",
            value_name="median_delta_g1_minus_g2",
            out_base=outdir / f"effect_heatmap_sample_type_{tax_level}",
            top_n=args.top_n,
        )

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
