#!/usr/bin/env python3
"""
asv_overlap_general.py
----------------------
Generalized CLI to generate UpSet & Venn-style set intersections for ASV-like tables.

What this expects
-----------------
1) A "counts" table TSV (wide):
   - rows = features (ASV IDs, OTUs, genes, etc.)
   - columns = samples
   - values = counts/abundance
   - first column is feature ID (index)

2) A metadata TSV:
   - must contain a sample id column that matches the counts table column names
     OR you can provide a mapping column (e.g. counts columns match "orig_sample")

3) A grouping column in metadata (e.g. body site, treatment, compartment, etc.)
   - you will tell the script which metadata column defines the grouping.

4) Suites (your custom Venn/UpSet definitions):
   - You can define any number of suites via repeated --suite JSON strings
   - Each suite picks:
        * which counts table ("raw" or "final")
        * which groups to include (list of group labels found in metadata[group_col])
        * which plots/tables to generate

Outputs
-------
For each suite, script writes:
- UpSet unique membership plot (counts of unique features)
- UpSet weighted plot (abundance-weighted; stacked by group)
- Venn plot (counts; 2-3 sets via matplotlib-venn if installed; 4-6 sets via venn if installed)
- Weighted Venn plot (2-3 sets only; abundance-weighted areas with exact sum labels)
- presence table TSV (exclusive membership lists)
- sum table TSV (exclusive membership abundance sums)

Dependencies
------------
Required:
- numpy, pandas, matplotlib, seaborn, upsetplot

Optional:
- matplotlib-venn (for 2-3 set Venn)
- venn (for 4-6 set Venn)

Examples
--------
Minimal (one counts table, one suite):
python asv_overlap_general.py \
  --counts-raw /path/counts.tsv \
  --metadata /path/metadata.tsv \
  --sample-col sample \
  --group-col type_group \
  --out-dir /path/out \
  --suite '{"name":"my3","counts":"raw","groups":["A","B","C"],"plots":["upset","upset_weighted","venn"],"tables":true}'

Two counts tables + multiple suites:
python asv_overlap_general.py \
  --counts-raw /path/ASV_target.tsv \
  --counts-final /path/ASV_final.tsv \
  --metadata /path/metadata_updated.tsv \
  --sample-col sample \
  --orig-sample-col orig_sample \
  --group-col type_group \
  --out-dir /path/out \
  --formats svg,pdf \
  --suite '{"name":"raw_5","counts":"raw","groups":["G1","G2","G3","G4","G5"],"plots":["upset","upset_weighted","venn"],"tables":true}' \
  --suite '{"name":"final_3","counts":"final","groups":["G2","G3","G5"],"plots":["upset","upset_weighted","venn"],"tables":true}'

Custom colors (optional):
  --palette '{"G1":"#CC79A7","G2":"#E69F00","G3":"#6A3D9A"}'

Notes on sample name normalization
----------------------------------
By default, this script keeps counts column names as-is.
If your counts columns contain paths or suffixes, use:
  --normalize-sample-names
which mimics your prior behavior:
  take basename, then strip last two '_' chunks.

"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import seaborn as sns
from upsetplot import from_contents, UpSet

# Optional venn backends
_HAVE_MPL_VENN = False
_HAVE_VENN = False
try:
    from matplotlib_venn import venn2, venn3  # 2–3 sets
    _HAVE_MPL_VENN = True
except Exception:
    pass
try:
    from venn import venn
    _HAVE_VENN = True
except Exception:
    pass


# ---------- Matplotlib / seaborn defaults ----------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.dpi"] = 600
plt.rcParams.update({"font.size": 12, "font.family": "Source Sans Pro"})
sns.set_theme()
sns.set_style("white")


# ---------- Utility ----------
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_sample_name(s: str) -> str:
    """
    Mimic your original normalization:
      - drop paths
      - strip trailing read pair-ish suffixes by removing last two '_' chunks
    """
    base = str(s).split("/")[-1]
    parts = base.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-2])
    return base


def to_bool_series(members: Mapping[str, set], elements: Iterable[str]) -> pd.DataFrame:
    """
    Build indicator matrix (rows=element, columns=groups) telling membership.
    """
    elements = list(elements)
    cols = list(members.keys())
    data = np.zeros((len(elements), len(cols)), dtype=bool)
    for j, g in enumerate(cols):
        s = members[g]
        data[:, j] = [e in s for e in elements]
    return pd.DataFrame(data, index=elements, columns=cols)


def exclusive_sets(name_to_set: Mapping[str, set]) -> Dict[Tuple[str, ...], List[str]]:
    """
    For each non-empty combination of groups, compute exactly-there elements:
    intersection(included) minus union(excluded).
    Returns dict with tuple of names -> sorted list of element IDs.
    """
    groups = list(name_to_set.keys())
    out: Dict[Tuple[str, ...], List[str]] = {}
    for r in range(1, len(groups) + 1):
        for combo in combinations(groups, r):
            inc = set.intersection(*(name_to_set[n] for n in combo))
            exc_names = [n for n in groups if n not in combo]
            if exc_names:
                exc = set.union(*(name_to_set[n] for n in exc_names))
                members = inc - exc
            else:
                members = inc
            if members:
                out[tuple(combo)] = sorted(members)
    return out


def exclusive_value_sums(
    name_to_set: Mapping[str, set],
    per_feature_total: Mapping[str, float],
) -> Dict[Tuple[str, ...], float]:
    """
    Sum feature totals within each exclusive overlap region.
    """
    exclusive = exclusive_sets(name_to_set)
    return {
        combo: float(sum(per_feature_total.get(feature_id, 0.0) for feature_id in feature_ids))
        for combo, feature_ids in exclusive.items()
    }


def venn3_subset_values(
    names: Sequence[str],
    combo_values: Mapping[Tuple[str, ...], float],
) -> Dict[str, float]:
    """
    Map tuple-based subset keys onto matplotlib-venn's binary subset ids.
    """
    subset_ids = ("100", "010", "110", "001", "101", "011", "111")
    subsets = {subset_id: 0.0 for subset_id in subset_ids}
    for combo, value in combo_values.items():
        key = "".join("1" if name in set(combo) else "0" for name in names)
        if key != "000":
            subsets[key] = float(value)
    return subsets


def venn2_subset_values(
    names: Sequence[str],
    combo_values: Mapping[Tuple[str, ...], float],
) -> Dict[str, float]:
    """
    Map tuple-based subset keys onto matplotlib-venn's binary subset ids.
    """
    subset_ids = ("10", "01", "11")
    subsets = {subset_id: 0.0 for subset_id in subset_ids}
    for combo, value in combo_values.items():
        key = "".join("1" if name in set(combo) else "0" for name in names)
        if key != "00":
            subsets[key] = float(value)
    return subsets


def compress_venn_layout_sizes(subsets: Mapping[str, float]) -> Dict[str, float]:
    """
    Keep all non-zero regions visible without letting extreme dynamic ranges
    collapse small intersections out of the diagram.
    """
    compressed = {}
    for subset_id, value in subsets.items():
        if value <= 0:
            compressed[subset_id] = 0.0
            continue
        compressed[subset_id] = max(float(np.log10(value + 1.0)), 0.4)
    return compressed


def format_venn_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def venn_fallback_position(diagram, subset_id: str) -> Optional[Tuple[float, float]]:
    """
    Approximate a label position when matplotlib-venn omits one for a tiny
    but non-zero region.
    """
    centers = getattr(diagram, "centers", None)
    if centers is None:
        return None

    points = [np.asarray(center, dtype=float) for center in centers]
    radii = getattr(diagram, "radii", None)
    mean_radius = float(np.mean(radii)) if radii is not None else 0.25
    offset = max(mean_radius * 0.32, 0.08)

    included = [idx for idx, flag in enumerate(subset_id) if flag == "1"]
    excluded = [idx for idx, flag in enumerate(subset_id) if flag == "0"]

    if len(included) == len(points):
        return tuple(np.mean(points, axis=0))

    if len(included) == 1 and excluded:
        origin = points[included[0]]
        away = origin - np.mean([points[idx] for idx in excluded], axis=0)
    elif len(included) == 2 and len(excluded) == 1:
        origin = np.mean([points[idx] for idx in included], axis=0)
        away = origin - points[excluded[0]]
    else:
        return None

    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        return tuple(origin)
    return tuple(origin + (away / norm) * offset)


def apply_venn_labels(diagram, subset_values: Mapping[str, float], ax) -> None:
    for subset_id, actual_value in subset_values.items():
        text = format_venn_value(actual_value) if actual_value > 0 else ""
        label = diagram.get_label_by_id(subset_id)
        if label is not None:
            label.set_text(text)
            continue
        if not text:
            continue
        pos = venn_fallback_position(diagram, subset_id)
        if pos is None:
            continue
        ax.text(pos[0], pos[1], text, ha="center", va="center", fontsize=6, color="#262626")


def plot_venn2_values(
    names: Sequence[str],
    subset_values: Mapping[str, float],
    colors: Mapping[str, str],
):
    layout_sizes = compress_venn_layout_sizes(subset_values)
    diagram = venn2(
        subsets=layout_sizes,
        set_labels=tuple(names),
        set_colors=(colors[names[0]], colors[names[1]]),
        alpha=0.6,
    )
    apply_venn_labels(diagram, subset_values, plt.gca())
    return diagram


def plot_venn3_values(
    names: Sequence[str],
    subset_values: Mapping[str, float],
    colors: Mapping[str, str],
):
    """
    Render a 3-set Venn using compressed layout sizes and exact displayed labels.
    """
    layout_sizes = compress_venn_layout_sizes(subset_values)
    diagram = venn3(
        subsets=layout_sizes,
        set_labels=tuple(names),
        set_colors=(colors[names[0]], colors[names[1]], colors[names[2]]),
        alpha=0.6,
    )
    apply_venn_labels(diagram, subset_values, plt.gca())
    return diagram


# ---------- IO ----------
def read_counts_table(path: Path, normalize_cols: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    if normalize_cols:
        df.columns = [normalize_sample_name(c) for c in df.columns]
    return df


def read_metadata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def melt_counts(count_df: pd.DataFrame) -> pd.DataFrame:
    st = count_df.stack().reset_index()
    st.columns = ["ASV_ID", "sample", "count"]
    return st


# ---------- UpSet builders ----------
def build_upset_unique(group_sets: Mapping[str, set]) -> pd.Series:
    """Series with MultiIndex booleans -> counts (unique elements)."""
    return from_contents(group_sets)


def build_upset_weighted_rows(
    group_sets: Mapping[str, set],
    per_group_values: Mapping[Tuple[str, str], float],
    group_order: Sequence[str],
) -> pd.DataFrame:
    """
    Build a DataFrame suitable for UpSet(..., sum_over='count') and
    add_stacked_bars(by='group', sum_over='count').

    Rows are ASV_ID×group (only where ASV_ID in that group's set).
    Columns are membership indicators for ALL groups (booleans),
    plus: ASV_ID, group, count
    """
    all_feats = sorted(set().union(*group_sets.values())) if group_sets else []
    indicators = to_bool_series(group_sets, all_feats)  # rows=FEATURE, cols=groups

    rows = []
    for g in group_order:
        members = group_sets.get(g, set())
        for feat in members:
            row = indicators.loc[feat].to_dict()
            row["ASV_ID"] = feat
            row["group"] = g
            row["count"] = float(per_group_values.get((g, feat), 0.0))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df[[*group_order, "ASV_ID", "group", "count"]]
    return df


# ---------- Plotters ----------
def savefig_multi(fig: plt.Figure, out_path_base: Path, name: str, formats: Sequence[str]) -> None:
    for ext in formats:
        fig.savefig(out_path_base.with_name(f"{out_path_base.stem}_{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_upset_unique(
    group_sets: Mapping[str, set],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
) -> None:
    rev_grp_sets = {g: s for g, s in reversed(list(group_sets.items()))}
    data = build_upset_unique(rev_grp_sets)
    upset = UpSet(
        data,
        subset_size="count",
        element_size=None,
        show_counts=True,
        sort_categories_by="input",
        min_subset_size=0,
    )
    for g, c in colors.items():
        upset.style_categories([g], bar_facecolor=c, bar_edgecolor="black")

    fig = plt.figure(figsize=(12, 8))
    mpl.rcParams["font.size"] = 6
    upset.plot(fig=fig)
    fig.suptitle(title, y=0.98)

    handles = [Patch(facecolor=colors[g], edgecolor="black", label=g) for g in group_sets.keys()]
    fig.legend(handles=handles, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    savefig_multi(fig, out_base, "upset", formats)


def plot_upset_weighted(
    group_sets: Mapping[str, set],
    per_group_values: Mapping[Tuple[str, str], float],
    group_order: Sequence[str],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
) -> None:
    rev_grp_sets = {g: s for g, s in reversed(list(group_sets.items()))}
    df = build_upset_weighted_rows(rev_grp_sets, per_group_values, group_order)

    contents = {c: set(df.loc[df[c].astype(bool), "ASV_ID"]) for c in group_order[::-1]}
    weights = df.groupby(["ASV_ID", "group"])["count"].sum().reset_index().set_index("ASV_ID")

    weights["group"] = pd.Categorical(weights["group"], categories=list(group_order), ordered=True)

    ser = from_contents(contents, data=weights)
    upset = UpSet(
        ser,
        sum_over="count",
        subset_size="sum",
        element_size=None,
        show_counts=True,
        sort_categories_by="input",
        min_subset_size=0,
        intersection_plot_elements=0,
    )

    for g, c in colors.items():
        upset.style_categories([g], bar_facecolor=c, bar_edgecolor="black")

    stack_colors = [colors[g] for g in group_order]

    upset.add_stacked_bars(
        by="group",
        sum_over="count",
        colors=stack_colors,   # <-- LIST, not dict/callable
        title="Abundance by Group",
        elements=10,
    )

    fig = plt.figure(figsize=(12, 8))
    mpl.rcParams["font.size"] = 6
    axes = upset.plot(fig=fig)

    ax_extra = axes.get("extra0", None)
    if ax_extra is not None:
        handles, labels = ax_extra.get_legend_handles_labels()
        order = list(group_order)
        handles = [handles[labels.index(o)] for o in order if o in labels]
        labels = [o for o in order if o in labels]
        ax_extra.legend(handles, labels, title="Group", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    fig.suptitle(title, y=1.02)
    savefig_multi(fig, out_base, "upset_weighted", formats)


def plot_venn(
    group_sets: Mapping[str, set],
    colors: Mapping[str, str],
    title: str,
    out_base: Path,
    formats: Sequence[str],
    weighted_labels: Optional[Mapping[Tuple[str, ...], float]] = None,
    plot_name: str = "venn",
) -> None:
    """
    2–3 sets => matplotlib-venn (if available)
    4–6 sets => venn package (if available)
    """
    names = list(group_sets.keys())
    sets = [group_sets[n] for n in names]

    if len(names) == 2 and _HAVE_MPL_VENN:
        fig = plt.figure(figsize=(6, 6))
        if weighted_labels is None:
            exclusive = exclusive_sets(group_sets)
            subset_values = venn2_subset_values(
                names,
                {combo: float(len(features)) for combo, features in exclusive.items()},
            )
        else:
            subset_values = venn2_subset_values(names, weighted_labels)
        plot_venn2_values(names, subset_values, colors)
        plt.title(title)
        savefig_multi(fig, out_base, plot_name, formats)
        return

    if len(names) == 3 and _HAVE_MPL_VENN:
        fig = plt.figure(figsize=(6, 6))
        if weighted_labels is None:
            exclusive = exclusive_sets(group_sets)
            subset_values = venn3_subset_values(
                names,
                {combo: float(len(features)) for combo, features in exclusive.items()},
            )
            plot_venn3_values(names, subset_values, colors)
        else:
            subset_values = venn3_subset_values(names, weighted_labels)
            plot_venn3_values(names, subset_values, colors)
        plt.title(title)
        savefig_multi(fig, out_base, plot_name, formats)
        return

    if weighted_labels is not None and len(names) > 3:
        print(f"[WARN] Weighted Venn plotting is only supported for 2-3 sets; got {len(names)}. Skipping.")
        return

    if 4 <= len(names) <= 6 and _HAVE_VENN:
        labels2sets = {n: set(s) for n, s in zip(names, sets)}
        if not any(len(s) for s in labels2sets.values()):
            print("All Venn sets are empty; skipping Venn plot.")
            return

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax = venn(
            labels2sets,
            ax=ax,
            cmap=[colors[n] for n in names],
            fontsize=8,
            alpha=0.45,
        )
        ax.set_title(title)
        savefig_multi(fig, out_base, plot_name, formats)
        plt.close(fig)
        return

    print(f"[WARN] Venn plotting not available for {len(names)} sets (need matplotlib-venn or venn). Skipping.")


def write_presence_and_sums(
    name_to_set: Mapping[str, set],
    per_feature_total: Mapping[str, float],
    out_base: Path,
    fname_prefix: str,
) -> None:
    """
    Write two TSVs:
      - <prefix>_venn_presence_table.tsv : rows = (grouping label, ASV_ID)
      - <prefix>_venn_sum_table.tsv      : rows = (grouping label, Sum_count)
    """
    exc = exclusive_sets(name_to_set)

    rows = [[(" + ".join(k) if len(k) > 1 else k[0]), feat] for k, ids in exc.items() for feat in ids]
    pres = pd.DataFrame(rows, columns=["grouping", "ASV_ID"])
    pres.to_csv(out_base.with_name(f"{out_base.stem}_{fname_prefix}_venn_presence_table.tsv"), sep="\t", index=False)

    rows2 = []
    for k, ids in exc.items():
        s = float(sum(per_feature_total.get(a, 0.0) for a in ids))
        rows2.append([(" + ".join(k) if len(k) > 1 else k[0]), s])
    sums = pd.DataFrame(rows2, columns=["grouping", "Sum_count"])
    sums.to_csv(out_base.with_name(f"{out_base.stem}_{fname_prefix}_venn_sum_table.tsv"), sep="\t", index=False)


# ---------- Core suite runner ----------
def run_suite(
    suite: dict,
    long_df: pd.DataFrame,
    group_col: str,
    out_dir: Path,
    formats: Sequence[str],
    palette: Mapping[str, str],
) -> None:
    """
    suite keys (JSON):
      name: str (required)
      groups: list[str] (required) - labels from metadata[group_col]
      plots: list[str] optional, any of: ["upset","upset_weighted","venn","venn_weighted"]
      tables: bool optional (default true)
      title: str optional (default derived)
    """
    name = suite["name"]
    groups = suite["groups"]
    plots = suite.get("plots", ["upset", "upset_weighted", "venn"])
    tables = bool(suite.get("tables", True))
    title = suite.get("title", f"{name}: feature overlap by {group_col}")

    # Filter positives
    pos = long_df[long_df["count"] > 0].copy()

    # Per-feature totals and per-(group, feature) totals
    feat_total = pos.groupby("ASV_ID")["count"].sum().to_dict()
    group_feat_total = pos.groupby([group_col, "ASV_ID"])["count"].sum().to_dict()

    present_groups = [g for g in groups if g in set(pos[group_col].astype(str).unique())]
    group_sets = {g: set(pos.loc[pos[group_col].astype(str) == g, "ASV_ID"]) for g in present_groups}

    if len(group_sets) < 2:
        print(f"[WARN] Suite '{name}': need ≥2 present groups; skipping.")
        return

    # Colors: use palette if provided; otherwise pull from matplotlib cycle
    if palette:
        colors = {g: palette[g] for g in present_groups if g in palette}
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        colors = {g: cycle[i % len(cycle)] for i, g in enumerate(present_groups)} if cycle else {g: "#4C72B0" for g in present_groups}

    ensure_dir(out_dir / "plots")
    ensure_dir(out_dir / "tables")
    plot_base = out_dir / "plots" / name
    table_base = out_dir / "tables" / name

    if "upset" in plots:
        plot_upset_unique(group_sets, colors, title, plot_base, formats)

    if "upset_weighted" in plots:
        plot_upset_weighted(group_sets, group_feat_total, present_groups, colors, title, plot_base, formats)

    if "venn" in plots:
        plot_venn(group_sets, colors, title, plot_base, formats)

    if "venn_weighted" in plots:
        plot_venn(
            group_sets,
            colors,
            f"{title} (weighted)",
            plot_base,
            formats,
            weighted_labels=exclusive_value_sums(group_sets, feat_total),
            plot_name="venn_weighted",
        )

    if tables:
        write_presence_and_sums(group_sets, feat_total, table_base, name)


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="General set overlap (UpSet & Venn) plotter")

    ap.add_argument("--counts-raw", default=None, help="TSV counts table (rows=feature, cols=samples)")
    ap.add_argument("--counts-final", default=None, help="Optional second TSV counts table (same shape style)")
    ap.add_argument("--metadata", required=True, help="Metadata TSV")

    ap.add_argument("--sample-col", required=True, help="Metadata column that matches counts column names (e.g. 'sample')")
    ap.add_argument(
        "--orig-sample-col",
        default=None,
        help="Optional metadata column to map counts sample name -> metadata sample key (e.g. 'orig_sample')",
    )
    ap.add_argument("--group-col", required=True, help="Metadata column that defines your group labels")

    ap.add_argument("--out-dir", required=True, help="Output directory root")
    ap.add_argument("--formats", default="svg,pdf", help="Comma-separated: svg,pdf,png")

    ap.add_argument(
        "--normalize-sample-names",
        action="store_true",
        help="Normalize counts column names like your old script (basename, strip last 2 '_' chunks)",
    )

    ap.add_argument(
        "--suite",
        action="append",
        default=[],
        help=(
            "Suite definition as JSON string. Repeatable. "
            "Example: '{\"name\":\"s1\",\"counts\":\"raw\",\"groups\":[\"A\",\"B\"],\"plots\":[\"upset\",\"venn\"],\"tables\":true}'"
        ),
    )

    ap.add_argument(
        "--palette",
        default=None,
        help="Optional JSON dict mapping group label -> color hex (e.g. '{\"A\":\"#CC79A7\",\"B\":\"#E69F00\"}')",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]

    # Palette
    palette = {}
    if args.palette:
        palette = json.loads(args.palette)
        if not isinstance(palette, dict):
            raise ValueError("--palette must be a JSON object mapping group->color")

    # Read metadata
    md = read_metadata(Path(args.metadata))

    # Build a sample-join key:
    # - counts long has column 'sample'
    # - metadata will be joined on either sample_col or orig_sample_col (if provided)
    md_sample_key = args.orig_sample_col if args.orig_sample_col else args.sample_col
    if md_sample_key not in md.columns:
        raise ValueError(f"Metadata missing join column '{md_sample_key}' (from --orig-sample-col/--sample-col).")

    # Validate group_col
    if args.group_col not in md.columns:
        raise ValueError(f"Metadata missing group column '{args.group_col}' (from --group-col).")

    # Read counts tables (optional raw/final)
    counts_raw = None
    counts_final = None
    if args.counts_raw:
        counts_raw = read_counts_table(Path(args.counts_raw), normalize_cols=args.normalize_sample_names)
    if args.counts_final:
        counts_final = read_counts_table(Path(args.counts_final), normalize_cols=args.normalize_sample_names)

    if counts_raw is None and counts_final is None:
        raise ValueError("You must provide at least one counts table via --counts-raw and/or --counts-final.")

    # Parse suites
    if not args.suite:
        raise ValueError(
            "No suites provided. Add at least one --suite JSON string defining name, counts, groups."
        )

    suites = []
    for s in args.suite:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("--suite must be a JSON object")
        if "name" not in obj or "counts" not in obj or "groups" not in obj:
            raise ValueError("Each suite must include: name, counts ('raw'|'final'), groups (list).")
        if obj["counts"] not in {"raw", "final"}:
            raise ValueError("suite.counts must be 'raw' or 'final'")
        if not isinstance(obj["groups"], list) or not all(isinstance(x, str) for x in obj["groups"]):
            raise ValueError("suite.groups must be a list of strings")
        suites.append(obj)

    # Pre-melt counts tables and attach metadata
    def build_long(count_df: pd.DataFrame) -> pd.DataFrame:
        long_df = melt_counts(count_df)
        # Join metadata
        # counts_long.sample should match md[md_sample_key]
        long_df = long_df.merge(md, how="left", left_on="sample", right_on=md_sample_key)
        return long_df

    long_raw = build_long(counts_raw) if counts_raw is not None else None
    long_final = build_long(counts_final) if counts_final is not None else None

    # Run suites
    for suite in suites:
        which = suite["counts"]
        long_df = long_raw if which == "raw" else long_final
        if long_df is None:
            print(f"[WARN] Suite '{suite['name']}': counts='{which}' requested but table not provided; skipping.")
            continue

        suite_out = out_dir

        print(f"[INFO] Running suite '{suite['name']}' on counts='{which}'")
        run_suite(
            suite=suite,
            long_df=long_df,
            group_col=args.group_col,
            out_dir=suite_out,
            formats=formats,
            palette=palette,
        )

    print("[OK] Done.")


if __name__ == "__main__":
    main()
