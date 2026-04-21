#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency, pearsonr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.dpi"] = 300
plt.rcParams.update({"font.size": 10})
sns.set_theme(style="white")


def die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        die(f"Missing file: {path}")
    if sep is None:
        sep = "\t" if p.suffix.lower() in {".tsv", ".tab", ".txt"} else ","
    return pd.read_csv(p, sep=sep, low_memory=False)


def parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x and x.strip()]


def parse_mapping(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    for token in str(text).split(","):
        tok = token.strip()
        if not tok or "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def parse_json_map(text: str | None) -> dict:
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception as exc:
        die(f"Failed to parse JSON mapping: {exc}")
    return obj if isinstance(obj, dict) else {}


def parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def normalize_palette_spec(value: object) -> dict[str, str]:
    """Accept either JSON object palettes or legacy comma key=value strings."""
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
    if isinstance(value, str):
        return parse_mapping(value)
    return {}


def normalize_order_spec(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return parse_csv(value)
    return []


def normalize_asv_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.split(";", n=1).str[0]


def natural_sort_key(value: object) -> tuple:
    text = str(value).strip()
    parts = re.split(r"(\d+)", text)
    out: list[object] = []
    for part in parts:
        if not part:
            continue
        out.append(int(part) if part.isdigit() else part.lower())
    return tuple(out)


def normalize_token(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if not text:
        return "NA"
    num = pd.to_numeric(text, errors="coerce")
    if pd.notna(num):
        f = float(num)
        if np.isfinite(f):
            if f.is_integer():
                return str(int(f))
            return f"{f:g}"
    return text


def build_join_key(df: pd.DataFrame, cols: Sequence[str], out_col: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"Join columns missing: {missing}")
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(normalize_token)
    out[out_col] = out[list(cols)].agg("|".join, axis=1)
    return out


def merge_biochem_assignments(
    meta: pd.DataFrame,
    biochem_df: pd.DataFrame,
    sample_col: str,
    biochem_sample_col: str,
    include_cols: Sequence[str] | None = None,
    meta_join_cols: Sequence[str] | None = None,
    biochem_join_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = meta.copy()
    out[sample_col] = out[sample_col].astype(str).str.strip()
    use_composite = bool(meta_join_cols or biochem_join_cols)
    meta_key = "__meta_join_key__"
    biochem_key = "__biochem_join_key__"

    if use_composite:
        left_cols = list(meta_join_cols or [])
        right_cols = list(biochem_join_cols or [])
        if len(left_cols) != len(right_cols):
            die("Composite biochem join requires same number of metadata and biochem columns.")
        out = build_join_key(out, left_cols, meta_key)
        bdf = build_join_key(biochem_df.copy(), right_cols, biochem_key)
        excluded = {biochem_key, "__merge_key__", *right_cols}
    else:
        if biochem_sample_col not in biochem_df.columns:
            die(f"Biochem assignments missing sample column: {biochem_sample_col}")
        out[meta_key] = out[sample_col].map(normalize_token)
        bdf = biochem_df.copy()
        bdf[biochem_sample_col] = bdf[biochem_sample_col].astype(str).str.strip()
        bdf = bdf[bdf[biochem_sample_col].ne("")]
        bdf[biochem_key] = bdf[biochem_sample_col].map(normalize_token)
        excluded = {biochem_key, "__merge_key__", biochem_sample_col}

    bdf = bdf.drop_duplicates(subset=[biochem_key], keep="first")
    use_cols = [c for c in (include_cols or [c for c in bdf.columns if c not in excluded]) if c in bdf.columns]
    if not use_cols:
        return out
    overlap_cols = [c for c in use_cols if c in out.columns and c not in {sample_col, meta_key}]
    if overlap_cols:
        # The biochem assignments table is authoritative for these group/feature columns.
        # Dropping overlaps avoids pandas _x/_y suffixes that make requested group columns disappear.
        out = out.drop(columns=overlap_cols)
    merge_df = bdf[[biochem_key] + use_cols].copy()
    merged = out.merge(merge_df, left_on=meta_key, right_on=biochem_key, how="left")
    return merged.drop(columns=[meta_key, biochem_key], errors="ignore")


def infer_biochem_feature_cols(df: pd.DataFrame, sample_col: str, excluded: Iterable[str]) -> list[str]:
    exclude = {sample_col, *excluded}
    cols: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 3:
            continue
        text = col.lower()
        if any(tok in text for tok in ["resp_", "umap", "pc", "component", "compartment", "cluster", "prob", "year", "month", "day"]):
            continue
        cols.append(col)
    return cols


def standardize_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vec = pd.to_numeric(out[col], errors="coerce")
        med = float(vec.median()) if vec.notna().any() else 0.0
        vec = vec.fillna(med)
        std = float(vec.std(ddof=0))
        if std > 0:
            out[col] = (vec - float(vec.mean())) / std
        else:
            out[col] = 0.0
    return out


def load_asv_counts(path: str) -> pd.DataFrame:
    counts = read_table(path, sep="\t")
    if "ASV_ID" in counts.columns:
        counts["ASV_ID"] = normalize_asv_id(counts["ASV_ID"])
        counts = counts.drop_duplicates(subset=["ASV_ID"]).set_index("ASV_ID")
    else:
        counts.index = normalize_asv_id(pd.Series(counts.index, index=counts.index))
        counts = counts[~counts.index.duplicated(keep="first")]
    return counts.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def build_palette(labels: Sequence[str], provided: dict[str, str], preferred_order: Sequence[str]) -> dict[str, str]:
    present = [str(x).strip() for x in labels if str(x).strip()]
    uniq: list[str] = []
    for val in present:
        if val not in uniq:
            uniq.append(val)
    ordered = [v for v in preferred_order if v in uniq]
    ordered.extend([v for v in uniq if v not in ordered])
    palette = {k: v for k, v in provided.items() if k in ordered}
    missing = [v for v in ordered if v not in palette]
    if missing:
        colors = sns.color_palette("husl", n_colors=max(len(missing), 3))
        for idx, val in enumerate(missing):
            palette[val] = mcolors.to_hex(colors[idx % len(colors)])
    return palette


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b = mcolors.to_rgb(color)
    return (r, g, b, alpha)


def compute_linkage(df: pd.DataFrame, metric: str, method: str):
    if df.shape[0] < 2:
        return None
    dist = pdist(df.to_numpy(), metric=metric)
    if not np.isfinite(dist).all():
        die("Non-finite distance encountered while computing dendrogram.")
    return linkage(dist, method=method, optimal_ordering=True)


def linkage_leaf_indices(link: np.ndarray, node_id: int | None = None) -> list[int]:
    n = link.shape[0] + 1
    if node_id is None:
        node_id = 2 * n - 2
    if node_id < n:
        return [int(node_id)]
    row = int(node_id - n)
    return linkage_leaf_indices(link, int(link[row, 0])) + linkage_leaf_indices(link, int(link[row, 1]))


def linkage_leaf_names(link: np.ndarray, labels: Sequence[str]) -> list[str]:
    return [str(labels[i]) for i in linkage_leaf_indices(link)]


def orient_linkage_to_target(link: np.ndarray, labels: Sequence[str], target_rank: dict[str, int]) -> np.ndarray:
    """Flip internal dendrogram branches to put leaves nearer their target-side ranks.

    This changes only left/right rotations of existing splits. It does not alter
    cluster membership, merge heights, or distances.
    """
    oriented = np.asarray(link, dtype=float).copy()
    n = oriented.shape[0] + 1
    fallback = len(target_rank) + 1

    def leaves(node_id: int) -> list[int]:
        if node_id < n:
            return [node_id]
        row = node_id - n
        return leaves(int(oriented[row, 0])) + leaves(int(oriented[row, 1]))

    def mean_target(indices: Sequence[int]) -> float:
        vals = [target_rank.get(str(labels[i]), fallback) for i in indices]
        return float(np.mean(vals)) if vals else float(fallback)

    for row in range(oriented.shape[0]):
        left_id = int(oriented[row, 0])
        right_id = int(oriented[row, 1])
        if mean_target(leaves(left_id)) > mean_target(leaves(right_id)):
            oriented[row, 0], oriented[row, 1] = oriented[row, 1], oriented[row, 0]
    return oriented


def untangle_linkages(
    left_link: np.ndarray,
    right_link: np.ndarray,
    left_labels: Sequence[str],
    right_labels: Sequence[str],
    passes: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    left_plot = np.asarray(left_link, dtype=float).copy()
    right_plot = np.asarray(right_link, dtype=float).copy()
    for _ in range(max(1, int(passes))):
        right_order = linkage_leaf_names(right_plot, right_labels)
        right_rank = {sample: idx for idx, sample in enumerate(right_order)}
        left_plot = orient_linkage_to_target(left_plot, left_labels, right_rank)

        left_order = linkage_leaf_names(left_plot, left_labels)
        left_rank = {sample: idx for idx, sample in enumerate(left_order)}
        right_plot = orient_linkage_to_target(right_plot, right_labels, left_rank)
    return left_plot, right_plot, linkage_leaf_names(left_plot, left_labels), linkage_leaf_names(right_plot, right_labels)


def cramers_v_from_table(table: pd.DataFrame) -> tuple[float, float, float, int]:
    if table.empty:
        return float("nan"), float("nan"), float("nan"), 0
    chi2, pvalue, _, _ = chi2_contingency(table.to_numpy())
    n = float(table.to_numpy().sum())
    if n <= 0:
        return float("nan"), float(pvalue), float(chi2), int(n)
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return float("nan"), float(pvalue), float(chi2), int(n)
    v = math.sqrt((chi2 / n) / denom)
    return float(v), float(pvalue), float(chi2), int(n)


def mantel_test(x: np.ndarray, y: np.ndarray, permutations: int = 999, seed: int = 42) -> tuple[float, float]:
    if len(x) != len(y) or len(x) == 0:
        return float("nan"), float("nan")
    obs = pearsonr(x, y).statistic
    rng = np.random.default_rng(seed)
    n = len(x)
    ge = 0
    for _ in range(permutations):
        idx = rng.permutation(n)
        perm = pearsonr(x[idx], y).statistic
        if abs(perm) >= abs(obs):
            ge += 1
    pvalue = (ge + 1.0) / (permutations + 1.0)
    return float(obs), float(pvalue)


def draw_color_strip(ax, order: list[str], labels: pd.Series, palette: dict[str, str], title: str, left: bool = True) -> None:
    ypos = {sample: 5 + 10 * idx for idx, sample in enumerate(order)}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10 * len(order))
    ax.invert_yaxis()
    for sample in order:
        y = ypos[sample] - 5
        color = palette.get(str(labels.get(sample, "")).strip(), "#d9d9d9")
        ax.add_patch(Rectangle((0, y), 1, 10, facecolor=color, edgecolor="white", linewidth=0.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, pad=6)


def draw_ribbons(ax, left_order: list[str], right_order: list[str], ribbon_colors: dict[str, tuple], line_width: float = 0.7) -> None:
    left_pos = {sample: 5 + 10 * idx for idx, sample in enumerate(left_order)}
    right_pos = {sample: 5 + 10 * idx for idx, sample in enumerate(right_order)}
    common = [s for s in left_order if s in right_pos]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10 * max(len(left_order), len(right_order)))
    ax.invert_yaxis()
    ax.axis("off")
    for sample in common:
        ax.plot([0, 1], [left_pos[sample], right_pos[sample]], color=ribbon_colors.get(sample, (0.7, 0.7, 0.7, 0.15)), linewidth=line_width, solid_capstyle="round")


def legend_from_palette(ax, palette: dict[str, str], title: str, max_cols: int = 1):
    handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    if not handles:
        return None
    ncol = max_cols if len(handles) > 14 else 1
    return ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), title=title, frameon=False, ncol=ncol)


def save_plot(fig, out_prefix: Path, extra_artists: list) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out_prefix}.{ext}", bbox_inches="tight", bbox_extra_artists=extra_artists)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot paired biochem-vs-ASV dendrogram ribbon comparisons.")
    ap.add_argument("--metadata", required=True, help="Metadata TSV with sample IDs.")
    ap.add_argument("--sample-top-modules", required=True, help="sample_top_modules.tsv")
    ap.add_argument("--asv-counts", required=True, help="ASV count table used for SPIEC-EASI")
    ap.add_argument("--biochem-assignments", required=True, help="Biochem assignments CSV/TSV")
    ap.add_argument("--sample-col", default="sampleID")
    ap.add_argument("--module-col", default="top_module")
    ap.add_argument("--group-cols", required=True, help="Comma-separated biochem grouping columns to compare to modules")
    ap.add_argument("--biochem-sample-col", default="cruise_year_month_depth")
    ap.add_argument("--biochem-feature-cols", default="", help="Optional comma-separated numeric biochem columns for left dendrogram")
    ap.add_argument("--meta-join-cols", default="", help="Optional metadata join columns")
    ap.add_argument("--biochem-join-cols", default="", help="Optional biochem join columns")
    ap.add_argument("--group-palette-map-json", default="{}")
    ap.add_argument("--group-order-map-json", default="{}")
    ap.add_argument("--biochem-metric", default="euclidean")
    ap.add_argument("--biochem-linkage", default="average")
    ap.add_argument("--asv-metric", default="euclidean")
    ap.add_argument("--asv-linkage", default="average")
    ap.add_argument("--untangle", default="true", help="Rotate valid dendrogram branches to reduce ribbon crossings.")
    ap.add_argument("--untangle-passes", type=int, default=4)
    ap.add_argument("--mantel-permutations", type=int, default=999)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    untangle_enabled = parse_bool(args.untangle, default=True)

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_table(args.metadata, sep="\t")
    biochem_df = read_table(args.biochem_assignments)
    top_modules = read_table(args.sample_top_modules, sep="\t")
    counts = load_asv_counts(args.asv_counts)

    if args.sample_col not in metadata.columns:
        die(f"Metadata missing sample column: {args.sample_col}")
    if args.sample_col not in top_modules.columns or args.module_col not in top_modules.columns:
        die(f"sample_top_modules missing {args.sample_col}/{args.module_col}")

    group_cols = parse_csv(args.group_cols)
    meta_join_cols = parse_csv(args.meta_join_cols)
    biochem_join_cols = parse_csv(args.biochem_join_cols)
    palette_map = parse_json_map(args.group_palette_map_json)
    order_map = parse_json_map(args.group_order_map_json)

    metadata[args.sample_col] = metadata[args.sample_col].astype(str).str.strip()
    top_modules[args.sample_col] = top_modules[args.sample_col].astype(str).str.strip()
    top_modules[args.module_col] = top_modules[args.module_col].astype(str).str.strip()

    include_cols = None
    if args.biochem_feature_cols:
        include_cols = list(dict.fromkeys(parse_csv(args.biochem_feature_cols) + group_cols))
    merged = merge_biochem_assignments(
        metadata,
        biochem_df,
        args.sample_col,
        args.biochem_sample_col,
        include_cols=include_cols,
        meta_join_cols=meta_join_cols,
        biochem_join_cols=biochem_join_cols,
    )
    merged = merged.merge(top_modules[[args.sample_col, args.module_col]], on=args.sample_col, how="inner")
    merged = merged[merged[args.sample_col].isin(counts.columns)].copy()
    merged = merged.drop_duplicates(subset=[args.sample_col]).set_index(args.sample_col, drop=False)

    feature_cols = parse_csv(args.biochem_feature_cols)
    if not feature_cols:
        feature_cols = infer_biochem_feature_cols(merged, args.sample_col, excluded=group_cols + [args.module_col])
    else:
        missing_features = [c for c in feature_cols if c not in merged.columns]
        if missing_features:
            print(f"[WARN] Skipping missing biochem feature columns: {','.join(missing_features)}")
        feature_cols = [c for c in feature_cols if c in merged.columns]
    if not feature_cols:
        die("No usable biochem feature columns found for dendrogram.")

    counts = counts.loc[:, [c for c in counts.columns if c in merged.index]]
    rel = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    asv_matrix_full = np.sqrt(rel.transpose())

    module_labels = merged[args.module_col].fillna("").astype(str)
    module_palette = build_palette(module_labels.tolist(), {}, sorted(module_labels.unique().tolist(), key=natural_sort_key))

    summary_rows = []
    stats_rows = []
    for group_col in group_cols:
        if group_col not in merged.columns:
            print(f"[WARN] Skipping missing group column: {group_col}")
            continue
        sub = merged[[args.sample_col, group_col, args.module_col] + feature_cols].copy()
        sub[group_col] = sub[group_col].fillna("").astype(str).str.strip()
        sub = sub[sub[group_col].ne("") & sub[args.module_col].ne("")].copy()
        if len(sub) < 4:
            print(f"[WARN] Skipping {group_col}: not enough overlapping samples ({len(sub)})")
            continue

        sample_ids = [s for s in sub[args.sample_col].tolist() if s in asv_matrix_full.index]
        sub = sub.set_index(args.sample_col).loc[sample_ids].copy()
        if len(sub) < 4:
            print(f"[WARN] Skipping {group_col}: not enough samples after ASV overlap ({len(sub)})")
            continue

        biochem_matrix = standardize_rows(sub[feature_cols])
        asv_matrix = asv_matrix_full.loc[sample_ids]
        left_link = compute_linkage(biochem_matrix, metric=args.biochem_metric, method=args.biochem_linkage)
        right_link = compute_linkage(asv_matrix, metric=args.asv_metric, method=args.asv_linkage)
        if left_link is None or right_link is None:
            continue
        left_plot_link = left_link
        right_plot_link = right_link
        if untangle_enabled:
            left_plot_link, right_plot_link, _, _ = untangle_linkages(
                left_link,
                right_link,
                biochem_matrix.index.tolist(),
                asv_matrix.index.tolist(),
                passes=args.untangle_passes,
            )

        left_dist = pdist(biochem_matrix.to_numpy(), metric=args.biochem_metric)
        right_dist = pdist(asv_matrix.to_numpy(), metric=args.asv_metric)

        group_values = sub[group_col].astype(str)
        group_palette = build_palette(
            group_values.tolist(),
            normalize_palette_spec(palette_map.get(group_col, {})),
            normalize_order_spec(order_map.get(group_col, [])),
        )
        ribbon_colors = {
            sample: rgba(group_palette.get(group_values.loc[sample], "#bdbdbd"), 0.22)
            for sample in sample_ids
        }

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(1, 5, width_ratios=[3.2, 0.25, 2.1, 0.25, 3.2], wspace=0.02)
        ax_left = fig.add_subplot(gs[0, 0])
        ax_left_strip = fig.add_subplot(gs[0, 1], sharey=ax_left)
        ax_mid = fig.add_subplot(gs[0, 2])
        ax_right_strip = fig.add_subplot(gs[0, 3], sharey=ax_left)
        ax_right = fig.add_subplot(gs[0, 4], sharey=ax_left)

        left_den = dendrogram(
            left_plot_link,
            labels=biochem_matrix.index.tolist(),
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#4a4a4a",
            ax=ax_left,
        )
        right_den = dendrogram(
            right_plot_link,
            labels=asv_matrix.index.tolist(),
            orientation="right",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#4a4a4a",
            ax=ax_right,
        )
        left_order = [biochem_matrix.index[i] for i in left_den["leaves"]]
        right_order = [asv_matrix.index[i] for i in right_den["leaves"]]

        draw_color_strip(ax_left_strip, left_order, group_values, group_palette, title=group_col)
        draw_color_strip(ax_right_strip, right_order, sub[args.module_col].astype(str), module_palette, title="ASV Module", left=False)
        draw_ribbons(ax_mid, left_order, right_order, ribbon_colors)

        for ax in (ax_left, ax_right):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax_left.set_title("Biochem Dendrogram", fontsize=11)
        ax_right.set_title("ASV Abundance Dendrogram", fontsize=11)
        ax_mid.set_title("Sample Links", fontsize=11)
        fig.suptitle(f"{group_col} vs ASV Modules", fontsize=14, y=0.98)

        group_legend = legend_from_palette(ax_right, group_palette, f"{group_col} groups", max_cols=2)
        module_legend = legend_from_palette(ax_right, module_palette, "Soft-assigned modules", max_cols=2)
        extras = [art for art in [group_legend, module_legend] if art is not None]
        if group_legend and module_legend:
            ax_right.add_artist(group_legend)
            module_legend.set_bbox_to_anchor((1.02, 0.45))

        out_prefix = plots_dir / f"sample_agreement_{re.sub(r'[^0-9A-Za-z._-]+', '_', group_col).strip('_').lower()}"
        save_plot(fig, out_prefix, extras)
        plt.close(fig)

        order_df = pd.DataFrame({
            args.sample_col: sorted(set(left_order).union(right_order), key=natural_sort_key),
        })
        order_df["left_rank"] = order_df[args.sample_col].map({s: i + 1 for i, s in enumerate(left_order)})
        order_df["right_rank"] = order_df[args.sample_col].map({s: i + 1 for i, s in enumerate(right_order)})
        order_df[group_col] = order_df[args.sample_col].map(group_values)
        order_df[args.module_col] = order_df[args.sample_col].map(sub[args.module_col])
        order_df.to_csv(tables_dir / f"sample_agreement_order_{re.sub(r'[^0-9A-Za-z._-]+', '_', group_col).strip('_').lower()}.tsv", sep="\t", index=False)

        ctab = pd.crosstab(sub[group_col].astype(str), sub[args.module_col].astype(str))
        ctab.to_csv(tables_dir / f"sample_agreement_crosstab_{re.sub(r'[^0-9A-Za-z._-]+', '_', group_col).strip('_').lower()}.tsv", sep="\t")
        ctab_row = ctab.div(ctab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        ctab_row.to_csv(tables_dir / f"sample_agreement_crosstab_rowfrac_{re.sub(r'[^0-9A-Za-z._-]+', '_', group_col).strip('_').lower()}.tsv", sep="\t")

        label_left = sub[group_col].astype(str).to_numpy()
        label_right = sub[args.module_col].astype(str).to_numpy()
        ari = adjusted_rand_score(label_left, label_right)
        nmi = normalized_mutual_info_score(label_left, label_right)
        ami = adjusted_mutual_info_score(label_left, label_right)
        cramers_v, chi2_pvalue, chi2_stat, chi2_n = cramers_v_from_table(ctab)
        mantel_r, mantel_p = mantel_test(left_dist, right_dist, permutations=args.mantel_permutations, seed=args.random_seed)
        cophen_left = np.asarray(pdist(biochem_matrix.to_numpy(), metric=args.biochem_metric), dtype=float)
        cophen_right = np.asarray(pdist(asv_matrix.to_numpy(), metric=args.asv_metric), dtype=float)
        cophenetic_r = pearsonr(cophen_left, cophen_right).statistic if len(cophen_left) and len(cophen_right) else float("nan")

        stats_rows.append({
            "group_col": group_col,
            "n_samples": int(len(sub)),
            "n_group_levels": int(sub[group_col].nunique()),
            "n_modules": int(sub[args.module_col].nunique()),
            "ari": float(ari),
            "nmi": float(nmi),
            "ami": float(ami),
            "cramers_v": float(cramers_v),
            "chi2_stat": float(chi2_stat),
            "chi2_pvalue": float(chi2_pvalue),
            "chi2_n": int(chi2_n),
            "mantel_r": float(mantel_r),
            "mantel_pvalue": float(mantel_p),
            "biochem_metric": args.biochem_metric,
            "asv_metric": args.asv_metric,
            "biochem_linkage": args.biochem_linkage,
            "asv_linkage": args.asv_linkage,
            "untangle": bool(untangle_enabled),
            "untangle_passes": int(args.untangle_passes),
            "mantel_permutations": int(args.mantel_permutations),
        })

        summary_rows.append({
            "group_col": group_col,
            "n_samples": int(len(sub)),
            "n_group_levels": int(sub[group_col].nunique()),
            "n_modules": int(sub[args.module_col].nunique()),
            "biochem_features": ",".join(feature_cols),
            "output_prefix": out_prefix.name,
            "untangle": bool(untangle_enabled),
            "untangle_passes": int(args.untangle_passes),
            "ari": float(ari),
            "nmi": float(nmi),
            "ami": float(ami),
            "cramers_v": float(cramers_v),
            "mantel_r": float(mantel_r),
            "mantel_pvalue": float(mantel_p),
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(tables_dir / "sample_agreement_summary.tsv", sep="\t", index=False)
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(tables_dir / "sample_agreement_stats.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
