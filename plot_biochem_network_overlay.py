#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.spatial import ConvexHull
    _HAS_HULL = True
except Exception:
    _HAS_HULL = False

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["savefig.dpi"] = 300
plt.rcParams.update({"font.size": 11})
sns.set_theme(style="white")
NOT_FOCUS_COLOR = "#D9D9D9"
MIXED_MAG_TAXONOMY = "Mixed MAG phyla"
UNKNOWN_MAG_TAXONOMY = "Unclassified MAG phylum"
MIXED_MAG_FAMILY = "Mixed MAG families"
UNKNOWN_MAG_FAMILY = "Unclassified MAG family"
UNKNOWN_MAG_MIMAG = "Unclassified MIMAG tier"
MIMAG_TIER_ORDER = ["low", "medium", "high"]
ALLOWED_MAG_MIMAG_TIERS = {"medium", "high"}
BETWEENNESS_THRESHOLD = 0.05
BETWEENNESS_HIGH_LABEL = f"Betweenness >= {BETWEENNESS_THRESHOLD:.2f}"
BETWEENNESS_LOW_LABEL = f"Betweenness < {BETWEENNESS_THRESHOLD:.2f}"
BETWEENNESS_PALETTE = {
    BETWEENNESS_HIGH_LABEL: "#4a4a4a",
    BETWEENNESS_LOW_LABEL: "#d0d0d0",
}
MIMAG_TIER_PALETTE = {
    "low": "#bdbdbd",
    "medium": "#7a7a7a",
    "high": "#3a3a3a",
    UNKNOWN_MAG_MIMAG: "#d9d9d9",
}
BIOCHEM_COLOR_MAP = {
    "Oxygen": "black",
    "Nitrogen": "#E7298A",
    "Nitrate": "#1B9E77",
    "Nitrite": "#66A61E",
    "Nitrous Oxide": "#0C5196",
    "Ammonium": "#7570B3",
    "Hydrogen Sulfide": "#D95F02",
    "Dimethyl Sulfide": "#E6AB02",
    "Methane": "violet",
    "Fe": "red",
    "Fluorescence": "limegreen",
    "Temperature": "gray",
    "Salinity": "darkviolet",
    "Phosphate": "brown",
    "Silicate": "peru",
    "Density": "tan",
}
SAMPLE_BUBBLE_Q_LOW = 0.01
SAMPLE_BUBBLE_Q_HIGH = 0.99
SAMPLE_BUBBLE_SIZE = 14.0


def die(msg: str) -> None:
    raise SystemExit(f"[ERROR] {msg}")


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        die(f"Missing file: {path}")
    if sep is None:
        sep = "\t" if path.endswith((".tsv", ".tab")) else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def normalize_asv_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.split(";", n=1).str[0]


def parse_mapping(text: str) -> dict[str, str]:
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


def parse_csv_list(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x and x.strip()]


def natural_module_sort_key(label: str) -> tuple:
    text = str(label).strip()
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def normalize_combo(label: str) -> str:
    parts = [p.strip() for p in str(label).split("+") if p and str(p).strip()]
    uniq: list[str] = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return "+".join(uniq)


def normalize_mimag_tier(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"med": "medium"})
    )


def get_betweenness_series(df: pd.DataFrame) -> pd.Series:
    if "Betweenness_norm" in df.columns:
        return pd.to_numeric(df["Betweenness_norm"], errors="coerce").fillna(0.0)
    if "Betweenness" in df.columns:
        return pd.to_numeric(df["Betweenness"], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def infer_group_name_from_path(path: str) -> str:
    stem = Path(path).name
    stem = re.sub(r"_indicator_species(_DULEG)?_summary\.tsv$", "", stem)
    return stem


def isa_summary_is_duleg(path: str) -> bool:
    return bool(re.search(r"_indicator_species_DULEG_summary\.tsv$", Path(path).name))


def collect_isa_summary_paths(isa_group_cols: list[str], summary_mode: str = "auto") -> dict[str, str]:
    variant_by_name: dict[str, dict[bool, str]] = {}
    for path in sorted(Path(".").glob("*_indicator_species*_summary.tsv")):
        name = infer_group_name_from_path(str(path))
        if name:
            variant_by_name.setdefault(name, {})[isa_summary_is_duleg(str(path))] = str(path.resolve())

    summary_by_name: dict[str, str] = {}
    prefer_duleg = True if summary_mode == "duleg" else False if summary_mode == "default" else None
    for name, variants in variant_by_name.items():
        if prefer_duleg is True:
            chosen = variants.get(True) or variants.get(False)
        elif prefer_duleg is False:
            chosen = variants.get(False) or variants.get(True)
        else:
            chosen = variants.get(False) or variants.get(True)
        if chosen:
            summary_by_name[name] = chosen

    ordered = [name for name in isa_group_cols if name in summary_by_name]
    for name in sorted(summary_by_name):
        if name not in ordered:
            ordered.append(name)
    return {name: summary_by_name[name] for name in ordered}


def scale_arrow(x: float, y: float, max_radius: float) -> tuple[float, float]:
    norm = math.sqrt((x * x) + (y * y))
    if norm <= 0:
        return 0.0, 0.0
    factor = max_radius / norm
    return x * factor, y * factor


def weighted_projection(sample_coords: pd.DataFrame, weights: pd.Series) -> tuple[float, float] | None:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    keep = w > 0
    if int(keep.sum()) == 0:
        return None
    sub_coords = sample_coords.loc[keep]
    sub_w = w.loc[keep]
    total = float(sub_w.sum())
    if total <= 0:
        return None
    x = float((sub_coords.iloc[:, 0] * sub_w).sum() / total)
    y = float((sub_coords.iloc[:, 1] * sub_w).sum() / total)
    return x, y


def build_palette(labels: list[str]) -> dict[str, str]:
    if not labels:
        return {}
    colors = sns.color_palette("husl", n_colors=max(3, len(labels)))
    return {label: mcolors.to_hex(colors[i % len(colors)]) for i, label in enumerate(labels)}


def infer_index_map_from_summary(summary_df: pd.DataFrame) -> dict[int, str]:
    s_cols = [str(c).strip() for c in summary_df.columns if str(c).strip().startswith("s.")]
    if s_cols:
        groups = [c.split("s.", 1)[1].strip() for c in s_cols if c.split("s.", 1)[1].strip()]
        mapping: dict[int, str] = {}
        n = len(groups)
        for i in range(1, (1 << n)):
            members = [groups[b] for b in range(n) if (i >> b) & 1]
            if members:
                mapping[i] = "+".join(members) if len(members) > 1 else members[0]
        if mapping:
            return mapping
    b_cols = [str(c).strip() for c in summary_df.columns if str(c).strip().endswith(".B")]
    groups = [c[:-2].strip() for c in b_cols if c[:-2].strip()]
    return {i + 1: g for i, g in enumerate(groups)}


def truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    norm = series.astype(str).str.strip().str.lower()
    return norm.isin({"1", "true", "t", "yes", "y"})


def labels_from_sign_columns(df: pd.DataFrame) -> pd.Series | None:
    s_cols = [str(c).strip() for c in df.columns if str(c).strip().startswith("s.")]
    if not s_cols:
        return None

    def _label(row: pd.Series) -> str:
        labels = []
        for col in s_cols:
            val = row.get(col)
            try:
                keep = float(val) > 0
            except Exception:
                keep = str(val).strip().lower() in {"1", "true", "t", "yes", "y"}
            if keep:
                labels.append(col.split("s.", 1)[1].strip())
        return normalize_combo("+".join(labels)) if labels else ""

    return df.apply(_label, axis=1)


def load_isa_overlay(summary_path: str | None) -> pd.DataFrame:
    if not summary_path or not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
        return pd.DataFrame(columns=["ASV_ID", "isa_label", "isa_strength", "isa_significant"])
    try:
        df = read_table(summary_path, sep="\t")
    except Exception:
        return pd.DataFrame(columns=["ASV_ID", "isa_label", "isa_strength", "isa_significant"])
    if "ASV" not in df.columns:
        return pd.DataFrame(columns=["ASV_ID", "isa_label", "isa_strength", "isa_significant"])
    out = df.rename(columns={"ASV": "ASV_ID"}).copy()
    out["ASV_ID"] = normalize_asv_id(out["ASV_ID"])
    b_cols = [c for c in out.columns if str(c).endswith(".B")]
    sign_labels = labels_from_sign_columns(out)
    if sign_labels is not None:
        out["isa_label"] = sign_labels
    else:
        index_map = infer_index_map_from_summary(out)
        out["isa_label"] = out["index"].map(index_map).fillna("")
    if b_cols:
        tmp = out[b_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["isa_strength"] = tmp.max(axis=1)
    else:
        out["isa_strength"] = pd.to_numeric(out.get("stat", 0.0), errors="coerce").fillna(0.0)
    out["isa_significant"] = out.get("significant", False)
    out["isa_significant"] = truthy_series(out["isa_significant"])
    out["isa_label"] = out["isa_label"].astype(str).map(normalize_combo)
    out.loc[~out["isa_significant"], "isa_label"] = ""
    return out[["ASV_ID", "isa_label", "isa_strength", "isa_significant"]].drop_duplicates(subset=["ASV_ID"])


def extract_rank(taxon: str, prefix: str) -> str:
    if pd.isna(taxon):
        return ""
    for part in str(taxon).split(";"):
        part = part.strip()
        if part.startswith(prefix):
            val = part.split("__", 1)[-1].strip()
            return val if val else ""
    return ""


def aggregate_mag_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ASV_ID", "mag_taxonomy_label", "mag_family_label", "mag_mimag_tier_label"])
    if "mag_phylum" not in df.columns:
        df = df.copy()
        df["mag_phylum"] = pd.NA
    if "mag_family" not in df.columns:
        df = df.copy()
        df["mag_family"] = pd.NA
    if "mag_mimag_tier" not in df.columns:
        df = df.copy()
        df["mag_mimag_tier"] = pd.NA
    df = df.copy()
    df["mag_mimag_tier"] = normalize_mimag_tier(df["mag_mimag_tier"])
    df = df[df["mag_mimag_tier"].isin(ALLOWED_MAG_MIMAG_TIERS)].copy()
    if df.empty:
        return pd.DataFrame(columns=["ASV_ID", "mag_taxonomy_label", "mag_family_label", "mag_mimag_tier_label"])

    def _agg(grp: pd.DataFrame) -> pd.Series:
        phyla = sorted({str(x).strip() for x in grp["mag_phylum"].dropna() if str(x).strip()})
        families = sorted({str(x).strip() for x in grp["mag_family"].dropna() if str(x).strip()})
        tiers = [str(x).strip().lower() for x in grp["mag_mimag_tier"].dropna() if str(x).strip()]
        if not phyla:
            phylum_label = UNKNOWN_MAG_TAXONOMY
        elif len(phyla) == 1:
            phylum_label = phyla[0]
        else:
            phylum_label = MIXED_MAG_TAXONOMY

        if not families:
            family_label = UNKNOWN_MAG_FAMILY
        elif len(families) == 1:
            family_label = families[0]
        else:
            family_label = MIXED_MAG_FAMILY

        if not tiers:
            mimag_label = UNKNOWN_MAG_MIMAG
        else:
            best_rank = max((MIMAG_TIER_ORDER.index(t) for t in tiers if t in MIMAG_TIER_ORDER), default=None)
            mimag_label = MIMAG_TIER_ORDER[best_rank] if best_rank is not None else UNKNOWN_MAG_MIMAG

        return pd.Series({
            "mag_taxonomy_label": phylum_label,
            "mag_family_label": family_label,
            "mag_mimag_tier_label": mimag_label,
        })

    return df.groupby("ASV_ID", dropna=False).apply(_agg).reset_index()


def convex_hull_polygon(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3 or not _HAS_HULL:
        return None
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return None


def place_label_at_tip(ax, feat: str, tipx: float, tipy: float, color: str, cloud_scale: float):
    sign = 1.0 if tipx >= 0 else -1.0
    x_pad = sign * 0.03 * cloud_scale
    sign_y = 1.0 if tipy >= 0 else -1.0
    y_pad = sign_y * 0.015 * cloud_scale
    text = ax.text(
        tipx * 1.06 + x_pad,
        tipy * 1.06 + y_pad,
        feat,
        fontsize=9,
        color=color,
        ha="left" if tipx >= 0 else "right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=6,
    )
    return text


def degree_marker_area(degree: float, degree_scale: float) -> float:
    deg = max(float(degree), 0.0)
    scale = max(float(degree_scale), 1.0) * 0.14
    return float(max(8.0, ((deg + 1.0) ** 1.6) * scale))


def build_degree_legend_values(observed_degrees: pd.Series) -> list[int]:
    vals = pd.to_numeric(observed_degrees, errors="coerce").dropna()
    if vals.empty:
        return [0, 5]
    max_deg = int(max(0, round(float(vals.max()))))
    step = 5 if max_deg <= 25 else 10
    legend_max = max(step, int(math.ceil(max_deg / float(step)) * step))
    return list(range(0, legend_max + step, step))


def overlay_degree_to_area(degree: float, degree_scale: float, degree_cap: float) -> float:
    # Keep the overlay smaller than the standalone network graphs, but use the
    # same nonlinear degree mapping and the same capped legend range.
    capped = min(max(0.0, float(degree)), float(degree_cap))
    return degree_marker_area(capped, max(float(degree_scale) * 0.38, 1.0))


def feature_slug(feature: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(feature).strip().lower()).strip("_")


def build_sample_biochem_handles(feature: str, values: pd.Series, vmin: float, vmax: float) -> list[Line2D]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return []
    return [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="0.75",
            markeredgecolor="#5a5a5a",
            markeredgewidth=0.35,
            markersize=math.sqrt(SAMPLE_BUBBLE_SIZE),
            alpha=0.35,
            label=f"Samples colored by {feature}",
        )
    ]


def build_sample_biochem_variants(meta: pd.DataFrame, sample_col: str, sample_cols: list[str]) -> list[dict]:
    variants: list[dict] = []
    if sample_col not in meta.columns:
        return variants
    meta_idx = meta.copy()
    meta_idx[sample_col] = meta_idx[sample_col].astype(str).str.strip()
    meta_idx = meta_idx.drop_duplicates(subset=[sample_col]).set_index(sample_col)
    for feature, base_color in BIOCHEM_COLOR_MAP.items():
        if feature not in meta_idx.columns:
            continue
        values = pd.to_numeric(meta_idx[feature], errors="coerce")
        sub = values.reindex(sample_cols).dropna()
        if sub.empty or sub.nunique() < 2:
            continue
        vmin = float(sub.quantile(0.05))
        vmax = float(sub.quantile(0.95))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        if vmax <= vmin:
            vmax = float(sub.max())
            vmin = float(sub.min())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            continue
        if vmax <= vmin:
            vmax = vmin + 1.0
        plot_df = pd.DataFrame({
            sample_col: sub.index.astype(str),
            "__sample_biochem_value": sub.values,
        })
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f"sample_{feature_slug(feature)}",
            ["#f4f4f4", "#dddddd", mcolors.to_hex(base_color)],
        )
        variants.append({
            "feature": feature,
            "slug": feature_slug(feature),
            "df": plot_df,
            "norm": norm,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
        })
    return variants


def degree_size_handles(legend_levels: list[int], degree_scale: float) -> list[Line2D]:
    if not legend_levels:
        return []
    handles: list[Line2D] = []
    degree_cap = max(legend_levels)
    for deg in legend_levels:
        area = overlay_degree_to_area(deg, degree_scale, degree_cap)
        size = math.sqrt(area)
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=size,
                label=f"Degree {deg}",
            )
        )
    return handles


def biochem_vector_legend_handles(core_vectors: pd.DataFrame, sparse_corr: pd.DataFrame) -> list[Line2D]:
    handles: list[Line2D] = []
    seen: set[tuple[str, str]] = set()

    for _, row in core_vectors.iterrows():
        feature = str(row["feature"]).strip()
        if not feature:
            continue
        key = (feature, "core")
        if key in seen:
            continue
        seen.add(key)
        color = BIOCHEM_COLOR_MAP.get(feature)
        if not color:
            continue
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.2, linestyle="-", label=f"{feature} (core)")
        )

    for _, row in sparse_corr.iterrows():
        feature = str(row["feature"]).strip()
        if not feature:
            continue
        key = (feature, "sparse")
        if key in seen:
            continue
        seen.add(key)
        color = BIOCHEM_COLOR_MAP.get(feature)
        if not color:
            continue
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.2, linestyle="--", label=f"{feature} (sparse)")
        )

    return handles


def wrap_legend_handles(handles: list, width: int = 28) -> list:
    """Wrap long legend labels so adjacent right-side legends cannot collide."""
    for handle in handles:
        label = str(handle.get_label())
        if not label or label.startswith("_"):
            continue
        handle.set_label(textwrap.fill(label, width=width, break_long_words=False, break_on_hyphens=False))
    return handles


def repel_texts(ax, text_artists, anchor_xy, *, max_iter: int = 250, pad_px: float = 2.5, step_px: float = 1.1, leader_px: float = 8.0):
    if not text_artists:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    disp_pos = []
    for text in text_artists:
        xd, yd = ax.transData.transform(text.get_position())
        disp_pos.append([float(xd), float(yd)])

    def bbox_i(i: int):
        x_data, y_data = ax.transData.inverted().transform(tuple(disp_pos[i]))
        text_artists[i].set_position((float(x_data), float(y_data)))
        return text_artists[i].get_window_extent(renderer=renderer).expanded(1.02, 1.15)

    for _ in range(int(max_iter)):
        moved = False
        bbs = [bbox_i(i) for i in range(len(text_artists))]
        for i in range(len(text_artists)):
            for j in range(i + 1, len(text_artists)):
                if not bbs[i].overlaps(bbs[j]):
                    continue
                dx = min(bbs[i].x1, bbs[j].x1) - max(bbs[i].x0, bbs[j].x0)
                dy = min(bbs[i].y1, bbs[j].y1) - max(bbs[i].y0, bbs[j].y0)
                if dx <= 0 or dy <= 0:
                    continue
                if dx < dy:
                    push = (dx / 2.0 + pad_px) * step_px
                    if bbs[i].x0 < bbs[j].x0:
                        disp_pos[i][0] -= push
                        disp_pos[j][0] += push
                    else:
                        disp_pos[i][0] += push
                        disp_pos[j][0] -= push
                else:
                    push = (dy / 2.0 + pad_px) * step_px
                    if bbs[i].y0 < bbs[j].y0:
                        disp_pos[i][1] -= push
                        disp_pos[j][1] += push
                    else:
                        disp_pos[i][1] += push
                        disp_pos[j][1] -= push
                moved = True
        if not moved:
            break

    for i, text in enumerate(text_artists):
        x_data, y_data = ax.transData.inverted().transform(tuple(disp_pos[i]))
        text.set_position((float(x_data), float(y_data)))

    for i, text in enumerate(text_artists):
        tx_d, ty_d = ax.transData.transform(text.get_position())
        ax_x, ax_y = anchor_xy[i]
        ax_d, ay_d = ax.transData.transform((ax_x, ax_y))
        dist = np.sqrt((tx_d - ax_d) ** 2 + (ty_d - ay_d) ** 2)
        if dist >= float(leader_px):
            ax.plot([ax_x, text.get_position()[0]], [ax_y, text.get_position()[1]], linewidth=0.8, color="0.7", zorder=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay SPIEC-EASI modules and MAG-linked ASVs onto the biochem PCA biplot.")
    ap.add_argument("--metadata", required=True, help="metadata_updated_micro.tsv with merged biochem PCs.")
    ap.add_argument("--asv-counts", required=True, help="ASV count table, features as rows and samples as columns.")
    ap.add_argument("--modules", required=True, help="spieceasi_modules_all.tsv")
    ap.add_argument("--node-features", required=True, help="spieceasi_node_features.csv")
    ap.add_argument("--asv-mag-pairing", required=True, help="asv2mag_pairing.tsv")
    ap.add_argument("--loadings", required=True, help="env_pca/tables/pca_loadings.csv")
    ap.add_argument("--sparse-corr", default=None, help="env_pca/tables/sparse_feature_pc_spearman.csv")
    ap.add_argument("--taxonomy", default=None, help="taxonomy TSV for phylum overlay.")
    ap.add_argument("--isa-group-cols", default="", help="Comma-separated ISA grouping columns in configured order.")
    ap.add_argument("--isa-palette-map-json", default="{}", help="JSON object mapping ISA group name -> palette string.")
    ap.add_argument("--isa-order-map-json", default="{}", help="JSON object mapping ISA group name -> ordered label list.")
    ap.add_argument("--isa-summary-mode", choices=["auto", "default", "duleg"], default="auto",
                    help="Preferred ISA summary variant when both default and DULEG summaries are staged.")
    ap.add_argument("--modes", nargs="+", default=["module"],
                    help="Point hue overlays to render on top of the base biochem/module hull plot. Supports groupN_isa modes.")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sample-col", default="sample", help="Sample column in metadata that matches ASV count columns.")
    ap.add_argument("--ordination-x", default="PC1", help="Metadata column for x-axis.")
    ap.add_argument("--ordination-y", default="PC2", help="Metadata column for y-axis.")
    ap.add_argument("--top-modules", type=int, default=8, help="Maximum number of modules to draw as hulls.")
    ap.add_argument("--top-vectors", type=int, default=10, help="Maximum number of biochem vectors to draw.")
    ap.add_argument("--module-hull-quantile", type=float, default=0.75, help="Quantile of module abundance used to select hull samples.")
    ap.add_argument("--min-samples-per-hull", type=int, default=4, help="Minimum highlighted samples needed to draw a module hull.")
    ap.add_argument("--min-asvs-per-module", type=int, default=3, help="Minimum ASVs in a module to consider it for hulls.")
    ap.add_argument("--degree-scale", type=float, default=80.0, help="Marker area scaling for Degree, matched to network plots.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    isa_group_cols = [x for x in parse_csv_list(args.isa_group_cols) if x]
    try:
        isa_palette_map = json.loads(args.isa_palette_map_json or "{}")
    except Exception:
        isa_palette_map = {}
    try:
        isa_order_map = json.loads(args.isa_order_map_json or "{}")
    except Exception:
        isa_order_map = {}

    meta = read_table(args.metadata, sep="\t")
    for col in [args.sample_col, args.ordination_x, args.ordination_y]:
        if col not in meta.columns:
            die(f"Metadata missing required column: {col}")
    meta[args.sample_col] = meta[args.sample_col].astype(str).str.strip()
    meta[args.ordination_x] = pd.to_numeric(meta[args.ordination_x], errors="coerce")
    meta[args.ordination_y] = pd.to_numeric(meta[args.ordination_y], errors="coerce")
    meta = meta.dropna(subset=[args.sample_col, args.ordination_x, args.ordination_y]).drop_duplicates(subset=[args.sample_col]).copy()
    if meta.empty:
        die("No metadata rows remain after ordination filtering.")

    counts = read_table(args.asv_counts, sep="\t")
    if "ASV_ID" not in counts.columns:
        die("ASV count table missing ASV_ID.")
    counts["ASV_ID"] = normalize_asv_id(counts["ASV_ID"])
    counts = counts.drop_duplicates(subset=["ASV_ID"]).set_index("ASV_ID")

    sample_cols = [c for c in counts.columns if str(c) in set(meta[args.sample_col])]
    if not sample_cols:
        die("No overlapping samples between ASV counts and metadata ordination.")
    counts = counts[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts = counts.loc[counts.sum(axis=1) > 0]
    rel = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

    sample_coords = meta.set_index(args.sample_col).loc[sample_cols, [args.ordination_x, args.ordination_y]]
    sample_biochem_variants = build_sample_biochem_variants(meta, args.sample_col, sample_cols)

    modules = read_table(args.modules, sep="\t")
    if "Taxon" not in modules.columns or "module_label" not in modules.columns:
        die("Module table missing Taxon/module_label.")
    modules["Taxon"] = normalize_asv_id(modules["Taxon"])
    modules["module_label"] = modules["module_label"].astype(str).str.strip()
    modules = modules.dropna(subset=["Taxon", "module_label"]).drop_duplicates(subset=["Taxon"])

    node_features = read_table(args.node_features)
    if "Taxon" not in node_features.columns:
        die("Node features table missing Taxon.")
    node_features["Taxon"] = normalize_asv_id(node_features["Taxon"])
    if "Degree" not in node_features.columns:
        node_features["Degree"] = 1.0
    node_features["Degree"] = pd.to_numeric(node_features["Degree"], errors="coerce").fillna(1.0)
    node_features["Betweenness"] = get_betweenness_series(node_features)
    node_features = node_features.drop_duplicates(subset=["Taxon"])

    pairing = read_table(args.asv_mag_pairing, sep="\t")
    if "ASV_ID" not in pairing.columns:
        die("ASV-MAG pairing table missing ASV_ID.")
    pairing["ASV_ID"] = normalize_asv_id(pairing["ASV_ID"])
    if "pairing_status" not in pairing.columns:
        pairing["pairing_status"] = np.where(pairing["genome_id"].notna(), "paired", "unpaired")
    pairing = pairing.loc[pairing["pairing_status"].astype(str).str.lower().ne("unpaired")].copy()
    if "mag_mimag_tier" in pairing.columns:
        pairing["mag_mimag_tier"] = normalize_mimag_tier(pairing["mag_mimag_tier"])
        pairing = pairing[pairing["mag_mimag_tier"].isin(ALLOWED_MAG_MIMAG_TIERS)].copy()
    else:
        pairing = pairing.iloc[0:0].copy()
    paired_asvs = sorted(set(pairing["ASV_ID"]))
    mag_taxonomy = aggregate_mag_taxonomy(pairing)

    loadings = read_table(args.loadings)
    feature_col = loadings.columns[0]
    loadings = loadings.rename(columns={feature_col: "feature"})
    if args.ordination_x not in loadings.columns or args.ordination_y not in loadings.columns:
        die(f"Loadings table missing {args.ordination_x}/{args.ordination_y}.")
    loadings["feature"] = loadings["feature"].astype(str).str.strip()
    loadings[args.ordination_x] = pd.to_numeric(loadings[args.ordination_x], errors="coerce")
    loadings[args.ordination_y] = pd.to_numeric(loadings[args.ordination_y], errors="coerce")
    loadings["arrow_strength"] = np.sqrt(loadings[args.ordination_x] ** 2 + loadings[args.ordination_y] ** 2)
    loadings = loadings.dropna(subset=[args.ordination_x, args.ordination_y]).sort_values("arrow_strength", ascending=False)
    loadings = loadings[loadings["feature"].isin(BIOCHEM_COLOR_MAP)].copy()
    core_vectors = loadings.head(12).copy()

    sparse_corr = pd.DataFrame(columns=["feature", args.ordination_x, args.ordination_y, "arrow_strength"])
    if args.sparse_corr and os.path.exists(args.sparse_corr):
        sparse_raw = read_table(args.sparse_corr)
        need_cols = {"feature", "PC", "spearman_r"}
        if need_cols.issubset(set(sparse_raw.columns)):
            sparse_raw = sparse_raw[sparse_raw["PC"].isin([args.ordination_x, args.ordination_y])].copy()
            if not sparse_raw.empty:
                sparse_corr = sparse_raw.pivot(index="feature", columns="PC", values="spearman_r").reset_index()
                for col in [args.ordination_x, args.ordination_y]:
                    if col not in sparse_corr.columns:
                        sparse_corr[col] = np.nan
                sparse_corr["feature"] = sparse_corr["feature"].astype(str).str.strip()
                sparse_corr[args.ordination_x] = pd.to_numeric(sparse_corr[args.ordination_x], errors="coerce")
                sparse_corr[args.ordination_y] = pd.to_numeric(sparse_corr[args.ordination_y], errors="coerce")
                sparse_corr["arrow_strength"] = np.sqrt(
                    sparse_corr[args.ordination_x].fillna(0.0) ** 2 +
                    sparse_corr[args.ordination_y].fillna(0.0) ** 2
                )
                sparse_corr = sparse_corr.dropna(subset=[args.ordination_x, args.ordination_y])
                sparse_corr = sparse_corr[sparse_corr["feature"].isin(BIOCHEM_COLOR_MAP)].copy()
                sparse_corr = sparse_corr.sort_values("arrow_strength", ascending=False).head(12).copy()

    taxonomy_df = pd.DataFrame(columns=["ASV_ID", "phylum"])
    if args.taxonomy and os.path.exists(args.taxonomy):
        taxonomy_df = read_table(args.taxonomy, sep="\t")
        if "Feature ID" in taxonomy_df.columns and "Taxon" in taxonomy_df.columns:
            taxonomy_df["ASV_ID"] = normalize_asv_id(taxonomy_df["Feature ID"])
            taxonomy_df["phylum"] = taxonomy_df["Taxon"].map(lambda x: extract_rank(x, "p__"))
            taxonomy_df = taxonomy_df[["ASV_ID", "phylum"]].drop_duplicates(subset=["ASV_ID"])

    summary_by_group = collect_isa_summary_paths(isa_group_cols, args.isa_summary_mode)
    isa_group_cols = [name for name in isa_group_cols if name in summary_by_group]
    if not isa_group_cols:
        isa_group_cols = list(summary_by_group.keys())
    isa_overlays: dict[str, pd.DataFrame] = {}
    for idx, group_name in enumerate(isa_group_cols, start=1):
        overlay = load_isa_overlay(summary_by_group[group_name]).rename(columns={
            "isa_label": f"group{idx}_isa_label",
            "isa_strength": f"group{idx}_isa_strength",
            "isa_significant": f"group{idx}_isa_significant",
        })
        isa_overlays[group_name] = overlay

    module_taxon = modules.merge(node_features[["Taxon", "Degree", "Betweenness"]], on="Taxon", how="left")
    module_taxon["Degree"] = pd.to_numeric(module_taxon["Degree"], errors="coerce").fillna(1.0)
    module_taxon["Betweenness"] = pd.to_numeric(module_taxon["Betweenness"], errors="coerce").fillna(0.0)
    module_taxon = module_taxon[module_taxon["Taxon"].isin(rel.index)].copy()

    module_sizes = module_taxon.groupby("module_label")["Taxon"].nunique().rename("n_module_asvs")
    module_taxon = module_taxon.merge(module_sizes, on="module_label", how="left")
    module_taxon = module_taxon[module_taxon["n_module_asvs"] >= max(1, args.min_asvs_per_module)].copy()

    module_summary_rows = []
    if not module_taxon.empty:
        module_by_label = module_taxon.groupby("module_label")["Taxon"].apply(list)
        module_sample_scores = []
        for module_label, taxa in module_by_label.items():
            module_rel = rel.loc[rel.index.intersection(taxa)]
            if module_rel.empty:
                continue
            score = module_rel.sum(axis=0)
            score_df = pd.DataFrame({
                args.sample_col: score.index.astype(str),
                "module_label": module_label,
                "module_abundance": score.values,
            })
            module_sample_scores.append(score_df)
            module_summary_rows.append({
                "module_label": module_label,
                "n_module_asvs": int(len(taxa)),
                "total_module_abundance": float(score.sum()),
            })
        module_sample_scores_df = pd.concat(module_sample_scores, ignore_index=True) if module_sample_scores else pd.DataFrame(columns=[args.sample_col, "module_label", "module_abundance"])
        module_summary_df = pd.DataFrame(module_summary_rows).sort_values(["total_module_abundance", "n_module_asvs"], ascending=[False, False])
        top_module_labels = module_summary_df["module_label"].head(max(1, args.top_modules)).tolist()
        top_module_labels = sorted(top_module_labels, key=natural_module_sort_key)
        module_palette = build_palette(top_module_labels)
    else:
        module_sample_scores_df = pd.DataFrame(columns=[args.sample_col, "module_label", "module_abundance"])
        module_summary_df = pd.DataFrame(columns=["module_label", "n_module_asvs", "total_module_abundance"])
        module_palette = {}

    all_network_rows = []
    if not module_taxon.empty:
        for _, row in module_taxon.iterrows():
            asv_id = row["Taxon"]
            proj = weighted_projection(sample_coords, rel.loc[asv_id, sample_cols])
            if proj is None:
                continue
            all_network_rows.append({
                "ASV_ID": asv_id,
                "module_label": row["module_label"],
                "Degree": float(row["Degree"]),
                "Betweenness": float(row["Betweenness"]),
                "betweenness_label": BETWEENNESS_HIGH_LABEL if float(row["Betweenness"]) >= BETWEENNESS_THRESHOLD else BETWEENNESS_LOW_LABEL,
                args.ordination_x: proj[0],
                args.ordination_y: proj[1],
            })
    if all_network_rows:
        all_network_df = pd.DataFrame(all_network_rows).drop_duplicates(subset=["ASV_ID"]).sort_values("Degree", ascending=False)
    else:
        all_network_df = pd.DataFrame(columns=["ASV_ID", "module_label", "Degree", "Betweenness", "betweenness_label", args.ordination_x, args.ordination_y])

    mag_df = all_network_df[all_network_df["ASV_ID"].isin(paired_asvs)].copy()
    non_mag_df = all_network_df[~all_network_df["ASV_ID"].isin(paired_asvs)].copy()
    if not mag_df.empty:
        mag_df = mag_df.merge(mag_taxonomy, on="ASV_ID", how="left")
        if not taxonomy_df.empty:
            mag_df = mag_df.merge(taxonomy_df, on="ASV_ID", how="left")
        for overlay in isa_overlays.values():
            if not overlay.empty:
                mag_df = mag_df.merge(overlay, on="ASV_ID", how="left")
    else:
        mag_df = pd.DataFrame(columns=[
            "ASV_ID", "module_label", "Degree", "Betweenness", "betweenness_label", args.ordination_x, args.ordination_y,
            "mag_taxonomy_label", "mag_family_label", "phylum",
        ])

    module_sample_scores_df.to_csv(tables_dir / "module_sample_scores.tsv", sep="\t", index=False)
    module_summary_df.to_csv(tables_dir / "module_summary.tsv", sep="\t", index=False)
    pd.DataFrame(columns=["module_label", args.sample_col, "module_abundance", args.ordination_x, args.ordination_y]).to_csv(
        tables_dir / "module_hull_samples.tsv", sep="\t", index=False
    )
    mag_df.to_csv(tables_dir / "mag_linked_network_asv_projection.tsv", sep="\t", index=False)

    def ordered_labels(series: pd.Series, preferred: list[str]) -> list[str]:
        vals = [str(x).strip() for x in series.dropna().tolist() if str(x).strip()]
        uniq: list[str] = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        out = [v for v in preferred if v in uniq]
        out.extend([v for v in uniq if v not in out])
        return out

    overlay_specs = {
        "module": {
            "column": "module_label",
            "title": "Module",
            "slug": "module",
            "palette": dict(module_palette),
            "order": list(module_palette.keys()),
            "signif_column": None,
        },
        "mag_pair": {
            "column": "mag_mimag_tier_label",
            "title": "MAG MIMAG Tier",
            "slug": "mag_pair",
            "palette": dict(MIMAG_TIER_PALETTE),
            "order": list(MIMAG_TIER_ORDER) + [UNKNOWN_MAG_MIMAG],
            "signif_column": None,
        },
        "mag_pair_tax": {
            "column": "mag_taxonomy_label",
            "title": "MAG Phylum",
            "slug": "mag_pair_tax",
            "palette": {},
            "order": [],
            "signif_column": None,
        },
        "mag_pair_phylum": {
            "column": "mag_taxonomy_label",
            "title": "MAG Phylum",
            "slug": "mag_pair_phylum",
            "palette": {},
            "order": [],
            "signif_column": None,
        },
        "mag_pair_family": {
            "column": "mag_family_label",
            "title": "MAG Family",
            "slug": "mag_pair_family",
            "palette": {},
            "order": [],
            "signif_column": None,
        },
        "phylum": {
            "column": "phylum",
            "title": "ASV Phylum",
            "slug": "phylum",
            "palette": {},
            "order": [],
            "signif_column": None,
        },
        "betweenness": {
            "column": "betweenness_label",
            "title": "Betweenness",
            "slug": "betweenness",
            "palette": dict(BETWEENNESS_PALETTE),
            "order": [BETWEENNESS_HIGH_LABEL, BETWEENNESS_LOW_LABEL],
            "signif_column": None,
        },
    }
    for idx, group_name in enumerate(isa_group_cols, start=1):
        overlay_specs[f"group{idx}_isa"] = {
            "column": f"group{idx}_isa_label",
            "title": group_name,
            "slug": re.sub(r"[^0-9A-Za-z._-]+", "_", f"{group_name}_isa").strip("_"),
            "palette": parse_mapping(str(isa_palette_map.get(group_name, ""))),
            "order": [str(x).strip() for x in (isa_order_map.get(group_name, []) or []) if str(x).strip()],
            "signif_column": f"group{idx}_isa_significant",
        }

    if not mag_df.empty:
        mag_df["__mag_pair"] = "MAG-linked"

    all_network_df.to_csv(tables_dir / "all_network_asv_projection.tsv", sep="\t", index=False)
    non_mag_df.to_csv(tables_dir / "non_mag_network_asv_projection.tsv", sep="\t", index=False)

    x_span = float(sample_coords[args.ordination_x].quantile(0.99) - sample_coords[args.ordination_x].quantile(0.01))
    y_span = float(sample_coords[args.ordination_y].quantile(0.99) - sample_coords[args.ordination_y].quantile(0.01))
    cloud_scale = 0.35 * float(min(x_span, y_span)) if np.isfinite(x_span) and np.isfinite(y_span) else 1.0
    if cloud_scale <= 0:
        cloud_scale = 1.0
    max_core = float(core_vectors["arrow_strength"].max()) if not core_vectors.empty else float("nan")
    max_sparse = float(sparse_corr["arrow_strength"].max()) if not sparse_corr.empty else float("nan")
    denom = np.nanmax([max_core, max_sparse])
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    arrow_scale = cloud_scale / denom

    unknown_modes = [mode for mode in args.modes if mode not in overlay_specs]
    if unknown_modes:
        die(f"Unsupported biochem overlay modes: {', '.join(unknown_modes)}")

    for mode in args.modes:
        spec = overlay_specs[mode]
        order: list[str] = []
        palette: dict[str, str] = {}
        point_df = pd.DataFrame()
        if not mag_df.empty:
            point_df = mag_df.copy()
            color_series = point_df.get(spec["column"], pd.Series(index=point_df.index, dtype=object)).fillna("").astype(str).str.strip()
            if spec["signif_column"] and spec["signif_column"] in point_df.columns:
                signif = point_df[spec["signif_column"]].fillna(False).astype(bool)
                color_series = color_series.where(signif, "")
            order = ordered_labels(color_series, spec["order"])
            if mode == "module":
                order = sorted(order, key=natural_module_sort_key)
            palette = dict(spec["palette"])
            missing_labels = [lab for lab in order if lab and lab not in palette]
            if missing_labels:
                palette.update(build_palette(missing_labels))
            point_df["__overlay_label"] = color_series
            point_df["__overlay_color"] = [palette.get(label, NOT_FOCUS_COLOR) if label else NOT_FOCUS_COLOR for label in color_series]
            point_df["__overlay_degree"] = pd.to_numeric(point_df["Degree"], errors="coerce").fillna(0.0).clip(lower=0.0)
            degree_legend_levels = build_degree_legend_values(point_df["__overlay_degree"])
            degree_cap = max(degree_legend_levels) if degree_legend_levels else 5
            point_df["__overlay_size"] = point_df["__overlay_degree"].map(
                lambda d: overlay_degree_to_area(d, args.degree_scale, degree_cap)
            )
        else:
            degree_legend_levels = []

        plot_variants = [{"kind": "base", "sample_spec": None}] + [
            {"kind": "sample_biochem", "sample_spec": sample_spec}
            for sample_spec in sample_biochem_variants
        ]

        for label_mode, suffix, extra_right in [
            (True, "", 0.76),
            (False, "_legend", 0.92),
        ]:
            for plot_variant in plot_variants:
                sample_spec = plot_variant["sample_spec"]
                variant_suffix = ""
                variant_title = ""
                if sample_spec is not None:
                    variant_suffix = f"_sample_{sample_spec['slug']}"
                    variant_title = f"\nSamples colored/sized by {sample_spec['feature']}"

                fig, ax = plt.subplots(figsize=(8.5, 7.0))

                if sample_spec is None:
                    ax.scatter(
                        sample_coords[args.ordination_x],
                        sample_coords[args.ordination_y],
                        s=14,
                        c="0.6",
                        alpha=0.35,
                        linewidths=0,
                        zorder=1,
                    )
                else:
                    sample_plot_df = sample_coords.reset_index().merge(
                        sample_spec["df"],
                        on=args.sample_col,
                        how="inner",
                    )
                    sample_colors = sample_spec["cmap"](sample_spec["norm"](sample_plot_df["__sample_biochem_value"].to_numpy()))
                    ax.scatter(
                        sample_plot_df[args.ordination_x],
                        sample_plot_df[args.ordination_y],
                        s=SAMPLE_BUBBLE_SIZE,
                        c=sample_colors,
                        alpha=1.0,
                        linewidths=0.30,
                        edgecolors="#5a5a5a",
                        zorder=1,
                    )
                    sm = cm.ScalarMappable(norm=sample_spec["norm"], cmap=sample_spec["cmap"])
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
                    cbar.set_label(sample_spec["feature"])

                if not non_mag_df.empty:
                    ax.scatter(
                        non_mag_df[args.ordination_x],
                        non_mag_df[args.ordination_y],
                        s=5,
                        c="#3f3f3f",
                        edgecolors="none",
                        alpha=0.85,
                        zorder=2,
                    )

                if not point_df.empty:
                    if mode == "betweenness":
                        low_mask = point_df["betweenness_label"].astype(str).eq(BETWEENNESS_LOW_LABEL)
                        high_mask = point_df["betweenness_label"].astype(str).eq(BETWEENNESS_HIGH_LABEL)
                        low_df = point_df.loc[low_mask].sort_values("Betweenness", ascending=True)
                        high_df = point_df.loc[high_mask].sort_values("Betweenness", ascending=True)
                        if not low_df.empty:
                            ax.scatter(
                                low_df[args.ordination_x],
                                low_df[args.ordination_y],
                                s=low_df["__overlay_size"],
                                c=low_df["__overlay_color"],
                                edgecolors="black",
                                linewidths=0.4,
                                alpha=0.95,
                                zorder=5,
                            )
                        if not high_df.empty:
                            # Halo underlay so the focal high-betweenness points
                            # stay visually on top wherever points overlap.
                            ax.scatter(
                                high_df[args.ordination_x],
                                high_df[args.ordination_y],
                                s=high_df["__overlay_size"] * 1.18,
                                c="white",
                                edgecolors="none",
                                alpha=0.98,
                                zorder=6,
                            )
                            ax.scatter(
                                high_df[args.ordination_x],
                                high_df[args.ordination_y],
                                s=high_df["__overlay_size"],
                                c=high_df["__overlay_color"],
                                edgecolors="black",
                                linewidths=0.55,
                                alpha=0.98,
                                zorder=7,
                            )
                    else:
                        ax.scatter(
                            point_df[args.ordination_x],
                            point_df[args.ordination_y],
                            s=point_df["__overlay_size"],
                            c=point_df["__overlay_color"],
                            edgecolors="black",
                            linewidths=0.4,
                            alpha=0.95,
                            zorder=5,
                        )

                label_texts = []
                label_anchors = []

                for _, row in core_vectors.iterrows():
                    feature = str(row["feature"])
                    col = BIOCHEM_COLOR_MAP.get(feature)
                    if not col:
                        continue
                    dx = float(row[args.ordination_x]) * arrow_scale
                    dy = float(row[args.ordination_y]) * arrow_scale
                    ax.arrow(
                        0, 0, dx, dy,
                        length_includes_head=True,
                        head_width=0.032 * cloud_scale,
                        linewidth=3.2,
                        color="white",
                        zorder=2,
                    )
                    ax.arrow(
                        0, 0, dx, dy,
                        length_includes_head=True,
                        head_width=0.03 * cloud_scale,
                        linewidth=2.2,
                        color=col,
                        zorder=3,
                    )
                    if label_mode:
                        label_texts.append(place_label_at_tip(ax, feature, dx, dy, col, cloud_scale))
                        label_anchors.append((dx, dy))

                for _, row in sparse_corr.iterrows():
                    feature = str(row["feature"])
                    col = BIOCHEM_COLOR_MAP.get(feature)
                    if not col:
                        continue
                    dx = float(row[args.ordination_x]) * arrow_scale
                    dy = float(row[args.ordination_y]) * arrow_scale
                    ax.plot([0, dx], [0, dy], linestyle="--", linewidth=3.2, color="white", zorder=2)
                    ax.plot([0, dx], [0, dy], linestyle="--", linewidth=2.2, color=col, zorder=3)
                    ax.scatter([dx], [dy], s=22, color=col, zorder=4)
                    if label_mode:
                        label_texts.append(place_label_at_tip(ax, feature, dx, dy, col, cloud_scale))
                        label_anchors.append((dx, dy))

                if label_mode:
                    repel_texts(ax, label_texts, label_anchors)

                overlay_handles = []
                if sample_spec is None:
                    overlay_handles.append(
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.6", markeredgecolor="none", alpha=0.35, markersize=5, label="Samples")
                    )
                else:
                    overlay_handles.extend(
                        build_sample_biochem_handles(
                            sample_spec["feature"],
                            sample_spec["df"]["__sample_biochem_value"],
                            sample_spec["vmin"],
                            sample_spec["vmax"],
                        )
                    )
                if not non_mag_df.empty:
                    overlay_handles.append(
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3f3f3f", markeredgecolor="none", markersize=3, label="Other network ASVs")
                    )
                if not point_df.empty:
                    if mode == "mag_pair":
                        overlay_handles.extend([
                            mpatches.Patch(facecolor=palette[label], edgecolor=palette[label], label=label.title())
                            for label in order if label
                        ])
                    else:
                        overlay_handles.append(
                            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.6, markersize=7, label="MAG-linked ASVs")
                        )
                        overlay_handles.extend([
                            mpatches.Patch(facecolor=palette[label], edgecolor=palette[label], label=label)
                            for label in order if label
                        ])
                    overlay_handles.extend(degree_size_handles(degree_legend_levels, args.degree_scale))

                legend_artists = []
                overlay_handles = wrap_legend_handles(overlay_handles, width=28)
                overlay_anchor = (1.18, 1.0) if sample_spec is not None else (1.02, 1.0)
                # Keep the vector legend in a separate right-side column. Long
                # overlay labels are wrapped above, and this larger offset
                # prevents the two legend columns from colliding in *_legend plots.
                biochem_anchor = (1.82, 1.0) if sample_spec is not None else (1.62, 1.0)
                overlay_legend = ax.legend(
                    handles=overlay_handles,
                    loc="upper left",
                    bbox_to_anchor=overlay_anchor,
                    frameon=False,
                    title=f"Overlay: {spec['title']}",
                )
                legend_artists.append(overlay_legend)

                if not label_mode:
                    biochem_handles = biochem_vector_legend_handles(core_vectors, sparse_corr)
                    if biochem_handles:
                        biochem_handles = wrap_legend_handles(biochem_handles, width=26)
                        ax.add_artist(overlay_legend)
                        biochem_legend = ax.legend(
                            handles=biochem_handles,
                            loc="upper left",
                            bbox_to_anchor=biochem_anchor,
                            frameon=False,
                            title="Biochem Vectors",
                        )
                        legend_artists.append(biochem_legend)

                ax.set_xlabel(args.ordination_x)
                ax.set_ylabel(args.ordination_y)
                ax.set_title(
                    f"{args.ordination_x} vs {args.ordination_y} biplot (core loadings + sparse correlations)\n"
                    f"MAG-linked ASV overlay: {spec['title']}{variant_title}"
                )
                ax.axhline(0, linewidth=0.8, color="0.8", zorder=0)
                ax.axvline(0, linewidth=0.8, color="0.8", zorder=0)
                ax.grid(False)
                fig.subplots_adjust(right=extra_right)

                base = plots_dir / f"biochem_network_overlay_{spec['slug']}_{args.ordination_x.lower()}_{args.ordination_y.lower()}{variant_suffix}{suffix}"
                for ext in ("png", "pdf", "svg"):
                    fig.savefig(f"{base}.{ext}", bbox_inches="tight", bbox_extra_artists=legend_artists)
                plt.close(fig)


if __name__ == "__main__":
    main()
