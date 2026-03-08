#!/usr/bin/env python3
"""
isa_plots_cli.py
Build significance tables and publication-grade scatter plots from
indicspecies outputs (multipatt sign tables), optionally overlaying
Venn membership and coloring by Phylum.

Inputs (typical)
---------------
--type-results   path/to/type_group_indicator_species_results.tsv
--status-results path/to/status_indicator_species_results.tsv
--venn           path/to/Three_types_venn_presence_table.tsv   (optional)
--taxonomy       path/to/taxonomy_updated.tsv                   (optional)

All key values (paths, thresholds, palettes, mappings) are configurable
via command line options. Outputs (SVG+PDF) and enriched TSVs are written
under --outdir.

Examples
--------
python isa_plots_cli.py \
  --type-results  /.../indicspecies/type_group_indicator_species_results.tsv \
  --status-results /.../indicspecies/status_indicator_species_results.tsv \
  --venn          /.../metadata/Three_types_venn_presence_table.tsv \
  --taxonomy      /.../metadata/taxonomy_updated.tsv \
  --outdir        /.../indicspecies \
  --p-thresh 0.05 --stat-thresh 0.0 \
  --type-index "1=BAL,2=Lung Brush,3=Oral Rinse,4=BAL+Lung Brush,5=BAL+Oral Rinse,6=Lung Brush+Oral Rinse,7=Oral Rinse+BAL+Lung Brush" \
  --type-palette "Oral Rinse=#6A3D9A,BAL=#0072B2,Lung Brush=#009E73,BAL+Oral Rinse=#F19CBB,BAL+Lung Brush=#00FFFF,Lung Brush+Oral Rinse=#C1EAAD,Oral Rinse+BAL+Lung Brush=#000000" \
  --status-index "1=Cancer,2=Non-Cancer,3=Cancer+Non-Cancer" \
  --status-palette "Cancer=#A50026,Non-Cancer=#FFFFFF,Cancer+Non-Cancer=#000000"
"""

import argparse
from pathlib import Path
import warnings
import os
from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

# ----------------------- Matplotlib/Seaborn defaults ------------------------
mpl.rcParams['pdf.fonttype'] = 42      # Keep text as text in PDF
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")

# Global significance alpha used for horizontal threshold lines in log-p plots.
SIG_LINE_ALPHA = None


# ------------------------------- Utilities ----------------------------------
def parse_mapping(s: str) -> dict:
    """
    Parse "A=#fff,B:#123,C=steelblue" or "1=BAL,2=Lung Brush" into dict.
    Returns {} if s is falsy/empty.
    """
    if not s:
        return {}
    out = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
        elif ":" in item:
            k, v = item.split(":", 1)
        else:
            warnings.warn(f"Ignoring malformed mapping: {item!r}")
            continue
        out[k.strip()] = v.strip()
    return out


def normalize_combo(label: str) -> str:
    """
    Normalize combo strings like "BAL + Lung Brush" -> "BAL + Lung Brush"
    (trim spaces around '+', sort parts alphabetically for stable mapping).
    """
    if not isinstance(label, str):
        return str(label)
    parts = [p.strip() for p in label.split("+")]
    parts = [p for p in parts if p]
    return " + ".join(sorted(parts)) if len(parts) > 1 else (parts[0] if parts else label)


def read_taxonomy_table(taxonomy_path: Path) -> pd.DataFrame:
    """
    Read taxonomy TSV produced earlier; create Domain..Species columns robustly.
    Expects columns: ASV_ID, Taxon (e.g., 'k__Bacteria; p__Firmicutes; ...').
    """
    tax_df = pd.read_csv(taxonomy_path, sep="\t", header=0)
    if "Feature ID" in tax_df.columns:
        tax_df['ASV_ID'] = tax_df['Feature ID'].astype(str).str.partition(';')[0]
    if "ASV_ID" not in tax_df.columns:
        raise ValueError("Taxonomy file must contain column 'ASV_ID'.")
    if "Taxon" not in tax_df.columns:
        raise ValueError("Taxonomy file must contain column 'Taxon'.")

    # Some pipelines append a trailing ";<something>" to ASV_ID – drop last ';...' safely
    tax_df["ASV_ID"] = tax_df["ASV_ID"].astype(str).apply(lambda s: s.split(";", 1)[0])

    def split_taxa_string(taxa_str: str, delimiter=';'):
        tax_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
        taxa_str = str(taxa_str)
        if taxa_str and taxa_str != 'Unassigned' and taxa_str.lower() != 'nan':
            parts = [part.strip().split('__', 1)[-1] for part in taxa_str.split(delimiter)]
        else:
            parts = ['Unassigned']
        return {lvl: (parts[i] if i < len(parts) else None) for i, lvl in enumerate(tax_levels)}

    split_cols = tax_df["Taxon"].apply(split_taxa_string).apply(pd.Series)
    tax_df = pd.concat([tax_df, split_cols], axis=1).set_index("ASV_ID")
    return tax_df


def ensure_cols(df: pd.DataFrame, required: list, where: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {where}: {missing}")


def compute_sig_table(
    sign_df: pd.DataFrame,
    index_map: dict,
    palette: dict,
    p_col: str = "p.value",
    q_col: str = "q.value",
    stat_col: str = "stat",
    idx_col: str = "index",
    p_thresh: float = 0.05,
    stat_thresh: float = 0.0,
    force_all_sig: bool = False,
    prefix: str = "type",
) -> pd.DataFrame:
    """
    From indicspecies sign table -> tidy table with log p, significance, label, color.
    - index_map: numeric string/int -> label (e.g., "1"->"BAL")
    - palette: label -> color
    """
    df = sign_df.copy()

    # coerce/validate key columns
    ensure_cols(df, [p_col, stat_col, idx_col], "indicspecies sign table")
    df[ p_col] = pd.to_numeric(df[p_col], errors="coerce")
    if q_col in df.columns:
        df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
    df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce").astype("Int64")

    # keep minimal columns, but preserve any group-membership columns too
    base_cols = ["ASV_ID"] if "ASV_ID" in df.columns else []
    keep_cols = base_cols + [c for c in df.columns if c not in base_cols]
    df = df[keep_cols]

    # Use the same significance metric for plotting and indicator calls:
    # prefer corrected q-values when available, otherwise fall back to p-values.
    sig_metric = q_col if q_col in df.columns else p_col
    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"{prefix}_log_p"] = (-np.log10(df[sig_metric])).replace([np.inf, -np.inf], np.nan).round(3)

    # significance thresholding
    if force_all_sig:
        df[f"{prefix}_significance"] = True
    else:
        df[f"{prefix}_significance"] = (df[sig_metric] < p_thresh) & (df[stat_col] > stat_thresh)

    # label from index_map
    def idx_to_label(x):
        if pd.isna(x):
            return "not_indicator"
        key = str(int(x))
        return index_map.get(key, index_map.get(int(x), f"index_{int(x)}"))

    labels = df[idx_col].apply(idx_to_label).astype(str)
    df[f"{prefix}_label"] = labels.where(df[f"{prefix}_significance"], "not_indicator")

    # colors
    pal = dict(palette)
    pal["not_indicator"] = "lightgray"
    df[f"{prefix}_color"] = df[f"{prefix}_label"].map(lambda k: pal.get(k, "lightgray"))

    # rename canonical columns for consistency
    df.rename(columns={
        p_col:  f"{prefix}_p_value",
        q_col:  f"{prefix}_q_value",
        stat_col: f"{prefix}_stat",
        idx_col:  f"{prefix}_index"
    }, inplace=True)

    return df


def plot_p_vs_stat_no_overlap(
    df: pd.DataFrame,
    output_file: Path,
    *,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    style_col: Optional[str] = None,
    type_palette: Optional[dict] = None,
    marker_dict: Optional[dict] = None,
    # NEW: point label controls
    label_col: Optional[str] = None,
    label_fontsize: int = 8,
    label_max_points: Optional[int] = None,  # e.g. 200 to avoid labeling everything
    label_bbox: bool = False,
    label_bbox_alpha: float = 0.7,
    label_bbox_color: str = "white",
    label_bbox_edgecolor: str = "none",
    label_bbox_pad: float = 0.15,
    label_bbox_boxstyle: str = "round",
    label_arrow: bool = True,
    label_arrow_color: str = "black",
    label_arrow_width: float = 1.2,
    label_arrow_alpha: float = 1.0,
    label_arrow_shrinkA: float = 6.0,
    label_arrow_shrinkB: float = 0.0,
    label_arrow_anchor: str = "text",
    adjust_expand_points=(1.4, 1.6),
    adjust_expand_text=(1.4, 1.6),
    adjust_force_points=(0.5, 0.8),
    adjust_force_text=(0.5, 0.8),
    adjust_lim: int = 300,
    adjust_avoid_self: bool = True,
    adjust_only_move_points: str = "xy",
    adjust_only_move_text: str = "xy",
    # Jitter controls (normalized units)
    min_dist_x=0.02,
    min_dist_y=0.03,
    step_x=0.35,
    step_y=0.35,
    anchor=0.05,
    iters=200, # this is how you set the deterministic jitter, use 0 for none and 200 for a good spread
    add_random_eps=(0.0, 0.0),  # will be deterministic if kept as-is (uses rng=0)
    # Visuals
    invert_y=False,
    point_size=50,
    alpha=0.85,
    show_legend=True,
    legend_color_title="Type",
    legend_marker_title="Status",
    # Layout
    plot_size_in=(8.0, 6.0),
    axes_pad_in=(0.8, 0.6, 0.3, 0.2),
    figure_edge_pad_in=0.25,
    legend_pad_in=0.45,
    legend_vgap_in=0.25,
    legend_fontsize=10,
):
    """Scatter x_col vs y_col with axis-wise repulsive jitter and fixed data area (deterministic)."""
    dd = df.copy()

    if x_col not in dd.columns or y_col not in dd.columns:
        raise ValueError(f"x_col={x_col!r} or y_col={y_col!r} not present in DataFrame")

    if label_col is not None and label_col not in dd.columns:
        raise ValueError(f"label_col={label_col!r} not present in DataFrame")

    dd[x_col] = pd.to_numeric(dd[x_col], errors="coerce")
    dd[y_col] = pd.to_numeric(dd[y_col], errors="coerce")
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    if dd.empty:
        warnings.warn(f"No data to plot for {output_file.name} after dropping NaN/inf.")
        return

    # --- IMPORTANT: enforce stable ordering so tie-breaks are stable across runs ---
    # Prefer ASV_ID if available; otherwise sort by the numeric coords (still deterministic).
    if "ASV_ID" in dd.columns:
        dd["ASV_ID"] = dd["ASV_ID"].astype(str)
        dd = dd.sort_values(["ASV_ID", x_col, y_col], kind="mergesort").reset_index(drop=True)
    else:
        dd = dd.sort_values([x_col, y_col], kind="mergesort").reset_index(drop=True)

    x = dd[x_col].to_numpy()
    y = dd[y_col].to_numpy()
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if xmin == xmax:
        xmin -= 0.05
        xmax += 0.05
    if ymin == ymax:
        ymin -= 0.05
        ymax += 0.05

    # normalize and repulse
    nx = (x - xmin) / (xmax - xmin)
    ny = (y - ymin) / (ymax - ymin)
    pos = np.stack([nx, ny], axis=1).astype(float)
    orig = pos.copy()

    n = len(pos)
    eye_mask = ~np.eye(n, dtype=bool)

    # Precompute deterministic tie-break sign matrices based purely on indices
    # tie_sign[i,j] is +1 if i>j else -1 (and 0 on diagonal, but diagonal masked anyway)
    idx = np.arange(n)
    tie_sign = np.sign(idx[:, None] - idx[None, :]).astype(float)
    # For i==j it's 0; but diagonal is excluded; still, make it nonzero just in case
    tie_sign[tie_sign == 0] = 1.0

    for _ in range(iters):
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        mask = eye_mask & (np.abs(dx) < min_dist_x) & (np.abs(dy) < min_dist_y)
        if not mask.any():
            break

        # Deterministic sign: if dx==0 (or dy==0) use index-based tie-break instead of RNG
        sign_x = np.sign(dx)
        sign_y = np.sign(dy)
        sign_x[sign_x == 0] = tie_sign[sign_x == 0]
        sign_y[sign_y == 0] = tie_sign[sign_y == 0]

        force_x = np.zeros_like(dx)
        force_y = np.zeros_like(dy)
        force_x[mask] = (min_dist_x - np.abs(dx[mask])) * sign_x[mask]
        force_y[mask] = (min_dist_y - np.abs(dy[mask])) * sign_y[mask]

        pos[:, 0] += step_x * force_x.sum(axis=1) - anchor * (pos[:, 0] - orig[:, 0])
        pos[:, 1] += step_y * force_y.sum(axis=1) - anchor * (pos[:, 1] - orig[:, 1])
        np.clip(pos, 0.0, 1.0, out=pos)

    # Optional epsilon jitter: already deterministic because RNG seed is fixed
    if add_random_eps != (0.0, 0.0):
        rng = np.random.default_rng(0)
        pos[:, 0] = np.clip(pos[:, 0] + rng.normal(0, add_random_eps[0], n), 0, 1)
        pos[:, 1] = np.clip(pos[:, 1] + rng.normal(0, add_random_eps[1], n), 0, 1)

    dd["_x_"] = pos[:, 0] * (xmax - xmin) + xmin
    dd["_y_"] = pos[:, 1] * (ymax - ymin) + ymin

    # Resolve hue palette
    palette = None
    if hue_col is not None:
        if type_palette:
            present = [h for h in dd[hue_col].dropna().unique().tolist() if h in type_palette]
            palette = {k: type_palette[k] for k in present}
        else:
            palette = None  # default seaborn

    # Resolve style markers
    markers = None
    if style_col is not None:
        cats = dd[style_col].dropna().unique().tolist()
        if marker_dict:
            markers = marker_dict
        else:
            default_markers = ["o", "s", "D", "X", "^", "v", "P", "*", "h", "H", "8", "p", "<", ">"]
            markers = {c: default_markers[i % len(default_markers)] for i, c in enumerate(cats)}

    # Legend handles
    color_handles = []
    if type_palette:
        for name in type_palette:
            if name != 'not_indicator':
                col = type_palette.get(name, "lightgray")
                color_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=col,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        markersize=8,
                        label=str(name),
                    )
                )

    marker_handles = []
    if show_legend and style_col is not None and markers:
        for name, mk in markers.items():
            if name != 'not_indicator':
                marker_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color="gray",
                        marker=mk,
                        linestyle="None",
                        markeredgewidth=0.5,
                        markersize=8,
                        label=str(name),
                    )
                )

    # Measure legend sizes
    def _legend_size_in(handles, title, fontsize):
        if not handles:
            return (0.0, 0.0)
        ftmp, axtmp = plt.subplots(figsize=(2, 2), dpi=100)
        leg = axtmp.legend(
            handles=handles, title=title, frameon=True, loc="upper left", fontsize=fontsize, title_fontsize=fontsize
        )
        ftmp.canvas.draw()
        bbox = leg.get_window_extent(ftmp.canvas.get_renderer())
        w_in = bbox.width / ftmp.dpi
        h_in = bbox.height / ftmp.dpi
        plt.close(ftmp)
        return (w_in, h_in)

    color_w, color_h = _legend_size_in(color_handles, legend_color_title, legend_fontsize)
    marker_w, marker_h = _legend_size_in(marker_handles, legend_marker_title, legend_fontsize)

    legend_w_in = max(color_w, marker_w) if show_legend else 0.0
    legend_h_in = (color_h if color_h else 0.0) + (marker_h if marker_h else 0.0)
    if show_legend and color_h and marker_h:
        legend_h_in += legend_vgap_in

    # Fixed data rectangle layout
    plot_w_in, plot_h_in = plot_size_in
    padL, padB, padR, padT = axes_pad_in
    pane_w_in = plot_w_in + padL + padR
    pane_h_in = plot_h_in + padT + padB

    fig_w_in = (
        figure_edge_pad_in
        + pane_w_in
        + ((legend_pad_in + legend_w_in) if (show_legend and legend_w_in > 0) else 0.0)
        + figure_edge_pad_in
    )
    fig_h_in = figure_edge_pad_in + max(pane_h_in, legend_h_in if show_legend else pane_h_in) + figure_edge_pad_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)

    ax_left = (figure_edge_pad_in + padL) / fig_w_in
    ax_bottom = (figure_edge_pad_in + padB) / fig_h_in
    ax_w = plot_w_in / fig_w_in
    ax_h = plot_h_in / fig_h_in
    ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])

    leg_ax = None
    if show_legend and legend_w_in > 0:
        leg_left = (figure_edge_pad_in + pane_w_in + legend_pad_in) / fig_w_in
        leg_w = legend_w_in / fig_w_in
        leg_ax = fig.add_axes(
            [leg_left, figure_edge_pad_in / fig_h_in, leg_w, 1.0 - 2 * figure_edge_pad_in / fig_h_in]
        )
        leg_ax.axis("off")

    # Draw threshold lines behind points.
    if SIG_LINE_ALPHA is not None and isinstance(SIG_LINE_ALPHA, (int, float)) and SIG_LINE_ALPHA > 0:
        if y_col.endswith("_log_p"):
            y_thr = -np.log10(SIG_LINE_ALPHA)
            ax.axhline(y=y_thr, linestyle="--", linewidth=1.0, color="gray", alpha=0.8, zorder=0)
    ax.axvline(x=0.25, linestyle="--", linewidth=1.0, color="gray", alpha=0.8, zorder=0)

    # Plot points
    sns.scatterplot(
        data=dd,
        x="_x_",
        y="_y_",
        hue=hue_col if hue_col else None,
        style=style_col if style_col else None,
        palette=palette,
        markers=markers,
        s=point_size,
        alpha=alpha,
        linewidth=0.5,
        edgecolor="black",
        legend=False,
        ax=ax,
        zorder=2,
    )
    ax.set_xlabel(x_col)
    if y_col.endswith("_log_p"):
        prefix = y_col[: -len("_log_p")]
        q_value_col = f"{prefix}_q_value"
        if q_value_col in dd.columns and pd.to_numeric(dd[q_value_col], errors="coerce").notna().any():
            ax.set_ylabel("-log10(q.value)")
        else:
            ax.set_ylabel("-log10(p.value)")
    else:
        ax.set_ylabel(y_col)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.tick_params(axis="both", which="both", length=4, width=1)

    # Labels: adjustText may introduce nondeterminism across platforms/versions.
    # If you need full determinism for labels too, set the env var ISA_DISABLE_ADJUSTTEXT=1
    # and this will use the deterministic fallback.
    if label_col is not None:
        lab = dd[label_col].astype("object").fillna("").astype(str)
        mask = lab.str.len() > 0
        dd_lab = dd.loc[mask, ["_x_", "_y_"]].copy()
        lab = lab.loc[mask]

        if label_max_points is not None and len(dd_lab) > label_max_points:
            dd_lab = dd_lab.iloc[:label_max_points].copy()
            lab = lab.iloc[:label_max_points]

        texts = []
        xspan = (xmax - xmin) if (xmax != xmin) else 1.0
        yspan = (ymax - ymin) if (ymax != ymin) else 1.0
        dx0 = 0.01 * xspan
        dy0 = 0.01 * yspan
        label_bbox_kwargs = None
        if label_bbox:
            label_bbox_kwargs = dict(
                boxstyle=f"{label_bbox_boxstyle},pad={label_bbox_pad}",
                facecolor=label_bbox_color,
                alpha=label_bbox_alpha,
                edgecolor=label_bbox_edgecolor,
            )
        arrowprops = None
        if label_arrow and label_arrow_anchor == "text":
            arrowprops = dict(
                arrowstyle="-",
                color=label_arrow_color,
                lw=label_arrow_width,
                alpha=label_arrow_alpha,
                shrinkA=label_arrow_shrinkA,
                shrinkB=label_arrow_shrinkB,
            )

        for (xv, yv), s in zip(dd_lab[["_x_", "_y_"]].to_numpy(), lab.to_numpy()):
            texts.append(
                ax.text(
                    xv + dx0,
                    yv + dy0,
                    s,
                    fontsize=label_fontsize,
                    bbox=label_bbox_kwargs,
                )
            )

        disable_adjust = (os.environ.get("ISA_DISABLE_ADJUSTTEXT", "0") == "1")
        used_adjust = False
        if not disable_adjust:
            try:
                from adjustText import adjust_text

                only_move = None
                if adjust_only_move_points != "none" or adjust_only_move_text != "none":
                    only_move = {}
                    if adjust_only_move_points != "none":
                        only_move["points"] = adjust_only_move_points
                    if adjust_only_move_text != "none":
                        only_move["text"] = adjust_only_move_text

                adjust_text(
                    texts,
                    ax=ax,
                    x=dd_lab["_x_"].to_numpy(),
                    y=dd_lab["_y_"].to_numpy(),
                    expand_points=tuple(adjust_expand_points),
                    expand_text=tuple(adjust_expand_text),
                    force_points=tuple(adjust_force_points),
                    force_text=tuple(adjust_force_text),
                    lim=adjust_lim,
                    avoid_self=adjust_avoid_self,
                    only_move=only_move,
                    arrowprops=arrowprops,
                )
                used_adjust = True
            except Exception:
                pass
        if label_arrow and label_arrow_anchor == "bbox":
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for (xv, yv), txt in zip(dd_lab[["_x_", "_y_"]].to_numpy(), texts):
                bbox = txt.get_window_extent(renderer=renderer)
                px, py = ax.transData.transform((xv, yv))
                cx = min(max(px, bbox.x0), bbox.x1)
                cy = min(max(py, bbox.y0), bbox.y1)
                x0, y0 = ax.transData.inverted().transform((cx, cy))
                ax.plot(
                    [xv, x0],
                    [yv, y0],
                    color=label_arrow_color,
                    lw=label_arrow_width,
                    alpha=label_arrow_alpha,
                )
        elif label_arrow and not used_adjust:
            for (xv, yv), txt in zip(dd_lab[["_x_", "_y_"]].to_numpy(), texts):
                tx, ty = txt.get_position()
                ax.plot(
                    [xv, tx],
                    [yv, ty],
                    color=label_arrow_color,
                    lw=label_arrow_width,
                    alpha=label_arrow_alpha,
                )

    # Legend pane
    if show_legend and leg_ax is not None:
        y_cursor = 1.0
        if color_handles:
            leg1 = leg_ax.legend(
                handles=color_handles,
                title=legend_color_title,
                loc="upper left",
                bbox_to_anchor=(0.0, y_cursor),
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )
            leg_ax.add_artist(leg1)
            y_cursor -= ((color_h if color_h else 0.0) + legend_vgap_in) / fig_h_in
        if marker_handles:
            leg_ax.legend(
                handles=marker_handles,
                title=legend_marker_title,
                loc="upper left",
                bbox_to_anchor=(0.0, y_cursor),
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=legend_fontsize,
            )

    # Save (SVG + PDF)
    output_file = Path(output_file)
    fig.savefig(output_file)
    if output_file.suffix.lower() == ".svg":
        fig.savefig(output_file.with_suffix(".pdf"))
    plt.close(fig)


# ------------------------------- Main ---------------------------------------
def main():
    global SIG_LINE_ALPHA
    ap = argparse.ArgumentParser(
        description="Refactored ISA plotting pipeline (indicspecies -> tidy tables + figures)."
    )
    ap.add_argument("--type-results", type=Path, required=True,
                    help="indicspecies sign table for type_group (TSV).")
    ap.add_argument("--status-results", type=Path, required=True,
                    help="indicspecies sign table for status (TSV).")
    ap.add_argument("--type-venn", type=Path, default=None,
                    help="Optional Type Venn presence table (cols: grouping, ASV_ID).")
    ap.add_argument("--status-venn", type=Path, default=None,
                    help="Optional Status Venn presence table (cols: grouping, ASV_ID).")
    ap.add_argument("--taxonomy", type=Path, default=None,
                    help="Optional taxonomy table (taxonomy_updated.tsv).")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Output directory for enriched TSVs and figures.")

    # Thresholds
    ap.add_argument("--p-thresh", type=float, default=0.05, help="p-value threshold (default: 0.05).")
    ap.add_argument("--stat-thresh", type=float, default=0.0, help="stat threshold (default: 0.0).")

    # Index maps
    ap.add_argument("--type-index", type=str, required=True,
                    help='Mapping of indicspecies "index" to labels, e.g. '
                         '"1=BAL,2=Lung Brush,3=Oral Rinse,4=BAL+Lung Brush,5=BAL+Oral Rinse,6=Lung Brush+Oral Rinse,7=Oral Rinse+BAL+Lung Brush"')
    ap.add_argument("--status-index", type=str, required=True,
                    help='Mapping for status "index", e.g. "1=Cancer,2=Non-Cancer,3=Cancer+Non-Cancer"')

    # Palettes
    ap.add_argument("--type-palette", type=str, required=True,
                    help='Color map for type labels, e.g. '
                         '"Oral Rinse=#6A3D9A,BAL=#0072B2,Lung Brush=#009E73,'
                         'BAL+Oral Rinse=#F19CBB,BAL+Lung Brush=#00FFFF,'
                         'Lung Brush+Oral Rinse=#C1EAAD,Oral Rinse+BAL+Lung Brush=#000000"')
    ap.add_argument("--status-palette", type=str, required=True,
                    help='Color map for status, e.g. "Cancer=#A50026,Non-Cancer=#FFFFFF,Cancer+Non-Cancer=#000000"')
    ap.add_argument("--status-markers", type=str, default="Cancer=X,Non-Cancer=D,Cancer+Non-Cancer=o",
                    help='Marker styles for status, e.g. "Cancer=X,Non-Cancer=D,Cancer+Non-Cancer=o"')

    # Column names in sign tables (robustness for variants)
    ap.add_argument("--p-col", default="p.value", help="Column name for p-values in sign tables (default: p.value).")
    ap.add_argument("--stat-col", default="stat", help="Column name for stat values in sign tables (default: stat).")
    ap.add_argument("--idx-col", default="index", help="Column name for index in sign tables (default: index).")

    # Plot layout
    ap.add_argument("--plot-width", type=float, default=8.0)
    ap.add_argument("--plot-height", type=float, default=6.0)

    # --- point label controls ---
    ap.add_argument("--label-col", default=None,
        help="DataFrame column to use as text labels for each plotted point (default: no labels).",
    )
    ap.add_argument("--label-fontsize", type=int, default=8,
        help="Font size for point labels (default: 8).",
    )
    ap.add_argument("--label-max-points", type=int, default=None,
        help="If set, only label the first N points (useful to avoid clutter).",
    )
    ap.add_argument("--label-bbox", action="store_true",
        help="Draw a light background box behind labels to improve readability.",
    )
    ap.add_argument("--label-bbox-alpha", type=float, default=0.7,
        help="Label background alpha (default: 0.7).",
    )
    ap.add_argument("--label-bbox-color", default="white",
        help="Label background color (default: white).",
    )
    ap.add_argument("--label-bbox-edgecolor", default="none",
        help="Label background edge color (default: none).",
    )
    ap.add_argument("--label-bbox-pad", type=float, default=0.15,
        help="Label background padding (default: 0.15).",
    )
    ap.add_argument("--label-bbox-boxstyle", default="round",
        help='Label background boxstyle (default: "round").',
    )
    ap.add_argument("--label-arrow", action="store_true", default=True,
        help="Draw connector lines from labels to points (default: on).",
    )
    ap.add_argument("--no-label-arrow", action="store_false", dest="label_arrow",
        help="Disable connector lines from labels to points.",
    )
    ap.add_argument("--label-arrow-color", default="black",
        help='Label connector line color (default: "black").',
    )
    ap.add_argument("--label-arrow-width", type=float, default=1.2,
        help="Label connector line width (default: 1.2).",
    )
    ap.add_argument("--label-arrow-alpha", type=float, default=1.0,
        help="Label connector line alpha (default: 1.0).",
    )
    ap.add_argument("--label-arrow-shrinkA", type=float, default=6.0,
        help="Arrow shrink at text end (default: 6.0).",
    )
    ap.add_argument("--label-arrow-shrinkB", type=float, default=0.0,
        help="Arrow shrink at point end (default: 0.0).",
    )
    ap.add_argument("--label-arrow-anchor", choices=["text", "bbox"], default="text",
        help='Where to anchor connector lines (default: "text"; use "bbox" to attach to label edge).',
    )
    ap.add_argument("--adjust-expand-points", type=float, nargs=2, default=(1.4, 1.6),
        metavar=("X", "Y"),
        help="adjustText expand for points (default: 1.4 1.6).",
    )
    ap.add_argument("--adjust-expand-text", type=float, nargs=2, default=(1.4, 1.6),
        metavar=("X", "Y"),
        help="adjustText expand for text (default: 1.4 1.6).",
    )
    ap.add_argument("--adjust-force-points", type=float, nargs=2, default=(0.5, 0.8),
        metavar=("X", "Y"),
        help="adjustText force for points (default: 0.5 0.8).",
    )
    ap.add_argument("--adjust-force-text", type=float, nargs=2, default=(0.5, 0.8),
        metavar=("X", "Y"),
        help="adjustText force for text (default: 0.5 0.8).",
    )
    ap.add_argument("--adjust-lim", type=int, default=300,
        help="adjustText iteration limit (default: 300).",
    )
    ap.add_argument("--adjust-avoid-self", action="store_true", default=True,
        help="Keep labels from colliding with their own points (default: on).",
    )
    ap.add_argument("--no-adjust-avoid-self", action="store_false", dest="adjust_avoid_self",
        help="Disable adjustText avoid_self.",
    )
    ap.add_argument("--adjust-only-move-points", choices=["x", "y", "xy", "none"], default="xy",
        help='Restrict adjustText movement for points (default: "xy").',
    )
    ap.add_argument("--adjust-only-move-text", choices=["x", "y", "xy", "none"], default="xy",
        help='Restrict adjustText movement for text (default: "xy").',
    )

    args = ap.parse_args()
    SIG_LINE_ALPHA = args.p_thresh

    # ---- Label locks (combined plots only; easy to find/adjust) ----
    LABEL_PVAL_CUTOFF = 100000000000000 #0.05
    LABEL_STAT_CUTOFF = 0 #0.25
    # Restrict type labels to specific groups (set to None to allow all types)
    # Example: {"BAL", "Lung Brush"}
    TYPE_LABEL_ALLOWLIST = {"Lung Brush", "BAL+Lung Brush", "Lung Brush+Oral Rinse", "BAL"}
    # Force labels for these status groups in combined plots (set to None to disable)
    STATUS_LABEL_ALLOWLIST = {"Cancer", "Non-Cancer"}

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    tables_outdir = outdir / "tables"
    tables_outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = outdir / "plots"
    plots_outdir.mkdir(parents=True, exist_ok=True)

    # ---- Read inputs ----
    tdf = pd.read_csv(args.type_results, sep="\t", header=0)
    if tdf.columns[0].lower() not in ("asv_id", "asv", "feature", "otu"):
        # indicspecies export often lacks ASV column name; make it explicit
        tdf.rename(columns={tdf.columns[0]: "ASV_ID"}, inplace=True)
    else:
        tdf.rename(columns={tdf.columns[0]: "ASV_ID"}, inplace=True)

    sdf = pd.read_csv(args.status_results, sep="\t", header=0)
    if sdf.columns[0].lower() not in ("asv_id", "asv", "feature", "otu"):
        sdf.rename(columns={sdf.columns[0]: "ASV_ID"}, inplace=True)
    else:
        sdf.rename(columns={sdf.columns[0]: "ASV_ID"}, inplace=True)

    type_venn_df = None
    if args.type_venn and args.type_venn.exists():
        type_venn_df = pd.read_csv(args.type_venn, sep="\t", header=0)
        ensure_cols(type_venn_df, ["ASV_ID", "grouping"], "Venn table")
        type_venn_df["grouping"] = type_venn_df["grouping"].astype(str).map(normalize_combo)

    status_venn_df = None
    if args.status_venn and args.status_venn.exists():
        status_venn_df = pd.read_csv(args.status_venn, sep="\t", header=0)
        ensure_cols(status_venn_df, ["ASV_ID", "grouping"], "Venn table")
        status_venn_df["grouping"] = status_venn_df["grouping"].astype(str).map(normalize_combo)

    tax_df = None
    if args.taxonomy:
        tax_df = read_taxonomy_table(args.taxonomy)

    # ---- Parse mappings/palettes ----
    type_index_map = parse_mapping(args.type_index)
    status_index_map = parse_mapping(args.status_index)
    # Allow integer keys too:
    type_index_map.update({int(k): v for k, v in list(type_index_map.items()) if k.isdigit()})
    status_index_map.update({int(k): v for k, v in list(status_index_map.items()) if k.isdigit()})

    type_palette = parse_mapping(args.type_palette)
    status_palette = parse_mapping(args.status_palette)
    # Ensure background class is always color-mapped for seaborn categorical plots.
    type_palette.setdefault("not_indicator", "lightgray")
    status_palette.setdefault("not_indicator", "lightgray")
    status_markers = parse_mapping(args.status_markers)

    # ---- Build significance tables ----
    type_sig = compute_sig_table(
        tdf, index_map=type_index_map, palette=type_palette,
        p_col=args.p_col, stat_col=args.stat_col, idx_col=args.idx_col,
        p_thresh=args.p_thresh, stat_thresh=args.stat_thresh,
        force_all_sig=False, prefix="type"
    )
    type_asv = type_sig["ASV_ID"].astype("object").fillna("").astype(str)
    type_sig_mask = (
        pd.to_numeric(type_sig["type_p_value"], errors="coerce") <= LABEL_PVAL_CUTOFF
    ) & (
        pd.to_numeric(type_sig["type_stat"], errors="coerce") >= LABEL_STAT_CUTOFF
    )
    if TYPE_LABEL_ALLOWLIST is not None:
        type_sig_mask = type_sig_mask & type_sig["type_label"].isin(TYPE_LABEL_ALLOWLIST)
    type_sig["label_type_sig"] = np.where(type_sig_mask.fillna(False), type_asv, "")
    type_sig.to_csv(tables_outdir / "type_group_ISA_enriched.tsv", sep="\t", index=False)

    status_sig = compute_sig_table(
        sdf, index_map=status_index_map, palette=status_palette,
        p_col=args.p_col, stat_col=args.stat_col, idx_col=args.idx_col,
        p_thresh=args.p_thresh, stat_thresh=args.stat_thresh,
        force_all_sig=False, prefix="status"
    )
    status_sig.to_csv(tables_outdir / "status_ISA_enriched.tsv", sep="\t", index=False)

    # ---- Type plot (ISA) ----
    plot_p_vs_stat_no_overlap(
        type_sig,
        plots_outdir / "type_group_ISA_plot.svg",
        x_col="type_stat", y_col="type_log_p",
        hue_col="type_label",
        type_palette=type_palette,
        plot_size_in=(args.plot_width, args.plot_height),
    )
    plot_p_vs_stat_no_overlap(
        type_sig,
        plots_outdir / "type_group_ISA_plot_labelled.svg",
        x_col="type_stat", y_col="type_log_p",
        hue_col="type_label",
        type_palette=type_palette,
        plot_size_in=(args.plot_width, args.plot_height),
        label_col="label_type_sig",
        label_fontsize=args.label_fontsize,
        label_max_points=args.label_max_points,
        label_bbox=args.label_bbox,
        label_bbox_alpha=args.label_bbox_alpha,
        label_bbox_color=args.label_bbox_color,
        label_bbox_edgecolor=args.label_bbox_edgecolor,
        label_bbox_pad=args.label_bbox_pad,
        label_bbox_boxstyle=args.label_bbox_boxstyle,
        label_arrow=args.label_arrow,
        label_arrow_color=args.label_arrow_color,
        label_arrow_width=args.label_arrow_width,
        label_arrow_alpha=args.label_arrow_alpha,
        label_arrow_shrinkA=args.label_arrow_shrinkA,
        label_arrow_shrinkB=args.label_arrow_shrinkB,
        label_arrow_anchor=args.label_arrow_anchor,
        adjust_expand_points=args.adjust_expand_points,
        adjust_expand_text=args.adjust_expand_text,
        adjust_force_points=args.adjust_force_points,
        adjust_force_text=args.adjust_force_text,
        adjust_lim=args.adjust_lim,
        adjust_avoid_self=args.adjust_avoid_self,
        adjust_only_move_points=args.adjust_only_move_points,
        adjust_only_move_text=args.adjust_only_move_text,
    )

    # ---- Type plot using Venn membership (optional; force all sig for color only) ----
    if type_venn_df is not None:
        tv_map = dict(zip(type_venn_df["ASV_ID"], type_venn_df["grouping"]))
        # Map Venn label -> normalized, then to index_map labels
        # Convert to the index label set used in palette (normalize for consistent keys)
        # If a Venn label isn't present in your palette, it'll fall back to lightgray.
        tv_sub = tdf.copy()
        if "ASV_ID" not in tv_sub.columns:
            tv_sub.rename(columns={tv_sub.columns[0]: "ASV_ID"}, inplace=True)
        tv_sub["ASV_ID"] = tv_sub["ASV_ID"].astype(str)
        tv_sub["__venn_label__"] = tv_sub["ASV_ID"].map(tv_map).dropna()

        tvenn_sig = compute_sig_table(
            tv_sub, index_map={}, palette={},  # labels come from Venn below
            p_col=args.p_col, stat_col=args.stat_col, idx_col=args.idx_col,
            p_thresh=args.p_thresh, stat_thresh=args.stat_thresh,
            force_all_sig=True, prefix="type"
        )
        tvenn_sig["type_label"] = tvenn_sig["ASV_ID"].map(tv_map).map(normalize_combo).fillna("not_indicator")
        # Convert Venn labels to palette keys if you used a different wording
        # Example mapping often needed:
        # "Oral Rinse + BAL" -> "BAL+Oral Rinse"
        tvenn_label_to_palette_key = {normalize_combo(k): k for k in type_palette.keys()}  # identity by default
        tvenn_sig["type_label"] = tvenn_sig["type_label"].map(lambda s: tvenn_label_to_palette_key.get(s, s))
        tvenn_sig["type_color"] = tvenn_sig["type_label"].map(lambda k: type_palette.get(k, "lightgray"))

        tvenn_sig.to_csv(tables_outdir / "type_group_Venn_enriched.tsv", sep="\t", index=False)
        plot_p_vs_stat_no_overlap(
            tvenn_sig,
            plots_outdir / "type_group_Venn_plot.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="type_label",
            type_palette=type_palette,
            plot_size_in=(args.plot_width, args.plot_height),
        )

    # ---- Status plot using Venn membership (optional; force all sig for color only) ----
    if status_venn_df is not None:
        sv_map = dict(zip(status_venn_df["ASV_ID"], status_venn_df["grouping"]))
        # Map Venn label -> normalized, then to index_map labels
        # Convert to the index label set used in palette (normalize for consistent keys)
        # If a Venn label isn't present in your palette, it'll fall back to lightgray.
        sv_sub = sdf.copy()
        if "ASV_ID" not in sv_sub.columns:
            sv_sub.rename(columns={sv_sub.columns[0]: "ASV_ID"}, inplace=True)
        sv_sub["ASV_ID"] = sv_sub["ASV_ID"].astype(str)
        sv_sub["__venn_label__"] = sv_sub["ASV_ID"].map(sv_map).dropna()

        svenn_sig = compute_sig_table(
            sv_sub, index_map={}, palette={},  # labels come from Venn below
            p_col=args.p_col, stat_col=args.stat_col, idx_col=args.idx_col,
            p_thresh=args.p_thresh, stat_thresh=args.stat_thresh,
            force_all_sig=True, prefix="status"
        )
        svenn_sig["status_label"] = svenn_sig["ASV_ID"].map(sv_map).map(normalize_combo).fillna("not_indicator")
        # Convert Venn labels to palette keys if you used a different wording
        # Example mapping often needed:
        # "Oral Rinse + BAL" -> "BAL+Oral Rinse"
        svenn_label_to_palette_key = {normalize_combo(k): k for k in status_palette.keys()}  # identity by default
        svenn_sig["status_label"] = svenn_sig["status_label"].map(lambda s: svenn_label_to_palette_key.get(s, s))
        svenn_sig["status_color"] = svenn_sig["status_label"].map(lambda k: status_palette.get(k, "lightgray"))

        svenn_sig.to_csv(tables_outdir / "status_group_Venn_enriched.tsv", sep="\t", index=False)
        plot_p_vs_stat_no_overlap(
            svenn_sig,
            plots_outdir / "status_group_Venn_plot.svg",
            x_col="status_stat", y_col="status_log_p",
            hue_col="status_label",
            type_palette=status_palette,
            plot_size_in=(args.plot_width, args.plot_height),
        )
    
    venn_combined = None
    if ((type_venn_df is not None) and (status_venn_df is not None)):
        # ---- Venn Combined tables/plots: join type + status on ASV ----
        venn_combined = pd.merge(tvenn_sig[["ASV_ID", "type_stat", "type_p_value", "type_log_p",
                                    "type_significance", "type_label", "type_color"]],
                            svenn_sig[["ASV_ID", "status_stat", "status_p_value", "status_log_p",
                                        "status_significance", "status_label"]],
                            on="ASV_ID", how="outer")
        venn_asv = venn_combined["ASV_ID"].astype("object").fillna("").astype(str)
        venn_type_sig = (
            pd.to_numeric(venn_combined["type_p_value"], errors="coerce") <= LABEL_PVAL_CUTOFF
        ) & (
            pd.to_numeric(venn_combined["type_stat"], errors="coerce") >= LABEL_STAT_CUTOFF
        )
        if TYPE_LABEL_ALLOWLIST is not None:
            venn_type_sig = venn_type_sig & venn_combined["type_label"].isin(TYPE_LABEL_ALLOWLIST)
        venn_status_sig = (
            pd.to_numeric(venn_combined["status_p_value"], errors="coerce") <= LABEL_PVAL_CUTOFF
        ) & (
            pd.to_numeric(venn_combined["status_stat"], errors="coerce") >= LABEL_STAT_CUTOFF
        )
        venn_type_sig = venn_type_sig.fillna(False)
        venn_status_sig = venn_status_sig.fillna(False)
        if STATUS_LABEL_ALLOWLIST is not None:
            venn_status_allow = venn_combined["status_label"].isin(STATUS_LABEL_ALLOWLIST)
        else:
            venn_status_allow = venn_status_sig
        venn_combined["label_type_sig"] = np.where(venn_type_sig, venn_asv, "")
        venn_combined["label_status_sig"] = np.where(venn_status_sig, venn_asv, "")
        venn_combined["label_status_all"] = np.where(venn_status_allow.fillna(False), venn_asv, "")
        venn_combined["label_any_sig"] = np.where((venn_type_sig | venn_status_sig), venn_asv, "")
        venn_combined.to_csv(tables_outdir / "Type_status_Venn_results.tsv", sep="\t", index=False)
    
    # ---- Status plot (ISA) ----
    plot_p_vs_stat_no_overlap(
        status_sig,
        plots_outdir / "status_ISA_plot.svg",
        x_col="status_stat", y_col="status_log_p",
        hue_col="status_label",
        type_palette=status_palette,
        plot_size_in=(args.plot_width, args.plot_height),
    )

    # ---- Combined tables/plots: join type + status on ASV ----
    combined = pd.merge(type_sig[["ASV_ID", "type_stat", "type_p_value", "type_log_p",
                                  "type_significance", "type_label", "type_color"]],
                        status_sig[["ASV_ID", "status_stat", "status_p_value", "status_log_p",
                                    "status_significance", "status_label"]],
                        on="ASV_ID", how="outer")
    # Label only significant points for combined plots (based on which palette is used)
    comb_asv = combined["ASV_ID"].astype("object").fillna("").astype(str)
    comb_type_sig = (
        pd.to_numeric(combined["type_p_value"], errors="coerce") <= LABEL_PVAL_CUTOFF
    ) & (
        pd.to_numeric(combined["type_stat"], errors="coerce") >= LABEL_STAT_CUTOFF
    )
    if TYPE_LABEL_ALLOWLIST is not None:
        comb_type_sig = comb_type_sig & combined["type_label"].isin(TYPE_LABEL_ALLOWLIST)
    comb_status_sig = (
        pd.to_numeric(combined["status_p_value"], errors="coerce") <= LABEL_PVAL_CUTOFF
    ) & (
        pd.to_numeric(combined["status_stat"], errors="coerce") >= LABEL_STAT_CUTOFF
    )
    comb_type_sig = comb_type_sig.fillna(False)
    comb_status_sig = comb_status_sig.fillna(False)
    if STATUS_LABEL_ALLOWLIST is not None:
        comb_status_allow = combined["status_label"].isin(STATUS_LABEL_ALLOWLIST)
    else:
        comb_status_allow = comb_status_sig
    combined["label_type_sig"] = np.where(comb_type_sig, comb_asv, "")
    combined["label_status_sig"] = np.where(comb_status_sig, comb_asv, "")
    combined["label_status_all"] = np.where(comb_status_allow.fillna(False), comb_asv, "")
    combined["label_any_sig"] = np.where((comb_type_sig | comb_status_sig), comb_asv, "")
    combined.to_csv(tables_outdir / "Type_status_ISA_results.tsv", sep="\t", index=False)

    plot_p_vs_stat_no_overlap(
        combined,
        plots_outdir / "Combined_ISA_plot.svg",
        x_col="type_stat", y_col="type_log_p",
        hue_col="status_label", # style_col="status_label",
        type_palette=status_palette, # marker_dict=status_markers,
        legend_color_title="Status", # legend_marker_title="Status",
        plot_size_in=(args.plot_width, args.plot_height),
    )

    plot_p_vs_stat_no_overlap(
        combined,
        plots_outdir / "Combined_ISA_plot_labelled.svg",
        x_col="type_stat", y_col="type_log_p",
        hue_col="status_label", # style_col="status_label",
        type_palette=status_palette, # marker_dict=status_markers,
        legend_color_title="Status", #legend_marker_title="Status",
        plot_size_in=(args.plot_width, args.plot_height),
        label_col="label_status_all",
        label_fontsize=args.label_fontsize,
        label_max_points=args.label_max_points,
        label_bbox=args.label_bbox,
        label_bbox_alpha=args.label_bbox_alpha,
        label_bbox_color=args.label_bbox_color,
        label_bbox_edgecolor=args.label_bbox_edgecolor,
        label_bbox_pad=args.label_bbox_pad,
        label_bbox_boxstyle=args.label_bbox_boxstyle,
        label_arrow=args.label_arrow,
        label_arrow_color=args.label_arrow_color,
        label_arrow_width=args.label_arrow_width,
        label_arrow_alpha=args.label_arrow_alpha,
        label_arrow_shrinkA=args.label_arrow_shrinkA,
        label_arrow_shrinkB=args.label_arrow_shrinkB,
        label_arrow_anchor=args.label_arrow_anchor,
        adjust_expand_points=args.adjust_expand_points,
        adjust_expand_text=args.adjust_expand_text,
        adjust_force_points=args.adjust_force_points,
        adjust_force_text=args.adjust_force_text,
        adjust_lim=args.adjust_lim,
        adjust_avoid_self=args.adjust_avoid_self,
        adjust_only_move_points=args.adjust_only_move_points,
        adjust_only_move_text=args.adjust_only_move_text,
    )

    if venn_combined is not None:
        plot_p_vs_stat_no_overlap(
            venn_combined,
            plots_outdir / "Combined_Venn_plot.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="status_label", # style_col="status_label",
            type_palette=status_palette, # marker_dict=status_markers,
            legend_color_title="Status", # legend_marker_title="Status",
            plot_size_in=(args.plot_width, args.plot_height),
        )

        plot_p_vs_stat_no_overlap(
            venn_combined,
            plots_outdir / "Combined_Venn_plot_labelled.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="status_label", # style_col="status_label",
            type_palette=status_palette, # marker_dict=status_markers,
            legend_color_title="Status", # legend_marker_title="Status",
            plot_size_in=(args.plot_width, args.plot_height),
            label_col="label_status_all",
            label_fontsize=args.label_fontsize,
            label_max_points=args.label_max_points,
            label_bbox=args.label_bbox,
            label_bbox_alpha=args.label_bbox_alpha,
            label_bbox_color=args.label_bbox_color,
            label_bbox_edgecolor=args.label_bbox_edgecolor,
            label_bbox_pad=args.label_bbox_pad,
            label_bbox_boxstyle=args.label_bbox_boxstyle,
            label_arrow=args.label_arrow,
            label_arrow_color=args.label_arrow_color,
            label_arrow_width=args.label_arrow_width,
            label_arrow_alpha=args.label_arrow_alpha,
            label_arrow_shrinkA=args.label_arrow_shrinkA,
            label_arrow_shrinkB=args.label_arrow_shrinkB,
            label_arrow_anchor=args.label_arrow_anchor,
            adjust_expand_points=args.adjust_expand_points,
            adjust_expand_text=args.adjust_expand_text,
            adjust_force_points=args.adjust_force_points,
            adjust_force_text=args.adjust_force_text,
            adjust_lim=args.adjust_lim,
            adjust_avoid_self=args.adjust_avoid_self,
            adjust_only_move_points=args.adjust_only_move_points,
            adjust_only_move_text=args.adjust_only_move_text,
        )

    # ---- Phylum-colored variants (if taxonomy provided) ----
    if tax_df is not None:
        # For Type ISA
        type_tax = type_sig.merge(tax_df, left_on="ASV_ID", right_index=True, how="left")
        phyla = type_tax["Phylum"].dropna().unique().tolist()
        phyl_pal = {p: c for p, c in zip(phyla, sns.color_palette('tab20', len(phyla)).as_hex())}
        type_tax.to_csv(tables_outdir / "type_group_ISA_with_taxonomy.tsv", sep="\t", index=False)
        plot_p_vs_stat_no_overlap(
            type_tax,
            plots_outdir / "type_group_ISA_plot_Phylum.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="Phylum", type_palette=phyl_pal,
            legend_color_title="Phylum",
            plot_size_in=(args.plot_width, args.plot_height),
        )

        # Combined + taxonomy
        comb_tax = combined.merge(tax_df, left_on="ASV_ID", right_index=True, how="left")
        comb_tax.to_csv(tables_outdir / "Combined_ISA_with_taxonomy.tsv", sep="\t", index=False)
        plot_p_vs_stat_no_overlap(
            comb_tax,
            plots_outdir / "Combined_ISA_plot_Phylum.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="Phylum", # style_col="status_label",
            type_palette=phyl_pal, # marker_dict=status_markers,
            legend_color_title="Phylum", # legend_marker_title="Status",
            plot_size_in=(args.plot_width, args.plot_height),
        )

        plot_p_vs_stat_no_overlap(
            comb_tax,
            plots_outdir / "Combined_ISA_plot_Phylum_labelled.svg",
            x_col="type_stat", y_col="type_log_p",
            hue_col="Phylum", # style_col="status_label",
            type_palette=phyl_pal, # marker_dict=status_markers,
            legend_color_title="Phylum", # legend_marker_title="Status",
            plot_size_in=(args.plot_width, args.plot_height),
            label_col="label_any_sig",
            label_fontsize=args.label_fontsize,
            label_max_points=args.label_max_points,           
            label_bbox=args.label_bbox,
            label_bbox_alpha=args.label_bbox_alpha,
            label_bbox_color=args.label_bbox_color,
            label_bbox_edgecolor=args.label_bbox_edgecolor,
            label_bbox_pad=args.label_bbox_pad,
            label_bbox_boxstyle=args.label_bbox_boxstyle,
            label_arrow=args.label_arrow,
            label_arrow_color=args.label_arrow_color,
            label_arrow_width=args.label_arrow_width,
            label_arrow_alpha=args.label_arrow_alpha,
            label_arrow_shrinkA=args.label_arrow_shrinkA,
            label_arrow_shrinkB=args.label_arrow_shrinkB,
            label_arrow_anchor=args.label_arrow_anchor,
            adjust_expand_points=args.adjust_expand_points,
            adjust_expand_text=args.adjust_expand_text,
            adjust_force_points=args.adjust_force_points,
            adjust_force_text=args.adjust_force_text,
            adjust_lim=args.adjust_lim,
            adjust_avoid_self=args.adjust_avoid_self,
            adjust_only_move_points=args.adjust_only_move_points,
            adjust_only_move_text=args.adjust_only_move_text,
        )

        # Venn Combined + taxonomy
        if venn_combined is not None:
            venn_comb_tax = venn_combined.merge(tax_df, left_on="ASV_ID", right_index=True, how="left")
            venn_comb_tax.to_csv(tables_outdir / "Combined_Venn_with_taxonomy.tsv", sep="\t", index=False)
            plot_p_vs_stat_no_overlap(
                venn_comb_tax,
                plots_outdir / "Combined_Venn_plot_Phylum.svg",
                x_col="type_stat", y_col="type_log_p",
                hue_col="Phylum", # style_col="status_label",
                type_palette=phyl_pal, # marker_dict=status_markers,
                legend_color_title="Phylum", # legend_marker_title="Status",
                plot_size_in=(args.plot_width, args.plot_height),
            )

            plot_p_vs_stat_no_overlap(
                venn_comb_tax,
                plots_outdir / "Combined_Venn_plot_Phylum_labelled.svg",
                x_col="type_stat", y_col="type_log_p",
                hue_col="Phylum", # style_col="status_label",
                type_palette=phyl_pal, # marker_dict=status_markers,
                legend_color_title="Phylum", # legend_marker_title="Status",
                plot_size_in=(args.plot_width, args.plot_height),
                label_col="label_any_sig",
                label_fontsize=args.label_fontsize,
                label_max_points=args.label_max_points,           
                label_bbox=args.label_bbox,
                label_bbox_alpha=args.label_bbox_alpha,
                label_bbox_color=args.label_bbox_color,
                label_bbox_edgecolor=args.label_bbox_edgecolor,
                label_bbox_pad=args.label_bbox_pad,
                label_bbox_boxstyle=args.label_bbox_boxstyle,
                label_arrow=args.label_arrow,
                label_arrow_color=args.label_arrow_color,
                label_arrow_width=args.label_arrow_width,
                label_arrow_alpha=args.label_arrow_alpha,
                label_arrow_shrinkA=args.label_arrow_shrinkA,
                label_arrow_shrinkB=args.label_arrow_shrinkB,
                label_arrow_anchor=args.label_arrow_anchor,
                adjust_expand_points=args.adjust_expand_points,
                adjust_expand_text=args.adjust_expand_text,
                adjust_force_points=args.adjust_force_points,
                adjust_force_text=args.adjust_force_text,
                adjust_lim=args.adjust_lim,
                adjust_avoid_self=args.adjust_avoid_self,
                adjust_only_move_points=args.adjust_only_move_points,
                adjust_only_move_text=args.adjust_only_move_text,
            )

    print(f"Done. Outputs in: {outdir}")

if __name__ == "__main__":
    main()
