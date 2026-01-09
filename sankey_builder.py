# file: data_loss_sankey.py
# edits:
#   1) Replace build_sankey() with the version below (Ctrl+F: "def build_sankey(")
#   2) Add build_sample_stage_df() helper (place it just ABOVE "def get_parser()")
#   3) In main() compute-mode, build sample_stage_df and pass it into build_sankey()
#
# NOTE: In manual mode, sample-level linking is unavailable, so the bottom block is omitted.

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional

import pandas as pd
import plotly.graph_objects as go

# Optional aesthetics (kept simple; plots are Plotly HTML)
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Global aesthetics (safe no-ops if MPL not used) ----------
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Source Sans Pro'
sns.set_theme()
sns.set_style("white")


# =========================
# Utility parsers / helpers
# =========================
def parse_kv_csv(s: str, val_cast=int) -> Dict[str, object]:
    """
    Parse 'A:1,B:2' into dict. Whitespace tolerated. Empty string -> {}.
    """
    out: Dict[str, object] = {}
    if not s:
        return out
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Expected key:value pair, got '{item}'")
        k, v = item.split(':', 1)
        k = k.strip()
        v = v.strip()
        out[k] = val_cast(v) if val_cast is not None else v
    return out


def parse_steps_csv(s: str) -> Tuple[List[str], List[int]]:
    """
    Parse 'StepA:100,StepB:90,...' -> (['StepA','StepB',...],[100,90,...])
    """
    d = parse_kv_csv(s, val_cast=int)
    return list(d.keys()), list(d.values())


def extract_sample_id_from_path(path_str: str, suffix_underscores: Optional[int] = None,
                                regex: Optional[str] = None) -> str:
    """
    Extract a sample id from a file path.
    - If regex is provided: return first capturing group.
    - Else if suffix_underscores is provided: chop that many underscore-delimited tokens from end.
      e.g., name='ABC_1_2_3_4.fastq.gz', n=4 -> 'ABC'
    - Else return basename without extension(s).
    """
    import re as _re
    base = os.path.basename(path_str)
    if regex:
        m = _re.search(regex, path_str)
        if not m or not m.groups():
            raise ValueError(f"Regex did not match or capture a group: {regex} for {path_str}")
        return m.group(1)
    if suffix_underscores is not None:
        stem = base
        # Remove common extensions
        for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq', '.gz', '.tsv', '.csv', '.txt'):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        parts = stem.split('_')
        if len(parts) <= suffix_underscores:
            return parts[0]
        return '_'.join(parts[: len(parts) - suffix_underscores])
    # Fallback: strip extensions
    if '.' in base:
        return base.split('.')[0]
    return base


def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


# =========================
# I/O readers (compute mode)
# =========================
def read_metadata(path: Path, samp_col: str, type_col: str,
                  keep_types: Optional[Sequence[str]]) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=0)
    if keep_types:
        df = df[df[type_col].isin(keep_types)].copy()
    # Make sure sample ids are strings
    df[samp_col] = df[samp_col].astype(str)
    return df


def read_fastq_stats(path: Path, samp_col: str,
                     id_suffix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Expects columns: file, num_seqs
    """
    df = pd.read_csv(path, sep='\t', header=0)
    if 'file' not in df or 'num_seqs' not in df:
        raise ValueError(f"{path} must contain columns: file, num_seqs")
    df[samp_col] = df['file'].apply(lambda x: extract_sample_id_from_path(x, id_suffix_underscores, id_regex))
    out = df.groupby(samp_col, as_index=False)['num_seqs'].sum()
    return out


def read_asv_matrix(path: Path, samp_col: str,
                    id_suffix_underscores: Optional[int], id_regex: Optional[str]) -> pd.DataFrame:
    """
    Input: wide matrix (rows=ASVs, columns=samples), counts.
    Returns long: [ASV_ID, samp_col, count] with count>0
    """
    df = pd.read_csv(path, sep='\t', header=0, index_col=0)
    long_df = df.stack().reset_index()
    long_df.columns = ['ASV_ID', 'sample_raw', 'count']
    long_df = long_df[long_df['count'] > 0].copy()
    long_df[samp_col] = long_df['sample_raw'].apply(lambda x: extract_sample_id_from_path(x, id_suffix_underscores, id_regex))
    long_df.drop(columns=['sample_raw'], inplace=True)
    return long_df


def group_counts_by_type(long_counts: pd.DataFrame, metadata: pd.DataFrame,
                         samp_col: str, type_col: str) -> pd.DataFrame:
    merged = long_counts.merge(metadata[[samp_col, type_col]], on=samp_col, how='inner')
    grp = merged.groupby(type_col, as_index=False)['count'].sum()
    grp.rename(columns={'count': 'num_reads'}, inplace=True)
    return grp


# =========================
# NEW: per-sample stage table (compute-mode)
# =========================
def build_sample_stage_df(
    meta: pd.DataFrame,
    raw_df: pd.DataFrame,
    filt_df: pd.DataFrame,
    asv_erc_long: pd.DataFrame,
    asv_decon_long: pd.DataFrame,
    asv_micro_long: pd.DataFrame,
    asv_mito_long: pd.DataFrame,
    samp_col: str,
    type_col: str,
) -> pd.DataFrame:
    """
    Returns a per-sample table linking each stage's counts to samples.
    Columns:
      - samp_col
      - type_col
      - qc_raw_pairs
      - error_correct_filtered_reads
      - asv_raw_reads
      - asv_decon_reads
      - asv_micro_reads
    """

    # base: unique samples + types from metadata (filtered to keep_types upstream)
    base = meta[[samp_col, type_col]].drop_duplicates().copy()
    base[samp_col] = base[samp_col].astype(str)

    # QC raw pairs
    raw_df = raw_df[[samp_col, "num_seqs"]].copy()
    raw_df[samp_col] = raw_df[samp_col].astype(str)
    raw_df["raw_reads"] = (raw_df["num_seqs"].fillna(0).astype("int64") // 2).astype("int64")
    raw_df = raw_df[[samp_col, "raw_reads"]]

    # Filtered reads (already single-end in your semantics)
    flt = filt_df[[samp_col, "num_seqs"]].copy()
    flt[samp_col] = flt[samp_col].astype(str)
    flt["qc_reads"] = flt["num_seqs"].fillna(0).astype("int64")
    flt = flt[[samp_col, "qc_reads"]]

    def asv_sum(long_df: pd.DataFrame, out_col: str) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame({samp_col: [], out_col: []})
        tmp = long_df.groupby(samp_col, as_index=False)["count"].sum().copy()
        tmp[samp_col] = tmp[samp_col].astype(str)
        tmp[out_col] = tmp["count"].fillna(0).astype("int64")
        return tmp[[samp_col, out_col]]

    a_erc = asv_sum(asv_erc_long, "error_correct_reads")
    a_dec = asv_sum(asv_decon_long, "asv_decon_reads")
    a_mic = asv_sum(asv_micro_long, "asv_micro_reads")
    a_mit = asv_sum(asv_mito_long, "asv_mito_reads")

    # Merge all stages onto base
    out = base.merge(raw_df, on=samp_col, how="left") \
              .merge(flt, on=samp_col, how="left") \
              .merge(a_erc, on=samp_col, how="left") \
              .merge(a_dec, on=samp_col, how="left") \
              .merge(a_mic, on=samp_col, how="left") \
              .merge(a_mit, on=samp_col, how="left")

    out['asv_unassigned_tax'] = out['asv_decon_reads'] - (out['asv_micro_reads'] + out['asv_mito_reads'])
    # Fill missing with 0 and cast to int
    for c in ["raw_reads", "qc_reads", "error_correct_reads", "asv_decon_reads", "asv_micro_reads", "asv_mito_reads", "asv_unassigned_tax"]:
        out[c] = out[c].fillna(0).astype("int64")

    # Nice default ordering: by type then sample
    out = out.sort_values([type_col, samp_col], kind="mergesort").reset_index(drop=True)
    return out


# =========================
# Sankey construction
# =========================
def build_sankey(
    steps: List[str],
    counts: List[int],
    lmp_in: Dict[str, int],
    lmp_out: Dict[str, int],
    palette: Dict[str, str],
    title: str,
    output_html: Path,
    labeled: bool,
    sample_stage_df: Optional[pd.DataFrame] = None,
    samp_col: str = "lmp_id",
    type_col: str = "type_group",
) -> None:
    """
    Two behaviors:
      1) If sample_stage_df is provided (compute-mode): build the NEW Sankey:
           raw split by type_group
           -> raw total
           -> after QC
           -> after error correction
           -> after decontamination
           -> after off-target filtering
           -> finished data
           -> finished split by type_group

         Semantics:
           - asv_decon_reads = target + non-target
           - finished/off-target output = asv_micro_reads
           - non-target removed at off-target step = asv_decon_reads - asv_micro_reads

      2) Otherwise (manual-mode): keep the OLD type-based Sankey behavior.

    Also: if sample_stage_df is provided, append the bottom sortable per-sample table block (unchanged).
    """

    # ---------------------------------------------------------------------
    # Helper: minimal HTML escape
    def _esc(x: object) -> str:
        s = str(x)
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace('"', "&quot;").replace("'", "&#39;"))

    # ---------------------------------------------------------------------
    # NEW: table-driven Sankey (compute-mode) — SPLIT BY TYPE_GROUP (NOT SAMPLE)
    if sample_stage_df is not None and not sample_stage_df.empty:
        df = sample_stage_df.copy()

        required_cols = {type_col, "raw_reads", "qc_reads", "error_correct_reads", "asv_decon_reads", "asv_micro_reads"}
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"sample_stage_df missing required columns: {missing}")

        df[type_col] = df[type_col].astype(str)
        for c in ["raw_reads", "qc_reads", "error_correct_reads", "asv_decon_reads", "asv_micro_reads"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

        # per-type (for split ends)
        by_type = df.groupby(type_col, as_index=False)[["raw_reads", "asv_micro_reads"]].sum()
        types = list(by_type[type_col].astype(str).tolist())

        raw_by_type = {t: int(by_type.loc[by_type[type_col] == t, "raw_reads"].iloc[0]) for t in types}
        fin_by_type = {t: int(by_type.loc[by_type[type_col] == t, "asv_micro_reads"].iloc[0]) for t in types}

        # totals (for the middle chain)
        total_raw = int(df["raw_reads"].sum())
        total_qc = int(df["qc_reads"].sum())
        total_ec = int(df["error_correct_reads"].sum())
        total_decon = int(df["asv_decon_reads"].sum())
        total_micro = int(df["asv_micro_reads"].sum())

        # Nodes / links
        nodes: List[Dict[str, str]] = []
        links: List[Dict[str, int]] = []
        link_colors: List[str] = []
        node_idx: Dict[str, int] = {}

        def add_node(key: str, label_txt: str, color: str) -> int:
            nodes.append({"key": key, "label": label_txt, "color": color})
            node_idx[key] = len(nodes) - 1
            return node_idx[key]

        def add_link(src_key: str, dst_key: str, value: int, color: str = "grey") -> None:
            if value <= 0:
                return
            links.append({"source": node_idx[src_key], "target": node_idx[dst_key], "value": int(value)})
            link_colors.append(color)

        # Left: raw split by type_group
        for t in types:
            v = raw_by_type.get(t, 0)
            col = palette.get(t, "black")
            lab = f"{t} ({v})" if labeled else ""
            add_node(f"raw_type::{t}", lab, col)

        # Middle: stage totals
        add_node("stage::raw_total", f"Raw total ({total_raw})" if labeled else "", "black")
        add_node("stage::qc_total", f"Quality Control ({total_qc})" if labeled else "", "black")
        add_node("stage::ec_total", f"Error Correction ({total_ec})" if labeled else "", "black")
        add_node("stage::decon_total", f"Decontamination ({total_decon})" if labeled else "", "black")
        add_node("stage::offtarget_total", f"Non-Target Filtering ({total_micro})" if labeled else "", "black")
        add_node("stage::finished_total", f"Finished Data ({total_micro})" if labeled else "", "black")

        # Right: finished split by type_group (asv_micro_reads)
        for t in types:
            v = fin_by_type.get(t, 0)
            col = palette.get(t, "black")
            lab = f"{t} ({v})" if labeled else ""
            add_node(f"fin_type::{t}", lab, col)

        # raw split -> raw total
        for t in types:
            add_link(f"raw_type::{t}", "stage::raw_total", raw_by_type.get(t, 0), "grey")

        # Chain totals with loss nodes
        add_link("stage::raw_total", "stage::qc_total", total_qc, "grey")
        if total_raw > total_qc:
            loss_val = total_raw - total_qc
            add_node("loss::after_raw", f"Loss after Quality Control ({loss_val})" if labeled else "", "lightgrey")
            add_link("stage::raw_total", "loss::after_raw", loss_val, "lightgrey")

        add_link("stage::qc_total", "stage::ec_total", total_ec, "grey")
        if total_qc > total_ec:
            loss_val = total_qc - total_ec
            add_node("loss::after_qc", f"Loss after Error Correction ({loss_val})" if labeled else "", "lightgrey")
            add_link("stage::qc_total", "loss::after_qc", loss_val, "lightgrey")

        add_link("stage::ec_total", "stage::decon_total", total_decon, "grey")
        if total_ec > total_decon:
            loss_val = total_ec - total_decon
            add_node("loss::after_ec", f"Loss after Decontamination ({loss_val})" if labeled else "", "lightgrey")
            add_link("stage::ec_total", "loss::after_ec", loss_val, "lightgrey")

        # This is the off-target removal step: decon (target+non-target) -> micro (target only)
        add_link("stage::decon_total", "stage::offtarget_total", total_micro, "grey")
        if total_decon > total_micro:
            loss_val = total_decon - total_micro
            add_node("loss::nontarget_removed", f"Lost after Non-target Filtering ({loss_val})" if labeled else "", "lightgrey")
            add_link("stage::decon_total", "loss::nontarget_removed", loss_val, "lightgrey")

        # Off-target -> Finished (same)
        add_link("stage::offtarget_total", "stage::finished_total", total_micro, "grey")

        # finished total -> finished split (by type_group)
        for t in types:
            add_link("stage::finished_total", f"fin_type::{t}", fin_by_type.get(t, 0), "grey")

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=18,
                line=dict(color="black", width=0.5),
                label=[n["label"] for n in nodes],
                color=[n["color"] for n in nodes],
            ),
            link=dict(
                source=[l["source"] for l in links],
                target=[l["target"] for l in links],
                value=[l["value"] for l in links],
                color=link_colors,
            ),
        )])
        fig.update_layout(title_text=title, font_size=12)
        output_html.parent.mkdir(parents=True, exist_ok=True)

        # Write HTML so we can append bottom table block (per-sample)
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")

        # Bottom block: keep as per-sample table; just fix stage_specs to match actual cols
        stage_specs = [
            ("Raw total", "raw_reads"),
            ("After QC", "qc_reads"),
            ("After Error Correction", "error_correct_reads"),
            ("After Decontamination", "asv_decon_reads"),
            ("After Off-Target Filtering / Finished", "asv_micro_reads"),
        ]

        required_for_block = {samp_col, type_col} | {c for _, c in stage_specs}
        missing_block = [c for c in required_for_block if c not in df.columns]
        if not missing_block:
            headers = [samp_col, type_col] + [c for _, c in stage_specs]
            totals = {label: int(df[col].sum()) for label, col in stage_specs}

            rows_html = []
            for _, r in df.iterrows():
                rows_html.append("<tr>" + "".join([f"<td>{_esc(r[h])}</td>" for h in headers]) + "</tr>")
            rows_html = "\n".join(rows_html)

            links_html = []
            for label, col in stage_specs:
                col_index = headers.index(col)
                links_html.append(
                    f'<li><a href="#sample_counts" onclick="sortSampleTable({col_index});">'
                    f'{_esc(label)}: {totals[label]}</a></li>'
                )
            links_html = "\n".join(links_html)

            bottom_block = f"""
<div id="counts_to_samples_block" style="max-width: 1100px; margin: 24px auto 48px auto; padding: 16px;">
  <a id="sample_counts"></a>
  <h2 style="margin: 0 0 8px 0;">Counts ↔ Samples (all stages)</h2>
  <p style="margin: 0 0 12px 0;">Click a stage total to jump here and sort the table by that stage.</p>

  <ul style="margin: 0 0 16px 18px;">
    {links_html}
  </ul>

  <style>
    #sample_counts_table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    #sample_counts_table th, #sample_counts_table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    #sample_counts_table th {{ position: sticky; top: 0; background: #f7f7f7; cursor: pointer; }}
    #sample_counts_table tr:nth-child(even) {{ background: #fafafa; }}
    .col-hilite {{ outline: 2px solid #000; }}
  </style>

  <table id="sample_counts_table">
    <thead>
      <tr>
        {''.join([f'<th onclick="sortSampleTable({i})">{_esc(h)}</th>' for i, h in enumerate(headers)])}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <script>
    function sortSampleTable(colIndex) {{
      const table = document.getElementById("sample_counts_table");
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);

      Array.from(table.tHead.rows[0].cells).forEach(th => th.classList.remove("col-hilite"));
      table.tHead.rows[0].cells[colIndex].classList.add("col-hilite");

      const isNumeric = rows.every(r => {{
        const v = r.cells[colIndex].innerText.trim();
        return v === "" || !isNaN(Number(v));
      }});

      rows.sort((a, b) => {{
        const av = a.cells[colIndex].innerText.trim();
        const bv = b.cells[colIndex].innerText.trim();

        if (isNumeric) {{
          const an = Number(av || 0);
          const bn = Number(bv || 0);
          if (bn !== an) return bn - an;
          return a.cells[0].innerText.localeCompare(b.cells[0].innerText);
        }} else {{
          const cmp = av.localeCompare(bv);
          if (cmp !== 0) return cmp;
          return a.cells[0].innerText.localeCompare(b.cells[0].innerText);
        }}
      }});

      rows.forEach(r => tbody.appendChild(r));
    }}
  </script>
</div>
"""
            if "</body>" in html:
                html = html.replace("</body>", bottom_block + "\n</body>")
            else:
                html += bottom_block

        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✔ Sankey saved: {output_html}")
        return

    # ---------------------------------------------------------------------
    # OLD: type-based Sankey (manual-mode / no sample_stage_df) — unchanged
    nodes: List[Dict[str, str]] = []
    links: List[Dict[str, int]] = []
    node_idx: Dict[Tuple[str, str], int] = {}
    link_colors: List[str] = []

    for k, v in lmp_in.items():
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": palette.get(k, "black")})
        node_idx[(k, "in")] = len(nodes) - 1

    for step, cnt in zip(steps, counts):
        nodes.append({"label": f"{step} ({cnt})" if labeled else "", "color": "black"})
        node_idx[(step, "proc")] = len(nodes) - 1

    for k, v in lmp_out.items():
        nodes.append({"label": f"{k} ({v})" if labeled else "", "color": palette.get(k, "black")})
        node_idx[(k, "out")] = len(nodes) - 1

    first_step = steps[0]
    for k, v in lmp_in.items():
        links.append({"source": node_idx[(k, "in")], "target": node_idx[(first_step, "proc")], "value": v})
        link_colors.append("grey")

    for i in range(len(steps) - 1):
        s, t = steps[i], steps[i + 1]
        links.append({"source": node_idx[(s, "proc")], "target": node_idx[(t, "proc")], "value": counts[i + 1]})
        link_colors.append("grey")

        if counts[i] > counts[i + 1]:
            loss_val = counts[i] - counts[i + 1]
            loss_label = f"Loss after {s} ({loss_val})" if labeled else ""
            nodes.append({"label": loss_label, "color": "lightgrey"})
            loss_idx = len(nodes) - 1
            links.append({"source": node_idx[(s, "proc")], "target": loss_idx, "value": loss_val})
            link_colors.append("lightgrey")

    last_step = steps[-1]
    for k, v in lmp_out.items():
        links.append({"source": node_idx[(last_step, "proc")], "target": node_idx[(k, "out")], "value": v})
        link_colors.append("grey")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"] for n in nodes],
        ),
        link=dict(
            source=[l["source"] for l in links],
            target=[l["target"] for l in links],
            value=[l["value"] for l in links],
            color=link_colors,
        ),
    )])
    fig.update_layout(title_text=title, font_size=12)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"✔ Sankey saved: {output_html}")


# =========================
# CLI
# =========================
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate Sankey diagrams for data loss/flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Mode selection (manual vs compute)
    p.add_argument("--steps", default="", help="Manual mode: 'StepA:100,StepB:90,...' (order preserved)")
    p.add_argument("--lmp-in", default="", help="Manual mode: input groups 'Type:count,TypeB:count,...'")
    p.add_argument("--lmp-out", default="", help="Manual mode: output groups 'Type:count,TypeB:count,...'")

    # --- Compute mode inputs
    io = p.add_argument_group("Compute Mode Inputs")
    io.add_argument("--data-dir", type=Path, help="Project root (used to resolve defaults)")
    io.add_argument("--sub-dir", default="spark_combined_output", help="Subdir under data-dir for outputs/stats")
    io.add_argument("--metadata", type=Path, help="TSV with sample metadata")
    io.add_argument("--samp-col", default="lmp_id", help="Sample column name in metadata")
    io.add_argument("--type-col", default="type_group", help="Grouping column in metadata")
    io.add_argument("--keep-types", default="Oral Rinse,Lung Brush,BAL,Skin Brush,Scope Flush",
                    help="Comma-separated list; if empty, keep all types")

    io.add_argument("--fastq-stats", default="stats/fastq_stats.tsv",
                    help="Path (relative to sub-dir or absolute) to raw fastq stats TSV")
    io.add_argument("--fastq-id-suffix-underscores", type=int, default=4,
                    help="Chop N underscore tokens from end to form sample id (raw fastq)")
    io.add_argument("--fastq-id-regex", default="",
                    help="Regex with one capture group to extract sample id from raw fastq 'file' path")

    io.add_argument("--filtered-stats", default="stats/filtered_fastqs.tsv",
                    help="Path to filtered fastq stats TSV")
    io.add_argument("--filtered-id-suffix-underscores", type=int, default=2,
                    help="Chop N underscore tokens (filtered reads)")
    io.add_argument("--filtered-id-regex", default="", help="Regex for filtered sample id extraction")

    io.add_argument("--asv-erc", default="ASVs/ASV_filtered.tsv", help="Error corrected ASV")
    io.add_argument("--asv-id-suffix-underscores", type=int, default=2,
                    help="Chop N underscore tokens (ASV matrices)")
    io.add_argument("--asv-id-regex", default="", help="Regex for ASV sample id extraction")

    io.add_argument("--asv-decon", default="ASVs/ASV_target.decon.tsv", help="ASV after decontamination")
    io.add_argument("--asv-micro", default="ASVs/ASV_target.micro.tsv", help="ASV microbial (finished)")
    io.add_argument("--asv-mito", default="ASVs/ASV_target.mito.tsv", help="ASV mitochondrial (finished)")

    # --- Appearance / output
    out = p.add_argument_group("Output")
    out.add_argument("--palette", default="Scope Flush:#E69F00,Skin Brush:#CC79A7,Lung Brush:#009E73,BAL:#0072B2,Oral Rinse:#6A3D9A,Failed-QC:lightgray",
                     help="Comma-separated 'Group:#HEX' list")
    out.add_argument("--title", default="Data Loss Flow", help="Plot title")
    out.add_argument("--output-prefix", default="data_loss_sankey",
                     help="Output prefix ('.html' appended automatically)")
    out.add_argument("--make-labeled", action="store_true", help="Create labeled-node HTML")
    out.add_argument("--make-unlabeled", action="store_true", help="Create unlabeled-node HTML")

    # --- Misc
    p.add_argument("--verbose", action="store_true", help="Verbose logs")

    return p


def main():
    args = get_parser().parse_args()

    # Palette
    palette = parse_kv_csv(args.palette, val_cast=None) if args.palette else {}

    # If manual steps passed, run manual mode
    manual_mode = bool(args.steps.strip())
    if manual_mode:
        steps, counts = parse_steps_csv(args.steps)
        lmp_in = parse_kv_csv(args.lmp_in, val_cast=int)
        lmp_out = parse_kv_csv(args.lmp_out, val_cast=int)
        if not args.make_labeled and not args.make_unlabeled:
            args.make_labeled = True  # default to at least one output

        out_pref = Path(args.output_prefix)
        if args.make_labeled:
            build_sankey(
                steps, counts, lmp_in, lmp_out, palette, args.title,
                out_pref.with_suffix(".label.html"), True,
                sample_stage_df=None,  # manual mode has no sample-level info
                samp_col=args.samp_col,
                type_col=args.type_col,
            )
        if args.make_unlabeled:
            build_sankey(
                steps, counts, lmp_in, lmp_out, palette, args.title,
                out_pref.with_suffix(".html"), False,
                sample_stage_df=None,
                samp_col=args.samp_col,
                type_col=args.type_col,
            )
        return

    # ---- Compute mode ----
    if not args.data_dir:
        raise SystemExit("--data-dir is required in compute mode")
    data_dir: Path = args.data_dir

    # Resolve default paths if relative
    def resolve(rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return data_dir / args.sub_dir / rel_or_abs

    metadata_path = args.metadata or (data_dir / "ref_db" / "spark_metadata.tsv")
    fastq_stats_path = resolve(args.fastq_stats)
    filtered_stats_path = resolve(args.filtered_stats)
    asv_erc_path = resolve(args.asv_erc)
    asv_decon_path = resolve(args.asv_decon)
    asv_micro_path = resolve(args.asv_micro)
    asv_mito_path = resolve(args.asv_mito)

    keep_types = [t.strip() for t in args.keep_types.split(',')] if args.keep_types.strip() else None

    if args.verbose:
        print(f"[i] Metadata: {metadata_path}")
        print(f"[i] Raw fastq stats: {fastq_stats_path}")
        print(f"[i] Filtered stats: {filtered_stats_path}")
        print(f"[i] ASV error-corr: {asv_erc_path}")
        print(f"[i] ASV decon: {asv_decon_path}")
        print(f"[i] ASV micro: {asv_micro_path}")
        print(f"[i] ASV mito: {asv_mito_path}")

    # Read metadata and filter by types
    meta = read_metadata(metadata_path, args.samp_col, args.type_col, keep_types)

    # Raw reads (pairs): sum num_seqs across files, then /2
    raw_df = read_fastq_stats(
        fastq_stats_path, args.samp_col,
        args.fastq_id_suffix_underscores,
        args.fastq_id_regex or None
    )
    raw_df = raw_df.loc[raw_df['sample'].isin(meta['sample'])]
    raw_reads_total = int(raw_df['num_seqs'].sum() // 2)
 
    # Filtered reads (already single-end counts in your script)
    filt_df = read_fastq_stats(
        filtered_stats_path, args.samp_col,
        args.filtered_id_suffix_underscores,
        args.filtered_id_regex or None
    )
    filt_df['sample'] = [x.split('.', 1)[0] for x in filt_df['sample']]
    filt_df = filt_df.loc[filt_df['sample'].isin(meta['sample'])]
    filt_reads_total = int(filt_df['num_seqs'].sum())

    # ASV matrices -> long -> merge -> sum
    asv_erc_long = read_asv_matrix(asv_erc_path, args.samp_col,
                                   args.asv_id_suffix_underscores, args.asv_id_regex or None)
    asv_decon_long = read_asv_matrix(asv_decon_path, args.samp_col,
                                     args.asv_id_suffix_underscores, args.asv_id_regex or None)
    asv_micro_long = read_asv_matrix(asv_micro_path, args.samp_col,
                                     args.asv_id_suffix_underscores, args.asv_id_regex or None)
    asv_mito_long = read_asv_matrix(asv_mito_path, args.samp_col,
                                     args.asv_id_suffix_underscores, args.asv_id_regex or None)
    
    # NEW: build per-sample stage table for bottom block
    sample_stage_df = build_sample_stage_df(
        meta=meta,
        raw_df=raw_df,
        filt_df=filt_df,
        asv_erc_long=asv_erc_long,
        asv_decon_long=asv_decon_long,
        asv_micro_long=asv_micro_long,
        asv_mito_long=asv_mito_long,
        samp_col=args.samp_col,
        type_col=args.type_col,
    )
    
    # also save sample-stage table to disk
    out_pref = (args.data_dir / args.sub_dir / "M6_downstream_analysis/metadata_summaries/tables" /args.output_prefix) if args.data_dir else Path(args.output_prefix)
    table_out = out_pref.with_suffix(".sample_stage_counts.tsv")
    table_out.parent.mkdir(parents=True, exist_ok=True)
    sample_stage_df.to_csv(table_out, sep="\t", index=False)
    print(f"✔ Sample-stage table saved: {table_out}")


    # Sum by type (group)
    raw_by_type = raw_df.merge(meta[[args.samp_col, args.type_col]], on=args.samp_col, how='inner') \
                        .groupby(args.type_col, as_index=False)['num_seqs'].sum()
    raw_by_type['num_reads'] = (raw_by_type['num_seqs'] // 2).astype(int)

    filt_by_type = filt_df.merge(meta[[args.samp_col, args.type_col]], on=args.samp_col, how='inner') \
                          .groupby(args.type_col, as_index=False)['num_seqs'].sum()
    filt_by_type['num_reads'] = filt_by_type['num_seqs'].astype(int)

    asv_raw_by_type = group_counts_by_type(asv_erc_long, meta, args.samp_col, args.type_col)
    asv_decon_by_type = group_counts_by_type(asv_decon_long, meta, args.samp_col, args.type_col)
    asv_micro_by_type = group_counts_by_type(asv_micro_long, meta, args.samp_col, args.type_col)

    # Totals (match original semantics)
    asv_raw_reads = int(asv_raw_by_type['num_reads'].sum())
    asv_decon_reads = int(asv_decon_by_type['num_reads'].sum())
    asv_micro_reads = int(asv_micro_by_type['num_reads'].sum())

    # Steps & counts
    steps = ['Quality Control', 'Error Correction', 'Decontamination',
             'Off-Target Filtering', 'Finished Data']
    counts = [raw_reads_total, filt_reads_total, asv_raw_reads, asv_decon_reads, asv_micro_reads]

    # Groups to carry through (types)
    if keep_types:
        types = keep_types
    else:
        types = list(sorted(meta[args.type_col].unique()))

    # Input and output dicts for sankey ends
    lmp_in = {t: int(raw_by_type.loc[raw_by_type[args.type_col] == t, 'num_reads'].sum()) for t in types}
    lmp_out = {t: int(asv_micro_by_type.loc[asv_micro_by_type[args.type_col] == t, 'num_reads'].sum()) for t in types}

    if args.verbose:
        print("[i] Steps:")
        for s, c in zip(steps, counts):
            print(f"  - {s}: {c}")
        print("[i] Inputs by type:", lmp_in)
        print("[i] Outputs by type:", lmp_out)

    # Outputs
    out_pref = (args.data_dir / args.sub_dir / "M6_downstream_analysis/metadata_summaries/plots" / args.output_prefix) if args.data_dir else Path(args.output_prefix)
    # default: generate both if none chosen
    if not args.make_labeled and not args.make_unlabeled:
        args.make_labeled = True
        args.make_unlabeled = True

    if args.make_labeled:
        build_sankey(
            steps, counts, lmp_in, lmp_out, palette, args.title,
            out_pref.with_suffix(".label.html"), True,
            sample_stage_df=sample_stage_df,  # NEW
            samp_col=args.samp_col,
            type_col=args.type_col,
        )
    if args.make_unlabeled:
        build_sankey(
            steps, counts, lmp_in, lmp_out, palette, args.title,
            out_pref.with_suffix(".html"), False,
            sample_stage_df=sample_stage_df,  # NEW
            samp_col=args.samp_col,
            type_col=args.type_col,
        )


if __name__ == "__main__":
    main()
