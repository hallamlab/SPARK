#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import env_eigenvectors as ee
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import env_eigenvectors.py from biochem_modeling. "
        "Run this script from the SPARK directory (or ensure biochem_modeling is on PYTHONPATH)."
    ) from e


DERIVED_SEASON_COL = ee.DERIVED_SEASON_COL
BIOCHEM_COLOR_MAP = ee.BIOCHEM_COLOR_MAP
plot_biplot_core_and_sparse = ee.plot_biplot_core_and_sparse
plot_scree = ee.plot_scree
plot_cumvar = ee.plot_cumvar
plot_pc_scatter = ee.plot_pc_scatter
plot_top_loadings = ee.plot_top_loadings
run_pc_selection = ee.run_pc_selection


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Standalone EOF pipeline extracted from env_eigenvectors. "
            "Consumes matrix_cleaned_with_sparse + core-feature definitions and writes eof_* outputs."
        )
    )
    ap.add_argument("--matrix-cleaned", required=True, help="Path to matrix_cleaned_with_sparse.csv")
    ap.add_argument("--outdir", required=True, help="Output directory (typically same as env_eigenvectors outdir)")
    ap.add_argument("--sep", default=",", help="Input delimiter (default ',').")

    group_core = ap.add_mutually_exclusive_group(required=True)
    group_core.add_argument(
        "--core-loadings",
        default=None,
        help="Path to pca_loadings.csv from env_eigenvectors (index is core feature names).",
    )
    group_core.add_argument(
        "--core-feature-cols",
        default=None,
        help="Comma-separated core feature columns (alternative to --core-loadings).",
    )

    ap.add_argument("--id-col", default=ee.DEFAULT_ID_COL)
    ap.add_argument("--time-col", default=ee.DERIVED_TIME_COL)
    ap.add_argument("--profile-col", default="Cruise")
    ap.add_argument("--anchor-by-col", default="ALL")
    ap.add_argument("--depth-col", default="Depth")
    ap.add_argument("--anchored-depth-col", default="Depth_anchored")

    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--eof-dropna-row-thresh", type=float, default=0.25)

    # Optional: sparse override
    ap.add_argument(
        "--sparse-feature-cols",
        default=None,
        help="Optional comma-separated sparse features; if omitted, inferred from numeric non-core, non-meta columns.",
    )

    # Keep same defaults as env_eigenvectors PC-selection
    ap.add_argument("--pc-selection", action="store_true")
    ap.add_argument("--pcsel-parallel-B", type=int, default=500)
    ap.add_argument("--pcsel-parallel-quantile", type=float, default=0.90)
    ap.add_argument("--pcsel-support-min-cov", type=float, default=0.50)
    ap.add_argument("--pcsel-support-median-cov", type=float, default=0.60)
    ap.add_argument("--pcsel-support-min-n", type=int, default=3)
    ap.add_argument("--pcsel-top-frac", type=float, default=0.15)
    ap.add_argument("--pcsel-top-min", type=int, default=3)
    ap.add_argument("--pcsel-feature-clusters", type=int, default=8)
    ap.add_argument("--pcsel-coherence-min-frac", type=float, default=0.30)
    ap.add_argument("--pcsel-block-col", type=str, default="Cruise")
    ap.add_argument("--pcsel-stability-R", type=int, default=200)
    ap.add_argument("--pcsel-stability-min-load-corr", type=float, default=0.70)
    ap.add_argument("--pcsel-stability-min-score-corr", type=float, default=0.65)
    ap.add_argument("--pcsel-entropy-max", type=float, default=0.85)
    ap.add_argument("--pcsel-pr-max", type=float, default=8.0)

    return ap.parse_args()


def _parse_col_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_core_features(args: argparse.Namespace) -> List[str]:
    if args.core_feature_cols:
        return _parse_col_list(args.core_feature_cols)

    load_df = pd.read_csv(args.core_loadings, sep=args.sep, index_col=0)
    core = [str(c) for c in load_df.index.tolist()]
    if not core:
        raise ValueError(f"No core features found in --core-loadings: {args.core_loadings}")
    return core


def _infer_sparse_features(df: pd.DataFrame, core_feats: List[str], args: argparse.Namespace) -> List[str]:
    if args.sparse_feature_cols:
        return [c for c in _parse_col_list(args.sparse_feature_cols) if c in df.columns]

    numeric_cols: List[str] = []
    for c in df.columns:
        if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            numeric_cols.append(c)

    exclude = set(core_feats)
    exclude.update(
        [
            args.id_col,
            args.time_col,
            args.profile_col,
            args.depth_col,
            args.anchored_depth_col,
            args.anchor_by_col,
            "Year",
            "Month",
            "Day",
            ee.DERIVED_SEASON_COL,
            "component",
            "max_prob",
            "entropy_norm",
            "knn_mean_dist",
        ]
    )
    sparse = []
    for c in numeric_cols:
        if c in exclude:
            continue
        if c.startswith("PC") or c.startswith("resp_"):
            continue
        sparse.append(c)
    return sparse


def run_eof_pipeline(
    cfg: RunConfig,
    meta: pd.DataFrame,
    X_imp: pd.DataFrame,
    feats_kept: List[str],
    tables_dir: str,
    plots_dir: str,
    df_filt: pd.DataFrame,
    sparse_feats: List[str],
) -> None:
    """EOF PCA pipeline extracted from main; logic unchanged."""

    # ==============================================================================
    # DROP-IN EOF BLOCK (extracted into run_eof_pipeline for readability)
    # ------------------------------------------------------------------------------
    # What this adds (NO changes to the existing script required):
    #   - Builds a CRUISE-level EOF feature matrix:
    #       rows    = cruises
    #       columns = (biochem_feature × anchored_depth_bin)
    #   - Runs the SAME style of PCA QC artifacts you already generate:
    #       * explained variance table
    #       * loadings table
    #       * eigenvector/score table
    #       * scree + cumulative variance plots
    #       * PC1 vs PC2 scatter plot (if time_col exists at cruise-level)
    #       * top-loadings plots for PC1..PC5
    #   - Writes outputs as "eof_*" files into your existing tables_dir / plots_dir
    #
    # Inputs consumed (already in-scope from your pipeline above):
    #   - cfg                  : your RunConfig / argparse config
    #   - meta                 : metadata dataframe (aligned with X_imp)
    #   - X_imp                : cleaned/imputed feature matrix (rows = cruise+depth samples)
    #   - feats_kept           : list[str] of feature columns used
    #   - tables_dir, plots_dir: output directories
    #
    # Required columns in meta to build EOF rows:
    #   - a "cruise id" column (chosen automatically, see logic below)
    #   - a depth column (prefers cfg.anchored_depth_col, else cfg.depth_col)
    #
    # Notes:
    #   - This does NOT change your existing compartment PCA.
    #   - This EOF PCA is run in parallel from the same cleaned/imputed data.
    # ==============================================================================

    def _pick_cruise_col_for_eof(meta_df: pd.DataFrame) -> str:
        """
        Choose a cruise identifier column without requiring script edits.
        Preference order:
        1) cfg.profile_col (if present)
        2) cfg.anchor_by_col (if present)
        3) cfg.id_col (if present)
        4) common fallbacks: 'Cruise', 'cruise', 'cruise_id'
        5) cfg.time_col (as a last resort; not ideal if multiple cruises share same date label)
        """
        # 1) depth interpolation block col
        if getattr(cfg, "profile_col", None) and cfg.profile_col in meta_df.columns:
            return cfg.profile_col

        # 2) anchor_by_col
        if getattr(cfg, "anchor_by_col", None) and cfg.anchor_by_col in meta_df.columns:
            return cfg.anchor_by_col

        # 3) id_col
        if getattr(cfg, "id_col", None) and cfg.id_col in meta_df.columns:
            return cfg.id_col

        # 4) common fallbacks
        for c in ["Cruise", "cruise", "cruise_id", "CruiseID", "cruiseID"]:
            if c in meta_df.columns:
                return c

        # 5) time col (last resort)
        if getattr(cfg, "time_col", None) and cfg.time_col in meta_df.columns:
            return cfg.time_col

        raise ValueError(
            "EOF block could not find a cruise identifier column. "
            "Ensure meta contains one of: cfg.profile_col, cfg.anchor_by_col, cfg.id_col, or 'Cruise'."
        )


    def _pick_depth_col_for_eof(meta_df: pd.DataFrame) -> str:
        """
        Prefer anchored depth if available; else raw depth.
        """
        if getattr(cfg, "anchored_depth_col", None) and cfg.anchored_depth_col in meta_df.columns:
            return cfg.anchored_depth_col
        if getattr(cfg, "depth_col", None) and cfg.depth_col in meta_df.columns:
            return cfg.depth_col
        raise ValueError(
            "EOF block could not find a depth column in meta. "
            "Expected cfg.anchored_depth_col or cfg.depth_col to be present in meta."
        )


    def _build_cruise_eof_matrix(
        meta_df: pd.DataFrame,
        X_df: pd.DataFrame,
        cruise_col: str,
        depth_col: str,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds cruise × (feature@depth) matrix using pivot_table.
        - If multiple rows exist for same (cruise, depth), averages them.
        - Returns:
            eof_X  : wide matrix (index = cruise, columns = feature@depth)
            cruise_meta : one row per cruise with representative metadata (time/season etc. if present)
        """
        # Work on aligned copies (your pipeline already aligns meta and X_imp rowwise)
        tmp = meta_df.copy()
        tmp[cruise_col] = tmp[cruise_col].astype(str)
        tmp[depth_col] = pd.to_numeric(tmp[depth_col], errors="coerce")

        Xw = X_df[feature_cols].copy()

        # Combine for pivoting
        comb = pd.concat([tmp[[cruise_col, depth_col]].reset_index(drop=True), Xw.reset_index(drop=True)], axis=1)

        # Ensure depth bins are "clean" and sorted; keep as numeric
        comb = comb.dropna(subset=[cruise_col, depth_col])
        comb[depth_col] = pd.to_numeric(comb[depth_col], errors="coerce")
        comb = comb.dropna(subset=[depth_col])

        # Build a multi-index pivot for each feature, then flatten to feature@depth columns
        depth_vals = np.sort(comb[depth_col].unique())

        wide_parts = []
        for feat in feature_cols:
            # pivot: cruise × depth
            p = comb.pivot_table(index=cruise_col, columns=depth_col, values=feat, aggfunc="mean")
            # ensure consistent depth columns
            p = p.reindex(columns=depth_vals)
            # flatten columns
            p.columns = [f"{feat}@{int(d) if float(d).is_integer() else d}" for d in p.columns.to_list()]
            wide_parts.append(p)

        eof_X = pd.concat(wide_parts, axis=1)
        eof_X.index.name = cruise_col

        # Create cruise-level meta (first non-null per cruise for selected columns)
        keep_meta_cols = []
        for c in [cfg.time_col, DERIVED_SEASON_COL, "Year", "Month", "Day"]:
            if c in meta_df.columns and c not in keep_meta_cols:
                keep_meta_cols.append(c)

        if keep_meta_cols:
            cruise_meta = (
                meta_df.copy()
                .assign(**{cruise_col: meta_df[cruise_col].astype(str)})
                .groupby(cruise_col, as_index=False)[keep_meta_cols]
                .first()
            )
            cruise_meta = cruise_meta.set_index(cruise_col)
        else:
            cruise_meta = pd.DataFrame(index=eof_X.index)

        return eof_X, cruise_meta


    def _run_eof_pca_and_write_outputs() -> None:
        eof_cruise_col = _pick_cruise_col_for_eof(meta)
        eof_depth_col = _pick_depth_col_for_eof(meta)

        # Build EOF matrix from the SAME cleaned/imputed feature values (X_imp)
        eof_X, eof_meta = _build_cruise_eof_matrix(
            meta_df=meta.reset_index(drop=True),
            X_df=X_imp.reset_index(drop=True),
            cruise_col=eof_cruise_col,
            depth_col=eof_depth_col,
            feature_cols=feats_kept,
        )

        # Write the raw cruise×(feature@depth) matrix (pre-impute at cruise level)
        eof_X.to_csv(os.path.join(tables_dir, "eof_cruise_feature_matrix.tsv"), sep="\t", index=True)

        # Missingness QC (cruise-level matrix)
        eof_missing = pd.DataFrame({
            "feature": eof_X.columns,
            "frac_missing": eof_X.isna().mean(axis=0).values,
            "n_missing": eof_X.isna().sum(axis=0).values,
            "n_total": int(eof_X.shape[0]),
        }).sort_values(["frac_missing", "feature"], ascending=[False, True])
        eof_missing.to_csv(os.path.join(tables_dir, "eof_missingness_feature.csv"), index=False)

        # ----------------------------------------------------------------------
        # EOF Cruise-level missingness audit + DROP cruises that are too incomplete
        #
        # Why:
        #   EOF modes can be dominated by sampling/missingness artifacts if cruises with
        #   sparse depth coverage (many missing biochem@depth features) are retained.
        #
        # What this does:
        #   - Computes, for each cruise (row), how many EOF features are missing.
        #   - Writes audit tables for transparency and reproducibility.
        #   - Drops cruises with frac_missing > threshold BEFORE imputation.
        #
        # Outputs:
        #   tables/eof_missingness_by_cruise.csv
        #   tables/eof_missingness_by_cruise_with_keep_flag.csv
        #   tables/eof_dropped_cruises.csv
        #   plots/eof_missingness_by_cruise_hist.png
        #
        # Notes:
        #   - This only acts at cruise level; your upstream cleaning already addresses
        #     feature quality globally.
        #   - Default threshold is conservative (<=25% missing feature@depth entries).
        #     You can tune via cfg.eof_dropna_row_thresh if you add that flag later.
        # ----------------------------------------------------------------------

        import matplotlib.pyplot as plt

        # Default: keep cruises with at least 75% of feature@depth entries present
        # If you later add cfg.eof_dropna_row_thresh, it will override this.
        eof_row_thresh = float(getattr(cfg, "eof_dropna_row_thresh", 0.25))  # max frac missing allowed

        # Cruise-level missingness (rows)
        eof_missing_by_cruise = pd.DataFrame({
            "cruise": eof_X.index.astype(str),
            "frac_missing": eof_X.isna().mean(axis=1).values,
            "n_missing": eof_X.isna().sum(axis=1).values,
            "n_total_features": int(eof_X.shape[1]),
        }).sort_values(["frac_missing", "cruise"], ascending=[False, True])

        eof_missing_by_cruise.to_csv(
            os.path.join(tables_dir, "eof_missingness_by_cruise.csv"),
            index=False,
        )

        # Keep/drop decision
        eof_missing_by_cruise["KEEP"] = eof_missing_by_cruise["frac_missing"] <= eof_row_thresh
        eof_missing_by_cruise["drop_reason"] = np.where(
            eof_missing_by_cruise["KEEP"],
            "",
            f"frac_missing>{eof_row_thresh:g}",
        )

        eof_missing_by_cruise.to_csv(
            os.path.join(tables_dir, "eof_missingness_by_cruise_with_keep_flag.csv"),
            index=False,
        )

        dropped_cruises = eof_missing_by_cruise.loc[~eof_missing_by_cruise["KEEP"], :].copy()
        dropped_cruises.to_csv(
            os.path.join(tables_dir, "eof_dropped_cruises.csv"),
            index=False,
        )

        # Plot histogram of cruise missingness to justify the threshold
        fig = plt.figure(figsize=(7.2, 4.6))
        ax = plt.gca()
        ax.hist(eof_missing_by_cruise["frac_missing"].values, bins=30)
        ax.axvline(eof_row_thresh, linestyle="--", linewidth=2.0)
        ax.set_xlabel("Cruise-level fraction missing (across all biochem@depth features)")
        ax.set_ylabel("Number of cruises")
        ax.set_title("EOF cruise completeness distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "eof_missingness_by_cruise_hist.png"), dpi=200)
        plt.close(fig)

        # Apply filtering BEFORE imputation/scaling/PCA
        keep_cruises = eof_missing_by_cruise.loc[eof_missing_by_cruise["KEEP"], "cruise"].astype(str).tolist()
        eof_X = eof_X.loc[eof_X.index.astype(str).isin(keep_cruises), :].copy()

        # Keep eof_meta aligned (it is indexed by cruise id)
        eof_meta = eof_meta.loc[eof_meta.index.astype(str).isin(keep_cruises), :].copy()

        # Drop feature@depth columns that are all-NA after cruise filtering
        all_nan_cols = eof_X.columns[eof_X.isna().all(axis=0)].tolist()
        n_all_nan = int(len(all_nan_cols))
        if all_nan_cols:
            pd.DataFrame({"feature": all_nan_cols}).to_csv(
                os.path.join(tables_dir, "eof_dropped_features_all_nan.csv"),
                index=False,
            )
            eof_X = eof_X.drop(columns=all_nan_cols)

        if eof_X.shape[1] == 0:
            raise ValueError(
                "After EOF filtering, no feature@depth columns remain (all NaN). "
                "Inspect tables/eof_dropped_features_all_nan.csv."
            )

        # Safety: make sure we still have enough cruises
        if eof_X.shape[0] < 3:
            raise ValueError(
                "After EOF cruise-level missingness filtering, <3 cruises remain. "
                "Relax cfg.eof_dropna_row_thresh or inspect tables/eof_dropped_cruises.csv."
            )

        # Log a small summary to qc_summary_eof later (we'll still write qc_summary_eof.json below)
        n_dropped = int(dropped_cruises.shape[0])
        n_kept = int(eof_X.shape[0])
        with open(os.path.join(cfg.outdir, "eof_cruise_drop_summary.json"), "w") as f:
            json.dump(
                {
                    "eof_dropna_row_thresh": eof_row_thresh,
                    "n_cruises_before": int(eof_missing_by_cruise.shape[0]),
                    "n_cruises_dropped": n_dropped,
                    "n_cruises_kept": n_kept,
                    "dropped_audit_file": os.path.join(tables_dir, "eof_dropped_cruises.csv"),
                },
                f,
                indent=2,
            )
        # ----------------------------------------------------------------------

        # Impute at cruise-level (median per feature) — robust, minimal assumptions
        eof_fill = eof_X.median(axis=0, skipna=True)
        eof_X_imp = eof_X.fillna(eof_fill)

        if eof_X_imp.isna().any().any():
            raise ValueError(
                "EOF matrix still has NaNs after median fill. "
                "Inspect tables/eof_dropped_features_all_nan.csv and eof_missingness_feature.csv."
            )

        eof_fill.rename("impute_value_median").to_csv(os.path.join(tables_dir, "eof_impute_values_median.tsv"), sep="\t", header=True)

        # Scale at cruise-level (z-score per feature@depth)
        eof_scaler = StandardScaler(with_mean=True, with_std=True)
        eof_scaled = eof_scaler.fit_transform(eof_X_imp.values)

        eof_scaled_df = pd.DataFrame(eof_scaled, index=eof_X_imp.index, columns=eof_X_imp.columns)
        eof_scaled_df.to_csv(os.path.join(tables_dir, "eof_matrix_scaled.tsv"), sep="\t", index=True)

        # PCA (EOF)
        eof_ncomp = min(cfg.n_components, eof_scaled.shape[1], eof_scaled.shape[0])
        eof_pca = PCA(n_components=eof_ncomp, random_state=cfg.random_state)
        eof_pca.fit(eof_scaled)

        # ----------------------------------------------------------------------
        # EOF quantitative PC retention metric #1: North's rule of thumb
        #
        # Purpose:
        #   Determines whether EOF eigenvalues (modes) are statistically separable
        #   given sampling variability (finite number of cruises).
        #
        # Rule:
        #   delta_lambda_k ≈ lambda_k * sqrt(2 / N)
        #   Mode k and k+1 are "not separable" if their error bars overlap:
        #     lambda_k - delta_k <= lambda_{k+1} + delta_{k+1}
        #
        # Output:
        #   tables/eof_north_rule.csv
        #   outdir/eof_north_rule_summary.json
        # ----------------------------------------------------------------------
        def _north_rule_table(pca_obj: PCA, n_samples: int) -> Tuple[pd.DataFrame, int]:
            lam = np.asarray(pca_obj.explained_variance_, dtype=float)
            pcs = [f"PC{i}" for i in range(1, len(lam) + 1)]
            if n_samples <= 1:
                # degenerate case
                df = pd.DataFrame({
                    "PC": pcs,
                    "eigenvalue": lam,
                    "delta_lambda": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "separable_from_next": np.nan,
                })
                return df, 1

            delta = lam * np.sqrt(2.0 / float(n_samples))
            lower = lam - delta
            upper = lam + delta

            separable = []
            for k in range(len(lam)):
                if k == len(lam) - 1:
                    separable.append(np.nan)
                else:
                    ok = bool(lower[k] > upper[k + 1])  # strict separation
                    separable.append(ok)

            df = pd.DataFrame({
                "PC": pcs,
                "eigenvalue": lam,
                "delta_lambda": delta,
                "lower": lower,
                "upper": upper,
                "separable_from_next": separable,
            })

            # Suggested keep count:
            # keep PCs until the first time separation fails (overlap occurs).
            # This is a conservative "distinct modes" cutoff.
            k_keep = 1
            for k in range(len(lam) - 1):
                if bool(separable[k]):
                    k_keep = k + 1
                else:
                    break

            return df, int(k_keep)

        n_cruises = int(eof_X_imp.shape[0])  # cruises == EOF observations
        north_df, north_k_keep = _north_rule_table(eof_pca, n_samples=n_cruises)
        north_df.to_csv(os.path.join(tables_dir, "eof_north_rule.csv"), index=False)

        with open(os.path.join(cfg.outdir, "eof_north_rule_summary.json"), "w") as f:
            json.dump(
                {
                    "n_cruises": n_cruises,
                    "north_rule_suggested_k_keep_distinct_modes": north_k_keep,
                    "interpretation": (
                        "This suggests how many EOF modes are statistically separable by North's rule "
                        "(eigenvalue error bars do not overlap). Use with vertical loading profiles."
                    ),
                },
                f,
                indent=2,
            )
        # ----------------------------------------------------------------------

        # Explained variance
        evr = eof_pca.explained_variance_ratio_
        ev = eof_pca.explained_variance_
        eof_ev_tbl = pd.DataFrame({
            "PC": [f"PC{i}" for i in range(1, len(evr) + 1)],
            "explained_variance": ev,
            "explained_variance_ratio": evr,
            "cumulative_ratio": np.cumsum(evr),
        })
        eof_ev_tbl.to_csv(os.path.join(tables_dir, "eof_pca_explained_variance.csv"), index=False)

        # Loadings (feature@depth × PC)
        eof_loadings = pd.DataFrame(
            eof_pca.components_.T,
            index=eof_X_imp.columns,
            columns=[f"PC{i}" for i in range(1, eof_pca.n_components_ + 1)],
        )
        eof_loadings.to_csv(os.path.join(tables_dir, "eof_pca_loadings.csv"))

        # ------------------------------------------------------------------------------
        # DROP-IN REPLACEMENT: block-aware split-half loading congruence with PC matching
        # ------------------------------------------------------------------------------

        def _greedy_one_to_one_match(abs_corr: np.ndarray) -> List[Tuple[int, int, float]]:
            """
            Greedy one-to-one matching between columns of two loading matrices.
            abs_corr shape = (kA, kB), entries in [0, 1]
            Returns list of (iA, iB, corr) with each iA and iB used at most once.
            """
            kA, kB = abs_corr.shape
            pairs = []
            used_A = set()
            used_B = set()

            # Flatten and sort by corr descending
            flat = [(i, j, abs_corr[i, j]) for i in range(kA) for j in range(kB)]
            flat.sort(key=lambda x: x[2], reverse=True)

            for i, j, c in flat:
                if i in used_A or j in used_B:
                    continue
                used_A.add(i)
                used_B.add(j)
                pairs.append((i, j, float(c)))
                if len(used_A) == kA or len(used_B) == kB:
                    break

            # If kA != kB, some PCs may remain unmatched; that's OK.
            return pairs


        def _make_block_balanced_split(
            idx: np.ndarray,
            block: Optional[pd.Series],
            rng: np.random.Generator,
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, int]]]:
            """
            Split indices into A/B.

            If block is provided:
            - split within each block so both halves contain representation from each block
            - blocks with < 2 items are handled by random assignment to avoid failures
            Returns:
            A_idx, B_idx, audit_counts
            """
            if block is None:
                perm = rng.permutation(idx)
                half = len(perm) // 2
                A = perm[:half]
                B = perm[half:]
                audit = {"__ALL__": {"n": int(len(idx)), "nA": int(len(A)), "nB": int(len(B))}}
                return A, B, audit

            # Align and coerce block labels
            b = block.reindex(idx)
            b = b.astype("object").fillna("NA").astype(str)

            A_list = []
            B_list = []
            audit: Dict[str, Dict[str, int]] = {}

            for lab in sorted(b.unique(), key=lambda z: str(z)):
                members = idx[(b.values == lab)]
                n = len(members)
                if n == 0:
                    continue

                perm = rng.permutation(members)

                # If only 1 sample in block, assign randomly (so we don't crash)
                if n == 1:
                    if rng.random() < 0.5:
                        A_list.append(perm)
                        nA = 1
                        nB = 0
                    else:
                        B_list.append(perm)
                        nA = 0
                        nB = 1
                else:
                    half = n // 2
                    # If odd, A gets floor(n/2), B gets remaining
                    A_part = perm[:half]
                    B_part = perm[half:]
                    A_list.append(A_part)
                    B_list.append(B_part)
                    nA = len(A_part)
                    nB = len(B_part)

                audit[str(lab)] = {"n": int(n), "nA": int(nA), "nB": int(nB)}

            A = np.concatenate(A_list) if len(A_list) else np.array([], dtype=idx.dtype)
            B = np.concatenate(B_list) if len(B_list) else np.array([], dtype=idx.dtype)

            # If one side is empty (pathological), fall back to random split
            if len(A) < 3 or len(B) < 3:
                perm = rng.permutation(idx)
                half = len(perm) // 2
                A = perm[:half]
                B = perm[half:]
                audit = {"__FALLBACK_RANDOM__": {"n": int(len(idx)), "nA": int(len(A)), "nB": int(len(B))}}

            return A, B, audit


        def _split_half_loading_congruence(
            X_scaled_df: pd.DataFrame,
            n_components: int,
            meta_df: Optional[pd.DataFrame] = None,
            block_col: Optional[str] = None,
            R: int = 200,
            random_state: int = 0,
            matching: str = "greedy",   # 'greedy' (default) is robust and no extra deps
            write_split_audit_path: Optional[str] = None,
        ) -> pd.DataFrame:
            """
            Block-aware split-half loading congruence with PC matching.

            What it does:
            - Splits cruises into A/B repeatedly (R times)
            - If block_col is provided, splits within blocks (e.g., Year) to avoid time-imbalance artifacts
            - Fits PCA on XA and XB
            - Compares loadings using ABS correlation (sign-invariant)
            - Matches PCs between halves using one-to-one matching (prevents false instability from PC swapping)

            Inputs:
            - X_scaled_df: rows = cruises, cols = features (e.g., feature@depth); values already scaled
            - n_components: number of PCs to evaluate
            - meta_df: cruise-level metadata indexed by cruise id (same index as X_scaled_df)
            - block_col: e.g., 'Year' (recommended) or 'Season' / 'YearSeason'
            - R: number of split repetitions
            - matching: currently 'greedy'
            - write_split_audit_path: if provided, writes per-split block counts audit CSV

            Output:
            DataFrame with per-PC median/mean abs loading corr, plus n_splits.
            Note: PCs are labeled PC1..PCk, but because of matching, "PCk" means the kth component
                    from the *A-side PCA* matched to its best partner in B (one-to-one).
            """
            rng = np.random.default_rng(int(random_state))
            idx = np.asarray(X_scaled_df.index)

            p = X_scaled_df.shape[1]
            comps = min(int(n_components), p, len(idx) - 1)
            pcs = [f"PC{i}" for i in range(1, comps + 1)]

            # Block series (optional)
            block_series = None
            if block_col is not None:
                if meta_df is None:
                    raise ValueError("block_col was provided but meta_df is None. Provide cruise-level meta_df.")
                if block_col not in meta_df.columns:
                    raise ValueError(f"block_col='{block_col}' not found in meta_df columns.")
                block_series = meta_df[block_col]

            rows = []
            audit_rows = []

            for r in range(int(R)):
                A, B, audit = _make_block_balanced_split(idx=idx, block=block_series, rng=rng)

                # Need at least 3 samples per half for PCA stability
                if len(A) < 3 or len(B) < 3:
                    continue

                XA = X_scaled_df.loc[A].values
                XB = X_scaled_df.loc[B].values

                pcaA = PCA(n_components=comps, random_state=int(rng.integers(0, 2**31 - 1)))
                pcaB = PCA(n_components=comps, random_state=int(rng.integers(0, 2**31 - 1)))
                pcaA.fit(XA)
                pcaB.fit(XB)

                LA = pcaA.components_.T  # (features × comps)
                LB = pcaB.components_.T

                # Build abs correlation matrix between loading vectors
                # abs_corr[kA, kB] = |corr(LA[:,kA], LB[:,kB])|
                abs_corr = np.full((comps, comps), np.nan, dtype=float)
                for i in range(comps):
                    a = LA[:, i]
                    sa = np.std(a)
                    if sa == 0:
                        continue
                    for j in range(comps):
                        b = LB[:, j]
                        sb = np.std(b)
                        if sb == 0:
                            continue
                        abs_corr[i, j] = float(np.abs(np.corrcoef(a, b)[0, 1]))

                # Match PCs across halves (one-to-one)
                if matching != "greedy":
                    raise ValueError("Only matching='greedy' is implemented in this drop-in.")
                pairs = _greedy_one_to_one_match(np.nan_to_num(abs_corr, nan=-1.0))

                # Record matched correlations.
                # We keep PC labels from A-side ordering: PC1..PCk (k in A) mapped to some PCj in B.
                # This fixes PC swapping/rotation issues.
                matched_map = {iA: (iB, corr) for iA, iB, corr in pairs}
                for k in range(comps):
                    pc = pcs[k]
                    if k in matched_map:
                        iB, c = matched_map[k]
                        rows.append({
                            "split": int(r),
                            "PC": pc,
                            "matched_PC_in_B": f"PC{int(iB)+1}",
                            "abs_loading_corr": float(c),
                        })
                    else:
                        rows.append({
                            "split": int(r),
                            "PC": pc,
                            "matched_PC_in_B": None,
                            "abs_loading_corr": np.nan,
                        })

                # Optional split audit
                if write_split_audit_path is not None:
                    for lab, d in audit.items():
                        audit_rows.append({
                            "split": int(r),
                            "block_col": str(block_col) if block_col is not None else "None",
                            "block_label": str(lab),
                            "n": int(d["n"]),
                            "nA": int(d["nA"]),
                            "nB": int(d["nB"]),
                        })

            df = pd.DataFrame(rows)
            if df.empty:
                out = pd.DataFrame({"PC": pcs, "median_abs_loading_corr": np.nan, "mean_abs_loading_corr": np.nan, "n_splits": 0})
                return out

            summ = (
                df.groupby("PC", as_index=False)["abs_loading_corr"]
                .agg(median_abs_loading_corr="median", mean_abs_loading_corr="mean", n_splits="count")
                .sort_values("PC", key=lambda s: s.map(lambda z: int(str(z).replace("PC", "")) if str(z).startswith("PC") else 10**9))
            )

            # Write audit if requested
            if write_split_audit_path is not None:
                audit_df = pd.DataFrame(audit_rows)
                if not audit_df.empty:
                    audit_df.to_csv(write_split_audit_path, index=False)

            return summ


        # ------------------------------------------------------------------------------
        # Build a DataFrame version of the EOF scaled matrix (for indexed splitting)
        # ------------------------------------------------------------------------------
        eof_scaled_df_for_splits = pd.DataFrame(
            eof_scaled,
            index=eof_X_imp.index,     # cruises
            columns=eof_X_imp.columns, # feature@depth
        )

        # Decide block_col for EOF splits:
        # Recommended: 'Year' if present at cruise level
        # (Set to None to reproduce the old behavior.)
        eof_block_col = None
        if "Year" in eof_meta.columns:
            eof_block_col = "Year"

        # Reasonable default: 200 splits (fast for cruise-level N)
        split_R = getattr(cfg, "pcsel_stability_R", 200)

        split_summ = _split_half_loading_congruence(
            X_scaled_df=eof_scaled_df_for_splits,
            n_components=int(eof_pca.n_components_),
            meta_df=eof_meta,                        # cruise-level meta (indexed by cruise id)
            block_col=eof_block_col,                 # None or 'Year'
            R=int(split_R),
            random_state=int(cfg.random_state),
            matching="greedy",
            write_split_audit_path=os.path.join(tables_dir, "eof_split_half_audit.csv"),
        )

        split_summ.to_csv(os.path.join(tables_dir, "eof_split_half_loading_congruence.csv"), index=False)


        # ----------------------------------------------------------------------
        # EOF diagnostic: vertical loading profiles
        #
        # Goal:
        #   For each EOF PC, visualize how loadings vary with depth, per variable.
        #   This is the core EOF interpretability check in oceanography:
        #     - coherent depth structure (e.g., oxycline, deep maxima, sulfidic bottom)
        #     - not random sign-flipping across adjacent depth bins
        #
        # Outputs:
        #   plots/eof_vertical_loading_profiles_PCk.png         (per-variable profiles, top variables)
        #   plots/eof_vertical_loading_meanabs_PCk.png          (mean abs loading vs depth across ALL variables)
        #   plots/eof_vertical_loading_weightedmean_PCk.png     (signed, magnitude-weighted mean vs depth)
        #   tables/eof_vertical_loading_summary_PCk.tsv         (depth-wise summaries)
        # ----------------------------------------------------------------------
        import matplotlib.pyplot as plt

        def _parse_feature_depth(index_vals: pd.Index) -> pd.DataFrame:
            s = pd.Series(index_vals.astype(str))
            parts = s.str.split("@", n=1, expand=True)
            var = parts[0]
            depth = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else pd.Series([np.nan] * len(s))
            out = pd.DataFrame({"var": var.values, "depth": depth.values})
            return out

        def _vertical_profile_plots(
            loadings_df: pd.DataFrame,
            pcs: List[str],
            out_plots_dir: str,
            out_tables_dir: str,
            top_vars: int = 10,
            max_pcs: int = 10,
        ) -> None:
            meta_ld = _parse_feature_depth(loadings_df.index)
            if meta_ld["depth"].isna().all():
                print("[EOF][WARN] Cannot plot vertical loading profiles: no '@depth' found in eof_loadings index.")
                return

            # Depth grid for plotting
            depths_sorted = np.sort(meta_ld["depth"].dropna().unique())

            pcs_use = [pc for pc in pcs if pc in loadings_df.columns][:max_pcs]
            if not pcs_use:
                return

            for pc in pcs_use:
                # Build long table: var, depth, loading for this PC
                tmp = meta_ld.copy()
                tmp["loading"] = loadings_df[pc].values
                tmp = tmp.dropna(subset=["depth"])
                tmp["depth"] = pd.to_numeric(tmp["depth"], errors="coerce")
                tmp = tmp.dropna(subset=["depth"])

                # Compute per-variable magnitude to choose top vars to show
                var_mag = (
                    tmp.groupby("var")["loading"]
                    .apply(lambda x: float(np.sqrt(np.sum(np.asarray(x) ** 2))))
                    .sort_values(ascending=False)
                )
                show_vars = var_mag.head(top_vars).index.tolist()

                # ---- Summary curves across ALL variables ----
                # mean absolute loading by depth (how "active" the PC is at each depth)
                mean_abs_by_depth = tmp.groupby("depth")["loading"].apply(lambda x: float(np.mean(np.abs(x)))).reindex(depths_sorted)

                # signed, magnitude-weighted mean loading by depth
                # (keeps directionality but downweights tiny/noisy variables)
                def _wmean_signed(x: pd.Series) -> float:
                    v = np.asarray(x, dtype=float)
                    w = np.abs(v)
                    if np.all(w == 0) or len(v) == 0:
                        return float(0.0)
                    return float(np.sum(w * v) / np.sum(w))

                wmean_by_depth = tmp.groupby("depth")["loading"].apply(_wmean_signed).reindex(depths_sorted)

                # Write summary table
                summ = pd.DataFrame({
                    "depth": depths_sorted,
                    "mean_abs_loading": mean_abs_by_depth.values,
                    "weighted_mean_signed_loading": wmean_by_depth.values,
                })
                summ.to_csv(os.path.join(out_tables_dir, f"eof_vertical_loading_summary_{pc}.tsv"), sep="\t", index=False)

                # ---- Plot 1: per-variable vertical loading profiles (top vars) ----
                fig = plt.figure(figsize=(7.2, 6.8))
                ax = plt.gca()

                for v in show_vars:
                    sub = tmp[tmp["var"] == v].groupby("depth", as_index=False)["loading"].mean()
                    sub = sub.sort_values("depth")
                    ax.plot(sub["loading"].values, sub["depth"].values, linewidth=1.5, label=str(v))

                ax.axvline(0.0, linestyle="--", linewidth=1.0)
                ax.set_xlabel(f"{pc} loading")
                ax.set_ylabel("Depth")
                ax.invert_yaxis()
                ax.set_title(f"EOF vertical loading profiles ({pc}) — top {top_vars} variables")
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(os.path.join(out_plots_dir, f"eof_vertical_loading_profiles_{pc}.png"), dpi=200)
                plt.close(fig)

                # ---- Plot 2: mean abs loading vs depth (all vars) ----
                fig = plt.figure(figsize=(6.8, 6.2))
                ax = plt.gca()
                ax.plot(mean_abs_by_depth.values, depths_sorted, linewidth=2.0)
                ax.set_xlabel(f"Mean |{pc} loading| across variables")
                ax.set_ylabel("Depth")
                ax.invert_yaxis()
                ax.set_title(f"EOF mean absolute loading vs depth ({pc})")
                fig.tight_layout()
                fig.savefig(os.path.join(out_plots_dir, f"eof_vertical_loading_meanabs_{pc}.png"), dpi=200)
                plt.close(fig)

                # ---- Plot 3: signed weighted-mean loading vs depth (all vars) ----
                fig = plt.figure(figsize=(6.8, 6.2))
                ax = plt.gca()
                ax.plot(wmean_by_depth.values, depths_sorted, linewidth=2.0)
                ax.axvline(0.0, linestyle="--", linewidth=1.0)
                ax.set_xlabel(f"Weighted mean signed {pc} loading (all variables)")
                ax.set_ylabel("Depth")
                ax.invert_yaxis()
                ax.set_title(f"EOF weighted signed mean loading vs depth ({pc})")
                fig.tight_layout()
                fig.savefig(os.path.join(out_plots_dir, f"eof_vertical_loading_weightedmean_{pc}.png"), dpi=200)
                plt.close(fig)

        # Generate vertical loading profile diagnostics for PC1..PC5
        _vertical_profile_plots(
            loadings_df=eof_loadings,
            pcs=[f"PC{i}" for i in range(1, eof_pca.n_components_ + 1)],
            out_plots_dir=plots_dir,
            out_tables_dir=tables_dir,
            top_vars=10,
            max_pcs=10,
        )

        # ----------------------------------------------------------------------
        # EOF loadings are indexed by "feature@depth". Your downstream biplot and
        # BIOCHEM_COLOR_MAP logic expects plain feature names (e.g., "O2", "NO3").
        #
        # We collapse loadings across depth into a single per-feature loading per PC
        # using an L2 norm across depths, while preserving sign using the depth-mean.
        # This yields a compact, defensible "overall contribution of each variable"
        # to each EOF mode.
        # ----------------------------------------------------------------------
        def _collapse_feature_depth_loadings(eof_loadings_df: pd.DataFrame) -> pd.DataFrame:
            idx = eof_loadings_df.index.to_series().astype(str)

            # Parse "var@depth" -> "var"; if no "@", keep as-is
            var = idx.str.split("@", n=1, expand=True)[0]

            pcs = list(eof_loadings_df.columns)
            out_rows = []
            for v in sorted(var.unique(), key=lambda z: str(z)):
                sub = eof_loadings_df.loc[var == v, pcs]

                # magnitude across depths (L2)
                mag = np.sqrt((sub.values ** 2).sum(axis=0))

                # sign from mean across depths (keeps interpretable +/- direction)
                sgn = np.sign(sub.mean(axis=0).values)
                sgn[sgn == 0] = 1.0

                out_rows.append(pd.Series(mag * sgn, index=pcs, name=v))

            out = pd.DataFrame(out_rows)
            return out

        eof_loadings_collapsed = _collapse_feature_depth_loadings(eof_loadings)
        eof_loadings_collapsed.to_csv(os.path.join(tables_dir, "eof_pca_loadings_collapsed_by_feature.csv"))
        # ----------------------------------------------------------------------

        # Scores (cruise × PC)
        eof_scores = eof_pca.transform(eof_scaled)
        eof_scores_df = pd.DataFrame(
            eof_scores,
            index=eof_X_imp.index,
            columns=[f"PC{i}" for i in range(1, eof_pca.n_components_ + 1)],
        )

        # Cruise-level eigenvectors table: meta + PCs (like your existing eigenvectors_scores.csv)
        eof_eigenvectors = pd.concat([eof_meta.reindex(eof_scores_df.index), eof_scores_df], axis=1).reset_index()
        eof_eigenvectors.to_csv(os.path.join(tables_dir, "eof_eigenvectors_scores_by_cruise.csv"), index=False)

        # ----------------------------------------------------------------------
        # EOF analog of: sparse_feature_pc_spearman.csv
        #
        # Here, EOF scores are cruise-level. Sparse features are depth-level.
        # We aggregate sparse feature values within each cruise (median across depths),
        # then compute Spearman correlation with EOF PC scores across cruises.
        # ----------------------------------------------------------------------
        eof_sparse_corr_rows = []
        if sparse_feats:
            # Aggregate sparse values per cruise (median across depths)
            cruise_col = eof_cruise_col  # already chosen for EOF
            tmp_meta = meta.reset_index(drop=True).copy()
            tmp_meta[cruise_col] = tmp_meta[cruise_col].astype(str)

            sparse_vals = df_filt[sparse_feats].reset_index(drop=True)  # depth-level sparse values
            sparse_with_cruise = pd.concat([tmp_meta[[cruise_col]], sparse_vals], axis=1)

            sparse_by_cruise = sparse_with_cruise.groupby(cruise_col, as_index=True).median(numeric_only=True)

            # Align to cruises present in eof_scores_df
            sparse_by_cruise = sparse_by_cruise.reindex(eof_scores_df.index)

            n_total_cruises = int(eof_scores_df.shape[0])

            for feat in sparse_by_cruise.columns:
                x = sparse_by_cruise[feat]
                for pc in eof_scores_df.columns:
                    y = eof_scores_df[pc]
                    m = x.notna() & y.notna()
                    n_used = int(m.sum())
                    if n_used >= 3:
                        r = float(x[m].corr(y[m], method="spearman"))
                    else:
                        r = np.nan
                    cov = float(n_used / n_total_cruises) if n_total_cruises > 0 else np.nan

                    eof_sparse_corr_rows.append({
                        "feature": str(feat),
                        "PC": str(pc),
                        "spearman_r": r,
                        "n_cruises_used": n_used,
                        "coverage": cov,
                    })

        eof_sparse_corr_df = pd.DataFrame(eof_sparse_corr_rows)
        eof_sparse_corr_df.to_csv(os.path.join(tables_dir, "eof_sparse_feature_pc_spearman.csv"), index=False)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # EOF analog of: pc1_vs_pc2_biplot_core_sparse.png
        # Uses collapsed core loadings (feature-level) + cruise-level sparse correlations.
        # ----------------------------------------------------------------------
        if "PC1" in eof_scores_df.columns and "PC2" in eof_scores_df.columns:
            eof_sparse_corr_mapped = eof_sparse_corr_df[
                eof_sparse_corr_df["feature"].isin(BIOCHEM_COLOR_MAP.keys())
            ].copy()

            # Your biplot expects:
            #  - scores_df with meta + PC1/PC2 columns
            #  - loadings_df indexed by feature name with PC columns
            plot_biplot_core_and_sparse(
                scores_df=eof_eigenvectors,              # contains meta + PCs
                loadings_df=eof_loadings_collapsed,      # feature-only loadings
                sparse_corr_df=eof_sparse_corr_mapped,
                outpath=os.path.join(plots_dir, "eof_pc1_vs_pc2_biplot_core_sparse.png"),
                top_core=12,
                top_sparse=12,
                min_core_norm=0.0,
                min_sparse_norm=0.0,
            )
        # ----------------------------------------------------------------------

        # QC summary (EOF-specific)
        eof_qc = {
            "eof_cruise_col": eof_cruise_col,
            "eof_depth_col": eof_depth_col,
            "n_cruises": int(eof_X_imp.shape[0]),
            "n_features_total": int(eof_X_imp.shape[1]),
            "n_features_dropped_all_nan": int(n_all_nan),
            "n_components_fit": int(eof_pca.n_components_),
            "impute_strategy_cruise_level": "median_per_feature@depth",
            "scaling": "zscore_per_feature@depth",
        }
        with open(os.path.join(cfg.outdir, "qc_summary_eof.json"), "w") as f:
            json.dump(eof_qc, f, indent=2)

        # Plots (reuse your existing plotting functions)
        plot_scree(eof_pca, os.path.join(plots_dir, "eof_scree.png"))
        plot_cumvar(eof_pca, os.path.join(plots_dir, "eof_cumulative_variance.png"))

        # PC scatter (only if time col is present at cruise level; function expects cfg.time_col column)
        # eof_eigenvectors has cruise id + (optional) time_col + PCs
        if cfg.time_col in eof_eigenvectors.columns:
            plot_pc_scatter(
                eof_eigenvectors,
                os.path.join(plots_dir, "eof_pc1_vs_pc2.png"),
                cfg.time_col,
            )

        # Top loadings plots (PC1..PC5)
        for i in range(1, min(6, eof_pca.n_components_ + 1)):
            pc = f"PC{i}"
            plot_top_loadings(eof_loadings, os.path.join(plots_dir, f"eof_top_loadings_{pc}.png"), pc=pc, top_n=25)

        # ----------------------------------------------------------------------
        # EOF PC selection (compatibility mode):
        #
        # Your run_pc_selection() expects feature names like 'Temperature', 'Oxygen', etc.
        # But EOF matrices have columns like 'Temperature@0', 'Temperature@5', ...
        # so cov/index mismatches occur.
        #
        # Solution:
        #   Build a cruise × feature matrix by collapsing across depth (median),
        #   then run PC selection on THAT matrix. This preserves the spirit of
        #   "cruise-level regime PCs" while matching expected feature naming.
        # ----------------------------------------------------------------------
        if cfg.pc_selection:
            # Collapse cruise × (feature@depth) -> cruise × feature
            # Using median across depths is robust and avoids depth-coverage artifacts.
            def _collapse_eof_matrix_to_features(eof_X_wide: pd.DataFrame) -> pd.DataFrame:
                cols = pd.Series(eof_X_wide.columns.astype(str))
                base = cols.str.split("@", n=1, expand=True)[0].values
                out = []
                for feat in sorted(pd.unique(base), key=lambda z: str(z)):
                    sub = eof_X_wide.loc[:, cols.str.startswith(f"{feat}@").values]
                    out.append(sub.median(axis=1, skipna=True).rename(feat))
                return pd.concat(out, axis=1)

            eof_feat_X = _collapse_eof_matrix_to_features(eof_X)  # pre-impute analog (still has NaNs)
            collapsed_all_nan = eof_feat_X.columns[eof_feat_X.isna().all(axis=0)].tolist()
            if collapsed_all_nan:
                pd.DataFrame({"feature": collapsed_all_nan}).to_csv(
                    os.path.join(tables_dir, "eof_dropped_features_collapsed_all_nan.csv"),
                    index=False,
                )
                eof_feat_X = eof_feat_X.drop(columns=collapsed_all_nan)

            if eof_feat_X.shape[1] == 0:
                raise ValueError(
                    "EOF collapsed feature matrix has no columns after dropping all-NaN features. "
                    "Inspect tables/eof_dropped_features_collapsed_all_nan.csv."
                )

            eof_feat_X.to_csv(os.path.join(tables_dir, "eof_cruise_feature_matrix_collapsed.tsv"), sep="\t", index=True)

            # Impute + scale
            eof_feat_fill = eof_feat_X.median(axis=0, skipna=True)
            eof_feat_X_imp = eof_feat_X.fillna(eof_feat_fill)

            if eof_feat_X_imp.isna().any().any():
                raise ValueError(
                    "EOF collapsed feature matrix still has NaNs after median fill. "
                    "Inspect tables/eof_dropped_features_collapsed_all_nan.csv."
                )

            eof_feat_scaler = StandardScaler(with_mean=True, with_std=True)
            eof_feat_scaled = eof_feat_scaler.fit_transform(eof_feat_X_imp.values)

            # Fit a PCA on the collapsed matrix for PC-selection diagnostics
            eof_feat_ncomp = min(cfg.n_components, eof_feat_scaled.shape[1], eof_feat_scaled.shape[0])
            eof_feat_pca = PCA(n_components=eof_feat_ncomp, random_state=cfg.random_state)
            eof_feat_pca.fit(eof_feat_scaled)

            # Loadings (plain feature names)
            eof_feat_loadings = pd.DataFrame(
                eof_feat_pca.components_.T,
                index=eof_feat_X_imp.columns,
                columns=[f"PC{i}" for i in range(1, eof_feat_pca.n_components_ + 1)],
            )
            eof_feat_loadings.to_csv(os.path.join(tables_dir, "eof_pca_loadings_collapsed_matrix.csv"))

            # Cruise-level meta aligned
            eof_meta_for_selection = eof_meta.reindex(eof_feat_X.index).reset_index(drop=False)

            # Run the same selection pipeline (now cov/index matches Tk names)
            run_pc_selection(
                cfg=cfg,
                tables_dir=tables_dir,
                plots_dir=plots_dir,
                X_preimpute=eof_feat_X,            # cruise × feature (with NaNs)
                X_scaled=eof_feat_scaled,          # numpy array
                meta=eof_meta_for_selection,
                pca=eof_feat_pca,
                loadings_df=eof_feat_loadings,
                file_prefix="eof_",
            )

        print("[EOF] Wrote EOF outputs:")
        print(f"      {os.path.join(tables_dir, 'eof_cruise_feature_matrix.tsv')}")
        print(f"      {os.path.join(tables_dir, 'eof_pca_explained_variance.csv')}")
        print(f"      {os.path.join(tables_dir, 'eof_pca_loadings.csv')}")
        print(f"      {os.path.join(tables_dir, 'eof_eigenvectors_scores_by_cruise.csv')}")
        print(f"      {os.path.join(plots_dir, 'eof_scree.png')}")
        print(f"      {os.path.join(plots_dir, 'eof_cumulative_variance.png')}")


    # ---- run EOF block ----
    _run_eof_pca_and_write_outputs()

def main() -> None:
    args = parse_args()
    tables_dir, plots_dir = ee.ensure_dirs(args.outdir)

    df = pd.read_csv(args.matrix_cleaned, sep=args.sep)
    if df.shape[0] == 0:
        raise ValueError(f"Input is empty: {args.matrix_cleaned}")

    core_feats = _load_core_features(args)
    missing_core = [c for c in core_feats if c not in df.columns]
    if missing_core:
        raise ValueError(f"Core features missing from matrix-cleaned: {missing_core[:10]}")

    # Rebuild meta / seasonal columns in the same style used by env_eigenvectors
    meta = ee.make_date_column(df.copy(), args.time_col)
    meta = ee.make_season_column(meta, time_col=args.time_col)

    # Core matrix (already cleaned/imputed upstream; still coerce to numeric defensively)
    X_imp = df[core_feats].copy().apply(pd.to_numeric, errors="coerce")
    if X_imp.isna().all(axis=1).any():
        # Keep behavior strict and transparent.
        n_bad = int(X_imp.isna().all(axis=1).sum())
        raise ValueError(f"{n_bad} rows have all-NaN core features; cannot run EOF reliably.")

    sparse_feats = _infer_sparse_features(df, core_feats, args)

    # df_filt is only used inside run_eof_pipeline for sparse-feature cruise aggregation
    num_cols = list(set(core_feats + sparse_feats))
    df_filt = df.copy()
    for c in num_cols:
        if c in df_filt.columns:
            df_filt[c] = pd.to_numeric(df_filt[c], errors="coerce")

    cfg = SimpleNamespace(
        outdir=args.outdir,
        id_col=args.id_col,
        time_col=args.time_col,
        profile_col=args.profile_col,
        anchor_by_col=args.anchor_by_col,
        depth_col=args.depth_col,
        anchored_depth_col=args.anchored_depth_col,
        n_components=int(args.n_components),
        random_state=int(args.random_state),
        eof_dropna_row_thresh=float(args.eof_dropna_row_thresh),
        pc_selection=bool(args.pc_selection),
        pcsel_parallel_B=int(args.pcsel_parallel_B),
        pcsel_parallel_quantile=float(args.pcsel_parallel_quantile),
        pcsel_support_min_cov=float(args.pcsel_support_min_cov),
        pcsel_support_median_cov=float(args.pcsel_support_median_cov),
        pcsel_support_min_n=int(args.pcsel_support_min_n),
        pcsel_top_frac=float(args.pcsel_top_frac),
        pcsel_top_min=int(args.pcsel_top_min),
        pcsel_feature_clusters=int(args.pcsel_feature_clusters),
        pcsel_coherence_min_frac=float(args.pcsel_coherence_min_frac),
        pcsel_block_col=str(args.pcsel_block_col),
        pcsel_stability_R=int(args.pcsel_stability_R),
        pcsel_stability_min_load_corr=float(args.pcsel_stability_min_load_corr),
        pcsel_stability_min_score_corr=float(args.pcsel_stability_min_score_corr),
        pcsel_entropy_max=float(args.pcsel_entropy_max),
        pcsel_pr_max=float(args.pcsel_pr_max),
    )

    with open(os.path.join(args.outdir, "run_config_eof_pipeline.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    run_eof_pipeline(
        cfg=cfg,
        meta=meta.reset_index(drop=True),
        X_imp=X_imp.reset_index(drop=True),
        feats_kept=core_feats,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
        df_filt=df_filt.reset_index(drop=True),
        sparse_feats=sparse_feats,
    )

    print(f"[OK] Standalone EOF pipeline wrote outputs to: {args.outdir}")
    print(f"      Tables: {tables_dir}")
    print(f"      Plots : {plots_dir}")


if __name__ == "__main__":
    main()
