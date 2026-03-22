#!/usr/bin/env python3
"""Multi-region Prithvi ablation study: tabular-only vs Prithvi-only vs fused.

Loads the unified multiregion training table (PA SWPA, PA NE, WV, OH, and
any future regions), extracts HLS chips and Prithvi embeddings for wells
from all regions, then trains three XGBoost model variants and compares
ROC-AUC and Average Precision.

Usage:
    python scripts/multiregion/run_prithvi_multiregion.py
    python scripts/multiregion/run_prithvi_multiregion.py --max-wells-per-region 500
    python scripts/multiregion/run_prithvi_multiregion.py --skip-hls  # reuse existing chips
    python scripts/multiregion/run_prithvi_multiregion.py --regions swpa_core wv_horizontal oh_utica
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gas_v1_common import parse_date, write_csv
from multiregion.labels import add_threshold_labels, compute_first_life_labels
from run_prithvi_gas_pipeline_v1 import (
    HLS_WINDOWS,
    build_model_pipeline,
    extract_prithvi_embeddings,
    materialize_hls_chips,
    search_hls_items,
    stratified_binary_downsample,
    write_json,
)

try:
    from pystac_client import Client
except ImportError:
    Client = None

# ---------------------------------------------------------------------------
# Feature definitions (unified across regions)
# ---------------------------------------------------------------------------
TABULAR_NUMERIC = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_deg",
    "relief_3px_m",
    "fault_distance_km",
    "well_count_2km",
    "well_count_5km",
    "nearest_well_km",
    "permit_count_2km",
    "permit_count_5km",
    "nearest_permit_km",
    "mature_f12_count_5km",
    "mature_f12_median_gas_5km",
    "mature_f12_p90_gas_5km",
    "mature_f12_count_10km",
    "mature_f12_median_gas_10km",
    "mature_f12_p90_gas_10km",
]

TABULAR_CATEGORICAL = [
    "geology_map_symbol",
    "geology_age",
    "geology_lith1",
]


# ---------------------------------------------------------------------------
# Load and unify PA + WV training data
# ---------------------------------------------------------------------------
def load_pa_wells(repo_root: Path, basin_id: str) -> pd.DataFrame:
    """Load PA v2 training table (or v1 + compute labels if v2 not ready)."""
    features_dir = repo_root / "data" / "features" / basin_id
    v2_path = features_dir / "gas_training_table_v2.csv"
    v1_path = features_dir / "gas_training_table_v1.csv"

    if v2_path.exists():
        df = pd.read_csv(v2_path, low_memory=False)
        # v2 already has true F12/F24 labels
        df = df.rename(columns={
            "latitude_decimal": "latitude",
            "longitude_decimal": "longitude",
            "nearest_well_distance_km": "nearest_well_km",
            "nearest_permit_distance_km": "nearest_permit_km",
            "formation_name": "formation",
        })
        df["region_id"] = "pa_" + basin_id.split("_")[0]
        df["state"] = "PA"
        return df

    if v1_path.exists():
        df = pd.read_csv(v1_path, low_memory=False)
        # v1 has broken labels — compute true F12/F24
        production = pd.read_csv(repo_root / "data/canonical/pa_mvp/production.csv")
        labels = compute_first_life_labels(
            well_apis=df["well_api"],
            production=production,
            method="day_prorated",
            well_api_col="permit_num",
            start_col="production_period_start_date",
            end_col="production_period_end_date",
            gas_col="gas_quantity",
            scope_col="production_scope",
            scope_value="unconventional",
        )
        labels = add_threshold_labels(labels)
        df = df.merge(labels, on="well_api", how="left")
        df = df.rename(columns={
            "latitude_decimal": "latitude",
            "longitude_decimal": "longitude",
            "nearest_well_distance_km": "nearest_well_km",
            "nearest_permit_distance_km": "nearest_permit_km",
        })
        df["region_id"] = "pa_" + basin_id.split("_")[0]
        df["state"] = "PA"
        return df

    raise FileNotFoundError(f"No PA training table found at {features_dir}")


def load_wv_wells(repo_root: Path) -> pd.DataFrame:
    """Load WV training table (already has true F12/F24 labels)."""
    wv_path = repo_root / "data/features/wv_horizontal_statewide/gas_training_table_wv_v0.csv"
    if not wv_path.exists():
        raise FileNotFoundError(f"WV training table not found: {wv_path}")

    df = pd.read_csv(wv_path, low_memory=False)
    # Rename WV columns to unified schema
    df = df.rename(columns={
        "producing_well_count_2km": "well_count_2km",
        "producing_well_count_5km": "well_count_5km",
        "nearest_producing_well_distance_km": "nearest_well_km",
        "nearest_permit_distance_km": "nearest_permit_km",
        "formation_name": "formation",
        "mature_f12_neighbor_count_5km": "mature_f12_count_5km",
        "mature_f12_neighbor_median_gas_5km": "mature_f12_median_gas_5km",
        "mature_f12_neighbor_p90_gas_5km": "mature_f12_p90_gas_5km",
        "mature_f12_neighbor_count_10km": "mature_f12_count_10km",
        "mature_f12_neighbor_median_gas_10km": "mature_f12_median_gas_10km",
        "mature_f12_neighbor_p90_gas_10km": "mature_f12_p90_gas_10km",
    })
    # WV uses month-based labels — f12_gas already present
    # Add threshold labels if missing
    if "label_f12_ge_100000" not in df.columns:
        df = add_threshold_labels(df)
    df["region_id"] = "wv_horizontal"
    df["state"] = "WV"
    return df


def load_unified_table(repo_root: Path, version: str = "v1",
                       regions: list[str] | None = None) -> pd.DataFrame:
    """Load the multiregion unified training table produced by merge.py.

    This is the preferred loader — it automatically includes all regions
    (PA SWPA, PA NE, WV, OH, and any future additions) with canonical
    column names already applied.
    """
    unified_path = repo_root / "data" / "features" / "multiregion" / f"unified_training_table_{version}.csv"
    if not unified_path.exists():
        raise FileNotFoundError(
            f"Unified training table not found: {unified_path}\n"
            "Run: python scripts/multiregion/merge.py first."
        )
    df = pd.read_csv(unified_path, low_memory=False)
    if regions:
        df = df[df["region_id"].isin(regions)].copy()
    return df


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, date_col: str, cutoff: str = "2020-01-01"):
    """Split into train (before cutoff) and test (on or after cutoff)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df[date_col] < cutoff_ts].copy()
    test = df[df[date_col] >= cutoff_ts].copy()
    return train, test


# ---------------------------------------------------------------------------
# Ablation: train 3 model variants
# ---------------------------------------------------------------------------
def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    model_name: str,
    random_state: int = 42,
) -> dict:
    """Train XGBoost and return metrics."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_train = train_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return {"model": model_name, "error": "degenerate labels in train set"}
    if len(np.unique(y_test)) < 2:
        return {"model": model_name, "error": "degenerate labels in test set"}

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    class_weight = float(neg / max(1, pos))

    pipeline = build_model_pipeline(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        class_weight_scale=class_weight,
        random_state=random_state,
    )

    pipeline.fit(train_df[feature_cols], y_train)
    y_prob = pipeline.predict_proba(test_df[feature_cols])[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_precision = float(average_precision_score(y_test, y_prob))

    # Per-region metrics
    per_region = {}
    if "region_id" in test_df.columns:
        for region in sorted(test_df["region_id"].unique()):
            mask = test_df["region_id"].values == region
            yt = y_test[mask]
            yp = y_prob[mask]
            if len(np.unique(yt)) < 2:
                per_region[region] = {"roc_auc": None, "avg_precision": None, "n": int(len(yt)), "pos": int(yt.sum())}
            else:
                per_region[region] = {
                    "roc_auc": float(roc_auc_score(yt, yp)),
                    "avg_precision": float(average_precision_score(yt, yp)),
                    "n": int(len(yt)),
                    "pos": int(yt.sum()),
                }

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "train_n": int(len(y_train)),
        "train_pos": pos,
        "train_pos_rate": round(pos / len(y_train), 4),
        "test_n": int(len(y_test)),
        "test_pos": int(y_test.sum()),
        "test_pos_rate": round(float(y_test.mean()), 4),
        "class_weight_scale": round(class_weight, 4),
        "per_region": per_region,
        "pipeline": pipeline,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-region Prithvi ablation study.")
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--max-wells-per-region", type=int, default=1500,
                        help="Max wells to select per region for HLS chip extraction.")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--temporal-cutoff", default="2020-01-01")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-hls", action="store_true",
                        help="Skip HLS download and chip extraction; reuse existing embeddings.")
    parser.add_argument("--skip-results-publish", action="store_true")
    parser.add_argument("--regions", nargs="*", default=None,
                        help="Specific region_ids to include (default: all in unified table).")
    parser.add_argument("--use-legacy-loaders", action="store_true",
                        help="Use per-state loaders instead of unified table (backwards compat).")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = repo_root / "data" / "features" / "multiregion"
    models_dir = repo_root / "models" / "multiregion"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 1. Load training data (prefer unified table)
    # ================================================================
    print("[1/6] Loading training data...")

    if not args.use_legacy_loaders:
        try:
            combined = load_unified_table(repo_root, regions=args.regions)
            print(f"  Loaded unified table: {len(combined):,} wells")
            for region in sorted(combined["region_id"].unique()):
                n = len(combined[combined["region_id"] == region])
                print(f"    {region}: {n:,} wells")
        except FileNotFoundError:
            print("  Unified table not found, falling back to per-state loaders.")
            args.use_legacy_loaders = True

    if args.use_legacy_loaders:
        regions_list = []
        for basin_id in ["swpa_core_washington_greene", "pa_northeast_susquehanna"]:
            try:
                pa = load_pa_wells(repo_root, basin_id)
                print(f"  PA ({basin_id}): {len(pa):,} wells")
                regions_list.append(pa)
            except FileNotFoundError as e:
                print(f"  PA ({basin_id}): SKIPPED - {e}")
        try:
            wv = load_wv_wells(repo_root)
            print(f"  WV: {len(wv):,} wells")
            regions_list.append(wv)
        except FileNotFoundError as e:
            print(f"  WV: SKIPPED - {e}")
        if not regions_list:
            raise SystemExit("No training data found.")
        combined = pd.concat(regions_list, ignore_index=True)

    print(f"  Combined: {len(combined):,} wells across {combined['region_id'].nunique()} regions")

    # ================================================================
    # 2. Filter to wells with valid labels
    # ================================================================
    print(f"\n[2/6] Filtering to wells with '{args.label_column}' labels...")
    combined["first_prod_date"] = parse_date(combined["first_prod_date"])
    available = combined[
        combined[args.label_column].notna()
        & combined["latitude"].notna()
        & combined["longitude"].notna()
        & combined["first_prod_date"].notna()
    ].copy()
    print(f"  Available with labels: {len(available):,} wells")
    for region in sorted(available["region_id"].unique()):
        n = len(available[available["region_id"] == region])
        pos = available.loc[available["region_id"] == region, args.label_column].astype(bool).sum()
        print(f"    {region}: {n:,} wells ({pos:,} positive = {pos/n*100:.1f}%)")

    # Downsample per region for HLS extraction
    selected_frames = []
    for region in available["region_id"].unique():
        region_df = available[available["region_id"] == region].copy()
        if len(region_df) > args.max_wells_per_region:
            region_df = stratified_binary_downsample(
                region_df, args.label_column,
                max_rows=args.max_wells_per_region,
                random_state=args.random_state,
            )
        selected_frames.append(region_df)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values("first_prod_date").reset_index(drop=True)
    print(f"  Selected for HLS: {len(selected):,} wells")

    # Add sample columns for HLS pipeline compatibility
    selected["sample_id"] = selected.apply(
        lambda r: f"well::{r['region_id']}::{r['well_api']}", axis=1
    )
    selected["sample_type"] = "well"
    selected["sample_key"] = selected["well_api"]
    selected["sample_longitude"] = selected["longitude"]
    selected["sample_latitude"] = selected["latitude"]

    # ================================================================
    # 3. HLS chip extraction + Prithvi embeddings
    # ================================================================
    chips_dir = output_dir / "hls_chips_multiregion_v1"
    embeddings_path = output_dir / "prithvi_embeddings_multiregion_v1.parquet"

    if args.skip_hls and embeddings_path.exists():
        print(f"\n[3/6] Reusing existing embeddings from {embeddings_path}")
        embeddings = pd.read_parquet(embeddings_path)
        embeddings = embeddings[embeddings["sample_id"].isin(selected["sample_id"])].copy()
        print(f"  Loaded {len(embeddings):,} embeddings")
    else:
        if Client is None:
            raise SystemExit("pystac_client not installed. Install with: pip install pystac-client")

        print(f"\n[3/6] Extracting HLS chips for {len(selected):,} wells...")

        # Compute bounding box across all regions
        min_lon = float(selected["sample_longitude"].min()) - 0.1
        min_lat = float(selected["sample_latitude"].min()) - 0.1
        max_lon = float(selected["sample_longitude"].max()) + 0.1
        max_lat = float(selected["sample_latitude"].max()) + 0.1
        print(f"  Bounding box: [{min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f}]")

        stac_client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        selected_items, hls_manifest = search_hls_items(
            stac_client,
            [min_lon, min_lat, max_lon, max_lat],
            selected[["sample_longitude", "sample_latitude"]].drop_duplicates().reset_index(drop=True),
        )
        write_json(output_dir / "hls_selection_manifest_multiregion_v1.json", hls_manifest)
        print(f"  HLS tiles found: {sum(len(tiles) for tiles in selected_items.values())} across {len(HLS_WINDOWS)} windows")

        # Materialize chips
        samples_for_chips = selected[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]].copy()
        chip_index, chip_meta = materialize_hls_chips(samples_for_chips, selected_items, chips_dir)
        chip_index.to_parquet(output_dir / "hls_chip_index_multiregion_v1.parquet", index=False)
        write_json(output_dir / "hls_chip_index_multiregion_v1_metadata.json", chip_meta)
        print(f"  Chips materialized: {len(chip_index):,} (failed: {chip_meta.get('failed_sample_count', 0)})")

        # Extract Prithvi embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Extracting Prithvi embeddings on {device}...")
        embeddings, embedding_meta = extract_prithvi_embeddings(
            chip_index=chip_index,
            batch_size=args.embedding_batch_size,
            device=device,
            existing_embeddings_path=embeddings_path if embeddings_path.exists() else None,
        )
        embeddings.to_parquet(embeddings_path, index=False)
        write_json(output_dir / "prithvi_embeddings_multiregion_v1_metadata.json", embedding_meta)
        print(f"  Embeddings: {len(embeddings):,} samples x {embedding_meta.get('embedding_dim', '?')} dims")

    # ================================================================
    # 4. Merge tabular + embeddings
    # ================================================================
    print("\n[4/6] Merging tabular features with Prithvi embeddings...")
    training_frame = selected.merge(
        embeddings, on=["sample_id", "sample_type", "sample_key"], how="inner"
    ).copy()
    embedding_cols = [c for c in training_frame.columns if c.startswith("embedding_")]
    print(f"  Merged: {len(training_frame):,} wells with {len(embedding_cols)} embedding dims")

    # Fill NaN in features
    for col in TABULAR_NUMERIC:
        if col in training_frame.columns:
            training_frame[col] = pd.to_numeric(training_frame[col], errors="coerce").fillna(-999)
        else:
            training_frame[col] = -999
    for col in TABULAR_CATEGORICAL:
        if col in training_frame.columns:
            training_frame[col] = training_frame[col].fillna("missing").astype(str)
        else:
            training_frame[col] = "missing"
    for col in embedding_cols:
        training_frame[col] = training_frame[col].fillna(0.0)

    # Ensure label is clean
    training_frame = training_frame.dropna(subset=[args.label_column]).copy()
    training_frame[args.label_column] = training_frame[args.label_column].astype(bool)

    # Save fused features
    fused_path = output_dir / "fused_features_multiregion_v1.parquet"
    training_frame.to_parquet(fused_path, index=False)

    # ================================================================
    # 5. Temporal split + ablation
    # ================================================================
    print(f"\n[5/6] Temporal split (cutoff: {args.temporal_cutoff})...")
    train_df, test_df = temporal_split(training_frame, "first_prod_date", args.temporal_cutoff)
    print(f"  Train: {len(train_df):,} wells | Test: {len(test_df):,} wells")

    if len(train_df) < 10 or len(test_df) < 10:
        raise SystemExit(f"Insufficient data after split: train={len(train_df)}, test={len(test_df)}")

    # Define feature sets for ablation
    tabular_features = [c for c in TABULAR_NUMERIC if c in training_frame.columns] + TABULAR_CATEGORICAL
    prithvi_features = embedding_cols
    fused_features = tabular_features + embedding_cols

    tabular_numeric_only = [c for c in TABULAR_NUMERIC if c in training_frame.columns]

    print("\n[6/6] Running ablation study (3 model variants)...")
    results = []

    # Variant 1: Tabular only
    print("\n  --- Variant 1: Tabular Only ---")
    r1 = train_and_evaluate(
        train_df, test_df, tabular_features,
        numeric_cols=tabular_numeric_only,
        categorical_cols=TABULAR_CATEGORICAL,
        label_col=args.label_column,
        model_name="tabular_only",
        random_state=args.random_state,
    )
    pipe1 = r1.pop("pipeline", None)
    results.append(r1)
    print(f"    ROC-AUC: {r1.get('roc_auc', 'N/A')}  |  Avg Precision: {r1.get('avg_precision', 'N/A')}")

    # Variant 2: Prithvi only
    print("\n  --- Variant 2: Prithvi Only ---")
    r2 = train_and_evaluate(
        train_df, test_df, prithvi_features,
        numeric_cols=embedding_cols,
        categorical_cols=[],
        label_col=args.label_column,
        model_name="prithvi_only",
        random_state=args.random_state,
    )
    pipe2 = r2.pop("pipeline", None)
    results.append(r2)
    print(f"    ROC-AUC: {r2.get('roc_auc', 'N/A')}  |  Avg Precision: {r2.get('avg_precision', 'N/A')}")

    # Variant 3: Fused (tabular + Prithvi)
    print("\n  --- Variant 3: Fused (Tabular + Prithvi) ---")
    r3 = train_and_evaluate(
        train_df, test_df, fused_features,
        numeric_cols=tabular_numeric_only + embedding_cols,
        categorical_cols=TABULAR_CATEGORICAL,
        label_col=args.label_column,
        model_name="fused",
        random_state=args.random_state,
    )
    pipe3 = r3.pop("pipeline", None)
    results.append(r3)
    print(f"    ROC-AUC: {r3.get('roc_auc', 'N/A')}  |  Avg Precision: {r3.get('avg_precision', 'N/A')}")

    # ================================================================
    # Summary
    # ================================================================
    # Compute lift
    tabular_auc = r1.get("roc_auc")
    fused_auc = r3.get("roc_auc")
    if tabular_auc and fused_auc and tabular_auc > 0:
        relative_lift = (fused_auc - tabular_auc) / tabular_auc * 100
    else:
        relative_lift = None

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "study": "multiregion_prithvi_ablation_v1",
        "label_column": args.label_column,
        "temporal_cutoff": args.temporal_cutoff,
        "max_wells_per_region": args.max_wells_per_region,
        "total_wells_with_embeddings": len(training_frame),
        "regions": sorted(training_frame["region_id"].unique().tolist()),
        "results": results,
        "lift_analysis": {
            "tabular_roc_auc": tabular_auc,
            "fused_roc_auc": fused_auc,
            "prithvi_relative_lift_pct": round(relative_lift, 2) if relative_lift is not None else None,
            "prithvi_adds_value": relative_lift is not None and relative_lift > 10,
            "gate_threshold_pct": 10,
            "verdict": (
                f"Prithvi adds {relative_lift:.1f}% relative lift — {'PASSES' if relative_lift > 10 else 'DOES NOT PASS'} the >10% gate"
                if relative_lift is not None
                else "Could not compute lift"
            ),
        },
        "output_paths": {
            "fused_features": str(fused_path),
            "embeddings": str(embeddings_path),
        },
    }

    # Save results
    results_path = output_dir / "prithvi_ablation_results_v1.json"
    write_json(results_path, summary)

    # Save best model
    if pipe3 is not None:
        model_path = models_dir / f"prithvi_fused_multiregion_v1_{args.label_column}.joblib"
        joblib.dump(pipe3, model_path)
        summary["output_paths"]["model"] = str(model_path)

    # Print final summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"  Label: {args.label_column}")
    print(f"  Regions: {', '.join(summary['regions'])}")
    print(f"  Wells: {summary['total_wells_with_embeddings']:,}")
    print(f"  Temporal cutoff: {args.temporal_cutoff}")
    print()
    print(f"  {'Model':<20} {'ROC-AUC':>10} {'Avg Prec':>10} {'Train':>8} {'Test':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        auc = f"{r['roc_auc']:.4f}" if r.get('roc_auc') else "N/A"
        ap = f"{r['avg_precision']:.4f}" if r.get('avg_precision') else "N/A"
        print(f"  {r['model']:<20} {auc:>10} {ap:>10} {r.get('train_n', 0):>8,} {r.get('test_n', 0):>8,}")
    print()
    lift = summary["lift_analysis"]
    print(f"  Prithvi relative lift: {lift.get('prithvi_relative_lift_pct', 'N/A')}%")
    print(f"  Verdict: {lift['verdict']}")
    print(f"\n  Results saved to: {results_path}")
    print("=" * 70)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
