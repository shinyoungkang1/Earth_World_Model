#!/usr/bin/env python3
"""Run the corrected Phase 4 Prithvi gas pipeline on a fixed cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from pystac_client import Client

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase4_common import (  # noqa: E402
    build_classification_pipeline,
    build_regression_pipeline,
    compute_classification_metrics,
    compute_regression_metrics,
    ensure_feature_columns,
    inverse_transform_target,
    load_json,
    parse_date,
    read_table,
    resolve_metadata_path,
    split_from_cohort,
    transform_target,
    write_json,
)
from run_prithvi_gas_pipeline_v1 import (  # noqa: E402
    build_hls_windows,
    extract_prithvi_embeddings,
    materialize_hls_chips,
    search_hls_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected Phase 4 Prithvi gas pipeline.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--cohort-path", required=True)
    parser.add_argument("--cohort-metadata-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--chips-dir", default=None)
    parser.add_argument("--chip-index-path", default=None)
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--target-column", default="f12_gas")
    parser.add_argument("--target-transform", choices=["log1p", "none"], default="log1p")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stac-api-url", default="https://planetarycomputer.microsoft.com/api/stac/v1")
    parser.add_argument("--skip-chip-build", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    return parser.parse_args()


def default_chip_root(cohort_path: Path) -> Path:
    return cohort_path.parent / f"{cohort_path.stem}_prithvi_hls"


def default_output_dir(cohort_path: Path) -> Path:
    return cohort_path.parent / f"{cohort_path.stem}_prithvi_phase4"


def summarize_split(train: pd.DataFrame, test: pd.DataFrame, split_mode: str, split_date: pd.Timestamp | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "split_mode": split_mode,
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
    }
    if not train.empty and "first_prod_date" in train.columns:
        summary["train_first_prod_start"] = train["first_prod_date"].min().strftime("%Y-%m-%d")
        summary["train_first_prod_end"] = train["first_prod_date"].max().strftime("%Y-%m-%d")
    if not test.empty and "first_prod_date" in test.columns:
        summary["test_first_prod_start"] = test["first_prod_date"].min().strftime("%Y-%m-%d")
        summary["test_first_prod_end"] = test["first_prod_date"].max().strftime("%Y-%m-%d")
    if split_date is not None:
        summary["temporal_holdout_cutoff_date"] = split_date.strftime("%Y-%m-%d")
    return summary


def score_grid_predictions(
    grid_frame: pd.DataFrame,
    feature_columns: list[str],
    model_pipeline,
    task_type: str,
    target_transform: str,
    derived_dir: Path,
) -> dict[str, Any]:
    if grid_frame.empty:
        return {"grid_scoring_skipped_reason": "No grid rows were present in the fixed cohort."}

    result = grid_frame.copy()
    if task_type == "classification":
        result["prediction_score"] = model_pipeline.predict_proba(result[feature_columns])[:, 1]
        rank_source = result["prediction_score"]
    else:
        pred_model = model_pipeline.predict(result[feature_columns])
        result["prediction_model_scale"] = pred_model
        result["prediction_raw"] = inverse_transform_target(pred_model, target_transform)
        rank_source = result["prediction_raw"]

    result["score_percentile"] = rank_source.rank(method="average", pct=True)
    result = result.sort_values("score_percentile", ascending=False).reset_index(drop=True)
    result["score_rank"] = np.arange(1, len(result) + 1)

    csv_path = derived_dir / "phase4_grid_predictions.csv"
    result.to_csv(csv_path, index=False)
    return {
        "row_count_grid_scored": int(len(result)),
        "grid_predictions_path": str(csv_path),
    }


def build_or_load_hls_index(
    cohort: pd.DataFrame,
    chip_root: Path,
    chip_index_path: Path,
    stac_api_url: str,
    skip_chip_build: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    chip_root.mkdir(parents=True, exist_ok=True)
    chips_dir = chip_root / "hls_chips_v1"
    manifest_path = chip_root / "hls_selection_manifest_v2.json"
    metadata_path = chip_root / "hls_chip_index_v2_metadata.json"

    if skip_chip_build and chip_index_path.exists():
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        return pd.read_parquet(chip_index_path), metadata

    catalog = Client.open(stac_api_url)
    manifest_groups: list[dict[str, Any]] = []
    chip_index_parts: list[pd.DataFrame] = []
    total_failures = 0

    for anchor_year, anchor_frame in cohort.groupby("anchor_year", sort=True):
        samples = anchor_frame[["sample_id", "sample_type", "sample_key", "sample_longitude", "sample_latitude"]].copy()
        samples = samples.reset_index(drop=True)
        windows = build_hls_windows(int(anchor_year))
        bbox = [
            float(samples["sample_longitude"].min()) - 0.05,
            float(samples["sample_latitude"].min()) - 0.05,
            float(samples["sample_longitude"].max()) + 0.05,
            float(samples["sample_latitude"].max()) + 0.05,
        ]
        sample_points = samples[["sample_longitude", "sample_latitude"]].drop_duplicates().reset_index(drop=True)
        selected_items, manifest = search_hls_items(catalog, bbox, sample_points, windows=windows)
        manifest_groups.append({"anchor_year": int(anchor_year), **manifest})
        chip_index_part, chip_meta_part = materialize_hls_chips(
            samples=samples,
            selected_items=selected_items,
            chips_dir=chips_dir,
            windows=windows,
            skip_existing=True,
            anchor_year=int(anchor_year),
        )
        chip_index_parts.append(chip_index_part)
        total_failures += int(chip_meta_part["failed_sample_count"])

    chip_index = (
        pd.concat(chip_index_parts, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        if chip_index_parts
        else pd.DataFrame()
    )
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "hls_chip_index_v2",
        "row_count": int(len(chip_index)),
        "failed_sample_count": int(total_failures),
        "anchor_year_counts": (
            {str(key): int(value) for key, value in chip_index["anchor_year"].value_counts().sort_index().items()}
            if (not chip_index.empty and "anchor_year" in chip_index.columns)
            else {}
        ),
        "chip_index_path": str(chip_index_path),
        "chips_dir": str(chips_dir),
        "manifest_path": str(manifest_path),
    }
    chip_index.to_parquet(chip_index_path, index=False)
    write_json(manifest_path, {"generated_at_utc": pd.Timestamp.utcnow().isoformat(), "groups": manifest_groups})
    write_json(metadata_path, metadata)
    return chip_index, metadata


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    cohort_path = Path(args.cohort_path)
    cohort_path = cohort_path if cohort_path.is_absolute() else (repo_root / cohort_path)
    cohort_meta_path = resolve_metadata_path(cohort_path, args.cohort_metadata_path)
    chip_root = Path(args.chips_dir) if args.chips_dir else default_chip_root(cohort_path)
    chip_root = chip_root if chip_root.is_absolute() else (repo_root / chip_root)
    chip_index_path = Path(args.chip_index_path) if args.chip_index_path else (chip_root / "hls_chip_index_v2.parquet")
    chip_index_path = chip_index_path if chip_index_path.is_absolute() else (repo_root / chip_index_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(cohort_path)
    output_dir = output_dir if output_dir.is_absolute() else (repo_root / output_dir)

    features_dir = output_dir / "features"
    models_dir = output_dir / "models"
    derived_dir = output_dir / "derived"
    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    model_device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    cohort = read_table(cohort_path)
    cohort_meta = load_json(cohort_meta_path)
    cohort["first_prod_date"] = parse_date(cohort["first_prod_date"])
    wells = cohort[cohort["sample_type"] == "well"].copy()
    grid = cohort[cohort["sample_type"] == "grid_cell"].copy()
    if args.target_column not in wells.columns:
        raise SystemExit(f"Target column {args.target_column!r} not found in cohort.")

    feature_columns_numeric = list(cohort_meta["feature_columns_numeric"])
    feature_columns_categorical = list(cohort_meta["feature_columns_categorical"])

    chip_index, chip_meta = build_or_load_hls_index(cohort, chip_root, chip_index_path, args.stac_api_url, args.skip_chip_build)
    chip_index = chip_index[chip_index["sample_id"].isin(cohort["sample_id"])].copy().reset_index(drop=True)
    if chip_index.empty:
        raise SystemExit("No HLS chip rows overlap the fixed cohort.")

    embeddings_path = features_dir / "prithvi_embeddings_v2.parquet"
    embeddings, embedding_meta = extract_prithvi_embeddings(
        chip_index=chip_index,
        batch_size=args.embedding_batch_size,
        device=model_device,
        existing_embeddings_path=embeddings_path,
    )
    embeddings.to_parquet(embeddings_path, index=False)
    write_json(features_dir / "prithvi_embeddings_v2_metadata.json", embedding_meta)

    training_frame = wells.merge(
        embeddings,
        on=["sample_id", "sample_type", "sample_key"],
        how="inner",
    ).copy()
    if training_frame.empty:
        raise SystemExit("No well rows retained after merging Prithvi embeddings with the fixed cohort.")

    training_frame = ensure_feature_columns(training_frame, feature_columns_numeric, feature_columns_categorical)
    embedding_columns = [column for column in training_frame.columns if column.startswith("embedding_")]
    feature_columns = feature_columns_numeric + feature_columns_categorical + embedding_columns
    if not embedding_columns:
        raise SystemExit("No Prithvi embedding columns were found after merging.")

    split_mode, chosen_quantile, split_date, train, test = split_from_cohort(
        training_frame,
        task_type=args.task_type,
        target_column=args.target_column,
        preferred_quantile=args.holdout_quantile,
    )

    metrics: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prithvi_phase4_v2",
        "cohort_path": str(cohort_path),
        "cohort_metadata_path": str(cohort_meta_path),
        "chip_index_path": str(chip_index_path),
        "task_type": args.task_type,
        "target_column": args.target_column,
        "target_transform": args.target_transform if args.task_type == "regression" else None,
        "row_count_cohort_total": int(len(cohort)),
        "row_count_wells_total": int(len(wells)),
        "row_count_grid_total": int(len(grid)),
        "row_count_with_embeddings": int(len(training_frame)),
        "feature_column_count_numeric": int(len(feature_columns_numeric)),
        "feature_column_count_categorical": int(len(feature_columns_categorical)),
        "feature_column_count_prithvi": int(len(embedding_columns)),
        "chip_build_failed_sample_count": int(chip_meta.get("failed_sample_count", 0)),
        "embeddings_path": str(embeddings_path),
        **summarize_split(train, test, split_mode, split_date),
    }
    if chosen_quantile is not None:
        metrics["temporal_holdout_quantile"] = float(chosen_quantile)

    fused_features_path = features_dir / "fused_features_prithvi_v2.parquet"
    if args.task_type == "classification":
        train[args.target_column] = train[args.target_column].astype(bool)
        test[args.target_column] = test[args.target_column].astype(bool)
        training_frame[args.target_column] = training_frame[args.target_column].astype(bool)
        positives = int(train[args.target_column].sum())
        negatives = int((~train[args.target_column]).sum())
        class_weight_scale = float(negatives / max(1, positives))
        model_pipeline = build_classification_pipeline(
            numeric_columns=feature_columns_numeric + embedding_columns,
            categorical_columns=feature_columns_categorical,
            class_weight_scale=class_weight_scale,
            random_state=args.random_state,
            device=model_device,
        )
        model_pipeline.fit(train[feature_columns], train[args.target_column])
        test_scores = model_pipeline.predict_proba(test[feature_columns])[:, 1]
        metrics.update(compute_classification_metrics(test[args.target_column].astype(int).to_numpy(), test_scores))

        final_class_weight_scale = float(
            (~training_frame[args.target_column]).sum() / max(1, training_frame[args.target_column].sum())
        )
        final_pipeline = build_classification_pipeline(
            numeric_columns=feature_columns_numeric + embedding_columns,
            categorical_columns=feature_columns_categorical,
            class_weight_scale=final_class_weight_scale,
            random_state=args.random_state,
            device=model_device,
        )
        final_pipeline.fit(training_frame[feature_columns], training_frame[args.target_column])
        model_path = models_dir / f"gas_prithvi_phase4_v2_classifier_{args.target_column}.joblib"
    else:
        y_train_raw = pd.to_numeric(train[args.target_column], errors="coerce").fillna(0.0).to_numpy()
        y_test_raw = pd.to_numeric(test[args.target_column], errors="coerce").fillna(0.0).to_numpy()
        y_train_model = transform_target(train[args.target_column], args.target_transform)
        y_test_model = transform_target(test[args.target_column], args.target_transform)
        model_pipeline = build_regression_pipeline(
            numeric_columns=feature_columns_numeric + embedding_columns,
            categorical_columns=feature_columns_categorical,
            random_state=args.random_state,
            device=model_device,
        )
        model_pipeline.fit(train[feature_columns], y_train_model)
        pred_model = model_pipeline.predict(test[feature_columns])
        pred_raw = inverse_transform_target(pred_model, args.target_transform)
        metrics.update(compute_regression_metrics(y_test_raw, pred_raw, y_test_model, pred_model))

        final_pipeline = build_regression_pipeline(
            numeric_columns=feature_columns_numeric + embedding_columns,
            categorical_columns=feature_columns_categorical,
            random_state=args.random_state,
            device=model_device,
        )
        final_pipeline.fit(
            training_frame[feature_columns],
            transform_target(training_frame[args.target_column], args.target_transform),
        )
        model_path = models_dir / f"gas_prithvi_phase4_v2_regression_{args.target_column}.joblib"

    joblib.dump(final_pipeline, model_path)
    training_frame.to_parquet(fused_features_path, index=False)
    metrics["model_path"] = str(model_path)
    metrics["fused_features_path"] = str(fused_features_path)

    if not args.skip_scoring:
        grid_frame = grid.merge(
            embeddings,
            on=["sample_id", "sample_type", "sample_key"],
            how="inner",
        ).copy()
        grid_frame = ensure_feature_columns(grid_frame, feature_columns_numeric, feature_columns_categorical)
        metrics.update(score_grid_predictions(grid_frame, feature_columns, final_pipeline, args.task_type, args.target_transform, derived_dir))
    else:
        metrics["grid_scoring_skipped_reason"] = "Explicitly disabled via --skip-scoring."

    metrics_path = models_dir / f"gas_prithvi_phase4_v2_{args.task_type}_{args.target_column}_metrics.json"
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
