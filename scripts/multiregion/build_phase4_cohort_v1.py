#!/usr/bin/env python3
"""Build a fixed well cohort for corrected Phase 4 gas experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multiregion.merge import load_and_prepare_region, load_region_registry  # noqa: E402
from multiregion.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402
from phase4_common import parse_date, write_json  # noqa: E402


def normalize_well_api(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        text = value.strip()
        return text or pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return pd.NA
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def stratified_binary_downsample(
    frame: pd.DataFrame,
    label_col: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.copy()
    positives = frame[frame[label_col] == True].copy()
    negatives = frame[frame[label_col] == False].copy()
    if positives.empty or negatives.empty:
        return frame.sample(n=max_rows, random_state=random_state).copy()

    target_each = max_rows // 2
    pos_n = min(len(positives), target_each)
    neg_n = min(len(negatives), target_each)
    remainder = max_rows - pos_n - neg_n
    if remainder > 0:
        if len(positives) - pos_n > len(negatives) - neg_n:
            pos_n = min(len(positives), pos_n + remainder)
        else:
            neg_n = min(len(negatives), neg_n + remainder)

    sampled = pd.concat(
        [
            positives.sample(n=pos_n, random_state=random_state),
            negatives.sample(n=neg_n, random_state=random_state),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed cohort for Phase 4 gas experiments.")
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--regions", nargs="*", default=["swpa_core", "pa_northeast", "wv_horizontal", "oh_utica"])
    parser.add_argument("--task-type", choices=["regression", "classification"], default="regression")
    parser.add_argument("--target-column", default="f12_gas")
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--target-transform", choices=["log1p", "none"], default="log1p")
    parser.add_argument("--split-mode", choices=["temporal", "leave-one-state-out", "leave-one-basin-out"], default="temporal")
    parser.add_argument("--holdout-date", default="2020-01-01")
    parser.add_argument("--holdout-state", default=None)
    parser.add_argument("--holdout-basin", default=None)
    parser.add_argument("--max-wells-per-region", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--anchor-date-column", default="first_prod_date")
    parser.add_argument("--anchor-year-offset", type=int, default=-1)
    parser.add_argument("--output-name", default="phase4_multiregion_wells_v1")
    return parser.parse_args()


def pick_target_availability_column(target_column: str) -> str | None:
    if target_column.startswith("f12_") or target_column == "f12_gas":
        return "label_f12_available"
    if target_column.startswith("f24_") or target_column == "f24_gas":
        return "label_f24_available"
    return None


def filter_trainable_rows(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    result["well_api"] = result["well_api"].map(normalize_well_api).astype("string")
    result["first_prod_date"] = parse_date(result["first_prod_date"])
    result["sample_id"] = result["well_api"].map(lambda value: f"well::{value}")
    result["sample_type"] = "well"
    result["sample_key"] = result["well_api"]
    result["sample_longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["sample_latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    required_mask = (
        result["sample_longitude"].notna()
        & result["sample_latitude"].notna()
        & result["first_prod_date"].notna()
    )
    if args.task_type == "classification":
        if args.label_column not in result.columns:
            raise ValueError(f"Label column {args.label_column!r} not found in cohort source data.")
        result[args.label_column] = result[args.label_column].astype("boolean")
        required_mask &= result[args.label_column].notna()
    else:
        if args.target_column not in result.columns:
            raise ValueError(f"Target column {args.target_column!r} not found in cohort source data.")
        result[args.target_column] = pd.to_numeric(result[args.target_column], errors="coerce")
        required_mask &= result[args.target_column].notna()
        availability_col = pick_target_availability_column(args.target_column)
        if availability_col and availability_col in result.columns:
            required_mask &= result[availability_col].eq(True)
    result = result[required_mask].copy()
    return result.reset_index(drop=True)


def sample_region_rows(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.max_wells_per_region <= 0 or len(frame) <= args.max_wells_per_region:
        return frame.copy().reset_index(drop=True)
    if args.task_type == "classification":
        return stratified_binary_downsample(frame, args.label_column, args.max_wells_per_region, args.random_state)
    return frame.sample(n=args.max_wells_per_region, random_state=args.random_state).copy().reset_index(drop=True)


def assign_split_group(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    result = frame.copy()
    if args.split_mode == "temporal":
        cutoff = pd.Timestamp(args.holdout_date)
        result["split_group"] = np.where(result["first_prod_date"] < cutoff, "train", "test")
    elif args.split_mode == "leave-one-state-out":
        if not args.holdout_state:
            raise ValueError("--holdout-state is required for leave-one-state-out.")
        holdout = args.holdout_state.upper()
        result["split_group"] = np.where(result["state"] == holdout, "test", "train")
    else:
        if not args.holdout_basin:
            raise ValueError("--holdout-basin is required for leave-one-basin-out.")
        result["split_group"] = np.where(result["basin"] == args.holdout_basin, "test", "train")
    train = result[result["split_group"] == "train"]
    test = result[result["split_group"] == "test"]
    if train.empty or test.empty:
        raise ValueError("Split assignment produced an empty train or test set.")
    return result


def normalize_parquet_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column]):
            result[column] = result[column].astype("string")
    return result


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    registry = load_region_registry(repo_root)
    region_map = {entry["region_id"]: entry for entry in registry["regions"]}

    selected_frames: list[pd.DataFrame] = []
    region_summaries: list[dict[str, object]] = []
    for region_id in args.regions:
        region = region_map.get(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")
        prepared, warnings = load_and_prepare_region(region, repo_root)
        trainable = filter_trainable_rows(prepared, args)
        sampled = sample_region_rows(trainable, args)
        sampled["region_id"] = region["region_id"]
        sampled["state"] = region["state"]
        sampled["basin"] = region["basin_config"]
        selected_frames.append(sampled)
        region_summaries.append(
            {
                "region_id": region["region_id"],
                "state": region["state"],
                "basin_config": region["basin_config"],
                "warning_count": int(len(warnings)),
                "row_count_available": int(len(trainable)),
                "row_count_selected": int(len(sampled)),
            }
        )

    cohort = pd.concat(selected_frames, ignore_index=True)
    cohort = cohort.sort_values(["region_id", "first_prod_date", "sample_id"]).reset_index(drop=True)
    cohort = assign_split_group(cohort, args)
    cohort[args.anchor_date_column] = parse_date(cohort[args.anchor_date_column])
    cohort["anchor_year"] = pd.to_numeric(cohort[args.anchor_date_column].dt.year + args.anchor_year_offset, errors="coerce")
    cohort = cohort[cohort["anchor_year"].notna()].copy().reset_index(drop=True)
    cohort["anchor_year"] = cohort["anchor_year"].astype(int)
    cohort = normalize_parquet_columns(cohort)

    output_dir = repo_root / "data" / "features" / "multiregion"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.output_name}.parquet"
    metadata_path = output_dir / f"{args.output_name}_metadata.json"

    cohort.to_parquet(output_path, index=False)

    target_key = args.target_column if args.task_type == "regression" else args.label_column
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "phase4_cohort_v1",
        "task_type": args.task_type,
        "target_column": target_key,
        "target_transform": args.target_transform if args.task_type == "regression" else None,
        "feature_columns_numeric": NUMERIC_FEATURES,
        "feature_columns_categorical": CATEGORICAL_FEATURES,
        "split_mode": args.split_mode,
        "holdout_date": args.holdout_date if args.split_mode == "temporal" else None,
        "holdout_state": args.holdout_state,
        "holdout_basin": args.holdout_basin,
        "max_wells_per_region": int(args.max_wells_per_region),
        "random_state": int(args.random_state),
        "anchor_date_column": args.anchor_date_column,
        "anchor_year_offset": int(args.anchor_year_offset),
        "row_count_total": int(len(cohort)),
        "row_count_train": int(cohort["split_group"].eq("train").sum()),
        "row_count_test": int(cohort["split_group"].eq("test").sum()),
        "row_count_by_region": {key: int(value) for key, value in cohort["region_id"].value_counts().sort_index().items()},
        "row_count_by_anchor_year": {str(key): int(value) for key, value in cohort["anchor_year"].value_counts().sort_index().items()},
        "regions": region_summaries,
        "output_path": str(output_path),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
