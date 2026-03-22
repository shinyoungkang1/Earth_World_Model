#!/usr/bin/env python3
"""Unified XGBoost training script for the multi-region dataset.

Supports three holdout strategies:
  - temporal:              train on wells first produced before 2020, test on the rest
  - leave-one-state-out:  train on all states except --holdout-state
  - leave-one-basin-out:  train on all basins except --holdout-basin

Outputs a JSON results file with per-region and pooled metrics.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "geology_map_symbol",
    "geology_age",
    "geology_lith1",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values so the pipeline can proceed."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(-999)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def _build_pipeline() -> Pipeline:
    """Return an sklearn Pipeline wrapping ColumnTransformer + XGBClassifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        tree_method="hist",
        eval_metric="aucpr",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def _compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, region_col: pd.Series | None = None
) -> dict:
    """Compute pooled and optionally per-region ROC-AUC and Average Precision."""
    results: dict = {}

    # Pooled metrics
    if len(np.unique(y_true)) < 2:
        results["pooled"] = {
            "roc_auc": None,
            "average_precision": None,
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
        }
    else:
        results["pooled"] = {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
        }

    # Per-region metrics
    if region_col is not None:
        per_region: dict = {}
        for region in sorted(region_col.unique()):
            mask = region_col == region
            yt = y_true[mask]
            yp = y_prob[mask]
            if len(np.unique(yt)) < 2:
                per_region[region] = {
                    "roc_auc": None,
                    "average_precision": None,
                    "n_samples": int(len(yt)),
                    "n_positive": int(yt.sum()),
                }
            else:
                per_region[region] = {
                    "roc_auc": float(roc_auc_score(yt, yp)),
                    "average_precision": float(average_precision_score(yt, yp)),
                    "n_samples": int(len(yt)),
                    "n_positive": int(yt.sum()),
                }
        results["per_region"] = per_region

    return results


# ---------------------------------------------------------------------------
# Holdout splits
# ---------------------------------------------------------------------------
def _split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on first_prod_date < 2020-01-01 vs >= 2020-01-01."""
    cutoff = pd.Timestamp("2020-01-01")
    df = df.copy()
    df["first_prod_date"] = pd.to_datetime(df["first_prod_date"], errors="coerce")
    train = df[df["first_prod_date"] < cutoff]
    test = df[df["first_prod_date"] >= cutoff]
    return train, test


def _split_leave_one_state_out(
    df: pd.DataFrame, holdout_state: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all states except *holdout_state*; test on that state."""
    holdout_upper = holdout_state.upper()
    train = df[df["state"] != holdout_upper]
    test = df[df["state"] == holdout_upper]
    return train, test


def _split_leave_one_basin_out(
    df: pd.DataFrame, holdout_basin: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on all basins except *holdout_basin*; test on that basin."""
    train = df[df["basin"] != holdout_basin]
    test = df[df["basin"] == holdout_basin]
    return train, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified XGBoost training for multi-region gas locator."
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/shin/Mineral_Gas_Locator"),
    )
    parser.add_argument(
        "--holdout",
        choices=["temporal", "leave-one-state-out", "leave-one-basin-out"],
        required=True,
        help="Holdout strategy for train/test split.",
    )
    parser.add_argument(
        "--holdout-state",
        type=str,
        default=None,
        help="State code to hold out (e.g. OH). Required for leave-one-state-out.",
    )
    parser.add_argument(
        "--holdout-basin",
        type=str,
        default=None,
        help="Basin name to hold out (e.g. Utica). Required for leave-one-basin-out.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label_f12_ge_1000000",
        help="Binary label column name.",
    )
    parser.add_argument(
        "--output-version",
        type=str,
        default="v1",
        help="Version tag appended to input filename.",
    )
    args = parser.parse_args()

    # --- Validate holdout args ---
    if args.holdout == "leave-one-state-out" and not args.holdout_state:
        parser.error("--holdout-state is required for leave-one-state-out mode.")
    if args.holdout == "leave-one-basin-out" and not args.holdout_basin:
        parser.error("--holdout-basin is required for leave-one-basin-out mode.")

    # --- Paths ---
    features_dir = args.repo_root / "data" / "features" / "multiregion"
    input_csv = features_dir / f"unified_training_table_{args.output_version}.csv"
    output_json = features_dir / f"training_results_{args.holdout}.json"

    print(f"[train] Reading {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[train] Loaded {len(df):,} rows, {df.columns.size} columns")

    if args.label_col not in df.columns:
        raise ValueError(
            f"Label column '{args.label_col}' not found. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # --- Split ---
    if args.holdout == "temporal":
        train_df, test_df = _split_temporal(df)
        region_key = "state"
    elif args.holdout == "leave-one-state-out":
        train_df, test_df = _split_leave_one_state_out(df, args.holdout_state)
        region_key = "state"
    elif args.holdout == "leave-one-basin-out":
        train_df, test_df = _split_leave_one_basin_out(df, args.holdout_basin)
        region_key = "basin"
    else:
        raise ValueError(f"Unknown holdout mode: {args.holdout}")

    print(f"[train] Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    if len(train_df) == 0:
        raise ValueError("Training set is empty after split.")
    if len(test_df) == 0:
        raise ValueError("Test set is empty after split.")

    # --- Prep features ---
    train_df = _prep_features(train_df)
    test_df = _prep_features(test_df)

    # Drop rows where the label is NaN (wells without enough production coverage)
    train_df = train_df.dropna(subset=[args.label_col]).copy()
    test_df = test_df.dropna(subset=[args.label_col]).copy()

    if len(train_df) == 0:
        raise ValueError("Training set is empty after dropping NaN labels.")
    if len(test_df) == 0:
        raise ValueError("Test set is empty after dropping NaN labels.")

    y_train = train_df[args.label_col].values.astype(int)
    y_test = test_df[args.label_col].values.astype(int)

    print(
        f"[train] Label distribution — train: "
        f"{y_train.sum():,}/{len(y_train):,} positive  |  "
        f"test: {y_test.sum():,}/{len(y_test):,} positive"
    )

    # --- Build & fit ---
    pipeline = _build_pipeline()
    print("[train] Fitting XGBoost pipeline …")
    pipeline.fit(train_df, y_train)

    # --- Predict & evaluate ---
    y_prob = pipeline.predict_proba(test_df)[:, 1]

    region_col = test_df[region_key] if region_key in test_df.columns else None
    metrics = _compute_metrics(y_test, y_prob, region_col=region_col)

    # --- Build output payload ---
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "holdout_mode": args.holdout,
        "holdout_state": args.holdout_state,
        "holdout_basin": args.holdout_basin,
        "label_col": args.label_col,
        "input_csv": str(input_csv),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"[train] Results written to {output_json}")
    pooled = metrics.get("pooled", {})
    print(
        f"[train] Pooled ROC-AUC: {pooled.get('roc_auc')}  |  "
        f"Avg Precision: {pooled.get('average_precision')}"
    )


if __name__ == "__main__":
    main()
