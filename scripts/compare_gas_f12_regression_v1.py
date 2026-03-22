#!/usr/bin/env python3
"""Compare tabular-only vs tabular+embedding regression on a fixed fused cohort.

This is meant for fairer local follow-up analysis after embeddings already exist.
It trains both models on the same cohort of wells, using the same temporal split,
and predicts actual `f12_gas` via regression on `log1p(f12_gas)` by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def choose_temporal_split(
    frame: pd.DataFrame,
    date_col: str,
    preferred_quantile: float,
) -> tuple[float, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    split_date = frame[date_col].quantile(preferred_quantile)
    train = frame[frame[date_col] <= split_date].copy()
    test = frame[frame[date_col] > split_date].copy()
    return preferred_quantile, split_date, train, test


def build_regression_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
    device: str,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        tree_method="hist",
        device=device,
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=0,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def compute_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_true_model: np.ndarray,
    y_pred_model: np.ndarray,
) -> dict[str, float | None]:
    spearman = pd.Series(y_true_raw).corr(pd.Series(y_pred_raw), method="spearman")
    return {
        "rmse_model_scale": float(math.sqrt(mean_squared_error(y_true_model, y_pred_model))),
        "mae_model_scale": float(mean_absolute_error(y_true_model, y_pred_model)),
        "r2_model_scale": float(r2_score(y_true_model, y_pred_model)),
        "rmse_raw": float(math.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
        "mae_raw": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "r2_raw": float(r2_score(y_true_raw, y_pred_raw)),
        "spearman_raw": float(spearman) if pd.notna(spearman) else None,
    }


def transform_target(values: pd.Series, transform: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float).to_numpy()
    if transform == "log1p":
        return np.log1p(numeric)
    if transform == "none":
        return numeric
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(values)
    if transform == "none":
        return values
    raise ValueError(f"Unsupported target transform: {transform}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tabular and fused regression on a fixed cohort.")
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--training-table-path", default="/tmp/regression_inputs/gas_training_table_v2.csv")
    parser.add_argument("--training-metadata-path", default="/tmp/regression_inputs/gas_training_table_v2_metadata.json")
    parser.add_argument("--fused-path", required=True)
    parser.add_argument("--cohort-name", required=True)
    parser.add_argument("--target-column", default="f12_gas")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--target-transform", choices=["log1p", "none"], default="log1p")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    models_dir = repo_root / "models" / args.basin_id
    models_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.training_table_path, low_memory=False)
    metadata = load_json(Path(args.training_metadata_path))
    fused = pd.read_parquet(args.fused_path)
    fused = fused[fused.get("sample_type", "well") == "well"].copy()

    if "sample_id" not in table.columns:
        table["sample_id"] = table["well_api"].map(lambda value: f"well::{value}")
    table["first_prod_date"] = parse_date(table["first_prod_date"])

    feature_cols_numeric = metadata["feature_columns_numeric"]
    feature_cols_categorical = metadata["feature_columns_categorical"]

    join_cols = [
        "sample_id",
        "well_api",
        "first_prod_date",
        args.target_column,
        "label_f12_available",
        *feature_cols_numeric,
        *feature_cols_categorical,
    ]
    available_join_cols = [col for col in join_cols if col in table.columns]
    cohort = fused.merge(
        table[available_join_cols].drop_duplicates(subset=["sample_id"]),
        on="sample_id",
        how="left",
        suffixes=("", "_table"),
    ).copy()

    if "first_prod_date" not in cohort.columns and "first_prod_date_table" in cohort.columns:
        cohort["first_prod_date"] = cohort["first_prod_date_table"]
    if args.target_column not in cohort.columns and f"{args.target_column}_table" in cohort.columns:
        cohort[args.target_column] = cohort[f"{args.target_column}_table"]
    if "label_f12_available" not in cohort.columns and "label_f12_available_table" in cohort.columns:
        cohort["label_f12_available"] = cohort["label_f12_available_table"]
    for column in feature_cols_numeric + feature_cols_categorical:
        table_column = f"{column}_table"
        if column not in cohort.columns and table_column in cohort.columns:
            cohort[column] = cohort[table_column]

    cohort["first_prod_date"] = parse_date(cohort["first_prod_date"])
    cohort[args.target_column] = pd.to_numeric(cohort[args.target_column], errors="coerce")
    cohort = cohort[
        cohort["label_f12_available"].eq(True)
        & cohort["first_prod_date"].notna()
        & cohort[args.target_column].notna()
    ].copy()
    cohort = cohort.sort_values("first_prod_date").reset_index(drop=True)
    if cohort.empty:
        raise SystemExit("No rows remain in the fused cohort after merging the regression target.")

    embedding_columns = [
        column for column in cohort.columns if column.startswith("embedding_") or column.startswith("ewm_embedding_")
    ]
    if not embedding_columns:
        raise SystemExit("No embedding columns found in the fused cohort.")

    chosen_quantile, split_date, train, test = choose_temporal_split(
        cohort,
        date_col="first_prod_date",
        preferred_quantile=args.holdout_quantile,
    )
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set.")

    y_train_raw = pd.to_numeric(train[args.target_column], errors="coerce").fillna(0.0).to_numpy()
    y_test_raw = pd.to_numeric(test[args.target_column], errors="coerce").fillna(0.0).to_numpy()
    y_train_model = transform_target(train[args.target_column], args.target_transform)
    y_test_model = transform_target(test[args.target_column], args.target_transform)

    base_features = feature_cols_numeric + feature_cols_categorical
    fused_features = base_features + embedding_columns

    baseline = build_regression_pipeline(
        numeric_columns=feature_cols_numeric,
        categorical_columns=feature_cols_categorical,
        random_state=args.random_state,
        device=args.device,
    )
    baseline.fit(train[base_features], y_train_model)
    baseline_pred_model = baseline.predict(test[base_features])
    baseline_pred_raw = inverse_transform_target(baseline_pred_model, args.target_transform)

    fused_model = build_regression_pipeline(
        numeric_columns=feature_cols_numeric + embedding_columns,
        categorical_columns=feature_cols_categorical,
        random_state=args.random_state,
        device=args.device,
    )
    fused_model.fit(train[fused_features], y_train_model)
    fused_pred_model = fused_model.predict(test[fused_features])
    fused_pred_raw = inverse_transform_target(fused_pred_model, args.target_transform)

    result = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_f12_regression_comparison_v1",
        "basin_id": args.basin_id,
        "cohort_name": args.cohort_name,
        "fused_path": str(Path(args.fused_path).resolve()),
        "target_column": args.target_column,
        "target_transform": args.target_transform,
        "row_count_cohort": int(len(cohort)),
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "embedding_column_count": int(len(embedding_columns)),
        "temporal_holdout_quantile": float(chosen_quantile),
        "temporal_holdout_cutoff_date": split_date.strftime("%Y-%m-%d"),
        "train_first_prod_start": train["first_prod_date"].min().strftime("%Y-%m-%d"),
        "train_first_prod_end": train["first_prod_date"].max().strftime("%Y-%m-%d"),
        "test_first_prod_start": test["first_prod_date"].min().strftime("%Y-%m-%d"),
        "test_first_prod_end": test["first_prod_date"].max().strftime("%Y-%m-%d"),
        "baseline": compute_metrics(y_test_raw, baseline_pred_raw, y_test_model, baseline_pred_model),
        "fused": compute_metrics(y_test_raw, fused_pred_raw, y_test_model, fused_pred_model),
        "notes": [
            "Baseline and fused models were trained on the same fused cohort.",
            "This isolates the value of adding embeddings better than comparing across different cohorts.",
        ],
    }

    baseline_model_path = models_dir / f"gas_f12_regression_baseline_{args.cohort_name}.joblib"
    fused_model_path = models_dir / f"gas_f12_regression_fused_{args.cohort_name}.joblib"
    metrics_path = models_dir / f"gas_f12_regression_compare_{args.cohort_name}_metrics.json"
    joblib.dump(baseline, baseline_model_path)
    joblib.dump(fused_model, fused_model_path)
    result["baseline_model_path"] = str(baseline_model_path)
    result["fused_model_path"] = str(fused_model_path)
    result["metrics_path"] = str(metrics_path)
    write_json(metrics_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
