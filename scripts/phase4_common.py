#!/usr/bin/env python3
"""Shared helpers for the corrected Phase 4 gas experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table extension for {path}")


def resolve_metadata_path(data_path: Path, explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    return data_path.with_name(f"{data_path.stem}_metadata.json")


def ensure_feature_columns(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in numeric_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in categorical_columns:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = result[column].astype(object)
        result[column] = result[column].where(pd.notna(result[column]), np.nan)
    return result


def temporal_split_candidates(preferred_quantile: float) -> list[float]:
    candidates = [preferred_quantile]
    for delta in [0.05, 0.1, 0.15, 0.2]:
        for direction in [-1, 1]:
            quantile = round(preferred_quantile + (delta * direction), 4)
            if 0.5 <= quantile < 0.98:
                candidates.append(quantile)
    ordered: list[float] = []
    seen: set[float] = set()
    for quantile in candidates:
        if quantile not in seen:
            seen.add(quantile)
            ordered.append(quantile)
    return ordered


def choose_temporal_split_classification(
    frame: pd.DataFrame,
    date_col: str,
    label_col: str,
    preferred_quantile: float,
) -> tuple[float, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    data = frame.sort_values(date_col).copy()
    for quantile in temporal_split_candidates(preferred_quantile):
        split_date = data[date_col].quantile(quantile)
        train = data[data[date_col] <= split_date].copy()
        test = data[data[date_col] > split_date].copy()
        if train.empty or test.empty:
            continue
        if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
            continue
        return quantile, split_date, train, test
    split_date = data[date_col].quantile(preferred_quantile)
    train = data[data[date_col] <= split_date].copy()
    test = data[data[date_col] > split_date].copy()
    return preferred_quantile, split_date, train, test


def choose_temporal_split_regression(
    frame: pd.DataFrame,
    date_col: str,
    preferred_quantile: float,
) -> tuple[float, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    data = frame.sort_values(date_col).copy()
    for quantile in temporal_split_candidates(preferred_quantile):
        split_date = data[date_col].quantile(quantile)
        train = data[data[date_col] <= split_date].copy()
        test = data[data[date_col] > split_date].copy()
        if train.empty or test.empty:
            continue
        return quantile, split_date, train, test
    split_date = data[date_col].quantile(preferred_quantile)
    train = data[data[date_col] <= split_date].copy()
    test = data[data[date_col] > split_date].copy()
    return preferred_quantile, split_date, train, test


def split_from_cohort(
    frame: pd.DataFrame,
    task_type: str,
    target_column: str,
    preferred_quantile: float,
    date_col: str = "first_prod_date",
) -> tuple[str, float | None, pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    if "split_group" in frame.columns:
        train = frame[frame["split_group"] == "train"].copy()
        test = frame[frame["split_group"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError("Cohort split_group produced an empty train or test set.")
        if task_type == "classification":
            if train[target_column].nunique() < 2 or test[target_column].nunique() < 2:
                raise ValueError("Cohort split_group produced a single-class train or test set.")
        return "cohort_split_group", None, None, train, test

    if task_type == "classification":
        quantile, split_date, train, test = choose_temporal_split_classification(
            frame,
            date_col=date_col,
            label_col=target_column,
            preferred_quantile=preferred_quantile,
        )
    else:
        quantile, split_date, train, test = choose_temporal_split_regression(
            frame,
            date_col=date_col,
            preferred_quantile=preferred_quantile,
        )
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set.")
    return "temporal_quantile", quantile, split_date, train, test


def build_classification_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    class_weight_scale: float,
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
    model = XGBClassifier(
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
        eval_metric="aucpr",
        scale_pos_weight=class_weight_scale,
        n_jobs=0,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


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


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def compute_regression_metrics(
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
