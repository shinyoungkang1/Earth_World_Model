#!/usr/bin/env python3
"""Train a first non-leaky gas baseline on static/location features only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--label-column", default="label_obs12_ge_250000")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    features_dir = repo_root / "data" / "features" / "pa_mvp"
    models_dir = repo_root / "models" / "pa_mvp"
    models_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(features_dir / "gas_training_table_v0.csv")
    label_col = args.label_column
    if label_col not in table.columns:
        raise SystemExit(f"Unknown label column: {label_col}")

    data = table[table["trainable_label_v0"] == True].copy()
    data = data[data[label_col].notna()].copy()
    data[label_col] = data[label_col].astype(bool)

    feature_cols_numeric = ["latitude_decimal", "longitude_decimal", "spud_year", "spud_month"]
    feature_cols_categorical = [
        "county_name",
        "municipality_name",
        "geology_map_symbol",
        "geology_age",
        "geology_lith1",
    ]

    groups = data["county_name"].fillna("UNKNOWN_COUNTY")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols_numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_cols_categorical,
            ),
        ]
    )
    model = LogisticRegression(max_iter=2000, solver="saga", class_weight="balanced")
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train = train[feature_cols_numeric + feature_cols_categorical]
    y_train = train[label_col]
    X_test = test[feature_cols_numeric + feature_cols_categorical]
    y_test = test[label_col]
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_v0",
        "model_type": "logistic_regression",
        "label_column": label_col,
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
        "average_precision_test": float(average_precision_score(y_test, test_scores)),
        "held_out_counties": sorted(test["county_name"].dropna().unique().tolist()),
        "feature_cols_numeric": feature_cols_numeric,
        "feature_cols_categorical": feature_cols_categorical,
    }

    model_path = models_dir / f"gas_baseline_v0_{label_col}.joblib"
    metrics_path = models_dir / f"gas_baseline_v0_{label_col}_metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "model_path": str(model_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
