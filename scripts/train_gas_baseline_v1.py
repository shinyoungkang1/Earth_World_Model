#!/usr/bin/env python3
"""Train the basin-scoped gas v1 model with a temporal holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gas_v1_common import load_basin_config, load_json, parse_date


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--label-column", default="label_recent12_ge_250000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    features_dir = repo_root / "data" / "features" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    models_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(features_dir / "gas_training_table_v1.csv")
    metadata = load_json(features_dir / "gas_training_table_v1_metadata.json")
    label_col = args.label_column
    if label_col not in table.columns:
        raise SystemExit(f"Unknown label column: {label_col}")

    data = table[(table["trainable_label_v1"] == True) & table["spud_date"].notna() & table[label_col].notna()].copy()
    data["spud_date"] = parse_date(data["spud_date"])
    data.loc[data["spud_date"] < pd.Timestamp("1950-01-01"), "spud_date"] = pd.NaT
    data = data[data["spud_date"].notna()].copy()
    data[label_col] = data[label_col].astype(bool)

    split_date = data["spud_date"].quantile(args.holdout_quantile)
    train = data[data["spud_date"] <= split_date].copy()
    test = data[data["spud_date"] > split_date].copy()
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set.")

    feature_cols_numeric = metadata["feature_columns_numeric"]
    feature_cols_categorical = metadata["feature_columns_categorical"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), feature_cols_numeric),
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
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=args.random_state,
        min_child_samples=20,
        reg_lambda=1.0,
        verbosity=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train = train[feature_cols_numeric + feature_cols_categorical]
    y_train = train[label_col]
    X_test = test[feature_cols_numeric + feature_cols_categorical]
    y_test = test[label_col]
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_.tolist()
    top_features = sorted(
        [{"feature": name, "importance": int(importance)} for name, importance in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )[:25]

    final_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=args.random_state,
                    min_child_samples=20,
                    reg_lambda=1.0,
                    verbosity=-1,
                ),
            ),
        ]
    )
    final_pipeline.fit(data[feature_cols_numeric + feature_cols_categorical], data[label_col])

    metrics = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_v1",
        "basin_id": args.basin_id,
        "basin_name": basin["display_name"],
        "model_type": "lightgbm_classifier",
        "label_column": label_col,
        "label_definition": metadata["label_definition"],
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
        "average_precision_test": float(average_precision_score(y_test, test_scores)),
        "temporal_holdout_quantile": args.holdout_quantile,
        "temporal_holdout_cutoff_date": split_date.strftime("%Y-%m-%d"),
        "train_spud_start": train["spud_date"].min().strftime("%Y-%m-%d"),
        "train_spud_end": train["spud_date"].max().strftime("%Y-%m-%d"),
        "test_spud_start": test["spud_date"].min().strftime("%Y-%m-%d"),
        "test_spud_end": test["spud_date"].max().strftime("%Y-%m-%d"),
        "feature_cols_numeric": feature_cols_numeric,
        "feature_cols_categorical": feature_cols_categorical,
        "top_feature_importances": top_features,
        "evaluation_note": "Metrics are from a temporal holdout only. The saved deployment model is retrained on the full basin-labeled dataset after evaluation.",
    }

    model_path = models_dir / f"gas_baseline_v1_{label_col}.joblib"
    metrics_path = models_dir / f"gas_baseline_v1_{label_col}_metrics.json"
    joblib.dump(final_pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "model_path": str(model_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
