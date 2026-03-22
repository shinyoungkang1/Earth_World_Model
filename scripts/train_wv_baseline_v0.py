#!/usr/bin/env python3
"""Train the first WV horizontal gas baseline model on GPU XGBoost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from wv_gas_common import load_basin_config, parse_date


def make_pipeline(random_state: int, scale_pos_weight: float) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), "drop"),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                "drop",
            ),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        device="cuda",
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="wv_horizontal_statewide")
    parser.add_argument("--label-column", default="label_f12_ge_2000000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    features_dir = repo_root / "data" / "features" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    models_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(features_dir / "gas_training_table_wv_v0.csv")
    metadata = json.loads((features_dir / "gas_training_table_wv_v0_metadata.json").read_text(encoding="utf-8"))
    if args.label_column not in table.columns:
        raise SystemExit(f"Unknown label column: {args.label_column}")

    data = table[
        table["label_f12_available"].eq(True)
        & table["first_prod_date"].notna()
        & table[args.label_column].notna()
    ].copy()
    data["first_prod_date"] = parse_date(data["first_prod_date"])
    data = data[data["first_prod_date"].notna()].copy()
    data[args.label_column] = data[args.label_column].astype(bool)

    split_date = data["first_prod_date"].quantile(args.holdout_quantile)
    train = data[data["first_prod_date"] <= split_date].copy()
    test = data[data["first_prod_date"] > split_date].copy()
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set.")
    if train[args.label_column].nunique() < 2 or test[args.label_column].nunique() < 2:
        raise SystemExit("Temporal split produced a single-class train or test set.")

    feature_cols_numeric = metadata["feature_columns_numeric"]
    feature_cols_categorical = metadata["feature_columns_categorical"]

    neg = int((~train[args.label_column]).sum())
    pos = int(train[args.label_column].sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0

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
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=args.random_state,
        tree_method="hist",
        device="cuda",
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train = train[feature_cols_numeric + feature_cols_categorical]
    y_train = train[args.label_column]
    X_test = test[feature_cols_numeric + feature_cols_categorical]
    y_test = test[args.label_column]
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_.tolist()
    top_features = sorted(
        [{"feature": name, "importance": float(importance)} for name, importance in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )[:25]

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
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
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=args.random_state,
                    tree_method="hist",
                    device="cuda",
                    scale_pos_weight=float((~data[args.label_column]).sum() / max(int(data[args.label_column].sum()), 1)),
                ),
            ),
        ]
    )
    final_pipeline.fit(data[feature_cols_numeric + feature_cols_categorical], data[args.label_column])

    metrics = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_wv_v0",
        "basin_id": args.basin_id,
        "basin_name": basin["display_name"],
        "model_type": "xgboost_classifier_gpu",
        "label_column": args.label_column,
        "label_definition": metadata["label_definition"],
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc_test": float(roc_auc_score(y_test, test_scores)),
        "average_precision_test": float(average_precision_score(y_test, test_scores)),
        "temporal_holdout_quantile": args.holdout_quantile,
        "temporal_holdout_cutoff_date": split_date.strftime("%Y-%m-%d"),
        "train_first_prod_start": train["first_prod_date"].min().strftime("%Y-%m-%d"),
        "train_first_prod_end": train["first_prod_date"].max().strftime("%Y-%m-%d"),
        "test_first_prod_start": test["first_prod_date"].min().strftime("%Y-%m-%d"),
        "test_first_prod_end": test["first_prod_date"].max().strftime("%Y-%m-%d"),
        "feature_cols_numeric": feature_cols_numeric,
        "feature_cols_categorical": feature_cols_categorical,
        "scale_pos_weight_train": scale_pos_weight,
        "top_feature_importances": top_features,
        "evaluation_note": "Metrics are from a temporal first-production holdout. The saved deployment model is retrained on the full labeled WV dataset after evaluation.",
    }

    model_path = models_dir / f"gas_baseline_wv_v0_{args.label_column}.joblib"
    metrics_path = models_dir / f"gas_baseline_wv_v0_{args.label_column}_metrics.json"
    joblib.dump(final_pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "model_path": str(model_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
