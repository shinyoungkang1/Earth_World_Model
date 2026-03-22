#!/usr/bin/env python3
"""Train the basin-scoped gas v2 model with true F12/F24 labels.

v2 fixes the broken v1 proxy labels (recent 12-month window) by using
day-prorated first-life production labels.  Also adds mature neighbor
features and switches to XGBoost (proven on WV with ROC-AUC 0.814).

Temporal holdout splits on first_prod_date (not spud_date like v1).

Usage:
    python scripts/train_gas_baseline_v2.py
    python scripts/train_gas_baseline_v2.py --label-column label_f12_ge_250000
    python scripts/train_gas_baseline_v2.py --basin-id pa_northeast_susquehanna
"""

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

from gas_v1_common import load_basin_config, parse_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the gas v2 model with true first-life labels.",
    )
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="XGBoost device (default: cuda).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    basin = load_basin_config(repo_root, args.basin_id)
    features_dir = repo_root / "data" / "features" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load v2 training table ----
    table = pd.read_csv(features_dir / "gas_training_table_v2.csv", low_memory=False)
    metadata = json.loads(
        (features_dir / "gas_training_table_v2_metadata.json").read_text(encoding="utf-8")
    )
    label_col = args.label_column
    if label_col not in table.columns:
        raise SystemExit(f"Unknown label column: {label_col}")

    # ---- Filter to trainable rows with true F12 labels ----
    data = table[
        table["label_f12_available"].eq(True)
        & table["first_prod_date"].notna()
        & table[label_col].notna()
    ].copy()
    data["first_prod_date"] = parse_date(data["first_prod_date"])
    data = data[data["first_prod_date"].notna()].copy()
    data[label_col] = data[label_col].astype(bool)

    print(f"Trainable rows with F12 labels: {len(data)}")
    print(f"Positive rate: {data[label_col].mean():.3f}")

    # ---- Temporal holdout on first_prod_date ----
    split_date = data["first_prod_date"].quantile(args.holdout_quantile)
    train = data[data["first_prod_date"] <= split_date].copy()
    test = data[data["first_prod_date"] > split_date].copy()
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set.")
    if train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
        raise SystemExit("Temporal split produced a single-class train or test set.")

    print(f"Train: {len(train)} rows ({train[label_col].mean():.3f} positive)")
    print(f"Test:  {len(test)} rows ({test[label_col].mean():.3f} positive)")
    print(f"Split date: {split_date.strftime('%Y-%m-%d')}")

    # ---- Feature columns from metadata ----
    feature_cols_numeric = metadata["feature_columns_numeric"]
    feature_cols_categorical = metadata["feature_columns_categorical"]

    # ---- Class weighting ----
    neg = int((~train[label_col]).sum())
    pos = int(train[label_col].sum())
    scale_pos_weight = float(neg / pos) if pos else 1.0

    # ---- Build pipeline ----
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
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
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="aucpr",
        random_state=args.random_state,
        tree_method="hist",
        device=args.device,
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    # ---- Train and evaluate ----
    X_train = train[feature_cols_numeric + feature_cols_categorical]
    y_train = train[label_col]
    X_test = test[feature_cols_numeric + feature_cols_categorical]
    y_test = test[label_col]
    pipeline.fit(X_train, y_train)
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_test, test_scores))
    avg_precision = float(average_precision_score(y_test, test_scores))
    print(f"\nROC-AUC (test):          {roc_auc:.4f}")
    print(f"Average Precision (test): {avg_precision:.4f}")

    # ---- Feature importances ----
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_.tolist()
    top_features = sorted(
        [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:25]

    # ---- Retrain final model on all data ----
    final_scale_pos_weight = float(
        (~data[label_col]).sum() / max(int(data[label_col].sum()), 1)
    )
    final_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
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
    final_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="aucpr",
        random_state=args.random_state,
        tree_method="hist",
        device=args.device,
        scale_pos_weight=final_scale_pos_weight,
    )
    final_pipeline = Pipeline(
        steps=[("preprocess", final_preprocess), ("model", final_model)]
    )
    final_pipeline.fit(
        data[feature_cols_numeric + feature_cols_categorical], data[label_col]
    )

    # ---- Save model and metrics ----
    metrics = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_training_table_v2",
        "basin_id": args.basin_id,
        "basin_name": basin["display_name"],
        "model_type": "xgboost_classifier",
        "model_version": "v2",
        "device": args.device,
        "label_column": label_col,
        "label_definition": metadata["label_definition"],
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc_test": roc_auc,
        "average_precision_test": avg_precision,
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
        "v1_comparison_note": (
            "v1 used broken recent-12-month proxy labels (ROC-AUC 0.524). "
            "v2 uses true day-prorated F12 first-life labels and adds mature "
            "neighbor features (proven on WV with ROC-AUC 0.814)."
        ),
        "evaluation_note": (
            "Metrics are from a temporal first-production holdout. "
            "The saved deployment model is retrained on the full labeled dataset after evaluation."
        ),
    }

    model_path = models_dir / f"gas_baseline_v2_{label_col}.joblib"
    metrics_path = models_dir / f"gas_baseline_v2_{label_col}_metrics.json"
    joblib.dump(final_pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nModel saved: {model_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
