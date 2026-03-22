#!/usr/bin/env python3
"""Publish the latest gas pipeline outputs into results/ for easy diffing."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_id_from_timestamp(ts: str) -> str:
    clean = ts.replace(":", "-")
    clean = clean.replace("+00:00", "Z")
    return clean


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/shin/Mineral_Gas_Locator")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    results_root = repo_root / "results"
    latest_dir = results_root / "latest"
    history_dir = results_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    training_meta_path = repo_root / "data/features/pa_mvp/gas_training_table_v0_metadata.json"
    scoring_meta_path = repo_root / "data/derived/prospect_layers/gas_prospect_cells_v0_metadata.json"
    model_metrics_path = repo_root / "models/pa_mvp/gas_baseline_v0_label_obs12_ge_250000_metrics.json"
    training_meta = load_json(training_meta_path)
    scoring_meta = load_json(scoring_meta_path)
    model_metrics = load_json(model_metrics_path)

    generated_at_utc = scoring_meta["generated_at_utc"]
    v1_training_meta_path = repo_root / "data/features/swpa_core_washington_greene/gas_training_table_v1_metadata.json"
    v1_scoring_meta_path = repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_v1_metadata.json"
    v1_model_metrics_path = repo_root / "models/swpa_core_washington_greene/gas_baseline_v1_label_recent12_ge_250000_metrics.json"
    has_v1 = v1_training_meta_path.exists() and v1_scoring_meta_path.exists() and v1_model_metrics_path.exists()
    v1_training_meta = load_json(v1_training_meta_path) if has_v1 else None
    v1_scoring_meta = load_json(v1_scoring_meta_path) if has_v1 else None
    v1_model_metrics = load_json(v1_model_metrics_path) if has_v1 else None
    if has_v1 and v1_scoring_meta["generated_at_utc"] > generated_at_utc:
        generated_at_utc = v1_scoring_meta["generated_at_utc"]

    prithvi_labels_meta_path = repo_root / "data/features/swpa_core_washington_greene/gas_labels_v2_metadata.json"
    prithvi_chip_meta_path = repo_root / "data/features/swpa_core_washington_greene/hls_chip_index_v1_metadata.json"
    prithvi_embedding_meta_path = repo_root / "data/features/swpa_core_washington_greene/prithvi_embeddings_v1_metadata.json"
    prithvi_scoring_meta_path = repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_prithvi_v1_metadata.json"
    prithvi_summary_path = repo_root / "data/derived/swpa_core_washington_greene/prithvi_gas_pipeline_v1_summary.json"
    prithvi_metric_candidates = sorted(
        (repo_root / "models/swpa_core_washington_greene").glob("gas_prithvi_xgboost_v1_*_metrics.json")
    )
    prithvi_model_metrics_path = prithvi_metric_candidates[-1] if prithvi_metric_candidates else None
    prithvi_model_path = (
        prithvi_model_metrics_path.with_name(prithvi_model_metrics_path.name.replace("_metrics.json", ".joblib"))
        if prithvi_model_metrics_path
        else None
    )
    has_prithvi = (
        prithvi_labels_meta_path.exists()
        and prithvi_chip_meta_path.exists()
        and prithvi_embedding_meta_path.exists()
        and prithvi_scoring_meta_path.exists()
        and prithvi_summary_path.exists()
        and prithvi_model_metrics_path is not None
        and prithvi_model_path is not None
        and prithvi_model_path.exists()
    )
    prithvi_labels_meta = load_json(prithvi_labels_meta_path) if has_prithvi else None
    prithvi_chip_meta = load_json(prithvi_chip_meta_path) if has_prithvi else None
    prithvi_embedding_meta = load_json(prithvi_embedding_meta_path) if has_prithvi else None
    prithvi_scoring_meta = load_json(prithvi_scoring_meta_path) if has_prithvi else None
    prithvi_model_metrics = load_json(prithvi_model_metrics_path) if has_prithvi else None
    prithvi_summary = load_json(prithvi_summary_path) if has_prithvi else None
    if has_prithvi and prithvi_scoring_meta["generated_at_utc"] > generated_at_utc:
        generated_at_utc = prithvi_scoring_meta["generated_at_utc"]

    wv_training_meta_path = repo_root / "data/features/wv_horizontal_statewide/gas_training_table_wv_v0_metadata.json"
    wv_scoring_meta_path = repo_root / "data/derived/wv_horizontal_statewide/gas_prospect_cells_wv_v0_metadata.json"
    wv_metric_candidates = sorted(
        (repo_root / "models/wv_horizontal_statewide").glob("gas_baseline_wv_v0_*_metrics.json")
    )
    wv_model_metrics_path = wv_metric_candidates[-1] if wv_metric_candidates else None
    wv_model_path = (
        wv_model_metrics_path.with_name(wv_model_metrics_path.name.replace("_metrics.json", ".joblib"))
        if wv_model_metrics_path
        else None
    )
    has_wv = (
        wv_training_meta_path.exists()
        and wv_scoring_meta_path.exists()
        and wv_model_metrics_path is not None
        and wv_model_path is not None
        and wv_model_path.exists()
    )
    wv_training_meta = load_json(wv_training_meta_path) if has_wv else None
    wv_scoring_meta = load_json(wv_scoring_meta_path) if has_wv else None
    wv_model_metrics = load_json(wv_model_metrics_path) if has_wv else None
    if has_wv and wv_scoring_meta["generated_at_utc"] > generated_at_utc:
        generated_at_utc = wv_scoring_meta["generated_at_utc"]

    snapshot_id = snapshot_id_from_timestamp(generated_at_utc)
    snapshot_dir = history_dir / snapshot_id

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    output_files = [
        repo_root / "data/features/pa_mvp/gas_training_table_v0.csv",
        training_meta_path,
        repo_root / "models/pa_mvp/gas_baseline_v0_label_obs12_ge_250000.joblib",
        model_metrics_path,
        repo_root / "data/derived/prospect_layers/gas_prospect_cells_v0.csv",
        repo_root / "data/derived/prospect_layers/gas_prospect_cells_v0.geojson",
        scoring_meta_path,
    ]
    if has_v1:
        output_files.extend(
            [
                repo_root / "data/features/swpa_core_washington_greene/gas_training_table_v1.csv",
                v1_training_meta_path,
                repo_root / "models/swpa_core_washington_greene/gas_baseline_v1_label_recent12_ge_250000.joblib",
                v1_model_metrics_path,
                repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_v1.csv",
                repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_v1.geojson",
                repo_root / "data/derived/swpa_core_washington_greene/top_prospects_v1.csv",
                v1_scoring_meta_path,
            ]
        )
    if has_prithvi:
        output_files.extend(
            [
                repo_root / "data/features/swpa_core_washington_greene/gas_labels_v2.parquet",
                prithvi_labels_meta_path,
                repo_root / "data/features/swpa_core_washington_greene/tabular_features_v2.parquet",
                repo_root / "data/features/swpa_core_washington_greene/tabular_candidate_cells_v2.parquet",
                repo_root / "data/features/swpa_core_washington_greene/hls_chip_index_v1.parquet",
                prithvi_chip_meta_path,
                repo_root / "data/features/swpa_core_washington_greene/prithvi_embeddings_v1.parquet",
                prithvi_embedding_meta_path,
                repo_root / "data/features/swpa_core_washington_greene/fused_features_v1.parquet",
                prithvi_model_path,
                prithvi_model_metrics_path,
                repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_prithvi_v1.csv",
                repo_root / "data/derived/swpa_core_washington_greene/gas_prospect_cells_prithvi_v1.geojson",
                repo_root / "data/derived/swpa_core_washington_greene/top_prospects_prithvi_v1.csv",
                prithvi_scoring_meta_path,
                prithvi_summary_path,
            ]
        )
    if has_wv:
        output_files.extend(
            [
                repo_root / "data/features/wv_horizontal_statewide/gas_training_table_wv_v0.csv",
                wv_training_meta_path,
                wv_model_path,
                wv_model_metrics_path,
                repo_root / "data/derived/wv_horizontal_statewide/gas_prospect_cells_wv_v0.csv",
                repo_root / "data/derived/wv_horizontal_statewide/gas_prospect_cells_wv_v0.geojson",
                repo_root / "data/derived/wv_horizontal_statewide/top_prospects_wv_v0.csv",
                wv_scoring_meta_path,
            ]
        )

    copied = []
    for src in output_files:
        rel = src.relative_to(repo_root)
        dst = snapshot_dir / rel
        copy_file(src, dst)
        copied.append(str(rel))

    summary = {
        "snapshot_id": snapshot_id,
        "generated_at_utc": generated_at_utc,
        "v0": {
            "training_dataset": training_meta,
            "model_metrics": model_metrics,
            "prospect_layer": scoring_meta,
        },
        "v1": (
            {
                "training_dataset": v1_training_meta,
                "model_metrics": v1_model_metrics,
                "prospect_layer": v1_scoring_meta,
            }
            if has_v1
            else None
        ),
        "prithvi_v1": (
            {
                "labels": prithvi_labels_meta,
                "hls_chip_index": prithvi_chip_meta,
                "embeddings": prithvi_embedding_meta,
                "model_metrics": prithvi_model_metrics,
                "prospect_layer": prithvi_scoring_meta,
                "pipeline_summary": prithvi_summary,
            }
            if has_prithvi
            else None
        ),
        "wv_v0": (
            {
                "training_dataset": wv_training_meta,
                "model_metrics": wv_model_metrics,
                "prospect_layer": wv_scoring_meta,
            }
            if has_wv
            else None
        ),
        "copied_files": copied,
    }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(snapshot_dir, latest_dir)

    index_path = results_root / "index.json"
    index = load_json(index_path) if index_path.exists() else {"snapshots": []}
    snapshots = [item for item in index.get("snapshots", []) if item.get("snapshot_id") != snapshot_id]
    snapshots.append(
        {
            "snapshot_id": snapshot_id,
            "generated_at_utc": generated_at_utc,
            "summary_path": f"history/{snapshot_id}/summary.json",
            "prospect_row_count": scoring_meta["row_count"],
            "score_max": scoring_meta["score_max"],
            "roc_auc_test": model_metrics["roc_auc_test"],
            "average_precision_test": model_metrics["average_precision_test"],
            "v1_prospect_row_count": v1_scoring_meta["row_count"] if has_v1 else None,
            "v1_score_max": v1_scoring_meta["score_max"] if has_v1 else None,
            "v1_roc_auc_test": v1_model_metrics["roc_auc_test"] if has_v1 else None,
            "prithvi_prospect_row_count": prithvi_scoring_meta["row_count"] if has_prithvi else None,
            "prithvi_score_max": prithvi_scoring_meta["score_max"] if has_prithvi else None,
            "prithvi_roc_auc_test": prithvi_model_metrics["roc_auc_test"] if has_prithvi else None,
            "prithvi_average_precision_test": prithvi_model_metrics["average_precision_test"] if has_prithvi else None,
            "wv_prospect_row_count": wv_scoring_meta["row_count"] if has_wv else None,
            "wv_score_max": wv_scoring_meta["score_max"] if has_wv else None,
            "wv_roc_auc_test": wv_model_metrics["roc_auc_test"] if has_wv else None,
            "wv_average_precision_test": wv_model_metrics["average_precision_test"] if has_wv else None,
        }
    )
    snapshots.sort(key=lambda item: item["generated_at_utc"])
    index = {"latest_snapshot_id": snapshot_id, "snapshots": snapshots}
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
