#!/usr/bin/env python3
"""Run a frozen downstream probe on SSL4EO-S12-downstream example data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from huggingface_hub import HfApi, hf_hub_download
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ewm.data.dataset import DEFAULT_DATES, SSL4EO_S1GRD_MEAN, SSL4EO_S1GRD_STD, SSL4EO_S2L2A_MEAN, SSL4EO_S2L2A_STD
from ewm.models.world_model import EarthWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", type=str, default="landcover_forest__regr")
    parser.add_argument("--network", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--repo-id", type=str, default="embed2scale/SSL4EO-S12-downstream")
    parser.add_argument("--repo-subdir", type=str, default="data_example")
    parser.add_argument("--part", type=str, default="part-000000")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ssl4eo_downstream_probe"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("earth_world_model/checkpoints/ssl4eo_downstream_probe"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def center_crop(frame: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = frame.shape
    y0 = (height - patch_size) // 2
    x0 = (width - patch_size) // 2
    return frame[:, y0 : y0 + patch_size, x0 : x0 + patch_size]


def build_model(checkpoint: dict, device: torch.device) -> EarthWorldModel:
    config = checkpoint["config"]
    model_cfg = config["model"]
    model = EarthWorldModel(
        embed_dim=model_cfg["embed_dim"],
        patch_px=model_cfg["patch_px"],
        temporal_depth=model_cfg["temporal_depth"],
        predictor_depth=model_cfg["predictor_depth"],
        num_heads=model_cfg["num_heads"],
        input_mode=model_cfg.get("input_mode", "s2s1"),
        hierarchical_layers=model_cfg.get("hierarchical_layers"),
        token_mode=model_cfg.get("token_mode", "pooled"),
        spatial_fusion=model_cfg.get("spatial_fusion", "early_mean"),
        use_modality_embeddings=model_cfg.get("use_modality_embeddings", False),
        use_patch_positional_encoding=model_cfg.get("use_patch_positional_encoding", False),
        temporal_block_type=model_cfg.get("temporal_block_type", "standard"),
        use_rope_temporal_attention=model_cfg.get("use_rope_temporal_attention", False),
        rope_base=model_cfg.get("rope_base", 10000.0),
        use_activation_checkpointing=model_cfg.get("use_activation_checkpointing", False),
    )
    model.to(device)
    return model


def load_probe_ids(repo_id: str, repo_subdir: str, part: str) -> set[str]:
    api = HfApi()
    items = list(api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=f"{repo_subdir}/s1/{part}", recursive=False, expand=True))
    return {item.path.split("/")[-1].replace(".zarr.zip", "") for item in items}


def apply_phase_balanced_limit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.reset_index(drop=True)
    required = {"cvpr_earthvision_phase_dev", "cvpr_earthvision_phase_eval"}
    if not required.issubset(frame.columns):
        return frame.head(limit).reset_index(drop=True)

    dev = frame[frame["cvpr_earthvision_phase_dev"].fillna(False)].copy()
    eval_ = frame[frame["cvpr_earthvision_phase_eval"].fillna(False)].copy()
    if dev.empty or eval_.empty:
        return frame.head(limit).reset_index(drop=True)

    dev_take = min(len(dev), max(1, limit // 2))
    eval_take = min(len(eval_), max(1, limit - dev_take))
    chosen = pd.concat([dev.head(dev_take), eval_.head(eval_take)], axis=0)

    if len(chosen) < limit:
        chosen_ids = set(chosen["id"].tolist())
        remainder = frame[~frame["id"].isin(chosen_ids)].head(limit - len(chosen))
        chosen = pd.concat([chosen, remainder], axis=0)
    return chosen.head(limit).reset_index(drop=True)


def load_labels(label_path: Path, available_ids: set[str], limit: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(label_path)
    frame = frame[frame["id"].isin(available_ids)].copy()
    if limit > 0:
        frame = apply_phase_balanced_limit(frame, limit)
    return frame.reset_index(drop=True)


def download_sample(repo_id: str, cache_dir: Path, repo_subdir: str, part: str, sample_id: str) -> tuple[Path, Path]:
    s1_file = f"{repo_subdir}/s1/{part}/{sample_id}.zarr.zip"
    s2_file = f"{repo_subdir}/s2l2a/{part}/{sample_id}.zarr.zip"
    s1_path = Path(hf_hub_download(repo_id, repo_type="dataset", filename=s1_file, local_dir=str(cache_dir)))
    s2_path = Path(hf_hub_download(repo_id, repo_type="dataset", filename=s2_file, local_dir=str(cache_dir)))
    return s1_path, s2_path


def load_modalities(s1_path: Path, s2_path: Path, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    ds_s1 = xr.open_zarr(str(s1_path))
    ds_s2 = xr.open_zarr(str(s2_path))
    s1 = torch.from_numpy(np.asarray(ds_s1["bands"].values)).float().squeeze(0)
    s2 = torch.from_numpy(np.asarray(ds_s2["bands"].values)).float().squeeze(0)
    s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
    s1 = torch.stack([center_crop(frame, patch_size) for frame in s1], dim=0)
    s2 = torch.stack([center_crop(frame, patch_size) for frame in s2], dim=0)
    s1 = (s1 - SSL4EO_S1GRD_MEAN.unsqueeze(0)) / (SSL4EO_S1GRD_STD.unsqueeze(0) + 1e-6)
    s2 = (s2 - SSL4EO_S2L2A_MEAN.unsqueeze(0)) / (SSL4EO_S2L2A_STD.unsqueeze(0) + 1e-6)
    return s1, s2


def compute_ewm_embedding(model: EarthWorldModel, device: torch.device, s1: torch.Tensor, s2: torch.Tensor) -> np.ndarray:
    dates = DEFAULT_DATES.view(1, -1).to(device)
    mask = torch.ones_like(dates, dtype=torch.bool, device=device)
    with torch.no_grad():
        timeline = model.encode_timeline(
            s2.unsqueeze(0).to(device),
            s1.unsqueeze(0).to(device),
            dates,
            mask,
        )
    return timeline.mean(dim=1).squeeze(0).cpu().numpy()


def compute_single_timestep_baseline(s1: torch.Tensor, s2: torch.Tensor) -> np.ndarray:
    first_s2 = s2[0]
    first_s1 = s1[0]
    features = [
        first_s2.mean(dim=(1, 2)),
        first_s2.std(dim=(1, 2)),
        first_s1.mean(dim=(1, 2)),
        first_s1.std(dim=(1, 2)),
    ]
    return torch.cat(features, dim=0).cpu().numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def fit_and_score(task_name: str, x_train: np.ndarray, x_eval: np.ndarray, y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    if task_name.endswith("__cls"):
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )
        clf.fit(x_train, y_train)
        scores = clf.predict_proba(x_eval)[:, 1]
        return classification_metrics(y_eval, scores)

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    reg.fit(x_train, y_train)
    preds = reg.predict(x_eval)
    return regression_metrics(y_eval, preds)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint, device)
    model.load_state_dict(checkpoint[args.network])
    model.eval()

    available_ids = load_probe_ids(args.repo_id, args.repo_subdir, args.part)
    label_path = Path(
        hf_hub_download(
            args.repo_id,
            repo_type="dataset",
            filename=f"labels/{args.task}.csv",
            local_dir=str(args.cache_dir),
        )
    )
    labels = load_labels(label_path, available_ids, limit=args.limit)
    if labels.empty:
        raise RuntimeError(f"No overlapping labeled samples found for task {args.task}")

    patch_size = int(checkpoint["config"]["data"]["patch_size"])
    records: list[dict[str, float | str | bool]] = []

    for row in labels.itertuples(index=False):
        sample_id = str(row.id)
        s1_path, s2_path = download_sample(args.repo_id, args.cache_dir, args.repo_subdir, args.part, sample_id)
        s1, s2 = load_modalities(s1_path, s2_path, patch_size)
        ewm_embedding = compute_ewm_embedding(model, device, s1, s2)
        baseline = compute_single_timestep_baseline(s1, s2)

        record: dict[str, float | str | bool] = {
            "id": sample_id,
            "label": float(row.label),
            "cvpr_earthvision_phase_dev": bool(getattr(row, "cvpr_earthvision_phase_dev", False)),
            "cvpr_earthvision_phase_eval": bool(getattr(row, "cvpr_earthvision_phase_eval", False)),
        }
        for idx, value in enumerate(ewm_embedding):
            record[f"ewm_{idx:04d}"] = float(value)
        for idx, value in enumerate(baseline):
            record[f"baseline_{idx:04d}"] = float(value)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    embeddings_path = args.output_dir / f"{args.task}__features.parquet"
    frame.to_parquet(embeddings_path, index=False)

    dev_mask = frame["cvpr_earthvision_phase_dev"].fillna(False).to_numpy(dtype=bool)
    eval_mask = frame["cvpr_earthvision_phase_eval"].fillna(False).to_numpy(dtype=bool)
    if not dev_mask.any() or not eval_mask.any():
        raise RuntimeError("Task does not have both dev and eval samples in the selected subset")

    y_dev = frame.loc[dev_mask, "label"].to_numpy(dtype=np.float32)
    y_eval = frame.loc[eval_mask, "label"].to_numpy(dtype=np.float32)
    ewm_cols = [column for column in frame.columns if column.startswith("ewm_")]
    baseline_cols = [column for column in frame.columns if column.startswith("baseline_")]
    ewm_dev = frame.loc[dev_mask, ewm_cols].to_numpy(dtype=np.float32)
    ewm_eval = frame.loc[eval_mask, ewm_cols].to_numpy(dtype=np.float32)
    baseline_dev = frame.loc[dev_mask, baseline_cols].to_numpy(dtype=np.float32)
    baseline_eval = frame.loc[eval_mask, baseline_cols].to_numpy(dtype=np.float32)

    ewm_metrics = fit_and_score(args.task, ewm_dev, ewm_eval, y_dev, y_eval)
    baseline_metrics = fit_and_score(args.task, baseline_dev, baseline_eval, y_dev, y_eval)

    summary = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "network": args.network,
        "repo_id": args.repo_id,
        "repo_subdir": args.repo_subdir,
        "part": args.part,
        "device": str(device),
        "sample_count": int(len(frame)),
        "dev_count": int(dev_mask.sum()),
        "eval_count": int(eval_mask.sum()),
        "ewm_metrics": ewm_metrics,
        "baseline_metrics": baseline_metrics,
        "embeddings_path": str(embeddings_path.resolve()),
    }
    summary_path = args.output_dir / f"{args.task}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
