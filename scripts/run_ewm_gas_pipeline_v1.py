#!/usr/bin/env python3
"""Run a Phase 4 gas pipeline with frozen Earth World Model embeddings.

This script is intentionally narrower than the Prithvi EO pipeline:

- it does not fetch satellite scenes on its own
- it consumes a prebuilt chip/sample index
- it extracts frozen EWM embeddings
- it trains an XGBoost gas model on top of tabular + EWM features
- it optionally scores candidate prospect cells when grid embeddings exist

Supported chip index modes:

- `hls_chip_index`
  - for checkpoints whose `input_mode` is `hls6`
  - expects the existing `hls_chip_index_v1.parquet` layout with `chip_path`
- `s2s1_npz_manifest`
  - for checkpoints whose `input_mode` is `s2s1`
  - expects a parquet with:
    - `sample_id`
    - `sample_type`
    - `sample_key`
    - one of `sample_path` or `chip_path`
  - the referenced `.npz` must contain:
    - `s2`: `[T, 12, H, W]`
    - `s1`: `[T, 2, H, W]`
    - optional `dates`: `[T]`
    - optional `mask`: `[T]`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dataset import DEFAULT_DATES, SSL4EO_S1GRD_MEAN, SSL4EO_S1GRD_STD, SSL4EO_S2L2A_MEAN, SSL4EO_S2L2A_STD  # noqa: E402
from ewm.models.world_model import EarthWorldModel  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_basin_config(repo_root: Path, basin_id: str) -> dict[str, Any]:
    return load_json(repo_root / "config" / "basins" / f"{basin_id}.json")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def polygon_from_bbox(minx: float, miny: float, maxx: float, maxy: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [[
            [float(minx), float(miny)],
            [float(maxx), float(miny)],
            [float(maxx), float(maxy)],
            [float(minx), float(maxy)],
            [float(minx), float(miny)],
        ]],
    }


def prospect_tier(percentile: float) -> str:
    if percentile >= 0.99:
        return "tier_1"
    if percentile >= 0.95:
        return "tier_2"
    if percentile >= 0.90:
        return "tier_3"
    return "tier_4"


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


def choose_temporal_split(
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


def center_crop(frame: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = frame.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds frame size {(height, width)}")
    y0 = (height - patch_size) // 2
    x0 = (width - patch_size) // 2
    return frame[:, y0 : y0 + patch_size, x0 : x0 + patch_size]


def build_model(checkpoint: dict[str, Any], device: torch.device) -> EarthWorldModel:
    model_cfg = checkpoint["config"]["model"]
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


def load_checkpoint_teacher_weights(model: EarthWorldModel, teacher_state: dict[str, Any]) -> None:
    adapted_state: dict[str, Any] = {}
    legacy_temporal_prefix = "temporal_encoder.layers."
    current_temporal_prefix = "temporal_encoder_layers."
    for key, value in teacher_state.items():
        if key.startswith(legacy_temporal_prefix):
            key = current_temporal_prefix + key[len(legacy_temporal_prefix) :]
        adapted_state[key] = value

    incompatible = model.load_state_dict(adapted_state, strict=False)
    allowed_missing = set()
    if not getattr(model.spatial, "use_modality_embeddings", False):
        allowed_missing.update(
            {
                "spatial.s2_mod_embed",
                "spatial.s1_mod_embed",
                "spatial.hls_mod_embed",
            }
        )

    disallowed_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    unexpected = sorted(incompatible.unexpected_keys)
    if disallowed_missing or unexpected:
        problems: list[str] = []
        if disallowed_missing:
            problems.append(f"missing keys: {disallowed_missing}")
        if unexpected:
            problems.append(f"unexpected keys: {unexpected}")
        raise RuntimeError("Checkpoint/model mismatch after compatibility remap: " + "; ".join(problems))


def resolve_index_path(
    repo_root: Path,
    features_dir: Path,
    args_index_path: str | None,
    input_mode: str,
) -> Path:
    if args_index_path:
        path = Path(args_index_path)
        return path if path.is_absolute() else (repo_root / path)
    if input_mode == "hls6":
        return features_dir / "hls_chip_index_v1.parquet"
    return features_dir / "ewm_s2s1_chip_index_v1.parquet"


def resolve_sample_path(repo_root: Path, row: pd.Series) -> Path:
    for column in ["sample_path", "chip_path"]:
        value = row.get(column)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else (repo_root / path)
    raise KeyError("Row is missing both 'sample_path' and 'chip_path'")


def load_hls_sample(npz_path: Path, patch_size: int) -> dict[str, torch.Tensor]:
    with np.load(npz_path, allow_pickle=False) as payload:
        chip = np.asarray(payload["chip"], dtype=np.float32)  # [6, 4, H, W]
        temporal_coords = np.asarray(payload["temporal_coords"], dtype=np.float32)  # [4, 2]
        valid_mask = np.asarray(payload["valid_mask"], dtype=np.uint8)  # [4, H, W]

    hls = torch.from_numpy(np.transpose(chip, (1, 0, 2, 3))).float()
    valid_pixels = torch.from_numpy(valid_mask.astype(np.float32))
    hls = torch.nan_to_num(hls, nan=0.0, posinf=0.0, neginf=0.0)
    hls = hls * valid_pixels.unsqueeze(1)
    hls = torch.stack([center_crop(frame, patch_size) for frame in hls], dim=0)
    valid_pixels = torch.stack(
        [center_crop(frame.unsqueeze(0), patch_size).squeeze(0) for frame in valid_pixels],
        dim=0,
    )
    dates = torch.from_numpy(np.rint(temporal_coords[:, 1]).astype(np.int32))
    mask = valid_pixels.reshape(valid_pixels.shape[0], -1).mean(dim=1) > 0.5
    return {"hls": hls, "dates": dates, "mask": mask}


def load_s2s1_sample(
    npz_path: Path,
    patch_size: int,
    s2s1_format: str,
) -> dict[str, torch.Tensor]:
    with np.load(npz_path, allow_pickle=False) as payload:
        s2 = torch.from_numpy(np.asarray(payload["s2"])).float()
        s1 = torch.from_numpy(np.asarray(payload["s1"])).float()
        if "dates" in payload.files:
            dates = torch.from_numpy(np.asarray(payload["dates"])).to(torch.int32)
        else:
            dates = DEFAULT_DATES[: s2.shape[0]].clone()
        if "mask" in payload.files:
            mask = torch.from_numpy(np.asarray(payload["mask"])).to(torch.bool)
        else:
            mask = torch.ones(s2.shape[0], dtype=torch.bool)

    if s2.ndim != 4 or s1.ndim != 4:
        raise ValueError(f"Expected 4D s2/s1 tensors in {npz_path}, got {tuple(s2.shape)} and {tuple(s1.shape)}")

    s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
    s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = torch.stack([center_crop(frame, patch_size) for frame in s2], dim=0)
    s1 = torch.stack([center_crop(frame, patch_size) for frame in s1], dim=0)

    if s2s1_format == "raw_ssl4eo":
        s2 = (s2 - SSL4EO_S2L2A_MEAN.unsqueeze(0)) / (SSL4EO_S2L2A_STD.unsqueeze(0) + 1e-6)
        s1 = (s1 - SSL4EO_S1GRD_MEAN.unsqueeze(0)) / (SSL4EO_S1GRD_STD.unsqueeze(0) + 1e-6)
    elif s2s1_format != "preprocessed":
        raise ValueError(f"Unsupported s2s1 format: {s2s1_format}")

    return {"s2": s2, "s1": s1, "dates": dates, "mask": mask}


def extract_ewm_embeddings(
    chip_index: pd.DataFrame,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    batch_size: int,
    device: str,
    repo_root: Path,
    embeddings_path: Path,
    s2s1_format: str,
    embedding_mode: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_mode = checkpoint["config"]["model"].get("input_mode", "s2s1")
    patch_size = int(checkpoint["config"]["data"]["patch_size"])
    cached = pd.DataFrame()
    if embeddings_path.exists():
        cached = pd.read_parquet(embeddings_path)
        cached = cached[cached["sample_id"].isin(chip_index["sample_id"])].copy()

    missing_ids = (
        sorted(set(chip_index["sample_id"]) - set(cached["sample_id"]))
        if not cached.empty
        else chip_index["sample_id"].tolist()
    )
    pending = chip_index[chip_index["sample_id"].isin(missing_ids)].copy().reset_index(drop=True)
    if pending.empty:
        frame = cached.sort_values("sample_id").reset_index(drop=True)
        metadata = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "dataset": "ewm_embeddings_v1",
            "checkpoint": str(checkpoint_path),
            "input_mode": input_mode,
            "embedding_dim": int(frame.filter(regex=r"^ewm_embedding_").shape[1]),
            "batch_size": int(batch_size),
            "device": device,
            "row_count": int(len(frame)),
            "reused_row_count": int(len(frame)),
            "new_row_count": 0,
        }
        return frame, metadata

    torch_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, torch_device)
    load_checkpoint_teacher_weights(model, checkpoint["teacher"])
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch_frame = pending.iloc[start : start + batch_size].copy()
            sample_ids: list[str] = []
            sample_types: list[str] = []
            sample_keys: list[str] = []
            batch_dates: list[torch.Tensor] = []
            batch_mask: list[torch.Tensor] = []

            if input_mode == "hls6":
                batch_hls: list[torch.Tensor] = []
                for row in batch_frame.itertuples(index=False):
                    sample = load_hls_sample(resolve_sample_path(repo_root, pd.Series(row._asdict())), patch_size)
                    batch_hls.append(sample["hls"])
                    batch_dates.append(sample["dates"])
                    batch_mask.append(sample["mask"])
                    sample_ids.append(str(row.sample_id))
                    sample_types.append(str(row.sample_type))
                    sample_keys.append(str(row.sample_key))

                embeddings_tensor = model.extract_embedding(
                    s2=None,
                    s1=None,
                    dates=torch.stack(batch_dates, dim=0).to(device=torch_device),
                    mask=torch.stack(batch_mask, dim=0).to(device=torch_device),
                    hls=torch.stack(batch_hls, dim=0).to(device=torch_device),
                    mode=embedding_mode,
                )
            elif input_mode == "s2s1":
                batch_s2: list[torch.Tensor] = []
                batch_s1: list[torch.Tensor] = []
                for row in batch_frame.itertuples(index=False):
                    sample = load_s2s1_sample(
                        resolve_sample_path(repo_root, pd.Series(row._asdict())),
                        patch_size,
                        s2s1_format=s2s1_format,
                    )
                    batch_s2.append(sample["s2"])
                    batch_s1.append(sample["s1"])
                    batch_dates.append(sample["dates"])
                    batch_mask.append(sample["mask"])
                    sample_ids.append(str(row.sample_id))
                    sample_types.append(str(row.sample_type))
                    sample_keys.append(str(row.sample_key))

                embeddings_tensor = model.extract_embedding(
                    s2=torch.stack(batch_s2, dim=0).to(device=torch_device),
                    s1=torch.stack(batch_s1, dim=0).to(device=torch_device),
                    dates=torch.stack(batch_dates, dim=0).to(device=torch_device),
                    mask=torch.stack(batch_mask, dim=0).to(device=torch_device),
                    mode=embedding_mode,
                )
            else:
                raise ValueError(f"Unsupported checkpoint input_mode: {input_mode}")

            embeddings = embeddings_tensor.cpu().numpy().astype(np.float32)
            for idx, sample_id in enumerate(sample_ids):
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "sample_type": sample_types[idx],
                    "sample_key": sample_keys[idx],
                }
                for col_idx in range(embeddings.shape[1]):
                    record[f"ewm_embedding_{col_idx:04d}"] = float(embeddings[idx, col_idx])
                rows.append(record)

    new_frame = pd.DataFrame(rows)
    frame = (
        pd.concat([cached, new_frame], ignore_index=True)
        .drop_duplicates(subset=["sample_id"], keep="last")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "ewm_embeddings_v1",
        "checkpoint": str(checkpoint_path),
        "input_mode": input_mode,
        "embedding_mode": embedding_mode,
        "embedding_dim": int(frame.filter(regex=r"^ewm_embedding_").shape[1]),
        "batch_size": int(batch_size),
        "device": str(torch_device),
        "row_count": int(len(frame)),
        "reused_row_count": int(len(cached)),
        "new_row_count": int(len(new_frame)),
        "output_path": str(embeddings_path),
    }
    return frame, metadata


def build_model_pipeline(
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


def score_grid_cells(
    basin: dict[str, Any],
    grid_features: pd.DataFrame,
    grid_embeddings: pd.DataFrame,
    model_pipeline: Pipeline,
    feature_columns: list[str],
    label_column: str,
    model_path: Path,
    metrics_path: Path,
    derived_dir: Path,
) -> tuple[Path, Path, Path]:
    merged = grid_features.merge(
        grid_embeddings,
        on=["sample_id", "sample_type", "sample_key"],
        how="inner",
    )
    merged["score"] = model_pipeline.predict_proba(merged[feature_columns])[:, 1]
    merged["score_percentile"] = merged["score"].rank(method="average", pct=True)
    merged["prospect_tier"] = merged["score_percentile"].map(prospect_tier)
    merged = merged.sort_values(["score"], ascending=False).reset_index(drop=True)
    merged["score_rank"] = np.arange(1, len(merged) + 1)

    out_csv = derived_dir / "gas_prospect_cells_ewm_v1.csv"
    merged.to_csv(out_csv, index=False)

    features = []
    for row in merged.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "geometry": polygon_from_bbox(row.bbox_west, row.bbox_south, row.bbox_east, row.bbox_north),
                "properties": {
                    "cell_id": row.sample_key,
                    "score": round(float(row.score), 6),
                    "score_percentile": round(float(row.score_percentile), 6),
                    "score_rank": int(row.score_rank),
                    "prospect_tier": row.prospect_tier,
                    "county_name": row.county_name,
                    "geology_name": row.geology_name,
                    "geology_lith1": row.geology_lith1,
                    "fault_distance_km": round(float(row.fault_distance_km), 3) if pd.notna(row.fault_distance_km) else None,
                    "well_count_5km": int(row.well_count_5km),
                    "permit_count_5km": int(row.permit_count_5km),
                },
            }
        )

    out_geojson = derived_dir / "gas_prospect_cells_ewm_v1.geojson"
    out_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )

    top_cols = [
        "sample_key",
        "score_rank",
        "score",
        "score_percentile",
        "prospect_tier",
        "county_name",
        "geology_name",
        "geology_lith1",
        "fault_distance_km",
        "well_count_5km",
        "permit_count_5km",
        "sample_longitude",
        "sample_latitude",
    ]
    top_path = derived_dir / "top_prospects_ewm_v1.csv"
    merged[top_cols].head(100).to_csv(top_path, index=False)

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_prospect_cells_ewm_v1",
        "basin_id": basin["basin_id"],
        "basin_name": basin["display_name"],
        "row_count": int(len(merged)),
        "score_min": float(merged["score"].min()),
        "score_p50": float(merged["score"].median()),
        "score_p90": float(merged["score"].quantile(0.9)),
        "score_max": float(merged["score"].max()),
        "tier_counts": {
            tier: int(count)
            for tier, count in merged["prospect_tier"].value_counts().sort_index().items()
        },
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "label_column": label_column,
        "output_csv_path": str(out_csv),
        "output_geojson_path": str(out_geojson),
        "top_prospects_path": str(top_path),
    }
    write_json(derived_dir / "gas_prospect_cells_ewm_v1_metadata.json", metadata)
    return out_csv, out_geojson, top_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 frozen-EWM gas pipeline.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--checkpoint", required=True, help="Path to the frozen EWM checkpoint to use.")
    parser.add_argument("--label-column", default="label_f12_ge_500000")
    parser.add_argument("--holdout-quantile", type=float, default=0.8)
    parser.add_argument("--max-training-wells", type=int, default=1600)
    parser.add_argument("--max-grid-cells", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--index-path",
        default=None,
        help="Optional chip/sample index parquet. Defaults depend on checkpoint input_mode.",
    )
    parser.add_argument(
        "--grid-features-path",
        default=None,
        help="Optional path override for candidate cell features. Defaults to data/derived/<basin>/gas_prospect_cells_v1.csv",
    )
    parser.add_argument(
        "--s2s1-format",
        default="raw_ssl4eo",
        choices=["preprocessed", "raw_ssl4eo"],
        help="How to interpret s2s1 manifest npz tensors. The Phase 4 materializer emits raw_ssl4eo.",
    )
    parser.add_argument("--skip-scoring", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else (repo_root / checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_mode = checkpoint["config"]["model"].get("input_mode", "s2s1")

    basin = load_basin_config(repo_root, args.basin_id)
    features_dir = repo_root / "data" / "features" / args.basin_id
    derived_dir = repo_root / "data" / "derived" / args.basin_id
    models_dir = repo_root / "models" / args.basin_id
    features_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tag = checkpoint_path.stem
    embeddings_path = features_dir / f"ewm_embeddings_v1_{checkpoint_tag}.parquet"
    model_path = models_dir / f"gas_ewm_xgboost_v1_{checkpoint_tag}_{args.label_column}.joblib"
    metrics_path = models_dir / f"gas_ewm_xgboost_v1_{checkpoint_tag}_{args.label_column}_metrics.json"
    fused_features_path = features_dir / f"fused_features_ewm_v1_{checkpoint_tag}.parquet"

    training_table = pd.read_csv(features_dir / "gas_training_table_v2.csv", low_memory=False)
    training_meta = load_json(features_dir / "gas_training_table_v2_metadata.json")
    if args.label_column not in training_table.columns:
        raise SystemExit(f"Unknown label column: {args.label_column}")

    training_table["first_prod_date"] = parse_date(training_table["first_prod_date"])
    wells = training_table.copy()
    wells["sample_id"] = wells["well_api"].map(lambda value: f"well::{value}")
    wells["sample_type"] = "well"
    wells["sample_key"] = wells["well_api"]
    wells["sample_longitude"] = wells["longitude_decimal"]
    wells["sample_latitude"] = wells["latitude_decimal"]

    wells_available = wells[
        wells["label_f12_available"].eq(True)
        & wells[args.label_column].notna()
        & wells["first_prod_date"].notna()
        & wells["sample_latitude"].notna()
        & wells["sample_longitude"].notna()
    ].copy()
    wells_available[args.label_column] = wells_available[args.label_column].astype(bool)
    wells_selected = stratified_binary_downsample(
        wells_available,
        args.label_column,
        max_rows=args.max_training_wells,
        random_state=args.random_state,
    )
    wells_selected = wells_selected.sort_values("first_prod_date").reset_index(drop=True)

    grid = pd.DataFrame()
    grid_features_path = (
        Path(args.grid_features_path)
        if args.grid_features_path
        else (derived_dir / "gas_prospect_cells_v1.csv")
    )
    if not args.skip_scoring and grid_features_path.exists():
        grid = pd.read_csv(grid_features_path)
        grid["sample_id"] = grid["cell_id"].map(lambda value: f"cell::{value}")
        grid["sample_type"] = "grid_cell"
        grid["sample_key"] = grid["cell_id"]
        grid["sample_longitude"] = grid["center_longitude"]
        grid["sample_latitude"] = grid["center_latitude"]
        grid["longitude_decimal"] = grid["center_longitude"]
        grid["latitude_decimal"] = grid["center_latitude"]
        if args.max_grid_cells and args.max_grid_cells > 0 and len(grid) > args.max_grid_cells:
            grid = grid.head(args.max_grid_cells).copy()

    combined_samples = wells_selected[["sample_id", "sample_type", "sample_key"]].copy()
    if not grid.empty:
        combined_samples = pd.concat(
            [combined_samples, grid[["sample_id", "sample_type", "sample_key"]]],
            ignore_index=True,
        ).drop_duplicates("sample_id")

    index_path = resolve_index_path(repo_root, features_dir, args.index_path, input_mode)
    if not index_path.exists():
        raise SystemExit(
            f"Chip/sample index not found: {index_path}. "
            f"For input_mode={input_mode!r}, provide --index-path or create the expected index first."
        )

    chip_index = pd.read_parquet(index_path)
    required_cols = {"sample_id", "sample_type", "sample_key"}
    missing_cols = sorted(required_cols - set(chip_index.columns))
    if missing_cols:
        raise SystemExit(f"Chip/sample index is missing required columns: {missing_cols}")
    chip_index = chip_index[chip_index["sample_id"].isin(combined_samples["sample_id"])].copy().reset_index(drop=True)
    if chip_index.empty:
        raise SystemExit("No overlapping chip/sample rows found between the chip index and selected wells/grid cells.")

    embeddings, embedding_meta = extract_ewm_embeddings(
        chip_index=chip_index,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        batch_size=args.embedding_batch_size,
        device=args.device,
        repo_root=repo_root,
        embeddings_path=embeddings_path,
        s2s1_format=args.s2s1_format,
    )
    embeddings.to_parquet(embeddings_path, index=False)
    write_json(features_dir / f"ewm_embeddings_v1_{checkpoint_tag}_metadata.json", embedding_meta)

    training_frame = wells_selected.merge(
        embeddings,
        on=["sample_id", "sample_type", "sample_key"],
        how="inner",
    ).copy()
    if training_frame.empty:
        raise SystemExit("No well rows retained after merging embeddings with the training table.")

    feature_columns_numeric = training_meta["feature_columns_numeric"]
    feature_columns_categorical = training_meta["feature_columns_categorical"]
    embedding_columns = [column for column in training_frame.columns if column.startswith("ewm_embedding_")]
    feature_columns = feature_columns_numeric + feature_columns_categorical + embedding_columns

    chosen_quantile, split_date, train, test = choose_temporal_split(
        training_frame,
        date_col="first_prod_date",
        label_col=args.label_column,
        preferred_quantile=args.holdout_quantile,
    )
    if train.empty or test.empty:
        raise SystemExit("Temporal split produced an empty train or test set.")
    if train[args.label_column].nunique() < 2 or test[args.label_column].nunique() < 2:
        raise SystemExit("Temporal split produced a single-class train or test set.")

    positives = int(train[args.label_column].sum())
    negatives = int((~train[args.label_column]).sum())
    class_weight_scale = float(negatives / max(1, positives))
    model_pipeline = build_model_pipeline(
        numeric_columns=feature_columns_numeric + embedding_columns,
        categorical_columns=feature_columns_categorical,
        class_weight_scale=class_weight_scale,
        random_state=args.random_state,
        device=args.device,
    )
    model_pipeline.fit(train[feature_columns], train[args.label_column])
    test_scores = model_pipeline.predict_proba(test[feature_columns])[:, 1]

    final_class_weight_scale = float(
        (~training_frame[args.label_column]).sum() / max(1, training_frame[args.label_column].sum())
    )
    final_pipeline = build_model_pipeline(
        numeric_columns=feature_columns_numeric + embedding_columns,
        categorical_columns=feature_columns_categorical,
        class_weight_scale=final_class_weight_scale,
        random_state=args.random_state,
        device=args.device,
    )
    final_pipeline.fit(training_frame[feature_columns], training_frame[args.label_column])
    joblib.dump(final_pipeline, model_path)
    training_frame.to_parquet(fused_features_path, index=False)

    roc_auc_test = (
        float(roc_auc_score(test[args.label_column], test_scores))
        if test[args.label_column].nunique() > 1
        else None
    )
    average_precision_test = (
        float(average_precision_score(test[args.label_column], test_scores))
        if test[args.label_column].nunique() > 1
        else None
    )

    metrics: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "gas_ewm_xgboost_v1",
        "basin_id": basin["basin_id"],
        "basin_name": basin["display_name"],
        "checkpoint": str(checkpoint_path),
        "input_mode": input_mode,
        "index_path": str(index_path),
        "label_column": args.label_column,
        "row_count_training_available": int(len(wells_available)),
        "row_count_training_selected": int(len(wells_selected)),
        "row_count_with_embeddings": int(len(training_frame)),
        "row_count_train": int(len(train)),
        "row_count_test": int(len(test)),
        "positive_rate_train": float(train[args.label_column].mean()),
        "positive_rate_test": float(test[args.label_column].mean()),
        "roc_auc_test": roc_auc_test,
        "average_precision_test": average_precision_test,
        "temporal_holdout_anchor": "first_prod_date",
        "temporal_holdout_quantile": chosen_quantile,
        "temporal_holdout_cutoff_date": split_date.strftime("%Y-%m-%d"),
        "train_first_prod_start": train["first_prod_date"].min().strftime("%Y-%m-%d"),
        "train_first_prod_end": train["first_prod_date"].max().strftime("%Y-%m-%d"),
        "test_first_prod_start": test["first_prod_date"].min().strftime("%Y-%m-%d"),
        "test_first_prod_end": test["first_prod_date"].max().strftime("%Y-%m-%d"),
        "feature_column_count_numeric": int(len(feature_columns_numeric)),
        "feature_column_count_categorical": int(len(feature_columns_categorical)),
        "feature_column_count_ewm": int(len(embedding_columns)),
        "embeddings_path": str(embeddings_path),
        "fused_features_path": str(fused_features_path),
        "model_path": str(model_path),
    }

    if not grid.empty:
        grid_embeddings = embeddings[embeddings["sample_type"] == "grid_cell"].copy()
        if not grid_embeddings.empty:
            score_csv_path, score_geojson_path, top_path = score_grid_cells(
                basin=basin,
                grid_features=grid,
                grid_embeddings=grid_embeddings,
                model_pipeline=final_pipeline,
                feature_columns=feature_columns,
                label_column=args.label_column,
                model_path=model_path,
                metrics_path=metrics_path,
                derived_dir=derived_dir,
            )
            metrics["grid_score_csv_path"] = str(score_csv_path)
            metrics["grid_score_geojson_path"] = str(score_geojson_path)
            metrics["top_prospects_path"] = str(top_path)
            metrics["row_count_grid_with_embeddings"] = int(len(grid_embeddings))
        else:
            metrics["grid_scoring_skipped_reason"] = "No grid embeddings were found in the chip/sample index."
    else:
        metrics["grid_scoring_skipped_reason"] = "No grid feature table was provided or found."

    write_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
