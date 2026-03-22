#!/usr/bin/env python3
"""Train the Earth World Model PoC on fake or manifest-backed temporal data."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ewm.data.dataset import (
    DenseTemporalNPZDataset,
    DenseTemporalZarrDataset,
    FakeTemporalDataset,
    HLSChipTemporalDataset,
    ManifestTemporalDataset,
    SSL4EOZarrDatafluxDataset,
    SSL4EOZarrDataset,
)
from ewm.models.world_model import EarthWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda", "tpu"], default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    return _expand_config_values(yaml.safe_load(path.read_text(encoding="utf-8")))


def _expand_config_values(value):
    if isinstance(value, dict):
        return {key: _expand_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\:-([^}]*)\}",
            lambda match: os.environ.get(match.group(1), match.group(2)),
            value,
        )
        return os.path.expanduser(os.path.expandvars(value))
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_bool_config(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_dataset_from_cfg(data_cfg: dict) -> torch.utils.data.Dataset:
    kind = data_cfg["kind"]
    force_single_process_io = resolve_bool_config(
        data_cfg.get("force_single_process_io"),
        default=resolve_bool_config(os.environ.get("EWM_FORCE_SINGLE_PROCESS_IO"), default=False),
    )
    if kind == "fake":
        return FakeTemporalDataset(
            num_samples=data_cfg.get("num_samples", 1024),
            timesteps=data_cfg.get("max_timestamps", 4),
            patch_size=data_cfg.get("patch_size", 128),
        )
    if kind == "manifest":
        return ManifestTemporalDataset(
            manifest_path=data_cfg["manifest_path"],
            patch_size=data_cfg.get("patch_size", 128),
            max_samples=data_cfg.get("max_samples"),
            sample_offset=data_cfg.get("sample_offset", 0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=data_cfg.get("repeat_factor", 1),
            force_single_process_io=force_single_process_io,
        )
    if kind == "hls_chip_index":
        return HLSChipTemporalDataset(
            index_path=data_cfg["index_path"],
            patch_size=data_cfg.get("patch_size", 224),
            max_samples=data_cfg.get("max_samples"),
            min_valid_fraction=data_cfg.get("min_valid_fraction", 0.0),
            sample_offset=data_cfg.get("sample_offset", 0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=data_cfg.get("repeat_factor", 1),
            force_single_process_io=force_single_process_io,
        )
    if kind == "ssl4eo_zarr":
        root_dir = data_cfg["root_dir"]
        use_dataflux = str(data_cfg.get("use_dataflux", os.environ.get("EWM_USE_DATAFLUX", "auto"))).strip().lower()
        if str(root_dir).startswith("gs://") and use_dataflux != "false":
            try:
                return SSL4EOZarrDatafluxDataset(
                    root_dir=root_dir,
                    split=data_cfg.get("split", "val"),
                    patch_size=data_cfg.get("patch_size", 128),
                    max_shards=data_cfg.get("max_shards"),
                    max_samples=data_cfg.get("max_samples"),
                    min_clear_fraction=data_cfg.get("min_clear_fraction", 0.8),
                    sample_offset=data_cfg.get("sample_offset", 0),
                    crop_mode=data_cfg.get("crop_mode", "center"),
                    repeat_factor=data_cfg.get("repeat_factor", 1),
                    assume_single_sample_per_shard=data_cfg.get("assume_single_sample_per_shard", False),
                    max_open_shards=data_cfg.get("max_open_shards", 16),
                    force_single_process_io=force_single_process_io,
                    project_name=data_cfg.get("project_name") or os.environ.get("EWM_GCP_PROJECT"),
                    dataflux_threads_per_process=data_cfg.get("dataflux_threads_per_process", 1),
                    dataflux_num_processes=data_cfg.get("dataflux_num_processes"),
                    dataflux_disable_compose=resolve_bool_config(data_cfg.get("dataflux_disable_compose"), default=False),
                )
            except ImportError:
                if use_dataflux == "true":
                    raise
        return SSL4EOZarrDataset(
            root_dir=root_dir,
            split=data_cfg.get("split", "val"),
            patch_size=data_cfg.get("patch_size", 128),
            max_shards=data_cfg.get("max_shards"),
            max_samples=data_cfg.get("max_samples"),
            min_clear_fraction=data_cfg.get("min_clear_fraction", 0.8),
            sample_offset=data_cfg.get("sample_offset", 0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=data_cfg.get("repeat_factor", 1),
            assume_single_sample_per_shard=data_cfg.get("assume_single_sample_per_shard", False),
            max_open_shards=data_cfg.get("max_open_shards", 16),
            force_single_process_io=force_single_process_io,
        )
    if kind == "dense_temporal_index":
        return DenseTemporalNPZDataset(
            index_path=data_cfg["index_path"],
            patch_size=data_cfg.get("patch_size", 128),
            max_samples=data_cfg.get("max_samples"),
            min_paired_frames=data_cfg.get("min_paired_frames", 1),
            min_paired_fraction=data_cfg.get("min_paired_fraction", 0.0),
            sample_offset=data_cfg.get("sample_offset", 0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=data_cfg.get("repeat_factor", 1),
            force_single_process_io=force_single_process_io,
        )
    if kind == "dense_temporal_zarr_index":
        return DenseTemporalZarrDataset(
            index_path=data_cfg["index_path"],
            patch_size=data_cfg.get("patch_size", 128),
            max_samples=data_cfg.get("max_samples"),
            min_paired_frames=data_cfg.get("min_paired_frames", 1),
            min_paired_fraction=data_cfg.get("min_paired_fraction", 0.0),
            sample_offset=data_cfg.get("sample_offset", 0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=data_cfg.get("repeat_factor", 1),
            max_open_shards=data_cfg.get("max_open_shards", 16),
            force_single_process_io=force_single_process_io,
        )
    raise ValueError(f"Unsupported data.kind: {kind}")


def build_dataset(config: dict, section: str = "data") -> torch.utils.data.Dataset:
    return build_dataset_from_cfg(config[section])


def make_loader(
    dataset: torch.utils.data.Dataset,
    config: dict,
    section: str = "data",
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    data_cfg = config[section]
    train_cfg = config["training"]
    requested_num_workers = int(data_cfg.get("num_workers", 0))
    num_workers = requested_num_workers
    if getattr(dataset, "requires_single_process_io", False) and num_workers > 0:
        num_workers = 0
    print(
        json.dumps(
            {
                "event": "loader_config",
                "section": section,
                "dataset_class": dataset.__class__.__name__,
                "requested_num_workers": requested_num_workers,
                "effective_num_workers": num_workers,
                "requires_single_process_io": bool(getattr(dataset, "requires_single_process_io", False)),
            }
        )
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=bool(num_workers),
    )


def make_model(config: dict, device: torch.device) -> EarthWorldModel:
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
    return model.to(device)


def resolve_backend(config: dict) -> tuple[str, torch.device]:
    backend = config["runtime"]["backend"]
    if backend == "cuda":
        print(
            json.dumps(
                {
                    "backend_probe": "cuda",
                    "python": sys.executable,
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                }
            )
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but no CUDA device is available")
        return backend, torch.device("cuda")
    if backend == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("TPU backend requested, but torch_xla is not installed") from exc
        import torch_xla.core.xla_model as xm

        return backend, xm.xla_device()
    return "cpu", torch.device("cpu")


def save_checkpoint(path: Path, backend: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "tpu":
        import torch_xla.core.xla_model as xm

        xm.save(payload, str(path))
    else:
        torch.save(payload, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path, backend: str):
    if backend == "tpu":
        try:
            import torch_xla.utils.serialization as xser

            return xser.load(str(path))
        except Exception:
            # Allow TPU runs to resume from regular torch.save checkpoints created on CPU/CUDA.
            return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu")


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device, backend: str) -> None:
    if backend == "cpu":
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_module_state(module: torch.nn.Module, state_dict: dict, label: str) -> None:
    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(
            json.dumps(
                {
                    "checkpoint_module": label,
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                },
                indent=2,
            )
        )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device, backend: str) -> dict[str, torch.Tensor]:
    if backend == "tpu":
        return batch
    return {key: value.to(device) for key, value in batch.items()}


def maybe_wrap_loader(loader: DataLoader, device: torch.device, backend: str):
    if backend != "tpu":
        return loader
    import torch_xla.distributed.parallel_loader as pl

    return pl.MpDeviceLoader(loader, device)


def autocast_dtype_from_config(config: dict) -> torch.dtype | None:
    precision = str(config["runtime"].get("autocast_dtype", "none")).lower()
    if precision in {"none", "off", "false"}:
        return None
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported runtime.autocast_dtype: {precision}")


def maybe_autocast(backend: str, device: torch.device, config: dict):
    autocast_dtype = autocast_dtype_from_config(config)
    if autocast_dtype is None:
        return nullcontext()
    if backend == "tpu":
        import torch_xla.amp as xla_amp

        return xla_amp.autocast(device)
    if backend == "cuda":
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def make_grad_scaler(backend: str, config: dict):
    autocast_dtype = autocast_dtype_from_config(config)
    if backend != "cuda" or autocast_dtype != torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


def scheduled_value(
    progress: float,
    *,
    schedule_type: str,
    start_value: float,
    peak_value: float,
    final_value: float,
    warmup_fraction: float,
    hold_fraction: float,
) -> float:
    progress = min(max(progress, 0.0), 1.0)
    if schedule_type == "fixed":
        return peak_value

    warmup_fraction = max(0.0, min(warmup_fraction, 1.0))
    hold_fraction = max(0.0, min(hold_fraction, 1.0 - warmup_fraction))

    if warmup_fraction > 0.0 and progress < warmup_fraction:
        warmup_progress = progress / max(warmup_fraction, 1.0e-8)
        return start_value + (peak_value - start_value) * warmup_progress

    hold_end = warmup_fraction + hold_fraction
    if progress < hold_end:
        return peak_value

    cosine_progress = (progress - hold_end) / max(1.0 - hold_end, 1.0e-8)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return final_value + (peak_value - final_value) * cosine_weight


def apply_training_schedules(
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    total_steps: int,
    training_cfg: dict[str, Any],
) -> tuple[float, float]:
    progress = global_step / max(total_steps - 1, 1)
    schedule_type = str(training_cfg.get("lr_schedule", "fixed")).lower()
    warmup_fraction = float(training_cfg.get("warmup_fraction", 0.0))
    hold_fraction = float(training_cfg.get("hold_fraction", 0.0))
    lr_start = float(training_cfg.get("start_lr", training_cfg["lr"]))
    lr_peak = float(training_cfg["lr"])
    lr_final = float(training_cfg.get("final_lr", lr_peak))
    wd_start = float(training_cfg.get("weight_decay", 0.0))
    wd_final = float(training_cfg.get("final_weight_decay", wd_start))

    lr_value = scheduled_value(
        progress,
        schedule_type=schedule_type,
        start_value=lr_start,
        peak_value=lr_peak,
        final_value=lr_final,
        warmup_fraction=warmup_fraction,
        hold_fraction=hold_fraction,
    )
    wd_value = scheduled_value(
        progress,
        schedule_type=schedule_type,
        start_value=wd_start,
        peak_value=wd_start,
        final_value=wd_final,
        warmup_fraction=warmup_fraction,
        hold_fraction=hold_fraction,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value
        param_group["weight_decay"] = wd_value
    return lr_value, wd_value


def temporal_mask_indices(timesteps: int, mask_ratio: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    num_masked = max(1, int(timesteps * mask_ratio))
    permutation_device = torch.device("cpu") if device.type == "xla" else device
    permutation = torch.randperm(timesteps, device=permutation_device)
    masked = permutation[:num_masked].sort().values
    visible = permutation[num_masked:].sort().values
    if permutation_device != device:
        masked = masked.to(device=device)
        visible = visible.to(device=device)
    return visible, masked


def all_temporal_mask_indices(timesteps: int, mask_ratio: float, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_masked = max(1, int(timesteps * mask_ratio))
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    all_steps = tuple(range(timesteps))
    for masked_steps in itertools.combinations(all_steps, num_masked):
        visible_steps = [step for step in all_steps if step not in masked_steps]
        visible = torch.tensor(visible_steps, device=device, dtype=torch.long)
        masked = torch.tensor(masked_steps, device=device, dtype=torch.long)
        pairs.append((visible, masked))
    return pairs


def student_visible_embeddings(
    student: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    visible_idx: torch.Tensor,
) -> torch.Tensor:
    s2 = batch.get("s2")
    s1 = batch.get("s1")
    hls = batch.get("hls")
    dates = batch["dates"]
    valid = batch["mask"]
    return student.encode_timeline(
        None if s2 is None else s2[:, visible_idx],
        None if s1 is None else s1[:, visible_idx],
        dates[:, visible_idx],
        valid[:, visible_idx],
        hls=None if hls is None else hls[:, visible_idx],
    )


def compute_teacher_embeddings(
    teacher: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor:
    return_hierarchical = bool(config["model"].get("hierarchical_layers"))
    return teacher.encode_timeline(
        batch.get("s2"),
        batch.get("s1"),
        batch["dates"],
        batch["mask"],
        hls=batch.get("hls"),
        return_hierarchical=return_hierarchical,
    )


def compute_masked_prediction_loss(
    student: EarthWorldModel,
    teacher_embeddings: torch.Tensor,
    batch: dict[str, torch.Tensor],
    visible_idx: torch.Tensor,
    masked_idx: torch.Tensor,
    backend: str,
    device: torch.device,
    config: dict,
) -> torch.Tensor:
    loss_terms = compute_prediction_losses(
        student,
        teacher_embeddings,
        batch,
        visible_idx,
        masked_idx,
        backend,
        device,
        config,
    )
    return loss_terms["total_loss"]


def temporal_context_weights(
    visible_idx: torch.Tensor,
    masked_idx: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sqrt_distance: bool = False,
) -> torch.Tensor:
    if visible_idx.numel() == 0 or masked_idx.numel() == 0:
        return torch.ones(max(1, visible_idx.numel()), device=device, dtype=dtype)
    visible = visible_idx.to(device=device, dtype=dtype)
    masked = masked_idx.to(device=device, dtype=dtype)
    distances = torch.cdist(visible.unsqueeze(-1), masked.unsqueeze(-1), p=1).amin(dim=-1)
    distances = distances.clamp_min(1.0)
    if sqrt_distance:
        distances = distances.sqrt()
    return 1.0 / distances


def expand_context_weights_for_tokens(weights: torch.Tensor | None, tokens_per_timestep: int) -> torch.Tensor | None:
    if weights is None or tokens_per_timestep <= 1:
        return weights
    return weights.repeat_interleave(tokens_per_timestep)


def select_teacher_targets(
    teacher: EarthWorldModel,
    teacher_embeddings: torch.Tensor,
    step_idx: torch.Tensor,
) -> torch.Tensor:
    if teacher.token_mode != "dense":
        return teacher_embeddings[:, step_idx, :]
    tokens_per_timestep = teacher.last_tokens_per_timestep
    token_indices = [
        torch.arange(
            int(step.item()) * tokens_per_timestep,
            (int(step.item()) + 1) * tokens_per_timestep,
            device=teacher_embeddings.device,
            dtype=torch.long,
        )
        for step in step_idx
    ]
    gathered = torch.cat(token_indices, dim=0)
    return teacher_embeddings[:, gathered, :]


def weighted_embedding_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    errors = F.mse_loss(
        F.normalize(predicted, dim=-1),
        F.normalize(target, dim=-1),
        reduction="none",
    )
    if weights is None:
        return errors.mean()
    weight_tensor = weights.view(1, -1, 1).to(device=errors.device, dtype=errors.dtype)
    weighted_errors = errors * weight_tensor
    normalizer = errors.shape[0] * errors.shape[-1] * weight_tensor.sum().clamp_min(1.0e-6)
    return weighted_errors.sum() / normalizer


def compute_prediction_losses(
    student: EarthWorldModel,
    teacher_embeddings: torch.Tensor,
    batch: dict[str, torch.Tensor],
    visible_idx: torch.Tensor,
    masked_idx: torch.Tensor,
    backend: str,
    device: torch.device,
    config: dict,
) -> dict[str, torch.Tensor]:
    dates = batch["dates"]
    training_cfg = config["training"]
    predict_visible_context = bool(training_cfg.get("predict_visible_context", False))
    context_loss_weight = float(training_cfg.get("context_loss_weight", 0.0))
    context_loss_distance_weighted = bool(training_cfg.get("context_loss_distance_weighted", False))
    context_loss_sqrt_distance = bool(training_cfg.get("context_loss_sqrt_distance", True))

    with maybe_autocast(backend, device, config):
        visible_embeddings = student_visible_embeddings(student, batch, visible_idx)
        predicted_visible, predicted_masked = student.predict_with_context(
            visible_embeddings,
            dates[:, visible_idx],
            dates[:, masked_idx],
        )
        masked_target = select_teacher_targets(student, teacher_embeddings, masked_idx).detach()
        masked_loss = weighted_embedding_mse(predicted_masked, masked_target)

        context_loss = torch.zeros_like(masked_loss)
        if predict_visible_context and visible_idx.numel() > 0 and context_loss_weight > 0.0:
            context_target = select_teacher_targets(student, teacher_embeddings, visible_idx).detach()
            context_weights = None
            if context_loss_distance_weighted:
                context_weights = temporal_context_weights(
                    visible_idx,
                    masked_idx,
                    device=predicted_visible.device,
                    dtype=predicted_visible.dtype,
                    sqrt_distance=context_loss_sqrt_distance,
                )
                context_weights = expand_context_weights_for_tokens(context_weights, student.last_tokens_per_timestep)
            context_loss = weighted_embedding_mse(predicted_visible, context_target, weights=context_weights)

        total_loss = masked_loss + (context_loss_weight * context_loss)

    return {
        "total_loss": total_loss,
        "masked_loss": masked_loss.detach(),
        "context_loss": context_loss.detach(),
        "context_loss_weight": torch.tensor(context_loss_weight, device=total_loss.device, dtype=total_loss.dtype),
        "predict_visible_context": torch.tensor(1 if predict_visible_context else 0, device=total_loss.device),
    }


def resolve_eval_config(config: dict) -> dict | None:
    raw_eval_cfg = config.get("eval")
    if not raw_eval_cfg or not raw_eval_cfg.get("enabled", False):
        return None

    eval_cfg = deepcopy(raw_eval_cfg)
    eval_data_cfg = deepcopy(config["data"])
    eval_data_cfg.update(eval_cfg.get("data", {}))
    eval_cfg["data"] = eval_data_cfg
    eval_cfg["batch_size"] = int(eval_cfg.get("batch_size", config["training"]["batch_size"]))
    eval_cfg["max_batches"] = int(eval_cfg.get("max_batches", 0))
    eval_cfg["every_epochs"] = int(eval_cfg.get("every_epochs", 1))
    eval_cfg["mask_mode"] = str(eval_cfg.get("mask_mode", "all")).lower()
    return eval_cfg


def evaluate(
    student: EarthWorldModel,
    teacher: EarthWorldModel,
    loader: DataLoader,
    backend: str,
    device: torch.device,
    config: dict,
    eval_cfg: dict,
) -> dict:
    mask_ratio = float(config["training"]["mask_ratio"])
    mask_mode = eval_cfg["mask_mode"]
    max_batches = int(eval_cfg.get("max_batches", 0))
    student.eval()
    teacher.eval()

    loss_total = 0.0
    masked_loss_total = 0.0
    context_loss_total = 0.0
    step_count = 0
    started_at = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device, backend)
            teacher_embeddings = compute_teacher_embeddings(teacher, batch, config)

            if mask_mode == "all":
                mask_pairs = all_temporal_mask_indices(batch["dates"].shape[1], mask_ratio, batch["dates"].device)
            elif mask_mode == "random":
                mask_pairs = [temporal_mask_indices(batch["dates"].shape[1], mask_ratio, batch["dates"].device)]
            else:
                raise ValueError(f"Unsupported eval.mask_mode: {mask_mode}")

            batch_losses = [
                compute_prediction_losses(student, teacher_embeddings, batch, visible_idx, masked_idx, backend, device, config)
                for visible_idx, masked_idx in mask_pairs
            ]
            batch_loss = torch.stack([loss_terms["total_loss"] for loss_terms in batch_losses]).mean()
            batch_masked_loss = torch.stack([loss_terms["masked_loss"] for loss_terms in batch_losses]).mean()
            batch_context_loss = torch.stack([loss_terms["context_loss"] for loss_terms in batch_losses]).mean()
            loss_total += float(batch_loss.cpu().item())
            masked_loss_total += float(batch_masked_loss.cpu().item())
            context_loss_total += float(batch_context_loss.cpu().item())
            step_count += 1

            if max_batches > 0 and step >= max_batches:
                break

    return {
        "mean_loss": loss_total / max(step_count, 1),
        "mean_masked_loss": masked_loss_total / max(step_count, 1),
        "mean_context_loss": context_loss_total / max(step_count, 1),
        "step_count": step_count,
        "duration_sec": time.time() - started_at,
        "mask_mode": mask_mode,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.backend is not None:
        config["runtime"]["backend"] = args.backend
    if args.manifest_path is not None:
        config["data"]["manifest_path"] = str(args.manifest_path)
        config["data"]["kind"] = "manifest"
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples

    set_seed(int(config["runtime"].get("seed", 42)))
    backend, device = resolve_backend(config)
    dataset = build_dataset(config, section="data")
    loader = make_loader(dataset, config, section="data", shuffle=True, drop_last=True)
    loader = maybe_wrap_loader(loader, device, backend)
    eval_cfg = resolve_eval_config(config)
    eval_dataset = None
    eval_loader = None
    if eval_cfg is not None:
        config["eval"]["data"] = eval_cfg["data"]
        eval_dataset = build_dataset_from_cfg(eval_cfg["data"])
        eval_config = deepcopy(config)
        eval_config["data"] = eval_cfg["data"]
        eval_loader = make_loader(
            eval_dataset,
            eval_config,
            section="data",
            batch_size=eval_cfg["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        eval_loader = maybe_wrap_loader(eval_loader, device, backend)

    student = make_model(config, device)
    teacher = deepcopy(student).to(device)
    teacher.requires_grad_(False)
    teacher.eval()

    # Performance: cuDNN autotuning + torch.compile (V-JEPA 2.1 pattern)
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True
        compile_model = config["training"].get("compile_model", False)
        if compile_model:
            print(json.dumps({"action": "compiling_models", "note": "student + teacher + predictor via torch.compile"}))
            torch._dynamo.config.optimize_ddp = False
            student = torch.compile(student)
            teacher = torch.compile(teacher)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    grad_scaler = make_grad_scaler(backend, config)

    checkpoint_dir = Path(config["runtime"]["checkpoint_dir"])
    epochs = int(config["training"]["epochs"])
    log_every = int(config["training"].get("log_every_steps", 10))
    save_every = int(config["training"].get("save_every_epochs", 1))
    max_steps_per_epoch = int(config["training"].get("max_steps_per_epoch", 0))
    mask_ratio = float(config["training"]["mask_ratio"])
    ema_start = float(config["training"].get("ema_momentum_start", 0.996))
    ema_end = float(config["training"].get("ema_momentum_end", 0.999))
    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * max(steps_per_epoch, 1))
    start_epoch = 0
    global_step = 0
    epoch_summaries: list[dict] = []
    eval_summaries: list[dict] = []
    best_validation_mean_loss = None
    resumed_from_checkpoint = None
    train_started_at = time.time()
    metrics_path = checkpoint_dir / "metrics.jsonl"

    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        checkpoint = load_checkpoint(resume_path, backend)
        load_module_state(student, checkpoint["student"], "student")
        load_module_state(teacher, checkpoint.get("teacher", checkpoint["student"]), "teacher")
        optimizer_resumed = True
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            move_optimizer_state(optimizer, device, backend)
        except ValueError as exc:
            optimizer_resumed = False
            print(
                json.dumps(
                    {
                        "resume_from": str(resume_path),
                        "optimizer_state_restored": False,
                        "warning": str(exc),
                    },
                    indent=2,
                )
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", start_epoch * max(steps_per_epoch, 1)))
        best_validation_mean_loss = checkpoint.get("best_validation_mean_loss")
        resumed_from_checkpoint = str(resume_path)
        print(
            json.dumps(
                {
                    "resume_from": resumed_from_checkpoint,
                    "start_epoch": start_epoch + 1,
                    "global_step": global_step,
                    "best_validation_mean_loss": best_validation_mean_loss,
                    "optimizer_state_restored": optimizer_resumed,
                },
                indent=2,
            )
        )

    print(json.dumps(
        {
            "backend": backend,
            "device": str(device),
            "rows": len(dataset),
            "base_rows": getattr(dataset, "base_row_count", len(dataset)),
            "steps_per_epoch": steps_per_epoch,
            "checkpoint_dir": str(checkpoint_dir),
            "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_base_rows": getattr(eval_dataset, "base_row_count", len(eval_dataset)) if eval_dataset is not None else 0,
            "start_epoch": start_epoch + 1,
        },
        indent=2,
    ))
    append_jsonl(
        metrics_path,
        {
            "event": "run_start",
            "timestamp": train_started_at,
            "backend": backend,
            "device": str(device),
            "row_count": len(dataset),
            "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
            "eval_row_count": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_base_row_count": getattr(eval_dataset, "base_row_count", len(eval_dataset))
            if eval_dataset is not None
            else 0,
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "start_epoch": start_epoch + 1,
            "resume_from_checkpoint": resumed_from_checkpoint,
            "checkpoint_dir": str(checkpoint_dir),
        },
    )

    for epoch in range(start_epoch, epochs):
        student.train()
        running_loss = None
        running_masked_loss = None
        running_context_loss = None
        epoch_loss_total = 0.0
        epoch_masked_loss_total = 0.0
        epoch_context_loss_total = 0.0
        epoch_step_count = 0
        epoch_started_at = time.time()

        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device, backend)
            dates = batch["dates"]
            visible_idx, masked_idx = temporal_mask_indices(dates.shape[1], mask_ratio, dates.device)
            current_lr, current_wd = apply_training_schedules(
                optimizer,
                global_step=global_step,
                total_steps=total_steps,
                training_cfg=config["training"],
            )

            with torch.no_grad():
                teacher_embeddings = compute_teacher_embeddings(teacher, batch, config)

            loss_terms = compute_prediction_losses(student, teacher_embeddings, batch, visible_idx, masked_idx, backend, device, config)
            loss = loss_terms["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

                if backend == "tpu":
                    import torch_xla.core.xla_model as xm

                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

            progress = global_step / total_steps
            momentum = ema_end - (ema_end - ema_start) * ((1 + math.cos(math.pi * progress)) / 2)
            with torch.no_grad():
                for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                    teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

            detached_loss = loss.detach()
            detached_masked_loss = loss_terms["masked_loss"]
            detached_context_loss = loss_terms["context_loss"]
            running_loss = detached_loss if running_loss is None else running_loss + detached_loss
            running_masked_loss = (
                detached_masked_loss
                if running_masked_loss is None
                else running_masked_loss + detached_masked_loss
            )
            running_context_loss = (
                detached_context_loss
                if running_context_loss is None
                else running_context_loss + detached_context_loss
            )
            epoch_loss_total += float(detached_loss.cpu().item())
            epoch_masked_loss_total += float(detached_masked_loss.cpu().item())
            epoch_context_loss_total += float(detached_context_loss.cpu().item())
            epoch_step_count += 1
            global_step += 1

            if step % log_every == 0:
                mean_loss = running_loss / log_every
                mean_masked_loss = running_masked_loss / log_every
                mean_context_loss = running_context_loss / log_every
                train_step_metrics = {
                    "event": "train_step",
                    "timestamp": time.time(),
                    "elapsed_sec": time.time() - train_started_at,
                    "epoch": epoch + 1,
                    "step_in_epoch": step,
                    "global_step": global_step,
                    "steps_per_epoch": steps_per_epoch,
                    "loss": float(mean_loss.cpu().item()),
                    "masked_loss": float(mean_masked_loss.cpu().item()),
                    "context_loss": float(mean_context_loss.cpu().item()),
                    "lr": float(current_lr),
                    "weight_decay": float(current_wd),
                    "momentum": float(momentum),
                }
                if backend == "tpu":
                    import torch_xla.core.xla_model as xm

                    xm.master_print(
                        f"epoch={epoch + 1} step={step}/{steps_per_epoch} "
                        f"loss={mean_loss.cpu().item():.6f} "
                        f"masked_loss={mean_masked_loss.cpu().item():.6f} "
                        f"context_loss={mean_context_loss.cpu().item():.6f} "
                        f"lr={current_lr:.6e} wd={current_wd:.6f} "
                        f"momentum={momentum:.6f}"
                    )
                else:
                    print(
                        f"epoch={epoch + 1} step={step}/{steps_per_epoch} "
                        f"loss={mean_loss.cpu().item():.6f} "
                        f"masked_loss={mean_masked_loss.cpu().item():.6f} "
                        f"context_loss={mean_context_loss.cpu().item():.6f} "
                        f"lr={current_lr:.6e} wd={current_wd:.6f} "
                        f"momentum={momentum:.6f}"
                    )
                append_jsonl(metrics_path, train_step_metrics)
                running_loss = None
                running_masked_loss = None
                running_context_loss = None

            if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                break

        epoch_duration_sec = time.time() - epoch_started_at
        epoch_mean_loss = epoch_loss_total / max(epoch_step_count, 1)
        epoch_summary = {
            "event": "epoch_summary",
            "timestamp": time.time(),
            "elapsed_sec": time.time() - train_started_at,
            "epoch": epoch + 1,
            "step_count": epoch_step_count,
            "mean_loss": epoch_mean_loss,
            "mean_masked_loss": epoch_masked_loss_total / max(epoch_step_count, 1),
            "mean_context_loss": epoch_context_loss_total / max(epoch_step_count, 1),
            "duration_sec": epoch_duration_sec,
        }
        if backend == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"epoch={epoch + 1} peak_gpu_mem_mb={peak_memory_mb:.1f}")
            torch.cuda.reset_peak_memory_stats(device)
            epoch_summary["peak_gpu_mem_mb"] = peak_memory_mb

        if eval_loader is not None and eval_cfg is not None and (epoch + 1) % eval_cfg["every_epochs"] == 0:
            eval_summary = evaluate(student, teacher, eval_loader, backend, device, config, eval_cfg)
            eval_summary["epoch"] = epoch + 1
            eval_summary["event"] = "validation"
            eval_summary["timestamp"] = time.time()
            eval_summary["elapsed_sec"] = time.time() - train_started_at
            eval_summaries.append(eval_summary)
            epoch_summary["validation_mean_loss"] = eval_summary["mean_loss"]
            epoch_summary["validation_mean_masked_loss"] = eval_summary["mean_masked_loss"]
            epoch_summary["validation_mean_context_loss"] = eval_summary["mean_context_loss"]
            epoch_summary["validation_step_count"] = eval_summary["step_count"]
            print(
                f"epoch={epoch + 1} validation_loss={eval_summary['mean_loss']:.6f} "
                f"validation_masked_loss={eval_summary['mean_masked_loss']:.6f} "
                f"validation_context_loss={eval_summary['mean_context_loss']:.6f} "
                f"validation_steps={eval_summary['step_count']}"
            )
            validation_mean_loss = float(eval_summary["mean_loss"])
            if best_validation_mean_loss is None or validation_mean_loss < best_validation_mean_loss:
                best_validation_mean_loss = validation_mean_loss
                best_path = checkpoint_dir / "ewm_best_val.pt"
                payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "best_validation_mean_loss": best_validation_mean_loss,
                }
                save_checkpoint(best_path, backend, payload)
                print(f"saved best-val checkpoint: {best_path}")
            append_jsonl(metrics_path, eval_summary)

        epoch_summaries.append(epoch_summary)
        append_jsonl(metrics_path, epoch_summary)

        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"ewm_epoch_{epoch + 1:03d}.pt"
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            save_checkpoint(checkpoint_path, backend, payload)
            print(f"saved checkpoint: {checkpoint_path}")

    summary_path = checkpoint_dir / "training_summary.json"
    summary = {
        "backend": backend,
        "device": str(device),
        "row_count": len(dataset),
        "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
        "eval_row_count": len(eval_dataset) if eval_dataset is not None else 0,
        "eval_base_row_count": getattr(eval_dataset, "base_row_count", len(eval_dataset)) if eval_dataset is not None else 0,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "global_steps_completed": global_step,
        "training_duration_sec": time.time() - train_started_at,
        "epoch_summaries": epoch_summaries,
        "eval_summaries": eval_summaries,
        "final_epoch_mean_loss": epoch_summaries[-1]["mean_loss"] if epoch_summaries else None,
        "best_validation_mean_loss": best_validation_mean_loss,
        "resume_from_checkpoint": resumed_from_checkpoint,
        "start_epoch": start_epoch + 1,
        "config": config,
    }
    write_json(summary_path, summary)
    append_jsonl(
        metrics_path,
        {
            "event": "run_end",
            "timestamp": time.time(),
            "elapsed_sec": time.time() - train_started_at,
            "summary_path": str(summary_path),
            "best_validation_mean_loss": best_validation_mean_loss,
            "global_steps_completed": global_step,
        },
    )
    print(f"wrote training summary: {summary_path}")


if __name__ == "__main__":
    main()
