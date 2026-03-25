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
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from ewm.data.dataset import (
    DenseTemporalNPZDataset,
    DenseTemporalZarrDataset,
    FakeTemporalDataset,
    HLSChipTemporalDataset,
    ManifestTemporalDataset,
    SSL4EOZarrDatafluxDataset,
    SSL4EOZarrDataset,
)
from ewm.losses import (
    adaptive_lambda,
    cramer_wold_sigreg,
    cross_covariance_loss,
    vicreg_regularization,
)
from ewm.models.world_model import EarthWorldModel


@dataclass
class SSL4EOShardBatchSampler:
    """Batch samples shard-locally to avoid ZIP shard thrash under global shuffle."""

    dataset: SSL4EOZarrDataset
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_replicas: int = 1
    rank: int = 0
    seed: int = 0
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive for SSL4EOShardBatchSampler")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive for SSL4EOShardBatchSampler")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas) for SSL4EOShardBatchSampler")
        self._groups = self._build_groups()

    def _build_groups(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        base_row_count = int(getattr(self.dataset, "base_row_count", len(self.dataset)))
        base_rows = self.dataset.rows[:base_row_count]
        for row_index, row in enumerate(base_rows):
            shard_index = int(row["shard_index"])
            groups.setdefault(shard_index, []).append(row_index)
        return [groups[key] for key in sorted(groups)]

    def __len__(self) -> int:
        per_repeat = 0
        global_batch_size = self.batch_size * self.num_replicas
        for group in self._groups:
            if self.drop_last:
                per_repeat += len(group) // global_batch_size
            else:
                per_repeat += math.ceil(len(group) / global_batch_size)
        return per_repeat * int(getattr(self.dataset, "repeat_factor", 1))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        repeat_factor = int(getattr(self.dataset, "repeat_factor", 1))
        global_batch_size = self.batch_size * self.num_replicas
        for _repeat_idx in range(repeat_factor):
            group_order = list(range(len(self._groups)))
            if self.shuffle:
                rng.shuffle(group_order)
            for group_idx in group_order:
                batch_indices = list(self._groups[group_idx])
                if self.shuffle and len(batch_indices) > 1:
                    rng.shuffle(batch_indices)
                for start in range(0, len(batch_indices), global_batch_size):
                    batch = batch_indices[start : start + global_batch_size]
                    if len(batch) < global_batch_size and self.drop_last:
                        continue
                    if self.num_replicas == 1:
                        yield batch
                        continue
                    local_start = self.rank * self.batch_size
                    local_batch = batch[local_start : local_start + self.batch_size]
                    if local_batch:
                        yield local_batch


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class EmbeddingMetadata:
    token_mode: str
    tokens_per_timestep: int


def should_use_ssl4eo_shard_sampler(dataset: torch.utils.data.Dataset, data_cfg: dict[str, Any], batch_size: int) -> bool:
    mode = str(data_cfg.get("batch_sampler", "auto")).strip().lower()
    if mode in {"off", "none", "disabled", "false", "0"}:
        return False
    if not isinstance(dataset, SSL4EOZarrDataset):
        return False
    if getattr(dataset, "assume_single_sample_per_shard", False):
        return False
    if batch_size <= 1:
        return False
    return True


def ewm_collate(batch):
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


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


_CONFIG_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_CONFIG_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)


def _coerce_expanded_config_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if _CONFIG_INT_PATTERN.fullmatch(stripped):
        return int(stripped)
    if _CONFIG_FLOAT_PATTERN.fullmatch(stripped):
        return float(stripped)
    return value


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
        expanded = os.path.expanduser(os.path.expandvars(value))
        return _coerce_expanded_config_scalar(expanded)
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


def resolve_int_config(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return int(stripped)
    return int(value)


def resolve_target_mode(config: dict) -> str:
    mode = str(config.get("training", {}).get("target_mode", "ema")).strip().lower()
    if mode not in {"ema", "self"}:
        raise ValueError(f"Unsupported training.target_mode: {mode}")
    return mode


def resolve_regularization_method(config: dict) -> str:
    method = str(config.get("regularization", {}).get("method", "none")).strip().lower()
    if method not in {"none", "sigreg", "vicreg"}:
        raise ValueError(f"Unsupported regularization.method: {method}")
    return method


def current_embedding_metadata(model: EarthWorldModel) -> EmbeddingMetadata:
    return EmbeddingMetadata(
        token_mode=str(model.token_mode),
        tokens_per_timestep=int(model.last_tokens_per_timestep),
    )


def build_dataset_from_cfg(data_cfg: dict) -> torch.utils.data.Dataset:
    kind = data_cfg["kind"]
    force_single_process_io = resolve_bool_config(
        data_cfg.get("force_single_process_io"),
        default=resolve_bool_config(os.environ.get("EWM_FORCE_SINGLE_PROCESS_IO"), default=False),
    )
    if kind == "fake":
        return FakeTemporalDataset(
            num_samples=resolve_int_config(data_cfg.get("num_samples"), default=1024),
            timesteps=resolve_int_config(data_cfg.get("max_timestamps"), default=4),
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
            temporal_subclip_length=resolve_int_config(data_cfg.get("temporal_subclip_length"), default=0),
            temporal_subclip_mode=data_cfg.get("temporal_subclip_mode", "random"),
            temporal_subclip_schedule=data_cfg.get("temporal_subclip_schedule"),
        )
    if kind == "manifest":
        return ManifestTemporalDataset(
            manifest_path=data_cfg["manifest_path"],
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
            max_samples=resolve_int_config(data_cfg.get("max_samples")),
            sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
            force_single_process_io=force_single_process_io,
        )
    if kind == "hls_chip_index":
        return HLSChipTemporalDataset(
            index_path=data_cfg["index_path"],
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=224),
            max_samples=resolve_int_config(data_cfg.get("max_samples")),
            min_valid_fraction=data_cfg.get("min_valid_fraction", 0.0),
            sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
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
                    patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
                    max_shards=resolve_int_config(data_cfg.get("max_shards")),
                    max_samples=resolve_int_config(data_cfg.get("max_samples")),
                    min_clear_fraction=data_cfg.get("min_clear_fraction", 0.8),
                    sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
                    crop_mode=data_cfg.get("crop_mode", "center"),
                    repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
                    assume_single_sample_per_shard=data_cfg.get("assume_single_sample_per_shard", False),
                    max_open_shards=resolve_int_config(data_cfg.get("max_open_shards"), default=16),
                    force_single_process_io=force_single_process_io,
                    project_name=data_cfg.get("project_name") or os.environ.get("EWM_GCP_PROJECT"),
                    dataflux_threads_per_process=resolve_int_config(
                        data_cfg.get("dataflux_threads_per_process"), default=1
                    ),
                    dataflux_num_processes=resolve_int_config(data_cfg.get("dataflux_num_processes")),
                    dataflux_disable_compose=resolve_bool_config(data_cfg.get("dataflux_disable_compose"), default=False),
                )
            except ImportError:
                if use_dataflux == "true":
                    raise
        return SSL4EOZarrDataset(
            root_dir=root_dir,
            split=data_cfg.get("split", "val"),
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
            max_shards=resolve_int_config(data_cfg.get("max_shards")),
            max_samples=resolve_int_config(data_cfg.get("max_samples")),
            min_clear_fraction=data_cfg.get("min_clear_fraction", 0.8),
            sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
            assume_single_sample_per_shard=data_cfg.get("assume_single_sample_per_shard", False),
            max_open_shards=resolve_int_config(data_cfg.get("max_open_shards"), default=16),
            force_single_process_io=force_single_process_io,
        )
    if kind == "dense_temporal_index":
        return DenseTemporalNPZDataset(
            index_path=data_cfg["index_path"],
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
            max_samples=resolve_int_config(data_cfg.get("max_samples")),
            min_paired_frames=resolve_int_config(data_cfg.get("min_paired_frames"), default=1),
            min_paired_fraction=data_cfg.get("min_paired_fraction", 0.0),
            sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
            temporal_subclip_length=resolve_int_config(data_cfg.get("temporal_subclip_length"), default=0),
            temporal_subclip_mode=data_cfg.get("temporal_subclip_mode", "random"),
            temporal_subclip_schedule=data_cfg.get("temporal_subclip_schedule"),
            force_single_process_io=force_single_process_io,
        )
    if kind == "dense_temporal_zarr_index":
        return DenseTemporalZarrDataset(
            index_path=data_cfg["index_path"],
            patch_size=resolve_int_config(data_cfg.get("patch_size"), default=128),
            max_samples=resolve_int_config(data_cfg.get("max_samples")),
            min_paired_frames=resolve_int_config(data_cfg.get("min_paired_frames"), default=1),
            min_paired_fraction=data_cfg.get("min_paired_fraction", 0.0),
            sample_offset=resolve_int_config(data_cfg.get("sample_offset"), default=0),
            crop_mode=data_cfg.get("crop_mode", "center"),
            repeat_factor=resolve_int_config(data_cfg.get("repeat_factor"), default=1),
            max_open_shards=resolve_int_config(data_cfg.get("max_open_shards"), default=16),
            temporal_subclip_length=resolve_int_config(data_cfg.get("temporal_subclip_length"), default=0),
            temporal_subclip_mode=data_cfg.get("temporal_subclip_mode", "random"),
            temporal_subclip_schedule=data_cfg.get("temporal_subclip_schedule"),
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
    distributed: DistributedContext | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    distributed = distributed or DistributedContext()
    data_cfg = config[section]
    train_cfg = config["training"]
    global_batch_size = int(batch_size or train_cfg["batch_size"])
    effective_batch_size = global_batch_size
    if distributed.enabled:
        if global_batch_size % distributed.world_size != 0:
            raise ValueError(
                f"{section} batch size {global_batch_size} must be divisible by WORLD_SIZE={distributed.world_size}"
            )
        effective_batch_size = global_batch_size // distributed.world_size
    requested_num_workers = int(data_cfg.get("num_workers", 0))
    num_workers = requested_num_workers
    if getattr(dataset, "requires_single_process_io", False) and num_workers > 0:
        num_workers = 0
    use_shard_sampler = should_use_ssl4eo_shard_sampler(dataset, data_cfg, effective_batch_size)
    if distributed.enabled and not drop_last:
        use_shard_sampler = False
    batch_sampler = None
    sampler = None
    if use_shard_sampler:
        batch_sampler = SSL4EOShardBatchSampler(
            dataset=dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            seed=int(config["runtime"].get("seed", 42)),
        )
    elif distributed.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(config["runtime"].get("seed", 42)),
        )
    prefetch_factor = None
    if num_workers > 0:
        prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    if distributed.is_main_process:
        print(
            json.dumps(
                {
                    "event": "loader_config",
                    "section": section,
                    "dataset_class": dataset.__class__.__name__,
                    "requested_num_workers": requested_num_workers,
                    "effective_num_workers": num_workers,
                    "requires_single_process_io": bool(getattr(dataset, "requires_single_process_io", False)),
                    "batch_size_global": global_batch_size,
                    "batch_size_per_rank": effective_batch_size,
                    "batch_sampler": batch_sampler.__class__.__name__ if batch_sampler is not None else None,
                    "sampler": sampler.__class__.__name__ if sampler is not None else None,
                    "distributed": distributed.enabled,
                    "world_size": distributed.world_size,
                    "prefetch_factor": prefetch_factor,
                }
            )
        )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "persistent_workers": bool(num_workers),
    }
    if isinstance(dataset, SSL4EOZarrDataset):
        loader_kwargs["collate_fn"] = ewm_collate
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = effective_batch_size
        loader_kwargs["shuffle"] = shuffle if sampler is None else False
        loader_kwargs["drop_last"] = drop_last
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
    return DataLoader(**loader_kwargs)


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
        fusion_num_heads=model_cfg.get("fusion_num_heads", max(1, int(model_cfg["num_heads"]) // 2)),
        use_modality_embeddings=resolve_bool_config(model_cfg.get("use_modality_embeddings", False), default=False),
        use_patch_positional_encoding=resolve_bool_config(model_cfg.get("use_patch_positional_encoding", False), default=False),
        use_missing_modality_embeddings=resolve_bool_config(model_cfg.get("use_missing_modality_embeddings", False), default=False),
        temporal_block_type=model_cfg.get("temporal_block_type", "standard"),
        use_rope_temporal_attention=resolve_bool_config(model_cfg.get("use_rope_temporal_attention", False), default=False),
        rope_base=model_cfg.get("rope_base", 10000.0),
        use_activation_checkpointing=resolve_bool_config(model_cfg.get("use_activation_checkpointing", False), default=False),
        tokenizer_type=model_cfg.get("tokenizer_type", "linear"),
        tokenizer_mode=model_cfg.get("tokenizer_mode"),
        auto_2d_max_timesteps=model_cfg.get("auto_2d_max_timesteps", 4),
        tubelet_size=model_cfg.get("tubelet_size", 2),
        separate_sensor_encoders=resolve_bool_config(model_cfg.get("separate_sensor_encoders", True), default=True),
        use_time_gap_features=resolve_bool_config(model_cfg.get("use_time_gap_features", True), default=True),
        use_sensor_timing_features=resolve_bool_config(model_cfg.get("use_sensor_timing_features", False), default=False),
        enable_latent_dynamics=resolve_bool_config(model_cfg.get("enable_latent_dynamics", True), default=True),
        dynamics_hidden_dim=model_cfg.get("dynamics_hidden_dim"),
    )
    return model.to(device)


class StudentStepModule(torch.nn.Module):
    def __init__(self, student: EarthWorldModel, backend: str, config: dict):
        super().__init__()
        self.student = student
        self.backend = backend
        self.config = config

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        target_embeddings: torch.Tensor | None,
        target_state_sequence: torch.Tensor | None,
        target_metadata: EmbeddingMetadata | None,
        mask_spec: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        device = next(self.student.parameters()).device
        return compute_prediction_losses(
            self.student,
            target_embeddings,
            target_state_sequence,
            target_metadata,
            batch,
            mask_spec,
            self.backend,
            device,
            self.config,
        )


def resolve_backend(config: dict) -> tuple[str, torch.device, DistributedContext]:
    backend = config["runtime"]["backend"]
    if backend == "cuda":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        distributed = DistributedContext(
            enabled=world_size > 1,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend_probe": "cuda",
                        "python": sys.executable,
                        "cuda_available": torch.cuda.is_available(),
                        "device_count": torch.cuda.device_count(),
                        "distributed": distributed.enabled,
                        "world_size": distributed.world_size,
                    }
                )
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but no CUDA device is available")
        if distributed.enabled:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} is out of range for {torch.cuda.device_count()} visible CUDA devices"
                )
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            return backend, torch.device("cuda", local_rank), distributed
        return backend, torch.device("cuda"), distributed
    if backend == "tpu":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("TPU backend requested, but torch_xla is not installed") from exc
        import torch_xla.core.xla_model as xm

        return backend, xm.xla_device(), DistributedContext()
    return "cpu", torch.device("cpu"), DistributedContext()


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


def barrier_if_needed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.barrier()


def destroy_distributed_process_group(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar_sum(value: float | int, device: torch.device, distributed: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if distributed.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def reduce_scalar_mean(
    value: torch.Tensor | float,
    *,
    distributed: DistributedContext,
    device: torch.device,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float64)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if distributed.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= distributed.world_size
    return tensor


def reduce_scalar_max(value: float, device: torch.device, distributed: DistributedContext) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if distributed.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def resolve_auxiliary_data_config(config: dict) -> dict | None:
    raw_aux_cfg = config.get("auxiliary_data")
    if not raw_aux_cfg:
        return None
    aux_cfg = deepcopy(raw_aux_cfg)
    if not resolve_bool_config(aux_cfg.get("enabled", True), default=True):
        return None
    aux_cfg.pop("enabled", None)
    return aux_cfg


def _mixed_stage_steps_per_epoch(primary_loader: DataLoader, auxiliary_loader: DataLoader | None, training_cfg: dict, epoch: int) -> int:
    max_steps_per_epoch = max(0, int(training_cfg.get("max_steps_per_epoch", 0)))
    if auxiliary_loader is None:
        base_steps = len(primary_loader)
        return min(base_steps, max_steps_per_epoch) if max_steps_per_epoch > 0 else base_steps
    mixed_stage_epochs = max(0, int(training_cfg.get("mixed_stage_epochs", 0)))
    if epoch >= mixed_stage_epochs:
        base_steps = len(primary_loader)
        return min(base_steps, max_steps_per_epoch) if max_steps_per_epoch > 0 else base_steps
    configured = int(training_cfg.get("mixed_stage_steps_per_epoch", 0))
    base_steps = configured if configured > 0 else len(primary_loader)
    return min(base_steps, max_steps_per_epoch) if max_steps_per_epoch > 0 else base_steps


def training_steps_schedule(primary_loader: DataLoader, auxiliary_loader: DataLoader | None, training_cfg: dict, epochs: int) -> list[int]:
    return [
        _mixed_stage_steps_per_epoch(primary_loader, auxiliary_loader, training_cfg, epoch)
        for epoch in range(max(0, int(epochs)))
    ]


def iter_training_batches(
    primary_loader: DataLoader,
    auxiliary_loader: DataLoader | None,
    *,
    epoch: int,
    training_cfg: dict,
    seed: int,
):
    mixed_stage_epochs = max(0, int(training_cfg.get("mixed_stage_epochs", 0)))
    auxiliary_fraction = float(training_cfg.get("mixed_stage_auxiliary_fraction", 0.5))
    auxiliary_fraction = min(max(auxiliary_fraction, 0.0), 1.0)
    steps_per_epoch = _mixed_stage_steps_per_epoch(primary_loader, auxiliary_loader, training_cfg, epoch)
    if auxiliary_loader is None or epoch >= mixed_stage_epochs or auxiliary_fraction <= 0.0:
        for batch in primary_loader:
            yield "primary", batch
        return

    rng = random.Random(int(seed) + int(epoch))
    primary_iter = iter(primary_loader)
    auxiliary_iter = iter(auxiliary_loader)
    for _step in range(steps_per_epoch):
        use_auxiliary = rng.random() < auxiliary_fraction
        source = "auxiliary" if use_auxiliary else "primary"
        active_loader = auxiliary_loader if use_auxiliary else primary_loader
        active_iter = auxiliary_iter if use_auxiliary else primary_iter
        try:
            batch = next(active_iter)
        except StopIteration:
            active_iter = iter(active_loader)
            batch = next(active_iter)
        if use_auxiliary:
            auxiliary_iter = active_iter
        else:
            primary_iter = active_iter
        yield source, batch


def try_git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def collect_run_manifest(
    *,
    args: argparse.Namespace,
    config: dict,
    backend: str,
    device: torch.device,
    distributed: DistributedContext,
    checkpoint_dir: Path,
    dataset: torch.utils.data.Dataset,
    auxiliary_dataset: torch.utils.data.Dataset | None,
    eval_dataset: torch.utils.data.Dataset | None,
    metrics_path: Path,
    loader: DataLoader,
    auxiliary_loader: DataLoader | None,
    eval_loader: DataLoader | None,
) -> dict[str, Any]:
    interesting_env_prefixes = ("EWM_",)
    interesting_env_names = {
        "CONFIG_PATH",
        "GCS_RUN_URI",
        "GCS_DATA_URI",
        "LOCAL_RUN_ROOT",
        "LOCAL_DATA_ROOT",
        "LOCAL_CHECKPOINT_DIR",
        "RUN_LOG_PATH",
        "DATA_ACCESS_MODE",
    }
    env_payload = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(interesting_env_prefixes) or key in interesting_env_names
    }
    project_root = Path(__file__).resolve().parents[1]
    return {
        "created_at": time.time(),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "backend": backend,
        "device": str(device),
        "distributed": {
            "enabled": distributed.enabled,
            "world_size": distributed.world_size,
            "rank": distributed.rank,
            "local_rank": distributed.local_rank,
        },
        "config_path": str(args.config.resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "metrics_path": str(metrics_path),
        "launcher_script": os.environ.get("EWM_LAUNCHER_SCRIPT"),
        "experiment_id": os.environ.get("EWM_EXPERIMENT_ID"),
        "run_label": os.environ.get("EWM_RUN_LABEL"),
        "git_commit": try_git_commit(project_root),
        "config": config,
        "loader": {
            "class_name": loader.__class__.__name__,
            "num_workers": int(getattr(loader, "num_workers", 0)),
            "batch_size": getattr(loader, "batch_size", None),
            "prefetch_factor": getattr(loader, "prefetch_factor", None),
            "pin_memory": bool(getattr(loader, "pin_memory", False)),
            "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
            "batch_sampler": loader.batch_sampler.__class__.__name__ if getattr(loader, "batch_sampler", None) else None,
            "sampler": loader.sampler.__class__.__name__ if getattr(loader, "sampler", None) else None,
        },
        "dataset": {
            "class_name": dataset.__class__.__name__,
            "row_count": len(dataset),
            "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
            "requires_single_process_io": bool(getattr(dataset, "requires_single_process_io", False)),
            "root_dir": getattr(dataset, "root_dir", None),
            "split": getattr(dataset, "split", None),
        },
        "auxiliary_dataset": {
            "class_name": auxiliary_dataset.__class__.__name__ if auxiliary_dataset is not None else None,
            "row_count": len(auxiliary_dataset) if auxiliary_dataset is not None else 0,
            "base_row_count": getattr(auxiliary_dataset, "base_row_count", len(auxiliary_dataset))
            if auxiliary_dataset is not None
            else 0,
            "requires_single_process_io": bool(getattr(auxiliary_dataset, "requires_single_process_io", False))
            if auxiliary_dataset is not None
            else False,
            "root_dir": getattr(auxiliary_dataset, "root_dir", None) if auxiliary_dataset is not None else None,
            "split": getattr(auxiliary_dataset, "split", None) if auxiliary_dataset is not None else None,
        },
        "eval_dataset": {
            "class_name": eval_dataset.__class__.__name__ if eval_dataset is not None else None,
            "row_count": len(eval_dataset) if eval_dataset is not None else 0,
            "base_row_count": getattr(eval_dataset, "base_row_count", len(eval_dataset)) if eval_dataset is not None else 0,
        },
        "auxiliary_loader": {
            "class_name": auxiliary_loader.__class__.__name__ if auxiliary_loader is not None else None,
            "num_workers": int(getattr(auxiliary_loader, "num_workers", 0)) if auxiliary_loader is not None else 0,
            "batch_size": getattr(auxiliary_loader, "batch_size", None) if auxiliary_loader is not None else None,
            "prefetch_factor": getattr(auxiliary_loader, "prefetch_factor", None) if auxiliary_loader is not None else None,
            "pin_memory": bool(getattr(auxiliary_loader, "pin_memory", False)) if auxiliary_loader is not None else False,
            "persistent_workers": bool(getattr(auxiliary_loader, "persistent_workers", False))
            if auxiliary_loader is not None
            else False,
            "batch_sampler": auxiliary_loader.batch_sampler.__class__.__name__
            if auxiliary_loader is not None and getattr(auxiliary_loader, "batch_sampler", None)
            else None,
            "sampler": auxiliary_loader.sampler.__class__.__name__
            if auxiliary_loader is not None and getattr(auxiliary_loader, "sampler", None)
            else None,
        },
        "eval_loader": {
            "class_name": eval_loader.__class__.__name__ if eval_loader is not None else None,
            "num_workers": int(getattr(eval_loader, "num_workers", 0)) if eval_loader is not None else 0,
            "batch_size": getattr(eval_loader, "batch_size", None) if eval_loader is not None else None,
            "prefetch_factor": getattr(eval_loader, "prefetch_factor", None) if eval_loader is not None else None,
            "pin_memory": bool(getattr(eval_loader, "pin_memory", False)) if eval_loader is not None else False,
            "persistent_workers": bool(getattr(eval_loader, "persistent_workers", False))
            if eval_loader is not None
            else False,
            "batch_sampler": eval_loader.batch_sampler.__class__.__name__
            if eval_loader is not None and getattr(eval_loader, "batch_sampler", None)
            else None,
            "sampler": eval_loader.sampler.__class__.__name__
            if eval_loader is not None and getattr(eval_loader, "sampler", None)
            else None,
        },
        "environment": env_payload,
    }


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


def schedule_fractions_from_cfg(
    *,
    total_steps: int,
    training_cfg: dict[str, Any],
) -> tuple[float, float]:
    warmup_steps = resolve_int_config(training_cfg.get("warmup_steps"), default=0) or 0
    hold_steps = resolve_int_config(training_cfg.get("hold_steps"), default=0) or 0
    if warmup_steps > 0 or hold_steps > 0:
        denom = max(int(total_steps) - 1, 1)
        warmup_fraction = min(max(float(warmup_steps) / float(denom), 0.0), 1.0)
        remaining = max(1.0 - warmup_fraction, 0.0)
        hold_fraction = min(max(float(hold_steps) / float(denom), 0.0), remaining)
        return warmup_fraction, hold_fraction
    warmup_fraction = float(training_cfg.get("warmup_fraction", 0.0))
    hold_fraction = float(training_cfg.get("hold_fraction", 0.0))
    return warmup_fraction, hold_fraction


def apply_training_schedules(
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    total_steps: int,
    training_cfg: dict[str, Any],
) -> tuple[float, float]:
    progress = global_step / max(total_steps - 1, 1)
    schedule_type = str(training_cfg.get("lr_schedule", "fixed")).lower()
    warmup_fraction, hold_fraction = schedule_fractions_from_cfg(
        total_steps=total_steps,
        training_cfg=training_cfg,
    )
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


def _dense_token_layout(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
) -> tuple[int, int, int, int, int, int]:
    source = batch.get("hls")
    if source is None:
        source = batch.get("s2")
    if source is None:
        raise ValueError("Dense block masking requires s2 or hls inputs")
    height = int(source.shape[-2])
    width = int(source.shape[-1])
    return model.spatial.dense_layout(
        height=height,
        width=width,
        timesteps=int(batch["dates"].shape[1]),
    )


def _dense_token_position_ids(dates: torch.Tensor, tokens_per_timestep: int) -> torch.Tensor:
    return dates.to(dtype=torch.long).unsqueeze(-1).expand(-1, -1, tokens_per_timestep).reshape(dates.shape[0], -1)


def _sample_block_extent(
    *,
    timesteps: int,
    grid_h: int,
    grid_w: int,
    spatial_scale: tuple[float, float],
    temporal_scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
) -> tuple[int, int, int]:
    temporal_ratio = float(torch.empty(1).uniform_(float(temporal_scale[0]), float(temporal_scale[1])).item())
    time_extent = max(1, int(round(timesteps * temporal_ratio)))

    spatial_ratio = float(torch.empty(1).uniform_(float(spatial_scale[0]), float(spatial_scale[1])).item())
    spatial_keep = max(1, int(round(grid_h * grid_w * spatial_ratio)))

    aspect = float(torch.empty(1).uniform_(float(aspect_ratio[0]), float(aspect_ratio[1])).item())
    height_extent = max(1, int(round(math.sqrt(spatial_keep * aspect))))
    width_extent = max(1, int(round(math.sqrt(spatial_keep / max(aspect, 1.0e-6)))))
    height_extent = min(height_extent, grid_h)
    width_extent = min(width_extent, grid_w)
    return time_extent, height_extent, width_extent


def dense_block_mask_spec(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, Any]:
    if model.token_mode != "dense":
        raise ValueError("dense_block_mask_spec requires dense token mode")
    dates = batch["dates"]
    grouped_dates = model.spatial.aggregate_temporal_dates(dates)
    timesteps = int(grouped_dates.shape[1])
    device = dates.device
    _group_count, tokens_per_timestep, patch_count, token_multiplier, grid_h, grid_w = _dense_token_layout(model, batch)
    total_tokens = timesteps * tokens_per_timestep
    visible_mask = torch.ones(total_tokens, dtype=torch.bool)

    training_cfg = config["training"]
    block_cfgs = training_cfg.get("block_mask_cfgs") or training_cfg.get("block_masks") or [
        {
            "num_blocks": 8,
            "spatial_scale": [0.10, 0.20],
            "temporal_scale": [0.125, 0.25],
            "aspect_ratio": [0.75, 1.5],
        },
        {
            "num_blocks": 2,
            "spatial_scale": [0.50, 0.70],
            "temporal_scale": [0.25, 0.50],
            "aspect_ratio": [0.75, 1.5],
        },
    ]

    for family in block_cfgs:
        num_blocks = max(0, int(family.get("num_blocks", 0)))
        if num_blocks == 0:
            continue
        spatial_scale = tuple(float(v) for v in family.get("spatial_scale", [0.15, 0.15]))
        temporal_scale = tuple(float(v) for v in family.get("temporal_scale", [1.0, 1.0]))
        aspect_ratio = tuple(float(v) for v in family.get("aspect_ratio", [0.75, 1.5]))
        for _ in range(num_blocks):
            time_extent, height_extent, width_extent = _sample_block_extent(
                timesteps=timesteps,
                grid_h=grid_h,
                grid_w=grid_w,
                spatial_scale=spatial_scale,
                temporal_scale=temporal_scale,
                aspect_ratio=aspect_ratio,
            )
            t0 = int(torch.randint(0, timesteps - time_extent + 1, (1,)).item())
            y0 = int(torch.randint(0, grid_h - height_extent + 1, (1,)).item())
            x0 = int(torch.randint(0, grid_w - width_extent + 1, (1,)).item())
            for step in range(t0, t0 + time_extent):
                step_offset = step * tokens_per_timestep
                for yy in range(y0, y0 + height_extent):
                    patch_row_offset = yy * grid_w
                    for xx in range(x0, x0 + width_extent):
                        patch_index = patch_row_offset + xx
                        for token_group in range(token_multiplier):
                            token_index = step_offset + (token_group * patch_count) + patch_index
                            visible_mask[token_index] = False

    masked_idx = torch.nonzero(~visible_mask, as_tuple=False).squeeze(1)
    visible_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(1)
    if masked_idx.numel() == 0 or visible_idx.numel() == 0:
        visible_idx, masked_idx = temporal_mask_indices(timesteps, float(training_cfg.get("mask_ratio", 0.5)), device)
        return {
            "mode": "temporal",
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
        }

    position_ids = _dense_token_position_ids(grouped_dates, tokens_per_timestep)
    return {
        "mode": "dense_block3d",
        "visible_idx": visible_idx.to(device=device),
        "masked_idx": masked_idx.to(device=device),
        "visible_position_ids": position_ids[:, visible_idx].to(device=device),
        "masked_position_ids": position_ids[:, masked_idx].to(device=device),
        "masked_local_token_indices": (masked_idx % tokens_per_timestep).to(device=device),
    }


def sample_mask_spec(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> dict[str, Any]:
    training_cfg = config["training"]
    masking_mode = str(training_cfg.get("masking_mode", "temporal_random")).lower()
    if model.token_mode == "dense" and masking_mode in {"dense_block3d", "block3d"}:
        return dense_block_mask_spec(model, batch, config)
    if model.spatial.resolve_tokenizer_type(int(batch["dates"].shape[1])) == "conv3d":
        raise ValueError("Conv3d tubelet tokenizers require training.masking_mode=dense_block3d")
    visible_idx, masked_idx = temporal_mask_indices(
        batch["dates"].shape[1],
        float(training_cfg.get("mask_ratio", 0.5)),
        batch["dates"].device,
    )
    return {
        "mode": "temporal",
        "visible_idx": visible_idx,
        "masked_idx": masked_idx,
    }


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
        s2_present=None if batch.get("s2_present") is None else batch["s2_present"][:, visible_idx],
        s1_present=None if batch.get("s1_present") is None else batch["s1_present"][:, visible_idx],
        s2_dates=None if batch.get("s2_dates") is None else batch["s2_dates"][:, visible_idx],
        s1_dates=None if batch.get("s1_dates") is None else batch["s1_dates"][:, visible_idx],
        hls=None if hls is None else hls[:, visible_idx],
    )


def compute_target_embeddings(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor:
    return_hierarchical = bool(config["model"].get("hierarchical_layers"))
    return model.encode_timeline(
        batch.get("s2"),
        batch.get("s1"),
        batch["dates"],
        batch["mask"],
        s2_present=batch.get("s2_present"),
        s1_present=batch.get("s1_present"),
        s2_dates=batch.get("s2_dates"),
        s1_dates=batch.get("s1_dates"),
        hls=batch.get("hls"),
        return_hierarchical=return_hierarchical,
    )


def compute_target_state_sequence(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor | None:
    dynamics_loss_weight = float(config["training"].get("dynamics_loss_weight", 0.0))
    if dynamics_loss_weight <= 0.0 or model.latent_rollout_head is None:
        return None
    return model.encode_state_sequence(
        batch.get("s2"),
        batch.get("s1"),
        batch["dates"],
        batch["mask"],
        hls=batch.get("hls"),
        s2_present=batch.get("s2_present"),
        s1_present=batch.get("s1_present"),
        s2_dates=batch.get("s2_dates"),
        s1_dates=batch.get("s1_dates"),
    )[0]


def compute_online_embeddings(
    model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    *,
    return_hierarchical: bool = False,
) -> tuple[torch.Tensor, EmbeddingMetadata]:
    embeddings = model.encode_timeline(
        batch.get("s2"),
        batch.get("s1"),
        batch["dates"],
        batch["mask"],
        s2_present=batch.get("s2_present"),
        s1_present=batch.get("s1_present"),
        s2_dates=batch.get("s2_dates"),
        s1_dates=batch.get("s1_dates"),
        hls=batch.get("hls"),
        return_hierarchical=return_hierarchical,
    )
    return embeddings, current_embedding_metadata(model)


def compute_masked_prediction_loss(
    student: EarthWorldModel,
    target_embeddings: torch.Tensor | None,
    target_state_sequence: torch.Tensor | None,
    target_metadata: EmbeddingMetadata | None,
    batch: dict[str, torch.Tensor],
    mask_spec: dict[str, Any],
    backend: str,
    device: torch.device,
    config: dict,
) -> torch.Tensor:
    loss_terms = compute_prediction_losses(
        student,
        target_embeddings,
        target_state_sequence,
        target_metadata,
        batch,
        mask_spec,
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


def dense_token_temporal_context_weights(
    visible_position_ids: torch.Tensor,
    masked_position_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sqrt_distance: bool = False,
) -> torch.Tensor:
    if visible_position_ids.numel() == 0 or masked_position_ids.numel() == 0:
        return torch.ones(
            (visible_position_ids.shape[0], max(1, visible_position_ids.shape[1])),
            device=device,
            dtype=dtype,
        )
    visible = visible_position_ids.to(device=device, dtype=dtype).unsqueeze(-1)
    masked = masked_position_ids.to(device=device, dtype=dtype).unsqueeze(-1)
    distances = torch.cdist(visible, masked, p=1).amin(dim=-1).clamp_min(1.0)
    if sqrt_distance:
        distances = distances.sqrt()
    return 1.0 / distances


def expand_context_weights_for_tokens(weights: torch.Tensor | None, tokens_per_timestep: int) -> torch.Tensor | None:
    if weights is None or tokens_per_timestep <= 1:
        return weights
    return weights.repeat_interleave(tokens_per_timestep)


def select_target_embeddings(
    reference_model: EarthWorldModel,
    target_embeddings: torch.Tensor,
    target_metadata: EmbeddingMetadata,
    step_idx: torch.Tensor | None = None,
    token_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if token_idx is not None:
        if target_metadata.token_mode != "dense":
            raise ValueError("token_idx selection is only supported for dense token mode")
        return target_embeddings[:, token_idx, :]
    if step_idx is None:
        raise ValueError("Either step_idx or token_idx must be provided")
    if target_metadata.token_mode != "dense":
        return target_embeddings[:, step_idx, :]
    tokens_per_timestep = int(target_metadata.tokens_per_timestep)
    token_indices = [
        torch.arange(
            int(step.item()) * tokens_per_timestep,
            (int(step.item()) + 1) * tokens_per_timestep,
            device=target_embeddings.device,
            dtype=torch.long,
        )
        for step in step_idx
    ]
    gathered = torch.cat(token_indices, dim=0)
    return target_embeddings[:, gathered, :]


def _pool_patch_validity(valid_mask: torch.Tensor, patch_px: int) -> torch.Tensor:
    if valid_mask.ndim != 4:
        raise ValueError(f"Expected validity mask [B, T, H, W], got {tuple(valid_mask.shape)}")
    batch, timesteps, height, width = valid_mask.shape
    pooled = F.avg_pool2d(
        valid_mask.to(dtype=torch.float32).reshape(batch * timesteps, 1, height, width),
        kernel_size=patch_px,
        stride=patch_px,
    )
    return pooled.reshape(batch, timesteps, -1)


def _combine_loss_weights(
    base_weights: torch.Tensor | None,
    extra_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if base_weights is None:
        return extra_weights
    if extra_weights is None:
        return base_weights
    extra = extra_weights.to(device=base_weights.device, dtype=base_weights.dtype)
    if extra.ndim == 1:
        extra = extra.view(1, -1)
    return base_weights * extra


def _dense_all_token_target_weights(
    reference_model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    tokens_per_timestep: int,
) -> torch.Tensor | None:
    patch_px = int(reference_model.spatial.patch_px)
    fusion_mode = str(reference_model.spatial.fusion_mode)
    input_mode = str(reference_model.input_mode)
    grouped_dates = reference_model.spatial.aggregate_temporal_dates(batch["dates"])
    grouped_steps = int(grouped_dates.shape[1])

    s2_valid_mask = reference_model.spatial.aggregate_temporal_valid_mask(batch.get("s2_valid_mask"))
    s1_valid_mask = reference_model.spatial.aggregate_temporal_valid_mask(batch.get("s1_valid_mask"))
    s2_present = reference_model.spatial.aggregate_temporal_presence(batch.get("s2_present"))
    s1_present = reference_model.spatial.aggregate_temporal_presence(batch.get("s1_present"))

    if input_mode == "s2s1" and fusion_mode == "late_concat":
        token_pieces: list[torch.Tensor] = []
        patch_count = tokens_per_timestep // 2
        if s2_valid_mask is not None:
            token_pieces.append(_pool_patch_validity(s2_valid_mask, patch_px))
        elif s2_present is not None:
            token_pieces.append(
                s2_present.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, patch_count)
            )
        if s1_valid_mask is not None:
            token_pieces.append(_pool_patch_validity(s1_valid_mask, patch_px))
        elif s1_present is not None:
            token_pieces.append(
                s1_present.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, patch_count)
            )
        if not token_pieces:
            return None
        return torch.cat(token_pieces, dim=-1).reshape(batch["dates"].shape[0], grouped_steps * tokens_per_timestep)

    if s2_valid_mask is not None:
        return _pool_patch_validity(s2_valid_mask, patch_px).reshape(batch["dates"].shape[0], grouped_steps * tokens_per_timestep)

    mask = reference_model.spatial.aggregate_temporal_boolean(batch.get("mask"))
    if mask is None:
        return None
    repeated = mask.to(dtype=torch.float32)
    if tokens_per_timestep > 1:
        repeated = repeated.unsqueeze(-1).expand(-1, -1, tokens_per_timestep).reshape(batch["dates"].shape[0], -1)
    return repeated


def select_target_weights(
    reference_model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    target_metadata: EmbeddingMetadata,
    step_idx: torch.Tensor | None = None,
    token_idx: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if token_idx is not None:
        if target_metadata.token_mode != "dense":
            raise ValueError("token_idx selection is only supported for dense token mode")
        weights = _dense_all_token_target_weights(
            reference_model,
            batch,
            int(target_metadata.tokens_per_timestep),
        )
        if weights is None:
            return None
        return weights[:, token_idx]
    if step_idx is None:
        raise ValueError("Either step_idx or token_idx must be provided")
    if target_metadata.token_mode != "dense":
        mask = batch.get("mask")
        if mask is None:
            return None
        return mask[:, step_idx].to(dtype=torch.float32)
    tokens_per_timestep = int(target_metadata.tokens_per_timestep)
    weights = _dense_all_token_target_weights(reference_model, batch, tokens_per_timestep)
    if weights is None:
        return None
    grouped_dates = reference_model.spatial.aggregate_temporal_dates(batch["dates"])
    reshaped = weights.reshape(batch["dates"].shape[0], grouped_dates.shape[1], tokens_per_timestep)
    return reshaped[:, step_idx].reshape(batch["dates"].shape[0], -1)


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
    if weights.ndim == 1:
        weight_tensor = weights.view(1, -1, 1)
    elif weights.ndim == 2:
        weight_tensor = weights.unsqueeze(-1)
    else:
        raise ValueError(f"Expected 1D or 2D weights, got {tuple(weights.shape)}")
    weight_tensor = weight_tensor.to(device=errors.device, dtype=errors.dtype)
    weighted_errors = errors * weight_tensor
    normalizer = errors.shape[-1] * weight_tensor.sum().clamp_min(1.0e-6)
    return weighted_errors.sum() / normalizer


def _all_embedding_weights(
    reference_model: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    embedding_metadata: EmbeddingMetadata,
) -> torch.Tensor | None:
    if embedding_metadata.token_mode == "dense":
        return _dense_all_token_target_weights(
            reference_model,
            batch,
            int(embedding_metadata.tokens_per_timestep),
        )
    mask = batch.get("mask")
    if mask is None:
        return None
    return mask.to(dtype=torch.float32)


def _flatten_regularization_rows(
    embeddings: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    max_samples: int = 0,
) -> torch.Tensor:
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    if weights is None:
        rows = flat
    else:
        valid = weights.reshape(-1) > 0
        if not torch.any(valid):
            return flat.new_zeros((0, flat.shape[-1]))
        rows = flat[valid]
    if max_samples > 0 and rows.shape[0] > max_samples:
        sample_idx = torch.randperm(rows.shape[0], device=rows.device)[:max_samples]
        rows = rows[sample_idx]
    return rows


def _resolve_subspace_dims(
    reg_cfg: dict[str, Any],
    embed_dim: int,
) -> tuple[int, int, int]:
    dims_cfg = reg_cfg.get("split_dims", {})
    s1_private = int(dims_cfg.get("s1_private", 0))
    shared = int(dims_cfg.get("shared", 0))
    s2_private = int(dims_cfg.get("s2_private", 0))
    if min(s1_private, shared, s2_private) < 0:
        raise ValueError(f"split_dims must be non-negative, got {(s1_private, shared, s2_private)}")
    if s1_private + shared + s2_private != int(embed_dim):
        raise ValueError(
            f"split_dims {(s1_private, shared, s2_private)} do not sum to embed_dim={embed_dim}"
        )
    return s1_private, shared, s2_private


def compute_regularization_losses(
    reference_model: EarthWorldModel,
    embeddings: torch.Tensor | None,
    embedding_metadata: EmbeddingMetadata | None,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings is None or embedding_metadata is None:
        reference = next(reference_model.parameters())
        zero = reference.new_zeros(())
        return zero, zero

    reg_cfg = config.get("regularization", {})
    method = resolve_regularization_method(config)
    if method == "none":
        zero = embeddings.new_zeros(())
        return zero, zero

    mode = str(reg_cfg.get("mode", "per_subspace")).strip().lower()
    if mode not in {"per_subspace", "global"}:
        raise ValueError(f"Unsupported regularization.mode: {mode}")

    max_samples = int(reg_cfg.get("max_samples", 0))
    token_weights = _all_embedding_weights(reference_model, batch, embedding_metadata)
    rows = _flatten_regularization_rows(embeddings, token_weights, max_samples=max_samples)
    zero = embeddings.new_zeros(())
    if rows.shape[0] < 2:
        return zero, zero

    base_lambda = float(reg_cfg.get("base_lambda", 1.0))
    adaptive_alpha = float(reg_cfg.get("adaptive_alpha", 1.0))
    n_projections = int(reg_cfg.get("n_projections", 1024))
    crosscov_mu = float(reg_cfg.get("crosscov_mu", 0.0))
    vicreg_variance_target = float(reg_cfg.get("vicreg_variance_target", 1.0))
    vicreg_variance_epsilon = float(reg_cfg.get("vicreg_variance_epsilon", 1.0e-4))
    vicreg_variance_weight = float(reg_cfg.get("vicreg_variance_weight", 25.0))
    vicreg_covariance_weight = float(reg_cfg.get("vicreg_covariance_weight", 1.0))

    def regularize_subspace(z: torch.Tensor) -> torch.Tensor:
        if z.shape[1] < 1:
            return zero
        if method == "sigreg":
            reg_loss, ep_statistic = cramer_wold_sigreg(z, n_projections=n_projections)
            lambda_value = adaptive_lambda(base_lambda, adaptive_alpha, ep_statistic)
            return lambda_value * reg_loss
        reg_loss, _variance_loss, _covariance_loss = vicreg_regularization(
            z,
            variance_target=vicreg_variance_target,
            variance_epsilon=vicreg_variance_epsilon,
            variance_weight=vicreg_variance_weight,
            covariance_weight=vicreg_covariance_weight,
        )
        return reg_loss * rows.new_tensor(base_lambda)

    if mode == "global":
        return regularize_subspace(rows), zero

    s1_private_dim, shared_dim, s2_private_dim = _resolve_subspace_dims(reg_cfg, rows.shape[1])
    s1_private = rows[:, :s1_private_dim]
    shared = rows[:, s1_private_dim : s1_private_dim + shared_dim]
    s2_private = rows[:, s1_private_dim + shared_dim : s1_private_dim + shared_dim + s2_private_dim]

    regularization_loss = (
        regularize_subspace(s1_private)
        + regularize_subspace(shared)
        + regularize_subspace(s2_private)
    )

    crosscov_penalty = zero
    if crosscov_mu > 0.0 and shared_dim > 0:
        if s1_private_dim > 0:
            crosscov_penalty = crosscov_penalty + (crosscov_mu * cross_covariance_loss(shared, s1_private))
        if s2_private_dim > 0:
            crosscov_penalty = crosscov_penalty + (crosscov_mu * cross_covariance_loss(shared, s2_private))
    return regularization_loss + crosscov_penalty, crosscov_penalty


def compute_dynamics_rollout_loss(
    student: EarthWorldModel,
    teacher_state_sequence: torch.Tensor | None,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor | None:
    if teacher_state_sequence is None or student.latent_rollout_head is None:
        return None
    horizon = int(config["training"].get("dynamics_rollout_horizon", 4))
    if horizon <= 0:
        return None
    student_states, grouped_dates = student.encode_state_sequence(
        batch.get("s2"),
        batch.get("s1"),
        batch["dates"],
        batch["mask"],
        hls=batch.get("hls"),
        s2_present=batch.get("s2_present"),
        s1_present=batch.get("s1_present"),
        s2_dates=batch.get("s2_dates"),
        s1_dates=batch.get("s1_dates"),
    )
    rollout_horizon = min(int(horizon), student_states.shape[1] - 1)
    if rollout_horizon <= 0:
        return None
    grouped_s2_dates = (
        None if batch.get("s2_dates") is None else student.spatial.aggregate_temporal_dates(batch["s2_dates"])
    )
    grouped_s1_dates = (
        None if batch.get("s1_dates") is None else student.spatial.aggregate_temporal_dates(batch["s1_dates"])
    )
    predicted = student.rollout_state_predictions(
        student_states,
        grouped_dates,
        horizon=rollout_horizon,
        s2_dates=grouped_s2_dates,
        s1_dates=grouped_s1_dates,
    )
    start_count = student_states.shape[1] - rollout_horizon
    target = torch.stack(
        [teacher_state_sequence[:, step + 1 : step + 1 + start_count, :] for step in range(rollout_horizon)],
        dim=2,
    ).detach()
    target_weights = None
    grouped_mask = student.spatial.aggregate_temporal_boolean(batch.get("mask"))
    if grouped_mask is not None:
        target_weights = torch.stack(
            [grouped_mask[:, step + 1 : step + 1 + start_count].to(torch.float32) for step in range(rollout_horizon)],
            dim=2,
        )
    return weighted_embedding_mse(
        predicted.reshape(predicted.shape[0], -1, predicted.shape[-1]),
        target.reshape(target.shape[0], -1, target.shape[-1]),
        weights=None if target_weights is None else target_weights.reshape(target_weights.shape[0], -1),
    )


def compute_cross_sensor_alignment_loss(
    student: EarthWorldModel,
    batch: dict[str, torch.Tensor],
    config: dict,
) -> torch.Tensor | None:
    if str(student.input_mode) != "s2s1":
        return None
    s2 = batch.get("s2")
    s1 = batch.get("s1")
    if s2 is None or s1 is None:
        return None
    s2_present = batch.get("s2_present")
    s1_present = batch.get("s1_present")
    s2_dates = batch.get("s2_dates")
    s1_dates = batch.get("s1_dates")
    paired_weights = None
    grouped_s2_present = student.spatial.aggregate_temporal_presence(s2_present)
    grouped_s1_present = student.spatial.aggregate_temporal_presence(s1_present)
    if grouped_s2_present is not None and grouped_s1_present is not None:
        paired_weights = (grouped_s2_present & grouped_s1_present).to(torch.float32)
    else:
        grouped_mask = student.spatial.aggregate_temporal_boolean(batch.get("mask"))
        if grouped_mask is not None:
            paired_weights = grouped_mask.to(torch.float32)
    if paired_weights is None or float(paired_weights.sum().detach().cpu().item()) <= 0.0:
        return None
    training_cfg = config.get("training", {})
    use_time_proximity = resolve_bool_config(training_cfg.get("cross_sensor_time_proximity", False), default=False)
    if use_time_proximity and s2_dates is not None and s1_dates is not None:
        grouped_s2_dates = student.spatial.aggregate_temporal_dates(s2_dates).to(torch.float32)
        grouped_s1_dates = student.spatial.aggregate_temporal_dates(s1_dates).to(torch.float32)
        valid_pair = (grouped_s2_dates > 0) & (grouped_s1_dates > 0)
        decay_days = max(1.0, float(training_cfg.get("cross_sensor_time_scale_days", 7.0)))
        day_delta = torch.abs(grouped_s2_dates - grouped_s1_dates)
        proximity = torch.exp(-day_delta / decay_days)
        paired_weights = paired_weights * proximity * valid_pair.to(torch.float32)
    if float(paired_weights.sum().detach().cpu().item()) <= 0.0:
        return None

    s2_only_states, _ = student.encode_state_sequence(
        s2,
        torch.zeros_like(s1),
        batch["dates"],
        batch["mask"],
        s2_present=s2_present,
        s1_present=None if s1_present is None else torch.zeros_like(s1_present, dtype=torch.bool),
        s2_dates=s2_dates,
        s1_dates=None if s1_dates is None else torch.full_like(s1_dates, -1),
    )
    s1_only_states, _ = student.encode_state_sequence(
        torch.zeros_like(s2),
        s1,
        batch["dates"],
        batch["mask"],
        s2_present=None if s2_present is None else torch.zeros_like(s2_present, dtype=torch.bool),
        s1_present=s1_present,
        s2_dates=None if s2_dates is None else torch.full_like(s2_dates, -1),
        s1_dates=s1_dates,
    )
    return weighted_embedding_mse(s2_only_states, s1_only_states, weights=paired_weights)


def compute_prediction_losses(
    student: EarthWorldModel,
    target_embeddings: torch.Tensor | None,
    target_state_sequence: torch.Tensor | None,
    target_metadata: EmbeddingMetadata | None,
    batch: dict[str, torch.Tensor],
    mask_spec: dict[str, Any],
    backend: str,
    device: torch.device,
    config: dict,
) -> dict[str, torch.Tensor]:
    dates = batch["dates"]
    training_cfg = config["training"]
    target_mode = resolve_target_mode(config)
    regularization_method = resolve_regularization_method(config)
    predict_visible_context = bool(training_cfg.get("predict_visible_context", False))
    context_loss_weight = float(training_cfg.get("context_loss_weight", 0.0))
    context_loss_distance_weighted = bool(training_cfg.get("context_loss_distance_weighted", False))
    context_loss_sqrt_distance = bool(training_cfg.get("context_loss_sqrt_distance", True))
    dynamics_loss_weight = float(training_cfg.get("dynamics_loss_weight", 0.0))
    cross_sensor_loss_weight = float(training_cfg.get("cross_sensor_loss_weight", 0.0))
    mask_mode = str(mask_spec.get("mode", "temporal")).lower()

    with maybe_autocast(backend, device, config):
        target_embeddings_local = target_embeddings
        target_state_sequence_local = target_state_sequence
        target_metadata_local = target_metadata
        regularization_source_embeddings = None
        regularization_source_metadata = None
        return_hierarchical_targets = bool(config["model"].get("hierarchical_layers"))

        if target_mode == "self":
            if return_hierarchical_targets:
                online_target_embeddings, target_metadata_local = compute_online_embeddings(
                    student,
                    batch,
                    return_hierarchical=True,
                )
                target_embeddings_local = online_target_embeddings.detach()
            else:
                online_target_embeddings, target_metadata_local = compute_online_embeddings(
                    student,
                    batch,
                    return_hierarchical=False,
                )
                target_embeddings_local = online_target_embeddings.detach()
                if regularization_method != "none":
                    regularization_source_embeddings = online_target_embeddings
                    regularization_source_metadata = target_metadata_local
            if dynamics_loss_weight > 0.0:
                online_state_sequence = compute_target_state_sequence(student, batch, config)
                target_state_sequence_local = None if online_state_sequence is None else online_state_sequence.detach()

        if regularization_method != "none" and regularization_source_embeddings is None:
            regularization_source_embeddings, regularization_source_metadata = compute_online_embeddings(
                student,
                batch,
                return_hierarchical=False,
            )

        if target_embeddings_local is None or target_metadata_local is None:
            raise ValueError(
                f"Missing target embeddings for training.target_mode={target_mode}. "
                "EMA runs must provide target embeddings; self runs should synthesize them."
            )

        if mask_mode == "dense_block3d":
            visible_token_idx = mask_spec["visible_idx"]
            masked_token_idx = mask_spec["masked_idx"]
            visible_position_ids = mask_spec["visible_position_ids"]
            masked_position_ids = mask_spec["masked_position_ids"]
            masked_local_token_indices = mask_spec["masked_local_token_indices"]
            visible_embeddings = student.encode_timeline(
                batch.get("s2"),
                batch.get("s1"),
                dates,
                batch["mask"],
                hls=batch.get("hls"),
                s2_present=batch.get("s2_present"),
                s1_present=batch.get("s1_present"),
                s2_dates=batch.get("s2_dates"),
                s1_dates=batch.get("s1_dates"),
                token_indices=visible_token_idx,
            )
            predicted_visible, predicted_masked = student.predict_with_context_tokens(
                visible_embeddings,
                visible_position_ids,
                masked_position_ids,
                masked_local_token_indices,
                visible_token_indices=visible_token_idx,
                masked_token_indices=masked_token_idx,
            )
            masked_target = select_target_embeddings(
                student,
                target_embeddings_local,
                target_metadata_local,
                token_idx=masked_token_idx,
            ).detach()
            masked_weights = select_target_weights(
                student,
                batch,
                target_metadata_local,
                token_idx=masked_token_idx,
            )
            masked_loss = weighted_embedding_mse(predicted_masked, masked_target, weights=masked_weights)

            context_loss = torch.zeros_like(masked_loss)
            if predict_visible_context and visible_token_idx.numel() > 0 and context_loss_weight > 0.0:
                context_target = select_target_embeddings(
                    student,
                    target_embeddings_local,
                    target_metadata_local,
                    token_idx=visible_token_idx,
                ).detach()
                context_weights = None
                if context_loss_distance_weighted:
                    context_weights = dense_token_temporal_context_weights(
                        visible_position_ids,
                        masked_position_ids,
                        device=predicted_visible.device,
                        dtype=predicted_visible.dtype,
                        sqrt_distance=context_loss_sqrt_distance,
                    )
                context_target_weights = select_target_weights(
                    student,
                    batch,
                    target_metadata_local,
                    token_idx=visible_token_idx,
                )
                context_weights = _combine_loss_weights(context_target_weights, context_weights)
                context_loss = weighted_embedding_mse(predicted_visible, context_target, weights=context_weights)
        else:
            visible_idx = mask_spec["visible_idx"]
            masked_idx = mask_spec["masked_idx"]
            visible_embeddings = student_visible_embeddings(student, batch, visible_idx)
            predicted_visible, predicted_masked = student.predict_with_context(
                visible_embeddings,
                dates[:, visible_idx],
                dates[:, masked_idx],
            )
            masked_target = select_target_embeddings(
                student,
                target_embeddings_local,
                target_metadata_local,
                step_idx=masked_idx,
            ).detach()
            masked_weights = select_target_weights(student, batch, target_metadata_local, step_idx=masked_idx)
            masked_loss = weighted_embedding_mse(predicted_masked, masked_target, weights=masked_weights)

            context_loss = torch.zeros_like(masked_loss)
            if predict_visible_context and visible_idx.numel() > 0 and context_loss_weight > 0.0:
                context_target = select_target_embeddings(
                    student,
                    target_embeddings_local,
                    target_metadata_local,
                    step_idx=visible_idx,
                ).detach()
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
                context_target_weights = select_target_weights(student, batch, target_metadata_local, step_idx=visible_idx)
                context_weights = _combine_loss_weights(context_target_weights, context_weights)
                context_loss = weighted_embedding_mse(predicted_visible, context_target, weights=context_weights)

        dynamics_loss = torch.zeros_like(masked_loss)
        if dynamics_loss_weight > 0.0:
            maybe_dynamics_loss = compute_dynamics_rollout_loss(
                student,
                target_state_sequence_local,
                batch,
                config,
            )
            if maybe_dynamics_loss is not None:
                dynamics_loss = maybe_dynamics_loss

        cross_sensor_loss = torch.zeros_like(masked_loss)
        if cross_sensor_loss_weight > 0.0:
            maybe_cross_sensor_loss = compute_cross_sensor_alignment_loss(student, batch, config)
            if maybe_cross_sensor_loss is not None:
                cross_sensor_loss = maybe_cross_sensor_loss

        regularization_loss, crosscov_penalty = compute_regularization_losses(
            student,
            regularization_source_embeddings,
            regularization_source_metadata,
            batch,
            config,
        )
        total_loss = (
            masked_loss
            + (context_loss_weight * context_loss)
            + (dynamics_loss_weight * dynamics_loss)
            + (cross_sensor_loss_weight * cross_sensor_loss)
            + regularization_loss
        )

    return {
        "total_loss": total_loss,
        "masked_loss": masked_loss.detach(),
        "context_loss": context_loss.detach(),
        "dynamics_loss": dynamics_loss.detach(),
        "cross_sensor_loss": cross_sensor_loss.detach(),
        "regularization_loss": regularization_loss.detach(),
        "crosscov_penalty": crosscov_penalty.detach(),
        "context_loss_weight": torch.tensor(context_loss_weight, device=total_loss.device, dtype=total_loss.dtype),
        "dynamics_loss_weight": torch.tensor(dynamics_loss_weight, device=total_loss.device, dtype=total_loss.dtype),
        "cross_sensor_loss_weight": torch.tensor(
            cross_sensor_loss_weight,
            device=total_loss.device,
            dtype=total_loss.dtype,
        ),
        "predict_visible_context": torch.tensor(1 if predict_visible_context else 0, device=total_loss.device),
    }


def compute_student_loss_terms(
    student: EarthWorldModel,
    student_stepper: torch.nn.Module | None,
    target_embeddings: torch.Tensor | None,
    target_state_sequence: torch.Tensor | None,
    target_metadata: EmbeddingMetadata | None,
    batch: dict[str, torch.Tensor],
    mask_spec: dict[str, Any],
    backend: str,
    device: torch.device,
    config: dict,
) -> dict[str, torch.Tensor]:
    if student_stepper is not None:
        return student_stepper(batch, target_embeddings, target_state_sequence, target_metadata, mask_spec)
    return compute_prediction_losses(
        student,
        target_embeddings,
        target_state_sequence,
        target_metadata,
        batch,
        mask_spec,
        backend,
        device,
        config,
    )


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
    student_stepper: torch.nn.Module | None,
    teacher: EarthWorldModel | None,
    loader: DataLoader,
    backend: str,
    device: torch.device,
    config: dict,
    eval_cfg: dict,
    distributed: DistributedContext,
) -> dict:
    mask_mode = eval_cfg["mask_mode"]
    max_batches = int(eval_cfg.get("max_batches", 0))
    student.eval()
    if student_stepper is not None:
        student_stepper.eval()
    if teacher is not None:
        teacher.eval()

    loss_total = 0.0
    masked_loss_total = 0.0
    context_loss_total = 0.0
    dynamics_loss_total = 0.0
    cross_sensor_loss_total = 0.0
    regularization_loss_total = 0.0
    crosscov_penalty_total = 0.0
    step_count = 0
    started_at = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device, backend)
            target_embeddings = None
            target_state_sequence = None
            target_metadata = None
            if teacher is not None:
                target_embeddings = compute_target_embeddings(teacher, batch, config)
                target_metadata = current_embedding_metadata(teacher)
                target_state_sequence = compute_target_state_sequence(teacher, batch, config)

            if mask_mode == "all":
                masking_mode = str(config["training"].get("masking_mode", "temporal_random")).lower()
                if student.token_mode == "dense" and masking_mode in {"dense_block3d", "block3d"}:
                    raise ValueError("eval.mask_mode=all is not supported with dense_block3d masking; use eval.mask_mode=random")
                mask_specs = [
                    {"mode": "temporal", "visible_idx": visible_idx, "masked_idx": masked_idx}
                    for visible_idx, masked_idx in all_temporal_mask_indices(
                        batch["dates"].shape[1],
                        float(config["training"]["mask_ratio"]),
                        batch["dates"].device,
                    )
                ]
            elif mask_mode == "random":
                mask_specs = [sample_mask_spec(student, batch, config)]
            else:
                raise ValueError(f"Unsupported eval.mask_mode: {mask_mode}")

            batch_losses = [
                compute_student_loss_terms(
                    student,
                    student_stepper,
                    target_embeddings,
                    target_state_sequence,
                    target_metadata,
                    batch,
                    mask_spec,
                    backend,
                    device,
                    config,
                )
                for mask_spec in mask_specs
            ]
            batch_loss = torch.stack([loss_terms["total_loss"] for loss_terms in batch_losses]).mean()
            batch_masked_loss = torch.stack([loss_terms["masked_loss"] for loss_terms in batch_losses]).mean()
            batch_context_loss = torch.stack([loss_terms["context_loss"] for loss_terms in batch_losses]).mean()
            batch_dynamics_loss = torch.stack([loss_terms["dynamics_loss"] for loss_terms in batch_losses]).mean()
            batch_cross_sensor_loss = torch.stack([loss_terms["cross_sensor_loss"] for loss_terms in batch_losses]).mean()
            batch_regularization_loss = torch.stack([loss_terms["regularization_loss"] for loss_terms in batch_losses]).mean()
            batch_crosscov_penalty = torch.stack([loss_terms["crosscov_penalty"] for loss_terms in batch_losses]).mean()
            loss_total += float(batch_loss.cpu().item())
            masked_loss_total += float(batch_masked_loss.cpu().item())
            context_loss_total += float(batch_context_loss.cpu().item())
            dynamics_loss_total += float(batch_dynamics_loss.cpu().item())
            cross_sensor_loss_total += float(batch_cross_sensor_loss.cpu().item())
            regularization_loss_total += float(batch_regularization_loss.cpu().item())
            crosscov_penalty_total += float(batch_crosscov_penalty.cpu().item())
            step_count += 1

            if max_batches > 0 and step >= max_batches:
                break

    if distributed.enabled:
        loss_total = reduce_scalar_sum(loss_total, device, distributed)
        masked_loss_total = reduce_scalar_sum(masked_loss_total, device, distributed)
        context_loss_total = reduce_scalar_sum(context_loss_total, device, distributed)
        dynamics_loss_total = reduce_scalar_sum(dynamics_loss_total, device, distributed)
        cross_sensor_loss_total = reduce_scalar_sum(cross_sensor_loss_total, device, distributed)
        regularization_loss_total = reduce_scalar_sum(regularization_loss_total, device, distributed)
        crosscov_penalty_total = reduce_scalar_sum(crosscov_penalty_total, device, distributed)
        step_count = int(round(reduce_scalar_sum(step_count, device, distributed)))

    return {
        "mean_loss": loss_total / max(step_count, 1),
        "mean_masked_loss": masked_loss_total / max(step_count, 1),
        "mean_context_loss": context_loss_total / max(step_count, 1),
        "mean_dynamics_loss": dynamics_loss_total / max(step_count, 1),
        "mean_cross_sensor_loss": cross_sensor_loss_total / max(step_count, 1),
        "mean_regularization_loss": regularization_loss_total / max(step_count, 1),
        "mean_crosscov_penalty": crosscov_penalty_total / max(step_count, 1),
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
    backend, device, distributed = resolve_backend(config)

    try:
        dataset = build_dataset(config, section="data")
        loader = make_loader(dataset, config, section="data", distributed=distributed, shuffle=True, drop_last=True)
        loader = maybe_wrap_loader(loader, device, backend)
        auxiliary_cfg = resolve_auxiliary_data_config(config)
        auxiliary_dataset = None
        auxiliary_loader = None
        if auxiliary_cfg is not None:
            auxiliary_dataset = build_dataset_from_cfg(auxiliary_cfg)
            auxiliary_loader_config = deepcopy(config)
            auxiliary_loader_config["data"] = auxiliary_cfg
            auxiliary_batch_size = resolve_int_config(
                auxiliary_cfg.get("batch_size", config["training"].get("auxiliary_batch_size", config["training"]["batch_size"])),
                default=int(config["training"]["batch_size"]),
            )
            auxiliary_loader = make_loader(
                auxiliary_dataset,
                auxiliary_loader_config,
                section="data",
                distributed=distributed,
                batch_size=auxiliary_batch_size,
                shuffle=True,
                drop_last=True,
            )
            auxiliary_loader = maybe_wrap_loader(auxiliary_loader, device, backend)
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
                distributed=distributed,
                batch_size=eval_cfg["batch_size"],
                shuffle=False,
                drop_last=False,
            )
            eval_loader = maybe_wrap_loader(eval_loader, device, backend)

        target_mode = resolve_target_mode(config)
        student = make_model(config, device)
        teacher = None
        if target_mode == "ema":
            teacher = deepcopy(student).to(device)
            teacher.requires_grad_(False)
            teacher.eval()

        # Performance: cuDNN autotuning + torch.compile (V-JEPA 2.1 pattern)
        if backend == "cuda":
            torch.backends.cudnn.benchmark = True
            compile_model = config["training"].get("compile_model", False)
            if compile_model and distributed.enabled and distributed.is_main_process:
                print(json.dumps({"warning": "compile_model disabled under CUDA DDP"}))
            elif compile_model:
                compile_note = "student + predictor via torch.compile"
                if teacher is not None:
                    compile_note = "student + teacher + predictor via torch.compile"
                print(json.dumps({"action": "compiling_models", "note": compile_note}))
                torch._dynamo.config.optimize_ddp = False
                student = torch.compile(student)
                if teacher is not None:
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
        ema_start = float(config["training"].get("ema_momentum_start", 0.996))
        ema_end = float(config["training"].get("ema_momentum_end", 0.999))
        steps_schedule = training_steps_schedule(loader, auxiliary_loader, config["training"], epochs)
        steps_per_epoch = len(loader)
        total_steps = max(1, sum(max(1, int(step_count)) for step_count in steps_schedule))
        start_epoch = 0
        global_step = 0
        epoch_summaries: list[dict] = []
        eval_summaries: list[dict] = []
        best_validation_mean_loss = None
        resumed_from_checkpoint = None
        train_started_at = time.time()
        metrics_path = checkpoint_dir / "metrics.jsonl"
        manifest_path = checkpoint_dir / "run_manifest.json"

        if args.resume_from is not None:
            resume_path = args.resume_from.resolve()
            checkpoint = load_checkpoint(resume_path, backend)
            load_module_state(student, checkpoint["student"], "student")
            if teacher is not None:
                load_module_state(teacher, checkpoint.get("teacher", checkpoint["student"]), "teacher")
            optimizer_resumed = True
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                move_optimizer_state(optimizer, device, backend)
            except ValueError as exc:
                optimizer_resumed = False
                if distributed.is_main_process:
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
            if distributed.is_main_process:
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

        student_stepper: torch.nn.Module | None = None
        if distributed.enabled:
            student_stepper = DDP(
                StudentStepModule(student, backend, config).to(device),
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

        if distributed.is_main_process:
            write_json(
                manifest_path,
                collect_run_manifest(
                    args=args,
                    config=config,
                    backend=backend,
                    device=device,
                    distributed=distributed,
                    checkpoint_dir=checkpoint_dir,
                    dataset=dataset,
                    auxiliary_dataset=auxiliary_dataset,
                    eval_dataset=eval_dataset,
                    metrics_path=metrics_path,
                    loader=loader,
                    auxiliary_loader=auxiliary_loader,
                    eval_loader=eval_loader,
                ),
            )

            print(json.dumps(
                {
                    "backend": backend,
                    "device": str(device),
                    "distributed": distributed.enabled,
                    "world_size": distributed.world_size,
                    "rows": len(dataset),
                    "base_rows": getattr(dataset, "base_row_count", len(dataset)),
                    "steps_per_epoch": steps_per_epoch,
                    "steps_schedule": steps_schedule,
                    "checkpoint_dir": str(checkpoint_dir),
                    "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
                    "eval_base_rows": getattr(eval_dataset, "base_row_count", len(eval_dataset))
                    if eval_dataset is not None
                    else 0,
                    "auxiliary_rows": len(auxiliary_dataset) if auxiliary_dataset is not None else 0,
                    "auxiliary_base_rows": getattr(auxiliary_dataset, "base_row_count", len(auxiliary_dataset))
                    if auxiliary_dataset is not None
                    else 0,
                    "start_epoch": start_epoch + 1,
                    "manifest_path": str(manifest_path),
                    "target_mode": target_mode,
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
                    "distributed": distributed.enabled,
                    "world_size": distributed.world_size,
                    "row_count": len(dataset),
                    "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
                    "eval_row_count": len(eval_dataset) if eval_dataset is not None else 0,
                    "eval_base_row_count": getattr(eval_dataset, "base_row_count", len(eval_dataset))
                    if eval_dataset is not None
                    else 0,
                    "auxiliary_row_count": len(auxiliary_dataset) if auxiliary_dataset is not None else 0,
                    "auxiliary_base_row_count": getattr(auxiliary_dataset, "base_row_count", len(auxiliary_dataset))
                    if auxiliary_dataset is not None
                    else 0,
                    "steps_per_epoch": steps_per_epoch,
                    "steps_schedule": steps_schedule,
                    "epochs": epochs,
                    "start_epoch": start_epoch + 1,
                    "resume_from_checkpoint": resumed_from_checkpoint,
                    "checkpoint_dir": str(checkpoint_dir),
                    "manifest_path": str(manifest_path),
                    "target_mode": target_mode,
                },
            )

        for epoch in range(start_epoch, epochs):
            set_loader_epoch(loader, epoch)
            if auxiliary_loader is not None:
                set_loader_epoch(auxiliary_loader, epoch)
            if eval_loader is not None:
                set_loader_epoch(eval_loader, epoch)
            student.train()
            if student_stepper is not None:
                student_stepper.train()
            running_loss = None
            running_masked_loss = None
            running_context_loss = None
            running_dynamics_loss = None
            running_cross_sensor_loss = None
            running_regularization_loss = None
            running_crosscov_penalty = None
            epoch_loss_total = 0.0
            epoch_masked_loss_total = 0.0
            epoch_context_loss_total = 0.0
            epoch_dynamics_loss_total = 0.0
            epoch_cross_sensor_loss_total = 0.0
            epoch_regularization_loss_total = 0.0
            epoch_crosscov_penalty_total = 0.0
            epoch_primary_batch_count = 0
            epoch_auxiliary_batch_count = 0
            epoch_step_count = 0
            epoch_started_at = time.time()
            epoch_target_steps = steps_schedule[epoch] if epoch < len(steps_schedule) else len(loader)

            for step, (data_source, batch) in enumerate(
                iter_training_batches(
                    loader,
                    auxiliary_loader,
                    epoch=epoch,
                    training_cfg=config["training"],
                    seed=int(config["runtime"].get("seed", 42)),
                ),
                start=1,
            ):
                batch = move_batch(batch, device, backend)
                if data_source == "auxiliary":
                    epoch_auxiliary_batch_count += 1
                else:
                    epoch_primary_batch_count += 1
                mask_spec = sample_mask_spec(student, batch, config)
                current_lr, current_wd = apply_training_schedules(
                    optimizer,
                    global_step=global_step,
                    total_steps=total_steps,
                    training_cfg=config["training"],
                )

                target_embeddings = None
                target_state_sequence = None
                target_metadata = None
                if teacher is not None:
                    with torch.no_grad():
                        target_embeddings = compute_target_embeddings(teacher, batch, config)
                        target_metadata = current_embedding_metadata(teacher)
                        target_state_sequence = compute_target_state_sequence(teacher, batch, config)

                loss_terms = compute_student_loss_terms(
                    student,
                    student_stepper,
                    target_embeddings,
                    target_state_sequence,
                    target_metadata,
                    batch,
                    mask_spec,
                    backend,
                    device,
                    config,
                )
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

                momentum = None
                if teacher is not None:
                    progress = global_step / total_steps
                    momentum = ema_end - (ema_end - ema_start) * ((1 + math.cos(math.pi * progress)) / 2)
                    with torch.no_grad():
                        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
                            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

                detached_loss = loss.detach()
                detached_masked_loss = loss_terms["masked_loss"]
                detached_context_loss = loss_terms["context_loss"]
                detached_dynamics_loss = loss_terms["dynamics_loss"]
                detached_cross_sensor_loss = loss_terms["cross_sensor_loss"]
                detached_regularization_loss = loss_terms["regularization_loss"]
                detached_crosscov_penalty = loss_terms["crosscov_penalty"]
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
                running_dynamics_loss = (
                    detached_dynamics_loss
                    if running_dynamics_loss is None
                    else running_dynamics_loss + detached_dynamics_loss
                )
                running_cross_sensor_loss = (
                    detached_cross_sensor_loss
                    if running_cross_sensor_loss is None
                    else running_cross_sensor_loss + detached_cross_sensor_loss
                )
                running_regularization_loss = (
                    detached_regularization_loss
                    if running_regularization_loss is None
                    else running_regularization_loss + detached_regularization_loss
                )
                running_crosscov_penalty = (
                    detached_crosscov_penalty
                    if running_crosscov_penalty is None
                    else running_crosscov_penalty + detached_crosscov_penalty
                )
                epoch_loss_total += float(detached_loss.cpu().item())
                epoch_masked_loss_total += float(detached_masked_loss.cpu().item())
                epoch_context_loss_total += float(detached_context_loss.cpu().item())
                epoch_dynamics_loss_total += float(detached_dynamics_loss.cpu().item())
                epoch_cross_sensor_loss_total += float(detached_cross_sensor_loss.cpu().item())
                epoch_regularization_loss_total += float(detached_regularization_loss.cpu().item())
                epoch_crosscov_penalty_total += float(detached_crosscov_penalty.cpu().item())
                epoch_step_count += 1
                global_step += 1

                if step % log_every == 0:
                    mean_loss = reduce_scalar_mean(running_loss / log_every, distributed=distributed, device=device)
                    mean_masked_loss = reduce_scalar_mean(
                        running_masked_loss / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    mean_context_loss = reduce_scalar_mean(
                        running_context_loss / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    mean_dynamics_loss = reduce_scalar_mean(
                        running_dynamics_loss / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    mean_cross_sensor_loss = reduce_scalar_mean(
                        running_cross_sensor_loss / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    mean_regularization_loss = reduce_scalar_mean(
                        running_regularization_loss / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    mean_crosscov_penalty = reduce_scalar_mean(
                        running_crosscov_penalty / log_every,
                        distributed=distributed,
                        device=device,
                    )
                    if distributed.is_main_process:
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
                            "dynamics_loss": float(mean_dynamics_loss.cpu().item()),
                            "cross_sensor_loss": float(mean_cross_sensor_loss.cpu().item()),
                            "regularization_loss": float(mean_regularization_loss.cpu().item()),
                            "crosscov_penalty": float(mean_crosscov_penalty.cpu().item()),
                            "lr": float(current_lr),
                            "weight_decay": float(current_wd),
                            "target_mode": target_mode,
                        }
                        if momentum is not None:
                            train_step_metrics["momentum"] = float(momentum)
                        if backend == "tpu":
                            import torch_xla.core.xla_model as xm

                            log_line = (
                                f"epoch={epoch + 1} step={step}/{epoch_target_steps} "
                                f"loss={mean_loss.cpu().item():.6f} "
                                f"masked_loss={mean_masked_loss.cpu().item():.6f} "
                                f"context_loss={mean_context_loss.cpu().item():.6f} "
                                f"dynamics_loss={mean_dynamics_loss.cpu().item():.6f} "
                                f"cross_sensor_loss={mean_cross_sensor_loss.cpu().item():.6f} "
                                f"regularization_loss={mean_regularization_loss.cpu().item():.6f} "
                                f"crosscov_penalty={mean_crosscov_penalty.cpu().item():.6f} "
                                f"lr={current_lr:.6e} wd={current_wd:.6f}"
                            )
                            if momentum is not None:
                                log_line += f" momentum={momentum:.6f}"
                            xm.master_print(log_line)
                        else:
                            log_line = (
                                f"epoch={epoch + 1} step={step}/{epoch_target_steps} "
                                f"loss={mean_loss.cpu().item():.6f} "
                                f"masked_loss={mean_masked_loss.cpu().item():.6f} "
                                f"context_loss={mean_context_loss.cpu().item():.6f} "
                                f"dynamics_loss={mean_dynamics_loss.cpu().item():.6f} "
                                f"cross_sensor_loss={mean_cross_sensor_loss.cpu().item():.6f} "
                                f"regularization_loss={mean_regularization_loss.cpu().item():.6f} "
                                f"crosscov_penalty={mean_crosscov_penalty.cpu().item():.6f} "
                                f"lr={current_lr:.6e} wd={current_wd:.6f}"
                            )
                            if momentum is not None:
                                log_line += f" momentum={momentum:.6f}"
                            print(log_line)
                        append_jsonl(metrics_path, train_step_metrics)
                    running_loss = None
                    running_masked_loss = None
                    running_context_loss = None
                    running_dynamics_loss = None
                    running_cross_sensor_loss = None
                    running_regularization_loss = None
                    running_crosscov_penalty = None

                if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                    break

            epoch_duration_sec = time.time() - epoch_started_at
            epoch_loss_total = reduce_scalar_sum(epoch_loss_total, device, distributed)
            epoch_masked_loss_total = reduce_scalar_sum(epoch_masked_loss_total, device, distributed)
            epoch_context_loss_total = reduce_scalar_sum(epoch_context_loss_total, device, distributed)
            epoch_dynamics_loss_total = reduce_scalar_sum(epoch_dynamics_loss_total, device, distributed)
            epoch_cross_sensor_loss_total = reduce_scalar_sum(epoch_cross_sensor_loss_total, device, distributed)
            epoch_regularization_loss_total = reduce_scalar_sum(epoch_regularization_loss_total, device, distributed)
            epoch_crosscov_penalty_total = reduce_scalar_sum(epoch_crosscov_penalty_total, device, distributed)
            epoch_primary_batch_count = int(round(reduce_scalar_sum(epoch_primary_batch_count, device, distributed)))
            epoch_auxiliary_batch_count = int(round(reduce_scalar_sum(epoch_auxiliary_batch_count, device, distributed)))
            epoch_step_count = int(round(reduce_scalar_sum(epoch_step_count, device, distributed)))
            epoch_mean_loss = epoch_loss_total / max(epoch_step_count, 1)
            epoch_summary = {
                "event": "epoch_summary",
                "timestamp": time.time(),
                "elapsed_sec": time.time() - train_started_at,
                "epoch": epoch + 1,
                "target_steps": epoch_target_steps,
                "step_count": epoch_step_count,
                "mean_loss": epoch_mean_loss,
                "mean_masked_loss": epoch_masked_loss_total / max(epoch_step_count, 1),
                "mean_context_loss": epoch_context_loss_total / max(epoch_step_count, 1),
                "mean_dynamics_loss": epoch_dynamics_loss_total / max(epoch_step_count, 1),
                "mean_cross_sensor_loss": epoch_cross_sensor_loss_total / max(epoch_step_count, 1),
                "mean_regularization_loss": epoch_regularization_loss_total / max(epoch_step_count, 1),
                "mean_crosscov_penalty": epoch_crosscov_penalty_total / max(epoch_step_count, 1),
                "primary_batch_count": epoch_primary_batch_count,
                "auxiliary_batch_count": epoch_auxiliary_batch_count,
                "mixed_stage_active": bool(auxiliary_loader is not None and epoch < int(config["training"].get("mixed_stage_epochs", 0))),
                "duration_sec": epoch_duration_sec,
                "target_mode": target_mode,
            }
            if backend == "cuda":
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_memory_mb = reduce_scalar_max(peak_memory_mb, device, distributed)
                if distributed.is_main_process:
                    print(f"epoch={epoch + 1} peak_gpu_mem_mb={peak_memory_mb:.1f}")
                torch.cuda.reset_peak_memory_stats(device)
                epoch_summary["peak_gpu_mem_mb"] = peak_memory_mb

            if eval_loader is not None and eval_cfg is not None and (epoch + 1) % eval_cfg["every_epochs"] == 0:
                eval_summary = evaluate(student, student_stepper, teacher, eval_loader, backend, device, config, eval_cfg, distributed)
                eval_summary["epoch"] = epoch + 1
                eval_summary["event"] = "validation"
                eval_summary["timestamp"] = time.time()
                eval_summary["elapsed_sec"] = time.time() - train_started_at
                if distributed.is_main_process:
                    eval_summaries.append(eval_summary)
                epoch_summary["validation_mean_loss"] = eval_summary["mean_loss"]
                epoch_summary["validation_mean_masked_loss"] = eval_summary["mean_masked_loss"]
                epoch_summary["validation_mean_context_loss"] = eval_summary["mean_context_loss"]
                epoch_summary["validation_mean_dynamics_loss"] = eval_summary["mean_dynamics_loss"]
                epoch_summary["validation_mean_cross_sensor_loss"] = eval_summary["mean_cross_sensor_loss"]
                epoch_summary["validation_mean_regularization_loss"] = eval_summary["mean_regularization_loss"]
                epoch_summary["validation_mean_crosscov_penalty"] = eval_summary["mean_crosscov_penalty"]
                epoch_summary["validation_step_count"] = eval_summary["step_count"]
                if distributed.is_main_process:
                    print(
                        f"epoch={epoch + 1} validation_loss={eval_summary['mean_loss']:.6f} "
                        f"validation_masked_loss={eval_summary['mean_masked_loss']:.6f} "
                        f"validation_context_loss={eval_summary['mean_context_loss']:.6f} "
                        f"validation_dynamics_loss={eval_summary['mean_dynamics_loss']:.6f} "
                        f"validation_cross_sensor_loss={eval_summary['mean_cross_sensor_loss']:.6f} "
                        f"validation_regularization_loss={eval_summary['mean_regularization_loss']:.6f} "
                        f"validation_crosscov_penalty={eval_summary['mean_crosscov_penalty']:.6f} "
                        f"validation_steps={eval_summary['step_count']}"
                    )
                validation_mean_loss = float(eval_summary["mean_loss"])
                if distributed.is_main_process and (
                    best_validation_mean_loss is None or validation_mean_loss < best_validation_mean_loss
                ):
                    best_validation_mean_loss = validation_mean_loss
                    best_path = checkpoint_dir / "ewm_best_val.pt"
                    payload = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "student": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "best_validation_mean_loss": best_validation_mean_loss,
                    }
                    if teacher is not None:
                        payload["teacher"] = teacher.state_dict()
                    save_checkpoint(best_path, backend, payload)
                    print(f"saved best-val checkpoint: {best_path}")
                if distributed.is_main_process:
                    append_jsonl(metrics_path, eval_summary)

            if distributed.is_main_process:
                epoch_summaries.append(epoch_summary)
                append_jsonl(metrics_path, epoch_summary)

            if distributed.is_main_process and (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f"ewm_epoch_{epoch + 1:03d}.pt"
                payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "best_validation_mean_loss": best_validation_mean_loss,
                }
                if teacher is not None:
                    payload["teacher"] = teacher.state_dict()
                save_checkpoint(checkpoint_path, backend, payload)
                print(f"saved checkpoint: {checkpoint_path}")

            barrier_if_needed(distributed)

        if distributed.is_main_process:
            summary_path = checkpoint_dir / "training_summary.json"
            summary = {
                "backend": backend,
                "device": str(device),
                "distributed": {
                    "enabled": distributed.enabled,
                    "world_size": distributed.world_size,
                },
                "row_count": len(dataset),
                "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
                "auxiliary_row_count": len(auxiliary_dataset) if auxiliary_dataset is not None else 0,
                "auxiliary_base_row_count": getattr(auxiliary_dataset, "base_row_count", len(auxiliary_dataset))
                if auxiliary_dataset is not None
                else 0,
                "eval_row_count": len(eval_dataset) if eval_dataset is not None else 0,
                "eval_base_row_count": getattr(eval_dataset, "base_row_count", len(eval_dataset))
                if eval_dataset is not None
                else 0,
                "steps_per_epoch": steps_per_epoch,
                "steps_schedule": steps_schedule,
                "epochs": epochs,
                "global_steps_completed": global_step,
                "training_duration_sec": time.time() - train_started_at,
                "epoch_summaries": epoch_summaries,
                "eval_summaries": eval_summaries,
                "final_epoch_mean_loss": epoch_summaries[-1]["mean_loss"] if epoch_summaries else None,
                "best_validation_mean_loss": best_validation_mean_loss,
                "resume_from_checkpoint": resumed_from_checkpoint,
                "start_epoch": start_epoch + 1,
                "manifest_path": str(manifest_path),
                "target_mode": target_mode,
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
                    "manifest_path": str(manifest_path),
                    "best_validation_mean_loss": best_validation_mean_loss,
                    "global_steps_completed": global_step,
                },
            )
            print(f"wrote training summary: {summary_path}")
        barrier_if_needed(distributed)
    finally:
        destroy_distributed_process_group(distributed)


if __name__ == "__main__":
    main()
