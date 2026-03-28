"""Datasets for the Earth World Model PoC."""

from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

S2_MEAN = torch.tensor(
    [1370, 1184, 1120, 1136, 1263, 1645, 1846, 1762, 2084, 1564, 1807, 1110],
    dtype=torch.float32,
).view(12, 1, 1)
S2_STD = torch.tensor(
    [633, 650, 590, 617, 535, 680, 738, 730, 688, 554, 635, 530],
    dtype=torch.float32,
).view(12, 1, 1)
S1_MEAN = torch.tensor([-12.54, -20.19], dtype=torch.float32).view(2, 1, 1)
S1_STD = torch.tensor([5.25, 5.73], dtype=torch.float32).view(2, 1, 1)
DEFAULT_DATES = torch.tensor([45, 136, 227, 318], dtype=torch.int32)
SSL4EO_S2L2A_MEAN = torch.tensor(
    [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625],
    dtype=torch.float32,
).view(12, 1, 1)
SSL4EO_S2L2A_STD = torch.tensor(
    [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307],
    dtype=torch.float32,
).view(12, 1, 1)
SSL4EO_S1GRD_MEAN = torch.tensor([-12.577, -20.265], dtype=torch.float32).view(2, 1, 1)
SSL4EO_S1GRD_STD = torch.tensor([5.179, 5.872], dtype=torch.float32).view(2, 1, 1)


def _normalize_temporal_subclip_schedule(raw_schedule: Any) -> list[dict[str, int]]:
    if raw_schedule is None or raw_schedule is False:
        return []
    if not isinstance(raw_schedule, (list, tuple)):
        raise ValueError("temporal_subclip_schedule must be a list of {start_epoch, length} mappings")
    normalized: list[dict[str, int]] = []
    for entry in raw_schedule:
        if not isinstance(entry, dict):
            raise ValueError("temporal_subclip_schedule entries must be dicts")
        start_epoch = int(entry.get("start_epoch", 0))
        length = int(entry.get("length", 0))
        if length <= 0:
            continue
        normalized.append({"start_epoch": max(0, start_epoch), "length": length})
    normalized.sort(key=lambda item: item["start_epoch"])
    return normalized


def _resolve_temporal_subclip_length(
    *,
    fixed_length: int,
    schedule: list[dict[str, int]],
    epoch: int,
) -> int:
    resolved = max(0, int(fixed_length))
    for item in schedule:
        if int(epoch) >= int(item["start_epoch"]):
            resolved = max(0, int(item["length"]))
        else:
            break
    return resolved


def _slice_temporal_tensor(value: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
    if value is None:
        return None
    return value[start:end]


def _day_of_year_from_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        day = int(value)
        return day if day > 0 else None
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        day = int(round(float(value)))
        return day if day > 0 else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("+-").isdigit():
        day = int(text)
        return day if day > 0 else None
    try:
        timestamp = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    return int(timestamp.dayofyear)


def _parse_frame_metadata_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _parse_frame_metadata_records(value.item())
        if value.size == 1:
            return _parse_frame_metadata_records(value.reshape(-1)[0].item())
        if value.dtype == object:
            return [dict(item) for item in value.tolist() if isinstance(item, dict)]
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _sensor_day_of_year_tensor(
    frame_metadata: Any,
    *,
    sensor_key: str,
    fallback_dates: torch.Tensor,
    present: torch.Tensor | None = None,
) -> torch.Tensor:
    fallback = fallback_dates.to(torch.int32).clone()
    present_mask = (
        torch.ones_like(fallback, dtype=torch.bool)
        if present is None
        else present.to(device=fallback.device, dtype=torch.bool)
    )
    result = torch.where(present_mask, fallback, torch.full_like(fallback, -1))
    records = _parse_frame_metadata_records(frame_metadata)
    if not records:
        return result
    for index, record in enumerate(records[: result.shape[0]]):
        if not bool(present_mask[index]):
            result[index] = -1
            continue
        candidate = None
        sensor_record = record.get(sensor_key)
        if isinstance(sensor_record, dict):
            for key in ("datetime", "timestamp", "date", "acquired"):
                candidate = sensor_record.get(key)
                if candidate:
                    break
        if candidate is None:
            for key in (f"{sensor_key}_datetime", f"{sensor_key}_timestamp", f"{sensor_key}_date"):
                candidate = record.get(key)
                if candidate:
                    break
        day_of_year = _day_of_year_from_value(candidate)
        if day_of_year is not None:
            result[index] = int(day_of_year)
    return result


def _temporal_subclip_start(total_steps: int, clip_length: int, mode: str) -> int:
    if clip_length <= 0 or total_steps <= clip_length:
        return 0
    mode = str(mode).lower()
    if mode == "center":
        return (total_steps - clip_length) // 2
    if mode == "random":
        return int(torch.randint(0, total_steps - clip_length + 1, (1,)).item())
    raise ValueError(f"Unsupported temporal_subclip_mode: {mode}")


def _load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _read_text(manifest_path).splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _center_crop(frame: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = frame.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds frame size {(height, width)}")
    y0 = (height - patch_size) // 2
    x0 = (width - patch_size) // 2
    return frame[:, y0 : y0 + patch_size, x0 : x0 + patch_size]


def _random_crop(frame: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = frame.shape
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds frame size {(height, width)}")
    if height == patch_size and width == patch_size:
        return frame
    y0 = int(torch.randint(0, height - patch_size + 1, (1,)).item())
    x0 = int(torch.randint(0, width - patch_size + 1, (1,)).item())
    return frame[:, y0 : y0 + patch_size, x0 : x0 + patch_size]


def _crop_frame(frame: torch.Tensor, patch_size: int, crop_mode: str) -> torch.Tensor:
    crop_mode = crop_mode.lower()
    if crop_mode == "center":
        return _center_crop(frame, patch_size)
    if crop_mode == "random":
        return _random_crop(frame, patch_size)
    raise ValueError(f"Unsupported crop mode: {crop_mode}")


def _crop_window(height: int, width: int, patch_size: int, crop_mode: str) -> tuple[int, int]:
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds frame size {(height, width)}")
    crop_mode = crop_mode.lower()
    if crop_mode == "center":
        return (height - patch_size) // 2, (width - patch_size) // 2
    if crop_mode == "random":
        if height == patch_size and width == patch_size:
            return 0, 0
        y0 = int(torch.randint(0, height - patch_size + 1, (1,)).item())
        x0 = int(torch.randint(0, width - patch_size + 1, (1,)).item())
        return y0, x0
    raise ValueError(f"Unsupported crop mode: {crop_mode}")


def _crop_sequence(sequence: torch.Tensor, patch_size: int, crop_mode: str) -> torch.Tensor:
    if sequence.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W] tensor, got {tuple(sequence.shape)}")
    _, _, height, width = sequence.shape
    y0, x0 = _crop_window(height, width, patch_size, crop_mode)
    return _crop_sequence_at(sequence, patch_size, y0, x0)


def _crop_sequence_at(sequence: torch.Tensor, patch_size: int, y0: int, x0: int) -> torch.Tensor:
    if sequence.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W] tensor, got {tuple(sequence.shape)}")
    _, _, height, width = sequence.shape
    if y0 < 0 or x0 < 0 or (y0 + patch_size) > height or (x0 + patch_size) > width:
        raise ValueError(f"Invalid crop window {(y0, x0, patch_size)} for frame size {(height, width)}")
    return sequence[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]


def _crop_mask_sequence(sequence: torch.Tensor, patch_size: int, crop_mode: str) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [T, H, W] tensor, got {tuple(sequence.shape)}")
    _, height, width = sequence.shape
    y0, x0 = _crop_window(height, width, patch_size, crop_mode)
    return _crop_mask_sequence_at(sequence, patch_size, y0, x0)


def _crop_mask_sequence_at(sequence: torch.Tensor, patch_size: int, y0: int, x0: int) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [T, H, W] tensor, got {tuple(sequence.shape)}")
    _, height, width = sequence.shape
    if y0 < 0 or x0 < 0 or (y0 + patch_size) > height or (x0 + patch_size) > width:
        raise ValueError(f"Invalid crop window {(y0, x0, patch_size)} for frame size {(height, width)}")
    return sequence[:, y0 : y0 + patch_size, x0 : x0 + patch_size]


def _read_binary(path: str) -> bytes:
    if path.startswith("gs://"):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError("gcsfs is required to read gs:// paths") from exc
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as handle:
            return handle.read()

    return Path(path).read_bytes()


def _read_text(path: str | Path) -> str:
    path_str = str(path)
    if path_str.startswith("gs://"):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError("gcsfs is required to read gs:// text paths") from exc
        fs = gcsfs.GCSFileSystem()
        with fs.open(path_str, "r", encoding="utf-8") as handle:
            return handle.read()
    return Path(path_str).read_text(encoding="utf-8")


def _read_zarr_group(path: str):
    if path.startswith("gs://"):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError("gcsfs is required to read gs:// Zarr paths") from exc
        fs = gcsfs.GCSFileSystem()
        return zarr.open_group(fs.get_mapper(path), mode="r")
    return zarr.open_group(path, mode="r")


def _is_gcs_path(path: str | Path) -> bool:
    return str(path).startswith("gs://")


def _normalize_gcs_path(path: str) -> str:
    if path.startswith("gs://"):
        return path.rstrip("/")
    return f"gs://{path.lstrip('/')}".rstrip("/")


def _parse_gcs_url(path: str | Path) -> tuple[str, str]:
    normalized = _normalize_gcs_path(str(path))
    without_scheme = normalized[len("gs://") :]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS path: {path}")
    return bucket, prefix.rstrip("/")


def _detect_gcp_project(explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    for key in ("EWM_GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        value = os.environ.get(key)
        if value:
            return value
    cloud_ml_project = os.environ.get("CLOUD_ML_PROJECT_ID")
    if cloud_ml_project:
        return cloud_ml_project
    try:
        import google.auth
    except ImportError:
        google = None  # type: ignore[assignment]
    if google is not None:
        try:
            _credentials, project_name = google.auth.default()
        except Exception:
            project_name = None
        if project_name:
            return project_name
    raise RuntimeError("Unable to determine GCP project for Dataflux; set EWM_GCP_PROJECT or GOOGLE_CLOUD_PROJECT")


def _path_name(path: str | Path) -> str:
    return str(path).rstrip("/").split("/")[-1]


def _path_stem(path: str | Path) -> str:
    return Path(_path_name(path)).stem


def _ssl4eo_shard_key(path: str | Path) -> str:
    name = _path_name(path)
    if name.endswith(".zarr.zip"):
        return name[: -len(".zarr.zip")]
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return Path(name).stem


def _is_ssl4eo_zip_path(path: str | Path) -> bool:
    return _path_name(path).endswith(".zarr.zip")


def _is_ssl4eo_dir_path(path: str | Path) -> bool:
    return _path_name(path).endswith(".zarr")


def _zip_fs_mapper(zip_fs, path_hint: str | Path):
    root_listing = zip_fs.ls("")
    top_level_files: set[str] = set()
    top_level_dirs: list[str] = []
    for item in root_listing:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip("/")
            item_type = str(item.get("type", ""))
        else:
            name = str(item).strip("/")
            item_type = ""
        if not name:
            continue
        if item_type == "directory":
            top_level_dirs.append(name)
        else:
            top_level_files.add(name)
    if {".zgroup", ".zattrs", ".zmetadata"} & top_level_files:
        return zip_fs.get_mapper("")
    if len(top_level_dirs) == 1:
        return zip_fs.get_mapper(top_level_dirs[0])
    return zip_fs.get_mapper(_path_stem(path_hint))


def _zip_store_mapper(path: str | Path):
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError("fsspec is required to read SSL4EO zip shards") from exc
    zip_fs = fsspec.filesystem("zip", fo=str(path))
    return _zip_fs_mapper(zip_fs, path)


def _zip_store_mapper_from_bytes(raw_bytes: bytes, path_hint: str | Path):
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError("fsspec is required to read SSL4EO zip shards") from exc
    buffer = io.BytesIO(raw_bytes)
    zip_fs = fsspec.filesystem("zip", fo=buffer)
    return _zip_fs_mapper(zip_fs, path_hint), buffer


def _zarr_store_mapper(path: str | Path):
    path_str = str(path)
    if _is_ssl4eo_zip_path(path_str):
        return _zip_store_mapper(path_str)
    if _is_gcs_path(path_str):
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError("gcsfs is required to read gs:// Zarr shards") from exc
        fs = gcsfs.GCSFileSystem()
        return fs.get_mapper(_normalize_gcs_path(path_str))
    return str(Path(path_str).resolve())


def _prepare_dense_temporal_sample(
    *,
    s2: torch.Tensor,
    s1: torch.Tensor,
    frame_mask: torch.Tensor,
    s2_frame_mask: torch.Tensor | None,
    s1_frame_mask: torch.Tensor | None,
    dates: torch.Tensor,
    s2_dates: torch.Tensor | None,
    s1_dates: torch.Tensor | None,
    s2_valid_mask: torch.Tensor,
    s1_valid_mask: torch.Tensor,
    patch_size: int,
    crop_mode: str,
) -> dict[str, torch.Tensor]:
    if s2.ndim != 4 or s2.shape[1] != 10:
        raise ValueError(f"Invalid dense s2 shape {tuple(s2.shape)}")
    if s1.ndim != 4 or s1.shape[1] != 2:
        raise ValueError(f"Invalid dense s1 shape {tuple(s1.shape)}")
    if s2.shape[0] != s1.shape[0]:
        raise ValueError(f"Mismatched timestep counts {s2.shape[0]} and {s1.shape[0]}")

    _, _, height, width = s2.shape
    y0, x0 = _crop_window(height, width, patch_size, crop_mode)
    s2 = _crop_sequence_at(s2, patch_size, y0, x0)
    s1 = _crop_sequence_at(s1, patch_size, y0, x0)
    s2_valid_mask = _crop_mask_sequence_at(s2_valid_mask, patch_size, y0, x0)
    s1_valid_mask = _crop_mask_sequence_at(s1_valid_mask, patch_size, y0, x0)

    full_s2 = torch.full((s2.shape[0], 12, s2.shape[2], s2.shape[3]), float("nan"), dtype=torch.float32)
    full_s2[:, 1] = s2[:, 0]   # B02
    full_s2[:, 2] = s2[:, 1]   # B03
    full_s2[:, 3] = s2[:, 2]   # B04
    full_s2[:, 4] = s2[:, 3]   # B05
    full_s2[:, 5] = s2[:, 4]   # B06
    full_s2[:, 6] = s2[:, 5]   # B07
    full_s2[:, 7] = s2[:, 6]   # B08
    full_s2[:, 8] = s2[:, 7]   # B8A
    full_s2[:, 10] = s2[:, 8]  # B11
    full_s2[:, 11] = s2[:, 9]  # B12

    full_s2_valid_mask = s2_valid_mask.clone()
    missing_band_fill = SSL4EO_S2L2A_MEAN[[0, 9]].view(1, 2, 1, 1)
    full_s2[:, [0, 9]] = missing_band_fill.expand(s2.shape[0], -1, s2.shape[2], s2.shape[3])

    full_s2 = torch.where(full_s2_valid_mask.unsqueeze(1), full_s2, SSL4EO_S2L2A_MEAN.unsqueeze(0))
    s1 = torch.where(s1_valid_mask.unsqueeze(1), s1, SSL4EO_S1GRD_MEAN.unsqueeze(0))

    full_s2 = (full_s2 - SSL4EO_S2L2A_MEAN.unsqueeze(0)) / (SSL4EO_S2L2A_STD.unsqueeze(0) + 1e-6)
    s1 = (s1 - SSL4EO_S1GRD_MEAN.unsqueeze(0)) / (SSL4EO_S1GRD_STD.unsqueeze(0) + 1e-6)
    s2_present = (
        s2_frame_mask.to(torch.bool)
        if s2_frame_mask is not None
        else s2_valid_mask.reshape(s2_valid_mask.shape[0], -1).any(dim=1)
    )
    s1_present = (
        s1_frame_mask.to(torch.bool)
        if s1_frame_mask is not None
        else s1_valid_mask.reshape(s1_valid_mask.shape[0], -1).any(dim=1)
    )
    timestep_mask = frame_mask.to(torch.bool) | s2_present | s1_present
    resolved_s2_dates = dates.to(torch.int32).clone() if s2_dates is None else s2_dates.to(torch.int32)
    resolved_s1_dates = dates.to(torch.int32).clone() if s1_dates is None else s1_dates.to(torch.int32)
    resolved_s2_dates = torch.where(s2_present, resolved_s2_dates, torch.full_like(resolved_s2_dates, -1))
    resolved_s1_dates = torch.where(s1_present, resolved_s1_dates, torch.full_like(resolved_s1_dates, -1))
    return {
        "s2": full_s2,
        "s1": s1,
        "dates": dates.to(torch.int32),
        "s2_dates": resolved_s2_dates,
        "s1_dates": resolved_s1_dates,
        "mask": timestep_mask,
        "paired_mask": frame_mask.to(torch.bool),
        "s2_present": s2_present,
        "s1_present": s1_present,
        "s2_valid_mask": s2_valid_mask.to(torch.bool),
        "s1_valid_mask": s1_valid_mask.to(torch.bool),
    }


class FakeTemporalDataset(Dataset):
    """Small fake dataset for Phase 0 sanity checks."""

    def __init__(
        self,
        num_samples: int = 1024,
        timesteps: int = 4,
        patch_size: int = 128,
        temporal_subclip_length: int = 0,
        temporal_subclip_mode: str = "random",
        temporal_subclip_schedule: list[dict[str, int]] | None = None,
    ):
        self.num_samples = int(num_samples)
        self.timesteps = int(timesteps)
        self.patch_size = int(patch_size)
        self.temporal_subclip_length = max(0, int(temporal_subclip_length))
        self.temporal_subclip_mode = str(temporal_subclip_mode).lower()
        self.temporal_subclip_schedule = _normalize_temporal_subclip_schedule(temporal_subclip_schedule)
        self.current_epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _current_temporal_subclip_length(self) -> int:
        return _resolve_temporal_subclip_length(
            fixed_length=self.temporal_subclip_length,
            schedule=self.temporal_subclip_schedule,
            epoch=self.current_epoch,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        base = torch.randn(12, self.patch_size, self.patch_size, generator=generator) * 0.5
        s2 = torch.stack(
            [
                base + 0.25 * torch.randn(base.shape, generator=generator, dtype=base.dtype) + (step * 0.1)
                for step in range(self.timesteps)
            ],
            dim=0,
        )
        s1 = torch.randn(self.timesteps, 2, self.patch_size, self.patch_size, generator=generator)
        mask = torch.ones(self.timesteps, dtype=torch.bool)
        dates = torch.linspace(1.0, 365.0, steps=self.timesteps, dtype=torch.float32).round().to(torch.int32)
        s2_dates = dates.clone()
        s1_dates = torch.clamp(dates + 2, max=366).to(torch.int32)
        s2_valid_mask = torch.ones(self.timesteps, self.patch_size, self.patch_size, dtype=torch.bool)
        s1_valid_mask = torch.ones(self.timesteps, self.patch_size, self.patch_size, dtype=torch.bool)
        clip_length = self._current_temporal_subclip_length()
        if clip_length > 0 and self.timesteps > clip_length:
            start = _temporal_subclip_start(self.timesteps, clip_length, self.temporal_subclip_mode)
            end = start + clip_length
            s2 = _slice_temporal_tensor(s2, start, end)
            s1 = _slice_temporal_tensor(s1, start, end)
            mask = _slice_temporal_tensor(mask, start, end)
            dates = _slice_temporal_tensor(dates, start, end)
            s2_dates = _slice_temporal_tensor(s2_dates, start, end)
            s1_dates = _slice_temporal_tensor(s1_dates, start, end)
            s2_valid_mask = _slice_temporal_tensor(s2_valid_mask, start, end)
            s1_valid_mask = _slice_temporal_tensor(s1_valid_mask, start, end)
        return {
            "s2": s2.float(),
            "s1": s1.float(),
            "dates": dates,
            "s2_dates": s2_dates,
            "s1_dates": s1_dates,
            "mask": mask,
            "paired_mask": mask,
            "s2_present": mask,
            "s1_present": mask,
            "s2_valid_mask": s2_valid_mask,
            "s1_valid_mask": s1_valid_mask,
        }


class ManifestTemporalDataset(Dataset):
    """Real-data dataset backed by a manifest of pre-extracted `.npz` samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        patch_size: int = 128,
        max_samples: int | None = None,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        force_single_process_io: bool = False,
    ):
        if _is_gcs_path(manifest_path):
            self.manifest_path = _normalize_gcs_path(str(manifest_path))
        else:
            self.manifest_path = str(Path(manifest_path).resolve())
        self.requires_single_process_io = _is_gcs_path(self.manifest_path) or _as_bool(force_single_process_io)
        self.patch_size = int(patch_size)
        self.crop_mode = str(crop_mode).lower()
        self.repeat_factor = max(1, int(repeat_factor))
        self.sample_offset = max(0, int(sample_offset))
        self.rows = _load_manifest(self.manifest_path)
        if self.sample_offset > 0:
            self.rows = self.rows[self.sample_offset :]
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[: int(max_samples)]
        self.base_row_count = len(self.rows)

    def __len__(self) -> int:
        return self.base_row_count * self.repeat_factor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index % self.base_row_count]
        raw = _read_binary(row["path"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as sample:
            s2 = torch.from_numpy(np.asarray(sample["s2"])).float()
            s1 = torch.from_numpy(np.asarray(sample["s1"])).float()
            dates = torch.from_numpy(np.asarray(sample["dates"])).to(torch.int32) if "dates" in sample.files else DEFAULT_DATES.clone()
            mask = torch.from_numpy(np.asarray(sample["mask"])).to(torch.bool) if "mask" in sample.files else torch.ones(4, dtype=torch.bool)
            if "s2_day_of_year" in sample.files:
                s2_dates = torch.from_numpy(np.asarray(sample["s2_day_of_year"])).to(torch.int32)
            elif "s2_dates" in sample.files:
                s2_dates = torch.from_numpy(np.asarray(sample["s2_dates"])).to(torch.int32)
            else:
                s2_dates = _sensor_day_of_year_tensor(row.get("frame_metadata_json"), sensor_key="s2", fallback_dates=dates, present=mask)
            if "s1_day_of_year" in sample.files:
                s1_dates = torch.from_numpy(np.asarray(sample["s1_day_of_year"])).to(torch.int32)
            elif "s1_dates" in sample.files:
                s1_dates = torch.from_numpy(np.asarray(sample["s1_dates"])).to(torch.int32)
            else:
                s1_dates = _sensor_day_of_year_tensor(row.get("frame_metadata_json"), sensor_key="s1", fallback_dates=dates, present=mask)

        if s2.ndim != 4 or s2.shape[:2] != (4, 12):
            raise ValueError(f"Row {index} has invalid s2 shape {tuple(s2.shape)}")
        if s1.ndim != 4 or s1.shape[:2] != (4, 2):
            raise ValueError(f"Row {index} has invalid s1 shape {tuple(s1.shape)}")

        s2 = torch.stack([_crop_frame(frame, self.patch_size, self.crop_mode) for frame in s2], dim=0)
        s1 = torch.stack([_crop_frame(frame, self.patch_size, self.crop_mode) for frame in s1], dim=0)

        s2 = (s2 - S2_MEAN.unsqueeze(0)) / (S2_STD.unsqueeze(0) + 1e-6)
        s1 = (s1 - S1_MEAN.unsqueeze(0)) / (S1_STD.unsqueeze(0) + 1e-6)

        return {"s2": s2, "s1": s1, "dates": dates, "s2_dates": s2_dates, "s1_dates": s1_dates, "mask": mask}


class HLSChipTemporalDataset(Dataset):
    """Dataset backed by the existing real HLS chip index from the gas pipeline."""

    def __init__(
        self,
        index_path: str | Path,
        patch_size: int = 224,
        max_samples: int | None = None,
        min_valid_fraction: float = 0.0,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        force_single_process_io: bool = False,
    ):
        if _is_gcs_path(index_path):
            self.index_path = _normalize_gcs_path(str(index_path))
        else:
            self.index_path = str(Path(index_path).resolve())
        self.requires_single_process_io = _is_gcs_path(self.index_path) or _as_bool(force_single_process_io)
        self.patch_size = int(patch_size)
        self.frame_count = 4
        self.crop_mode = str(crop_mode).lower()
        self.repeat_factor = max(1, int(repeat_factor))
        self.sample_offset = max(0, int(sample_offset))
        frame = pd.read_parquet(self.index_path)
        if min_valid_fraction > 0:
            frame = frame[frame["valid_fraction"].fillna(0.0) >= float(min_valid_fraction)].copy()
        if self.sample_offset > 0:
            frame = frame.iloc[self.sample_offset :].copy()
        if max_samples is not None and max_samples > 0:
            frame = frame.head(int(max_samples)).copy()
        self.rows = frame.reset_index(drop=True)
        self.base_row_count = len(self.rows)

    def __len__(self) -> int:
        return self.base_row_count * self.repeat_factor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index % self.base_row_count]
        raw = _read_binary(str(row["chip_path"]))
        with np.load(io.BytesIO(raw), allow_pickle=False) as sample:
            chip = np.asarray(sample["chip"], dtype=np.float32)  # [6, 4, H, W]
            temporal_coords = np.asarray(sample["temporal_coords"], dtype=np.float32)  # [4, 2]
            valid_mask = np.asarray(sample["valid_mask"], dtype=np.uint8)  # [4, H, W]

        if chip.shape[0] != 6 or chip.shape[1] != self.frame_count:
            raise ValueError(f"Row {index} has invalid chip shape {chip.shape}")

        hls = torch.from_numpy(np.transpose(chip, (1, 0, 2, 3))).float()  # [4, 6, H, W]
        valid_pixels = torch.from_numpy(valid_mask.astype(np.float32))  # [4, H, W]
        hls = torch.nan_to_num(hls, nan=0.0, posinf=0.0, neginf=0.0)
        hls = hls * valid_pixels.unsqueeze(1)
        hls = torch.stack([_crop_frame(frame, self.patch_size, self.crop_mode) for frame in hls], dim=0)
        valid_pixels = torch.stack(
            [_crop_frame(frame.unsqueeze(0), self.patch_size, self.crop_mode).squeeze(0) for frame in valid_pixels],
            dim=0,
        )

        dates = torch.from_numpy(np.rint(temporal_coords[:, 1]).astype(np.int32))
        frame_validity = valid_pixels.reshape(self.frame_count, -1).mean(dim=1) > 0.5

        return {"hls": hls, "dates": dates, "mask": frame_validity}


class DenseTemporalNPZDataset(Dataset):
    """Dataset backed by dense temporal per-sequence `.npz` outputs."""

    def __init__(
        self,
        index_path: str | Path,
        patch_size: int = 128,
        max_samples: int | None = None,
        min_paired_frames: int = 1,
        min_paired_fraction: float = 0.0,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        temporal_subclip_length: int = 0,
        temporal_subclip_mode: str = "random",
        temporal_subclip_schedule: list[dict[str, int]] | None = None,
        force_single_process_io: bool = False,
    ):
        if _is_gcs_path(index_path):
            self.index_path = _normalize_gcs_path(str(index_path))
        else:
            self.index_path = str(Path(index_path).resolve())
        self.requires_single_process_io = _is_gcs_path(self.index_path) or _as_bool(force_single_process_io)
        self.patch_size = int(patch_size)
        self.crop_mode = str(crop_mode).lower()
        self.repeat_factor = max(1, int(repeat_factor))
        self.sample_offset = max(0, int(sample_offset))
        self.temporal_subclip_length = max(0, int(temporal_subclip_length))
        self.temporal_subclip_mode = str(temporal_subclip_mode).lower()
        self.temporal_subclip_schedule = _normalize_temporal_subclip_schedule(temporal_subclip_schedule)
        self.current_epoch = 0

        frame = pd.read_parquet(self.index_path)
        required_columns = {"sequence_path", "paired_frame_count", "frame_count"}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Dense temporal index is missing required columns: {missing_columns}")

        frame = frame[frame["paired_frame_count"].fillna(0).astype(int) >= int(min_paired_frames)].copy()
        if min_paired_fraction > 0:
            paired_fraction = frame["paired_frame_count"].fillna(0).astype(float) / frame["frame_count"].clip(lower=1).astype(float)
            frame = frame[paired_fraction >= float(min_paired_fraction)].copy()
        if self.sample_offset > 0:
            frame = frame.iloc[self.sample_offset :].copy()
        if max_samples is not None and max_samples > 0:
            frame = frame.head(int(max_samples)).copy()
        self.rows = frame.reset_index(drop=True)
        self.base_row_count = len(self.rows)

    def __len__(self) -> int:
        return self.base_row_count * self.repeat_factor

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _current_temporal_subclip_length(self) -> int:
        return _resolve_temporal_subclip_length(
            fixed_length=self.temporal_subclip_length,
            schedule=self.temporal_subclip_schedule,
            epoch=self.current_epoch,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index % self.base_row_count]
        raw = _read_binary(str(row["sequence_path"]))
        with np.load(io.BytesIO(raw), allow_pickle=False) as sample:
            s2 = torch.from_numpy(np.asarray(sample["s2"])).float()
            s1 = torch.from_numpy(np.asarray(sample["s1"])).float()
            frame_mask = (
                torch.from_numpy(np.asarray(sample["frame_mask"])).to(torch.bool)
                if "frame_mask" in sample.files
                else torch.ones(s2.shape[0], dtype=torch.bool)
            )
            dates = (
                torch.from_numpy(np.asarray(sample["day_of_year"])).to(torch.int32)
                if "day_of_year" in sample.files
                else DEFAULT_DATES[: s2.shape[0]].clone()
            )
            frame_metadata_value = (
                np.asarray(sample["frame_metadata_json"])
                if "frame_metadata_json" in sample.files
                else row.get("frame_metadata_json")
            )
            s2_valid_mask = (
                torch.from_numpy(np.asarray(sample["s2_valid_mask"])).to(torch.bool)
                if "s2_valid_mask" in sample.files
                else torch.isfinite(s2).all(dim=1)
            )
            s1_valid_mask = (
                torch.from_numpy(np.asarray(sample["s1_valid_mask"])).to(torch.bool)
                if "s1_valid_mask" in sample.files
                else torch.isfinite(s1).all(dim=1)
            )
            s2_frame_mask = (
                torch.from_numpy(np.asarray(sample["s2_frame_mask"])).to(torch.bool)
                if "s2_frame_mask" in sample.files
                else s2_valid_mask.reshape(s2_valid_mask.shape[0], -1).any(dim=1)
            )
            s1_frame_mask = (
                torch.from_numpy(np.asarray(sample["s1_frame_mask"])).to(torch.bool)
                if "s1_frame_mask" in sample.files
                else s1_valid_mask.reshape(s1_valid_mask.shape[0], -1).any(dim=1)
            )
            if "s2_day_of_year" in sample.files:
                s2_dates = torch.from_numpy(np.asarray(sample["s2_day_of_year"])).to(torch.int32)
            elif "s2_dates" in sample.files:
                s2_dates = torch.from_numpy(np.asarray(sample["s2_dates"])).to(torch.int32)
            else:
                s2_dates = _sensor_day_of_year_tensor(frame_metadata_value, sensor_key="s2", fallback_dates=dates, present=s2_frame_mask)
            if "s1_day_of_year" in sample.files:
                s1_dates = torch.from_numpy(np.asarray(sample["s1_day_of_year"])).to(torch.int32)
            elif "s1_dates" in sample.files:
                s1_dates = torch.from_numpy(np.asarray(sample["s1_dates"])).to(torch.int32)
            else:
                s1_dates = _sensor_day_of_year_tensor(frame_metadata_value, sensor_key="s1", fallback_dates=dates, present=s1_frame_mask)
        clip_length = self._current_temporal_subclip_length()
        if clip_length > 0 and s2.shape[0] > clip_length:
            start = _temporal_subclip_start(s2.shape[0], clip_length, self.temporal_subclip_mode)
            end = start + clip_length
            s2 = _slice_temporal_tensor(s2, start, end)
            s1 = _slice_temporal_tensor(s1, start, end)
            frame_mask = _slice_temporal_tensor(frame_mask, start, end)
            dates = _slice_temporal_tensor(dates, start, end)
            s2_dates = _slice_temporal_tensor(s2_dates, start, end)
            s1_dates = _slice_temporal_tensor(s1_dates, start, end)
            s2_valid_mask = _slice_temporal_tensor(s2_valid_mask, start, end)
            s1_valid_mask = _slice_temporal_tensor(s1_valid_mask, start, end)
            s2_frame_mask = _slice_temporal_tensor(s2_frame_mask, start, end)
            s1_frame_mask = _slice_temporal_tensor(s1_frame_mask, start, end)
        return _prepare_dense_temporal_sample(
            s2=s2,
            s1=s1,
            frame_mask=frame_mask,
            s2_frame_mask=s2_frame_mask,
            s1_frame_mask=s1_frame_mask,
            dates=dates,
            s2_dates=s2_dates,
            s1_dates=s1_dates,
            s2_valid_mask=s2_valid_mask,
            s1_valid_mask=s1_valid_mask,
            patch_size=self.patch_size,
            crop_mode=self.crop_mode,
        )


class DenseTemporalZarrDataset(Dataset):
    """Dataset backed by dense temporal shard-level Zarr outputs."""

    def __init__(
        self,
        index_path: str | Path,
        patch_size: int = 128,
        max_samples: int | None = None,
        min_paired_frames: int = 1,
        min_paired_fraction: float = 0.0,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        max_open_shards: int = 16,
        temporal_subclip_length: int = 0,
        temporal_subclip_mode: str = "random",
        temporal_subclip_schedule: list[dict[str, int]] | None = None,
        force_single_process_io: bool = False,
    ):
        if _is_gcs_path(index_path):
            self.index_path = _normalize_gcs_path(str(index_path))
        else:
            self.index_path = str(Path(index_path).resolve())
        self.requires_single_process_io = _is_gcs_path(self.index_path) or _as_bool(force_single_process_io)
        self.patch_size = int(patch_size)
        self.crop_mode = str(crop_mode).lower()
        self.repeat_factor = max(1, int(repeat_factor))
        self.sample_offset = max(0, int(sample_offset))
        self.max_open_shards = max(1, int(max_open_shards))
        self.temporal_subclip_length = max(0, int(temporal_subclip_length))
        self.temporal_subclip_mode = str(temporal_subclip_mode).lower()
        self.temporal_subclip_schedule = _normalize_temporal_subclip_schedule(temporal_subclip_schedule)
        self.current_epoch = 0
        self._cache: OrderedDict[str, Any] = OrderedDict()

        frame = pd.read_parquet(self.index_path)
        required_columns = {"zarr_shard_path", "row_in_shard", "paired_frame_count", "frame_count"}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Dense temporal Zarr index is missing required columns: {missing_columns}")

        frame = frame[frame["paired_frame_count"].fillna(0).astype(int) >= int(min_paired_frames)].copy()
        if min_paired_fraction > 0:
            paired_fraction = frame["paired_frame_count"].fillna(0).astype(float) / frame["frame_count"].clip(lower=1).astype(float)
            frame = frame[paired_fraction >= float(min_paired_fraction)].copy()
        if self.sample_offset > 0:
            frame = frame.iloc[self.sample_offset :].copy()
        if max_samples is not None and max_samples > 0:
            frame = frame.head(int(max_samples)).copy()
        self.rows = frame.reset_index(drop=True)
        self.base_row_count = len(self.rows)

    def _clear_cache(self) -> None:
        while self._cache:
            _path, group = self._cache.popitem(last=False)
            close = getattr(group, "close", None)
            if callable(close):
                close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def _open_group(self, path: str):
        if path in self._cache:
            group = self._cache.pop(path)
            self._cache[path] = group
            return group
        if len(self._cache) >= self.max_open_shards:
            self._cache.popitem(last=False)
        group = _read_zarr_group(path)
        self._cache[path] = group
        return group

    def __len__(self) -> int:
        return self.base_row_count * self.repeat_factor

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(0, int(epoch))

    def _current_temporal_subclip_length(self) -> int:
        return _resolve_temporal_subclip_length(
            fixed_length=self.temporal_subclip_length,
            schedule=self.temporal_subclip_schedule,
            epoch=self.current_epoch,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index % self.base_row_count]
        group = self._open_group(str(row["zarr_shard_path"]))
        row_idx = int(row["row_in_shard"])

        frame_count = int(np.asarray(group["frame_count"][row_idx]))
        s2 = torch.from_numpy(np.asarray(group["s2"][row_idx, :frame_count])).float()
        s1 = torch.from_numpy(np.asarray(group["s1"][row_idx, :frame_count])).float()
        frame_mask = torch.from_numpy(np.asarray(group["frame_mask"][row_idx, :frame_count])).to(torch.bool)
        s2_frame_mask = torch.from_numpy(np.asarray(group["s2_frame_mask"][row_idx, :frame_count])).to(torch.bool)
        s1_frame_mask = torch.from_numpy(np.asarray(group["s1_frame_mask"][row_idx, :frame_count])).to(torch.bool)
        dates = torch.from_numpy(np.asarray(group["day_of_year"][row_idx, :frame_count])).to(torch.int32)
        s2_valid_mask = torch.from_numpy(np.asarray(group["s2_valid_mask"][row_idx, :frame_count])).to(torch.bool)
        s1_valid_mask = torch.from_numpy(np.asarray(group["s1_valid_mask"][row_idx, :frame_count])).to(torch.bool)
        if "s2_day_of_year" in group:
            s2_dates = torch.from_numpy(np.asarray(group["s2_day_of_year"][row_idx, :frame_count])).to(torch.int32)
        elif "s2_dates" in group:
            s2_dates = torch.from_numpy(np.asarray(group["s2_dates"][row_idx, :frame_count])).to(torch.int32)
        else:
            frame_metadata_value = (
                np.asarray(group["frame_metadata_json"][row_idx]) if "frame_metadata_json" in group else row.get("frame_metadata_json")
            )
            s2_dates = _sensor_day_of_year_tensor(frame_metadata_value, sensor_key="s2", fallback_dates=dates, present=s2_frame_mask)
        if "s1_day_of_year" in group:
            s1_dates = torch.from_numpy(np.asarray(group["s1_day_of_year"][row_idx, :frame_count])).to(torch.int32)
        elif "s1_dates" in group:
            s1_dates = torch.from_numpy(np.asarray(group["s1_dates"][row_idx, :frame_count])).to(torch.int32)
        else:
            frame_metadata_value = (
                np.asarray(group["frame_metadata_json"][row_idx]) if "frame_metadata_json" in group else row.get("frame_metadata_json")
            )
            s1_dates = _sensor_day_of_year_tensor(frame_metadata_value, sensor_key="s1", fallback_dates=dates, present=s1_frame_mask)
        clip_length = self._current_temporal_subclip_length()
        if clip_length > 0 and s2.shape[0] > clip_length:
            start = _temporal_subclip_start(s2.shape[0], clip_length, self.temporal_subclip_mode)
            end = start + clip_length
            s2 = _slice_temporal_tensor(s2, start, end)
            s1 = _slice_temporal_tensor(s1, start, end)
            frame_mask = _slice_temporal_tensor(frame_mask, start, end)
            s2_frame_mask = _slice_temporal_tensor(s2_frame_mask, start, end)
            s1_frame_mask = _slice_temporal_tensor(s1_frame_mask, start, end)
            dates = _slice_temporal_tensor(dates, start, end)
            s2_dates = _slice_temporal_tensor(s2_dates, start, end)
            s1_dates = _slice_temporal_tensor(s1_dates, start, end)
            s2_valid_mask = _slice_temporal_tensor(s2_valid_mask, start, end)
            s1_valid_mask = _slice_temporal_tensor(s1_valid_mask, start, end)

        return _prepare_dense_temporal_sample(
            s2=s2,
            s1=s1,
            frame_mask=frame_mask,
            s2_frame_mask=s2_frame_mask,
            s1_frame_mask=s1_frame_mask,
            dates=dates,
            s2_dates=s2_dates,
            s1_dates=s1_dates,
            s2_valid_mask=s2_valid_mask,
            s1_valid_mask=s1_valid_mask,
            patch_size=self.patch_size,
            crop_mode=self.crop_mode,
        )


class SSL4EOZarrDataset(Dataset):
    """Minimal paired S2L2A + S1GRD loader for downloaded SSL4EO Zarr shards."""

    returns_batched_getitems = True

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "val",
        patch_size: int = 128,
        max_shards: int | None = None,
        max_samples: int | None = None,
        min_clear_fraction: float = 0.8,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        assume_single_sample_per_shard: bool = False,
        max_open_shards: int = 16,
        force_single_process_io: bool = False,
    ):
        if _is_gcs_path(root_dir):
            self.root_dir = _normalize_gcs_path(str(root_dir))
        else:
            self.root_dir = str(Path(root_dir).resolve())
        self.requires_single_process_io = _is_gcs_path(self.root_dir) or _as_bool(force_single_process_io)
        self.split = split
        self.patch_size = int(patch_size)
        self.min_clear_fraction = float(min_clear_fraction)
        self.sample_offset = max(0, int(sample_offset))
        self.crop_mode = str(crop_mode).lower()
        self.repeat_factor = max(1, int(repeat_factor))
        self.assume_single_sample_per_shard = bool(assume_single_sample_per_shard)
        self.max_open_shards = max(1, int(max_open_shards))
        self._xr = self._import_xarray()
        self._cache: OrderedDict[str, Any] = OrderedDict()

        s2_paths = self._list_shards("S2L2A")
        s1_paths = self._list_shards("S1GRD")
        if max_shards is not None and max_shards > 0:
            s2_paths = s2_paths[: int(max_shards)]
            s1_paths = s1_paths[: int(max_shards)]
        if not s2_paths or not s1_paths:
            raise FileNotFoundError(f"No SSL4EO shard pairs found under {self.root_dir}/{split}")
        if len(s2_paths) != len(s1_paths):
            raise ValueError("S2L2A and S1GRD shard counts do not match")

        self.shard_pairs = []
        for s2_path, s1_path in zip(s2_paths, s1_paths):
            if _ssl4eo_shard_key(s2_path) != _ssl4eo_shard_key(s1_path):
                raise ValueError(f"Mismatched SSL4EO pair: {_path_name(s2_path)} vs {_path_name(s1_path)}")
            self.shard_pairs.append((s2_path, s1_path))
        self.rows: list[dict[str, Any]] = []
        if self.assume_single_sample_per_shard:
            for shard_index, (s2_path, _s1_path) in enumerate(self.shard_pairs):
                self.rows.append(
                    {
                        "shard_index": shard_index,
                        "sample_index": None,
                        "sample_id": _ssl4eo_shard_key(s2_path),
                    }
                )
        else:
            self.rows = self._build_rows()

        if self.sample_offset > 0:
            self.rows = self.rows[self.sample_offset :]
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[: int(max_samples)]
        self.base_row_count = len(self.rows)
        self._clear_cache()

    def _probe_shard_rows(self, s2_path: str | Path, s1_path: str | Path) -> tuple[int, list[str]]:
        ds_s2 = self._open_dataset(s2_path)
        ds_s1 = self._open_dataset(s1_path)
        if "sample" in ds_s2.dims and int(ds_s2.sizes.get("sample", 0)) > 1:
            s2_samples = np.asarray(ds_s2.coords["sample"].values)
            s1_samples = np.asarray(ds_s1.coords["sample"].values)
            if not np.array_equal(s2_samples, s1_samples):
                raise ValueError(f"Sample coordinates do not align for {_path_name(s2_path)} and {_path_name(s1_path)}")
            return int(len(s2_samples)), [str(sample_id) for sample_id in s2_samples]
        sample_coord = ds_s2.coords.get("sample")
        sample_id = str(sample_coord.item()) if sample_coord is not None else _path_stem(s2_path)
        return 1, [sample_id]

    def _build_rows_uniform(self, template_sample_ids: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if len(template_sample_ids) == 1:
            for shard_index, (s2_path, _s1_path) in enumerate(self.shard_pairs):
                rows.append(
                    {
                        "shard_index": shard_index,
                        "sample_index": None,
                        "sample_id": _ssl4eo_shard_key(s2_path),
                    }
                )
            return rows

        for shard_index, (s2_path, _s1_path) in enumerate(self.shard_pairs):
            shard_name = _ssl4eo_shard_key(s2_path)
            for sample_index, sample_id in enumerate(template_sample_ids):
                rows.append(
                    {
                        "shard_index": shard_index,
                        "sample_index": int(sample_index),
                        "sample_id": f"{shard_name}::{sample_id}",
                    }
                )
        return rows

    def _build_rows_exact(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for shard_index, (s2_path, s1_path) in enumerate(self.shard_pairs):
            sample_count, sample_ids = self._probe_shard_rows(s2_path, s1_path)
            if sample_count > 1:
                shard_name = _ssl4eo_shard_key(s2_path)
                for sample_index, sample_id in enumerate(sample_ids):
                    rows.append(
                        {
                            "shard_index": shard_index,
                            "sample_index": int(sample_index),
                            "sample_id": f"{shard_name}::{sample_id}",
                        }
                    )
                continue
            rows.append(
                {
                    "shard_index": shard_index,
                    "sample_index": None,
                    "sample_id": sample_ids[0],
                }
            )
        return rows

    def _build_rows(self) -> list[dict[str, Any]]:
        first_count, first_sample_ids = self._probe_shard_rows(*self.shard_pairs[0])
        if len(self.shard_pairs) == 1:
            return self._build_rows_uniform(first_sample_ids)

        last_count, last_sample_ids = self._probe_shard_rows(*self.shard_pairs[-1])
        if first_count == last_count and len(first_sample_ids) == len(last_sample_ids):
            return self._build_rows_uniform(first_sample_ids)
        return self._build_rows_exact()

    def _import_xarray(self):
        try:
            import xarray as xr
        except ImportError as exc:
            raise RuntimeError("xarray is required for SSL4EO Zarr loading") from exc
        return xr

    def _list_shards(self, modality: str) -> list[str | Path]:
        if _is_gcs_path(self.root_dir):
            try:
                import gcsfs
            except ImportError as exc:
                raise RuntimeError("gcsfs is required to list gs:// SSL4EO shards") from exc
            fs = gcsfs.GCSFileSystem()
            pattern = f"{self.root_dir}/{self.split}/{modality}/*.zarr.zip"
            matches = fs.glob(pattern)
            return sorted(_normalize_gcs_path(match) for match in matches if str(match).endswith(".zarr.zip"))
        split_dir = Path(self.root_dir) / self.split / modality
        zip_paths = sorted(split_dir.glob("*.zarr.zip"))
        dir_paths = sorted(path for path in split_dir.glob("*.zarr") if path.is_dir())
        by_key: dict[str, Path] = {}
        for path in zip_paths:
            by_key[_ssl4eo_shard_key(path)] = path
        # Prefer plain .zarr directories over zipped shards when both exist.
        for path in dir_paths:
            by_key[_ssl4eo_shard_key(path)] = path
        return [by_key[key] for key in sorted(by_key)]

    def _open_dataset(self, path: str | Path):
        key = str(path)
        if key in self._cache:
            dataset = self._cache.pop(key)
            self._cache[key] = dataset
            return dataset

        if len(self._cache) >= self.max_open_shards:
            _, stale_dataset = self._cache.popitem(last=False)
            close = getattr(stale_dataset, "close", None)
            if callable(close):
                close()

        dataset = self._xr.open_zarr(_zarr_store_mapper(path))
        self._cache[key] = dataset
        return dataset

    def _clear_cache(self) -> None:
        while self._cache:
            _path, dataset = self._cache.popitem(last=False)
            close = getattr(dataset, "close", None)
            if callable(close):
                close()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return self.base_row_count * self.repeat_factor

    def _select_sample(self, dataset, sample_index: int | None):
        if "sample" not in dataset.dims:
            return dataset
        if sample_index is None:
            # For micro-PoC configs that intentionally treat each shard as one
            # training row, use the first sample deterministically.
            return dataset.isel(sample=0, drop=True)
        return dataset.isel(sample=sample_index, drop=True)

    def _select_samples(self, dataset, sample_indices: list[int | None]):
        if "sample" not in dataset.dims:
            return dataset
        if not sample_indices:
            raise ValueError("sample_indices must not be empty")
        if any(sample_index is None for sample_index in sample_indices):
            if any(sample_index is not None for sample_index in sample_indices):
                raise ValueError("Mixed None and integer sample indices are not supported")
            return dataset.isel(sample=[0], drop=False)
        return dataset.isel(sample=[int(sample_index) for sample_index in sample_indices], drop=False)

    def _time_values(self, dataset) -> np.ndarray:
        if "time_" in dataset.variables:
            return np.asarray(dataset["time_"].values)
        if "time" in dataset.coords:
            return np.asarray(dataset["time"].values)
        if "time" in dataset.variables:
            return np.asarray(dataset["time"].values)
        raise KeyError("SSL4EO dataset is missing a time coordinate")

    def _data_array_values(self, data_array, preferred_order: tuple[str, ...], expected_ndim: int) -> np.ndarray:
        dims = list(getattr(data_array, "dims", ()))
        transpose_order = [dim for dim in preferred_order if dim in dims] + [dim for dim in dims if dim not in preferred_order]
        if transpose_order and transpose_order != dims:
            data_array = data_array.transpose(*transpose_order)
        values = np.asarray(data_array.values)
        if values.ndim == expected_ndim - 1:
            values = np.expand_dims(values, axis=0)
        if values.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim} dims for {getattr(data_array, 'name', 'array')}, got {values.ndim}")
        return values

    def _dates_tensor(self, time_values: np.ndarray, batch_size: int) -> torch.Tensor:
        times = np.asarray(time_values)
        times = np.squeeze(times)
        if times.ndim == 0:
            times = times.reshape(1, 1)
        elif times.ndim == 1:
            times = np.broadcast_to(times.reshape(1, -1), (batch_size, times.shape[0]))
        elif times.ndim == 2:
            if times.shape[0] == batch_size:
                pass
            elif times.shape[1] == batch_size:
                times = times.T
            elif times.shape[0] == 1:
                times = np.broadcast_to(times, (batch_size, times.shape[1]))
            else:
                raise ValueError(f"Unexpected SSL4EO time shape {times.shape} for batch size {batch_size}")
        else:
            raise ValueError(f"Unexpected SSL4EO time rank {times.ndim}")
        timestamps = pd.to_datetime(times.reshape(-1))
        day_of_year = np.asarray(timestamps.dayofyear, dtype=np.int32).reshape(batch_size, -1)
        return torch.from_numpy(day_of_year)

    def _crop_ssl4eo_batch(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        clear_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if s2.ndim != 5 or s1.ndim != 5 or clear_pixels.ndim != 4:
            raise ValueError(
                f"Expected batched SSL4EO tensors [B,T,C,H,W]/[B,T,H,W], got {tuple(s2.shape)}, {tuple(s1.shape)}, {tuple(clear_pixels.shape)}"
            )
        batch_size, _, _, height, width = s2.shape
        if self.crop_mode.lower() == "center":
            y0, x0 = _crop_window(height, width, self.patch_size, self.crop_mode)
            return (
                s2[:, :, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size],
                s1[:, :, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size],
                clear_pixels[:, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size],
            )

        cropped_s2: list[torch.Tensor] = []
        cropped_s1: list[torch.Tensor] = []
        cropped_clear: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            # Keep one crop window per sample so S2/S1/cloud remain spatially aligned.
            y0, x0 = _crop_window(height, width, self.patch_size, self.crop_mode)
            cropped_s2.append(s2[batch_index, :, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size])
            cropped_s1.append(s1[batch_index, :, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size])
            cropped_clear.append(clear_pixels[batch_index, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size])
        return torch.stack(cropped_s2, dim=0), torch.stack(cropped_s1, dim=0), torch.stack(cropped_clear, dim=0)

    def _fetch_rows_batch(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("rows must not be empty")
        shard_index = int(rows[0]["shard_index"])
        if any(int(row["shard_index"]) != shard_index for row in rows):
            raise ValueError("SSL4EO batched fetch expects rows from a single shard")
        s2_path, s1_path = self.shard_pairs[shard_index]
        ds_s2 = self._open_dataset(s2_path)
        ds_s1 = self._open_dataset(s1_path)
        sample_indices = [row["sample_index"] for row in rows]

        batch_s2 = self._select_samples(ds_s2, sample_indices)
        batch_s1 = self._select_samples(ds_s1, sample_indices)

        s2 = torch.from_numpy(self._data_array_values(batch_s2["bands"], ("sample", "time", "band", "y", "x"), 5)).float()
        s1 = torch.from_numpy(self._data_array_values(batch_s1["bands"], ("sample", "time", "band", "y", "x"), 5)).float()
        cloud = torch.from_numpy(self._data_array_values(batch_s2["cloud_mask"], ("sample", "time", "y", "x"), 4)).float()
        dates = self._dates_tensor(self._time_values(batch_s2), batch_size=s2.shape[0])

        clear_pixels = cloud.eq(0)
        clear_fraction = clear_pixels.reshape(clear_pixels.shape[0], clear_pixels.shape[1], -1).float().mean(dim=2)
        frame_mask = clear_fraction >= self.min_clear_fraction

        s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
        s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
        s2, s1, clear_pixels = self._crop_ssl4eo_batch(s2, s1, clear_pixels)

        s2_fill = SSL4EO_S2L2A_MEAN.view(1, 1, 12, 1, 1)
        s1_mean = SSL4EO_S1GRD_MEAN.view(1, 1, 2, 1, 1)
        s1_std = SSL4EO_S1GRD_STD.view(1, 1, 2, 1, 1)
        s2_std = SSL4EO_S2L2A_STD.view(1, 1, 12, 1, 1)
        s2 = torch.where(clear_pixels.unsqueeze(2), s2, s2_fill)
        s2 = (s2 - s2_fill) / (s2_std + 1e-6)
        s1 = (s1 - s1_mean) / (s1_std + 1e-6)

        return {
            "s2": s2,
            "s1": s1,
            "dates": dates.to(torch.int32),
            "s2_dates": dates.to(torch.int32),
            "s1_dates": dates.to(torch.int32),
            "mask": frame_mask.to(torch.bool),
            "paired_mask": frame_mask.to(torch.bool),
            "s2_present": frame_mask.to(torch.bool),
            "s1_present": frame_mask.to(torch.bool),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self.__getitems__([int(index)])
        return {key: value[0] for key, value in batch.items()}

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        if not indices:
            raise ValueError("indices must not be empty")
        normalized_indices = [int(index) for index in indices]
        rows = [self.rows[index % self.base_row_count] for index in normalized_indices]
        grouped_rows: OrderedDict[int, list[tuple[int, dict[str, Any]]]] = OrderedDict()
        for position, row in enumerate(rows):
            grouped_rows.setdefault(int(row["shard_index"]), []).append((position, row))

        if len(grouped_rows) == 1:
            return self._fetch_rows_batch([row for _position, row in next(iter(grouped_rows.values()))])

        ordered: dict[str, list[torch.Tensor | None]] = {
            "s2": [None] * len(rows),
            "s1": [None] * len(rows),
            "dates": [None] * len(rows),
            "s2_dates": [None] * len(rows),
            "s1_dates": [None] * len(rows),
            "mask": [None] * len(rows),
            "paired_mask": [None] * len(rows),
            "s2_present": [None] * len(rows),
            "s1_present": [None] * len(rows),
        }
        for entries in grouped_rows.values():
            group_batch = self._fetch_rows_batch([row for _position, row in entries])
            for batch_offset, (position, _row) in enumerate(entries):
                for key in ordered:
                    ordered[key][position] = group_batch[key][batch_offset]

        stacked: dict[str, torch.Tensor] = {}
        for key, values in ordered.items():
            if any(value is None for value in values):
                raise RuntimeError(f"Incomplete SSL4EO batched fetch for key {key}")
            stacked[key] = torch.stack([value for value in values if value is not None], dim=0)
        return stacked


class SSL4EOZarrDatafluxDataset(SSL4EOZarrDataset):
    """Paired SSL4EO loader streaming ZIP shards from GCS via Dataflux."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "val",
        patch_size: int = 128,
        max_shards: int | None = None,
        max_samples: int | None = None,
        min_clear_fraction: float = 0.8,
        sample_offset: int = 0,
        crop_mode: str = "center",
        repeat_factor: int = 1,
        assume_single_sample_per_shard: bool = False,
        max_open_shards: int = 16,
        force_single_process_io: bool = False,
        project_name: str | None = None,
        dataflux_threads_per_process: int = 1,
        dataflux_num_processes: int | None = None,
        dataflux_disable_compose: bool = False,
    ):
        self.project_name = _detect_gcp_project(project_name)
        self._dataflux = self._import_dataflux()
        self._dataflux_threads_per_process = max(1, int(dataflux_threads_per_process))
        self._dataflux_num_processes = None if dataflux_num_processes in (None, 0) else max(1, int(dataflux_num_processes))
        self._dataflux_disable_compose = bool(dataflux_disable_compose)
        self._dataflux_datasets: dict[str, Any] = {}
        self._dataflux_object_index: dict[str, int] = {}
        super().__init__(
            root_dir=root_dir,
            split=split,
            patch_size=patch_size,
            max_shards=max_shards,
            max_samples=max_samples,
            min_clear_fraction=min_clear_fraction,
            sample_offset=sample_offset,
            crop_mode=crop_mode,
            repeat_factor=repeat_factor,
            assume_single_sample_per_shard=assume_single_sample_per_shard,
            max_open_shards=max_open_shards,
            force_single_process_io=force_single_process_io,
        )
        self.requires_single_process_io = _as_bool(force_single_process_io)

    def _import_dataflux(self):
        try:
            from dataflux_pytorch import dataflux_mapstyle_dataset
        except ImportError as exc:
            raise ImportError("gcs-torch-dataflux is required for SSL4EO Dataflux loading") from exc
        return dataflux_mapstyle_dataset

    def _list_shards(self, modality: str) -> list[str]:
        if not _is_gcs_path(self.root_dir):
            raise ValueError("SSL4EOZarrDatafluxDataset requires a gs:// root_dir")
        bucket_name, prefix = _parse_gcs_url(self.root_dir)
        object_prefix = "/".join(part for part in (prefix, self.split, modality) if part).rstrip("/") + "/"
        config_kwargs: dict[str, Any] = {
            "prefix": object_prefix,
            "sort_listing_results": True,
            "threads_per_process": self._dataflux_threads_per_process,
            "disable_compose": self._dataflux_disable_compose,
        }
        if self._dataflux_num_processes is not None:
            config_kwargs["num_processes"] = self._dataflux_num_processes
        dataset = self._dataflux.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=bucket_name,
            config=self._dataflux.Config(**config_kwargs),
        )
        self._dataflux_datasets[modality] = dataset
        matches: list[str] = []
        for idx, (object_name, _size_bytes) in enumerate(dataset.objects):
            if not str(object_name).endswith(".zarr.zip"):
                continue
            normalized = _normalize_gcs_path(f"gs://{bucket_name}/{object_name}")
            matches.append(normalized)
            self._dataflux_object_index[normalized] = idx
        return matches

    def _resolve_modality(self, path: str | Path) -> str:
        path_str = str(path)
        for modality in ("S2L2A", "S1GRD"):
            token = f"/{modality}/"
            if token in path_str:
                return modality
        raise KeyError(f"Unable to infer SSL4EO modality from path: {path_str}")

    def _insert_cached_bytes(self, path: str, raw_bytes: bytes):
        if path in self._cache:
            bundle = self._cache.pop(path)
            self._cache[path] = bundle
            return bundle["dataset"]
        while len(self._cache) >= self.max_open_shards:
            _stale_path, stale_bundle = self._cache.popitem(last=False)
            close = getattr(stale_bundle["dataset"], "close", None)
            if callable(close):
                close()
        mapper, backing_buffer = _zip_store_mapper_from_bytes(raw_bytes, path)
        dataset = self._xr.open_zarr(mapper)
        self._cache[path] = {"dataset": dataset, "backing_buffer": backing_buffer}
        return dataset

    def _prefetch_missing_paths(self, paths: list[str]) -> None:
        grouped: dict[str, list[tuple[str, int]]] = {"S2L2A": [], "S1GRD": []}
        for path in paths:
            normalized = _normalize_gcs_path(path)
            if normalized in self._cache:
                bundle = self._cache.pop(normalized)
                self._cache[normalized] = bundle
                continue
            modality = self._resolve_modality(normalized)
            grouped[modality].append((normalized, self._dataflux_object_index[normalized]))

        for modality, entries in grouped.items():
            if not entries:
                continue
            dataset = self._dataflux_datasets[modality]
            unique_entries: list[tuple[str, int]] = []
            seen_paths: set[str] = set()
            for path, object_idx in entries:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                unique_entries.append((path, object_idx))
            raw_payloads = dataset.__getitems__([object_idx for _path, object_idx in unique_entries])
            for (path, _object_idx), raw_bytes in zip(unique_entries, raw_payloads):
                self._insert_cached_bytes(path, raw_bytes)

    def _open_dataset(self, path: str | Path):
        normalized = _normalize_gcs_path(str(path))
        if normalized in self._cache:
            bundle = self._cache.pop(normalized)
            self._cache[normalized] = bundle
            return bundle["dataset"]
        modality = self._resolve_modality(normalized)
        dataset = self._dataflux_datasets[modality]
        object_idx = self._dataflux_object_index[normalized]
        raw_bytes = dataset[object_idx]
        return self._insert_cached_bytes(normalized, raw_bytes)

    def _clear_cache(self) -> None:
        while self._cache:
            _path, bundle = self._cache.popitem(last=False)
            close = getattr(bundle["dataset"], "close", None)
            if callable(close):
                close()

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        if not indices:
            return []
        prefetch_paths: list[str] = []
        for index in indices:
            row = self.rows[index % self.base_row_count]
            s2_path, s1_path = self.shard_pairs[row["shard_index"]]
            prefetch_paths.extend((str(s2_path), str(s1_path)))
        self._prefetch_missing_paths(prefetch_paths)
        return [self.__getitem__(index) for index in indices]
