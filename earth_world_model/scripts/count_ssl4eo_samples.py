#!/usr/bin/env python3
"""Count real SSL4EO samples across paired Zarr shards.

This is the first gating step before a medium-scale run. The current micro-PoC
used `assume_single_sample_per_shard: true`, which can undercount available
training examples substantially when shards contain multiple samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def shard_key(path: Path) -> str:
    name = path.name
    if name.endswith(".zarr.zip"):
        return name[: -len(".zarr.zip")]
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return path.stem


def resolve_modality_paths(root_dir: Path, split: str, modality: str) -> list[Path]:
    split_dir = root_dir / split / modality
    zip_paths = sorted(split_dir.glob("*.zarr.zip"))
    dir_paths = sorted(path for path in split_dir.glob("*.zarr") if path.is_dir())
    by_key: dict[str, Path] = {}
    for path in zip_paths:
        by_key[shard_key(path)] = path
    for path in dir_paths:
        by_key[shard_key(path)] = path
    return [by_key[key] for key in sorted(by_key)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def import_xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray is required to inspect SSL4EO Zarr shards") from exc
    return xr


def open_zarr_target(path: Path):
    xr = import_xarray()
    if path.name.endswith(".zarr.zip"):
        try:
            import fsspec
        except ImportError as exc:
            raise RuntimeError("fsspec is required to inspect zipped SSL4EO shards") from exc
        zip_fs = fsspec.filesystem("zip", fo=str(path))
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
            mapper = zip_fs.get_mapper("")
        elif len(top_level_dirs) == 1:
            mapper = zip_fs.get_mapper(top_level_dirs[0])
        else:
            mapper = zip_fs.get_mapper(path.stem)
        return xr.open_zarr(mapper)
    return xr.open_zarr(str(path))


def resolve_shard_pairs(root_dir: Path, split: str, max_shards: int) -> list[tuple[Path, Path]]:
    s2_paths = resolve_modality_paths(root_dir, split, "S2L2A")
    s1_paths = resolve_modality_paths(root_dir, split, "S1GRD")
    if max_shards > 0:
        s2_paths = s2_paths[:max_shards]
        s1_paths = s1_paths[:max_shards]
    if not s2_paths or not s1_paths:
        raise FileNotFoundError(f"No SSL4EO shard pairs found under {root_dir / split}")
    if len(s2_paths) != len(s1_paths):
        raise ValueError(f"Mismatched shard counts for split={split}: {len(s2_paths)} S2 vs {len(s1_paths)} S1")

    pairs: list[tuple[Path, Path]] = []
    for s2_path, s1_path in zip(s2_paths, s1_paths):
        if shard_key(s2_path) != shard_key(s1_path):
            raise ValueError(f"Mismatched shard pair names: {s2_path.name} vs {s1_path.name}")
        pairs.append((s2_path, s1_path))
    return pairs


def count_shard_samples(xr: Any, s2_path: Path, s1_path: Path) -> dict[str, Any]:
    ds_s2 = open_zarr_target(s2_path)
    ds_s1 = open_zarr_target(s1_path)
    try:
        if "sample" in ds_s2.dims and int(ds_s2.sizes.get("sample", 0)) > 1:
            s2_samples = np.asarray(ds_s2.coords["sample"].values)
            s1_samples = np.asarray(ds_s1.coords["sample"].values)
            if not np.array_equal(s2_samples, s1_samples):
                raise ValueError(f"Sample coordinates do not align for {s2_path.name} and {s1_path.name}")
            sample_count = int(len(s2_samples))
            first_sample_id = str(s2_samples[0]) if sample_count else None
            last_sample_id = str(s2_samples[-1]) if sample_count else None
        else:
            sample_coord = ds_s2.coords.get("sample")
            sample_count = 1
            sample_id = str(sample_coord.item()) if sample_coord is not None else s2_path.stem
            first_sample_id = sample_id
            last_sample_id = sample_id
    finally:
        close_s2 = getattr(ds_s2, "close", None)
        if callable(close_s2):
            close_s2()
        close_s1 = getattr(ds_s1, "close", None)
        if callable(close_s1):
            close_s1()

    return {
        "shard_name": s2_path.name,
        "sample_count": sample_count,
        "first_sample_id": first_sample_id,
        "last_sample_id": last_sample_id,
    }


def apply_limits(total_samples: int, sample_offset: int, max_samples: int) -> dict[str, int]:
    remaining = max(0, total_samples - max(0, sample_offset))
    capped = remaining if max_samples <= 0 else min(remaining, max_samples)
    return {
        "sample_offset": int(max(0, sample_offset)),
        "max_samples": int(max_samples),
        "samples_after_offset": int(remaining),
        "samples_after_limits": int(capped),
    }


def summarize_split(
    root_dir: Path,
    split: str,
    max_shards: int,
    sample_offset: int,
    max_samples: int,
) -> dict[str, Any]:
    xr = import_xarray()
    shard_pairs = resolve_shard_pairs(root_dir, split, max_shards=max_shards)
    shard_summaries = [count_shard_samples(xr, s2_path, s1_path) for s2_path, s1_path in shard_pairs]
    total_samples = int(sum(item["sample_count"] for item in shard_summaries))
    one_per_shard = int(len(shard_summaries))
    limited = apply_limits(total_samples, sample_offset=sample_offset, max_samples=max_samples)
    top_multi_sample = sorted(
        [item for item in shard_summaries if int(item["sample_count"]) > 1],
        key=lambda item: int(item["sample_count"]),
        reverse=True,
    )[:10]

    return {
        "split": split,
        "root_dir": str(root_dir.resolve()),
        "shard_count": one_per_shard,
        "sample_count_real": total_samples,
        "sample_count_if_single_sample_per_shard": one_per_shard,
        "scale_factor_vs_single_sample_per_shard": (
            round(total_samples / one_per_shard, 4) if one_per_shard > 0 else None
        ),
        **limited,
        "top_multi_sample_shards": top_multi_sample,
    }


def main() -> None:
    args = parse_args()
    summaries = [
        summarize_split(
            root_dir=args.root_dir,
            split=split,
            max_shards=args.max_shards,
            sample_offset=args.sample_offset,
            max_samples=args.max_samples,
        )
        for split in args.splits
    ]
    payload = {
        "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "dataset": "ssl4eo_sample_count",
        "root_dir": str(args.root_dir.resolve()),
        "splits": summaries,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
