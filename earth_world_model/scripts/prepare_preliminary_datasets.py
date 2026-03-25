#!/usr/bin/env python3
"""Prepare preliminary yearly + SSL4EO pilot datasets on a CPU prep VM."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from google.cloud import storage
from huggingface_hub import hf_hub_download
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = REPO_ROOT / "earth_world_model" / "scripts" / "process_earth_engine_exact_chip_exports.py"
DEFAULT_SSL4EO_REPO_ID = "embed2scale/SSL4EO-S12-v1.1-Zarr"
DEFAULT_YEARLY_RAW_ROOT = "/home/shin/scratch/ee_interactive_100_1000_v1/raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare yearly_400 and ssl4eo_400 pilot datasets.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory on the mounted data disk.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ewm_prelim_400_prep"))
    parser.add_argument("--yearly-name", default="yearly_400")
    parser.add_argument("--ssl4eo-name", default="ssl4eo_400")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument(
        "--yearly-raw-root",
        dest="yearly_raw_roots",
        action="append",
        type=str,
        default=None,
        help="Completed yearly raw sample root. Can be passed multiple times.",
    )
    parser.add_argument("--yearly-train-count", type=int, default=400)
    parser.add_argument("--yearly-val-count", type=int, default=32)
    parser.add_argument("--yearly-workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    parser.add_argument("--ssl4eo-repo-id", default=DEFAULT_SSL4EO_REPO_ID)
    parser.add_argument("--ssl4eo-cache-dir", type=Path, default=None)
    parser.add_argument("--ssl4eo-workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    parser.add_argument("--ssl4eo-train-count", type=int, default=400)
    parser.add_argument("--ssl4eo-val-count", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_converter_module():
    spec = importlib.util.spec_from_file_location("ee_exact_chip_converter", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load converter module from {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EE_CONVERTER = load_converter_module()


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def is_gcs_uri(value: str) -> bool:
    return str(value).startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not is_gcs_uri(uri):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    bucket_and_path = uri[5:]
    if "/" not in bucket_and_path:
        return bucket_and_path, ""
    bucket, prefix = bucket_and_path.split("/", 1)
    return bucket, prefix.rstrip("/")


def gcs_join(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix.lstrip("/")
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


def ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing directory without --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    combined.to_parquet(path, index=False)


def load_existing_sample_ids(index_path: Path, sequences_dir: Path) -> set[str]:
    sample_ids: set[str] = set()
    if index_path.exists():
        df = pd.read_parquet(index_path, columns=["sample_id"])
        sample_ids.update(str(value) for value in df["sample_id"].tolist())
    if sequences_dir.exists():
        for sequence_path in sequences_dir.glob("*.npz"):
            stem = sequence_path.stem
            if stem.startswith("ee__") and "__" in stem[4:]:
                sample_part = stem[4:]
                sample_id = sample_part.rsplit("__", 1)[0]
                if sample_id:
                    sample_ids.add(sample_id)
    return sample_ids


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sample_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sample_is_complete(sample_dir: Path) -> bool:
    payload = load_sample_payload(sample_dir / f"{sample_dir.name}.json")
    if payload is None:
        return False
    chunk_results = payload.get("chunk_results") or []
    if not chunk_results:
        return False
    return all(int(item.get("status_code", 0)) == 200 for item in chunk_results)


def gcs_sample_is_complete(
    client: storage.Client,
    *,
    bucket_name: str,
    prefix: str,
    sample_id: str,
) -> tuple[bool, dict[str, Any] | None]:
    bucket = client.bucket(bucket_name)
    json_blob = bucket.blob(gcs_join(prefix, f"{sample_id}/{sample_id}.json"))
    if not json_blob.exists():
        return False, None
    try:
        payload = json.loads(json_blob.download_as_text())
    except Exception:
        return False, None
    chunk_results = payload.get("chunk_results") or []
    if not chunk_results:
        return False, payload
    return all(int(item.get("status_code", 0)) == 200 for item in chunk_results), payload


def discover_completed_yearly_samples(raw_roots: list[Path | str]) -> list[dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    gcs_client: storage.Client | None = None
    for raw_root in raw_roots:
        raw_root_str = str(raw_root)
        if is_gcs_uri(raw_root_str):
            bucket_name, prefix = parse_gcs_uri(raw_root_str)
            if gcs_client is None:
                gcs_client = storage.Client()
            sample_prefix_iter = gcs_client.list_blobs(bucket_name, prefix=gcs_join(prefix, ""), delimiter="/")
            list(sample_prefix_iter)
            for sample_prefix in sorted(sample_prefix_iter.prefixes):
                sample_id = Path(sample_prefix.rstrip("/")).name
                if sample_id in discovered:
                    continue
                is_complete, _payload = gcs_sample_is_complete(
                    gcs_client,
                    bucket_name=bucket_name,
                    prefix=prefix,
                    sample_id=sample_id,
                )
                if not is_complete:
                    continue
                tif_count = sum(
                    1
                    for blob in gcs_client.list_blobs(bucket_name, prefix=gcs_join(prefix, f"{sample_id}/"))
                    if blob.name.endswith(".tif")
                )
                if tif_count <= 0:
                    continue
                discovered[sample_id] = {
                    "sample_id": sample_id,
                    "source_dir": f"gs://{bucket_name}/{gcs_join(prefix, sample_id)}",
                    "source_root": f"gs://{bucket_name}/{prefix}" if prefix else f"gs://{bucket_name}",
                    "tif_count": int(tif_count),
                }
            continue

        raw_root_path = Path(raw_root_str)
        if not raw_root_path.exists():
            continue
        for sample_dir in sorted(path for path in raw_root_path.iterdir() if path.is_dir()):
            sample_id = sample_dir.name
            if sample_id in discovered:
                continue
            if not sample_is_complete(sample_dir):
                continue
            tif_paths = sorted(sample_dir.glob("*.tif"))
            if not tif_paths:
                continue
            discovered[sample_id] = {
                "sample_id": sample_id,
                "source_dir": str(sample_dir.resolve()),
                "source_root": str(raw_root_path.resolve()),
                "tif_count": len(tif_paths),
            }
    return [discovered[key] for key in sorted(discovered)]


def stage_yearly_record(
    *,
    record: dict[str, Any],
    staging_dir: Path,
    overwrite: bool,
    gcs_client: storage.Client | None,
) -> Path:
    ensure_dir(staging_dir)
    sample_id = str(record["sample_id"])
    target_dir = staging_dir / sample_id
    expected_tif_count = int(record.get("tif_count", 0))

    if target_dir.exists():
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            tif_count = len(list(target_dir.glob("*.tif")))
            json_ok = (target_dir / f"{sample_id}.json").exists() or not is_gcs_uri(str(record["source_dir"]))
            if tif_count >= expected_tif_count and json_ok:
                return target_dir
            shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = str(record["source_dir"])
    if is_gcs_uri(source_dir):
        if gcs_client is None:
            raise RuntimeError("gcs_client is required for gs:// yearly sources")
        bucket_name, prefix = parse_gcs_uri(source_dir)
        for blob in sorted(gcs_client.list_blobs(bucket_name, prefix=gcs_join(prefix, "")), key=lambda item: item.name):
            if blob.name.endswith("/"):
                continue
            filename = Path(blob.name).name
            if not filename.endswith(".tif") and not filename.endswith(".json"):
                continue
            blob.download_to_filename(str(target_dir / filename))
        return target_dir

    source_path = Path(source_dir)
    for tif_path in sorted(source_path.glob("*.tif")):
        target_path = target_dir / tif_path.name
        if not target_path.exists():
            target_path.symlink_to(tif_path)
    json_path = source_path / f"{sample_id}.json"
    if json_path.exists():
        target_json = target_dir / json_path.name
        if not target_json.exists():
            target_json.symlink_to(json_path)
    return target_dir


def process_yearly_record(
    *,
    record: dict[str, Any],
    staging_dir: Path,
    output_dir: Path,
    year: int,
    chip_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_dir = str(record["source_dir"])
    gcs_client = storage.Client() if is_gcs_uri(source_dir) else None
    staged_sample_dir = stage_yearly_record(
        record=record,
        staging_dir=staging_dir,
        overwrite=overwrite,
        gcs_client=gcs_client,
    )
    sample_id = str(record["sample_id"])
    tif_paths = sorted(staged_sample_dir.glob("*.tif"))
    yearly_record = EE_CONVERTER.read_ee_tif_records(
        sample_id=sample_id,
        tif_paths=tif_paths,
        manifests={},
        locations={},
        year=int(year),
        chip_size=int(chip_size),
    )
    npz_rows = EE_CONVERTER.write_npz_records(output_dir, [yearly_record])
    return {
        "sample_id": sample_id,
        "npz_rows": npz_rows,
        "failed_rows": [failure for failure in yearly_record.get("failed_chunks", [])],
        "normalization_rows": [event for event in yearly_record.get("normalization_events", [])],
        "staged_sample_dir": str(staged_sample_dir),
    }


def write_yearly_progress(
    *,
    progress_path: Path,
    split_name: str,
    target_count: int,
    completed_count: int,
    sample_id: str | None,
) -> None:
    payload = {
        "split": split_name,
        "target_count": int(target_count),
        "completed_count": int(completed_count),
        "last_completed_sample_id": sample_id,
    }
    write_json(progress_path, payload)


def convert_yearly_split_incremental(
    *,
    records: list[dict[str, Any]],
    split_name: str,
    staging_dir: Path,
    output_dir: Path,
    year: int,
    chip_size: int,
    yearly_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    ensure_dir(staging_dir)
    sequences_dir = output_dir / "sequences"
    ensure_dir(sequences_dir)

    index_path = output_dir / "dense_temporal_index.parquet"
    failed_path = output_dir / "failed_chunks.parquet"
    normalization_path = output_dir / "normalization_events.parquet"
    progress_path = output_dir / "progress.json"

    completed_sample_ids = load_existing_sample_ids(index_path, sequences_dir)
    processed = 0
    skipped = 0
    pending_records: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in completed_sample_ids:
            skipped += 1
            staged_existing = staging_dir / sample_id
            if staged_existing.exists():
                shutil.rmtree(staged_existing)
            write_yearly_progress(
                progress_path=progress_path,
                split_name=split_name,
                target_count=len(records),
                completed_count=len(completed_sample_ids),
                sample_id=sample_id,
            )
            continue
        pending_records.append(record)

    worker_count = max(1, int(yearly_workers))
    if worker_count == 1:
        for record in pending_records:
            result = process_yearly_record(
                record=record,
                staging_dir=staging_dir,
                output_dir=output_dir,
                year=year,
                chip_size=chip_size,
                overwrite=overwrite,
            )
            append_parquet_rows(index_path, result["npz_rows"])
            append_parquet_rows(failed_path, result["failed_rows"])
            append_parquet_rows(normalization_path, result["normalization_rows"])
            completed_sample_ids.add(str(result["sample_id"]))
            processed += 1
            shutil.rmtree(Path(result["staged_sample_dir"]), ignore_errors=True)
            write_yearly_progress(
                progress_path=progress_path,
                split_name=split_name,
                target_count=len(records),
                completed_count=len(completed_sample_ids),
                sample_id=str(result["sample_id"]),
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_record = {
                executor.submit(
                    process_yearly_record,
                    record=record,
                    staging_dir=staging_dir,
                    output_dir=output_dir,
                    year=year,
                    chip_size=chip_size,
                    overwrite=overwrite,
                ): record
                for record in pending_records
            }
            for future in concurrent.futures.as_completed(future_to_record):
                record = future_to_record[future]
                sample_id = str(record["sample_id"])
                result = future.result()
                append_parquet_rows(index_path, result["npz_rows"])
                append_parquet_rows(failed_path, result["failed_rows"])
                append_parquet_rows(normalization_path, result["normalization_rows"])
                completed_sample_ids.add(sample_id)
                processed += 1
                shutil.rmtree(Path(result["staged_sample_dir"]), ignore_errors=True)
                write_yearly_progress(
                    progress_path=progress_path,
                    split_name=split_name,
                    target_count=len(records),
                    completed_count=len(completed_sample_ids),
                    sample_id=sample_id,
                )

    summary = {
        "split": split_name,
        "target_count": int(len(records)),
        "completed_count": int(len(completed_sample_ids)),
        "processed_this_run": int(processed),
        "skipped_existing": int(skipped),
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "progress_path": str(progress_path),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def write_yearly_selection(selection_path: Path, rows: list[dict[str, Any]]) -> None:
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_count": len(rows),
        "samples": rows,
    }
    selection_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(selection_path.with_suffix(".parquet"), index=False)


def ssl4eo_shard_key(path: Path) -> str:
    name = path.name
    if name.endswith(".zarr.zip"):
        return name[: -len(".zarr.zip")]
    if name.endswith(".zarr"):
        return name[: -len(".zarr")]
    return path.stem


def ssl4eo_target_pair_paths(*, target_root: Path, split: str, filename: str) -> tuple[Path, Path]:
    return (
        target_root / split / "S2L2A" / filename,
        target_root / split / "S1GRD" / filename,
    )


def ssl4eo_pair_exists(*, target_root: Path, split: str, filename: str) -> bool:
    s2_path, s1_path = ssl4eo_target_pair_paths(target_root=target_root, split=split, filename=filename)
    return s2_path.exists() and s1_path.exists()


def load_ssl4eo_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    rows = df.to_dict(orient="records")
    return {str(row["filename"]): dict(row) for row in rows}


def write_ssl4eo_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    pd.DataFrame(ordered).to_parquet(path, index=False)


def write_ssl4eo_progress(
    *,
    progress_path: Path,
    split: str,
    target_samples: int,
    selected_shard_count: int,
    cumulative_samples: int,
    last_filename: str | None,
) -> None:
    payload = {
        "split": split,
        "target_samples": int(target_samples),
        "selected_shard_count": int(selected_shard_count),
        "cumulative_samples": int(cumulative_samples),
        "last_filename": last_filename,
    }
    write_json(progress_path, payload)


def open_ssl4eo_target(path: Path):
    import xarray as xr

    if path.name.endswith(".zarr.zip"):
        import fsspec

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
            return xr.open_zarr(zip_fs.get_mapper(""))
        if len(top_level_dirs) == 1:
            return xr.open_zarr(zip_fs.get_mapper(top_level_dirs[0]))
        return xr.open_zarr(zip_fs.get_mapper(path.stem))
    return xr.open_zarr(str(path))


def count_ssl4eo_samples_for_pair(s2_path: Path, s1_path: Path) -> dict[str, Any]:
    ds_s2 = open_ssl4eo_target(s2_path)
    ds_s1 = open_ssl4eo_target(s1_path)
    try:
        if "sample" in ds_s2.dims and int(ds_s2.sizes.get("sample", 0)) > 1:
            s2_samples = np.asarray(ds_s2.coords["sample"].values)
            s1_samples = np.asarray(ds_s1.coords["sample"].values)
            if not np.array_equal(s2_samples, s1_samples):
                raise ValueError(f"Sample coordinates do not align for {s2_path.name} and {s1_path.name}")
            sample_count = int(len(s2_samples))
        else:
            sample_count = 1
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
    }


def load_ssl4eo_split_list(repo_id: str, split: str, cache_dir: Path) -> list[str]:
    split_file = hf_hub_download(
        repo_id,
        repo_type="dataset",
        filename=f"splits/ssl4eos12_{split}.txt",
        local_dir=str(cache_dir),
    )
    return [
        line.strip()
        for line in Path(split_file).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def download_ssl4eo_shard_pair(
    *,
    repo_id: str,
    split: str,
    filename: str,
    cache_dir: Path,
    target_root: Path,
) -> tuple[Path, Path]:
    paths: dict[str, Path] = {}
    for modality in ("S2L2A", "S1GRD"):
        repo_file = f"{split}/{modality}/{filename}"
        downloaded = Path(
            hf_hub_download(
                repo_id,
                repo_type="dataset",
                filename=repo_file,
                local_dir=str(cache_dir),
            )
        )
        target_path = target_root / split / modality / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(downloaded, target_path)
        paths[modality] = target_path
    return paths["S2L2A"], paths["S1GRD"]


def materialize_ssl4eo_shard_pair(
    *,
    repo_id: str,
    split: str,
    filename: str,
    cache_dir: Path,
    target_root: Path,
) -> dict[str, Any]:
    s2_target, s1_target = ssl4eo_target_pair_paths(target_root=target_root, split=split, filename=filename)
    if not (s2_target.exists() and s1_target.exists()):
        s2_target, s1_target = download_ssl4eo_shard_pair(
            repo_id=repo_id,
            split=split,
            filename=filename,
            cache_dir=cache_dir,
            target_root=target_root,
        )
    stats = count_ssl4eo_samples_for_pair(s2_target, s1_target)
    return {
        "filename": filename,
        "sample_count": int(stats["sample_count"]),
        "shard_key": ssl4eo_shard_key(s2_target),
        "s2_path": str(s2_target),
        "s1_path": str(s1_target),
    }


def prepare_ssl4eo_split(
    *,
    repo_id: str,
    split: str,
    target_samples: int,
    output_root: Path,
    cache_dir: Path,
    ssl4eo_workers: int,
) -> dict[str, Any]:
    if target_samples <= 0:
        return {
            "split": split,
            "target_samples": 0,
            "sample_count_real": 0,
            "downloaded_shards": [],
        }
    filenames = load_ssl4eo_split_list(repo_id, split, cache_dir)
    manifest_path = output_root / f"{split}_shards.parquet"
    progress_path = output_root / f"{split}_progress.json"
    manifest_rows = load_ssl4eo_manifest(manifest_path)
    downloaded_rows: list[dict[str, Any]] = []
    cumulative = 0
    worker_count = max(1, int(ssl4eo_workers))
    idx = 0

    while idx < len(filenames) and cumulative < target_samples:
        batch: list[str] = []
        while idx < len(filenames) and len(batch) < worker_count and cumulative < target_samples:
            filename = filenames[idx]
            idx += 1
            existing = manifest_rows.get(filename)
            if existing and ssl4eo_pair_exists(target_root=output_root, split=split, filename=filename):
                cumulative += int(existing["sample_count"])
                downloaded_rows.append(
                    {
                        "filename": filename,
                        "sample_count": int(existing["sample_count"]),
                        "cumulative_samples": int(cumulative),
                        "shard_key": str(existing["shard_key"]),
                    }
                )
                write_ssl4eo_progress(
                    progress_path=progress_path,
                    split=split,
                    target_samples=target_samples,
                    selected_shard_count=len(downloaded_rows),
                    cumulative_samples=cumulative,
                    last_filename=filename,
                )
            else:
                batch.append(filename)

        if not batch:
            continue

        if worker_count == 1:
            batch_results = {
                filename: materialize_ssl4eo_shard_pair(
                    repo_id=repo_id,
                    split=split,
                    filename=filename,
                    cache_dir=cache_dir,
                    target_root=output_root,
                )
                for filename in batch
            }
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_filename = {
                    executor.submit(
                        materialize_ssl4eo_shard_pair,
                        repo_id=repo_id,
                        split=split,
                        filename=filename,
                        cache_dir=cache_dir,
                        target_root=output_root,
                    ): filename
                    for filename in batch
                }
                batch_results = {
                    future_to_filename[future]: future.result()
                    for future in concurrent.futures.as_completed(future_to_filename)
                }

        for filename in batch:
            result = batch_results[filename]
            manifest_rows[filename] = {
                "filename": filename,
                "sample_count": int(result["sample_count"]),
                "shard_key": str(result["shard_key"]),
                "s2_path": str(result["s2_path"]),
                "s1_path": str(result["s1_path"]),
            }
        write_ssl4eo_manifest(manifest_path, manifest_rows)

        for filename in batch:
            result = batch_results[filename]
            cumulative += int(result["sample_count"])
            downloaded_rows.append(
                {
                    "filename": filename,
                    "sample_count": int(result["sample_count"]),
                    "cumulative_samples": int(cumulative),
                    "shard_key": str(result["shard_key"]),
                }
            )
            write_ssl4eo_progress(
                progress_path=progress_path,
                split=split,
                target_samples=target_samples,
                selected_shard_count=len(downloaded_rows),
                cumulative_samples=cumulative,
                last_filename=filename,
            )
            if cumulative >= target_samples:
                break

    if cumulative < target_samples:
        raise RuntimeError(
            f"Unable to reach target_samples={target_samples} for split={split}; only collected {cumulative}"
        )
    return {
        "split": split,
        "target_samples": int(target_samples),
        "sample_count_real": int(cumulative),
        "downloaded_shard_count": int(len(downloaded_rows)),
        "manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "downloaded_shards": downloaded_rows,
    }


def main() -> int:
    args = parse_args()
    raw_roots = args.yearly_raw_roots or [DEFAULT_YEARLY_RAW_ROOT]
    output_root = args.output_root.resolve()
    work_dir = args.work_dir.resolve()
    ssl4eo_cache_dir = (
        args.ssl4eo_cache_dir.resolve()
        if args.ssl4eo_cache_dir is not None
        else (work_dir / "ssl4eo_zarr_downloads").resolve()
    )
    yearly_root = output_root / args.yearly_name
    ssl4eo_root = output_root / args.ssl4eo_name
    work_dir.mkdir(parents=True, exist_ok=True)
    ssl4eo_cache_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    completed = discover_completed_yearly_samples(raw_roots)
    required_yearly = int(args.yearly_train_count) + int(args.yearly_val_count)
    if len(completed) < required_yearly:
        raise RuntimeError(
            f"Need {required_yearly} completed yearly samples, found {len(completed)} across {[str(p) for p in raw_roots]}"
        )

    yearly_train = completed[: int(args.yearly_train_count)]
    yearly_val = completed[int(args.yearly_train_count) : required_yearly]
    yearly_stage_root = work_dir / "yearly_staging"
    train_stage = yearly_stage_root / "train"
    val_stage = yearly_stage_root / "val"
    write_yearly_selection(yearly_root / "train_selection.json", yearly_train)
    write_yearly_selection(yearly_root / "val_selection.json", yearly_val)
    yearly_train_summary = convert_yearly_split_incremental(
        records=yearly_train,
        split_name="train",
        staging_dir=train_stage,
        output_dir=yearly_root / "train",
        year=int(args.year),
        chip_size=int(args.chip_size),
        yearly_workers=int(args.yearly_workers),
        overwrite=args.overwrite,
    )
    yearly_val_summary = convert_yearly_split_incremental(
        records=yearly_val,
        split_name="val",
        staging_dir=val_stage,
        output_dir=yearly_root / "val",
        year=int(args.year),
        chip_size=int(args.chip_size),
        yearly_workers=int(args.yearly_workers),
        overwrite=args.overwrite,
    )

    ssl4eo_train_summary = prepare_ssl4eo_split(
        repo_id=args.ssl4eo_repo_id,
        split="train",
        target_samples=int(args.ssl4eo_train_count),
        output_root=ssl4eo_root,
        cache_dir=ssl4eo_cache_dir,
        ssl4eo_workers=int(args.ssl4eo_workers),
    )
    ssl4eo_val_summary = prepare_ssl4eo_split(
        repo_id=args.ssl4eo_repo_id,
        split="val",
        target_samples=int(args.ssl4eo_val_count),
        output_root=ssl4eo_root,
        cache_dir=ssl4eo_cache_dir,
        ssl4eo_workers=int(args.ssl4eo_workers),
    )

    summary = {
        "output_root": str(output_root),
        "work_dir": str(work_dir),
        "yearly": {
            "raw_roots": [
                str(Path(path).resolve()) if not is_gcs_uri(str(path)) else str(path)
                for path in raw_roots
            ],
            "available_completed_count": int(len(completed)),
            "train_count": int(len(yearly_train)),
            "val_count": int(len(yearly_val)),
            "train_output": str((yearly_root / "train").resolve()),
            "val_output": str((yearly_root / "val").resolve()),
            "train_summary": yearly_train_summary,
            "val_summary": yearly_val_summary,
        },
        "ssl4eo": {
            "repo_id": args.ssl4eo_repo_id,
            "cache_dir": str(ssl4eo_cache_dir),
            "output_root": str(ssl4eo_root.resolve()),
            "train": ssl4eo_train_summary,
            "val": ssl4eo_val_summary,
        },
    }
    (output_root / "preliminary_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
