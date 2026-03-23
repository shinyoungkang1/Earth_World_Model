#!/usr/bin/env python3
"""Materialize SSL4EO *.zarr.zip shards into plain .zarr directories."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--modalities", nargs="+", default=["S2L2A", "S1GRD"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    return parser.parse_args()


def resolve_zip_paths(root_dir: Path, splits: list[str], modalities: list[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        for modality in modalities:
            paths.extend(sorted((root_dir / split / modality).glob("*.zarr.zip")))
    return paths


def materialize_one(zip_path: Path, *, delete_zip_after_extract: bool) -> dict[str, object]:
    dest_dir = zip_path.with_suffix("")
    if (dest_dir / ".zgroup").exists():
        if delete_zip_after_extract and zip_path.exists():
            zip_path.unlink()
        return {
            "status": "existing",
            "zip_path": str(zip_path),
            "dest_dir": str(dest_dir),
            "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else None,
        }

    tmp_dir = dest_dir.with_name(dest_dir.name + ".partial")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    if not (tmp_dir / ".zgroup").exists():
        raise RuntimeError(f"Extracted shard is missing .zgroup: {zip_path}")
    tmp_dir.rename(dest_dir)
    if delete_zip_after_extract:
        zip_path.unlink()

    return {
        "status": "materialized",
        "zip_path": str(zip_path),
        "dest_dir": str(dest_dir),
        "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else None,
        "duration_sec": round(time.time() - started_at, 4),
    }


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    zip_paths = resolve_zip_paths(root_dir, list(args.splits), list(args.modalities))
    if args.limit > 0:
        zip_paths = zip_paths[: int(args.limit)]
    if not zip_paths:
        raise FileNotFoundError(f"No *.zarr.zip shards found under {root_dir}")

    started_at = time.time()
    payloads: list[dict[str, object]] = []
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                materialize_one,
                path,
                delete_zip_after_extract=bool(args.delete_zip_after_extract),
            ): path
            for path in zip_paths
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            payloads.append(result)
            completed += 1
            if completed <= 5 or completed % 25 == 0 or completed == len(zip_paths):
                print(
                    json.dumps(
                        {
                            "event": "materialize_progress",
                            "completed": completed,
                            "total": len(zip_paths),
                            **result,
                        }
                    ),
                    flush=True,
                )

    summary = {
        "event": "materialize_complete",
        "root_dir": str(root_dir),
        "splits": list(args.splits),
        "modalities": list(args.modalities),
        "workers": int(args.workers),
        "delete_zip_after_extract": bool(args.delete_zip_after_extract),
        "total_shards": len(zip_paths),
        "materialized_count": sum(1 for item in payloads if item["status"] == "materialized"),
        "existing_count": sum(1 for item in payloads if item["status"] == "existing"),
        "duration_sec": round(time.time() - started_at, 4),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
