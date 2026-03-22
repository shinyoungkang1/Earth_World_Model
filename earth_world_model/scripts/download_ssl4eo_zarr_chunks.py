#!/usr/bin/env python3
"""Download official SSL4EO Zarr chunk files into a local corpus and/or GCS."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


DEFAULT_MODALITIES = ("S1GRD", "S2L2A")
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class GCSMirror:
    def __init__(self, base_uri: str, *, quiet: bool = False, backend: str = "auto"):
        if not base_uri.startswith("gs://"):
            raise ValueError(f"GCS URI must start with gs://, got: {base_uri}")
        self.base_uri = base_uri.rstrip("/")
        self.quiet = bool(quiet)
        self.backend = str(backend)
        self.fs = None
        if self.backend not in {"auto", "gcloud", "gcsfs"}:
            raise ValueError(f"Unsupported GCS backend: {self.backend}")
        if self.backend == "auto":
            self.backend = "gcloud" if shutil.which("gcloud") else "gcsfs"
        if self.backend == "gcsfs":
            try:
                import gcsfs
            except ImportError as exc:
                raise RuntimeError("gcsfs backend requested, but gcsfs is not installed") from exc
            self.fs = gcsfs.GCSFileSystem()
        elif shutil.which("gcloud") is None:
            raise RuntimeError("gcloud backend requested, but gcloud is not available on PATH")

    def object_uri(self, relative_path: Path) -> str:
        return f"{self.base_uri}/{relative_path.as_posix().lstrip('/')}"

    def exists(self, relative_path: Path) -> bool:
        object_uri = self.object_uri(relative_path)
        if self.backend == "gcsfs":
            return bool(self.fs.isfile(object_uri))
        result = subprocess.run(
            ["gcloud", "storage", "objects", "describe", object_uri],
            stdout=subprocess.DEVNULL if self.quiet else None,
            stderr=subprocess.DEVNULL if self.quiet else None,
            check=False,
        )
        return result.returncode == 0

    def copy_file(self, source_path: Path, relative_path: Path) -> str:
        object_uri = self.object_uri(relative_path)
        if self.backend == "gcsfs":
            with source_path.open("rb") as handle_in, self.fs.open(object_uri, "wb") as handle_out:
                handle_out.write(handle_in.read())
            return object_uri
        subprocess.run(
            ["gcloud", "storage", "cp", "--no-clobber", str(source_path), object_uri],
            stdout=subprocess.DEVNULL if self.quiet else None,
            stderr=subprocess.DEVNULL if self.quiet else None,
            check=True,
        )
        return object_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="embed2scale/SSL4EO-S12-v1.1-Zarr")
    parser.add_argument("--root-dir", type=Path, default=Path("data/raw/ssl4eo_zarr_50k"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ssl4eo_zarr_downloads"))
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--gcs-uri", type=str, default=None)
    parser.add_argument("--gcs-backend", choices=["auto", "gcloud", "gcsfs"], default="auto")
    parser.add_argument("--skip-local-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=1)
    return parser.parse_args()


def copy_to_targets(
    source_path: Path,
    *,
    relative_path: Path,
    local_path: Path | None,
    gcs_mirror: GCSMirror | None,
) -> dict[str, Any]:
    wrote_local = False
    wrote_gcs = False
    local_exists = local_path is not None and local_path.exists()
    gcs_exists = gcs_mirror.exists(relative_path) if gcs_mirror is not None else False
    if local_exists and (gcs_mirror is None or gcs_exists):
        return {
            "local_path": str(local_path.resolve()) if local_path is not None else None,
            "gcs_uri": gcs_mirror.object_uri(relative_path) if gcs_mirror is not None else None,
            "wrote_local": False,
            "wrote_gcs": False,
        }

    if local_path is not None and not local_exists:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(source_path.read_bytes())
        wrote_local = True
    if gcs_mirror is not None and not gcs_exists:
        gcs_mirror.copy_file(source_path, relative_path)
        wrote_gcs = True

    return {
        "local_path": str(local_path.resolve()) if local_path is not None else None,
        "gcs_uri": gcs_mirror.object_uri(relative_path) if gcs_mirror is not None else None,
        "wrote_local": wrote_local,
        "wrote_gcs": wrote_gcs,
    }


def ensure_split_list(
    repo_id: str,
    split: str,
    root_dir: Path,
    cache_dir: Path,
    *,
    write_local_copy: bool,
    gcs_mirror: GCSMirror | None,
) -> tuple[dict[str, Any], list[str]]:
    filename = f"splits/ssl4eos12_{split}.txt"
    relative_path = Path("splits") / Path(filename).name
    split_dir = root_dir / "splits"
    downloaded = Path(hf_hub_download(repo_id, repo_type="dataset", filename=filename, local_dir=str(cache_dir)))
    record = copy_to_targets(
        downloaded,
        relative_path=relative_path,
        local_path=(split_dir / downloaded.name) if write_local_copy else None,
        gcs_mirror=gcs_mirror,
    )
    lines = [line.strip() for line in downloaded.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return record, lines


def build_download_record(
    *,
    repo_id: str,
    cache_dir: Path,
    split: str,
    modality: str,
    filename: str,
    root_dir: Path,
    write_local_copy: bool,
    gcs_mirror: GCSMirror | None,
    dry_run: bool,
) -> dict[str, Any]:
    destination_dir = root_dir / split / modality
    repo_file = f"{split}/{modality}/{filename}"
    relative_path = Path(split) / modality / filename
    if dry_run:
        return {
            "modality": modality,
            "filename": filename,
            "repo_file": repo_file,
            "local_path": str((destination_dir / filename).resolve()) if write_local_copy else None,
            "gcs_uri": gcs_mirror.object_uri(relative_path) if gcs_mirror is not None else None,
            "wrote_local": False,
            "wrote_gcs": False,
        }
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id,
                repo_type="dataset",
                filename=repo_file,
                local_dir=str(cache_dir),
            )
        )
    except EntryNotFoundError as exc:
        raise FileNotFoundError(f"Missing dataset file in {repo_id}: {repo_file}") from exc
    result = copy_to_targets(
        downloaded,
        relative_path=relative_path,
        local_path=(destination_dir / filename) if write_local_copy else None,
        gcs_mirror=gcs_mirror,
    )
    return {
        "modality": modality,
        "filename": filename,
        "repo_file": repo_file,
        **result,
    }


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    args.parallel_workers = max(1, int(args.parallel_workers))
    if args.parallel_workers > 1:
        # Multiple concurrent hf_hub_download calls are noisy by default.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    gcs_uri = args.gcs_uri or os.getenv("GCS_DATA_URI") or os.getenv("EWM_GCS_DATA_URI")
    if args.skip_local_copy and not gcs_uri:
        raise ValueError("--skip-local-copy requires --gcs-uri or GCS_DATA_URI in .env")
    write_local_copy = not args.skip_local_copy
    if write_local_copy:
        args.root_dir.mkdir(parents=True, exist_ok=True)
    gcs_mirror = (
        GCSMirror(gcs_uri, quiet=args.parallel_workers > 1, backend=args.gcs_backend) if gcs_uri else None
    )

    split_list_record, split_filenames = ensure_split_list(
        args.repo_id,
        args.split,
        args.root_dir,
        args.cache_dir,
        write_local_copy=write_local_copy,
        gcs_mirror=gcs_mirror,
    )
    selected = split_filenames[max(0, args.offset) :]
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError(f"No filenames selected for split={args.split} offset={args.offset} limit={args.limit}")

    summary: dict[str, Any] = {
        "repo_id": args.repo_id,
        "root_dir": str(args.root_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "gcs_uri": gcs_uri,
        "gcs_backend": args.gcs_backend,
        "write_local_copy": write_local_copy,
        "dry_run": bool(args.dry_run),
        "split": args.split,
        "offset": int(args.offset),
        "limit": int(args.limit),
        "parallel_workers": int(args.parallel_workers),
        "selected_count": len(selected),
        "modalities": list(args.modalities),
        "split_list": split_list_record,
        "first_selected": selected[:3],
        "last_selected": selected[-3:],
        "downloads": [],
    }

    tasks = [(modality, filename) for filename in selected for modality in args.modalities]
    if args.parallel_workers == 1:
        for modality, filename in tasks:
            summary["downloads"].append(
                build_download_record(
                    repo_id=args.repo_id,
                    cache_dir=args.cache_dir,
                    split=args.split,
                    modality=modality,
                    filename=filename,
                    root_dir=args.root_dir,
                    write_local_copy=write_local_copy,
                    gcs_mirror=gcs_mirror,
                    dry_run=bool(args.dry_run),
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            futures = [
                executor.submit(
                    build_download_record,
                    repo_id=args.repo_id,
                    cache_dir=args.cache_dir,
                    split=args.split,
                    modality=modality,
                    filename=filename,
                    root_dir=args.root_dir,
                    write_local_copy=write_local_copy,
                    gcs_mirror=gcs_mirror,
                    dry_run=bool(args.dry_run),
                )
                for modality, filename in tasks
            ]
            total = len(futures)
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                summary["downloads"].append(record)
                print(
                    f"[{completed}/{total}] "
                    f"{record['modality']} {record['filename']} "
                    f"local={record['wrote_local']} gcs={record['wrote_gcs']}"
                )

    summary["downloads"].sort(key=lambda item: (item["modality"], item["filename"]))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
