#!/usr/bin/env python3
"""Download and extract paired SSL4EO Zarr shard tarballs into a local corpus."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
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
    def __init__(self, base_uri: str):
        if not base_uri.startswith("gs://"):
            raise ValueError(f"GCS URI must start with gs://, got: {base_uri}")
        try:
            import gcsfs
        except ImportError as exc:
            raise RuntimeError("gcsfs is required when --gcs-uri is used") from exc
        self.base_uri = base_uri.rstrip("/")
        self.fs = gcsfs.GCSFileSystem()

    def object_uri(self, relative_path: Path) -> str:
        return f"{self.base_uri}/{relative_path.as_posix().lstrip('/')}"

    def exists(self, relative_path: Path) -> bool:
        return bool(self.fs.exists(self.object_uri(relative_path)))

    def write_bytes(self, relative_path: Path, payload: bytes) -> str:
        object_uri = self.object_uri(relative_path)
        with self.fs.open(object_uri, "wb") as handle:
            handle.write(payload)
        return object_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="embed2scale/SSL4EO-S12-v1.1")
    parser.add_argument("--root-dir", type=Path, default=Path("data/raw/ssl4eo_zarr_minimal"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ssl4eo_zarr_downloads"))
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--gcs-uri", type=str, default=None)
    parser.add_argument("--skip-local-copy", action="store_true")
    parser.add_argument("--keep-tars", action="store_true")
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

    payload = source_path.read_bytes()
    if local_path is not None and not local_exists:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(payload)
        wrote_local = True
    if gcs_mirror is not None and not gcs_exists:
        gcs_mirror.write_bytes(relative_path, payload)
        wrote_gcs = True

    return {
        "local_path": str(local_path.resolve()) if local_path is not None else None,
        "gcs_uri": gcs_mirror.object_uri(relative_path) if gcs_mirror is not None else None,
        "wrote_local": wrote_local,
        "wrote_gcs": wrote_gcs,
    }


def ensure_split_lists(
    repo_id: str,
    root_dir: Path,
    cache_dir: Path,
    *,
    write_local_copy: bool,
    gcs_mirror: GCSMirror | None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    split_dir = root_dir / "splits"
    for split_name in ("train", "val"):
        filename = f"splits/ssl4eos12_{split_name}.txt"
        relative_path = Path("splits") / Path(filename).name
        try:
            downloaded = Path(hf_hub_download(repo_id, repo_type="dataset", filename=filename, local_dir=str(cache_dir)))
        except EntryNotFoundError:
            existing_local = split_dir / Path(filename).name if write_local_copy else None
            gcs_exists = gcs_mirror.exists(relative_path) if gcs_mirror is not None else False
            if (existing_local is not None and existing_local.exists()) or gcs_exists:
                outputs.append(
                    {
                        "local_path": str(existing_local.resolve()) if existing_local is not None and existing_local.exists() else None,
                        "gcs_uri": gcs_mirror.object_uri(relative_path) if gcs_exists and gcs_mirror is not None else None,
                        "wrote_local": False,
                        "wrote_gcs": False,
                    }
                )
            continue
        outputs.append(
            copy_to_targets(
                downloaded,
                relative_path=relative_path,
                local_path=(split_dir / downloaded.name) if write_local_copy else None,
                gcs_mirror=gcs_mirror,
            )
        )
    return outputs


def shard_tar_name(shard_id: int) -> str:
    return f"ssl4eos12_shard_{int(shard_id):06d}.tar"


def safe_extract_tar(
    tar_path: Path,
    destination_dir: Path,
    *,
    split: str,
    modality: str,
    write_local_copy: bool,
    gcs_mirror: GCSMirror | None,
) -> dict[str, int]:
    extracted = 0
    skipped = 0
    uploaded_gcs = 0
    skipped_existing_gcs = 0
    if write_local_copy:
        destination_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            if not member_name.endswith(".zarr.zip"):
                continue
            relative_path = Path(split) / modality / member_name
            target_path = destination_dir / member_name if write_local_copy else None
            local_exists = target_path is not None and target_path.exists()
            gcs_exists = gcs_mirror.exists(relative_path) if gcs_mirror is not None else False
            if local_exists and (gcs_mirror is None or gcs_exists):
                skipped += 1
                continue
            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                skipped += 1
                continue
            payload = extracted_file.read()
            if target_path is not None:
                if local_exists:
                    skipped += 1
                else:
                    target_path.write_bytes(payload)
                    extracted += 1
            if gcs_mirror is not None:
                if gcs_exists:
                    skipped_existing_gcs += 1
                else:
                    gcs_mirror.write_bytes(relative_path, payload)
                    uploaded_gcs += 1
    return {
        "extracted_files": extracted,
        "skipped_existing_files": skipped,
        "uploaded_gcs_files": uploaded_gcs,
        "skipped_existing_gcs_files": skipped_existing_gcs,
    }


def main() -> None:
    load_env_file(REPO_ROOT / ".env")
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    gcs_uri = args.gcs_uri or os.getenv("GCS_DATA_URI") or os.getenv("EWM_GCS_DATA_URI")
    if args.skip_local_copy and not gcs_uri:
        raise ValueError("--skip-local-copy requires --gcs-uri or GCS_DATA_URI in .env")
    write_local_copy = not args.skip_local_copy
    if write_local_copy:
        args.root_dir.mkdir(parents=True, exist_ok=True)
    gcs_mirror = GCSMirror(gcs_uri) if gcs_uri else None

    split_lists = ensure_split_lists(
        args.repo_id,
        args.root_dir,
        args.cache_dir,
        write_local_copy=write_local_copy,
        gcs_mirror=gcs_mirror,
    )
    summary: dict[str, object] = {
        "repo_id": args.repo_id,
        "root_dir": str(args.root_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "gcs_uri": gcs_uri,
        "write_local_copy": write_local_copy,
        "split": args.split,
        "shards": [int(value) for value in args.shards],
        "modalities": list(args.modalities),
        "split_lists": split_lists,
        "downloads": [],
    }

    for modality in args.modalities:
        modality_dir = args.root_dir / args.split / modality
        for shard_id in args.shards:
            tar_name = shard_tar_name(shard_id)
            repo_file = f"{args.split}/{modality}/{tar_name}"
            tar_path = Path(
                hf_hub_download(
                    args.repo_id,
                    repo_type="dataset",
                    filename=repo_file,
                    local_dir=str(args.cache_dir),
                )
            )
            extract_summary = safe_extract_tar(
                tar_path,
                modality_dir,
                split=args.split,
                modality=modality,
                write_local_copy=write_local_copy,
                gcs_mirror=gcs_mirror,
            )
            record = {
                "modality": modality,
                "shard_id": int(shard_id),
                "tar_path": str(tar_path.resolve()),
                "destination_dir": str(modality_dir.resolve()) if write_local_copy else None,
                "gcs_destination_prefix": gcs_mirror.object_uri(Path(args.split) / modality) if gcs_mirror is not None else None,
            }
            record.update(extract_summary)
            summary["downloads"].append(record)
            if not args.keep_tars and tar_path.exists():
                tar_path.unlink()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
