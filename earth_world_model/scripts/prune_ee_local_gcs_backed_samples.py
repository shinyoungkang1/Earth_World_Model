#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete local EE sample dirs whose sample IDs already exist in GCS."
    )
    parser.add_argument("--raw-root", required=True, help="Local raw sample root containing ee_batch_* dirs.")
    parser.add_argument("--gcs-root", required=True, help="GCS raw prefix, e.g. gs://bucket/prefix/raw/")
    parser.add_argument("--apply", action="store_true", help="Actually delete matching local dirs.")
    return parser.parse_args()


def list_gcs_ids(gcs_root: str) -> set[str]:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", gcs_root],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to list {gcs_root}")
    return {line.rstrip("/").split("/")[-1] for line in proc.stdout.splitlines() if line.strip()}


def dir_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                pass
    return total


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    gcs_ids = list_gcs_ids(args.gcs_root)

    matches: list[tuple[str, Path, int]] = []
    for sample_dir in sorted(raw_root.iterdir()):
        if sample_dir.is_dir() and sample_dir.name in gcs_ids:
            matches.append((sample_dir.name, sample_dir, dir_size_bytes(sample_dir)))

    total_bytes = sum(size for _, _, size in matches)
    print(f"gcs_ids={len(gcs_ids)}")
    print(f"local_dirs_backed_by_gcs={len(matches)}")
    print(f"total_bytes={total_bytes}")
    if matches:
        print("first_five=" + ",".join(sample_id for sample_id, _, _ in matches[:5]))

    if args.apply:
        for _, sample_dir, _ in matches:
            shutil.rmtree(sample_dir, ignore_errors=True)
        print("applied=1")
    else:
        print("applied=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
