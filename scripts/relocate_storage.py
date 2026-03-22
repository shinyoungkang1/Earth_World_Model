#!/usr/bin/env python3
"""Relocate repo storage directories into an external folder and leave symlinks behind.

This is intended for synced folders such as Google Drive. The codebase continues
to read and write `data/`, `models/`, and `results/` inside the repo, but those
paths can point at directories stored elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = ("data", "models", "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move repo storage directories into an external folder and replace them with symlinks."
    )
    parser.add_argument(
        "--repo-root",
        default=str(DEFAULT_REPO_ROOT),
        help="Repo root containing data/, models/, and results/.",
    )
    parser.add_argument(
        "--storage-root",
        required=True,
        help="External storage root, for example a Google Drive synced folder.",
    )
    parser.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        choices=ALLOWED_DIRS,
        help="Directory to relocate. Can be repeated. Defaults to data.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the move. Without this flag the script only prints the plan.",
    )
    return parser.parse_args()


def same_location(a: Path, b: Path) -> bool:
    try:
        return a.resolve(strict=True) == b.resolve(strict=True)
    except FileNotFoundError:
        return False


def describe_directory(path: Path) -> dict[str, object]:
    if not path.exists() and not path.is_symlink():
        return {"exists": False}
    if path.is_symlink():
        return {"exists": True, "kind": "symlink", "target": str(path.resolve(strict=False))}
    if path.is_dir():
        child_count = sum(1 for _ in path.iterdir())
        return {"exists": True, "kind": "directory", "child_count": child_count}
    return {"exists": True, "kind": "file"}


def validate_target(repo_root: Path, storage_root: Path, name: str) -> dict[str, object]:
    source = repo_root / name
    target = storage_root / name
    plan: dict[str, object] = {
        "name": name,
        "source": str(source),
        "target": str(target),
        "source_state": describe_directory(source),
        "target_state": describe_directory(target),
    }

    if source.is_symlink():
        if same_location(source, target):
            plan["action"] = "already_linked"
            return plan
        raise RuntimeError(f"{source} is already a symlink, but not to {target}")

    if not source.exists():
        raise RuntimeError(f"{source} does not exist")

    if source.is_file():
        raise RuntimeError(f"{source} is a file, expected a directory")

    if target.is_symlink():
        raise RuntimeError(f"{target} is a symlink; refusing to merge or overwrite it")

    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"{target} exists but is not a directory")
        plan["action"] = "sync_and_link"
        return plan

    plan["action"] = "copy_and_link"
    return plan


def files_match(source: Path, target: Path) -> bool:
    if not target.exists() or not target.is_file():
        return False
    return source.stat().st_size == target.stat().st_size


def sync_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in os.walk(source):
        root_path = Path(root)
        rel_root = root_path.relative_to(source)
        target_root = target / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        for dirname in dirnames:
            (target_root / dirname).mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source_path = root_path / filename
            target_path = target_root / filename
            if files_match(source_path, target_path):
                continue
            if target_path.exists():
                target_path.unlink()
            # Google Drive mounts can reject metadata-preserving copies; copy bytes only.
            shutil.copyfile(source_path, target_path)


def remove_local_directory(path: Path) -> None:
    shutil.rmtree(path)


def relocate_directory(source: Path, target: Path) -> None:
    backup = source.with_name(f"{source.name}.local_backup")
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"Backup path already exists: {backup}")

    sync_tree(source, target)
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
    except Exception:
        if source.is_symlink():
            source.unlink()
        backup.rename(source)
        raise
    remove_local_directory(backup)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    storage_root = Path(args.storage_root).expanduser().resolve()
    dir_names = args.dirs or ["data"]

    plans: list[dict[str, object]] = []
    try:
        for name in dir_names:
            plans.append(validate_target(repo_root, storage_root, name))
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "plans": plans}, indent=2))
        return 1

    if not args.apply:
        print(json.dumps({"ok": True, "apply": False, "plans": plans}, indent=2))
        return 0

    completed: list[dict[str, str]] = []
    try:
        for plan in plans:
            if plan["action"] == "already_linked":
                completed.append(
                    {
                        "name": str(plan["name"]),
                        "action": "already_linked",
                        "source": str(plan["source"]),
                        "target": str(plan["target"]),
                    }
                )
                continue
            source = Path(str(plan["source"]))
            target = Path(str(plan["target"]))
            relocate_directory(source, target)
            completed.append(
                {
                    "name": str(plan["name"]),
                    "action": "moved_and_linked",
                    "source": str(source),
                    "target": str(target),
                }
            )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "completed": completed}, indent=2))
        return 1

    print(json.dumps({"ok": True, "apply": True, "completed": completed}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
