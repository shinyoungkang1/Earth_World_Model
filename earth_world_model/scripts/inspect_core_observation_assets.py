#!/usr/bin/env python3
"""Inspect whether the core S1/S2 observation corpus is already present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the core yearly S1/S2 and SSL4EO observation artifacts already exist.",
    )
    parser.add_argument("--yearly-root", default="")
    parser.add_argument("--ssl4eo-root", default="")
    parser.add_argument("--yearly-train-index-path", default="")
    parser.add_argument("--yearly-val-index-path", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def resolve_yearly_indices(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    yearly_root = Path(args.yearly_root) if args.yearly_root else None
    train_index = Path(args.yearly_train_index_path) if args.yearly_train_index_path else None
    val_index = Path(args.yearly_val_index_path) if args.yearly_val_index_path else None
    if yearly_root is not None:
        train_index = train_index or (yearly_root / "train" / "dense_temporal_index.parquet")
        val_index = val_index or (yearly_root / "val" / "dense_temporal_index.parquet")
    return train_index, val_index


def build_status(args: argparse.Namespace) -> dict:
    train_index, val_index = resolve_yearly_indices(args)
    ssl4eo_root = Path(args.ssl4eo_root) if args.ssl4eo_root else None

    checks: list[dict] = []

    def add_check(label: str, path: Path | None, *, required: bool = True) -> None:
        checks.append(
            {
                "label": label,
                "path": str(path) if path is not None else None,
                "exists": bool(path is not None and path.exists()),
                "required": bool(required),
            }
        )

    add_check("yearly_train_index", train_index)
    add_check("yearly_val_index", val_index)
    if ssl4eo_root is not None:
        add_check("ssl4eo_train_s2", ssl4eo_root / "train" / "S2L2A")
        add_check("ssl4eo_train_s1", ssl4eo_root / "train" / "S1GRD")
        add_check("ssl4eo_val_s2", ssl4eo_root / "val" / "S2L2A")
        add_check("ssl4eo_val_s1", ssl4eo_root / "val" / "S1GRD")

    configured = [check for check in checks if check["path"] is not None]
    missing = [check for check in configured if check["required"] and not check["exists"]]
    ready = None if not configured else (len(missing) == 0)

    return {
        "dataset": "core_observation_status_v1",
        "yearly_root": str(Path(args.yearly_root)) if args.yearly_root else None,
        "ssl4eo_root": str(Path(args.ssl4eo_root)) if args.ssl4eo_root else None,
        "configured_check_count": int(len(configured)),
        "ready": ready,
        "checks": checks,
        "missing_required": missing,
        "recovery_scripts": {
            "prepare_preliminary_datasets": "earth_world_model/scripts/prepare_preliminary_datasets.py",
            "gcp_yearly10k_ssl4eo50k_prep": "earth_world_model/scripts/gcp_prepare_yearly10k_ssl4eo50k_to_ssd.sh",
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_status(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.require_ready and payload["ready"] is not True:
        raise SystemExit("Core observation assets are not fully ready.")


if __name__ == "__main__":
    main()
