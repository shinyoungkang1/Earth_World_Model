#!/usr/bin/env python3
"""Merge sharded Phase 4 EWM S2/S1 chip indexes into the canonical index paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded EWM S2/S1 chip indexes.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--basin-id", default="swpa_core_washington_greene")
    parser.add_argument("--features-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    if args.features_dir:
        features_dir = Path(args.features_dir)
        features_dir = features_dir if features_dir.is_absolute() else (repo_root / features_dir)
    else:
        features_dir = repo_root / "data" / "features" / args.basin_id
    features_dir = features_dir.resolve()

    shard_paths = sorted(features_dir.glob("ewm_s2s1_chip_index_v1__shard_*.parquet"))
    if not shard_paths:
        raise SystemExit(f"No sharded chip index parquet files found in {features_dir}")

    frames = [pd.read_parquet(path) for path in shard_paths]
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["sample_id"], keep="last")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )

    metadata_paths = sorted(features_dir.glob("ewm_s2s1_chip_index_v1_metadata__shard_*.json"))
    metadata_rows = [load_json(path) for path in metadata_paths]

    sample_type_counts: dict[str, int] = {}
    for key, value in merged["sample_type"].value_counts().items():
        sample_type_counts[str(key)] = int(value)

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset": "ewm_s2s1_chip_index_v1",
        "row_count": int(len(merged)),
        "sample_type_counts": sample_type_counts,
        "source_shard_count": int(len(shard_paths)),
        "source_index_paths": [str(path) for path in shard_paths],
        "source_metadata_paths": [str(path) for path in metadata_paths],
        "failed_sample_count": int(sum(int(row.get("failed_sample_count", 0)) for row in metadata_rows)),
        "reused_existing_chip_count": int(sum(int(row.get("reused_existing_chip_count", 0)) for row in metadata_rows)),
        "frame_count": int(metadata_rows[0].get("frame_count", 0)) if metadata_rows else 0,
        "chip_size_px": int(metadata_rows[0].get("chip_size_px", 0)) if metadata_rows else 0,
        "modalities": metadata_rows[0].get("modalities", []) if metadata_rows else [],
        "s2_band_order": metadata_rows[0].get("s2_band_order", []) if metadata_rows else [],
        "s1_band_order": metadata_rows[0].get("s1_band_order", []) if metadata_rows else [],
        "min_s2_clear_fraction": (
            float(metadata_rows[0].get("min_s2_clear_fraction", 0.0)) if metadata_rows else 0.0
        ),
        "min_s1_valid_fraction": (
            float(metadata_rows[0].get("min_s1_valid_fraction", 0.0)) if metadata_rows else 0.0
        ),
    }

    out_path = features_dir / "ewm_s2s1_chip_index_v1.parquet"
    out_meta_path = features_dir / "ewm_s2s1_chip_index_v1_metadata.json"
    merged.to_parquet(out_path, index=False)
    payload["output_path"] = str(out_path)
    write_json(out_meta_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
