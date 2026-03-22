#!/usr/bin/env python3
"""Build a manifest for pre-extracted SSL4EO temporal `.npz` samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def validate_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as sample:
        keys = set(sample.files)
        missing = {"s2", "s1"} - keys
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")

        s2 = np.asarray(sample["s2"])
        s1 = np.asarray(sample["s1"])
        dates = np.asarray(sample["dates"]) if "dates" in keys else None
        mask = np.asarray(sample["mask"]) if "mask" in keys else None

        if s2.ndim != 4 or s2.shape[0] != 4 or s2.shape[1] != 12:
            raise ValueError(f"{path} has invalid s2 shape: {s2.shape}")
        if s1.ndim != 4 or s1.shape[0] != 4 or s1.shape[1] != 2:
            raise ValueError(f"{path} has invalid s1 shape: {s1.shape}")
        if dates is not None and (dates.ndim != 1 or dates.shape[0] != 4):
            raise ValueError(f"{path} has invalid dates shape: {dates.shape}")
        if mask is not None and (mask.ndim != 1 or mask.shape[0] != 4):
            raise ValueError(f"{path} has invalid mask shape: {mask.shape}")

    return {
        "height": int(s2.shape[2]),
        "width": int(s2.shape[3]),
        "has_dates": dates is not None,
        "has_mask": mask is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="**/*.npz")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples_root = args.samples_root.resolve()
    sample_paths = sorted(samples_root.glob(args.glob))
    if args.limit > 0:
        sample_paths = sample_paths[: args.limit]

    if not sample_paths:
        raise SystemExit(f"No sample files matched {args.glob} under {samples_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    heights = set()
    widths = set()
    dates_count = 0
    mask_count = 0

    for sample_path in sample_paths:
        stats = validate_npz(sample_path)
        heights.add(stats["height"])
        widths.add(stats["width"])
        dates_count += int(stats["has_dates"])
        mask_count += int(stats["has_mask"])

        sample_id = sample_path.relative_to(samples_root).with_suffix("").as_posix()
        rows.append(
            {
                "sample_id": sample_id,
                "path": str(sample_path.resolve()),
            }
        )

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    metadata_path = args.output.with_suffix(".metadata.json")
    metadata = {
        "samples_root": str(samples_root),
        "manifest_path": str(args.output.resolve()),
        "row_count": len(rows),
        "height_values": sorted(heights),
        "width_values": sorted(widths),
        "rows_with_dates": dates_count,
        "rows_with_mask": mask_count,
        "glob": args.glob,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {args.output}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
