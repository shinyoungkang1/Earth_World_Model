#!/usr/bin/env python3
"""Benchmark SSL4EO dataloader startup and early-batch throughput."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from earth_world_model.train_tpu import build_dataset_from_cfg, load_config, make_loader, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--section", default="data")
    parser.add_argument("--override-root-dir", default=None)
    parser.add_argument("--override-max-samples", type=int, default=None)
    parser.add_argument("--override-sample-offset", type=int, default=None)
    parser.add_argument("--override-num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def batch_shapes(batch: dict) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in batch.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[key] = list(shape)
    return shapes


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config)
    if args.section not in config:
        raise KeyError(f"Missing config section: {args.section}")
    data_cfg = dict(config[args.section])
    if args.override_root_dir is not None:
        data_cfg["root_dir"] = args.override_root_dir
    if args.override_max_samples is not None:
        data_cfg["max_samples"] = int(args.override_max_samples)
    if args.override_sample_offset is not None:
        data_cfg["sample_offset"] = int(args.override_sample_offset)
    if args.override_num_workers is not None:
        data_cfg["num_workers"] = int(args.override_num_workers)
    config[args.section] = data_cfg

    started_at = time.time()
    dataset = build_dataset_from_cfg(data_cfg)
    dataset_init_sec = time.time() - started_at

    loader_started_at = time.time()
    loader = make_loader(
        dataset,
        config,
        section=args.section,
        batch_size=args.batch_size,
        shuffle=bool(args.shuffle),
        drop_last=True,
    )
    loader_init_sec = time.time() - loader_started_at

    iter_started_at = time.time()
    iterator = iter(loader)
    iterator_init_sec = time.time() - iter_started_at

    batch_times_sec: list[float] = []
    first_batch_shapes: dict[str, list[int]] | None = None
    observed_batches = 0
    for _ in range(max(0, int(args.max_batches))):
        batch_started_at = time.time()
        batch = next(iterator)
        batch_elapsed_sec = time.time() - batch_started_at
        batch_times_sec.append(batch_elapsed_sec)
        observed_batches += 1
        if first_batch_shapes is None:
            first_batch_shapes = batch_shapes(batch)

    payload = {
        "config_path": str(args.config.resolve()),
        "section": args.section,
        "root_dir": data_cfg.get("root_dir"),
        "dataset_kind": data_cfg.get("kind"),
        "dataset_requires_single_process_io": bool(getattr(dataset, "requires_single_process_io", False)),
        "row_count": len(dataset),
        "base_row_count": getattr(dataset, "base_row_count", len(dataset)),
        "num_workers_requested": args.override_num_workers if args.override_num_workers is not None else data_cfg.get("num_workers", 0),
        "num_workers_effective": loader.num_workers,
        "batch_size_effective": loader.batch_size,
        "dataset_init_sec": round(dataset_init_sec, 4),
        "loader_init_sec": round(loader_init_sec, 4),
        "iterator_init_sec": round(iterator_init_sec, 4),
        "observed_batches": observed_batches,
        "first_batch_sec": round(batch_times_sec[0], 4) if batch_times_sec else None,
        "mean_batch_sec": round(statistics.mean(batch_times_sec), 4) if batch_times_sec else None,
        "median_batch_sec": round(statistics.median(batch_times_sec), 4) if batch_times_sec else None,
        "batch_times_sec": [round(value, 4) for value in batch_times_sec],
        "first_batch_shapes": first_batch_shapes,
        "timestamp": time.time(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
