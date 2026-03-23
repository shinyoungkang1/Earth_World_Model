#!/usr/bin/env python3
"""Backfill a run_manifest.json for an existing checkpoint directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "earth_world_model" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import earth_world_model.train_tpu as train_tpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write run_manifest.json for an existing run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda", "tpu"], default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def resolve_manifest_device(backend: str) -> torch.device:
    if backend == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    config = train_tpu.load_config(args.config)
    if args.backend is not None:
        config["runtime"]["backend"] = args.backend
    checkpoint_dir = args.checkpoint_dir.resolve()
    config["runtime"]["checkpoint_dir"] = str(checkpoint_dir)

    dataset = train_tpu.build_dataset(config, section="data")
    loader = train_tpu.make_loader(dataset, config, section="data", shuffle=True, drop_last=True)

    eval_cfg = train_tpu.resolve_eval_config(config)
    eval_dataset = None
    eval_loader = None
    if eval_cfg is not None:
        config["eval"]["data"] = eval_cfg["data"]
        eval_dataset = train_tpu.build_dataset_from_cfg(eval_cfg["data"])
        eval_config = dict(config)
        eval_config["data"] = eval_cfg["data"]
        eval_loader = train_tpu.make_loader(
            eval_dataset,
            eval_config,
            section="data",
            batch_size=eval_cfg["batch_size"],
            shuffle=False,
            drop_last=False,
        )

    backend = str(config["runtime"].get("backend", "cpu"))
    manifest_path = (args.manifest_path or (checkpoint_dir / "run_manifest.json")).resolve()
    manifest = train_tpu.collect_run_manifest(
        args=argparse.Namespace(config=args.config.resolve()),
        config=config,
        backend=backend,
        device=resolve_manifest_device(backend),
        checkpoint_dir=checkpoint_dir,
        dataset=dataset,
        eval_dataset=eval_dataset,
        metrics_path=checkpoint_dir / "metrics.jsonl",
        loader=loader,
        eval_loader=eval_loader,
    )
    train_tpu.write_json(manifest_path, manifest)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
