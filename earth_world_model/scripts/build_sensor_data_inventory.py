#!/usr/bin/env python3
"""Build a unified inventory/resume manifest for the sensor data program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_core_observation_assets import build_status as build_core_status


ALLOWED_PRODUCTS = ("modis", "chirps", "smap", "era5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect existing core/HLS/static/decoder outputs and write a unified inventory manifest.",
    )
    parser.add_argument("--yearly-root", default="")
    parser.add_argument("--ssl4eo-root", default="")
    parser.add_argument("--yearly-train-index-path", default="")
    parser.add_argument("--yearly-val-index-path", default="")
    parser.add_argument("--hls-index-path", default="")
    parser.add_argument("--obs-events-parquet", default="")
    parser.add_argument("--obs-events-metadata-json", default="")
    parser.add_argument("--static-parquet", default="")
    parser.add_argument("--static-metadata-json", default="")
    parser.add_argument("--targets-dir", default="")
    parser.add_argument("--benchmarks-dir", default="")
    parser.add_argument("--products", default="modis,chirps,smap,era5")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _parse_products(raw: str) -> list[str]:
    products: list[str] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token not in ALLOWED_PRODUCTS:
            raise SystemExit(f"Unknown product {token!r}; allowed values are {list(ALLOWED_PRODUCTS)}")
        if token not in products:
            products.append(token)
    return products


def _path_payload(path: Path | None) -> dict[str, object]:
    return {
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and path.exists()),
    }


def _component_ready(*checks: bool) -> bool | None:
    if not checks:
        return None
    return all(checks)


def build_inventory(args: argparse.Namespace) -> dict[str, object]:
    output_json = Path(args.output_json)
    status_dir = output_json.parent
    products = _parse_products(args.products)

    core_status = build_core_status(args)

    hls_index_path = Path(args.hls_index_path) if args.hls_index_path else None
    obs_events_parquet = Path(args.obs_events_parquet) if args.obs_events_parquet else None
    obs_events_metadata = Path(args.obs_events_metadata_json) if args.obs_events_metadata_json else None
    static_parquet = Path(args.static_parquet) if args.static_parquet else None
    static_metadata = Path(args.static_metadata_json) if args.static_metadata_json else None
    targets_dir = Path(args.targets_dir) if args.targets_dir else None
    benchmarks_dir = Path(args.benchmarks_dir) if args.benchmarks_dir else None

    hls = {
        "index": _path_payload(hls_index_path),
        "obs_events_parquet": _path_payload(obs_events_parquet),
        "obs_events_metadata_json": _path_payload(obs_events_metadata),
    }
    hls["ready_input"] = _component_ready(bool(hls["index"]["exists"]))
    hls["ready_output"] = _component_ready(
        bool(hls["obs_events_parquet"]["exists"]),
        bool(hls["obs_events_metadata_json"]["exists"]),
    )

    static_context = {
        "parquet": _path_payload(static_parquet),
        "metadata_json": _path_payload(static_metadata),
    }
    static_context["ready_output"] = _component_ready(
        bool(static_context["parquet"]["exists"]),
        bool(static_context["metadata_json"]["exists"]),
    )

    decoder_products: dict[str, object] = {}
    for product in products:
        parquet_path = (targets_dir / f"{product}_daily.parquet") if targets_dir is not None else None
        metadata_path = (targets_dir / f"{product}_daily_metadata.json") if targets_dir is not None else None
        decoder_products[product] = {
            "parquet": _path_payload(parquet_path),
            "metadata_json": _path_payload(metadata_path),
            "ready_output": _component_ready(
                bool(parquet_path is not None and parquet_path.exists()),
                bool(metadata_path is not None and metadata_path.exists()),
            ),
        }

    benchmark_parquet = (benchmarks_dir / "decoder_target_benchmark_v1.parquet") if benchmarks_dir is not None else None
    benchmark_metadata = (benchmarks_dir / "decoder_target_benchmark_v1_metadata.json") if benchmarks_dir is not None else None
    benchmark = {
        "parquet": _path_payload(benchmark_parquet),
        "metadata_json": _path_payload(benchmark_metadata),
    }
    benchmark["ready_output"] = _component_ready(
        bool(benchmark_parquet is not None and benchmark_parquet.exists()),
        bool(benchmark_metadata is not None and benchmark_metadata.exists()),
    )

    return {
        "dataset": "sensor_data_inventory_v1",
        "status_dir": str(status_dir),
        "components": {
            "core_observations": core_status,
            "hls": hls,
            "static_context": static_context,
            "decoder_targets": {
                "targets_dir": str(targets_dir) if targets_dir is not None else None,
                "products": decoder_products,
                "benchmark": benchmark,
            },
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_inventory(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
