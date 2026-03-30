#!/usr/bin/env python3
"""Build an Earth Engine interactive-point CSV from candidate regions or sampled center tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sample_id,latitude,longitude CSV from candidate regions.")
    parser.add_argument("--regions-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining rows.")
    parser.add_argument("--sample-id-start", type=int, default=0)
    parser.add_argument("--sample-id-prefix", default="ee_batch_")
    parser.add_argument(
        "--force-regenerate-sample-ids",
        action="store_true",
        help="Regenerate sample_id even if the input table already contains one.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported regions format for {path}. Expected .csv or .parquet")


def main() -> int:
    args = parse_args()
    df = read_table(args.regions_path)
    if args.offset > 0:
        df = df.iloc[int(args.offset) :].copy()
    if int(args.limit) > 0:
        df = df.iloc[: int(args.limit)].copy()
    if df.empty:
        raise SystemExit("No region rows selected")

    df = df.reset_index(drop=True)
    if "sample_id" in df.columns and not args.force_regenerate_sample_ids:
        df["sample_id"] = df["sample_id"].astype(str)
    else:
        start = int(args.sample_id_start)
        df["sample_id"] = [f"{args.sample_id_prefix}{start + idx:08d}" for idx in range(len(df))]
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "latitude": df["center_latitude"],
            "longitude": df["center_longitude"],
            "region_id": df["region_id"],
            "continent_bucket": df["continent_bucket"],
            "country_bucket": df["country_bucket"],
            "latitude_band": df["latitude_band"],
        }
    )
    passthrough_columns = [
        "sampling_design",
        "sampling_group_id",
        "cluster_id",
        "cluster_anchor_region_id",
        "cluster_anchor_grid_x",
        "cluster_anchor_grid_y",
        "cluster_dx_idx",
        "cluster_dy_idx",
        "cluster_grid_radius",
        "cluster_grid_side",
        "bucket_id",
        "grid_x",
        "grid_y",
    ]
    for column in passthrough_columns:
        if column in df.columns and column not in out.columns:
            out[column] = df[column]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print({"rows": int(len(out)), "output_path": str(args.output_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
