#!/usr/bin/env python3
"""Build an Earth Engine interactive-point CSV from candidate regions."""

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.regions_path)
    if args.offset > 0:
        df = df.iloc[int(args.offset) :].copy()
    if int(args.limit) > 0:
        df = df.iloc[: int(args.limit)].copy()
    if df.empty:
        raise SystemExit("No region rows selected")

    start = int(args.sample_id_start)
    df = df.reset_index(drop=True)
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
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print({"rows": int(len(out)), "output_path": str(args.output_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
