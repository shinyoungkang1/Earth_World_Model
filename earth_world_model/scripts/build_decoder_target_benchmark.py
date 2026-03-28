#!/usr/bin/env python3
"""Build an aligned decoder-target benchmark from per-product daily target tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


ID_COLUMNS = ("sample_id", "date")
PRIMARY_TARGET_SOURCES = ("modis", "chirps", "smap")


@dataclass(frozen=True)
class SourceFrame:
    name: str
    path: Path
    frame: "pd.DataFrame"
    value_columns: list[str]
    duplicate_rows: int


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pandas is required to build the decoder target benchmark.") from exc
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an aligned decoder benchmark from daily target parquet files.",
    )
    parser.add_argument("--modis-path", default="", help="Daily MODIS LST parquet.")
    parser.add_argument("--chirps-path", default="", help="Daily CHIRPS parquet.")
    parser.add_argument("--smap-path", default="", help="Daily SMAP parquet.")
    parser.add_argument("--era5-path", default="", help="Daily ERA5-Land parquet.")
    parser.add_argument(
        "--join",
        choices=("outer", "inner"),
        default="outer",
        help="How to join source tables on sample_id/date.",
    )
    parser.add_argument(
        "--require-sources",
        default="modis,chirps,smap",
        help="Comma-separated sources that must be present on a row to keep it.",
    )
    parser.add_argument(
        "--forecast-horizons-days",
        default="",
        help="Optional comma-separated future target horizons in days, e.g. 7,14,30.",
    )
    parser.add_argument(
        "--output-parquet",
        required=True,
        help="Output aligned benchmark parquet path.",
    )
    parser.add_argument(
        "--output-metadata-json",
        required=True,
        help="Output metadata JSON path.",
    )
    return parser.parse_args()


def parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_horizons(raw: str) -> list[int]:
    horizons = []
    for token in parse_csv_tokens(raw):
        value = int(token)
        if value <= 0:
            raise SystemExit(f"Forecast horizon must be positive, got {value}")
        horizons.append(value)
    return sorted(set(horizons))


def load_source_frame(name: str, raw_path: str) -> SourceFrame | None:
    pd = require_pandas()
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"{name} parquet not found: {path}")
    frame = pd.read_parquet(path).copy()
    missing = [column for column in ID_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{name} parquet missing required columns {missing}: {path}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        bad_rows = int(frame["date"].isna().sum())
        raise SystemExit(f"{name} parquet contains {bad_rows} rows with invalid date values: {path}")
    duplicate_mask = frame.duplicated(subset=list(ID_COLUMNS), keep="last")
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        frame = frame.loc[~duplicate_mask].copy()
    value_columns = [column for column in frame.columns if column not in ID_COLUMNS]
    renamed = {
        column: f"{name}_{column}"
        for column in value_columns
    }
    frame = frame.rename(columns=renamed)
    value_columns = [renamed[column] for column in value_columns]
    frame[f"has_{name}"] = True
    value_columns.append(f"has_{name}")
    return SourceFrame(
        name=name,
        path=path,
        frame=frame,
        value_columns=value_columns,
        duplicate_rows=duplicate_rows,
    )


def merge_sources(sources: list[SourceFrame], *, join: str) -> pd.DataFrame:
    if not sources:
        raise SystemExit("At least one source parquet is required.")
    merged = sources[0].frame.copy()
    for source in sources[1:]:
        merged = merged.merge(source.frame, on=list(ID_COLUMNS), how=join)
    merged = merged.sort_values(["sample_id", "date"]).reset_index(drop=True)
    for source in sources:
        has_column = f"has_{source.name}"
        if has_column in merged.columns:
            merged[has_column] = merged[has_column].fillna(False).astype(bool)
    return merged


def add_future_targets(
    benchmark: pd.DataFrame,
    *,
    sources: list[SourceFrame],
    horizons_days: list[int],
) -> pd.DataFrame:
    pd = require_pandas()
    if not horizons_days:
        return benchmark
    output = benchmark.copy()
    for source in sources:
        value_columns = [
            column for column in source.value_columns
            if not column.startswith("has_")
        ]
        if not value_columns:
            continue
        source_base = source.frame[["sample_id", "date", *value_columns]].copy()
        for horizon_days in horizons_days:
            shifted = source_base.copy()
            shifted["date"] = shifted["date"] - pd.to_timedelta(horizon_days, unit="D")
            shifted = shifted.rename(
                columns={
                    column: f"{column}_plus_{horizon_days}d"
                    for column in value_columns
                }
            )
            output = output.merge(
                shifted,
                on=["sample_id", "date"],
                how="left",
            )
    return output


def filter_required_sources(
    benchmark: pd.DataFrame,
    *,
    required_sources: list[str],
) -> pd.DataFrame:
    pd = require_pandas()
    if not required_sources:
        return benchmark
    mask = pd.Series(True, index=benchmark.index)
    for source in required_sources:
        column = f"has_{source}"
        if column not in benchmark.columns:
            raise SystemExit(f"Required source {source!r} is not present in the merged benchmark.")
        mask &= benchmark[column].fillna(False).astype(bool)
    return benchmark.loc[mask].copy().reset_index(drop=True)


def build_metadata(
    *,
    benchmark: pd.DataFrame,
    sources: list[SourceFrame],
    required_sources: list[str],
    horizons_days: list[int],
    join: str,
) -> dict[str, Any]:
    pd = require_pandas()
    row_count = int(len(benchmark))
    if row_count:
        min_date = str(pd.Timestamp(benchmark["date"].min()).date())
        max_date = str(pd.Timestamp(benchmark["date"].max()).date())
    else:
        min_date = None
        max_date = None
    per_source = {}
    for source in sources:
        has_column = f"has_{source.name}"
        coverage_count = int(benchmark[has_column].fillna(False).sum()) if has_column in benchmark.columns else 0
        per_source[source.name] = {
            "path": str(source.path),
            "row_count_input": int(len(source.frame)),
            "duplicate_rows_dropped": int(source.duplicate_rows),
            "value_columns": [column for column in source.value_columns if not column.startswith("has_")],
            "coverage_count_in_benchmark": coverage_count,
            "coverage_fraction_in_benchmark": (coverage_count / row_count) if row_count else 0.0,
        }
    primary_count_columns = [f"has_{name}" for name in PRIMARY_TARGET_SOURCES if f"has_{name}" in benchmark.columns]
    if primary_count_columns:
        target_count = benchmark[primary_count_columns].fillna(False).astype(int).sum(axis=1)
        target_count_distribution = {
            str(int(key)): int(value)
            for key, value in target_count.value_counts().sort_index().items()
        }
    else:
        target_count_distribution = {}
    return {
        "dataset": "decoder_target_benchmark_v1",
        "row_count": row_count,
        "date_min": min_date,
        "date_max": max_date,
        "join": join,
        "required_sources": required_sources,
        "forecast_horizons_days": horizons_days,
        "primary_target_sources": list(PRIMARY_TARGET_SOURCES),
        "primary_target_count_distribution": target_count_distribution,
        "sources": per_source,
        "columns": list(benchmark.columns),
    }


def main() -> None:
    args = parse_args()
    sources = [
        source
        for source in (
            load_source_frame("modis", args.modis_path),
            load_source_frame("chirps", args.chirps_path),
            load_source_frame("smap", args.smap_path),
            load_source_frame("era5", args.era5_path),
        )
        if source is not None
    ]
    required_sources = parse_csv_tokens(args.require_sources)
    horizons_days = parse_horizons(args.forecast_horizons_days)

    benchmark = merge_sources(sources, join=args.join)
    benchmark = add_future_targets(benchmark, sources=sources, horizons_days=horizons_days)
    benchmark = filter_required_sources(benchmark, required_sources=required_sources)

    output_parquet = Path(args.output_parquet)
    output_metadata_json = Path(args.output_metadata_json)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_json.parent.mkdir(parents=True, exist_ok=True)

    benchmark.to_parquet(output_parquet, index=False)
    metadata = build_metadata(
        benchmark=benchmark,
        sources=sources,
        required_sources=required_sources,
        horizons_days=horizons_days,
        join=args.join,
    )
    output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output_parquet": str(output_parquet), "rows": int(len(benchmark))}, indent=2))


if __name__ == "__main__":
    main()
