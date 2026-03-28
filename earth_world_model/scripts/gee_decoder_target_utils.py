#!/usr/bin/env python3
"""Shared utilities for daily decoder-target extraction from Earth Engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from ee_stage_utils import initialize_ee, require_ee

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class DailyRequestBatch:
    date: date
    frame: "pd.DataFrame"


def require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pandas is required for decoder-target extraction scripts.") from exc
    return pd


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_scale_meters: float,
    default_region_side_meters: float = 2560.0,
) -> None:
    parser.add_argument("--project", required=True, help="GCP project for Earth Engine.")
    parser.add_argument(
        "--locations-path",
        required=True,
        help="CSV or parquet with sample_id and coordinates.",
    )
    parser.add_argument(
        "--requests-path",
        default="",
        help="Optional CSV or parquet with explicit sample_id/date rows to extract.",
    )
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    parser.add_argument(
        "--request-sample-id-column",
        default="sample_id",
        help="sample_id column in --requests-path.",
    )
    parser.add_argument(
        "--request-date-column",
        default="date",
        help="Date column in --requests-path.",
    )
    parser.add_argument(
        "--start-date",
        default="",
        help="Inclusive YYYY-MM-DD start date. Required if --requests-path is omitted.",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="Inclusive YYYY-MM-DD end date. Required if --requests-path is omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Samples per EE reduceRegions request.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit loaded samples.")
    parser.add_argument("--max-requests", type=int, default=0, help="Limit explicit request rows.")
    parser.add_argument(
        "--aggregation",
        choices=("point", "square_mean"),
        default="point",
        help="Spatial aggregation for each sample/date target.",
    )
    parser.add_argument(
        "--region-side-meters",
        type=float,
        default=default_region_side_meters,
        help="Square side length used when aggregation=square_mean.",
    )
    parser.add_argument(
        "--scale-meters",
        type=float,
        default=default_scale_meters,
        help="ReduceRegions nominal scale in meters.",
    )
    parser.add_argument(
        "--time-shard-mode",
        choices=("daily", "year"),
        default="year",
        help="Internal extraction sharding. 'year' keeps daily outputs but executes one time-series request per sample batch per year.",
    )
    parser.add_argument(
        "--max-dates-per-time-series-request",
        type=int,
        default=31,
        help="When time_shard_mode=year, split each year's dates into bounded chunks to avoid Earth Engine graph-depth limits.",
    )
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-metadata-json", required=True)
    parser.add_argument("--authenticate", action="store_true")


def _read_table(path: Path) -> pd.DataFrame:
    pd = require_pandas()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported table format for {path}. Expected .csv or .parquet")


def load_locations(
    *,
    path: Path,
    sample_id_column: str,
    latitude_column: str,
    longitude_column: str,
    max_samples: int = 0,
) -> pd.DataFrame:
    pd = require_pandas()
    frame = _read_table(path).copy()
    missing = [column for column in (sample_id_column, latitude_column, longitude_column) if column not in frame.columns]
    if missing:
        raise SystemExit(f"Location table missing columns {missing}: {path}")
    frame = frame.rename(
        columns={
            sample_id_column: "sample_id",
            latitude_column: "latitude",
            longitude_column: "longitude",
        }
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame = frame.dropna(subset=["sample_id", "latitude", "longitude"]).copy()
    frame = frame.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    if max_samples > 0:
        frame = frame.iloc[: int(max_samples)].copy()
    if frame.empty:
        raise SystemExit(f"No usable sample rows loaded from {path}")
    return frame


def load_request_index(
    *,
    path: Path,
    sample_id_column: str,
    date_column: str,
    max_requests: int = 0,
) -> pd.DataFrame:
    pd = require_pandas()
    frame = _read_table(path).copy()
    missing = [column for column in (sample_id_column, date_column) if column not in frame.columns]
    if missing:
        raise SystemExit(f"Request table missing columns {missing}: {path}")
    frame = frame.rename(columns={sample_id_column: "sample_id", date_column: "date"})
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["sample_id", "date"]).copy()
    frame = frame.drop_duplicates(subset=["sample_id", "date"], keep="first").reset_index(drop=True)
    if max_requests > 0:
        frame = frame.iloc[: int(max_requests)].copy()
    if frame.empty:
        raise SystemExit(f"No usable request rows loaded from {path}")
    return frame


def parse_date_arg(raw: str, *, field_name: str) -> date:
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"{field_name} must be YYYY-MM-DD, got {raw!r}") from exc


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_request_batches(
    *,
    samples: pd.DataFrame,
    requests: pd.DataFrame | None,
    start_date: date | None,
    end_date: date | None,
) -> list[DailyRequestBatch]:
    pd = require_pandas()
    if requests is not None:
        merged = requests.merge(samples, on="sample_id", how="inner")
        if merged.empty:
            raise SystemExit("No explicit request rows matched the loaded sample locations.")
        batches = []
        for request_date, frame in merged.groupby("date", sort=True):
            batches.append(
                DailyRequestBatch(
                    date=pd.Timestamp(request_date).date(),
                    frame=frame[["sample_id", "latitude", "longitude"]].reset_index(drop=True),
                )
            )
        return batches
    if start_date is None or end_date is None:
        raise SystemExit("Either --requests-path or both --start-date and --end-date are required.")
    if end_date < start_date:
        raise SystemExit(f"end-date {end_date} is earlier than start-date {start_date}")
    return [
        DailyRequestBatch(date=request_date, frame=samples.copy())
        for request_date in iter_dates(start_date, end_date)
    ]


def build_request_frame(
    *,
    samples: pd.DataFrame,
    requests: pd.DataFrame | None,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    pd = require_pandas()
    if requests is not None:
        merged = requests.merge(samples, on="sample_id", how="inner")
        if merged.empty:
            raise SystemExit("No explicit request rows matched the loaded sample locations.")
        merged = merged[["sample_id", "latitude", "longitude", "date"]].copy()
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
        merged = merged.dropna(subset=["sample_id", "latitude", "longitude", "date"]).copy()
        merged = merged.drop_duplicates(subset=["sample_id", "date"], keep="first").reset_index(drop=True)
        return merged
    if start_date is None or end_date is None:
        raise SystemExit("Either --requests-path or both --start-date and --end-date are required.")
    if end_date < start_date:
        raise SystemExit(f"end-date {end_date} is earlier than start-date {start_date}")
    dates = list(iter_dates(start_date, end_date))
    if not dates:
        raise SystemExit("No dates selected for extraction.")
    request_dates = pd.DataFrame({"date": pd.to_datetime([value.isoformat() for value in dates]).normalize()})
    sample_frame = samples.copy()
    sample_frame["_join_key"] = 1
    request_dates["_join_key"] = 1
    merged = sample_frame.merge(request_dates, on="_join_key", how="inner").drop(columns="_join_key")
    merged = merged[["sample_id", "latitude", "longitude", "date"]].copy()
    return merged


def request_frame_to_batches(request_frame: pd.DataFrame) -> list[DailyRequestBatch]:
    pd = require_pandas()
    batches: list[DailyRequestBatch] = []
    for request_date, frame in request_frame.groupby("date", sort=True):
        batches.append(
            DailyRequestBatch(
                date=pd.Timestamp(request_date).date(),
                frame=frame[["sample_id", "latitude", "longitude"]].reset_index(drop=True),
            )
        )
    return batches


def group_request_dates_by_year(request_frame: pd.DataFrame) -> list[tuple[int, list[date]]]:
    pd = require_pandas()
    if request_frame.empty:
        return []
    unique_dates = (
        pd.to_datetime(request_frame["date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    grouped: list[tuple[int, list[date]]] = []
    for year_value, year_dates in unique_dates.groupby(unique_dates.dt.year, sort=True):
        grouped.append((int(year_value), [pd.Timestamp(value).date() for value in year_dates.tolist()]))
    return grouped


def chunk_dates(dates: list[date], max_dates_per_chunk: int) -> list[list[date]]:
    if max_dates_per_chunk <= 0:
        return [dates]
    return [dates[start : start + max_dates_per_chunk] for start in range(0, len(dates), max_dates_per_chunk)]


def chunk_frame(frame: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    if batch_size <= 0:
        raise SystemExit(f"batch-size must be positive, got {batch_size}")
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size].copy()


def build_feature_collection(
    ee: Any,
    *,
    frame: pd.DataFrame,
    aggregation: str,
    region_side_meters: float,
) -> Any:
    features = []
    half_side = max(0.0, float(region_side_meters)) / 2.0
    for row in frame.itertuples(index=False):
        lon = float(row.longitude)
        lat = float(row.latitude)
        geometry = ee.Geometry.Point([lon, lat])
        if aggregation == "square_mean":
            geometry = geometry.buffer(half_side).bounds()
        features.append(
            ee.Feature(
                geometry,
                {
                    "sample_id": str(row.sample_id),
                },
            )
        )
    return ee.FeatureCollection(features)


def reducer_for_aggregation(ee: Any, aggregation: str) -> Any:
    if aggregation == "point":
        return ee.Reducer.first()
    if aggregation == "square_mean":
        return ee.Reducer.mean()
    raise SystemExit(f"Unsupported aggregation mode: {aggregation}")


def collect_reduced_rows(
    *,
    reduced_fc: Any,
    request_date: date | None,
    product_name: str | None,
) -> list[dict[str, Any]]:
    payload = reduced_fc.getInfo()
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = dict(feature.get("properties") or {})
        sample_id = props.pop("sample_id", None)
        if sample_id is None:
            continue
        row_date = props.pop("date", None)
        if row_date is None and request_date is not None:
            row_date = request_date.isoformat()
        if row_date is None:
            continue
        row_product = props.pop("source_product", None)
        if row_product is None and product_name is not None:
            row_product = product_name
        row: dict[str, Any] = {
            "sample_id": str(sample_id),
            "date": str(row_date),
            "source_product": str(row_product) if row_product is not None else None,
        }
        for key, value in props.items():
            row[key] = value
        rows.append(row)
    return rows


def write_rows_with_metadata(
    *,
    rows: list[dict[str, Any]],
    output_parquet: Path,
    output_metadata_json: Path,
    metadata: dict[str, Any],
) -> None:
    pd = require_pandas()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if not frame.empty and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame.to_parquet(output_parquet, index=False)
    output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _requested_pair_frame(request_frame: pd.DataFrame) -> pd.DataFrame:
    pd = require_pandas()
    pairs = request_frame[["sample_id", "date"]].copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["date"] = pd.to_datetime(pairs["date"], errors="coerce").dt.normalize()
    pairs = pairs.dropna(subset=["sample_id", "date"]).drop_duplicates(subset=["sample_id", "date"], keep="first")
    return pairs.reset_index(drop=True)


def _filter_rows_to_requested_pairs(rows: list[dict[str, Any]], request_frame: pd.DataFrame) -> list[dict[str, Any]]:
    pd = require_pandas()
    if not rows:
        return rows
    frame = pd.DataFrame(rows)
    if frame.empty:
        return rows
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    pairs = _requested_pair_frame(request_frame)
    frame = frame.merge(pairs, on=["sample_id", "date"], how="inner")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame.to_dict(orient="records")


def _request_dates_with_image(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    valid_dates: set[str] = set()
    for row in rows:
        target_valid = row.get("target_valid")
        if target_valid is None:
            continue
        try:
            if float(target_valid) > 0:
                valid_dates.add(str(row.get("date")))
        except (TypeError, ValueError):
            continue
    return len(valid_dates)


def reduce_time_series_over_batch(
    ee: Any,
    *,
    request_dates: list[date],
    sample_batch: pd.DataFrame,
    aggregation: str,
    region_side_meters: float,
    reducer: Any,
    scale_meters: float,
    product_name: str,
    build_daily_image_server: Callable[[Any, date], Any],
) -> list[dict[str, Any]]:
    if not request_dates:
        return []
    fc = build_feature_collection(
        ee,
        frame=sample_batch,
        aggregation=aggregation,
        region_side_meters=region_side_meters,
    )
    merged_fc = None
    for request_date in request_dates:
        image = build_daily_image_server(ee, request_date)
        reduced = image.reduceRegions(
            collection=fc,
            reducer=reducer,
            scale=scale_meters,
        )

        request_date_iso = request_date.isoformat()

        def _annotate(feature):
            return ee.Feature(feature).set("date", request_date_iso).set("source_product", product_name)

        annotated = reduced.map(_annotate)
        merged_fc = annotated if merged_fc is None else merged_fc.merge(annotated)
    if merged_fc is None:
        return []
    return collect_reduced_rows(
        reduced_fc=merged_fc,
        request_date=None,
        product_name=None,
    )


def run_daily_extraction(
    *,
    product_name: str,
    dataset_id: str,
    args: argparse.Namespace,
    build_daily_image_server: Callable[[Any, date], Any],
) -> None:
    ee = require_ee()
    initialize_ee(ee, project=args.project, authenticate=bool(args.authenticate))

    samples = load_locations(
        path=Path(args.locations_path),
        sample_id_column=args.sample_id_column,
        latitude_column=args.latitude_column,
        longitude_column=args.longitude_column,
        max_samples=int(args.max_samples),
    )
    requests = (
        load_request_index(
            path=Path(args.requests_path),
            sample_id_column=args.request_sample_id_column,
            date_column=args.request_date_column,
            max_requests=int(args.max_requests),
        )
        if args.requests_path
        else None
    )
    start_date = parse_date_arg(args.start_date, field_name="start-date") if args.start_date else None
    end_date = parse_date_arg(args.end_date, field_name="end-date") if args.end_date else None
    request_frame = build_request_frame(
        samples=samples,
        requests=requests,
        start_date=start_date,
        end_date=end_date,
    )

    rows: list[dict[str, Any]] = []
    reducer = reducer_for_aggregation(ee, args.aggregation)
    if args.time_shard_mode == "daily":
        request_batches = request_frame_to_batches(request_frame)
        for request_batch in request_batches:
            date_frame = request_batch.frame
            for sample_batch in chunk_frame(date_frame, int(args.batch_size)):
                rows.extend(
                    reduce_time_series_over_batch(
                        ee,
                        request_dates=[request_batch.date],
                        sample_batch=sample_batch,
                        aggregation=args.aggregation,
                        region_side_meters=float(args.region_side_meters),
                        reducer=reducer,
                        scale_meters=float(args.scale_meters),
                        product_name=product_name,
                        build_daily_image_server=build_daily_image_server,
                    )
                )
    else:
        for _year_value, year_dates in group_request_dates_by_year(request_frame):
            for date_chunk in chunk_dates(year_dates, int(args.max_dates_per_time_series_request)):
                for sample_batch in chunk_frame(samples, int(args.batch_size)):
                    rows.extend(
                        reduce_time_series_over_batch(
                            ee,
                            request_dates=date_chunk,
                            sample_batch=sample_batch,
                            aggregation=args.aggregation,
                            region_side_meters=float(args.region_side_meters),
                            reducer=reducer,
                            scale_meters=float(args.scale_meters),
                            product_name=product_name,
                            build_daily_image_server=build_daily_image_server,
                        )
                    )
        rows = _filter_rows_to_requested_pairs(rows, request_frame)

    request_date_count = int(request_frame["date"].nunique())
    time_shard_count = int(len(group_request_dates_by_year(request_frame))) if args.time_shard_mode == "year" else request_date_count
    output_parquet = Path(args.output_parquet)
    output_metadata_json = Path(args.output_metadata_json)
    metadata = {
        "dataset": product_name,
        "source_dataset_id": dataset_id,
        "locations_path": str(Path(args.locations_path)),
        "requests_path": str(Path(args.requests_path)) if args.requests_path else None,
        "sample_count": int(len(samples)),
        "request_date_count": request_date_count,
        "request_row_count": int(len(request_frame)),
        "request_dates_with_image": int(_request_dates_with_image(rows)),
        "time_shard_mode": str(args.time_shard_mode),
        "time_shard_count": time_shard_count,
        "max_dates_per_time_series_request": int(args.max_dates_per_time_series_request),
        "row_count_output": int(len(rows)),
        "aggregation": args.aggregation,
        "region_side_meters": float(args.region_side_meters),
        "scale_meters": float(args.scale_meters),
        "output_parquet": str(output_parquet),
        "output_metadata_json": str(output_metadata_json),
    }
    write_rows_with_metadata(
        rows=rows,
        output_parquet=output_parquet,
        output_metadata_json=output_metadata_json,
        metadata=metadata,
    )
    print(json.dumps({"dataset": product_name, "rows": int(len(rows)), "output_parquet": str(output_parquet)}, indent=2))
