"""Shared helpers for the WV horizontal gas pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gas_v1_common import (
    build_balltree,
    count_neighbors_before_date,
    count_neighbors_current,
    load_basin_config,
    nearest_neighbor_before_date_km,
    nearest_neighbor_current_km,
    parse_date,
    polygon_from_bbox,
    prospect_tier,
    write_csv,
)


def county_code_string(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype(str).str.zfill(3).replace("<NA>", pd.NA)


def clean_flag(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    values = values.replace({"<na>": pd.NA, "nan": pd.NA, "": pd.NA})
    return values


def clean_formation(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.replace({"<NA>": pd.NA, "nan": pd.NA, "NA": pd.NA, "": pd.NA})
    return values


def first_valid(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else pd.NA


def last_valid(series: pd.Series):
    values = series.dropna()
    return values.iloc[-1] if not values.empty else pd.NA


def deduplicate_wells(wells: pd.DataFrame) -> pd.DataFrame:
    frame = wells.copy()
    for col in ["permit_issued_date", "completion_date", "record_date"]:
        frame[col] = parse_date(frame[col])
    rows = []
    for well_api, group in frame.groupby("well_api", dropna=True):
        ordered = group.sort_values(
            ["record_date", "permit_issued_date", "completion_date", "objectid"],
            na_position="last",
        )
        rows.append(
            {
                "well_api": well_api,
                "objectid": last_valid(ordered["objectid"]),
                "county_code": county_code_string(pd.Series([last_valid(ordered["county"])])).iloc[0],
                "longitude": first_valid(ordered["longitude"]),
                "latitude": first_valid(ordered["latitude"]),
                "permit_event_count": int(ordered["permit_id"].nunique()),
                "permit_issued_date": ordered["permit_issued_date"].min(),
                "completion_date": ordered["completion_date"].max(),
                "record_date": ordered["record_date"].max(),
                "operator_name": last_valid(ordered["operator_name"]),
                "well_type": last_valid(ordered["well_type"]),
                "well_use": last_valid(ordered["well_use"]),
                "well_depth_class": last_valid(ordered["well_depth_class"]),
                "permit_type": last_valid(ordered["permit_type"]),
                "well_status": last_valid(ordered["well_status"]),
                "farm_name": last_valid(ordered["farm_name"]),
                "well_number": last_valid(ordered["well_number"]),
                "marcellus_flag": clean_flag(pd.Series([last_valid(ordered["marcellus"])])).iloc[0],
                "formation_name": clean_formation(pd.Series([last_valid(ordered["formation_name"])])).iloc[0],
            }
        )
    dedup = pd.DataFrame(rows)
    dedup["longitude"] = pd.to_numeric(dedup["longitude"], errors="coerce")
    dedup["latitude"] = pd.to_numeric(dedup["latitude"], errors="coerce")
    dedup = dedup.dropna(subset=["longitude", "latitude"]).copy()
    return dedup


def build_first_life_summary(production: pd.DataFrame) -> pd.DataFrame:
    frame = production.copy()
    frame["period_start_date"] = parse_date(frame["period_start_date"])
    for col in ["gas_quantity", "oil_quantity", "water_quantity", "ngl_quantity"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = frame.sort_values(["well_api", "period_start_date"]).copy()

    nonzero = frame[frame["gas_quantity"] > 0].copy()
    first_prod = (
        nonzero.groupby("well_api", dropna=True)
        .agg(first_prod_date=("period_start_date", "min"))
        .reset_index()
    )
    frame = frame.merge(first_prod, on="well_api", how="inner")
    frame["months_since_first"] = (
        (frame["period_start_date"].dt.year - frame["first_prod_date"].dt.year) * 12
        + (frame["period_start_date"].dt.month - frame["first_prod_date"].dt.month)
    )

    summary = (
        frame.groupby("well_api", dropna=True)
        .agg(
            first_prod_date=("first_prod_date", "min"),
            latest_prod_date=("period_start_date", "max"),
            observed_months_all=("period_start_date", "nunique"),
            observed_gas_all=("gas_quantity", "sum"),
        )
        .reset_index()
    )

    for window in [12, 24]:
        subset = frame[(frame["months_since_first"] >= 0) & (frame["months_since_first"] < window)].copy()
        agg = (
            subset.groupby("well_api", dropna=True)
            .agg(
                **{
                    f"f{window}_gas": ("gas_quantity", "sum"),
                    f"f{window}_oil": ("oil_quantity", "sum"),
                    f"f{window}_water": ("water_quantity", "sum"),
                    f"f{window}_ngl": ("ngl_quantity", "sum"),
                    f"covered_months_f{window}": ("period_start_date", "nunique"),
                    f"last_month_f{window}": ("period_start_date", "max"),
                }
            )
            .reset_index()
        )
        summary = summary.merge(agg, on="well_api", how="left")
        summary[f"label_f{window}_available"] = summary[f"covered_months_f{window}"].fillna(0).ge(window)

    return summary


def neighbor_value_stats_before_date(
    target_lat: float,
    target_lon: float,
    target_date: pd.Timestamp,
    source_frame: pd.DataFrame,
    source_tree,
    date_col: str,
    lat_col: str,
    lon_col: str,
    value_col: str,
    radius_km: float,
) -> dict:
    if pd.isna(target_date) or source_tree is None or source_frame.empty:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs = source_tree.query_radius(coords_rad, r=radius_km / 6371.0088, return_distance=False)[0]
    if len(idxs) == 0:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    subset = source_frame.iloc[idxs].copy()
    subset = subset[parse_date(subset[date_col]) < target_date]
    values = pd.to_numeric(subset[value_col], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    return {
        "count": int(len(values)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
    }


def neighbor_value_stats_current(
    target_lat: float,
    target_lon: float,
    source_frame: pd.DataFrame,
    source_tree,
    value_col: str,
    radius_km: float,
) -> dict:
    if source_tree is None or source_frame.empty:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    coords_rad = np.deg2rad([[float(target_lat), float(target_lon)]])
    idxs = source_tree.query_radius(coords_rad, r=radius_km / 6371.0088, return_distance=False)[0]
    if len(idxs) == 0:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    values = pd.to_numeric(source_frame.iloc[idxs][value_col], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "median": np.nan, "p90": np.nan}
    return {
        "count": int(len(values)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
    }


def nearest_row(frame: pd.DataFrame, tree, latitude: float, longitude: float) -> pd.Series:
    coords_rad = np.deg2rad([[float(latitude), float(longitude)]])
    idx = int(tree.query(coords_rad, k=1, return_distance=False)[0][0])
    return frame.iloc[idx]


__all__ = [
    "build_balltree",
    "build_first_life_summary",
    "clean_formation",
    "clean_flag",
    "count_neighbors_before_date",
    "count_neighbors_current",
    "county_code_string",
    "deduplicate_wells",
    "load_basin_config",
    "nearest_neighbor_before_date_km",
    "nearest_neighbor_current_km",
    "nearest_row",
    "neighbor_value_stats_before_date",
    "neighbor_value_stats_current",
    "parse_date",
    "polygon_from_bbox",
    "prospect_tier",
    "write_csv",
]
