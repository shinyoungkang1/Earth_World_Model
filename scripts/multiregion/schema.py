"""Unified training table schema for multi-region gas pipeline.

Defines the canonical column names, types, and validation rules so that
per-state training tables can be merged into a single dataset.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column definitions
# ---------------------------------------------------------------------------

# Metadata columns (not features, but preserved for provenance and filtering)
METADATA_COLUMNS = [
    "well_api",
    "region_id",
    "state",
    "basin",
    "county_name",
    "formation",
    "operator_name",
    "well_status",
    "spud_date",
    "first_prod_date",
    "context_date",
]

# Numeric features used in training
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_deg",
    "relief_3px_m",
    "fault_distance_km",
    "well_count_2km",
    "well_count_5km",
    "nearest_well_km",
    "permit_count_2km",
    "permit_count_5km",
    "nearest_permit_km",
    "mature_f12_count_5km",
    "mature_f12_median_gas_5km",
    "mature_f12_p90_gas_5km",
    "mature_f12_count_10km",
    "mature_f12_median_gas_10km",
    "mature_f12_p90_gas_10km",
]

# Categorical features used in training
CATEGORICAL_FEATURES = [
    "geology_map_symbol",
    "geology_age",
    "geology_lith1",
]

# Label columns
LABEL_COLUMNS = [
    "f12_gas",
    "f24_gas",
    "label_f12_available",
    "label_f24_available",
    # F12 thresholds
    "label_f12_ge_100000",
    "label_f12_ge_250000",
    "label_f12_ge_500000",
    "label_f12_ge_1000000",
    "label_f12_ge_2000000",
    # F24 thresholds
    "label_f24_ge_250000",
    "label_f24_ge_500000",
    "label_f24_ge_1000000",
    "label_f24_ge_2000000",
    "label_f24_ge_5000000",
]

ALL_COLUMNS = METADATA_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + LABEL_COLUMNS

# ---------------------------------------------------------------------------
# Per-state column renames (source name -> canonical name)
# ---------------------------------------------------------------------------

PA_COLUMN_RENAMES = {
    "latitude_decimal": "latitude",
    "longitude_decimal": "longitude",
    "well_count_2km": "well_count_2km",
    "well_count_5km": "well_count_5km",
    "nearest_well_distance_km": "nearest_well_km",
    "permit_count_2km": "permit_count_2km",
    "permit_count_5km": "permit_count_5km",
    "nearest_permit_distance_km": "nearest_permit_km",
    "county_name": "county_name",
    "formation_name": "formation",
    "geology_map_symbol": "geology_map_symbol",
    "geology_age": "geology_age",
    "geology_lith1": "geology_lith1",
}

WV_COLUMN_RENAMES = {
    "latitude": "latitude",
    "longitude": "longitude",
    "producing_well_count_2km": "well_count_2km",
    "producing_well_count_5km": "well_count_5km",
    "nearest_producing_well_distance_km": "nearest_well_km",
    "permit_count_2km": "permit_count_2km",
    "permit_count_5km": "permit_count_5km",
    "nearest_permit_distance_km": "nearest_permit_km",
    "county_code": "county_name",
    "formation_name": "formation",
    "mature_f12_neighbor_count_5km": "mature_f12_count_5km",
    "mature_f12_neighbor_median_gas_5km": "mature_f12_median_gas_5km",
    "mature_f12_neighbor_p90_gas_5km": "mature_f12_p90_gas_5km",
    "mature_f12_neighbor_count_10km": "mature_f12_count_10km",
    "mature_f12_neighbor_median_gas_10km": "mature_f12_median_gas_10km",
    "mature_f12_neighbor_p90_gas_10km": "mature_f12_p90_gas_10km",
}

# Generic rename for new states (OH, CO, TX) that use canonical names from the start
NEW_STATE_COLUMN_RENAMES = {
    "latitude": "latitude",
    "longitude": "longitude",
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_training_table(df: pd.DataFrame, region_id: str) -> list[str]:
    """Validate a training table against the unified schema.

    Returns a list of warning messages. Empty list means valid.
    """
    warnings = []

    # Check required metadata
    for col in ["well_api", "region_id", "latitude", "longitude"]:
        if col not in df.columns:
            warnings.append(f"[{region_id}] Missing required column: {col}")

    # Check numeric feature columns exist
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            warnings.append(f"[{region_id}] Missing numeric feature: {col}")

    # Check categorical feature columns exist
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            warnings.append(f"[{region_id}] Missing categorical feature: {col}")

    # Check label columns exist
    for col in ["f12_gas", "f24_gas", "label_f12_available", "label_f24_available"]:
        if col not in df.columns:
            warnings.append(f"[{region_id}] Missing label column: {col}")

    # Sanity checks on data
    if "latitude" in df.columns:
        bad_lat = df["latitude"].dropna()
        if len(bad_lat) > 0 and (bad_lat.min() < 20 or bad_lat.max() > 55):
            warnings.append(
                f"[{region_id}] Latitude range [{bad_lat.min():.2f}, {bad_lat.max():.2f}] outside expected CONUS range [20, 55]"
            )

    if "longitude" in df.columns:
        bad_lon = df["longitude"].dropna()
        if len(bad_lon) > 0 and (bad_lon.min() < -130 or bad_lon.max() > -65):
            warnings.append(
                f"[{region_id}] Longitude range [{bad_lon.min():.2f}, {bad_lon.max():.2f}] outside expected CONUS range [-130, -65]"
            )

    if "well_api" in df.columns:
        dup_count = df["well_api"].duplicated().sum()
        if dup_count > 0:
            warnings.append(f"[{region_id}] {dup_count} duplicate well_api values")

    return warnings


def align_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns per the mapping and ensure all unified columns exist (fill missing with NaN)."""
    result = df.rename(columns=rename_map)
    for col in ALL_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    return result[ALL_COLUMNS].copy()


def add_region_metadata(
    df: pd.DataFrame,
    region_id: str,
    state: str,
    basin: str,
) -> pd.DataFrame:
    """Set region_id, state, and basin columns."""
    result = df.copy()
    result["region_id"] = region_id
    result["state"] = state
    result["basin"] = basin
    return result
