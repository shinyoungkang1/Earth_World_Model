"""Shared first-life label computation for all regions.

Two methods:
  - day_prorated: for PA-style period data with explicit start/end dates
  - month_based:  for WV/OH/CO/TX-style monthly production rows

Both return the same output schema so downstream code is method-agnostic.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Output schema (every method returns a DataFrame with these columns)
# ---------------------------------------------------------------------------
LABEL_COLUMNS = [
    "well_api",
    "first_prod_date",
    "f12_gas",
    "f24_gas",
    "covered_days_f12",
    "covered_days_f24",
    "label_f12_available",
    "label_f24_available",
]

F12_THRESHOLDS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000]
F24_THRESHOLDS = [250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# ---------------------------------------------------------------------------
# Method 1: Day-prorated (PA-style period data)
# ---------------------------------------------------------------------------

def _compute_day_prorated(
    well_apis: pd.Series,
    production: pd.DataFrame,
    well_api_col: str = "permit_num",
    start_col: str = "production_period_start_date",
    end_col: str = "production_period_end_date",
    gas_col: str = "gas_quantity",
    scope_col: str | None = "production_scope",
    scope_value: str | None = "unconventional",
    f12_coverage_threshold: int = 330,
    f24_coverage_threshold: int = 695,
) -> pd.DataFrame:
    """Prorated cumulative gas over first 365/730 days from first production.

    Extracted from run_prithvi_gas_pipeline_v1.py:189-263.
    """
    prod = production.copy()
    if scope_col and scope_value:
        prod = prod[prod[scope_col] == scope_value].copy()
    prod = prod[prod[well_api_col].isin(well_apis)].copy()
    prod[start_col] = _parse_date(prod[start_col])
    prod[end_col] = _parse_date(prod[end_col])
    prod[gas_col] = pd.to_numeric(prod[gas_col], errors="coerce").fillna(0.0)

    rows = []
    for well_api, frame in prod.groupby(well_api_col, sort=False):
        frame = frame.sort_values(start_col)
        first_prod = frame[start_col].min()
        if pd.isna(first_prod):
            continue

        for horizon_days, horizon_name in [(365, "f12"), (730, "f24")]:
            window_end = first_prod + pd.Timedelta(days=horizon_days - 1)
            overlap_start = frame[start_col].clip(lower=first_prod)
            overlap_end = frame[end_col].clip(upper=window_end)
            overlap_days = (overlap_end - overlap_start).dt.days + 1
            period_days = (frame[end_col] - frame[start_col]).dt.days + 1
            valid = overlap_days.gt(0) & period_days.gt(0)
            covered_days = float(overlap_days.where(valid, 0).sum())
            cum_gas = float(
                (
                    frame[gas_col]
                    * (overlap_days.where(valid, 0) / period_days.where(valid, 1))
                )
                .fillna(0.0)
                .sum()
            )
            rows.append(
                {
                    "well_api": well_api,
                    "horizon": horizon_name,
                    "first_prod_date": first_prod,
                    "covered_days": covered_days,
                    "cum_gas": cum_gas,
                }
            )

    if not rows:
        empty = pd.DataFrame(columns=LABEL_COLUMNS)
        for api in well_apis:
            empty = pd.concat(
                [empty, pd.DataFrame([{"well_api": api}])],
                ignore_index=True,
            )
        return empty

    summary = pd.DataFrame(rows)
    wide = summary.pivot(
        index="well_api", columns="horizon", values=["first_prod_date", "covered_days", "cum_gas"]
    )
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()

    labels = pd.DataFrame({"well_api": well_apis}).merge(wide, on="well_api", how="left")
    labels["first_prod_date"] = _parse_date(labels.get("first_prod_date_f12"))
    labels["f12_gas"] = labels.get("cum_gas_f12", pd.NA)
    labels["f24_gas"] = labels.get("cum_gas_f24", pd.NA)
    labels["covered_days_f12"] = labels.get("covered_days_f12", 0)
    labels["covered_days_f24"] = labels.get("covered_days_f24", 0)
    labels["label_f12_available"] = labels["covered_days_f12"].fillna(0).ge(f12_coverage_threshold)
    labels["label_f24_available"] = labels["covered_days_f24"].fillna(0).ge(f24_coverage_threshold)

    return labels[LABEL_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Method 2: Month-based (WV / OH / CO / TX style)
# ---------------------------------------------------------------------------

def _compute_month_based(
    well_apis: pd.Series,
    production: pd.DataFrame,
    well_api_col: str = "well_api",
    period_col: str = "period_start_date",
    gas_col: str = "gas_quantity",
    f12_months: int = 12,
    f24_months: int = 24,
) -> pd.DataFrame:
    """Monthly window labels anchored on first non-zero gas month.

    Extracted from wv_gas_common.py:91-140.
    """
    frame = production.copy()
    frame[period_col] = _parse_date(frame[period_col])
    frame[gas_col] = pd.to_numeric(frame[gas_col], errors="coerce").fillna(0.0)
    frame = frame[frame[well_api_col].isin(well_apis)].copy()
    frame = frame.sort_values([well_api_col, period_col])

    nonzero = frame[frame[gas_col] > 0].copy()
    first_prod = (
        nonzero.groupby(well_api_col, dropna=True)
        .agg(first_prod_date=(period_col, "min"))
        .reset_index()
    )
    frame = frame.merge(first_prod, on=well_api_col, how="inner")
    frame["months_since_first"] = (
        (frame[period_col].dt.year - frame["first_prod_date"].dt.year) * 12
        + (frame[period_col].dt.month - frame["first_prod_date"].dt.month)
    )

    summary = (
        frame.groupby(well_api_col, dropna=True)
        .agg(first_prod_date=("first_prod_date", "min"))
        .reset_index()
    )

    for window, months in [("f12", f12_months), ("f24", f24_months)]:
        subset = frame[
            (frame["months_since_first"] >= 0) & (frame["months_since_first"] < months)
        ].copy()
        agg = (
            subset.groupby(well_api_col, dropna=True)
            .agg(
                **{
                    f"{window}_gas": (gas_col, "sum"),
                    f"covered_months_{window}": (period_col, "nunique"),
                }
            )
            .reset_index()
        )
        summary = summary.merge(agg, on=well_api_col, how="left")
        # Convert covered months to approximate days for unified schema
        summary[f"covered_days_{window}"] = summary[f"covered_months_{window}"].fillna(0) * 30.44
        summary[f"label_{window}_available"] = summary[f"covered_months_{window}"].fillna(0).ge(months)

    summary = summary.rename(columns={well_api_col: "well_api"})
    labels = pd.DataFrame({"well_api": well_apis}).merge(summary, on="well_api", how="left")
    return labels[LABEL_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def compute_first_life_labels(
    well_apis: pd.Series,
    production: pd.DataFrame,
    method: str = "day_prorated",
    **kwargs,
) -> pd.DataFrame:
    """Compute first-life F12/F24 labels using the specified method.

    Parameters
    ----------
    well_apis : pd.Series
        Series of well API numbers to compute labels for.
    production : pd.DataFrame
        Production records.
    method : str
        'day_prorated' (PA-style) or 'month_based' (WV/OH/CO/TX-style).
    **kwargs
        Forwarded to the underlying method.

    Returns
    -------
    pd.DataFrame
        Columns: well_api, first_prod_date, f12_gas, f24_gas,
                 covered_days_f12, covered_days_f24,
                 label_f12_available, label_f24_available
    """
    if method == "day_prorated":
        return _compute_day_prorated(well_apis, production, **kwargs)
    if method == "month_based":
        return _compute_month_based(well_apis, production, **kwargs)
    raise ValueError(f"Unknown label method: {method!r}. Use 'day_prorated' or 'month_based'.")


def add_threshold_labels(
    labels: pd.DataFrame,
    f12_thresholds: list[int] | None = None,
    f24_thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Add binary threshold label columns (label_f12_ge_*, label_f24_ge_*)."""
    df = labels.copy()
    for threshold in (f12_thresholds or F12_THRESHOLDS):
        df[f"label_f12_ge_{threshold}"] = (df["f12_gas"] >= threshold).where(
            df["label_f12_available"], pd.NA
        )
    for threshold in (f24_thresholds or F24_THRESHOLDS):
        df[f"label_f24_ge_{threshold}"] = (df["f24_gas"] >= threshold).where(
            df["label_f24_available"], pd.NA
        )
    return df
