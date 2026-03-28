#!/usr/bin/env python3
"""Extract daily ERA5-Land forcing targets for decoder benchmarking."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from gee_decoder_target_utils import add_common_args, run_daily_extraction


PRODUCT_NAME = "era5_land_daily_v1"
DATASET_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract daily ERA5-Land forcing targets from Earth Engine.")
    add_common_args(parser, default_scale_meters=11132.0)
    return parser.parse_args()


def _constant_band(ee, value: float, name: str):
    return ee.Image.constant([float(value)]).rename([name]).toFloat()


def _masked_bands(ee, band_names: list[str]):
    return ee.Image.constant([0.0] * len(band_names)).rename(band_names).toFloat().updateMask(ee.Image.constant(0))


def build_daily_image_server(ee, request_date: date):
    start = ee.Date(request_date.isoformat())
    end = start.advance(1, "day")
    daily = ee.ImageCollection(DATASET_ID).filterDate(start, end)
    image_count = daily.size()

    def _actual_image():
        image = ee.Image(daily.first())
        temperature_2m = image.select("temperature_2m").subtract(273.15).rename("temperature_2m_c")
        total_precip = image.select("total_precipitation_sum").multiply(1000.0).rename("total_precip_mm")
        soil_temp = image.select("soil_temperature_level_1").subtract(273.15).rename("soil_temperature_c")
        surface_pressure = image.select("surface_pressure").rename("surface_pressure_pa")
        shortwave = image.select("surface_solar_radiation_downwards_sum").rename("shortwave_radiation_j_m2")
        wind_u = image.select("u_component_of_wind_10m").rename("wind_u_10m_m_s")
        wind_v = image.select("v_component_of_wind_10m").rename("wind_v_10m_m_s")
        target_valid = _constant_band(ee, 1.0, "target_valid")
        image_count_band = ee.Image.constant(image_count).rename("image_count").toFloat()
        return (
            temperature_2m
            .addBands(total_precip)
            .addBands(soil_temp)
            .addBands(surface_pressure)
            .addBands(shortwave)
            .addBands(wind_u)
            .addBands(wind_v)
            .addBands(target_valid)
            .addBands(image_count_band)
        )

    empty = _masked_bands(
        ee,
        [
            "temperature_2m_c",
            "total_precip_mm",
            "soil_temperature_c",
            "surface_pressure_pa",
            "shortwave_radiation_j_m2",
            "wind_u_10m_m_s",
            "wind_v_10m_m_s",
        ],
    ).addBands(_constant_band(ee, 0.0, "target_valid")).addBands(_constant_band(ee, 0.0, "image_count"))
    return ee.Image(ee.Algorithms.If(image_count.gt(0), _actual_image(), empty))


def main() -> None:
    args = parse_args()
    run_daily_extraction(
        product_name=PRODUCT_NAME,
        dataset_id=DATASET_ID,
        args=args,
        build_daily_image_server=build_daily_image_server,
    )


if __name__ == "__main__":
    main()
