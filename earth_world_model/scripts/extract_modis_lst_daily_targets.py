#!/usr/bin/env python3
"""Extract daily MODIS LST targets for decoder benchmarking."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from gee_decoder_target_utils import add_common_args, run_daily_extraction


PRODUCT_NAME = "modis_lst_daily_v1"
DATASET_ID = "MODIS/061/MOD11A1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract daily MODIS LST targets from Earth Engine.")
    add_common_args(parser, default_scale_meters=1000.0)
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
        lst_day = (
            daily.select("LST_Day_1km")
            .mean()
            .multiply(0.02)
            .subtract(273.15)
            .rename("surface_temp_c")
        )
        lst_night = (
            daily.select("LST_Night_1km")
            .mean()
            .multiply(0.02)
            .subtract(273.15)
            .rename("surface_temp_night_c")
        )
        qc_day = daily.select("QC_Day").mode().rename("qc_day")
        qc_night = daily.select("QC_Night").mode().rename("qc_night")
        day_quality_ok = qc_day.bitwiseAnd(3).lte(1).rename("day_quality_ok")
        night_quality_ok = qc_night.bitwiseAnd(3).lte(1).rename("night_quality_ok")
        target_valid = day_quality_ok.rename("target_valid")
        image_count_band = ee.Image.constant(image_count).rename("image_count").toFloat()
        return (
            lst_day
            .addBands(lst_night)
            .addBands(qc_day.toFloat())
            .addBands(qc_night.toFloat())
            .addBands(day_quality_ok.toFloat())
            .addBands(night_quality_ok.toFloat())
            .addBands(target_valid.toFloat())
            .addBands(image_count_band)
        )

    empty = (
        _masked_bands(
            ee,
            [
                "surface_temp_c",
                "surface_temp_night_c",
                "qc_day",
                "qc_night",
                "day_quality_ok",
                "night_quality_ok",
            ],
        )
        .addBands(_constant_band(ee, 0.0, "target_valid"))
        .addBands(_constant_band(ee, 0.0, "image_count"))
    )
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
