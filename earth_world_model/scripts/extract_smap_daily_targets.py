#!/usr/bin/env python3
"""Extract daily SMAP soil-moisture targets for decoder benchmarking."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from gee_decoder_target_utils import add_common_args, run_daily_extraction


PRODUCT_NAME = "smap_daily_v1"
DATASET_ID = "NASA/SMAP/SPL4SMGP/008"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract daily SMAP targets from Earth Engine.")
    add_common_args(parser, default_scale_meters=11000.0)
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
        sm_surface = daily.select("sm_surface").mean().rename("soil_moisture_surface")
        sm_rootzone = daily.select("sm_rootzone").mean().rename("soil_moisture_rootzone")
        target_valid = _constant_band(ee, 1.0, "target_valid")
        image_count_band = ee.Image.constant(image_count).rename("image_count").toFloat()
        return sm_surface.addBands(sm_rootzone).addBands(target_valid).addBands(image_count_band)

    empty = _masked_bands(ee, ["soil_moisture_surface", "soil_moisture_rootzone"]).addBands(
        _constant_band(ee, 0.0, "target_valid")
    ).addBands(_constant_band(ee, 0.0, "image_count"))
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
