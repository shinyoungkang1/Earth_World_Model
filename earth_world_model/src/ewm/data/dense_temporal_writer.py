from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc


def _string_array(values: list[str]) -> np.ndarray:
    max_len = max((len(str(value)) for value in values), default=1)
    dtype = f"<U{max_len}"
    return np.asarray([str(value) for value in values], dtype=dtype)


def _json_string_array(values: list[Any]) -> np.ndarray:
    encoded = [json.dumps(value, separators=(",", ":")) for value in values]
    return _string_array(encoded)


def write_dense_temporal_zarr_shard(
    shard_path: Path,
    records: list[dict[str, Any]],
    *,
    chunk_sequences: int | None = None,
    compressor_clevel: int = 5,
) -> list[dict[str, Any]]:
    if not records:
        return []

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    max_frames = max(int(record["frame_count"]) for record in records)
    sequence_count = len(records)
    s2_bands = int(records[0]["s2"].shape[1])
    s1_bands = int(records[0]["s1"].shape[1])
    chip_size = int(records[0]["s2"].shape[2])
    mask_height = int(records[0]["s2_valid_mask"].shape[1])
    mask_width = int(records[0]["s2_valid_mask"].shape[2])

    s2 = np.full((sequence_count, max_frames, s2_bands, chip_size, chip_size), np.nan, dtype=np.float32)
    s1 = np.full((sequence_count, max_frames, s1_bands, chip_size, chip_size), np.nan, dtype=np.float32)
    frame_mask = np.zeros((sequence_count, max_frames), dtype=bool)
    s2_frame_mask = np.zeros((sequence_count, max_frames), dtype=bool)
    s1_frame_mask = np.zeros((sequence_count, max_frames), dtype=bool)
    s2_valid_mask = np.zeros((sequence_count, max_frames, mask_height, mask_width), dtype=bool)
    s1_valid_mask = np.zeros((sequence_count, max_frames, mask_height, mask_width), dtype=bool)
    day_of_year = np.full((sequence_count, max_frames), -1, dtype=np.int16)
    bin_index = np.full((sequence_count, max_frames), -1, dtype=np.int16)
    frame_count = np.zeros((sequence_count,), dtype=np.int16)
    paired_frame_count = np.zeros((sequence_count,), dtype=np.int16)
    anchor_year = np.zeros((sequence_count,), dtype=np.int16)
    latitude = np.zeros((sequence_count,), dtype=np.float32)
    longitude = np.zeros((sequence_count,), dtype=np.float32)

    sequence_ids: list[str] = []
    sample_ids: list[str] = []
    search_group_ids: list[str] = []
    frame_metadata: list[Any] = []

    zarr_index_rows: list[dict[str, Any]] = []
    for row_idx, record in enumerate(records):
        steps = int(record["frame_count"])
        s2[row_idx, :steps] = record["s2"]
        s1[row_idx, :steps] = record["s1"]
        frame_mask[row_idx, :steps] = record["frame_mask"]
        s2_frame_mask[row_idx, :steps] = record["s2_frame_mask"]
        s1_frame_mask[row_idx, :steps] = record["s1_frame_mask"]
        s2_valid_mask[row_idx, :steps] = record["s2_valid_mask"]
        s1_valid_mask[row_idx, :steps] = record["s1_valid_mask"]
        day_of_year[row_idx, :steps] = record["day_of_year"]
        bin_index[row_idx, :steps] = record["bin_index"]
        frame_count[row_idx] = steps
        paired_frame_count[row_idx] = int(record["paired_frame_count"])
        anchor_year[row_idx] = int(record["anchor_year"])
        latitude[row_idx] = float(record["latitude"])
        longitude[row_idx] = float(record["longitude"])
        sequence_ids.append(str(record["sequence_id"]))
        sample_ids.append(str(record["sample_id"]))
        search_group_ids.append(str(record.get("search_group_id") or ""))
        frame_metadata.append(record.get("frame_metadata", []))

        zarr_index_rows.append(
            {
                "sequence_id": str(record["sequence_id"]),
                "sample_id": str(record["sample_id"]),
                "search_group_id": str(record.get("search_group_id") or ""),
                "anchor_year": int(record["anchor_year"]),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "zarr_shard_path": str(shard_path),
                "row_in_shard": int(row_idx),
                "frame_count": steps,
                "paired_frame_count": int(record["paired_frame_count"]),
                "frame_metadata_json": json.dumps(record.get("frame_metadata", [])),
            }
        )

    sequence_chunk = min(sequence_count, chunk_sequences or max(1, min(32, sequence_count)))
    compressor = Blosc(cname="zstd", clevel=int(compressor_clevel), shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(shard_path), mode="w")
    root.attrs.update(
        {
            "schema_version": 1,
            "sequence_count": int(sequence_count),
            "max_frames": int(max_frames),
            "s2_bands": int(s2_bands),
            "s1_bands": int(s1_bands),
            "chip_size": int(chip_size),
        }
    )

    root.create_dataset("s2", data=s2, chunks=(sequence_chunk, 1, s2_bands, chip_size, chip_size), compressor=compressor)
    root.create_dataset("s1", data=s1, chunks=(sequence_chunk, 1, s1_bands, chip_size, chip_size), compressor=compressor)
    root.create_dataset("frame_mask", data=frame_mask, chunks=(sequence_chunk, max_frames), compressor=compressor)
    root.create_dataset("s2_frame_mask", data=s2_frame_mask, chunks=(sequence_chunk, max_frames), compressor=compressor)
    root.create_dataset("s1_frame_mask", data=s1_frame_mask, chunks=(sequence_chunk, max_frames), compressor=compressor)
    root.create_dataset(
        "s2_valid_mask",
        data=s2_valid_mask,
        chunks=(sequence_chunk, 1, mask_height, mask_width),
        compressor=compressor,
    )
    root.create_dataset(
        "s1_valid_mask",
        data=s1_valid_mask,
        chunks=(sequence_chunk, 1, mask_height, mask_width),
        compressor=compressor,
    )
    root.create_dataset("day_of_year", data=day_of_year, chunks=(sequence_chunk, max_frames), compressor=compressor)
    root.create_dataset("bin_index", data=bin_index, chunks=(sequence_chunk, max_frames), compressor=compressor)
    root.create_dataset("frame_count", data=frame_count, chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("paired_frame_count", data=paired_frame_count, chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("anchor_year", data=anchor_year, chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("latitude", data=latitude, chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("longitude", data=longitude, chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("sequence_id", data=_string_array(sequence_ids), chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("sample_id", data=_string_array(sample_ids), chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("search_group_id", data=_string_array(search_group_ids), chunks=(sequence_chunk,), compressor=compressor)
    root.create_dataset("frame_metadata_json", data=_json_string_array(frame_metadata), chunks=(sequence_chunk,), compressor=compressor)
    return zarr_index_rows
