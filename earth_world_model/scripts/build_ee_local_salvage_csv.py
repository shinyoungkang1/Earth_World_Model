#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CSV of locally complete EE samples that are still missing from GCS."
    )
    parser.add_argument("--raw-root", required=True, help="Local raw sample root containing ee_batch_* dirs.")
    parser.add_argument("--master-csv", required=True, help="Master sample CSV with sample_id,latitude,longitude.")
    parser.add_argument("--gcs-root", required=True, help="GCS raw prefix, e.g. gs://bucket/prefix/raw/")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    return parser.parse_args()


def list_gcs_ids(gcs_root: str) -> set[str]:
    proc = subprocess.run(
        ["gcloud", "storage", "ls", gcs_root],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Failed to list {gcs_root}")
    return {line.rstrip("/").split("/")[-1] for line in proc.stdout.splitlines() if line.strip()}


def is_complete_payload(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    chunk_results = payload.get("chunk_results") or []
    return bool(chunk_results) and all(int(item.get("status_code") or 0) == 200 for item in chunk_results)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    master_csv = Path(args.master_csv)
    output_csv = Path(args.output_csv)

    gcs_ids = list_gcs_ids(args.gcs_root)
    local_complete_ids: set[str] = set()
    for sample_dir in sorted(raw_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        payload_path = sample_dir / f"{sample_id}.json"
        if payload_path.exists() and is_complete_payload(payload_path) and sample_id not in gcs_ids:
            local_complete_ids.add(sample_id)

    rows: list[dict[str, str]] = []
    with master_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sample_id"] in local_complete_ids:
                rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "latitude", "longitude"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"gcs_ids={len(gcs_ids)}")
    print(f"local_complete_not_in_gcs={len(local_complete_ids)}")
    print(f"csv_rows={len(rows)}")
    print(output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
