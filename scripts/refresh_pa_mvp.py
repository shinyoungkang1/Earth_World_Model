#!/usr/bin/env python3
"""Stateful refresh runner for the current gas MVP source catalog."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def should_run(source: dict, repo_root: Path, force_static: bool) -> bool:
    mode = source.get("default_mode", "always")
    if mode == "always":
        return True
    if force_static:
        return True
    if mode == "if_missing":
        return not all((repo_root / rel).exists() for rel in source.get("raw_manifests", []))
    return False


def derive_source_marker(source_id: str, repo_root: Path) -> dict:
    if source_id == "pa_dep_inventory":
        manifest = load_json(repo_root / "data/raw/pa_dep/download_manifest.json")
        item = next(x for x in manifest["downloads"] if x["dataset"] == "pa_dep_unconventional_well_inventory")
        csv_path = Path(item["output_path"])
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "row_count": count_csv_rows(csv_path),
            "raw_path": str(csv_path),
        }

    if source_id == "pa_dep_production":
        manifest = load_json(repo_root / "data/raw/pa_dep/download_manifest.json")
        items = [x for x in manifest["downloads"] if x["dataset"] == "pa_dep_production"]
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "period_count": len(items),
            "period_ids": [x["period_id"] for x in items],
            "actions": {x["period_id"]: x.get("action") for x in items},
        }

    if source_id == "pa_dep_permits":
        manifest = load_json(repo_root / "data/raw/arcgis_download_manifest.json")
        item = manifest["downloads"][0]
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "feature_count": item["feature_count"],
            "raw_path": item["output_path"],
            "action": item.get("action"),
        }

    if source_id == "pa_dcnr_bedrock_geology":
        manifest = load_json(repo_root / "data/raw/dcnr/download_manifest_bedrock_geology_chunks.json")
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "feature_count": manifest["feature_count"],
            "chunk_count": manifest["chunk_count"],
            "action": manifest.get("action"),
        }

    if source_id == "usgs_dem_arc1":
        manifest = load_json(repo_root / "data/raw/usgs/download_manifest_arc1.json")
        elevation_items = [x for x in manifest["downloads"] if x["dataset"] == "usgs_elevation_1"]
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "tile_count": len(manifest["tiles"]),
            "tiles": manifest["tiles"],
            "download_actions": {
                x["tile"]: {"tif": x.get("tif_action"), "xml": x.get("xml_action")} for x in elevation_items
            },
        }

    if source_id == "usgs_geophysics":
        manifest = load_json(repo_root / "data/raw/usgs/download_manifest_geophysics.json")
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "downloads": {
                x["dataset"]: {"size_bytes": x["size_bytes"], "action": x.get("action")}
                for x in manifest["downloads"]
            },
        }

    if source_id in {"satellite_landsat_catalog", "satellite_sentinel2_catalog"}:
        manifest = load_json(
            repo_root / "data/raw/satellite_catalog" / source_id / "download_manifest.json"
        )
        return {
            "checked_at_utc": manifest["checked_at_utc"],
            "collection": manifest["collection"],
            "new_item_count": manifest["new_item_count"],
            "total_catalog_count": manifest["total_catalog_count"],
            "last_seen_datetime": manifest["last_seen_datetime"],
        }

    if source_id == "satellite_sentinel2_quicklooks":
        manifest = load_json(
            repo_root / "data/raw/satellite_assets/satellite_sentinel2_quicklooks/download_manifest.json"
        )
        return {
            "checked_at_utc": manifest["checked_at_utc"],
            "selected_scene_count": manifest["selected_scene_count"],
            "downloaded_count": manifest["downloaded_count"],
            "reused_count": manifest["reused_count"],
            "composite_count": len(manifest.get("composites", [])),
        }

    if source_id == "planet_psscene_catalog":
        manifest = load_json(
            repo_root / "data/raw/planet_catalog/planet_psscene_catalog/download_manifest.json"
        )
        return {
            "checked_at_utc": manifest["checked_at_utc"],
            "fetched_item_count": manifest["fetched_item_count"],
            "new_item_count": manifest["new_item_count"],
            "total_catalog_count": manifest["total_catalog_count"],
            "latest_acquired": manifest.get("latest_acquired"),
            "draft_item_count": manifest.get("draft_item_count"),
        }

    if source_id == "planet_basemaps_catalog":
        manifest = load_json(
            repo_root / "data/raw/satellite_catalog/planet_basemaps_catalog/download_manifest.json"
        )
        return {
            "checked_at_utc": manifest["checked_at_utc"],
            "configured": manifest["configured"],
            "authorized": manifest["authorized"],
            "mosaic_count_sample": manifest.get("mosaic_count_sample", 0),
            "http_statuses": manifest.get("http_statuses", []),
        }

    if source_id == "wv_dep_horizontal_wells":
        manifest = load_json(repo_root / "data/raw/wv_dep/download_manifest.json")
        wells = next(x for x in manifest["downloads"] if x["dataset"] == "wv_dep_horizontal_wells")
        laterals = next(x for x in manifest["downloads"] if x["dataset"] == "wv_dep_horizontal_laterals")
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "wells_feature_count": wells["feature_count"],
            "laterals_feature_count": laterals["feature_count"],
            "wells_raw_path": wells["output_path"],
            "laterals_raw_path": laterals["output_path"],
        }

    if source_id == "wv_dep_h6a_production":
        manifest = load_json(repo_root / "data/raw/wv_dep/download_manifest.json")
        items = [x for x in manifest["downloads"] if x["dataset"] == "wv_dep_h6a_production"]
        return {
            "downloaded_at_utc": manifest["downloaded_at_utc"],
            "year_count": len(items),
            "years": [x["year"] for x in items],
            "actions": {str(x["year"]): x.get("action") for x in items},
        }

    return {}


def derive_normalization_marker(normalizer_argv: list[str], repo_root: Path) -> dict:
    script_name = Path(normalizer_argv[-1]).name
    if script_name == "normalize_pa_mvp.py":
        registry = load_json(repo_root / "data/canonical/pa_mvp/source_registry.json")
    elif script_name == "normalize_wv_mvp.py":
        registry = load_json(repo_root / "data/canonical/wv_mvp/source_registry.json")
    else:
        return {"normalizer_argv": normalizer_argv, "marker": None}
    return {
        "normalizer_argv": normalizer_argv,
        "normalized_at_utc": registry["normalized_at_utc"],
        "datasets": {x["dataset"]: x for x in registry["outputs"]},
    }


def run_command(argv: list[str], repo_root: Path) -> str:
    proc = subprocess.run(
        argv,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/shin/Mineral_Gas_Locator",
        help="Repository root used to resolve configs and state.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh all active sources using catalog default behavior.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Specific source_id to refresh. May be passed multiple times.",
    )
    parser.add_argument(
        "--force-static",
        action="store_true",
        help="Also run sources whose default mode is if_missing.",
    )
    args = parser.parse_args()

    if not args.all and not args.source:
        raise SystemExit("Use --all or --source <source_id>.")

    repo_root = Path(args.repo_root)
    catalog = load_json(repo_root / "config/source_catalog.json")
    selected_ids = set(args.source)

    state_path = repo_root / "data/state/source_refresh_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prior_state = load_json(state_path) if state_path.exists() else {"sources": {}}
    state = {
        "updated_at_utc": utc_now(),
        "catalog_version": catalog["catalog_version"],
        "sources": prior_state.get("sources", {}),
    }

    normalizers_to_run: dict[tuple[str, ...], list[str]] = {}
    for source in catalog["sources"]:
        source_id = source["source_id"]
        selected = args.all or source_id in selected_ids
        if not selected:
            continue

        entry = {
            "status": source["status"],
            "refresh_strategy": source["refresh_strategy"],
            "ontology_entities": source["ontology_entities"],
            "default_mode": source["default_mode"],
            "last_attempted_at_utc": utc_now(),
        }

        if source["status"] not in {"active", "manual"}:
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "planned"
            state["sources"][source_id] = entry
            continue

        if source["status"] == "manual" and source_id not in selected_ids:
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "manual_only"
            if all((repo_root / rel).exists() for rel in source.get("raw_manifests", [])):
                entry["last_marker"] = derive_source_marker(source_id, repo_root)
            state["sources"][source_id] = entry
            continue

        if not should_run(source, repo_root, args.force_static) and source_id not in selected_ids:
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "default_if_missing_already_satisfied"
            entry["last_marker"] = derive_source_marker(source_id, repo_root)
            state["sources"][source_id] = entry
            continue

        stdout = run_command(source["trigger_argv"], repo_root)
        entry["run_status"] = "success"
        entry["last_successful_at_utc"] = utc_now()
        entry["trigger_argv"] = source["trigger_argv"]
        entry["last_marker"] = derive_source_marker(source_id, repo_root)
        entry["stdout_excerpt"] = stdout[-2000:]
        state["sources"][source_id] = entry
        normalizer_argv = source.get("normalizer_argv")
        if source["status"] == "active" and normalizer_argv:
            normalizers_to_run[tuple(normalizer_argv)] = normalizer_argv

    if normalizers_to_run:
        state["normalizations"] = {}
        for normalizer_argv in normalizers_to_run.values():
            run_command(normalizer_argv, repo_root)
            marker = derive_normalization_marker(normalizer_argv, repo_root)
            state["normalizations"][Path(normalizer_argv[-1]).stem] = marker
            if Path(normalizer_argv[-1]).name == "normalize_pa_mvp.py":
                state["normalization"] = marker

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"refresh failed: {exc}", file=sys.stderr)
        raise
