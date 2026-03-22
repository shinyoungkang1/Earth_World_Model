#!/usr/bin/env python3
"""Generalized stateful refresh runner for all states in the gas pipeline.

Extends the PA-MVP refresh logic (scripts/refresh_pa_mvp.py) with support for
OH, CO, and TX source markers and normalizers while maintaining full backwards
compatibility with the existing PA and WV refresh behavior.

Usage:
    python scripts/refresh.py --all
    python scripts/refresh.py --source pa_dep_inventory --source oh_odnr_wells
    python scripts/refresh.py --all --force-static
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def count_csv_rows(path: Path) -> int:
    """Count data rows (excluding header) in a CSV file."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def should_run(source: dict, repo_root: Path, force_static: bool) -> bool:
    """Decide whether a source should be refreshed based on its default_mode."""
    mode = source.get("default_mode", "always")
    if mode == "always":
        return True
    if force_static:
        return True
    if mode == "if_missing":
        return not all(
            (repo_root / rel).exists() for rel in source.get("raw_manifests", [])
        )
    return False


# ---------------------------------------------------------------------------
# Source marker derivation
# ---------------------------------------------------------------------------

def _derive_marker_pa_dep_inventory(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/pa_dep/download_manifest.json")
    item = next(
        x for x in manifest["downloads"]
        if x["dataset"] == "pa_dep_unconventional_well_inventory"
    )
    csv_path = Path(item["output_path"])
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "row_count": count_csv_rows(csv_path),
        "raw_path": str(csv_path),
    }


def _derive_marker_pa_dep_production(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/pa_dep/download_manifest.json")
    items = [x for x in manifest["downloads"] if x["dataset"] == "pa_dep_production"]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "period_count": len(items),
        "period_ids": [x["period_id"] for x in items],
        "actions": {x["period_id"]: x.get("action") for x in items},
    }


def _derive_marker_pa_dep_permits(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/arcgis_download_manifest.json")
    item = manifest["downloads"][0]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "feature_count": item["feature_count"],
        "raw_path": item["output_path"],
        "action": item.get("action"),
    }


def _derive_marker_pa_dcnr_bedrock_geology(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root / "data/raw/dcnr/download_manifest_bedrock_geology_chunks.json"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "feature_count": manifest["feature_count"],
        "chunk_count": manifest["chunk_count"],
        "action": manifest.get("action"),
    }


def _derive_marker_usgs_dem_arc1(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/usgs/download_manifest_arc1.json")
    elevation_items = [
        x for x in manifest["downloads"] if x["dataset"] == "usgs_elevation_1"
    ]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "tile_count": len(manifest["tiles"]),
        "tiles": manifest["tiles"],
        "download_actions": {
            x["tile"]: {"tif": x.get("tif_action"), "xml": x.get("xml_action")}
            for x in elevation_items
        },
    }


def _derive_marker_usgs_geophysics(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root / "data/raw/usgs/download_manifest_geophysics.json"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "downloads": {
            x["dataset"]: {"size_bytes": x["size_bytes"], "action": x.get("action")}
            for x in manifest["downloads"]
        },
    }


def _derive_marker_usgs_dem_national(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root / "data/raw/usgs/download_manifest_dem_national.json"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "tile_count": len(manifest.get("tiles", [])),
        "tiles": manifest.get("tiles", []),
        "download_count": len(manifest.get("downloads", [])),
    }


def _derive_marker_satellite_catalog(source_id: str, repo_root: Path) -> dict:
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


def _derive_marker_sentinel2_quicklooks(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root
        / "data/raw/satellite_assets/satellite_sentinel2_quicklooks/download_manifest.json"
    )
    return {
        "checked_at_utc": manifest["checked_at_utc"],
        "selected_scene_count": manifest["selected_scene_count"],
        "downloaded_count": manifest["downloaded_count"],
        "reused_count": manifest["reused_count"],
        "composite_count": len(manifest.get("composites", [])),
    }


def _derive_marker_planet_psscene(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root
        / "data/raw/planet_catalog/planet_psscene_catalog/download_manifest.json"
    )
    return {
        "checked_at_utc": manifest["checked_at_utc"],
        "fetched_item_count": manifest["fetched_item_count"],
        "new_item_count": manifest["new_item_count"],
        "total_catalog_count": manifest["total_catalog_count"],
        "latest_acquired": manifest.get("latest_acquired"),
        "draft_item_count": manifest.get("draft_item_count"),
    }


def _derive_marker_planet_basemaps(repo_root: Path) -> dict:
    manifest = load_json(
        repo_root
        / "data/raw/satellite_catalog/planet_basemaps_catalog/download_manifest.json"
    )
    return {
        "checked_at_utc": manifest["checked_at_utc"],
        "configured": manifest["configured"],
        "authorized": manifest["authorized"],
        "mosaic_count_sample": manifest.get("mosaic_count_sample", 0),
        "http_statuses": manifest.get("http_statuses", []),
    }


def _derive_marker_wv_dep_horizontal_wells(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/wv_dep/download_manifest.json")
    wells = next(
        x for x in manifest["downloads"] if x["dataset"] == "wv_dep_horizontal_wells"
    )
    laterals = next(
        x
        for x in manifest["downloads"]
        if x["dataset"] == "wv_dep_horizontal_laterals"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "wells_feature_count": wells["feature_count"],
        "laterals_feature_count": laterals["feature_count"],
        "wells_raw_path": wells["output_path"],
        "laterals_raw_path": laterals["output_path"],
    }


def _derive_marker_wv_dep_h6a_production(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/wv_dep/download_manifest.json")
    items = [
        x for x in manifest["downloads"] if x["dataset"] == "wv_dep_h6a_production"
    ]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "year_count": len(items),
        "years": [x["year"] for x in items],
        "actions": {str(x["year"]): x.get("action") for x in items},
    }


def _derive_marker_oh_odnr_wells(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/oh_odnr/download_manifest.json")
    wells = next(
        x for x in manifest["downloads"] if x["dataset"] == "oh_odnr_horizontal_wells"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "feature_count": wells.get("feature_count", wells.get("row_count")),
        "raw_path": wells["output_path"],
        "action": wells.get("action"),
    }


def _derive_marker_oh_odnr_production(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/oh_odnr/download_manifest.json")
    items = [
        x for x in manifest["downloads"] if x["dataset"] == "oh_odnr_production"
    ]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "file_count": len(items),
        "actions": {
            x.get("output_path", str(i)): x.get("action")
            for i, x in enumerate(items)
        },
    }


def _derive_marker_co_cogcc_wells(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/co_cogcc/download_manifest.json")
    wells = next(
        x for x in manifest["downloads"] if x["dataset"] == "co_cogcc_wells"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "feature_count": wells.get("feature_count", wells.get("row_count")),
        "raw_path": wells["output_path"],
        "action": wells.get("action"),
    }


def _derive_marker_co_cogcc_production(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/co_cogcc/download_manifest.json")
    items = [
        x for x in manifest["downloads"] if x["dataset"] == "co_cogcc_production"
    ]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "file_count": len(items),
        "actions": {
            x.get("output_path", str(i)): x.get("action")
            for i, x in enumerate(items)
        },
    }


def _derive_marker_tx_rrc_wells(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/tx_rrc/download_manifest.json")
    wells = next(
        x for x in manifest["downloads"] if x["dataset"] == "tx_rrc_horizontal_wells"
    )
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "feature_count": wells.get("feature_count", wells.get("row_count")),
        "raw_path": wells["output_path"],
        "action": wells.get("action"),
    }


def _derive_marker_tx_rrc_production(repo_root: Path) -> dict:
    manifest = load_json(repo_root / "data/raw/tx_rrc/download_manifest.json")
    items = [
        x for x in manifest["downloads"] if x["dataset"] == "tx_rrc_production"
    ]
    return {
        "downloaded_at_utc": manifest["downloaded_at_utc"],
        "file_count": len(items),
        "actions": {
            x.get("output_path", str(i)): x.get("action")
            for i, x in enumerate(items)
        },
    }


def derive_source_marker(source_id: str, repo_root: Path) -> dict:
    """Derive a provenance marker for a source based on its download manifest.

    Returns a dict summarizing what was downloaded, or empty dict if unknown.
    """
    # PA sources
    if source_id == "pa_dep_inventory":
        return _derive_marker_pa_dep_inventory(repo_root)
    if source_id == "pa_dep_production":
        return _derive_marker_pa_dep_production(repo_root)
    if source_id == "pa_dep_permits":
        return _derive_marker_pa_dep_permits(repo_root)
    if source_id == "pa_dcnr_bedrock_geology":
        return _derive_marker_pa_dcnr_bedrock_geology(repo_root)

    # USGS sources
    if source_id == "usgs_dem_arc1":
        return _derive_marker_usgs_dem_arc1(repo_root)
    if source_id == "usgs_geophysics":
        return _derive_marker_usgs_geophysics(repo_root)
    if source_id == "usgs_dem_national":
        return _derive_marker_usgs_dem_national(repo_root)

    # Satellite sources
    if source_id in {"satellite_landsat_catalog", "satellite_sentinel2_catalog"}:
        return _derive_marker_satellite_catalog(source_id, repo_root)
    if source_id == "satellite_sentinel2_quicklooks":
        return _derive_marker_sentinel2_quicklooks(repo_root)

    # Planet sources
    if source_id == "planet_psscene_catalog":
        return _derive_marker_planet_psscene(repo_root)
    if source_id == "planet_basemaps_catalog":
        return _derive_marker_planet_basemaps(repo_root)

    # WV sources
    if source_id == "wv_dep_horizontal_wells":
        return _derive_marker_wv_dep_horizontal_wells(repo_root)
    if source_id == "wv_dep_h6a_production":
        return _derive_marker_wv_dep_h6a_production(repo_root)

    # OH sources
    if source_id == "oh_odnr_wells":
        return _derive_marker_oh_odnr_wells(repo_root)
    if source_id == "oh_odnr_production":
        return _derive_marker_oh_odnr_production(repo_root)

    # CO sources
    if source_id == "co_cogcc_wells":
        return _derive_marker_co_cogcc_wells(repo_root)
    if source_id == "co_cogcc_production":
        return _derive_marker_co_cogcc_production(repo_root)

    # TX sources
    if source_id == "tx_rrc_wells":
        return _derive_marker_tx_rrc_wells(repo_root)
    if source_id == "tx_rrc_production":
        return _derive_marker_tx_rrc_production(repo_root)

    return {}


# ---------------------------------------------------------------------------
# Normalization marker derivation
# ---------------------------------------------------------------------------

# Mapping from normalizer script name to its source registry path
_NORMALIZER_REGISTRIES: dict[str, str] = {
    "normalize_pa_mvp.py": "data/canonical/pa_mvp/source_registry.json",
    "normalize_wv_mvp.py": "data/canonical/wv_mvp/source_registry.json",
    "normalize_ohio.py": "data/canonical/oh_mvp/source_registry.json",
    "normalize_colorado.py": "data/canonical/co_mvp/source_registry.json",
    "normalize_texas.py": "data/canonical/tx_mvp/source_registry.json",
}


def derive_normalization_marker(normalizer_argv: list[str], repo_root: Path) -> dict:
    """Derive a provenance marker for a normalizer run.

    Reads the source_registry.json produced by the normalizer to capture
    what datasets were written and when.
    """
    script_name = Path(normalizer_argv[-1]).name
    registry_rel = _NORMALIZER_REGISTRIES.get(script_name)

    if registry_rel is None:
        return {"normalizer_argv": normalizer_argv, "marker": None}

    registry = load_json(repo_root / registry_rel)
    return {
        "normalizer_argv": normalizer_argv,
        "normalized_at_utc": registry["normalized_at_utc"],
        "datasets": {x["dataset"]: x for x in registry["outputs"]},
    }


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def run_command(argv: list[str], repo_root: Path) -> str:
    """Run a subprocess command and return its stdout."""
    proc = subprocess.run(
        argv,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for the generalized refresh runner."""
    parser = argparse.ArgumentParser(
        description="Stateful refresh runner for the gas pipeline source catalog (all states).",
    )
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
    state: dict = {
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

        entry: dict = {
            "status": source["status"],
            "refresh_strategy": source["refresh_strategy"],
            "ontology_entities": source["ontology_entities"],
            "default_mode": source["default_mode"],
            "last_attempted_at_utc": utc_now(),
        }

        # Skip planned/future sources
        if source["status"] not in {"active", "manual"}:
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "planned"
            state["sources"][source_id] = entry
            continue

        # Skip manual sources unless explicitly requested
        if source["status"] == "manual" and source_id not in selected_ids:
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "manual_only"
            if all(
                (repo_root / rel).exists()
                for rel in source.get("raw_manifests", [])
            ):
                entry["last_marker"] = derive_source_marker(source_id, repo_root)
            state["sources"][source_id] = entry
            continue

        # Skip if_missing sources that are already satisfied
        if (
            not should_run(source, repo_root, args.force_static)
            and source_id not in selected_ids
        ):
            entry["run_status"] = "skipped"
            entry["skip_reason"] = "default_if_missing_already_satisfied"
            entry["last_marker"] = derive_source_marker(source_id, repo_root)
            state["sources"][source_id] = entry
            continue

        # Execute the source trigger
        stdout = run_command(source["trigger_argv"], repo_root)
        entry["run_status"] = "success"
        entry["last_successful_at_utc"] = utc_now()
        entry["trigger_argv"] = source["trigger_argv"]
        entry["last_marker"] = derive_source_marker(source_id, repo_root)
        entry["stdout_excerpt"] = stdout[-2000:]
        state["sources"][source_id] = entry

        # Queue normalizer if applicable
        normalizer_argv = source.get("normalizer_argv")
        if source["status"] == "active" and normalizer_argv:
            normalizers_to_run[tuple(normalizer_argv)] = normalizer_argv

    # Run each unique normalizer once
    if normalizers_to_run:
        state["normalizations"] = {}
        for normalizer_argv in normalizers_to_run.values():
            run_command(normalizer_argv, repo_root)
            marker = derive_normalization_marker(normalizer_argv, repo_root)
            state["normalizations"][Path(normalizer_argv[-1]).stem] = marker
            # Backwards compatibility: also write top-level "normalization"
            # for the PA normalizer, matching refresh_pa_mvp.py behavior
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
