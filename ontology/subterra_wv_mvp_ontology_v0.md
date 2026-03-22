# SubTerra WV MVP Ontology v0

## Scope

This ontology defines the first West Virginia gas MVP slice using the normalized files in `data/canonical/wv_mvp/`.

It is intentionally narrow:

- one state
- gas-first
- horizontal-well focused
- explicit provenance back to official WVDEP raw assets

## Canonical Sources

| Entity Family | Canonical File |
| --- | --- |
| Wells | `data/canonical/wv_mvp/wells.csv` |
| Permits | `data/canonical/wv_mvp/permits.csv` and `permits.geojson` |
| Laterals | `data/canonical/wv_mvp/laterals.csv` and `laterals.geojson` |
| Production | `data/canonical/wv_mvp/production.csv` |
| Production assets | `data/canonical/wv_mvp/production_assets.csv` |
| Raw provenance | `data/raw/wv_dep/download_manifest.json` |

## Source Contract

- `config/source_catalog.json` is the shared source-to-ontology contract.
- `data/state/source_refresh_state.json` stores the latest WV source markers and normalization marker.
- `scripts/download_wv_dep.py` acquires WV horizontal wells, laterals, and H6A production files.
- `scripts/normalize_wv_mvp.py` converts the raw assets into canonical tables.

## Core Entities

| Entity | Canonical key | Geometry | Notes |
| --- | --- | --- | --- |
| `Well` | `well_api` | point via `longitude`,`latitude` | Horizontal-only WVDEP well inventory from the official ArcGIS oil/gas service. |
| `Permit` | `permit_id` | point | Derived subset of `Well` where `well_status` is `Permit Issued` or `Permit Application`. |
| `LateralFeature` | `permit_id` + `well_api` | line | Simplified horizontal lateral geometry from the WVDEP ArcGIS oil/gas service. |
| `ProductionRecord` | `well_api` + `production_year` + `production_month` | none | Monthly long-form production fact table derived from WVDEP H6A files. |
| `ProductionAsset` | `year` + `source_file` | none | Asset inventory for the downloaded H6A Excel/ZIP files. |
| `RawAsset` | `output_path` | none | Downloaded raw asset recorded in the WV raw manifest. |

## Relationships

| From | Relation | To | Join Rule |
| --- | --- | --- | --- |
| `Permit` | identifies | `Well` | `permit.well_api = well.well_api` when available |
| `LateralFeature` | references | `Well` | `lateral.well_api = well.well_api` |
| `ProductionRecord` | references | `Well` | `production.well_api = well.well_api` |
| `ProductionAsset` | provenance_for | `ProductionRecord` | `source_file` lineage |
| `RawAsset` | provenance_for | all WV canonical entities | raw manifest lineage |

## Normalization Rules

- ArcGIS geometries are normalized to GeoJSON in EPSG:4326.
- Raw field names are converted to `snake_case`.
- WV H6A spreadsheets are converted into monthly long-form rows.
- For partial-year H6A assets such as `Q3`, months after the reported quarter are written as null, not zero.

## Current Constraints

- This WV slice is horizontal-well focused and does not attempt to cover the full legacy WV well universe.
- Permit geometry is currently derived from the horizontal-well service subset rather than a dedicated permit layer.
- The WV production table is built only from H6A horizontal-well assets starting in 2018.
- The canonical WV tables are not yet wired into the frontend or training pipeline.

## Next Extensions

1. Add WV geology and terrain features aligned to the same canonical keys.
2. Add a basin/play AOI for the WV gas slice.
3. Join WV production and lateral context into a second-state feature table.
4. Extend the cross-state ontology with shared `State`, `Play`, and `ProspectArea` entities.
