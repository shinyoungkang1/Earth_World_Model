# SubTerra PA MVP Ontology v0

## Scope

This ontology defines the entity model for the first Pennsylvania gas MVP using the normalized files in `data/canonical/pa_mvp/`.

It is intentionally narrow:

- one state
- one gas-first ontology
- explicit provenance back to raw files
- a coarse prospect-cell entity for the current gas baseline output

## Canonical Sources

| Entity Family | Canonical File |
| --- | --- |
| Wells | `data/canonical/pa_mvp/wells.csv` |
| Permits | `data/canonical/pa_mvp/permits.csv` and `data/canonical/pa_mvp/permits.geojson` |
| Production | `data/canonical/pa_mvp/production.csv` |
| Bedrock polygons | `data/canonical/pa_mvp/bedrock_geology.geojson` |
| Bedrock units | `data/canonical/pa_mvp/bedrock_geology_units.csv` |
| Raster scenes | `data/canonical/pa_mvp/raster_scenes_sentinel2.csv` and `raster_scenes_landsat.csv` |
| Raster catalogs | `data/canonical/pa_mvp/raster_catalogs.csv` |
| Prospect scoring outputs | `data/derived/prospect_layers/gas_prospect_cells_v0.csv` and `gas_prospect_cells_v0.geojson` |
| Geology reference tables | `data/canonical/pa_mvp/paunits.csv`, `paage.csv`, `palith.csv`, `paref.csv`, `paref_link.csv` |
| Raw provenance | `data/canonical/pa_mvp/raw_asset_status.csv` and `raw_archive_members.csv` |

## Source Contract

- `config/source_catalog.json` is the source-to-ontology contract.
- `data/state/source_refresh_state.json` is the operational refresh state.
- `scripts/refresh_pa_mvp.py` is the default orchestrator that refreshes active sources and reruns normalization.
- manual sources can also be tracked through the same state file when run explicitly with `--source`.
- Each source catalog entry declares `refresh_strategy`, `default_mode`, `ontology_entities`, raw manifests, and canonical outputs.

## Core Entities

| Entity | Canonical key | Geometry | Notes |
| --- | --- | --- | --- |
| `Well` | `well_api` | point-like via `latitude_decimal`,`longitude_decimal` | Operational inventory record from PA DEP. |
| `Permit` | `authorization_id` | point via GeoJSON and `longitude`,`latitude` | Permit issuance record with `well_api`, `permit_issued_date`, and `spud_date`; `well_api` is not unique in this permit table. |
| `ProductionRecord` | `permit_num` + `og_production_periods_id` + `well_no` | point-like via `latitude_decimal`,`longitude_decimal` | Period-level production fact table. |
| `BedrockGeologyPolygon` | `objectid` | polygon | Spatial geology feature keyed to a geology unit symbol. |
| `BedrockGeologyUnit` | `map_symbol` | none | Unit dictionary for polygons and downstream feature engineering. |
| `RawAsset` | `raw_path` | none | Downloaded raw file with status and manifest lineage. |
| `ArchiveMember` | `archive_path` + `member_name` | none | Member-level inventory for ZIP/KMZ/DS9 bundles. |
| `RasterTile` | `dataset` + `tile` | raster footprint | DEM tiles are tracked through the raw manifest. |
| `GeophysicsBundle` | `output_path` | archive | DS9 gravity and magnetic grids currently remain bundle-level, not feature-level. |
| `FaultArchive` | `output_path` | archive | Fault datasets currently remain archive-level until shapefile parsing is added. |
| `RasterScene` | `scene_id` | bbox footprint | Scene-level satellite catalog item from Sentinel-2 or Landsat. |
| `RasterCatalog` | `source_id` + `collection` | none | Catalog-level refresh record for a satellite source. |
| `RasterAsset` | `scene_id` + `asset_kind` | image footprint via scene bbox | Materialized local quicklook or other downloaded raster asset tied to a scene. |
| `RasterComposite` | `composite_id` | raster | Derived preview mosaic or later analysis-ready composite built from multiple scenes. |
| `AcquisitionOrderDraft` | `draft_name` | optional AOI only | Non-submitted order specification for downloading commercial imagery products. |
| `ProspectCell` | `cell_id` | polygon | Coarse scored grid cell for the current gas baseline prospect surface. |

## Relationships

| From | Relation | To | Join Rule |
| --- | --- | --- | --- |
| `Permit` | identifies | `Well` | `permit.well_api = well.well_api` |
| `ProductionRecord` | references | `Well` | `production.permit_num = well.well_api` where the PA API/permit formatting matches |
| `ProductionRecord` | references | `Permit` | `production.permit_num = permit.well_api` |
| `ProductionRecord` | references | `Well` | `production.ogo_num = well.ogo_id` as a secondary linkage |
| `BedrockGeologyPolygon` | typed_as | `BedrockGeologyUnit` | `polygon.map_symbol = unit.map_symbol` |
| `Well` | falls_within | `BedrockGeologyPolygon` | spatial join in EPSG:4326 |
| `Permit` | falls_within | `BedrockGeologyPolygon` | spatial join in EPSG:4326 |
| `RasterTile` | covers | `Well`, `Permit`, `BedrockGeologyPolygon` | spatial footprint intersection |
| `RasterScene` | belongs_to | `RasterCatalog` | `scene.source_id = catalog.source_id` |
| `RasterAsset` | derived_from | `RasterScene` | `asset.scene_id = scene.scene_id` |
| `RasterScene` | covers | `Well`, `Permit`, `BedrockGeologyPolygon` | spatial bbox or geometry intersection |
| `RasterComposite` | derived_from | `RasterScene` | derivation lineage once composites exist |
| `AcquisitionOrderDraft` | references | `RasterScene` | item IDs in the draft resolve to catalog scene IDs |
| `ProspectCell` | falls_within | `BedrockGeologyPolygon` | cell center is assigned to the covering geology polygon where available |
| `ProspectCell` | nearest_context | `Well` | current v0 scoring assigns county and municipality from the nearest unconventional well |
| `RawAsset` | provenance_for | all canonical entities | file lineage from raw manifests |
| `ArchiveMember` | contained_in | `FaultArchive`, `GeophysicsBundle` | archive membership |

## Normalization Rules

- Spatial coordinates are normalized to EPSG:4326 in canonical point and polygon outputs.
- Dates are normalized to ISO `YYYY-MM-DD`.
- Boolean flags are normalized to `true` / `false` / null where possible.
- Raw source field names are converted to `snake_case`, but domain semantics are preserved.
- Raw archives are not unpacked into feature tables unless the environment can parse them reliably.

## Current Constraints

- `well_api` is unique in the current well inventory, but not in the permit table. `authorization_id` remains the canonical permit key, while `well_api` is treated as a join field.
- `well_api` and `permit_num` are treated as the same practical join key in Pennsylvania, but this should be validated before cross-state expansion.
- Fault and geophysics datasets are only normalized to archive metadata. They still need shapefile/grid parsers before they become first-class spatial entities.
- The direct USGS `PAgeol_lcc.zip` download is invalid in this workspace. The ontology therefore treats the PASDA-hosted DCNR bedrock polygons as the active geology geometry source.
- Sentinel-2 quicklook thumbnails can now be materialized as local `RasterAsset` records, and a low-fidelity preview mosaic can be derived from them.
- The current quicklook mosaic is a browse layer, not an analysis-ready remote-sensing product.
- Planet PSScene catalog search is now available through the Data API, and the pipeline can generate non-submitted draft orders for later commercial download.
- Landsat assets remain catalog-only for now because the current CDSE item metadata exposes auth-gated product downloads rather than a directly reusable public quicklook URL in this workspace.
- Planet basemaps are represented as a manual catalog/status source until the account has the necessary basemap/mosaic entitlement for the target AOI.
- `ProspectCell` is currently a coarse model surface built from static/location features only. It should be treated as an internal ranking layer, not as a direct gas-detection product.

## Next Extensions

1. Add `Operator` as a first-class entity keyed by `client_id` with resolved name variants.
2. Add `Basin` and `Play` entities once the MVP basin is selected.
3. Add `ProspectArea` aggregations and basin-specific AOI entities after the first scored-cell workflow is stable.
4. Promote `FaultFeature` and `GeophysicsGrid` from archive-level assets to parsed spatial entities.
5. Promote the quicklook pipeline from browse imagery to analysis-ready composites once raster tooling and authenticated asset access are in place.
