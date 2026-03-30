# Regional Multiscale Data Design (`#8`)

## Purpose

`#8` is the first data program that should move beyond the current single-tile world model.

It should introduce:

- local multiscale context,
- broader regional context,
- multiyear observation histories,
- and a benchmark design that is defensible at an ICLR / CVPR level.

This is a data-design step, not just another training ablation.

## Final Recommendation

The correct local asset is:

- collect a full **`5x5` high-resolution local field**,
- derive **`9` valid target-centered windows** from the inner `3x3`,
- use the outer `16` tiles as halo context,
- and collect one **broader low-resolution regional window** beyond the local field.

This is stronger than collecting only `3x3`, because:

- a `3x3` asset supports only one boundary-safe target,
- a `5x5` asset supports `9` valid target-centered local windows,
- later runs can reuse the same collected asset for stronger local-field supervision without recollection.

## What We Are Optimizing For

The design should satisfy four constraints:

1. It must remain scientifically interpretable.
2. It must permit clean ablations.
3. It must be collection-feasible at paper scale.
4. It must avoid leakage across space, time, and neighboring regions.

## Year Range

Recommended first paper years:

- `2019`
- `2020`
- `2021`

For decoder targets with forecasting horizons such as `+7d`, `+14d`, and `+30d`, target support should extend through:

- `2022-01-30`

That gives:

- `3` annual cycles,
- interannual variation,
- and complete end-of-period forecast targets.

## Spatial Unit

### Center Tile

Keep the current center tile:

- side length: `2560 m`
- nominal resolution: `10 m`
- nominal raster size: `256 x 256`

This remains the main local reference tile.

### Local High-Resolution Field

Collect the full local **`5x5`** field:

- `25` tiles total,
- centered on the current center tile,
- same tile size and resolution as the current local runs.

So each region sample contains:

- `1` center tile,
- `8` tiles in the inner `3x3`,
- `16` outer-ring halo tiles.

### Valid Local Training Windows

From the collected `5x5`, derive the `9` valid target-centered windows from the inner `3x3`.

Each derived local window contains:

- `1` target tile,
- `8` immediate local-support neighbors,
- `16` halo tiles.

This is the correct way to increase training units without boundary cheating.

## Larger Regional Context

On top of the local `5x5`, collect one broader low-resolution regional window centered on the same location.

Recommended regional window:

- side length: `25.6 km`
- resolution: `80 m`

This yields roughly:

- `320 x 320` low-resolution pixels

and extends materially beyond the local `12.8 km x 12.8 km` high-resolution field.

The regional window should be treated as:

- coarse contextual evidence,
- not another high-resolution peer tile.

## Local Vs Mesoscale

These are different levels and should not be conflated.

- the local `5x5` solves the immediate neighborhood problem inside one region sample,
- the broader low-resolution window provides beyond-local landscape and regime context.

At paper scale we also want a minority of **nearby center regions** to probe mesoscale coherence across adjacent local worlds. That clustered-center design is specified in [paper_scale_10k_sampling_design.md](/Users/shin/Projects/EO_WM/manuscript/paper_scale_10k_sampling_design.md).

## Coordinate System And Geometry

Neighborhood geometry must not be defined by naive latitude/longitude offsets.

Instead:

1. define offsets in meters,
2. apply those offsets geodesically or in a local projected CRS,
3. define local and regional windows in metric space,
4. convert back to WGS84 only when needed for Earth Engine calls.

The intent is metric neighborhood fidelity, not degree-based approximations.

## Sensors

Required:

- `S2`
- `S1`

Optional but strongly preferred:

- `HLS`

Recommended treatment:

- `S1/S2` use the grouped regional GEE collection path,
- `HLS` remains a separate aligned observation-event source,
- `HLS` should improve the asset where available, but should not be a hard completeness gate for the whole dataset.

## Targets And Supervision

### Daily Decoder Targets

For the first regional benchmark, keep daily targets at the **center region only**:

- `MODIS LST`
- `CHIRPS`
- `SMAP`
- optional `ERA5-Land`

This keeps the first regional claim clean:

- multiscale context improves the center prediction

rather than:

- we already supervise a fully interacting regional field.

### Static Context

Extract static context for all local tiles:

- elevation
- slope
- aspect
- WorldCover

## Collection And Processing Path

The `#8` data program should be:

1. build a canonical regional manifest,
2. derive grouped request tables,
3. collect one local high-resolution `5x5` supertile per region and one broader low-resolution regional window,
4. split the local supertile back into the `25` constituent tiles during post-processing,
5. derive the `9` valid target-centered inner-window definitions,
6. extract center-only daily targets and static context using the same `region_id` / `center_sample_id`.

This matters because the old exact-chip processor center-crops exports back to the center tile, which would destroy the local neighborhood signal. A dedicated grouped post-processing path is required.

## Data Products

The `#8` data program should produce:

1. `local_obs_events_5x5/`
2. `local_training_windows/`
3. `regional_context_obs/`
4. `hls_obs_events/`
5. `static_context/`
6. `state_targets_daily_center/`
7. `benchmarks/`

All keyed by a shared manifest.

## Manifest Design

We need one canonical manifest with:

- `region_id`
- `center_sample_id`
- `tile_id`
- `tile_role`
- `latitude`
- `longitude`
- `grid_dx_idx`
- `grid_dy_idx`
- `grid_ring_index`
- `is_valid_local_target`
- `is_inner_local_tile`
- `is_halo_tile`
- `split`

This manifest should generate:

- the grouped local `5x5` collection rows,
- the low-resolution regional-context rows,
- the local training-window membership tables,
- the static-context extraction points,
- and the center decoder-target extraction points.

## Split Design

For isolated regions, the split unit is:

- `region_id`

For clustered center regions, the split unit is:

- `cluster_id`

And all years for the same region or cluster must stay in the same split.

Recommended top-level split:

- train: `70%`
- val: `15%`
- test: `15%`

This preserves:

- spatial generalization,
- temporal generalization,
- and leakage-safe regional evaluation.
