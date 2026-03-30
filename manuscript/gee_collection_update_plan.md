# GEE Collection Update Plan

## Goal

Update the data-collection stack from a single-year local chip collector into a year-sharded program that can support the next large run.

For the first paper-scale regional build, the target is now the `10k` design recorded in [paper_scale_10k_sampling_design.md](/Users/shin/Projects/EO_WM/manuscript/paper_scale_10k_sampling_design.md):

- `8,002` isolated centers
- `1,998` clustered centers arranged as `222` mesoscale `3x3` center clusters
- local `5x5` high-resolution collection per center
- broader `25.6 km` low-resolution regional context

The right collection structure is:

1. multiyear `S1/S2` exact-chip collection,
2. `HLS` observation-event integration,
3. static context extraction,
4. daily decoder-target extraction.

## Why The Current Collector Is Not Enough

The live exact-chip collector currently does:

- `S1/S2`
- one anchor year
- central tile only

That is enough for the current local yearly corpus, but not for the next multiyear data package.

## Implemented Update

### A. Multiyear S1/S2 Collection Orchestration

The collection should remain year-sharded rather than trying to issue one giant multiyear Earth Engine request.

We now support a multiyear orchestration layer that:

- takes one fixed `sample_id` / location manifest,
- loops over years,
- launches the existing exact-chip yearly collector once per year,
- writes per-year stage roots and logs,
- optionally uploads per-sample outputs to year-specific GCS prefixes.

Conceptually:

\[
\text{collect\_s1s2}(i, \mathcal{Y}) = \{x_{i,t}^{(s2)}, x_{i,t}^{(s1)} \mid t \in \cup_{y \in \mathcal{Y}} \mathcal{T}_y \}
\]

where each year \(y\) is collected independently but shares the same sample geometry.

### B. Static Context

Static context is extracted once per `sample_id`:

- elevation
- slope
- aspect
- WorldCover class

This forms a one-time table:

\[
c_i = \text{static}(i)
\]

that can later condition the local world model.

### C. HLS

`HLS` should not be forced into the exact same weekly S1/S2 collector.

Instead:

- reuse or stage the HLS chip/index artifacts,
- convert them into `obs_events`,
- align by `sample_id` and timestamp.

### D. Daily Decoder Targets

Daily targets remain separate from chip collection:

- `MODIS LST`
- `CHIRPS`
- `SMAP`
- optional `ERA5-Land`

They are aligned later by `sample_id` and date.

## Practical Dataset Structure

The intended next large-run data package is:

- `obs_events_s1s2_multiyear/`
- `obs_events_hls/`
- `static_context/`
- `state_targets_daily/`
- `benchmarks/`

All pieces share the same base sample manifest:

- `sample_id`
- `latitude`
- `longitude`

## What This Enables

This is still a local world-model dataset, but it fixes the main collection limitation of the current pipeline:

- local inputs become multiyear,
- HLS is aligned as a proper sensor source,
- static priors are available,
- decoder targets remain daily.

The grouped `5x5` plus broader regional-context path is now the regional extension layer on top of this base multiyear stack. The mesoscale clustered-center sampling design sits above that collection layer and should be applied through the regional manifest builder.
