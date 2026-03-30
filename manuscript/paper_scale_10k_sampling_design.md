# Paper-Scale `10k` Regional Sampling Design

## Decision

For the first serious multiscale paper dataset, the right scale is:

- **`10,000` center regions**
- **`3` years**
- **local high-resolution `5x5` collection**
- **`9` valid target-centered inner-window targets per region-year**
- **one broader low-resolution regional-context window**

Do **not** start with `100k`.

At the new `5x5` scale, `100k` is an infrastructure program, not a first paper dataset.

## Why `10k` Is Enough

With the new local asset:

- each region-year gives `25` local tiles,
- each region-year yields `9` valid target-centered local windows,
- each target window has:
  - `1` target tile,
  - `8` immediate local-support neighbors,
  - `16` halo tiles,
  - broader low-resolution regional context.

So for:

- `10k` centers
- `3` years

we get:

\[
10{,}000 \times 3 = 30{,}000 \text{ region-years}
\]

\[
30{,}000 \times 9 = 270{,}000 \text{ valid target-centered local windows}
\]

That is already a very large and scientifically meaningful dataset.

## Final Paper-Scale Sampling Goal

The paper-scale dataset should be:

- `10,000` center regions total
- `8,002` globally stratified isolated centers
- `1,998` clustered centers

The clustered part should not be random noise. It is there to give us limited evidence about beyond-neighborhood spatial coherence.

## Cluster Verdict

For the clustered-center subset, the correct first paper design is:

- **`3x3` center clusters**
- not `2x2`

Reason:

- the local `5x5` already handles the immediate neighborhood problem,
- the clustered-center subset is for mesoscale overlap beyond one local field,
- and `3x3` is the first symmetric mesoscale cluster geometry.

## Sampling Principle

We need two levels of sampling:

1. **global center diversity**
2. **local and mesoscale structure**

So the correct design is:

- sample center regions broadly across the Earth for diversity,
- but collect a structured `5x5` + broader-context field around each center,
- and reserve a minority of centers in explicit mesoscale clusters.

This is stronger than:

- pure global random centers only, and
- stronger than collecting giant contiguous mosaics everywhere.

## Exact `10k` Split

### A. Isolated Backbone: `8,000` Centers

These are the main paper dataset.

They should be:

- globally distributed,
- stratified,
- and separated enough that they are not just many nearly-duplicate neighboring fields.

Recommended target strata:

- continent / macro-region
- major climate bucket
- dominant land-cover bucket
- elevation band

The backbone exists to guarantee:

- global diversity,
- climate diversity,
- land-cover diversity,
- terrain diversity.

### B. Mesoscale Clusters: `1,998` Centers

These are for beyond-neighborhood scaling.

Use:

- `222` cluster anchors
- each anchor expands to a **`3x3` center cluster**
- `9` centers per cluster

So:

\[
222 \times 9 = 1{,}998 \text{ clustered centers}
\]

The remaining centers are isolated:

\[
10{,}000 - 1{,}998 = 8{,}002
\]

## Cluster Geometry

Each clustered center should use:

- center-to-center spacing of **`12.8 km`**

Why `12.8 km`:

- one local high-resolution `5x5` field is `12.8 km x 12.8 km`
- so neighboring clustered centers are **just beyond one local field width**
- this gives us a clean mesoscale design:
  - local `5x5` fields do not fully duplicate each other,
  - broader regional windows overlap strongly,
  - we can test beyond-local coherence without wasting too much storage on near-duplicates.

So the clustered design is:

- local field: `12.8 km`
- center spacing: `12.8 km`
- cluster layout: `3x3` centers
- broader low-resolution regional field: **beyond `12.8 km`**, recommended `25.6 km`

This means one mesoscale cluster has an effective union footprint of about:

\[
3 \times 12.8 \text{ km} = 38.4 \text{ km}
\]

which is roughly a `15x15` local-tile union field.

## Years

Recommended first paper years:

- **`2019`**
- **`2020`**
- **`2021`**

And for daily decoder targets with forecasting horizons:

- collect target support through **`2022-01-30`**

Reason:

- `3` annual cycles are enough to support:
  - seasonal recurrence,
  - interannual variation,
  - decoder forecasting,
  - world-model claims stronger than a single-year study.

## Region-Level Counts

For the full `10k x 3 year` paper dataset:

- region-years:

\[
10{,}000 \times 3 = 30{,}000
\]

- local tile sequences:

\[
30{,}000 \times 25 = 750{,}000
\]

- broader regional-context sequences:

\[
30{,}000 \times 1 = 30{,}000
\]

- valid target-centered windows:

\[
30{,}000 \times 9 = 270{,}000
\]

These are already paper-scale numbers.

## Decoder Target Counts

If daily decoder targets are center-only and we use:

- `2019-01-01` through `2022-01-30`

then the number of daily rows per product is roughly:

\[
10{,}000 \times 1{,}126 \approx 11.26 \text{ million rows}
\]

per target product.

That is large but manageable.

## Data Products To Collect

### A. Observation Inputs

- `S2`
- `S1`
- `HLS` where available

Important:

- `S1/S2` use the grouped regional GEE path
- `HLS` stays a separate event/index integration path
- `HLS` does **not** currently use the grouped `5x5` collector

### B. Static Context

- elevation
- slope
- aspect
- WorldCover

### C. Daily Decoder Targets

- MODIS LST
- CHIRPS
- SMAP
- optional ERA5-Land

## Split Policy

Never split individual centers from the same mesoscale cluster across train/val/test.

The split unit should be:

- isolated center id for isolated regions
- **cluster id** for clustered regions

And all years for the same center or cluster must stay in the same split.

Recommended top-level split:

- train: `70%`
- val: `15%`
- test: `15%`

Split balancing should still preserve the same broad strata:

- geography
- climate
- land cover
- elevation

## Storage Reality

At the upgraded `5x5` scale, raw storage is very large.

A rough first-order estimate from the measured pilot implies:

- about **`2.14 GB` raw per region-year**

So for the `10k x 3 year` paper dataset:

\[
30{,}000 \times 2.14 \text{ GB} \approx 64.2 \text{ TB raw}
\]

That is too large to keep casually in Standard GCS forever, and far too large for a single external `8 TB` SSD archive.

## Storage Policy

The right storage policy is:

1. use fast local scratch / attached SSD for active collection and processing
2. process immediately into compact sequence artifacts
3. keep:
   - processed sequence artifacts,
   - manifests,
   - indices,
   - decoder/static tables
4. delete raw weekly TIFF exports after validation unless explicitly needed

So the long-term archive should be:

- processed data
- not full raw grouped exports

## Sharding Plan

Do **not** try to collect all `10k` at once.

Use shards of roughly:

- `250` to `500` centers per shard

Why:

- `500` centers across `3` years is on the order of a few TB of raw temporary data
- that fits a rolling scratch workflow much better than trying to hold tens of TB at once

Recommended unit:

- one shard = `500` centers
- collect/process/delete raw
- then move to the next shard

For `10k` centers, that is:

- `20` shards if using `500` centers each

## What The Clustered Fraction Buys Us

The clustered `20%` does **not** replace the global dataset.

It gives us:

- limited evidence about beyond-neighborhood scaling
- overlapping broader-context fields
- local-to-mesoscale consistency checks

without turning the whole paper into a contiguous regional simulation project.

That is the right level of ambition for the first multiscale paper.

## Final Recommendation

The correct paper-scale data program is:

- `10k` centers, not `100k`
- `3` years: `2019-2021`
- `8,002` isolated stratified centers
- `1,998` clustered centers in `222` mesoscale `3x3` clusters
- local `5x5` high-resolution asset
- `9` valid target-centered windows per region-year
- broader low-resolution regional context at `25.6 km`
- center-only decoder targets
- process raw immediately and do **not** archive all raw TIFFs long term

This is the strongest balance of:

- scientific defensibility,
- dataset size,
- collection feasibility,
- and storage cost.
