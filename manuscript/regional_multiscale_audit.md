# Regional Multiscale Audit (`#8`)

## Reviewer-Style Findings

### 1. Collecting only four neighbors is not a durable data design

If we collect only:

- center
- north
- south
- east
- west

then any later attempt to evaluate diagonal or halo influence requires recollection.

At a CVPR / ICLR level, that is weak because the collection asset itself encodes an early architectural guess.

The better design is:

- collect full `5x5`,
- derive inner `3x3` target-centered windows later.

### 2. A `3x3` local asset is not future-proof enough

A `3x3` collection supports only one boundary-safe target: the center tile.

If later runs want:

- more than one valid local target,
- auxiliary supervision on nearby tiles,
- or explicit local interaction dynamics,

then `3x3` forces recollection.

The better design is:

- collect `5x5`,
- use the outer ring as halo,
- and supervise the inner `3x3` progressively.

### 3. The old exact-chip processor is incompatible with regional collection

Current processing in [process_earth_engine_exact_chip_exports.py](/Users/shin/Projects/EO_WM/earth_world_model/scripts/process_earth_engine_exact_chip_exports.py) always center-crops exported rasters back to `chip_size`.

That means:

- if we export a `5x5` supertile today,
- the old converter discards the surrounding tiles.

So `#8` requires:

- a grouped regional collector,
- and a grouped post-processor that splits the supertile back into `25` local tiles.

### 4. Split units must respect clustered center design

For isolated regions, the split unit is:

- `region_id`

For clustered mesoscale samples, the split unit must be:

- `cluster_id`

Anything weaker leaks nearby geography across train/val/test and is not reviewer-safe.

### 5. Center-only decoder targets are the correct first evaluation

The first regional claim should be:

- multiscale context improves the center prediction

not:

- we already supervise a fully interacting regional field.

So center-only decoder targets remain scientifically cleaner for the first regional paper.

### 6. The local and regional scales must remain separate

The local `5x5` and the broader regional window solve different problems:

- local `5x5` captures immediate neighborhood and halo context,
- larger low-resolution context captures broader landscape and regime structure.

These should be collected separately, not conflated into one giant high-resolution crop.

### 7. Geometry must be metric, not naive lat/lon offsets

For global data, neighborhood offsets must be defined in meters, not simple degree deltas.

Even if the implementation uses geodesic offsets rather than a heavyweight projection stack, the design intent must remain:

- define neighborhoods metrically
- avoid latitude-dependent distortion

### 8. `HLS` should remain optional in the first regional build

Requiring complete `HLS` coverage for all samples and timesteps would shrink the dataset and complicate the first paper unnecessarily.

`HLS` should be:

- strongly preferred,
- integrated where available,
- but not a hard completeness gate.

### 9. The cleanest local collection primitive is one grouped `5x5` supertile

For the high-resolution local context, a single grouped `5x5` collection is better than `25` separate tile requests because it gives:

- consistent temporal pairing,
- fewer requests,
- simpler downstream grouping,
- and a clean mapping to derived inner-window targets.

### 10. The paper-scale mesoscale sampler should use `3x3` center clusters

The local `5x5` already handles immediate neighborhood structure.

The clustered-center sampler serves a different role:

- modest beyond-local mesoscale overlap between nearby region centers.

For that purpose, the first paper-scale clustered design should be:

- `3x3` center clusters,
- not `2x2`,
- because `3x3` is the first symmetric mesoscale cluster geometry.

### 11. Regional post-processing must preserve both tiles and derived windows

Reviewer-level correctness depends on preserving:

- the collected local `25` tiles,
- the broader low-resolution regional context,
- and the derived `9` valid target-centered windows from the inner `3x3`.

If we only keep the grouped raw supertile without this structure, the benchmark is not actually usable for multiscale training.

## Audit Decision

The strongest first regional design is:

- collect local `5x5`,
- derive inner-`3x3` valid target windows,
- collect one broader low-resolution regional window,
- keep decoder targets center-only,
- keep `HLS` optional,
- use `3x3` clustered center groups for the paper-scale mesoscale subset,
- and split by `region_id` or `cluster_id` as appropriate.

Anything weaker is likely to be challenged by reviewers as either:

- under-designed,
- leaky,
- or not actually testing multiscale context correctly.
