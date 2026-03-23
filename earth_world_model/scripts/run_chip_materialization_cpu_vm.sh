#!/usr/bin/env bash
# =============================================================================
# Materialize S1+S2 satellite chips for all wells across all regions.
#
# This script is designed to run on a cheap GCP CPU VM (e2-standard-8).
# It builds a full well cohort, then downloads and processes satellite
# imagery from Microsoft Planetary Computer (free, no auth needed).
#
# Usage:
#   # On the VM after setup:
#   bash earth_world_model/scripts/run_chip_materialization_cpu_vm.sh
#
#   # With custom regions:
#   REGIONS="swpa_core pa_northeast wv_horizontal" \
#     bash earth_world_model/scripts/run_chip_materialization_cpu_vm.sh
#
#   # With sharding (run on multiple VMs):
#   SHARD_INDEX=0 SHARD_COUNT=4 bash earth_world_model/scripts/run_chip_materialization_cpu_vm.sh
#   SHARD_INDEX=1 SHARD_COUNT=4 bash earth_world_model/scripts/run_chip_materialization_cpu_vm.sh
#   ...
#
# Environment variables:
#   REPO_ROOT           - Project root (default: auto-detect)
#   REGIONS             - Space-separated region IDs (default: all 4 regions)
#   MAX_WELLS_PER_REGION - 0 = no cap (default: 0 = all wells)
#   SHARD_INDEX         - This VM's shard index (default: 0)
#   SHARD_COUNT         - Total number of shards/VMs (default: 1)
#   GCS_OUTPUT_URI      - GCS path to sync chips (default: auto from .env or hardcoded)
#   COHORT_NAME         - Output cohort filename stem (default: auto-generated)
#   SPLIT_MODE          - temporal | leave-one-state-out | leave-one-basin-out (default: temporal)
#   HOLDOUT_DATE        - Temporal split date (default: 2020-01-01)
#   SKIP_COHORT_BUILD   - Set to 1 to skip cohort build (reuse existing)
#   SKIP_GCS_SYNC       - Set to 1 to skip uploading to GCS
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Regions
REGIONS="${REGIONS:-swpa_core pa_northeast wv_horizontal oh_utica}"
MAX_WELLS_PER_REGION="${MAX_WELLS_PER_REGION:-0}"
RANDOM_STATE="${RANDOM_STATE:-42}"

# Split config
SPLIT_MODE="${SPLIT_MODE:-temporal}"
HOLDOUT_DATE="${HOLDOUT_DATE:-2020-01-01}"
TASK_TYPE="${TASK_TYPE:-regression}"
TARGET_COLUMN="${TARGET_COLUMN:-f12_gas}"

# Sharding
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

# GCS
GCS_BUCKET="${GCS_BUCKET:-omois-earth-world-model-phase2-20260320-11728}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:-gs://$GCS_BUCKET/earth_world_model/data/features/multiregion}"

# Output naming
REGION_TAG=$(echo "$REGIONS" | tr ' ' '_')
WELL_TAG="all"
if [ "$MAX_WELLS_PER_REGION" -gt 0 ]; then
  WELL_TAG="${MAX_WELLS_PER_REGION}per"
fi
COHORT_NAME="${COHORT_NAME:-phase4_${REGION_TAG}_${WELL_TAG}_v1}"

SKIP_COHORT_BUILD="${SKIP_COHORT_BUILD:-0}"
SKIP_GCS_SYNC="${SKIP_GCS_SYNC:-0}"

# Paths
DATA_DIR="$REPO_ROOT/data/features/multiregion"
COHORT_PATH="$DATA_DIR/${COHORT_NAME}.parquet"
COHORT_META="$DATA_DIR/${COHORT_NAME}_metadata.json"
CHIP_OUTPUT_DIR="$DATA_DIR/${COHORT_NAME}_ewm_s2s1"

echo "============================================================"
echo "EWM Chip Materialization"
echo "============================================================"
echo "  Repo root:    $REPO_ROOT"
echo "  Regions:      $REGIONS"
echo "  Max wells/region: $MAX_WELLS_PER_REGION (0=all)"
echo "  Cohort:       $COHORT_NAME"
echo "  Shard:        $SHARD_INDEX of $SHARD_COUNT"
echo "  Split:        $SPLIT_MODE (holdout: $HOLDOUT_DATE)"
echo "  GCS output:   $GCS_OUTPUT_URI"
echo "============================================================"

# ── Step 1: Download training tables from GCS if not local ──
echo ""
echo "Step 1: Ensuring training tables are available..."
for region in $REGIONS; do
  region_dir="$REPO_ROOT/data/features/$region"
  if [ ! -d "$region_dir" ] || [ -z "$(ls "$region_dir"/*.csv 2>/dev/null)" ]; then
    echo "  Downloading $region from GCS..."
    mkdir -p "$region_dir"
    gsutil -m cp "gs://$GCS_BUCKET/earth_world_model/data/features/$region/*" "$region_dir/" 2>&1 | tail -2
  else
    echo "  $region: already local"
  fi
done

# Also need canonical data for OH if building its training table
if echo "$REGIONS" | grep -q "oh_utica"; then
  oh_canonical="$REPO_ROOT/data/canonical/oh_mvp"
  if [ ! -d "$oh_canonical" ]; then
    echo "  Downloading OH canonical data..."
    mkdir -p "$oh_canonical"
    gsutil -m cp "gs://$GCS_BUCKET/earth_world_model/data/canonical/oh_mvp/*" "$oh_canonical/" 2>&1 | tail -2
  fi
fi

# ── Step 2: Build cohort ──
if [ "$SKIP_COHORT_BUILD" != "1" ]; then
  echo ""
  echo "Step 2: Building cohort ($COHORT_NAME)..."
  REGION_ARGS=""
  for region in $REGIONS; do
    REGION_ARGS="$REGION_ARGS --regions $region"
  done

  $PYTHON_BIN "$REPO_ROOT/scripts/multiregion/build_phase4_cohort_v1.py" \
    --repo-root "$REPO_ROOT" \
    $REGION_ARGS \
    --max-wells-per-region "$MAX_WELLS_PER_REGION" \
    --task-type "$TASK_TYPE" \
    --target-column "$TARGET_COLUMN" \
    --split-mode "$SPLIT_MODE" \
    --holdout-date "$HOLDOUT_DATE" \
    --random-state "$RANDOM_STATE" \
    --output-name "$COHORT_NAME"

  echo "  Cohort saved: $COHORT_PATH"
else
  echo ""
  echo "Step 2: Skipping cohort build (SKIP_COHORT_BUILD=1)"
  if [ ! -f "$COHORT_PATH" ]; then
    echo "  ERROR: Cohort not found at $COHORT_PATH"
    exit 1
  fi
fi

# Upload cohort to GCS
if [ "$SKIP_GCS_SYNC" != "1" ]; then
  echo "  Syncing cohort to GCS..."
  gsutil cp "$COHORT_PATH" "$GCS_OUTPUT_URI/" 2>&1 | tail -1
  gsutil cp "$COHORT_META" "$GCS_OUTPUT_URI/" 2>&1 | tail -1
fi

# ── Step 3: Materialize chips ──
echo ""
echo "Step 3: Materializing S1+S2 chips (shard $SHARD_INDEX of $SHARD_COUNT)..."
$PYTHON_BIN "$REPO_ROOT/scripts/materialize_ewm_s2s1_chips_v1.py" \
  --repo-root "$REPO_ROOT" \
  --cohort-path "$COHORT_PATH" \
  --output-dir "$CHIP_OUTPUT_DIR" \
  --imagery-anchor-mode year_from_date_column \
  --anchor-date-column first_prod_date \
  --anchor-year-offset -1 \
  --skip-grid \
  --skip-existing \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  2>&1 | tee "$CHIP_OUTPUT_DIR/materialize_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.log"

# ── Step 4: Sync to GCS ──
if [ "$SKIP_GCS_SYNC" != "1" ]; then
  echo ""
  echo "Step 4: Syncing chips to GCS..."
  gsutil -m rsync -r "$CHIP_OUTPUT_DIR/" "$GCS_OUTPUT_URI/${COHORT_NAME}_ewm_s2s1/" 2>&1 | tail -5
  echo "  Sync complete."
fi

echo ""
echo "============================================================"
echo "DONE"
echo "  Cohort: $COHORT_PATH"
echo "  Chips:  $CHIP_OUTPUT_DIR/"
echo "  GCS:    $GCS_OUTPUT_URI/${COHORT_NAME}_ewm_s2s1/"
echo "============================================================"
