#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ID="${PROJECT_ID:-omois-483220}"
ZONE="${ZONE:-us-central1-f}"
PREP_VM_NAME="${PREP_VM_NAME:-ewm-prelim400-prep-cpu}"
A100_VM_NAME="${A100_VM_NAME:-ewm-phase2-a100}"
DATA_DISK_NAME="${DATA_DISK_NAME:-ewm-prelim400-data}"
DATA_DISK_SIZE_GB="${DATA_DISK_SIZE_GB:-500}"
DATA_DISK_DEVICE_NAME="${DATA_DISK_DEVICE_NAME:-ewm-training-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ewm-data-disk/ewm_prelim_400_v1}"
YEARLY_RAW_ROOTS="${YEARLY_RAW_ROOTS:-gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_exact_chip52w_batch100:gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_interactive_exact_chip_100_10000_v1/raw}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
YEARLY_TRAIN_COUNT="${YEARLY_TRAIN_COUNT:-400}"
YEARLY_VAL_COUNT="${YEARLY_VAL_COUNT:-32}"
SSL4EO_TRAIN_COUNT="${SSL4EO_TRAIN_COUNT:-400}"
SSL4EO_VAL_COUNT="${SSL4EO_VAL_COUNT:-128}"
OVERWRITE="${OVERWRITE:-0}"
GCS_RUNS_ROOT="${GCS_RUNS_ROOT:-gs://omois-earth-world-model-phase2-20260320-11728/earth_world_model/runs/prelim_400}"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  PREP_VM_NAME="$PREP_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  DATA_DISK_SIZE_GB="$DATA_DISK_SIZE_GB" \
  DATA_DISK_DEVICE_NAME="$DATA_DISK_DEVICE_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_create_ssl4eo_prep_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$PREP_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_sync_repo_bundle_to_vm.sh"

gcloud compute ssh "$PREP_VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc 'cd ~/workspace/Mineral_Gas_Locator && \
    env \
      OUTPUT_ROOT=$OUTPUT_ROOT \
      YEARLY_RAW_ROOTS=$YEARLY_RAW_ROOTS \
      YEARLY_NAME=$YEARLY_NAME \
      SSL4EO_NAME=$SSL4EO_NAME \
      YEARLY_TRAIN_COUNT=$YEARLY_TRAIN_COUNT \
      YEARLY_VAL_COUNT=$YEARLY_VAL_COUNT \
      SSL4EO_TRAIN_COUNT=$SSL4EO_TRAIN_COUNT \
      SSL4EO_VAL_COUNT=$SSL4EO_VAL_COUNT \
      OVERWRITE=$OVERWRITE \
      FORMAT_IF_NEEDED=1 \
      bash earth_world_model/scripts/run_prepare_preliminary_datasets_cpu_vm.sh'"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$PREP_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_detach_ssl4eo_data_disk.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  DATA_DISK_NAME="$DATA_DISK_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_attach_ssl4eo_data_disk_to_a100.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  VM_NAME="$A100_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_sync_repo_bundle_to_vm.sh"

env \
  PROJECT_ID="$PROJECT_ID" \
  ZONE="$ZONE" \
  A100_VM_NAME="$A100_VM_NAME" \
  bash "$PROJECT_ROOT/earth_world_model/scripts/gcp_mount_training_data_disk_on_a100.sh"

gcloud compute ssh "$A100_VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="bash -lc 'cd ~/workspace/Mineral_Gas_Locator && \
    nohup env GCS_RUNS_ROOT=$GCS_RUNS_ROOT YEARLY_NAME=$YEARLY_NAME SSL4EO_NAME=$SSL4EO_NAME YEARLY_TRAIN_MAX_SAMPLES=$YEARLY_TRAIN_COUNT YEARLY_VAL_MAX_SAMPLES=$YEARLY_VAL_COUNT SSL4EO_TRAIN_MAX_SAMPLES=$SSL4EO_TRAIN_COUNT SSL4EO_VAL_MAX_SAMPLES=$SSL4EO_VAL_COUNT bash earth_world_model/scripts/run_preliminary_three_runs_gpu_vm.sh \
    > /tmp/ewm_prelim400_launcher.log 2>&1 < /dev/null &'"

echo "Launched prelim400 prep + three A100 runs."
