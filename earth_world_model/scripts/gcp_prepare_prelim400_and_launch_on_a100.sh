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
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/home/shin/workspace/Mineral_Gas_Locator}"
BOOTSTRAP_PREP_VM="${BOOTSTRAP_PREP_VM:-1}"
SKIP_LAUNCH_A100="${SKIP_LAUNCH_A100:-0}"
DETACH_PREP_RUN="${DETACH_PREP_RUN:-0}"
PREP_REQUIREMENTS_PATH="${PREP_REQUIREMENTS_PATH:-earth_world_model/requirements-prep.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ewm-data-disk/ewm_prelim_400_v1}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT}_work}"
YEARLY_RAW_ROOTS="${YEARLY_RAW_ROOTS:-gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_exact_chip52w_batch100|gs://omois-earth-world-model-phase2-20260320-11728/earth_engine_interactive_exact_chip_100_10000_v1/raw}"
YEARLY_NAME="${YEARLY_NAME:-yearly_400}"
SSL4EO_NAME="${SSL4EO_NAME:-ssl4eo_400}"
YEARLY_TRAIN_COUNT="${YEARLY_TRAIN_COUNT:-400}"
YEARLY_VAL_COUNT="${YEARLY_VAL_COUNT:-32}"
YEARLY_WORKERS="${YEARLY_WORKERS:-4}"
SSL4EO_WORKERS="${SSL4EO_WORKERS:-4}"
SSL4EO_TRAIN_COUNT="${SSL4EO_TRAIN_COUNT:-400}"
SSL4EO_VAL_COUNT="${SSL4EO_VAL_COUNT:-128}"
OVERWRITE="${OVERWRITE:-0}"
GCS_RUNS_ROOT="${GCS_RUNS_ROOT:-gs://omois-earth-world-model-phase2-20260320-11728/earth_world_model/runs/prelim_400}"

quote_for_bash() {
  printf '%q' "$1"
}

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

if [[ "$BOOTSTRAP_PREP_VM" == "1" ]]; then
  echo "[prep] Bootstrapping prep VM Python environment"
  gcloud compute ssh "$PREP_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="bash -lc 'sudo apt-get update && sudo apt-get install -y python3-pip && \
      cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
      python3 -m pip install --upgrade pip && \
      python3 -m pip install -r $(quote_for_bash "$PREP_REQUIREMENTS_PATH")'"
fi

echo "[prep] Starting dataset build on $PREP_VM_NAME"
if [[ "$DETACH_PREP_RUN" == "1" ]]; then
  gcloud compute ssh "$PREP_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="bash -lc 'cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
      env \
        OUTPUT_ROOT=$(quote_for_bash "$OUTPUT_ROOT") \
        WORK_DIR=$(quote_for_bash "$WORK_DIR") \
        YEARLY_RAW_ROOTS=$(quote_for_bash "$YEARLY_RAW_ROOTS") \
        YEARLY_NAME=$(quote_for_bash "$YEARLY_NAME") \
        SSL4EO_NAME=$(quote_for_bash "$SSL4EO_NAME") \
        YEARLY_TRAIN_COUNT=$(quote_for_bash "$YEARLY_TRAIN_COUNT") \
        YEARLY_VAL_COUNT=$(quote_for_bash "$YEARLY_VAL_COUNT") \
        YEARLY_WORKERS=$(quote_for_bash "$YEARLY_WORKERS") \
        SSL4EO_WORKERS=$(quote_for_bash "$SSL4EO_WORKERS") \
        SSL4EO_TRAIN_COUNT=$(quote_for_bash "$SSL4EO_TRAIN_COUNT") \
        SSL4EO_VAL_COUNT=$(quote_for_bash "$SSL4EO_VAL_COUNT") \
        OVERWRITE=$(quote_for_bash "$OVERWRITE") \
        FORMAT_IF_NEEDED=1 \
        bash earth_world_model/scripts/launch_prepare_preliminary_datasets_bg.sh'"
else
  gcloud compute ssh "$PREP_VM_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="bash -lc 'cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
      env \
        OUTPUT_ROOT=$(quote_for_bash "$OUTPUT_ROOT") \
        WORK_DIR=$(quote_for_bash "$WORK_DIR") \
        YEARLY_RAW_ROOTS=$(quote_for_bash "$YEARLY_RAW_ROOTS") \
        YEARLY_NAME=$(quote_for_bash "$YEARLY_NAME") \
        SSL4EO_NAME=$(quote_for_bash "$SSL4EO_NAME") \
        YEARLY_TRAIN_COUNT=$(quote_for_bash "$YEARLY_TRAIN_COUNT") \
        YEARLY_VAL_COUNT=$(quote_for_bash "$YEARLY_VAL_COUNT") \
        YEARLY_WORKERS=$(quote_for_bash "$YEARLY_WORKERS") \
        SSL4EO_WORKERS=$(quote_for_bash "$SSL4EO_WORKERS") \
        SSL4EO_TRAIN_COUNT=$(quote_for_bash "$SSL4EO_TRAIN_COUNT") \
        SSL4EO_VAL_COUNT=$(quote_for_bash "$SSL4EO_VAL_COUNT") \
        OVERWRITE=$(quote_for_bash "$OVERWRITE") \
        FORMAT_IF_NEEDED=1 \
        bash earth_world_model/scripts/run_prepare_preliminary_datasets_cpu_vm.sh'"
fi

if [[ "$SKIP_LAUNCH_A100" == "1" ]]; then
  echo "Preparation finished on $PREP_VM_NAME. Skipping A100 launch."
  exit 0
fi

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
  --command="bash -lc 'cd $(quote_for_bash "$REMOTE_REPO_ROOT") && \
    nohup env GCS_RUNS_ROOT=$(quote_for_bash "$GCS_RUNS_ROOT") YEARLY_NAME=$(quote_for_bash "$YEARLY_NAME") SSL4EO_NAME=$(quote_for_bash "$SSL4EO_NAME") YEARLY_TRAIN_MAX_SAMPLES=$(quote_for_bash "$YEARLY_TRAIN_COUNT") YEARLY_VAL_MAX_SAMPLES=$(quote_for_bash "$YEARLY_VAL_COUNT") SSL4EO_TRAIN_MAX_SAMPLES=$(quote_for_bash "$SSL4EO_TRAIN_COUNT") SSL4EO_VAL_MAX_SAMPLES=$(quote_for_bash "$SSL4EO_VAL_COUNT") bash earth_world_model/scripts/run_preliminary_three_runs_gpu_vm.sh \
    > /tmp/ewm_prelim400_launcher.log 2>&1 < /dev/null &'"

echo "Launched dataset prep plus three A100 runs."
