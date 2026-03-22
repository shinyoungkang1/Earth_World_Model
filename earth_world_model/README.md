# Earth World Model PoC

This directory contains the first executable scaffold for the `<$500`
real-data TPU PoC described in:

- [earth_world_model_tpu_v5e_plan.md](/home/shin/Mineral_Gas_Locator/Docs/03-20-2026/earth_world_model_tpu_v5e_plan.md)
- [earth_world_model_phase0_phase1_checklist.md](/home/shin/Mineral_Gas_Locator/Docs/03-20-2026/earth_world_model_phase0_phase1_checklist.md)

Current scope:

- fake-data Phase 0 training
- manifest-based real-data Phase 1 loading
- minimal proper SSL4EO Zarr validation-shard loading
- simple multi-sensor spatial encoder
- temporal JEPA-style training loop

This is intentionally narrow. It is not yet a full SSL4EO raw-shard downloader or
a full benchmark suite.

## Layout

```text
earth_world_model/
  configs/
  scripts/
  src/ewm/
  train_tpu.py
  requirements.txt
```

## Run Phase 0

```bash
bash earth_world_model/scripts/run_train_entry.sh \
  --config earth_world_model/configs/fake_micro.yaml
```

## Build A Phase 1 Manifest

The current Phase 1 loader expects pre-extracted `.npz` samples where each file
contains:

- `s2`: `[4, 12, H, W]`
- `s1`: `[4, 2, H, W]`
- optional `dates`: `[4]`
- optional `mask`: `[4]`

Create a manifest:

```bash
python earth_world_model/scripts/build_ssl4eo_manifest.py \
  --samples-root /path/to/ssl4eo_subset \
  --output earth_world_model/manifests/ssl4eo_debug.jsonl
```

Then run the real-data debug config:

```bash
bash earth_world_model/scripts/run_train_entry.sh \
  --config earth_world_model/configs/real_loader_debug.yaml \
  --manifest-path earth_world_model/manifests/ssl4eo_debug.jsonl \
  --max-samples 256
```

## Minimal Proper SSL4EO Test

The repo now also supports a small direct SSL4EO Zarr test using real validation
shards downloaded into:

- `data/raw/ssl4eo_zarr_minimal`

Current downloaded subset:

- `splits/ssl4eos12_train.txt`
- `splits/ssl4eos12_val.txt`
- `val/S2L2A/ssl4eos12_val_seasonal_data_000001.zarr.zip`
- `val/S2L2A/ssl4eos12_val_seasonal_data_000002.zarr.zip`
- `val/S1GRD/ssl4eos12_val_seasonal_data_000001.zarr.zip`
- `val/S1GRD/ssl4eos12_val_seasonal_data_000002.zarr.zip`

Run the minimal SSL4EO CUDA smoke test:

```bash
PYTHONPATH=earth_world_model/src:. PYTHONUNBUFFERED=1 python - <<'PY'
import sys
from earth_world_model.train_tpu import main
sys.argv = ['train_tpu.py', '--config', 'earth_world_model/configs/ssl4eo_zarr_val_cuda_minimal.yaml']
main()
PY
```

This uses:

- [ssl4eo_zarr_val_cuda_minimal.yaml](/home/shin/Mineral_Gas_Locator/earth_world_model/configs/ssl4eo_zarr_val_cuda_minimal.yaml)

and writes checkpoints to:

- `earth_world_model/checkpoints/ssl4eo_zarr_val_cuda_minimal/`

## Download SSL4EO Shards To GCS

The shard downloader can now read a dataset bucket prefix from the repo `.env`.

Example `.env` entry:

```bash
GCS_DATA_URI=gs://YOUR_BUCKET/earth_world_model/data/raw/ssl4eo_zarr_minimal
```

Then download directly into `GCS` while skipping the repo-local extracted copy:

```bash
python earth_world_model/scripts/download_ssl4eo_zarr_shards.py \
  --split train \
  --shards 1 2 3 \
  --skip-local-copy
```

The script still uses a temporary local cache for Hugging Face tar downloads,
but it uploads the extracted shard zips and split lists to `GCS`
automatically. You can also override the env var explicitly with `--gcs-uri`.

## TPU Setup

Use:

- [setup_tpu.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/setup_tpu.sh)

It installs `torch_xla` and the Python dependencies needed for this PoC.

## GCS + TPU VM Phase 2

For the cloud path, use:

- [ssl4eo_zarr_trainval_phase2_scale_tpu_vm.yaml](/home/shin/Mineral_Gas_Locator/earth_world_model/configs/ssl4eo_zarr_trainval_phase2_scale_tpu_vm.yaml)
- [upload_ssl4eo_to_gcs.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/upload_ssl4eo_to_gcs.sh)
- [run_phase2_tpu_vm.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/run_phase2_tpu_vm.sh)
- [earth_world_model_gcs_tpu_v5e_runbook.md](/home/shin/Mineral_Gas_Locator/Docs/03-20-2026/earth_world_model_gcs_tpu_v5e_runbook.md)

The intended flow is:

1. Upload the SSL4EO corpus and the saved Phase 2 seed checkpoint to `GCS`.
2. Create a `v5litepod-1` TPU VM in the same region as the bucket.
3. Stage the corpus from `GCS` onto the TPU VM's local disk.
4. Train locally on the TPU VM and sync checkpoints back to `GCS`.
5. Stop or delete the TPU VM after the run.
