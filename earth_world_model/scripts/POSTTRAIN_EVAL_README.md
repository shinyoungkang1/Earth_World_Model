# Post-Train Eval Manifest

Use this doc with [`POSTTRAIN_EVAL_MANIFEST_TEMPLATE.yaml`](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/POSTTRAIN_EVAL_MANIFEST_TEMPLATE.yaml) to record the exact artifacts and data paths needed to validate a training run later from a fresh machine.

## What Training Already Syncs To GCS

The phase-2 training launchers already sync these artifacts to `GCS_RUN_URI` during and at the end of training:

- `checkpoints/`
- `train.log`
- `metrics.jsonl`
- `training_summary.json`
- `run_manifest.json`

That sync is handled by:

- [run_phase2_gpu_vm.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/run_phase2_gpu_vm.sh)
- [sync_run_artifacts_loop.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/sync_run_artifacts_loop.sh)

So we do not need to put checkpoints or large result directories in git.

## What To Record Per Run

Fill one manifest per training run with:

- the run GCS root
- the exact checkpoint URI to validate
- the training summary / metrics / run-manifest URIs
- the repo config path used for the run
- the prepared yearly index paths
- the prepared SSL4EO root, if needed
- optional benchmark dataset locations for flood / DFC2020 / BigEarthNet v2 / BioMassters

## Minimum Needed For Later Validation

For the current eval bundle in [run_paper3_eval_bundle.sh](/home/shin/Mineral_Gas_Locator/earth_world_model/scripts/run_paper3_eval_bundle.sh):

- always required:
  - checkpoint path
  - config path
  - yearly index path
  - output directory

- needed only for specific evals:
  - flood eval: `FLOOD_DATA_DIR`
  - SSL4EO downstream probes: benchmark data can be downloaded from Hugging Face
  - DFC2020 / BigEarthNet v2 / BioMassters: local benchmark manifests and data roots

## Recommended Workflow

1. Train with a launcher that sets `GCS_RUN_URI`.
2. After launch, create a copy of the manifest template for that run.
3. Fill in the concrete GCS artifact URIs and local benchmark/index paths.
4. Commit the manifest if it only contains lightweight paths and metadata.
5. Keep large artifacts in GCS or attached disks, not in git.

## Example Eval Launch

```bash
env \
  CHECKPOINT_PATH=/path/to/ewm_best_val.pt \
  CONFIG_PATH=/path/to/config.yaml \
  INDEX_PATH=/path/to/yearly_10000/train/dense_temporal_index.parquet \
  OUTPUT_ROOT=/tmp/ewm_eval_run \
  DEVICE=cuda \
  bash earth_world_model/scripts/run_paper3_eval_bundle.sh
```

## Notes

- `earth_world_model/local_paper2_viz` is scratch output and should stay out of git.
- If you only have the repo on a laptop, pull the checkpoint and summaries from GCS and point the eval script at the prepared dataset / benchmark locations available on the remote machine.
