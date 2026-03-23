# EWM Experiment Registry

This directory is the source of truth for named Earth World Model experiments.

## Why this exists

The repo already had:

- runnable configs in `earth_world_model/configs/`
- launch scripts in `earth_world_model/scripts/`
- planning notes in `Docs/Future/`

What it did not have was one stable place to answer:

- which variants are actually available now
- which ones are runnable only after dense data materialization
- which variables each run is meant to isolate
- what command family launches each run

That is what `registry.yaml` is for.

## Files

- `registry.yaml`
  - canonical experiment ids
  - readiness status
  - config path
  - launcher kind
  - paper-facing fields such as sample budget, cadence, recipe, and continuation parent

## How to use it

Use the helper script:

```bash
python earth_world_model/scripts/run_registered_experiment.py list
python earth_world_model/scripts/run_registered_experiment.py show ssl4eo_50k_stagec_v1
python earth_world_model/scripts/run_registered_experiment.py compare ssl4eo_50k_control_v1 ssl4eo_50k_stagec_v1
python earth_world_model/scripts/run_registered_experiment.py command ssl4eo_50k_stagec_v1
python earth_world_model/scripts/run_registered_experiment.py command ssl4eo_50k_stagec_v1 --cuda-ddp-procs 2
python earth_world_model/scripts/run_registered_experiment.py probe-command ssl4eo_50k_stagec_v1
```

## Naming rule

Experiment ids are intentionally explicit:

- data source and scale first
- recipe stage next
- continuation parent when relevant
- version suffix last

Example:

- `dense_cdse_5k_biweekly_stagec_continue_v1`

That is easier to cite in notes, run logs, and a later paper appendix than a
free-form run directory name.

## Storage layout

Keep using the existing split:

- configs:
  `earth_world_model/configs/`
- infra launchers:
  `earth_world_model/scripts/`
- experiment definitions:
  `earth_world_model/experiments/`
- narrative runbooks:
  `Docs/Future/`

That separation keeps "what the model is", "how the job is launched", and "why
this variant exists" from getting mixed together.
