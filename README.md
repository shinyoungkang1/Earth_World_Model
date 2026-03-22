# Earth World Model — Satellite-Based Gas Prospectivity

A self-supervised temporal vision model for Earth observation, applied to natural gas prospectivity prediction across the Appalachian Basin.

## Overview

This project combines **tabular well/geology features** with **satellite-derived embeddings** (Sentinel-2 optical + Sentinel-1 SAR) to predict gas production potential at any location — including undeveloped frontier areas where traditional well data is sparse.

### Architecture

The Earth World Model (EWM) uses a **V-JEPA 2.1-inspired** temporal Joint Embedding Predictive Architecture:

1. **Spatial Encoder** — patchifies multi-sensor satellite imagery (S2 12-band + S1 2-band)
2. **Temporal Encoder** — 6-layer transformer processing seasonal observations (4 timestamps)
3. **JEPA Predictor** — 3-layer transformer trained via masked temporal prediction (student-teacher EMA)
4. **Attentive Probe** — V-JEPA 2.1-style cross-attention pooling for downstream task adaptation

### Key Features

- **5 embedding extraction modes**: mean, l2_mean, predictor passthrough, multi-layer, multi-mask ensemble
- **V-JEPA 2.1 attentive probe**: learnable query tokens with cross-attention for task-specific embeddings
- **Dataflux GCS streaming**: optional high-throughput data loading from Google Cloud Storage
- **torch.compile support**: 1.3-1.5x training speedup on PyTorch 2.0+
- **Stage A-D training recipes**: progressive architecture upgrades (context loss → hierarchical supervision → dense tokens → RoPE)

## Project Structure

```
earth_world_model/          # Core EWM model and training
  src/ewm/models/
    world_model.py          # EarthWorldModel with extract_embedding()
    attentive_pooler.py     # V-JEPA 2.1-style AttentivePooler
    encoder.py              # SpatialEncoder (S2+S1 patchification)
    transformer_blocks.py   # RoPE transformer blocks
  train_tpu.py              # Training loop (CUDA/TPU/CPU)
  configs/                  # Training configs (Stage A-D)
  experiments/              # Experiment registry
  scripts/                  # VM setup, GCS sync, probes

scripts/                    # Gas pipeline scripts
  run_ewm_gas_pipeline_v2.py      # EWM embeddings + XGBoost pipeline
  train_ewm_attentive_probe.py    # Train attentive probe on gas data
  materialize_ewm_s2s1_chips_v1.py  # Fetch satellite chips
  multiregion/              # Multi-state data processing (PA, WV, OH, TX)

config/basins/              # Basin configuration (8 basins)
app/                        # FastAPI map viewer
models/                     # Trained model metrics
ontology/                   # Data ontology definitions
```

## Regions

| Region | State | Wells | Status |
|--------|-------|-------|--------|
| SWPA Core (Washington/Greene) | PA | 5,961 | Ready |
| PA Northeast (Susquehanna) | PA | 11,825 | Ready |
| WV Horizontal Statewide | WV | 7,969 | Ready |
| OH Utica Statewide | OH | 4,583 | Ready |
| TX Permian/Eagle Ford/Haynesville | TX | — | Blocked (data) |
| CO DJ Basin | CO | — | Blocked (API) |

## Quick Start

### Train EWM (on GCP A100)

```bash
# Setup VM
INSTALL_DATAFLUX=1 bash earth_world_model/scripts/setup_gpu_vm.sh

# Train Stage D (dense tokens + RoPE)
python earth_world_model/train_tpu.py \
  --config earth_world_model/configs/ssl4eo_zarr_trainval_50k_vjepa21_staged_rope_cuda_gpu_vm.yaml
```

### Extract Embeddings & Run Gas Pipeline

```bash
# Extract with different modes: mean, l2_mean, predictor, multilayer, ensemble
python scripts/run_ewm_gas_pipeline_v2.py \
  --cohort-path data/features/multiregion/cohort.parquet \
  --checkpoint checkpoints/ewm_best_val.pt \
  --index-path data/features/multiregion/chip_index.parquet \
  --embedding-mode l2_mean \
  --task-type regression \
  --target-column f12_gas
```

### Train Attentive Probe

```bash
python scripts/train_ewm_attentive_probe.py \
  --cohort-path data/features/multiregion/cohort.parquet \
  --checkpoint checkpoints/ewm_best_val.pt \
  --index-path data/features/multiregion/chip_index.parquet \
  --probe-depth 1 --epochs 200
```

## Current Results (2026-03-22)

3-region regression (PA + SWPA + WV, 636 wells, f12_gas target):

| Method | Spearman | R²(raw) |
|--------|----------|---------|
| **Tabular only** | **0.6622** | **-0.2460** |
| + EWM l2_mean | 0.6277 | -0.6991 |
| + EWM attentive probe | 0.5699 | -0.7217 |

**Finding**: The micro-PoC encoder (5,120 pretrain samples) has not yet learned useful geological representations. Tabular features (neighbor production, fault distance, geology) dominate. The 50k-500k SSL4EO pretraining experiments are needed to realize the satellite signal.

## Tabular Features

18 numeric + 3 categorical features per well:
- **Location**: latitude, longitude, elevation, slope, terrain relief
- **Geology**: fault distance, bedrock map symbol, geological age, lithology
- **Well proximity**: well/permit counts at 2km/5km, nearest well/permit distance
- **Production history**: mature F12 neighbor count/median/P90 at 5km/10km

## References

- [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482) — Meta FAIR, 2026
- [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) — IBM/NASA foundation model
- [SSL4EO-S12](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1) — Sentinel-1/2 pretraining data
- [Hydrocarbon Microseepage Detection via Sentinel-2](https://www.e3s-conferences.org/articles/e3sconf/abs/2018/48/e3sconf_icenis18_03021/e3sconf_icenis18_03021.html)
