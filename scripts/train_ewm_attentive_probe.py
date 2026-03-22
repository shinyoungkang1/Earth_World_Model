#!/usr/bin/env python3
"""Train a V-JEPA 2.1-style attentive probe on cached EWM encoder outputs.

This script:
1. Loads a frozen EWM checkpoint and a fixed well cohort
2. Extracts and caches encoder timeline outputs [N, T, D] for all wells
3. Trains an AttentivePooler + Linear(D, 1) regressor on log1p(f12_gas)
4. Exports the trained probe weights for use in the embedding pipeline

Usage
-----
python scripts/train_ewm_attentive_probe.py \
    --cohort-path data/features/multiregion/phase4_multiregion_regression_3region_v1.parquet \
    --checkpoint checkpoints/ewm_best_val.pt \
    --index-path data/features/multiregion/.../ewm_s2s1_chip_index_v1.parquet \
    --target-column f12_gas \
    --output-dir models/attentive_probe/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))

from ewm.data.dataset import (  # noqa: E402
    DEFAULT_DATES,
    SSL4EO_S1GRD_MEAN,
    SSL4EO_S1GRD_STD,
    SSL4EO_S2L2A_MEAN,
    SSL4EO_S2L2A_STD,
)
from ewm.models.attentive_pooler import AttentiveRegressor  # noqa: E402
from ewm.models.world_model import EarthWorldModel  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_ewm_gas_pipeline_v1 import (  # noqa: E402
    build_model,
    load_checkpoint_teacher_weights,
    load_hls_sample,
    load_s2s1_sample,
    resolve_sample_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attentive probe on cached EWM features.")
    parser.add_argument("--cohort-path", required=True, help="Parquet with well cohort")
    parser.add_argument("--cohort-metadata-path", default=None)
    parser.add_argument("--checkpoint", required=True, help="EWM checkpoint .pt file")
    parser.add_argument("--index-path", required=True, help="Chip index parquet")
    parser.add_argument("--target-column", default="f12_gas")
    parser.add_argument("--target-transform", choices=["log1p", "none"], default="log1p")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "models" / "attentive_probe"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--s2s1-format", choices=["preprocessed", "raw_ssl4eo"], default="raw_ssl4eo")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    # Probe architecture
    parser.add_argument("--probe-depth", type=int, default=1, help="AttentivePooler depth (1=minimal)")
    parser.add_argument("--probe-num-heads", type=int, default=4)
    parser.add_argument("--probe-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--probe-dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5, help="K-fold CV for validation")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def extract_timelines(
    model: EarthWorldModel,
    chip_index: pd.DataFrame,
    cohort: pd.DataFrame,
    repo_root: Path,
    device: torch.device,
    batch_size: int,
    s2s1_format: str,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract frozen encoder timeline outputs for all wells with chips.

    Returns
    -------
    timelines : Tensor [N, T, D]
    sample_ids : ndarray of str
    """
    input_mode = model.input_mode
    config_patch_size = 128  # default for SSL4EO chips

    matched = chip_index[chip_index["sample_id"].isin(cohort["sample_id"])].copy().reset_index(drop=True)
    print(f"  Extracting timelines for {len(matched)} samples...")

    all_timelines: list[torch.Tensor] = []
    all_ids: list[str] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(matched), batch_size):
            batch_frame = matched.iloc[start : start + batch_size]
            batch_dates: list[torch.Tensor] = []
            batch_mask: list[torch.Tensor] = []
            sample_ids: list[str] = []

            if input_mode == "hls6":
                batch_hls: list[torch.Tensor] = []
                for row in batch_frame.itertuples(index=False):
                    sample = load_hls_sample(resolve_sample_path(repo_root, pd.Series(row._asdict())), config_patch_size)
                    batch_hls.append(sample["hls"])
                    batch_dates.append(sample["dates"])
                    batch_mask.append(sample["mask"])
                    sample_ids.append(str(row.sample_id))

                timeline = model.encode_timeline(
                    s2=None, s1=None,
                    dates=torch.stack(batch_dates, dim=0).to(device),
                    mask=torch.stack(batch_mask, dim=0).to(device),
                    hls=torch.stack(batch_hls, dim=0).to(device),
                )
            elif input_mode == "s2s1":
                batch_s2: list[torch.Tensor] = []
                batch_s1: list[torch.Tensor] = []
                for row in batch_frame.itertuples(index=False):
                    sample = load_s2s1_sample(
                        resolve_sample_path(repo_root, pd.Series(row._asdict())),
                        config_patch_size,
                        s2s1_format=s2s1_format,
                    )
                    batch_s2.append(sample["s2"])
                    batch_s1.append(sample["s1"])
                    batch_dates.append(sample["dates"])
                    batch_mask.append(sample["mask"])
                    sample_ids.append(str(row.sample_id))

                timeline = model.encode_timeline(
                    s2=torch.stack(batch_s2, dim=0).to(device),
                    s1=torch.stack(batch_s1, dim=0).to(device),
                    dates=torch.stack(batch_dates, dim=0).to(device),
                    mask=torch.stack(batch_mask, dim=0).to(device),
                )
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            all_timelines.append(timeline.cpu())
            all_ids.extend(sample_ids)

            if (start // batch_size) % 10 == 0:
                print(f"    batch {start // batch_size + 1}/{(len(matched) + batch_size - 1) // batch_size}")

    return torch.cat(all_timelines, dim=0), np.array(all_ids)


def train_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[AttentiveRegressor, dict[str, float]]:
    """Train an AttentiveRegressor on the given split."""
    embed_dim = features.shape[-1]
    probe = AttentiveRegressor(
        embed_dim=embed_dim,
        num_queries=1,
        num_heads=args.probe_num_heads,
        mlp_ratio=args.probe_mlp_ratio,
        depth=args.probe_depth,
        dropout=args.probe_dropout,
    )

    train_ds = TensorDataset(features[train_mask], targets[train_mask])
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)

    val_features = features[val_mask]
    val_targets = targets[val_mask]

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        probe.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = probe(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.shape[0]
        train_loss /= train_mask.sum()
        scheduler.step()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_features).squeeze(-1)
            val_loss = criterion(val_pred, val_targets).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if patience_counter >= args.patience:
            print(f"    early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    # Compute final val metrics
    probe.eval()
    with torch.no_grad():
        val_pred = probe(val_features).squeeze(-1).numpy()
        val_true = val_targets.numpy()
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(val_true, val_pred)

    metrics = {
        "val_mse": float(best_val_loss),
        "val_spearman": float(spearman) if np.isfinite(spearman) else 0.0,
    }
    return probe, metrics


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cohort and checkpoint...")
    cohort = pd.read_parquet(args.cohort_path)
    chip_index = pd.read_parquet(args.index_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    torch_device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, torch_device)
    load_checkpoint_teacher_weights(model, checkpoint["teacher"])

    print("Extracting frozen encoder timelines...")
    timelines, sample_ids = extract_timelines(
        model, chip_index, cohort, repo_root, torch_device,
        args.embedding_batch_size, args.s2s1_format,
    )
    print(f"  Cached timelines: {timelines.shape}")

    # Build targets aligned with extracted timelines
    wells = cohort[cohort["sample_type"] == "well"].copy()
    id_to_target = dict(zip(wells["sample_id"].astype(str), wells[args.target_column].astype(float)))
    id_to_split = dict(zip(wells["sample_id"].astype(str), wells.get("split_group", pd.Series("train", index=wells.index))))

    targets = []
    split_labels = []
    valid_mask = []
    for sid in sample_ids:
        if sid in id_to_target and np.isfinite(id_to_target[sid]):
            val = id_to_target[sid]
            targets.append(np.log1p(val) if args.target_transform == "log1p" else val)
            split_labels.append(id_to_split.get(sid, "train"))
            valid_mask.append(True)
        else:
            targets.append(0.0)
            split_labels.append("skip")
            valid_mask.append(False)

    targets = np.array(targets, dtype=np.float32)
    split_labels = np.array(split_labels)
    valid_mask = np.array(valid_mask)

    # Filter to valid wells only
    valid_timelines = timelines[valid_mask]
    valid_targets = torch.tensor(targets[valid_mask], dtype=torch.float32)
    valid_splits = split_labels[valid_mask]

    train_mask = valid_splits == "train"
    test_mask = valid_splits == "test"

    print(f"  Training wells: {train_mask.sum()}, Test wells: {test_mask.sum()}")

    # K-fold CV on train set for probe training
    rng = np.random.RandomState(args.random_state)
    train_indices = np.where(train_mask)[0]
    rng.shuffle(train_indices)
    fold_size = len(train_indices) // args.num_folds

    all_fold_metrics: list[dict[str, float]] = []
    best_fold_probe = None
    best_fold_spearman = -float("inf")

    for fold in range(args.num_folds):
        print(f"\nFold {fold + 1}/{args.num_folds}")
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < args.num_folds - 1 else len(train_indices)
        val_idx = train_indices[val_start:val_end]
        train_idx = np.concatenate([train_indices[:val_start], train_indices[val_end:]])

        fold_train_mask = np.zeros(len(valid_timelines), dtype=bool)
        fold_val_mask = np.zeros(len(valid_timelines), dtype=bool)
        fold_train_mask[train_idx] = True
        fold_val_mask[val_idx] = True

        probe, fold_metrics = train_probe(valid_timelines, valid_targets, fold_train_mask, fold_val_mask, args)
        all_fold_metrics.append(fold_metrics)
        print(f"  Fold {fold + 1} val_spearman: {fold_metrics['val_spearman']:.4f}")

        if fold_metrics["val_spearman"] > best_fold_spearman:
            best_fold_spearman = fold_metrics["val_spearman"]
            best_fold_probe = probe

    # Report CV results
    mean_spearman = np.mean([m["val_spearman"] for m in all_fold_metrics])
    std_spearman = np.std([m["val_spearman"] for m in all_fold_metrics])
    print(f"\nCV Results: Spearman = {mean_spearman:.4f} +/- {std_spearman:.4f}")

    # Train final probe on ALL training data
    print("\nTraining final probe on all training data...")
    # Use test set as validation for early stopping
    final_probe, final_metrics = train_probe(valid_timelines, valid_targets, train_mask, test_mask, args)

    # Evaluate on held-out test set
    final_probe.eval()
    with torch.no_grad():
        test_pred = final_probe(valid_timelines[test_mask]).squeeze(-1).numpy()
        test_true = valid_targets[test_mask].numpy()
        from scipy.stats import spearmanr
        test_spearman, _ = spearmanr(test_true, test_pred)

        # Raw-scale metrics
        if args.target_transform == "log1p":
            test_pred_raw = np.expm1(test_pred)
            test_true_raw = np.expm1(test_true)
        else:
            test_pred_raw = test_pred
            test_true_raw = test_true

        from sklearn.metrics import mean_squared_error, r2_score
        test_rmse_raw = float(np.sqrt(mean_squared_error(test_true_raw, test_pred_raw)))
        test_r2_raw = float(r2_score(test_true_raw, test_pred_raw))

    print(f"\nFinal Test Metrics:")
    print(f"  Spearman (raw): {test_spearman:.4f}")
    print(f"  R² (raw):       {test_r2_raw:.4f}")
    print(f"  RMSE (raw):     {test_rmse_raw:.2f}")

    # Save probe weights
    probe_path = output_dir / "attentive_probe.pt"
    torch.save({
        "state_dict": final_probe.pooler.state_dict(),
        "config": {
            "embed_dim": int(valid_timelines.shape[-1]),
            "num_queries": 1,
            "num_heads": args.probe_num_heads,
            "mlp_ratio": args.probe_mlp_ratio,
            "depth": args.probe_depth,
            "dropout": args.probe_dropout,
        },
    }, probe_path)
    print(f"\nSaved probe weights: {probe_path}")

    # Save full metrics
    results = {
        "cv_mean_spearman": float(mean_spearman),
        "cv_std_spearman": float(std_spearman),
        "cv_fold_metrics": all_fold_metrics,
        "test_spearman_raw": float(test_spearman) if np.isfinite(test_spearman) else 0.0,
        "test_r2_raw": float(test_r2_raw),
        "test_rmse_raw": float(test_rmse_raw),
        "train_count": int(train_mask.sum()),
        "test_count": int(test_mask.sum()),
        "timeline_shape": list(valid_timelines.shape),
        "probe_config": {
            "depth": args.probe_depth,
            "num_heads": args.probe_num_heads,
            "mlp_ratio": args.probe_mlp_ratio,
            "dropout": args.probe_dropout,
        },
        "training_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "target_column": args.target_column,
            "target_transform": args.target_transform,
        },
        "probe_path": str(probe_path),
    }
    results_path = output_dir / "attentive_probe_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {results_path}")

    # Also extract and save the embeddings from the trained probe
    print("\nExtracting embeddings with trained probe...")
    final_probe.eval()
    with torch.no_grad():
        all_embeddings = final_probe.extract_embedding(timelines[valid_mask]).numpy()

    embed_records = []
    valid_sample_ids = sample_ids[valid_mask]
    for i, sid in enumerate(valid_sample_ids):
        record: dict[str, Any] = {"sample_id": sid}
        for col_idx in range(all_embeddings.shape[1]):
            record[f"ewm_embedding_{col_idx:04d}"] = float(all_embeddings[i, col_idx])
        embed_records.append(record)

    embed_df = pd.DataFrame(embed_records)
    embed_path = output_dir / "attentive_probe_embeddings.parquet"
    embed_df.to_parquet(embed_path, index=False)
    print(f"Saved embeddings: {embed_path} ({len(embed_df)} rows, {all_embeddings.shape[1]} dims)")


if __name__ == "__main__":
    main()
