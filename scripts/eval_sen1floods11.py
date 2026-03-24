#!/usr/bin/env python3
"""Evaluate frozen EWM encoder on Sen1Floods11 flood segmentation benchmark.

Approach:
1. Load each 512x512 S1+S2 chip
2. Tile into 128x128 patches (4x4 grid = 16 patches per chip)
3. Run frozen EWM encoder on each patch → [16, T*patches_or_T, D] features
4. Pool temporal tokens → [16, D] per patch
5. Reshape to spatial feature map [4, 4, D]
6. Upsample + linear head → per-pixel flood/no-flood prediction
7. Evaluate IoU on test split

Usage:
    python scripts/eval_sen1floods11.py \
        --checkpoint checkpoints/50k_stagec_1024/ewm_best_val.pt \
        --data-dir data/benchmarks/sen1floods11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "earth_world_model" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ewm.data.dataset import SSL4EO_S1GRD_MEAN, SSL4EO_S1GRD_STD, SSL4EO_S2L2A_MEAN, SSL4EO_S2L2A_STD
from ewm.models.world_model import EarthWorldModel
from run_ewm_gas_pipeline_v1 import build_model, load_checkpoint_teacher_weights


# Sen1Floods11 S2 has 13 bands; our encoder expects 12 (drop B10 at index 10)
S2_BAND_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
PATCH_SIZE = 128
GRID_H, GRID_W = 4, 4  # 512 / 128 = 4


class Sen1Floods11Dataset(Dataset):
    """Sen1Floods11 hand-labeled flood dataset."""

    def __init__(self, data_dir: Path, split_file: Path):
        self.data_dir = Path(data_dir)
        self.chip_ids = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "Country_ChipID_S1Hand.tif,Country_ChipID_LabelHand.tif"
                s1_name = line.split(",")[0]
                chip_id = s1_name.replace("_S1Hand.tif", "")
                self.chip_ids.append(chip_id)

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        chip_id = self.chip_ids[idx]

        # Load S1 (2 bands, 512x512, float32, dB)
        with rasterio.open(self.data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif") as src:
            s1 = src.read().astype(np.float32)  # [2, 512, 512]
        s1 = np.nan_to_num(s1, nan=0.0)

        # Load S2 (13 bands, 512x512, uint16 → float32)
        with rasterio.open(self.data_dir / "S2Hand" / f"{chip_id}_S2Hand.tif") as src:
            s2_full = src.read().astype(np.float32)  # [13, 512, 512]
        s2 = s2_full[S2_BAND_INDICES]  # [12, 512, 512]
        # Keep raw DN values — encoder normalization handles the scaling

        # Load label (1 band, 512x512, int16: -1=nodata, 0=land, 1=water)
        with rasterio.open(self.data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
            label = src.read(1).astype(np.int16)  # [512, 512]

        return {
            "s2": torch.from_numpy(s2),      # [12, 512, 512]
            "s1": torch.from_numpy(s1),      # [2, 512, 512]
            "label": torch.from_numpy(label.astype(np.int64)),  # [512, 512]
            "chip_id": chip_id,
        }


class UNetDecoderBlock(nn.Module):
    """Single U-Net decoder block: upsample + concat skip + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Resize skip if needed
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetSegmentationHead(nn.Module):
    """U-Net decoder with skip connections from input image.

    Takes [B, 64, 64, D] encoder features + [B, 14, 512, 512] raw input
    and produces [B, num_classes, 512, 512].

    Skip connections at 3 scales from the raw input:
      - 256×256 (2× downsample of input)
      - 128×128 (4× downsample)
      - 64×64 (8× downsample — same as encoder features)
    """

    def __init__(self, embed_dim: int, input_channels: int = 14, num_classes: int = 2, base_ch: int = 64):
        super().__init__()
        # Encoder-side skip feature extractors (lightweight convs on raw input at multiple scales)
        self.skip_64 = nn.Sequential(
            nn.Conv2d(input_channels, base_ch, kernel_size=8, stride=8),  # 512→64
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )
        self.skip_128 = nn.Sequential(
            nn.Conv2d(input_channels, base_ch, kernel_size=4, stride=4),  # 512→128
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )
        self.skip_256 = nn.Sequential(
            nn.Conv2d(input_channels, base_ch // 2, kernel_size=2, stride=2),  # 512→256
            nn.BatchNorm2d(base_ch // 2), nn.ReLU(),
        )

        # Bottleneck: reduce embed_dim to manageable size
        self.bottleneck = nn.Sequential(
            nn.Conv2d(embed_dim, base_ch * 4, kernel_size=1),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(),
        )

        # Decoder blocks: 64→128→256→512
        self.dec1 = UNetDecoderBlock(base_ch * 4, base_ch, base_ch * 2)  # 64→128, skip from skip_128
        self.dec2 = UNetDecoderBlock(base_ch * 2, base_ch // 2, base_ch)  # 128→256, skip from skip_256
        self.dec3 = UNetDecoderBlock(base_ch, 0, base_ch // 2)  # 256→512, no skip

        # For dec3 we don't have a skip, so override
        self.up3 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch // 2), nn.ReLU(),
        )

        self.head = nn.Conv2d(base_ch // 2, num_classes, kernel_size=1)

    def forward(self, dense_features: torch.Tensor, raw_input: torch.Tensor) -> torch.Tensor:
        """
        dense_features: [B, 64, 64, D] from frozen encoder
        raw_input: [B, 14, 512, 512] concatenated S2+S1
        """
        # Build skip connections from raw input
        s64 = self.skip_64(raw_input)    # [B, base_ch, 64, 64]
        s128 = self.skip_128(raw_input)  # [B, base_ch, 128, 128]
        s256 = self.skip_256(raw_input)  # [B, base_ch//2, 256, 256]

        # Bottleneck on encoder features
        x = dense_features.permute(0, 3, 1, 2)  # [B, D, 64, 64]
        x = self.bottleneck(x)  # [B, base_ch*4, 64, 64]

        # Decoder with skip connections
        x = self.dec1(x, s128)  # [B, base_ch*2, 128, 128]
        x = self.dec2(x, s256)  # [B, base_ch, 256, 256]

        # Final upsample (no skip at 512)
        x = self.up3(x)         # [B, base_ch//2, 512, 512]
        x = self.conv3(x)

        return self.head(x)     # [B, num_classes, 512, 512]


def extract_patch_features(
    model: EarthWorldModel,
    s2: torch.Tensor,
    s1: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Extract dense spatial features for each 128x128 patch from a 512x512 chip.

    For dense token mode: each 128x128 patch → [256, D] spatial tokens (16×16 grid)
    Reassembled: 4×4 patches × 16×16 tokens = 64×64 feature map

    Args:
        s2: [B, 12, 512, 512]
        s1: [B, 2, 512, 512]

    Returns:
        [B, 64, 64, embed_dim]
    """
    B = s2.shape[0]
    embed_dim = model.embed_dim
    # With late_concat: S2 (256 tokens) + S1 (256 tokens) = 512 tokens per timestep
    # We mean-pool the S2 and S1 token groups back to 256 spatial tokens
    patches_per_side = PATCH_SIZE // model.spatial.patch_px  # 128/8 = 16
    spatial_patches = patches_per_side * patches_per_side  # 256

    # Move inputs to device first
    s2 = s2.to(device)
    s1 = s1.to(device)

    # Normalize S2 (our encoder expects SSL4EO normalization)
    s2_mean = SSL4EO_S2L2A_MEAN.clone().detach().to(device).view(1, 12, 1, 1)
    s2_std = SSL4EO_S2L2A_STD.clone().detach().to(device).view(1, 12, 1, 1)
    s2_norm = (s2 - s2_mean) / (s2_std + 1e-6)

    s1_mean = SSL4EO_S1GRD_MEAN.clone().detach().to(device).view(1, 2, 1, 1)
    s1_std = SSL4EO_S1GRD_STD.clone().detach().to(device).view(1, 2, 1, 1)
    s1_norm = (s1 - s1_mean) / (s1_std + 1e-6)

    # Tile into 128x128 patches
    s2_patches = s2_norm.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    s2_patches = s2_patches.permute(0, 2, 3, 1, 4, 5).reshape(B * GRID_H * GRID_W, 12, PATCH_SIZE, PATCH_SIZE)

    s1_patches = s1_norm.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    s1_patches = s1_patches.permute(0, 2, 3, 1, 4, 5).reshape(B * GRID_H * GRID_W, 2, PATCH_SIZE, PATCH_SIZE)

    # Sen1Floods11 is a single timestamp — wrap as T=1
    s2_temporal = s2_patches.unsqueeze(1)
    s1_temporal = s1_patches.unsqueeze(1)
    dates = torch.zeros(B * GRID_H * GRID_W, 1, dtype=torch.long, device=device)
    mask = torch.ones(B * GRID_H * GRID_W, 1, dtype=torch.bool, device=device)

    # Run encoder in sub-batches
    all_features = []
    sub_batch = 8  # smaller for dense mode (more tokens per sample)
    for i in range(0, s2_temporal.shape[0], sub_batch):
        j = min(i + sub_batch, s2_temporal.shape[0])
        with torch.inference_mode():
            timeline = model.encode_timeline(
                s2=s2_temporal[i:j],
                s1=s1_temporal[i:j],
                dates=dates[i:j],
                mask=mask[i:j],
            )
            # Dense mode with T=1 and late_concat: [sub_B, 512, D]
            # 512 = 256 S2 tokens + 256 S1 tokens
            # Mean-pool the two modality groups → [sub_B, 256, D]
            sb = timeline.shape[0]
            total_tokens = timeline.shape[1]
            if total_tokens == 2 * spatial_patches:
                # late_concat: first half S2, second half S1 → mean
                s2_tokens = timeline[:, :spatial_patches, :]
                s1_tokens = timeline[:, spatial_patches:, :]
                merged = (s2_tokens + s1_tokens) / 2.0
            elif total_tokens == spatial_patches:
                merged = timeline
            else:
                # Fallback: just take first spatial_patches tokens
                merged = timeline[:, :spatial_patches, :]
            spatial = merged.reshape(sb, patches_per_side, patches_per_side, embed_dim)
            all_features.append(spatial.cpu())

    # [B*16, 16, 16, D] → reassemble to [B, 4, 16, 4, 16, D] → [B, 64, 64, D]
    features = torch.cat(all_features, dim=0)
    features = features.reshape(B, GRID_H, GRID_W, patches_per_side, patches_per_side, embed_dim)
    features = features.permute(0, 1, 3, 2, 4, 5)
    features = features.reshape(B, GRID_H * patches_per_side, GRID_W * patches_per_side, embed_dim)
    return features


def compute_iou(pred: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Compute IoU for water and non-water classes."""
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    results = {}
    for cls, name in [(0, "non_water"), (1, "water")]:
        tp = ((pred_valid == cls) & (target_valid == cls)).sum()
        fp = ((pred_valid == cls) & (target_valid != cls)).sum()
        fn = ((pred_valid != cls) & (target_valid == cls)).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        results[f"iou_{name}"] = float(iou)
    results["mean_iou"] = (results["iou_non_water"] + results["iou_water"]) / 2
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data/benchmarks/sen1floods11"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--t-max", type=int, default=0, help="CosineAnnealingLR T_max (0=same as epochs)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading checkpoint...", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint, device)
    load_checkpoint_teacher_weights(model, checkpoint["teacher"])
    model.eval()
    embed_dim = model.embed_dim
    print(f"  embed_dim={embed_dim}, token_mode={model.token_mode}")

    # Load datasets
    train_ds = Sen1Floods11Dataset(data_dir, data_dir / "splits/flood_train_data.csv")
    val_ds = Sen1Floods11Dataset(data_dir, data_dir / "splits/flood_valid_data.csv")
    test_ds = Sen1Floods11Dataset(data_dir, data_dir / "splits/flood_test_data.csv")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Step 1: Pre-extract features for all splits (frozen encoder)
    print("\nStep 1: Extracting frozen encoder features...", flush=True)

    def extract_all(dataset, label):
        all_features, all_labels, all_raw = [], [], []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, batch in enumerate(loader):
            features = extract_patch_features(model, batch["s2"], batch["s1"], device)
            all_features.append(features)  # [1, 64, 64, D]
            all_labels.append(batch["label"])  # [1, 512, 512]
            # Store raw S2+S1 concatenated for U-Net skip connections
            raw = torch.cat([batch["s2"], batch["s1"]], dim=1)  # [1, 14, 512, 512]
            all_raw.append(raw)
            if (i + 1) % 50 == 0:
                print(f"    {label}: {i+1}/{len(dataset)}", flush=True)
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_raw, dim=0)

    # Bolivia held-out event
    bolivia_split = data_dir / "splits/flood_bolivia_data.csv"
    bolivia_ds = Sen1Floods11Dataset(data_dir, bolivia_split) if bolivia_split.exists() else None

    train_features, train_labels, train_raw = extract_all(train_ds, "train")
    val_features, val_labels, val_raw = extract_all(val_ds, "val")
    test_features, test_labels, test_raw = extract_all(test_ds, "test")
    if bolivia_ds is not None:
        bolivia_features, bolivia_labels, bolivia_raw = extract_all(bolivia_ds, "bolivia")
    print(f"  Train features: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}")
    if bolivia_ds is not None:
        print(f"  Bolivia: {bolivia_features.shape}")

    # Free GPU memory from encoder
    del model
    torch.cuda.empty_cache()

    # Step 2: Train U-Net segmentation head
    print("\nStep 2: Training U-Net segmentation head...", flush=True)
    seg_head = UNetSegmentationHead(embed_dim=embed_dim, input_channels=14, num_classes=2, base_ch=64).to(device)
    print(f"  Seg head params: {sum(p.numel() for p in seg_head.parameters()):,}")
    optimizer = torch.optim.AdamW(seg_head.parameters(), lr=args.lr, weight_decay=0.01)
    t_max = args.t_max if args.t_max > 0 else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # Class weights (flood is rare ~13%, land ~20%, nodata ~66%)
    # Only train on valid pixels (label >= 0)
    best_val_iou = 0.0
    best_state = None

    for epoch in range(args.epochs):
        seg_head.train()
        epoch_loss = 0.0
        perm = torch.randperm(len(train_features))

        for i in range(0, len(train_features), args.batch_size):
            idx = perm[i:i + args.batch_size]
            feat = train_features[idx].to(device)       # [B, 64, 64, D]
            raw = train_raw[idx].to(device)             # [B, 14, 512, 512]
            lbl = train_labels[idx].to(device)           # [B, 512, 512]

            logits = seg_head(feat, raw)                  # [B, 2, 512, 512]

            # Mask out nodata pixels (label == -1)
            valid = lbl >= 0
            if valid.sum() == 0:
                continue

            loss = F.cross_entropy(logits, lbl.clamp(min=0), ignore_index=-1)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg_head.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validate
        seg_head.eval()
        all_pred, all_true, all_valid = [], [], []
        with torch.no_grad():
            for i in range(0, len(val_features), args.batch_size):
                feat = val_features[i:i + args.batch_size].to(device)
                raw = val_raw[i:i + args.batch_size].to(device)
                lbl = val_labels[i:i + args.batch_size]
                logits = seg_head(feat, raw)
                pred = logits.argmax(dim=1).cpu().numpy()
                lbl_np = lbl.numpy()
                valid = lbl_np >= 0
                all_pred.append(pred)
                all_true.append(lbl_np)
                all_valid.append(valid)

        pred_all = np.concatenate(all_pred)
        true_all = np.concatenate(all_true)
        valid_all = np.concatenate(all_valid)
        val_metrics = compute_iou(pred_all, true_all, valid_all)

        if val_metrics["iou_water"] > best_val_iou:
            best_val_iou = val_metrics["iou_water"]
            best_state = {k: v.clone() for k, v in seg_head.state_dict().items()}

        # Compute val loss
        val_loss = 0.0
        with torch.no_grad():
            for vi in range(0, len(val_features), args.batch_size):
                vf = val_features[vi:vi + args.batch_size].to(device)
                vr = val_raw[vi:vi + args.batch_size].to(device)
                vl = val_labels[vi:vi + args.batch_size].to(device)
                vlogits = seg_head(vf, vr)
                val_loss += F.cross_entropy(vlogits, vl.clamp(min=0), ignore_index=-1).item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_water_IoU={val_metrics['iou_water']:.4f} "
                  f"val_mIoU={val_metrics['mean_iou']:.4f}", flush=True)

    # Step 3: Evaluate on test set
    print("\nStep 3: Test evaluation...", flush=True)
    if best_state is not None:
        seg_head.load_state_dict(best_state)
    else:
        print("  WARNING: no improvement during training, using final weights")
    seg_head.eval()

    all_pred, all_true, all_valid = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_features), args.batch_size):
            feat = test_features[i:i + args.batch_size].to(device)
            raw = test_raw[i:i + args.batch_size].to(device)
            lbl = test_labels[i:i + args.batch_size]
            logits = seg_head(feat, raw)
            pred = logits.argmax(dim=1).cpu().numpy()
            lbl_np = lbl.numpy()
            valid = lbl_np >= 0
            all_pred.append(pred)
            all_true.append(lbl_np)
            all_valid.append(valid)

    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)
    valid_all = np.concatenate(all_valid)
    test_metrics = compute_iou(pred_all, true_all, valid_all)

    print(f"\n{'='*60}")
    print(f"SEN1FLOODS11 RESULTS — EWM {embed_dim}-dim")
    print(f"{'='*60}")
    print(f"  Water IoU:     {test_metrics['iou_water']:.4f}")
    print(f"  Non-water IoU: {test_metrics['iou_non_water']:.4f}")
    print(f"  Mean IoU:      {test_metrics['mean_iou']:.4f}")
    # Bolivia held-out event evaluation
    bolivia_metrics = None
    if bolivia_ds is not None:
        print("\nStep 4: Bolivia held-out event evaluation...", flush=True)
        all_pred, all_true, all_valid = [], [], []
        with torch.no_grad():
            for i in range(0, len(bolivia_features), args.batch_size):
                feat = bolivia_features[i:i + args.batch_size].to(device)
                raw = bolivia_raw[i:i + args.batch_size].to(device)
                lbl = bolivia_labels[i:i + args.batch_size]
                logits = seg_head(feat, raw)
                pred = logits.argmax(dim=1).cpu().numpy()
                lbl_np = lbl.numpy()
                valid = lbl_np >= 0
                all_pred.append(pred)
                all_true.append(lbl_np)
                all_valid.append(valid)
        pred_all = np.concatenate(all_pred)
        true_all = np.concatenate(all_true)
        valid_all = np.concatenate(all_valid)
        bolivia_metrics = compute_iou(pred_all, true_all, valid_all)

    print(f"\n{'='*60}")
    print(f"SEN1FLOODS11 RESULTS — EWM {embed_dim}-dim")
    print(f"{'='*60}")
    print(f"  Test set (90 chips):")
    print(f"    Water IoU:     {test_metrics['iou_water']:.4f}")
    print(f"    Non-water IoU: {test_metrics['iou_non_water']:.4f}")
    print(f"    Mean IoU:      {test_metrics['mean_iou']:.4f}")
    if bolivia_metrics is not None:
        print(f"  Bolivia held-out (15 chips, unseen flood event):")
        print(f"    Water IoU:     {bolivia_metrics['iou_water']:.4f}")
        print(f"    Non-water IoU: {bolivia_metrics['iou_non_water']:.4f}")
        print(f"    Mean IoU:      {bolivia_metrics['mean_iou']:.4f}")
    print(f"\nComparison:")
    print(f"  Prithvi-EO-1.0 (100M):  Water IoU ≈ 0.8046 (test), ≈ 0.8666 mIoU (Bolivia)")
    print(f"  Prithvi-EO-2.0 (600M):  Water IoU ≈ 0.84+")
    print(f"  Our EWM ({embed_dim}-dim): Water IoU = {test_metrics['iou_water']:.4f} (test)", end="")
    if bolivia_metrics is not None:
        print(f", {bolivia_metrics['iou_water']:.4f} (Bolivia)")
    else:
        print()

    # Save results
    import json
    results = {
        "benchmark": "Sen1Floods11",
        "checkpoint": str(args.checkpoint),
        "embed_dim": embed_dim,
        "test_iou_water": test_metrics["iou_water"],
        "test_iou_non_water": test_metrics["iou_non_water"],
        "test_mean_iou": test_metrics["mean_iou"],
        "best_val_iou_water": best_val_iou,
        "train_chips": len(train_ds),
        "test_chips": len(test_ds),
        "epochs": args.epochs,
        "seg_head_params": sum(p.numel() for p in seg_head.parameters()),
    }
    if bolivia_metrics is not None:
        results["bolivia_iou_water"] = bolivia_metrics["iou_water"]
        results["bolivia_iou_non_water"] = bolivia_metrics["iou_non_water"]
        results["bolivia_mean_iou"] = bolivia_metrics["mean_iou"]
        results["bolivia_chips"] = len(bolivia_ds)
    out_path = REPO_ROOT / "results" / "sen1floods11_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
