"""Simple fixed-shape multi-sensor spatial encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SensorProjector(nn.Module):
    """Map flattened sensor patches into a shared embedding space."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(inputs)


class SpatialEncoder(nn.Module):
    """Patchify S2/S1 inputs and fuse them with mean pooling."""

    def __init__(
        self,
        embed_dim: int = 128,
        patch_px: int = 8,
        input_mode: str = "s2s1",
        fusion_mode: str = "early_mean",
        use_modality_embeddings: bool = False,
        use_patch_positional_encoding: bool = False,
    ):
        super().__init__()
        self.patch_px = int(patch_px)
        self.input_mode = input_mode
        self.fusion_mode = str(fusion_mode)
        self.use_modality_embeddings = bool(use_modality_embeddings)
        self.use_patch_positional_encoding = bool(use_patch_positional_encoding)
        if self.fusion_mode not in {"early_mean", "late_concat"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        self.s2_proj = SensorProjector(12 * self.patch_px * self.patch_px, embed_dim)
        self.s1_proj = SensorProjector(2 * self.patch_px * self.patch_px, embed_dim)
        self.hls_proj = SensorProjector(6 * self.patch_px * self.patch_px, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = int(embed_dim)
        self.s2_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s1_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hls_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_modality_embeddings:
            nn.init.normal_(self.s2_mod_embed, std=1.0e-6)
            nn.init.normal_(self.s1_mod_embed, std=1.0e-6)
            nn.init.normal_(self.hls_mod_embed, std=1.0e-6)

    def patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        patch = self.patch_px
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Input size {(height, width)} is not divisible by patch size {patch}")
        grid_h = height // patch
        grid_w = width // patch
        patches = inputs.reshape(batch, channels, grid_h, patch, grid_w, patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, grid_h * grid_w, channels * patch * patch)

    def patch_position_template(self, token_count: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid_size = int(math.isqrt(token_count))
        if grid_size * grid_size != token_count:
            positions = torch.arange(token_count, device=device, dtype=dtype).unsqueeze(-1)
            div = torch.exp(
                torch.arange(0, self.embed_dim, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / self.embed_dim)
            )
            template = torch.zeros(1, token_count, self.embed_dim, device=device, dtype=dtype)
            template[:, :, 0::2] = torch.sin(positions * div)
            template[:, :, 1::2] = torch.cos(positions * div)
            return template

        ys = torch.arange(grid_size, device=device, dtype=dtype)
        xs = torch.arange(grid_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)

        half_dim = self.embed_dim // 2
        div = torch.exp(
            torch.arange(0, half_dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(half_dim, 1))
        )
        pos_y = coords[:, 0:1]
        pos_x = coords[:, 1:2]
        encoding_y = torch.zeros(token_count, half_dim, device=device, dtype=dtype)
        encoding_x = torch.zeros(token_count, self.embed_dim - half_dim, device=device, dtype=dtype)
        encoding_y[:, 0::2] = torch.sin(pos_y * div)
        encoding_y[:, 1::2] = torch.cos(pos_y * div)
        encoding_x[:, 0::2] = torch.sin(pos_x * div[: encoding_x[:, 0::2].shape[1]])
        encoding_x[:, 1::2] = torch.cos(pos_x * div[: encoding_x[:, 1::2].shape[1]])
        return torch.cat([encoding_y, encoding_x], dim=-1).unsqueeze(0)

    def _apply_token_enrichments(
        self,
        tokens: torch.Tensor,
        *,
        modality: str,
    ) -> torch.Tensor:
        result = tokens
        if self.use_patch_positional_encoding:
            result = result + self.patch_position_template(
                result.shape[1],
                device=result.device,
                dtype=result.dtype,
            )
        if self.use_modality_embeddings:
            if modality == "s2":
                result = result + self.s2_mod_embed.to(device=result.device, dtype=result.dtype)
            elif modality == "s1":
                result = result + self.s1_mod_embed.to(device=result.device, dtype=result.dtype)
            elif modality == "hls":
                result = result + self.hls_mod_embed.to(device=result.device, dtype=result.dtype)
        return result

    def encode_modalities(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if hls is not None:
            hls_tokens = self.hls_proj(self.patchify(hls))
            pieces.append(self._apply_token_enrichments(hls_tokens, modality="hls"))
            return pieces

        if s2 is None or s1 is None:
            raise ValueError("s2 and s1 are required when hls is not provided")
        s2_tokens = self.s2_proj(self.patchify(s2))
        s1_tokens = self.s1_proj(self.patchify(s1))
        pieces.append(self._apply_token_enrichments(s2_tokens, modality="s2"))
        pieces.append(self._apply_token_enrichments(s1_tokens, modality="s1"))
        return pieces

    def step_token_template(self, tokens_per_step: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.fusion_mode == "late_concat" and self.input_mode == "s2s1":
            if tokens_per_step % 2 != 0:
                raise ValueError(f"Expected even tokens_per_step for s2s1 late_concat, got {tokens_per_step}")
            patch_count = tokens_per_step // 2
            base = self.patch_position_template(patch_count, device=device, dtype=dtype)
            pieces = []
            s2 = base
            s1 = base
            if self.use_modality_embeddings:
                s2 = s2 + self.s2_mod_embed.to(device=device, dtype=dtype)
                s1 = s1 + self.s1_mod_embed.to(device=device, dtype=dtype)
            pieces.extend([s2, s1])
            return torch.cat(pieces, dim=1)

        base = self.patch_position_template(tokens_per_step, device=device, dtype=dtype)
        if self.use_modality_embeddings:
            if self.input_mode == "hls6":
                base = base + self.hls_mod_embed.to(device=device, dtype=dtype)
            else:
                base = base + self.s2_mod_embed.to(device=device, dtype=dtype)
        return base

    def fuse_tokens(self, pieces: list[torch.Tensor]) -> torch.Tensor:
        if not pieces:
            raise ValueError("No modality tokens were provided")
        if self.fusion_mode == "early_mean":
            return torch.stack(pieces, dim=0).mean(dim=0)
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        combined = self.fuse_tokens(self.encode_modalities(s2=s2, s1=s1, hls=hls))
        combined = self.norm(combined)
        if return_tokens:
            return combined
        return combined
