"""Explicit 2D and 3D tokenizers for multi-sensor EO inputs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ewm.models.transformer_blocks import CrossModalFusionBlock


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


class ConvPatchEmbed2D(nn.Module):
    """Convolutional per-frame patch embedding."""

    def __init__(self, in_channels: int, embed_dim: int, patch_px: int):
        super().__init__()
        self.patch_px = int(patch_px)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_px,
            stride=self.patch_px,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"ConvPatchEmbed2D expects [B, C, H, W], got {tuple(inputs.shape)}")
        outputs = self.proj(inputs)
        batch, channels, grid_h, grid_w = outputs.shape
        return outputs.reshape(batch, channels, grid_h * grid_w).transpose(1, 2)


class ConvPatchEmbed3D(nn.Module):
    """Convolutional tubelet embedding over time and space."""

    def __init__(self, in_channels: int, embed_dim: int, patch_px: int, tubelet_size: int):
        super().__init__()
        self.patch_px = int(patch_px)
        self.tubelet_size = int(tubelet_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(self.tubelet_size, self.patch_px, self.patch_px),
            stride=(self.tubelet_size, self.patch_px, self.patch_px),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(f"ConvPatchEmbed3D expects [B, C, T, H, W], got {tuple(inputs.shape)}")
        outputs = self.proj(inputs)
        batch, channels, groups, grid_h, grid_w = outputs.shape
        return outputs.permute(0, 2, 3, 4, 1).reshape(batch, groups, grid_h * grid_w, channels)


class SpatialEncoder(nn.Module):
    """Patchify S2/S1 inputs with explicit 2D or 3D tokenizers."""

    def __init__(
        self,
        embed_dim: int = 128,
        patch_px: int = 8,
        input_mode: str = "s2s1",
        fusion_mode: str = "early_mean",
        fusion_num_heads: int = 4,
        use_modality_embeddings: bool = False,
        use_patch_positional_encoding: bool = False,
        use_missing_modality_embeddings: bool = False,
        tokenizer_type: str = "linear",
        tubelet_size: int = 2,
    ):
        super().__init__()
        self.patch_px = int(patch_px)
        self.input_mode = str(input_mode)
        self.fusion_mode = str(fusion_mode)
        self.fusion_num_heads = max(1, int(fusion_num_heads))
        self.use_modality_embeddings = bool(use_modality_embeddings)
        self.use_patch_positional_encoding = bool(use_patch_positional_encoding)
        self.use_missing_modality_embeddings = bool(use_missing_modality_embeddings)
        self.tokenizer_type = str(tokenizer_type).lower()
        self.tubelet_size = max(1, int(tubelet_size))
        if self.fusion_mode not in {"early_mean", "late_concat", "cross_attend"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if self.tokenizer_type not in {"linear", "conv2d", "conv3d"}:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")

        self.embed_dim = int(embed_dim)
        if self.tokenizer_type == "linear":
            self.s2_proj = SensorProjector(12 * self.patch_px * self.patch_px, embed_dim)
            self.s1_proj = SensorProjector(2 * self.patch_px * self.patch_px, embed_dim)
            self.hls_proj = SensorProjector(6 * self.patch_px * self.patch_px, embed_dim)
        elif self.tokenizer_type == "conv2d":
            self.s2_proj = ConvPatchEmbed2D(12, embed_dim, self.patch_px)
            self.s1_proj = ConvPatchEmbed2D(2, embed_dim, self.patch_px)
            self.hls_proj = ConvPatchEmbed2D(6, embed_dim, self.patch_px)
        else:
            self.s2_proj = ConvPatchEmbed3D(12, embed_dim, self.patch_px, self.tubelet_size)
            self.s1_proj = ConvPatchEmbed3D(2, embed_dim, self.patch_px, self.tubelet_size)
            self.hls_proj = ConvPatchEmbed3D(6, embed_dim, self.patch_px, self.tubelet_size)

        self.norm = nn.LayerNorm(embed_dim)
        self.s2_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s1_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hls_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s2_missing_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s1_missing_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_modality_embeddings:
            nn.init.normal_(self.s2_mod_embed, std=1.0e-6)
            nn.init.normal_(self.s1_mod_embed, std=1.0e-6)
            nn.init.normal_(self.hls_mod_embed, std=1.0e-6)
        if self.use_missing_modality_embeddings:
            nn.init.normal_(self.s2_missing_embed, std=1.0e-6)
            nn.init.normal_(self.s1_missing_embed, std=1.0e-6)
        self.cross_modal_fusion = None
        if self.input_mode == "s2s1" and self.fusion_mode == "cross_attend":
            self.cross_modal_fusion = CrossModalFusionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.fusion_num_heads,
                mlp_ratio=2.0,
                dropout=0.1,
            )

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

    def _apply_token_enrichments(
        self,
        tokens: torch.Tensor,
        *,
        modality: str,
    ) -> torch.Tensor:
        result = tokens
        if self.use_patch_positional_encoding:
            patch_token_count = result.shape[-2]
            patch_template = self.patch_position_template(
                patch_token_count,
                device=result.device,
                dtype=result.dtype,
            )
            if result.ndim == 4:
                patch_template = patch_template.unsqueeze(1)
            result = result + patch_template
        if self.use_modality_embeddings:
            if modality == "s2":
                mod_embed = self.s2_mod_embed
            elif modality == "s1":
                mod_embed = self.s1_mod_embed
            elif modality == "hls":
                mod_embed = self.hls_mod_embed
            else:
                raise ValueError(f"Unsupported modality {modality!r}")
            mod_embed = mod_embed.to(device=result.device, dtype=result.dtype)
            if result.ndim == 4:
                mod_embed = mod_embed.unsqueeze(1)
            result = result + mod_embed
        return result

    def _missing_embed(self, modality: str, *, device: torch.device, dtype: torch.dtype, ndims: int) -> torch.Tensor:
        if modality == "s2":
            missing = self.s2_missing_embed
        elif modality == "s1":
            missing = self.s1_missing_embed
        else:
            raise ValueError(f"Unsupported modality {modality!r}")
        missing = missing.to(device=device, dtype=dtype)
        if ndims == 4:
            missing = missing.unsqueeze(1)
        return missing

    def _project_step_tokens(self, inputs: torch.Tensor, *, modality: str) -> torch.Tensor:
        if self.tokenizer_type == "linear":
            if modality == "s2":
                tokens = self.s2_proj(self.patchify(inputs))
            elif modality == "s1":
                tokens = self.s1_proj(self.patchify(inputs))
            elif modality == "hls":
                tokens = self.hls_proj(self.patchify(inputs))
            else:
                raise ValueError(f"Unsupported modality {modality!r}")
        elif self.tokenizer_type == "conv2d":
            if modality == "s2":
                tokens = self.s2_proj(inputs)
            elif modality == "s1":
                tokens = self.s1_proj(inputs)
            elif modality == "hls":
                tokens = self.hls_proj(inputs)
            else:
                raise ValueError(f"Unsupported modality {modality!r}")
        else:
            raise ValueError("Per-step token projection is not supported for conv3d tokenizers")
        return self._apply_token_enrichments(tokens, modality=modality)

    def _project_sequence_tokens(self, inputs: torch.Tensor, *, modality: str) -> torch.Tensor:
        if self.tokenizer_type != "conv3d":
            raise ValueError("Sequence token projection is only supported for conv3d tokenizers")
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        if modality == "s2":
            tokens = self.s2_proj(inputs)
        elif modality == "s1":
            tokens = self.s1_proj(inputs)
        elif modality == "hls":
            tokens = self.hls_proj(inputs)
        else:
            raise ValueError(f"Unsupported modality {modality!r}")
        return self._apply_token_enrichments(tokens, modality=modality)

    def _group_count(self, timesteps: int) -> int:
        timesteps = int(timesteps)
        if self.tokenizer_type != "conv3d":
            return timesteps
        if timesteps % self.tubelet_size != 0:
            raise ValueError(
                f"Conv3d tubelet tokenizer requires timesteps divisible by tubelet_size={self.tubelet_size}, got {timesteps}"
            )
        return timesteps // self.tubelet_size

    def aggregate_temporal_dates(self, dates: torch.Tensor) -> torch.Tensor:
        if self.tokenizer_type != "conv3d":
            return dates
        groups = self._group_count(dates.shape[1])
        return dates.to(torch.float32).reshape(dates.shape[0], groups, self.tubelet_size).mean(dim=2).round().to(torch.int64)

    def aggregate_temporal_boolean(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None or self.tokenizer_type != "conv3d":
            return value
        groups = self._group_count(value.shape[1])
        return value.to(torch.bool).reshape(value.shape[0], groups, self.tubelet_size).any(dim=2)

    def aggregate_temporal_presence(self, value: torch.Tensor | None) -> torch.Tensor | None:
        return self.aggregate_temporal_boolean(value)

    def aggregate_temporal_valid_mask(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None or self.tokenizer_type != "conv3d":
            return value
        groups = self._group_count(value.shape[1])
        reshaped = value.to(torch.float32).reshape(value.shape[0], groups, self.tubelet_size, value.shape[2], value.shape[3])
        return reshaped.mean(dim=2)

    def dense_layout(self, *, height: int, width: int, timesteps: int) -> tuple[int, int, int, int, int, int]:
        if height % self.patch_px != 0 or width % self.patch_px != 0:
            raise ValueError(f"Input size {(height, width)} is not divisible by patch size {self.patch_px}")
        grid_h = height // self.patch_px
        grid_w = width // self.patch_px
        patch_count = grid_h * grid_w
        token_multiplier = 1
        if self.input_mode == "s2s1" and self.fusion_mode == "late_concat":
            token_multiplier = 2
        tokens_per_timestep = patch_count * token_multiplier
        temporal_groups = self._group_count(timesteps)
        return temporal_groups, tokens_per_timestep, patch_count, token_multiplier, grid_h, grid_w

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

    def encode_step_modalities(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if hls is not None:
            pieces.append(self._project_step_tokens(hls, modality="hls"))
            return pieces

        if s2 is None or s1 is None:
            raise ValueError("s2 and s1 are required when hls is not provided")
        s2_tokens = self._project_step_tokens(s2, modality="s2")
        s1_tokens = self._project_step_tokens(s1, modality="s1")
        if self.use_missing_modality_embeddings and s2_present is not None:
            missing = (~s2_present.to(torch.bool)).view(-1, 1, 1).to(device=s2_tokens.device, dtype=s2_tokens.dtype)
            s2_tokens = s2_tokens + (missing * self._missing_embed("s2", device=s2_tokens.device, dtype=s2_tokens.dtype, ndims=s2_tokens.ndim))
        if self.use_missing_modality_embeddings and s1_present is not None:
            missing = (~s1_present.to(torch.bool)).view(-1, 1, 1).to(device=s1_tokens.device, dtype=s1_tokens.dtype)
            s1_tokens = s1_tokens + (missing * self._missing_embed("s1", device=s1_tokens.device, dtype=s1_tokens.dtype, ndims=s1_tokens.ndim))
        pieces.append(s2_tokens)
        pieces.append(s1_tokens)
        return pieces

    def encode_sequence_modalities(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.tokenizer_type != "conv3d":
            raise ValueError("encode_sequence_modalities is only supported for conv3d tokenizers")
        pieces: list[torch.Tensor] = []
        if hls is not None:
            pieces.append(self._project_sequence_tokens(hls, modality="hls"))
            return pieces

        if s2 is None or s1 is None:
            raise ValueError("s2 and s1 are required when hls is not provided")
        s2_tokens = self._project_sequence_tokens(s2, modality="s2")
        s1_tokens = self._project_sequence_tokens(s1, modality="s1")
        grouped_s2_present = self.aggregate_temporal_presence(s2_present)
        grouped_s1_present = self.aggregate_temporal_presence(s1_present)
        if self.use_missing_modality_embeddings and grouped_s2_present is not None:
            missing = (~grouped_s2_present.to(torch.bool)).unsqueeze(-1).unsqueeze(-1).to(device=s2_tokens.device, dtype=s2_tokens.dtype)
            s2_tokens = s2_tokens + (missing * self._missing_embed("s2", device=s2_tokens.device, dtype=s2_tokens.dtype, ndims=s2_tokens.ndim))
        if self.use_missing_modality_embeddings and grouped_s1_present is not None:
            missing = (~grouped_s1_present.to(torch.bool)).unsqueeze(-1).unsqueeze(-1).to(device=s1_tokens.device, dtype=s1_tokens.dtype)
            s1_tokens = s1_tokens + (missing * self._missing_embed("s1", device=s1_tokens.device, dtype=s1_tokens.dtype, ndims=s1_tokens.ndim))
        pieces.append(s2_tokens)
        pieces.append(s1_tokens)
        return pieces

    def step_token_template(self, tokens_per_step: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.fusion_mode == "late_concat" and self.input_mode == "s2s1":
            if tokens_per_step % 2 != 0:
                raise ValueError(f"Expected even tokens_per_step for s2s1 late_concat, got {tokens_per_step}")
            patch_count = tokens_per_step // 2
            base = self.patch_position_template(patch_count, device=device, dtype=dtype)
            s2 = base
            s1 = base
            if self.use_modality_embeddings:
                s2 = s2 + self.s2_mod_embed.to(device=device, dtype=dtype)
                s1 = s1 + self.s1_mod_embed.to(device=device, dtype=dtype)
            return torch.cat([s2, s1], dim=1)

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
        reference = pieces[0]
        if reference.ndim == 3:
            cat_dim = 1
        elif reference.ndim == 4:
            cat_dim = 2
        else:
            raise ValueError(f"Unsupported token rank {reference.ndim}")
        if self.fusion_mode == "cross_attend":
            if len(pieces) != 2:
                raise ValueError("cross_attend fusion requires exactly two modality token tensors")
            left, right = pieces
            if reference.ndim == 4:
                batch, timesteps, patch_tokens, embed_dim = left.shape
                fused = self.cross_modal_fusion(
                    left.reshape(batch * timesteps, patch_tokens, embed_dim),
                    right.reshape(batch * timesteps, patch_tokens, embed_dim),
                )
                return fused.reshape(batch, timesteps, patch_tokens, embed_dim)
            return self.cross_modal_fusion(left, right)
        if self.fusion_mode == "early_mean":
            return torch.stack(pieces, dim=0).mean(dim=0)
        return torch.cat(pieces, dim=cat_dim)

    def encode_step_tokens(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.tokenizer_type == "conv3d":
            raise ValueError("encode_step_tokens is not supported for conv3d tokenizers")
        combined = self.fuse_tokens(
            self.encode_step_modalities(s2=s2, s1=s1, hls=hls, s2_present=s2_present, s1_present=s1_present)
        )
        return self.norm(combined)

    def encode_sequence_tokens(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.tokenizer_type != "conv3d":
            raise ValueError("encode_sequence_tokens is only supported for conv3d tokenizers")
        combined = self.fuse_tokens(
            self.encode_sequence_modalities(s2=s2, s1=s1, hls=hls, s2_present=s2_present, s1_present=s1_present)
        )
        return self.norm(combined)

    def forward(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        if self.tokenizer_type == "conv3d":
            raise ValueError("SpatialEncoder.forward is only supported for per-step tokenizers; use encode_sequence_tokens for conv3d")
        combined = self.encode_step_tokens(
            s2=s2,
            s1=s1,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
        )
        if return_tokens:
            return combined
        return combined
