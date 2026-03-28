"""Explicit 2D and 3D tokenizers for multi-sensor EO inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SensorAdapterSpec:
    name: str
    in_channels: int
    supports_2d: bool
    supports_3d: bool
    supports_missing_embedding: bool = True


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
        tokenizer_mode: str | None = None,
        auto_2d_max_timesteps: int = 4,
        tubelet_size: int = 2,
        separate_sensor_encoders: bool = True,
    ):
        super().__init__()
        self.patch_px = int(patch_px)
        self.input_mode = str(input_mode)
        self.fusion_mode = str(fusion_mode)
        self.fusion_num_heads = max(1, int(fusion_num_heads))
        self.use_modality_embeddings = bool(use_modality_embeddings)
        self.use_patch_positional_encoding = bool(use_patch_positional_encoding)
        self.use_missing_modality_embeddings = bool(use_missing_modality_embeddings)
        raw_tokenizer_type = str(tokenizer_type).lower()
        self.tokenizer_type = raw_tokenizer_type
        self.tokenizer_mode = None if tokenizer_mode is None else str(tokenizer_mode).lower()
        self.auto_2d_max_timesteps = max(1, int(auto_2d_max_timesteps))
        self.tubelet_size = max(1, int(tubelet_size))
        self.separate_sensor_encoders = bool(separate_sensor_encoders)
        if self.fusion_mode not in {"early_mean", "late_concat", "cross_attend"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        if raw_tokenizer_type not in {"linear", "conv2d", "conv3d"}:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")
        if self.tokenizer_mode not in {None, "auto", "force_2d", "force_3d"}:
            raise ValueError(f"Unsupported tokenizer_mode: {self.tokenizer_mode}")
        if self.tokenizer_mode is not None and raw_tokenizer_type == "linear":
            raise ValueError("tokenizer_mode routing is not supported with tokenizer_type=linear")
        effective_fixed_tokenizer = raw_tokenizer_type
        if self.tokenizer_mode == "force_2d":
            effective_fixed_tokenizer = "conv2d"
        elif self.tokenizer_mode == "force_3d":
            effective_fixed_tokenizer = "conv3d"
        if self.tokenizer_mode in {"force_2d", "force_3d"}:
            self.tokenizer_type = effective_fixed_tokenizer

        self.embed_dim = int(embed_dim)
        self.s2_proj: nn.Module | None = None
        self.s1_proj: nn.Module | None = None
        self.hls_proj: nn.Module | None = None
        self.joint_proj: nn.Module | None = None
        self.s2_proj_2d: nn.Module | None = None
        self.s1_proj_2d: nn.Module | None = None
        self.hls_proj_2d: nn.Module | None = None
        self.joint_proj_2d: nn.Module | None = None
        self.s2_proj_3d: nn.Module | None = None
        self.s1_proj_3d: nn.Module | None = None
        self.hls_proj_3d: nn.Module | None = None
        self.joint_proj_3d: nn.Module | None = None
        self.extra_linear_proj = nn.ModuleDict()
        self.extra_proj_2d = nn.ModuleDict()
        self.extra_proj_3d = nn.ModuleDict()
        self.extra_mod_embeds = nn.ParameterDict()
        self.extra_missing_embeds = nn.ParameterDict()
        self._sensor_specs: dict[str, SensorAdapterSpec] = {}
        if self.tokenizer_mode == "auto":
            self.s2_proj_2d = ConvPatchEmbed2D(12, embed_dim, self.patch_px)
            self.s1_proj_2d = ConvPatchEmbed2D(2, embed_dim, self.patch_px)
            self.joint_proj_2d = ConvPatchEmbed2D(14, embed_dim, self.patch_px)
            self.hls_proj_2d = ConvPatchEmbed2D(6, embed_dim, self.patch_px)
            self.s2_proj_3d = ConvPatchEmbed3D(12, embed_dim, self.patch_px, self.tubelet_size)
            self.s1_proj_3d = ConvPatchEmbed3D(2, embed_dim, self.patch_px, self.tubelet_size)
            self.joint_proj_3d = ConvPatchEmbed3D(14, embed_dim, self.patch_px, self.tubelet_size)
            self.hls_proj_3d = ConvPatchEmbed3D(6, embed_dim, self.patch_px, self.tubelet_size)
            self.s2_proj = self.s2_proj_2d
            self.s1_proj = self.s1_proj_2d
            self.joint_proj = self.joint_proj_2d
            self.hls_proj = self.hls_proj_2d
        elif effective_fixed_tokenizer == "linear":
            self.s2_proj = SensorProjector(12 * self.patch_px * self.patch_px, embed_dim)
            self.s1_proj = SensorProjector(2 * self.patch_px * self.patch_px, embed_dim)
            self.joint_proj = SensorProjector(14 * self.patch_px * self.patch_px, embed_dim)
            self.hls_proj = SensorProjector(6 * self.patch_px * self.patch_px, embed_dim)
        elif effective_fixed_tokenizer == "conv2d":
            self.s2_proj = ConvPatchEmbed2D(12, embed_dim, self.patch_px)
            self.s1_proj = ConvPatchEmbed2D(2, embed_dim, self.patch_px)
            self.joint_proj = ConvPatchEmbed2D(14, embed_dim, self.patch_px)
            self.hls_proj = ConvPatchEmbed2D(6, embed_dim, self.patch_px)
            self.s2_proj_2d = self.s2_proj
            self.s1_proj_2d = self.s1_proj
            self.joint_proj_2d = self.joint_proj
            self.hls_proj_2d = self.hls_proj
        else:
            self.s2_proj = ConvPatchEmbed3D(12, embed_dim, self.patch_px, self.tubelet_size)
            self.s1_proj = ConvPatchEmbed3D(2, embed_dim, self.patch_px, self.tubelet_size)
            self.joint_proj = ConvPatchEmbed3D(14, embed_dim, self.patch_px, self.tubelet_size)
            self.hls_proj = ConvPatchEmbed3D(6, embed_dim, self.patch_px, self.tubelet_size)
            self.s2_proj_3d = self.s2_proj
            self.s1_proj_3d = self.s1_proj
            self.joint_proj_3d = self.joint_proj
            self.hls_proj_3d = self.hls_proj

        self.norm = nn.LayerNorm(embed_dim)
        self.s2_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s1_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.joint_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.hls_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s2_missing_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.s1_missing_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_modality_embeddings:
            nn.init.normal_(self.s2_mod_embed, std=1.0e-6)
            nn.init.normal_(self.s1_mod_embed, std=1.0e-6)
            nn.init.normal_(self.joint_mod_embed, std=1.0e-6)
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
        self._register_builtin_sensor_specs()

    def _register_builtin_sensor_specs(self) -> None:
        self._sensor_specs["s2"] = SensorAdapterSpec("s2", in_channels=12, supports_2d=True, supports_3d=True)
        self._sensor_specs["s1"] = SensorAdapterSpec("s1", in_channels=2, supports_2d=True, supports_3d=True)
        self._sensor_specs["joint"] = SensorAdapterSpec(
            "joint",
            in_channels=14,
            supports_2d=True,
            supports_3d=True,
            supports_missing_embedding=False,
        )
        self._sensor_specs["hls"] = SensorAdapterSpec(
            "hls",
            in_channels=6,
            supports_2d=True,
            supports_3d=True,
            supports_missing_embedding=False,
        )

    def describe_sensor_adapters(self) -> dict[str, SensorAdapterSpec]:
        return dict(self._sensor_specs)

    def sensor_names(self) -> tuple[str, ...]:
        return tuple(self._sensor_specs.keys())

    def has_sensor_adapter(self, name: str) -> bool:
        return str(name).strip().lower() in self._sensor_specs

    def register_runtime_sensor_adapter(
        self,
        name: str,
        *,
        in_channels: int,
        supports_2d: bool = True,
        supports_3d: bool = True,
        supports_missing_embedding: bool = True,
    ) -> None:
        sensor_name = str(name).strip().lower()
        if sensor_name in self._sensor_specs:
            raise ValueError(f"Sensor adapter {sensor_name!r} is already registered")
        if not supports_2d and not supports_3d:
            raise ValueError("A sensor adapter must support at least one tokenizer mode")
        channels = int(in_channels)
        if channels <= 0:
            raise ValueError(f"in_channels must be positive, got {channels}")
        self._sensor_specs[sensor_name] = SensorAdapterSpec(
            name=sensor_name,
            in_channels=channels,
            supports_2d=bool(supports_2d),
            supports_3d=bool(supports_3d),
            supports_missing_embedding=bool(supports_missing_embedding),
        )
        self.extra_linear_proj[sensor_name] = SensorProjector(channels * self.patch_px * self.patch_px, self.embed_dim)
        if supports_2d:
            self.extra_proj_2d[sensor_name] = ConvPatchEmbed2D(channels, self.embed_dim, self.patch_px)
        if supports_3d:
            self.extra_proj_3d[sensor_name] = ConvPatchEmbed3D(channels, self.embed_dim, self.patch_px, self.tubelet_size)
        self.extra_mod_embeds[sensor_name] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.use_modality_embeddings:
            nn.init.normal_(self.extra_mod_embeds[sensor_name], std=1.0e-6)
        if supports_missing_embedding:
            self.extra_missing_embeds[sensor_name] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            if self.use_missing_modality_embeddings:
                nn.init.normal_(self.extra_missing_embeds[sensor_name], std=1.0e-6)

    def resolve_tokenizer_type(self, timesteps: int | None = None) -> str:
        if self.tokenizer_mode == "force_2d":
            return "conv2d"
        if self.tokenizer_mode == "force_3d":
            return "conv3d"
        if self.tokenizer_mode == "auto":
            if timesteps is None:
                raise ValueError("timesteps are required when tokenizer_mode=auto")
            return "conv2d" if int(timesteps) <= self.auto_2d_max_timesteps else "conv3d"
        return self.tokenizer_type

    def _projector_for(self, modality: str, tokenizer_type: str) -> nn.Module:
        tokenizer_type = str(tokenizer_type).lower()
        if modality in self.extra_linear_proj:
            if tokenizer_type == "linear":
                projector = self.extra_linear_proj[modality]
            elif tokenizer_type == "conv2d":
                projector = self.extra_proj_2d[modality] if modality in self.extra_proj_2d else None
            elif tokenizer_type == "conv3d":
                projector = self.extra_proj_3d[modality] if modality in self.extra_proj_3d else None
            else:
                raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")
        else:
            if tokenizer_type == "linear":
                projector_map = {
                    "s2": self.s2_proj,
                    "s1": self.s1_proj,
                    "joint": self.joint_proj,
                    "hls": self.hls_proj,
                }
            elif tokenizer_type == "conv2d":
                projector_map = {
                    "s2": self.s2_proj if self.tokenizer_mode is None else self.s2_proj_2d,
                    "s1": self.s1_proj if self.tokenizer_mode is None else self.s1_proj_2d,
                    "joint": self.joint_proj if self.tokenizer_mode is None else self.joint_proj_2d,
                    "hls": self.hls_proj if self.tokenizer_mode is None else self.hls_proj_2d,
                }
            elif tokenizer_type == "conv3d":
                projector_map = {
                    "s2": self.s2_proj if self.tokenizer_mode is None else self.s2_proj_3d,
                    "s1": self.s1_proj if self.tokenizer_mode is None else self.s1_proj_3d,
                    "joint": self.joint_proj if self.tokenizer_mode is None else self.joint_proj_3d,
                    "hls": self.hls_proj if self.tokenizer_mode is None else self.hls_proj_3d,
                }
            else:
                raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")
            projector = projector_map.get(modality)
        if projector is None:
            raise ValueError(f"Unsupported modality {modality!r} for tokenizer_type={tokenizer_type}")
        return projector

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
            elif modality == "joint":
                mod_embed = self.joint_mod_embed
            elif modality in self.extra_mod_embeds:
                mod_embed = self.extra_mod_embeds[modality]
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
        elif modality in self.extra_missing_embeds:
            missing = self.extra_missing_embeds[modality]
        else:
            raise ValueError(f"Unsupported modality {modality!r}")
        missing = missing.to(device=device, dtype=dtype)
        if ndims == 4:
            missing = missing.unsqueeze(1)
        return missing

    def _project_step_tokens(self, inputs: torch.Tensor, *, modality: str, tokenizer_type: str | None = None) -> torch.Tensor:
        if tokenizer_type is None:
            resolved_tokenizer = self.tokenizer_type if self.tokenizer_mode is None else "conv2d"
        else:
            resolved_tokenizer = str(tokenizer_type).lower()
        if resolved_tokenizer == "linear":
            tokens = self._projector_for(modality, resolved_tokenizer)(self.patchify(inputs))
        elif resolved_tokenizer == "conv2d":
            tokens = self._projector_for(modality, resolved_tokenizer)(inputs)
        else:
            raise ValueError("Per-step token projection is not supported for conv3d tokenizers")
        return self._apply_token_enrichments(tokens, modality=modality)

    def _project_sequence_tokens(self, inputs: torch.Tensor, *, modality: str, tokenizer_type: str | None = None) -> torch.Tensor:
        if tokenizer_type is None:
            resolved_tokenizer = self.tokenizer_type if self.tokenizer_mode is None else "conv3d"
        else:
            resolved_tokenizer = str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            raise ValueError("Sequence token projection is only supported for conv3d tokenizers")
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self._projector_for(modality, resolved_tokenizer)(inputs)
        return self._apply_token_enrichments(tokens, modality=modality)

    @staticmethod
    def _apply_presence_mask_step(inputs: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        if present is None:
            return inputs
        present_mask = present.to(dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1, 1)
        return inputs * present_mask

    @staticmethod
    def _apply_presence_mask_sequence(inputs: torch.Tensor, present: torch.Tensor | None) -> torch.Tensor:
        if present is None:
            return inputs
        present_mask = present.to(dtype=inputs.dtype, device=inputs.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return inputs * present_mask

    def _group_count(self, timesteps: int, tokenizer_type: str | None = None) -> int:
        timesteps = int(timesteps)
        resolved_tokenizer = self.resolve_tokenizer_type(timesteps) if tokenizer_type is None else str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            return timesteps
        if timesteps % self.tubelet_size != 0:
            raise ValueError(
                f"Conv3d tubelet tokenizer requires timesteps divisible by tubelet_size={self.tubelet_size}, got {timesteps}"
            )
        return timesteps // self.tubelet_size

    def aggregate_temporal_dates(self, dates: torch.Tensor, tokenizer_type: str | None = None) -> torch.Tensor:
        resolved_tokenizer = self.resolve_tokenizer_type(dates.shape[1]) if tokenizer_type is None else str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            return dates
        groups = self._group_count(dates.shape[1], tokenizer_type=resolved_tokenizer)
        reshaped = dates.to(torch.float32).reshape(dates.shape[0], groups, self.tubelet_size)
        valid = reshaped > 0
        counts = valid.sum(dim=2)
        summed = torch.where(valid, reshaped, torch.zeros_like(reshaped)).sum(dim=2)
        averaged = torch.where(
            counts > 0,
            summed / counts.clamp_min(1).to(torch.float32),
            torch.full_like(summed, -1.0),
        )
        return averaged.round().to(torch.int64)

    def aggregate_temporal_boolean(self, value: torch.Tensor | None, tokenizer_type: str | None = None) -> torch.Tensor | None:
        if value is None:
            return value
        resolved_tokenizer = self.resolve_tokenizer_type(value.shape[1]) if tokenizer_type is None else str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            return value
        groups = self._group_count(value.shape[1], tokenizer_type=resolved_tokenizer)
        return value.to(torch.bool).reshape(value.shape[0], groups, self.tubelet_size).any(dim=2)

    def aggregate_temporal_presence(self, value: torch.Tensor | None, tokenizer_type: str | None = None) -> torch.Tensor | None:
        return self.aggregate_temporal_boolean(value, tokenizer_type=tokenizer_type)

    def aggregate_temporal_valid_mask(self, value: torch.Tensor | None, tokenizer_type: str | None = None) -> torch.Tensor | None:
        if value is None:
            return value
        resolved_tokenizer = self.resolve_tokenizer_type(value.shape[1]) if tokenizer_type is None else str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            return value
        groups = self._group_count(value.shape[1], tokenizer_type=resolved_tokenizer)
        reshaped = value.to(torch.float32).reshape(value.shape[0], groups, self.tubelet_size, value.shape[2], value.shape[3])
        return reshaped.mean(dim=2)

    def dense_layout(self, *, height: int, width: int, timesteps: int) -> tuple[int, int, int, int, int, int]:
        if height % self.patch_px != 0 or width % self.patch_px != 0:
            raise ValueError(f"Input size {(height, width)} is not divisible by patch size {self.patch_px}")
        grid_h = height // self.patch_px
        grid_w = width // self.patch_px
        patch_count = grid_h * grid_w
        token_multiplier = 1
        if self.input_mode == "s2s1" and self.separate_sensor_encoders and self.fusion_mode == "late_concat":
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
        extra_sensors: dict[str, torch.Tensor] | None = None,
        extra_sensor_present: dict[str, torch.Tensor] | None = None,
        tokenizer_type: str | None = None,
    ) -> list[torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if hls is not None:
            pieces.append(self._project_step_tokens(hls, modality="hls", tokenizer_type=tokenizer_type))
        if s2 is not None or s1 is not None:
            if s2 is None or s1 is None:
                raise ValueError("s2 and s1 must either both be provided or both be omitted")
            if not self.separate_sensor_encoders:
                joint_inputs = torch.cat(
                    [
                        self._apply_presence_mask_step(s2, s2_present),
                        self._apply_presence_mask_step(s1, s1_present),
                    ],
                    dim=1,
                )
                pieces.append(self._project_step_tokens(joint_inputs, modality="joint", tokenizer_type=tokenizer_type))
            else:
                s2_tokens = self._project_step_tokens(s2, modality="s2", tokenizer_type=tokenizer_type)
                s1_tokens = self._project_step_tokens(s1, modality="s1", tokenizer_type=tokenizer_type)
                if self.use_missing_modality_embeddings and s2_present is not None:
                    missing = (~s2_present.to(torch.bool)).view(-1, 1, 1).to(device=s2_tokens.device, dtype=s2_tokens.dtype)
                    s2_tokens = s2_tokens + (missing * self._missing_embed("s2", device=s2_tokens.device, dtype=s2_tokens.dtype, ndims=s2_tokens.ndim))
                if self.use_missing_modality_embeddings and s1_present is not None:
                    missing = (~s1_present.to(torch.bool)).view(-1, 1, 1).to(device=s1_tokens.device, dtype=s1_tokens.dtype)
                    s1_tokens = s1_tokens + (missing * self._missing_embed("s1", device=s1_tokens.device, dtype=s1_tokens.dtype, ndims=s1_tokens.ndim))
                pieces.append(s2_tokens)
                pieces.append(s1_tokens)
        for sensor_name, sensor_inputs in sorted((extra_sensors or {}).items()):
            if not self.has_sensor_adapter(sensor_name):
                raise ValueError(f"Unknown runtime sensor adapter {sensor_name!r}")
            sensor_tokens = self._project_step_tokens(sensor_inputs, modality=sensor_name, tokenizer_type=tokenizer_type)
            sensor_present = None if extra_sensor_present is None else extra_sensor_present.get(sensor_name)
            if self.use_missing_modality_embeddings and sensor_present is not None:
                missing = (~sensor_present.to(torch.bool)).view(-1, 1, 1).to(device=sensor_tokens.device, dtype=sensor_tokens.dtype)
                sensor_tokens = sensor_tokens + (
                    missing * self._missing_embed(sensor_name, device=sensor_tokens.device, dtype=sensor_tokens.dtype, ndims=sensor_tokens.ndim)
                )
            pieces.append(sensor_tokens)
        if not pieces:
            raise ValueError("At least one sensor input is required")
        return pieces

    def encode_sequence_modalities(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        extra_sensors: dict[str, torch.Tensor] | None = None,
        extra_sensor_present: dict[str, torch.Tensor] | None = None,
        tokenizer_type: str | None = None,
    ) -> list[torch.Tensor]:
        if tokenizer_type is None:
            if s2 is not None:
                time_source = s2
            elif hls is not None:
                time_source = hls
            elif extra_sensors:
                time_source = next(iter(extra_sensors.values()))
            else:
                raise ValueError("At least one sequence input is required")
            resolved_tokenizer = self.resolve_tokenizer_type(time_source.shape[1])
        else:
            resolved_tokenizer = str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            raise ValueError("encode_sequence_modalities is only supported for conv3d tokenizers")
        pieces: list[torch.Tensor] = []
        if hls is not None:
            pieces.append(self._project_sequence_tokens(hls, modality="hls", tokenizer_type=resolved_tokenizer))
        if s2 is not None or s1 is not None:
            if s2 is None or s1 is None:
                raise ValueError("s2 and s1 must either both be provided or both be omitted")
            if not self.separate_sensor_encoders:
                joint_inputs = torch.cat(
                    [
                        self._apply_presence_mask_sequence(s2, s2_present),
                        self._apply_presence_mask_sequence(s1, s1_present),
                    ],
                    dim=2,
                )
                pieces.append(self._project_sequence_tokens(joint_inputs, modality="joint", tokenizer_type=resolved_tokenizer))
            else:
                s2_tokens = self._project_sequence_tokens(s2, modality="s2", tokenizer_type=resolved_tokenizer)
                s1_tokens = self._project_sequence_tokens(s1, modality="s1", tokenizer_type=resolved_tokenizer)
                grouped_s2_present = self.aggregate_temporal_presence(s2_present, tokenizer_type=resolved_tokenizer)
                grouped_s1_present = self.aggregate_temporal_presence(s1_present, tokenizer_type=resolved_tokenizer)
                if self.use_missing_modality_embeddings and grouped_s2_present is not None:
                    missing = (~grouped_s2_present.to(torch.bool)).unsqueeze(-1).unsqueeze(-1).to(device=s2_tokens.device, dtype=s2_tokens.dtype)
                    s2_tokens = s2_tokens + (missing * self._missing_embed("s2", device=s2_tokens.device, dtype=s2_tokens.dtype, ndims=s2_tokens.ndim))
                if self.use_missing_modality_embeddings and grouped_s1_present is not None:
                    missing = (~grouped_s1_present.to(torch.bool)).unsqueeze(-1).unsqueeze(-1).to(device=s1_tokens.device, dtype=s1_tokens.dtype)
                    s1_tokens = s1_tokens + (missing * self._missing_embed("s1", device=s1_tokens.device, dtype=s1_tokens.dtype, ndims=s1_tokens.ndim))
                pieces.append(s2_tokens)
                pieces.append(s1_tokens)
        for sensor_name, sensor_inputs in sorted((extra_sensors or {}).items()):
            if not self.has_sensor_adapter(sensor_name):
                raise ValueError(f"Unknown runtime sensor adapter {sensor_name!r}")
            sensor_tokens = self._project_sequence_tokens(sensor_inputs, modality=sensor_name, tokenizer_type=resolved_tokenizer)
            grouped_present = None
            if extra_sensor_present is not None:
                grouped_present = self.aggregate_temporal_presence(extra_sensor_present.get(sensor_name), tokenizer_type=resolved_tokenizer)
            if self.use_missing_modality_embeddings and grouped_present is not None:
                missing = (~grouped_present.to(torch.bool)).unsqueeze(-1).unsqueeze(-1).to(device=sensor_tokens.device, dtype=sensor_tokens.dtype)
                sensor_tokens = sensor_tokens + (
                    missing * self._missing_embed(sensor_name, device=sensor_tokens.device, dtype=sensor_tokens.dtype, ndims=sensor_tokens.ndim)
                )
            pieces.append(sensor_tokens)
        if not pieces:
            raise ValueError("At least one sensor input is required")
        return pieces

    def step_token_template(self, tokens_per_step: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.fusion_mode == "late_concat" and self.input_mode == "s2s1" and self.separate_sensor_encoders:
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
            elif self.input_mode == "s2s1" and not self.separate_sensor_encoders:
                base = base + self.joint_mod_embed.to(device=device, dtype=dtype)
            else:
                base = base + self.s2_mod_embed.to(device=device, dtype=dtype)
        return base

    def fuse_tokens(self, pieces: list[torch.Tensor]) -> torch.Tensor:
        if not pieces:
            raise ValueError("No modality tokens were provided")
        if len(pieces) == 1:
            return pieces[0]
        reference = pieces[0]
        if reference.ndim == 3:
            cat_dim = 1
        elif reference.ndim == 4:
            cat_dim = 2
        else:
            raise ValueError(f"Unsupported token rank {reference.ndim}")
        if self.fusion_mode == "cross_attend":
            if self.cross_modal_fusion is None:
                raise RuntimeError("cross_attend fusion requested, but cross_modal_fusion is not initialized")
            fused = pieces[0]
            for other in pieces[1:]:
                if reference.ndim == 4:
                    batch, timesteps, patch_tokens, embed_dim = fused.shape
                    fused = self.cross_modal_fusion(
                        fused.reshape(batch * timesteps, patch_tokens, embed_dim),
                        other.reshape(batch * timesteps, patch_tokens, embed_dim),
                    ).reshape(batch, timesteps, patch_tokens, embed_dim)
                else:
                    fused = self.cross_modal_fusion(fused, other)
            return fused
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
        extra_sensors: dict[str, torch.Tensor] | None = None,
        extra_sensor_present: dict[str, torch.Tensor] | None = None,
        tokenizer_type: str | None = None,
    ) -> torch.Tensor:
        resolved_tokenizer = self.tokenizer_type if tokenizer_type is None else str(tokenizer_type).lower()
        if self.tokenizer_mode is not None and tokenizer_type is None:
            resolved_tokenizer = "conv2d"
        if resolved_tokenizer == "conv3d":
            raise ValueError("encode_step_tokens is not supported for conv3d tokenizers")
        combined = self.fuse_tokens(
            self.encode_step_modalities(
                s2=s2,
                s1=s1,
                hls=hls,
                s2_present=s2_present,
                s1_present=s1_present,
                extra_sensors=extra_sensors,
                extra_sensor_present=extra_sensor_present,
                tokenizer_type=resolved_tokenizer,
            )
        )
        return self.norm(combined)

    def encode_sequence_tokens(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        extra_sensors: dict[str, torch.Tensor] | None = None,
        extra_sensor_present: dict[str, torch.Tensor] | None = None,
        tokenizer_type: str | None = None,
    ) -> torch.Tensor:
        if tokenizer_type is None:
            if s2 is not None:
                time_source = s2
            elif hls is not None:
                time_source = hls
            elif extra_sensors:
                time_source = next(iter(extra_sensors.values()))
            else:
                raise ValueError("At least one sequence input is required")
            resolved_tokenizer = self.resolve_tokenizer_type(time_source.shape[1])
        else:
            resolved_tokenizer = str(tokenizer_type).lower()
        if resolved_tokenizer != "conv3d":
            raise ValueError("encode_sequence_tokens is only supported for conv3d tokenizers")
        combined = self.fuse_tokens(
            self.encode_sequence_modalities(
                s2=s2,
                s1=s1,
                hls=hls,
                s2_present=s2_present,
                s1_present=s1_present,
                extra_sensors=extra_sensors,
                extra_sensor_present=extra_sensor_present,
                tokenizer_type=resolved_tokenizer,
            )
        )
        return self.norm(combined)

    def forward(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        extra_sensors: dict[str, torch.Tensor] | None = None,
        extra_sensor_present: dict[str, torch.Tensor] | None = None,
        return_tokens: bool = False,
        tokenizer_type: str | None = None,
    ) -> torch.Tensor:
        resolved_tokenizer = self.tokenizer_type if tokenizer_type is None else str(tokenizer_type).lower()
        if self.tokenizer_mode is not None and tokenizer_type is None:
            resolved_tokenizer = "conv2d"
        if resolved_tokenizer == "conv3d":
            raise ValueError("SpatialEncoder.forward is only supported for per-step tokenizers; use encode_sequence_tokens for conv3d")
        combined = self.encode_step_tokens(
            s2=s2,
            s1=s1,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
            extra_sensors=extra_sensors,
            extra_sensor_present=extra_sensor_present,
            tokenizer_type=resolved_tokenizer,
        )
        if return_tokens:
            return combined
        return combined
