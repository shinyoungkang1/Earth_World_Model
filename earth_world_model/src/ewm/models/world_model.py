"""Temporal JEPA-style world model for the Earth World Model PoC."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ewm.models.encoder import SpatialEncoder
from ewm.models.transformer_blocks import FactorizedRopeBlock, RopeTransformerBlock


class TemporalMetadataEncoding(nn.Module):
    """Encode EO acquisition timing with cyclic and interval-aware features."""

    def __init__(self, embed_dim: int, *, use_time_gap_features: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.use_time_gap_features = bool(use_time_gap_features)
        self.proj = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        dates: torch.Tensor,
        *,
        span_days: torch.Tensor | None = None,
        valid_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dates = dates.to(torch.float32)
        batch, timesteps = dates.shape
        phase = (2.0 * math.pi * dates) / 366.0
        delta_prev = torch.zeros_like(dates)
        delta_next = torch.zeros_like(dates)
        if timesteps > 1:
            delta_prev[:, 1:] = dates[:, 1:] - dates[:, :-1]
            delta_next[:, :-1] = dates[:, 1:] - dates[:, :-1]
            delta_prev[:, 0] = delta_prev[:, 1]
            delta_next[:, -1] = delta_next[:, -2]
        if not self.use_time_gap_features:
            delta_prev.zero_()
            delta_next.zero_()
        progress = torch.linspace(0.0, 1.0, timesteps, device=dates.device, dtype=torch.float32).unsqueeze(0)
        progress = progress.expand(batch, -1)
        if span_days is None:
            span_days = torch.zeros_like(dates)
        else:
            span_days = span_days.to(device=dates.device, dtype=torch.float32)
        if not self.use_time_gap_features:
            span_days = torch.zeros_like(dates)
        if valid_fraction is None:
            valid_fraction = torch.ones_like(dates)
        else:
            valid_fraction = valid_fraction.to(device=dates.device, dtype=torch.float32)
        features = torch.stack(
            [
                torch.sin(phase),
                torch.cos(phase),
                dates / 366.0,
                progress,
                delta_prev / 366.0,
                delta_next / 366.0,
                span_days / 366.0,
                valid_fraction,
            ],
            dim=-1,
        )
        return self.proj(features)


class LatentRolloutHead(nn.Module):
    """Autonomous latent transition prior used for short rollout consistency."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim or embed_dim)
        self.transition = nn.GRUCell(self.embed_dim * 2, self.hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def rollout(
        self,
        state_sequence: torch.Tensor,
        temporal_context: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        if state_sequence.ndim != 3:
            raise ValueError(f"state_sequence must be [B, T, D], got {tuple(state_sequence.shape)}")
        batch, timesteps, embed_dim = state_sequence.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")
        rollout_horizon = max(0, min(int(horizon), timesteps - 1))
        if rollout_horizon == 0:
            return state_sequence.new_zeros((batch, 0, 0, self.embed_dim))
        start_count = timesteps - rollout_horizon
        hidden = state_sequence[:, :start_count, :]
        outputs: list[torch.Tensor] = []
        for step in range(rollout_horizon):
            cond = temporal_context[:, step + 1 : step + 1 + start_count, :]
            gru_input = torch.cat([hidden, cond], dim=-1).reshape(batch * start_count, -1)
            hidden = self.transition(gru_input, hidden.reshape(batch * start_count, -1)).reshape(batch, start_count, -1)
            outputs.append(self.out_proj(hidden))
        return torch.stack(outputs, dim=2)


class EarthWorldModel(nn.Module):
    """Small temporal embedding model for fixed-shape S2/S1 sequences."""

    def __init__(
        self,
        embed_dim: int = 128,
        patch_px: int = 8,
        temporal_depth: int = 4,
        predictor_depth: int = 2,
        num_heads: int = 4,
        input_mode: str = "s2s1",
        hierarchical_layers: list[int] | tuple[int, ...] | None = None,
        token_mode: str = "pooled",
        spatial_fusion: str = "early_mean",
        fusion_num_heads: int = 4,
        use_modality_embeddings: bool = False,
        use_patch_positional_encoding: bool = False,
        use_missing_modality_embeddings: bool = False,
        temporal_block_type: str = "standard",
        use_rope_temporal_attention: bool = False,
        rope_base: float = 10000.0,
        use_activation_checkpointing: bool = False,
        tokenizer_type: str = "linear",
        tubelet_size: int = 2,
        separate_sensor_encoders: bool = True,
        use_time_gap_features: bool = True,
        enable_latent_dynamics: bool = True,
        dynamics_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.temporal_depth = int(temporal_depth)
        self.input_mode = input_mode
        self.temporal_block_type = str(temporal_block_type)
        if self.temporal_block_type not in {"standard", "stage_d_rope", "factorized_rope"}:
            raise ValueError(f"Unsupported temporal_block_type: {self.temporal_block_type}")
        self.use_rope_temporal_attention = bool(use_rope_temporal_attention)
        self.rope_base = float(rope_base)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.use_time_gap_features = bool(use_time_gap_features)
        self.token_mode = str(token_mode)
        if self.token_mode not in {"pooled", "dense"}:
            raise ValueError(f"Unsupported token_mode: {self.token_mode}")
        self.enable_latent_dynamics = bool(enable_latent_dynamics)
        self.spatial = SpatialEncoder(
            embed_dim=embed_dim,
            patch_px=patch_px,
            input_mode=input_mode,
            fusion_mode=spatial_fusion,
            fusion_num_heads=fusion_num_heads,
            use_modality_embeddings=use_modality_embeddings,
            use_patch_positional_encoding=use_patch_positional_encoding,
            use_missing_modality_embeddings=use_missing_modality_embeddings,
            tokenizer_type=tokenizer_type,
            tubelet_size=tubelet_size,
            separate_sensor_encoders=separate_sensor_encoders,
        )
        self.temporal_pos = TemporalMetadataEncoding(embed_dim, use_time_gap_features=use_time_gap_features)
        self._last_tokens_per_timestep = 1

        normalized_hierarchical = self._normalize_layer_indices(hierarchical_layers, self.temporal_depth)
        self.hierarchical_layers = normalized_hierarchical
        if self.temporal_block_type == "factorized_rope":
            self.temporal_encoder_layers = nn.ModuleList(
                [
                    FactorizedRopeBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        dropout=0.1,
                        use_rope=self.use_rope_temporal_attention,
                        rope_base=self.rope_base,
                    )
                    for _ in range(temporal_depth)
                ]
            )
        elif self.temporal_block_type == "stage_d_rope":
            self.temporal_encoder_layers = nn.ModuleList(
                [
                    RopeTransformerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=4.0,
                        dropout=0.1,
                        use_rope=self.use_rope_temporal_attention,
                        rope_base=self.rope_base,
                    )
                    for _ in range(temporal_depth)
                ]
            )
        else:
            self.temporal_encoder_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=0.1,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(temporal_depth)
                ]
            )
        self.hierarchical_norms = nn.ModuleDict(
            {str(layer_idx): nn.LayerNorm(embed_dim) for layer_idx in self.hierarchical_layers}
        )

        if self.temporal_block_type == "factorized_rope":
            self.predictor = nn.ModuleList(
                [
                    FactorizedRopeBlock(
                        embed_dim=embed_dim,
                        num_heads=max(1, num_heads // 2),
                        mlp_ratio=2.0,
                        dropout=0.1,
                        use_rope=self.use_rope_temporal_attention,
                        rope_base=self.rope_base,
                    )
                    for _ in range(predictor_depth)
                ]
            )
        elif self.temporal_block_type == "stage_d_rope":
            self.predictor = nn.ModuleList(
                [
                    RopeTransformerBlock(
                        embed_dim=embed_dim,
                        num_heads=max(1, num_heads // 2),
                        mlp_ratio=2.0,
                        dropout=0.1,
                        use_rope=self.use_rope_temporal_attention,
                        rope_base=self.rope_base,
                    )
                    for _ in range(predictor_depth)
                ]
            )
        else:
            predictor_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=max(1, num_heads // 2),
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        target_levels = max(1, len(self.hierarchical_layers))
        self.pred_head = nn.Linear(embed_dim, embed_dim * target_levels)
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.latent_rollout_head = (
            LatentRolloutHead(embed_dim=embed_dim, hidden_dim=dynamics_hidden_dim)
            if self.enable_latent_dynamics
            else None
        )

    @staticmethod
    def _normalize_layer_indices(
        hierarchical_layers: list[int] | tuple[int, ...] | None,
        temporal_depth: int,
    ) -> list[int]:
        if not hierarchical_layers:
            return []
        normalized: list[int] = []
        for value in hierarchical_layers:
            idx = int(value)
            if idx < 0:
                idx = temporal_depth + idx
            if idx < 0 or idx >= temporal_depth:
                raise ValueError(
                    f"hierarchical layer index {value} resolves to {idx}, outside [0, {temporal_depth})"
                )
            if idx not in normalized:
                normalized.append(idx)
        return sorted(normalized)

    def _group_temporal_spans(self, dates: torch.Tensor) -> torch.Tensor:
        if self.spatial.tokenizer_type != "conv3d":
            return torch.zeros_like(dates, dtype=torch.float32)
        groups = self.spatial._group_count(dates.shape[1])
        grouped = dates.to(torch.float32).reshape(dates.shape[0], groups, self.spatial.tubelet_size)
        return grouped[:, :, -1] - grouped[:, :, 0]

    def _group_temporal_valid_fraction(self, mask: torch.Tensor | None, dates: torch.Tensor) -> torch.Tensor | None:
        if mask is None:
            return None
        if self.spatial.tokenizer_type != "conv3d":
            return mask.to(torch.float32)
        groups = self.spatial._group_count(dates.shape[1])
        reshaped = mask.to(torch.float32).reshape(mask.shape[0], groups, self.spatial.tubelet_size)
        return reshaped.mean(dim=2)

    def _temporal_metadata_embeddings(
        self,
        dates: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grouped_dates = self.spatial.aggregate_temporal_dates(dates)
        span_days = self._group_temporal_spans(dates)
        valid_fraction = self._group_temporal_valid_fraction(mask, dates)
        return self.temporal_pos(grouped_dates, span_days=span_days, valid_fraction=valid_fraction)

    @staticmethod
    def _dense_visibility_mask(
        batch_size: int,
        timesteps: int,
        tokens_per_timestep: int,
        *,
        device: torch.device,
        visible_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if visible_token_indices is None:
            return None
        total_tokens = timesteps * tokens_per_timestep
        visibility = torch.zeros(total_tokens, device=device, dtype=torch.bool)
        visibility[visible_token_indices] = True
        visibility = visibility.view(1, timesteps, tokens_per_timestep).expand(batch_size, -1, -1)
        return visibility

    @staticmethod
    def _flatten_dense_sequence(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError(f"Expected [B, T, P, D], got {tuple(hidden.shape)}")
        return hidden.reshape(hidden.shape[0], hidden.shape[1] * hidden.shape[2], hidden.shape[3])

    def _build_structured_timeline_inputs(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        visible_token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.token_mode != "dense":
            raise ValueError("Structured timeline inputs are only available in dense token mode")
        if self.spatial.tokenizer_type == "conv3d":
            if hls is not None:
                timeline = self.spatial.encode_sequence_tokens(hls=hls)
            else:
                if s2 is None or s1 is None:
                    raise ValueError("s2 and s1 are required when hls is not provided")
                timeline = self.spatial.encode_sequence_tokens(
                    s2=s2,
                    s1=s1,
                    s2_present=s2_present,
                    s1_present=s1_present,
                )
            grouped_dates = self.spatial.aggregate_temporal_dates(dates)
            grouped_mask = self.spatial.aggregate_temporal_boolean(mask)
        else:
            step_embeddings = []
            for timestep in range(dates.shape[1]):
                if hls is not None:
                    step_embeddings.append(self.spatial.encode_step_tokens(hls=hls[:, timestep]))
                else:
                    if s2 is None or s1 is None:
                        raise ValueError("s2 and s1 are required when hls is not provided")
                    step_embeddings.append(
                        self.spatial.encode_step_tokens(
                            s2=s2[:, timestep],
                            s1=s1[:, timestep],
                            s2_present=None if s2_present is None else s2_present[:, timestep],
                            s1_present=None if s1_present is None else s1_present[:, timestep],
                        )
                    )
            timeline = torch.stack(step_embeddings, dim=1)
            grouped_dates = dates
            grouped_mask = mask
        self._last_tokens_per_timestep = int(timeline.shape[2])
        timeline = timeline + self._temporal_metadata_embeddings(dates, mask=mask).unsqueeze(2)
        visibility_mask = self._dense_visibility_mask(
            timeline.shape[0],
            grouped_dates.shape[1],
            int(timeline.shape[2]),
            device=timeline.device,
            visible_token_indices=visible_token_indices,
        )
        if grouped_mask is not None:
            frame_visibility = grouped_mask.to(device=timeline.device, dtype=torch.bool).unsqueeze(-1).expand_as(timeline[..., 0])
            visibility_mask = frame_visibility if visibility_mask is None else (visibility_mask & frame_visibility)
        if visibility_mask is not None:
            timeline = timeline * visibility_mask.unsqueeze(-1).to(dtype=timeline.dtype)
        return timeline, grouped_dates.to(dtype=torch.long), visibility_mask

    def _run_temporal_encoder(
        self,
        timeline: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        return_hierarchical: bool = False,
    ) -> torch.Tensor:
        hidden = timeline
        hierarchical_outputs: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.temporal_encoder_layers):
            if isinstance(layer, FactorizedRopeBlock):
                if self.use_activation_checkpointing and self.training:
                    hidden = activation_checkpoint(
                        lambda tensor, pos, vis: layer(tensor, temporal_position_ids=pos, visibility_mask=vis),
                        hidden,
                        position_ids,
                        visibility_mask,
                        use_reentrant=False,
                    )
                else:
                    hidden = layer(hidden, temporal_position_ids=position_ids, visibility_mask=visibility_mask)
            elif isinstance(layer, RopeTransformerBlock):
                if self.use_activation_checkpointing and self.training:
                    hidden = activation_checkpoint(
                        lambda tensor, pos: layer(tensor, position_ids=pos),
                        hidden,
                        position_ids,
                        use_reentrant=False,
                    )
                else:
                    hidden = layer(hidden, position_ids=position_ids)
            else:
                hidden = layer(hidden)
            if return_hierarchical and layer_idx in self.hierarchical_layers:
                normalized = self.hierarchical_norms[str(layer_idx)](hidden)
                if normalized.ndim == 4:
                    normalized = self._flatten_dense_sequence(normalized)
                hierarchical_outputs[layer_idx] = normalized

        if return_hierarchical and self.hierarchical_layers:
            return torch.cat([hierarchical_outputs[layer_idx] for layer_idx in self.hierarchical_layers], dim=-1)
        return hidden

    def _run_predictor(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.predictor, nn.ModuleList):
            for layer in self.predictor:
                if isinstance(layer, FactorizedRopeBlock):
                    if self.use_activation_checkpointing and self.training:
                        hidden = activation_checkpoint(
                            lambda tensor, pos, vis: layer(tensor, temporal_position_ids=pos, visibility_mask=vis),
                            hidden,
                            position_ids,
                            visibility_mask,
                            use_reentrant=False,
                        )
                    else:
                        hidden = layer(hidden, temporal_position_ids=position_ids, visibility_mask=visibility_mask)
                else:
                    if self.use_activation_checkpointing and self.training:
                        hidden = activation_checkpoint(
                            lambda tensor, pos: layer(tensor, position_ids=pos),
                            hidden,
                            position_ids,
                            use_reentrant=False,
                        )
                    else:
                        hidden = layer(hidden, position_ids=position_ids)
            return hidden
        return self.predictor(hidden)

    def _timeline_position_ids(
        self,
        dates: torch.Tensor,
        *,
        tokens_per_timestep: int,
    ) -> torch.Tensor:
        position_ids = dates.to(dtype=torch.long)
        if self.token_mode == "dense":
            position_ids = position_ids.unsqueeze(-1).expand(-1, -1, tokens_per_timestep).reshape(position_ids.shape[0], -1)
        return position_ids

    def _build_timeline_inputs(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_rope_positions = self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention
        temporal_metadata = self._temporal_metadata_embeddings(dates, mask=mask)
        if self.spatial.tokenizer_type == "conv3d":
            if hls is not None:
                timeline = self.spatial.encode_sequence_tokens(hls=hls)
            else:
                if s2 is None or s1 is None:
                    raise ValueError("s2 and s1 are required when hls is not provided")
                timeline = self.spatial.encode_sequence_tokens(
                    s2=s2,
                    s1=s1,
                    s2_present=s2_present,
                    s1_present=s1_present,
                )
            encoded_dates = self.spatial.aggregate_temporal_dates(dates)
            encoded_mask = self.spatial.aggregate_temporal_boolean(mask)
            if self.token_mode == "dense":
                self._last_tokens_per_timestep = int(timeline.shape[2])
                if not use_rope_positions:
                    timeline = timeline + temporal_metadata.unsqueeze(2)
                if encoded_mask is not None:
                    timeline = timeline * encoded_mask.unsqueeze(-1).unsqueeze(-1).float()
                position_ids = self._timeline_position_ids(encoded_dates, tokens_per_timestep=self._last_tokens_per_timestep)
                timeline = timeline.reshape(timeline.shape[0], timeline.shape[1] * timeline.shape[2], timeline.shape[3])
                if token_indices is not None:
                    timeline = timeline[:, token_indices, :]
                    position_ids = position_ids[:, token_indices]
                return timeline, position_ids

            self._last_tokens_per_timestep = 1
            timeline = timeline.mean(dim=2)
            if not use_rope_positions:
                timeline = timeline + temporal_metadata
            if encoded_mask is not None:
                timeline = timeline * encoded_mask.unsqueeze(-1).float()
            position_ids = self._timeline_position_ids(encoded_dates, tokens_per_timestep=1)
            if token_indices is not None:
                raise ValueError("token_indices is only supported in dense token mode")
            return timeline, position_ids

        step_embeddings = []
        for timestep in range(dates.shape[1]):
            if hls is not None:
                spatial_tokens = self.spatial.encode_step_tokens(hls=hls[:, timestep])
            else:
                if s2 is None or s1 is None:
                    raise ValueError("s2 and s1 are required when hls is not provided")
                spatial_tokens = self.spatial.encode_step_tokens(
                    s2=s2[:, timestep],
                    s1=s1[:, timestep],
                    s2_present=None if s2_present is None else s2_present[:, timestep],
                    s1_present=None if s1_present is None else s1_present[:, timestep],
                )

            if self.token_mode == "dense":
                if use_rope_positions:
                    step_tokens = spatial_tokens
                else:
                    temporal = temporal_metadata[:, timestep : timestep + 1].unsqueeze(1)
                    step_tokens = spatial_tokens + temporal.squeeze(2)
                step_embeddings.append(step_tokens)
            else:
                pooled = spatial_tokens.mean(dim=1)
                step_embeddings.append(pooled)

        if self.token_mode == "dense":
            timeline = torch.stack(step_embeddings, dim=1)
            self._last_tokens_per_timestep = int(timeline.shape[2])
            if mask is not None:
                timeline = timeline * mask.unsqueeze(-1).unsqueeze(-1).float()
            position_ids = self._timeline_position_ids(dates, tokens_per_timestep=self._last_tokens_per_timestep)
            timeline = timeline.reshape(timeline.shape[0], timeline.shape[1] * timeline.shape[2], timeline.shape[3])
            if token_indices is not None:
                timeline = timeline[:, token_indices, :]
                position_ids = position_ids[:, token_indices]
            return timeline, position_ids

        self._last_tokens_per_timestep = 1
        timeline = torch.stack(step_embeddings, dim=1)
        if not use_rope_positions:
            timeline = timeline + temporal_metadata
        if mask is not None:
            timeline = timeline * mask.unsqueeze(-1).float()
        position_ids = self._timeline_position_ids(dates, tokens_per_timestep=1)
        if token_indices is not None:
            raise ValueError("token_indices is only supported in dense token mode")
        return timeline, position_ids

    def encode_timeline(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        token_indices: torch.Tensor | None = None,
        return_hierarchical: bool = False,
        return_structured: bool = False,
    ) -> torch.Tensor:
        if self.token_mode == "dense" and self.temporal_block_type == "factorized_rope":
            timeline, grouped_dates, visibility_mask = self._build_structured_timeline_inputs(
                s2=s2,
                s1=s1,
                dates=dates,
                mask=mask,
                hls=hls,
                s2_present=s2_present,
                s1_present=s1_present,
                visible_token_indices=token_indices,
            )
            encoded = self._run_temporal_encoder(
                timeline,
                position_ids=grouped_dates,
                visibility_mask=visibility_mask,
                return_hierarchical=return_hierarchical,
            )
            if return_hierarchical:
                if token_indices is not None:
                    return encoded[:, token_indices, :]
                return encoded
            if return_structured:
                return encoded
            flattened = self._flatten_dense_sequence(encoded)
            if token_indices is not None:
                flattened = flattened[:, token_indices, :]
            return flattened
        timeline, position_ids = self._build_timeline_inputs(
            s2=s2,
            s1=s1,
            dates=dates,
            mask=mask,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
            token_indices=token_indices,
        )
        encoded = self._run_temporal_encoder(timeline, position_ids=position_ids, return_hierarchical=return_hierarchical)
        if return_structured:
            if self.token_mode != "dense":
                raise ValueError("return_structured is only supported in dense token mode")
            tokens_per_step = self.last_tokens_per_timestep
            timesteps = position_ids.shape[1]
            return encoded.reshape(encoded.shape[0], timesteps, tokens_per_step, encoded.shape[-1])
        return encoded

    def predict_masked(
        self,
        visible_embeddings: torch.Tensor,
        visible_dates: torch.Tensor,
        masked_dates: torch.Tensor,
    ) -> torch.Tensor:
        _visible_predictions, masked_predictions = self.predict_with_context(
            visible_embeddings,
            visible_dates,
            masked_dates,
        )
        return masked_predictions

    def predict_with_context(
        self,
        visible_embeddings: torch.Tensor,
        visible_dates: torch.Tensor,
        masked_dates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = visible_embeddings.shape[0]
        visible_steps = visible_dates.shape[1]
        masked_steps = masked_dates.shape[1]

        if self.token_mode == "dense":
            tokens_per_step = max(1, visible_embeddings.shape[1] // max(visible_steps, 1))
            visible_token_count = visible_embeddings.shape[1]
            step_template = self.spatial.step_token_template(
                tokens_per_step,
                device=visible_embeddings.device,
                dtype=visible_embeddings.dtype,
            ).unsqueeze(1)
            mask_tokens = self.mask_token.view(1, 1, 1, -1).to(device=visible_embeddings.device, dtype=visible_embeddings.dtype)
            mask_tokens = (mask_tokens + step_template).expand(batch, masked_steps, tokens_per_step, -1)
            if not (self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention):
                temporal = self.temporal_pos(masked_dates).unsqueeze(2)
                mask_tokens = mask_tokens + temporal
            mask_tokens = mask_tokens.reshape(batch, masked_steps * tokens_per_step, -1)
            combined_position_ids = torch.cat(
                [
                    self._timeline_position_ids(visible_dates, tokens_per_timestep=tokens_per_step),
                    self._timeline_position_ids(masked_dates, tokens_per_timestep=tokens_per_step),
                ],
                dim=1,
            )
        else:
            visible_token_count = visible_embeddings.shape[1]
            mask_tokens = self.mask_token.view(1, 1, -1).expand(batch, masked_steps, -1)
            if not (self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention):
                mask_tokens = mask_tokens + self.temporal_pos(masked_dates)
            combined_position_ids = torch.cat(
                [
                    self._timeline_position_ids(visible_dates, tokens_per_timestep=1),
                    self._timeline_position_ids(masked_dates, tokens_per_timestep=1),
                ],
                dim=1,
            )
        combined = torch.cat([visible_embeddings, mask_tokens], dim=1)
        predicted = self._run_predictor(combined, position_ids=combined_position_ids)
        predicted = self.pred_head(predicted)
        return predicted[:, :visible_token_count, :], predicted[:, -mask_tokens.shape[1] :, :]

    def predict_with_context_tokens(
        self,
        visible_embeddings: torch.Tensor,
        visible_position_ids: torch.Tensor,
        masked_position_ids: torch.Tensor,
        masked_local_token_indices: torch.Tensor,
        *,
        visible_token_indices: torch.Tensor | None = None,
        masked_token_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_mode != "dense":
            raise ValueError("predict_with_context_tokens is only supported in dense token mode")
        batch = visible_embeddings.shape[0]
        visible_token_count = visible_embeddings.shape[1]
        tokens_per_step = self.last_tokens_per_timestep
        if self.temporal_block_type == "factorized_rope":
            if visible_token_indices is None or masked_token_indices is None:
                raise ValueError("Structured dense prediction requires visible_token_indices and masked_token_indices")
            total_tokens = int(max(visible_token_indices.max().item(), masked_token_indices.max().item()) + 1)
            total_steps = total_tokens // tokens_per_step
            full_tokens = visible_embeddings.new_zeros(batch, total_tokens, visible_embeddings.shape[-1])
            full_tokens[:, visible_token_indices, :] = visible_embeddings
            full_position_ids = visible_position_ids.new_zeros(batch, total_tokens)
            full_position_ids[:, visible_token_indices] = visible_position_ids
            template = self.spatial.step_token_template(
                tokens_per_step,
                device=visible_embeddings.device,
                dtype=visible_embeddings.dtype,
            ).expand(batch, -1, -1)
            gather_index = masked_local_token_indices.to(device=visible_embeddings.device, dtype=torch.long)
            gather_index = gather_index.view(1, -1, 1).expand(batch, -1, template.shape[-1])
            local_template = torch.gather(template, dim=1, index=gather_index)
            mask_tokens = self.mask_token.view(1, 1, -1).to(device=visible_embeddings.device, dtype=visible_embeddings.dtype)
            mask_tokens = mask_tokens + local_template + self.temporal_pos(masked_position_ids)
            full_tokens[:, masked_token_indices, :] = mask_tokens
            full_position_ids[:, masked_token_indices] = masked_position_ids
            structured = full_tokens.view(batch, total_steps, tokens_per_step, -1)
            grouped_position_ids = full_position_ids.view(batch, total_steps, tokens_per_step)[:, :, 0]
            predicted = self._run_predictor(structured, position_ids=grouped_position_ids)
            predicted = self.pred_head(predicted)
            predicted = self._flatten_dense_sequence(predicted)
            return predicted[:, visible_token_indices, :], predicted[:, masked_token_indices, :]
        template = self.spatial.step_token_template(
            tokens_per_step,
            device=visible_embeddings.device,
            dtype=visible_embeddings.dtype,
        ).expand(batch, -1, -1)
        gather_index = masked_local_token_indices.to(device=visible_embeddings.device, dtype=torch.long)
        gather_index = gather_index.view(1, -1, 1).expand(batch, -1, template.shape[-1])
        local_template = torch.gather(template, dim=1, index=gather_index)
        mask_tokens = self.mask_token.view(1, 1, -1).to(device=visible_embeddings.device, dtype=visible_embeddings.dtype)
        mask_tokens = mask_tokens + local_template
        if not (self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention):
            mask_tokens = mask_tokens + self.temporal_pos(masked_position_ids)
        combined = torch.cat([visible_embeddings, mask_tokens], dim=1)
        combined_position_ids = torch.cat([visible_position_ids, masked_position_ids], dim=1)
        predicted = self._run_predictor(combined, position_ids=combined_position_ids)
        predicted = self.pred_head(predicted)
        return predicted[:, :visible_token_count, :], predicted[:, visible_token_count:, :]

    def _state_sequence_from_dense_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        tokens_per_step = self.last_tokens_per_timestep
        if embeddings.shape[1] % tokens_per_step != 0:
            raise ValueError(
                f"Dense embedding sequence length {embeddings.shape[1]} is not divisible by tokens_per_step={tokens_per_step}"
            )
        timesteps = embeddings.shape[1] // tokens_per_step
        return embeddings.reshape(embeddings.shape[0], timesteps, tokens_per_step, embeddings.shape[-1]).mean(dim=2)

    def encode_state_sequence(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_mode == "dense":
            if self.temporal_block_type == "factorized_rope":
                structured = self.encode_timeline(
                    s2=s2,
                    s1=s1,
                    dates=dates,
                    mask=mask,
                    hls=hls,
                    s2_present=s2_present,
                    s1_present=s1_present,
                    return_structured=True,
                )
                grouped_dates = self.spatial.aggregate_temporal_dates(dates)
                return structured.mean(dim=2), grouped_dates
            dense = self.encode_timeline(
                s2=s2,
                s1=s1,
                dates=dates,
                mask=mask,
                hls=hls,
                s2_present=s2_present,
                s1_present=s1_present,
            )
            grouped_dates = self.spatial.aggregate_temporal_dates(dates)
            return self._state_sequence_from_dense_embeddings(dense), grouped_dates
        pooled = self.encode_timeline(
            s2=s2,
            s1=s1,
            dates=dates,
            mask=mask,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
        )
        return pooled, dates

    def rollout_state_predictions(
        self,
        state_sequence: torch.Tensor,
        dates: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        if self.latent_rollout_head is None:
            raise RuntimeError("Latent dynamics are disabled for this model")
        temporal_context = self.temporal_pos(dates)
        return self.latent_rollout_head.rollout(state_sequence, temporal_context, horizon=horizon)

    @property
    def last_tokens_per_timestep(self) -> int:
        return int(self._last_tokens_per_timestep)

    # ------------------------------------------------------------------
    # Embedding extraction modes for downstream tasks
    # ------------------------------------------------------------------

    _EMBEDDING_MODES = {"mean", "l2_mean", "predictor", "multilayer", "ensemble"}

    @torch.no_grad()
    def extract_embedding(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
        mode: str = "mean",
    ) -> torch.Tensor:
        """Extract a fixed-size embedding from an input sequence.

        Modes
        -----
        mean       : simple temporal mean-pool of encoder output (baseline).
        l2_mean    : L2-normalize each timestep, then mean-pool.
        predictor  : pass encoder output through pretrained predictor, then mean-pool.
        multilayer : concatenate mean-pooled outputs from encoder layers 1, 3, 5.
        ensemble   : multi-mask predictor ensemble (C(T,T//2) mask patterns).

        Returns ``[B, D]`` for most modes, ``[B, D * num_layers]`` for multilayer.
        """
        if mode not in self._EMBEDDING_MODES:
            raise ValueError(f"Unsupported embedding mode {mode!r}; choose from {self._EMBEDDING_MODES}")

        if mode == "multilayer":
            return self._extract_multilayer(s2, s1, dates, mask, hls, s2_present, s1_present)

        if mode == "ensemble":
            return self._extract_ensemble(s2, s1, dates, mask, hls, s2_present, s1_present)

        # Modes that operate on the full encoder timeline
        timeline, position_ids = self._build_timeline_inputs(
            s2=s2,
            s1=s1,
            dates=dates,
            mask=mask,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
        )

        if mode == "l2_mean":
            return F.normalize(timeline, dim=-1).mean(dim=1)

        if mode == "predictor":
            pred_out = self._run_predictor(timeline, position_ids=position_ids)
            pred_out = self.pred_head(pred_out)
            return pred_out.mean(dim=1)

        # default: "mean"
        return timeline.mean(dim=1)

    def _extract_multilayer(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate mean-pooled outputs from encoder layers 1, 3, 5."""
        timeline, position_ids = self._build_timeline_inputs(
            s2=s2,
            s1=s1,
            dates=dates,
            mask=mask,
            hls=hls,
            s2_present=s2_present,
            s1_present=s1_present,
        )

        # Run encoder layer-by-layer, collect intermediates
        extract_layers = {1, 3, 5} if self.temporal_depth >= 6 else set(range(0, self.temporal_depth, max(1, self.temporal_depth // 3)))
        hidden = timeline
        intermediates: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.temporal_encoder_layers):
            if isinstance(layer, RopeTransformerBlock):
                hidden = layer(hidden, position_ids=position_ids)
            else:
                hidden = layer(hidden)
            if layer_idx in extract_layers:
                intermediates.append(hidden.mean(dim=1))

        if not intermediates:
            intermediates.append(hidden.mean(dim=1))
        return torch.cat(intermediates, dim=-1)

    def _extract_ensemble(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        s2_present: torch.Tensor | None = None,
        s1_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-mask predictor ensemble matching JEPA training distribution.

        For T timesteps with 50% masking, generates C(T, T//2) mask patterns.
        Each pattern encodes visible timesteps, then runs predict_with_context.
        Averages the visible-position predictor outputs across all patterns.
        """
        if self.spatial.tokenizer_type == "conv3d":
            raise ValueError("ensemble extraction is not supported for conv3d tubelet tokenizers")
        T = dates.shape[1]
        num_visible = max(1, T // 2)
        all_patterns = list(itertools.combinations(range(T), num_visible))

        accumulated = torch.zeros(dates.shape[0], self.embed_dim, device=dates.device, dtype=dates.dtype if dates.is_floating_point() else torch.float32)

        for visible_idx in all_patterns:
            visible_idx_t = torch.tensor(visible_idx, device=dates.device, dtype=torch.long)
            masked_idx_t = torch.tensor([t for t in range(T) if t not in visible_idx], device=dates.device, dtype=torch.long)

            # Encode only visible timesteps
            visible_s2 = s2[:, visible_idx_t] if s2 is not None else None
            visible_s1 = s1[:, visible_idx_t] if s1 is not None else None
            visible_hls = hls[:, visible_idx_t] if hls is not None else None
            visible_s2_present = s2_present[:, visible_idx_t] if s2_present is not None else None
            visible_s1_present = s1_present[:, visible_idx_t] if s1_present is not None else None
            visible_dates = dates[:, visible_idx_t]
            visible_mask = mask[:, visible_idx_t] if mask is not None else None
            masked_dates = dates[:, masked_idx_t]

            visible_emb = self.encode_timeline(
                s2=visible_s2, s1=visible_s1, dates=visible_dates,
                mask=visible_mask, hls=visible_hls, s2_present=visible_s2_present, s1_present=visible_s1_present,
            )

            # Run predictor with visible + mask tokens (training distribution)
            vis_pred, _masked_pred = self.predict_with_context(
                visible_emb, visible_dates, masked_dates,
            )

            # Use visible-position predictor output (context prediction)
            accumulated = accumulated + vis_pred.mean(dim=1)

        return accumulated / len(all_patterns)
