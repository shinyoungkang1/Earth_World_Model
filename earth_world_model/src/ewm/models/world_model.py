"""Temporal JEPA-style world model for the Earth World Model PoC."""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ewm.models.encoder import SpatialEncoder
from ewm.models.transformer_blocks import RopeTransformerBlock


class TemporalPositionalEncoding(nn.Module):
    """Encode day-of-year integers into embedding vectors."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        batch, timesteps = dates.shape
        embed_dim = self.embed_dim
        positions = dates.float().unsqueeze(-1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, device=dates.device, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encodings = torch.zeros(batch, timesteps, embed_dim, device=dates.device)
        encodings[:, :, 0::2] = torch.sin(positions * div)
        encodings[:, :, 1::2] = torch.cos(positions * div)
        return self.proj(encodings)


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
        use_modality_embeddings: bool = False,
        use_patch_positional_encoding: bool = False,
        temporal_block_type: str = "standard",
        use_rope_temporal_attention: bool = False,
        rope_base: float = 10000.0,
        use_activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.temporal_depth = int(temporal_depth)
        self.input_mode = input_mode
        self.temporal_block_type = str(temporal_block_type)
        if self.temporal_block_type not in {"standard", "stage_d_rope"}:
            raise ValueError(f"Unsupported temporal_block_type: {self.temporal_block_type}")
        self.use_rope_temporal_attention = bool(use_rope_temporal_attention)
        self.rope_base = float(rope_base)
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.token_mode = str(token_mode)
        if self.token_mode not in {"pooled", "dense"}:
            raise ValueError(f"Unsupported token_mode: {self.token_mode}")
        self.spatial = SpatialEncoder(
            embed_dim=embed_dim,
            patch_px=patch_px,
            input_mode=input_mode,
            fusion_mode=spatial_fusion,
            use_modality_embeddings=use_modality_embeddings,
            use_patch_positional_encoding=use_patch_positional_encoding,
        )
        self.temporal_pos = TemporalPositionalEncoding(embed_dim)
        self._last_tokens_per_timestep = 1

        normalized_hierarchical = self._normalize_layer_indices(hierarchical_layers, self.temporal_depth)
        self.hierarchical_layers = normalized_hierarchical
        if self.temporal_block_type == "stage_d_rope":
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

        if self.temporal_block_type == "stage_d_rope":
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

    def _run_temporal_encoder(
        self,
        timeline: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        return_hierarchical: bool = False,
    ) -> torch.Tensor:
        hidden = timeline
        hierarchical_outputs: dict[int, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.temporal_encoder_layers):
            if isinstance(layer, RopeTransformerBlock):
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
                hierarchical_outputs[layer_idx] = self.hierarchical_norms[str(layer_idx)](hidden)

        if return_hierarchical and self.hierarchical_layers:
            return torch.cat([hierarchical_outputs[layer_idx] for layer_idx in self.hierarchical_layers], dim=-1)
        return hidden

    def _run_predictor(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(self.predictor, nn.ModuleList):
            for layer in self.predictor:
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

    def encode_timeline(
        self,
        s2: torch.Tensor | None,
        s1: torch.Tensor | None,
        dates: torch.Tensor,
        mask: torch.Tensor | None = None,
        hls: torch.Tensor | None = None,
        return_hierarchical: bool = False,
    ) -> torch.Tensor:
        step_embeddings = []
        use_rope_positions = self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention
        for timestep in range(dates.shape[1]):
            if hls is not None:
                spatial_tokens = self.spatial(hls=hls[:, timestep], return_tokens=True)
            else:
                if s2 is None or s1 is None:
                    raise ValueError("s2 and s1 are required when hls is not provided")
                spatial_tokens = self.spatial(s2=s2[:, timestep], s1=s1[:, timestep], return_tokens=True)

            if self.token_mode == "dense":
                if use_rope_positions:
                    step_tokens = spatial_tokens
                else:
                    temporal = self.temporal_pos(dates[:, timestep : timestep + 1]).unsqueeze(1)
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
        else:
            self._last_tokens_per_timestep = 1
            timeline = torch.stack(step_embeddings, dim=1)
            if not use_rope_positions:
                timeline = timeline + self.temporal_pos(dates)
            if mask is not None:
                timeline = timeline * mask.unsqueeze(-1).float()
            position_ids = self._timeline_position_ids(dates, tokens_per_timestep=1)
        return self._run_temporal_encoder(timeline, position_ids=position_ids, return_hierarchical=return_hierarchical)

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
            return self._extract_multilayer(s2, s1, dates, mask, hls)

        if mode == "ensemble":
            return self._extract_ensemble(s2, s1, dates, mask, hls)

        # Modes that operate on the full encoder timeline
        timeline = self.encode_timeline(s2=s2, s1=s1, dates=dates, mask=mask, hls=hls)

        if mode == "l2_mean":
            return F.normalize(timeline, dim=-1).mean(dim=1)

        if mode == "predictor":
            position_ids = self._timeline_position_ids(
                dates, tokens_per_timestep=self._last_tokens_per_timestep,
            )
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
    ) -> torch.Tensor:
        """Concatenate mean-pooled outputs from encoder layers 1, 3, 5."""
        # Build the pre-encoder timeline (same logic as encode_timeline up to
        # the temporal encoder call).
        step_embeddings: list[torch.Tensor] = []
        use_rope = self.temporal_block_type == "stage_d_rope" and self.use_rope_temporal_attention
        for timestep in range(dates.shape[1]):
            if hls is not None:
                spatial_tokens = self.spatial(hls=hls[:, timestep], return_tokens=True)
            else:
                if s2 is None or s1 is None:
                    raise ValueError("s2 and s1 are required when hls is not provided")
                spatial_tokens = self.spatial(s2=s2[:, timestep], s1=s1[:, timestep], return_tokens=True)
            if self.token_mode == "dense":
                step_tokens = spatial_tokens if use_rope else spatial_tokens + self.temporal_pos(dates[:, timestep:timestep + 1]).unsqueeze(1).squeeze(2)
                step_embeddings.append(step_tokens)
            else:
                step_embeddings.append(spatial_tokens.mean(dim=1))

        if self.token_mode == "dense":
            timeline = torch.stack(step_embeddings, dim=1)
            tokens_per_ts = int(timeline.shape[2])
            if mask is not None:
                timeline = timeline * mask.unsqueeze(-1).unsqueeze(-1).float()
            position_ids = self._timeline_position_ids(dates, tokens_per_timestep=tokens_per_ts)
            timeline = timeline.reshape(timeline.shape[0], timeline.shape[1] * timeline.shape[2], timeline.shape[3])
        else:
            timeline = torch.stack(step_embeddings, dim=1)
            if not use_rope:
                timeline = timeline + self.temporal_pos(dates)
            if mask is not None:
                timeline = timeline * mask.unsqueeze(-1).float()
            position_ids = self._timeline_position_ids(dates, tokens_per_timestep=1)

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
    ) -> torch.Tensor:
        """Multi-mask predictor ensemble matching JEPA training distribution.

        For T timesteps with 50% masking, generates C(T, T//2) mask patterns.
        Each pattern encodes visible timesteps, then runs predict_with_context.
        Averages the visible-position predictor outputs across all patterns.
        """
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
            visible_dates = dates[:, visible_idx_t]
            visible_mask = mask[:, visible_idx_t] if mask is not None else None
            masked_dates = dates[:, masked_idx_t]

            visible_emb = self.encode_timeline(
                s2=visible_s2, s1=visible_s1, dates=visible_dates,
                mask=visible_mask, hls=visible_hls,
            )

            # Run predictor with visible + mask tokens (training distribution)
            vis_pred, _masked_pred = self.predict_with_context(
                visible_emb, visible_dates, masked_dates,
            )

            # Use visible-position predictor output (context prediction)
            accumulated = accumulated + vis_pred.mean(dim=1)

        return accumulated / len(all_patterns)
