from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ewm.models.rope import RotaryEmbedding


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RopeSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = bool(use_rope)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = float(dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope else None

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence_length, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            if position_ids is None:
                position_ids = torch.arange(sequence_length, device=hidden.device, dtype=torch.long).unsqueeze(0)
                position_ids = position_ids.expand(batch, -1)
            q, k = self.rope.apply(q, k, position_ids=position_ids)

        attn_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, sequence_length):
                raise ValueError(
                    f"key_padding_mask must be [B, L], got {tuple(key_padding_mask.shape)} for hidden {tuple(hidden.shape)}"
                )
            valid = (~key_padding_mask).to(device=hidden.device, dtype=torch.bool)
            attn_mask = (valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(3))
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape(batch, sequence_length, self.embed_dim)
        return self.proj(attn)


class RopeTransformerBlock(nn.Module):
    """Pre-norm transformer block with optional RoPE attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RopeSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), position_ids=position_ids, key_padding_mask=key_padding_mask)
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class CrossModalFusionBlock(nn.Module):
    """Bidirectional cross-attention followed by gated fusion."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_s2 = nn.LayerNorm(embed_dim)
        self.norm_s1 = nn.LayerNorm(embed_dim)
        self.s2_to_s1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.s1_to_s2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, s2_tokens: torch.Tensor, s1_tokens: torch.Tensor) -> torch.Tensor:
        if s2_tokens.shape != s1_tokens.shape:
            raise ValueError(
                f"CrossModalFusionBlock expects aligned token shapes, got {tuple(s2_tokens.shape)} and {tuple(s1_tokens.shape)}"
            )
        s2_norm = self.norm_s2(s2_tokens)
        s1_norm = self.norm_s1(s1_tokens)
        s2_ctx = s2_tokens + self.s2_to_s1(s2_norm, s1_norm, s1_norm, need_weights=False)[0]
        s1_ctx = s1_tokens + self.s1_to_s2(s1_norm, s2_norm, s2_norm, need_weights=False)[0]
        gate = torch.sigmoid(self.gate(torch.cat([s2_ctx, s1_ctx], dim=-1)))
        fused = gate * s2_ctx + (1.0 - gate) * s1_ctx
        fused = fused + self.out_mlp(self.out_norm(fused))
        return fused


class FactorizedRopeBlock(nn.Module):
    """Alternating spatial and temporal attention over a dense spatiotemporal grid."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.spatial_block = RopeTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_rope=False,
            rope_base=rope_base,
        )
        self.temporal_block = RopeTransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_rope=use_rope,
            rope_base=rope_base,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        temporal_position_ids: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError(f"FactorizedRopeBlock expects [B, T, P, D], got {tuple(hidden.shape)}")
        batch, timesteps, patch_tokens, embed_dim = hidden.shape
        if visibility_mask is not None and visibility_mask.shape != (batch, timesteps, patch_tokens):
            raise ValueError(
                f"visibility_mask must be [B, T, P], got {tuple(visibility_mask.shape)} for hidden {tuple(hidden.shape)}"
            )

        spatial_hidden = hidden.reshape(batch * timesteps, patch_tokens, embed_dim)
        spatial_mask = None
        if visibility_mask is not None:
            spatial_mask = ~visibility_mask.reshape(batch * timesteps, patch_tokens).to(torch.bool)
        spatial_hidden = self.spatial_block(spatial_hidden, key_padding_mask=spatial_mask)
        hidden = spatial_hidden.reshape(batch, timesteps, patch_tokens, embed_dim)

        temporal_hidden = hidden.permute(0, 2, 1, 3).reshape(batch * patch_tokens, timesteps, embed_dim)
        temporal_mask = None
        if visibility_mask is not None:
            temporal_mask = ~visibility_mask.permute(0, 2, 1).reshape(batch * patch_tokens, timesteps).to(torch.bool)
        temporal_ids = None
        if temporal_position_ids is not None:
            if temporal_position_ids.shape != (batch, timesteps):
                raise ValueError(
                    f"temporal_position_ids must be [B, T], got {tuple(temporal_position_ids.shape)} for hidden {tuple(hidden.shape)}"
                )
            temporal_ids = temporal_position_ids.unsqueeze(1).expand(-1, patch_tokens, -1).reshape(batch * patch_tokens, timesteps)
        temporal_hidden = self.temporal_block(
            temporal_hidden,
            position_ids=temporal_ids,
            key_padding_mask=temporal_mask,
        )
        hidden = temporal_hidden.reshape(batch, patch_tokens, timesteps, embed_dim).permute(0, 2, 1, 3).contiguous()
        return hidden
