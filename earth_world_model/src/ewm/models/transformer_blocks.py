from __future__ import annotations

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

    def forward(self, hidden: torch.Tensor, *, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, sequence_length, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, sequence_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            if position_ids is None:
                position_ids = torch.arange(sequence_length, device=hidden.device, dtype=torch.long).unsqueeze(0)
                position_ids = position_ids.expand(batch, -1)
            q, k = self.rope.apply(q, k, position_ids=position_ids)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
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

    def forward(self, hidden: torch.Tensor, *, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), position_ids=position_ids)
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden
