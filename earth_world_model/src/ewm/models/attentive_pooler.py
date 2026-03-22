"""V-JEPA 2.1-style attentive pooler for frozen encoder features.

Adapted from ``facebookresearch/vjepa2/src/models/attentive_pooler.py``.
The core idea: instead of mean-pooling encoder outputs, use learnable query
tokens that cross-attend to the full spatial-temporal feature map, producing
a task-aware summary embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentivePooler(nn.Module):
    """Aggregate a variable-length token sequence into a fixed-size embedding.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input tokens and output embedding.
    num_queries : int
        Number of learnable query tokens.  Use 1 for a single summary vector.
    num_heads : int
        Number of attention heads in cross-attention and self-attention blocks.
    mlp_ratio : float
        MLP expansion ratio (hidden_dim = embed_dim * mlp_ratio).
    depth : int
        Total number of attention layers.  The first ``depth - 1`` layers are
        self-attention on the queries; the last layer is cross-attention from
        queries to the encoder features.
    dropout : float
        Dropout probability in attention and MLP layers.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_queries: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)

        # Learnable query tokens (V-JEPA 2.1: initialized to zeros + trunc_normal)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Self-attention blocks to refine queries before cross-attending
        self.self_attn_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(max(0, depth - 1))
            ]
        )

        # Cross-attention: queries attend to encoder features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_norm_q = nn.LayerNorm(embed_dim)
        self.cross_norm_kv = nn.LayerNorm(embed_dim)

        # MLP after cross-attention (post-norm residual)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.cross_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.cross_mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Aggregate encoder features into query embeddings.

        Parameters
        ----------
        encoder_features : Tensor
            Shape ``[B, seq_len, embed_dim]`` — frozen encoder output tokens.

        Returns
        -------
        Tensor
            Shape ``[B, embed_dim]`` when ``num_queries == 1``, otherwise
            ``[B, num_queries, embed_dim]``.
        """
        batch = encoder_features.shape[0]
        queries = self.query_tokens.expand(batch, -1, -1)  # [B, Q, D]

        # Self-attention on queries (refine before cross-attending)
        for block in self.self_attn_blocks:
            queries = block(queries)

        # Cross-attention: queries attend to ALL encoder features
        q = self.cross_norm_q(queries)
        kv = self.cross_norm_kv(encoder_features)
        attended, _ = self.cross_attn(q, kv, kv)  # [B, Q, D]
        queries = queries + attended  # residual

        # MLP with residual
        queries = queries + self.cross_mlp(self.cross_mlp_norm(queries))

        if self.num_queries == 1:
            return queries.squeeze(1)  # [B, D]
        return queries


class AttentiveRegressor(nn.Module):
    """AttentivePooler + linear regression head for probe training.

    This wraps the pooler with a final linear layer for training on a
    regression target (e.g. log1p(f12_gas)).  After training, discard the
    linear head and use the pooler output as the embedding.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_queries: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Return scalar prediction [B, 1]."""
        pooled = self.pooler(encoder_features)  # [B, D]
        return self.head(pooled)  # [B, 1]

    def extract_embedding(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Return the pooler output [B, D] without the regression head."""
        return self.pooler(encoder_features)
