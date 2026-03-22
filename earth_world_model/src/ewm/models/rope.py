from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """Simple RoPE helper for self-attention heads."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding requires even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        position_ids: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids.to(device=device, dtype=torch.float32)
        freqs = positions.unsqueeze(-1) * self.inv_freq.to(device=device)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        cos = torch.cos(emb).to(dtype=dtype)
        sin = torch.sin(emb).to(dtype=dtype)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(position_ids, device=q.device, dtype=q.dtype)
        return ((q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin))
