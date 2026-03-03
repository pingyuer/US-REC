"""Temporal Transformer for long-sequence pose refinement.

Implements a causal Transformer encoder with **sliding-window attention**
to support US scan sequences of 400–600+ frames without O(T²) memory.

Design choices:
- Causal masking ensures frame *i* only attends to frames ≤ *i*.
- Window attention (default w=64) keeps memory linear in T.
- Sinusoidal positional encoding (easy to extend to rotary later).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Positional encoding ────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


# ─── Sliding-window causal attention ─────────────────────────────────────────

def _build_sliding_window_causal_mask(
    T: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Build a boolean attention mask for sliding-window causal attention.

    Returns (T, T) bool mask where ``True`` means **blocked** (following the
    convention of ``torch.nn.MultiheadAttention(attn_mask)`` where True=ignore).

    Position *i* can attend to positions ``max(0, i - window_size + 1) .. i``.
    """
    arange = torch.arange(T, device=device)
    j = arange.unsqueeze(0)   # (1, T) — column / key position
    i = arange.unsqueeze(1)   # (T, 1) — row   / query position
    # mask[i,j] = True means BLOCK attention from query i to key j.
    # Causal: block j > i (no attending to future).
    # Window: block j < i - window_size + 1 (no attending beyond window).
    mask = (j > i) | (j < i - window_size + 1)
    return mask


class SlidingWindowTransformerLayer(nn.Module):
    """Single transformer encoder layer with sliding-window causal attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        window_size: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Cache mask per sequence length
        self._cached_mask: torch.Tensor | None = None
        self._cached_T: int = -1

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if T != self._cached_T or self._cached_mask is None or self._cached_mask.device != device:
            self._cached_mask = _build_sliding_window_causal_mask(
                T, self.window_size, device
            )
            self._cached_T = T
        return self._cached_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, D) → (B, T, D)"""
        T = x.size(1)
        mask = self._get_mask(T, x.device)
        # Pre-norm style
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ─── Full temporal transformer ──────────────────────────────────────────────

class TemporalPoseTransformer(nn.Module):
    """Stack of sliding-window causal transformer layers.

    Parameters
    ----------
    d_model : int
        Token dimension (must match FrameEncoder output).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    dim_feedforward : int
        FFN hidden dimension.
    dropout : float
        Dropout rate.
    window_size : int
        Sliding-window size for causal attention.
    max_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        window_size: int = 64,
        max_len: int = 4096,
    ):
        super().__init__()
        self.pos_emb = SinusoidalPosEmb(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                SlidingWindowTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    window_size=window_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, T, D) — per-frame feature tokens

        Returns
        -------
        ctx : (B, T, D) — context-enriched tokens
        """
        x = self.pos_emb(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
