"""Temporal Transformer for long-sequence pose refinement.

Implements a causal Transformer encoder with **sliding-window attention**
to support US scan sequences of 400–600+ frames without O(T²) memory.

Design choices:
- Causal masking ensures frame *i* only attends to frames ≤ *i*.
- Window attention (default w=64) keeps memory linear in T.
- Sinusoidal positional encoding (easy to extend to rotary later).
- Optional **Transformer-XL style memory tokens**: cached hidden states
  from the previous segment are prepended to the current segment as
  read-only context, enabling cross-segment information flow without
  re-computing the full scan.  Memory tokens are always detached from
  the computation graph (recurrence without BPTT).
"""

from __future__ import annotations

import math
from typing import Optional

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

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x : (B, T, D),  offset : global sequence start position."""
        return x + self.pe[:, offset : offset + x.size(1)]


# ─── Attention masks ─────────────────────────────────────────────────────────

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
    mask = (j > i) | (j < i - window_size + 1)
    return mask


def _build_memory_causal_mask(
    T: int, M: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Attention mask for M memory tokens followed by T frame tokens.

    Memory tokens can be used as context by any frame but are not
    themselves updated (they are detached read-only inputs).

    Layout of the (M+T, M+T) mask (True = blocked):
    - Memory queries (rows 0..M-1): attend to all earlier/same memory tokens
      (standard causal attention among memories).
    - Frame  queries (rows M..M+T-1): can ALWAYS attend to all memory tokens
      (no window restriction on memory); standard causal sliding window
      within the T frame tokens.
    """
    L = M + T
    arange = torch.arange(L, device=device)
    col_j = arange.unsqueeze(0)   # (1, L) — key
    row_i = arange.unsqueeze(1)   # (L, 1) — query

    # Start with fully blocked
    mask = torch.ones(L, L, dtype=torch.bool, device=device)

    # Memory-on-memory: causal (j <= i, both < M)
    mem_q = row_i < M
    mem_k = col_j < M
    mask = torch.where(mem_q & mem_k & (col_j <= row_i), torch.zeros_like(mask), mask)

    # Frame-on-memory: always allow (memory is always "past")
    frame_q = row_i >= M
    mask = torch.where(frame_q & mem_k, torch.zeros_like(mask), mask)

    # Frame-on-frame: causal sliding window (local coords within T)
    frame_k = col_j >= M
    local_i = row_i - M   # offset into frame tokens
    local_j = col_j - M
    frame_causal_ok = (local_j <= local_i) & (local_j >= local_i - window_size + 1)
    mask = torch.where(frame_q & frame_k & frame_causal_ok, torch.zeros_like(mask), mask)

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

        # Cache masks per sequence length
        self._cached_mask: torch.Tensor | None = None
        self._cached_key: tuple[int, int] = (-1, -1)  # (T, M)

    def _get_mask(self, T: int, device: torch.device, M: int = 0) -> torch.Tensor:
        key = (T + M, M)
        if key != self._cached_key or self._cached_mask is None or self._cached_mask.device != device:
            if M > 0:
                self._cached_mask = _build_memory_causal_mask(T, M, self.window_size, device)
            else:
                self._cached_mask = _build_sliding_window_causal_mask(T, self.window_size, device)
            self._cached_key = key
        return self._cached_mask

    def forward(self, x: torch.Tensor, M: int = 0) -> torch.Tensor:
        """x : (B, M+T, D) → (B, M+T, D).  M = number of memory tokens."""
        L = x.size(1)
        T = L - M
        mask = self._get_mask(T, x.device, M=M)
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
    memory_size : int
        Number of memory tokens to carry between segments (Transformer-XL
        style).  0 disables the memory mechanism (default, trains identically
        to the original design).  When > 0 the transformer accepts an optional
        ``memory`` argument and returns ``(ctx, new_memory)`` from ``forward()``.
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
        memory_size: int = 0,
    ):
        super().__init__()
        self.memory_size = max(0, int(memory_size))
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

    def forward(
        self,
        tokens: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        tokens : (B, T, D) — per-frame feature tokens (current segment)
        memory : (B, M, D) or None — cached context from previous segment
            (detached; only used as keys/values via the memory attention mask)
        pos_offset : int — global position of the first frame token (for
            sinusoidal PE when processing the sequence in chunks)

        Returns
        -------
        ctx : (B, T, D) — context-enriched frame tokens
        new_memory : (B, M, D) or None — updated memory to pass to the next
            segment (= last ``memory_size`` ctx tokens, detached).
            None when ``memory_size == 0``.
        """
        B, T, D = tokens.shape
        device = tokens.device

        if memory is not None and self.memory_size > 0:
            # Prepend memory (detached) to frame tokens.
            # Memory tokens get positional encodings 0..M-1;
            # frame tokens get encodings pos_offset..pos_offset+T-1.
            M = memory.size(1)
            mem_pe = self.pos_emb(memory.detach(), offset=0)          # (B, M, D)
            tok_pe = self.pos_emb(tokens, offset=pos_offset)          # (B, T, D)
            x = torch.cat([mem_pe, tok_pe], dim=1)                    # (B, M+T, D)
        else:
            M = 0
            x = self.pos_emb(tokens, offset=pos_offset)               # (B, T, D)

        for layer in self.layers:
            x = layer(x, M=M)

        x = self.final_norm(x)
        ctx = x[:, M:]  # (B, T, D) — only frame positions

        # Build new memory from the last memory_size frame tokens
        if self.memory_size > 0:
            take = min(self.memory_size, T)
            new_memory = ctx[:, -take:].detach()                      # (B, take, D)
        else:
            new_memory = None

        return ctx, new_memory
