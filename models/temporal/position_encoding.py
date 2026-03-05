"""Position encodings for the V2 dual-stream transformer.

Implements:
- **RotaryPositionEncoding (RoPE)**: rotation-based relative PE for the
  *short* stream.  Applied to Q/K before attention — naturally captures
  relative distance within the sliding-window causal mask.
  Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
  Position Embedding", 2021.

- **RealIndexSinusoidalPosEmb**: absolute sinusoidal PE keyed to *real*
  frame indices (e.g. ``t + m*s``) for the *long* stream.  Tells the
  model the physical temporal gap between sparse anchor tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ─── Rotary Position Encoding (RoPE) ────────────────────────────────────────

def _build_rope_freqs(d_head: int, max_len: int = 8192, base: float = 10000.0) -> torch.Tensor:
    """Pre-compute RoPE frequency pairs for half-dimension rotation.

    Returns (max_len, d_head) of cosine/sine interleaved per pair.
    """
    assert d_head % 2 == 0, "d_head must be even for RoPE"
    half = d_head // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(0, max_len, dtype=torch.float32)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_len, half)
    return angles  # (max_len, half) — pass to cos/sin at runtime


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE).

    Pre-computes sin/cos tables and applies the rotation to Q/K tensors.
    Supports *position_ids* for non-contiguous indices.

    Parameters
    ----------
    d_head : int
        Per-head dimension (must be even).
    max_len : int
        Maximum cached sequence length.
    base : float
        Base frequency (default 10000).
    """

    def __init__(self, d_head: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_len = max_len
        angles = _build_rope_freqs(d_head, max_len, base)  # (max_len, half)
        cos_cache = torch.cos(angles)  # (max_len, half)
        sin_cache = torch.sin(angles)  # (max_len, half)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """Return (cos, sin) tables shaped (T, half) or (B, T, half)."""
        if position_ids is not None:
            # position_ids: (B, T) or (T,)
            cos = self.cos_cache.to(device)[position_ids]  # (..., half)
            sin = self.sin_cache.to(device)[position_ids]
            return cos, sin
        cos = self.cos_cache[:seq_len].to(device)
        sin = self.sin_cache[:seq_len].to(device)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x0, x1, x2, x3, …] → [-x1, x0, -x3, x2, …]."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Parameters
        ----------
        q : (B, n_heads, T, d_head)
        k : (B, n_heads, T, d_head)
        position_ids : (B, T) or None
            When ``None`` uses contiguous 0..T-1.

        Returns
        -------
        q_rot, k_rot : same shapes as input.
        """
        T = q.size(2)
        cos, sin = self._get_cos_sin(T, q.device, position_ids)

        # Expand cos/sin to match (B, 1, T, half) for broadcasting
        if cos.dim() == 2:
            # (T, half) → (1, 1, T, half)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            # (B, T, half) → (B, 1, T, half)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Duplicate cos/sin to full d_head: [cos, cos] for the paired rotation
        cos = torch.cat([cos, cos], dim=-1)  # (…, d_head)
        sin = torch.cat([sin, sin], dim=-1)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─── Real-Index Sinusoidal PE ────────────────────────────────────────────────

class RealIndexSinusoidalPosEmb(nn.Module):
    """Sinusoidal PE using *actual* frame indices instead of 0..T-1.

    For the long branch with stride ``s``, feeding position_ids =
    ``[t, t+s, t+2s, …, t+(k-1)*s]`` encodes the real temporal gap
    between sparse anchor frames.

    Functionally equivalent to the standard sinusoidal PE but dynamically
    computed from ``position_ids`` rather than a lookup table, so it
    supports arbitrary (non-contiguous) integer positions.

    Parameters
    ----------
    d_model : int
        Token / embedding dimension.
    max_len : int
        Maximum position index (for sanity).
    """

    def __init__(self, d_model: int, max_len: int = 65536):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Pre-compute div_term once
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)  # (D/2,)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add sinusoidal PE computed from ``position_ids`` to ``x``.

        Parameters
        ----------
        x : (B, T, D) — token embeddings
        position_ids : (B, T) or (T,) LongTensor of real frame indices.
            When ``None`` falls back to 0..T-1.

        Returns
        -------
        x + pe : (B, T, D)
        """
        B, T, D = x.shape
        device = x.device

        if position_ids is None:
            position_ids = torch.arange(T, device=device, dtype=torch.float32)
        else:
            position_ids = position_ids.float()

        # Ensure shape is (B, T)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)  # (B, T)

        # Build PE: (B, T, D/2) via outer product with div_term
        div_term = self.div_term.to(device)  # (D/2,)
        angles = position_ids.unsqueeze(-1) * div_term.unsqueeze(0).unsqueeze(0)  # (B, T, D/2)

        pe = torch.zeros(B, T, D, device=device, dtype=x.dtype)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)

        return x + pe
