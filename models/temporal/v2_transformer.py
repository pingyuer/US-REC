"""V2 Temporal Transformer — per-branch PE and attention strategies.

*Short* branch:  RoPE (relative PE) + sliding-window causal attention.
*Long*  branch:  Real-index sinusoidal PE + global (optionally dilated) attention.

The transformer stack is shared in structure — only the *PE injection*
and *attention mask construction* differ between branches.

Design
------
- ``V2TransformerLayer``: a single transformer encoder layer that
  receives pre-built Q, K (possibly rotated by RoPE) and an attention
  mask.
- ``V2TemporalPoseTransformer``: the full stack, parameterised by
  ``branch_mode ∈ {"short", "long"}`` which selects:
    * PE: ``RotaryPositionEncoding`` / ``RealIndexSinusoidalPosEmb``
    * Mask: ``sliding_window_causal`` / ``global_causal`` / ``dilated``
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal.position_encoding import (
    RotaryPositionEncoding,
    RealIndexSinusoidalPosEmb,
)
from models.temporal.temporal_transformer import (
    SinusoidalPosEmb,
    _build_sliding_window_causal_mask,
)


# ─── Attention masks for long branch ────────────────────────────────────────

def _build_global_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Full causal mask: position *i* attends to all positions ≤ *i*.

    Returns (T, T) bool where True = **blocked**.
    """
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def _build_dilated_causal_mask(
    T: int, window_size: int, dilation: int, device: torch.device,
) -> torch.Tensor:
    """Dilated sliding-window causal mask (Longformer-style).

    Position *i* attends to {j : j ≤ i  AND  (i - j) mod dilation == 0
    AND (i - j) / dilation < window_size}.

    Combined with full global tokens at anchors (the caller can override
    specific rows to all-attend).

    Returns (T, T) bool with True = **blocked**.
    """
    arange = torch.arange(T, device=device)
    j = arange.unsqueeze(0)  # (1, T)
    i = arange.unsqueeze(1)  # (T, 1)
    diff = i - j  # (T, T) — positive when i > j
    causal = j <= i
    dilated = (diff % dilation) == 0
    in_window = diff < (window_size * dilation)
    allow = causal & dilated & in_window
    return ~allow  # True = blocked


# ─── V2 Transformer Layer ───────────────────────────────────────────────────

class V2TransformerLayer(nn.Module):
    """Transformer layer that accepts *external* Q/K (for RoPE) and mask.

    For the short branch, Q/K are rotated by RoPE externally and passed
    in; the layer's self-attention uses them directly.  For the long
    branch, Q/K come from the standard linear projections (no external
    rotation needed — PE was already added to tokens).

    Architecture is identical to ``SlidingWindowTransformerLayer`` but
    *decouples* the mask and Q/K computation from the layer itself so
    the caller controls these.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        # Q/K/V projections (needed for RoPE path which applies rotation
        # *between* projection and dot-product)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

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

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last_linear = self.ffn[-2]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope: Optional[RotaryPositionEncoding] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D)
        attn_mask : (T, T) bool — True = blocked
        rope : optional RoPE to apply to Q/K
        position_ids : (B, T) for RoPE position resolution

        Returns
        -------
        (B, T, D)
        """
        B, T, D = x.shape

        residual = x
        x_n = self.norm1(x)

        # Q/K/V projections → (B, n_heads, T, d_head)
        q = self.q_proj(x_n).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_n).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_n).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K if provided
        if rope is not None:
            q, k = rope(q, k, position_ids=position_ids)

        # Scaled dot-product attention with mask
        # attn_mask: (T, T) bool True=blocked → convert to float -inf mask
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply mask: True positions → -inf
        attn_weights = attn_weights.masked_fill(
            attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)          # (B, H, T, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        x = residual + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ─── Full V2 Temporal Transformer ───────────────────────────────────────────

class V2TemporalPoseTransformer(nn.Module):
    """Branch-aware temporal transformer for the V2 kroot dual system.

    Parameters
    ----------
    d_model : int
        Token dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Stacked transformer layers.
    dim_feedforward : int
        FFN hidden dim.
    dropout : float
        Dropout rate.
    branch_mode : str
        ``"short"`` → RoPE + sliding-window causal attention.
        ``"long"``  → Real-index sinusoidal PE + global/dilated attention.
    window_size : int
        Sliding window size (short branch) or dilated window size (long).
    dilation : int
        Dilation factor for long-branch dilated attention. 1 = no dilation
        (equivalent to global causal with window_size).  Only used when
        ``attention_mode="dilated"``.
    attention_mode : str
        ``"sliding_window"`` (default for short),
        ``"global"`` (full causal for long),
        ``"dilated"`` (dilated sliding-window for long).
    max_len : int
        Maximum position for PE tables.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        branch_mode: str = "short",
        window_size: int = 64,
        dilation: int = 1,
        attention_mode: Optional[str] = None,
        max_len: int = 8192,
    ):
        super().__init__()
        self.branch_mode = branch_mode
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ── Select PE strategy ─────────────────────────────────────
        if branch_mode == "short":
            # RoPE — applied per-head to Q/K; no additive PE needed
            self.rope = RotaryPositionEncoding(
                d_head=self.d_head, max_len=max_len,
            )
            self.additive_pe = None
        else:
            # Real-index sinusoidal — additive PE
            self.rope = None
            self.additive_pe = RealIndexSinusoidalPosEmb(d_model, max_len=max_len)

        # ── Select attention mode ─────────────────────────────────
        if attention_mode is None:
            attention_mode = "sliding_window" if branch_mode == "short" else "global"
        self.attention_mode = attention_mode
        self.window_size = window_size
        self.dilation = max(1, int(dilation))

        # ── Transformer layers ─────────────────────────────────────
        self.layers = nn.ModuleList([
            V2TransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Mask cache ─────────────────────────────────────────────
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_cache_key: tuple = ()

    def _build_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build and cache the attention mask for current sequence length."""
        key = (T, self.attention_mode, self.window_size, self.dilation, device)
        if key == self._mask_cache_key and self._mask_cache is not None:
            return self._mask_cache

        if self.attention_mode == "sliding_window":
            mask = _build_sliding_window_causal_mask(T, self.window_size, device)
        elif self.attention_mode == "global":
            mask = _build_global_causal_mask(T, device)
        elif self.attention_mode == "dilated":
            mask = _build_dilated_causal_mask(T, self.window_size, self.dilation, device)
        else:
            raise ValueError(f"Unknown attention_mode: {self.attention_mode!r}")

        self._mask_cache = mask
        self._mask_cache_key = key
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, T, D) — per-frame feature tokens.
        position_ids : (B, T) LongTensor — real frame indices.
            * Short branch: optional (defaults to 0..T-1 for RoPE).
            * Long branch: **required** — real frame indices for correct PE.

        Returns
        -------
        ctx : (B, T, D) — context-enriched tokens.
        """
        B, T, D = tokens.shape
        device = tokens.device

        # ── Apply additive PE (long branch) ──────────────────────
        if self.additive_pe is not None:
            x = self.additive_pe(tokens, position_ids=position_ids)
        else:
            x = tokens  # RoPE is applied inside each layer

        # ── Build attention mask ─────────────────────────────────
        mask = self._build_mask(T, device)

        # ── Transformer layers ───────────────────────────────────
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=mask,
                rope=self.rope,
                position_ids=position_ids,
            )

        return self.final_norm(x)
