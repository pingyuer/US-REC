"""Scan-level contextualisation for VQ anchor memories.

This module lets anchor-level VQ codes exchange information across the whole
scan before they are pooled into ``g`` or queried by local pose tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.temporal.temporal_transformer import SinusoidalPosEmb


class ScanContextEncoder(nn.Module):
    """Contextualise VQ anchor codes with bidirectional self-attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.pos_emb = SinusoidalPosEmb(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return contextualised anchor memory.

        Parameters
        ----------
        z_q : (B, M, D)
            Quantised anchor codes.
        mask : (B, M) bool or None
            True marks valid anchors.
        """
        x = self.pos_emb(z_q)
        key_padding_mask = ~mask if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out_norm(x + z_q)
