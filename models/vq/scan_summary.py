"""Scan Summary Pool — lightweight aggregation of anchor VQ codes → g.

g is a scan-level global summary vector derived entirely from the
quantised anchor codes (z_q), NOT raw frame features.  This ensures
g represents a dictionary-compressed scan-level abstract.

Two pool types are implemented:
  * ``attention`` — weighted sum with learned MLP + softmax scores.
  * ``latent``    — a small set of learnable latent tokens cross-attend
                    to the anchor codes, then are averaged to produce g.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionPool(nn.Module):
    """Attention-weighted pooling: g = Σ α_j · MLP(z_q_j).

    Produces a single global summary vector from a variable-length
    sequence of anchor VQ codes.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )
        self.score = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        z_q  : (B, M, D) anchor VQ codes (should be detached)
        mask : (B, M) bool — True = valid anchor, False = padding

        Returns
        -------
        g : (B, D_out) scan-level summary
        """
        values = self.mlp(z_q)        # (B, M, D_out)
        scores = self.score(z_q)       # (B, M, 1)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            # Guard against all-masked rows (all -inf → NaN after softmax)
            all_masked = ~mask.any(dim=1, keepdim=True).unsqueeze(-1)  # (B,1,1)
            scores = scores.masked_fill(all_masked, 0.0)

        alpha = F.softmax(scores, dim=1)  # (B, M, 1)
        g = (alpha * values).sum(dim=1)   # (B, D_out)
        return g


class _LatentPool(nn.Module):
    """Latent-bottleneck pooling: learnable latent tokens cross-attend
    to anchor VQ codes, then are averaged.

    This is a lightweight single-layer cross-attention, NOT a full
    transformer. Suitable for a future V2 extension.
    """

    def __init__(self, d_in: int, d_out: int, n_latents: int = 8, n_heads: int = 4):
        super().__init__()
        self.n_latents = n_latents
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_out) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_out,
            num_heads=n_heads,
            kdim=d_in,
            vdim=d_in,
            batch_first=True,
            dropout=0.0,
        )
        self.norm_q = nn.LayerNorm(d_out)
        self.norm_kv = nn.LayerNorm(d_in)

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        z_q  : (B, M, D_in) anchor VQ codes
        mask : (B, M) bool — True = valid

        Returns
        -------
        g : (B, D_out)
        """
        B = z_q.size(0)
        q = self.norm_q(self.latents.expand(B, -1, -1))  # (B, L, D_out)
        kv = self.norm_kv(z_q)                            # (B, M, D_in)

        # key_padding_mask: True = ignore
        kpm = ~mask if mask is not None else None

        out, _ = self.cross_attn(q, kv, kv, key_padding_mask=kpm)  # (B, L, D_out)
        g = out.mean(dim=1)  # (B, D_out)
        return g


class ScanSummaryPool(nn.Module):
    """Aggregates anchor VQ codes into a scan-level summary vector g.

    Parameters
    ----------
    d_in : int
        Dimension of quantised anchor codes (code_dim).
    d_out : int
        Dimension of the summary vector g.
    pool_type : str
        ``"attention"`` (default, V1) or ``"latent"`` (V2).
    n_latents : int
        Number of latent tokens (only used when pool_type="latent").
    n_heads : int
        Number of attention heads for latent pooling.
    """

    def __init__(
        self,
        d_in: int = 256,
        d_out: int = 256,
        pool_type: str = "attention",
        n_latents: int = 8,
        n_heads: int = 4,
    ):
        super().__init__()
        self.pool_type = pool_type

        if pool_type == "attention":
            self.pool = _AttentionPool(d_in, d_out)
        elif pool_type == "latent":
            self.pool = _LatentPool(d_in, d_out, n_latents=n_latents, n_heads=n_heads)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type!r}")

        self.out_norm = nn.LayerNorm(d_out)

    def forward(
        self,
        z_q: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_q  : (B, M, D_in) quantised anchor VQ codes (should be detached!)
        mask : (B, M) bool — True = valid anchor

        Returns
        -------
        g : (B, D_out) scan-level summary vector
        """
        g = self.pool(z_q, mask)
        g = self.out_norm(g)
        return g
