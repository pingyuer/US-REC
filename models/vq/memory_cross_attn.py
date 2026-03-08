"""VQ Memory Cross-Attention — each local token queries the scan VQ memory.

Gives each local-transformer token global context c_t by soft-attending
over the VQ memory (anchor z_q vectors).  c_t tells the model which
scan-level prototypes are relevant for the current spatial context.

This is distinct from the summary g:
  * g = low-frequency global overview (one vector for the whole scan)
  * c_t = per-token global context (what anchors relate to *this* position)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQMemoryCrossAttn(nn.Module):
    """Single-layer cross-attention from local tokens to VQ memory.

    Parameters
    ----------
    d_model : int
        Dimension of local transformer tokens (query).
    d_memory : int
        Dimension of VQ memory entries (key / value).
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_memory: int = 256,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=d_memory,
            vdim=d_memory,
            batch_first=True,
            dropout=dropout,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_memory)

        # Relative anchor position embedding (scalar position -> memory dim).
        # Zero-init on the last layer keeps behavior unchanged at step 0.
        self.pos_proj = nn.Sequential(
            nn.Linear(1, d_memory),
            nn.GELU(),
            nn.Linear(d_memory, d_memory),
        )

        # Post cross-attention feed-forward + gate
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        # Learnable gate: sigmoid(-3)≈0.05 → starts as near-no-op
        self.gate = nn.Parameter(torch.tensor(-3.0))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.pos_proj[-1], nn.Linear):
            nn.init.zeros_(self.pos_proj[-1].weight)
            nn.init.zeros_(self.pos_proj[-1].bias)

    def forward(
        self,
        local_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        local_tokens : (B, T, D_model) — output from local transformer
        memory       : (B, M, D_memory) — VQ memory (anchor z_q codes)
        memory_mask  : (B, M) bool — True = valid memory entry
        memory_pos   : (B, M) float in [0, 1] — normalized anchor positions

        Returns
        -------
        c_t       : (B, T, D_model) — per-token global context
        attn_weights : (B, T, M) — attention weights (for logging)
        """
        q = self.norm_q(local_tokens)      # (B, T, D)
        kv = memory
        if memory_pos is not None:
            kv = kv + self.pos_proj(memory_pos.unsqueeze(-1).to(kv.dtype))
        kv = self.norm_kv(kv)               # (B, M, D_mem)

        # key_padding_mask: True = ignore
        kpm = ~memory_mask if memory_mask is not None else None

        attn_out, attn_weights = self.cross_attn(
            q, kv, kv,
            key_padding_mask=kpm,
            need_weights=True,
            average_attn_weights=True,
        )  # attn_out: (B, T, D), attn_weights: (B, T, M)

        # Gated residual: starts near 0, gradually learns to inject memory info
        gate = torch.sigmoid(self.gate)
        c_t = gate * attn_out  # (B, T, D)

        # FFN refinement
        c_t = c_t + self.ffn(self.norm_ffn(c_t))

        return c_t, attn_weights
