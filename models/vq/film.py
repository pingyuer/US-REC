"""FiLM Conditioner — Feature-wise Linear Modulation from scan summary g.

Uses the scan-level summary g to modulate local transformer features via
    y = γ(g) * x + β(g)

This injects low-frequency global prior into every local token without
architectural changes to the transformer itself.

Optionally also supports a simpler "global_token" mode where g is
prepended/added to the local sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation conditioned on scan summary g.

    Parameters
    ----------
    d_model : int
        Dimension of local tokens to modulate.
    d_cond : int
        Dimension of conditioning vector g.
    use_film : bool
        Apply affine FiLM transform (γ * x + β). Default True.
    use_global_token : bool
        Additively inject g as a bias. Default False.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_cond: int = 256,
        use_film: bool = True,
        use_global_token: bool = False,
    ):
        super().__init__()
        self.use_film = use_film
        self.use_global_token = use_global_token

        if use_film:
            # Project g → (γ, β) pair
            self.film_proj = nn.Sequential(
                nn.LayerNorm(d_cond),
                nn.Linear(d_cond, d_model * 2),
            )
            # Initialise γ≈1, β≈0 so FiLM starts as identity
            nn.init.zeros_(self.film_proj[1].weight)
            with torch.no_grad():
                bias = torch.zeros(d_model * 2)
                bias[:d_model] = 1.0  # γ init = 1
                self.film_proj[1].bias = nn.Parameter(bias)

        if use_global_token:
            self.global_proj = nn.Sequential(
                nn.LayerNorm(d_cond),
                nn.Linear(d_cond, d_model),
            )
            nn.init.xavier_uniform_(self.global_proj[1].weight)
            nn.init.zeros_(self.global_proj[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D) local features
        g : (B, D_cond) scan summary

        Returns
        -------
        x_out : (B, T, D) modulated features
        """
        out = x

        if self.use_film:
            film_params = self.film_proj(g)          # (B, 2*D)
            gamma, beta = film_params.chunk(2, dim=-1)  # each (B, D)
            gamma = gamma.unsqueeze(1)               # (B, 1, D)
            beta = beta.unsqueeze(1)                 # (B, 1, D)
            out = gamma * out + beta

        if self.use_global_token:
            g_tok = self.global_proj(g).unsqueeze(1)  # (B, 1, D)
            out = out + g_tok

        return out
