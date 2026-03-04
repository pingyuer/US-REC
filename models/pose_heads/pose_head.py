"""Pose prediction heads for long-sequence models.

- LocalPoseHead: predict T_{i-1 <- i} from context tokens [h_{i-1}, h_i]
- MultiIntervalHead: predict T_{i-Δ <- i} for multiple intervals Δ

Both heads produce (B, T, 4, 4) rigid transforms using a configurable
rotation parameterization (rot6d / quat / se3_expmap).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation import (
    rotation_rep_to_rotmat,
    make_se3,
    ROTATION_REP_DIM,
)


class _PoseDecoder(nn.Module):
    """MLP: concatenated context → pose parameters → SE(3) matrix.

    Three-layer MLP with LayerNorm and residual connection for stable
    gradient flow.  Output bias is set to identity so the initial
    prediction is the identity transform (near-zero loss at init).
    """

    def __init__(self, d_input: int, d_hidden: int, rotation_rep: str = "rot6d"):
        super().__init__()
        self.rotation_rep = rotation_rep
        rot_dim = ROTATION_REP_DIM[rotation_rep] - 3  # rotation-only dim
        self.rot_dim = rot_dim
        out_dim = rot_dim + 3  # rotation params + translation

        # Three-layer MLP with LayerNorm + GELU
        self.norm_in = nn.LayerNorm(d_input)
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.fc_out = nn.Linear(d_hidden, out_dim)

        self._init_weights()
        self._init_identity_bias()

    def _init_weights(self) -> None:
        """Xavier init for hidden layers — prevents vanishing / exploding
        activations at initialisation."""
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _init_identity_bias(self) -> None:
        """Set the output layer bias so the initial prediction is near-identity.

        This avoids random rotations at initialisation which create large
        initial geodesic loss and slow convergence.
        """
        # Very small weights → output dominated by bias (near-identity)
        nn.init.normal_(self.fc_out.weight, std=1e-4)
        bias = torch.zeros(self.rot_dim + 3)
        if self.rotation_rep == "rot6d":
            # rot6d identity: first column [1,0,0], second column [0,1,0]
            bias[:6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        elif self.rotation_rep == "quat":
            # quaternion identity: [1, 0, 0, 0]
            bias[:4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # translation stays at zero (identity)
        self.fc_out.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (..., d_input)

        Returns
        -------
        T : (..., 4, 4) rigid transformation matrix
        """
        h = self.norm_in(x)
        h = F.gelu(self.fc1(h))
        # Residual through hidden layers (if dimensions match)
        h2 = self.norm1(h)
        h2 = F.gelu(self.fc2(h2))
        h = h + self.norm2(h2)              # residual connection
        raw = self.fc_out(h)                # (..., rot_dim + 3)

        rot_param = raw[..., : self.rot_dim]        # (..., rot_dim)
        t = raw[..., self.rot_dim :]                # (..., 3)
        R = rotation_rep_to_rotmat(rot_param, self.rotation_rep)  # (..., 3, 3)
        return make_se3(R, t)                       # (..., 4, 4)


class LocalPoseHead(nn.Module):
    """Predict per-frame local transform T_{i-1 <- i}.

    Concatenates context tokens of adjacent frames and decodes to SE(3).

    Output convention:
        - result[:, 0] = I (identity for frame 0)
        - result[:, i] = T_{i-1 <- i} for i >= 1
    """

    def __init__(self, d_model: int = 256, d_hidden: int = 256, rotation_rep: str = "rot6d"):
        super().__init__()
        self.decoder = _PoseDecoder(d_model * 2, d_hidden, rotation_rep)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ctx : (B, T, D) context tokens from transformer

        Returns
        -------
        local_T : (B, T, 4, 4) local transforms (frame 0 = identity)
        """
        B, T, D = ctx.shape
        device, dtype = ctx.device, ctx.dtype

        # Concatenate [h_{i-1}, h_i] for i = 1..T-1
        pairs = torch.cat([ctx[:, :-1], ctx[:, 1:]], dim=-1)  # (B, T-1, 2D)
        pred = self.decoder(pairs)  # (B, T-1, 4, 4)

        # Prepend identity for frame 0
        eye = torch.eye(4, device=device, dtype=dtype).expand(B, 1, 4, 4)
        return torch.cat([eye, pred], dim=1)  # (B, T, 4, 4)


class MultiIntervalHead(nn.Module):
    """Predict auxiliary transforms T_{i-Δ <- i} for multiple intervals Δ.

    For each interval Δ, concatenates context tokens [h_{i-Δ}, h_i] for all
    valid positions i ≥ Δ and decodes to SE(3).  Positions i < Δ are filled
    with identity.

    Parameters
    ----------
    intervals : sequence of int
        Set of intervals, e.g. [2, 4, 8, 16].
    d_model : int
        Token dimension.
    d_hidden : int
        MLP hidden dimension (shared or per-interval).
    rotation_rep : str
        Rotation parameterization.
    share_decoder : bool
        If True, share a single decoder across all intervals.
    """

    def __init__(
        self,
        intervals: Sequence[int] = (2, 4, 8, 16),
        d_model: int = 256,
        d_hidden: int = 256,
        rotation_rep: str = "rot6d",
        share_decoder: bool = False,
    ):
        super().__init__()
        self.intervals = list(intervals)
        if share_decoder:
            shared = _PoseDecoder(d_model * 2, d_hidden, rotation_rep)
            self.decoders = nn.ModuleDict(
                {str(d): shared for d in self.intervals}
            )
        else:
            self.decoders = nn.ModuleDict(
                {str(d): _PoseDecoder(d_model * 2, d_hidden, rotation_rep)
                 for d in self.intervals}
            )

    def forward(self, ctx: torch.Tensor) -> dict[int, torch.Tensor]:
        """
        Parameters
        ----------
        ctx : (B, T, D) context tokens

        Returns
        -------
        aux_T : dict[int, Tensor]
            Maps Δ → (B, T, 4, 4).  Positions i < Δ are identity.
        """
        B, T, D = ctx.shape
        device, dtype = ctx.device, ctx.dtype
        eye = torch.eye(4, device=device, dtype=dtype)

        result: dict[int, torch.Tensor] = {}
        for delta in self.intervals:
            if delta >= T:
                # Not enough frames for this interval — fill with identity
                result[delta] = eye.expand(B, T, 4, 4).clone()
                continue

            # Valid indices: i = delta .. T-1; reference: i - delta
            ctx_ref = ctx[:, : T - delta]    # (B, T-delta, D)
            ctx_cur = ctx[:, delta:]         # (B, T-delta, D)
            pairs = torch.cat([ctx_ref, ctx_cur], dim=-1)  # (B, T-delta, 2D)

            decoder = self.decoders[str(delta)]
            pred = decoder(pairs)            # (B, T-delta, 4, 4)

            # Pad with identity for positions i < delta
            pad = eye.expand(B, delta, 4, 4).clone()
            result[delta] = torch.cat([pad, pred], dim=1)  # (B, T, 4, 4)

        return result


class MultiIntervalMask:
    """Utility to build boolean masks for valid positions per interval.

    mask[delta][i] = True if i >= delta (valid for loss computation).
    """

    @staticmethod
    def build(T: int, intervals: Sequence[int], device: torch.device) -> dict[int, torch.Tensor]:
        arange = torch.arange(T, device=device)
        return {d: (arange >= d) for d in intervals}
