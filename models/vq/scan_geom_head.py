"""Scan Geometry Head — predict coarse trajectory from scan summary g.

Used for L_geom auxiliary loss: from g, regress an 8-point coarse
trajectory target (relative translations + rotations at 8 equally
spaced time points along the scan).

The target is auto-generated from GT global transforms each step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScanGeomHead(nn.Module):
    """Small MLP to regress coarse scan trajectory from g.

    Parameters
    ----------
    d_in : int
        Dimension of scan summary g.
    n_waypoints : int
        Number of equally spaced time points (default 8).
    output_per_wp : int
        Output dims per waypoint.  Default 6 = translation xyz (3) +
        rotation axis-angle magnitude (3) relative to first frame.
    """

    def __init__(
        self,
        d_in: int = 256,
        n_waypoints: int = 8,
        output_per_wp: int = 6,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.output_per_wp = output_per_wp
        out_dim = n_waypoints * output_per_wp

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        g : (B, D_in)

        Returns
        -------
        pred_traj : (B, n_waypoints, output_per_wp)
        """
        raw = self.mlp(g)  # (B, N*O)
        return raw.reshape(g.size(0), self.n_waypoints, self.output_per_wp)


# ─── Loss utilities ──────────────────────────────────────────────────────────

def build_geom_target(
    gt_global_T: torch.Tensor,
    n_waypoints: int = 8,
) -> torch.Tensor:
    """Auto-generate coarse trajectory target from GT global transforms.

    Parameters
    ----------
    gt_global_T : (B, T, 4, 4)
        Ground-truth global transforms (T_{0←i}).
    n_waypoints : int

    Returns
    -------
    target : (B, n_waypoints, 6)
        Per-waypoint: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        Translation = relative to first frame.
        Rotation = axis-angle (Rodriguez) relative to first frame.
    """
    B, T, _, _ = gt_global_T.shape
    device, dtype = gt_global_T.device, gt_global_T.dtype

    # Pick n_waypoints equally spaced indices (including first and last)
    if T <= n_waypoints:
        indices = torch.arange(T, device=device)
        # Pad with last index
        if T < n_waypoints:
            pad = torch.full((n_waypoints - T,), T - 1, device=device, dtype=torch.long)
            indices = torch.cat([indices, pad])
    else:
        indices = torch.linspace(0, T - 1, n_waypoints, device=device).long()

    # Gather waypoint transforms: (B, n_wp, 4, 4)
    wp_T = gt_global_T[:, indices]  # (B, n_wp, 4, 4)

    # Relative to first frame: T_{0←wp} @ inv(T_{0←0}) = T_{0←wp}  (frame 0 is identity)
    # If frame 0 transform is not identity, compute relative:
    T0_inv = torch.inverse(wp_T[:, 0:1])  # (B, 1, 4, 4)
    rel_T = T0_inv @ wp_T                 # (B, n_wp, 4, 4)

    # Extract translation (relative to frame 0)
    trans = rel_T[:, :, :3, 3]  # (B, n_wp, 3)

    # Extract rotation as axis-angle (compact 3D)
    R = rel_T[:, :, :3, :3]    # (B, n_wp, 3, 3)
    aa = _rotmat_to_axis_angle(R)  # (B, n_wp, 3)

    target = torch.cat([trans, aa], dim=-1)  # (B, n_wp, 6)
    return target


def _rotmat_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle (Rodriguez) representation.

    Parameters
    ----------
    R : (..., 3, 3) rotation matrices

    Returns
    -------
    aa : (..., 3) axis-angle vectors (direction = axis, magnitude = angle)
    """
    # Using the formula: θ = arccos((tr(R)-1)/2), axis from skew-symmetric part
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    traces = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_angle = (traces - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (N,)

    # Axis from skew-symmetric part of R
    # r = [R32-R23, R13-R31, R21-R12] / (2*sin(θ))
    axis = torch.stack([
        R_flat[:, 2, 1] - R_flat[:, 1, 2],
        R_flat[:, 0, 2] - R_flat[:, 2, 0],
        R_flat[:, 1, 0] - R_flat[:, 0, 1],
    ], dim=-1)  # (N, 3)

    sin_angle = torch.sin(angle).clamp(min=1e-7).unsqueeze(-1)  # (N, 1)
    axis = axis / (2.0 * sin_angle)

    # Scale by angle
    aa = axis * angle.unsqueeze(-1)  # (N, 3)
    return aa.reshape(*batch_shape, 3)


def geom_loss(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
) -> torch.Tensor:
    """L1 loss between predicted and target coarse trajectory.

    Parameters
    ----------
    pred_traj   : (B, n_wp, 6)
    target_traj : (B, n_wp, 6)

    Returns
    -------
    loss : scalar
    """
    return F.l1_loss(pred_traj, target_traj)


def consistency_loss(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Cosine consistency loss between two scan summary views.

    Parameters
    ----------
    g1, g2 : (B, D) scan summaries from two different anchor subsets

    Returns
    -------
    loss : scalar  (1 - cos_sim, so 0 = perfectly consistent)
    """
    sim = F.cosine_similarity(g1, g2, dim=-1)  # (B,)
    return (1.0 - sim).mean()
