"""Unified pose-loss primitives — single source for rotation + translation losses.

Eliminates duplication between ``dual_loss._se3_pose_loss`` (chordal + smooth-L1)
and ``longseq_loss._se3_pose_loss`` (geodesic + MSE).  Both can now import from here.

Public API
----------
``chordal_rotation_loss(R_pred, R_gt)`` — Frobenius ‖R₁ − R₂‖_F
``geodesic_rotation_loss(R_pred, R_gt)`` — acos-based geodesic distance
``se3_chordal_loss(pred_T, gt_T, …)`` — chordal rot + smooth-L1 trans
``se3_geodesic_loss(pred_T, gt_T, …)`` — geodesic rot + MSE trans
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─── Rotation losses ────────────────────────────────────────────────────────

def chordal_rotation_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Chordal distance: ‖R₁ − R₂‖_F.

    Everywhere-differentiable, avoids the acos singularity of geodesic.
    Relationship: d_chord = 2√2 sin(θ/2).

    Parameters
    ----------
    R_pred, R_gt : (..., 3, 3)

    Returns
    -------
    Scalar — mean chordal distance.
    """
    diff = R_pred - R_gt
    frob = torch.sqrt((diff * diff).sum(dim=(-2, -1)).clamp(min=1e-8))
    return frob.mean()


def geodesic_rotation_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic (angular) distance between rotation matrices.

    Delegates to ``utils.rotation_loss.geodesic_loss`` when available,
    falls back to a direct implementation otherwise.
    """
    try:
        from utils.rotation_loss import geodesic_loss  # noqa: PLC0415
        return geodesic_loss(R_pred, R_gt)
    except ImportError:
        # Inline fallback: angle from trace
        R_diff = R_pred.transpose(-2, -1) @ R_gt
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_angle = (trace - 1.0).clamp(-1.0, 1.0) / 2.0
        return torch.acos(cos_angle).mean()


# ─── Combined SE(3) losses ──────────────────────────────────────────────────

def se3_chordal_loss(
    pred_T: torch.Tensor,
    gt_T: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chordal rotation + Smooth-L1 translation loss (Dual / KRoot path).

    Parameters
    ----------
    pred_T, gt_T : (..., 4, 4)
    mask : optional bool mask of valid positions
    """
    R_pred = pred_T[..., :3, :3]
    R_gt = gt_T[..., :3, :3]
    t_pred = pred_T[..., :3, 3]
    t_gt = gt_T[..., :3, 3]

    if mask is not None:
        R_pred = R_pred[mask]
        R_gt = R_gt[mask]
        t_pred = t_pred[mask]
        t_gt = t_gt[mask]

    if R_pred.numel() == 0:
        zero = torch.tensor(0.0, device=pred_T.device, dtype=pred_T.dtype)
        return zero, {"rot_loss": 0.0, "trans_loss": 0.0}

    rot_l = chordal_rotation_loss(R_pred, R_gt)
    trans_l = F.smooth_l1_loss(t_pred, t_gt, beta=1.0)
    total = rot_weight * rot_l + trans_weight * trans_l
    return total, {
        "rot_loss": float(rot_l.detach()),
        "trans_loss": float(trans_l.detach()),
    }


def se3_geodesic_loss(
    pred_T: torch.Tensor,
    gt_T: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Geodesic rotation + MSE translation loss (LongSeq path).

    Parameters
    ----------
    pred_T, gt_T : (..., 4, 4)
    mask : optional bool mask of valid positions
    """
    R_pred = pred_T[..., :3, :3]
    R_gt = gt_T[..., :3, :3]
    t_pred = pred_T[..., :3, 3]
    t_gt = gt_T[..., :3, 3]

    if mask is not None:
        R_pred = R_pred[mask]
        R_gt = R_gt[mask]
        t_pred = t_pred[mask]
        t_gt = t_gt[mask]

    if R_pred.numel() == 0:
        zero = torch.tensor(0.0, device=pred_T.device, dtype=pred_T.dtype)
        return zero, {"rot_loss": 0.0, "trans_loss": 0.0}

    rot_l = geodesic_rotation_loss(R_pred, R_gt)
    trans_l = F.mse_loss(t_pred, t_gt)
    total = rot_weight * rot_l + trans_weight * trans_l
    return total, {
        "rot_loss": float(rot_l.detach().item()),
        "trans_loss": float(trans_l.detach().item()),
    }
