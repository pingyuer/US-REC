"""Rotation-aware losses for pose regression.

Provides geodesic rotation loss, quaternion inner-product loss, and a
combined pose loss that sums weighted rotation + translation terms.

All functions are pure-PyTorch and fully differentiable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.rotation import normalize_quat, quat_sign_align

_EPS = 1e-6


# ─── Rotation losses ────────────────────────────────────────────────────────

def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Geodesic (angular) distance between two rotation matrices.

    .. math::
        \\theta = \\arccos\\!\\left(\\frac{\\mathrm{tr}(R_{gt}^T R_{pred}) - 1}{2}\\right)

    Parameters
    ----------
    R_pred, R_gt : (..., 3, 3)

    Returns
    -------
    Scalar — mean angular error in **radians**.
    """
    R_rel = torch.matmul(R_gt.transpose(-2, -1), R_pred)
    # trace of (..., 3, 3) → diagonal sum
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    # Clamp strictly inside [-1, 1] to avoid NaN from acos
    cos_angle = torch.clamp(cos_angle, -1.0 + _EPS, 1.0 - _EPS)
    angle = torch.acos(cos_angle)  # radians
    # For identical rotations trace≈3 → cos_angle≈1 → angle≈0; the eps
    # clamp introduces a tiny residual (~0.001 rad).  Zero it out.
    angle = torch.where(cos_angle > 1.0 - 2 * _EPS, torch.zeros_like(angle), angle)
    return angle.mean()


def quat_inner_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Quaternion inner-product loss (sign-invariant).

    .. math::
        L = 1 - |\\langle q_{pred}, q_{gt} \\rangle|

    Parameters
    ----------
    q_pred, q_gt : (..., 4)  — [w, x, y, z], need not be unit-length.

    Returns
    -------
    Scalar — mean loss in [0, 1].
    """
    q_pred = normalize_quat(q_pred)
    q_gt = normalize_quat(q_gt)
    dot = (q_pred * q_gt).sum(dim=-1)
    return (1.0 - dot.abs()).mean()


# ─── Translation losses ─────────────────────────────────────────────────────

def l1_translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(t_pred, t_gt)


def l2_translation_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(t_pred, t_gt)


# ─── Combined pose loss ─────────────────────────────────────────────────────

def pose_loss(
    *,
    R_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_pred: torch.Tensor,
    t_gt: torch.Tensor,
    rot_loss_type: str = "geodesic",
    trans_loss_type: str = "l2",
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    q_pred: torch.Tensor | None = None,
    q_gt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted rotation + translation loss.

    Parameters
    ----------
    R_pred, R_gt : (B, 3, 3) rotation matrices
    t_pred, t_gt : (B, 3) translation vectors
    rot_loss_type : "geodesic" | "quat_inner"
    trans_loss_type : "l1" | "l2"
    rot_weight, trans_weight : scalar weights
    q_pred, q_gt : optional quaternions for quat_inner loss

    Returns
    -------
    total_loss : scalar tensor
    breakdown : dict with rot_loss, trans_loss values (detached floats)
    """
    # Rotation
    if rot_loss_type == "geodesic":
        rot_l = geodesic_loss(R_pred, R_gt)
    elif rot_loss_type == "quat_inner":
        if q_pred is None or q_gt is None:
            raise ValueError("quat_inner loss requires q_pred and q_gt")
        rot_l = quat_inner_loss(q_pred, q_gt)
    else:
        raise ValueError(f"Unknown rot_loss_type: {rot_loss_type}")

    # Translation
    if trans_loss_type == "l1":
        trans_l = l1_translation_loss(t_pred, t_gt)
    elif trans_loss_type == "l2":
        trans_l = l2_translation_loss(t_pred, t_gt)
    else:
        raise ValueError(f"Unknown trans_loss_type: {trans_loss_type}")

    total = rot_weight * rot_l + trans_weight * trans_l
    breakdown = {
        "rot_loss": float(rot_l.detach().item()),
        "trans_loss": float(trans_l.detach().item()),
    }
    return total, breakdown
