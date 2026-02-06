# trainers/metrics/pose.py
"""Pose metrics with explicit units and stable rotation error."""

from __future__ import annotations

from typing import Optional
import warnings

import torch


def translation_error_mm(
    pred_trans: torch.Tensor, gt_trans: torch.Tensor, *, scale: float = 1.0
) -> torch.Tensor:
    """
    L2 translation error in mm.

    scale applies a unit conversion factor (default 1.0).
    """
    diff = (pred_trans - gt_trans) * float(scale)
    return torch.linalg.norm(diff, dim=-1)


def _project_to_so3(rot: torch.Tensor) -> torch.Tensor:
    u, _, v = torch.linalg.svd(rot)
    r = u @ v.transpose(-1, -2)
    det = torch.det(r)
    if torch.any(det < 0):
        u_adj = u.clone()
        u_adj[..., :, -1] *= -1.0
        r = u_adj @ v.transpose(-1, -2)
    return r


def rotation_error_deg(
    pred_rot: torch.Tensor,
    gt_rot: torch.Tensor,
    *,
    check_valid: bool = False,
    orthonormalize: bool = False,
    eps: float = 1e-6,
    valid_tol: float = 1e-2,
    det_tol: float = 0.9,
    raise_on_invalid: bool = False,
) -> torch.Tensor:
    """
    SO(3) geodesic rotation error in degrees.
    """
    if pred_rot.shape[-2:] != (3, 3) or gt_rot.shape[-2:] != (3, 3):
        raise ValueError("Rotation input must have shape (..., 3, 3).")

    pred = pred_rot
    gt = gt_rot
    if orthonormalize:
        pred = _project_to_so3(pred)
        gt = _project_to_so3(gt)

    if check_valid:
        eye = torch.eye(3, device=pred.device, dtype=pred.dtype)
        pred_err = torch.linalg.norm(pred.transpose(-1, -2) @ pred - eye, dim=(-2, -1))
        gt_err = torch.linalg.norm(gt.transpose(-1, -2) @ gt - eye, dim=(-2, -1))
        pred_det = torch.det(pred)
        gt_det = torch.det(gt)
        invalid = bool(
            (pred_err.max() > valid_tol)
            or (gt_err.max() > valid_tol)
            or (pred_det.min() < det_tol)
            or (gt_det.min() < det_tol)
        )
        if invalid:
            msg = "Rotation matrices appear invalid (RtR or det out of tolerance)."
            if raise_on_invalid:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)

    rel = pred @ gt.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / torch.pi)


def se3_translation_error(pred_T: torch.Tensor, gt_T: torch.Tensor) -> torch.Tensor:
    return translation_error_mm(pred_T[..., :3, 3], gt_T[..., :3, 3])


def se3_rotation_error_deg(
    pred_T: torch.Tensor, gt_T: torch.Tensor, **kwargs
) -> torch.Tensor:
    return rotation_error_deg(pred_T[..., :3, :3], gt_T[..., :3, :3], **kwargs)


__all__ = [
    "translation_error_mm",
    "rotation_error_deg",
    "se3_translation_error",
    "se3_rotation_error_deg",
]
