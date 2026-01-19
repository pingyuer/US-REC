from __future__ import annotations

import torch


def _to_rotation_matrix(rot: torch.Tensor) -> torch.Tensor:
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Rotation input must have shape (..., 3, 3).")
    return rot


def translation_error(pred_trans: torch.Tensor, gt_trans: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 translation error (meters or mm) for batched translations.

    Args:
        pred_trans: Tensor of shape (..., 3) translation vectors.
        gt_trans: Tensor of shape (..., 3) translation vectors.

    Returns:
        Tensor of shape (...) with L2 translation error.

    Example:
        >>> err = translation_error(pred_t, gt_t)
    """
    diff = pred_trans - gt_trans
    return torch.linalg.norm(diff, dim=-1)


def rotation_error(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    """
    Compute SO(3) geodesic rotation error in degrees.

    Args:
        pred_rot: Tensor of shape (..., 3, 3) rotation matrices.
        gt_rot: Tensor of shape (..., 3, 3) rotation matrices.

    Returns:
        Tensor of shape (...) with rotation error in degrees.

    Example:
        >>> rot_err = rotation_error(pred_R, gt_R)
    """
    pred_rot = _to_rotation_matrix(pred_rot)
    gt_rot = _to_rotation_matrix(gt_rot)
    rel = pred_rot @ gt_rot.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / torch.pi)


def se3_error(pred_T: torch.Tensor, gt_T: torch.Tensor) -> torch.Tensor:
    """
    Combined SE(3) error using translation + rotation (deg) in a single scalar.

    Args:
        pred_T: Tensor of shape (..., 4, 4) predicted transforms.
        gt_T: Tensor of shape (..., 4, 4) ground-truth transforms.

    Returns:
        Tensor of shape (...) with combined error value.

    Example:
        >>> se3 = se3_error(pred_T, gt_T)
    """
    pred_t = pred_T[..., :3, 3]
    gt_t = gt_T[..., :3, 3]
    pred_R = pred_T[..., :3, :3]
    gt_R = gt_T[..., :3, :3]
    trans_err = translation_error(pred_t, gt_t)
    rot_err = rotation_error(pred_R, gt_R)
    return trans_err + rot_err
