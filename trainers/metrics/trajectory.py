# trainers/metrics/trajectory.py
"""Trajectory metrics with explicit units and SE(3) analytic inverse."""

from __future__ import annotations

import torch

from .pose import translation_error_mm, rotation_error_deg


def _se3_inv(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t[..., None]).squeeze(-1)
    eye = torch.zeros_like(T)
    eye[..., :3, :3] = R_inv
    eye[..., :3, 3] = t_inv
    eye[..., 3, 3] = 1.0
    return eye


def endpoint_rpe_translation_mm(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor
) -> torch.Tensor:
    """Endpoint relative pose error (translation, mm) between start and end."""
    pred_delta = _se3_inv(pred_Ts[:, 0]) @ pred_Ts[:, -1]
    gt_delta = _se3_inv(gt_Ts[:, 0]) @ gt_Ts[:, -1]
    return translation_error_mm(pred_delta[..., :3, 3], gt_delta[..., :3, 3])


def endpoint_rpe_rotation_deg(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor
) -> torch.Tensor:
    """Endpoint relative pose error (rotation, deg) between start and end."""
    pred_delta = _se3_inv(pred_Ts[:, 0]) @ pred_Ts[:, -1]
    gt_delta = _se3_inv(gt_Ts[:, 0]) @ gt_Ts[:, -1]
    return rotation_error_deg(pred_delta[..., :3, :3], gt_delta[..., :3, :3])


def end_to_start_rpe_translation_mm(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor
) -> torch.Tensor:
    """End-to-start relative pose error (translation, mm)."""
    pred_loop = _se3_inv(pred_Ts[:, -1]) @ pred_Ts[:, 0]
    gt_loop = _se3_inv(gt_Ts[:, -1]) @ gt_Ts[:, 0]
    return translation_error_mm(pred_loop[..., :3, 3], gt_loop[..., :3, 3])


def end_to_start_rpe_rotation_deg(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor
) -> torch.Tensor:
    """End-to-start relative pose error (rotation, deg)."""
    pred_loop = _se3_inv(pred_Ts[:, -1]) @ pred_Ts[:, 0]
    gt_loop = _se3_inv(gt_Ts[:, -1]) @ gt_Ts[:, 0]
    return rotation_error_deg(pred_loop[..., :3, :3], gt_loop[..., :3, :3])


def rpe_translation_mm(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor, *, delta: int = 1
) -> torch.Tensor:
    """Relative pose error (translation, mm) over delta steps."""
    if delta <= 0:
        raise ValueError("delta must be >= 1")
    pred_rel = _se3_inv(pred_Ts[:, :-delta]) @ pred_Ts[:, delta:]
    gt_rel = _se3_inv(gt_Ts[:, :-delta]) @ gt_Ts[:, delta:]
    err = translation_error_mm(pred_rel[..., :3, 3], gt_rel[..., :3, 3])
    return err.mean(dim=-1)


def rpe_rotation_deg(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor, *, delta: int = 1
) -> torch.Tensor:
    """Relative pose error (rotation, deg) over delta steps."""
    if delta <= 0:
        raise ValueError("delta must be >= 1")
    pred_rel = _se3_inv(pred_Ts[:, :-delta]) @ pred_Ts[:, delta:]
    gt_rel = _se3_inv(gt_Ts[:, :-delta]) @ gt_Ts[:, delta:]
    err = rotation_error_deg(pred_rel[..., :3, :3], gt_rel[..., :3, :3])
    return err.mean(dim=-1)


def endpoint_drift_rate(
    pred_Ts: torch.Tensor, gt_Ts: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Endpoint translation error normalized by path length (gt)."""
    path = gt_Ts[:, 1:, :3, 3] - gt_Ts[:, :-1, :3, 3]
    path_len = torch.linalg.norm(path, dim=-1).sum(dim=-1)
    endpoint = endpoint_rpe_translation_mm(pred_Ts, gt_Ts)
    return endpoint / (path_len + eps)


__all__ = [
    "endpoint_rpe_translation_mm",
    "endpoint_rpe_rotation_deg",
    "end_to_start_rpe_translation_mm",
    "end_to_start_rpe_rotation_deg",
    "rpe_translation_mm",
    "rpe_rotation_deg",
    "endpoint_drift_rate",
]
