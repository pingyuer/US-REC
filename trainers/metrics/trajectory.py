from __future__ import annotations

import torch

from .pose import se3_error


def _batched_inv(T: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(T)


def cumulative_drift(
    pred_transforms: torch.Tensor, gt_transforms: torch.Tensor
) -> torch.Tensor:
    """
    Measure drift over a sequence by comparing start->end transforms.

    Args:
        pred_transforms: Tensor of shape (B, T, 4, 4).
        gt_transforms: Tensor of shape (B, T, 4, 4).

    Returns:
        Tensor of shape (B,) with drift error.

    Example:
        >>> drift = cumulative_drift(pred_Ts, gt_Ts)
    """
    if pred_transforms.shape[-2:] != (4, 4) or gt_transforms.shape[-2:] != (4, 4):
        raise ValueError("Transforms must have shape (B, T, 4, 4).")
    pred_delta = _batched_inv(pred_transforms[:, 0]) @ pred_transforms[:, -1]
    gt_delta = _batched_inv(gt_transforms[:, 0]) @ gt_transforms[:, -1]
    return se3_error(pred_delta, gt_delta)


def loop_closure_error(
    pred_transforms: torch.Tensor, gt_transforms: torch.Tensor
) -> torch.Tensor:
    """
    Compute loop closure error by comparing final loop return transforms.

    Args:
        pred_transforms: Tensor of shape (B, T, 4, 4).
        gt_transforms: Tensor of shape (B, T, 4, 4).

    Returns:
        Tensor of shape (B,) with loop closure error.

    Example:
        >>> loop_err = loop_closure_error(pred_Ts, gt_Ts)
    """
    pred_loop = _batched_inv(pred_transforms[:, -1]) @ pred_transforms[:, 0]
    gt_loop = _batched_inv(gt_transforms[:, -1]) @ gt_transforms[:, 0]
    return se3_error(pred_loop, gt_loop)
