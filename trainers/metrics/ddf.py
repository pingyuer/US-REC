# trainers/metrics/ddf.py
"""DDF metrics with explicit units and EPE."""

from __future__ import annotations

from typing import Optional, Sequence
import warnings

import torch


def _infer_vec_dim(tensor: torch.Tensor) -> int:
    if tensor.ndim >= 2 and tensor.shape[1] == 3:
        return 1
    if tensor.ndim >= 1 and tensor.shape[-1] == 3:
        return -1
    raise ValueError("Cannot infer vec_dim (expected channel=3 or last_dim=3).")


def _apply_spacing(diff: torch.Tensor, spacing: Sequence[float], vec_dim: int) -> torch.Tensor:
    spacing_t = torch.tensor(spacing, device=diff.device, dtype=diff.dtype)
    if spacing_t.numel() != 3:
        raise ValueError("spacing must be a 3-tuple (sx, sy, sz).")
    shape = [1] * diff.ndim
    shape[vec_dim] = 3
    return diff * spacing_t.view(shape)


def ddf_rmse_all_dims(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    """RMSE over all elements of the displacement field (unitless)."""
    diff = pred_ddf - gt_ddf
    return torch.sqrt(torch.mean(diff * diff))


def ddf_mae_all_dims(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    """MAE over all elements of the displacement field (unitless)."""
    diff = pred_ddf - gt_ddf
    return torch.mean(torch.abs(diff))


def ddf_epe_mean(
    pred_ddf: torch.Tensor,
    gt_ddf: torch.Tensor,
    *,
    vec_dim: Optional[int] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Mean endpoint error (EPE) over displacement vectors.

    If spacing is provided, EPE is computed in mm; otherwise in vox/px.
    """
    diff = pred_ddf - gt_ddf
    vec_dim = _infer_vec_dim(diff) if vec_dim is None else int(vec_dim)
    if spacing is not None:
        diff = _apply_spacing(diff, spacing, vec_dim)
    epe = torch.linalg.norm(diff, dim=vec_dim)
    return epe.mean()


def ddf_epe_vox(
    pred_ddf: torch.Tensor,
    gt_ddf: torch.Tensor,
    *,
    vec_dim: Optional[int] = None,
) -> torch.Tensor:
    """Mean EPE in vox/px."""
    return ddf_epe_mean(pred_ddf, gt_ddf, vec_dim=vec_dim, spacing=None)


def ddf_epe_mm(
    pred_ddf: torch.Tensor,
    gt_ddf: torch.Tensor,
    *,
    spacing: Sequence[float],
    vec_dim: Optional[int] = None,
) -> torch.Tensor:
    """Mean EPE in mm."""
    return ddf_epe_mean(pred_ddf, gt_ddf, vec_dim=vec_dim, spacing=spacing)


def ddf_rmse(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    warnings.warn("ddf_rmse is deprecated; use ddf_rmse_all_dims.", RuntimeWarning)
    return ddf_rmse_all_dims(pred_ddf, gt_ddf)


def ddf_mae(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    warnings.warn("ddf_mae is deprecated; use ddf_mae_all_dims.", RuntimeWarning)
    return ddf_mae_all_dims(pred_ddf, gt_ddf)


__all__ = [
    "ddf_rmse_all_dims",
    "ddf_mae_all_dims",
    "ddf_epe_mean",
    "ddf_epe_vox",
    "ddf_epe_mm",
    "ddf_rmse",
    "ddf_mae",
]
