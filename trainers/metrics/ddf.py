from __future__ import annotations

import torch


def ddf_rmse(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    """
    Compute RMSE for dense displacement fields.

    Args:
        pred_ddf: Tensor of shape (B, 3, ...) or (..., 3).
        gt_ddf: Tensor of shape (B, 3, ...) or (..., 3).

    Returns:
        Scalar tensor RMSE.

    Example:
        >>> rmse = ddf_rmse(pred_ddf, gt_ddf)
    """
    diff = pred_ddf - gt_ddf
    return torch.sqrt(torch.mean(diff ** 2))


def ddf_mae(pred_ddf: torch.Tensor, gt_ddf: torch.Tensor) -> torch.Tensor:
    """
    Compute MAE for dense displacement fields.

    Args:
        pred_ddf: Tensor of shape (B, 3, ...) or (..., 3).
        gt_ddf: Tensor of shape (B, 3, ...) or (..., 3).

    Returns:
        Scalar tensor MAE.

    Example:
        >>> mae = ddf_mae(pred_ddf, gt_ddf)
    """
    diff = pred_ddf - gt_ddf
    return torch.mean(torch.abs(diff))
