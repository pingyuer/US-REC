from __future__ import annotations

import torch


def _flatten_volume(volume: torch.Tensor) -> torch.Tensor:
    if volume.dim() <= 1:
        return volume
    return volume.reshape(volume.shape[0], -1) if volume.dim() > 1 else volume


def volume_ncc(vol_pred: torch.Tensor, vol_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalized cross-correlation for 3D volumes.

    Args:
        vol_pred: Tensor of shape (B, ...) or (...).
        vol_gt: Tensor of shape (B, ...) or (...).
        eps: Numerical stability epsilon.

    Returns:
        Tensor of shape (B,) with NCC values.

    Example:
        >>> ncc = volume_ncc(pred_vol, gt_vol)
    """
    if vol_pred.dim() == vol_gt.dim() == 0:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)
    if vol_pred.dim() == 1:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)
    pred = vol_pred.reshape(vol_pred.shape[0], -1)
    gt = vol_gt.reshape(vol_gt.shape[0], -1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    gt = gt - gt.mean(dim=1, keepdim=True)
    numerator = (pred * gt).sum(dim=1)
    denom = torch.sqrt((pred ** 2).sum(dim=1) * (gt ** 2).sum(dim=1))
    ncc = numerator / (denom + eps)
    zero_mask = denom <= eps
    if zero_mask.any():
        same_mask = ((pred - gt).abs().sum(dim=1) <= eps)
        ncc = torch.where(zero_mask & same_mask, torch.ones_like(ncc), ncc)
        ncc = torch.where(zero_mask & ~same_mask, torch.zeros_like(ncc), ncc)
    return ncc


def volume_ssim(vol_pred: torch.Tensor, vol_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global SSIM for 3D volumes (simple, non-windowed).

    Args:
        vol_pred: Tensor of shape (B, ...) or (...).
        vol_gt: Tensor of shape (B, ...) or (...).
        eps: Numerical stability epsilon.

    Returns:
        Tensor of shape (B,) with SSIM values.

    Example:
        >>> ssim = volume_ssim(pred_vol, gt_vol)
    """
    if vol_pred.dim() == vol_gt.dim() == 0:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)
    if vol_pred.dim() == 1:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)
    pred = vol_pred.reshape(vol_pred.shape[0], -1)
    gt = vol_gt.reshape(vol_gt.shape[0], -1)

    mu_x = pred.mean(dim=1)
    mu_y = gt.mean(dim=1)
    sigma_x = pred.var(dim=1, unbiased=False)
    sigma_y = gt.var(dim=1, unbiased=False)
    sigma_xy = ((pred - mu_x[:, None]) * (gt - mu_y[:, None])).mean(dim=1)

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return num / (den + eps)


def volume_dice(vol_pred: torch.Tensor, vol_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice score for binary 3D volumes.

    Args:
        vol_pred: Tensor of shape (B, ...) or (...). Thresholded at 0.5 if float.
        vol_gt: Tensor of shape (B, ...) or (...). Thresholded at 0.5 if float.
        eps: Numerical stability epsilon.

    Returns:
        Tensor of shape (B,) with Dice scores.

    Example:
        >>> dice = volume_dice(pred_vol, gt_vol)
    """
    if vol_pred.dim() == vol_gt.dim() == 0:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)
    if vol_pred.dim() == 1:
        vol_pred = vol_pred.unsqueeze(0)
        vol_gt = vol_gt.unsqueeze(0)

    pred = vol_pred.reshape(vol_pred.shape[0], -1)
    gt = vol_gt.reshape(vol_gt.shape[0], -1)

    if pred.dtype.is_floating_point:
        pred = pred > 0.5
    if gt.dtype.is_floating_point:
        gt = gt > 0.5

    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    return (2 * inter + eps) / (denom + eps)
