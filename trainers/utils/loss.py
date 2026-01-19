"""Loss helpers for rec/rec-reg trainer."""

from __future__ import annotations

import torch

from utils.utils_ori import compute_plane_normal, angle_between_planes


def compute_loss(
    *,
    loss_type,
    labels,
    pred_pts,
    frames,
    step,
    criterion,
    img_loss,
    regularization,
    reg_loss_weight,
    ddf_dirc,
    conv_coords,
    option,
    device,
    scatter_pts_registration,
    scatter_pts_interpolation,
    wrapped_pred_dist_fn,
    convR_batched=None,
    minxyz_all=None,
    rigid_only: bool = True,
):
    """Compute loss, auxiliary losses, and metrics for a batch."""
    loss1 = criterion(pred_pts, labels)
    dist = ((pred_pts - labels) ** 2).sum(dim=2).sqrt().mean()
    wrap_dist = torch.tensor(0.0, device=device)
    loss2 = loss1
    ddf = None
    gt_volume = None
    pred_volume = None
    non_rigid_loss_types = {"reg", "rec_reg", "wraped", "rec_volume", "rec_volume10000", "volume_only"}

    if rigid_only and loss_type in non_rigid_loss_types:
        # VoxelMorph / non-rigid registration is intentionally disabled in rigid-only mode.
        loss = loss1
        loss2 = torch.tensor(0.0, device=device)
        extras = {}
        return loss, loss1, loss2, dist, wrap_dist, extras

    if loss_type == "MSE_points":
        loss = loss1
        loss2 = loss
    elif loss_type == "Plane_norm":
        normal_gt = compute_plane_normal(labels)
        normal_np = compute_plane_normal(pred_pts)
        cos_value = angle_between_planes(normal_gt, normal_np)
        loss = loss1 - sum(sum(cos_value))
    elif loss_type in {"reg", "rec_reg", "wraped"}:
        gt_volume, pred_volume, warped, ddf = scatter_pts_registration(labels, pred_pts, frames, step)
        if ddf_dirc == "Move" and conv_coords == "optimised_coord":
            wrap_mseloss, wrap_dist, _ = wrapped_pred_dist_fn(
                ddf,
                pred_pts,
                labels,
                option,
                frames.shape[2],
                frames.shape[3],
                convR_batched,
                minxyz_all,
                device,
            )
        if ddf_dirc == "Fix":
            loss2 = img_loss(torch.squeeze(warped, 1), gt_volume) + regularization(ddf)
        elif ddf_dirc == "Move":
            loss2 = img_loss(torch.squeeze(warped, 1), pred_volume) + regularization(ddf)
        if loss_type == "reg":
            loss = loss2
        elif loss_type == "rec_reg":
            loss = loss1 + reg_loss_weight * loss2
        elif loss_type == "wraped" and ddf_dirc == "Move":
            loss = wrap_mseloss + regularization(ddf)
    elif loss_type == "rec_volume":
        gt_volume, pred_volume = scatter_pts_interpolation(labels, pred_pts, frames, step)
        loss2 = criterion(pred_volume, gt_volume)
        loss = loss1 + loss2
    elif loss_type == "rec_volume10000":
        gt_volume, pred_volume = scatter_pts_interpolation(labels, pred_pts, frames, step)
        loss2 = criterion(pred_volume, gt_volume)
        loss = loss1 + reg_loss_weight * loss2
    elif loss_type == "volume_only":
        gt_volume, pred_volume = scatter_pts_interpolation(labels, pred_pts, frames, step)
        loss = criterion(pred_volume, gt_volume)
        loss2 = loss
    else:
        loss = loss1

    extras = {}
    if loss_type in {"reg", "rec_reg", "wraped"}:
        extras["ddf"] = ddf
    if loss_type in {"rec_volume", "rec_volume10000", "volume_only"}:
        extras["gt_volume"] = gt_volume
        extras["pred_volume"] = pred_volume

    return loss, loss1, loss2, dist, wrap_dist, extras
