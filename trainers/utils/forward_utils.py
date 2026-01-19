"""Forward helpers for rec/rec-reg trainer."""

from __future__ import annotations

import torch

from utils.rec_ops import ConvPose


def unpack_batch(batch):
    """Unpack batch into (frames, tforms, tforms_inv)."""
    if isinstance(batch, dict):
        frames = batch["frames"]
        tforms = batch["tforms"]
        tforms_inv = batch["tforms_inv"]
    else:
        frames, tforms, tforms_inv = batch
    return frames, tforms, tforms_inv


def build_pred_transforms(transform_prediction, outputs, device):
    """Convert model outputs into full transforms with identity prepended."""
    pred_transfs = transform_prediction(outputs)
    predframe0 = torch.eye(4, 4)[None, ...].repeat(pred_transfs.shape[0], 1, 1, 1).to(device)
    return torch.cat((predframe0, pred_transfs), 1)


def points_from_transforms(
    *,
    img_pro_coord,
    tform_calib_R_T,
    tform_calib,
    image_points,
    tforms_each_frame2frame0,
    pred_transfs,
):
    """Compute labels/pred points from transforms."""
    if img_pro_coord == "img_coord":
        labels = torch.matmul(
            torch.linalg.inv(tform_calib_R_T),
            torch.matmul(tforms_each_frame2frame0, torch.matmul(tform_calib, image_points)),
        )[:, :, 0:3, ...]
        pred_pts = torch.matmul(
            torch.linalg.inv(tform_calib_R_T),
            torch.matmul(pred_transfs, torch.matmul(tform_calib, image_points)),
        )[:, :, 0:3, ...]
    else:
        labels = torch.matmul(
            tforms_each_frame2frame0, torch.matmul(tform_calib, image_points)
        )[:, :, 0:3, ...]
        pred_pts = torch.matmul(
            pred_transfs, torch.matmul(tform_calib, image_points)
        )[:, :, 0:3, ...]
    return labels, pred_pts


def convpose_if_needed(
    *,
    conv_coords,
    img_pro_coord,
    tforms_each_frame2frame0,
    pred_transfs,
    tform_calib,
    image_points,
    labels,
    pred_pts,
    device,
):
    """Apply ConvPose when optimised_coord + pro_coord is enabled."""
    convR_batched = None
    minxyz_all = None
    if conv_coords == "optimised_coord" and img_pro_coord == "pro_coord":
        ori_pts = torch.matmul(tforms_each_frame2frame0, torch.matmul(tform_calib, image_points)).permute(0, 1, 3, 2)
        pre = torch.matmul(pred_transfs, torch.matmul(tform_calib, image_points)).permute(0, 1, 3, 2)
        labels, pred_pts, convR_batched, minxyz_all = ConvPose(
            labels, ori_pts, pre, "auto_PCA", device
        )
    elif conv_coords == "optimised_coord" and img_pro_coord != "pro_coord":
        raise RuntimeError("optimised_coord must be used when pro_coord")
    return labels, pred_pts, convR_batched, minxyz_all
