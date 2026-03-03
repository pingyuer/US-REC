"""Interpolation/registration helpers for rec/rec-reg trainer."""

from __future__ import annotations

from utils.funcs import compute_common_volume
from utils.interpolation import interpolation_3D_pytorch_batched


def scatter_pts_interpolation(
    *,
    labels,
    pred_pts,
    frames,
    step,
    device,
    option,
    intepoletion_method,
    intepoletion_volume,
    enable_voxel: bool = False,
):
    """Scatter points to volumes for interpolation."""
    if not enable_voxel:
        return None, None

    common_volume = None
    if option == "common_volume":
        common_volume = compute_common_volume(labels, pred_pts, device)

    gt_volume, gt_volume_position = interpolation_3D_pytorch_batched(
        scatter_pts=labels,
        frames=frames,
        time_log=None,
        saved_folder_test=None,
        scan_name="gt_step" + str(step),
        device=device,
        option=intepoletion_method,
        volume_size=intepoletion_volume,
        volume_position=common_volume,
    )

    pred_volume, pred_volume_position = interpolation_3D_pytorch_batched(
        scatter_pts=pred_pts,
        frames=frames,
        time_log=None,
        saved_folder_test=None,
        scan_name="pred_step" + str(step),
        device=device,
        option=intepoletion_method,
        volume_size=intepoletion_volume,
        volume_position=common_volume,
    )

    return gt_volume, pred_volume


def scatter_pts_registration(
    *,
    labels,
    pred_pts,
    frames,
    step,
    device,
    option,
    intepoletion_method,
    intepoletion_volume,
    voxel_morph_net,
    use_deform: bool = False,
    enable_voxel: bool = False,
):
    """Scatter points and apply registration via VoxelMorph net."""
    if not (use_deform and enable_voxel):
        return None, None, None, None
    if voxel_morph_net is None:
        raise NotImplementedError("Nonrigid disabled")

    gt_volume, pred_volume = scatter_pts_interpolation(
        labels=labels,
        pred_pts=pred_pts,
        frames=frames,
        step=step,
        device=device,
        option=option,
        intepoletion_method=intepoletion_method,
        intepoletion_volume=intepoletion_volume,
        enable_voxel=enable_voxel,
    )

    warped, ddf = voxel_morph_net(
        moving=pred_volume.unsqueeze(1),
        fixed=gt_volume.unsqueeze(1),
    )
    return gt_volume, pred_volume, warped, ddf
