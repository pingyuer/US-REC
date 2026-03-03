"""Reconstruction/registration helper ops (canonical location)."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_dimention(label_pred_type, num_points_each_frame=None, num_frames=None, type_option=None, rotation_rep="se3_expmap"):
    if type_option == "pred":
        num_frames = num_frames - 1

    # Per-pair dimension depends on rotation representation for "parameter" type
    from utils.rotation import ROTATION_REP_DIM
    if label_pred_type == "parameter":
        per_pair = ROTATION_REP_DIM.get(rotation_rep, 6)
        return per_pair * num_frames

    type_dim_dict = {
        "transform": 12 * num_frames,
        "point": 3 * 4 * num_frames,  # predict four corner points, and then intepolete the other points in a frame
        "quaternion": 7 * num_frames,
    }
    return type_dim_dict[label_pred_type]


def data_pairs_adjacent(num_frames):
    """
    Build adjacent frame pairs with an identity anchor in front.

    Returns pairs shaped (num_frames, 2):
    - pair[0] = [0, 0] (identity for frame0)
    - pair[i] = [i-1, i] for i >= 1 (adjacent frame relation)
    """
    n = int(num_frames)
    if n <= 0:
        raise ValueError("num_frames must be >= 1")
    pairs = [[0, 0]]
    pairs.extend([[i - 1, i] for i in range(1, n)])
    return torch.tensor(pairs, dtype=torch.long)


def scatter_plot_3D(data, save_folder, save_name):
    """Plot 3D scatter points."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[:, 0, :], data[:, 1, :], data[:, 2, :], marker="o")
    plt.show()


def union_volome(gt_volume_position, pred_volume, pred_volume_position):
    """Crop pred_volume to the bounding box of gt_volume_position."""
    gt_X = gt_volume_position[0]
    gt_Y = gt_volume_position[1]
    gt_Z = gt_volume_position[2]
    min_x = torch.min(gt_X)
    max_x = torch.max(gt_X)
    min_y = torch.min(gt_Y)
    max_y = torch.max(gt_Y)
    min_z = torch.min(gt_Z)
    max_z = torch.max(gt_Z)

    pred_X = torch.zeros((pred_volume.shape[0], pred_volume.shape[1], pred_volume.shape[2]))
    pred_X = pred_volume_position[0] + pred_X[:-1, :-1, :-1]
    pred_Y = pred_volume_position[1]
    pred_Z = pred_volume_position[2]

    inside_min_x = torch.where(pred_X > min_x, 1.0, 0.0)
    inside_max_x = torch.where(pred_X < max_x, 1.0, 0.0)
    inside_min_y = torch.where(pred_Y > min_y, 1.0, 0.0)
    inside_max_y = torch.where(pred_Y < max_y, 1.0, 0.0)
    inside_min_z = torch.where(pred_Z > min_z, 1.0, 0.0)
    inside_max_z = torch.where(pred_Z < max_z, 1.0, 0.0)

    return pred_volume * inside_min_x * inside_max_x * inside_min_y * inside_max_y * inside_min_z * inside_max_z


def calculateConvPose_batched(pts_batched, option, device):
    for i_batch in range(pts_batched.shape[0]):
        ConvR = calculateConvPose(pts_batched[i_batch, ...], option, device)
        ConvR = ConvR[None, ...]
        if i_batch == 0:
            ConvR_batched = ConvR
        else:
            ConvR_batched = torch.cat((ConvR_batched, ConvR), 0)
    return ConvR_batched


def calculateConvPose(pts, option, device):
    """Calculate roto-translation matrix to a convenient reference frame.

    Parameters
    ----------
    pts : torch.Tensor  (num_frames, 3, num_points)
    option : str
        'auto_PCA' — PCA on all US image corners.
        'first_last_frames_centroid' — X axis from first to last centroid.
    device : torch.device
    """
    if option == "auto_PCA":
        with torch.no_grad():
            pts1 = pts.permute(0, 2, 1).reshape([-1, 3])
            U, s = pca(torch.transpose(pts1, 0, 1))
            convR = torch.vstack(
                (
                    torch.hstack((U, torch.zeros((3, 1)).to(device))),
                    torch.tensor([0, 0, 0, 1]).to(device),
                )
            )
    elif option == "first_last_frames_centroid":
        C0 = torch.mean(pts[0, :, :], 1)  # 3
        C1 = torch.mean(pts[-1, :, :], 1)  # 3
        X = C1 - C0
        Ytemp = pts[0, :, 0] - pts[0, :, 1]
        Z = torch.cross(X, Ytemp)
        Y = torch.cross(Z, X)
        X = X / torch.linalg.norm(X)
        Y = Y / torch.linalg.norm(Y)
        Z = Z / torch.linalg.norm(Z)
        M = torch.transpose(torch.stack((X, Y, Z), 0), 0, 1)
        convR = torch.transpose(
            torch.vstack(
                (torch.hstack((M, torch.zeros((3, 1)).to(device))), torch.tensor([0, 0, 0, 1]).to(device))
            ),
            0,
            1,
        )
    else:
        raise ValueError(f"Unknown option: {option!r}")

    return convR


def ConvPose(labels, ori_pts, pre, option_method, device):
    convR_batched = calculateConvPose_batched(labels, option=option_method, device=device)

    for i_batch in range(convR_batched.shape[0]):
        labels_i = torch.matmul(ori_pts[i_batch, ...], convR_batched[i_batch, ...])[None, ...]
        minx = torch.min(labels_i[..., 0])
        miny = torch.min(labels_i[..., 1])
        minz = torch.min(labels_i[..., 2])
        labels_i[..., 0] -= minx
        labels_i[..., 1] -= miny
        labels_i[..., 2] -= minz
        pred_pts_i = torch.matmul(pre[i_batch, ...], convR_batched[i_batch, ...])[None, ...]

        minxyz = torch.from_numpy(np.array([minx.item(), miny.item(), minz.item()]))

        pred_pts_i[..., 0] -= minx
        pred_pts_i[..., 1] -= miny
        pred_pts_i[..., 2] -= minz

        if i_batch == 0:
            labels_opt = labels_i
            pred_pts_opt = pred_pts_i
            minxyz_all = minxyz
        else:
            labels_opt = torch.cat((labels_opt, labels_i), 0)
            pred_pts_opt = torch.cat((pred_pts_opt, pred_pts_i), 0)
            minxyz_all = torch.cat((minxyz_all, minxyz), 0)

    labels_opt = labels_opt[:, :, :, 0:3].permute(0, 1, 3, 2)
    pred_pts_opt = pred_pts_opt[:, :, :, 0:3].permute(0, 1, 3, 2)
    return labels_opt, pred_pts_opt, convR_batched, minxyz_all


def pca(D):
    """Run PCA on data matrix via SVD of covariance.

    Parameters
    ----------
    D : torch.Tensor  (Nv, No)
    """
    cov = torch.cov(D)
    U, s, V = torch.linalg.svd(cov)
    return U, s
