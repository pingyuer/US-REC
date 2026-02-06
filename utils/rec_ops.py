"""Reconstruction/registration helper ops."""

import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_dimention(label_pred_type, num_points_each_frame=None, num_frames=None, type_option=None):
    if type_option == "pred":
        num_frames = num_frames - 1

    type_dim_dict = {
        "transform": 12 * num_frames,
        "parameter": 6 * num_frames,
        "point": 3 * 4 * num_frames,  # predict four corner points, and then intepolete the other points in a frame
        "quaternion": 7 * num_frames,
    }
    return type_dim_dict[label_pred_type]  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]



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
    # plot 3D scatter points

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[:, 0, :], data[:, 1, :], data[:, 2, :], marker="o")
    plt.show()
    # plt.savefig(save_folder+'/'+save_name)



def union_volome(gt_volume_position, pred_volume, pred_volume_position):
    # crop the volume2 based on volume 1
    # get the boundary of ground truth volume
    #  not completed
    gt_X = gt_volume_position[0]
    gt_Y = gt_volume_position[1]
    gt_Z = gt_volume_position[2]
    min_x = torch.min(gt_X)
    max_x = torch.max(gt_X)
    min_y = torch.min(gt_Y)
    max_y = torch.max(gt_Y)
    min_z = torch.min(gt_Z)
    max_z = torch.max(gt_Z)

    #  the length of each dimention is larger than the volume, because of the torch.ceil operation, we need to
    #  add one additional length for each dimention to allow the torch.ceil opration
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
        # ConvR = ConvR.repeat(pts_batched[i_batch,...].shape[0], 1,1)[None,...]
        ConvR = ConvR[None, ...]
        if i_batch == 0:
            ConvR_batched = ConvR
        else:
            ConvR_batched = torch.cat((ConvR_batched, ConvR), 0)
    return ConvR_batched



def calculateConvPose(pts, option, device):
    """Calculate roto-translation matrix from global reference frame to *convenient* reference frame.
    Voxel-array dimensions are calculated in this new refence frame. This rotation is important whenever the US scans sihouette is remarkably
    oblique to some axis of the global reference frame. In this case, the voxel-array dimensions (calculated by the smallest parallelepipedon
    wrapping all the realigned scans), calculated in the global refrence frame, would not be optimal, i.e. larger than necessary.

    .. image:: diag_scan_direction.png
        :scale: 30 %

    Parameters
    ----------
    convR : mixed
        Roto-translation matrix.
        If str, it specifies the method for automatically calculate the matrix.
        If 'auto_PCA', PCA is performed on all US image corners. The x, y and z of the new convenient reference frame are represented by the eigenvectors out of the PCA.
        If 'first_last_frames_centroid', the convenent reference frame is expressed as:

        - x from first image centroid to last image centroid
        - z orthogonal to x and the axis and the vector joining the top-left corner to the top-right corner of the first image
        - y orthogonal to z and x

        If np.ndarray, it must be manually specified as a 4 x 4 affine matrix.

    """
    # pts = torch.reshape(pts,(pts.shape[0],-1,3))
    # pts = torch.permute(pts, (2, 0, 1))

    # Calculating best pose automatically, if necessary
    # ivx = np.array(self.voxFrames)
    if option == "auto_PCA":
        # Perform PCA on image corners
        # print ('Performing PCA on images corners...')
        with torch.no_grad():
            pts1 = pts.permute(0, 2, 1).reshape([-1, 3])  # .cpu().numpy()
            U, s = pca(torch.transpose(pts1, 0, 1))
            # Build convenience affine matrix
            convR = torch.vstack(
                (
                    torch.hstack((U, torch.zeros((3, 1)).to(device))),
                    torch.tensor([0, 0, 0, 1]).to(device),
                )
            )  # .T
            # convR = torch.from_numpy(convR).to(torch.float32).to(device)
        # print ('PCA perfomed')
    elif option == "first_last_frames_centroid":
        # Search connection from first image centroid to last image centroid (X)
        # print ('Performing convenient reference frame calculation based on first and last image centroids...')
        C0 = torch.mean(pts[0, :, :], 1)  # 3
        C1 = torch.mean(pts[-1, :, :], 1)  # 3
        X = C1 - C0
        # Define Y and Z axis
        Ytemp = pts[0, :, 0] - pts[0, :, 1]  # from top-left corner to top-right corner of the first image

        Z = torch.cross(X, Ytemp)
        Y = torch.cross(Z, X)
        # Normalize axis length
        X = X / torch.linalg.norm(X)
        Y = Y / torch.linalg.norm(Y)
        Z = Z / torch.linalg.norm(Z)
        # Create rotation matrix
        # M = np.array([X, Y, Z]).T
        M = torch.transpose(torch.stack((X, Y, Z), 0), 0, 1)
        # Build convenience affine matrix
        # convR = np.vstack((np.hstack((M,np.zeros((3,1)))),[0,0,0,1])).T
        convR = torch.transpose(
            torch.vstack(
                (torch.hstack((M, torch.zeros((3, 1)).to(device))), torch.tensor([0, 0, 0, 1]).to(device))
            ),
            0,
            1,
        )
        # print ('Convenient reference frame calculated')

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

        # return for future use
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
    """Run Principal Component Analysis on data matrix. It performs SVD
    decomposition on data covariance matrix.

    Parameters
    ----------
    D : np.ndarray
        Nv x No matrix, where Nv is the number of variables
        and No the number of observations.

    Returns
    -------
    list
        U, s as out of SVD (``see np.linalg.svd``)

    """
    cov = torch.cov(D)
    U, s, V = torch.linalg.svd(cov)
    return U, s
