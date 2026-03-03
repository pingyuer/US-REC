
import torch
import heapq
try:
    import MDAnalysis.lib.transformations as MDA
except ImportError:  # optional dependency; raise when actually used
    MDA = None
import numpy as np
# import pytorch3d.transforms
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch import linalg as LA


def reference_image_points(image_size, density=2):
    """
    Build image points in pixel coordinates with top-left origin.

    :param image_size: (H, W)
    :param density: (nH, nW), sampling density along H/W; int -> square grid.
    """
    if isinstance(density, int):
        density = (density, density)

    h = int(image_size[0])
    w = int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image_size={image_size}; expected positive (H,W).")
    dh = int(density[0])
    dw = int(density[1])
    if dh <= 0 or dw <= 0:
        raise ValueError(f"Invalid density={density}; expected positive integers.")

    # Legal pixel range is [0, H-1] x [0, W-1].
    x_vals = torch.linspace(0.0, float(h - 1), dh)
    y_vals = torch.linspace(0.0, float(w - 1), dw)
    image_points = torch.cartesian_prod(x_vals, y_vals).t()

    image_points = torch.cat([
        image_points,
        torch.zeros(1, image_points.shape[1], dtype=image_points.dtype),
        torch.ones(1, image_points.shape[1], dtype=image_points.dtype)
    ], axis=0)

    max_x = float(image_points[0].max().item())
    max_y = float(image_points[1].max().item())
    if max_x > (h - 1) + 1e-6 or max_y > (w - 1) + 1e-6:
        raise ValueError(
            f"reference_image_points out of bounds: max_x={max_x}, max_y={max_y}, "
            f"expected <= ({h - 1}, {w - 1})"
        )
    return image_points


def add_scalars_rec_volume(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss_all']
    train_epoch_dist = loss_dists['train_epoch_dist']
    epoch_loss_val = loss_dists['epoch_loss_val_all']
    epoch_dist_val = loss_dists['epoch_dist_val']
    train_epoch_loss_reg = loss_dists['train_epoch_loss_reg']
    epoch_loss_val_reg = loss_dists['epoch_loss_val_reg']
    train_epoch_loss_rec = loss_dists['train_epoch_loss_rec']
    epoch_loss_val_rec = loss_dists['epoch_loss_val_rec']


    
    writer.add_scalars('loss_rec_all', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss_rec_all', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)

    writer.add_scalars('loss_rec_volume', {'train_loss': train_epoch_loss_reg},epoch)
    writer.add_scalars('loss_rec_volume', {'val_loss': epoch_loss_val_reg},epoch)

    writer.add_scalars('loss_rec', {'train_loss': train_epoch_loss_rec},epoch)
    writer.add_scalars('loss_rec', {'val_loss': epoch_loss_val_rec},epoch)

    

def add_scalars_wrap_dist(writer,epoch, loss_dists,model_name):
    # loss in training and val
    train_epoch_loss = loss_dists['train_wrap_dist']
    epoch_loss_val = loss_dists['val_wrap_dist']

    writer.add_scalars('wrap_dist_'+model_name, {'train_wrap_dist_'+model_name: train_epoch_loss},epoch)
    writer.add_scalars('wrap_dist_'+model_name, {'val_wrap_dist_'+model_name: epoch_loss_val},epoch)


def compute_plane_normal(pts):
    # Create vectors from the points
    vector1 = pts[:,:,:,1]-pts[:,:,:,0]
    vector2 = pts[:,:,:,2]-pts[:,:,:,0]
    
    # Compute the cross product of vector1 and vector2
    cross_product = torch.linalg.cross(vector1, vector2)
    
    # Normalize the cross product to get the plane's normal vector
    matrix_norm = LA.norm(cross_product, dim= 2)
    normal_vector = cross_product / matrix_norm.unsqueeze(2).repeat(1, 1, 3)
    
    return normal_vector



def angle_between_planes(normal_vector1, normal_vector2):
    # compute the cos value between two norm vertorc of two planes
   
    # Calculate the dot product of the two normal vectors
    normal_vector1 = normal_vector1.to(torch.float)
    normal_vector2 = normal_vector2.to(torch.float)

    dot_product = torch.sum(normal_vector1 * normal_vector2, dim=(2))

    # dot_product = torch.dot(normal_vector1, normal_vector2)
    
    # Calculate the magnitudes of the two normal vectors
   
    magnitude1 = LA.norm(normal_vector1, dim= 2)
    magnitude2 = LA.norm(normal_vector2, dim= 2)
    
    # Calculate the cos value using the dot product and magnitudes
    cos_value = dot_product / (magnitude1 * magnitude2)
    # np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
    return cos_value


def add_scalars_reg(writer, epoch, loss_dists, model_name):
    """Write registration-only loss scalars to TensorBoard writer."""
    train_loss = loss_dists['train_epoch_loss_reg_only']
    val_loss = loss_dists['epoch_loss_val_reg_only']
    writer.add_scalars('loss_reg_only_' + model_name, {'train_loss': train_loss}, epoch)
    writer.add_scalars('loss_reg_only_' + model_name, {'val_loss': val_loss}, epoch)


def add_scalars_reg_T(writer, epoch, dist_dicts, model_name):
    """Write registration translation-distance scalars to TensorBoard writer."""
    train_dist = dist_dicts['train_dist_reg_T']
    val_dist = dist_dicts['val_dist_reg_T']
    writer.add_scalars('dist_reg_T_' + model_name, {'train_dist': train_dist}, epoch)
    writer.add_scalars('dist_reg_T_' + model_name, {'val_dist': val_dist}, epoch)
