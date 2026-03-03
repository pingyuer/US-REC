"""Pairwise frame model factory (EfficientNet / ResNet backbones)."""

import torch
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
)
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
from utils.transform import PredictionTransform


def build_model(opt, in_frames, pred_dim, label_dim, image_points, tform_calib, tform_calib_R_T):
    """Build a pairwise-frame pose regression model.

    :param opt: config namespace; opt.model_name selects the backbone.
    :param in_frames: number of input frames (input channels for the first conv).
    :param pred_dim: output dimension of the final linear layer.
    :param label_dim: label dimension (unused here, kept for API consistency).
    :param image_points: image corner points tensor (unused here).
    :param tform_calib: calibration transform (unused here).
    :param tform_calib_R_T: calibration R_T transform (unused here).
    """
    if opt.model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias,
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim,
        )

    elif opt.model_name == "efficientnet_b6":
        model = efficientnet_b6(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias,
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim,
        )

    elif opt.model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias,
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim,
        )

    elif opt.model_name[:6] == "resnet":
        model = resnet101()
        model.conv1 = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=model.conv1.bias,
        )
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features,
            out_features=pred_dim,
        )

    else:
        raise ValueError(f"Unknown model_name: {opt.model_name!r}")

    return model
