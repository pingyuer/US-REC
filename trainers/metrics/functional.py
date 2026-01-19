from __future__ import annotations

import numpy as np
import torch


def iou_score(output, target):
    """
    Legacy binary IoU + Dice helper.

    - output: logits or probabilities, any shape, thresholded at 0.5 after sigmoid
    - target: binary mask (0/1)
    Returns: (iou, dice)
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    return float(iou), float(dice)

