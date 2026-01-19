"""BatchNorm helpers for rec/rec-reg trainer."""

from __future__ import annotations

import torch
import torch.nn as nn


def switch_off_batch_norm(model, voxel_morph_net, batch_norm_flag: str) -> None:
    """Disable BatchNorm layers when batch_norm_flag == 'BNoff'."""
    if batch_norm_flag != "BNoff":
        return

    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_parameters()
            mod.eval()
            with torch.no_grad():
                mod.weight.fill_(1.0)
                mod.bias.zero_()
                mod.momentum = 1

    for mod in voxel_morph_net.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_parameters()
            mod.eval()
            with torch.no_grad():
                mod.weight.fill_(1.0)
                mod.bias.zero_()
                mod.momentum = 1
