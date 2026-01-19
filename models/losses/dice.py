from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DiceLoss", "DiceCELoss"]


def _one_hot(
    target: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert `target` (B,H,W) to one-hot (B,C,H,W). Returns (one_hot, valid_mask).

    valid_mask is (B,1,H,W) bool when ignore_index is used, else None.
    """
    if target.dim() != 3:
        raise ValueError(f"Expected target shape (B,H,W), got {tuple(target.shape)}")

    valid_mask = None
    if ignore_index is not None:
        valid_mask = (target != ignore_index).unsqueeze(1)
        target = target.clone()
        target[target == ignore_index] = 0

    one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    return one_hot, valid_mask


@dataclass
class _DiceConfig:
    include_background: bool = True
    smooth: float = 1e-5


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    - logits: (B,C,H,W)
    - target: (B,H,W) with class indices
    """

    def __init__(
        self,
        *,
        include_background: bool = True,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = _DiceConfig(include_background=bool(include_background), smooth=float(smooth))
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError(f"Expected logits shape (B,C,H,W), got {tuple(logits.shape)}")
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)

        target_oh, valid_mask = _one_hot(target, num_classes, ignore_index=self.ignore_index)

        if not self.cfg.include_background and num_classes > 1:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        if valid_mask is not None:
            probs = probs * valid_mask
            target_oh = target_oh * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dims)
        denom = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.cfg.smooth) / (denom + self.cfg.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
        ignore_index: Optional[int] = None,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice = DiceLoss(
            include_background=include_background,
            ignore_index=ignore_index,
            smooth=smooth,
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index is not None else -100)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight:
            loss = loss + self.ce_weight * self.ce(logits, target.long())
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss

