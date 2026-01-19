from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class ConfusionMatrix:
    num_classes: int
    ignore_index: Optional[int] = None

    def __post_init__(self) -> None:
        self.num_classes = int(self.num_classes)
        self.reset()

    def reset(self) -> None:
        self._confmat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update confusion matrix.

        - preds: (B,H,W) class indices or (B,C,H,W) logits/probs
        - target: (B,H,W) class indices
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        if preds.dim() != 3 or target.dim() != 3:
            raise ValueError(f"Expected preds/target (B,H,W), got {tuple(preds.shape)} and {tuple(target.shape)}")

        preds = preds.detach().to(torch.int64).view(-1)
        target = target.detach().to(torch.int64).view(-1)

        if self.ignore_index is not None:
            mask = target != int(self.ignore_index)
            preds = preds[mask]
            target = target[mask]

        k = (target >= 0) & (target < self.num_classes)
        target = target[k]
        preds = preds[k]
        idx = target * self.num_classes + preds
        conf = torch.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self._confmat += conf.cpu()

    def compute(self, metrics: Iterable[str] = ("mIoU", "Dice", "Accuracy")) -> dict[str, float]:
        """
        Compute requested metrics from accumulated confusion matrix.
        """
        conf = self._confmat.to(torch.float64)
        tp = torch.diag(conf)
        fp = conf.sum(0) - tp
        fn = conf.sum(1) - tp
        denom = conf.sum()

        out: dict[str, float] = {}
        requested = {m.lower() for m in metrics}

        if "accuracy" in requested:
            out["Accuracy"] = float((tp.sum() / denom).item()) if denom > 0 else 0.0

        if "miou" in requested or "iou" in requested:
            iou = tp / (tp + fp + fn + 1e-12)
            out["mIoU"] = float(iou.mean().item())

        if "dice" in requested:
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-12)
            out["Dice"] = float(dice.mean().item())

        return out

