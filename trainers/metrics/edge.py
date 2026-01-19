from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:  # optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def edge_f1score(output, target, cls: int, *, threshold: float = 0.5) -> float:
    """
    Edge F1 score for (B, C, H, W) binary masks/logits.

    Notes:
    - Requires OpenCV. If unavailable, raises RuntimeError.
    - This is intentionally kept as a functional helper for legacy scripts.
    """
    if cv2 is None:
        raise RuntimeError("edge_f1score requires opencv-python (cv2) installed")

    if torch.is_tensor(output):
        output = torch.sigmoid(output).detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()

    output = (output >= threshold).astype(np.uint8)
    f1_total = 0.0
    batch = len(output)

    for i in range(batch):
        for c in range(int(cls)):
            output_edge = cv2.Canny((output[i, c] * 255).squeeze().astype("uint8"), 100, 200)
            target_edge = cv2.Canny((target[i, c] * 255).squeeze().astype("uint8"), 100, 200)

            tp = int(((target_edge == 255) & (output_edge == 255)).sum())
            fp = int(((target_edge != 255) & (output_edge == 255)).sum())
            fn = int(((target_edge == 255) & (output_edge != 255)).sum())

            pre = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1_total += (2 * pre * rec) / (pre + rec + 1e-12)

    return float(f1_total / batch)

