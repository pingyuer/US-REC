"""Export GPE-vs-frame drift curves.

These curves help explain why LPE may be small while GPE is large
(drift accumulates over frames).

Usage (standalone)::

    python -m viz.drift_curve \
        --pred pred_global.pt --gt gt_global.pt \
        --calib calib.csv --out-dir results/

Programmatic::

    from viz.drift_curve import export_drift_curve
    export_drift_curve(pred_global, gt_global, tform_calib, image_points,
                       out_dir="results/", scan_id="scan01")
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import torch

from metrics.compose import local_from_global


def _per_frame_point_error(
    gt_global: torch.Tensor,
    pred_global: torch.Tensor,
    tform_calib: torch.Tensor,
    image_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return mean 3-D point error per frame (mm).  Shape ``(T,)``."""
    T = gt_global.shape[0]
    device = gt_global.device

    # Ensure all tensors are on the same device as gt_global
    tform_calib = tform_calib.to(device=device, dtype=gt_global.dtype)

    if image_points is not None:
        pts = image_points.to(device=device, dtype=gt_global.dtype)
    else:
        # Fall back to a sparse 4-corner grid
        pts = torch.tensor(
            [[0, 0, 99, 99], [0, 99, 0, 99], [0, 0, 0, 0], [1, 1, 1, 1]],
            device=device,
            dtype=gt_global.dtype,
        )

    pts_tool = torch.matmul(tform_calib, pts)  # (4, K)
    gt_pts = torch.matmul(gt_global, pts_tool)[:, :3, :]  # (T, 3, K)
    pred_pts = torch.matmul(pred_global, pts_tool)[:, :3, :]
    per_frame = torch.linalg.norm(pred_pts - gt_pts, dim=1).mean(dim=1)  # (T,)
    return per_frame


def export_drift_curve(
    pred_global: torch.Tensor,
    gt_global: torch.Tensor,
    tform_calib: torch.Tensor,
    image_points: Optional[torch.Tensor] = None,
    *,
    out_dir: Union[str, Path],
    scan_id: str = "scan",
    save_csv: bool = True,
    save_png: bool = True,
) -> dict[str, str]:
    """Write GPE-per-frame drift curve.

    Returns dict mapping format name to file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpe_per_frame = _per_frame_point_error(
        gt_global, pred_global, tform_calib, image_points
    ).cpu().tolist()

    # Also compute LPE per frame for comparison
    gt_local = local_from_global(gt_global)
    pred_local = local_from_global(pred_global)
    lpe_per_frame = _per_frame_point_error(
        gt_local, pred_local, tform_calib, image_points
    ).cpu().tolist()

    T = len(gpe_per_frame)
    written: dict[str, str] = {}

    if save_csv:
        csv_path = out_dir / f"{scan_id}_drift_curve.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "GPE_mm", "LPE_mm"])
            for i in range(T):
                writer.writerow([i, f"{gpe_per_frame[i]:.6f}", f"{lpe_per_frame[i]:.6f}"])
        written["csv"] = str(csv_path)

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            frames = list(range(T))
            ax.plot(frames, gpe_per_frame, "b-", label="GPE (global)")
            ax.plot(frames, lpe_per_frame, "r--", alpha=0.7, label="LPE (local)")
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Point error (mm)")
            ax.set_title(f"{scan_id} — drift curve (GPE vs LPE)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            png_path = out_dir / f"{scan_id}_drift_curve.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            written["png"] = str(png_path)
        except ImportError:
            pass

    return written
