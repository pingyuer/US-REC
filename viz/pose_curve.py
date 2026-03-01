"""Export per-scan pose curves (translation norm & rotation angle vs frame).

Usage (standalone)::

    python -m viz.pose_curve --transforms pred_global.pt --out-dir results/

Programmatic::

    from viz.pose_curve import export_pose_curve
    export_pose_curve(global_T, out_dir="results/scan01", scan_id="scan01")
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Union

import torch


def _rotation_angle_deg(R: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation angle (degrees) from a batch of 3x3 rotation matrices."""
    # R: (..., 3, 3)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / 3.141592653589793)


def export_pose_curve(
    global_T: torch.Tensor,
    *,
    out_dir: Union[str, Path],
    scan_id: str = "scan",
    save_csv: bool = True,
    save_png: bool = True,
) -> dict[str, str]:
    """Write translation-norm and rotation-angle vs frame index.

    Parameters
    ----------
    global_T : (T, 4, 4)
        Global transforms ``T_{0<-i}``.
    out_dir : path
        Directory to write outputs.
    scan_id : str
        Prefix for file names.
    save_csv, save_png : bool
        Which formats to save.

    Returns
    -------
    dict with keys ``"csv"`` and / or ``"png"`` pointing to written files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = global_T.shape[0]
    trans_norm = torch.linalg.norm(global_T[:, :3, 3], dim=-1).cpu().tolist()
    rot_deg = _rotation_angle_deg(global_T[:, :3, :3]).cpu().tolist()

    written: dict[str, str] = {}

    # ---- CSV ----
    if save_csv:
        csv_path = out_dir / f"{scan_id}_pose_curve.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "translation_norm_mm", "rotation_deg"])
            for i in range(T):
                writer.writerow([i, f"{trans_norm[i]:.6f}", f"{rot_deg[i]:.6f}"])
        written["csv"] = str(csv_path)

    # ---- PNG ----
    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            frames = list(range(T))
            ax1.plot(frames, trans_norm, "b-")
            ax1.set_ylabel("Translation norm (mm)")
            ax1.set_title(f"{scan_id} — pose curve")
            ax1.grid(True, alpha=0.3)

            ax2.plot(frames, rot_deg, "r-")
            ax2.set_ylabel("Rotation angle (deg)")
            ax2.set_xlabel("Frame index")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            png_path = out_dir / f"{scan_id}_pose_curve.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            written["png"] = str(png_path)
        except ImportError:
            pass  # matplotlib not available

    return written
