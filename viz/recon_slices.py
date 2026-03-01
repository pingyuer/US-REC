"""Export reconstruction volume slices for qualitative evaluation.

Saves axial / sagittal / coronal centre-slice images (PNG) suitable
for insertion into a paper or presentation.

Usage (standalone)::

    python -m viz.recon_slices --volume pred_vol.pt --out-dir results/

Programmatic::

    from viz.recon_slices import export_recon_slices
    export_recon_slices(volume, out_dir="results/", scan_id="scan01")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch


def export_recon_slices(
    volume: torch.Tensor,
    *,
    out_dir: Union[str, Path],
    scan_id: str = "scan",
    slice_indices: Optional[dict[str, int]] = None,
    planes: Sequence[str] = ("axial", "sagittal", "coronal"),
    gt_volume: Optional[torch.Tensor] = None,
) -> dict[str, str]:
    """Save centre-slice images from a 3-D volume.

    Parameters
    ----------
    volume : (D, H, W) or (1, D, H, W) float tensor
    out_dir : output directory
    scan_id : prefix for filenames
    slice_indices : optional dict mapping plane name -> index
    planes : which planes to extract
    gt_volume : optional ground-truth volume for side-by-side comparison

    Returns
    -------
    dict mapping plane name to PNG path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if volume.ndim == 4:
        volume = volume.squeeze(0)
    assert volume.ndim == 3, f"Expected 3-D volume, got shape {volume.shape}"
    D, H, W = volume.shape

    if gt_volume is not None:
        if gt_volume.ndim == 4:
            gt_volume = gt_volume.squeeze(0)

    default_indices = {
        "axial": D // 2,
        "sagittal": H // 2,
        "coronal": W // 2,
    }
    if slice_indices:
        default_indices.update(slice_indices)

    written: dict[str, str] = {}

    def _extract(vol: torch.Tensor, plane: str, idx: int) -> torch.Tensor:
        if plane == "axial":
            return vol[idx, :, :]
        elif plane == "sagittal":
            return vol[:, idx, :]
        elif plane == "coronal":
            return vol[:, :, idx]
        raise ValueError(f"Unknown plane {plane}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Without matplotlib, save raw tensors as .pt
        for plane in planes:
            idx = default_indices.get(plane, 0)
            sl = _extract(volume, plane, idx).cpu()
            pt_path = out_dir / f"{scan_id}_{plane}_slice.pt"
            torch.save(sl, pt_path)
            written[plane] = str(pt_path)
        return written

    for plane in planes:
        idx = default_indices.get(plane, 0)
        pred_slice = _extract(volume, plane, idx).cpu().numpy()

        if gt_volume is not None:
            gt_slice = _extract(gt_volume, plane, idx).cpu().numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(gt_slice, cmap="gray")
            ax1.set_title(f"GT — {plane} (idx={idx})")
            ax1.axis("off")
            ax2.imshow(pred_slice, cmap="gray")
            ax2.set_title(f"Pred — {plane} (idx={idx})")
            ax2.axis("off")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(pred_slice, cmap="gray")
            ax.set_title(f"{scan_id} — {plane} (idx={idx})")
            ax.axis("off")

        fig.tight_layout()
        png_path = out_dir / f"{scan_id}_{plane}_slice.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        written[plane] = str(png_path)

    return written
