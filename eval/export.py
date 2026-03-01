"""Export evaluation results as per-scan JSON and optional NPZ.

Programmatic::

    from eval.export import export_results
    export_results(metrics, out_dir="eval_output/", save_json=True, save_npz=True)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def export_results(
    metrics: dict[str, Any],
    *,
    out_dir: Union[str, Path],
    save_json: bool = True,
    save_npz: bool = False,
    pred_transforms: Optional[Any] = None,
    gt_transforms: Optional[Any] = None,
) -> dict[str, str]:
    """Export evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Metrics dict as returned by ``RecEvaluator.run()``.
        May contain ``tusrec_per_scan`` (list of per-scan dicts).
    out_dir : path
        Root output directory.
    save_json : bool
        Save per-scan JSON and summary JSON.
    save_npz : bool
        Save pred/gt transforms as .npz for reproducibility.
    pred_transforms, gt_transforms : optional array-like
        If provided and save_npz is True, save to npz.

    Returns
    -------
    dict mapping output type to file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if save_json:
        # Per-scan JSON
        per_scan = metrics.get("tusrec_per_scan")
        if per_scan:
            per_scan_path = out_dir / "tusrec_per_scan.json"
            with open(per_scan_path, "w") as f:
                json.dump(per_scan, f, indent=2, default=_json_serializable)
            written["per_scan_json"] = str(per_scan_path)

        # Summary JSON (scalar metrics only)
        summary = {
            k: v
            for k, v in metrics.items()
            if k != "tusrec_per_scan" and isinstance(v, (int, float))
        }
        summary_path = out_dir / "eval_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_serializable)
        written["summary_json"] = str(summary_path)

    if save_npz and pred_transforms is not None and gt_transforms is not None:
        npz_path = out_dir / "transforms.npz"
        np.savez_compressed(
            npz_path,
            pred=np.asarray(pred_transforms),
            gt=np.asarray(gt_transforms),
        )
        written["npz"] = str(npz_path)

    return written


def _json_serializable(obj):
    """Fallback serialiser for numpy types."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
