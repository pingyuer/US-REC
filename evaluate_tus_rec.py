"""Evaluate TUS-REC metrics from saved predictions.

Usage:
    python evaluate_tus_rec.py --pred pred_T.pt --gt gt_T.pt
    python evaluate_tus_rec.py --pred pred_T.npy --gt gt_T.npy \
        --pred-vol pred_vol.pt --gt-vol gt_vol.pt
"""

import argparse
from typing import Optional

import numpy as np
import torch

from metrics import (
    translation_error,
    rotation_error,
    se3_error,
    cumulative_drift,
    loop_closure_error,
    ddf_rmse,
    ddf_mae,
    volume_ncc,
    volume_ssim,
    volume_dice,
)


def _load_tensor(path: str) -> torch.Tensor:
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu")
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path))
    raise ValueError(f"Unsupported file extension: {path}")


def _ensure_batched(T: torch.Tensor) -> torch.Tensor:
    if T.dim() == 3:
        return T.unsqueeze(0)
    return T


def _print_row(name: str, value: Optional[float]) -> None:
    if value is None:
        print(f"{name:24s}: n/a")
    else:
        print(f"{name:24s}: {value:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TUS-REC metrics")
    parser.add_argument("--pred", required=True, help="Predicted transforms (T,4,4) or (B,T,4,4)")
    parser.add_argument("--gt", required=True, help="GT transforms (T,4,4) or (B,T,4,4)")
    parser.add_argument("--pred-ddf", default=None, help="Predicted DDF tensor")
    parser.add_argument("--gt-ddf", default=None, help="GT DDF tensor")
    parser.add_argument("--pred-vol", default=None, help="Predicted volume tensor")
    parser.add_argument("--gt-vol", default=None, help="GT volume tensor")
    args = parser.parse_args()

    pred_T = _ensure_batched(_load_tensor(args.pred).float())
    gt_T = _ensure_batched(_load_tensor(args.gt).float())

    trans_err = translation_error(pred_T[..., :3, 3], gt_T[..., :3, 3]).mean().item()
    rot_err = rotation_error(pred_T[..., :3, :3], gt_T[..., :3, :3]).mean().item()
    se3_err = se3_error(pred_T, gt_T).mean().item()
    drift = cumulative_drift(pred_T, gt_T).mean().item()
    loop_err = loop_closure_error(pred_T, gt_T).mean().item()

    _print_row("translation_error", trans_err)
    _print_row("rotation_error", rot_err)
    _print_row("se3_error", se3_err)
    _print_row("cumulative_drift", drift)
    _print_row("loop_closure_error", loop_err)

    if args.pred_ddf and args.gt_ddf:
        pred_ddf = _load_tensor(args.pred_ddf).float()
        gt_ddf = _load_tensor(args.gt_ddf).float()
        _print_row("ddf_rmse", ddf_rmse(pred_ddf, gt_ddf).item())
        _print_row("ddf_mae", ddf_mae(pred_ddf, gt_ddf).item())
    else:
        _print_row("ddf_rmse", None)
        _print_row("ddf_mae", None)

    if args.pred_vol and args.gt_vol:
        pred_vol = _load_tensor(args.pred_vol).float()
        gt_vol = _load_tensor(args.gt_vol).float()
        _print_row("volume_ncc", volume_ncc(pred_vol, gt_vol).mean().item())
        _print_row("volume_ssim", volume_ssim(pred_vol, gt_vol).mean().item())
        _print_row("volume_dice", volume_dice(pred_vol, gt_vol).mean().item())
    else:
        _print_row("volume_ncc", None)
        _print_row("volume_ssim", None)
        _print_row("volume_dice", None)


if __name__ == "__main__":
    main()
