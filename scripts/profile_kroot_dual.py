#!/usr/bin/env python
"""Smoke profiler for the kroot_dual_joint training pipeline.

Runs 50 training steps with synthetic data and prints per-step timing:
  - t_batch:   time to fetch one batch from the DataLoader
  - t_fwd_bwd: time for forward + backward on both branches
  - GPU util:  optional (if pynvml is available)

Usage:
    python scripts/profile_kroot_dual.py [--steps 50] [--device cuda]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is importable
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.datasets.kroot_dual_joint import SyntheticJointKRootDualDataset
from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import local_from_global


# ---------------------------------------------------------------------------
# Optional GPU-util helper
# ---------------------------------------------------------------------------
def _try_gpu_util() -> str:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return f"{util.gpu}%"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Profile kroot_dual_joint pipeline")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--frames-per-scan", type=int, default=512)
    parser.add_argument("--num-scans", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    k = args.k

    print(f"[profile] device={device}  k={k}  steps={args.steps}  "
          f"bs={args.batch_size}  workers={args.num_workers}")
    print(f"[profile] synthetic: {args.num_scans} scans × {args.frames_per_scan} frames "
          f"@ {args.height}×{args.width}")

    # ── Dataset & DataLoader ────────────────────────────────────────
    ds = SyntheticJointKRootDualDataset(
        num_scans=args.num_scans,
        frames_per_scan=args.frames_per_scan,
        height=args.height,
        width=args.width,
        k=k,
        mode="train",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # ── Models ──────────────────────────────────────────────────────
    short_model = LongSeqPoseModel(
        backbone="efficientnet_b0", in_channels=1, token_dim=128,
        n_heads=4, n_layers=2, dim_feedforward=256, window_size=k,
        dropout=0.0, rotation_rep="rot6d", pretrained_backbone=False,
    ).to(device)
    long_model = LongSeqPoseModel(
        backbone="efficientnet_b0", in_channels=1, token_dim=128,
        n_heads=4, n_layers=2, dim_feedforward=256, window_size=k,
        dropout=0.0, rotation_rep="rot6d", pretrained_backbone=False,
    ).to(device)

    short_opt = torch.optim.AdamW(short_model.parameters(), lr=1e-4)
    long_opt = torch.optim.AdamW(long_model.parameters(), lr=1e-4)

    # ── Warm-up (one step to JIT-compile / allocate) ────────────────
    print("[profile] warming up …")
    it = iter(loader)
    batch = next(it)
    sf = batch["short"]["frames"].to(device, non_blocking=True).float() / 255.0
    out = short_model(sf)
    out["pred_local_T"].sum().backward()
    short_opt.step(); short_opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Profile loop ────────────────────────────────────────────────
    print(f"\n{'step':>5s}  {'t_batch':>8s}  {'t_fwd_bwd':>10s}  {'gpu_util':>9s}")
    print("-" * 42)

    times_batch: list[float] = []
    times_fwd: list[float] = []

    step = 0
    it = iter(loader)
    while step < args.steps:
        # ── Batch fetch ────────────
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t_batch = time.perf_counter() - t0

        # ── Forward + backward ─────
        t1 = time.perf_counter()

        # Short
        sf = batch["short"]["frames"].to(device, non_blocking=True)
        sg = batch["short"]["gt_global_T"].to(device, non_blocking=True)
        if sf.dtype != torch.float32:
            sf = sf.float()
        if sf.max() > 1.1:
            sf = sf / 255.0
        out_s = short_model(sf)
        gt_local_s = local_from_global(sg)
        pred_R = out_s["pred_local_T"][:, 1:, :3, :3]
        gt_R = gt_local_s[:, 1:, :3, :3]
        loss_s = chordal_rotation_loss(pred_R, gt_R)
        loss_s.backward()

        # Long
        lf = batch["long"]["frames"].to(device, non_blocking=True)
        lg = batch["long"]["gt_global_T"].to(device, non_blocking=True)
        if lf.dtype != torch.float32:
            lf = lf.float()
        if lf.max() > 1.1:
            lf = lf / 255.0
        out_l = long_model(lf)
        gt_local_l = local_from_global(lg)
        pred_R_l = out_l["pred_local_T"][:, 1:, :3, :3]
        gt_R_l = gt_local_l[:, 1:, :3, :3]
        loss_l = chordal_rotation_loss(pred_R_l, gt_R_l)
        loss_l.backward()

        short_opt.step(); short_opt.zero_grad()
        long_opt.step(); long_opt.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t1

        gpu_util = _try_gpu_util()

        times_batch.append(t_batch)
        times_fwd.append(t_fwd)

        step += 1
        print(f"{step:5d}  {t_batch:8.3f}s  {t_fwd:10.3f}s  {gpu_util:>9s}")

    # ── Summary ─────────────────────────────────────────────────────
    if times_batch:
        avg_b = sum(times_batch) / len(times_batch)
        avg_f = sum(times_fwd) / len(times_fwd)
        p50_b = sorted(times_batch)[len(times_batch) // 2]
        p50_f = sorted(times_fwd)[len(times_fwd) // 2]
        print(f"\n[summary] t_batch  avg={avg_b:.3f}s  p50={p50_b:.3f}s")
        print(f"[summary] t_fwd    avg={avg_f:.3f}s  p50={p50_f:.3f}s")
        print(f"[summary] total    avg={avg_b + avg_f:.3f}s/step")


if __name__ == "__main__":
    main()
