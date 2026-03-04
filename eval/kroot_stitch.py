"""K-root stitch: Long-anchor-based global + Short dense refinement + SE(3) interpolation.

The core function ``stitch_long_base_short_refine`` fuses predictions from
two independently-trained models (ShortTransformer and LongTransformer) into
a single globally-consistent trajectory.

Algorithm
---------
1. **Long pass**: run the LongTransformer on sparse anchor frames
   (stride *s*) with sliding windows of *k* tokens.  Accumulate
   predictions to get ``T_long_global[a]`` for every anchor ``a``.

2. **Short pass**: run the ShortTransformer on consecutive-frame
   windows (token length *k*, stride *k-overlap*) to get dense
   local transforms ``T_{i-1<-i}`` everywhere.

3. **Per-interval composition**: in each anchor interval ``[a, b)``,
   compose short-local transforms into a segment-global with the
   anchor start as origin:
   ``T_short_seg[i] = prod(T_short_local[a+1..i])``  → ``T_short_seg[a] = I``.

4. **Left alignment**: ``T_candidate[i] = T_long_global[a] @ T_short_seg[i]``.

5. **Two-endpoint SE(3) correction** (optional):
   ``C_right = T_long_global[b] @ inv(T_candidate[b])``
   Interpolate ``C(i)`` from ``I`` to ``C_right`` in Lie-algebra:
   ``log_C = t * log(C_right)``  →  ``T_fused[i] = exp(log_C) @ T_candidate[i]``.

6. **Last segment** (no right anchor): constant correction from left only.

Public API
----------
``stitch_long_base_short_refine(short_model, long_model, scan_frames, gt_global, k, s, ...)``
``stitch_from_predictions(short_local_T, long_local_T, anchor_indices, ...)``
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from metrics.compose import compose_global_from_local, local_from_global
from eval.dual_fusion import _se3_log, _se3_exp


# ---------------------------------------------------------------------------
# Core stitching (works on already-predicted transforms)
# ---------------------------------------------------------------------------

def stitch_from_predictions(
    short_local_T: torch.Tensor,
    long_anchor_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    T: int,
    *,
    enable_endpoint_interp: bool = True,
) -> torch.Tensor:
    """Stitch long anchor globals + short dense locals into fused globals.

    Parameters
    ----------
    short_local_T : (T, 4, 4)
        Dense Δ=1 local transforms (frame 0 = I).
    long_anchor_global : (M, 4, 4)
        Global transforms at anchor frames (anchor 0 = I).
    anchor_indices : (M,) LongTensor
        Frame indices of anchors.
    T : int
        Total frames.
    enable_endpoint_interp : bool
        Apply two-endpoint SE(3) correction interpolation.

    Returns
    -------
    fused_global : (T, 4, 4)
    """
    device = short_local_T.device
    dtype = short_local_T.dtype
    M = anchor_indices.shape[0]

    # 1. Compose short dense global from locals
    short_global = compose_global_from_local(short_local_T.unsqueeze(0)).squeeze(0)  # (T, 4, 4)

    fused = torch.zeros(T, 4, 4, device=device, dtype=dtype)
    fused[:, :, :] = torch.eye(4, device=device, dtype=dtype)

    for m_idx in range(M):
        a = int(anchor_indices[m_idx].item())
        b = int(anchor_indices[m_idx + 1].item()) if m_idx + 1 < M else T

        # T_long_global at anchor a
        T_long_a = long_anchor_global[m_idx]  # (4, 4)

        for i in range(a, b):
            # T_short_seg[i] relative to anchor a:
            # T_short_seg[i] = inv(short_global[a]) @ short_global[i]
            T_seg_i = torch.linalg.inv(short_global[a]) @ short_global[i]  # (4, 4)
            # T_candidate[i] = T_long_global[a] @ T_seg_i
            fused[i] = T_long_a @ T_seg_i

        # Two-endpoint correction
        if enable_endpoint_interp and m_idx + 1 < M:
            T_long_b = long_anchor_global[m_idx + 1]
            # C_right = T_long_global[b] @ inv(T_candidate[b])
            # But T_candidate[b] was computed in the NEXT segment's first step...
            # Actually b is the start of next segment; we need the value at b
            # from current segment's perspective.
            T_seg_b = torch.linalg.inv(short_global[a]) @ short_global[b if b < T else T - 1]
            T_candidate_b = T_long_a @ T_seg_b
            C_right = T_long_b @ torch.linalg.inv(T_candidate_b)

            # Interpolate C from I to C_right in SE(3) log space
            log_C = _se3_log(C_right.unsqueeze(0)).squeeze(0)  # (6,)

            interval_len = b - a
            if interval_len > 0:
                for i in range(a, b):
                    alpha = float(i - a) / max(1, interval_len)
                    log_C_interp = alpha * log_C
                    C_interp = _se3_exp(log_C_interp.unsqueeze(0)).squeeze(0)  # (4, 4)
                    fused[i] = C_interp @ fused[i]

    return fused


# ---------------------------------------------------------------------------
# Model-based stitching (runs inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def stitch_long_base_short_refine(
    short_model: torch.nn.Module,
    long_model: torch.nn.Module,
    scan_frames: torch.Tensor,
    k: int,
    s: int,
    device: torch.device,
    *,
    short_overlap: int = 8,
    long_window_stride: Optional[int] = None,
    enable_endpoint_interp: bool = True,
) -> Dict[str, torch.Tensor]:
    """Full stitch pipeline: run both models, then fuse.

    Parameters
    ----------
    short_model : trained ShortTransformer
    long_model : trained LongTransformer
    scan_frames : (T_total, H, W) full scan frames
    k : int — token window length for both models
    s : int — sparse stride for long model
    device : torch device
    short_overlap : int — overlap for short sliding window
    long_window_stride : int or None — stride in token space for long sliding window
    enable_endpoint_interp : bool

    Returns
    -------
    dict with:
        "fused_global"  : (T, 4, 4)
        "short_global"  : (T, 4, 4)  — short-only accumulated globals
        "long_global"   : (M, 4, 4)  — long anchor globals
        "anchor_indices": (M,)
        "short_local"   : (T, 4, 4)
        "long_local"    : (M, 4, 4)
    """
    short_model.eval()
    long_model.eval()
    T_total = scan_frames.shape[0]

    if long_window_stride is None:
        long_window_stride = max(1, k - 1)

    # ── Normalise frames ─────────────────────────────────────────
    frames_norm = scan_frames.float()
    if frames_norm.max() > 1.1:
        frames_norm = frames_norm / 255.0

    # ── Step 1: Short model — dense local transforms ─────────────
    short_local = _run_short_sliding_window(
        short_model, frames_norm, k, short_overlap, device,
    )  # (T, 4, 4)
    short_global = compose_global_from_local(short_local.unsqueeze(0)).squeeze(0)

    # ── Step 2: Long model — sparse anchor transforms ────────────
    anchor_indices, long_local = _run_long_sliding_window(
        long_model, frames_norm, k, s, long_window_stride, device,
    )  # anchor_indices (M,), long_local (M, 4, 4)
    long_global = compose_global_from_local(long_local.unsqueeze(0)).squeeze(0)

    # ── Step 3: Stitch ───────────────────────────────────────────
    fused_global = stitch_from_predictions(
        short_local_T=short_local,
        long_anchor_global=long_global,
        anchor_indices=anchor_indices,
        T=T_total,
        enable_endpoint_interp=enable_endpoint_interp,
    )

    return {
        "fused_global": fused_global,
        "short_global": short_global,
        "long_global": long_global,
        "anchor_indices": anchor_indices,
        "short_local": short_local,
        "long_local": long_local,
    }


# ---------------------------------------------------------------------------
# Sliding window inference helpers
# ---------------------------------------------------------------------------

def _run_short_sliding_window(
    model: torch.nn.Module,
    frames: torch.Tensor,
    k: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    """Run short model with sliding window, return (T, 4, 4) local transforms.

    For each window [start, start+k), extract predicted locals and stitch
    into a single local transform tensor via last-write-wins (or averaging
    in overlap areas → simplified to last-write for robustness).
    """
    T = frames.shape[0]
    local_T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(T, -1, -1).clone()

    stride = max(1, k - overlap)
    starts = list(range(0, max(1, T - k + 1), stride))
    # Ensure we cover the end
    if starts and starts[-1] + k < T and T >= k:
        starts.append(T - k)

    for start in starts:
        end = min(start + k, T)
        chunk = frames[start:end].unsqueeze(0).to(device)  # (1, W, H, W_img)
        out = model(chunk)
        pred_local = out["pred_local_T"].squeeze(0)  # (W, 4, 4)

        # Write predicted locals into the global array
        # local_T[i] = T_{i-1 <- i}, so pred_local[0] = I (skip)
        for j in range(1, pred_local.shape[0]):
            frame_idx = start + j
            if frame_idx < T:
                local_T[frame_idx] = pred_local[j].to(device)

    return local_T


def _run_long_sliding_window(
    model: torch.nn.Module,
    frames: torch.Tensor,
    k: int,
    s: int,
    window_stride_tokens: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run long model with sliding window on sparse anchors.

    Returns anchor_indices (M,) and long_local (M, 4, 4).
    """
    T = frames.shape[0]

    # Build full anchor index list
    all_anchors = list(range(0, T, s))
    M = len(all_anchors)
    anchor_indices = torch.tensor(all_anchors, dtype=torch.long, device=device)

    # Allocate local transforms (anchor domain)
    long_local = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(M, -1, -1).clone()

    if M <= 1:
        return anchor_indices, long_local

    # Sliding window in anchor-token space
    stride_t = max(1, window_stride_tokens)
    starts_t = list(range(0, max(1, M - k + 1), stride_t))
    if starts_t and starts_t[-1] + k < M and M >= k:
        starts_t.append(M - k)
    # If M < k, use a single window of whatever we have
    if M < k:
        starts_t = [0]

    for t_start in starts_t:
        t_end = min(t_start + k, M)
        window_anchors = all_anchors[t_start:t_end]  # frame indices for this window
        if not window_anchors:
            continue

        # Gather sparse frames
        idx_tensor = torch.tensor(window_anchors, dtype=torch.long)
        chunk = frames[idx_tensor].unsqueeze(0).to(device)  # (1, W_tok, H, W_img)
        out = model(chunk)
        pred_local = out["pred_local_T"].squeeze(0)  # (W_tok, 4, 4)

        # Write into the anchor local array
        for j in range(1, pred_local.shape[0]):
            anchor_pos = t_start + j
            if anchor_pos < M:
                long_local[anchor_pos] = pred_local[j].to(device)

    return anchor_indices, long_local


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_stitch_metrics(
    fused_global: torch.Tensor,
    short_global: torch.Tensor,
    long_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    gt_global: torch.Tensor,
    *,
    tform_calib: Optional[torch.Tensor] = None,
    image_size: Tuple[int, int] = (480, 640),
    frames: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute GPE/LPE for fused, short_only, and long_only predictions.

    Returns a flat dict of metrics.
    """
    T = gt_global.shape[0]
    device = gt_global.device
    metrics: Dict[str, float] = {}

    gt_local = local_from_global(gt_global)

    # ── Per-variant SE(3) translation metrics ────────────────────
    for name, pred_g in [("fused", fused_global), ("short_only", short_global)]:
        pred_l = local_from_global(pred_g)
        lpe = (pred_l[1:, :3, 3] - gt_local[1:, :3, 3]).norm(dim=-1).mean()
        gpe = (pred_g[:, :3, 3] - gt_global[:, :3, 3]).norm(dim=-1).mean()
        drift = (pred_g[-1, :3, 3] - gt_global[-1, :3, 3]).norm()
        metrics[f"gpe_mm_{name}"] = float(gpe.item())
        metrics[f"lpe_mm_{name}"] = float(lpe.item())
        metrics[f"drift_last_mm_{name}"] = float(drift.item())

    # Long-only: interpolate anchor globals into full sequence (piecewise constant)
    M = anchor_indices.shape[0]
    long_full = torch.zeros(T, 4, 4, device=device, dtype=gt_global.dtype)
    for m in range(M):
        a = int(anchor_indices[m].item())
        b = int(anchor_indices[m + 1].item()) if m + 1 < M else T
        for i in range(a, b):
            long_full[i] = long_global[m]

    long_l = local_from_global(long_full)
    metrics["gpe_mm_long_only"] = float(
        (long_full[:, :3, 3] - gt_global[:, :3, 3]).norm(dim=-1).mean().item()
    )
    metrics["lpe_mm_long_only"] = float(
        (long_l[1:, :3, 3] - gt_local[1:, :3, 3]).norm(dim=-1).mean().item()
    )
    metrics["drift_last_mm_long_only"] = float(
        (long_full[-1, :3, 3] - gt_global[-1, :3, 3]).norm().item()
    )

    # ── Official TUS-REC DDF metrics (if calibration available) ──
    if tform_calib is not None and frames is not None:
        try:
            from trainers.metrics.tusrec import compute_tusrec_metrics as _tusrec
            for name, pred_g in [("fused", fused_global), ("short_only", short_global)]:
                tr = _tusrec(
                    frames=frames,
                    gt_transforms=gt_global,
                    pred_transforms=pred_g,
                    calib={"tform_calib": tform_calib},
                    compute_scores=True,
                )
                for k_m, v_m in tr.items():
                    if v_m is not None:
                        metrics[f"tusrec_{k_m}_{name}"] = float(v_m)
        except Exception as exc:
            warnings.warn(f"[kroot_stitch] TUS-REC metrics failed: {exc}")

    metrics["num_frames"] = int(T)
    metrics["num_anchors"] = int(M)
    return metrics


# ---------------------------------------------------------------------------
# Debug CSV export
# ---------------------------------------------------------------------------

def export_debug_csv(
    scan_id: str,
    fused_global: torch.Tensor,
    short_global: torch.Tensor,
    long_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    gt_global: torch.Tensor,
    out_dir: str,
) -> str:
    """Export per-frame debug CSV with ||t||_mm and rot_deg for each variant.

    Returns path to the written CSV file.
    """
    T = gt_global.shape[0]
    M = anchor_indices.shape[0]

    # Expand long to full frames (piecewise constant for intermediate frames)
    long_full = torch.zeros(T, 4, 4, dtype=gt_global.dtype, device=gt_global.device)
    for m in range(M):
        a = int(anchor_indices[m].item())
        b = int(anchor_indices[m + 1].item()) if m + 1 < M else T
        for i in range(a, b):
            long_full[i] = long_global[m]

    rows = []
    for i in range(T):
        row = {"frame": i}
        for name, pred_g in [("fused", fused_global), ("short_only", short_global.to(gt_global.device)), ("long_only", long_full), ("gt", gt_global)]:
            t_err = (pred_g[i, :3, 3] - gt_global[i, :3, 3]).norm().item() if name != "gt" else 0.0
            # Translation magnitude for each variant
            t_mag = pred_g[i, :3, 3].norm().item()
            # Rough rotation angle (Frobenius of R - I → approximate angle)
            R = pred_g[i, :3, :3]
            trace_val = R[0, 0] + R[1, 1] + R[2, 2]
            cos_a = max(-1.0, min(1.0, (float(trace_val) - 1.0) / 2.0))
            rot_deg = math.degrees(math.acos(cos_a))
            row[f"t_err_mm_{name}"] = f"{t_err:.4f}" if name != "gt" else "0.0"
            row[f"t_mag_mm_{name}"] = f"{t_mag:.4f}"
            row[f"rot_deg_{name}"] = f"{rot_deg:.4f}"
        rows.append(row)

    safe_id = scan_id.replace("/", "_").replace("\\", "_")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"stitch_debug_{safe_id}.csv"

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)
