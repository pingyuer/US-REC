"""Dual-path loss: Dense (Δ=1) + Sparse (Δ=k) SE(3) pose losses.

No multi-interval auxiliary losses — each branch receives dedicated supervision
from its own ground-truth local transforms.

Loss improvements for stable training:
- Rotation: **chordal distance** (Frobenius norm ||R_pred - R_gt||_F) instead
  of geodesic (acos).  Chordal distance is smooth everywhere (no acos gradient
  singularity), and is monotonically related to geodesic distance for angles
  in [0, π].  For small angles, chordal ≈ geodesic.
- Translation: **Smooth L1 (Huber)** instead of MSE.  Reduces sensitivity to
  outlier frames that would cause large gradient spikes with squared error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from metrics.compose import local_from_global
from models.losses.pose_loss import (  # noqa: F401 — re-export for backward compat
    chordal_rotation_loss,
    se3_chordal_loss as _se3_pose_loss,
)


# ─── Public API ──────────────────────────────────────────────────────────────

def dual_loss(
    pred_local_T: torch.Tensor,
    pred_sparse_T: torch.Tensor,
    gt_global_T: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined dense + sparse loss.

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4)
        Dense Δ=1 predicted local transforms (frame 0 = I).
    pred_sparse_T : (B, M, 4, 4)
        Sparse Δ=k predicted local transforms among anchor frames (anchor 0 = I).
    gt_global_T : (B, T, 4, 4)
        Ground-truth global transforms (frame 0 = I).
    anchor_indices : (M,)
        LongTensor of anchor frame indices in original sequence.
    rot_weight, trans_weight : float
        Weights for rotation / translation components.
    dense_weight, sparse_weight : float
        Weights for the two branches.

    Returns
    -------
    loss : scalar
    breakdown : dict with dense_loss, sparse_loss, and per-component details.
    """
    # ── Dense loss (Δ=1) ────────────────────────────────────────────
    gt_local_T = local_from_global(gt_global_T)  # (B, T, 4, 4)
    # Mask out frame 0 (identity)
    dense_mask = torch.zeros(pred_local_T.shape[:2], dtype=torch.bool,
                             device=pred_local_T.device)
    dense_mask[:, 1:] = True
    dense_loss, dense_bd = _se3_pose_loss(
        pred_local_T, gt_local_T, mask=dense_mask,
        rot_weight=rot_weight, trans_weight=trans_weight,
    )

    # ── Sparse loss (Δ=k) ──────────────────────────────────────────
    # Build GT sparse locals from anchor globals
    gt_anchor_global = gt_global_T[:, anchor_indices]  # (B, M, 4, 4)
    gt_sparse_local = local_from_global(gt_anchor_global)  # (B, M, 4, 4)
    # Mask out anchor 0 (identity)
    sparse_mask = torch.zeros(pred_sparse_T.shape[:2], dtype=torch.bool,
                              device=pred_sparse_T.device)
    sparse_mask[:, 1:] = True
    sparse_loss, sparse_bd = _se3_pose_loss(
        pred_sparse_T, gt_sparse_local, mask=sparse_mask,
        rot_weight=rot_weight, trans_weight=trans_weight,
    )

    # ── Combined ────────────────────────────────────────────────────
    total = dense_weight * dense_loss + sparse_weight * sparse_loss

    breakdown = {
        "dense_loss": float(dense_loss.detach()),
        "sparse_loss": float(sparse_loss.detach()),
        "dense_rot": dense_bd["rot_loss"],
        "dense_trans": dense_bd["trans_loss"],
        "sparse_rot": sparse_bd["rot_loss"],
        "sparse_trans": sparse_bd["trans_loss"],
    }
    return total, breakdown
