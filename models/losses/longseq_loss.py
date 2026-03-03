"""Multi-interval auxiliary loss + consistency constraint for long-sequence pose.

This module implements:
    1. **Main local loss** (Δ=1): L_local — standard SE(3) pose loss on T_{i-1←i}
    2. **Auxiliary losses** (Δ∈{2,4,8,16}): L_aux — multi-interval transform supervision
       using ground-truth  gt_T_{i-Δ←i} = inv(gt_global[i-Δ]) @ gt_global[i] with masking.
    3. **Consistency constraint**: L_consist — cycle consistency
       T̂_{i-2←i} ≈ T̂_{i-2←i-1} @ T̂_{i-1←i}
    4. **Sequence loss combiner**: weighted sum with configurable weights.

References:
    Liqi/TBME: multi-interval auxiliary supervision reduces drift (GPE).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from utils.rotation_loss import geodesic_loss


# ─── helpers ─────────────────────────────────────────────────────────────────

def _se3_pose_loss(
    pred_T: torch.Tensor,
    gt_T: torch.Tensor,
    mask: torch.Tensor | None = None,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SE(3) pose loss (geodesic rotation + L2 translation).

    Parameters
    ----------
    pred_T, gt_T : (B, T, 4, 4) or (N, 4, 4)
    mask : optional boolean tensor of valid positions (same leading dims)
    rot_weight, trans_weight : scalar weights

    Returns
    -------
    total : scalar loss
    breakdown : dict with rot_loss, trans_loss floats
    """
    R_pred = pred_T[..., :3, :3]
    R_gt = gt_T[..., :3, :3]
    t_pred = pred_T[..., :3, 3]
    t_gt = gt_T[..., :3, 3]

    if mask is not None:
        # Flatten masked entries
        R_pred = R_pred[mask]
        R_gt = R_gt[mask]
        t_pred = t_pred[mask]
        t_gt = t_gt[mask]

    if R_pred.numel() == 0:
        zero = torch.tensor(0.0, device=pred_T.device, dtype=pred_T.dtype)
        return zero, {"rot_loss": 0.0, "trans_loss": 0.0}

    rot_l = geodesic_loss(R_pred, R_gt)
    trans_l = F.mse_loss(t_pred, t_gt)

    total = rot_weight * rot_l + trans_weight * trans_l
    breakdown = {
        "rot_loss": float(rot_l.detach().item()),
        "trans_loss": float(trans_l.detach().item()),
    }
    return total, breakdown


# ─── Ground-truth multi-interval transforms ─────────────────────────────────

def gt_interval_transform(
    gt_global_T: torch.Tensor,
    delta: int,
) -> torch.Tensor:
    """Compute ground-truth interval transform T_{i-Δ←i} from global transforms.

    gt_T_{i-delta <- i} = inv(gt_global[i-delta]) @ gt_global[i]

    Parameters
    ----------
    gt_global_T : (B, T, 4, 4)  with gt_global_T[:, 0] = I
    delta : int  interval size

    Returns
    -------
    interval_T : (B, T, 4, 4)  where positions i < delta are filled with I
    """
    B, T, _, _ = gt_global_T.shape
    device, dtype = gt_global_T.device, gt_global_T.dtype
    eye = torch.eye(4, device=device, dtype=dtype)

    if delta >= T:
        return eye.unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4).clone()

    # inv(global[i-delta]) @ global[i]  for i = delta..T-1
    gt_ref = gt_global_T[:, :T - delta]   # (B, T-delta, 4, 4) — global[i-delta]
    gt_cur = gt_global_T[:, delta:]        # (B, T-delta, 4, 4) — global[i]
    inv_ref = torch.linalg.inv(gt_ref)
    interval = inv_ref @ gt_cur            # (B, T-delta, 4, 4)

    # Pad with identity for i < delta
    pad = eye.expand(B, delta, 4, 4).clone()
    return torch.cat([pad, interval], dim=1)  # (B, T, 4, 4)


# ─── Main API ────────────────────────────────────────────────────────────────

def local_pose_loss(
    pred_local_T: torch.Tensor,
    gt_global_T: torch.Tensor,
    *,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Main loss (Δ=1): local frame-to-frame pose error.

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4)  with pred_local_T[:, 0] = I
    gt_global_T  : (B, T, 4, 4)  ground-truth global transforms

    Returns
    -------
    loss : scalar
    breakdown : dict with rot_loss, trans_loss
    """
    from metrics.compose import local_from_global

    gt_local_T = local_from_global(gt_global_T)  # (B, T, 4, 4)
    # Only compute on i >= 1 (frame 0 is identity, skip it)
    mask = torch.zeros(pred_local_T.shape[:2], dtype=torch.bool,
                       device=pred_local_T.device)
    mask[:, 1:] = True
    return _se3_pose_loss(pred_local_T, gt_local_T, mask=mask,
                          rot_weight=rot_weight, trans_weight=trans_weight)


def auxiliary_pose_loss(
    pred_aux_T: dict[int, torch.Tensor],
    gt_global_T: torch.Tensor,
    intervals: Sequence[int] = (2, 4, 8, 16),
    *,
    base_weight: float = 0.5,
    decay: float = 0.5,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary multi-interval loss: Σ_Δ w_Δ · L(T̂_{i-Δ←i}, T*_{i-Δ←i}).

    Weight schedule: w_Δ = base_weight * decay^(k) where k is interval rank
    (k=0 for smallest Δ).

    Parameters
    ----------
    pred_aux_T : dict[Δ → (B, T, 4, 4)]
    gt_global_T : (B, T, 4, 4) ground-truth global transforms
    intervals : ordered list of intervals
    base_weight : weight for first (smallest) interval
    decay : multiplicative decay per interval rank
    rot_weight, trans_weight : pose loss component weights

    Returns
    -------
    total_aux_loss : scalar
    breakdown : dict with per-interval losses
    """
    device = gt_global_T.device
    dtype = gt_global_T.dtype
    total = torch.tensor(0.0, device=device, dtype=dtype)
    breakdown: dict[str, float] = {}
    T = gt_global_T.shape[1]

    for rank, delta in enumerate(sorted(intervals)):
        pred_T = pred_aux_T.get(delta)
        if pred_T is None:
            continue
        gt_T = gt_interval_transform(gt_global_T, delta)  # (B, T, 4, 4)

        # Mask: only valid for i >= delta
        mask = torch.zeros(pred_T.shape[:2], dtype=torch.bool, device=device)
        if delta < T:
            mask[:, delta:] = True

        w = base_weight * (decay ** rank)
        loss_d, bd = _se3_pose_loss(pred_T, gt_T, mask=mask,
                                     rot_weight=rot_weight, trans_weight=trans_weight)
        # Scale-normalise: translation MSE grows as Δ² while rotation
        # (geodesic) stays roughly constant.  Dividing by Δ keeps the
        # auxiliary contribution on the same order as the local (Δ=1)
        # loss so that the decay schedule controls relative importance.
        total = total + w * loss_d / delta
        breakdown[f"aux_delta{delta}_rot"] = bd["rot_loss"]
        breakdown[f"aux_delta{delta}_trans"] = bd["trans_loss"]
        breakdown[f"aux_delta{delta}_weight"] = w

    breakdown["aux_total"] = float(total.detach().item())
    return total, breakdown


def consistency_loss(
    pred_local_T: torch.Tensor,
    pred_aux_T: dict[int, torch.Tensor],
    *,
    delta_check: int = 2,
) -> torch.Tensor:
    """Cycle-consistency: T̂_{i-2←i} ≈ T̂_{i-2←i-1} @ T̂_{i-1←i}.

    Compares the auxiliary Δ=2 prediction against the composition of
    two consecutive local (Δ=1) predictions.  Penalises the Frobenius
    norm difference.

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4)  — local transforms (Δ=1)
    pred_aux_T : dict[Δ → (B, T, 4, 4)]
    delta_check : int  — which interval to enforce consistency on (default 2)

    Returns
    -------
    loss : scalar  (mean Frobenius norm of difference)
    """
    aux_2 = pred_aux_T.get(delta_check)
    if aux_2 is None:
        return torch.tensor(0.0, device=pred_local_T.device, dtype=pred_local_T.dtype)

    B, T = pred_local_T.shape[:2]
    if T < 3:
        return torch.tensor(0.0, device=pred_local_T.device, dtype=pred_local_T.dtype)

    # T̂_{i-2←i-1}: pred_local for the pair (i-2, i-1) — only makes sense
    # if we also have a Δ=1 head, which is pred_local_T.
    # But pred_local_T[:, i] = T_{i-1←i}.
    # We need T_{i-2←i-1} = pred_local_T[:, i-1] for i >= 2.
    # And T_{i-1←i} = pred_local_T[:, i] for i >= 2.
    # Composition: T_{i-2←i-1} @ T_{i-1←i} = pred_local_T[:, i-1] @ pred_local_T[:, i]

    composed = pred_local_T[:, 1:-1] @ pred_local_T[:, 2:]  # (B, T-2, 4, 4)
    direct = aux_2[:, 2:]  # (B, T-2, 4, 4) — aux prediction for positions i >= 2

    diff = composed - direct
    # Mean Frobenius distance
    fro = torch.norm(diff.reshape(-1, 4, 4), dim=(-2, -1))  # (N,)
    return fro.mean()


# ─── Combined sequence loss ─────────────────────────────────────────────────

def longseq_loss(
    pred_local_T: torch.Tensor,
    pred_aux_T: dict[int, torch.Tensor],
    gt_global_T: torch.Tensor,
    *,
    intervals: Sequence[int] = (2, 4, 8, 16),
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    aux_base_weight: float = 0.5,
    aux_decay: float = 0.5,
    consistency_weight: float = 0.1,
    consistency_delta: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss for long-sequence pose estimation.

    total = L_local + Σ L_aux_Δ + λ_consist · L_consist

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4) predicted local transforms
    pred_aux_T : dict[Δ → (B, T, 4, 4)] predicted auxiliary transforms
    gt_global_T : (B, T, 4, 4) ground-truth global transforms
    intervals : auxiliary interval set
    rot_weight, trans_weight : SE(3) loss component weights
    aux_base_weight : base weight for auxiliary losses
    aux_decay : decay factor per interval rank
    consistency_weight : weight for consistency loss
    consistency_delta : interval to enforce consistency on

    Returns
    -------
    total_loss : scalar
    breakdown : dict with all individual loss terms
    """
    # 1. Main local loss (Δ=1)
    l_local, bd_local = local_pose_loss(
        pred_local_T, gt_global_T,
        rot_weight=rot_weight, trans_weight=trans_weight,
    )

    # 2. Auxiliary losses
    l_aux, bd_aux = auxiliary_pose_loss(
        pred_aux_T, gt_global_T,
        intervals=intervals,
        base_weight=aux_base_weight,
        decay=aux_decay,
        rot_weight=rot_weight,
        trans_weight=trans_weight,
    )

    # 3. Consistency
    l_consist = consistency_loss(
        pred_local_T, pred_aux_T,
        delta_check=consistency_delta,
    )

    total = l_local + l_aux + consistency_weight * l_consist

    breakdown = {
        "loss_local": float(l_local.detach().item()),
        "loss_local_rot": bd_local["rot_loss"],
        "loss_local_trans": bd_local["trans_loss"],
        "loss_aux": float(l_aux.detach().item()),
        "loss_consistency": float(l_consist.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    breakdown.update(bd_aux)
    return total, breakdown
