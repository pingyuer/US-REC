"""Multi-interval auxiliary loss + consistency constraint for long-sequence pose.

This module implements:
    1. **Main local loss** (Δ=1): points-based L2 loss on T_{i-1←i}
    2. **Auxiliary losses** (Δ∈{2,4,8,16}): points-based multi-interval supervision
       using ground-truth  gt_T_{i-Δ←i} = inv(gt_global[i-Δ]) @ gt_global[i].
    3. **Consistency constraint**: L_consist — cycle consistency in point space
       T̂_{i-2←i} @ P ≈ T̂_{i-2←i-1} @ T̂_{i-1←i} @ P
    4. **DDF surrogate loss**: random pixel subsampling of the GP-DDF, i.e.
       sample K random pixel coordinates, map through tform_calib to tool-mm
       space, apply global transforms, measure L2 error in mm.  This directly
       matches the official TUS-REC GPE oracle and makes the training loss
       numerically consistent with the evaluation criterion.
    5. **Sequence loss combiner**: weighted sum with configurable weights.

Two loss modes are supported (via ``loss_mode`` argument):
    * ``"points"``  (default, recommended): projects reference points through
      pred/gt transforms and measures L2 distance.  Both rotation and
      translation errors contribute naturally, matching the GPE metric.
    * ``"se3"``     (ablation only): geodesic rotation + MSE translation.

References:
    Liqi/TBME: multi-interval auxiliary supervision reduces drift (GPE).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from utils.rotation_loss import geodesic_loss


# ─── Reference point helpers ─────────────────────────────────────────────────

def make_ref_points(
    scale_mm: float = 20.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 4 image-plane corner reference points in homogeneous coordinates.

    Points are placed at ±scale_mm on the X and Y axes (Z=0, W=1), giving
    a square footprint that makes both rotation and translation errors visible.

    Returns
    -------
    pts : (4, 4) — each row is a homogeneous 3D point [x, y, z, 1]
    """
    s = float(scale_mm)
    pts = torch.tensor(
        [
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s,  s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )  # (4, 4)
    return pts


def _apply_transforms_to_points(
    T: torch.Tensor,
    pts: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply (batch of) SE(3) transforms to reference points.

    Parameters
    ----------
    T    : (..., 4, 4)  transforms
    pts  : (K, 4)       reference points in homogeneous coordinates
    mask : optional boolean tensor matching T's leading dims; only valid
           entries are returned when provided.

    Returns
    -------
    pos3d : shape (..., K, 3)  or (N_valid, K, 3) when mask given
        3D positions of points after applying each transform.
    """
    # pts.T : (4, K);  T : (..., 4, 4)
    # PyTorch matmul broadcasts: (..., 4, 4) @ (4, K) → (..., 4, K)
    pts_hom = pts.T.to(device=T.device, dtype=T.dtype)  # (4, K)
    pos = T @ pts_hom                                    # (..., 4, K)
    pos3d = pos[..., :3, :].transpose(-2, -1)            # (..., K, 3)
    if mask is not None:
        pos3d = pos3d[mask]
    return pos3d


def _points_loss_from_transforms(
    pred_T: torch.Tensor,
    gt_T: torch.Tensor,
    pts: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L2 error between predicted and GT point positions (mean over K and N).

    Parameters
    ----------
    pred_T, gt_T : (..., 4, 4)
    pts  : (K, 4) reference points
    mask : optional boolean mask (same leading dims as T)

    Returns
    -------
    loss : scalar
    """
    pred_pos = _apply_transforms_to_points(pred_T, pts, mask)  # (N, K, 3)
    gt_pos   = _apply_transforms_to_points(gt_T,   pts, mask)  # (N, K, 3)
    if pred_pos.numel() == 0:
        return torch.tensor(0.0, device=pred_T.device, dtype=pred_T.dtype)
    return F.mse_loss(pred_pos, gt_pos)


# ─── SE(3) parameter loss (ablation only) ────────────────────────────────────

def _se3_pose_loss(
    pred_T: torch.Tensor,
    gt_T: torch.Tensor,
    mask: torch.Tensor | None = None,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Geodesic rotation + L2 translation loss (ablation / legacy mode).

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

# ─── Per-mode loss functions ─────────────────────────────────────────────────

def local_pose_loss(
    pred_local_T: torch.Tensor,
    gt_global_T: torch.Tensor,
    *,
    loss_mode: str = "points",
    ref_pts: torch.Tensor | None = None,
    ref_pts_scale_mm: float = 20.0,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Main loss (Δ=1): local frame-to-frame pose error.

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4)  with pred_local_T[:, 0] = I
    gt_global_T  : (B, T, 4, 4)  ground-truth global transforms
    loss_mode    : "points" (default) | "se3"
    ref_pts      : (K, 4) reference points; generated from scale if None
    ref_pts_scale_mm : half-span of reference square (mm)
    rot_weight, trans_weight : SE(3) weights (only used in se3 mode)

    Returns
    -------
    loss : scalar
    breakdown : dict with diagnostic floats
    """
    from metrics.compose import local_from_global

    gt_local_T = local_from_global(gt_global_T)  # (B, T, 4, 4)

    # Only compute on i >= 1 (frame 0 is identity)
    mask = torch.zeros(pred_local_T.shape[:2], dtype=torch.bool,
                       device=pred_local_T.device)
    mask[:, 1:] = True

    if loss_mode == "points":
        if ref_pts is None:
            ref_pts = make_ref_points(
                ref_pts_scale_mm,
                device=pred_local_T.device,
                dtype=pred_local_T.dtype,
            )
        loss = _points_loss_from_transforms(pred_local_T, gt_local_T, ref_pts, mask)
        return loss, {"rot_loss": 0.0, "trans_loss": float(loss.detach().item())}
    else:
        return _se3_pose_loss(pred_local_T, gt_local_T, mask=mask,
                              rot_weight=rot_weight, trans_weight=trans_weight)


def auxiliary_pose_loss(
    pred_aux_T: dict[int, torch.Tensor],
    gt_global_T: torch.Tensor,
    intervals: Sequence[int] = (2, 4, 8, 16),
    *,
    loss_mode: str = "points",
    ref_pts: torch.Tensor | None = None,
    ref_pts_scale_mm: float = 20.0,
    base_weight: float = 0.5,
    decay: float = 0.5,
    aux_scale: str = "none",
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
    loss_mode : "points" | "se3"
    ref_pts : (K, 4) reference points; generated if None (points mode)
    ref_pts_scale_mm : half-span of reference square (mm)
    base_weight : weight for first (smallest) interval
    decay : multiplicative decay per interval rank
    aux_scale : scaling for auxiliary loss magnitude:
        "none"  — no scaling (recommended for points mode)
        "delta" — divide by Δ  (legacy default for SE(3) mode)
        "delta2"— divide by Δ²
    rot_weight, trans_weight : SE(3) weights (only used in se3 mode)

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

    if loss_mode == "points" and ref_pts is None:
        ref_pts = make_ref_points(ref_pts_scale_mm, device=device, dtype=dtype)

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

        if loss_mode == "points":
            loss_d = _points_loss_from_transforms(pred_T, gt_T, ref_pts, mask)
            breakdown[f"aux_delta{delta}_pts"] = float(loss_d.detach().item())
        else:
            loss_d, bd = _se3_pose_loss(pred_T, gt_T, mask=mask,
                                        rot_weight=rot_weight,
                                        trans_weight=trans_weight)
            breakdown[f"aux_delta{delta}_rot"] = bd["rot_loss"]
            breakdown[f"aux_delta{delta}_trans"] = bd["trans_loss"]

        # Optional normalisation by interval size.
        # For points mode the loss already encodes absolute mm error so /Δ
        # scaling is usually counterproductive.  For SE(3) mode the legacy
        # /Δ normalisation keeps rotation and translation on the same order.
        if aux_scale == "delta":
            loss_d = loss_d / max(1, delta)
        elif aux_scale == "delta2":
            loss_d = loss_d / max(1, delta * delta)
        # else "none": no scaling

        total = total + w * loss_d
        breakdown[f"aux_delta{delta}_weight"] = w

    breakdown["aux_total"] = float(total.detach().item())
    return total, breakdown


def consistency_loss(
    pred_local_T: torch.Tensor,
    pred_aux_T: dict[int, torch.Tensor],
    *,
    delta_check: int = 2,
    loss_mode: str = "points",
    ref_pts: torch.Tensor | None = None,
    ref_pts_scale_mm: float = 20.0,
) -> torch.Tensor:
    """Cycle-consistency: T̂_{i-2←i} @ P ≈ T̂_{i-2←i-1} @ T̂_{i-1←i} @ P.

    Compares the auxiliary Δ=delta_check prediction against the composition of
    two consecutive local (Δ=1) predictions, either in matrix space (Frobenius)
    or in point space (L2 on mapped points).

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4)  — local transforms (Δ=1)
    pred_aux_T : dict[Δ → (B, T, 4, 4)]
    delta_check : int  — which interval to enforce consistency on (default 2)
    loss_mode : "points" | "se3"
    ref_pts : (K, 4) reference points (points mode only)
    ref_pts_scale_mm : half-span of reference square (mm)

    Returns
    -------
    loss : scalar
    """
    aux_dc = pred_aux_T.get(delta_check)
    if aux_dc is None:
        return torch.tensor(0.0, device=pred_local_T.device, dtype=pred_local_T.dtype)

    B, T = pred_local_T.shape[:2]
    if T < delta_check + 1:
        return torch.tensor(0.0, device=pred_local_T.device, dtype=pred_local_T.dtype)

    # Compose delta_check local transforms to build T̂_{i-Δ←i}
    # For delta_check=2:
    #   composed[:, i] = pred_local_T[:, i-1] @ pred_local_T[:, i]
    # For general delta_check, slide a product window.
    composed = torch.eye(4, device=pred_local_T.device,
                          dtype=pred_local_T.dtype).expand(B, 1, 4, 4).clone()
    for k in range(delta_check):
        offset = delta_check - k
        composed = pred_local_T[:, offset - 1 : T - k - 1] @ \
                   pred_local_T[:, offset : T - k]
        if k == 0:
            composed_all = composed
        else:
            composed_all = composed_all @ pred_local_T[:, delta_check - k : T - k]

    # Simplification: for delta_check == 2 only (common case)
    if delta_check == 2:
        composed_all = pred_local_T[:, 1:-1] @ pred_local_T[:, 2:]

    direct = aux_dc[:, delta_check:]  # (B, T-delta_check, 4, 4)

    if loss_mode == "points":
        if ref_pts is None:
            ref_pts = make_ref_points(
                ref_pts_scale_mm,
                device=pred_local_T.device,
                dtype=pred_local_T.dtype,
            )
        return _points_loss_from_transforms(composed_all, direct, ref_pts)
    else:
        diff = composed_all - direct
        fro = torch.norm(diff.reshape(-1, 4, 4), dim=(-2, -1))
        return fro.mean()


# ─── DDF surrogate loss ──────────────────────────────────────────────────────

def ddf_surrogate_loss(
    pred_global_T: torch.Tensor,
    gt_global_T: torch.Tensor,
    tform_calib: torch.Tensor,
    *,
    image_size: tuple[int, int] = (480, 640),
    num_points: int = 1024,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Dense-displacement-field surrogate loss using random pixel subsampling.

    Matches the official TUS-REC GPE criterion: for each frame *i*, sample
    *K* random pixel coordinates, map them to tool-frame 3D via ``tform_calib``,
    then apply the global transform T_{0←i} to get world-space 3D positions.
    The L2 error between predicted and GT world-space positions (in mm) is the
    DDF surrogate.

    Parameters
    ----------
    pred_global_T : (B, T, 4, 4)  predicted global transforms (frame 0 = I)
    gt_global_T   : (B, T, 4, 4)  ground-truth global transforms
    tform_calib   : (4, 4)  calibration matrix mapping pixel coords → tool mm
        Convention: ``pts_tool = tform_calib @ [u, v, 0, 1]^T``
    image_size : (H, W)  image dimensions for pixel sampling
    num_points : int  number of random pixels to sample per batch
    generator : optional torch.Generator for reproducible sampling

    Returns
    -------
    loss : scalar MSE in mm² (differentiable w.r.t. pred_global_T)
    """
    B, T, _, _ = pred_global_T.shape
    H, W = image_size
    device = pred_global_T.device
    dtype = pred_global_T.dtype

    # Sample random pixel coordinates (u=col, v=row)
    u = torch.rand(num_points, generator=generator, device=device, dtype=dtype) * (W - 1)
    v = torch.rand(num_points, generator=generator, device=device, dtype=dtype) * (H - 1)
    zeros = torch.zeros(num_points, device=device, dtype=dtype)
    ones  = torch.ones(num_points, device=device, dtype=dtype)
    # pts_pixel : (4, K)  — homogeneous pixel coordinates [u, v, 0, 1]
    pts_pixel = torch.stack([u, v, zeros, ones], dim=0)  # (4, K)

    # Map to tool-frame mm: (4, 4) @ (4, K) → (4, K)
    tform_c = tform_calib.to(device=device, dtype=dtype)  # (4, 4)
    pts_tool = tform_c @ pts_pixel  # (4, K)

    # Apply global transforms:  (B, T, 4, 4) @ (4, K) → (B, T, 4, K)
    # keep first 3 rows → (B, T, 3, K) → (B, T, K, 3)
    pred_world = (pred_global_T @ pts_tool)[..., :3, :].transpose(-2, -1)  # (B,T,K,3)
    gt_world   = (gt_global_T   @ pts_tool)[..., :3, :].transpose(-2, -1)  # (B,T,K,3)

    return F.mse_loss(pred_world, gt_world)


# ─── Combined sequence loss ─────────────────────────────────────────────────

def longseq_loss(
    pred_local_T: torch.Tensor,
    pred_aux_T: dict[int, torch.Tensor],
    gt_global_T: torch.Tensor,
    *,
    intervals: Sequence[int] = (2, 4, 8, 16),
    loss_mode: str = "points",
    ref_pts_scale_mm: float = 20.0,
    rot_weight: float = 1.0,
    trans_weight: float = 1.0,
    aux_base_weight: float = 0.5,
    aux_decay: float = 0.5,
    aux_scale: str = "none",
    consistency_weight: float = 0.1,
    consistency_delta: int = 2,
    ddf_sample_weight: float = 0.0,
    ddf_num_points: int = 1024,
    ddf_tform_calib: torch.Tensor | None = None,
    ddf_image_size: tuple[int, int] = (480, 640),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss for long-sequence pose estimation.

    total = L_local + Σ L_aux_Δ + λ_consist · L_consist

    Parameters
    ----------
    pred_local_T : (B, T, 4, 4) predicted local transforms
    pred_aux_T : dict[Δ → (B, T, 4, 4)] predicted auxiliary transforms
    gt_global_T : (B, T, 4, 4) ground-truth global transforms
    intervals : auxiliary interval set
    loss_mode : "points" (default) | "se3" (ablation)
    ref_pts_scale_mm : half-span of reference square in mm (points mode)
    rot_weight, trans_weight : SE(3) weights (se3 mode only)
    aux_base_weight : base weight for auxiliary losses
    aux_decay : decay factor per interval rank
    aux_scale : scaling for aux loss: "none" | "delta" | "delta2"
    consistency_weight : weight for consistency loss
    consistency_delta : interval to enforce consistency on
    ddf_sample_weight : weight for DDF surrogate loss (0 = disabled)
    ddf_num_points : number of random pixels to sample for DDF loss
    ddf_tform_calib : (4, 4) calibration matrix; DDF loss is skipped if None
    ddf_image_size : (H, W) image dimensions for DDF pixel sampling

    Returns
    -------
    total_loss : scalar
    breakdown : dict with all individual loss terms
    """
    device = gt_global_T.device
    dtype = gt_global_T.dtype

    # Shared reference points (avoid re-creating per sub-loss)
    ref_pts = (
        make_ref_points(ref_pts_scale_mm, device=device, dtype=dtype)
        if loss_mode == "points"
        else None
    )

    # 1. Main local loss (Δ=1)
    l_local, bd_local = local_pose_loss(
        pred_local_T, gt_global_T,
        loss_mode=loss_mode,
        ref_pts=ref_pts,
        ref_pts_scale_mm=ref_pts_scale_mm,
        rot_weight=rot_weight,
        trans_weight=trans_weight,
    )

    # 2. Auxiliary losses
    l_aux, bd_aux = auxiliary_pose_loss(
        pred_aux_T, gt_global_T,
        intervals=intervals,
        loss_mode=loss_mode,
        ref_pts=ref_pts,
        ref_pts_scale_mm=ref_pts_scale_mm,
        base_weight=aux_base_weight,
        decay=aux_decay,
        aux_scale=aux_scale,
        rot_weight=rot_weight,
        trans_weight=trans_weight,
    )

    # 3. Consistency
    l_consist = consistency_loss(
        pred_local_T, pred_aux_T,
        delta_check=consistency_delta,
        loss_mode=loss_mode,
        ref_pts=ref_pts,
        ref_pts_scale_mm=ref_pts_scale_mm,
    )

    # 4. DDF surrogate (only when calibration is provided and weight > 0)
    need_ddf = ddf_sample_weight > 0.0 and ddf_tform_calib is not None
    if need_ddf:
        # Reconstruct global from local for the DDF loss.
        from metrics.compose import compose_global_from_local
        pred_global_T = compose_global_from_local(pred_local_T)
        l_ddf = ddf_surrogate_loss(
            pred_global_T,
            gt_global_T,
            tform_calib=ddf_tform_calib,
            image_size=ddf_image_size,
            num_points=ddf_num_points,
        )
    else:
        l_ddf = torch.tensor(0.0, device=device, dtype=dtype)

    total = l_local + l_aux + consistency_weight * l_consist + ddf_sample_weight * l_ddf

    breakdown = {
        "loss_local": float(l_local.detach().item()),
        "loss_local_rot": bd_local["rot_loss"],
        "loss_local_trans": bd_local["trans_loss"],
        "loss_aux": float(l_aux.detach().item()),
        "loss_consistency": float(l_consist.detach().item()),
        "loss_ddf": float(l_ddf.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    breakdown.update(bd_aux)
    return total, breakdown
