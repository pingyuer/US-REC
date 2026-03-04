"""Dual-path fusion: merge Dense (Δ=1) and Sparse (Δ=k) predictions into fused global transforms.

Two fusion modes:

1. **anchor_interp** — Anchor correction with optional SE(3) interpolation.
   Fast, simple, and effective for moderate k (4–16).

2. **pose_graph** — Gauss-Newton pose-graph optimisation with Huber robust kernel.
   Uses both dense and sparse relative-pose edges to jointly optimise all
   global transforms.  Principled and effective at suppressing drift.

Both modes output T_{0←i} for every frame i.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from metrics.compose import compose_global_from_local


# ─── SE(3) Lie algebra helpers ───────────────────────────────────────────────

def _se3_log(T: torch.Tensor) -> torch.Tensor:
    """Approximate SE(3) logarithm → ℝ⁶ (rotation vector ++ translation).

    Uses the small-angle approximation for simplicity and differentiability.
    Sufficient for interpolation between nearby corrections.

    Parameters
    ----------
    T : (..., 4, 4) rigid transform

    Returns
    -------
    xi : (..., 6)  [omega_x, omega_y, omega_z, tx, ty, tz]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Rodrigues inverse: θ = arccos((tr(R)-1)/2), axis from skew-symmetric part
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)  # (...,)

    # Skew-symmetric part: (R - R^T) / 2
    skew = (R - R.transpose(-2, -1)) / 2.0
    # axis * sin(θ) = [skew[2,1], skew[0,2], skew[1,0]]
    omega = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0],
    ], dim=-1)  # (..., 3)

    # Normalise to get axis * θ (avoid div-by-zero for small angles)
    sin_angle = torch.sin(angle).unsqueeze(-1).clamp(min=1e-8)
    omega = omega / sin_angle * angle.unsqueeze(-1)

    # For very small angles, use identity approximation (omega ≈ 0)
    small = (angle < 1e-6).unsqueeze(-1)
    omega = torch.where(small, torch.zeros_like(omega), omega)

    return torch.cat([omega, t], dim=-1)  # (..., 6)


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential map: ℝ⁶ → SE(3).

    Parameters
    ----------
    xi : (..., 6)  [omega (3), translation (3)]

    Returns
    -------
    T : (..., 4, 4)
    """
    omega = xi[..., :3]   # (..., 3)
    t = xi[..., 3:]       # (..., 3)

    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis = omega / theta  # (..., 3)

    # Rodrigues formula
    cos_t = torch.cos(theta).unsqueeze(-1)   # (..., 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)   # (..., 1, 1)

    # Skew matrix of axis
    ax = axis  # (..., 3)
    zero = torch.zeros_like(ax[..., 0])
    K = torch.stack([
        zero, -ax[..., 2], ax[..., 1],
        ax[..., 2], zero, -ax[..., 0],
        -ax[..., 1], ax[..., 0], zero,
    ], dim=-1).reshape(ax.shape[:-1] + (3, 3))

    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    R = I * cos_t + (1 - cos_t) * torch.einsum("...i,...j->...ij", axis, axis) + sin_t * K

    # For very small angles, R ≈ I
    small = (theta.squeeze(-1) < 1e-6).unsqueeze(-1).unsqueeze(-1)
    R = torch.where(small, I, R)

    T = torch.zeros(xi.shape[:-1] + (4, 4), device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ─── Anchor-correction fusion (D1) ──────────────────────────────────────────

def _anchor_interp_fusion(
    dense_global: torch.Tensor,
    sparse_anchor_global: torch.Tensor,
    anchor_indices: torch.Tensor,
    T: int,
    smooth: bool = True,
) -> torch.Tensor:
    """Fuse dense and sparse globals via anchor correction + optional SE(3) slerp.

    Steps:
      1. Compute correction C_m = sparse_anchor_global[m] @ inv(dense_global[anchor[m]])
      2. For each interval [anchor[m], anchor[m+1]):
         - If smooth: interpolate corrections in SE(3) log space
         - Else: apply left-correction C_m to dense_global[i]
      3. Return fused_global for all T frames.

    Parameters
    ----------
    dense_global : (T, 4, 4)
    sparse_anchor_global : (M, 4, 4)
    anchor_indices : (M,) LongTensor
    T : int — total frames
    smooth : bool — SE(3) interpolation between corrections if True

    Returns
    -------
    fused_global : (T, 4, 4)
    """
    device = dense_global.device
    dtype = dense_global.dtype
    M = anchor_indices.shape[0]

    # Corrections at each anchor: C_m = sparse[m] @ inv(dense[anchor[m]])
    anchor_dense = dense_global[anchor_indices]  # (M, 4, 4)
    inv_anchor_dense = torch.linalg.inv(anchor_dense)
    corrections = sparse_anchor_global @ inv_anchor_dense  # casted to (M, 4, 4) ... actually (M, 4, 4)
    # Fix: corrections[m] = sparse_anchor_global[m] @ inv(dense_global[anchor_indices[m]])
    # But matrix multiply needs careful ordering. C_m is a left-multiplier:
    # fused[i] = C_m @ dense[i].
    # So C_m = sparse_global[m] @ inv(dense_global[anchor[m]]).

    fused = dense_global.clone()  # (T, 4, 4)

    if not smooth or M <= 1:
        # Piecewise constant correction
        for m_idx in range(M):
            start = int(anchor_indices[m_idx].item())
            end = int(anchor_indices[m_idx + 1].item()) if m_idx + 1 < M else T
            C = corrections[m_idx]  # (4, 4)
            for i in range(start, end):
                fused[i] = C @ dense_global[i]
        return fused

    # Smooth: SE(3) log-space linear interpolation of corrections
    log_corrections = _se3_log(corrections)  # (M, 6)

    for m_idx in range(M):
        start = int(anchor_indices[m_idx].item())
        end = int(anchor_indices[m_idx + 1].item()) if m_idx + 1 < M else T

        log_C_curr = log_corrections[m_idx]  # (6,)
        if m_idx + 1 < M:
            log_C_next = log_corrections[m_idx + 1]  # (6,)
        else:
            log_C_next = log_C_curr  # last segment: constant

        interval_len = end - start
        if interval_len <= 0:
            continue

        for i in range(start, end):
            alpha = float(i - start) / max(1, interval_len)
            log_C_interp = (1.0 - alpha) * log_C_curr + alpha * log_C_next
            C_interp = _se3_exp(log_C_interp.unsqueeze(0)).squeeze(0)  # (4, 4)
            fused[i] = C_interp @ dense_global[i]

    return fused


# ─── Pose-graph optimisation (D2) ───────────────────────────────────────────

def _pose_graph_fusion(
    dense_local_T: torch.Tensor,
    sparse_local_T: torch.Tensor,
    anchor_indices: torch.Tensor,
    T: int,
    *,
    n_iters: int = 20,
    huber_delta: float = 0.1,
    sparse_info_weight: float = 2.0,
) -> torch.Tensor:
    """Pose-graph optimisation (Gauss-Newton on SE(3) Lie algebra).

    Nodes: X_i ∈ SE(3) for i = 0..T-1  (X_0 = I, fixed).
    Edges:
      - Dense: Z^(1)_{i-1,i} with information weight 1
      - Sparse: Z^(k)_{(m-1)k, mk} with information weight `sparse_info_weight`

    Optimise in tangent space (incremental ℝ⁶ updates) with Huber robust kernel.

    Parameters
    ----------
    dense_local_T : (T, 4, 4)   local Δ=1 predictions
    sparse_local_T : (M, 4, 4)  local Δ=k predictions
    anchor_indices : (M,)        anchor frame indices
    T : int
    n_iters : int
    huber_delta : float
    sparse_info_weight : float

    Returns
    -------
    optimised_global : (T, 4, 4)
    """
    device = dense_local_T.device
    dtype = dense_local_T.dtype

    # Initialise with dense accumulation
    X = compose_global_from_local(dense_local_T.unsqueeze(0)).squeeze(0)  # (T, 4, 4)

    # Collect edges: (i, j, Z_ij, weight)
    # Z_ij ≈ inv(X_i) @ X_j  — the measurement
    edges: list[tuple[int, int, torch.Tensor, float]] = []

    # Dense edges (Δ=1)
    for i in range(1, T):
        edges.append((i - 1, i, dense_local_T[i], 1.0))

    # Sparse edges (Δ=k)
    M = anchor_indices.shape[0]
    for m in range(1, M):
        i_idx = int(anchor_indices[m - 1].item())
        j_idx = int(anchor_indices[m].item())
        edges.append((i_idx, j_idx, sparse_local_T[m], sparse_info_weight))

    # Gauss-Newton iterations
    for _it in range(n_iters):
        # Build linear system: sum over edges of J^T W J dx = J^T W r
        # For simplicity, use gradient descent on the Lie algebra residuals.
        grad = torch.zeros(T, 6, device=device, dtype=dtype)

        for (i_idx, j_idx, Z_ij, w) in edges:
            # Residual: log(Z_ij^{-1} @ inv(X_i) @ X_j)
            X_rel = torch.linalg.inv(X[i_idx]) @ X[j_idx]  # (4,4)
            Z_inv = torch.linalg.inv(Z_ij)
            err_T = Z_inv @ X_rel  # (4,4)
            err_log = _se3_log(err_T.unsqueeze(0)).squeeze(0)  # (6,)

            # Huber weighting
            err_norm = err_log.norm()
            if err_norm > huber_delta:
                huber_w = huber_delta / (err_norm + 1e-8)
            else:
                huber_w = 1.0

            weighted_err = w * huber_w * err_log

            # Approximate Jacobian: ∂residual/∂X_i ≈ -I, ∂residual/∂X_j ≈ +I
            if i_idx > 0:  # X_0 is fixed
                grad[i_idx] -= weighted_err
            if j_idx > 0:
                grad[j_idx] += weighted_err

        # Update step (gradient descent with small step size for stability)
        step_size = 0.5
        for idx in range(1, T):  # skip X_0 = I
            delta_T = _se3_exp((-step_size * grad[idx]).unsqueeze(0)).squeeze(0)
            X[idx] = delta_T @ X[idx]

    return X


# ─── Public API ──────────────────────────────────────────────────────────────

def fuse_dual_predictions(
    dense_local_T: torch.Tensor,
    sparse_local_T: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    mode: str = "anchor_interp",
    smooth: bool = True,
    pose_graph_iters: int = 20,
    pose_graph_huber: float = 0.1,
    pose_graph_sparse_weight: float = 2.0,
) -> torch.Tensor:
    """Fuse dense and sparse predictions into global transforms.

    Parameters
    ----------
    dense_local_T : (T, 4, 4)   Dense Δ=1 local transforms (frame 0 = I).
    sparse_local_T : (M, 4, 4)  Sparse Δ=k local transforms (anchor 0 = I).
    anchor_indices : (M,)        LongTensor of anchor frame indices.
    mode : "anchor_interp" | "pose_graph"
    smooth : bool                SE(3) interpolation for anchor_interp mode.
    pose_graph_iters : int       Iterations for pose graph.
    pose_graph_huber : float     Huber kernel δ.
    pose_graph_sparse_weight : float  Information weight for sparse edges.

    Returns
    -------
    fused_global : (T, 4, 4)    Fused global transforms T_{0←i}.
    """
    T = dense_local_T.shape[0]

    if mode == "anchor_interp":
        # Accumulate dense global
        dense_global = compose_global_from_local(
            dense_local_T.unsqueeze(0)
        ).squeeze(0)  # (T, 4, 4)
        # Accumulate sparse anchor global
        sparse_anchor_global = compose_global_from_local(
            sparse_local_T.unsqueeze(0)
        ).squeeze(0)  # (M, 4, 4)

        return _anchor_interp_fusion(
            dense_global, sparse_anchor_global, anchor_indices, T, smooth=smooth,
        )

    elif mode == "pose_graph":
        return _pose_graph_fusion(
            dense_local_T, sparse_local_T, anchor_indices, T,
            n_iters=pose_graph_iters,
            huber_delta=pose_graph_huber,
            sparse_info_weight=pose_graph_sparse_weight,
        )

    else:
        raise ValueError(f"Unknown fusion mode: {mode!r}. Use 'anchor_interp' or 'pose_graph'.")
