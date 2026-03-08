"""SE(3) Pose-Graph Optimisation — pluggable backend for fused trajectory.

Nodes  : T_i ∈ SE(3), one per frame.  T_0 = I fixed (gauge freedom).
Edges  :
  * Short edges  (weight w_short) — one per consecutive frame pair
    from the ShortTransformer's local predictions.
  * Long  edges  (weight w_long)  — one per adjacent anchor pair
    from the LongTransformer (span s frames each).

The optimiser solves (Gauss-Newton, iterative):

    min_{T_1..T_{T-1}}  w_s * Σ_i ||e_short_i||²  +  w_l * Σ_m ||e_long_m||²

where each residual is a 6-vector in se(3):

    e = log( z^{-1} @ T_{a}^{-1} @ T_{b} )      (left-multiplicative error)

Solver: sparse Cholesky via scipy / torch.linalg.lstsq (fallback).

Usage::

    from eval.pose_graph import pose_graph_refine
    fused_refined = pose_graph_refine(
        short_local,          # (T, 4, 4) Δ=1 locals from short model
        long_local,           # (M, 4, 4) locals in anchor space
        anchor_indices,       # (M,)      frame indices of anchors
        init_global=fused_global,   # (T, 4, 4) initialisation (optional)
        w_short=1.0,
        w_long=0.5,
        n_iters=5,
    )
    # fused_refined: (T, 4, 4)
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

from metrics.compose import compose_global_from_local


# ─── SE(3) helpers ───────────────────────────────────────────────────────────

def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """SO(3) → so(3) axis-angle vector. R: (..., 3, 3) → (..., 3)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_a = ((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_a)
    skew = (R - R.transpose(-2, -1)) / 2.0
    omega = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    sin_a = torch.sin(angle).unsqueeze(-1).clamp(min=1e-8)
    omega = torch.where(
        (angle < 1e-6).unsqueeze(-1),
        torch.zeros_like(omega),
        omega / sin_a * angle.unsqueeze(-1),
    )
    return omega


def _se3_log(T: torch.Tensor) -> torch.Tensor:
    """SE(3) → R^6. T: (..., 4, 4) → (..., 6) = [omega, t]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    omega = _so3_log(R)
    return torch.cat([omega, t], dim=-1)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """so(3) axis-angle → SO(3). omega: (..., 3) → (..., 3, 3)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    ax = axis
    z = torch.zeros_like(ax[..., 0])
    K = torch.stack([
        z, -ax[..., 2], ax[..., 1],
        ax[..., 2], z, -ax[..., 0],
        -ax[..., 1], ax[..., 0], z,
    ], dim=-1).reshape(ax.shape[:-1] + (3, 3))
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    R = (
        I * cos_t.unsqueeze(-1)
        + (1 - cos_t).unsqueeze(-1) * torch.einsum("...i,...j->...ij", axis, axis)
        + sin_t.unsqueeze(-1) * K
    )
    return R


def _se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """R^6 → SE(3). xi: (..., 6) = [omega, t] → (..., 4, 4)."""
    omega = xi[..., :3]
    t = xi[..., 3:]
    R = _so3_exp(omega)
    batch = xi.shape[:-1]
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).expand(batch + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


# ─── Pose graph residual helpers ─────────────────────────────────────────────

def _edge_residual(
    T_a: torch.Tensor,
    T_b: torch.Tensor,
    z_ab: torch.Tensor,
) -> torch.Tensor:
    """6-vector residual for edge (a→b) with measurement z_ab.

    Convention: z_ab = T_{a←b} (prev_from_curr).
    Pred relative: T_a^{-1} @ T_b  (= T_{a←b} predicted by poses).
    Residual:  log( z_ab^{-1} @ T_a^{-1} @ T_b )
    """
    pred_rel = torch.linalg.inv(T_a) @ T_b  # (4,4)
    err_T = torch.linalg.inv(z_ab) @ pred_rel  # (4,4)
    return _se3_log(err_T)  # (6,)


# ─── Main optimiser ───────────────────────────────────────────────────────────

def pose_graph_refine(
    short_local: torch.Tensor,
    long_local: torch.Tensor,
    anchor_indices: torch.Tensor,
    *,
    init_global: Optional[torch.Tensor] = None,
    w_short: float = 1.0,
    w_long: float = 0.3,
    n_iters: int = 5,
    huber_delta: float = 5.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Gauss-Newton pose-graph optimisation.

    Parameters
    ----------
    short_local : (T, 4, 4)  — local transforms T_{i-1←i} from short model.
    long_local  : (M, 4, 4)  — local transforms T_{m-1←m} in anchor space.
    anchor_indices : (M,) LongTensor — frame indices of anchors.
    init_global : (T, 4, 4) or None — initialisation; defaults to composing short_local.
    w_short : float — weight for short edges.
    w_long  : float — weight for long edges.
    n_iters : int   — number of GN iterations.
    huber_delta : float — Huber kernel threshold (mm / rad equivalent).
    verbose : bool  — print per-iteration residual.

    Returns
    -------
    refined : (T, 4, 4) — optimised global transforms.
    """
    device = short_local.device
    dtype = short_local.dtype
    T = short_local.shape[0]
    M = anchor_indices.shape[0]

    if T <= 1:
        return short_local.clone()

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    if init_global is not None:
        poses = init_global.clone().to(device=device, dtype=dtype)
    else:
        poses = compose_global_from_local(short_local.unsqueeze(0)).squeeze(0).clone()

    # Compute anchor-space global transforms for long edges
    long_global_anchors = compose_global_from_local(long_local.unsqueeze(0)).squeeze(0)  # (M,4,4)

    # T_0 is fixed — optimise poses[1..T-1], giving N_free = T-1 variables × 6 dof
    N_free = T - 1
    DOF = 6

    # ------------------------------------------------------------------
    # Build edge list once (vertex indices into the FREE variables, i.e. frame−1)
    # ------------------------------------------------------------------
    # Short edges: (i-1, i) for i in 1..T-1
    # Each edge: node_a = i-1, node_b = i
    #   free-var index: i-1 → i-2 (or fixed if i-1=0), i → i-1
    edges: list[tuple] = []  # (type, frame_a, frame_b, measurement_z, weight)

    for i in range(1, T):
        edges.append(("short", i - 1, i, short_local[i], w_short))

    # Long edges: anchor a_m to a_{m+1}
    # Measurement: long_global_anchors[m+1] = global at anchor m+1 relative to anchor 0.
    # But we need the RELATIVE measurement between two anchor frames.
    # T_{a_m ← a_{m+1}} = long_global_anchors[m]^{-1} @ long_global_anchors[m+1]
    for m in range(M - 1):
        a_m = int(anchor_indices[m].item())
        a_n = int(anchor_indices[m + 1].item())
        if a_m >= T or a_n >= T:
            continue
        # Relative measurement in frame space
        z_long_mn = torch.linalg.inv(long_global_anchors[m]) @ long_global_anchors[m + 1]
        edges.append(("long", a_m, a_n, z_long_mn, w_long))

    # ------------------------------------------------------------------
    # Gauss-Newton iterations
    # ------------------------------------------------------------------
    for it in range(n_iters):
        # Accumulate J^T W J and J^T W r
        # Use dense accumulation for moderate T (<= 2000)
        # For large T use block-tridiagonal solver.
        try:
            poses = _gn_step_dense(poses, edges, N_free, DOF, huber_delta)
        except Exception as exc:
            warnings.warn(f"[pose_graph] GN iteration {it} failed: {exc}; stopping early")
            break

        if verbose:
            # Compute total residual
            total_sq = 0.0
            for _type, fa, fb, z, w in edges:
                r = _edge_residual(poses[fa], poses[fb], z)
                total_sq += float(w * (r ** 2).sum().item())
            print(f"[pose_graph] iter {it}  total_residual²={total_sq:.4f}")

    return poses


def _total_residual(poses: torch.Tensor, edges: list, huber_delta: float) -> float:
    """Sum of weighted Huber residuals for all edges."""
    total = 0.0
    for _type, fa, fb, z, w in edges:
        r = _edge_residual(poses[fa], poses[fb], z)
        r_norm = float(r.norm().item())
        if r_norm > huber_delta:
            cost = huber_delta * (2 * r_norm - huber_delta)
        else:
            cost = r_norm ** 2
        total += w * cost
    return total


def _apply_delta(poses: torch.Tensor, delta: torch.Tensor, N_free: int, DOF: int, step: float = 1.0) -> torch.Tensor:
    """Apply a GN step delta (scaled by `step`) to poses[1..T-1]."""
    new_poses = poses.clone()
    for fi in range(N_free):
        xi = step * delta[fi * DOF: (fi + 1) * DOF]
        dT = _se3_exp(xi.unsqueeze(0)).squeeze(0)
        new_poses[fi + 1] = dT @ poses[fi + 1]
    return new_poses


def _gn_step_dense(
    poses: torch.Tensor,
    edges: list,
    N_free: int,
    DOF: int,
    huber_delta: float,
) -> torch.Tensor:
    """Single GN step (Levenberg-Marquardt) with backtracking line search.

    Solves:  (H + λ diag(H)) Δ = -g   where H = J^T W J,  g = J^T W r
    Applies: T_i ← exp(α Δ_i) @ T_i   with Armijo backtracking on α.
    """
    device = poses.device
    dtype = poses.dtype
    dim = N_free * DOF

    H = torch.zeros(dim, dim, device=device, dtype=dtype)
    g = torch.zeros(dim, device=device, dtype=dtype)  # gradient g = J^T W r

    for _type, fa, fb, z, w in edges:
        T_a = poses[fa]
        T_b = poses[fb]

        r = _edge_residual(T_a, T_b, z)  # (6,)

        # Huber weight
        r_norm = float(r.norm().item())
        wr = w * (huber_delta / (r_norm + 1e-8) if r_norm > huber_delta else 1.0)

        # Numerical Jacobians wrt left perturbation of T_fa and T_fb
        eps = 1e-5
        J_a = _numerical_jac(poses, fa, fb, z, eps, device, dtype, wrt="a")  # (6,6)
        J_b = _numerical_jac(poses, fa, fb, z, eps, device, dtype, wrt="b")  # (6,6)

        # Free-variable indices (frame 0 is fixed → free index = frame - 1)
        free_a = fa - 1  # -1 means fixed
        free_b = fb - 1

        # Accumulate gradient and Hessian
        if free_a >= 0:
            ra = free_a * DOF
            g[ra:ra+DOF] += wr * (J_a.t() @ r)
            H[ra:ra+DOF, ra:ra+DOF] += wr * (J_a.t() @ J_a)
        if free_b >= 0:
            rb = free_b * DOF
            g[rb:rb+DOF] += wr * (J_b.t() @ r)
            H[rb:rb+DOF, rb:rb+DOF] += wr * (J_b.t() @ J_b)
        if free_a >= 0 and free_b >= 0:
            ra = free_a * DOF
            rb = free_b * DOF
            cross = wr * (J_a.t() @ J_b)
            H[ra:ra+DOF, rb:rb+DOF] += cross
            H[rb:rb+DOF, ra:ra+DOF] += cross.t()

    # Adaptive Levenberg-Marquardt damping: λ = 0.1 * mean(diag(H))
    diag_mean = H.diagonal().abs().mean().clamp(min=1e-6)
    H.diagonal().add_(0.1 * diag_mean)

    # Solve H Δ = -g
    try:
        delta = torch.linalg.solve(H, -g)
    except Exception:
        delta = torch.linalg.lstsq(H, (-g).unsqueeze(-1)).solution.squeeze(-1)

    # Armijo backtracking line search
    cost0 = _total_residual(poses, edges, huber_delta)
    slope = float((-g @ delta).item())  # should be negative (descent direction)
    alpha = 1.0
    c = 0.1  # Armijo constant
    for _ in range(10):
        candidate = _apply_delta(poses, delta, N_free, DOF, step=alpha)
        cost_new = _total_residual(candidate, edges, huber_delta)
        if cost_new <= cost0 + c * alpha * slope:
            return candidate
        alpha *= 0.5

    # If backtracking fails, still return best candidate (conservative step)
    return _apply_delta(poses, delta, N_free, DOF, step=alpha)


def _numerical_jac(
    poses: torch.Tensor,
    fa: int,
    fb: int,
    z: torch.Tensor,
    eps: float,
    device: torch.device,
    dtype: torch.dtype,
    wrt: str,  # "a" or "b"
) -> torch.Tensor:
    """6×6 numerical Jacobian of edge residual wrt left perturbation of T_fa or T_fb."""
    r0 = _edge_residual(poses[fa], poses[fb], z)
    J = torch.zeros(6, 6, device=device, dtype=dtype)
    for k in range(6):
        xi = torch.zeros(6, device=device, dtype=dtype)
        xi[k] = eps
        dT = _se3_exp(xi.unsqueeze(0)).squeeze(0)
        if wrt == "a":
            r_pert = _edge_residual(dT @ poses[fa], poses[fb], z)
        else:
            r_pert = _edge_residual(poses[fa], dT @ poses[fb], z)
        J[:, k] = (r_pert - r0) / eps
    return J  # (6, 6)
