"""Rotation representation utilities.

Supports conversion between different rotation parameterizations and SE(3)
construction.  All quaternions use **[w, x, y, z]** convention throughout
this module and the wider project.

References
----------
* Zhou et al., "On the Continuity of Rotation Representations in Neural
  Networks", CVPR 2019 — rot6d (6D continuous representation).
* Quaternion conventions follow Hamilton ordering [w, x, y, z].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ─── Quaternion helpers ──────────────────────────────────────────────────────

_EPS = 1e-7


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Normalize a quaternion to unit length.

    Parameters
    ----------
    q : (..., 4) tensor  — [w, x, y, z]

    Returns
    -------
    (..., 4) unit quaternion
    """
    return q / (q.norm(dim=-1, keepdim=True) + _EPS)


def quat_sign_align(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
    """Flip *q* so that it lies in the same hemisphere as *q_ref*.

    Solves the q ≡ -q ambiguity before loss computation.

    Parameters
    ----------
    q, q_ref : (..., 4) tensors  — [w, x, y, z]
    """
    dot = (q * q_ref).sum(dim=-1, keepdim=True)
    return torch.where(dot < 0, -q, q)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion to rotation matrix.

    Parameters
    ----------
    q : (..., 4) tensor  — [w, x, y, z]

    Returns
    -------
    (..., 3, 3) rotation matrix
    """
    q = normalize_quat(q)
    w, x, y, z = q.unbind(dim=-1)

    # Precompute products
    xx = x * x;  yy = y * y;  zz = z * z
    xy = x * y;  xz = x * z;  yz = y * z
    wx = w * x;  wy = w * y;  wz = w * z

    R = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to unit quaternion [w, x, y, z].

    Parameters
    ----------
    R : (..., 3, 3) rotation matrix

    Returns
    -------
    (..., 4) unit quaternion  — [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1, min=_EPS)) * 2  # s = 4*w
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], min=_EPS)) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2[mask2]

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2], min=_EPS)) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3[mask3]

    # Case 4: else
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1], min=_EPS)) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    q = normalize_quat(q)
    return q.reshape(batch_shape + (4,))


# ─── rot6d helpers (Zhou et al. CVPR 2019) ──────────────────────────────────

def rot6d_to_rotmat(r6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix.

    Parameters
    ----------
    r6d : (..., 6) tensor  — first two columns of the rotation matrix (a, b)

    Returns
    -------
    (..., 3, 3) rotation matrix (orthonormal, det ≈ +1)
    """
    a = r6d[..., :3]
    b = r6d[..., 3:]

    r1 = F.normalize(a, dim=-1)
    # Gram-Schmidt: b - (r1·b)r1
    dot = (r1 * b).sum(dim=-1, keepdim=True)
    r2 = F.normalize(b - dot * r1, dim=-1)
    r3 = torch.cross(r1, r2, dim=-1)

    R = torch.stack([r1, r2, r3], dim=-1)  # (..., 3, 3) columns are r1,r2,r3
    return R


# ─── SE(3) construction ─────────────────────────────────────────────────────

def make_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Assemble rotation matrix and translation into a 4×4 rigid transform.

    Parameters
    ----------
    R : (..., 3, 3)
    t : (..., 3)

    Returns
    -------
    (..., 4, 4) homogeneous rigid transformation matrix
    """
    batch_shape = R.shape[:-2]
    T = torch.zeros(batch_shape + (4, 4), device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ─── Param → SE(3) dispatchers ──────────────────────────────────────────────

def rotation_rep_to_rotmat(rot_param: torch.Tensor, rotation_rep: str) -> torch.Tensor:
    """Convert rotation parameters to rotation matrix based on representation.

    Parameters
    ----------
    rot_param : (..., k) where k depends on rotation_rep
        - "quat": k=4  [w, x, y, z]
        - "rot6d": k=6  [a1, a2, a3, b1, b2, b3]
        - "se3_expmap": k=3  [rx, ry, rz] Euler angles (ZYX)
    rotation_rep : one of {"quat", "rot6d", "se3_expmap"}
    """
    if rotation_rep == "quat":
        return quat_to_rotmat(normalize_quat(rot_param))
    elif rotation_rep == "rot6d":
        return rot6d_to_rotmat(rot_param)
    elif rotation_rep == "se3_expmap":
        return _euler_zyx_to_rotmat(rot_param)
    else:
        raise ValueError(f"Unknown rotation_rep: {rotation_rep}")


def _euler_zyx_to_rotmat(params: torch.Tensor) -> torch.Tensor:
    """Convert ZYX Euler angles to rotation matrix.

    Matches the existing ``PredictionTransform.param_to_transform`` convention.

    Parameters
    ----------
    params : (..., 3) — [rz, ry, rx]
    """
    cos_x = torch.cos(params[..., 2])
    sin_x = torch.sin(params[..., 2])
    cos_y = torch.cos(params[..., 1])
    sin_y = torch.sin(params[..., 1])
    cos_z = torch.cos(params[..., 0])
    sin_z = torch.sin(params[..., 0])

    r00 = cos_y * cos_z
    r01 = sin_x * sin_y * cos_z - cos_x * sin_z
    r02 = cos_x * sin_y * cos_z + sin_x * sin_z
    r10 = cos_y * sin_z
    r11 = sin_x * sin_y * sin_z + cos_x * cos_z
    r12 = cos_x * sin_y * sin_z - sin_x * cos_z
    r20 = -sin_y
    r21 = sin_x * cos_y
    r22 = cos_x * cos_y

    R = torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    return R.reshape(params.shape[:-1] + (3, 3))


# ─── Rotation-specific output dimensions ────────────────────────────────────

ROTATION_REP_DIM = {
    "se3_expmap": 6,   # 3 euler + 3 trans
    "quat": 7,         # 4 quat + 3 trans
    "rot6d": 9,        # 6 rot6d + 3 trans
}


def get_pose_output_dim(rotation_rep: str, num_pairs: int) -> int:
    """Compute total output dimension for the pose head.

    Returns per-pair dim × num_pairs.
    """
    per_pair = ROTATION_REP_DIM[rotation_rep]
    return per_pair * num_pairs
