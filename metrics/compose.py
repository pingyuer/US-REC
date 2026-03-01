"""Authoritative local <-> global transform composition.

This module is the **single source of truth** for accumulating local
pair-wise transforms into global frame-0 transforms and vice versa.

Every call-site in the repository **must** import from here (or via
``metrics`` / ``trainers.metrics`` which re-export these symbols).
No other file is allowed to re-implement the accumulation loop.

Transform conventions
---------------------
* **Global transform** ``T_{0<-i}``  (aliased ``global_T[i]``):
  maps a point in frame *i* coordinates to frame-0 coordinates.
  ``global_T[0] = I``.

* **Local transform** ``T_{i-1<-i}``  (aliased ``T_prev_from_curr[i]``):
  maps a point in frame *i* to frame *i-1*.
  ``local_T[0] = I`` (identity placeholder).
"""

from __future__ import annotations

import torch

__all__ = ["compose_global_from_local", "local_from_global"]


# ------------------------------------------------------------------
# local -> global
# ------------------------------------------------------------------

def compose_global_from_local(
    local_T: torch.Tensor,
    convention: str = "prev_from_curr",
) -> torch.Tensor:
    """Accumulate local pair transforms into global frame-0 transforms.

    This is the **single authoritative implementation** for
    local -> global conversion.  Every call-site in the repo must use
    this function.

    Parameters
    ----------
    local_T : Tensor of shape ``(T, 4, 4)`` or ``(B, T, 4, 4)``
        Per-frame local rigid transforms.  ``local_T[..., 0, :, :]``
        **must** be the identity (frame-0 has no predecessor).
    convention : {"prev_from_curr", "curr_from_prev"}
        * ``"prev_from_curr"`` — ``local_T[i] = T_{i-1 <- i}``
          (maps frame *i* **into** frame *i-1*).  This is what the
          tracker / label pipeline produces.
        * ``"curr_from_prev"`` — ``local_T[i] = T_{i <- i-1}``
          (maps frame *i-1* into frame *i*).  Will be inverted
          internally before accumulation.

    Returns
    -------
    global_T : Tensor, same shape as *local_T*
        ``global_T[..., i, :, :] = T_{0 <- i}``  with ``global_T[..., 0] = I``.

    Accumulation rule (for ``prev_from_curr``):
        ``global[0] = I``
        ``global[i] = global[i-1] @ local[i]``   for i >= 1
    """
    if convention not in ("prev_from_curr", "curr_from_prev"):
        raise ValueError(f"Unknown convention: {convention!r}")

    batched = local_T.ndim == 4
    if not batched:
        local_T = local_T.unsqueeze(0)  # (1, T, 4, 4)

    B, T, _, _ = local_T.shape
    if T == 0:
        return local_T.clone()

    # If the inputs are curr_from_prev, invert to get prev_from_curr.
    if convention == "curr_from_prev":
        local_pfc = local_T.clone()
        local_pfc[:, 1:] = torch.linalg.inv(local_T[:, 1:])
    else:
        local_pfc = local_T  # already prev_from_curr

    # Accumulate: global[i] = global[i-1] @ local_pfc[i]
    global_T = torch.zeros_like(local_pfc)
    global_T[:, 0] = torch.eye(4, device=local_T.device, dtype=local_T.dtype)
    for i in range(1, T):
        global_T[:, i] = torch.matmul(global_T[:, i - 1], local_pfc[:, i])

    if not batched:
        global_T = global_T.squeeze(0)
    return global_T


# ------------------------------------------------------------------
# global -> local
# ------------------------------------------------------------------

def local_from_global(transforms: torch.Tensor) -> torch.Tensor:
    """Derive local ``T_{i-1<-i}`` (prev_from_curr) from global ``T_{0<-i}``.

    Parameters
    ----------
    transforms : (T, 4, 4) or (B, T, 4, 4) — global transforms with
        ``transforms[..., 0] = I``.

    Returns
    -------
    local : same shape as input
        ``local[..., 0] = I``,
        ``local[..., i] = inv(global[i-1]) @ global[i]``
        i.e. ``T_{i-1 <- i}`` (prev_from_curr).
    """
    batched = transforms.ndim == 4
    if not batched:
        transforms = transforms.unsqueeze(0)

    B, T = transforms.shape[:2]
    if T <= 1:
        out = transforms.clone()
        if not batched:
            out = out.squeeze(0)
        return out

    inv_prev = torch.linalg.inv(transforms[:, :-1])
    local_tail = torch.matmul(inv_prev, transforms[:, 1:])
    identity = torch.eye(4, device=transforms.device, dtype=transforms.dtype)
    identity = identity.unsqueeze(0).unsqueeze(0).expand(B, 1, 4, 4)
    local = torch.cat([identity, local_tail], dim=1)

    if not batched:
        local = local.squeeze(0)
    return local
