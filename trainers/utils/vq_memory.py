"""Utilities for mask-aware VQ-memory batching.

These helpers keep scan-specific padding / masking logic out of
``VQMemoryTrainer`` so the training loop stays focused on optimisation.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from models.vq.scan_geom_head import build_geom_target


def valid_length(mask_row: torch.Tensor | None, fallback: int) -> int:
    """Return the number of valid time steps from a boolean mask."""
    if mask_row is None:
        return int(fallback)
    return max(1, int(mask_row.long().sum().item()))


def pad_sequence_list(
    tensors: list[torch.Tensor],
    *,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length tensors on dim 0 and return the validity mask."""
    if not tensors:
        raise ValueError("Expected at least one tensor to pad.")

    max_len = max(int(t.shape[0]) for t in tensors)
    out_shape = (len(tensors), max_len) + tuple(tensors[0].shape[1:])
    out = tensors[0].new_full(out_shape, pad_value)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool, device=tensors[0].device)
    for i, tensor in enumerate(tensors):
        length = int(tensor.shape[0])
        out[i, :length] = tensor
        mask[i, :length] = True
    return out, mask


def build_batched_scan_cache(
    model,
    scan_frames: torch.Tensor,
    scan_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Encode padded full scans into a batchable VQ-memory cache."""
    z_q_list: list[torch.Tensor] = []
    z_q_detached_list: list[torch.Tensor] = []
    h_all_list: list[torch.Tensor] = []
    vq_indices_list: list[torch.Tensor] = []
    anchor_pos_list: list[torch.Tensor] = []
    commit_losses: list[torch.Tensor] = []

    B = scan_frames.shape[0]
    for b in range(B):
        scan_len = valid_length(scan_mask[b] if scan_mask is not None else None, scan_frames.shape[1])
        cache_b = model.encode_scan_anchors(scan_frames[b:b + 1, :scan_len])
        z_q_list.append(cache_b["z_q"].squeeze(0))
        z_q_detached_list.append(cache_b["z_q_detached"].squeeze(0))
        h_all_list.append(cache_b["h_all"].squeeze(0))
        vq_indices_list.append(cache_b["vq_indices"].squeeze(0))
        if "anchor_pos" in cache_b:
            anchor_pos_list.append(cache_b["anchor_pos"].squeeze(0))
        else:
            if "anchor_idx" in cache_b:
                idx = cache_b["anchor_idx"].to(cache_b["z_q"].dtype)
                denom = float(max(1, scan_len - 1))
                anchor_pos = (idx / denom).reshape(-1)
            else:
                m = int(cache_b["z_q"].shape[1])
                anchor_pos = torch.linspace(
                    0.0, 1.0, max(1, m),
                    device=cache_b["z_q"].device,
                    dtype=cache_b["z_q"].dtype,
                )
            anchor_pos_list.append(anchor_pos)
        commit_losses.append(cache_b["commit_loss"])

    z_q, memory_mask = pad_sequence_list(z_q_list)
    z_q_detached, _ = pad_sequence_list(z_q_detached_list)
    h_all, scan_valid_mask = pad_sequence_list(h_all_list)
    vq_indices, _ = pad_sequence_list(vq_indices_list, pad_value=-1)
    anchor_pos, _ = pad_sequence_list(anchor_pos_list, pad_value=0.0)
    z_ctx = model.contextualize_memory(z_q, mask=memory_mask)

    return {
        "z_q": z_q,
        "z_ctx": z_ctx,
        "z_q_detached": z_q_detached,
        "g": model.summary_pool(z_ctx, mask=memory_mask),
        "memory_mask": memory_mask,
        "commit_loss": torch.stack(commit_losses).mean(),
        "vq_indices": vq_indices.long(),
        "anchor_pos": anchor_pos,
        "h_all": h_all,
        "scan_mask": scan_valid_mask,
    }


def build_consistency_views(
    model,
    *,
    anchor_stride: int,
    scan_frames: torch.Tensor,
    ratio: float = 0.6,
    h_all: torch.Tensor | None = None,
    scan_mask: torch.Tensor | None = None,
    rng: random.Random | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build two random anchor-subset views for summary consistency."""
    if h_all is None:
        h_all = model.encoder.encode_sequence(scan_frames)

    rng = rng or random.Random()
    views_1: list[torch.Tensor] = []
    views_2: list[torch.Tensor] = []
    commit_1: list[torch.Tensor] = []
    commit_2: list[torch.Tensor] = []

    def _make_view(h_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        valid_steps = h_valid.shape[1]
        all_idx = list(range(0, valid_steps, anchor_stride))
        n_anchors = len(all_idx)
        n_sample = max(2, int(n_anchors * ratio))
        selected = sorted(rng.sample(all_idx, min(n_sample, n_anchors)))
        idx_t = torch.tensor(selected, device=h_valid.device)
        h_anchors = h_valid[:, idx_t]
        vq_out = model.vq(model.proj_vq(h_anchors))
        z_ctx = model.contextualize_memory(vq_out["z_q"])
        return {
            "g": model.summary_pool(z_ctx),
            "commit_loss": vq_out["commit_loss"],
            "anchor_idx": idx_t,
        }

    B = h_all.shape[0]
    for b in range(B):
        scan_len = valid_length(scan_mask[b] if scan_mask is not None else None, h_all.shape[1])
        h_valid = h_all[b:b + 1, :scan_len]
        view1 = _make_view(h_valid)
        view2 = _make_view(h_valid)
        views_1.append(view1["g"].squeeze(0))
        views_2.append(view2["g"].squeeze(0))
        commit_1.append(view1["commit_loss"])
        commit_2.append(view2["commit_loss"])

    return (
        {"g": torch.stack(views_1, dim=0), "commit_loss": torch.stack(commit_1).mean()},
        {"g": torch.stack(views_2, dim=0), "commit_loss": torch.stack(commit_2).mean()},
    )


def build_masked_geom_targets(
    scan_gt: torch.Tensor,
    n_waypoints: int,
    scan_gt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build per-scan geometry targets while ignoring padded suffixes."""
    targets: list[torch.Tensor] = []
    B = scan_gt.shape[0]
    for b in range(B):
        scan_len = valid_length(scan_gt_mask[b] if scan_gt_mask is not None else None, scan_gt.shape[1])
        targets.append(
            build_geom_target(scan_gt[b:b + 1, :scan_len], n_waypoints=n_waypoints).squeeze(0)
        )
    return torch.stack(targets, dim=0)