"""TUS-REC metric helpers (rigid-only compatible)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def _ensure_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, device=device)
    return torch.as_tensor(value, device=device)


def _infer_image_size(frames: torch.Tensor, image_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if image_size is not None:
        return int(image_size[0]), int(image_size[1])
    if frames.ndim == 3:
        return int(frames.shape[1]), int(frames.shape[2])
    if frames.ndim == 4 and frames.shape[1] in {1, 3}:
        return int(frames.shape[2]), int(frames.shape[3])
    raise ValueError("frames must be (N,H,W) or (N,C,H,W) when image_size is not provided")


def _make_points_grid(x_vals: torch.Tensor, y_vals: torch.Tensor) -> torch.Tensor:
    grid = torch.cartesian_prod(x_vals, y_vals).t()
    zeros = torch.zeros((1, grid.shape[1]), device=grid.device)
    ones = torch.ones((1, grid.shape[1]), device=grid.device)
    return torch.cat([grid, zeros, ones], dim=0)


def _mean_point_error(
    *,
    gt_transforms: torch.Tensor,
    pred_transforms: torch.Tensor,
    tform_calib: torch.Tensor,
    image_size: Tuple[int, int],
    chunk_rows: int,
) -> float:
    num_frames = int(gt_transforms.shape[0])
    if num_frames <= 1:
        return 0.0

    height, width = image_size
    device = gt_transforms.device
    x_full = torch.linspace(0, height, height, device=device)
    y_full = torch.linspace(0, width, width, device=device)

    total = torch.tensor(0.0, device=device)
    count = 0
    for start in range(0, height, chunk_rows):
        x_vals = x_full[start : start + chunk_rows]
        points = _make_points_grid(x_vals, y_full)
        points_tool = torch.matmul(tform_calib, points)

        gt_pts = torch.matmul(gt_transforms, points_tool)[:, 0:3, :]
        pred_pts = torch.matmul(pred_transforms, points_tool)[:, 0:3, :]
        diff = pred_pts - gt_pts
        dist = torch.sqrt((diff ** 2).sum(dim=1))
        dist = dist[1:, :]
        total += dist.sum()
        count += dist.numel()

    if count == 0:
        return 0.0
    return float((total / count).item())


def _group_landmarks(
    landmarks: torch.Tensor, num_frames: int
) -> list[torch.Tensor]:
    grouped: list[torch.Tensor] = [torch.empty((0, 2), device=landmarks.device) for _ in range(num_frames)]
    if landmarks.ndim == 2:
        if landmarks.shape[1] < 3:
            return grouped
        frame_idx = landmarks[:, 0].long()
        coords = landmarks[:, 1:3]
        for idx in range(num_frames):
            mask = frame_idx == idx
            if mask.any():
                grouped[idx] = coords[mask]
        return grouped
    if landmarks.ndim == 3:
        for idx in range(min(num_frames, landmarks.shape[0])):
            coords = landmarks[idx]
            if coords.numel() == 0:
                continue
            if coords.shape[-1] >= 2:
                grouped[idx] = coords[:, 0:2]
        return grouped
    return grouped


def _mean_landmark_error(
    *,
    gt_transforms: torch.Tensor,
    pred_transforms: torch.Tensor,
    tform_calib: torch.Tensor,
    landmarks: Optional[torch.Tensor],
) -> Optional[float]:
    if landmarks is None or landmarks.numel() == 0:
        return None

    num_frames = int(gt_transforms.shape[0])
    grouped = _group_landmarks(landmarks, num_frames)
    device = gt_transforms.device

    total = torch.tensor(0.0, device=device)
    count = 0
    for frame_idx in range(1, num_frames):
        coords = grouped[frame_idx]
        if coords.numel() == 0:
            continue
        points = torch.cat(
            [
                coords.t(),
                torch.zeros((1, coords.shape[0]), device=device),
                torch.ones((1, coords.shape[0]), device=device),
            ],
            dim=0,
        )
        points_tool = torch.matmul(tform_calib, points)
        gt_pts = torch.matmul(gt_transforms[frame_idx], points_tool)[0:3, :]
        pred_pts = torch.matmul(pred_transforms[frame_idx], points_tool)[0:3, :]
        dist = torch.sqrt(((pred_pts - gt_pts) ** 2).sum(dim=0))
        total += dist.sum()
        count += int(dist.numel())

    if count == 0:
        return None
    return float((total / count).item())


def _local_from_global(transforms: torch.Tensor) -> torch.Tensor:
    if transforms.shape[0] <= 1:
        return transforms.clone()
    inv_prev = torch.linalg.inv(transforms[:-1])
    local = torch.matmul(inv_prev, transforms[1:])
    identity = torch.eye(4, device=transforms.device, dtype=transforms.dtype).unsqueeze(0)
    return torch.cat([identity, local], dim=0)


def _normalized(metric: Optional[float], largest: Optional[float]) -> Optional[float]:
    if metric is None or largest is None:
        return None
    if largest == 0:
        return 0.0
    return float((metric - largest) / (0.0 - largest))


def compute_tusrec_metrics(
    *,
    frames: torch.Tensor,
    gt_transforms: torch.Tensor,
    pred_transforms: torch.Tensor,
    calib: Dict[str, Any],
    landmarks: Optional[torch.Tensor] = None,
    image_points: Optional[torch.Tensor] = None,
    image_size: Optional[Tuple[int, int]] = None,
    chunk_rows: int = 64,
    runtime_s: Optional[float] = None,
    compute_normalized: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute TUS-REC metrics for a single scan sequence.

    gt_transforms/pred_transforms: (N,4,4) mapping frame i -> frame 0.
    calib: expects "tform_calib" mapping image pixels to tool (mm) coordinates.
    landmarks: optional landmarks, either (K,3) [frame_idx,x,y] or (N,K,2/3).
    """
    device = gt_transforms.device
    tform_calib = _ensure_tensor(calib["tform_calib"], device=device)
    frames_t = _ensure_tensor(frames, device=device)
    image_size = _infer_image_size(frames_t, image_size)

    if image_points is not None:
        _ensure_tensor(image_points, device=device)

    gt_global = _ensure_tensor(gt_transforms, device=device)
    pred_global = _ensure_tensor(pred_transforms, device=device)
    gt_local = _local_from_global(gt_global)
    pred_local = _local_from_global(pred_global)

    gpe = _mean_point_error(
        gt_transforms=gt_global,
        pred_transforms=pred_global,
        tform_calib=tform_calib,
        image_size=image_size,
        chunk_rows=chunk_rows,
    )
    lpe = _mean_point_error(
        gt_transforms=gt_local,
        pred_transforms=pred_local,
        tform_calib=tform_calib,
        image_size=image_size,
        chunk_rows=chunk_rows,
    )
    gle = _mean_landmark_error(
        gt_transforms=gt_global,
        pred_transforms=pred_global,
        tform_calib=tform_calib,
        landmarks=landmarks,
    )
    lle = _mean_landmark_error(
        gt_transforms=gt_local,
        pred_transforms=pred_local,
        tform_calib=tform_calib,
        landmarks=landmarks,
    )

    gpe_norm = gle_norm = lpe_norm = lle_norm = None
    final_score = None
    if compute_normalized:
        identity = torch.eye(4, device=device, dtype=gt_global.dtype).unsqueeze(0).repeat(
            gt_global.shape[0], 1, 1
        )
        gpe_largest = _mean_point_error(
            gt_transforms=gt_global,
            pred_transforms=identity,
            tform_calib=tform_calib,
            image_size=image_size,
            chunk_rows=chunk_rows,
        )
        lpe_largest = _mean_point_error(
            gt_transforms=gt_local,
            pred_transforms=identity,
            tform_calib=tform_calib,
            image_size=image_size,
            chunk_rows=chunk_rows,
        )
        gle_largest = _mean_landmark_error(
            gt_transforms=gt_global,
            pred_transforms=identity,
            tform_calib=tform_calib,
            landmarks=landmarks,
        )
        lle_largest = _mean_landmark_error(
            gt_transforms=gt_local,
            pred_transforms=identity,
            tform_calib=tform_calib,
            landmarks=landmarks,
        )

        gpe_norm = _normalized(gpe, gpe_largest)
        gle_norm = _normalized(gle, gle_largest)
        lpe_norm = _normalized(lpe, lpe_largest)
        lle_norm = _normalized(lle, lle_largest)

        if None not in (gpe_norm, gle_norm, lpe_norm, lle_norm):
            final_score = float(0.25 * (gpe_norm + gle_norm + lpe_norm + lle_norm))

    return {
        "GPE_mm": gpe,
        "GLE_mm": gle,
        "LPE_mm": lpe,
        "LLE_mm": lle,
        "GPE_norm": gpe_norm,
        "GLE_norm": gle_norm,
        "LPE_norm": lpe_norm,
        "LLE_norm": lle_norm,
        "final_score": final_score,
        "runtime_s_per_scan": float(runtime_s) if runtime_s is not None else None,
    }
