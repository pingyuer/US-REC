"""TUS-REC metrics with explicit global/local semantics and score outputs."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

_COORD_PRINTED = False


def _ensure_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device)
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
    xy = torch.cartesian_prod(x_vals, y_vals).t()
    zeros = torch.zeros((1, xy.shape[1]), device=xy.device, dtype=xy.dtype)
    ones = torch.ones((1, xy.shape[1]), device=xy.device, dtype=xy.dtype)
    return torch.cat([xy, zeros, ones], dim=0)


def _mean_point_error(
    *,
    gt_transforms: torch.Tensor,
    pred_transforms: torch.Tensor,
    tform_calib: torch.Tensor,
    image_size: Tuple[int, int],
    chunk_rows: int,
    start_frame: int,
    image_points: Optional[torch.Tensor] = None,
) -> float:
    """Mean 3D point error (mm) from start_frame..N-1."""
    num_frames = int(gt_transforms.shape[0])
    if num_frames <= 1 or start_frame >= num_frames:
        return 0.0

    device = gt_transforms.device
    total = torch.tensor(0.0, device=device, dtype=gt_transforms.dtype)
    count = 0

    if image_points is not None:
        points = image_points.to(device=device, dtype=gt_transforms.dtype)
        if points.ndim != 2 or points.shape[0] != 4:
            raise ValueError("image_points must have shape (4, K) in homogeneous pixel coordinates.")
        points_tool = torch.matmul(tform_calib, points)
        gt_pts = torch.matmul(gt_transforms, points_tool)[:, 0:3, :]
        pred_pts = torch.matmul(pred_transforms, points_tool)[:, 0:3, :]
        dist = torch.linalg.norm(pred_pts - gt_pts, dim=1)
        dist = dist[start_frame:, :]
        total += dist.sum()
        count += int(dist.numel())
    else:
        height, width = image_size
        x_full = torch.arange(0, height, device=device, dtype=gt_transforms.dtype)
        y_full = torch.arange(0, width, device=device, dtype=gt_transforms.dtype)
        for start in range(0, height, int(chunk_rows)):
            x_vals = x_full[start : start + int(chunk_rows)]
            points = _make_points_grid(x_vals, y_full)
            points_tool = torch.matmul(tform_calib, points)
            gt_pts = torch.matmul(gt_transforms, points_tool)[:, 0:3, :]
            pred_pts = torch.matmul(pred_transforms, points_tool)[:, 0:3, :]
            dist = torch.linalg.norm(pred_pts - gt_pts, dim=1)
            dist = dist[start_frame:, :]
            total += dist.sum()
            count += int(dist.numel())
    if count == 0:
        return 0.0
    return float((total / count).item())


def _group_landmarks(landmarks: torch.Tensor, num_frames: int) -> list[torch.Tensor]:
    grouped: list[torch.Tensor] = [torch.empty((0, 2), device=landmarks.device) for _ in range(num_frames)]
    if landmarks.ndim == 2:
        if landmarks.shape[1] < 3:
            return grouped
        frame_idx = landmarks[:, 0].long()
        coords = landmarks[:, 1:3]
        valid = (frame_idx >= 0) & (frame_idx < num_frames)
        frame_idx = frame_idx[valid]
        coords = coords[valid]
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
    start_frame: int,
) -> Optional[float]:
    if landmarks is None or landmarks.numel() == 0:
        return None

    num_frames = int(gt_transforms.shape[0])
    grouped = _group_landmarks(landmarks, num_frames)
    device = gt_transforms.device
    total = torch.tensor(0.0, device=device, dtype=gt_transforms.dtype)
    count = 0
    for frame_idx in range(max(0, int(start_frame)), num_frames):
        coords = grouped[frame_idx]
        if coords.numel() == 0:
            continue
        points = torch.cat(
            [
                coords.t().to(dtype=gt_transforms.dtype),
                torch.zeros((1, coords.shape[0]), device=device, dtype=gt_transforms.dtype),
                torch.ones((1, coords.shape[0]), device=device, dtype=gt_transforms.dtype),
            ],
            dim=0,
        )
        points_tool = torch.matmul(tform_calib, points)
        gt_pts = torch.matmul(gt_transforms[frame_idx], points_tool)[0:3, :]
        pred_pts = torch.matmul(pred_transforms[frame_idx], points_tool)[0:3, :]
        dist = torch.linalg.norm(pred_pts - gt_pts, dim=0)
        total += dist.sum()
        count += int(dist.numel())
    if count == 0:
        return None
    return float((total / count).item())


def _local_from_global(transforms: torch.Tensor) -> torch.Tensor:
    """Convert T_i (frame i -> frame 0) to L_i (frame i -> frame i-1)."""
    if transforms.shape[0] <= 1:
        return transforms.clone()
    inv_prev = torch.linalg.inv(transforms[:-1])
    local = torch.matmul(inv_prev, transforms[1:])
    identity = torch.eye(4, device=transforms.device, dtype=transforms.dtype).unsqueeze(0)
    return torch.cat([identity, local], dim=0)


def _score_from_error(metric: Optional[float], largest: Optional[float], *, eps: float = 1e-8) -> Optional[float]:
    if metric is None or largest is None:
        return None
    score = 1.0 - (float(metric) / (float(largest) + float(eps)))
    return float(max(0.0, min(1.0, score)))


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
    compute_scores: bool = True,
    enforce_lp_gp_distinct: bool = False,
    distinct_tol: float = 1e-3,
) -> Dict[str, Optional[float]]:
    """
    Compute TUS-REC metrics for one scan.

    Global transforms: T_i maps frame i -> frame 0.
    Local transforms:  L_i maps frame i -> frame i-1 with L_0 = I.
    """
    device = gt_transforms.device
    tform_calib = _ensure_tensor(calib["tform_calib"], device=device)
    frames_t = _ensure_tensor(frames, device=device)
    image_size = _infer_image_size(frames_t, image_size)

    points = _ensure_tensor(image_points, device=device) if image_points is not None else None
    if points is not None:
        h, w = image_size
        max_x = float(points[0].max().item())
        max_y = float(points[1].max().item())
        if max_x > (h - 1) + 1e-6 or max_y > (w - 1) + 1e-6:
            raise ValueError(
                f"image_points out of bounds for image_size={image_size}: "
                f"max_x={max_x}, max_y={max_y}"
            )

    gt_global = _ensure_tensor(gt_transforms, device=device)
    pred_global = _ensure_tensor(pred_transforms, device=device)
    gt_local = _local_from_global(gt_global)
    pred_local = _local_from_global(pred_global)

    global _COORD_PRINTED
    if not _COORD_PRINTED:
        _COORD_PRINTED = True
        trans = tform_calib[:3, 3].detach().cpu().tolist()
        diag = torch.diagonal(tform_calib[:3, :3]).detach().cpu().tolist()
        if points is not None:
            pmin = points[:2, :].min(dim=1).values.detach().cpu().tolist()
            pmax = points[:2, :].max(dim=1).values.detach().cpu().tolist()
            print(
                f"[tusrec] image_points xy min/max={pmin}/{pmax}, "
                f"calib diag={diag}, trans={trans}"
            )
        else:
            print(
                f"[tusrec] dense grid over image_size(H,W)={image_size}, "
                f"calib diag={diag}, trans={trans}"
            )

    # Global metrics evaluate frame i->0 for i>=1.
    gpe = _mean_point_error(
        gt_transforms=gt_global,
        pred_transforms=pred_global,
        tform_calib=tform_calib,
        image_size=image_size,
        chunk_rows=chunk_rows,
        start_frame=1,
        image_points=points,
    )
    gle = _mean_landmark_error(
        gt_transforms=gt_global,
        pred_transforms=pred_global,
        tform_calib=tform_calib,
        landmarks=landmarks,
        start_frame=1,
    )

    # Local metrics evaluate frame i->i-1 for i>=1.
    lpe = _mean_point_error(
        gt_transforms=gt_local,
        pred_transforms=pred_local,
        tform_calib=tform_calib,
        image_size=image_size,
        chunk_rows=chunk_rows,
        start_frame=1,
        image_points=points,
    )
    lle = _mean_landmark_error(
        gt_transforms=gt_local,
        pred_transforms=pred_local,
        tform_calib=tform_calib,
        landmarks=landmarks,
        start_frame=1,
    )

    gpe_score = gle_score = lpe_score = lle_score = None
    final_score = None
    if compute_scores:
        identity = torch.eye(4, device=device, dtype=gt_global.dtype).unsqueeze(0).repeat(
            gt_global.shape[0], 1, 1
        )
        gpe_largest = _mean_point_error(
            gt_transforms=gt_global,
            pred_transforms=identity,
            tform_calib=tform_calib,
            image_size=image_size,
            chunk_rows=chunk_rows,
            start_frame=1,
            image_points=points,
        )
        lpe_largest = _mean_point_error(
            gt_transforms=gt_local,
            pred_transforms=identity,
            tform_calib=tform_calib,
            image_size=image_size,
            chunk_rows=chunk_rows,
            start_frame=1,
            image_points=points,
        )
        gle_largest = _mean_landmark_error(
            gt_transforms=gt_global,
            pred_transforms=identity,
            tform_calib=tform_calib,
            landmarks=landmarks,
            start_frame=1,
        )
        lle_largest = _mean_landmark_error(
            gt_transforms=gt_local,
            pred_transforms=identity,
            tform_calib=tform_calib,
            landmarks=landmarks,
            start_frame=1,
        )

        gpe_score = _score_from_error(gpe, gpe_largest)
        gle_score = _score_from_error(gle, gle_largest)
        lpe_score = _score_from_error(lpe, lpe_largest)
        lle_score = _score_from_error(lle, lle_largest)
        available = [v for v in (gpe_score, gle_score, lpe_score, lle_score) if v is not None]
        if available:
            final_score = float(max(0.0, min(1.0, sum(available) / len(available))))

    if (
        enforce_lp_gp_distinct
        and gpe is not None
        and lpe is not None
        and float(max(abs(gpe), abs(lpe))) > 1e-6
        and abs(float(gpe) - float(lpe)) < float(distinct_tol)
    ):
        raise ValueError(
            f"GPE and LPE are unexpectedly close (GPE={gpe:.6f}, LPE={lpe:.6f}); "
            "check LP/GP definition and transform semantics."
        )

    return {
        "GPE_mm": gpe,
        "GLE_mm": gle,
        "LPE_mm": lpe,
        "LLE_mm": lle,
        "GPE_score": gpe_score,
        "GLE_score": gle_score,
        "LPE_score": lpe_score,
        "LLE_score": lle_score,
        "final_score": final_score,
        "runtime_s_per_scan": float(runtime_s) if runtime_s is not None else None,
    }


__all__ = ["compute_tusrec_metrics"]
