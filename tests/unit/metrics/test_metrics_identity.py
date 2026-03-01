import pytest
pytestmark = pytest.mark.unit

import torch

from trainers.metrics import compute_tusrec_metrics


def _make_transforms(num_frames: int) -> torch.Tensor:
    transforms = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    for idx in range(1, num_frames):
        transforms[idx, 0, 3] = float(idx) * 2.0
    return transforms


def test_tusrec_metrics_identity_and_largest():
    num_frames = 4
    frames = torch.zeros((num_frames, 8, 8))
    tform_calib = torch.eye(4)
    gt = _make_transforms(num_frames)

    metrics_gt = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=gt,
        calib={"tform_calib": tform_calib},
        compute_scores=True,
        chunk_rows=4,
    )
    assert metrics_gt["GPE_mm"] == 0.0
    assert metrics_gt["LPE_mm"] == 0.0

    identity = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    metrics_identity = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=identity,
        calib={"tform_calib": tform_calib},
        compute_scores=True,
        chunk_rows=4,
    )
    assert metrics_identity["GPE_score"] < 1e-6
    assert metrics_identity["LPE_score"] < 1e-6
    if metrics_identity["final_score"] is not None:
        assert 0.0 <= metrics_identity["final_score"] <= 1.0
