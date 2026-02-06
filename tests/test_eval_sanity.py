import torch

from trainers.metrics import compute_tusrec_metrics


def _make_transforms(num_frames: int, shift_mm: float = 0.0) -> torch.Tensor:
    transforms = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    if shift_mm != 0.0:
        transforms[:, 0, 3] += shift_mm
    return transforms


def test_tusrec_gt_zero_and_shift_growth():
    num_frames = 3
    frames = torch.zeros((num_frames, 8, 8))
    tform_calib = torch.eye(4)
    gt = _make_transforms(num_frames)

    metrics_gt = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=gt,
        calib={"tform_calib": tform_calib},
        compute_scores=False,
        chunk_rows=4,
    )
    assert metrics_gt["GPE_mm"] == 0.0
    assert metrics_gt["LPE_mm"] == 0.0

    pred_shift = _make_transforms(num_frames, shift_mm=10.0)
    metrics_shift = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=pred_shift,
        calib={"tform_calib": tform_calib},
        compute_scores=False,
        chunk_rows=4,
    )
    assert metrics_shift["GPE_mm"] >= 9.0
    assert metrics_shift["LPE_mm"] >= 9.0
