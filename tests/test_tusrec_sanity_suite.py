import torch

from trainers.metrics import (
    compute_tusrec_metrics,
    endpoint_rpe_translation_mm,
    rpe_translation_mm,
    se3_translation_error_mm,
)
from trainers.metrics.tusrec import _local_from_global


def _make_seq(num_frames: int) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1, 1)
    for i in range(1, num_frames):
        T[i, 0, 3] = float(i) * 2.0
    return T


def test_gt_to_gt_metrics_and_scores():
    frames = torch.zeros((4, 8, 8), dtype=torch.float32)
    gt = _make_seq(4)
    metrics = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=gt,
        calib={"tform_calib": torch.eye(4)},
        compute_scores=True,
    )
    assert metrics["GPE_mm"] == 0.0
    assert metrics["LPE_mm"] == 0.0
    assert metrics["GPE_score"] == 1.0
    assert metrics["LPE_score"] == 1.0
    assert metrics["final_score"] == 1.0


def test_translation_plus_10mm_injection():
    gt = _make_seq(4)
    pred = gt.clone()
    pred[1:, 0, 3] += 10.0
    pred_b = pred.unsqueeze(0)
    gt_b = gt.unsqueeze(0)
    assert se3_translation_error_mm(pred_b, gt_b).mean().item() >= 9.0
    assert endpoint_rpe_translation_mm(pred_b, gt_b).mean().item() >= 9.0
    metrics = compute_tusrec_metrics(
        frames=torch.zeros((4, 8, 8), dtype=torch.float32),
        gt_transforms=gt,
        pred_transforms=pred,
        calib={"tform_calib": torch.eye(4)},
        compute_scores=True,
    )
    assert metrics["GPE_mm"] >= 9.0
    assert metrics["LPE_mm"] >= 2.5
    assert 0.0 <= float(metrics["GPE_score"]) <= 1.0
    assert 0.0 <= float(metrics["LPE_score"]) <= 1.0
    assert 0.0 <= float(metrics["final_score"]) <= 1.0


def test_local_injection_distinguishes_gpe_lpe_and_local_reconstruction():
    gt = _make_seq(5)
    pred = gt.clone()
    pred[3, 0, 3] += 7.0
    metrics = compute_tusrec_metrics(
        frames=torch.zeros((5, 8, 8), dtype=torch.float32),
        gt_transforms=gt,
        pred_transforms=pred,
        calib={"tform_calib": torch.eye(4)},
        compute_scores=False,
    )
    assert abs(float(metrics["GPE_mm"]) - float(metrics["LPE_mm"])) > 1e-3

    local = _local_from_global(gt)
    for i in range(1, gt.shape[0]):
        lhs = gt[i]
        rhs = gt[i - 1] @ local[i]
        assert torch.allclose(lhs, rhs, atol=1e-5)

    gt_b = gt.unsqueeze(0)
    pred_b = pred.unsqueeze(0)
    assert rpe_translation_mm(pred_b, gt_b, delta=1).mean().item() > 0.0


def test_chunk_rows_consistency():
    gt = _make_seq(4)
    pred = gt.clone()
    pred[2, 0, 3] += 3.0
    frames = torch.zeros((4, 32, 32), dtype=torch.float32)
    m16 = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=pred,
        calib={"tform_calib": torch.eye(4)},
        chunk_rows=16,
        compute_scores=True,
    )
    m64 = compute_tusrec_metrics(
        frames=frames,
        gt_transforms=gt,
        pred_transforms=pred,
        calib={"tform_calib": torch.eye(4)},
        chunk_rows=64,
        compute_scores=True,
    )
    for key in ("GPE_mm", "LPE_mm", "GPE_score", "LPE_score", "final_score"):
        assert abs(float(m16[key]) - float(m64[key])) < 1e-6
