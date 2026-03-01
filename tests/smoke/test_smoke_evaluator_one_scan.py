"""Smoke test 4: evaluator on 1 synthetic scan (few frames).

Constructs a fake scan with known transforms, runs through
``compute_tusrec_metrics``, and checks that the output JSON fields
contain the expected keys.
"""

import pytest
import torch

from metrics.compose import compose_global_from_local
from trainers.metrics.tusrec import compute_tusrec_metrics


def _make_synthetic_scan(num_frames: int = 10, H: int = 32, W: int = 32):
    """Build a tiny deterministic scan for smoke testing."""
    torch.manual_seed(0)

    frames = torch.randint(0, 255, (num_frames, H, W), dtype=torch.float32)

    # Local transforms: small +0.5mm x-shift per frame
    local_gt = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
    for i in range(1, num_frames):
        local_gt[i, 0, 3] = 0.5

    global_gt = compose_global_from_local(local_gt)

    # Predicted: add small noise to the local transforms
    local_pred = local_gt.clone()
    local_pred[1:, 0, 3] += torch.randn(num_frames - 1) * 0.05
    global_pred = compose_global_from_local(local_pred)

    # Simple identity calibration (pixel = mm, no rotation)
    calib = torch.eye(4)
    calib[0, 0] = 0.1  # 0.1 mm/pixel
    calib[1, 1] = 0.1

    return frames, global_gt, global_pred, calib


@pytest.mark.smoke
class TestSmokeEvaluatorOneScan:
    """Run compute_tusrec_metrics on a single synthetic scan."""

    def test_metrics_fields(self):
        frames, gt_global, pred_global, calib = _make_synthetic_scan(num_frames=10)

        result = compute_tusrec_metrics(
            frames=frames,
            gt_transforms=gt_global,
            pred_transforms=pred_global,
            calib={"tform_calib": calib},
            compute_scores=True,
        )

        # Required fields
        for key in ("GPE_mm", "LPE_mm", "final_score"):
            assert key in result, f"Missing field: {key}"

        # GPE/LPE should be finite positive (we added noise)
        assert result["GPE_mm"] is not None and result["GPE_mm"] >= 0
        assert result["LPE_mm"] is not None and result["LPE_mm"] >= 0
        assert result["final_score"] is not None

    def test_identity_pred_gives_zero_error(self):
        """If pred == gt, errors should be ~0."""
        frames, gt_global, _, calib = _make_synthetic_scan(num_frames=5)

        result = compute_tusrec_metrics(
            frames=frames,
            gt_transforms=gt_global,
            pred_transforms=gt_global.clone(),
            calib={"tform_calib": calib},
        )

        assert result["GPE_mm"] is not None
        assert result["GPE_mm"] < 1e-4, f"GPE not near zero: {result['GPE_mm']}"
        assert result["LPE_mm"] is not None
        assert result["LPE_mm"] < 1e-4, f"LPE not near zero: {result['LPE_mm']}"

    def test_runtime_field(self):
        """If runtime_s is provided, it flows through."""
        frames, gt_global, pred_global, calib = _make_synthetic_scan(num_frames=5)

        result = compute_tusrec_metrics(
            frames=frames,
            gt_transforms=gt_global,
            pred_transforms=pred_global,
            calib={"tform_calib": calib},
            runtime_s=1.23,
        )
        assert result.get("runtime_s_per_scan") == pytest.approx(1.23)
