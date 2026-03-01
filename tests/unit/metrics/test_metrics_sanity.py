import pytest
pytestmark = pytest.mark.unit

# tests/test_metrics_sanity.py
import math

import pytest
import torch

from trainers.metrics.ddf import ddf_epe_mm, ddf_epe_vox
from trainers.metrics.pose import rotation_error_deg
from trainers.metrics.trajectory import endpoint_rpe_translation_mm, rpe_translation_mm


def _make_T(translation):
    t = torch.tensor(translation, dtype=torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, 3] = t
    return T


def test_ddf_epe_vox_and_mm():
    pred = torch.zeros(1, 3, 2, 2, 2)
    gt = torch.zeros_like(pred)
    assert ddf_epe_vox(pred, gt) == 0
    pred[:, 0] += 1.0
    assert torch.allclose(ddf_epe_vox(pred, gt), torch.tensor(1.0))
    assert torch.allclose(ddf_epe_mm(pred, gt, spacing=(2.0, 2.0, 2.0)), torch.tensor(2.0))


def test_rotation_error_small_angle_and_invalid_warns():
    angle_deg = 10.0
    angle = math.radians(angle_deg)
    rot = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pred = rot.unsqueeze(0)
    gt = torch.eye(3).unsqueeze(0)
    err = rotation_error_deg(pred, gt)
    assert torch.allclose(err, torch.tensor([angle_deg]), atol=1e-3)

    bad = pred.clone()
    bad[0, 0, 0] = 2.0
    with pytest.warns(RuntimeWarning):
        rotation_error_deg(bad, gt, check_valid=True)


def test_rpe_translation_delta1():
    gt = torch.stack([_make_T([0.0, 0.0, 0.0]), _make_T([1.0, 0.0, 0.0])], dim=0)
    pred = torch.stack([_make_T([0.0, 0.0, 0.0]), _make_T([11.0, 0.0, 0.0])], dim=0)
    gt = gt.unsqueeze(0)
    pred = pred.unsqueeze(0)
    endpoint = endpoint_rpe_translation_mm(pred, gt)
    rpe = rpe_translation_mm(pred, gt, delta=1)
    assert torch.allclose(endpoint, torch.tensor([10.0]))
    assert torch.allclose(rpe, torch.tensor([10.0]))
