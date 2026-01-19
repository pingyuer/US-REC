import torch

from trainers.metrics import (
    cumulative_drift,
    ddf_mae,
    ddf_rmse,
    loop_closure_error,
    rotation_error,
    se3_error,
    translation_error,
    volume_dice,
    volume_ncc,
    volume_ssim,
)


def _make_T(translation):
    t = torch.tensor(translation, dtype=torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    T[:3, 3] = t
    return T


def test_translation_error_zero():
    pred = torch.zeros(2, 3)
    gt = torch.zeros(2, 3)
    err = translation_error(pred, gt)
    assert torch.allclose(err, torch.zeros(2))


def test_rotation_error_identity():
    pred = torch.eye(3).repeat(2, 1, 1)
    gt = torch.eye(3).repeat(2, 1, 1)
    err = rotation_error(pred, gt)
    assert torch.allclose(err, torch.zeros(2))


def test_se3_error_nonzero():
    pred = _make_T([1.0, 0.0, 0.0]).unsqueeze(0)
    gt = _make_T([0.0, 0.0, 0.0]).unsqueeze(0)
    err = se3_error(pred, gt)
    assert err.item() > 0


def test_trajectory_metrics_zero():
    pred = torch.stack([_make_T([0.0, 0.0, 0.0]), _make_T([1.0, 0.0, 0.0])], dim=0)
    pred = pred.unsqueeze(0)
    gt = pred.clone()
    drift = cumulative_drift(pred, gt)
    loop = loop_closure_error(pred, gt)
    assert torch.allclose(drift, torch.zeros(1))
    assert torch.allclose(loop, torch.zeros(1))


def test_ddf_metrics_zero():
    pred = torch.zeros(2, 3, 4, 4, 4)
    gt = torch.zeros(2, 3, 4, 4, 4)
    assert ddf_rmse(pred, gt) == 0
    assert ddf_mae(pred, gt) == 0


def test_volume_metrics_basic():
    pred = torch.ones(2, 4, 4, 4)
    gt = torch.ones(2, 4, 4, 4)
    ncc = volume_ncc(pred, gt)
    ssim = volume_ssim(pred, gt)
    dice = volume_dice(pred, gt)
    assert torch.allclose(ncc, torch.ones(2))
    assert torch.allclose(ssim, torch.ones(2))
    assert torch.allclose(dice, torch.ones(2))
