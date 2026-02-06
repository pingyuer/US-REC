import torch

from trainers.metrics import (
    ddf_mae_all_dims,
    ddf_rmse_all_dims,
    endpoint_rpe_rotation_deg,
    endpoint_rpe_translation_mm,
    end_to_start_rpe_translation_mm,
    end_to_start_rpe_rotation_deg,
    rotation_error_deg,
    se3_rotation_error_deg,
    se3_translation_error_mm,
    translation_error_mm,
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
    err = translation_error_mm(pred, gt)
    assert torch.allclose(err, torch.zeros(2))


def test_rotation_error_identity():
    pred = torch.eye(3).repeat(2, 1, 1)
    gt = torch.eye(3).repeat(2, 1, 1)
    err = rotation_error_deg(pred, gt)
    assert torch.allclose(err, torch.zeros(2))


def test_se3_errors_nonzero():
    pred = _make_T([1.0, 0.0, 0.0]).unsqueeze(0)
    gt = _make_T([0.0, 0.0, 0.0]).unsqueeze(0)
    trans_err = se3_translation_error_mm(pred, gt)
    rot_err = se3_rotation_error_deg(pred, gt)
    assert trans_err.item() > 0
    assert rot_err.item() == 0


def test_trajectory_metrics_zero():
    pred = torch.stack([_make_T([0.0, 0.0, 0.0]), _make_T([1.0, 0.0, 0.0])], dim=0)
    pred = pred.unsqueeze(0)
    gt = pred.clone()
    drift_t = endpoint_rpe_translation_mm(pred, gt)
    drift_r = endpoint_rpe_rotation_deg(pred, gt)
    loop_t = end_to_start_rpe_translation_mm(pred, gt)
    loop_r = end_to_start_rpe_rotation_deg(pred, gt)
    assert torch.allclose(drift_t, torch.zeros(1))
    assert torch.allclose(drift_r, torch.zeros(1))
    assert torch.allclose(loop_t, torch.zeros(1))
    assert torch.allclose(loop_r, torch.zeros(1))


def test_ddf_metrics_zero():
    pred = torch.zeros(2, 3, 4, 4, 4)
    gt = torch.zeros(2, 3, 4, 4, 4)
    assert ddf_rmse_all_dims(pred, gt) == 0
    assert ddf_mae_all_dims(pred, gt) == 0


def test_volume_metrics_basic():
    pred = torch.ones(2, 4, 4, 4)
    gt = torch.ones(2, 4, 4, 4)
    ncc = volume_ncc(pred, gt)
    ssim = volume_ssim(pred, gt)
    dice = volume_dice(pred, gt)
    assert torch.allclose(ncc, torch.ones(2))
    assert torch.allclose(ssim, torch.ones(2))
    assert torch.allclose(dice, torch.ones(2))
