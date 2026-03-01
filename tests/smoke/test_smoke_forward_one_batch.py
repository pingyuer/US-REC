"""Smoke test 2: single forward pass.

Constructs a minimal batch (synthetic data), builds a model from config,
and verifies a single forward pass produces the correct output shape
with no NaN/Inf values.
"""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "demo_rec24_ete.yml"


def _build_minimal_model(cfg):
    """Build the model used in rec training with minimal frame count."""
    from utils.network import build_model
    from utils.rec_ops import compute_dimention, data_pairs_adjacent
    from utils.utils_ori import reference_image_points
    from trainers.utils.calibration import load_calibration

    num_samples = 2  # pair-based: 2 frames
    data_pairs = data_pairs_adjacent(num_samples)

    # Calibration (uses config path, falls back to identity)
    calib_file = OmegaConf.select(cfg, "dataset.calib_file") or ""
    resample = int(OmegaConf.select(cfg, "dataset.resample_factor") or 4)
    try:
        _, _, tform_calib = load_calibration(calib_file, resample, torch.device("cpu"))
    except Exception:
        tform_calib = torch.eye(4)

    # Small image for speed
    H, W = 32, 32
    image_points = reference_image_points((H, W), (H, W))

    pred_type = str(cfg.model.get("pred_type", "parameter"))
    label_type = str(cfg.model.get("label_type", "transform"))
    pred_dim = compute_dimention(pred_type, image_points.shape[1], num_samples, "pred")

    class _Opt:
        """Minimal 'opt' namespace expected by build_model."""
        model_name = str(cfg.model.name)

    opt = _Opt()
    model = build_model(
        opt,
        in_frames=num_samples,
        pred_dim=pred_dim,
        label_dim=0,
        image_points=image_points,
        tform_calib=tform_calib,
        tform_calib_R_T=tform_calib,
    )
    return model, num_samples, H, W


@pytest.mark.smoke
class TestSmokeForwardOneBatch:
    """One-batch forward pass through rec model."""

    def test_forward_no_nan(self):
        cfg = OmegaConf.load(str(CONFIG_PATH))
        model, num_samples, H, W = _build_minimal_model(cfg)
        model.eval()

        batch = torch.randn(1, num_samples, H, W)  # (B, frames, H, W)
        with torch.no_grad():
            out = model(batch)

        assert out is not None, "Model returned None"
        if torch.is_tensor(out):
            assert not torch.isnan(out).any(), "NaN in output"
            assert not torch.isinf(out).any(), "Inf in output"
            assert out.shape[0] == 1, f"Batch dim mismatch: {out.shape}"
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if torch.is_tensor(o):
                    assert not torch.isnan(o).any(), f"NaN in output[{i}]"
                    assert not torch.isinf(o).any(), f"Inf in output[{i}]"
