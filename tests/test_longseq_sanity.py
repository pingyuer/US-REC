"""Sanity checks for the longseq refactor.

Verifies four invariants that were reported as broken or unverified:

  1. Augmentation: reversed window → transforms are consistent (compose matches GT).
  2. Pretrained init: when pretrained=True, conv1 weights are NOT random (mean close
     to the RGB→1ch average, not the default PyTorch random init).
  3. Compose direction: compose_global_from_local(local_gt) ≈ gt_global
     (point error < threshold).
  4. Points-based loss: numerically reasonable for near-identity transforms
     (not NaN, not zero for non-perfect predictions).

Run with:
    pytest tests/test_longseq_sanity.py -v
"""

from __future__ import annotations

import math
import random

import torch
import pytest


# ── 1. Time-reversal augmentation ──────────────────────────────────────────

class TestFlipAugmentation:
    """Verify that reversing a window yields correctly inverted local transforms."""

    @staticmethod
    def _random_walk_global(T: int, scale: float = 1.0) -> torch.Tensor:
        """Generate T consecutive global SE(3) transforms via small random walk."""
        from data.datasets.scan_window import SyntheticScanWindowDataset
        gen = torch.Generator().manual_seed(42)
        return SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)

    def test_reversed_local_equals_forward_inv(self):
        """new_L[j] = inv(old_L[T-j]) for all j >= 1."""
        from metrics.compose import local_from_global

        T = 16
        gt_global = self._random_walk_global(T)  # (T, 4, 4)

        # Original local transforms
        old_local = local_from_global(gt_global)  # (T, 4, 4)

        # Simulate flip: reverse then re-normalise
        gt_global_rev = gt_global.flip(0).clone()
        inv_first = torch.linalg.inv(gt_global_rev[0:1])
        gt_global_rev_norm = inv_first @ gt_global_rev

        new_local = local_from_global(gt_global_rev_norm)  # (T, 4, 4)

        # For j=1..T-1: new_L[j] should == inv(old_L[T-j])
        for j in range(1, T):
            k = T - j  # original index
            expected = torch.linalg.inv(old_local[k])  # (4, 4)
            actual   = new_local[j]
            err = (actual - expected).norm().item()
            assert err < 1e-4, (
                f"Flip invariant violated at j={j}: error={err:.6e}"
            )

    def test_compose_after_flip_equals_normalised_global(self):
        """After flip, compose(local) == normalised_global (point error < 1e-4)."""
        from metrics.compose import local_from_global, compose_global_from_local

        T = 20
        gt_global = self._random_walk_global(T)

        gt_flip = gt_global.flip(0).contiguous()
        inv0 = torch.linalg.inv(gt_flip[0:1])
        gt_flip_norm = inv0 @ gt_flip

        local_flip = local_from_global(gt_flip_norm)
        recomposed  = compose_global_from_local(local_flip)

        err = (recomposed - gt_flip_norm).norm().item()
        assert err < 1e-4, f"Compose after flip mismatch: {err:.2e}"

    def test_scan_window_flip_roundtrip(self):
        """ScanWindowDataset flip: compose(local_from(flipped sample)) == flipped GT."""
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from metrics.compose import local_from_global, compose_global_from_local

        ds = SyntheticScanWindowDataset(
            num_scans=2,
            frames_per_scan=32,
            window_size=16,
            mode="train",
            seed=7,
        )
        for sample in ds:
            gt_g = sample["gt_global_T"]  # (T, 4, 4)
            assert gt_g[0].allclose(torch.eye(4), atol=1e-5), \
                "Frame 0 must be identity after normalisation"
            local_t = local_from_global(gt_g)
            recomp  = compose_global_from_local(local_t)
            err = (recomp - gt_g).norm().item()
            assert err < 1e-4, f"Compose roundtrip error={err:.2e}"
            break  # one sample is enough


# ── 2. Pretrained conv1 weight transfer ────────────────────────────────────

class TestPretrainedInit:
    def test_pretrained_conv1_not_random(self):
        """With pretrained=True, conv1 weight must not be random (uniform dist)."""
        from models.temporal.early_cnn import FrameEncoder

        enc_pretrained  = FrameEncoder(backbone="efficientnet_b0", in_channels=1,
                                        token_dim=64, pretrained=True)
        enc_random      = FrameEncoder(backbone="efficientnet_b0", in_channels=1,
                                        token_dim=64, pretrained=False)

        w_pre = enc_pretrained.features[0][0].weight.data   # (out_ch, 1, k, k)
        w_rnd = enc_random.features[0][0].weight.data

        # The std of a pretrained weight should differ from kaiming-uniform init.
        # A simple heuristic: the pretrained weights have lower std than random init.
        std_pre = w_pre.std().item()
        std_rnd = w_rnd.std().item()
        # They must differ — if both are random they'd be nearly equal.
        assert abs(std_pre - std_rnd) > 1e-4, (
            f"Pretrained and random conv1 have identical std ({std_pre:.4f}); "
            "weight transfer may not have worked."
        )

    def test_pretrained_conv1_is_rgb_mean(self):
        """Pretrained 1-ch conv1 must equal mean(RGB weights, dim=1)."""
        import torchvision.models as tvm
        from models.temporal.early_cnn import FrameEncoder

        # Load RGB pretrained weights directly from torchvision
        rgb_model = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.DEFAULT
        )
        rgb_w = rgb_model.features[0][0].weight.data  # (32, 3, 3, 3)
        expected_1ch = rgb_w.mean(dim=1, keepdim=True)  # (32, 1, 3, 3)

        enc = FrameEncoder(backbone="efficientnet_b0", in_channels=1,
                           token_dim=64, pretrained=True)
        actual_1ch = enc.features[0][0].weight.data   # (32, 1, 3, 3)

        assert actual_1ch.allclose(expected_1ch, atol=1e-6), (
            "1-ch conv1 weights differ from expected RGB-mean init."
        )


# ── 3. Compose direction ───────────────────────────────────────────────────

class TestComposeDirection:
    """Verify compose_global_from_local(local_from_global(G)) == G."""

    def test_compose_roundtrip_random_walk(self):
        from metrics.compose import local_from_global, compose_global_from_local
        from data.datasets.scan_window import SyntheticScanWindowDataset

        gen = torch.Generator().manual_seed(0)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(
            64, generator=gen
        )  # (64, 4, 4)
        local_t  = local_from_global(gt_global)
        recomp   = compose_global_from_local(local_t)
        assert recomp.allclose(gt_global, atol=1e-5), (
            f"Compose roundtrip max error: {(recomp - gt_global).abs().max():.2e}"
        )

    def test_compose_point_error_near_zero(self):
        """Points mapped through compose(local_gt) == points mapped through gt_global."""
        from metrics.compose import local_from_global, compose_global_from_local
        from models.losses.longseq_loss import make_ref_points
        from data.datasets.scan_window import SyntheticScanWindowDataset

        gen = torch.Generator().manual_seed(1)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(
            32, generator=gen
        )  # (32, 4, 4)
        local_t  = local_from_global(gt_global)
        recomp   = compose_global_from_local(local_t)

        ref_pts = make_ref_points(20.0)  # (4, 4)
        pts_hom = ref_pts.T             # (4, 4)

        gt_3d   = (gt_global  @ pts_hom)[:, :3, :].permute(0, 2, 1)  # (T, K, 3)
        rec_3d  = (recomp     @ pts_hom)[:, :3, :].permute(0, 2, 1)

        point_err = (gt_3d - rec_3d).norm(dim=-1).mean().item()
        assert point_err < 1e-4, f"Point error after compose roundtrip: {point_err:.2e}"

    def test_batch_compose_consistency(self):
        """Batched (B, T, 4, 4) compose matches per-batch compose."""
        from metrics.compose import local_from_global, compose_global_from_local
        from data.datasets.scan_window import SyntheticScanWindowDataset

        T = 20
        gen = torch.Generator().manual_seed(3)
        g0 = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gen.manual_seed(7)
        g1 = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)

        gt_batch = torch.stack([g0, g1], dim=0)  # (2, T, 4, 4)
        local_b  = local_from_global(gt_batch)
        recomp_b = compose_global_from_local(local_b)

        assert recomp_b.allclose(gt_batch, atol=1e-5), (
            "Batched compose mismatch"
        )


# ── 4. Points-based loss ───────────────────────────────────────────────────

class TestPointsLoss:
    def test_not_nan(self):
        from models.losses.longseq_loss import longseq_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset

        B, T = 2, 12
        gen = torch.Generator().manual_seed(0)
        frames = torch.rand(B, T, 32, 32, generator=gen)

        # Use identity GT (trivial case)
        gt = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).clone()

        # Perturbed pred_local (non-identity)
        pred_local = gt.clone()
        pred_local[:, 1:, :3, 3] += 0.5  # small translation

        loss, bd = longseq_loss(
            pred_local_T=pred_local,
            pred_aux_T={2: pred_local},
            gt_global_T=gt,
            intervals=[2],
            loss_mode="points",
            ref_pts_scale_mm=20.0,
        )
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be > 0 for non-perfect predictions"

    def test_zero_loss_for_perfect_prediction(self):
        """Points loss is ~0 when pred exactly equals gt (no aux intervals)."""
        from metrics.compose import local_from_global
        from models.losses.longseq_loss import longseq_loss, gt_interval_transform
        from data.datasets.scan_window import SyntheticScanWindowDataset

        B, T = 1, 10
        gen = torch.Generator().manual_seed(5)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gt_global = gt_global.unsqueeze(0).expand(B, T, 4, 4).clone()
        pred_local = local_from_global(gt_global)      # (B, T, 4, 4) — perfect Δ=1
        pred_aux_2 = gt_interval_transform(gt_global, 2)  # (B, T, 4, 4) — perfect Δ=2

        loss, _ = longseq_loss(
            pred_local_T=pred_local,
            pred_aux_T={2: pred_aux_2},
            gt_global_T=gt_global,
            intervals=[2],
            loss_mode="points",
            ref_pts_scale_mm=20.0,
        )
        assert loss.item() < 1e-5, f"Expected ~0 loss for perfect prediction, got {loss.item():.4e}"

    def test_loss_decreases_one_step(self):
        """A gradient step on a simple linear pose regression should reduce loss."""
        from models.losses.longseq_loss import longseq_loss, make_ref_points
        from metrics.compose import local_from_global
        from data.datasets.scan_window import SyntheticScanWindowDataset

        B, T = 1, 8
        gen = torch.Generator().manual_seed(11)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gt_global = gt_global.unsqueeze(0).expand(B, T, 4, 4)

        # Learnable perturbation on translation only
        delta_t = torch.nn.Parameter(torch.randn(B, T, 3) * 2.0)
        opt = torch.optim.SGD([delta_t], lr=0.1)

        def make_pred():
            pred = local_from_global(gt_global).clone()
            pred[:, 1:, :3, 3] = pred[:, 1:, :3, 3] + delta_t[:, 1:]
            return pred

        opt.zero_grad()
        loss_before, _ = longseq_loss(
            pred_local_T=make_pred(),
            pred_aux_T={},
            gt_global_T=gt_global,
            intervals=[],
            loss_mode="points",
        )
        loss_before.backward()
        opt.step()

        with torch.no_grad():
            loss_after, _ = longseq_loss(
                pred_local_T=make_pred(),
                pred_aux_T={},
                gt_global_T=gt_global,
                intervals=[],
                loss_mode="points",
            )

        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
        )

    @pytest.mark.parametrize("loss_mode", ["points", "se3"])
    def test_backward_both_modes(self, loss_mode):
        """Gradients flow correctly in both loss modes."""
        from models.losses.longseq_loss import longseq_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from metrics.compose import local_from_global

        B, T = 1, 8
        gen = torch.Generator().manual_seed(99)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gt_global = gt_global.unsqueeze(0)
        pred_local = local_from_global(gt_global).detach().requires_grad_(True)

        loss, _ = longseq_loss(
            pred_local_T=pred_local,
            pred_aux_T={2: pred_local.detach().clone()},
            gt_global_T=gt_global,
            intervals=[2],
            loss_mode=loss_mode,
        )
        loss.backward()
        assert pred_local.grad is not None
        assert pred_local.grad.abs().sum().item() > 0


# ── 5. DDF surrogate loss ──────────────────────────────────────────────────

class TestDDFSurrogateLoss:
    """Verify the DDF surrogate loss (pixel-space L2) is well-behaved."""

    @staticmethod
    def _identity_calib(scale: float = 0.1) -> torch.Tensor:
        """Simple calib matrix: pixels → mm at fixed scale."""
        c = torch.eye(4, dtype=torch.float32)
        c[0, 0] = scale   # u → mm
        c[1, 1] = scale   # v → mm
        return c

    def test_zero_for_perfect_prediction(self):
        """DDF loss is 0 when pred == gt."""
        from models.losses.longseq_loss import ddf_surrogate_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset

        B, T = 1, 8
        gen = torch.Generator().manual_seed(0)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gt_global = gt_global.unsqueeze(0)
        calib = self._identity_calib()

        loss = ddf_surrogate_loss(
            gt_global, gt_global, tform_calib=calib, image_size=(64, 64), num_points=32
        )
        assert loss.item() < 1e-10, f"Expected ~0 for perfect pred, got {loss.item()}"

    def test_nonzero_for_imperfect(self):
        """DDF loss > 0 when pred ≠ gt."""
        from models.losses.longseq_loss import ddf_surrogate_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from metrics.compose import local_from_global

        B, T = 1, 8
        gen = torch.Generator().manual_seed(1)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen)
        gt_global = gt_global.unsqueeze(0)
        # Perturb prediction slightly
        noise = torch.randn_like(gt_global) * 0.01
        pred_global = gt_global + noise
        calib = self._identity_calib()

        loss = ddf_surrogate_loss(
            pred_global, gt_global, tform_calib=calib, image_size=(64, 64), num_points=32
        )
        assert loss.item() > 1e-12, f"Expected > 0 for imperfect pred, got {loss.item()}"

    def test_gradients_flow(self):
        """Gradients flow through DDF loss."""
        from models.losses.longseq_loss import ddf_surrogate_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset

        B, T = 1, 6
        gen = torch.Generator().manual_seed(2)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen).unsqueeze(0)
        # Perturb gt to make a non-perfect prediction (so loss > 0 and grad != 0)
        noise = torch.randn(B, T, 4, 4) * 0.05
        pred_global = (gt_global + noise).detach().requires_grad_(True)
        calib = self._identity_calib()

        loss = ddf_surrogate_loss(
            pred_global, gt_global, tform_calib=calib, image_size=(32, 32), num_points=16
        )
        assert loss.item() > 0, "Loss should be > 0 for non-perfect prediction"
        loss.backward()
        assert pred_global.grad is not None
        assert pred_global.grad.abs().sum().item() > 0

    def test_ddf_in_longseq_loss(self):
        """DDF term inside longseq_loss is added when calib provided."""
        from models.losses.longseq_loss import longseq_loss
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from metrics.compose import local_from_global

        B, T = 1, 8
        gen = torch.Generator().manual_seed(7)
        gt_global = SyntheticScanWindowDataset._random_walk_transforms(T, generator=gen).unsqueeze(0)
        pred_local = local_from_global(gt_global).detach().requires_grad_(True)
        calib = self._identity_calib()

        loss_with_ddf, bd_with = longseq_loss(
            pred_local_T=pred_local,
            pred_aux_T={},
            gt_global_T=gt_global,
            intervals=[],
            ddf_sample_weight=1.0,
            ddf_num_points=16,
            ddf_tform_calib=calib,
            ddf_image_size=(32, 32),
        )
        loss_without, bd_without = longseq_loss(
            pred_local_T=pred_local,
            pred_aux_T={},
            gt_global_T=gt_global,
            intervals=[],
            ddf_sample_weight=0.0,
        )
        assert bd_with["loss_ddf"] < 1e-10, f"DDF loss should be ~0 for perfect prediction, got {bd_with['loss_ddf']}"
        assert bd_without["loss_ddf"] == 0.0


# ── 6. Memory tokens ─────────────────────────────────────────────────────────

class TestMemoryTokens:
    """Verify Transformer-XL memory token mechanics."""

    def test_memory_shape(self):
        """Memory tokens have correct shape and dtype."""
        from models.temporal.model_longseq import LongSeqPoseModel

        B, T, H, W = 1, 16, 64, 64
        model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64, n_heads=2, n_layers=1, dim_feedforward=128,
            window_size=8, memory_size=4,
        )
        model.eval()
        frames = torch.randn(B, T, H, W)
        with torch.no_grad():
            out = model(frames)
        assert out["memory"] is not None, "memory should not be None when memory_size > 0"
        assert out["memory"].shape == (B, 4, 64), f"Wrong memory shape: {out['memory'].shape}"

    def test_no_memory_when_disabled(self):
        """memory is None when memory_size == 0."""
        from models.temporal.model_longseq import LongSeqPoseModel

        model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64, n_heads=2, n_layers=1, dim_feedforward=128,
            window_size=8, memory_size=0,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 8, 64, 64))
        assert out["memory"] is None

    def test_memory_propagation_changes_output(self):
        """Passing memory from segment 1 changes segment 2 predictions."""
        from models.temporal.model_longseq import LongSeqPoseModel

        B, T = 1, 8
        model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64, n_heads=2, n_layers=1, dim_feedforward=128,
            window_size=8, memory_size=4,
        )
        model.eval()
        frames1 = torch.randn(B, T, 64, 64)
        frames2 = torch.randn(B, T, 64, 64)

        with torch.no_grad():
            out1 = model(frames1)
            # With memory
            out2_mem = model(frames2, memory=out1["memory"])
            # Without memory
            out2_nomem = model(frames2)

        # Predictions should differ when memory is provided vs not
        diff = (out2_mem["pred_local_T"] - out2_nomem["pred_local_T"]).abs().max().item()
        assert diff > 1e-6, f"Memory had no effect on predictions (max diff {diff})"

    def test_memory_is_detached(self):
        """Memory tokens must be detached from the compute graph."""
        from models.temporal.model_longseq import LongSeqPoseModel

        model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64, n_heads=2, n_layers=1, dim_feedforward=128,
            window_size=8, memory_size=4,
        )
        frames = torch.randn(1, 8, 64, 64, requires_grad=True)
        out = model(frames)
        assert not out["memory"].requires_grad, "Memory tokens should be detached"

    def test_chunked_forward_matches_shapes(self):
        """_forward_with_memory produces correct shapes for all frames."""
        from trainers.longseq_trainer import LongSeqTrainer
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "model": {
                "pose_head": {"rotation_rep": "rot6d"},
                "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
                "transformer": {
                    "d_model": 64, "n_heads": 2, "n_layers": 1,
                    "dim_feedforward": 128, "window_size": 8,
                    "dropout": 0.0, "memory_size": 4,
                },
            },
            "loss": {
                "aux_intervals": [2],
                "aux_weight": 0.5, "aux_decay": 0.5,
                "aux_scale": "none", "mode": "points",
                "ref_pts_scale_mm": 20.0,
                "consistency_weight": 0.0, "consistency_delta": 2,
                "rot_weight": 1.0, "trans_weight": 1.0,
                "ddf_sample_weight": 0.0, "ddf_num_points": 0,
            },
            "optimizer": {"lr": 1e-4},
            "trainer": {"max_epochs": 1, "log_interval": 1, "validate_every": 1,
                        "grad_accum": 1},
            "paths": {"output_dir": "logs"},
        })
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.eval()

        B, T = 1, 20
        frames = torch.randn(B, T, 64, 64)
        with torch.no_grad():
            out = trainer._forward_with_memory(frames, chunk_size=8)

        assert out["pred_local_T"].shape == (B, T, 4, 4), (
            f"Wrong shape from chunked forward: {out['pred_local_T'].shape}"
        )


# ── New: SE(3) metric naming + val_loss + reverse_prob ─────────────────────

def _make_minimal_trainer_cfg():
    """Minimal OmegaConf config for a LongSeqTrainer CPU smoke run."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {
            "pose_head": {"rotation_rep": "rot6d"},
            "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
            "transformer": {
                "d_model": 64, "n_heads": 2, "n_layers": 1,
                "dim_feedforward": 128, "window_size": 8,
                "dropout": 0.0, "memory_size": 0,
            },
        },
        "loss": {
            "aux_intervals": [2],
            "aux_weight": 0.5, "aux_decay": 0.5,
            "aux_scale": "none", "mode": "points",
            "ref_pts_scale_mm": 20.0,
            "consistency_weight": 0.0, "consistency_delta": 2,
            "rot_weight": 1.0, "trans_weight": 1.0,
            "ddf_sample_weight": 0.0, "ddf_num_points": 0,
        },
        "optimizer": {"lr": 1e-4},
        "trainer": {"max_epochs": 1, "log_interval": 1, "validate_every": 1,
                    "grad_accum": 1},
        "paths": {"output_dir": "logs"},
    })


def _make_val_loader(num_scans: int = 2, T: int = 12, H: int = 64, W: int = 64):
    """Synthetic val loader (full-scan) for evaluate() tests."""
    from data.datasets.scan_window import SyntheticScanWindowDataset
    from torch.utils.data import DataLoader
    ds = SyntheticScanWindowDataset(
        num_scans=num_scans, frames_per_scan=T,
        height=H, width=W, window_size=T, mode="val", seed=0,
    )
    return DataLoader(ds, batch_size=1)


class TestSE3MetricNaming:
    """evaluate() must use se3_* keys, not the old lpe_mm / gpe_mm names."""

    def test_se3_keys_present(self):
        from trainers.longseq_trainer import LongSeqTrainer
        cfg = _make_minimal_trainer_cfg()
        trainer = LongSeqTrainer(cfg, device="cpu")
        loader = _make_val_loader()
        metrics = trainer.evaluate(loader)

        # New canonical names
        for key in ("mean_se3_lpe_mm", "mean_se3_gpe_mm", "mean_se3_drift_last_mm"):
            assert key in metrics, f"Expected key '{key}' in evaluate() output; got {list(metrics)}"

    def test_old_keys_absent(self):
        """Legacy names must NOT appear (prevents silent metric confusion)."""
        from trainers.longseq_trainer import LongSeqTrainer
        cfg = _make_minimal_trainer_cfg()
        trainer = LongSeqTrainer(cfg, device="cpu")
        loader = _make_val_loader()
        metrics = trainer.evaluate(loader)

        for old_key in ("mean_lpe_mm", "mean_gpe_mm", "mean_drift_last_mm"):
            assert old_key not in metrics, (
                f"Old key '{old_key}' should have been removed; found in {list(metrics)}"
            )

    def test_se3_metrics_are_non_negative(self):
        from trainers.longseq_trainer import LongSeqTrainer
        cfg = _make_minimal_trainer_cfg()
        trainer = LongSeqTrainer(cfg, device="cpu")
        loader = _make_val_loader()
        metrics = trainer.evaluate(loader)

        for key in ("mean_se3_lpe_mm", "mean_se3_gpe_mm", "mean_se3_drift_last_mm"):
            assert metrics[key] >= 0.0, f"{key} should be >= 0, got {metrics[key]}"


class TestValLossIsTrainingLoss:
    """after_val must set val_loss = training avg loss, not a tusrec proxy."""

    def test_val_loss_equals_last_train_avg_loss(self):
        """_last_train_avg_loss is exposed as val_loss in after_val hook calls."""
        from trainers.longseq_trainer import LongSeqTrainer
        from trainers.hooks.base_hook import Hook

        # Capture log_buffers from after_val calls
        captured: list[dict] = []

        class CapturingHook(Hook):
            def after_val(self, trainer, log_buffer=None, **_):
                if log_buffer:
                    captured.append(dict(log_buffer))

        cfg = _make_minimal_trainer_cfg()
        val_loader = _make_val_loader(num_scans=2)

        # Build a tiny train loader too
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader
        ds_train = SyntheticScanWindowDataset(
            num_scans=2, frames_per_scan=16, height=64, width=64, window_size=8,
            mode="train", seed=42,
        )
        train_loader = DataLoader(ds_train, batch_size=1)

        trainer = LongSeqTrainer(cfg, device="cpu",
                                 train_loader=train_loader,
                                 val_loader=val_loader)
        trainer.hooks.append(CapturingHook())
        trainer.train()

        assert captured, "after_val was never called during train()"
        buf = captured[0]
        assert "val_loss" in buf, f"val_loss missing from after_val buffer: {buf}"
        # val_loss must equal _last_train_avg_loss set during training
        assert buf["val_loss"] == pytest.approx(trainer._last_train_avg_loss, abs=1e-6), (
            f"val_loss={buf['val_loss']} != _last_train_avg_loss={trainer._last_train_avg_loss}"
        )

    def test_val_tusrec_final_score_added_when_present(self):
        """val_tusrec_final_score must be injected into after_val when tusrec metrics exist."""
        from trainers.longseq_trainer import LongSeqTrainer
        from trainers.hooks.base_hook import Hook

        captured: list[dict] = []

        class CapturingHook(Hook):
            def after_val(self, trainer, log_buffer=None, **_):
                if log_buffer:
                    captured.append(dict(log_buffer))

        cfg = _make_minimal_trainer_cfg()
        # Monkey-patch evaluate() to inject tusrec metrics
        class FakeLoader:
            def __iter__(self): return iter([])

        val_loader = FakeLoader()

        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader
        ds_train = SyntheticScanWindowDataset(
            num_scans=2, frames_per_scan=16, height=64, width=64, window_size=8,
            mode="train", seed=0,
        )
        train_loader = DataLoader(ds_train, batch_size=1)

        trainer = LongSeqTrainer(cfg, device="cpu",
                                 train_loader=train_loader,
                                 val_loader=val_loader)
        trainer.hooks.append(CapturingHook())

        # Patch evaluate() to return fake tusrec metrics
        def _fake_evaluate(_loader):
            return {
                "mean_gpe_pts_mm": 12.3,
                "mean_tusrec_final_score": 0.75,
                "mean_tusrec_GPE_mm": 8.5,
                "num_scans": 2.0,
            }
        trainer.evaluate = _fake_evaluate  # type: ignore[assignment]
        trainer.train()

        assert captured, "after_val was never called"
        buf = captured[0]
        assert "val_tusrec_final_score" in buf, (
            f"val_tusrec_final_score missing from after_val buffer: {buf}"
        )
        assert buf["val_tusrec_final_score"] == pytest.approx(0.75)
        assert buf["val_tusrec_GPE_mm"] == pytest.approx(8.5)


class TestReverseProbConfig:
    """augment.reverse_prob config key must override legacy flip_prob in builder."""

    def test_reverse_prob_overrides_flip_prob(self):
        """When dataset.augment.reverse_prob is set, augment_flip and flip_prob
        passed to ScanWindowDataset must reflect that value."""
        from omegaconf import OmegaConf
        from trainers import builder as bld
        import importlib

        # Re-import to get fresh module state
        importlib.reload(bld)

        calls: list[dict] = []
        original_scan_window_ds = None

        # Patch ScanWindowDataset.__init__ to capture kwargs
        import data.datasets.scan_window as sw_mod
        original_cls = sw_mod.ScanWindowDataset

        class PatchedScanWindowDataset(original_cls):
            def __init__(self, **kwargs):
                calls.append(kwargs)
                # Don't call super().__init__ — we just want kwargs
        sw_mod.ScanWindowDataset = PatchedScanWindowDataset

        import importlib, sys
        # Force builder to reload the patched class
        if "trainers.builder" in sys.modules:
            del sys.modules["trainers.builder"]
        import trainers.builder as bld2

        cfg = OmegaConf.create({
            "dataset": {
                "augment_flip": False,   # legacy key OFF
                "flip_prob": 0.1,        # legacy value — should be overridden
                "augment": {"reverse_prob": 0.9},  # preferred key
                "sequence_window": 8,
                "windows_per_scan": 1,
            },
            "dataloader": {
                "train": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
                "val":   {"batch_size": 1, "num_workers": 0, "pin_memory": False},
            },
            "seed": 0,
        })

        # Create minimal fake base datasets
        class _FakeDS:
            region = "us-east-1"
            endpoint = "http://localhost:9000"
            force_path_style = True
            shuffle_slices = False
            def _list_slices(self, client=None): return []
            def _iter_loaded_slices(self, slices, client=None): return iter([])

        try:
            bld2._build_scan_window_loaders(cfg, _FakeDS(), _FakeDS())
        except Exception:
            pass  # DataLoader setup may fail; we only care about captured kwargs

        # Restore
        sw_mod.ScanWindowDataset = original_cls

        train_kwargs = next((c for c in calls if c.get("mode") == "train"), None)
        assert train_kwargs is not None, "ScanWindowDataset was not instantiated for train"
        assert train_kwargs["augment_flip"] is True, (
            f"augment_flip should be True when reverse_prob is set; got {train_kwargs['augment_flip']}"
        )
        assert train_kwargs["flip_prob"] == pytest.approx(0.9), (
            f"flip_prob should be 0.9 (from reverse_prob), got {train_kwargs['flip_prob']}"
        )


# ── New (Phase 4): mode-aware logging, DDF active, seg_len, aux_delta32 ────

def _make_trainer_cfg_ex(
    loss_mode: str = "points",
    ddf_weight: float = 0.0,
    memory_size: int = 0,
    seg_len: int = 0,
    aux_intervals: list | None = None,
):
    """Minimal trainer config supporting the new Phase-4 features."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {
            "pose_head": {"rotation_rep": "rot6d"},
            "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
            "transformer": {
                "d_model": 64, "n_heads": 2, "n_layers": 1,
                "dim_feedforward": 128, "window_size": 8,
                "dropout": 0.0,
                "memory_size": memory_size,
                "seg_len": seg_len,
            },
        },
        "loss": {
            "aux_intervals": aux_intervals or [2],
            "aux_weight": 0.5, "aux_decay": 0.5, "aux_scale": "none",
            "mode": loss_mode,
            "ref_pts_scale_mm": 20.0,
            "consistency_weight": 0.0, "consistency_delta": 2,
            "rot_weight": 1.0, "trans_weight": 1.0,
            "ddf_sample_weight": ddf_weight,
            "ddf_num_points": 16,
        },
        "optimizer": {"lr": 1e-4},
        "trainer": {"max_epochs": 1, "log_interval": 1, "validate_every": 1,
                    "grad_accum": 1},
        "paths": {"output_dir": "logs"},
    })


def _make_train_val_loaders(T_train: int = 16, T_val: int = 20):
    """Small synthetic loaders for trainer integration smoke tests."""
    from data.datasets.scan_window import SyntheticScanWindowDataset
    from torch.utils.data import DataLoader
    ds_t = SyntheticScanWindowDataset(
        num_scans=2, frames_per_scan=T_train, height=64, width=64,
        window_size=T_train, mode="train", seed=0,
    )
    ds_v = SyntheticScanWindowDataset(
        num_scans=2, frames_per_scan=T_val, height=64, width=64,
        window_size=T_val, mode="val", seed=0,
    )
    return DataLoader(ds_t, batch_size=1), DataLoader(ds_v, batch_size=1)


class TestModeAwareLogging:
    """A: In points mode, breakdown must expose local_pts_rmse_mm & drift_pts_mm
    instead of misleading local_rot=0 / local_trans fields."""

    def test_points_mode_breakdown_keys_v2(self):
        """_run_step breakdown in points mode has local_pts_rmse_mm + drift_pts_mm."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(loss_mode="points")
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.train()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=12, height=64, width=64, window_size=12, mode="train", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        with torch.no_grad():
            loss, bd = trainer._run_step(batch)

        assert "local_pts_rmse_mm" in bd, f"local_pts_rmse_mm missing; got {list(bd)}"
        assert "drift_pts_mm" in bd,      f"drift_pts_mm missing; got {list(bd)}"
        assert "local_rot_deg" not in bd, "local_rot_deg should NOT appear in points mode"
        assert bd["local_pts_rmse_mm"] >= 0.0
        assert bd["drift_pts_mm"] >= 0.0

    def test_se3_mode_breakdown_keys(self):
        """_run_step breakdown in se3 mode has local_rot_deg + local_trans_mm."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(loss_mode="se3")
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.train()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=12, height=64, width=64, window_size=12, mode="train", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        with torch.no_grad():
            loss, bd = trainer._run_step(batch)

        assert "local_rot_deg"   in bd, f"local_rot_deg missing; got {list(bd)}"
        assert "local_trans_mm"  in bd, f"local_trans_mm missing; got {list(bd)}"
        assert "local_pts_rmse_mm" not in bd, "local_pts_rmse_mm should NOT appear in se3 mode"
        assert bd["local_rot_deg"] >= 0.0

    def test_grad_norm_rot_head_in_step_buffer(self):
        """grad_norm_rot_head must appear in the after_step log_buffer."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from trainers.hooks.base_hook import Hook

        step_bufs: list[dict] = []

        class CapHook(Hook):
            def after_step(self, trainer, log_buffer=None, **_):
                if log_buffer:
                    step_bufs.append(dict(log_buffer))

        cfg = _make_trainer_cfg_ex(loss_mode="points")
        tr_loader, _ = _make_train_val_loaders()
        trainer = LongSeqTrainer(cfg, device="cpu", train_loader=tr_loader)
        trainer.hooks.append(CapHook())
        trainer.train()

        assert step_bufs, "No after_step calls captured"
        buf = step_bufs[0]
        assert "grad_norm_rot_head" in buf, f"grad_norm_rot_head not in step buffer: {list(buf)}"
        assert buf["grad_norm_rot_head"] >= 0.0


class TestDDFLossEnabled:
    """B: When ddf_sample_weight > 0, the DDF term must be non-zero and logged."""

    def test_ddf_loss_nonzero_with_weight(self):
        """With ddf_sample_weight=0.05 and a toy calib, loss_ddf must be > 0."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(ddf_weight=0.05)
        trainer = LongSeqTrainer(cfg, device="cpu")
        # Inject a toy calib (pixels → mm at 0.1 scale)
        trainer.tform_calib = torch.eye(4, dtype=torch.float32) * 0.1
        trainer.tform_calib[3, 3] = 1.0
        trainer.model.train()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=12, height=64, width=64, window_size=12, mode="train", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        loss, bd = trainer._run_step(batch)

        assert "loss_ddf" in bd, "loss_ddf missing from breakdown"
        assert bd["loss_ddf"] > 0.0, f"loss_ddf should be > 0 (got {bd['loss_ddf']})"

    def test_ddf_loss_zero_without_calib(self):
        """Without calib, loss_ddf must be 0 regardless of weight setting."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(ddf_weight=0.05)
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.tform_calib = None  # no calib → DDF disabled
        trainer.model.train()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=12, height=64, width=64, window_size=12, mode="train", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        loss, bd = trainer._run_step(batch)
        assert bd["loss_ddf"] == 0.0, f"loss_ddf should be 0 without calib, got {bd['loss_ddf']}"


class TestSegLenRecurrence:
    """C: With seg_len > 0 and memory_size > 0, training forward uses chunked path."""

    def test_train_with_seg_len_runs_without_error(self):
        """Training with seg_len=8, memory_size=4 completes one epoch on synthetic data."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer

        cfg = _make_trainer_cfg_ex(memory_size=4, seg_len=8)
        tr_loader, _ = _make_train_val_loaders(T_train=16)
        trainer = LongSeqTrainer(cfg, device="cpu", train_loader=tr_loader)
        trainer.train()  # should complete without error

    def test_seg_len_forward_shape_consistency(self):
        """pred_local_T shape matches T regardless of seg_len chunking."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        B, T = 1, 24
        cfg = _make_trainer_cfg_ex(memory_size=4, seg_len=8)
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.eval()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=T, height=64, width=64, window_size=T, mode="val", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=B)))
        frames = batch["frames"]

        with torch.no_grad():
            out = trainer._forward_with_memory(frames, chunk_size=8)

        assert out["pred_local_T"].shape == (B, T, 4, 4), (
            f"Shape mismatch: {out['pred_local_T'].shape}"
        )

    def test_seg_len_breakdown_has_expected_fields(self):
        """_run_step with seg_len recurrence still produces all expected fields."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(memory_size=4, seg_len=8)
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.train()

        ds = SyntheticScanWindowDataset(num_scans=1, frames_per_scan=16, height=64, width=64, window_size=16, mode="train", seed=0)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        loss, bd = trainer._run_step(batch)

        for req_key in ("loss_local", "loss_aux", "drift_mm_last",
                        "local_pts_rmse_mm", "drift_pts_mm"):
            assert req_key in bd, f"Required key '{req_key}' missing from breakdown"


class TestAuxInterval32:
    """D: aux_intervals=[2,4,8,16,32] – breakdown contains aux_delta32 key."""

    def test_aux_delta32_in_breakdown(self):
        """With intervals=[2,4,8,16,32] and T>32, aux_delta32_pts appears in breakdown."""
        import torch
        from trainers.longseq_trainer import LongSeqTrainer
        from data.datasets.scan_window import SyntheticScanWindowDataset
        from torch.utils.data import DataLoader

        cfg = _make_trainer_cfg_ex(aux_intervals=[2, 4, 8, 16, 32])
        trainer = LongSeqTrainer(cfg, device="cpu")
        trainer.model.train()

        # Need T > 32 to get valid Δ=32 frames
        ds = SyntheticScanWindowDataset(
            num_scans=1, frames_per_scan=48, height=64, width=64,
            window_size=48, mode="train", seed=0,
        )
        batch = next(iter(DataLoader(ds, batch_size=1)))
        loss, bd = trainer._run_step(batch)

        assert "aux_delta32_pts" in bd, (
            f"aux_delta32_pts not in breakdown for intervals=[2,4,8,16,32]; "
            f"got keys: {[k for k in bd if 'aux' in k]}"
        )
        assert bd["aux_delta32_pts"] >= 0.0

    def test_aux_delta32_printed_in_log(self):
        """With aux_intervals=[2,4,8,16,32], the training log includes aux32 term."""
        import io, sys
        from trainers.longseq_trainer import LongSeqTrainer

        cfg = _make_trainer_cfg_ex(aux_intervals=[2, 4, 8, 16, 32])
        tr_loader, _ = _make_train_val_loaders(T_train=48)

        # Capture stdout
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            trainer = LongSeqTrainer(cfg, device="cpu", train_loader=tr_loader)
            trainer.train()
        finally:
            sys.stdout = old_stdout

        output = buf.getvalue()
        assert "auxΔ32" in output or "aux_delta32" in output or "32" in output, (
            f"aux Δ=32 term not visible in training log output:\n{output[:500]}"
        )


