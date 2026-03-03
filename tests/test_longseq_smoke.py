"""Smoke tests for the long-sequence pose estimation pipeline.

Tests (per spec §F):
    1. Random forward (B=1,T=16) → correct output shapes, no NaN
    2. Multi-interval mask correctness (i<Δ excluded from loss)
    3. compose_global_from_local sanity: inv(global[i-1])@global[i] ≈ local[i]
    4. Loss computation: all terms differentiable, no NaN
    5. Consistency loss: cycle constraint gradient flows
    6. Full mini-train: few steps on synthetic data, loss decreases
"""

from __future__ import annotations

import pytest
import torch

# Ensure project root is on path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── 1. Forward pass shape + NaN check ──────────────────────────────────────

class TestForwardPass:
    """Verify LongSeqPoseModel forward produces correct shapes and no NaN."""

    @pytest.fixture(autouse=True)
    def _model(self):
        from models.temporal.model_longseq import LongSeqPoseModel
        self.model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            in_channels=1,
            token_dim=64,           # small for speed
            n_heads=2,
            n_layers=2,
            dim_feedforward=128,
            window_size=16,
            dropout=0.0,
            rotation_rep="rot6d",
            aux_intervals=[2, 4, 8],
        )
        self.model.eval()

    def test_output_shapes(self):
        B, T, H, W = 1, 16, 64, 64
        frames = torch.rand(B, T, H, W)
        out = self.model(frames)

        assert out["pred_local_T"].shape == (B, T, 4, 4)
        assert out["tokens"].shape == (B, T, 64)
        assert out["ctx"].shape == (B, T, 64)
        for delta in [2, 4, 8]:
            assert delta in out["pred_aux_T"]
            assert out["pred_aux_T"][delta].shape == (B, T, 4, 4)

    def test_no_nan(self):
        B, T, H, W = 1, 16, 64, 64
        frames = torch.rand(B, T, H, W)
        out = self.model(frames)

        assert not torch.isnan(out["pred_local_T"]).any(), "NaN in pred_local_T"
        assert not torch.isnan(out["tokens"]).any(), "NaN in tokens"
        assert not torch.isnan(out["ctx"]).any(), "NaN in ctx"
        for delta, aux_t in out["pred_aux_T"].items():
            assert not torch.isnan(aux_t).any(), f"NaN in pred_aux_T[{delta}]"

    def test_frame0_identity(self):
        """pred_local_T[:, 0] must be identity."""
        B, T, H, W = 1, 16, 64, 64
        frames = torch.rand(B, T, H, W)
        out = self.model(frames)
        eye = torch.eye(4)
        assert torch.allclose(out["pred_local_T"][0, 0], eye, atol=1e-5)

    def test_aux_frame_below_delta_is_identity(self):
        """pred_aux_T[delta][:, i<delta] must be identity."""
        B, T, H, W = 1, 16, 64, 64
        frames = torch.rand(B, T, H, W)
        out = self.model(frames)
        eye = torch.eye(4)
        for delta in [2, 4, 8]:
            for i in range(delta):
                assert torch.allclose(
                    out["pred_aux_T"][delta][0, i], eye, atol=1e-5
                ), f"pred_aux_T[{delta}][:, {i}] should be identity"

    def test_batch_gt_1(self):
        """Model handles B > 1."""
        B, T, H, W = 2, 8, 64, 64
        frames = torch.rand(B, T, H, W)
        out = self.model(frames)
        assert out["pred_local_T"].shape == (B, T, 4, 4)

    def test_5d_input(self):
        """Model accepts (B,T,C,H,W) with C=1."""
        B, T, C, H, W = 1, 8, 1, 64, 64
        frames = torch.rand(B, T, C, H, W)
        out = self.model(frames)
        assert out["pred_local_T"].shape == (B, T, 4, 4)


# ─── 2. Mask correctness ────────────────────────────────────────────────────

class TestMultiIntervalMask:
    def test_mask_valid_positions(self):
        from models.pose_heads.pose_head import MultiIntervalMask
        T = 20
        intervals = [2, 4, 8, 16]
        masks = MultiIntervalMask.build(T, intervals, device=torch.device("cpu"))

        for delta in intervals:
            m = masks[delta]
            assert m.shape == (T,)
            # i < delta → False (invalid)
            for i in range(min(delta, T)):
                assert not m[i].item(), f"mask[{delta}][{i}] should be False"
            # i >= delta → True (valid)
            for i in range(delta, T):
                assert m[i].item(), f"mask[{delta}][{i}] should be True"

    def test_delta_exceeds_T(self):
        from models.pose_heads.pose_head import MultiIntervalMask
        masks = MultiIntervalMask.build(4, [8], device=torch.device("cpu"))
        assert not masks[8].any(), "All should be False when delta > T"


# ─── 3. compose_global_from_local sanity ─────────────────────────────────────

class TestComposeGlobal:
    def test_roundtrip(self):
        """inv(global[i-1]) @ global[i] ≈ local[i]."""
        from metrics.compose import compose_global_from_local, local_from_global

        T = 10
        eye = torch.eye(4)
        local = [eye]
        for _ in range(T - 1):
            # small random rotation + translation
            R = torch.eye(3) + 0.01 * torch.randn(3, 3)
            U, _, Vh = torch.linalg.svd(R)
            R = U @ Vh
            if torch.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vh
            step = torch.eye(4)
            step[:3, :3] = R
            step[:3, 3] = torch.randn(3) * 0.5
            local.append(step)
        local_T = torch.stack(local, dim=0)  # (T, 4, 4)

        global_T = compose_global_from_local(local_T)
        recovered = local_from_global(global_T)

        assert torch.allclose(recovered, local_T, atol=1e-4), (
            f"Round-trip error: max diff = {(recovered - local_T).abs().max():.6f}"
        )

    def test_frame0_is_identity(self):
        from metrics.compose import compose_global_from_local
        local_T = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)
        global_T = compose_global_from_local(local_T)
        assert torch.allclose(global_T[0], torch.eye(4))


# ─── 4. Loss computation ────────────────────────────────────────────────────

class TestLossComputation:
    @pytest.fixture(autouse=True)
    def _setup(self):
        from models.temporal.model_longseq import LongSeqPoseModel
        self.model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64,
            n_heads=2,
            n_layers=2,
            dim_feedforward=128,
            window_size=16,
            dropout=0.0,
            rotation_rep="rot6d",
            aux_intervals=[2, 4],
        )

    def test_loss_no_nan(self):
        from models.losses.longseq_loss import longseq_loss
        B, T, H, W = 1, 12, 64, 64
        frames = torch.rand(B, T, H, W)
        # Random-walk ground truth
        gt_global = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        for i in range(1, T):
            step = torch.eye(4)
            step[:3, 3] = torch.randn(3) * 0.5
            gt_global[:, i] = gt_global[:, i - 1] @ step

        out = self.model(frames)
        loss, bd = longseq_loss(
            pred_local_T=out["pred_local_T"],
            pred_aux_T=out["pred_aux_T"],
            gt_global_T=gt_global,
            intervals=[2, 4],
        )
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_loss_backward(self):
        """Loss gradients flow back to model parameters."""
        from models.losses.longseq_loss import longseq_loss
        B, T, H, W = 1, 8, 64, 64
        frames = torch.rand(B, T, H, W)
        gt_global = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)

        out = self.model(frames)
        loss, _ = longseq_loss(
            pred_local_T=out["pred_local_T"],
            pred_aux_T=out["pred_aux_T"],
            gt_global_T=gt_global,
            intervals=[2, 4],
        )
        loss.backward()
        # Check at least one parameter has grad
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        assert has_grad, "No gradients flowed to model parameters"


# ─── 5. Consistency loss ────────────────────────────────────────────────────

class TestConsistencyLoss:
    def test_perfect_consistency(self):
        """When pred_aux exactly equals composition, consistency loss = 0."""
        from models.losses.longseq_loss import consistency_loss
        B, T = 1, 8
        pred_local = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        # Perfect aux: compose of two locals
        composed = pred_local[:, :-2] @ pred_local[:, 1:-1]  # (B, T-2, 4, 4) — all I
        aux_2 = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        aux_2[:, 2:] = composed  # paste composed into valid positions

        loss = consistency_loss(pred_local, {2: aux_2}, delta_check=2)
        assert loss.item() < 1e-6, f"Expected near-zero, got {loss.item()}"

    def test_consistency_gradient(self):
        from models.losses.longseq_loss import consistency_loss
        B, T = 1, 8
        pred_local = torch.randn(B, T, 4, 4, requires_grad=True)
        aux_2 = torch.randn(B, T, 4, 4, requires_grad=True)
        loss = consistency_loss(pred_local, {2: aux_2}, delta_check=2)
        loss.backward()
        assert pred_local.grad is not None


# ─── 6. Mini-train: loss decreases on synthetic data ────────────────────────

class TestMiniTrain:
    def test_loss_decreases(self):
        from models.temporal.model_longseq import LongSeqPoseModel
        from models.losses.longseq_loss import longseq_loss

        torch.manual_seed(0)
        model = LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=32,
            n_heads=2,
            n_layers=1,
            dim_feedforward=64,
            window_size=8,
            dropout=0.0,
            rotation_rep="rot6d",
            aux_intervals=[2],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, T, H, W = 1, 8, 32, 32
        frames = torch.rand(B, T, H, W)
        gt_global = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        for i in range(1, T):
            step = torch.eye(4)
            step[:3, 3] = torch.randn(3) * 0.3
            gt_global[:, i] = gt_global[:, i - 1] @ step

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(frames)
            loss, _ = longseq_loss(
                pred_local_T=out["pred_local_T"],
                pred_aux_T=out["pred_aux_T"],
                gt_global_T=gt_global,
                intervals=[2],
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


# ─── 7. GT interval transform ───────────────────────────────────────────────

class TestGTIntervalTransform:
    def test_delta1_equals_local(self):
        """gt_interval_transform with delta=1 should equal local_from_global."""
        from models.losses.longseq_loss import gt_interval_transform
        from metrics.compose import local_from_global

        B, T = 1, 10
        gt_global = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        for i in range(1, T):
            step = torch.eye(4)
            step[:3, 3] = torch.randn(3) * 0.5
            gt_global[:, i] = gt_global[:, i - 1] @ step

        interval_1 = gt_interval_transform(gt_global, delta=1)
        local = local_from_global(gt_global)
        # They should match for i >= 1
        assert torch.allclose(interval_1[:, 1:], local[:, 1:], atol=1e-4)


# ─── 8. Sliding-window causal mask ──────────────────────────────────────────

class TestSlidingWindowMask:
    def test_causal(self):
        """Frame i should not attend to any frame j > i."""
        from models.temporal.temporal_transformer import _build_sliding_window_causal_mask
        T, W = 16, 8
        mask = _build_sliding_window_causal_mask(T, W, torch.device("cpu"))
        for i in range(T):
            for j in range(i + 1, T):
                assert mask[i, j].item(), f"Frame {i} should not attend to future frame {j}"

    def test_window(self):
        """Frame i should attend to frames max(0, i-W+1)..i."""
        from models.temporal.temporal_transformer import _build_sliding_window_causal_mask
        T, W = 16, 4
        mask = _build_sliding_window_causal_mask(T, W, torch.device("cpu"))
        for i in range(T):
            start = max(0, i - W + 1)
            for j in range(start, i + 1):
                assert not mask[i, j].item(), (
                    f"Frame {i} should attend to frame {j} (window {start}..{i})"
                )
            # Outside window (before start) should be blocked
            for j in range(0, start):
                assert mask[i, j].item(), (
                    f"Frame {i} should NOT attend to frame {j} (outside window)"
                )


# ─── 9. Synthetic dataset ───────────────────────────────────────────────────

class TestSyntheticDataset:
    def test_yields_correct_keys(self):
        from data.datasets.scan_window import SyntheticScanWindowDataset
        ds = SyntheticScanWindowDataset(
            num_scans=2, frames_per_scan=16, height=32, width=32,
            window_size=8, mode="train", seed=42,
        )
        for sample in ds:
            assert "frames" in sample
            assert "gt_global_T" in sample
            assert "meta" in sample
            T = sample["frames"].shape[0]
            assert sample["gt_global_T"].shape == (T, 4, 4)
            # Frame 0 should be identity
            assert torch.allclose(sample["gt_global_T"][0], torch.eye(4), atol=1e-5)
            break  # just check first

    def test_eval_full_scan(self):
        from data.datasets.scan_window import SyntheticScanWindowDataset
        ds = SyntheticScanWindowDataset(
            num_scans=1, frames_per_scan=20, height=32, width=32,
            window_size=8, mode="val", seed=42,
        )
        for sample in ds:
            assert sample["frames"].shape[0] == 20
            assert sample["meta"]["window_size"] == 20
            break
