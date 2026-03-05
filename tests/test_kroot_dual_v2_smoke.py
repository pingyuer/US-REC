"""Smoke tests for the V2 kroot dual system.

Verifies:
1. V2 position encodings (RoPE + real-index sinusoidal) produce correct shapes.
2. V2 transformer forward pass works for both branch modes.
3. V2 model forward pass produces expected output keys and shapes.
4. V2 dataset yields samples with position_ids.
5. V2 trainer instantiation and single-step training.
"""

from __future__ import annotations

import math

import pytest
import torch


# ─── PE tests ────────────────────────────────────────────────────────────────

class TestRoPE:
    def test_shape(self):
        from models.temporal.position_encoding import RotaryPositionEncoding

        rope = RotaryPositionEncoding(d_head=64, max_len=1024)
        B, H, T, D = 2, 4, 32, 64
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == (B, H, T, D)
        assert k_rot.shape == (B, H, T, D)

    def test_with_position_ids(self):
        from models.temporal.position_encoding import RotaryPositionEncoding

        rope = RotaryPositionEncoding(d_head=64, max_len=1024)
        B, H, T, D = 2, 4, 16, 64
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        pos_ids = torch.arange(10, 10 + T).unsqueeze(0).expand(B, -1)
        q_rot, k_rot = rope(q, k, position_ids=pos_ids)
        assert q_rot.shape == (B, H, T, D)

    def test_relative_invariance(self):
        """Dot product q[i]·k[j] should depend only on |i-j| under RoPE."""
        from models.temporal.position_encoding import RotaryPositionEncoding

        rope = RotaryPositionEncoding(d_head=32, max_len=256)
        q = torch.randn(1, 1, 8, 32)
        k = torch.randn(1, 1, 8, 32)
        q1, k1 = rope(q, k, position_ids=torch.arange(8).unsqueeze(0))
        q2, k2 = rope(q, k, position_ids=torch.arange(100, 108).unsqueeze(0))
        # The relative dot products should be similar (shifted by same offset)
        dot1 = (q1[0, 0, 2] * k1[0, 0, 0]).sum()
        dot2 = (q2[0, 0, 2] * k2[0, 0, 0]).sum()
        assert torch.allclose(dot1, dot2, atol=1e-4)


class TestRealIndexSinusoidalPE:
    def test_shape(self):
        from models.temporal.position_encoding import RealIndexSinusoidalPosEmb

        pe = RealIndexSinusoidalPosEmb(d_model=256)
        x = torch.randn(2, 16, 256)
        pos_ids = torch.arange(0, 16 * 8, 8).unsqueeze(0).expand(2, -1)
        out = pe(x, position_ids=pos_ids)
        assert out.shape == (2, 16, 256)

    def test_no_position_ids(self):
        from models.temporal.position_encoding import RealIndexSinusoidalPosEmb

        pe = RealIndexSinusoidalPosEmb(d_model=128)
        x = torch.randn(1, 10, 128)
        out = pe(x)  # should default to 0..9
        assert out.shape == (1, 10, 128)


# ─── V2 Transformer tests ───────────────────────────────────────────────────

class TestV2Transformer:
    def test_short_branch(self):
        from models.temporal.v2_transformer import V2TemporalPoseTransformer

        tf = V2TemporalPoseTransformer(
            d_model=64, n_heads=4, n_layers=2,
            dim_feedforward=128, branch_mode="short",
            window_size=16,
        )
        tokens = torch.randn(2, 16, 64)
        ctx = tf(tokens)
        assert ctx.shape == (2, 16, 64)

    def test_long_branch_global(self):
        from models.temporal.v2_transformer import V2TemporalPoseTransformer

        tf = V2TemporalPoseTransformer(
            d_model=64, n_heads=4, n_layers=2,
            dim_feedforward=128, branch_mode="long",
            attention_mode="global",
        )
        tokens = torch.randn(2, 16, 64)
        pos_ids = torch.arange(0, 16 * 8, 8).unsqueeze(0).expand(2, -1)
        ctx = tf(tokens, position_ids=pos_ids)
        assert ctx.shape == (2, 16, 64)

    def test_long_branch_dilated(self):
        from models.temporal.v2_transformer import V2TemporalPoseTransformer

        tf = V2TemporalPoseTransformer(
            d_model=64, n_heads=4, n_layers=2,
            dim_feedforward=128, branch_mode="long",
            attention_mode="dilated", window_size=8, dilation=2,
        )
        tokens = torch.randn(2, 16, 64)
        pos_ids = torch.arange(0, 16 * 4, 4).unsqueeze(0).expand(2, -1)
        ctx = tf(tokens, position_ids=pos_ids)
        assert ctx.shape == (2, 16, 64)


# ─── V2 Model tests ─────────────────────────────────────────────────────────

class TestV2Model:
    @pytest.fixture(params=["short", "long"])
    def branch(self, request):
        return request.param

    def test_forward(self, branch):
        from models.v2_dual_pose_model import V2LongSeqPoseModel

        model = V2LongSeqPoseModel(
            backbone="efficientnet_b0",
            token_dim=64,
            n_heads=4,
            n_layers=1,
            dim_feedforward=128,
            branch_mode=branch,
            window_size=8,
        )
        B, T, H, W = 1, 8, 64, 64
        frames = torch.randn(B, T, H, W)
        pos_ids = torch.arange(T).unsqueeze(0) * (1 if branch == "short" else 8)
        out = model(frames, position_ids=pos_ids)

        assert "pred_local_T" in out
        assert out["pred_local_T"].shape == (B, T, 4, 4)
        assert out["tokens"].shape == (B, T, 64)
        assert out["ctx"].shape == (B, T, 64)
        # frame 0 local should be identity
        assert torch.allclose(out["pred_local_T"][0, 0], torch.eye(4), atol=0.1)


# ─── V2 Dataset tests ───────────────────────────────────────────────────────

class TestV2Dataset:
    def test_stride_from_L_target(self):
        from data.datasets.kroot_dual_joint_v2 import SyntheticV2JointKRootDualDataset

        k = 64
        L_target = 504
        ds = SyntheticV2JointKRootDualDataset(
            k=k, L_target_frames=L_target,
            frames_per_scan=600, num_scans=1, height=32, width=32,
        )
        expected_s = math.ceil(L_target / (k - 1))
        assert ds.s == expected_s, f"Expected s={expected_s}, got {ds.s}"

    def test_yields_position_ids(self):
        from data.datasets.kroot_dual_joint_v2 import SyntheticV2JointKRootDualDataset

        ds = SyntheticV2JointKRootDualDataset(
            k=8, L_target_frames=56,
            frames_per_scan=200, num_scans=1, height=32, width=32,
            mode="train",
        )
        sample = next(iter(ds))
        assert "position_ids" in sample["short"], "Short branch missing position_ids"
        assert "position_ids" in sample["long"], "Long branch missing position_ids"
        assert sample["short"]["position_ids"].shape[0] == 8
        assert sample["long"]["position_ids"].shape[0] == 8

    def test_val_mode(self):
        from data.datasets.kroot_dual_joint_v2 import SyntheticV2JointKRootDualDataset

        ds = SyntheticV2JointKRootDualDataset(
            k=8, L_target_frames=56,
            frames_per_scan=200, num_scans=1, height=32, width=32,
            mode="val",
        )
        sample = next(iter(ds))
        assert "position_ids" in sample["short"]
        assert "position_ids" in sample["long"]
        # Val yields full scan
        T = sample["meta"]["total_frames"]
        assert sample["short"]["frames"].shape[0] == T

    def test_scan_adaptive(self):
        """When scan is shorter than L_target, s should be clamped."""
        from data.datasets.kroot_dual_joint_v2 import SyntheticV2JointKRootDualDataset

        ds = SyntheticV2JointKRootDualDataset(
            k=64, L_target_frames=5000,  # very large
            frames_per_scan=200, num_scans=1, height=32, width=32,
            mode="train",
        )
        # Computed s = ceil(5000/63) = 80, but scan only has 200 frames
        # Need (k-1)*s + 1 ≤ 200 → s ≤ (200-1)/63 ≈ 3.1 → s=3
        # The dataset should adaptively clamp s
        samples = list(ds)
        assert len(samples) > 0, "Should produce at least one sample from adaptive clamping"


# ─── V2 Trainer smoke test ───────────────────────────────────────────────────

class TestV2Trainer:
    def test_instantiation(self):
        """V2 trainer instantiates with synthetic data and runs one step."""
        from data.datasets.kroot_dual_joint_v2 import SyntheticV2JointKRootDualDataset
        from trainers.kroot_dual_v2_trainer import KRootDualV2Trainer
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "seed": 0,
            "kroot": {"k": 8, "L_target_frames": 56},
            "model": {
                "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
                "pose_head": {"rotation_rep": "rot6d"},
                "transformer": {
                    "d_model": 64, "n_heads": 4, "n_layers": 1,
                    "dim_feedforward": 128, "window_size": 8, "dropout": 0.0,
                },
                "short": {"transformer": {"attention_mode": "sliding_window"}},
                "long": {"transformer": {"attention_mode": "global", "dilation": 1}},
            },
            "optimizer": {
                "lr_rec": 1e-3, "weight_decay": 0, "betas": [0.9, 0.999],
                "min_lr": 1e-6, "warmup_steps": 0, "encoder_lr_mult": 1.0,
            },
            "loss": {
                "rot_weight": 1.0, "trans_weight": 1.0,
                "short_weight": 1.0, "long_weight": 1.0,
            },
            "trainer": {
                "max_epochs": 1, "log_interval": 1, "validate_every": 0,
                "grad_accum": 1, "max_grad_norm": 1.0, "ema_decay": 0,
            },
            "eval": {"csv_dir": None},
        })

        ds = SyntheticV2JointKRootDualDataset(
            k=8, L_target_frames=56,
            frames_per_scan=200, num_scans=1, height=64, width=64,
            mode="train",
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=1)

        trainer = KRootDualV2Trainer(
            cfg, device="cpu", joint_train_loader=loader,
        )

        # Run one step
        batch = next(iter(loader))
        trainer.short_model.train()
        trainer.long_model.train()
        loss, metrics = trainer._run_step(batch)
        assert loss.requires_grad
        assert "short_loss" in metrics
        assert "long_loss" in metrics
        loss.backward()
