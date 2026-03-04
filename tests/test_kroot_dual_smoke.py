"""Smoke tests for KRootDualTrainer — joint training + stitch eval.

Exercises the full unified pipeline:
  synthetic data → dual trainer (train both branches) → stitch eval → metrics
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


# ── Helpers ──────────────────────────────────────────────────────────────────

K = 16          # small for fast testing
S = 4
H, W = 16, 16
T_SHORT = 64    # frames per synthetic scan for short windows
T_LONG = 128    # frames per synthetic scan for long windows (needs k*s = 64 span)


def _make_cfg(**overrides):
    """Minimal config for KRootDualTrainer."""
    cfg = OmegaConf.create({
        "seed": 42,
        "kroot": {"k": K, "s": S},
        "model": {
            "type": "kroot_dual",
            "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
            "transformer": {
                "d_model": 32, "n_heads": 2, "n_layers": 1,
                "dim_feedforward": 64, "window_size": K, "dropout": 0.0,
            },
            "short": {"transformer": {"n_layers": 1, "window_size": K}},
            "long": {"transformer": {"n_layers": 1, "window_size": K}},
            "pose_head": {"rotation_rep": "rot6d"},
        },
        "optimizer": {
            "lr_rec": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999],
            "min_lr": 1e-6, "warmup_steps": 0, "encoder_lr_mult": 0.1,
        },
        "loss": {"rot_weight": 1.0, "trans_weight": 1.0, "short_weight": 1.0, "long_weight": 1.0},
        "trainer": {
            "max_epochs": 1, "log_interval": 1, "validate_every": 99,
            "grad_accum": 1, "max_grad_norm": 1.0, "ema_decay": 0.0,
        },
        "train": {"max_steps": 2},
        "stitch": {
            "enable_endpoint_interp": True, "short_overlap": 4,
            "long_window_stride": max(1, K - 1),
        },
        "paths": {"output_dir": tempfile.mkdtemp()},
    })
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def _make_loaders():
    """Build synthetic short/long train loaders + val loader."""
    from data.datasets.dual_kroot_window import (
        SyntheticShortWindowDataset,
        SyntheticLongWindowDataset,
    )

    short_train = SyntheticShortWindowDataset(
        num_scans=2, frames_per_scan=T_SHORT,
        height=H, width=W, k=K, overlap=4, mode="train",
    )
    long_train = SyntheticLongWindowDataset(
        num_scans=2, frames_per_scan=T_LONG,
        height=H, width=W, k=K, s=S, mode="train",
    )
    val_ds = SyntheticShortWindowDataset(
        num_scans=1, frames_per_scan=T_SHORT,
        height=H, width=W, k=K, overlap=4, mode="val",
    )
    return (
        DataLoader(short_train, batch_size=1),
        DataLoader(long_train, batch_size=1),
        DataLoader(val_ds, batch_size=1),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

class TestKRootDualTrainer:
    """Smoke-level tests for unified dual training."""

    def test_import(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        assert KRootDualTrainer is not None

    def test_init(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        assert trainer.short_model is not None
        assert trainer.long_model is not None
        assert trainer.k == K
        assert trainer.s == S

    def test_train_smoke(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        trainer.train()
        assert trainer.global_step >= 1

    def test_evaluate_returns_metrics(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        metrics = trainer.evaluate(val_l)
        assert "mean_gpe_mm_fused" in metrics
        assert "mean_gpe_mm_short_only" in metrics
        assert "mean_gpe_mm_long_only" in metrics
        assert "mean_drift_last_mm_fused" in metrics

    def test_checkpoint_roundtrip(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer1 = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        trainer1.train()

        ckpt_path = os.path.join(cfg.paths.output_dir, "test_ckpt.pt")
        trainer1.save_checkpoint(ckpt_path, tag="test")
        assert os.path.exists(ckpt_path)

        # Reload into a fresh trainer
        trainer2 = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.epoch == trainer1.epoch
        assert trainer2.global_step == trainer1.global_step

        # Verify both models produce same output
        with torch.no_grad():
            dummy = torch.rand(1, K, H, W)
            trainer1.short_model.eval()
            trainer2.short_model.eval()
            out1_s = trainer1.short_model(dummy)["pred_local_T"]
            out2_s = trainer2.short_model(dummy)["pred_local_T"]
            assert torch.allclose(out1_s, out2_s, atol=1e-4), \
                f"Short model mismatch: max diff = {(out1_s - out2_s).abs().max()}"

            trainer1.long_model.eval()
            trainer2.long_model.eval()
            out1_l = trainer1.long_model(dummy)["pred_local_T"]
            out2_l = trainer2.long_model(dummy)["pred_local_T"]
            assert torch.allclose(out1_l, out2_l, atol=1e-4), \
                f"Long model mismatch: max diff = {(out1_l - out2_l).abs().max()}"

    def test_model_wrapper(self):
        """The .model property should expose both models for CheckpointHook."""
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        model = trainer.model
        assert isinstance(model, torch.nn.ModuleDict)
        assert "short" in model
        assert "long" in model

    def test_train_with_ema(self):
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg(**{"trainer": {"ema_decay": 0.99}})
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        trainer.train()
        assert trainer.short_ema is not None
        assert trainer.long_ema is not None
        # EMA eval should work
        metrics = trainer.evaluate(val_l)
        assert "mean_gpe_mm_fused" in metrics

    def test_train_then_eval_full_pipeline(self):
        """Full pipeline: train both → stitch eval → check all metric keys."""
        from trainers.kroot_dual_trainer import KRootDualTrainer
        cfg = _make_cfg()
        short_tl, long_tl, val_l = _make_loaders()
        trainer = KRootDualTrainer(
            cfg, device="cpu",
            short_train_loader=short_tl,
            long_train_loader=long_tl,
            val_loader=val_l,
        )
        trainer.train()
        metrics = trainer.evaluate(val_l)

        expected_keys = [
            "mean_gpe_mm_fused", "mean_gpe_mm_short_only", "mean_gpe_mm_long_only",
            "mean_lpe_mm_fused", "mean_lpe_mm_short_only", "mean_lpe_mm_long_only",
            "mean_drift_last_mm_fused", "mean_drift_last_mm_short_only", "mean_drift_last_mm_long_only",
        ]
        for k in expected_keys:
            assert k in metrics, f"Missing key: {k}"
            assert isinstance(metrics[k], float), f"Key {k} should be float, got {type(metrics[k])}"


# ── Standalone runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("K-root Dual Trainer smoke test (standalone)")
    print("=" * 60)

    t = TestKRootDualTrainer()
    for name in sorted(dir(t)):
        if name.startswith("test_"):
            print(f"\n--- {name} ---")
            getattr(t, name)()
            print(f"  ✓ {name} passed")

    print("\n" + "=" * 60)
    print("All dual trainer smoke tests passed!")
    print("=" * 60)
