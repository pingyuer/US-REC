"""Smoke test for the K-root dual short/long + stitch pipeline.

Tests:
1. ShortWindowDataset and LongWindowDataset yield correct shapes
2. KRootTrainer can train short and long branches (synthetic data)
3. kroot_stitch can fuse short + long predictions into a global trajectory
4. Metrics are computed for short_only / long_only / fused
5. Debug CSV is exported

Run::

    pytest tests/test_kroot_stitch_smoke.py -v
    python tests/test_kroot_stitch_smoke.py  # standalone
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
K = 16
S = 4  # int(round(sqrt(16)))
FRAMES_PER_SCAN = 128
NUM_SCANS = 2
H, W = 32, 32  # small for speed


def _random_walk_transforms(T: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    eye = torch.eye(4)
    out = [eye]
    for _ in range(T - 1):
        angle = torch.randn(3, generator=gen) * 0.02
        t = torch.randn(3, generator=gen) * 0.5
        theta = angle.norm()
        if theta > 1e-8:
            k = angle / theta
            K_mat = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = torch.eye(3) + torch.sin(theta) * K_mat + (1 - torch.cos(theta)) * (K_mat @ K_mat)
        else:
            R = torch.eye(3)
        step = torch.eye(4)
        step[:3, :3] = R
        step[:3, 3] = t
        out.append(out[-1] @ step)
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# 1. Dataset tests
# ---------------------------------------------------------------------------

class TestShortWindowDataset:
    def test_synthetic_shape(self):
        from data.datasets.dual_kroot_window import SyntheticShortWindowDataset
        ds = SyntheticShortWindowDataset(
            num_scans=NUM_SCANS, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, overlap=4, mode="train", seed=0,
        )
        samples = list(ds)
        assert len(samples) > 0, "No samples produced"
        for s in samples:
            assert s["frames"].shape[0] == K, f"Expected k={K}, got {s['frames'].shape[0]}"
            assert s["gt_global_T"].shape == (K, 4, 4)
            # Frame 0 should be identity
            assert torch.allclose(s["gt_global_T"][0], torch.eye(4), atol=1e-5)

    def test_synthetic_val_full_scan(self):
        from data.datasets.dual_kroot_window import SyntheticShortWindowDataset
        ds = SyntheticShortWindowDataset(
            num_scans=1, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, mode="val", seed=0,
        )
        samples = list(ds)
        assert len(samples) == 1
        assert samples[0]["frames"].shape[0] == FRAMES_PER_SCAN

    def test_no_remainder_exceeds_k(self):
        """Ensure no training sample has more than k tokens."""
        from data.datasets.dual_kroot_window import SyntheticShortWindowDataset
        ds = SyntheticShortWindowDataset(
            num_scans=4, frames_per_scan=100,  # not divisible by k
            height=H, width=W, k=K, overlap=0, mode="train", seed=0,
        )
        for s in ds:
            assert s["frames"].shape[0] == K


class TestLongWindowDataset:
    def test_synthetic_shape(self):
        from data.datasets.dual_kroot_window import SyntheticLongWindowDataset
        ds = SyntheticLongWindowDataset(
            num_scans=NUM_SCANS, frames_per_scan=FRAMES_PER_SCAN * 4,
            height=H, width=W, k=K, s=S, mode="train", seed=0,
        )
        samples = list(ds)
        assert len(samples) > 0
        for s in samples:
            assert s["frames"].shape[0] == K
            assert s["gt_global_T"].shape == (K, 4, 4)
            assert s["idx_long"].shape[0] == K
            # Check stride between consecutive indices
            diffs = s["idx_long"][1:] - s["idx_long"][:-1]
            assert (diffs == S).all(), f"Expected stride {S}, got {diffs.tolist()}"

    def test_synthetic_val_full(self):
        from data.datasets.dual_kroot_window import SyntheticLongWindowDataset
        ds = SyntheticLongWindowDataset(
            num_scans=1, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, s=S, mode="val", seed=0,
        )
        samples = list(ds)
        assert len(samples) == 1
        assert samples[0]["frames"].shape[0] == FRAMES_PER_SCAN


# ---------------------------------------------------------------------------
# 2. KRootTrainer smoke test
# ---------------------------------------------------------------------------

def _make_cfg(branch: str) -> OmegaConf:
    """Build a minimal OmegaConf for KRootTrainer smoke test."""
    return OmegaConf.create({
        "seed": 42,
        "kroot": {"branch": branch, "k": K, "s": S},
        "model": {
            "type": "kroot",
            "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
            "transformer": {
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 1,
                "dim_feedforward": 128,
                "window_size": K,
                "dropout": 0.0,
                "seg_len": 0,
                "memory_size": 0,
            },
            "pose_head": {"rotation_rep": "rot6d"},
        },
        "optimizer": {"lr_rec": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "loss": {"rot_weight": 1.0, "trans_weight": 1.0},
        "trainer": {
            "max_epochs": 1,
            "log_interval": 1,
            "validate_every": 1,
            "grad_accum": 1,
            "max_grad_norm": 1.0,
        },
        "train": {"max_steps": 2},
        "paths": {"output_dir": "logs"},
    })


class TestKRootTrainer:
    def test_short_train_smoke(self):
        from data.datasets.dual_kroot_window import SyntheticShortWindowDataset
        from trainers.kroot_trainer import KRootTrainer
        from torch.utils.data import DataLoader

        cfg = _make_cfg("short")
        ds = SyntheticShortWindowDataset(
            num_scans=2, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, overlap=0, mode="train",
        )
        ds_val = SyntheticShortWindowDataset(
            num_scans=1, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, mode="val",
        )
        trainer = KRootTrainer(
            cfg, device="cpu",
            train_loader=DataLoader(ds, batch_size=1),
            val_loader=DataLoader(ds_val, batch_size=1),
        )
        trainer.train()
        assert trainer.global_step >= 1

    def test_long_train_smoke(self):
        from data.datasets.dual_kroot_window import SyntheticLongWindowDataset
        from trainers.kroot_trainer import KRootTrainer
        from torch.utils.data import DataLoader

        cfg = _make_cfg("long")
        ds = SyntheticLongWindowDataset(
            num_scans=2, frames_per_scan=FRAMES_PER_SCAN * 4,
            height=H, width=W, k=K, s=S, mode="train",
        )
        ds_val = SyntheticLongWindowDataset(
            num_scans=1, frames_per_scan=FRAMES_PER_SCAN * 4,
            height=H, width=W, k=K, s=S, mode="val",
        )
        trainer = KRootTrainer(
            cfg, device="cpu",
            train_loader=DataLoader(ds, batch_size=1),
            val_loader=DataLoader(ds_val, batch_size=1),
        )
        trainer.train()
        assert trainer.global_step >= 1


# ---------------------------------------------------------------------------
# 3. Stitch test
# ---------------------------------------------------------------------------

class TestKRootStitch:
    def _make_models(self):
        """Create two tiny LongSeqPoseModel instances (short + long)."""
        from models.temporal.model_longseq import LongSeqPoseModel
        kwargs = dict(
            backbone="efficientnet_b0",
            in_channels=1,
            token_dim=64,
            n_heads=2,
            n_layers=1,
            dim_feedforward=128,
            window_size=K,
            dropout=0.0,
            rotation_rep="rot6d",
            aux_intervals=[],
            memory_size=0,
        )
        short_model = LongSeqPoseModel(**kwargs)
        long_model = LongSeqPoseModel(**kwargs)
        return short_model, long_model

    def test_stitch_from_predictions(self):
        from eval.kroot_stitch import stitch_from_predictions
        from metrics.compose import compose_global_from_local

        T = 64
        # Synthetic short locals (identity + small random)
        short_local = torch.eye(4).unsqueeze(0).expand(T, -1, -1).clone()
        for i in range(1, T):
            short_local[i, :3, 3] = torch.randn(3) * 0.1

        anchor_indices = torch.arange(0, T, S)
        M = anchor_indices.shape[0]
        # Synthetic long anchor globals
        long_global = compose_global_from_local(short_local.unsqueeze(0)).squeeze(0)
        long_anchor_global = long_global[anchor_indices]

        fused = stitch_from_predictions(
            short_local, long_anchor_global, anchor_indices, T, enable_endpoint_interp=True,
        )
        assert fused.shape == (T, 4, 4)
        # Frame 0 should be identity
        assert torch.allclose(fused[0], torch.eye(4), atol=1e-4)

    def test_stitch_model_based(self):
        """Test stitch_long_base_short_refine with tiny models."""
        from eval.kroot_stitch import stitch_long_base_short_refine

        short_model, long_model = self._make_models()
        T = 64
        frames = torch.rand(T, H, W)
        result = stitch_long_base_short_refine(
            short_model, long_model, frames, k=K, s=S,
            device=torch.device("cpu"), short_overlap=4,
        )
        assert result["fused_global"].shape == (T, 4, 4)
        assert result["short_global"].shape == (T, 4, 4)
        assert len(result["anchor_indices"]) > 0

    def test_compute_metrics(self):
        from eval.kroot_stitch import compute_stitch_metrics
        from metrics.compose import compose_global_from_local

        T = 64
        gt_global = _random_walk_transforms(T, seed=10)
        # Simulate predictions with some noise
        pred_local = torch.eye(4).unsqueeze(0).expand(T, -1, -1).clone()
        for i in range(1, T):
            pred_local[i, :3, 3] = torch.randn(3) * 0.1
        short_global = compose_global_from_local(pred_local.unsqueeze(0)).squeeze(0)
        fused_global = short_global.clone()  # no long correction

        anchor_indices = torch.arange(0, T, S)
        long_global = gt_global[anchor_indices]  # "perfect" anchors

        metrics = compute_stitch_metrics(
            fused_global=fused_global,
            short_global=short_global,
            long_global=long_global,
            anchor_indices=anchor_indices,
            gt_global=gt_global,
        )
        assert "gpe_mm_fused" in metrics
        assert "gpe_mm_short_only" in metrics
        assert "gpe_mm_long_only" in metrics
        assert "drift_last_mm_fused" in metrics

    def test_export_csv(self):
        from eval.kroot_stitch import export_debug_csv
        from metrics.compose import compose_global_from_local

        T = 32
        gt_global = _random_walk_transforms(T, seed=20)
        short_global = _random_walk_transforms(T, seed=21)
        fused_global = short_global.clone()
        anchor_indices = torch.arange(0, T, S)
        long_global = gt_global[anchor_indices]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = export_debug_csv(
                "test/scan_0", fused_global, short_global, long_global,
                anchor_indices, gt_global, out_dir=tmpdir,
            )
            assert os.path.isfile(csv_path)
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == T
                assert "t_err_mm_fused" in rows[0]


# ---------------------------------------------------------------------------
# 4. End-to-end smoke: train short + long, then stitch
# ---------------------------------------------------------------------------

class TestEndToEndSmoke:
    def test_full_pipeline(self):
        """Train short + long on synth data, then stitch and compute metrics."""
        from data.datasets.dual_kroot_window import (
            SyntheticShortWindowDataset,
            SyntheticLongWindowDataset,
        )
        from trainers.kroot_trainer import KRootTrainer
        from eval.kroot_stitch import (
            stitch_long_base_short_refine,
            compute_stitch_metrics,
            export_debug_csv,
        )
        from torch.utils.data import DataLoader

        # Train short
        cfg_short = _make_cfg("short")
        OmegaConf.update(cfg_short, "train.max_steps", 2)
        ds_short = SyntheticShortWindowDataset(
            num_scans=2, frames_per_scan=FRAMES_PER_SCAN,
            height=H, width=W, k=K, mode="train",
        )
        trainer_short = KRootTrainer(
            cfg_short, device="cpu",
            train_loader=DataLoader(ds_short, batch_size=1),
        )
        trainer_short.train()

        # Train long
        cfg_long = _make_cfg("long")
        OmegaConf.update(cfg_long, "train.max_steps", 2)
        ds_long = SyntheticLongWindowDataset(
            num_scans=2, frames_per_scan=FRAMES_PER_SCAN * 4,
            height=H, width=W, k=K, s=S, mode="train",
        )
        trainer_long = KRootTrainer(
            cfg_long, device="cpu",
            train_loader=DataLoader(ds_long, batch_size=1),
        )
        trainer_long.train()

        # Stitch on synthetic val scan
        T = 64
        frames = torch.rand(T, H, W) * 255
        gt_global = _random_walk_transforms(T, seed=99)

        result = stitch_long_base_short_refine(
            short_model=trainer_short.model,
            long_model=trainer_long.model,
            scan_frames=frames,
            k=K, s=S,
            device=torch.device("cpu"),
            short_overlap=4,
        )

        metrics = compute_stitch_metrics(
            fused_global=result["fused_global"],
            short_global=result["short_global"],
            long_global=result["long_global"],
            anchor_indices=result["anchor_indices"],
            gt_global=gt_global,
        )

        print("\n=== K-root Stitch Smoke Results ===")
        for key in sorted(metrics):
            if isinstance(metrics[key], float):
                print(f"  {key:35s}: {metrics[key]:.6f}")
            else:
                print(f"  {key:35s}: {metrics[key]}")

        # All three variants should have metrics
        assert "gpe_mm_fused" in metrics
        assert "gpe_mm_short_only" in metrics
        assert "gpe_mm_long_only" in metrics
        assert "drift_last_mm_fused" in metrics
        assert "drift_last_mm_short_only" in metrics
        assert "drift_last_mm_long_only" in metrics

        # Export CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = export_debug_csv(
                "smoke/scan_0",
                result["fused_global"],
                result["short_global"],
                result["long_global"],
                result["anchor_indices"],
                gt_global,
                out_dir=tmpdir,
            )
            assert os.path.isfile(csv_path)
            print(f"  Debug CSV: {csv_path}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("K-root stitch smoke test (standalone)")
    print("=" * 60)

    print("\n--- ShortWindowDataset ---")
    TestShortWindowDataset().test_synthetic_shape()
    TestShortWindowDataset().test_no_remainder_exceeds_k()
    print("  PASS")

    print("\n--- LongWindowDataset ---")
    TestLongWindowDataset().test_synthetic_shape()
    print("  PASS")

    print("\n--- KRootTrainer short ---")
    TestKRootTrainer().test_short_train_smoke()
    print("  PASS")

    print("\n--- KRootTrainer long ---")
    TestKRootTrainer().test_long_train_smoke()
    print("  PASS")

    print("\n--- Stitch from predictions ---")
    TestKRootStitch().test_stitch_from_predictions()
    print("  PASS")

    print("\n--- Stitch model-based ---")
    TestKRootStitch().test_stitch_model_based()
    print("  PASS")

    print("\n--- Compute metrics ---")
    TestKRootStitch().test_compute_metrics()
    print("  PASS")

    print("\n--- Export CSV ---")
    TestKRootStitch().test_export_csv()
    print("  PASS")

    print("\n--- Full pipeline ---")
    TestEndToEndSmoke().test_full_pipeline()
    print("  PASS")

    print("\n" + "=" * 60)
    print("All K-root stitch smoke tests passed!")
    print("=" * 60)
