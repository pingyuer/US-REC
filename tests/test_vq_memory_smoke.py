"""Smoke tests for VQ-Memory Conditioned Local Pose Transformer.

Tests cover:
  * VQTokenizerHead quantisation + EMA update
  * ScanSummaryPool (attention + latent)
  * VQMemoryCrossAttn
  * FiLMConditioner
  * ScanGeomHead + geom_loss + consistency_loss
  * VQMemoryPoseModel end-to-end
  * VQMemoryTrainer forward + backward
"""

import pytest
import torch
from omegaconf import OmegaConf


# ── VQTokenizerHead ──────────────────────────────────────────────────────

class TestVQTokenizer:
    def test_forward_shape(self):
        from models.vq.vq_tokenizer import VQTokenizerHead
        vq = VQTokenizerHead(code_dim=32, codebook_size=16, ema_decay=0.99)
        z_e = torch.randn(4, 32)
        out = vq(z_e)
        assert out["z_q"].shape == (4, 32)
        assert out["indices"].shape == (4,)
        assert out["commit_loss"].shape == ()
        assert out["z_q_detached"].shape == (4, 32)

    def test_batch_forward(self):
        from models.vq.vq_tokenizer import VQTokenizerHead
        vq = VQTokenizerHead(code_dim=32, codebook_size=16)
        z_e = torch.randn(2, 5, 32)  # (B, M, D)
        out = vq(z_e)
        assert out["z_q"].shape == (2, 5, 32)
        assert out["indices"].shape == (2, 5)

    def test_ema_updates_codebook(self):
        from models.vq.vq_tokenizer import VQTokenizerHead
        vq = VQTokenizerHead(code_dim=8, codebook_size=4, ema_decay=0.9)
        vq.train()
        cb_before = vq.embeddings.clone()
        z_e = torch.randn(10, 8)
        vq(z_e)
        # Codebook should have changed after EMA update
        assert not torch.allclose(vq.embeddings, cb_before, atol=1e-6)

    def test_codebook_usage(self):
        from models.vq.vq_tokenizer import VQTokenizerHead
        vq = VQTokenizerHead(code_dim=8, codebook_size=4)
        vq.train()
        vq.reset_usage()
        z_e = torch.randn(20, 8)
        vq(z_e)
        assert vq.codebook_usage > 0.0

    def test_straight_through_gradient(self):
        from models.vq.vq_tokenizer import VQTokenizerHead
        vq = VQTokenizerHead(code_dim=8, codebook_size=4)
        z_e = torch.randn(4, 8, requires_grad=True)
        out = vq(z_e)
        out["z_q"].sum().backward()
        assert z_e.grad is not None


# ── ScanSummaryPool ──────────────────────────────────────────────────────

class TestScanSummaryPool:
    def test_attention_pool(self):
        from models.vq.scan_summary import ScanSummaryPool
        pool = ScanSummaryPool(d_in=32, d_out=64, pool_type="attention")
        z_q = torch.randn(2, 8, 32)
        g = pool(z_q)
        assert g.shape == (2, 64)

    def test_latent_pool(self):
        from models.vq.scan_summary import ScanSummaryPool
        pool = ScanSummaryPool(d_in=32, d_out=64, pool_type="latent", n_latents=4, n_heads=2)
        z_q = torch.randn(2, 8, 32)
        g = pool(z_q)
        assert g.shape == (2, 64)

    def test_with_mask(self):
        from models.vq.scan_summary import ScanSummaryPool
        pool = ScanSummaryPool(d_in=32, d_out=64, pool_type="attention")
        z_q = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 6:] = False  # mask out last 2 for batch 0
        g = pool(z_q, mask=mask)
        assert g.shape == (2, 64)


# ── VQMemoryCrossAttn ────────────────────────────────────────────────────

class TestVQMemoryCrossAttn:
    def test_forward_shape(self):
        from models.vq.memory_cross_attn import VQMemoryCrossAttn
        xattn = VQMemoryCrossAttn(d_model=64, d_memory=32, n_heads=2)
        local_tokens = torch.randn(2, 16, 64)
        memory = torch.randn(2, 8, 32)
        c_t, weights = xattn(local_tokens, memory)
        assert c_t.shape == (2, 16, 64)
        assert weights.shape == (2, 16, 8)

    def test_gate_starts_near_zero(self):
        from models.vq.memory_cross_attn import VQMemoryCrossAttn
        xattn = VQMemoryCrossAttn(d_model=64, d_memory=32, n_heads=2)
        # Gate init is -3, sigmoid(-3) ≈ 0.047 — near-no-op start
        assert torch.sigmoid(xattn.gate).item() < 0.1

    def test_respects_memory_mask(self):
        from models.vq.memory_cross_attn import VQMemoryCrossAttn
        xattn = VQMemoryCrossAttn(d_model=32, d_memory=16, n_heads=2)
        local_tokens = torch.randn(1, 4, 32)
        memory = torch.randn(1, 3, 16)
        memory_mask = torch.tensor([[True, True, False]])
        _, weights = xattn(local_tokens, memory, memory_mask=memory_mask)
        assert weights[..., 2].abs().max().item() < 1e-5


# ── FiLMConditioner ──────────────────────────────────────────────────────

class TestFiLMConditioner:
    def test_film_only(self):
        from models.vq.film import FiLMConditioner
        film = FiLMConditioner(d_model=64, d_cond=32, use_film=True, use_global_token=False)
        x = torch.randn(2, 16, 64)
        g = torch.randn(2, 32)
        out = film(x, g)
        assert out.shape == (2, 16, 64)

    def test_global_token_only(self):
        from models.vq.film import FiLMConditioner
        film = FiLMConditioner(d_model=64, d_cond=32, use_film=False, use_global_token=True)
        x = torch.randn(2, 16, 64)
        g = torch.randn(2, 32)
        out = film(x, g)
        assert out.shape == (2, 16, 64)

    def test_film_starts_as_identity(self):
        from models.vq.film import FiLMConditioner
        film = FiLMConditioner(d_model=16, d_cond=16, use_film=True, use_global_token=False)
        x = torch.randn(1, 4, 16)
        g = torch.zeros(1, 16)  # zero conditioning → γ≈1, β≈0
        out = film(x, g)
        assert torch.allclose(out, x, atol=0.05)  # approximate identity


class TestScanContextEncoder:
    def test_forward_shape(self):
        from models.vq.scan_context import ScanContextEncoder
        enc = ScanContextEncoder(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64, dropout=0.0)
        z_q = torch.randn(2, 8, 32)
        out = enc(z_q)
        assert out.shape == (2, 8, 32)

    def test_respects_mask(self):
        from models.vq.scan_context import ScanContextEncoder
        enc = ScanContextEncoder(d_model=16, n_heads=4, n_layers=1, dim_feedforward=32, dropout=0.0)
        z_q = torch.randn(1, 4, 16)
        mask = torch.tensor([[True, True, False, False]])
        out = enc(z_q, mask=mask)
        assert out.shape == (1, 4, 16)
        assert torch.isfinite(out).all()


# ── ScanGeomHead + losses ────────────────────────────────────────────────

class TestScanGeomHead:
    def test_forward_shape(self):
        from models.vq.scan_geom_head import ScanGeomHead
        head = ScanGeomHead(d_in=64, n_waypoints=8, output_per_wp=6)
        g = torch.randn(2, 64)
        pred = head(g)
        assert pred.shape == (2, 8, 6)

    def test_build_geom_target(self):
        from models.vq.scan_geom_head import build_geom_target
        gt = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 32, -1, -1).clone()
        target = build_geom_target(gt, n_waypoints=8)
        assert target.shape == (2, 8, 6)
        # Identity transforms → all zeros
        assert target.abs().max() < 0.01

    def test_geom_loss(self):
        from models.vq.scan_geom_head import geom_loss
        pred = torch.randn(2, 8, 6)
        target = torch.randn(2, 8, 6)
        loss = geom_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_consistency_loss(self):
        from models.vq.scan_geom_head import consistency_loss
        g1 = torch.randn(4, 64)
        loss_diff = consistency_loss(g1, torch.randn(4, 64))
        loss_same = consistency_loss(g1, g1)
        assert loss_same.item() < 0.01  # same → cos_sim ≈ 1
        assert loss_diff.item() > loss_same.item()


# ── VQMemoryPoseModel ────────────────────────────────────────────────────

class TestVQMemoryPoseModel:
    @pytest.fixture
    def model(self):
        from models.vq.vq_memory_model import VQMemoryPoseModel
        return VQMemoryPoseModel(
            backbone="efficientnet_b0",
            in_channels=1,
            token_dim=32,
            code_dim=32,
            codebook_size=8,
            ema_decay=0.99,
            anchor_stride=4,
            n_heads=2,
            n_layers=1,
            dim_feedforward=64,
            window_size=8,
            dropout=0.0,
            rotation_rep="rot6d",
            aux_intervals=[2],
            pretrained_backbone=False,
            memory_size=0,
            pool_type="attention",
            n_geom_waypoints=4,
            use_film=True,
            use_global_token=False,
            use_memory_cross_attn=True,
            memory_n_heads=2,
        )

    def test_forward_with_scan_frames(self, model):
        B, T, H, W = 1, 8, 32, 32
        frames = torch.randn(B, T, H, W)
        out = model(frames, scan_frames=frames)
        assert out["pred_local_T"].shape == (B, T, 4, 4)
        assert out["g"].shape == (B, 32)
        assert out["pred_geom"] is not None
        assert out["attn_weights"] is not None

    def test_forward_with_cache(self, model):
        B, T, H, W = 1, 8, 32, 32
        frames = torch.randn(B, T, H, W)
        cache = model.encode_scan_anchors(frames)
        assert cache["z_ctx"].shape == cache["z_q"].shape
        out = model(frames, scan_vq_cache=cache)
        assert out["pred_local_T"].shape == (B, T, 4, 4)

    def test_forward_without_vq(self, model):
        """When no scan context is available, falls back to vanilla transformer."""
        B, T, H, W = 1, 8, 32, 32
        frames = torch.randn(B, T, H, W)
        out = model(frames)
        assert out["pred_local_T"].shape == (B, T, 4, 4)
        assert out["g"] is None  # no VQ context
        assert out["pred_geom"] is None

    def test_backward(self, model):
        model.train()
        B, T, H, W = 1, 8, 32, 32
        frames = torch.randn(B, T, H, W)
        out = model(frames, scan_frames=frames)
        loss = out["pred_local_T"].sum() + out["commit_loss"]
        loss.backward()
        # Check gradients flow to encoder
        for p in model.encoder.parameters():
            if p.requires_grad and p.grad is not None:
                assert p.grad.abs().sum() > 0
                break

    def test_proj_vq_and_proj_tf_are_independent(self, model):
        """Verify proj_vq and proj_tf don't share parameters."""
        vq_params = set(id(p) for p in model.proj_vq.parameters())
        tf_params = set(id(p) for p in model.proj_tf.parameters())
        assert vq_params.isdisjoint(tf_params)


# ── VQMemoryTrainer ──────────────────────────────────────────────────────

class TestVQMemoryTrainer:
    @pytest.fixture
    def cfg(self):
        return OmegaConf.create({
            "seed": 0,
            "paths": {"output_dir": "logs"},
            "model": {
                "type": "vq_memory",
                "pose_head": {"rotation_rep": "rot6d"},
                "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
                "transformer": {
                    "d_model": 32, "n_heads": 2, "n_layers": 1,
                    "dim_feedforward": 64, "window_size": 8, "dropout": 0.0,
                    "memory_size": 0,
                },
                "anchor_stride": 4,
                "n_geom_waypoints": 4,
                "image_size_h": 32, "image_size_w": 32,
            },
            "vq": {
                "codebook_size": 8, "code_dim": 32, "ema_decay": 0.99,
                "anchor_stride": 4, "n_geom_waypoints": 4,
            },
            "summary": {"pool_type": "attention", "latent_num": 4},
            "memory": {"use_cross_attn": True, "num_heads": 2},
            "conditioning": {"use_film": True, "use_global_token": False},
            "optimizer": {
                "lr_rec": 1e-3, "lr": 1e-3, "weight_decay": 0,
                "encoder_lr_mult": 1.0, "warmup_steps": 0,
                "min_lr": 1e-6,
            },
            "loss": {
                "mode": "points", "ref_pts_scale_mm": 20.0,
                "rot_weight": 1.0, "trans_weight": 1.0,
                "aux_intervals": [2], "aux_weight": 0.5,
                "aux_decay": 0.5, "aux_scale": "none",
                "consistency_weight": 0.0, "consistency_delta": 2,
                "ddf_sample_weight": 0.0, "ddf_num_points": 0,
                "lambda_cons": 0.1, "lambda_geom": 0.1,
                "lambda_commit": 0.25, "cons_sample_ratio": 0.6,
            },
            "eval": {"metric_mode": "points"},
            "trainer": {
                "name": "trainers.vq_memory_trainer.VQMemoryTrainer",
                "max_epochs": 1, "log_interval": 1, "validate_every": 1,
                "ema_decay": 0, "grad_accum": 1, "max_grad_norm": 1.0,
            },
        })

    def test_trainer_init(self, cfg):
        from trainers.vq_memory_trainer import VQMemoryTrainer
        trainer = VQMemoryTrainer(cfg, device="cpu")
        assert hasattr(trainer, "model")
        assert hasattr(trainer, "optimizer")

    def test_run_step(self, cfg):
        from trainers.vq_memory_trainer import VQMemoryTrainer
        trainer = VQMemoryTrainer(cfg, device="cpu")
        trainer.model.train()
        batch = {
            "frames": torch.randn(1, 8, 32, 32),
            "gt_global_T": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1).clone(),
        }
        loss, metrics = trainer._run_step(batch)
        assert loss.requires_grad
        assert "loss_cons" in metrics
        assert "loss_geom" in metrics
        assert "vq_codebook_usage" in metrics
        assert "summary_g_norm" in metrics

    def test_run_step_with_padded_scan_context(self, cfg):
        from trainers.vq_memory_trainer import VQMemoryTrainer
        trainer = VQMemoryTrainer(cfg, device="cpu")
        trainer.model.train()
        batch = {
            "frames": torch.randn(2, 8, 32, 32),
            "gt_global_T": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 8, -1, -1).clone(),
            "scan_frames": torch.randn(2, 10, 32, 32),
            "scan_frames_mask": torch.tensor([
                [True] * 10,
                [True] * 6 + [False] * 4,
            ]),
            "scan_gt_global_T": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 10, -1, -1).clone(),
            "scan_gt_global_T_mask": torch.tensor([
                [True] * 10,
                [True] * 6 + [False] * 4,
            ]),
        }
        loss, metrics = trainer._run_step(batch)
        assert loss.requires_grad
        assert metrics["summary_g_norm"] >= 0.0

    def test_evaluate(self, cfg):
        from trainers.vq_memory_trainer import VQMemoryTrainer
        trainer = VQMemoryTrainer(cfg, device="cpu")

        # Minimal val loader
        class FakeLoader:
            def __iter__(self):
                yield {
                    "frames": torch.randn(1, 8, 32, 32),
                    "gt_global_T": torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 8, -1, -1).clone(),
                }
        metrics = trainer.evaluate(FakeLoader())
        assert "mean_gpe_pts_mm" in metrics
        assert "mean_lpe_pts_mm" in metrics
