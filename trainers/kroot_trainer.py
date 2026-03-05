"""K-root Short/Long independent trainer.

Trains a single LongSeqPoseModel (no auxΔ, no consistency) for either:
* **short** mode — contiguous frames, Δ=1 local supervision
* **long** mode  — sparse frames at stride *s*, Δ=1 (in sparse domain) supervision

The two modes share the same model architecture and loss;
only the dataset and stride differ.  They are trained with
separate configs and produce separate checkpoints.

Now inherits from :class:`BaseTrainer` for shared boilerplate (hooks,
checkpoint, gradient accumulation, train loop skeleton).

Usage::

    trainer = KRootTrainer(cfg, device=..., train_loader=..., val_loader=...)
    trainer.train()
    trainer.evaluate(loader)
"""

from __future__ import annotations

import math
import os
import time
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import compose_global_from_local, local_from_global
from trainers.base_trainer import BaseTrainer
from trainers.common import cfg_get, load_tform_calib


class KRootTrainer(BaseTrainer):
    """Self-contained trainer for one branch (short or long) of the K-root scheme.

    Inherits generic train loop, hooks, checkpoint from :class:`BaseTrainer`.
    Uses constant LR (no warm-up / cosine decay) to match original behaviour.

    Parameters
    ----------
    cfg : OmegaConf config
    device : torch device
    train_loader, val_loader : DataLoader or None
    """

    def __init__(
        self,
        cfg,
        *,
        device: str | torch.device = "cpu",
        train_loader=None,
        val_loader=None,
    ):
        super().__init__(cfg, device=device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Branch mode ──────────────────────────────────────────────
        self.branch = str(cfg_get(cfg, "kroot.branch", "short")).lower()
        assert self.branch in ("short", "long"), f"Unknown branch: {self.branch}"
        self.k = int(cfg_get(cfg, "kroot.k", 64))
        self.s = int(cfg_get(cfg, "kroot.s", 0))
        if self.s <= 0:
            self.s = int(round(math.sqrt(self.k)))

        # ── Model ────────────────────────────────────────────────────
        self.rotation_rep = str(cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ── Optimizer ────────────────────────────────────────────────
        lr = float(cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 1e-4)))
        weight_decay = float(cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.base_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay, betas=betas,
        )

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(cfg_get(cfg, "loss.trans_weight", 1.0))

        # ── Calibration (for eval metrics) ───────────────────────────
        self.tform_calib: torch.Tensor | None = load_tform_calib(
            cfg, device=self.device, warn_prefix="KRootTrainer",
        )

    # ── Model construction ───────────────────────────────────────────

    def _build_model(self) -> LongSeqPoseModel:
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))

        return LongSeqPoseModel(
            backbone=backbone,
            in_channels=1,
            token_dim=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_ff,
            window_size=window_size,
            dropout=dropout,
            rotation_rep=self.rotation_rep,
            aux_intervals=[],  # no auxΔ
            pretrained_backbone=pretrained,
            memory_size=0,
        )

    # ── BaseTrainer overrides ────────────────────────────────────────

    def _update_lr(self) -> float:
        """KRootTrainer uses constant LR (no warmup / cosine decay)."""
        return float(self.optimizer.param_groups[0].get("lr", self.base_lr))

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        tag = f"KRoot-{self.branch}"
        return (
            f"[{tag} epoch {epoch}  step {n_optim_steps}]  "
            f"loss={avg_accum:.4f}  "
            f"rot={metrics.get('rot_loss', 0):.4f}  "
            f"trans={metrics.get('trans_loss', 0):.4f}  "
            f"drift={metrics.get('drift_mm_last', 0):.2f}mm"
        )

    # ── Single training step ─────────────────────────────────────────

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + chordal-rot + smooth-L1-trans loss."""
        frames = batch["frames"].to(self.device)
        gt_global = batch["gt_global_T"].to(self.device)

        if frames.max() > 1.1:
            frames = frames / 255.0

        out = self.model(frames)
        pred_local_T = out["pred_local_T"]  # (B, T, 4, 4)

        # Derive GT locals from GT globals
        gt_local_T = local_from_global(gt_global)  # (B, T, 4, 4)

        # Loss on frames 1..T-1 (frame 0 is identity)
        pred_R = pred_local_T[:, 1:, :3, :3]
        gt_R = gt_local_T[:, 1:, :3, :3]
        pred_t = pred_local_T[:, 1:, :3, 3]
        gt_t = gt_local_T[:, 1:, :3, 3]

        rot_loss = chordal_rotation_loss(pred_R, gt_R)
        trans_loss = F.smooth_l1_loss(pred_t, gt_t, beta=1.0)
        loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

        with torch.no_grad():
            pred_global = compose_global_from_local(pred_local_T)
            drift = (pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]).norm(dim=-1).mean()

        breakdown = {
            "rot_loss": float(rot_loss.item()),
            "trans_loss": float(trans_loss.item()),
            "drift_mm_last": float(drift.item()),
        }
        return loss, breakdown

    # ── Evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Evaluate on full-scan loader; compute SE(3) GPE/LPE."""
        self.model.eval()
        scan_metrics: list[dict[str, float]] = []

        for batch in loader:
            frames = batch["frames"].to(self.device)
            gt_global = batch["gt_global_T"].to(self.device)

            if frames.max() > 1.1:
                frames = frames / 255.0

            out = self.model(frames)
            pred_local_T = out["pred_local_T"]
            pred_global = compose_global_from_local(pred_local_T)
            gt_local = local_from_global(gt_global)

            B, T = pred_local_T.shape[:2]
            for b in range(B):
                lpe = (pred_local_T[b, 1:, :3, 3] - gt_local[b, 1:, :3, 3]).norm(dim=-1).mean()
                gpe = (pred_global[b, :, :3, 3] - gt_global[b, :, :3, 3]).norm(dim=-1).mean()
                drift = (pred_global[b, -1, :3, 3] - gt_global[b, -1, :3, 3]).norm()
                scan_metrics.append({
                    "lpe_mm": float(lpe.item()),
                    "gpe_mm": float(gpe.item()),
                    "drift_last_mm": float(drift.item()),
                    "num_frames": int(T),
                })

        if not scan_metrics:
            return {"mean_lpe_mm": 0.0, "mean_gpe_mm": 0.0, "mean_drift_last_mm": 0.0}

        agg: dict[str, float] = {}
        for key in ("lpe_mm", "gpe_mm", "drift_last_mm"):
            vals = [s[key] for s in scan_metrics if key in s]
            if vals:
                agg[f"mean_{key}"] = sum(vals) / len(vals)
        agg["num_scans"] = float(len(scan_metrics))
        return agg
