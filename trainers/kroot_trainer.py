"""K-root Short/Long independent trainer.

Trains a single LongSeqPoseModel (no auxΔ, no consistency) for either:
* **short** mode — contiguous frames, Δ=1 local supervision
* **long** mode  — sparse frames at stride *s*, Δ=1 (in sparse domain) supervision

The two modes share the same model architecture and loss;
only the dataset and stride differ.  They are trained with
separate configs and produce separate checkpoints.

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
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import compose_global_from_local, local_from_global
from trainers.hooks.base_hook import Hook


def _cfg_get(cfg, path, default=None):
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


class KRootTrainer:
    """Self-contained trainer for one branch (short or long) of the K-root scheme.

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
        self.cfg = cfg
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.hooks: list[Hook] = []
        self.epoch = 0
        self.global_step = 0

        # ── Branch mode ──────────────────────────────────────────────
        self.branch = str(_cfg_get(cfg, "kroot.branch", "short")).lower()
        assert self.branch in ("short", "long"), f"Unknown branch: {self.branch}"
        self.k = int(_cfg_get(cfg, "kroot.k", 64))
        self.s = int(_cfg_get(cfg, "kroot.s", 0))
        if self.s <= 0:
            self.s = int(round(math.sqrt(self.k)))

        # ── Model ────────────────────────────────────────────────────
        self.rotation_rep = str(_cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ── Optimizer ────────────────────────────────────────────────
        lr = float(_cfg_get(cfg, "optimizer.lr_rec", _cfg_get(cfg, "optimizer.lr", 1e-4)))
        weight_decay = float(_cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(_cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(_cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(_cfg_get(cfg, "loss.trans_weight", 1.0))

        # ── Training schedule ────────────────────────────────────────
        self.num_epochs = int(_cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = _cfg_get(cfg, "train.max_steps") or _cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(_cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(_cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(_cfg_get(cfg, "paths.output_dir", "logs"))
        self.grad_accum = max(1, int(_cfg_get(cfg, "trainer.grad_accum", 1) or 1))
        self.max_grad_norm = float(_cfg_get(cfg, "trainer.max_grad_norm", 1.0))

        # ── Calibration (for eval metrics) ───────────────────────────
        self.tform_calib: torch.Tensor | None = self._load_calib()

    # ── calibration ─────────────────────────────────────────────────

    def _load_calib(self) -> torch.Tensor | None:
        calib_file = _cfg_get(self.cfg, "dataset.calib_file")
        if not calib_file:
            return None
        try:
            from trainers.utils.calibration import load_calibration
            resample_factor = float(_cfg_get(self.cfg, "dataset.resample_factor", 1.0))
            _, _, tform_calib = load_calibration(calib_file, resample_factor, device=self.device)
            if not isinstance(tform_calib, torch.Tensor):
                import numpy as np
                tform_calib = torch.as_tensor(tform_calib, dtype=torch.float32)
            return tform_calib.float().to(self.device)
        except Exception as exc:
            warnings.warn(f"[KRootTrainer] Could not load calibration: {exc}")
            return None

    # ── model construction ───────────────────────────────────────────

    def _build_model(self) -> LongSeqPoseModel:
        backbone = str(_cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(_cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(_cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(_cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(_cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(_cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(_cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        pretrained = bool(_cfg_get(self.cfg, "model.encoder.pretrained", False))

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

    # ── hooks ────────────────────────────────────────────────────────

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(self, **kwargs)

    # ── single training step ─────────────────────────────────────────

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + chordal-rot + smooth-L1-trans loss."""
        import torch.nn.functional as F

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

    # ── train loop ───────────────────────────────────────────────────

    def train(self):
        if self.train_loader is None:
            raise ValueError("No train_loader provided")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")
        tag = f"KRoot-{self.branch}"

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            n_steps = 0

            self.call_hooks("before_epoch")
            ds = getattr(self.train_loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            self.optimizer.zero_grad()

            for step, batch in enumerate(self.train_loader):
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break
                self.global_step += 1
                n_steps += 1
                self.call_hooks("before_step")

                loss, metrics = self._run_step(batch)
                (loss / self.grad_accum).backward()

                if n_steps % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += float(loss.item())

                if n_steps % self.log_interval == 0 or n_steps == 1:
                    print(
                        f"[{tag} epoch {epoch}  step {n_steps}]  "
                        f"loss={loss.item():.4f}  "
                        f"rot={metrics['rot_loss']:.4f}  "
                        f"trans={metrics['trans_loss']:.4f}  "
                        f"drift={metrics['drift_mm_last']:.2f}mm"
                    )

                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                self.call_hooks(
                    "after_step",
                    log_buffer={
                        "mode": "train",
                        "epoch": epoch + 1,
                        "iter": step + 1,
                        "global_step": self.global_step,
                        "loss": float(loss.item()),
                        "lr": lr,
                        **metrics,
                    },
                )

            avg_loss = epoch_loss / max(1, n_steps)
            print(f"[{tag} epoch {epoch}] avg_loss={avg_loss:.4f}")
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.call_hooks("after_epoch", log_buffer={"epoch": epoch + 1, "train_loss": avg_loss, "lr": lr})

            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                self.last_eval_metrics = val_metrics
                print(f"[{tag} val epoch {epoch}] {val_metrics}")
                self.call_hooks(
                    "after_val",
                    log_buffer={"epoch": epoch + 1, "val_loss": avg_loss, **val_metrics},
                )

        self.call_hooks("after_train")
        self.call_hooks("after_run")

    # ── evaluation ───────────────────────────────────────────────────

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
