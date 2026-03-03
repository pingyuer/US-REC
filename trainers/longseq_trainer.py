"""Long-sequence trainer for Early-CNN + Temporal Transformer pose models.

This trainer works with :class:`LongSeqPoseModel` and the ``ScanWindowDataset``
(or ``SyntheticScanWindowDataset`` for smoke tests).  It keeps the baseline
pairwise path completely untouched; the original ``rec_trainer.py`` is not
modified.

Usage (from config):
    trainer.name: trainers.longseq_trainer.LongSeqTrainer
    model.type: longseq_transformer
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.longseq_loss import longseq_loss
from metrics.compose import compose_global_from_local, local_from_global
from trainers.hooks.base_hook import Hook


def _cfg_get(cfg, path, default=None):
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


class LongSeqTrainer:
    """Self-contained trainer for the long-sequence pose model.

    Lifecycle::

        trainer = LongSeqTrainer(cfg, device=...)
        trainer.train()           # or
        trainer.evaluate(loader)  # standalone eval
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

        # ---- model hyper-parameters -----------------------------------------
        self.rotation_rep = str(_cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ---- optimizer -------------------------------------------------------
        lr = float(_cfg_get(cfg, "optimizer.lr_rec", _cfg_get(cfg, "optimizer.lr", 1e-4)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ---- loss config -----------------------------------------------------
        self.rot_weight = float(_cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(_cfg_get(cfg, "loss.trans_weight", 1.0))
        self.aux_intervals = list(_cfg_get(cfg, "loss.aux_intervals", [2, 4, 8, 16]))
        self.aux_base_weight = float(_cfg_get(cfg, "loss.aux_weight", 0.5))
        self.aux_decay = float(_cfg_get(cfg, "loss.aux_decay", 0.5))
        self.consistency_weight = float(_cfg_get(cfg, "loss.consistency_weight", 0.1))
        self.consistency_delta = int(_cfg_get(cfg, "loss.consistency_delta", 2))

        # ---- training schedule -----------------------------------------------
        self.num_epochs = int(_cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = _cfg_get(cfg, "train.max_steps") or _cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(_cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(_cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(_cfg_get(cfg, "paths.output_dir", "logs"))

    # ── model construction ───────────────────────────────────────────

    def _build_model(self) -> LongSeqPoseModel:
        backbone = str(_cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(_cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(_cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(_cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(_cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(_cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(_cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        aux_intervals = list(_cfg_get(self.cfg, "loss.aux_intervals", [2, 4, 8, 16]))
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
            aux_intervals=aux_intervals,
            pretrained_backbone=pretrained,
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
        """Forward + loss on a single sequence batch.

        Parameters
        ----------
        batch : dict with ``frames`` (B,T,H,W) and ``gt_global_T`` (B,T,4,4)

        Returns
        -------
        loss : scalar tensor (differentiable)
        metrics : dict of detached float metrics
        """
        frames = batch["frames"].to(self.device)          # (B, T, H, W)
        gt_global = batch["gt_global_T"].to(self.device)   # (B, T, 4, 4)

        if frames.max() > 1.1:
            frames = frames / 255.0

        out = self.model(frames)
        pred_local_T = out["pred_local_T"]       # (B, T, 4, 4)
        pred_aux_T = out["pred_aux_T"]           # dict[Δ → (B, T, 4, 4)]

        loss, breakdown = longseq_loss(
            pred_local_T=pred_local_T,
            pred_aux_T=pred_aux_T,
            gt_global_T=gt_global,
            intervals=self.aux_intervals,
            rot_weight=self.rot_weight,
            trans_weight=self.trans_weight,
            aux_base_weight=self.aux_base_weight,
            aux_decay=self.aux_decay,
            consistency_weight=self.consistency_weight,
            consistency_delta=self.consistency_delta,
        )

        # Compute per-frame global pose for diagnostics
        with torch.no_grad():
            pred_global = compose_global_from_local(pred_local_T)  # (B, T, 4, 4)
            # Translation drift at last frame
            drift_t = (
                pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]
            ).norm(dim=-1).mean()
            breakdown["drift_mm_last"] = float(drift_t.item())

        return loss, breakdown

    # ── train loop ───────────────────────────────────────────────────

    def train(self):
        if self.train_loader is None:
            raise ValueError("No train_loader provided")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            n_steps = 0

            self.call_hooks("before_epoch")

            # Update dataset epoch for proper shuffle randomisation
            ds = getattr(self.train_loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            for step, batch in enumerate(self.train_loader):
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break
                self.global_step += 1
                n_steps += 1
                self.call_hooks("before_step")

                self.optimizer.zero_grad()
                loss, metrics = self._run_step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += float(loss.item())

                if n_steps % self.log_interval == 0 or n_steps == 1:
                    print(
                        f"[LongSeq epoch {epoch}  step {n_steps}]  "
                        f"loss={loss.item():.4f}  local_rot={metrics.get('loss_local_rot', 0):.4f}  "
                        f"local_trans={metrics.get('loss_local_trans', 0):.4f}  "
                        f"drift_mm={metrics.get('drift_mm_last', 0):.2f}"
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
            print(f"[LongSeq epoch {epoch}] avg_loss={avg_loss:.4f}")
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.call_hooks(
                "after_epoch",
                log_buffer={
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "lr": lr,
                },
            )

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f"[LongSeq val epoch {epoch}] {val_metrics}")
                # Store for VizHook; fire after_val so MLflowHook can save best checkpoint.
                self.last_eval_metrics = val_metrics
                self.call_hooks(
                    "after_val",
                    log_buffer={
                        "epoch": epoch + 1,
                        # Use mean_gpe_mm as primary metric; MLflowHook maps
                        # missing final_score to val_loss (mode=min → lower=better).
                        "val_loss": float(val_metrics.get("mean_gpe_mm",
                            val_metrics.get("mean_drift_last_mm", 0.0))),
                        **{k: float(v) for k, v in val_metrics.items()
                           if isinstance(v, (int, float))},
                    },
                )

        self.call_hooks("after_train")
        self.call_hooks("after_run")

    # ── evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Evaluate long-sequence model on a full-scan loader.

        Returns
        -------
        dict with mean drift, LPE, GPE metrics over all scans.
        """
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
                # LPE: mean local translation error (frames 1..T-1)
                local_terr = (
                    pred_local_T[b, 1:, :3, 3] - gt_local[b, 1:, :3, 3]
                ).norm(dim=-1).mean()
                # GPE: mean global translation error
                global_terr = (
                    pred_global[b, :, :3, 3] - gt_global[b, :, :3, 3]
                ).norm(dim=-1).mean()
                # Drift at last frame
                drift = (
                    pred_global[b, -1, :3, 3] - gt_global[b, -1, :3, 3]
                ).norm()

                scan_metrics.append({
                    "lpe_mm": float(local_terr.item()),
                    "gpe_mm": float(global_terr.item()),
                    "drift_last_mm": float(drift.item()),
                    "num_frames": int(T),
                })

        if not scan_metrics:
            return {"lpe_mm": 0.0, "gpe_mm": 0.0, "drift_last_mm": 0.0}

        agg: dict[str, float] = {}
        for key in scan_metrics[0]:
            if key == "num_frames":
                continue
            agg[f"mean_{key}"] = sum(s[key] for s in scan_metrics) / len(scan_metrics)
        agg["num_scans"] = float(len(scan_metrics))
        return agg
