"""Dual-path Transformer trainer (Dense Δ=1 + Sparse Δ=k).

Self-contained trainer for :class:`DualPoseModel`.  No multi-interval
auxiliary losses — each branch has dedicated supervision.

Now inherits from :class:`BaseTrainer` for shared boilerplate (hooks,
checkpoint, gradient accumulation, LR schedule, train loop skeleton).

Training stability features (all via BaseTrainer):
- **AdamW** with decoupled weight decay.
- **Linear warm-up + cosine annealing** LR schedule.
- **Gradient accumulation** to simulate larger effective batch sizes.
- **Gradient clipping** (max-norm) before optimizer step.
- **EMA** model for smoother evaluation.

Lifecycle::

    trainer = DualTrainer(cfg, device=..., train_loader=..., val_loader=...)
    trainer.train()
    trainer.evaluate(loader)

The eval path calls :func:`eval.dual_fusion.fuse_dual_predictions` to
produce fused global transforms, then computes TUS-REC metrics.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.dual_pose_model import DualPoseModel
from models.losses.dual_loss import dual_loss
from metrics.compose import compose_global_from_local, local_from_global
from eval.dual_fusion import fuse_dual_predictions
from trainers.base_trainer import BaseTrainer
from trainers.common import cfg_get, EMA, warmup_cosine_lr, load_tform_calib


class DualTrainer(BaseTrainer):
    """Trainer for the dual-path (Dense + Sparse) pose model.

    Inherits generic train loop, hooks, checkpoint from :class:`BaseTrainer`.
    Adds: DualPoseModel construction, dual fusion evaluation, EMA.

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

        # ── Model hyper-parameters ───────────────────────────────────
        self.rotation_rep = str(cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.k_stride = int(cfg_get(cfg, "model.k_stride", 8))
        self.model = self._build_model().to(self.device)

        # ── Optimizer: AdamW with decoupled weight decay ─────────────
        lr = float(cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 1e-4)))
        weight_decay = float(cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.base_lr = lr
        self.min_lr = float(cfg_get(cfg, "optimizer.min_lr", 1e-6))

        # Separate param groups: encoder (lower LR) vs. transformer+head
        encoder_params = list(self.model.encoder.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith("encoder.") and p.requires_grad]
        encoder_lr_mult = float(cfg_get(cfg, "optimizer.encoder_lr_mult", 0.1))
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"},
            {"params": other_params, "lr": lr, "name": "tf_heads"},
        ], weight_decay=weight_decay, betas=betas)

        # ── LR schedule ─────────────────────────────────────────────
        self.warmup_steps = int(cfg_get(cfg, "optimizer.warmup_steps", 200))

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(cfg_get(cfg, "loss.trans_weight", 1.0))
        self.dense_weight = float(cfg_get(cfg, "loss.dense_weight", 1.0))
        self.sparse_weight = float(cfg_get(cfg, "loss.sparse_weight", 1.0))

        # ── EMA ──────────────────────────────────────────────────────
        ema_decay = float(cfg_get(cfg, "trainer.ema_decay", 0.999))
        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None

        # ── Fusion config (used at eval time) ────────────────────────
        self.fusion_mode = str(cfg_get(cfg, "fusion.mode", "anchor_interp"))
        self.fusion_smooth = bool(cfg_get(cfg, "fusion.anchor_smooth", True))

        # ── Calibration ──────────────────────────────────────────────
        self.tform_calib: torch.Tensor | None = load_tform_calib(
            cfg, device=self.device, warn_prefix="DualTrainer"
        )

    # ── Model construction ───────────────────────────────────────────

    def _build_model(self) -> DualPoseModel:
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(cfg_get(self.cfg, "model.transformer.d_model",
                                 cfg_get(self.cfg, "model.dense.transformer.d_model", 256)))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))

        # Dense branch config
        dense_n_heads = int(cfg_get(self.cfg, "model.dense.transformer.n_heads", 4))
        dense_n_layers = int(cfg_get(self.cfg, "model.dense.transformer.n_layers", 4))
        dense_dim_ff = int(cfg_get(self.cfg, "model.dense.transformer.dim_feedforward", 1024))
        dense_window = int(cfg_get(self.cfg, "model.dense.transformer.window_size", 64))

        # Sparse branch config
        sparse_n_heads = int(cfg_get(self.cfg, "model.sparse.transformer.n_heads", 4))
        sparse_n_layers = int(cfg_get(self.cfg, "model.sparse.transformer.n_layers", 4))
        sparse_dim_ff = int(cfg_get(self.cfg, "model.sparse.transformer.dim_feedforward", 1024))
        sparse_window = int(cfg_get(self.cfg, "model.sparse.transformer.window_size", 256))

        dropout = float(cfg_get(self.cfg, "model.transformer.dropout",
                                 cfg_get(self.cfg, "model.dense.transformer.dropout", 0.1)))

        return DualPoseModel(
            backbone=backbone,
            in_channels=1,
            token_dim=token_dim,
            k_stride=self.k_stride,
            dense_n_heads=dense_n_heads,
            dense_n_layers=dense_n_layers,
            dense_dim_ff=dense_dim_ff,
            dense_window=dense_window,
            sparse_n_heads=sparse_n_heads,
            sparse_n_layers=sparse_n_layers,
            sparse_dim_ff=sparse_dim_ff,
            sparse_window=sparse_window,
            dropout=dropout,
            rotation_rep=self.rotation_rep,
            pretrained_backbone=pretrained,
        )

    # ── Hooks ────────────────────────────────────────────────────────
    # register_hook / call_hooks inherited from BaseTrainer

    # ── Checkpoint save / load (with EMA support) ────────────────────

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        import os as _os
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            payload["ema_shadow"] = self.ema.shadow
        torch.save(payload, path)
        print(f"[DualTF] checkpoint saved → {path}  (tag={tag})")

    def load_full_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        model_state = None
        for key in ("model", "state_dict", "network"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                model_state = state[key]
                break
        if model_state is None:
            model_state = state
        self.model.load_state_dict(model_state)
        if "optimizer" in state and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:
                print(f"[DualTF] optimizer state load failed ({exc}), starting fresh optimizer")
        if "ema_shadow" in state and self.ema is not None:
            self.ema.shadow = state["ema_shadow"]
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        print(f"[DualTF] checkpoint loaded ← {path}  (epoch={self.epoch}, step={self.global_step})")

    # ── Single training step ─────────────────────────────────────────

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss on a single sequence batch."""
        frames = batch["frames"].to(self.device)          # (B, T, H, W)
        gt_global = batch["gt_global_T"].to(self.device)   # (B, T, 4, 4)

        if frames.max() > 1.1:
            frames = frames / 255.0

        out = self.model(frames)
        pred_local_T = out["pred_local_T"]       # (B, T, 4, 4)
        pred_sparse_T = out["pred_sparse_T"]     # (B, M, 4, 4)
        anchor_indices = out["anchor_indices"]   # (M,)

        loss, breakdown = dual_loss(
            pred_local_T=pred_local_T,
            pred_sparse_T=pred_sparse_T,
            gt_global_T=gt_global,
            anchor_indices=anchor_indices,
            rot_weight=self.rot_weight,
            trans_weight=self.trans_weight,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )

        # Diagnostic: drift at last frame (dense-only accumulation)
        with torch.no_grad():
            pred_global = compose_global_from_local(pred_local_T)
            drift_t = (
                pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]
            ).norm(dim=-1).mean()
            breakdown["drift_mm_last"] = float(drift_t.item())
            breakdown["k_stride"] = self.k_stride
            breakdown["fusion_mode"] = self.fusion_mode

        return loss, breakdown

    # ── Train loop customisations ────────────────────────────────────
    # The generic loop is inherited from BaseTrainer.train().
    # _update_lr / _estimate_total_steps are inherited from BaseTrainer.

    def _after_optim_step(self) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        return (
            f"[DualTF epoch {epoch}  step {n_optim_steps}]  "
            f"loss={avg_accum:.4f}  "
            f"dense={metrics.get('dense_loss', 0):.4f}  "
            f"sparse={metrics.get('sparse_loss', 0):.4f}  "
            f"drift_mm={metrics.get('drift_mm_last', 0):.2f}  "
            f"lr={current_lr:.2e}  "
            f"k={self.k_stride}"
        )

    # ── Evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Evaluate dual-path model with fusion on a full-scan loader.

        Uses EMA weights if available for smoother evaluation.
        Produces fused global transforms via anchor-correction or pose-graph
        optimisation, then computes GPE/LPE metrics.
        """
        # Apply EMA weights for evaluation
        ema_backup = None
        if self.ema is not None:
            ema_backup = self.ema.apply(self.model)

        self.model.eval()
        scan_metrics: list[dict[str, float]] = []
        scan_globals: dict[str, dict] = {}  # VizHook needs per-scan pred/gt
        scan_counter = 0

        for batch in loader:
            frames = batch["frames"].to(self.device)
            gt_global = batch["gt_global_T"].to(self.device)

            if frames.max() > 1.1:
                frames = frames / 255.0

            out = self.model(frames)
            pred_local_T = out["pred_local_T"]
            pred_sparse_T = out["pred_sparse_T"]
            anchor_indices = out["anchor_indices"]

            B, T = pred_local_T.shape[:2]
            for b in range(B):
                fused_global = fuse_dual_predictions(
                    dense_local_T=pred_local_T[b],      # (T, 4, 4)
                    sparse_local_T=pred_sparse_T[b],    # (M, 4, 4)
                    anchor_indices=anchor_indices,       # (M,)
                    mode=self.fusion_mode,
                    smooth=self.fusion_smooth,
                )  # (T, 4, 4)

                gt_g = gt_global[b]  # (T, 4, 4)

                # Fused local from fused global
                fused_local = local_from_global(fused_global)
                gt_local = local_from_global(gt_g)

                # LPE: mean local translation error (frames 1..T-1)
                lpe = (
                    fused_local[1:, :3, 3] - gt_local[1:, :3, 3]
                ).norm(dim=-1).mean()

                # GPE: mean global translation error
                gpe = (
                    fused_global[:, :3, 3] - gt_g[:, :3, 3]
                ).norm(dim=-1).mean()

                # Drift at last frame
                drift = (
                    fused_global[-1, :3, 3] - gt_g[-1, :3, 3]
                ).norm()

                # Also compute dense-only metrics for comparison
                dense_global = compose_global_from_local(pred_local_T[b].unsqueeze(0)).squeeze(0)
                gpe_dense = (
                    dense_global[:, :3, 3] - gt_g[:, :3, 3]
                ).norm(dim=-1).mean()

                scan_metrics.append({
                    "lpe_mm": float(lpe.item()),
                    "gpe_mm": float(gpe.item()),
                    "drift_last_mm": float(drift.item()),
                    "gpe_dense_only_mm": float(gpe_dense.item()),
                    "num_frames": int(T),
                    "k_stride": self.k_stride,
                    "fusion_mode": self.fusion_mode,
                })

                # Store for VizHook: per-scan pred/gt global transforms
                meta = batch.get("meta")
                if isinstance(meta, dict):
                    sid = meta.get("scan_id") or meta.get("scan_name") or f"scan_{scan_counter}"
                elif isinstance(meta, list) and b < len(meta) and isinstance(meta[b], dict):
                    sid = meta[b].get("scan_id") or meta[b].get("scan_name") or f"scan_{scan_counter}"
                else:
                    sid = f"scan_{scan_counter}"
                if isinstance(sid, (list, tuple)):
                    sid = "/".join(str(s) for s in sid)
                sid = str(sid)
                scan_globals[sid] = {
                    "pred": fused_global.detach().cpu(),
                    "gt": gt_g.detach().cpu(),
                }
                scan_counter += 1

        if not scan_metrics:
            return {"lpe_mm": 0.0, "gpe_mm": 0.0, "drift_last_mm": 0.0}

        agg: dict[str, float] = {}
        for key in ("lpe_mm", "gpe_mm", "drift_last_mm", "gpe_dense_only_mm"):
            vals = [s[key] for s in scan_metrics if key in s]
            if vals:
                agg[f"mean_{key}"] = sum(vals) / len(vals)
        agg["num_scans"] = float(len(scan_metrics))
        agg["k_stride"] = float(self.k_stride)
        agg["fusion_mode"] = self.fusion_mode  # type: ignore[assignment]

        # Expose per-scan global transforms for VizHook
        agg["scan_globals"] = scan_globals  # type: ignore[assignment]

        # Restore original (non-EMA) weights
        if ema_backup is not None:
            EMA.restore(self.model, ema_backup)

        return agg
