"""Dual-path Transformer trainer (Dense Δ=1 + Sparse Δ=k).

Self-contained trainer for :class:`DualPoseModel`.  No multi-interval
auxiliary losses — each branch has dedicated supervision.

Training stability features:
- **AdamW** with decoupled weight decay (prevents unbounded weight growth).
- **Linear warm-up + cosine annealing** LR schedule.
- **Gradient accumulation** to simulate larger effective batch sizes.
- **Gradient clipping** (max-norm) before optimizer step.
- **EMA (Exponential Moving Average)** model for smoother evaluation.

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
from trainers.hooks.base_hook import Hook


def _cfg_get(cfg, path, default=None):
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


# ─── EMA helper ──────────────────────────────────────────────────────────────

class _EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy that is updated as:
      shadow = decay * shadow + (1 - decay) * current
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Apply EMA weights; return original weights for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ─── LR schedule: linear warm-up + cosine decay ─────────────────────────────

def _warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int,
                       base_lr: float, min_lr: float = 1e-6) -> float:
    """Compute LR at given step."""
    if step < warmup_steps:
        # Linear warm-up from min_lr to base_lr
        return min_lr + (base_lr - min_lr) * step / max(1, warmup_steps)
    # Cosine annealing from base_lr to min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class DualTrainer:
    """Trainer for the dual-path (Dense + Sparse) pose model.

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

        # ── Model hyper-parameters ───────────────────────────────────
        self.rotation_rep = str(_cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.k_stride = int(_cfg_get(cfg, "model.k_stride", 8))
        self.model = self._build_model().to(self.device)

        # ── Optimizer: AdamW with decoupled weight decay ─────────────
        lr = float(_cfg_get(cfg, "optimizer.lr_rec", _cfg_get(cfg, "optimizer.lr", 1e-4)))
        weight_decay = float(_cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(_cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.base_lr = lr
        self.min_lr = float(_cfg_get(cfg, "optimizer.min_lr", 1e-6))

        # Separate param groups: encoder (lower LR) vs. transformer+head
        encoder_params = list(self.model.encoder.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith("encoder.") and p.requires_grad]
        encoder_lr_mult = float(_cfg_get(cfg, "optimizer.encoder_lr_mult", 0.1))
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"},
            {"params": other_params, "lr": lr, "name": "tf_heads"},
        ], weight_decay=weight_decay, betas=betas)

        # ── LR schedule ─────────────────────────────────────────────
        self.warmup_steps = int(_cfg_get(cfg, "optimizer.warmup_steps", 200))

        # ── Gradient accumulation ────────────────────────────────────
        self.grad_accum_steps = int(_cfg_get(cfg, "trainer.grad_accum_steps", 4))

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(_cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(_cfg_get(cfg, "loss.trans_weight", 1.0))
        self.dense_weight = float(_cfg_get(cfg, "loss.dense_weight", 1.0))
        self.sparse_weight = float(_cfg_get(cfg, "loss.sparse_weight", 1.0))

        # ── Training schedule ────────────────────────────────────────
        self.num_epochs = int(_cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = _cfg_get(cfg, "train.max_steps") or _cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(_cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(_cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(_cfg_get(cfg, "paths.output_dir", "logs"))

        # ── Gradient clipping ────────────────────────────────────────
        self.max_grad_norm = float(_cfg_get(cfg, "trainer.max_grad_norm", 1.0))

        # ── EMA ──────────────────────────────────────────────────────
        ema_decay = float(_cfg_get(cfg, "trainer.ema_decay", 0.999))
        self.ema = _EMA(self.model, decay=ema_decay) if ema_decay > 0 else None

        # ── Fusion config (used at eval time) ────────────────────────
        self.fusion_mode = str(_cfg_get(cfg, "fusion.mode", "anchor_interp"))
        self.fusion_smooth = bool(_cfg_get(cfg, "fusion.anchor_smooth", True))

        # ── Calibration (for VizHook drift curves + future DDF eval) ─
        self.tform_calib: torch.Tensor | None = self._load_calib()

    # ── Calibration loader ───────────────────────────────────────────

    def _load_calib(self) -> "torch.Tensor | None":
        """Load tform_calib from config (mirrors LongSeqTrainer._load_calib)."""
        import warnings
        calib_file = _cfg_get(self.cfg, "dataset.calib_file")
        if not calib_file:
            return None
        try:
            from trainers.utils.calibration import load_calibration  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            resample_factor = float(_cfg_get(self.cfg, "dataset.resample_factor", 1.0))
            _, _, tform_calib = load_calibration(calib_file, resample_factor, device=self.device)
            if not isinstance(tform_calib, torch.Tensor):
                tform_calib = torch.as_tensor(np.array(tform_calib), dtype=torch.float32)
            return tform_calib.float().to(self.device)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[DualTrainer] Could not load calibration from {calib_file!r}: {exc}"
            )
            return None

    # ── Model construction ───────────────────────────────────────────

    def _build_model(self) -> DualPoseModel:
        backbone = str(_cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(_cfg_get(self.cfg, "model.transformer.d_model",
                                 _cfg_get(self.cfg, "model.dense.transformer.d_model", 256)))
        pretrained = bool(_cfg_get(self.cfg, "model.encoder.pretrained", False))

        # Dense branch config
        dense_n_heads = int(_cfg_get(self.cfg, "model.dense.transformer.n_heads", 4))
        dense_n_layers = int(_cfg_get(self.cfg, "model.dense.transformer.n_layers", 4))
        dense_dim_ff = int(_cfg_get(self.cfg, "model.dense.transformer.dim_feedforward", 1024))
        dense_window = int(_cfg_get(self.cfg, "model.dense.transformer.window_size", 64))

        # Sparse branch config
        sparse_n_heads = int(_cfg_get(self.cfg, "model.sparse.transformer.n_heads", 4))
        sparse_n_layers = int(_cfg_get(self.cfg, "model.sparse.transformer.n_layers", 4))
        sparse_dim_ff = int(_cfg_get(self.cfg, "model.sparse.transformer.dim_feedforward", 1024))
        sparse_window = int(_cfg_get(self.cfg, "model.sparse.transformer.window_size", 256))

        dropout = float(_cfg_get(self.cfg, "model.transformer.dropout",
                                 _cfg_get(self.cfg, "model.dense.transformer.dropout", 0.1)))

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

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(self, **kwargs)

    # ── Checkpoint save / load (full resume) ─────────────────────────

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        """Save model, optimizer, EMA, epoch, and global_step for resume."""
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
        """Restore model, optimizer, EMA, epoch, global_step from a full checkpoint."""
        state = torch.load(path, map_location=self.device)
        # Model weights — try known wrapper keys, then assume flat
        model_state = None
        for key in ("model", "state_dict", "network"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                model_state = state[key]
                break
        if model_state is None:
            model_state = state  # assume flat state_dict
        self.model.load_state_dict(model_state)
        # Optimizer
        if "optimizer" in state and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception as exc:
                print(f"[DualTF] optimizer state load failed ({exc}), starting fresh optimizer")
        # EMA
        if "ema_shadow" in state and self.ema is not None:
            self.ema.shadow = state["ema_shadow"]
        # Epoch / step
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

    # ── LR update ────────────────────────────────────────────────────

    def _update_lr(self) -> float:
        """Update LR based on global_step with warm-up + cosine schedule."""
        total_steps = self._estimate_total_steps()
        new_lr = _warmup_cosine_lr(
            self.global_step, self.warmup_steps, total_steps,
            self.base_lr, self.min_lr,
        )
        for pg in self.optimizer.param_groups:
            if pg.get("name") == "encoder":
                encoder_mult = float(_cfg_get(self.cfg, "optimizer.encoder_lr_mult", 0.1))
                pg["lr"] = new_lr * encoder_mult
            else:
                pg["lr"] = new_lr
        return new_lr

    def _estimate_total_steps(self) -> int:
        """Estimate total optimizer steps for the cosine schedule."""
        if self.max_steps is not None:
            return self.max_steps
        # Rough estimate from loader length
        try:
            loader_len = len(self.train_loader)
        except TypeError:
            loader_len = 100  # iterable dataset fallback
        steps_per_epoch = max(1, loader_len // self.grad_accum_steps)
        return steps_per_epoch * self.num_epochs

    # ── Train loop ───────────────────────────────────────────────────

    def train(self):
        if self.train_loader is None:
            raise ValueError("No train_loader provided")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")

        start_epoch = self.epoch  # supports resume from checkpoint
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            n_steps = 0          # micro-steps (forward passes)
            n_optim_steps = 0    # optimizer steps (after accumulation)

            self.call_hooks("before_epoch")

            ds = getattr(self.train_loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            self.optimizer.zero_grad()
            accum_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break
                n_steps += 1
                self.call_hooks("before_step")

                loss, metrics = self._run_step(batch)
                # Scale loss by accumulation steps for correct averaging
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()
                accum_loss += float(loss.item())

                # Optimizer step every grad_accum_steps micro-steps
                if n_steps % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    n_optim_steps += 1

                    # Update LR schedule
                    current_lr = self._update_lr()

                    # Update EMA
                    if self.ema is not None:
                        self.ema.update(self.model)

                    avg_accum = accum_loss / self.grad_accum_steps
                    epoch_loss += avg_accum
                    accum_loss = 0.0

                    if n_optim_steps % self.log_interval == 0 or n_optim_steps == 1:
                        print(
                            f"[DualTF epoch {epoch}  step {n_optim_steps}]  "
                            f"loss={avg_accum:.4f}  "
                            f"dense={metrics.get('dense_loss', 0):.4f}  "
                            f"sparse={metrics.get('sparse_loss', 0):.4f}  "
                            f"drift_mm={metrics.get('drift_mm_last', 0):.2f}  "
                            f"lr={current_lr:.2e}  "
                            f"k={self.k_stride}"
                        )

                    self.call_hooks(
                        "after_step",
                        log_buffer={
                            "mode": "train",
                            "epoch": epoch + 1,
                            "iter": n_optim_steps,
                            "global_step": self.global_step,
                            "loss": avg_accum,
                            "lr": current_lr,
                            **metrics,
                        },
                    )

            # Handle any remaining accumulated gradients
            if n_steps % self.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                n_optim_steps += 1
                if self.ema is not None:
                    self.ema.update(self.model)
                epoch_loss += accum_loss / (n_steps % self.grad_accum_steps)

            avg_loss = epoch_loss / max(1, n_optim_steps)
            print(f"[DualTF epoch {epoch}] avg_loss={avg_loss:.4f}  optim_steps={n_optim_steps}")
            # Read current LR from the tf_heads param group (index 1 = non-encoder).
            _epoch_lr = float(
                next((pg["lr"] for pg in self.optimizer.param_groups
                      if pg.get("name") == "tf_heads"),
                     self.optimizer.param_groups[-1]["lr"])
            )
            self.call_hooks(
                "after_epoch",
                log_buffer={
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "lr": _epoch_lr,
                },
            )

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                self.call_hooks("before_val")
                val_metrics = self.evaluate(self.val_loader)
                self.last_eval_metrics = val_metrics  # VizHook reads this
                print(f"[DualTF val epoch {epoch}] {val_metrics}")
                self.call_hooks(
                    "after_val",
                    log_buffer={
                        "epoch": epoch + 1,
                        "val_loss": val_metrics.get("mean_gpe_mm", 0.0),
                        **val_metrics,
                    },
                )

        self.call_hooks("after_train")
        self.call_hooks("after_run")

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
            _EMA.restore(self.model, ema_backup)

        return agg
