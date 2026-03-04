"""K-root Dual Trainer — joint training of Short + Long branches with stitch eval.

Trains two **independent** ``LongSeqPoseModel`` instances simultaneously:
- **Short model**: contiguous k-frame windows, predicts Δ=1 local transforms.
- **Long model**: sparse k-token windows at stride s, predicts Δ=1 (sparse-domain) locals.

Each training step draws one batch from the short loader and one batch from
the long loader, computing separate losses for each model.  Both models use
AdamW with linear warm-up + cosine annealing LR and optional EMA.

Evaluation runs the full stitch pipeline from ``eval.kroot_stitch``:
long anchors provide the global skeleton, short provides dense refinement,
fused via SE(3) log-space endpoint correction.

Usage::

    trainer = KRootDualTrainer(
        cfg, device=...,
        short_train_loader=..., long_train_loader=..., val_loader=...,
    )
    trainer.train()
    trainer.evaluate(loader)

Single config → single command::

    python main_rec.py --config configs/demo_rec24_ete_kroot_dual.yml
"""

from __future__ import annotations

import copy
import math
import os
import time
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import compose_global_from_local, local_from_global
from eval.kroot_stitch import (
    stitch_long_base_short_refine,
    compute_stitch_metrics,
    export_debug_csv,
)
from trainers.hooks.base_hook import Hook


def _cfg_get(cfg, path, default=None):
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


# ─── EMA helper ──────────────────────────────────────────────────────────────

class _EMA:
    """Exponential Moving Average of model parameters."""

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


# ─── LR schedule ────────────────────────────────────────────────────────────

def _warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int,
                       base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ==========================================================================
# KRootDualTrainer
# ==========================================================================

class KRootDualTrainer:
    """Joint trainer for both K-root branches (short + long).

    Parameters
    ----------
    cfg : OmegaConf config
    device : str | torch.device
    short_train_loader : DataLoader for ShortWindowDataset (train)
    long_train_loader : DataLoader for LongWindowDataset (train)
    val_loader : DataLoader for full-scan validation (ShortWindowDataset in val mode)
    """

    def __init__(
        self,
        cfg,
        *,
        device: str | torch.device = "cpu",
        short_train_loader=None,
        long_train_loader=None,
        val_loader=None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.short_train_loader = short_train_loader
        self.long_train_loader = long_train_loader
        self.val_loader = val_loader

        self.hooks: list[Hook] = []
        self.epoch = 0
        self.global_step = 0

        # ── K-root params ────────────────────────────────────────────
        self.k = int(_cfg_get(cfg, "kroot.k", 64))
        self.s = int(_cfg_get(cfg, "kroot.s", 0))
        if self.s <= 0:
            self.s = int(round(math.sqrt(self.k)))

        # ── Models ───────────────────────────────────────────────────
        self.rotation_rep = str(_cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.short_model = self._build_model("short").to(self.device)
        self.long_model = self._build_model("long").to(self.device)

        # ── Optimizers (separate for each model) ─────────────────────
        lr = float(_cfg_get(cfg, "optimizer.lr_rec", _cfg_get(cfg, "optimizer.lr", 5e-4)))
        weight_decay = float(_cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(_cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.base_lr = lr
        self.min_lr = float(_cfg_get(cfg, "optimizer.min_lr", 1e-6))

        encoder_lr_mult = float(_cfg_get(cfg, "optimizer.encoder_lr_mult", 0.1))

        def _make_optimizer(model: nn.Module) -> torch.optim.AdamW:
            encoder_params = [p for n, p in model.named_parameters()
                              if n.startswith("encoder.") and p.requires_grad]
            other_params = [p for n, p in model.named_parameters()
                           if not n.startswith("encoder.") and p.requires_grad]
            groups = []
            if encoder_params:
                groups.append({"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"})
            if other_params:
                groups.append({"params": other_params, "lr": lr, "name": "tf_heads"})
            return torch.optim.AdamW(groups, weight_decay=weight_decay, betas=betas)

        self.short_optimizer = _make_optimizer(self.short_model)
        self.long_optimizer = _make_optimizer(self.long_model)

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(_cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(_cfg_get(cfg, "loss.trans_weight", 1.0))
        self.short_loss_weight = float(_cfg_get(cfg, "loss.short_weight", 1.0))
        self.long_loss_weight = float(_cfg_get(cfg, "loss.long_weight", 1.0))

        # ── Training schedule ────────────────────────────────────────
        self.num_epochs = int(_cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = _cfg_get(cfg, "train.max_steps") or _cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(_cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(_cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(_cfg_get(cfg, "paths.output_dir", "logs"))

        # ── Gradient accumulation & clipping ─────────────────────────
        self.grad_accum = max(1, int(_cfg_get(cfg, "trainer.grad_accum", 4) or 4))
        self.max_grad_norm = float(_cfg_get(cfg, "trainer.max_grad_norm", 1.0))

        # ── Warm-up ──────────────────────────────────────────────────
        self.warmup_steps = int(_cfg_get(cfg, "optimizer.warmup_steps", 200))

        # ── EMA ──────────────────────────────────────────────────────
        ema_decay = float(_cfg_get(cfg, "trainer.ema_decay", 0.999))
        self.short_ema = _EMA(self.short_model, decay=ema_decay) if ema_decay > 0 else None
        self.long_ema = _EMA(self.long_model, decay=ema_decay) if ema_decay > 0 else None

        # ── Stitch config ────────────────────────────────────────────
        self.enable_endpoint_interp = bool(_cfg_get(cfg, "stitch.enable_endpoint_interp", True))
        self.stitch_short_overlap = int(_cfg_get(cfg, "stitch.short_overlap", 8))
        self.stitch_long_window_stride = int(
            _cfg_get(cfg, "stitch.long_window_stride", max(1, self.k - 1))
        )

        # ── Calibration ─────────────────────────────────────────────
        self.tform_calib: torch.Tensor | None = self._load_calib()

    # ── Calibration ─────────────────────────────────────────────────

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
                tform_calib = torch.as_tensor(np.array(tform_calib), dtype=torch.float32)
            return tform_calib.float().to(self.device)
        except Exception as exc:
            warnings.warn(f"[KRootDual] Could not load calibration: {exc}")
            return None

    # ── Model construction ───────────────────────────────────────────

    def _build_model(self, branch: str) -> LongSeqPoseModel:
        """Build a LongSeqPoseModel for the given branch.

        Config lookup order:  model.<branch>.transformer.* → model.transformer.*
        """
        backbone = str(_cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        pretrained = bool(_cfg_get(self.cfg, "model.encoder.pretrained", False))

        def _tf(key, default):
            # Try branch-specific first, then shared
            v = _cfg_get(self.cfg, f"model.{branch}.transformer.{key}")
            if v is None:
                v = _cfg_get(self.cfg, f"model.transformer.{key}", default)
            return v

        token_dim = int(_tf("d_model", 256))
        n_heads = int(_tf("n_heads", 4))
        n_layers = int(_tf("n_layers", 4))
        dim_ff = int(_tf("dim_feedforward", 1024))
        window_size = int(_tf("window_size", self.k))
        dropout = float(_tf("dropout", 0.1))

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
            aux_intervals=[],
            pretrained_backbone=pretrained,
            memory_size=0,
        )

    # ── Hooks ────────────────────────────────────────────────────────

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(self, **kwargs)

    # ── Checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "short_model": self.short_model.state_dict(),
            "long_model": self.long_model.state_dict(),
            "short_optimizer": self.short_optimizer.state_dict(),
            "long_optimizer": self.long_optimizer.state_dict(),
        }
        if self.short_ema is not None:
            payload["short_ema_shadow"] = self.short_ema.shadow
        if self.long_ema is not None:
            payload["long_ema_shadow"] = self.long_ema.shadow
        torch.save(payload, path)
        print(f"[KRootDual] checkpoint saved → {path}  (tag={tag})")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "short_model" in state:
            self.short_model.load_state_dict(state["short_model"])
        if "long_model" in state:
            self.long_model.load_state_dict(state["long_model"])
        if "short_optimizer" in state:
            try:
                self.short_optimizer.load_state_dict(state["short_optimizer"])
            except Exception:
                pass
        if "long_optimizer" in state:
            try:
                self.long_optimizer.load_state_dict(state["long_optimizer"])
            except Exception:
                pass
        if "short_ema_shadow" in state and self.short_ema is not None:
            self.short_ema.shadow = state["short_ema_shadow"]
        if "long_ema_shadow" in state and self.long_ema is not None:
            self.long_ema.shadow = state["long_ema_shadow"]
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        print(f"[KRootDual] checkpoint loaded ← {path}  (epoch={self.epoch}, step={self.global_step})")

    # Also support the standard `model` key for compatibility with the checkpoint hook
    @property
    def model(self):
        """Expose short_model as the primary 'model' for CheckpointHook compatibility.

        The full save/load uses save_checkpoint/load_checkpoint above,
        but single-model hooks (CheckpointHook) need a `.model` attr.
        We wrap both models into a ModuleDict so state_dict includes both.
        """
        if not hasattr(self, "_model_wrapper"):
            self._model_wrapper = nn.ModuleDict({
                "short": self.short_model,
                "long": self.long_model,
            })
        return self._model_wrapper

    # ── Single forward + loss for one branch ─────────────────────────

    def _forward_branch(
        self,
        model: LongSeqPoseModel,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass + chordal-rot + smooth-L1-trans loss for one branch."""
        frames = batch["frames"].to(self.device)
        gt_global = batch["gt_global_T"].to(self.device)

        if frames.max() > 1.1:
            frames = frames / 255.0

        out = model(frames)
        pred_local_T = out["pred_local_T"]

        gt_local_T = local_from_global(gt_global)

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

        metrics = {
            "rot_loss": float(rot_loss.item()),
            "trans_loss": float(trans_loss.item()),
            "drift_mm": float(drift.item()),
        }
        return loss, metrics

    # ── LR update ────────────────────────────────────────────────────

    def _update_lr(self) -> float:
        total_steps = self._estimate_total_steps()
        new_lr = _warmup_cosine_lr(
            self.global_step, self.warmup_steps, total_steps,
            self.base_lr, self.min_lr,
        )
        encoder_mult = float(_cfg_get(self.cfg, "optimizer.encoder_lr_mult", 0.1))
        for opt in (self.short_optimizer, self.long_optimizer):
            for pg in opt.param_groups:
                if pg.get("name") == "encoder":
                    pg["lr"] = new_lr * encoder_mult
                else:
                    pg["lr"] = new_lr
        return new_lr

    def _estimate_total_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        try:
            loader_len = len(self.short_train_loader)
        except TypeError:
            loader_len = 100
        steps_per_epoch = max(1, loader_len // self.grad_accum)
        return steps_per_epoch * self.num_epochs

    # ── Train loop ───────────────────────────────────────────────────

    def train(self):
        if self.short_train_loader is None or self.long_train_loader is None:
            raise ValueError("Both short_train_loader and long_train_loader are required")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            self.short_model.train()
            self.long_model.train()
            epoch_short_loss = 0.0
            epoch_long_loss = 0.0
            n_optim_steps = 0
            n_micro_steps = 0

            self.call_hooks("before_epoch")

            # Set epoch for both datasets
            for loader in (self.short_train_loader, self.long_train_loader):
                ds = getattr(loader, "dataset", None)
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(epoch)

            self.short_optimizer.zero_grad()
            self.long_optimizer.zero_grad()

            short_iter = iter(self.short_train_loader)
            long_iter = iter(self.long_train_loader)

            accum_short = 0.0
            accum_long = 0.0
            step_done = False

            while not step_done:
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break

                # ── Get batches ──────────────────────────────────────
                try:
                    short_batch = next(short_iter)
                except StopIteration:
                    break
                try:
                    long_batch = next(long_iter)
                except StopIteration:
                    # Long loader exhausted before short — restart it
                    long_iter = iter(self.long_train_loader)
                    try:
                        long_batch = next(long_iter)
                    except StopIteration:
                        break

                n_micro_steps += 1
                self.call_hooks("before_step")

                # ── Short forward + backward ─────────────────────────
                short_loss, short_metrics = self._forward_branch(self.short_model, short_batch)
                (self.short_loss_weight * short_loss / self.grad_accum).backward()
                accum_short += float(short_loss.item())

                # ── Long forward + backward ──────────────────────────
                long_loss, long_metrics = self._forward_branch(self.long_model, long_batch)
                (self.long_loss_weight * long_loss / self.grad_accum).backward()
                accum_long += float(long_loss.item())

                # ── Optimizer step every grad_accum micro-steps ──────
                if n_micro_steps % self.grad_accum == 0:
                    for model, opt in [
                        (self.short_model, self.short_optimizer),
                        (self.long_model, self.long_optimizer),
                    ]:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=self.max_grad_norm
                        )
                        opt.step()
                        opt.zero_grad()

                    self.global_step += 1
                    n_optim_steps += 1
                    current_lr = self._update_lr()

                    # EMA update
                    if self.short_ema is not None:
                        self.short_ema.update(self.short_model)
                    if self.long_ema is not None:
                        self.long_ema.update(self.long_model)

                    avg_s = accum_short / self.grad_accum
                    avg_l = accum_long / self.grad_accum
                    epoch_short_loss += avg_s
                    epoch_long_loss += avg_l
                    accum_short = accum_long = 0.0

                    if n_optim_steps % self.log_interval == 0 or n_optim_steps == 1:
                        print(
                            f"[KRootDual epoch {epoch}  step {n_optim_steps}]  "
                            f"short={avg_s:.4f}(rot={short_metrics['rot_loss']:.4f} "
                            f"trans={short_metrics['trans_loss']:.4f} "
                            f"drift={short_metrics['drift_mm']:.1f}mm)  "
                            f"long={avg_l:.4f}(rot={long_metrics['rot_loss']:.4f} "
                            f"trans={long_metrics['trans_loss']:.4f} "
                            f"drift={long_metrics['drift_mm']:.1f}mm)  "
                            f"lr={current_lr:.2e}"
                        )

                    self.call_hooks(
                        "after_step",
                        log_buffer={
                            "mode": "train",
                            "epoch": epoch + 1,
                            "iter": n_optim_steps,
                            "global_step": self.global_step,
                            "loss": avg_s + avg_l,
                            "short_loss": avg_s,
                            "long_loss": avg_l,
                            "short_rot": short_metrics["rot_loss"],
                            "short_trans": short_metrics["trans_loss"],
                            "short_drift_mm": short_metrics["drift_mm"],
                            "long_rot": long_metrics["rot_loss"],
                            "long_trans": long_metrics["trans_loss"],
                            "long_drift_mm": long_metrics["drift_mm"],
                            "lr": current_lr,
                        },
                    )

            # Handle remaining accumulated gradients
            if n_micro_steps % self.grad_accum != 0:
                rem = n_micro_steps % self.grad_accum
                for model, opt in [
                    (self.short_model, self.short_optimizer),
                    (self.long_model, self.long_optimizer),
                ]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                    opt.step()
                    opt.zero_grad()
                self.global_step += 1
                n_optim_steps += 1
                if self.short_ema is not None:
                    self.short_ema.update(self.short_model)
                if self.long_ema is not None:
                    self.long_ema.update(self.long_model)
                epoch_short_loss += accum_short / rem
                epoch_long_loss += accum_long / rem

            avg_s = epoch_short_loss / max(1, n_optim_steps)
            avg_l = epoch_long_loss / max(1, n_optim_steps)
            print(
                f"[KRootDual epoch {epoch}] avg short={avg_s:.4f}  avg long={avg_l:.4f}  "
                f"optim_steps={n_optim_steps}"
            )
            _lr = float(self.short_optimizer.param_groups[-1]["lr"])
            self.call_hooks(
                "after_epoch",
                log_buffer={
                    "epoch": epoch + 1,
                    "train_loss": avg_s + avg_l,
                    "short_loss": avg_s,
                    "long_loss": avg_l,
                    "lr": _lr,
                },
            )

            # ── Validation ───────────────────────────────────────────
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                self.call_hooks("before_val")
                val_metrics = self.evaluate(self.val_loader)
                self.last_eval_metrics = val_metrics
                print(f"[KRootDual val epoch {epoch}] {val_metrics}")
                self.call_hooks(
                    "after_val",
                    log_buffer={
                        "epoch": epoch + 1,
                        "val_loss": val_metrics.get("mean_gpe_mm_fused", 0.0),
                        **val_metrics,
                    },
                )

        self.call_hooks("after_train")
        self.call_hooks("after_run")

    # ── Evaluation with stitch fusion ────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader=None) -> dict[str, float]:
        """Evaluate both models with stitch fusion on full-scan data.

        For each scan in the loader:
        1. Run short + long inference via sliding windows
        2. Stitch: long anchors + short refinement + SE(3) endpoint correction
        3. Compute GPE/LPE/drift for fused / short_only / long_only
        """
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return {}

        # Apply EMA weights
        short_backup = long_backup = None
        if self.short_ema is not None:
            short_backup = self.short_ema.apply(self.short_model)
        if self.long_ema is not None:
            long_backup = self.long_ema.apply(self.long_model)

        self.short_model.eval()
        self.long_model.eval()

        all_metrics: list[dict[str, float]] = []
        scan_globals: dict[str, dict] = {}
        scan_count = 0

        for batch in loader:
            frames = batch["frames"].to(self.device)        # (1, T, H, W)
            gt_global = batch["gt_global_T"].to(self.device) # (1, T, 4, 4)

            B = frames.shape[0]
            for b in range(B):
                scan_frames = frames[b]      # (T, H, W)
                gt_g = gt_global[b]          # (T, 4, 4)

                result = stitch_long_base_short_refine(
                    short_model=self.short_model,
                    long_model=self.long_model,
                    scan_frames=scan_frames,
                    k=self.k,
                    s=self.s,
                    device=self.device,
                    short_overlap=self.stitch_short_overlap,
                    long_window_stride=self.stitch_long_window_stride,
                    enable_endpoint_interp=self.enable_endpoint_interp,
                )

                metrics = compute_stitch_metrics(
                    fused_global=result["fused_global"],
                    short_global=result["short_global"],
                    long_global=result["long_global"],
                    anchor_indices=result["anchor_indices"],
                    gt_global=gt_g,
                    tform_calib=self.tform_calib,
                    frames=scan_frames if self.tform_calib is not None else None,
                )
                all_metrics.append(metrics)

                # Store per-scan globals for VizHook
                meta = batch.get("meta")
                sid = f"scan_{scan_count}"
                if isinstance(meta, dict):
                    raw = meta.get("scan_id") or meta.get("scan_name")
                    if isinstance(raw, (list, tuple)):
                        sid = str(raw[b]) if b < len(raw) else str(raw[0])
                    elif raw is not None:
                        sid = str(raw)
                scan_globals[sid] = {
                    "pred": result["fused_global"].detach().cpu(),
                    "gt": gt_g.detach().cpu(),
                }
                scan_count += 1

        # Restore original (non-EMA) weights
        if short_backup is not None:
            _EMA.restore(self.short_model, short_backup)
        if long_backup is not None:
            _EMA.restore(self.long_model, long_backup)

        if not all_metrics:
            return {"mean_gpe_mm_fused": 0.0}

        # Aggregate
        agg: dict[str, Any] = {}
        float_keys = sorted({k for m in all_metrics for k in m if isinstance(m[k], (int, float))})
        for key in float_keys:
            vals = [m[key] for m in all_metrics if key in m]
            if vals:
                agg[f"mean_{key}"] = sum(vals) / len(vals)

        agg["num_scans"] = float(scan_count)
        agg["scan_globals"] = scan_globals  # type: ignore

        return agg
