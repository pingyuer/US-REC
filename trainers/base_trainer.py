"""BaseTrainer — abstract base class that captures the 80 % shared boilerplate.

Concrete trainers override:
    ``_build_models()`` → construct + return model(s)
    ``_build_optimizers()`` → construct + return optimizer(s)
    ``_run_step(batch)`` → forward + loss (returns ``(loss, metrics_dict)``)
    ``evaluate(loader)`` → full-scan evaluation (returns ``metrics_dict``)
    ``_models_and_optimizers()`` → pairs for grad clip / step / zero_grad

Optional overrides:
    ``_on_train_start()`` — extra setup before the epoch loop
    ``_on_epoch_start(epoch)`` — per-epoch setup (e.g. set_epoch on dataset)
    ``_format_step_log(…)`` — customize per-step console output
    ``_after_optim_step()`` — EMA update, additional bookkeeping
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from trainers.common import cfg_get, EMA, warmup_cosine_lr
from trainers.hooks.base_hook import Hook


class BaseTrainer(ABC):
    """Training skeleton shared by LongSeq / Dual / KRoot / KRootDual trainers.

    Lifecycle::

        trainer = ConcreteTrainer(cfg, device=…, …)
        trainer.train()
        trainer.evaluate(loader)
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(
        self,
        cfg,
        *,
        device: str | torch.device = "cpu",
    ):
        self.cfg = cfg
        self.device = torch.device(device)

        self.hooks: list[Hook] = []
        self.epoch: int = 0
        self.global_step: int = 0
        self._last_train_avg_loss: float = 0.0

        # ── Training schedule (read from config) ─────────────────────
        self.num_epochs = int(cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = cfg_get(cfg, "train.max_steps") or cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(cfg_get(cfg, "paths.output_dir", "logs"))

        # ── Gradient accumulation & clipping ─────────────────────────
        self.grad_accum = max(1, int(
            cfg_get(cfg, "trainer.grad_accum",
                    cfg_get(cfg, "trainer.grad_accum_steps", 1)) or 1
        ))
        self.max_grad_norm = float(cfg_get(cfg, "trainer.max_grad_norm", 1.0))

        # ── LR schedule ─────────────────────────────────────────────
        self.base_lr = float(
            cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 1e-4))
        )
        self.min_lr = float(cfg_get(cfg, "optimizer.min_lr", 1e-6))
        self.warmup_steps = int(cfg_get(cfg, "optimizer.warmup_steps", 0))

    # ------------------------------------------------------------------
    # Hook management (identical across all trainers)
    # ------------------------------------------------------------------

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(self, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss on a single micro-batch.

        Returns ``(loss, metrics_dict)`` where *loss* is differentiable.
        """

    @abstractmethod
    def evaluate(self, loader) -> dict[str, float]:
        """Full-scan evaluation.  Called inside ``torch.no_grad()``."""

    # Override to yield ``(model, optimizer)`` pairs for the generic
    # clip-step-zero loop.  Default: single ``(self.model, self.optimizer)``.
    def _models_and_optimizers(self) -> Sequence[tuple[nn.Module, torch.optim.Optimizer]]:
        return [(self.model, self.optimizer)]

    # ------------------------------------------------------------------
    # Generic train loop
    # ------------------------------------------------------------------

    def _get_train_loader(self):
        """Return the primary training DataLoader.

        Override in multi-loader trainers (e.g. KRootDual legacy mode).
        """
        return getattr(self, "train_loader", None)

    def _get_val_loader(self):
        return getattr(self, "val_loader", None)

    def _set_models_train(self) -> None:
        """Put all models into training mode."""
        for m, _ in self._models_and_optimizers():
            m.train()

    def _zero_all(self) -> None:
        for _, opt in self._models_and_optimizers():
            opt.zero_grad()

    def _clip_step_zero(self) -> None:
        """Gradient clip → optimizer step → zero_grad for all model/opt pairs."""
        for m, opt in self._models_and_optimizers():
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=self.max_grad_norm)
            opt.step()
            opt.zero_grad()

    def _update_lr(self) -> float:
        """Update LR for all optimizers using warmup + cosine schedule."""
        total = self._estimate_total_steps()
        new_lr = warmup_cosine_lr(
            self.global_step, self.warmup_steps, total,
            self.base_lr, self.min_lr,
        )
        encoder_mult = float(cfg_get(self.cfg, "optimizer.encoder_lr_mult", 1.0))
        for _, opt in self._models_and_optimizers():
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
            loader_len = len(self._get_train_loader())
        except TypeError:
            loader_len = 100
        steps_per_epoch = max(1, loader_len // self.grad_accum)
        return steps_per_epoch * self.num_epochs

    # Hooks into the train loop that subclasses may override
    def _on_train_start(self) -> None:
        """Called once before the epoch loop."""

    def _on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch (set_epoch, etc.)."""
        loader = self._get_train_loader()
        if loader is not None:
            ds = getattr(loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def _after_optim_step(self) -> None:
        """Called after each optimizer step (EMA update, etc.)."""

    def _pre_clip_metrics(self) -> dict[str, float]:
        """Compute extra metrics (e.g. gradient norms) BEFORE clip+step+zero.

        Called right before ``_clip_step_zero()`` when gradients are still
        available.  The returned dict is merged into the ``after_step``
        ``log_buffer``.
        """
        return {}

    def _build_val_log_buffer(
        self, val_metrics: dict, avg_loss: float, epoch: int,
    ) -> dict:
        """Build the ``log_buffer`` dict for after_val hooks.

        Override to rename / augment keys (e.g. ``mean_tusrec_*`` →
        ``val_tusrec_*``).
        """
        buf: dict = {
            "epoch": epoch + 1,
            "val_loss": val_metrics.get(
                "mean_gpe_mm",
                val_metrics.get("mean_gpe_pts_mm", avg_loss),
            ),
            **val_metrics,
        }
        return buf

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        """Format the per-step console log line."""
        return (
            f"[{self.__class__.__name__} epoch {epoch}  step {n_optim_steps}]  "
            f"loss={avg_accum:.4f}  lr={current_lr:.2e}"
        )

    def train(self):
        """Generic training loop with gradient accumulation.

        The loop structure is:
        1. Iterate micro-batches from ``_get_train_loader()``
        2. Call ``_run_step(batch)`` for forward + loss
        3. Scale loss by ``1/grad_accum`` and ``backward()``
        4. Every ``grad_accum`` micro-steps: clip, step, EMA, LR, log
        5. End-of-epoch: validation if due
        """
        loader = self._get_train_loader()
        if loader is None:
            raise ValueError(f"No train loader provided to {self.__class__.__name__}")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")
        self._on_train_start()

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.epoch = epoch
            self._set_models_train()
            self._on_epoch_start(epoch)
            self.call_hooks("before_epoch")

            self._zero_all()
            epoch_loss = 0.0
            n_micro = 0
            n_optim = 0
            accum_loss = 0.0

            for step, batch in enumerate(loader):
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    break

                n_micro += 1
                self.call_hooks("before_step")

                loss, metrics = self._run_step(batch)
                (loss / self.grad_accum).backward()
                accum_loss += float(loss.item())

                if n_micro % self.grad_accum == 0:
                    extra_metrics = self._pre_clip_metrics()
                    self._clip_step_zero()
                    self.global_step += 1
                    n_optim += 1

                    current_lr = self._update_lr()
                    self._after_optim_step()

                    avg_accum = accum_loss / self.grad_accum
                    epoch_loss += avg_accum
                    accum_loss = 0.0

                    if n_optim % self.log_interval == 0 or n_optim == 1:
                        msg = self._format_step_log(epoch, n_optim, avg_accum, metrics, current_lr)
                        print(msg)

                    self.call_hooks(
                        "after_step",
                        log_buffer={
                            "mode": "train",
                            "epoch": epoch + 1,
                            "iter": n_optim,
                            "global_step": self.global_step,
                            "loss": avg_accum,
                            "lr": current_lr,
                            **metrics,
                            **extra_metrics,
                        },
                    )

            # Flush remaining accumulated gradients
            if n_micro % self.grad_accum != 0:
                self._clip_step_zero()
                self.global_step += 1
                n_optim += 1
                self._after_optim_step()
                rem = n_micro % self.grad_accum
                epoch_loss += accum_loss / rem

            avg_loss = epoch_loss / max(1, n_optim)
            self._last_train_avg_loss = avg_loss
            print(f"[{self.__class__.__name__} epoch {epoch}] avg_loss={avg_loss:.4f}  optim_steps={n_optim}")

            _lr = self._get_current_lr()
            self.call_hooks(
                "after_epoch",
                log_buffer={"epoch": epoch + 1, "train_loss": avg_loss, "lr": _lr},
            )

            # Validation
            val_loader = self._get_val_loader()
            if val_loader is not None and (epoch + 1) % self.val_every == 0:
                self.call_hooks("before_val")
                val_metrics = self.evaluate(val_loader)
                self.last_eval_metrics = val_metrics
                print(f"[{self.__class__.__name__} val epoch {epoch}] {val_metrics}")
                val_log = self._build_val_log_buffer(val_metrics, avg_loss, epoch)
                self.call_hooks("after_val", log_buffer=val_log)

        self.call_hooks("after_train")
        self.call_hooks("after_run")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        """Save model(s), optimizer(s), epoch, global_step.

        Subclasses that hold EMA or extra state should override and call
        ``super().save_checkpoint(...)`` or build their own payload.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        pairs = self._models_and_optimizers()
        if len(pairs) == 1:
            m, o = pairs[0]
            payload["model"] = m.state_dict()
            payload["optimizer"] = o.state_dict()
        else:
            for i, (m, o) in enumerate(pairs):
                payload[f"model_{i}"] = m.state_dict()
                payload[f"optimizer_{i}"] = o.state_dict()
        torch.save(payload, path)
        print(f"[{self.__class__.__name__}] checkpoint saved → {path}  (tag={tag})")

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint (model + optimizer + epoch/step)."""
        state = torch.load(path, map_location=self.device)
        pairs = self._models_and_optimizers()
        if len(pairs) == 1:
            m, o = pairs[0]
            model_state = state.get("model") or state.get("state_dict") or state
            m.load_state_dict(model_state)
            if "optimizer" in state:
                try:
                    o.load_state_dict(state["optimizer"])
                except Exception:
                    pass
        else:
            for i, (m, o) in enumerate(pairs):
                ms = state.get(f"model_{i}")
                if ms:
                    m.load_state_dict(ms)
                os_ = state.get(f"optimizer_{i}")
                if os_:
                    try:
                        o.load_state_dict(os_)
                    except Exception:
                        pass
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        print(f"[{self.__class__.__name__}] checkpoint loaded ← {path}  (epoch={self.epoch}, step={self.global_step})")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_current_lr(self) -> float:
        """Read LR from the first non-encoder param group."""
        for _, opt in self._models_and_optimizers():
            for pg in opt.param_groups:
                if pg.get("name") != "encoder":
                    return float(pg["lr"])
            # fallback
            return float(opt.param_groups[-1]["lr"])
        return 0.0
