"""
Trainer class for medical image segmentation.
Inspired by mmsegmentation architecture.
"""

import torch
import torch.nn as nn
from typing import Any, Optional

from omegaconf import DictConfig

from .builder import build_optimizer
from .hooks.base_hook import Hook
from models import build_loss
from .evaluator import Evaluator


class Trainer:
    """
    Single-device trainer with hook callbacks.
    
    Args:
        model: segmentation model
        train_loader: training dataloader
        val_loader: validation dataloader
        cfg: OmegaConf config
        device: training device (default: "cuda")
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: DictConfig,
        device="cuda",
        ctx: Optional[Any] = None,
        test_loader=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        self.ctx = ctx

        self.hooks: list[Hook] = []
        self.evaluator = Evaluator(device=device)
        self._evaluator_callbacks: list[Any] = []
        
        # Training state
        self.epoch = 0
        self.global_step = 0  # global iteration counter
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.last_val_metrics = {}
        
        # Setup training components
        self._build_loss()
        self._build_optimizer_and_scheduler()

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: int(getattr(h, "priority", 50)), reverse=True)

    def register_hooks(self, hooks: list[Hook]) -> None:
        for hook in hooks:
            self.register_hook(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for hook in self.hooks:
            fn = getattr(hook, event, None)
            if callable(fn):
                fn(self, **kwargs)

    def add_evaluator_callback(self, callback: Any) -> None:
        """
        Register a callback object consumed by Evaluator during val/test.

        This is the public API hooks should use (instead of mutating private
        attributes) to avoid tight coupling.
        """
        if callback is None:
            return
        if not hasattr(self, "_evaluator_callbacks") or self._evaluator_callbacks is None:
            self._evaluator_callbacks = []
        self._evaluator_callbacks.append(callback)

    def _take_evaluator_callbacks(self) -> list[Any]:
        callbacks = list(getattr(self, "_evaluator_callbacks", None) or [])
        if hasattr(self, "_evaluator_callbacks"):
            self._evaluator_callbacks.clear()
        return callbacks

    def _build_loss(self):
        """Build loss function from config."""
        loss_cfg = self.cfg.get("loss")
        
        # Default to CrossEntropyLoss if not specified
        if loss_cfg is None:
            self.criterion = nn.CrossEntropyLoss()
            print("[Trainer] Using default CrossEntropyLoss")
            return

        try:
            self.criterion = build_loss(loss_cfg)
        except Exception as exc:
            loss_type = loss_cfg.get("type", "CrossEntropyLoss")
            print(f"[Trainer] Loss build failed for '{loss_type}': {exc}; falling back to CrossEntropyLoss")
            self.criterion = nn.CrossEntropyLoss(ignore_index=loss_cfg.get("ignore_index", -100))
    
    def _build_optimizer_and_scheduler(self):
        """Build optimizer and learning rate scheduler from config."""
        # Optimizer - use the factory function
        optim_cfg = self.cfg.get("optimizer")
        if optim_cfg is None:
            # Fallback to AdamW with default lr
            import torch.optim as optim
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
            print("[Trainer] Using default AdamW optimizer")
        else:
            self.optimizer = build_optimizer(optim_cfg, self.model)
        
        # Learning rate scheduler (optional)
        sched_cfg = self.cfg.get("lr_scheduler")
        if sched_cfg and sched_cfg.get("type") == "PolyLR":
            from torch.optim.lr_scheduler import LambdaLR
            max_epochs = self.cfg.trainer.get("max_epochs", 100)
            power = sched_cfg.get("power", 0.9)
            
            def poly_decay(epoch):
                return max(0, (1 - epoch / max_epochs) ** power)
            
            self.scheduler = LambdaLR(self.optimizer, poly_decay)
        else:
            self.scheduler = None
    
    def train_one_epoch(self):
        """Run one epoch of training."""
        if self.train_loader is None:
            raise ValueError("train_loader is None; cannot run train_one_epoch()")
        self.model.train()
        epoch_loss = 0.0
        max_steps = None
        try:
            max_steps = self.cfg.trainer.get("max_steps")
        except Exception:
            max_steps = None
        max_steps = int(max_steps) if max_steps is not None else None
        num_seen_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if max_steps is not None and self.global_step >= max_steps:
                break
            self.global_step += 1
            num_seen_batches += 1
            self.call_hooks("before_step")
            
            images, masks = self.evaluator._unpack_batch(batch)
            
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()

            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.call_hooks(
                "after_step",
                log_buffer={
                    "mode": "train",
                    "epoch": self.epoch + 1,
                    "iter": batch_idx + 1,
                    "global_step": self.global_step,
                    "loss": float(loss.item()),
                    "lr": lr,
                },
            )
        
        avg_loss = epoch_loss / max(1, num_seen_batches)
        
        # Step scheduler
        if self.scheduler:
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self):
        """Run validation (delegates to Evaluator)."""
        callbacks = self._take_evaluator_callbacks()
        metrics = self.evaluator.run(
            model=self.model,
            loader=self.val_loader,
            criterion=self.criterion,
            cfg=self.cfg,
            mode="val",
            epoch=int(self.epoch + 1),
            ctx=self.ctx,
            callbacks=callbacks,
        )
        self.last_val_metrics = metrics
        return metrics

    def test(self):
        """Run test (delegates to Evaluator)."""
        callbacks = self._take_evaluator_callbacks()
        metrics = self.evaluator.run(
            model=self.model,
            loader=self.test_loader,
            criterion=self.criterion,
            cfg=self.cfg,
            mode="test",
            epoch=int(self.epoch + 1),
            ctx=self.ctx,
            callbacks=callbacks,
        )
        self.last_test_metrics = metrics
        return metrics

    def run(self, mode: str = "train"):
        mode = str(mode).strip().lower()
        self.call_hooks("before_run", mode=mode)
        try:
            if mode == "train":
                return self.train()
            if mode == "val":
                self.call_hooks("before_val")
                metrics = self.validate()
                self.call_hooks("after_val", log_buffer=metrics)
                return metrics
            if mode == "test":
                self.call_hooks("before_test")
                metrics = self.test()
                self.call_hooks("after_test", log_buffer=metrics)
                return metrics
            raise ValueError(f"Unknown mode: {mode}")
        except BaseException as exc:
            self.call_hooks("on_exception", exc=exc)
            raise
        finally:
            self.call_hooks("after_run", mode=mode)
    
    def train(self):
        """Main training loop."""
        max_epochs = self.cfg.trainer.get("max_epochs", 10)
        eval_interval = self.cfg.trainer.get("eval_interval", 1)
        max_steps = self.cfg.trainer.get("max_steps", None)
        max_steps = int(max_steps) if max_steps is not None else None

        self.call_hooks("before_train")
        for epoch in range(max_epochs):
            self.epoch = epoch

            self.call_hooks("before_epoch")
            train_loss = self.train_one_epoch()
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.call_hooks(
                "after_epoch",
                log_buffer={
                    "epoch": epoch + 1,
                    "max_epochs": max_epochs,
                    "train_loss": float(train_loss),
                    "lr": lr,
                },
            )
            
            # Validate
            if eval_interval and (epoch + 1) % int(eval_interval) == 0:
                self.call_hooks("before_val")
                val_metrics = self.validate()
                is_best = False
                
                # Simple best model tracking
                if val_metrics['val_loss'] < self.best_metric:
                    self.best_metric = val_metrics['val_loss']
                    self.best_epoch = epoch
                    is_best = True

                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                self.call_hooks(
                    "after_val",
                    log_buffer={
                        **{k: float(v) for k, v in val_metrics.items()},
                        "epoch": epoch + 1,
                        "max_epochs": max_epochs,
                        "lr": lr,
                        "is_best": is_best,
                        "best_val_loss": float(self.best_metric),
                        "best_epoch": int(self.best_epoch + 1),
                    },
                )

            if max_steps is not None and self.global_step >= max_steps:
                break

        self.call_hooks("after_train")
