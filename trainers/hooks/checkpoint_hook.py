import os

import torch

from .base_hook import Hook
from .registry import register_hook


@register_hook("CheckpointHook")
class CheckpointHook(Hook):
    """
    Hook that saves model checkpoints.

    - Periodic checkpoints: `after_epoch` with `interval`
    - Best checkpoint: `after_val` (uses `is_best` if provided, or compares `metric_name`)
    - Last checkpoint: `after_train`
    """

    def __init__(
        self,
        interval: int = 0,
        dirpath: str = "./checkpoints",
        metric_name: str | None = "val_loss",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
        best_filename: str = "best_model.pth",
        last_filename: str = "last_model.pth",
        save_optimizer: bool = False,
    ):
        """
        interval: save every N epochs (0 disables periodic checkpoints)
        metric_name: metric key in `log_buffer` to decide improvement (default: val_loss)
        mode: "max" or "min"
        save_best/save_last: write best/last checkpoints
        save_optimizer: include optimizer state_dict in checkpoint file
        """
        self.interval = int(interval)
        self.dirpath = dirpath
        self.metric_name = metric_name
        self.mode = mode
        self.save_best = bool(save_best)
        self.save_last = bool(save_last)
        self.best_filename = str(best_filename)
        self.last_filename = str(last_filename)
        self.save_optimizer = bool(save_optimizer)

        self.best = None
        os.makedirs(dirpath, exist_ok=True)

    def before_run(self, trainer, mode: str = "train"):
        ctx = getattr(trainer, "ctx", None)
        if ctx is None:
            return
        if self.dirpath != "./checkpoints":
            return
        run_dir = getattr(ctx, "run_dir", None)
        if not run_dir:
            return
        self.dirpath = os.path.join(str(run_dir), "checkpoints")
        os.makedirs(self.dirpath, exist_ok=True)

    def after_train(self, trainer):
        if not self.save_last:
            return
        self._save(
            trainer,
            path=os.path.join(self.dirpath, self.last_filename),
            tag="last",
        )

    def after_epoch(self, trainer, log_buffer=None):
        if self.interval <= 0:
            return
        epoch = trainer.epoch + 1
        if (epoch) % self.interval == 0:
            self._save(
                trainer,
                path=os.path.join(self.dirpath, f"epoch{epoch}.pth"),
                tag="periodic",
            )

    def after_val(self, trainer, log_buffer=None):
        if not self.save_best:
            return
        if not log_buffer:
            return

        is_best = bool(log_buffer.get("is_best", False))
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        val_metric = None
        if self.metric_name and self.metric_name in log_buffer:
            try:
                val_metric = float(log_buffer[self.metric_name])
            except Exception:
                val_metric = None

        if not is_best and val_metric is not None:
            improved = False
            if self.best is None:
                improved = True
            else:
                if self.mode == "max" and val_metric > self.best:
                    improved = True
                if self.mode == "min" and val_metric < self.best:
                    improved = True
            is_best = improved

        if is_best:
            if val_metric is not None:
                self.best = val_metric
            self._save(
                trainer,
                path=os.path.join(self.dirpath, self.best_filename),
                tag="best",
                extra={"epoch": epoch, "metric_name": self.metric_name, "metric": val_metric},
            )

    def _save(self, trainer, *, path: str, tag: str, extra: dict | None = None) -> None:
        payload = {
            "tag": tag,
            "epoch": int(getattr(trainer, "epoch", 0)) + 1,
            "global_step": int(getattr(trainer, "global_step", 0)),
            "model": trainer.model.state_dict(),
        }
        if self.save_optimizer and getattr(trainer, "optimizer", None) is not None:
            payload["optimizer"] = trainer.optimizer.state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
