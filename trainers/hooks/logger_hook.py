from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from .base_hook import Hook
from .registry import register_hook


@register_hook("LoggerHook")
class LoggerHook(Hook):
    """
    Minimal logger hook: local heartbeat (file/console) only.

    Produces two files inside the run directory:
      * ``train.log``    — continuous step / epoch / val heartbeats.
      * ``metrics.log``  — one structured block per validation epoch,
                           containing every numeric metric in log_buffer.
                           Serves as an offline backup before MLflow push.
    """

    priority = 90

    def __init__(
        self,
        interval: int = 50,
        log_file: Optional[str] = None,
        console: bool = True,
        mlflow_enabled: bool = True,
        upload_run_dir: bool = True,
        delete_local_run_dir: bool = False,
        artifact_path: str = "run",
        flush_interval: int = 20,
    ):
        self.interval = int(interval)
        self.log_file = log_file
        self.console = bool(console)
        self.mlflow_enabled = bool(mlflow_enabled)
        self.upload_run_dir = bool(upload_run_dir)
        self.delete_local_run_dir = bool(delete_local_run_dir)
        self.artifact_path = str(artifact_path)
        self.flush_interval = int(flush_interval)
        self._fp = None
        self._metrics_fp = None
        self._write_count = 0

    def before_run(self, trainer, mode: str = "train"):
        self._open_log(trainer)
        self._writeln(f"[run] mode={mode}")

    def before_train(self, trainer):
        self._open_log(trainer)

    def after_train(self, trainer):
        # Keep for backward compatibility; prefer after_run() for archive/cleanup.
        self._close_log()

    def after_run(self, trainer, mode: str = "train"):
        self._close_log()

    def after_step(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        step = int(log_buffer.get("global_step", 0))
        if step <= 0 or (step % self.interval) != 0:
            return

        loss = log_buffer.get("loss", None)
        lr = log_buffer.get("lr", None)
        epoch = log_buffer.get("epoch", None)
        loss_str = f"{loss:.6f}" if loss is not None else "n/a"
        lr_str = f"{lr:.6g}" if lr is not None else "n/a"
        msg = f"[train] epoch={epoch} step={step} loss={loss_str} lr={lr_str}"
        self._writeln(msg)

    def after_epoch(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        train_loss = log_buffer.get("train_loss", None)
        lr = log_buffer.get("lr", None)
        loss_str = f"{train_loss:.6f}" if train_loss is not None else "n/a"
        lr_str = f"{lr:.6g}" if lr is not None else "n/a"
        msg = f"[epoch] epoch={epoch} train_loss={loss_str} lr={lr_str}"
        self._writeln(msg)

    def after_val(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        is_best = bool(log_buffer.get("is_best", False))
        val_loss = log_buffer.get("val_loss", None)
        best_val_loss = log_buffer.get("best_val_loss", None)
        best_epoch = log_buffer.get("best_epoch", None)

        msg = f"[val] epoch={epoch} val_loss={val_loss:.6f}" if val_loss is not None else f"[val] epoch={epoch}"
        if is_best:
            msg += f" (best val_loss={best_val_loss:.6f} @ epoch={best_epoch})"
        self._writeln(msg)

        # Write all numeric metrics to the metrics log (offline MLflow backup)
        self._write_metrics_block(epoch, log_buffer)

    def after_test(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        test_loss = log_buffer.get("test_loss", None)
        msg = f"[test] epoch={epoch} test_loss={test_loss:.6f}" if test_loss is not None else f"[test] epoch={epoch}"
        self._writeln(msg)

    def _writeln(self, msg: str) -> None:
        line = msg.rstrip("\n") + "\n"
        if self.console:
            print(msg)
        if self._fp is not None:
            self._fp.write(line)
            self._write_count += 1
            if self._write_count % self.flush_interval == 0:
                self._fp.flush()

    def _write_metrics_block(self, epoch: int, buf: dict) -> None:
        """Append a structured metrics block to metrics.log."""
        if self._metrics_fp is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n=== val epoch={epoch}  [{ts}] ==="]
        # Separate keys: put val_loss / best_val_loss first, then the rest sorted
        priority_keys = ["val_loss", "best_val_loss", "best_epoch", "lr"]
        skip_keys = {"epoch", "is_best", "scan_globals", "tusrec_per_scan"}
        seen: set[str] = set()
        for k in priority_keys:
            if k in buf and buf[k] is not None:
                v = buf[k]
                lines.append(f"  {k:<40s}: {v:.6g}" if isinstance(v, float) else f"  {k:<40s}: {v}")
                seen.add(k)
        for k in sorted(buf.keys()):
            if k in seen or k in skip_keys:
                continue
            v = buf[k]
            if isinstance(v, float):
                lines.append(f"  {k:<40s}: {v:.6g}")
            elif isinstance(v, int):
                lines.append(f"  {k:<40s}: {v}")
        self._metrics_fp.write("\n".join(lines) + "\n")
        self._metrics_fp.flush()

    def _open_log(self, trainer) -> None:
        if self._fp is not None:
            return
        if self.log_file:
            path = Path(self.log_file)
        elif getattr(trainer, "ctx", None) is not None and getattr(trainer.ctx, "log_file", None) is not None:
            path = Path(trainer.ctx.log_file)
        else:
            path = Path("logs/train.log")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("a", encoding="utf-8")
        self._writeln(f"[run] log_file={path}")

        # Open companion metrics.log in the same directory
        metrics_path = path.parent / "metrics.log"
        self._metrics_fp = metrics_path.open("a", encoding="utf-8")
        self._writeln(f"[run] metrics_log={metrics_path}")

    def _close_log(self) -> None:
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None
        if self._metrics_fp is not None:
            self._metrics_fp.flush()
            self._metrics_fp.close()
            self._metrics_fp = None
