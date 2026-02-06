from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from .base_hook import Hook
from .registry import register_hook
from utils.loggers import BaseExperimentLogger, MLflowExperimentLogger, NoOpExperimentLogger


def _ensure_resolvers() -> None:
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt))


def _flatten_config(cfg: Any) -> Dict[str, str]:
    def _recurse(value, prefix=""):
        entries = {}
        if isinstance(value, dict):
            for key, field in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                entries.update(_recurse(field, next_prefix))
        else:
            low = prefix.lower()
            if any(s in low for s in ("secret", "password", "token", "access_key")):
                entries[prefix] = "***"
            else:
                entries[prefix] = "None" if value is None else str(value)
        return entries

    if OmegaConf.is_config(cfg):
        _ensure_resolvers()
        container = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        container = cfg
    else:
        container = {}
    return _recurse(container)


def _get_git_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _safe_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def _resolve_run_name(template: Optional[str], cfg: Any) -> Optional[str]:
    if not template:
        return template
    resolved = template
    if "${now:" in resolved:
        while "${now:" in resolved:
            start = resolved.find("${now:")
            end = resolved.find("}", start)
            if end == -1:
                break
            fmt = resolved[start + len("${now:") : end]
            resolved = resolved[:start] + datetime.now().strftime(fmt) + resolved[end + 1 :]
    if "${model.name}" in resolved:
        model_name = OmegaConf.select(cfg, "model.name") if OmegaConf.is_config(cfg) else None
        if model_name is not None:
            resolved = resolved.replace("${model.name}", str(model_name))
    return resolved


@register_hook("MLflowHook")
class MLflowHook(Hook):
    priority = 80

    def __init__(
        self,
        *,
        cfg: Any,
        logger: Optional[BaseExperimentLogger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or self._build_logger(cfg)
        self.best_metric_name = None
        self.best_metric_value = None
        self.best_mode = None
        self._warned = False

    def _build_logger(self, cfg: Any) -> BaseExperimentLogger:
        mlflow_cfg = OmegaConf.select(cfg, "logging.mlflow")
        if mlflow_cfg is None and hasattr(cfg, "get"):
            mlflow_cfg = cfg.get("mlflow")
        if mlflow_cfg:
            _ensure_resolvers()
            mlflow_cfg = OmegaConf.to_container(mlflow_cfg, resolve=True)
        else:
            mlflow_cfg = {}
        enabled = bool(mlflow_cfg.get("enabled", bool(mlflow_cfg)))
        if not enabled:
            return NoOpExperimentLogger()
        try:
            return MLflowExperimentLogger(
                tracking_uri=str(mlflow_cfg.get("tracking_uri") or ""),
                experiment_name=str(mlflow_cfg.get("experiment_name") or ""),
                artifact_subdir=mlflow_cfg.get("artifact_subdir", "artifacts"),
                log_system_metrics=bool(mlflow_cfg.get("log_system_metrics", False)),
                register_model=bool(mlflow_cfg.get("register_model", False)),
                log_model_format=str(mlflow_cfg.get("log_model_format", "state_dict")),
            )
        except Exception:
            if not self._warned:
                print("[mlflow] disabled (logger init failed)")
                self._warned = True
            return NoOpExperimentLogger()

    def _resolve_best_metric(self, log_buffer: dict) -> tuple[str, str]:
        mlflow_cfg = OmegaConf.select(self.cfg, "logging.mlflow")
        if mlflow_cfg is None and hasattr(self.cfg, "get"):
            mlflow_cfg = self.cfg.get("mlflow")
        mlflow_cfg = OmegaConf.to_container(mlflow_cfg, resolve=True) if mlflow_cfg else {}
        metric_name = str(mlflow_cfg.get("best_metric") or "")
        if not metric_name:
            metric_name = "final_score" if "final_score" in log_buffer else "val_loss"
        mode = str(mlflow_cfg.get("best_mode") or "")
        if not mode:
            mode = "max" if metric_name in {"final_score"} else "min"
        return metric_name, mode

    def before_run(self, trainer, mode: str = "train"):
        mlflow_cfg = OmegaConf.select(self.cfg, "logging.mlflow")
        if mlflow_cfg is None and hasattr(self.cfg, "get"):
            mlflow_cfg = self.cfg.get("mlflow")
        if mlflow_cfg:
            _ensure_resolvers()
            mlflow_cfg = OmegaConf.to_container(mlflow_cfg, resolve=True)
        else:
            mlflow_cfg = {}
        run_name = _resolve_run_name(mlflow_cfg.get("run_name"), self.cfg)
        tags = mlflow_cfg.get("tags") or {}
        try:
            print(
                "[mlflow] resolved config summary:",
                f"tracking_uri={getattr(self.logger, 'tracking_uri', None)}",
                f"experiment_name={getattr(self.logger, 'experiment_name', None)}",
                f"run_name={run_name}",
            )
            self.logger.start_run(run_name=run_name, tags=tags)
        except ValueError:
            raise
        except Exception:
            if not self._warned:
                print("[mlflow] disabled (start_run failed)")
                self._warned = True
            self.logger = NoOpExperimentLogger()

        params = _flatten_config(self.cfg)
        params["command"] = " ".join(sys.argv)
        git_hash = _get_git_hash()
        if git_hash:
            params["git_commit"] = git_hash
        device = getattr(trainer, "device", None)
        if device is not None:
            params["device"] = str(device)
            params["cuda_available"] = str(torch.cuda.is_available())
        params["metric_spec"] = (
            "translation_error_mm,rotation_error_deg,se3_trans_mm,se3_rot_deg,"
            "endpoint_rpe_mm,endpoint_rpe_deg,end_to_start_rpe_mm,end_to_start_rpe_deg,wrap_dist_enabled,"
            "GPE_mm,GLE_mm,LPE_mm,LLE_mm,GPE_score,GLE_score,LPE_score,LLE_score,"
            "runtime_s_per_scan,runtime_forward_s_per_scan,runtime_e2e_s_per_scan,final_score"
        )
        self.logger.log_params(params)
        self._log_calib_summary(trainer)

    def after_step(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        step = int(log_buffer.get("global_step", 0))
        metrics = {}
        if "loss" in log_buffer:
            metrics["train/loss"] = float(log_buffer["loss"])
        if "lr" in log_buffer:
            metrics["train/lr"] = float(log_buffer["lr"])
        if metrics:
            self.logger.log_metrics(metrics, step=step)

    def after_epoch(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        metrics = {}
        if "train_loss" in log_buffer:
            metrics["train/epoch_loss"] = float(log_buffer["train_loss"])
        if "lr" in log_buffer:
            metrics["train/epoch_lr"] = float(log_buffer["lr"])
        if metrics:
            self.logger.log_metrics(metrics, step=epoch)

    def after_val(self, trainer, log_buffer=None):
        if not log_buffer:
            return
        epoch = int(log_buffer.get("epoch", trainer.epoch + 1))
        metrics = {}
        tusrec_rows = log_buffer.get("tusrec_per_scan")
        for key, value in log_buffer.items():
            if key in {"epoch", "max_epochs", "is_best", "tusrec_per_scan"}:
                continue
            if isinstance(value, (int, float)):
                metrics[f"val/{key}" if key != "val_loss" else "val/loss"] = float(value)
        if metrics:
            self.logger.log_metrics(metrics, step=epoch)

        metric_name, mode = self._resolve_best_metric(log_buffer)
        metric_value = log_buffer.get(metric_name)
        if isinstance(metric_value, (int, float)):
            improved = False
            if self.best_metric_value is None:
                improved = True
            elif mode == "min" and metric_value < self.best_metric_value:
                improved = True
            elif mode == "max" and metric_value > self.best_metric_value:
                improved = True
            if improved:
                self.best_metric_name = metric_name
                self.best_metric_value = float(metric_value)
                self.best_mode = mode
                self._save_best_checkpoint(trainer, metric_name, epoch)
                self.logger.log_metrics({f"best/{metric_name}": self.best_metric_value}, step=epoch)

        if tusrec_rows:
            self._log_tusrec_rows(trainer, tusrec_rows)

    def after_run(self, trainer, mode: str = "train"):
        self._save_last_checkpoint(trainer)
        self._log_config(trainer)
        self.logger.close()

    def on_exception(self, trainer, exc: BaseException):
        try:
            self.logger.set_tags({"run_status": "failed"})
        finally:
            self.logger.close()

    def _save_best_checkpoint(self, trainer, metric_name: str, epoch: int) -> None:
        ctx = getattr(trainer, "ctx", None)
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        if not run_dir:
            return
        ckpt_dir = Path(run_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"best_{metric_name}.pt"
        torch.save(_safe_state_dict(trainer.model), path)
        self.logger.log_artifact(str(path), artifact_path="checkpoints/best")

    def _save_last_checkpoint(self, trainer) -> None:
        ctx = getattr(trainer, "ctx", None)
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        if not run_dir:
            return
        ckpt_dir = Path(run_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "last.pt"
        torch.save(_safe_state_dict(trainer.model), path)
        self.logger.log_artifact(str(path), artifact_path="checkpoints/last")

    def _log_config(self, trainer) -> None:
        ctx = getattr(trainer, "ctx", None)
        config_file = getattr(ctx, "config_file", None) if ctx is not None else None
        if not config_file:
            return
        self.logger.log_artifact(str(config_file), artifact_path="configs")

    def _log_tusrec_rows(self, trainer, rows: list[dict[str, Any]]) -> None:
        ctx = getattr(trainer, "ctx", None)
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        if not run_dir:
            return
        out_dir = Path(run_dir) / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "val_tusrec_per_scan.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        self.logger.log_artifact(str(out_path), artifact_path="metrics")

    def _log_calib_summary(self, trainer) -> None:
        tform_calib = getattr(trainer, "tform_calib", None)
        tform_scale = getattr(trainer, "tform_calib_scale", None)
        if tform_calib is None:
            return
        ctx = getattr(trainer, "ctx", None)
        run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
        if not run_dir:
            return
        out_dir = Path(run_dir) / "calib"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "tform_calib": torch.as_tensor(tform_calib).detach().cpu().tolist(),
            "tform_calib_scale": torch.as_tensor(tform_scale).detach().cpu().tolist()
            if tform_scale is not None
            else None,
        }
        out_path = out_dir / "calib_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.log_artifact(str(out_path), artifact_path="calib")
