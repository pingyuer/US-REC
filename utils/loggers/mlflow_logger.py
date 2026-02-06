"""MLflow-backed experiment logger."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import BaseExperimentLogger

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class MLflowExperimentLogger(BaseExperimentLogger):
    def __init__(
        self,
        *,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        artifact_subdir: str = "artifacts",
        log_system_metrics: bool = False,
        register_model: bool = False,
        log_model_format: str = "state_dict",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or "default"
        self.artifact_subdir = artifact_subdir.strip("/") if artifact_subdir else "artifacts"
        self.log_system_metrics = bool(log_system_metrics)
        self.register_model = bool(register_model)
        self.log_model_format = str(log_model_format)
        self._disabled = mlflow is None

    def _validate_value(self, name: str, value: Optional[str]) -> None:
        if value is None:
            return
        if "${" in value or "oc.env" in value:
            raise ValueError(f"MLflow config '{name}' not resolved: {value}")

    def _validate_tracking_uri(self, tracking_uri: str) -> None:
        if not tracking_uri:
            raise ValueError("MLflow tracking_uri is empty")
        if "${" in tracking_uri or "oc.env" in tracking_uri:
            raise ValueError(f"MLflow tracking_uri not resolved: {tracking_uri}")
        if not (tracking_uri.startswith("http://") or tracking_uri.startswith("https://")):
            raise ValueError(f"MLflow tracking_uri must be http(s): {tracking_uri}")

    def start_run(self, *, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        if self._disabled:
            return
        self._validate_tracking_uri(str(self.tracking_uri))
        self._validate_value("experiment_name", str(self.experiment_name))
        if run_name is not None:
            self._validate_value("run_name", str(run_name))
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            if self.experiment_name:
                mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=run_name, tags=tags)
            if self.log_system_metrics and hasattr(mlflow, "log_system_metrics"):
                mlflow.log_system_metrics()
        except Exception:
            self._disabled = True

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            mlflow.log_params(params)
        except Exception:
            self._disabled = True

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        if self._disabled:
            return
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value), step=step)
        except Exception:
            self._disabled = True

    def log_artifact(self, path: str, *, artifact_path: Optional[str] = None) -> None:
        if self._disabled:
            return
        try:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        except Exception:
            self._disabled = True

    def log_model(self, model: Any, *, artifact_path: Optional[str] = None, name: str = "model.pt") -> None:
        if self._disabled:
            return
        try:
            if self.log_model_format == "mlflow.pytorch":
                mlflow.pytorch.log_model(model, artifact_path=artifact_path or name)
                return
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / name
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state, out_path)
                mlflow.log_artifact(str(out_path), artifact_path=artifact_path)
        except Exception:
            self._disabled = True

    def set_tags(self, tags: Dict[str, str]) -> None:
        if self._disabled:
            return
        try:
            mlflow.set_tags(tags)
        except Exception:
            self._disabled = True

    def flush(self) -> None:
        return

    def close(self) -> None:
        if self._disabled:
            return
        try:
            mlflow.end_run()
        except Exception:
            self._disabled = True

    @staticmethod
    def download_artifact(*, run_id: str, artifact_path: str) -> Optional[str]:
        if mlflow is None:
            return None
        try:
            return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        except Exception:
            return None
