"""Experiment logger interfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional


class BaseExperimentLogger:
    def start_run(self, *, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str, *, artifact_path: Optional[str] = None) -> None:
        pass

    def log_model(self, model: Any, *, artifact_path: Optional[str] = None, name: str = "model.pt") -> None:
        pass

    def set_tags(self, tags: Dict[str, str]) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class NoOpExperimentLogger(BaseExperimentLogger):
    def start_run(self, *, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> None:
        return

    def log_params(self, params: Dict[str, Any]) -> None:
        return

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        return

    def log_artifact(self, path: str, *, artifact_path: Optional[str] = None) -> None:
        return

    def log_model(self, model: Any, *, artifact_path: Optional[str] = None, name: str = "model.pt") -> None:
        return

    def set_tags(self, tags: Dict[str, str]) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return
