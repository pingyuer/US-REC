from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf


def _safe_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-", "."})


@dataclass(frozen=True)
class TrainingContext:
    """
    Per-run context (no globals).

    This holds shared, mostly-readonly run metadata and paths (log dir, config
    snapshot, etc.). Training state (epoch/step/metrics) stays on Trainer.
    """

    exp_name: str
    run_name: str
    root_dir: Path
    run_dir: Path
    log_file: Path
    config_file: Path
    mlflow_run_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        *,
        exp_name: str,
        run_name: Optional[str] = None,
        root_dir: str | Path = "logs",
        timestamp: Optional[str] = None,
    ) -> "TrainingContext":
        root_path = Path(root_dir)
        exp_name = _safe_name(exp_name or "default")
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = _safe_name(run_name or timestamp)
        run_dir = root_path / exp_name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            exp_name=exp_name,
            run_name=run_name,
            root_dir=root_path,
            run_dir=run_dir,
            log_file=run_dir / "train.log",
            config_file=run_dir / "config.yaml",
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        root_dir: str | Path = "logs",
        run_name: Optional[str] = None,
    ) -> "TrainingContext":
        exp_name = None
        if cfg is not None:
            exp_name = OmegaConf.select(cfg, "experiment.name")
            if exp_name is None:
                exp_name = OmegaConf.select(cfg, "mlflow.experiment_name")
        return cls.create(exp_name=exp_name or "default", run_name=run_name, root_dir=root_dir)

    def save_config(self, cfg: Any) -> None:
        if cfg is None:
            return
        # Keep env interpolations unresolved to avoid leaking secrets into artifacts.
        OmegaConf.save(cfg, str(self.config_file), resolve=False)

    def with_mlflow_run_id(self, run_id: Optional[str]) -> "TrainingContext":
        return TrainingContext(
            exp_name=self.exp_name,
            run_name=self.run_name,
            root_dir=self.root_dir,
            run_dir=self.run_dir,
            log_file=self.log_file,
            config_file=self.config_file,
            mlflow_run_id=run_id,
        )
