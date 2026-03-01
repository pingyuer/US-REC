"""Model checkpoint loading with MLflow-first fallback to local disk.

Priority order
--------------
1. **MLflow** — explicit ``checkpoint.mlflow_run_id`` in config, *or* automatic
   search for the best finished run in the configured experiment
   (when ``checkpoint.mlflow_search_best`` is ``true``).
2. **Local disk** — explicit ``checkpoint.local_path``, *or* automatic scan of
   ``<run_dir>/checkpoints/best/`` then ``<run_dir>/checkpoints/``.

Configuration (in YAML under ``checkpoint:``):

.. code-block:: yaml

    checkpoint:
      # --- MLflow source ---
      mlflow_run_id: null          # pin a specific run; null → search best run
      mlflow_artifact: "checkpoints/best"  # artifact sub-path inside the run
      mlflow_search_best: true     # auto-search best run when run_id is null

      # --- Local fallback ---
      local_path: null             # explicit .pt/.pth file; null → auto-scan

      # --- Behaviour ---
      load_on_eval: true           # load before --eval-only
      load_on_resume: true         # load when retain_epoch > 0 (training resume)
      strict: true                 # strict=True for load_state_dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mlflow_download(*, run_id: str, artifact_path: str) -> Optional[str]:
    """Download *artifact_path* from *run_id* and return its local path."""
    try:
        import mlflow  # noqa: PLC0415
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
        return local
    except Exception as exc:
        print(f"[checkpoint] MLflow download failed ({exc})")
        return None


def _mlflow_best_run_id(
    *,
    tracking_uri: str,
    experiment_name: str,
    metric_name: str,
    mode: str,
) -> Optional[str]:
    """Return the run_id of the best *finished* run in the experiment, or None."""
    try:
        import mlflow  # noqa: PLC0415
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"[checkpoint] MLflow experiment {experiment_name!r} not found")
            return None
        order = "DESC" if mode == "max" else "ASC"
        # The MLflowHook logs the best value under "best/<metric_name>"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=[f"metrics.`best/{metric_name}` {order}"],
            max_results=1,
        )
        if not runs:
            print(f"[checkpoint] MLflow: no finished runs in {experiment_name!r}")
            return None
        return runs[0].info.run_id
    except Exception as exc:
        print(f"[checkpoint] MLflow search_runs failed ({exc})")
        return None


def _resolve_local_path(
    run_dir: Optional[str], local_path: Optional[str]
) -> Optional[Path]:
    """Return an existing local checkpoint file, or None."""
    if local_path:
        p = Path(local_path)
        if p.exists():
            return p
        print(f"[checkpoint] local_path={local_path!r} not found")
        return None
    if run_dir:
        ckpt_dir = Path(run_dir) / "checkpoints"
        # MLflowHook writes best/<metric>.pt
        candidates = sorted((ckpt_dir / "best").glob("*.pt")) if (ckpt_dir / "best").is_dir() else []
        if not candidates:
            candidates = sorted(ckpt_dir.glob("best_*.pt"))
        if not candidates:
            candidates = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth"))
        if candidates:
            return candidates[0]
    return None


def _pick_file_from(path_str: str) -> Optional[Path]:
    """Given a path that may be a file or a download directory, return a .pt file."""
    p = Path(path_str)
    if p.is_file():
        return p
    if p.is_dir():
        pts = sorted(p.glob("*.pt")) + sorted(p.glob("*.pth"))
        if pts:
            return pts[0]
        print(f"[checkpoint] downloaded dir {path_str!r} contains no .pt/.pth files")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: torch.nn.Module,
    cfg,
    device: torch.device,
    *,
    ctx=None,
    strict: bool = True,
) -> bool:
    """Load model weights from MLflow (preferred) or local disk.

    Parameters
    ----------
    model:
        The ``torch.nn.Module`` whose weights to replace.
    cfg:
        OmegaConf config (the full project config).
    device:
        Target device for ``torch.load``.
    ctx:
        Optional ``TrainingContext``; provides ``run_dir`` for local fallback.
    strict:
        Default value for ``load_state_dict(strict=...)``.
        Overridden by ``checkpoint.strict`` in config.

    Returns
    -------
    bool
        ``True`` when weights were successfully loaded, ``False`` otherwise.
    """
    ckpt_cfg = OmegaConf.select(cfg, "checkpoint") or {}
    if OmegaConf.is_config(ckpt_cfg):
        ckpt_cfg = OmegaConf.to_container(ckpt_cfg, resolve=True)

    mlflow_run_id: Optional[str] = ckpt_cfg.get("mlflow_run_id") or None
    mlflow_artifact: str = str(ckpt_cfg.get("mlflow_artifact") or "checkpoints/best")
    mlflow_search_best: bool = bool(ckpt_cfg.get("mlflow_search_best", True))
    local_path: Optional[str] = ckpt_cfg.get("local_path") or None
    strict_load: bool = bool(ckpt_cfg.get("strict", strict))

    run_dir: Optional[str] = None
    if ctx is not None:
        run_dir = str(getattr(ctx, "run_dir", None) or "") or None

    raw_path: Optional[str] = None

    # ------------------------------------------------------------------
    # 1. MLflow
    # ------------------------------------------------------------------
    if mlflow_run_id:
        print(
            f"[checkpoint] MLflow: run_id={mlflow_run_id!r} "
            f"artifact={mlflow_artifact!r}"
        )
        raw_path = _mlflow_download(
            run_id=mlflow_run_id, artifact_path=mlflow_artifact
        )

    if raw_path is None and mlflow_search_best:
        # Read individual keys with OmegaConf.select so that unrelated
        # interpolations (e.g. run_name: ${now:...}) are never resolved here.
        mlflow_enabled = OmegaConf.select(cfg, "logging.mlflow.enabled", default=False)
        tracking_uri = str(OmegaConf.select(cfg, "logging.mlflow.tracking_uri") or "")
        experiment_name = str(OmegaConf.select(cfg, "logging.mlflow.experiment_name") or "")
        metric_name = str(OmegaConf.select(cfg, "logging.mlflow.best_metric") or "final_score")
        mode = str(OmegaConf.select(cfg, "logging.mlflow.best_mode") or "max")
        if bool(mlflow_enabled) and tracking_uri and experiment_name:
            print(
                f"[checkpoint] MLflow: searching best run in "
                f"{experiment_name!r} "
                f"(metric={metric_name}, mode={mode})"
            )
            best_run_id = _mlflow_best_run_id(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                metric_name=metric_name,
                mode=mode,
            )
            if best_run_id:
                print(f"[checkpoint] MLflow: best run_id={best_run_id!r}")
                raw_path = _mlflow_download(
                    run_id=best_run_id, artifact_path=mlflow_artifact
                )

    # ------------------------------------------------------------------
    # 2. Local fallback
    # ------------------------------------------------------------------
    if raw_path is None:
        local = _resolve_local_path(run_dir, local_path)
        if local is not None:
            raw_path = str(local)
            print(f"[checkpoint] local: {raw_path}")

    if raw_path is None:
        print("[checkpoint] no checkpoint found — model keeps random init weights")
        return False

    # ------------------------------------------------------------------
    # Resolve to a single file, then load
    # ------------------------------------------------------------------
    resolved = _pick_file_from(raw_path)
    if resolved is None:
        print(f"[checkpoint] could not resolve a .pt file from {raw_path!r}")
        return False

    try:
        state = torch.load(str(resolved), map_location=device)
        # Unwrap common checkpoint wrappers
        for key in ("model", "state_dict", "network"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        model.load_state_dict(state, strict=strict_load)
        print(f"[checkpoint] ✅ weights loaded from {resolved}")
        return True
    except Exception as exc:
        print(f"[checkpoint] ❌ load failed: {exc}")
        return False
