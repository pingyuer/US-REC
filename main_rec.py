import importlib
import inspect
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import mlflow
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data.builder import build_dataset
from trainers.context import TrainingContext
from trainers.hooks import LoggerHook
from trainers.rec_trainer import Train_Rec_Reg_Model  # compat for tests


def _resolve_class(name_or_obj: Any):
    if not isinstance(name_or_obj, str):
        return name_or_obj
    module, cls_name = name_or_obj.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, cls_name)


def load_config(config_path: str, overrides: list[str]) -> Any:
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)
    return base_cfg


def load_dotenv(path: str = ".env", *, override: bool = False) -> None:
    """Minimal .env loader (no external dependency)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def _flatten_config(cfg) -> Dict[str, str]:
    def _recurse(value, prefix=""):
        entries = {}
        if isinstance(value, dict):
            for key, field in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                entries.update(_recurse(field, next_prefix))
        else:
            entries[prefix] = "None" if value is None else str(value)
        return entries

    container = OmegaConf.to_container(cfg, resolve=True)
    return _recurse(container)


def _mlflow_run_active() -> bool:
    return mlflow.active_run() is not None


def _log_params(params: dict, *, batch_size: int = 64):
    if not _mlflow_run_active():
        return
    items = list(params.items())
    for idx in range(0, len(items), batch_size):
        mlflow.log_params(dict(items[idx : idx + batch_size]))


def _log_config_artifact(cfg: OmegaConf):
    if not _mlflow_run_active():
        return
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    temp_path = temp_file.name
    temp_file.close()
    OmegaConf.save(cfg, temp_path)
    try:
        mlflow.log_artifact(temp_path, artifact_path="configs")
    finally:
        os.remove(temp_path)


def _build_rec_trainer(cfg, *, save_path, dset_train, dset_val, device, writer):
    trainer_cfg = OmegaConf.select(cfg, "trainer.rec_trainer") or OmegaConf.select(cfg, "rec_trainer") or {}
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True) if trainer_cfg else {}
    trainer_name = trainer_cfg.pop(
        "name",
        OmegaConf.select(cfg, "trainer.name") or "trainers.rec_trainer.Train_Rec_Reg_Model",
    )
    TrainerCls = _resolve_class(trainer_name)

    params = {
        "cfg": cfg,
        "save_path": save_path,
        "non_improve_maxmum": trainer_cfg.pop("non_improve_maxmum", 1e10),
        "reg_loss_weight": trainer_cfg.pop("reg_loss_weight", 1000),
        "val_loss_min": trainer_cfg.pop("val_loss_min", 1e10),
        "val_dist_min": trainer_cfg.pop("val_dist_min", 1e10),
        "val_loss_min_reg": trainer_cfg.pop("val_loss_min_reg", 1e10),
        "dset_train": dset_train,
        "dset_val": dset_val,
        "dset_train_reg": trainer_cfg.pop("dset_train_reg", None),
        "dset_val_reg": trainer_cfg.pop("dset_val_reg", None),
        "device": device,
        "writer": writer,
        "option": trainer_cfg.pop("option", "common_volume"),
    }

    # Pass only accepted init args to avoid accidental config typos.
    signature = inspect.signature(TrainerCls.__init__)
    allowed = {k for k in signature.parameters if k != "self"}
    final_params = {k: v for k, v in {**params, **trainer_cfg}.items() if k in allowed}
    return TrainerCls(**final_params)

def main():
    # Expect --config passed via sys.argv; keep parity with main.py style
    if "--config" not in sys.argv:
        print("Usage: python main_rec.py --config <yaml> [key=value ...]")
        sys.exit(1)
    config_idx = sys.argv.index("--config") + 1
    config_path = sys.argv[config_idx]
    overrides = sys.argv[config_idx + 1 :] if len(sys.argv) > config_idx + 1 else []

    load_dotenv(".env", override=False)
    cfg = load_config(config_path, overrides)
    ctx = TrainingContext.from_cfg(cfg, root_dir="logs")
    ctx.save_config(cfg)
    runtime_cfg = OmegaConf.select(cfg, "runtime") or {}
    save_path = str(ctx.run_dir)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "saved_model"), exist_ok=True)

    gpu_ids = runtime_cfg.get("gpu_ids")
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(str(ctx.run_dir / "tb"))

    dset_train = build_dataset(cfg, split="train")
    dset_val = build_dataset(cfg, split="val")
    dset_test = build_dataset(cfg, split="test")

    train_rec_reg_model = _build_rec_trainer(
        cfg,
        save_path=save_path,
        dset_train=dset_train,
        dset_val=dset_val,
        device=device,
        writer=writer,
    )
    train_rec_reg_model.ctx = ctx

    mlflow_node = OmegaConf.select(cfg, "logging.mlflow")
    if mlflow_node is None and hasattr(cfg, "get"):
        mlflow_node = cfg.get("mlflow")
    if mlflow_node is None:
        mlflow_cfg = {}
    elif OmegaConf.is_config(mlflow_node):
        mlflow_cfg = OmegaConf.to_container(mlflow_node, resolve=True) or {}
    elif isinstance(mlflow_node, dict):
        mlflow_cfg = mlflow_node
    else:
        mlflow_cfg = {}
    trainer_cfg = cfg.get("trainer") or {}
    log_interval = int(trainer_cfg.get("log_interval", 50))
    logger_hook = LoggerHook(
        interval=log_interval,
        log_file=str(ctx.log_file),
        console=True,
        mlflow_enabled=bool(mlflow_cfg),
        upload_run_dir=bool(OmegaConf.select(cfg, "mlflow.archive_run_dir") or False),
        delete_local_run_dir=bool(OmegaConf.select(cfg, "mlflow.delete_local_run_dir") or False),
        artifact_path=str(OmegaConf.select(cfg, "mlflow.artifact_path") or "run"),
    )
    train_rec_reg_model.register_hook(logger_hook)
    if not mlflow_cfg:
        train_rec_reg_model.multi_model()
        train_rec_reg_model.train_rec_model()
        return

    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name") or "default"
    run_name = mlflow_cfg.get("run_name") or f"{experiment_name}-{int(time.time())}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=mlflow_cfg.get("tags")) as active_run:
        run_id = active_run.info.run_id
        train_rec_reg_model.ctx = ctx.with_mlflow_run_id(run_id)
        _log_params(_flatten_config(cfg))
        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        train_rec_reg_model.multi_model()
        train_rec_reg_model.train_rec_model()
        _log_config_artifact(cfg)


if __name__ == "__main__":
    main()
