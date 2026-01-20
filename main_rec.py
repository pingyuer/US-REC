import importlib
import inspect
import os
import sys
from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data.builder import build_dataset
from trainers.context import TrainingContext
from trainers.hooks import LoggerHook, MLflowHook
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
    trainer_cfg = cfg.get("trainer") or {}
    log_interval = int(trainer_cfg.get("log_interval", 50))
    logger_hook = LoggerHook(
        interval=log_interval,
        log_file=str(ctx.log_file),
        console=True,
        mlflow_enabled=False,
        upload_run_dir=False,
        delete_local_run_dir=False,
        artifact_path="run",
    )
    train_rec_reg_model.register_hook(logger_hook)
    train_rec_reg_model.register_hook(MLflowHook(cfg=cfg))
    train_rec_reg_model.multi_model()
    train_rec_reg_model.train_rec_model()


if __name__ == "__main__":
    main()
