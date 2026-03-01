"""Builder for evaluation components.

Constructs model, trainer (for transform/calibration state), evaluator
and data loader so callers only need to call ``evaluator.run()``.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data.builder import build_dataset
from trainers.builder import (
    _resolve_class,
    build_rec_trainer,
    build_training_context,
    limit_dataset,
    load_dotenv,
    resolve_run_dirs,
)
from trainers.rec_evaluator import RecEvaluator


def build_evaluator(cfg: Any) -> dict[str, Any]:
    """Build all components needed for evaluation.

    Returns
    -------
    dict with keys: trainer, evaluator, model, loader, device, cfg
    """
    load_dotenv(".env", override=False)

    ctx = build_training_context(cfg)
    dirs = resolve_run_dirs(cfg, ctx)
    save_path = str(dirs["run_dir"])
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "saved_model"), exist_ok=True)

    runtime_cfg = OmegaConf.select(cfg, "runtime") or {}
    gpu_ids = runtime_cfg.get("gpu_ids")
    if gpu_ids:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets — apply sampling limits
    max_scans = OmegaConf.select(cfg, "eval.max_scans")
    max_frames = OmegaConf.select(cfg, "data.max_frames_per_scan")

    dset_train = build_dataset(cfg, split="train")
    dset_val = build_dataset(cfg, split="val")
    dset_test = build_dataset(cfg, split="test")

    if max_scans is not None or max_frames is not None:
        dset_test = limit_dataset(
            dset_test, max_scans=max_scans, max_frames_per_scan=max_frames
        )

    writer = SummaryWriter(str(dirs["log_dir"] / "tb"))

    trainer = build_rec_trainer(
        cfg,
        save_path=save_path,
        dset_train=dset_train,
        dset_val=dset_val,
        device=device,
        writer=writer,
    )
    trainer.ctx = ctx

    # Build test loader
    from torch.utils.data import DataLoader, IterableDataset

    dl_cfg = OmegaConf.select(cfg, "dataloader.test") or {}
    batch_size = int(dl_cfg.get("batch_size", 1))
    num_workers = int(dl_cfg.get("num_workers", 0))

    if isinstance(dset_test, IterableDataset):
        loader = DataLoader(
            dset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
    else:
        loader = DataLoader(
            dset_test,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

    evaluator = RecEvaluator(device=device)

    return {
        "trainer": trainer,
        "evaluator": evaluator,
        "model": trainer.model,
        "loader": loader,
        "device": device,
        "cfg": cfg,
        "ctx": ctx,
    }
