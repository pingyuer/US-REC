"""Component builders for optimizers and training infrastructure."""

import importlib
import inspect
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.optimizer import Optimizer

from data.builder import build_dataset
from trainers.context import TrainingContext
from trainers.hooks import LoggerHook, MLflowHook, VizHook
from trainers.hooks.checkpoint_hook import CheckpointHook


def _vq_memory_collate(batch: list[dict]) -> dict:
    """Collate for VQMemoryScanWindowDataset with variable-length fields.

    - ``frames`` and ``gt_global_T`` are **truncated** to the min length in
      the batch (safe because all windows are ≥ min_tile_size).
    - ``scan_frames`` and ``scan_gt_global_T`` are **padded** to the max
      length with boolean masks (True = valid, False = padding).
    """
    if not batch:
        return {}

    result: dict = {}
    all_keys = list(batch[0].keys())

    # Keys to truncate (window-level — must all be the same length for loss)
    trunc_keys = {"frames", "gt_global_T"}
    # Keys to pad (scan-level — variable length is expected)
    pad_keys = {"scan_frames", "scan_gt_global_T"}

    _tensor_keys = [k for k in all_keys
                    if torch.is_tensor(batch[0][k]) and batch[0][k].ndim >= 1]

    for key in _tensor_keys:
        tensors = [s[key].contiguous() for s in batch]
        lengths = [t.shape[0] for t in tensors]

        if key in trunc_keys:
            # Truncate to shortest → no padding needed, no singular matrices
            min_len = min(lengths)
            result[key] = torch.stack([t[:min_len] for t in tensors])
        elif key in pad_keys:
            # Pad to longest + add mask
            max_len = max(lengths)
            padded = []
            mask = []
            for t, L in zip(tensors, lengths):
                if L < max_len:
                    pad_shape = (max_len - L,) + t.shape[1:]
                    t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                padded.append(t)
                m = torch.zeros(max_len, dtype=torch.bool)
                m[:L] = True
                mask.append(m)
            result[key] = torch.stack(padded)
            result[f"{key}_mask"] = torch.stack(mask)
        else:
            # Other tensors: try stack (same shape) or pad
            if all(L == lengths[0] for L in lengths):
                result[key] = torch.stack(tensors)
            else:
                max_len = max(lengths)
                padded = []
                for t, L in zip(tensors, lengths):
                    if L < max_len:
                        pad_shape = (max_len - L,) + t.shape[1:]
                        t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
                    padded.append(t)
                result[key] = torch.stack(padded)

    # Non-tensor keys
    for key in all_keys:
        if key in result:
            continue
        vals = [s[key] for s in batch]
        if isinstance(vals[0], (int, float)):
            result[key] = torch.tensor(vals)
        else:
            result[key] = vals

    return result


# Optimizer registry for easy extension
OPTIMIZER_REGISTRY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "rmsprop": RMSprop,
}


def build_optimizer(
    cfg: Union[DictConfig, dict], model: torch.nn.Module
) -> Optimizer:
    """
    Build a PyTorch optimizer from an OmegaConf configuration.

    Parameters
    ----------
    cfg : DictConfig or dict
        OmegaConf node, typically `cfg.optimizer`.
        Example YAML:
        optimizer:
          type: AdamW
          lr: 1e-4
          weight_decay: 1e-2
          params:
            - name: backbone
              lr_mult: 0.1
            - name: head
              lr_mult: 1.0
    model : nn.Module
        Model whose parameters to optimize.

    Returns
    -------
    torch.optim.Optimizer
    """
    # Ensure DictConfig type
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # === Parse basic parameters ===
    opt_type: str = str(cfg.get("type", "AdamW")).lower()
    if opt_type not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"[build_optimizer] Unsupported optimizer type '{opt_type}'. "
            f"Supported types: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    base_lr = float(cfg.get("lr", 1e-3))
    base_wd = float(cfg.get("weight_decay", 0.0))

    # === Parameter groups definition ===
    param_groups = []
    if OmegaConf.select(cfg, "params") is not None:
        for group_cfg in cfg.params:
            name = group_cfg.get("name", None)
            lr_mult = float(group_cfg.get("lr_mult", 1.0))
            wd_mult = float(group_cfg.get("wd_mult", 1.0))

            # Match parameters by name pattern
            matched_params = [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and (name is None or name in n)
            ]
            if not matched_params:
                print(f"[build_optimizer] No parameters matched group '{name}'.")
                continue

            param_groups.append(
                {
                    "params": matched_params,
                    "lr": base_lr * lr_mult,
                    "weight_decay": base_wd * wd_mult,
                }
            )
    else:
        # Default: single group with all trainable parameters
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    # === Optimizer-specific parameters ===
    kwargs = {"lr": base_lr, "weight_decay": base_wd}
    if opt_type == "sgd":
        kwargs.update(
            {
                "momentum": float(cfg.get("momentum", 0.9)),
                "nesterov": bool(cfg.get("nesterov", False)),
            }
        )
    elif opt_type in {"adam", "adamw"}:
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        kwargs.update(
            {
                "betas": betas,
                "eps": float(cfg.get("eps", 1e-8)),
            }
        )
    elif opt_type == "rmsprop":
        kwargs.update(
            {
                "alpha": float(cfg.get("alpha", 0.99)),
                "momentum": float(cfg.get("momentum", 0.9)),
            }
        )

    # === Construct optimizer ===
    optimizer_cls = OPTIMIZER_REGISTRY[opt_type]
    optimizer = optimizer_cls(param_groups, **kwargs)

    # === Log ===
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[build_optimizer] ✅ Built {opt_type.upper()} | "
        f"param_groups={len(param_groups)}, lr={base_lr}, wd={base_wd}, "
        f"trainable_params={total_params:,}"
    )

    return optimizer


def load_dotenv(path: str = ".env", *, override: bool = False) -> None:
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


def _resolve_class(name_or_obj: Any):
    if not isinstance(name_or_obj, str):
        return name_or_obj
    module, cls_name = name_or_obj.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, cls_name)


def build_datasets(cfg: Any) -> tuple[Any, Any, Any]:
    return (
        build_dataset(cfg, split="train"),
        build_dataset(cfg, split="val"),
        build_dataset(cfg, split="test"),
    )


def build_training_context(cfg: Any, *, root_dir: str | None = None) -> TrainingContext:
    root = root_dir or OmegaConf.select(cfg, "paths.output_dir") or "logs"
    return TrainingContext.from_cfg(cfg, root_dir=root)


def resolve_run_dirs(cfg: Any, ctx: TrainingContext) -> dict[str, Path]:
    paths_cfg = OmegaConf.select(cfg, "paths") or {}
    run_dir = Path(ctx.run_dir)
    log_dir = Path(paths_cfg.get("log_dir") or (run_dir / "logs"))
    ckpt_dir = Path(paths_cfg.get("ckpt_dir") or (run_dir / "saved_model"))
    cache_dir = Path(paths_cfg.get("cache_dir") or (run_dir / "cache"))
    for path in (log_dir, ckpt_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log_dir": log_dir,
        "ckpt_dir": ckpt_dir,
        "cache_dir": cache_dir,
    }


def _build_scan_window_loaders(
    cfg: Any,
    dset_train: Any,
    dset_val: Any,
) -> tuple[Any, Any]:
    """Wrap raw TUS-REC datasets into ScanWindowDataset → DataLoader for longseq."""
    from data.datasets.scan_window import ScanWindowDataset  # noqa: PLC0415
    from torch.utils.data import DataLoader

    ds_cfg = OmegaConf.select(cfg, "dataset") or {}
    window_size = int(ds_cfg.get("sequence_window", 128))
    windows_per_scan = int(ds_cfg.get("windows_per_scan", 1))
    sampling_mode = str(ds_cfg.get("sampling_mode", "tiled"))
    tile_overlap = int(ds_cfg.get("tile_overlap", 0))
    min_tile_size = int(ds_cfg.get("min_tile_size", 0))
    max_tiles_per_scan = int(ds_cfg.get("max_tiles_per_scan", 0))
    seed = int(OmegaConf.select(cfg, "seed") or 0)
    augment_flip = bool(ds_cfg.get("augment_flip", False))
    flip_prob = float(ds_cfg.get("flip_prob", 0.5))
    # augment.reverse_prob (preferred key) overrides the legacy augment_flip/flip_prob pair.
    # Example config:  dataset: { augment: { reverse_prob: 0.5 } }
    _reverse_prob = OmegaConf.select(cfg, "dataset.augment.reverse_prob")
    if _reverse_prob is not None:
        augment_flip = True
        flip_prob = float(_reverse_prob)

    dl_cfg = OmegaConf.select(cfg, "dataloader") or {}
    train_dl_cfg = OmegaConf.to_container(dl_cfg.get("train") or {}, resolve=True) if dl_cfg else {}
    val_dl_cfg = OmegaConf.to_container(dl_cfg.get("val") or {}, resolve=True) if dl_cfg else {}

    # ── Optional calibration ────────────────────────────────────────────────
    # Load tform_calib so the dataset can expose it per-sample for DDF loss /
    # official TUS-REC metric computation in the trainer's evaluate().
    tform_calib = None
    image_size: tuple[int, int] | None = None
    calib_file = ds_cfg.get("calib_file") or OmegaConf.select(cfg, "dataset.calib_file")
    if calib_file:
        try:
            from trainers.utils.calibration import load_calibration  # noqa: PLC0415
            resample_factor = float(ds_cfg.get("resample_factor", 1.0))
            _, _, tform_calib = load_calibration(calib_file, resample_factor, device="cpu")
            # tform_calib is returned as a torch.Tensor (4,4) or numpy; ensure tensor
            if not isinstance(tform_calib, torch.Tensor):
                import numpy as np  # noqa: PLC0415
                tform_calib = torch.as_tensor(np.array(tform_calib), dtype=torch.float32)
            tform_calib = tform_calib.float()
        except Exception as exc:
            import warnings
            warnings.warn(f"[builder] Could not load calibration from {calib_file!r}: {exc}")
            tform_calib = None
    # Image size from config (H, W)
    cfg_img_h = OmegaConf.select(cfg, "dataset.image_size_h") or OmegaConf.select(cfg, "model.image_size_h")
    cfg_img_w = OmegaConf.select(cfg, "dataset.image_size_w") or OmegaConf.select(cfg, "model.image_size_w")
    if cfg_img_h and cfg_img_w:
        image_size = (int(cfg_img_h), int(cfg_img_w))

    # ── Detect VQ memory mode → use VQMemoryScanWindowDataset ───────────────
    seq_mode = str(ds_cfg.get("sequence_mode", "scan_window")).lower()
    use_vq_memory_ds = seq_mode == "vq_memory"
    if use_vq_memory_ds:
        from data.datasets.vq_memory_window import VQMemoryScanWindowDataset  # noqa: PLC0415
        DatasetCls = VQMemoryScanWindowDataset
    else:
        DatasetCls = ScanWindowDataset

    # Common dataset kwargs
    _ds_kwargs = dict(
        window_size=window_size,
        windows_per_scan=windows_per_scan,
        sampling_mode=sampling_mode,
        tile_overlap=tile_overlap,
        min_tile_size=min_tile_size,
        max_tiles_per_scan=max_tiles_per_scan,
        seed=seed,
        augment_flip=augment_flip,
        flip_prob=flip_prob,
        calib_matrix=tform_calib,
        image_size=image_size,
    )
    if use_vq_memory_ds:
        _ds_kwargs["anchor_stride"] = int(ds_cfg.get("anchor_stride",
            OmegaConf.select(cfg, "vq.anchor_stride") or
            OmegaConf.select(cfg, "model.anchor_stride") or 8))

    # Custom collate for VQ memory mode: pads variable-length scan_frames
    collate_fn = _vq_memory_collate if use_vq_memory_ds else None

    train_loader = val_loader = None
    if dset_train is not None:
        sw_train = DatasetCls(base_dataset=dset_train, mode="train", **_ds_kwargs)
        persistent = bool(train_dl_cfg.get("persistent_workers", False))
        nw = int(train_dl_cfg.get("num_workers", 0))
        prefetch = int(train_dl_cfg.get("prefetch_factor", 2)) if nw > 0 else None
        train_loader = DataLoader(
            sw_train,
            batch_size=int(train_dl_cfg.get("batch_size", 1)),
            num_workers=nw,
            pin_memory=bool(train_dl_cfg.get("pin_memory", False)),
            persistent_workers=persistent and nw > 0,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )
    if dset_val is not None:
        val_kwargs = {k: v for k, v in _ds_kwargs.items()
                      if k not in ("augment_flip", "flip_prob")}
        val_kwargs["windows_per_scan"] = 1
        sw_val = DatasetCls(base_dataset=dset_val, mode="val", **val_kwargs)
        val_loader = DataLoader(
            sw_val,
            batch_size=int(val_dl_cfg.get("batch_size", 1)),
            num_workers=int(val_dl_cfg.get("num_workers", 0)),
            pin_memory=bool(val_dl_cfg.get("pin_memory", False)),
            collate_fn=collate_fn,
        )
    return train_loader, val_loader


def build_rec_trainer(
    cfg: Any,
    save_path: str,
    dset_train: Any,
    dset_val: Any,
    device: torch.device,
    writer: Any = None,
) -> Any:
    # Lazy import to avoid pulling in the full rec_trainer / monai chain at
    # package import time (breaks tests that only need trainers.metrics).
    from trainers.rec_trainer import Train_Rec_Reg_Model  # noqa: PLC0415
    trainer_cfg = OmegaConf.select(cfg, "trainer.rec_trainer") or OmegaConf.select(cfg, "rec_trainer") or {}
    trainer_cfg = OmegaConf.to_container(trainer_cfg, resolve=True) if trainer_cfg else {}
    trainer_name = trainer_cfg.pop(
        "name",
        OmegaConf.select(cfg, "trainer.name") or "trainers.rec_trainer.Train_Rec_Reg_Model",
    )
    TrainerCls = _resolve_class(trainer_name)

    # Detect longseq trainer: it expects train_loader/val_loader, not raw datasets.
    sig = inspect.signature(TrainerCls.__init__)
    uses_loaders = "train_loader" in sig.parameters

    if uses_loaders:
        train_loader, val_loader = _build_scan_window_loaders(cfg, dset_train, dset_val)
        params = {
            "cfg": cfg,
            "device": device,
            "train_loader": train_loader,
            "val_loader": val_loader,
        }
    else:
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

    allowed = {k for k in sig.parameters if k != "self"}
    final_params = {k: v for k, v in {**params, **trainer_cfg}.items() if k in allowed}
    return TrainerCls(**final_params)


def build_hooks(cfg: Any, ctx: TrainingContext, trainer: Any = None) -> Sequence[Any]:
    trainer_cfg = cfg.get("trainer") or {}
    log_interval = int(trainer_cfg.get("log_interval", 50))

    # Build MLflowHook first so its logger can be shared with VizHook.
    mlflow_hook = MLflowHook(cfg=cfg)

    hooks: list[Any] = [
        LoggerHook(
            interval=log_interval,
            log_file=str(ctx.log_file),
            console=True,
            mlflow_enabled=False,
            upload_run_dir=False,
            delete_local_run_dir=False,
            artifact_path="run",
        ),
        mlflow_hook,
    ]

    viz_cfg = OmegaConf.select(cfg, "viz") or {}
    viz_enabled = bool(viz_cfg.get("enabled", False))
    if viz_enabled:
        out_dir = viz_cfg.get("out_dir", None)
        hooks.append(
            VizHook(
                out_dir=out_dir,
                drift_curve=bool(viz_cfg.get("drift_curve", True)),
                pose_curve=bool(viz_cfg.get("pose_curve", True)),
                recon_slices=bool(viz_cfg.get("recon_slices", False)),
                save_png=bool(viz_cfg.get("save_png", True)),
                save_csv=bool(viz_cfg.get("save_csv", True)),
                trainer=trainer,
                logger=mlflow_hook.logger,
            )
        )

    # CheckpointHook — save best & last model checkpoints
    ckpt_cfg = OmegaConf.select(cfg, "checkpoint") or {}
    run_dir = str(getattr(ctx, "run_dir", None) or ".")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    hooks.append(
        CheckpointHook(
            interval=int(ckpt_cfg.get("save_interval", 0)),
            dirpath=ckpt_dir,
            metric_name=str(ckpt_cfg.get("best_metric", "val_loss")),
            mode=str(ckpt_cfg.get("best_mode", "min")),
            save_best=True,
            save_last=True,
            save_optimizer=bool(ckpt_cfg.get("save_optimizer", True)),
        )
    )

    return hooks


def _scan_id_from_sample(sample: Any, sample_index: int) -> str:
    meta = None
    if isinstance(sample, dict):
        meta = sample.get("meta")
    if isinstance(meta, dict):
        value = meta.get("scan_id") or meta.get("scan_name")
    elif isinstance(meta, list) and meta:
        first = meta[0]
        if isinstance(first, dict):
            value = first.get("scan_id") or first.get("scan_name")
        else:
            value = None
    else:
        value = None
    if isinstance(value, (list, tuple)):
        value = "/".join(str(v) for v in value)
    if value is None:
        value = f"sample_{sample_index}"
    return str(value)


class _SampleLimiter(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: Iterable,
        max_scans: Optional[int] = None,
        max_frames_per_scan: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.max_scans = max_scans
        self.max_frames_per_scan = max_frames_per_scan

    def __iter__(self):
        counts: dict[str, int] = {}
        total = 0
        for sample in iter(self.dataset):
            sid = _scan_id_from_sample(sample, total)
            if sid not in counts:
                if self.max_scans is not None and len(counts) >= self.max_scans:
                    break
                counts[sid] = 0
            if self.max_frames_per_scan is not None and counts[sid] >= self.max_frames_per_scan:
                continue
            counts[sid] += 1
            total += 1
            yield sample


def limit_dataset(
    dataset: Iterable,
    *,
    max_scans: Optional[int] = None,
    max_frames_per_scan: Optional[int] = None,
) -> Iterable:
    if max_scans is None and max_frames_per_scan is None:
        return dataset
    return _SampleLimiter(dataset, max_scans=max_scans, max_frames_per_scan=max_frames_per_scan)
