from .transforms.registry import get_transform_cls
from torchvision.transforms import Compose
import importlib
from omegaconf import DictConfig, OmegaConf
import inspect
from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate


def _resolve_class(name_or_obj):
    """Resolve a class from a string import path or return the class directly."""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    module, cls_name = name_or_obj.rsplit('.', 1)
    mod = importlib.import_module(module)
    return getattr(mod, cls_name)


def default_collate_fn(batch):
    """
    Custom collate function that handles variable-sized images and masks by padding.
    Used for segmentation datasets where image/mask sizes may vary.
    """
    if not batch:
        return batch
    if not isinstance(batch[0], dict) or "image" not in batch[0] or "mask" not in batch[0]:
        return default_collate(batch)

    # Find max dimensions across batch
    max_h = max(item['image'].shape[1] for item in batch)
    max_w = max(item['image'].shape[2] for item in batch)
    
    padded_images = []
    padded_masks = []
    
    for item in batch:
        img = item['image']  # Shape: [C, H, W]
        mask = item['mask']  # Shape: [H, W]
        
        img_h, img_w = img.shape[1], img.shape[2]
        mask_h, mask_w = mask.shape[0], mask.shape[1]
        
        # Pad image if needed
        if img_h < max_h or img_w < max_w:
            pad_img = torch.nn.functional.pad(
                img,
                (0, max_w - img_w, 0, max_h - img_h),
                mode='constant',
                value=0
            )
        else:
            pad_img = img
        padded_images.append(pad_img)
        
        # Pad mask if needed
        if mask_h < max_h or mask_w < max_w:
            pad_mask = torch.nn.functional.pad(
                mask.unsqueeze(0),
                (0, max_w - mask_w, 0, max_h - mask_h),
                mode='constant',
                value=0
            ).squeeze(0)
        else:
            pad_mask = mask
        padded_masks.append(pad_mask)
    
    # Stack all
    images = torch.stack(padded_images)
    masks = torch.stack(padded_masks)
    
    # Ensure masks are long type for CrossEntropyLoss
    if masks.dtype != torch.long:
        masks = masks.long()
    
    # Keep metadata
    result = {
        'image': images,
        'mask': masks,
    }
    
    # Add any other keys (like 'meta')
    if 'meta' in batch[0]:
        result['meta'] = [item['meta'] for item in batch]
    
    return result

def build_dataset(cfg: DictConfig, split: str = "train"):
    """
    Build a dataset instance from OmegaConf config.
    The dataset section is expected to be like:
        dataset:
          name: mypkg.datasets.voc.VOCDataset
          img_dir: {train: /path/train, val: /path/val}
          ann_file: {train: /path/train.json, val: /path/val.json}
          transforms: {train: [...], val: [...]}
          batch_size: {train: 16, val: 8}
    """
    # 1. Build pipeline first
    pipeline = build_pipeline(cfg, split=split)
    ds_cfg = cfg.dataset
    DatasetCls = _resolve_class(ds_cfg.name)

    # 2. Pick split-specific values (if DictConfig with per-split mapping)
    kwargs = {}
    for k, v in ds_cfg.items():
        if k == "name":
            continue
        if k == "sampling" and isinstance(v, DictConfig):
            for sk, sv in v.items():
                if isinstance(sv, DictConfig) and split in sv:
                    kwargs[sk] = sv[split]
                else:
                    kwargs[sk] = sv
            continue
        if isinstance(v, DictConfig) and split in v:
            kwargs[k] = v[split]
        else:
            kwargs[k] = v

    # 3. Automatically inject split/mode if the constructor requires it
    params = inspect.signature(DatasetCls.__init__).parameters
    if "split" in params and "split" not in kwargs:
        kwargs["split"] = split
    if "mode" in params and "mode" not in kwargs:
        kwargs["mode"] = split

    # 4. Add pipeline to kwargs
    if "pipeline" in params:
        kwargs["pipeline"] = pipeline

    # 5. Instantiate dataset
    dataset = DatasetCls(**kwargs)
    return dataset


def build_dataloader(
    dataset,
    cfg: DictConfig,
    split: str = "train",
    collate_fn: Optional[Callable] = None,
    sampler: Optional[torch.utils.data.Sampler] = None,
):
    """
    Build a DataLoader given a dataset and config section.

    Args:
        dataset: A torch.utils.data.Dataset instance
        cfg: OmegaConf config containing dataloader parameters
        split: "train" | "val" | "test"
        collate_fn: Optional custom collate function (defaults to default_collate_fn for padding)
        sampler: Optional sampler (overrides internal sampler)
    Returns:
        torch.utils.data.DataLoader
    """

    # ---- 1. Get dataloader settings from cfg ----
    dl_cfg = cfg.get("dataloader", {})
    split_cfg = dl_cfg.get(split, {}) if split in dl_cfg else {}

    batch_size = split_cfg.get("batch_size", 1)
    num_workers = split_cfg.get("num_workers", 2)
    shuffle = split_cfg.get("shuffle", split == "train")
    pin_memory = split_cfg.get("pin_memory", True)
    drop_last = split_cfg.get("drop_last", split == "train")

    # Use default collate_fn if not provided
    if collate_fn is None:
        collate_fn = default_collate_fn

    # ---- 2. If distributed, build DistributedSampler ----
    if sampler is None and torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # disable shuffle when sampler is used

    # ---- 3. Optional reproducibility: seed workers ----
    def seed_worker(worker_id):
        import numpy as np
        worker_seed = (torch.initial_seed() + worker_id) % 2**32
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # ---- 4. Build DataLoader ----
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
    )
    return loader

def build_pipeline(cfg: DictConfig, split: str = "train"):
    """
    Build a transform pipeline from config using the global registry.
    Supports custom transforms, torchvision.transforms.v2, and Albumentations.
    """
    if "pipeline" not in cfg:
        return None

    pipe_cfg = cfg.pipeline
    lib_type = pipe_cfg.get("type", "registry")  # default: use registry-based lookup
    transforms_cfg = pipe_cfg.get(split, [])

    transforms_list = []
    for t_cfg in transforms_cfg:
        t_type = t_cfg["type"]
        params = {k: v for k, v in t_cfg.items() if k != "type"}

        # Step 1: If albumentations type is specified, try that first
        if lib_type.lower() == "albumentations":
            import albumentations as A
            from albumentations.pytorch import ToTensorV2

            if hasattr(A, t_type):
                TransformCls = getattr(A, t_type)
                transforms_list.append(TransformCls(**params))
                continue
            elif t_type == "ToTensorV2":
                transforms_list.append(ToTensorV2(**params))
                continue
        
        # Step 2: Try custom registry / torchvision.v2
        try:
            TransformCls = get_transform_cls(t_type)
            transforms_list.append(TransformCls(**params))
            continue
        except KeyError:
            pass

        # Step 3: Fallback to Albumentations if not found yet
        if lib_type.lower() == "albumentations":
            from albumentations.pytorch import ToTensorV2

            if t_type == "ToTensorV2":
                transforms_list.append(ToTensorV2(**params))
            else:
                raise KeyError(f"Transform {t_type} not found in Albumentations.")
        else:
            raise KeyError(f"Transform {t_type} not found in registry or Albumentations.")

    # Step 3: Compose pipeline
    if lib_type.lower() == "albumentations":
        import albumentations as A

        composed = A.Compose(transforms_list)
    else:
        # torchvision v2 transforms are callable sequences
        from torchvision.transforms.v2 import Compose
        composed = Compose(transforms_list)

    return composed
