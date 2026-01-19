from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch

from .registry import register_transform


@register_transform("FinalizeSegSample")
class FinalizeSegSample:
    """
    Normalize a sample dict into the canonical training schema:
      {"image": Tensor[C,H,W] float32, "mask": Tensor[H,W] long, "meta": dict}

    It also drops any extra keys to keep dataloader batches small.
    """

    def __init__(
        self,
        *,
        image_key: str = "image",
        mask_key: str = "mask",
        meta_key: str = "meta",
        keep_keys: Optional[Iterable[str]] = None,
        allow_missing_mask: bool = False,
    ):
        self.image_key = str(image_key)
        self.mask_key = str(mask_key)
        self.meta_key = str(meta_key)
        self.allow_missing_mask = bool(allow_missing_mask)
        self.keep_keys = set(keep_keys) if keep_keys is not None else None

    def __call__(self, sample: dict) -> dict:
        if not isinstance(sample, dict):
            raise TypeError(f"FinalizeSegSample expects dict but got {type(sample)}")

        if self.image_key not in sample:
            raise KeyError(f"FinalizeSegSample missing '{self.image_key}' in sample")

        image = sample[self.image_key]
        image_t = _to_image_tensor(image)

        if self.mask_key in sample:
            mask = sample[self.mask_key]
            mask_t = _to_mask_tensor(mask)
        else:
            if not self.allow_missing_mask:
                raise KeyError(f"FinalizeSegSample missing '{self.mask_key}' in sample")
            _, h, w = image_t.shape
            mask_t = torch.zeros((h, w), dtype=torch.long)

        meta = sample.get(self.meta_key)
        if not isinstance(meta, dict):
            meta = {} if meta is None else {"value": meta}

        out = {self.image_key: image_t, self.mask_key: mask_t, self.meta_key: meta}

        if self.keep_keys:
            for k in self.keep_keys:
                if k in sample and k not in out:
                    out[k] = sample[k]
        return out


def _to_image_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        t = value
        if t.ndim == 3:
            return t.contiguous().float()
        if t.ndim == 4 and t.shape[0] == 1:
            return t[0].contiguous().float()
        raise ValueError(f"Unexpected image tensor shape: {tuple(t.shape)}")

    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected image ndarray shape: {arr.shape}")
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    raise TypeError(f"Unsupported image type: {type(value)}")


def _to_mask_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        t = value
        if t.ndim == 2:
            return t.contiguous().long()
        if t.ndim == 3 and t.shape[0] == 1:
            return t[0].contiguous().long()
        raise ValueError(f"Unexpected mask tensor shape: {tuple(t.shape)}")

    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected mask ndarray shape: {arr.shape}")
        return torch.from_numpy(arr).contiguous().long()

    raise TypeError(f"Unsupported mask type: {type(value)}")

