"""Shared trainer utilities — single-source for EMA, LR schedules, config access.

Every trainer variant imports these instead of copy-pasting the same helpers.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf


# ─── Config access ───────────────────────────────────────────────────────────

def cfg_get(cfg, path: str, default=None):
    """Safe OmegaConf selector with default fallback.

    Equivalent to ``OmegaConf.select(cfg, path)`` but returns *default*
    when the key is missing **or** its value is ``None``.
    """
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


# ─── EMA (Exponential Moving Average) ───────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy updated as:
      ``shadow = decay * shadow + (1 - decay) * current``

    Usage::

        ema = EMA(model, decay=0.999)
        # … after each optimizer step:
        ema.update(model)
        # … during evaluation:
        backup = ema.apply(model)   # install EMA weights
        ...                         # run eval
        EMA.restore(model, backup)  # restore originals
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Install EMA weights; return backup of originals for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore original weights from a backup dict."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ─── LR schedule: linear warm-up + cosine annealing ─────────────────────────

def warmup_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> float:
    """Compute learning rate at *step* with linear warm-up → cosine decay."""
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ─── Calibration loading ────────────────────────────────────────────────────

def load_tform_calib(
    cfg,
    *,
    device: str | torch.device = "cpu",
    warn_prefix: str = "Trainer",
) -> Optional[torch.Tensor]:
    """Load ``tform_calib`` from ``dataset.calib_file`` in the config.

    Returns ``None`` when no calibration is configured or loading fails.
    All 5+ duplicated implementations now delegate here.
    """
    calib_file = cfg_get(cfg, "dataset.calib_file")
    if not calib_file:
        return None
    try:
        from trainers.utils.calibration import load_calibration  # noqa: PLC0415

        resample_factor = float(cfg_get(cfg, "dataset.resample_factor", 1.0))
        _, _, tform_calib = load_calibration(calib_file, resample_factor, device=device)
        if not isinstance(tform_calib, torch.Tensor):
            import numpy as np  # noqa: PLC0415
            tform_calib = torch.as_tensor(np.array(tform_calib), dtype=torch.float32)
        return tform_calib.float().to(device)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[{warn_prefix}] Could not load calibration from {calib_file!r}: {exc}"
        )
        return None


# ─── K-root stride resolution ───────────────────────────────────────────────

def resolve_kroot_stride(cfg, *, k: int) -> int:
    """Resolve the kroot stride *s* from config, with ``sqrt(k)`` fallback.

    Lookup order (first non-None wins):
        1. ``kroot.s``
        2. ``dataset.long_stride``
        3. ``round(sqrt(k))``

    For V2 configs that specify ``kroot.L_target_frames`` the caller
    should compute ``s = ceil(L_target / (k - 1))`` *before* invoking
    this helper (or pass the result as ``kroot.s``).
    """
    s_raw = cfg_get(cfg, "kroot.s") or cfg_get(cfg, "dataset.long_stride")
    if s_raw is not None and int(s_raw) > 0:
        return int(s_raw)
    return int(round(math.sqrt(k)))
