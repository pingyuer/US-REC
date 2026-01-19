"""HJUNet-aware model registry and factory."""

from typing import Any, Dict
import torch.nn as nn

from .unet import UNet
from .unext import UNext
from .hjunet import HJUNet as HJUNetClass
from .losses import BCEDiceLoss, DiceCELoss, DiceLoss, LovaszHingeLoss

__all__ = ["build_model", "build_loss", "UNext", "UNet", "HJUNet"]


def _merge_model_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.copy()
    model_cfg.pop("name", None)
    return model_cfg


def _normalize_hjunet_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize legacy/nested HJUNet configs into HJUNet(**kwargs).

    Supports configs like:
      model:
        name: hjunet_xxx
        backbone: {embed_dims: [...], decoder_dims: [...]}
        probabilistic: {...}
        tf_decoder: {enabled: true, num_queries: 16, ...}
    """
    out: Dict[str, Any] = {}

    # Direct passthrough for already-flat kwargs.
    for key in (
        "num_classes",
        "in_channels",
        "img_size",
        "embed_dims",
        "decoder_dims",
        "depths",
        "drop_rate",
        "attn_drop_rate",
        "drop_path_rate",
        "decoder",
        "probabilistic",
    ):
        if key in params:
            out[key] = params[key]

    backbone = params.get("backbone") or {}
    if isinstance(backbone, dict):
        out.setdefault("embed_dims", backbone.get("embed_dims"))
        out.setdefault("decoder_dims", backbone.get("decoder_dims"))

    tf_decoder = params.get("tf_decoder") or {}
    if isinstance(tf_decoder, dict) and bool(tf_decoder.get("enabled", False)):
        decoder_cfg = {k: v for k, v in tf_decoder.items() if k != "enabled"}
        decoder_cfg["type"] = decoder_cfg.get("type", "tf_decoder")
        out["decoder"] = decoder_cfg

    # If probabilistic is present at top-level, keep it; otherwise map legacy key.
    if "probabilistic" not in out and isinstance(params.get("probabilistic"), dict):
        out["probabilistic"] = params["probabilistic"]

    # Handle legacy key.
    if "in_channels" not in out and "input_channels" in params:
        out["in_channels"] = params["input_channels"]

    # Drop unknown keys (shift/scconv/gates/etc.) for now.
    return {k: v for k, v in out.items() if v is not None}


def build_model(cfg: Dict[str, Any]):
    name = cfg.get("name", "hjunet")
    params = _merge_model_cfg(cfg)

    if name == "hjunet" or name.startswith("hjunet_"):
        return HJUNet(**_normalize_hjunet_params(params))
    if name == "unext":
        return UNext(**params)
    if name == "unet":
        return UNet(**params)

    raise ValueError(f"Unknown model: {name}")


def HJUNet(**params):
    # HJUNet wraps the three-stage UNext backbone and allows parameter overrides
    return HJUNetClass(**params)


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """Build loss function from config."""
    loss_type = cfg.get("type", "CrossEntropyLoss")
    params = {k: v for k, v in cfg.items() if k != "type"}
    
    if loss_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**params)
    elif loss_type == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss(**params)
    elif loss_type == "BCEDiceLoss":
        return BCEDiceLoss()
    elif loss_type == "LovaszHingeLoss":
        return LovaszHingeLoss()
    elif loss_type == "DiceLoss":
        return DiceLoss(**params)
    elif loss_type == "DiceCELoss":
        return DiceCELoss(**params)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
