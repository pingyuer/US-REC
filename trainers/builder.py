"""Optimizer builder factory."""

import torch
from typing import Any, Union

from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.optimizer import Optimizer
from omegaconf import DictConfig, OmegaConf


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
