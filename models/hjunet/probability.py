from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _parse_ops(value: Iterable | None) -> list[str]:
    if not value:
        return []
    ops: list[str] = []
    for token in str(value).split("_"):
        token = token.strip().lower()
        if token == "mix":
            continue
        if token in {"add", "mul", "cat", "norm"}:
            ops.append(token)
    return ops


def _build_conv_block(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    if kind == "cbl_linear":
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
    if kind == "vgg_linear":
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class ProbabilitySampler(nn.Module):
    def __init__(
        self,
        in_channels,
        sample_channels,
        prior="cbl_linear",
        posterior="cbl_linear",
        use_layernorm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.prior_block = _build_conv_block(prior, in_channels, sample_channels)
        self.posterior_block = _build_conv_block(posterior, sample_channels, sample_channels)
        self.use_layernorm = use_layernorm
        self.sample_norm = norm_layer(sample_channels) if use_layernorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prior_block(x)
        x = self.posterior_block(x)
        if self.sample_norm is not None:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.sample_norm(x)
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class ProbabilityFusion(nn.Module):
    def __init__(
        self,
        target_channels,
        sample_channels,
        operations=None,
        use_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.sample_proj = nn.Conv2d(sample_channels, target_channels, kernel_size=1, bias=False)
        self.operations = operations or ["add"]
        self.cat_proj = None
        if any(op == "cat" for op in self.operations):
            self.cat_proj = nn.Conv2d(target_channels * 2, target_channels, kernel_size=1, bias=False)
        self.use_norm = use_norm
        self.norm = norm_layer(target_channels) if use_norm else None

    def forward(self, x: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        sample = F.interpolate(sample, size=x.shape[-2:], mode="bilinear", align_corners=False)
        sample = self.sample_proj(sample)
        out = x
        for op in self.operations:
            if op == "add":
                out = out + sample
            elif op == "mul":
                out = out * sample
            elif op == "cat" and self.cat_proj is not None:
                out = torch.cat([out, sample], dim=1)
                out = self.cat_proj(out)
        if self.norm is not None:
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2)
            out = self.norm(out)
            out = out.transpose(1, 2).view(B, C, H, W)
        return out


__all__ = ["ProbabilityFusion", "ProbabilitySampler", "_parse_ops"]
