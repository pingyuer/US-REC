"""Early-CNN frame encoder for long-sequence pose estimation.

Extracts a per-frame feature token from a single ultrasound frame using
a lightweight EfficientNet backbone (default: efficientnet_b0).

The encoder is used *per-frame* (not per-pair), so the first conv layer
accepts 1-channel (grayscale) input by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
)


class FrameEncoder(nn.Module):
    """Extract a D-dimensional token from a single US frame.

    Parameters
    ----------
    backbone : str
        ``"efficientnet_b0"`` (D=1280) or ``"efficientnet_b1"`` (D=1280).
    in_channels : int
        Number of input channels (1 for greyscale US).
    token_dim : int or None
        If given, project the backbone feature to this dimension.
        If ``None``, output the raw backbone feature (1280 for EB0/EB1).
    pretrained : bool
        Whether to use ImageNet pre-trained weights.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        token_dim: int | None = 256,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = efficientnet_b0(weights=weights)
        elif backbone == "efficientnet_b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            net = efficientnet_b1(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Patch first conv for single-channel input
        first_conv = net.features[0][0]
        if first_conv.in_channels != in_channels:
            net.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

        self.features = net.features       # conv backbone
        self.avgpool = net.avgpool          # adaptive avgpool
        self.backbone_dim = net.classifier[1].in_features  # 1280

        # Optional projection to token_dim
        if token_dim is not None and token_dim != self.backbone_dim:
            self.proj = nn.Linear(self.backbone_dim, token_dim)
            self.out_dim = token_dim
        else:
            self.proj = None
            self.out_dim = self.backbone_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) single-frame tensor

        Returns
        -------
        token : (B, D)
        """
        feat = self.features(x)             # (B, backbone_dim, h, w)
        feat = self.avgpool(feat)           # (B, backbone_dim, 1, 1)
        feat = feat.flatten(1)              # (B, backbone_dim)
        if self.proj is not None:
            feat = self.proj(feat)          # (B, token_dim)
        return feat

    def encode_sequence(
        self,
        frames: torch.Tensor,
        chunk_size: int = 16,
    ) -> torch.Tensor:
        """Encode a sequence of frames independently, in memory-efficient chunks.

        Instead of passing all T frames through the CNN backbone at once
        (which stores O(T) activation maps for backprop), this processes
        ``chunk_size`` frames at a time and uses **gradient checkpointing**
        so that only one chunk's worth of intermediate activations is live
        at any moment.

        Parameters
        ----------
        frames : (B, T, C, H, W) or (B, T, H, W) — greyscale assumed if 4D
        chunk_size : int
            Number of frames to encode per micro-batch.  Lower values save
            more memory at the cost of an extra recomputation during backward.

        Returns
        -------
        tokens : (B, T, D)
        """
        if frames.ndim == 4:
            # (B, T, H, W) → (B, T, 1, H, W)
            frames = frames.unsqueeze(2)
        B, T, C, H, W = frames.shape

        token_chunks: list[torch.Tensor] = []
        for t0 in range(0, T, chunk_size):
            chunk = frames[:, t0 : t0 + chunk_size]          # (B, chunk, C, H, W)
            flat = chunk.reshape(-1, C, H, W)                # (B*chunk, C, H, W)
            if self.training:
                tok = torch.utils.checkpoint.checkpoint(
                    self.forward, flat, use_reentrant=False,
                )
            else:
                tok = self.forward(flat)                      # (B*chunk, D)
            token_chunks.append(tok.reshape(B, -1, tok.size(-1)))

        return torch.cat(token_chunks, dim=1)                 # (B, T, D)
