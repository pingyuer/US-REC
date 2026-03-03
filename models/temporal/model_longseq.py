"""Long-sequence pose model combining Early-CNN + Temporal Transformer + Pose Heads.

This module assembles the full pipeline:
  1. FrameEncoder (per-frame CNN features)
  2. TemporalPoseTransformer (causal sliding-window attention)
  3. LocalPoseHead (Δ=1 local transforms)
  4. MultiIntervalHead (auxiliary transforms for Δ > 1)

The output is a dict with:
  - "pred_local_T": (B, T, 4, 4) — local transforms (frame 0 = I)
  - "pred_aux_T": dict[int, (B, T, 4, 4)] — auxiliary transforms per Δ
  - "tokens": (B, T, D) — raw frame tokens (for debugging / probing)
  - "ctx": (B, T, D) — transformer context tokens
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from models.temporal.early_cnn import FrameEncoder
from models.temporal.temporal_transformer import TemporalPoseTransformer
from models.pose_heads.pose_head import LocalPoseHead, MultiIntervalHead


class LongSeqPoseModel(nn.Module):
    """End-to-end long-sequence pose estimation model.

    Parameters
    ----------
    backbone : str
        CNN backbone name for FrameEncoder.
    in_channels : int
        Per-frame input channels (1 for greyscale).
    token_dim : int
        Frame token dimension (also transformer d_model).
    n_heads, n_layers, dim_feedforward : int
        Transformer hyperparameters.
    window_size : int
        Sliding-window size for causal attention.
    dropout : float
        Dropout rate for transformer.
    rotation_rep : str
        Rotation parameterization: "rot6d" | "quat" | "se3_expmap".
    aux_intervals : sequence of int
        Multi-interval Δ values for auxiliary supervision.
    share_aux_decoder : bool
        Whether auxiliary decoders share weights.
    pretrained_backbone : bool
        Use ImageNet pretrained backbone weights.
    memory_size : int
        Transformer-XL style memory tokens.  0 = disabled (default).  When > 0
        the model carries ``memory_size`` hidden states from the previous
        segment as context.  Pass ``memory`` to ``forward()`` and retrieve
        ``new_memory`` from the returned dict to chain segments.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        token_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        window_size: int = 64,
        dropout: float = 0.1,
        rotation_rep: str = "rot6d",
        aux_intervals: Sequence[int] = (2, 4, 8, 16),
        share_aux_decoder: bool = False,
        pretrained_backbone: bool = False,
        memory_size: int = 0,
    ):
        super().__init__()
        self.rotation_rep = rotation_rep
        self.aux_intervals = list(aux_intervals)
        self.memory_size = max(0, int(memory_size))

        # Stage 1: per-frame CNN encoder
        self.encoder = FrameEncoder(
            backbone=backbone,
            in_channels=in_channels,
            token_dim=token_dim,
            pretrained=pretrained_backbone,
        )

        # Stage 2: temporal transformer
        self.transformer = TemporalPoseTransformer(
            d_model=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            window_size=window_size,
            memory_size=memory_size,
        )

        # Stage 3: pose heads
        self.local_head = LocalPoseHead(
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
        )
        self.aux_head = MultiIntervalHead(
            intervals=aux_intervals,
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
            share_decoder=share_aux_decoder,
        ) if aux_intervals else None

    def forward(
        self,
        frames: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
        """
        Parameters
        ----------
        frames : (B, T, H, W) or (B, T, C, H, W)
            Sequence of US frames (greyscale assumed if 4D).
        memory : (B, M, D) or None
            Optional cached context from a previous segment
            (Transformer-XL recurrence).  Only used when
            ``memory_size > 0``.
        pos_offset : int
            Global start position of this segment for positional encoding.

        Returns
        -------
        dict with keys:
            "pred_local_T" : (B, T, 4, 4) local transforms
            "pred_aux_T"   : dict[Δ → (B, T, 4, 4)] auxiliary transforms
            "tokens"       : (B, T, D) raw frame tokens
            "ctx"          : (B, T, D) transformer context
            "memory"       : (B, M, D) or None — updated memory tokens
        """
        # 1. Per-frame encoding
        tokens = self.encoder.encode_sequence(frames)        # (B, T, D)

        # 2. Temporal context (returns ctx + optional new_memory)
        ctx, new_memory = self.transformer(tokens, memory=memory, pos_offset=pos_offset)

        # 3. Local pose prediction
        pred_local_T = self.local_head(ctx)                  # (B, T, 4, 4)

        # 4. Auxiliary multi-interval predictions
        pred_aux_T: dict[int, torch.Tensor] = {}
        if self.aux_head is not None:
            pred_aux_T = self.aux_head(ctx)

        return {
            "pred_local_T": pred_local_T,
            "pred_aux_T": pred_aux_T,
            "tokens": tokens,
            "ctx": ctx,
            "memory": new_memory,
        }
