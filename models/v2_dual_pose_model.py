"""V2 Long-Sequence Pose Model — branch-aware encoder + V2 transformer + pose heads.

Extends :class:`LongSeqPoseModel` by swapping the temporal transformer
for :class:`V2TemporalPoseTransformer` which supports:

* **Short branch**: RoPE + sliding-window causal attention.
* **Long  branch**: Real-index sinusoidal PE + global/dilated attention.

The CNN encoder and pose heads are identical to the V1 model, so V1
checkpoints can seed the encoder and head weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from models.temporal.early_cnn import FrameEncoder
from models.temporal.v2_transformer import V2TemporalPoseTransformer
from models.pose_heads.pose_head import LocalPoseHead, MultiIntervalHead


class V2LongSeqPoseModel(nn.Module):
    """V2 end-to-end long-sequence pose model with branch-specific PE/attention.

    Parameters
    ----------
    backbone : str
        CNN backbone for :class:`FrameEncoder`.
    in_channels : int
        Input channels (1 = greyscale US).
    token_dim : int
        Frame token dimension (= transformer d_model).
    n_heads, n_layers, dim_feedforward : int
        Transformer hyper-parameters.
    dropout : float
    branch_mode : str
        ``"short"`` → RoPE + sliding-window causal.
        ``"long"``  → real-index sinusoidal + global/dilated.
    window_size : int
        Attention window size.
    dilation : int
        Dilation for long-branch dilated attention (1 = global).
    attention_mode : str or None
        Override auto-detected attention mode.
    rotation_rep : str
        ``"rot6d"`` | ``"quat"`` | ``"se3_expmap"``.
    aux_intervals : sequence of int
        Multi-interval Δ values for auxiliary supervision.
    share_aux_decoder : bool
    pretrained_backbone : bool
    max_len : int
        Maximum position for PE tables.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        token_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        branch_mode: str = "short",
        window_size: int = 64,
        dilation: int = 1,
        attention_mode: Optional[str] = None,
        rotation_rep: str = "rot6d",
        aux_intervals: Sequence[int] = (),
        share_aux_decoder: bool = False,
        pretrained_backbone: bool = False,
        max_len: int = 8192,
    ):
        super().__init__()
        self.rotation_rep = rotation_rep
        self.aux_intervals = list(aux_intervals)
        self.branch_mode = branch_mode

        # Stage 1: per-frame CNN encoder (shared architecture across branches)
        self.encoder = FrameEncoder(
            backbone=backbone,
            in_channels=in_channels,
            token_dim=token_dim,
            pretrained=pretrained_backbone,
        )

        # Stage 2: V2 temporal transformer (branch-aware)
        self.transformer = V2TemporalPoseTransformer(
            d_model=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            branch_mode=branch_mode,
            window_size=window_size,
            dilation=dilation,
            attention_mode=attention_mode,
            max_len=max_len,
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
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
        """
        Parameters
        ----------
        frames : (B, T, H, W) or (B, T, C, H, W)
        position_ids : (B, T) LongTensor — real frame indices (for long branch PE).
            For the short branch these can be ``None`` (defaults to 0..T-1).

        Returns
        -------
        dict:
            "pred_local_T" : (B, T, 4, 4)
            "pred_aux_T"   : dict[Δ → (B, T, 4, 4)]
            "tokens"       : (B, T, D) raw frame tokens
            "ctx"          : (B, T, D) transformer context
        """
        # 1. Per-frame encoding
        tokens = self.encoder.encode_sequence(frames)  # (B, T, D)

        # 2. V2 temporal context
        ctx = self.transformer(tokens, position_ids=position_ids)  # (B, T, D)

        # 3. Pose prediction
        pred_local_T = self.local_head(ctx)  # (B, T, 4, 4)

        pred_aux_T: dict[int, torch.Tensor] = {}
        if self.aux_head is not None:
            pred_aux_T = self.aux_head(ctx)

        return {
            "pred_local_T": pred_local_T,
            "pred_aux_T": pred_aux_T,
            "tokens": tokens,
            "ctx": ctx,
        }
