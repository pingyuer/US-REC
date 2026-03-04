"""Dual-path Transformer pose model: shared CNN encoder + Dense (Δ=1) & Sparse (Δ=k) branches.

Architecture::

    FrameEncoder (shared)
        ↓
    z_i tokens (B, T, D)
        ├── DenseTransformer → DenseHead → pred_local_T (Δ=1)
        └── SparseTransformer(z_{0,k,2k,...}) → SparseHead → pred_sparse_T (Δ=k)

The Dense branch predicts every adjacent pair T_{i-1←i}.
The Sparse branch operates on a sub-sampled sequence at stride *k* and
predicts T_{(m-1)k ← mk} among anchor frames.

No multi-interval auxiliary losses (auxΔ) — each branch has its own
dedicated supervision.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from models.temporal.early_cnn import FrameEncoder
from models.temporal.temporal_transformer import TemporalPoseTransformer
from models.pose_heads.pose_head import LocalPoseHead


class DualPoseModel(nn.Module):
    """Dual-path (Dense + Sparse) pose estimation model.

    Parameters
    ----------
    backbone : str
        CNN backbone for FrameEncoder.
    in_channels : int
        Input channels (1 = greyscale).
    token_dim : int
        Frame token / transformer d_model.
    k_stride : int
        Sparse branch stride (Δ=k).
    dense_n_heads, dense_n_layers, dense_dim_ff, dense_window : int
        Dense transformer hyper-parameters.
    sparse_n_heads, sparse_n_layers, sparse_dim_ff, sparse_window : int
        Sparse transformer hyper-parameters.  Use large window or 0 for full
        attention on the (short) sparse subsequence.
    dropout : float
        Shared dropout rate.
    rotation_rep : str
        "rot6d" recommended.
    pretrained_backbone : bool
        Use ImageNet pretrained weights.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_channels: int = 1,
        token_dim: int = 256,
        k_stride: int = 8,
        # Dense transformer
        dense_n_heads: int = 4,
        dense_n_layers: int = 4,
        dense_dim_ff: int = 1024,
        dense_window: int = 64,
        # Sparse transformer
        sparse_n_heads: int = 4,
        sparse_n_layers: int = 4,
        sparse_dim_ff: int = 1024,
        sparse_window: int = 256,
        # Common
        dropout: float = 0.1,
        rotation_rep: str = "rot6d",
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.k_stride = k_stride
        self.rotation_rep = rotation_rep

        # ── Shared encoder ──────────────────────────────────────────
        self.encoder = FrameEncoder(
            backbone=backbone,
            in_channels=in_channels,
            token_dim=token_dim,
            pretrained=pretrained_backbone,
        )

        # ── Dense branch (Δ=1) ─────────────────────────────────────
        self.dense_transformer = TemporalPoseTransformer(
            d_model=token_dim,
            n_heads=dense_n_heads,
            n_layers=dense_n_layers,
            dim_feedforward=dense_dim_ff,
            dropout=dropout,
            window_size=dense_window,
        )
        self.dense_head = LocalPoseHead(
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
        )

        # ── Sparse branch (Δ=k) ────────────────────────────────────
        # The sparse subsequence is much shorter (T/k frames), so we can
        # afford a larger (or full) attention window.
        self.sparse_transformer = TemporalPoseTransformer(
            d_model=token_dim,
            n_heads=sparse_n_heads,
            n_layers=sparse_n_layers,
            dim_feedforward=sparse_dim_ff,
            dropout=dropout,
            window_size=sparse_window,
        )
        self.sparse_head = LocalPoseHead(
            d_model=token_dim,
            d_hidden=token_dim,
            rotation_rep=rotation_rep,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        frames : (B, T, H, W) or (B, T, C, H, W)

        Returns
        -------
        dict with:
            "pred_local_T"  : (B, T, 4, 4)   Dense Δ=1 local transforms
            "pred_sparse_T" : (B, M, 4, 4)   Sparse Δ=k local transforms
                              among anchor frames (M = number of anchors)
            "anchor_indices": (M_a,)          LongTensor of anchor frame indices
            "tokens"        : (B, T, D)       raw frame tokens
            "dense_ctx"     : (B, T, D)       dense context tokens
            "sparse_ctx"    : (B, M, D)       sparse context tokens
        """
        # 1. Shared per-frame encoding
        tokens = self.encoder.encode_sequence(frames)  # (B, T, D)
        B, T, D = tokens.shape

        # 2. Dense branch — full sequence
        dense_ctx, _ = self.dense_transformer(tokens)  # (B, T, D)
        pred_local_T = self.dense_head(dense_ctx)      # (B, T, 4, 4)

        # 3. Sparse branch — sub-sampled at stride k
        k = self.k_stride
        anchor_indices = torch.arange(0, T, k, device=tokens.device)  # (M,)
        sparse_tokens = tokens[:, anchor_indices]                      # (B, M, D)
        sparse_ctx, _ = self.sparse_transformer(sparse_tokens)         # (B, M, D)
        pred_sparse_T = self.sparse_head(sparse_ctx)                   # (B, M, 4, 4)

        return {
            "pred_local_T": pred_local_T,
            "pred_sparse_T": pred_sparse_T,
            "anchor_indices": anchor_indices,
            "tokens": tokens,
            "dense_ctx": dense_ctx,
            "sparse_ctx": sparse_ctx,
        }
