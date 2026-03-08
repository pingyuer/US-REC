"""VQ Tokenizer Head — EMA-updated vector quantisation for anchor features.

Converts continuous anchor features to discrete codebook entries.
Uses Exponential Moving Average (EMA) to update codebook, avoiding
backpropagation through the codebook itself (memory & stability).

Straight-through estimator passes gradients to the encoder projection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQTokenizerHead(nn.Module):
    """Vector Quantisation with EMA codebook updates.

    Parameters
    ----------
    code_dim : int
        Dimensionality of each codebook vector.
    codebook_size : int
        Number of entries in the codebook (K).
    ema_decay : float
        Exponential moving average decay for codebook updates.
    commitment_weight : float
        Weight for the commitment loss ||z_e - sg(z_q)||^2.
    epsilon : float
        Small constant for Laplace smoothing in EMA updates.
    """

    def __init__(
        self,
        code_dim: int = 256,
        codebook_size: int = 512,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.epsilon = epsilon

        # Codebook embeddings (not optimised by gradient — updated via EMA)
        self.register_buffer("embeddings", torch.randn(codebook_size, code_dim))
        nn.init.xavier_uniform_(self.embeddings.unsqueeze(0))
        self.embeddings = self.embeddings.squeeze(0)  # (K, D)

        # EMA tracking buffers
        self.register_buffer("ema_count", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", self.embeddings.clone())

        # Track codebook usage (for logging)
        self.register_buffer("_usage_counts", torch.zeros(codebook_size, dtype=torch.long))

        # Dead code revival: if smoothed count < threshold, replace embedding
        self._dead_threshold = 1.0

    def reset_usage(self) -> None:
        """Reset codebook usage tracking (call at epoch start)."""
        self._usage_counts.zero_()

    @property
    def codebook_usage(self) -> float:
        """Fraction of codebook entries used since last reset."""
        return float((self._usage_counts > 0).float().mean().item())

    def _quantise(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest codebook entry for each input vector.

        Parameters
        ----------
        z_e : (N, D) continuous encoder output

        Returns
        -------
        z_q : (N, D) quantised vectors
        indices : (N,) codebook indices
        """
        # Pairwise L2 distance: ||z_e - e||^2 = ||z_e||^2 - 2 <z_e, e> + ||e||^2
        d = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2 * z_e @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=-1, keepdim=True).t()
        )  # (N, K)
        indices = d.argmin(dim=-1)  # (N,)
        z_q = self.embeddings[indices]  # (N, D)
        return z_q, indices

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA codebook update (called only during training)."""
        # One-hot assignment
        onehot = F.one_hot(indices, self.codebook_size).float()  # (N, K)

        # Update usage counts
        counts = onehot.sum(dim=0)  # (K,)
        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

        # Update weight sums
        dw = onehot.t() @ z_e  # (K, D)
        self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Laplace smoothing — avoids dead codes collapsing
        n = self.ema_count.sum()
        smoothed = (
            (self.ema_count + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Update embeddings
        self.embeddings.data.copy_(self.ema_weight / smoothed.unsqueeze(1))

        # ── Dead code revival ────────────────────────────────────────
        # If a code's smoothed count is below threshold, replace it with
        # a randomly chosen encoder output (+ small jitter) to break
        # symmetry and revive dead entries.
        dead_mask = smoothed < self._dead_threshold
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            # Sample replacement vectors from current batch
            rand_idx = torch.randint(0, z_e.size(0), (n_dead,), device=z_e.device)
            replacement = z_e[rand_idx].clone()
            replacement += torch.randn_like(replacement) * 1e-3  # jitter
            self.embeddings.data[dead_mask] = replacement
            self.ema_weight[dead_mask] = replacement
            self.ema_count[dead_mask] = smoothed.mean()  # reset to avg count

        # Track usage for logging
        self._usage_counts += (counts > 0).long()

    def forward(
        self, z_e: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Quantise input features and compute commitment loss.

        Parameters
        ----------
        z_e : (N, D) or (B, M, D) continuous features from proj_vq

        Returns
        -------
        dict with:
            z_q          : (same shape as z_e) quantised vectors (straight-through)
            indices      : (N,) or (B, M) codebook indices
            commit_loss  : scalar commitment loss
            z_q_detached : (same shape) quantised vectors detached from encoder grad
        """
        input_shape = z_e.shape
        flat = z_e.reshape(-1, self.code_dim)  # (N, D)

        z_q, indices = self._quantise(flat)

        # Commitment loss: push encoder output toward codebook entries
        commit_loss = F.mse_loss(flat, z_q.detach())

        # EMA update (training only, no grad)
        if self.training:
            self._ema_update(flat.detach(), indices)

        # Straight-through estimator: gradient flows to encoder via z_e
        z_q_st = flat + (z_q - flat).detach()

        # Reshape back
        z_q_st = z_q_st.reshape(input_shape)
        z_q_detached = z_q.detach().reshape(input_shape)
        indices = indices.reshape(input_shape[:-1])

        return {
            "z_q": z_q_st,
            "indices": indices,
            "commit_loss": commit_loss * self.commitment_weight,
            "z_q_detached": z_q_detached,
        }
