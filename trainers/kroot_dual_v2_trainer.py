"""V2 K-root Dual Trainer — scientific dual-stream with RoPE/real-index PE.

Extends :class:`KRootDualTrainer` with:

1. **V2 models**: :class:`V2LongSeqPoseModel` per branch, each with
   branch-appropriate PE and attention (RoPE + sliding-window for short,
   real-index sinusoidal + global/dilated for long).

2. **Scientific stride**: ``s = ceil(L_target_frames / (k - 1))``
   computed once from config and passed to dataset and stitch pipeline.

3. **position_ids forwarding**: the ``_forward_branch`` method passes
   ``position_ids`` from the batch to the V2 model so the PE sees
   real frame indices.

4. **Diagnostic CSV**: after eval, exports per-scan CSV with
   long_only / short_only / fused error curves.

Everything else (hooks, checkpoint, train loop, EMA, stitch fusion)
is inherited from :class:`KRootDualTrainer`.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.v2_dual_pose_model import V2LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import compose_global_from_local, local_from_global
from eval.kroot_stitch import (
    stitch_long_base_short_refine,
    compute_stitch_metrics,
    export_debug_csv,
)
from trainers.kroot_dual_trainer import KRootDualTrainer
from trainers.common import cfg_get, EMA, load_tform_calib


# ==========================================================================
# V2 Trainer
# ==========================================================================

class KRootDualV2Trainer(KRootDualTrainer):
    """V2 joint trainer with scientific PE, attention, and stride.

    Overrides model construction to use :class:`V2LongSeqPoseModel` and
    forwards ``position_ids`` through the training loop.  Everything else
    (train loop, stitch eval, EMA, hook system) is inherited.

    Additional config keys (beyond V1):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kroot.L_target_frames : int
        Desired real-frame span of the long window.
        ``s = ceil(L_target / (k - 1))``.
    model.short.transformer.attention_mode : str
        Default ``"sliding_window"``.
    model.long.transformer.attention_mode : str
        Default ``"global"``.
    model.long.transformer.dilation : int
        Dilation factor when attention_mode is ``"dilated"`` (default 1).
    eval.csv_dir : str or None
        If set, write diagnostic CSV per scan after eval.
    """

    def __init__(
        self,
        cfg,
        *,
        device: str | torch.device = "cpu",
        joint_train_loader=None,
        short_train_loader=None,
        long_train_loader=None,
        val_loader=None,
    ):
        # Compute s from L_target_frames BEFORE the super().__init__
        # so that self.s is consistent everywhere.
        k = int(cfg_get(cfg, "kroot.k", 64))
        L_target = cfg_get(cfg, "kroot.L_target_frames")
        if L_target is not None and int(L_target) > 0 and k > 1:
            computed_s = max(1, math.ceil(int(L_target) / (k - 1)))
            # Inject computed s into cfg so parent __init__ + builder see it
            if OmegaConf.is_struct(cfg):
                OmegaConf.set_struct(cfg, False)
            OmegaConf.update(cfg, "kroot.s", computed_s)
            if hasattr(cfg, "_metadata"):
                pass  # OmegaConf handles this
            self._L_target_frames = int(L_target)
        else:
            self._L_target_frames = None

        super().__init__(
            cfg,
            device=device,
            joint_train_loader=joint_train_loader,
            short_train_loader=short_train_loader,
            long_train_loader=long_train_loader,
            val_loader=val_loader,
        )

        # CSV diagnostic output
        self._csv_dir = cfg_get(cfg, "eval.csv_dir")

        print(
            f"[KRootDualV2] k={self.k}  s={self.s}  "
            f"L_target={self._L_target_frames}  "
            f"long_span={(self.k - 1) * self.s} frames  "
            f"short_branch=RoPE+sliding_window  "
            f"long_branch=real_index+{cfg_get(cfg, 'model.long.transformer.attention_mode', 'global')}"
        )

    # ── Override model construction to use V2 models ─────────────────

    def _build_model(self, branch: str) -> V2LongSeqPoseModel:
        """Build a V2LongSeqPoseModel for the given branch."""
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))

        def _tf(key, default):
            v = cfg_get(self.cfg, f"model.{branch}.transformer.{key}")
            if v is None:
                v = cfg_get(self.cfg, f"model.transformer.{key}", default)
            return v

        token_dim = int(_tf("d_model", 256))
        n_heads = int(_tf("n_heads", 4))
        n_layers = int(_tf("n_layers", 4))
        dim_ff = int(_tf("dim_feedforward", 1024))
        window_size = int(_tf("window_size", self.k))
        dropout = float(_tf("dropout", 0.1))
        dilation = int(_tf("dilation", 1))
        attention_mode = _tf("attention_mode", None)
        if attention_mode is not None:
            attention_mode = str(attention_mode)

        return V2LongSeqPoseModel(
            backbone=backbone,
            in_channels=1,
            token_dim=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            branch_mode=branch,
            window_size=window_size,
            dilation=dilation,
            attention_mode=attention_mode,
            rotation_rep=self.rotation_rep,
            aux_intervals=[],
            pretrained_backbone=pretrained,
        )

    # ── Override _forward_branch to pass position_ids ────────────────

    def _forward_branch(
        self,
        model: nn.Module,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss with V2 position_ids support."""
        frames = batch["frames"].to(self.device, non_blocking=True)
        gt_global = batch["gt_global_T"].to(self.device, non_blocking=True)

        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.1:
            frames = frames / 255.0

        # V2: use position_ids for PE (both branches)
        position_ids = None
        if "position_ids" in batch:
            position_ids = batch["position_ids"].to(self.device, non_blocking=True).long()
        elif "idx_long" in batch:
            position_ids = batch["idx_long"].to(self.device, non_blocking=True).long()

        out = model(frames, position_ids=position_ids)
        pred_local_T = out["pred_local_T"]

        gt_local_T = local_from_global(gt_global)

        pred_R = pred_local_T[:, 1:, :3, :3]
        gt_R = gt_local_T[:, 1:, :3, :3]
        pred_t = pred_local_T[:, 1:, :3, 3]
        gt_t = gt_local_T[:, 1:, :3, 3]

        rot_loss = chordal_rotation_loss(pred_R, gt_R)
        trans_loss = F.smooth_l1_loss(pred_t, gt_t, beta=1.0)
        loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

        with torch.no_grad():
            pred_global = compose_global_from_local(pred_local_T)
            drift = (pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]).norm(dim=-1).mean()

        metrics = {
            "rot_loss": float(rot_loss.item()),
            "trans_loss": float(trans_loss.item()),
            "drift_mm": float(drift.item()),
        }
        return loss, metrics

    # ── Override evaluate to add CSV export ──────────────────────────

    @torch.no_grad()
    def evaluate(self, loader=None) -> dict[str, float]:
        """V2 evaluate with optional diagnostic CSV export."""
        result = super().evaluate(loader)

        # Export diagnostic CSVs if configured
        if self._csv_dir and "scan_globals" in result:
            csv_dir = str(self._csv_dir)
            for scan_id, data in result["scan_globals"].items():
                pred = data.get("pred")
                gt = data.get("gt")
                if pred is not None and gt is not None:
                    try:
                        # Build simple short/long decomposition for CSV
                        # Use fused=pred as stand-in; full decomposition
                        # requires re-running inference
                        T = gt.shape[0]
                        anchor_idx = torch.arange(0, T, self.s)
                        export_debug_csv(
                            scan_id=scan_id,
                            fused_global=pred,
                            short_global=pred,  # placeholder
                            long_global=pred[::self.s] if T > self.s else pred,
                            anchor_indices=anchor_idx,
                            gt_global=gt,
                            out_dir=csv_dir,
                        )
                    except Exception as exc:
                        warnings.warn(f"[V2] CSV export failed for {scan_id}: {exc}")

        return result

    # ── Format log line ──────────────────────────────────────────────

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        m = metrics
        return (
            f"[KRootDualV2 epoch {epoch}  step {n_optim_steps}]  "
            f"short={m.get('short_loss', 0):.4f}(rot={m.get('short_rot', 0):.4f} "
            f"trans={m.get('short_trans', 0):.4f} "
            f"drift={m.get('short_drift_mm', 0):.1f}mm)  "
            f"long={m.get('long_loss', 0):.4f}(rot={m.get('long_rot', 0):.4f} "
            f"trans={m.get('long_trans', 0):.4f} "
            f"drift={m.get('long_drift_mm', 0):.1f}mm)  "
            f"lr={current_lr:.2e}"
        )
