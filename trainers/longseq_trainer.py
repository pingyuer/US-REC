"""Long-sequence trainer for Early-CNN + Temporal Transformer pose models.

This trainer works with :class:`LongSeqPoseModel` and the ``ScanWindowDataset``
(or ``SyntheticScanWindowDataset`` for smoke tests).  It keeps the baseline
pairwise path completely untouched; the original ``rec_trainer.py`` is not
modified.

Now inherits from :class:`BaseTrainer` for shared boilerplate (hooks,
checkpoint, gradient accumulation, LR schedule, train loop skeleton).

Usage (from config):
    trainer.name: trainers.longseq_trainer.LongSeqTrainer
    model.type: longseq_transformer
"""

from __future__ import annotations

import math
import os
import time
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.longseq_loss import longseq_loss, make_ref_points, ddf_surrogate_loss
from metrics.compose import compose_global_from_local, local_from_global
from trainers.base_trainer import BaseTrainer
from trainers.common import cfg_get, load_tform_calib


class LongSeqTrainer(BaseTrainer):
    """Self-contained trainer for the long-sequence pose model.

    Inherits generic train loop, hooks, checkpoint from :class:`BaseTrainer`.
    Adds: Transformer-XL memory chunking, multi-Δ aux losses, DDF eval.

    Lifecycle::

        trainer = LongSeqTrainer(cfg, device=...)
        trainer.train()           # or
        trainer.evaluate(loader)  # standalone eval
    """

    def __init__(
        self,
        cfg,
        *,
        device: str | torch.device = "cpu",
        train_loader=None,
        val_loader=None,
    ):
        super().__init__(cfg, device=device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._last_train_avg_loss: float = 0.0

        # ---- model hyper-parameters -----------------------------------------
        self.rotation_rep = str(cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ---- optimizer -------------------------------------------------------
        lr = float(cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 1e-4)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ---- loss config -----------------------------------------------------
        self.rot_weight = float(cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(cfg_get(cfg, "loss.trans_weight", 1.0))
        self.aux_intervals = list(cfg_get(cfg, "loss.aux_intervals", [2, 4, 8, 16]))
        self.aux_base_weight = float(cfg_get(cfg, "loss.aux_weight", 0.5))
        self.aux_decay = float(cfg_get(cfg, "loss.aux_decay", 0.5))
        self.aux_scale = str(cfg_get(cfg, "loss.aux_scale", "none"))
        self.consistency_weight = float(cfg_get(cfg, "loss.consistency_weight", 0.1))
        self.consistency_delta = int(cfg_get(cfg, "loss.consistency_delta", 2))
        self.loss_mode = str(cfg_get(cfg, "loss.mode", "points"))
        self.ref_pts_scale_mm = float(cfg_get(cfg, "loss.ref_pts_scale_mm", 20.0))
        self.ddf_sample_weight = float(cfg_get(cfg, "loss.ddf_sample_weight", 0.0))
        self.ddf_num_points = int(cfg_get(cfg, "loss.ddf_num_points", 1024))

        # ---- calibration (for DDF loss + TUS-REC eval metrics) --------------
        self.tform_calib: torch.Tensor | None = load_tform_calib(
            cfg, device=self.device, warn_prefix="LongSeqTrainer"
        )

        # ---- image dimensions for DDF loss pixel sampling -------------------
        img_h = cfg_get(cfg, "dataset.image_size_h") or cfg_get(cfg, "model.image_size_h")
        img_w = cfg_get(cfg, "dataset.image_size_w") or cfg_get(cfg, "model.image_size_w")
        self.ddf_image_size: tuple[int, int] = (
            (int(img_h), int(img_w)) if (img_h and img_w) else (480, 640)
        )

        # ---- evaluation metric mode -----------------------------------------
        self.eval_metric_mode = str(cfg_get(cfg, "eval.metric_mode", "points"))

        # Segment length for Transformer-XL style recurrence
        self.seg_len = int(cfg_get(cfg, "model.transformer.seg_len", 0) or 0)

    # ── model construction ───────────────────────────────────────────

    def _build_model(self) -> LongSeqPoseModel:
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        aux_intervals = list(cfg_get(self.cfg, "loss.aux_intervals", [2, 4, 8, 16]))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))
        memory_size = int(cfg_get(self.cfg, "model.transformer.memory_size", 0) or 0)

        return LongSeqPoseModel(
            backbone=backbone,
            in_channels=1,
            token_dim=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_ff,
            window_size=window_size,
            dropout=dropout,
            rotation_rep=self.rotation_rep,
            aux_intervals=aux_intervals,
            pretrained_backbone=pretrained,
            memory_size=memory_size,
        )

    # ── model forward helpers ────────────────────────────────────────

    def _forward_with_memory(
        self, frames: torch.Tensor, chunk_size: int = 0
    ) -> dict:
        """Run model forward, optionally chunked with Transformer-XL memory.

        When ``memory_size > 0`` and ``chunk_size > 0``, splits ``frames``
        into ``chunk_size``-frame segments and passes memory between them.
        This simulates online inference and gives the model cross-segment
        context at evaluation time.

        Parameters
        ----------
        frames : (B, T, H, W)
        chunk_size : int  chunk size in frames (0 = full scan in one pass)

        Returns
        -------
        dict matching ``LongSeqPoseModel.forward()`` output, with
        ``pred_local_T`` spanning the full T frames.
        """
        memory_size = getattr(self.model, "memory_size", 0)
        if chunk_size <= 0 or memory_size == 0:
            # Single forward pass (backward compatible)
            return self.model(frames)

        B, T, H, W = frames.shape
        all_local: list[torch.Tensor] = []
        all_aux: dict[int, list[torch.Tensor]] = {}
        memory = None
        pos_offset = 0

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_frames = frames[:, start:end]
            out = self.model(chunk_frames, memory=memory, pos_offset=pos_offset)
            all_local.append(out["pred_local_T"])
            for delta, aux_t in out.get("pred_aux_T", {}).items():
                all_aux.setdefault(delta, []).append(aux_t)
            memory = out["memory"]
            pos_offset += (end - start)

        pred_local_T = torch.cat(all_local, dim=1)   # (B, T, 4, 4)
        pred_aux_T = {
            d: torch.cat(chunks, dim=1) for d, chunks in all_aux.items()
        }
        return {
            "pred_local_T": pred_local_T,
            "pred_aux_T": pred_aux_T,
            "tokens": None,
            "ctx": None,
            "memory": memory,
        }

    # register_hook / call_hooks inherited from BaseTrainer

    # ── single training step ─────────────────────────────────────────

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss on a single sequence batch.

        Parameters
        ----------
        batch : dict with ``frames`` (B,T,H,W) and ``gt_global_T`` (B,T,4,4)

        Returns
        -------
        loss : scalar tensor (differentiable)
        metrics : dict of detached float metrics
        """
        frames = batch["frames"].to(self.device)          # (B, T, H, W)
        gt_global = batch["gt_global_T"].to(self.device)   # (B, T, 4, 4)

        if frames.max() > 1.1:
            frames = frames / 255.0

        # Use segmented Transformer-XL forward when seg_len and memory are configured.
        # Within each segment, gradients flow normally; memory tokens are stop-grad
        # (temporal_transformer.py detaches new_memory before returning).
        if self.seg_len > 0 and getattr(self.model, "memory_size", 0) > 0:
            out = self._forward_with_memory(frames, chunk_size=self.seg_len)
        else:
            out = self.model(frames)
        pred_local_T = out["pred_local_T"]       # (B, T, 4, 4)
        pred_aux_T = out["pred_aux_T"]           # dict[Δ → (B, T, 4, 4)]

        loss, breakdown = longseq_loss(
            pred_local_T=pred_local_T,
            pred_aux_T=pred_aux_T,
            gt_global_T=gt_global,
            intervals=self.aux_intervals,
            loss_mode=self.loss_mode,
            ref_pts_scale_mm=self.ref_pts_scale_mm,
            rot_weight=self.rot_weight,
            trans_weight=self.trans_weight,
            aux_base_weight=self.aux_base_weight,
            aux_decay=self.aux_decay,
            aux_scale=self.aux_scale,
            consistency_weight=self.consistency_weight,
            consistency_delta=self.consistency_delta,
            ddf_sample_weight=self.ddf_sample_weight,
            ddf_num_points=self.ddf_num_points,
            ddf_tform_calib=self.tform_calib,
            ddf_image_size=self.ddf_image_size,
        )

        # Compute per-frame global pose for diagnostics
        with torch.no_grad():
            pred_global = compose_global_from_local(pred_local_T)  # (B, T, 4, 4)
            # SE(3) translation drift at last frame
            drift_t = (
                pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]
            ).norm(dim=-1).mean()
            breakdown["drift_mm_last"] = float(drift_t.item())

            if self.loss_mode == "points":
                # Points-based drift at last frame (matches GPE evaluation direction)
                _ref = make_ref_points(
                    self.ref_pts_scale_mm, device=self.device, dtype=torch.float32
                )  # (K, 4)
                _pts_hom = _ref.T  # (4, K)
                _pred_last = (
                    pred_global[:, -1] @ _pts_hom
                )[:, :3, :].permute(0, 2, 1)   # (B, K, 3)
                _gt_last = (
                    gt_global[:, -1] @ _pts_hom
                )[:, :3, :].permute(0, 2, 1)
                breakdown["drift_pts_mm"] = float(
                    (_pred_last - _gt_last).norm(dim=-1).mean().item()
                )
                # RMSE of local points loss (sqrt of MSE in mm^2 → mm)
                breakdown["local_pts_rmse_mm"] = float(
                    breakdown["loss_local"] ** 0.5
                )
            else:
                # SE(3) mode: convert geodesic rotation loss to degrees
                breakdown["local_rot_deg"] = float(
                    math.degrees(breakdown["loss_local_rot"])
                )
                breakdown["local_trans_mm"] = float(
                    breakdown["loss_local_trans"] ** 0.5
                )

        return loss, breakdown

    # ── train loop customisations ────────────────────────────────────
    # The generic loop is inherited from BaseTrainer.train().
    # We override _format_step_log for LongSeq-specific console output,
    # _pre_clip_metrics for rotation-head gradient norms, and
    # _build_val_log_buffer for tusrec metric renaming.

    def _pre_clip_metrics(self) -> dict[str, float]:
        """Compute gradient norm for the pose head before clip+step."""
        total_norm_sq = 0.0
        found = False
        for name, param in self.model.named_parameters():
            if name.startswith("local_head.") and param.grad is not None:
                total_norm_sq += param.grad.data.norm().item() ** 2
                found = True
        extras: dict[str, float] = {}
        if found:
            extras["grad_norm_rot_head"] = total_norm_sq ** 0.5
        return extras

    def _build_val_log_buffer(
        self, val_metrics: dict, avg_loss: float, epoch: int,
    ) -> dict:
        """Use training avg_loss as val_loss; rename ``mean_tusrec_*`` → ``val_tusrec_*``."""
        buf: dict = {
            "epoch": epoch + 1,
            "val_loss": avg_loss,
            **val_metrics,
        }
        # Promote tusrec metrics to "val_" namespace for hook consumers
        for key in list(buf):
            if key.startswith("mean_tusrec_"):
                buf["val_" + key[5:]] = buf[key]  # mean_tusrec_X → val_tusrec_X
        return buf

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        if self.loss_mode == "points":
            mode_str = (
                f"local_pts_rmse={metrics.get('local_pts_rmse_mm', 0.0):.6f}mm  "
                f"drift_pts={metrics.get('drift_pts_mm', 0.0):.6f}mm"
            )
        else:
            mode_str = (
                f"local_rot={metrics.get('local_rot_deg', 0.0):.6f}deg  "
                f"local_trans={metrics.get('local_trans_mm', 0.0):.6f}mm"
            )
        aux_parts = "  ".join(
            f"auxΔ{d}={metrics.get(f'aux_delta{d}_pts' if self.loss_mode == 'points' else f'aux_delta{d}_rot', 0.0):.5f}"
            for d in sorted(self.aux_intervals)
            if f"aux_delta{d}_pts" in metrics or f"aux_delta{d}_rot" in metrics
        )
        base = (
            f"[LongSeq epoch {epoch}  step {n_optim_steps}]  "
            f"loss={avg_accum:.6f}  {mode_str}  "
            f"ddf={metrics.get('loss_ddf', 0.0):.6f}  "
            f"consist={metrics.get('loss_consistency', 0.0):.5f}  "
            f"drift_se3={metrics.get('drift_mm_last', 0.0):.3f}mm  "
            f"lr={current_lr:.2e}"
        )
        return base + (f"  [{aux_parts}]" if aux_parts else "")

    # ── evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Evaluate long-sequence model on a full-scan loader.

        Two metric modes are supported (config key ``eval.metric_mode``):

        ``"points"`` (default)
            Four reference-point L2 error in mm, fast and calib-free:
            * ``gpe_pts_mm`` — global point error (frames 0..T-1 vs GT)
            * ``lpe_pts_mm`` — local  point error (frame-to-frame)
            * ``drift_pts_mm`` — point error at last frame only

        ``"tusrec_ddf"``
            Official TUS-REC Challenge metrics via ``compute_tusrec_metrics()``:
            * ``GPE_mm``, ``LPE_mm``, ``GLE_mm``, ``LLE_mm``
            * ``GPE_score``, ``LPE_score``, ``final_score``
            Requires ``tform_calib`` to be loaded (``dataset.calib_file``).
            Falls back to "points" mode if calib is unavailable.

        Both modes also report SE(3) translation-only metrics
        (``se3_lpe_mm``, ``se3_gpe_mm``, ``se3_drift_last_mm``).  These measure
        L2 distance between predicted and GT translation vectors and are *not*
        the same as the DDF-based TUS-REC LPE/GPE metrics.

        Returns
        -------
        dict with aggregated metrics over all scans in the loader.
        """
        self.model.eval()

        use_tusrec = (
            self.eval_metric_mode == "tusrec_ddf"
            and self.tform_calib is not None
        )
        if self.eval_metric_mode == "tusrec_ddf" and self.tform_calib is None:
            warnings.warn(
                "[LongSeqTrainer.evaluate] eval_metric_mode='tusrec_ddf' but no "
                "tform_calib loaded.  Falling back to 'points' mode.  Set "
                "dataset.calib_file in the config to enable DDF metrics."
            )

        # Reference points shared across all scans (used in both modes for
        # fast proxy metrics alongside the official ones)
        ref_pts = make_ref_points(
            self.ref_pts_scale_mm,
            device=self.device,
            dtype=torch.float32,
        )  # (K, 4) homogeneous

        if use_tusrec:
            from trainers.metrics.tusrec import compute_tusrec_metrics  # noqa: PLC0415

        scan_metrics: list[dict[str, float]] = []

        for batch in loader:
            frames = batch["frames"].to(self.device)          # (B, T, H, W)
            gt_global = batch["gt_global_T"].to(self.device)  # (B, T, 4, 4)

            if frames.max() > 1.1:
                frames = frames / 255.0

            out = self._forward_with_memory(
                frames,
                chunk_size=int(cfg_get(self.cfg, "model.transformer.window_size", 64)),
            )
            pred_local_T = out["pred_local_T"]
            pred_global = compose_global_from_local(pred_local_T)
            gt_local = local_from_global(gt_global)

            pts_hom = ref_pts.T.to(device=self.device, dtype=torch.float32)  # (4, K)

            B, T = pred_local_T.shape[:2]
            for b in range(B):
                sm: dict[str, float] = {}

                # ── Points-based metrics (always computed) ────────────────────
                gt_pts_g   = (gt_global[b]   @ pts_hom)[:, :3, :].permute(0, 2, 1)  # (T,K,3)
                pred_pts_g = (pred_global[b] @ pts_hom)[:, :3, :].permute(0, 2, 1)  # (T,K,3)
                sm["gpe_pts_mm"]   = float((pred_pts_g - gt_pts_g).norm(dim=-1).mean().item())
                sm["drift_pts_mm"] = float((pred_pts_g[-1] - gt_pts_g[-1]).norm(dim=-1).mean().item())

                gt_pts_l   = (gt_local[b, 1:]     @ pts_hom)[:, :3, :].permute(0, 2, 1)
                pred_pts_l = (pred_local_T[b, 1:] @ pts_hom)[:, :3, :].permute(0, 2, 1)
                sm["lpe_pts_mm"] = float((pred_pts_l - gt_pts_l).norm(dim=-1).mean().item())

                # ── SE(3) translation-only error (NOT the official DDF LPE/GPE) ───
                # Named se3_* to distinguish from the DDF-based tusrec_* metrics.
                sm["se3_lpe_mm"]        = float((pred_local_T[b, 1:, :3, 3] - gt_local[b, 1:, :3, 3]).norm(dim=-1).mean().item())
                sm["se3_gpe_mm"]        = float((pred_global[b, :, :3, 3]   - gt_global[b, :, :3, 3]).norm(dim=-1).mean().item())
                sm["se3_drift_last_mm"] = float((pred_global[b, -1, :3, 3]  - gt_global[b, -1, :3, 3]).norm().item())

                # ── Official TUS-REC DDF metrics (optional) ───────────────────
                if use_tusrec:
                    try:
                        tusrec = compute_tusrec_metrics(
                            frames=frames[b],          # (T, H, W)
                            gt_transforms=gt_global[b],    # (T, 4, 4)  T_{0←i}
                            pred_transforms=pred_global[b],
                            calib={"tform_calib": self.tform_calib},
                            compute_scores=True,
                        )
                        for k, v in tusrec.items():
                            if v is not None:
                                sm[f"tusrec_{k}"] = float(v)
                    except Exception as exc:  # noqa: BLE001
                        warnings.warn(f"[evaluate] compute_tusrec_metrics failed: {exc}")

                sm["num_frames"] = int(T)
                scan_metrics.append(sm)

        if not scan_metrics:
            base = {
                "gpe_pts_mm": 0.0, "lpe_pts_mm": 0.0, "drift_pts_mm": 0.0,
                "se3_lpe_mm": 0.0, "se3_gpe_mm": 0.0, "se3_drift_last_mm": 0.0,
            }
            if use_tusrec:
                base.update({"tusrec_GPE_mm": 0.0, "tusrec_LPE_mm": 0.0, "tusrec_final_score": 0.0})
            return base

        agg: dict[str, float] = {}
        # Union of all keys (DDF metrics may only appear in some scans)
        all_keys = {k for s in scan_metrics for k in s if k != "num_frames"}
        for key in all_keys:
            vals = [s[key] for s in scan_metrics if key in s]
            agg[f"mean_{key}"] = sum(vals) / len(vals)
        agg["num_scans"] = float(len(scan_metrics))
        return agg

