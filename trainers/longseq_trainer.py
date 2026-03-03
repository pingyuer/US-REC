"""Long-sequence trainer for Early-CNN + Temporal Transformer pose models.

This trainer works with :class:`LongSeqPoseModel` and the ``ScanWindowDataset``
(or ``SyntheticScanWindowDataset`` for smoke tests).  It keeps the baseline
pairwise path completely untouched; the original ``rec_trainer.py`` is not
modified.

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
from trainers.hooks.base_hook import Hook


def _cfg_get(cfg, path, default=None):
    val = OmegaConf.select(cfg, path)
    return default if val is None else val


class LongSeqTrainer:
    """Self-contained trainer for the long-sequence pose model.

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
        self.cfg = cfg
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.hooks: list[Hook] = []
        self.epoch = 0
        self.global_step = 0
        self._last_train_avg_loss: float = 0.0

        # ---- model hyper-parameters -----------------------------------------
        self.rotation_rep = str(_cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ---- optimizer -------------------------------------------------------
        lr = float(_cfg_get(cfg, "optimizer.lr_rec", _cfg_get(cfg, "optimizer.lr", 1e-4)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ---- loss config -----------------------------------------------------
        self.rot_weight = float(_cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(_cfg_get(cfg, "loss.trans_weight", 1.0))
        self.aux_intervals = list(_cfg_get(cfg, "loss.aux_intervals", [2, 4, 8, 16]))
        self.aux_base_weight = float(_cfg_get(cfg, "loss.aux_weight", 0.5))
        self.aux_decay = float(_cfg_get(cfg, "loss.aux_decay", 0.5))
        self.aux_scale = str(_cfg_get(cfg, "loss.aux_scale", "none"))
        self.consistency_weight = float(_cfg_get(cfg, "loss.consistency_weight", 0.1))
        self.consistency_delta = int(_cfg_get(cfg, "loss.consistency_delta", 2))
        # loss mode: "points" (default, aligned with GPE metric) or "se3" (ablation)
        self.loss_mode = str(_cfg_get(cfg, "loss.mode", "points"))
        self.ref_pts_scale_mm = float(_cfg_get(cfg, "loss.ref_pts_scale_mm", 20.0))
        # DDF surrogate loss: samples K random pixels from the image plane,
        # maps through tform_calib, applies global T, measures L2 error in mm.
        # Requires calibration to be loaded; weight=0 disables it.
        self.ddf_sample_weight = float(_cfg_get(cfg, "loss.ddf_sample_weight", 0.0))
        self.ddf_num_points = int(_cfg_get(cfg, "loss.ddf_num_points", 1024))

        # ---- calibration (for DDF loss + TUS-REC eval metrics) --------------
        # Loaded from cfg.dataset.calib_file exactly like rec_trainer does.
        self.tform_calib: torch.Tensor | None = self._load_calib()

        # ---- image dimensions for DDF loss pixel sampling -------------------
        img_h = _cfg_get(cfg, "dataset.image_size_h") or _cfg_get(cfg, "model.image_size_h")
        img_w = _cfg_get(cfg, "dataset.image_size_w") or _cfg_get(cfg, "model.image_size_w")
        self.ddf_image_size: tuple[int, int] = (
            (int(img_h), int(img_w)) if (img_h and img_w) else (480, 640)
        )

        # ---- evaluation metric mode -----------------------------------------
        # "points"     — ref-point L2 in mm (fast, no calib needed)
        # "tusrec_ddf" — official TUS-REC DDF metrics (GPE_mm/LPE_mm/score)
        self.eval_metric_mode = str(_cfg_get(cfg, "eval.metric_mode", "points"))

        # ---- training schedule -----------------------------------------------
        self.num_epochs = int(_cfg_get(cfg, "trainer.max_epochs", 1) or 1)
        self.max_steps = _cfg_get(cfg, "train.max_steps") or _cfg_get(cfg, "trainer.max_steps")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        self.log_interval = int(_cfg_get(cfg, "trainer.log_interval", 50))
        self.val_every = int(_cfg_get(cfg, "trainer.validate_every", 1) or 1)
        self.save_path = str(_cfg_get(cfg, "paths.output_dir", "logs"))
        # Gradient accumulation: accumulate over N micro-batches before stepping.
        self.grad_accum = max(1, int(_cfg_get(cfg, "trainer.grad_accum", 1) or 1))
        # Segment length for Transformer-XL style recurrence:
        # split each training sequence into chunks of seg_len and pass memory
        # tokens between segments.  0 = disabled (single-pass forward).
        # mem_len is a documentation alias for memory_size (set at model level).
        self.seg_len = int(_cfg_get(cfg, "model.transformer.seg_len", 0) or 0)

    # ── calibration ─────────────────────────────────────────────────

    def _load_calib(self) -> torch.Tensor | None:
        """Load calibration tform_calib from config, if calib_file is provided."""
        calib_file = _cfg_get(self.cfg, "dataset.calib_file")
        if not calib_file:
            return None
        try:
            from trainers.utils.calibration import load_calibration  # noqa: PLC0415
            resample_factor = float(_cfg_get(self.cfg, "dataset.resample_factor", 1.0))
            _, _, tform_calib = load_calibration(calib_file, resample_factor, device=self.device)
            if not isinstance(tform_calib, torch.Tensor):
                import numpy as np  # noqa: PLC0415
                tform_calib = torch.as_tensor(tform_calib, dtype=torch.float32)
            return tform_calib.float().to(self.device)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[LongSeqTrainer] Could not load calibration from {calib_file!r}: {exc}")
            return None

    # ── model construction ───────────────────────────────────────────

    def _build_model(self) -> LongSeqPoseModel:
        backbone = str(_cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(_cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(_cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(_cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(_cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(_cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(_cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        aux_intervals = list(_cfg_get(self.cfg, "loss.aux_intervals", [2, 4, 8, 16]))
        pretrained = bool(_cfg_get(self.cfg, "model.encoder.pretrained", False))
        memory_size = int(_cfg_get(self.cfg, "model.transformer.memory_size", 0) or 0)

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

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for h in self.hooks:
            fn = getattr(h, event, None)
            if callable(fn):
                fn(self, **kwargs)

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

    # ── train loop ───────────────────────────────────────────────────

    def train(self):
        if self.train_loader is None:
            raise ValueError("No train_loader provided")

        self.call_hooks("before_run", mode="train")
        self.call_hooks("before_train")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.model.train()
            epoch_loss = 0.0
            n_steps = 0

            self.call_hooks("before_epoch")

            # Update dataset epoch for proper shuffle randomisation
            ds = getattr(self.train_loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            self.optimizer.zero_grad()  # initialise grad buffer at epoch start

            for step, batch in enumerate(self.train_loader):
                reached_max = (
                    self.max_steps is not None
                    and self.global_step >= self.max_steps
                )
                if reached_max:
                    # Flush any pending accumulated gradients before stopping
                    if n_steps % self.grad_accum != 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    break
                self.global_step += 1
                n_steps += 1
                self.call_hooks("before_step")

                # Gradient accumulation: scale loss, backward every step,
                # but clip + step only every self.grad_accum micro-batches.
                loss, metrics = self._run_step(batch)
                (loss / self.grad_accum).backward()

                # ── Gradient norm of rotation head (computed after backward) ─────────
                # Confirms that gradients flow to the rotation sub-network.
                _rot_head = getattr(self.model, "local_head", None)
                if _rot_head is not None:
                    _rg = [
                        p.grad.detach().norm()
                        for p in _rot_head.parameters()
                        if p.grad is not None
                    ]
                    metrics["grad_norm_rot_head"] = (
                        float(torch.stack(_rg).norm().item()) if _rg else 0.0
                    )
                else:
                    metrics["grad_norm_rot_head"] = 0.0

                should_step = (
                    n_steps % self.grad_accum == 0
                    or (
                        self.max_steps is not None
                        and self.global_step >= self.max_steps
                    )
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss += float(loss.item())

                if n_steps % self.log_interval == 0 or n_steps == 1:
                    # Mode-aware logging:
                    #   points mode → local_pts_rmse_mm + drift_pts_mm
                    #   se3 mode    → local_rot_deg + local_trans_mm
                    if self.loss_mode == "points":
                        _mode_str = (
                            f"local_pts_rmse={metrics.get('local_pts_rmse_mm', 0.0):.6f}mm  "
                            f"drift_pts={metrics.get('drift_pts_mm', 0.0):.6f}mm"
                        )
                    else:
                        _mode_str = (
                            f"local_rot={metrics.get('local_rot_deg', 0.0):.6f}deg  "
                            f"local_trans={metrics.get('local_trans_mm', 0.0):.6f}mm"
                        )
                    # Per-delta aux losses for multi-interval supervision
                    _aux_parts = "  ".join(
                        f"auxΔ{d}={metrics.get(f'aux_delta{d}_pts' if self.loss_mode == 'points' else f'aux_delta{d}_rot', 0.0):.5f}"
                        for d in sorted(self.aux_intervals)
                        if f"aux_delta{d}_pts" in metrics or f"aux_delta{d}_rot" in metrics
                    )
                    print(
                        f"[LongSeq epoch {epoch}  step {n_steps}]  "
                        f"loss={loss.item():.6f}  {_mode_str}  "
                        f"ddf={metrics.get('loss_ddf', 0.0):.6f}  "
                        f"consist={metrics.get('loss_consistency', 0.0):.5f}  "
                        f"drift_se3={metrics.get('drift_mm_last', 0.0):.3f}mm  "
                        f"grad_rot={metrics.get('grad_norm_rot_head', 0.0):.5f}"
                        + (f"  [{_aux_parts}]" if _aux_parts else "")
                    )

                lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                self.call_hooks(
                    "after_step",
                    log_buffer={
                        "mode": "train",
                        "epoch": epoch + 1,
                        "iter": step + 1,
                        "global_step": self.global_step,
                        "loss": float(loss.item()),
                        "lr": lr,
                        **metrics,
                    },
                )

            avg_loss = epoch_loss / max(1, n_steps)
            self._last_train_avg_loss = avg_loss
            print(f"[LongSeq epoch {epoch}] avg_loss={avg_loss:.4f}")
            lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
            self.call_hooks(
                "after_epoch",
                log_buffer={
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "lr": lr,
                },
            )

            # Validation
            if self.val_loader is not None and (epoch + 1) % self.val_every == 0:
                val_metrics = self.evaluate(self.val_loader)
                print(f"[LongSeq val epoch {epoch}] {val_metrics}")
                # Store for VizHook; fire after_val so MLflowHook can save best checkpoint.
                self.last_eval_metrics = val_metrics
                # Build after_val log_buffer.
                # val_loss tracks the *training* objective (lower → better ckpt).
                # val_tusrec_* expose official TUS-REC metrics for separate best
                # checkpoint tracking (val_tusrec_final_score → higher is better).
                _after_val_buf: dict[str, Any] = {
                    "epoch": epoch + 1,
                    "val_loss": self._last_train_avg_loss,
                    **{k: float(v) for k, v in val_metrics.items()
                       if isinstance(v, (int, float))},
                }
                # Expose official TUS-REC metrics under canonical names so that
                # mlflow / checkpoint hooks can use them as best-metric targets.
                if "mean_tusrec_final_score" in val_metrics:
                    _after_val_buf["val_tusrec_final_score"] = float(val_metrics["mean_tusrec_final_score"])
                if "mean_tusrec_GPE_mm" in val_metrics:
                    _after_val_buf["val_tusrec_GPE_mm"] = float(val_metrics["mean_tusrec_GPE_mm"])
                self.call_hooks(
                    "after_val",
                    log_buffer=_after_val_buf,
                )

        self.call_hooks("after_train")
        self.call_hooks("after_run")

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
                chunk_size=int(_cfg_get(self.cfg, "model.transformer.window_size", 64)),
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

