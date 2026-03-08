"""VQ-Memory Conditioned Local Pose Transformer — Trainer.

Extends the LongSeqTrainer with:
  * Scan-level VQ cache (built once per scan, shared across windows)
  * L_cons (multi-view consistency of scan summary g)
  * L_geom (coarse trajectory regression from g)
  * VQ codebook usage logging
  * Additional eval diagnostics (memory attention entropy, g norm)

The trainer reuses the existing ScanWindowDataset + eval pipeline.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F

from models.vq.vq_memory_model import VQMemoryPoseModel
from models.vq.scan_geom_head import build_geom_target, geom_loss, consistency_loss
from models.losses.longseq_loss import longseq_loss, make_ref_points
from metrics.compose import compose_global_from_local, local_from_global
from trainers.base_trainer import BaseTrainer
from trainers.common import cfg_get, EMA, load_tform_calib
from trainers.utils.vq_memory import (
    valid_length,
    build_batched_scan_cache,
    build_consistency_views,
    build_masked_geom_targets,
)


class VQMemoryTrainer(BaseTrainer):
    """Trainer for VQ-Memory Conditioned Local Pose Transformer.

    Lifecycle::

        trainer = VQMemoryTrainer(cfg, device=..., train_loader=..., val_loader=...)
        trainer.train()
        trainer.evaluate(loader)
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

        # ── Model config ─────────────────────────────────────────────────
        self.rotation_rep = str(cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.model = self._build_model().to(self.device)

        # ── Optimizer ─────────────────────────────────────────────────────
        lr = float(cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 1e-4)))
        wd = float(cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        encoder_lr_mult = float(cfg_get(cfg, "optimizer.encoder_lr_mult", 0.1))

        # Build param groups: encoder gets lower LR
        encoder_params = list(self.model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in encoder_ids]

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"},
            {"params": other_params, "lr": lr, "name": "main"},
        ], weight_decay=wd)

        # ── EMA ──────────────────────────────────────────────────────────
        ema_decay = float(cfg_get(cfg, "trainer.ema_decay", 0.0))
        self.ema = EMA(self.model, decay=ema_decay) if ema_decay > 0 else None

        # ── Loss config ──────────────────────────────────────────────────
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
        self.ddf_use_rmse = bool(cfg_get(cfg, "loss.ddf_use_rmse", True))
        self.ddf_loss_max = float(cfg_get(cfg, "loss.ddf_loss_max", 0.0))

        # VQ-specific loss weights
        self.lambda_cons = float(cfg_get(cfg, "loss.lambda_cons", 0.1))
        self.lambda_geom = float(cfg_get(cfg, "loss.lambda_geom", 0.1))
        self.lambda_commit = float(cfg_get(cfg, "loss.lambda_commit", 0.25))
        self.lambda_endpoint = float(cfg_get(cfg, "loss.lambda_endpoint", 0.15))
        self.lambda_contrast = float(cfg_get(cfg, "loss.lambda_contrast", 0.05))
        self.contrast_temp = float(cfg_get(cfg, "loss.contrast_temperature", 0.2))

        # VQ config
        self.anchor_stride = int(cfg_get(cfg, "vq.anchor_stride",
                                          cfg_get(cfg, "model.anchor_stride", 8)))
        self.cons_sample_ratio = float(cfg_get(cfg, "loss.cons_sample_ratio", 0.6))
        self.n_geom_waypoints = int(cfg_get(cfg, "model.n_geom_waypoints",
                                             cfg_get(cfg, "vq.n_geom_waypoints", 8)))

        # ── Calibration ──────────────────────────────────────────────────
        self.tform_calib: torch.Tensor | None = load_tform_calib(
            cfg, device=self.device, warn_prefix="VQMemoryTrainer",
        )

        img_h = cfg_get(cfg, "dataset.image_size_h") or cfg_get(cfg, "model.image_size_h")
        img_w = cfg_get(cfg, "dataset.image_size_w") or cfg_get(cfg, "model.image_size_w")
        self.ddf_image_size: tuple[int, int] = (
            (int(img_h), int(img_w)) if (img_h and img_w) else (480, 640)
        )

        # ── Eval config ──────────────────────────────────────────────────
        self.eval_metric_mode = str(cfg_get(cfg, "eval.metric_mode", "points"))
        self.seg_len = int(cfg_get(cfg, "model.transformer.seg_len", 0) or 0)

        # ── Logging file for VQ diagnostics ──────────────────────────────
        self._vq_log_path: str | None = None

    # ── Model construction ────────────────────────────────────────────

    def _build_model(self) -> VQMemoryPoseModel:
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        token_dim = int(cfg_get(self.cfg, "model.transformer.d_model", 256))
        n_heads = int(cfg_get(self.cfg, "model.transformer.n_heads", 4))
        n_layers = int(cfg_get(self.cfg, "model.transformer.n_layers", 4))
        dim_ff = int(cfg_get(self.cfg, "model.transformer.dim_feedforward", 1024))
        window_size = int(cfg_get(self.cfg, "model.transformer.window_size", 64))
        dropout = float(cfg_get(self.cfg, "model.transformer.dropout", 0.1))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))
        memory_size = int(cfg_get(self.cfg, "model.transformer.memory_size", 0) or 0)

        # VQ config
        code_dim = int(cfg_get(self.cfg, "vq.code_dim", token_dim))
        codebook_size = int(cfg_get(self.cfg, "vq.codebook_size", 512))
        ema_decay = float(cfg_get(self.cfg, "vq.ema_decay", 0.99))
        anchor_stride = int(cfg_get(self.cfg, "vq.anchor_stride",
                                     cfg_get(self.cfg, "model.anchor_stride", 8)))

        # Summary config
        pool_type = str(cfg_get(self.cfg, "summary.pool_type", "attention"))
        n_latents = int(cfg_get(self.cfg, "summary.latent_num", 8))

        # Memory/conditioning config
        use_film = bool(cfg_get(self.cfg, "conditioning.use_film", True))
        use_global_token = bool(cfg_get(self.cfg, "conditioning.use_global_token", False))
        use_cross_attn = bool(cfg_get(self.cfg, "memory.use_cross_attn", True))
        mem_n_heads = int(cfg_get(self.cfg, "memory.num_heads", 4))
        scan_ctx_layers = int(cfg_get(self.cfg, "memory.context_layers", 2))
        scan_ctx_heads = int(cfg_get(self.cfg, "memory.context_heads", mem_n_heads))

        n_geom_waypoints = int(cfg_get(self.cfg, "vq.n_geom_waypoints", 8))

        aux_intervals = list(cfg_get(self.cfg, "loss.aux_intervals", [2, 4, 8, 16]))

        # Encoder input resize (None = native resolution)
        enc_h = cfg_get(self.cfg, "model.encoder.input_h")
        enc_w = cfg_get(self.cfg, "model.encoder.input_w")
        encoder_input_size = (int(enc_h), int(enc_w)) if enc_h and enc_w else None

        return VQMemoryPoseModel(
            backbone=backbone,
            in_channels=1,
            token_dim=token_dim,
            code_dim=code_dim,
            codebook_size=codebook_size,
            ema_decay=ema_decay,
            anchor_stride=anchor_stride,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_ff,
            window_size=window_size,
            dropout=dropout,
            rotation_rep=self.rotation_rep,
            aux_intervals=aux_intervals,
            pretrained_backbone=pretrained,
            memory_size=memory_size,
            encoder_input_size=encoder_input_size,
            pool_type=pool_type,
            n_latents=n_latents,
            n_geom_waypoints=n_geom_waypoints,
            use_film=use_film,
            use_global_token=use_global_token,
            use_memory_cross_attn=use_cross_attn,
            memory_n_heads=mem_n_heads,
            scan_context_layers=scan_ctx_layers,
            scan_context_heads=scan_ctx_heads,
        )

    # ── Training step ────────────────────────────────────────────────

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss on a single batch.

        # The batch contains a scan-window (128 frames) plus the full scan
        # frames for building VQ cache. If scan_frames is padded, we respect
        # its mask and only encode the valid prefix of each scan.
        """
        frames = batch["frames"].to(self.device)            # (B, T, H, W)
        gt_global = batch["gt_global_T"].to(self.device)     # (B, T, 4, 4)

        if frames.max() > 1.1:
            frames = frames / 255.0

        # Use scan_frames if provided, else use window frames as minimal scan
        scan_frames = batch.get("scan_frames")
        scan_frames_mask = batch.get("scan_frames_mask")
        if scan_frames is not None:
            scan_frames = scan_frames.to(self.device)
            if scan_frames.max() > 1.1:
                scan_frames = scan_frames / 255.0
            if scan_frames_mask is not None:
                scan_frames_mask = scan_frames_mask.to(self.device)
        else:
            scan_frames = frames
            scan_frames_mask = None

        # For scan-level GT: use scan_gt_global_T if available, else window GT
        scan_gt = batch.get("scan_gt_global_T")
        scan_gt_mask = batch.get("scan_gt_global_T_mask")
        if scan_gt is not None:
            scan_gt = scan_gt.to(self.device)
            if scan_gt_mask is not None:
                scan_gt_mask = scan_gt_mask.to(self.device)
        else:
            scan_gt = gt_global
            scan_gt_mask = None

        # ── 1. Build scan VQ cache ───────────────────────────────────
        scan_cache = build_batched_scan_cache(self.model, scan_frames, scan_frames_mask)

        # ── 2. Forward local window ──────────────────────────────────
        out = self.model(
            frames=frames,
            scan_vq_cache=scan_cache,
        )
        pred_local_T = out["pred_local_T"]
        pred_aux_T = out["pred_aux_T"]

        # ── 3. Main pose loss ────────────────────────────────────────
        loss_pose, breakdown = longseq_loss(
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
            ddf_use_rmse=self.ddf_use_rmse,
            ddf_loss_max=self.ddf_loss_max,
        )

        # ── 4. L_cons: consistency loss ──────────────────────────────
        view1, view2 = build_consistency_views(
            self.model,
            anchor_stride=self.anchor_stride,
            scan_frames=scan_frames,
            ratio=self.cons_sample_ratio,
            h_all=scan_cache.get("h_all"),
            scan_mask=scan_cache.get("scan_mask", scan_frames_mask),
        )
        l_cons = consistency_loss(view1["g"], view2["g"])
        l_contrast = self._inbatch_contrastive_loss(
            view1["g"], view2["g"], temperature=self.contrast_temp,
        )

        # ── 5. L_geom: geometry loss ────────────────────────────────
        pred_geom = out.get("pred_geom")
        if pred_geom is not None:
            geom_target = build_masked_geom_targets(
                scan_gt,
                self.n_geom_waypoints,
                scan_gt_mask,
            )
            # Detach g through geom head: g is NOT trained via pose loss
            # but it is trained by the geometry objective itself.
            l_geom = geom_loss(pred_geom, geom_target.detach())
        else:
            l_geom = torch.tensor(0.0, device=self.device)

        # ── 6. Commitment loss ───────────────────────────────────────
        commit_loss = out.get("commit_loss", torch.tensor(0.0, device=self.device))
        # Also add commit losses from consistency views
        commit_loss = commit_loss + view1["commit_loss"] + view2["commit_loss"]

        # ── 7. Global endpoint drift constraint ──────────────────────
        pred_global = compose_global_from_local(pred_local_T)
        l_endpoint = (
            pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]
        ).norm(dim=-1).mean()

        # ── 8. Total loss ────────────────────────────────────────────
        loss_total = (
            loss_pose
            + self.lambda_cons * l_cons
            + self.lambda_contrast * l_contrast
            + self.lambda_geom * l_geom
            + self.lambda_commit * commit_loss
            + self.lambda_endpoint * l_endpoint
        )

        # ── 9. Diagnostics ───────────────────────────────────────────
        with torch.no_grad():
            drift_t = (
                pred_global[:, -1, :3, 3] - gt_global[:, -1, :3, 3]
            ).norm(dim=-1).mean()
            breakdown["drift_mm_last"] = float(drift_t.item())

            if self.loss_mode == "points":
                _ref = make_ref_points(
                    self.ref_pts_scale_mm, device=self.device, dtype=torch.float32,
                )
                _pts_hom = _ref.T
                _pred_last = (pred_global[:, -1] @ _pts_hom)[:, :3, :].permute(0, 2, 1)
                _gt_last = (gt_global[:, -1] @ _pts_hom)[:, :3, :].permute(0, 2, 1)
                breakdown["drift_pts_mm"] = float(
                    (_pred_last - _gt_last).norm(dim=-1).mean().item()
                )
                breakdown["local_pts_rmse_mm"] = float(
                    breakdown["loss_local"] ** 0.5
                )

            # VQ-specific metrics
            breakdown["loss_cons"] = float(l_cons.item())
            breakdown["loss_contrast"] = float(l_contrast.item())
            breakdown["loss_geom"] = float(l_geom.item())
            breakdown["loss_commit"] = float(commit_loss.item())
            breakdown["loss_endpoint"] = float(l_endpoint.item())
            breakdown["vq_codebook_usage"] = self.model.vq.codebook_usage
            breakdown["summary_g_norm"] = float(
                scan_cache["g"].norm(dim=-1).mean().item()
            )

            # Memory attention entropy (if available)
            attn_w = out.get("attn_weights")
            if attn_w is not None and attn_w.numel() > 0:
                # Average entropy over batch and tokens
                attn_w_clamped = attn_w.clamp(min=1e-8)
                entropy = -(attn_w_clamped * attn_w_clamped.log()).sum(dim=-1).mean()
                breakdown["memory_attn_entropy"] = float(entropy.item())

        return loss_total, breakdown

    @staticmethod
    def _inbatch_contrastive_loss(
        g1: torch.Tensor,
        g2: torch.Tensor,
        *,
        temperature: float = 0.2,
    ) -> torch.Tensor:
        """Symmetric InfoNCE over batch: positives are matched scan views."""
        bsz = g1.shape[0]
        if bsz < 2:
            return torch.zeros((), device=g1.device, dtype=g1.dtype)

        z1 = F.normalize(g1, dim=-1)
        z2 = F.normalize(g2, dim=-1)
        logits = z1 @ z2.transpose(0, 1)
        logits = logits / max(1e-6, float(temperature))
        labels = torch.arange(bsz, device=g1.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    # ── EMA update ───────────────────────────────────────────────────

    def _after_optim_step(self) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

    # ── Epoch hooks ──────────────────────────────────────────────────

    def _on_epoch_start(self, epoch: int) -> None:
        super()._on_epoch_start(epoch)
        # Reset VQ usage tracking per epoch
        self.model.vq.reset_usage()

    # ── Logging ──────────────────────────────────────────────────────

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
            mode_str = f"loss_local={metrics.get('loss_local', 0.0):.6f}"

        return (
            f"[VQMemory epoch {epoch}  step {n_optim_steps}]  "
            f"loss={avg_accum:.6f}  {mode_str}  "
            f"L_cons={metrics.get('loss_cons', 0.0):.5f}  "
            f"L_ctr={metrics.get('loss_contrast', 0.0):.5f}  "
            f"L_geom={metrics.get('loss_geom', 0.0):.5f}  "
            f"L_end={metrics.get('loss_endpoint', 0.0):.5f}  "
            f"vq_usage={metrics.get('vq_codebook_usage', 0.0):.3f}  "
            f"g_norm={metrics.get('summary_g_norm', 0.0):.3f}  "
            f"mem_ent={metrics.get('memory_attn_entropy', 0.0):.3f}  "
            f"drift_se3={metrics.get('drift_mm_last', 0.0):.3f}mm  "
            f"lr={current_lr:.2e}"
        )

    def _build_val_log_buffer(
        self, val_metrics: dict, avg_loss: float, epoch: int,
    ) -> dict:
        buf: dict = {
            "epoch": epoch + 1,
            "val_loss": val_metrics.get(
                "mean_gpe_mm",
                val_metrics.get("mean_gpe_pts_mm", avg_loss),
            ),
            **val_metrics,
        }
        for key in list(buf):
            if key.startswith("mean_tusrec_"):
                buf["val_" + key[5:]] = buf[key]
        return buf

    def _save_vq_log(self, epoch: int, metrics: dict) -> None:
        """Append VQ diagnostics to a local text file."""
        if self._vq_log_path is None:
            run_dir = str(cfg_get(self.cfg, "paths.output_dir", "logs"))
            os.makedirs(run_dir, exist_ok=True)
            self._vq_log_path = os.path.join(run_dir, "vq_diagnostics.txt")
            with open(self._vq_log_path, "w") as f:
                f.write("epoch\tvq_usage\tL_cons\tL_geom\tg_norm\tmem_entropy\n")

        with open(self._vq_log_path, "a") as f:
            f.write(
                f"{epoch}\t"
                f"{metrics.get('vq_codebook_usage', 0):.4f}\t"
                f"{metrics.get('loss_cons', 0):.6f}\t"
                f"{metrics.get('loss_geom', 0):.6f}\t"
                f"{metrics.get('summary_g_norm', 0):.4f}\t"
                f"{metrics.get('memory_attn_entropy', 0):.4f}\n"
            )

    # ── Evaluation ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        """Evaluate VQ-Memory model on full-scan loader.

        Uses the same evaluation pipeline as LongSeqTrainer with
        additional VQ diagnostics.
        """
        self.model.eval()
        if self.ema is not None:
            backup = self.ema.apply(self.model)

        use_tusrec = (
            self.eval_metric_mode == "tusrec_ddf"
            and self.tform_calib is not None
        )
        if self.eval_metric_mode == "tusrec_ddf" and self.tform_calib is None:
            warnings.warn(
                "[VQMemoryTrainer.evaluate] eval_metric_mode='tusrec_ddf' but no "
                "tform_calib. Falling back to 'points' mode."
            )

        ref_pts = make_ref_points(
            self.ref_pts_scale_mm, device=self.device, dtype=torch.float32,
        )

        if use_tusrec:
            from trainers.metrics.tusrec import compute_tusrec_metrics  # noqa

        scan_metrics: list[dict[str, float]] = []

        for batch in loader:
            frames = batch["frames"].to(self.device)
            gt_global = batch["gt_global_T"].to(self.device)

            if frames.max() > 1.1:
                frames = frames / 255.0

            scan_mask = batch.get("scan_frames_mask")
            if scan_mask is not None:
                scan_mask = scan_mask.to(self.device)

            # Build scan VQ cache for the full scan
            scan_cache = build_batched_scan_cache(self.model, frames, scan_mask)

            # Forward with VQ cache
            out = self.model(
                frames=frames,
                scan_vq_cache=scan_cache,
            )
            pred_local_T = out["pred_local_T"]
            pred_global = compose_global_from_local(pred_local_T)
            gt_local = local_from_global(gt_global)

            pts_hom = ref_pts.T.to(device=self.device, dtype=torch.float32)

            B, T = pred_local_T.shape[:2]
            for b in range(B):
                sm: dict[str, float] = {}

                # Points-based metrics
                gt_pts_g = (gt_global[b] @ pts_hom)[:, :3, :].permute(0, 2, 1)
                pred_pts_g = (pred_global[b] @ pts_hom)[:, :3, :].permute(0, 2, 1)
                sm["gpe_pts_mm"] = float((pred_pts_g - gt_pts_g).norm(dim=-1).mean().item())
                sm["drift_pts_mm"] = float((pred_pts_g[-1] - gt_pts_g[-1]).norm(dim=-1).mean().item())

                gt_pts_l = (gt_local[b, 1:] @ pts_hom)[:, :3, :].permute(0, 2, 1)
                pred_pts_l = (pred_local_T[b, 1:] @ pts_hom)[:, :3, :].permute(0, 2, 1)
                sm["lpe_pts_mm"] = float((pred_pts_l - gt_pts_l).norm(dim=-1).mean().item())

                # SE(3) translation-only error
                sm["se3_lpe_mm"] = float(
                    (pred_local_T[b, 1:, :3, 3] - gt_local[b, 1:, :3, 3]).norm(dim=-1).mean().item()
                )
                sm["se3_gpe_mm"] = float(
                    (pred_global[b, :, :3, 3] - gt_global[b, :, :3, 3]).norm(dim=-1).mean().item()
                )
                sm["se3_drift_last_mm"] = float(
                    (pred_global[b, -1, :3, 3] - gt_global[b, -1, :3, 3]).norm().item()
                )

                # TUS-REC DDF metrics
                if use_tusrec:
                    try:
                        from trainers.metrics.tusrec import compute_tusrec_metrics  # noqa
                        tusrec = compute_tusrec_metrics(
                            frames=frames[b],
                            gt_transforms=gt_global[b],
                            pred_transforms=pred_global[b],
                            calib={"tform_calib": self.tform_calib},
                            compute_scores=True,
                        )
                        for k, v in tusrec.items():
                            if v is not None:
                                sm[f"tusrec_{k}"] = float(v)
                    except Exception as exc:
                        warnings.warn(f"[evaluate] compute_tusrec_metrics failed: {exc}")

                # VQ diagnostics
                sm["vq_codebook_usage"] = self.model.vq.codebook_usage
                if scan_cache["g"] is not None:
                    sm["summary_g_norm"] = float(
                        scan_cache["g"][b].norm().item() if B > 1 else scan_cache["g"].norm(dim=-1).mean().item()
                    )

                sm["num_frames"] = int(T)
                scan_metrics.append(sm)

        # Restore from EMA
        if self.ema is not None:
            EMA.restore(self.model, backup)

        if not scan_metrics:
            base = {
                "gpe_pts_mm": 0.0, "lpe_pts_mm": 0.0, "drift_pts_mm": 0.0,
                "se3_lpe_mm": 0.0, "se3_gpe_mm": 0.0, "se3_drift_last_mm": 0.0,
            }
            if use_tusrec:
                base.update({"tusrec_GPE_mm": 0.0, "tusrec_LPE_mm": 0.0, "tusrec_final_score": 0.0})
            return base

        # Aggregation
        agg: dict[str, float] = {}
        all_keys = {k for s in scan_metrics for k in s if k != "num_frames"}
        for key in all_keys:
            vals = [s[key] for s in scan_metrics if key in s]
            agg[f"mean_{key}"] = sum(vals) / len(vals)
        agg["num_scans"] = float(len(scan_metrics))

        # Save VQ diagnostic log
        self._save_vq_log(self.epoch, agg)

        return agg

    # ── Checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            payload["ema_shadow"] = self.ema.shadow
        torch.save(payload, path)
        print(f"[VQMemoryTrainer] checkpoint saved → {path} (tag={tag})")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        model_state = state.get("model") or state.get("state_dict") or state
        self.model.load_state_dict(model_state, strict=False)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        if self.ema is not None and "ema_shadow" in state:
            self.ema.shadow = state["ema_shadow"]
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        print(f"[VQMemoryTrainer] checkpoint loaded ← {path} (epoch={self.epoch})")

    # Make model accessible as property for hook compatibility
    def load_full_checkpoint(self, path: str) -> None:
        self.load_checkpoint(path)
