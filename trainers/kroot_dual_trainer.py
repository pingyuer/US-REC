"""K-root Dual Trainer — joint training of Short + Long branches with stitch eval.

Trains two **independent** ``LongSeqPoseModel`` instances simultaneously:
- **Short model**: contiguous k-frame windows, predicts Δ=1 local transforms.
- **Long model**: sparse k-token windows at stride s, predicts Δ=1 (sparse-domain) locals.

Each training step draws one batch from the short loader and one batch from
the long loader, computing separate losses for each model.  Both models use
AdamW with linear warm-up + cosine annealing LR and optional EMA.

Now inherits from :class:`BaseTrainer` for shared boilerplate (hooks,
checkpoint, gradient accumulation, LR schedule, train loop skeleton).

Evaluation runs the full stitch pipeline from ``eval.kroot_stitch``:
long anchors provide the global skeleton, short provides dense refinement,
fused via SE(3) log-space endpoint correction.

Usage::

    trainer = KRootDualTrainer(
        cfg, device=...,
        short_train_loader=..., long_train_loader=..., val_loader=...,
    )
    trainer.train()
    trainer.evaluate(loader)

Single config → single command::

    python main_rec.py --config configs/demo_rec24_ete_kroot_dual.yml
"""

from __future__ import annotations

import copy
import math
import os
import time
import warnings
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.temporal.model_longseq import LongSeqPoseModel
from models.losses.dual_loss import chordal_rotation_loss
from metrics.compose import compose_global_from_local, local_from_global
from eval.kroot_stitch import (
    stitch_long_base_short_refine,
    compute_stitch_metrics,
    export_debug_csv,
)
from trainers.base_trainer import BaseTrainer
from trainers.common import cfg_get, EMA, load_tform_calib, resolve_kroot_stride


# ─── Helper: zip loaders (legacy dual-loader mode) ──────────────────────────

class _ZipCycleLoader:
    """Zip short + long loaders; cycle long if shorter.

    Yields ``{"short": short_batch, "long": long_batch}`` — same format
    as the joint loader so ``_run_step`` sees a uniform interface.
    """

    def __init__(self, short_loader, long_loader):
        self.short = short_loader
        self.long = long_loader

    def __iter__(self):
        long_iter = iter(self.long)
        for short_batch in self.short:
            try:
                long_batch = next(long_iter)
            except StopIteration:
                long_iter = iter(self.long)
                long_batch = next(long_iter)
            yield {"short": short_batch, "long": long_batch}

    def __len__(self):
        return len(self.short)


# ==========================================================================
# KRootDualTrainer
# ==========================================================================

class KRootDualTrainer(BaseTrainer):
    """Joint trainer for both K-root branches (short + long).

    Inherits generic train loop, hooks, checkpoint from :class:`BaseTrainer`.
    Adds: dual-model construction, branch-specific forward, stitch eval, EMA.

    Supports two loader modes:
    1. **Joint loader** (preferred): a single ``joint_train_loader`` that
       yields ``{"short": …, "long": …, "meta": …}`` — one scan read,
       one batch, one backward.
    2. **Legacy dual loaders**: separate ``short_train_loader`` +
       ``long_train_loader``.  Kept for backward compatibility but slower.

    Parameters
    ----------
    cfg : OmegaConf config
    device : str | torch.device
    joint_train_loader : DataLoader for JointKRootDualDataset (preferred)
    short_train_loader : DataLoader for ShortWindowDataset (legacy)
    long_train_loader : DataLoader for LongWindowDataset (legacy)
    val_loader : DataLoader for full-scan validation
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
        super().__init__(cfg, device=device)
        self.joint_train_loader = joint_train_loader
        self.short_train_loader = short_train_loader
        self.long_train_loader = long_train_loader
        self.val_loader = val_loader
        self._use_joint = joint_train_loader is not None

        # Signal to main_rec.py / build_hooks that this trainer has its own
        # evaluate() that understands the joint {"short": …, "long": …} batch
        # format and must NOT be dispatched to the generic RecEvaluator.
        self.fusion_mode = "stitch"

        # ── K-root params ────────────────────────────────────────────
        self.k = int(cfg_get(cfg, "kroot.k", 64))
        self.s = resolve_kroot_stride(cfg, k=self.k)

        # ── Models ───────────────────────────────────────────────────
        self.rotation_rep = str(cfg_get(cfg, "model.pose_head.rotation_rep", "rot6d"))
        self.short_model = self._build_model("short").to(self.device)
        self.long_model = self._build_model("long").to(self.device)

        # ── Optimizers (separate for each model) ─────────────────────
        lr = float(cfg_get(cfg, "optimizer.lr_rec", cfg_get(cfg, "optimizer.lr", 5e-4)))
        weight_decay = float(cfg_get(cfg, "optimizer.weight_decay", 1e-2))
        betas = tuple(cfg_get(cfg, "optimizer.betas", (0.9, 0.999)))
        self.base_lr = lr
        self.min_lr = float(cfg_get(cfg, "optimizer.min_lr", 1e-6))

        encoder_lr_mult = float(cfg_get(cfg, "optimizer.encoder_lr_mult", 0.1))

        def _make_optimizer(model: nn.Module) -> torch.optim.AdamW:
            encoder_params = [p for n, p in model.named_parameters()
                              if n.startswith("encoder.") and p.requires_grad]
            other_params = [p for n, p in model.named_parameters()
                           if not n.startswith("encoder.") and p.requires_grad]
            groups = []
            if encoder_params:
                groups.append({"params": encoder_params, "lr": lr * encoder_lr_mult, "name": "encoder"})
            if other_params:
                groups.append({"params": other_params, "lr": lr, "name": "tf_heads"})
            return torch.optim.AdamW(groups, weight_decay=weight_decay, betas=betas)

        self.short_optimizer = _make_optimizer(self.short_model)
        self.long_optimizer = _make_optimizer(self.long_model)

        # ── Loss config ──────────────────────────────────────────────
        self.rot_weight = float(cfg_get(cfg, "loss.rot_weight", 1.0))
        self.trans_weight = float(cfg_get(cfg, "loss.trans_weight", 1.0))
        self.short_loss_weight = float(cfg_get(cfg, "loss.short_weight", 1.0))
        self.long_loss_weight = float(cfg_get(cfg, "loss.long_weight", 1.0))

        # ── Override grad_accum default (KRootDual historically uses 4) ──
        self.grad_accum = max(1, int(cfg_get(cfg, "trainer.grad_accum", 4) or 4))

        # ── EMA ──────────────────────────────────────────────────────
        ema_decay = float(cfg_get(cfg, "trainer.ema_decay", 0.999))
        self.short_ema = EMA(self.short_model, decay=ema_decay) if ema_decay > 0 else None
        self.long_ema = EMA(self.long_model, decay=ema_decay) if ema_decay > 0 else None

        # ── Stitch config ────────────────────────────────────────────
        self.enable_endpoint_interp = bool(cfg_get(cfg, "stitch.enable_endpoint_interp", True))
        self.stitch_short_overlap = int(cfg_get(cfg, "stitch.short_overlap", 8))
        self.stitch_long_window_stride = int(
            cfg_get(cfg, "stitch.long_window_stride", max(1, self.k - 1))
        )
        # ── Eval diagnostics & pose-graph backend ─────────────────────
        # diagnostics_level:
        #   0 = silent, 1 = shape/NaN/divergence checks,
        #   2 = level-1 + GT-baseline score,  3 = level-2 + DDF direction
        self.diagnostics_level = int(cfg_get(cfg, "eval.diagnostics_level", 1))
        # pose_graph_refine: run GN pose-graph as additional post-processing
        self.pose_graph_refine = bool(cfg_get(cfg, "eval.pose_graph_refine", False))
        self.pg_w_short = float(cfg_get(cfg, "eval.pg_w_short", 1.0))
        self.pg_w_long = float(cfg_get(cfg, "eval.pg_w_long", 0.3))
        self.pg_n_iters = int(cfg_get(cfg, "eval.pg_n_iters", 5))
        # ── Calibration ─────────────────────────────────────────────
        self.tform_calib: torch.Tensor | None = load_tform_calib(
            cfg, device=self.device, warn_prefix="KRootDual",
        )

    # ── Model construction ───────────────────────────────────────────

    def _build_model(self, branch: str) -> LongSeqPoseModel:
        """Build a LongSeqPoseModel for the given branch.

        Config lookup order:  model.<branch>.transformer.* → model.transformer.*
        """
        backbone = str(cfg_get(self.cfg, "model.encoder.backbone", "efficientnet_b0"))
        pretrained = bool(cfg_get(self.cfg, "model.encoder.pretrained", False))

        def _tf(key, default):
            # Try branch-specific first, then shared
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
            aux_intervals=[],
            pretrained_backbone=pretrained,
            memory_size=0,
        )

    # ── Hooks: inherited from BaseTrainer ─────────────────────────────

    # ── Checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str, *, tag: str = "manual") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "short_model": self.short_model.state_dict(),
            "long_model": self.long_model.state_dict(),
            "short_optimizer": self.short_optimizer.state_dict(),
            "long_optimizer": self.long_optimizer.state_dict(),
        }
        if self.short_ema is not None:
            payload["short_ema_shadow"] = self.short_ema.shadow
        if self.long_ema is not None:
            payload["long_ema_shadow"] = self.long_ema.shadow
        torch.save(payload, path)
        print(f"[KRootDual] checkpoint saved → {path}  (tag={tag})")

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        if "short_model" in state:
            self.short_model.load_state_dict(state["short_model"])
        if "long_model" in state:
            self.long_model.load_state_dict(state["long_model"])
        if "short_optimizer" in state:
            try:
                self.short_optimizer.load_state_dict(state["short_optimizer"])
            except Exception:
                pass
        if "long_optimizer" in state:
            try:
                self.long_optimizer.load_state_dict(state["long_optimizer"])
            except Exception:
                pass
        if "short_ema_shadow" in state and self.short_ema is not None:
            self.short_ema.shadow = state["short_ema_shadow"]
        if "long_ema_shadow" in state and self.long_ema is not None:
            self.long_ema.shadow = state["long_ema_shadow"]
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        print(f"[KRootDual] checkpoint loaded ← {path}  (epoch={self.epoch}, step={self.global_step})")

    # Also support the standard `model` key for compatibility with the checkpoint hook
    @property
    def model(self):
        """Expose short_model as the primary 'model' for CheckpointHook compatibility.

        The full save/load uses save_checkpoint/load_checkpoint above,
        but single-model hooks (CheckpointHook) need a `.model` attr.
        We wrap both models into a ModuleDict so state_dict includes both.
        """
        if not hasattr(self, "_model_wrapper"):
            self._model_wrapper = nn.ModuleDict({
                "short": self.short_model,
                "long": self.long_model,
            })
        return self._model_wrapper

    # ── Single forward + loss for one branch ─────────────────────────

    def _forward_branch(
        self,
        model: LongSeqPoseModel,
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass + chordal-rot + smooth-L1-trans loss for one branch.

        Dtype promotion (uint8 → float32, /255) happens here on GPU via
        non_blocking transfer to overlap CPU↔GPU copies with compute.
        """
        frames = batch["frames"].to(self.device, non_blocking=True)
        gt_global = batch["gt_global_T"].to(self.device, non_blocking=True)

        # Promote uint8 → float on GPU (avoids CPU .float().clone())
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.1:
            frames = frames / 255.0

        if "idx_long" in batch:
            pos_ids = batch["idx_long"].to(self.device, non_blocking=True).long()
        elif "frame_pos" in batch:
            pos_ids = batch["frame_pos"].to(self.device, non_blocking=True).long()
        else:
            pos_ids = None

        # V1 LongSeqPoseModel accepts (frames, memory, pos_offset).
        # V2 V2LongSeqPoseModel accepts (frames, position_ids).
        # Dispatch based on model signature.
        import inspect as _inspect
        sig = _inspect.signature(model.forward)
        if "position_ids" in sig.parameters:
            out = model(frames, position_ids=pos_ids)
        else:
            out = model(frames)
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

    # ── BaseTrainer overrides ────────────────────────────────────────

    def _models_and_optimizers(self) -> Sequence[tuple[nn.Module, torch.optim.Optimizer]]:
        return [
            (self.short_model, self.short_optimizer),
            (self.long_model, self.long_optimizer),
        ]

    def _get_train_loader(self):
        """Return a single iterable that yields ``{"short": …, "long": …}``.

        In joint mode: the joint loader already provides this format.
        In legacy mode: wraps both loaders with :class:`_ZipCycleLoader`.
        """
        if self._use_joint:
            return self.joint_train_loader
        if self.short_train_loader is not None and self.long_train_loader is not None:
            return _ZipCycleLoader(self.short_train_loader, self.long_train_loader)
        return None

    def _get_val_loader(self):
        return self.val_loader

    def _on_epoch_start(self, epoch: int) -> None:
        active_loaders = (
            [self.joint_train_loader] if self._use_joint
            else [self.short_train_loader, self.long_train_loader]
        )
        for loader in active_loaders:
            if loader is None:
                continue
            ds = getattr(loader, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def _run_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward + loss on both branches for a single micro-batch.

        ``batch`` is ``{"short": …, "long": …}`` regardless of loader mode.
        Returns ``(total_loss, combined_metrics)`` where *total_loss* is
        ``short_weight * short_loss + long_weight * long_loss``.
        """
        short_batch = batch["short"]
        long_batch = batch["long"]

        short_loss, sm = self._forward_branch(self.short_model, short_batch)
        long_loss, lm = self._forward_branch(self.long_model, long_batch)

        total_loss = self.short_loss_weight * short_loss + self.long_loss_weight * long_loss

        metrics = {
            "short_loss": float(short_loss.item()),
            "long_loss": float(long_loss.item()),
            "short_rot": sm["rot_loss"],
            "short_trans": sm["trans_loss"],
            "short_drift_mm": sm["drift_mm"],
            "long_rot": lm["rot_loss"],
            "long_trans": lm["trans_loss"],
            "long_drift_mm": lm["drift_mm"],
        }
        return total_loss, metrics

    def _after_optim_step(self) -> None:
        if self.short_ema is not None:
            self.short_ema.update(self.short_model)
        if self.long_ema is not None:
            self.long_ema.update(self.long_model)

    def _format_step_log(
        self, epoch: int, n_optim_steps: int, avg_accum: float,
        metrics: dict[str, float], current_lr: float,
    ) -> str:
        m = metrics
        return (
            f"[KRootDual epoch {epoch}  step {n_optim_steps}]  "
            f"short={m.get('short_loss', 0):.4f}(rot={m.get('short_rot', 0):.4f} "
            f"trans={m.get('short_trans', 0):.4f} "
            f"drift={m.get('short_drift_mm', 0):.1f}mm)  "
            f"long={m.get('long_loss', 0):.4f}(rot={m.get('long_rot', 0):.4f} "
            f"trans={m.get('long_trans', 0):.4f} "
            f"drift={m.get('long_drift_mm', 0):.1f}mm)  "
            f"lr={current_lr:.2e}"
        )

    # ── Evaluation with stitch fusion ────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader=None) -> dict[str, float]:
        """Evaluate both models with stitch fusion on full-scan data.

        Automatically runs pipeline diagnostics (level = self.diagnostics_level)
        and optionally applies a pose-graph refinement pass.
        """
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return {}

        # ── Log eval split provenance ──────────────────────────────
        ds = getattr(loader, "dataset", None)
        ds_cls = type(ds).__name__ if ds is not None else "?"
        print(
            f"[evaluate] split=val  dataset_class={ds_cls}  "
            f"diagnostics_level={self.diagnostics_level}  "
            f"pose_graph_refine={self.pose_graph_refine}"
        )

        # Apply EMA weights
        short_backup = long_backup = None
        if self.short_ema is not None:
            short_backup = self.short_ema.apply(self.short_model)
        if self.long_ema is not None:
            long_backup = self.long_ema.apply(self.long_model)

        self.short_model.eval()
        self.long_model.eval()

        all_metrics: list[dict[str, float]] = []
        scan_globals: dict[str, dict] = {}
        scan_count = 0
        first_scan_diagnosed = False  # run full diagnostics on first scan only

        for batch in loader:
            frames = batch["frames"].to(self.device)        # (1, T, H, W)
            gt_global = batch["gt_global_T"].to(self.device) # (1, T, 4, 4)

            B = frames.shape[0]
            for b in range(B):
                scan_frames = frames[b]      # (T, H, W)
                gt_g = gt_global[b]          # (T, 4, 4)

                result = stitch_long_base_short_refine(
                    short_model=self.short_model,
                    long_model=self.long_model,
                    scan_frames=scan_frames,
                    k=self.k,
                    s=self.s,
                    device=self.device,
                    short_overlap=self.stitch_short_overlap,
                    long_window_stride=self.stitch_long_window_stride,
                    enable_endpoint_interp=self.enable_endpoint_interp,
                )

                # ── Optional: pose-graph refinement ─────────────────
                if self.pose_graph_refine:
                    try:
                        from eval.pose_graph import pose_graph_refine as _pg  # noqa: PLC0415
                        fused_pg = _pg(
                            short_local=result["short_local"],
                            long_local=result["long_local"],
                            anchor_indices=result["anchor_indices"],
                            init_global=result["fused_global"],
                            w_short=self.pg_w_short,
                            w_long=self.pg_w_long,
                            n_iters=self.pg_n_iters,
                        )
                        result["fused_pg_global"] = fused_pg
                    except Exception as _pg_exc:
                        warnings.warn(f"[evaluate] pose_graph failed: {_pg_exc}")

                # ── Diagnostics (full on scan 0; summary only thereafter) ──
                if self.diagnostics_level > 0:
                    _diag_level = self.diagnostics_level if not first_scan_diagnosed else 1
                    meta = batch.get("meta")
                    _sid = f"scan_{scan_count}"
                    if isinstance(meta, dict):
                        raw = meta.get("scan_id") or meta.get("scan_name")
                        if isinstance(raw, (list, tuple)):
                            _sid = str(raw[b]) if b < len(raw) else str(raw[0])
                        elif raw is not None:
                            _sid = str(raw)
                    try:
                        from eval.diagnostics import run_pipeline_diagnostics  # noqa: PLC0415
                        diag = run_pipeline_diagnostics(
                            fused_global=result["fused_global"],
                            short_global=result["short_global"],
                            long_global=result["long_global"],
                            gt_global=gt_g,
                            anchor_indices=result["anchor_indices"],
                            tform_calib=self.tform_calib,
                            scan_id=_sid,
                            diagnostics_level=_diag_level,
                            frames=scan_frames if _diag_level >= 2 else None,
                        )
                        print(diag["summary"])
                        first_scan_diagnosed = True
                    except Exception as _diag_exc:
                        warnings.warn(f"[evaluate] diagnostics failed: {_diag_exc}")

                metrics = compute_stitch_metrics(
                    fused_global=result["fused_global"],
                    short_global=result["short_global"],
                    long_global=result["long_global"],
                    anchor_indices=result["anchor_indices"],
                    gt_global=gt_g,
                    tform_calib=self.tform_calib,
                    frames=scan_frames if self.tform_calib is not None else None,
                )

                # If pose-graph ran, also compute its metrics
                if "fused_pg_global" in result:
                    try:
                        from eval.kroot_stitch import compute_stitch_metrics as _csm  # noqa
                        pg_met = _csm(
                            fused_global=result["fused_pg_global"],
                            short_global=result["short_global"],
                            long_global=result["long_global"],
                            anchor_indices=result["anchor_indices"],
                            gt_global=gt_g,
                            tform_calib=self.tform_calib,
                            frames=scan_frames if self.tform_calib is not None else None,
                        )
                        # Prefix pg_ to avoid overwriting stitch metrics
                        for k_m, v_m in pg_met.items():
                            if isinstance(v_m, float):
                                metrics[f"pg_{k_m}"] = v_m
                    except Exception as _exc:
                        warnings.warn(f"[evaluate] pg metrics failed: {_exc}")

                all_metrics.append(metrics)

                # Store per-scan globals for VizHook
                meta = batch.get("meta")
                sid = f"scan_{scan_count}"
                if isinstance(meta, dict):
                    raw = meta.get("scan_id") or meta.get("scan_name")
                    if isinstance(raw, (list, tuple)):
                        sid = str(raw[b]) if b < len(raw) else str(raw[0])
                    elif raw is not None:
                        sid = str(raw)
                scan_globals[sid] = {
                    "pred": result["fused_global"].detach().cpu(),
                    "gt": gt_g.detach().cpu(),
                }
                scan_count += 1

        print(f"[evaluate] finished  num_scans={scan_count}  (metric accumulators are local to this call)")

        # Restore original (non-EMA) weights
        if short_backup is not None:
            EMA.restore(self.short_model, short_backup)
        if long_backup is not None:
            EMA.restore(self.long_model, long_backup)

        if not all_metrics:
            return {"mean_gpe_mm_fused": 0.0}

        # Aggregate
        agg: dict[str, Any] = {}
        float_keys = sorted({k for m in all_metrics for k in m if isinstance(m[k], (int, float))})
        for key in float_keys:
            vals = [m[key] for m in all_metrics if key in m]
            if vals:
                agg[f"mean_{key}"] = sum(vals) / len(vals)

        agg["num_scans"] = float(scan_count)
        agg["scan_globals"] = scan_globals  # type: ignore

        # Print compact divergence summary across all scans
        for metric in (
            "mean_divergence_fused_vs_short_t_mm",
            "mean_divergence_fused_vs_long_t_mm",
            "mean_gpe_mm_fused",
            "mean_tusrec_GPE_mm_fused",
            "mean_tusrec_LPE_mm_fused",
            "mean_tusrec_final_score_fused",
        ):
            if metric in agg:
                print(f"[evaluate] {metric}: {agg[metric]:.4f}")

        return agg
