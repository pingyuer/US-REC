"""VizHook — write visualisation artefacts after each evaluation run.

This hook bridges the ``viz.*`` modules into the training / evaluation pipeline.
It works in two modes:

1. **RecEvaluator callback** (eval-only path): implements ``on_end`` which is
   called by :meth:`RecEvaluator.run` after all batches are processed.

2. **Trainer hook** (training path): implements ``after_val`` / ``after_test``
   which are fired by the trainer after the evaluator returns.

In both cases, per-scan global transforms are read from
``metrics["scan_globals"]`` (populated by :class:`RecEvaluator`).

Config key: ``viz``

Example YAML::

    viz:
      enabled: true
      drift_curve: true      # GPE/LPE per-frame curve
      pose_curve: true       # translation & rotation norm vs frame
      recon_slices: false    # axial/sagittal/coronal slices (needs volume)
      save_png: true
      save_csv: true
      out_dir: null          # defaults to <run_dir>/viz/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from .base_hook import Hook
from .registry import register_hook


@register_hook("VizHook")
class VizHook(Hook):
    """Write drift curves, pose curves, and reconstruction slices after eval."""

    priority = 60  # runs after MLflowHook (80) saves metrics

    def __init__(
        self,
        *,
        out_dir: Union[str, Path, None] = None,
        drift_curve: bool = True,
        pose_curve: bool = True,
        recon_slices: bool = False,
        save_png: bool = True,
        save_csv: bool = True,
        trainer: Any = None,
    ) -> None:
        self.out_dir = Path(out_dir) if out_dir else None
        self.drift_curve = drift_curve
        self.pose_curve = pose_curve
        self.recon_slices = recon_slices
        self.save_png = save_png
        self.save_csv = save_csv
        self.trainer = trainer  # set by build_hooks or main_rec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_base_out_dir(self, trainer_obj: Any, mode: str) -> Path:
        if self.out_dir:
            return self.out_dir / mode
        if trainer_obj is not None and hasattr(trainer_obj, "save_path"):
            return Path(trainer_obj.save_path) / "viz" / mode
        return Path("viz_output") / mode

    def _run_viz(
        self,
        metrics: dict,
        trainer_obj: Any,
        mode: str,
        epoch: Optional[int],
    ) -> None:
        scan_globals: dict | None = metrics.get("scan_globals")
        if not scan_globals:
            return

        base_out = self._get_base_out_dir(trainer_obj, mode)
        epoch_tag = f"epoch{epoch:04d}" if epoch is not None else "eval"

        tform_calib = getattr(trainer_obj, "tform_calib", None) if trainer_obj else None
        image_points = getattr(trainer_obj, "image_points", None) if trainer_obj else None

        written_dirs: list[str] = []

        for sid, sg in scan_globals.items():
            pred_global = sg["pred"]
            gt_global = sg["gt"]
            out_dir = base_out / epoch_tag / sid.replace("/", "_")

            # ---- pose curve (pred only) --------------------------------
            if self.pose_curve:
                try:
                    from viz.pose_curve import export_pose_curve
                    result = export_pose_curve(
                        pred_global,
                        out_dir=out_dir,
                        scan_id=sid,
                        save_png=self.save_png,
                        save_csv=self.save_csv,
                    )
                    written_dirs.append(str(out_dir))
                except Exception as exc:
                    print(f"[VizHook] pose_curve failed for '{sid}': {exc}")

            # ---- drift curve (pred vs gt) ------------------------------
            if self.drift_curve and tform_calib is not None:
                try:
                    from viz.drift_curve import export_drift_curve
                    export_drift_curve(
                        pred_global,
                        gt_global,
                        tform_calib,
                        image_points,
                        out_dir=out_dir,
                        scan_id=sid,
                        save_png=self.save_png,
                        save_csv=self.save_csv,
                    )
                except Exception as exc:
                    print(f"[VizHook] drift_curve failed for '{sid}': {exc}")
            elif self.drift_curve and tform_calib is None:
                print("[VizHook] drift_curve skipped — trainer.tform_calib not available")

            # ---- recon slices (optional; needs volume in scan_globals) --
            if self.recon_slices and "pred_volume" in sg:
                try:
                    from viz.recon_slices import export_recon_slices
                    export_recon_slices(
                        sg["pred_volume"],
                        gt_volume=sg.get("gt_volume"),
                        out_dir=out_dir,
                        scan_id=sid,
                    )
                except Exception as exc:
                    print(f"[VizHook] recon_slices failed for '{sid}': {exc}")

        if written_dirs:
            unique = sorted(set(written_dirs))
            print(f"[VizHook] Visualisations written → {'; '.join(unique)}")

    # ------------------------------------------------------------------
    # RecEvaluator callback interface  (on_start / on_batch / on_end)
    # ------------------------------------------------------------------

    def on_end(
        self,
        *,
        metrics: dict,
        mode: str = "val",
        epoch: Optional[int] = None,
        ctx: Any = None,
        loader: Any = None,
    ) -> None:
        self._run_viz(metrics, self.trainer, mode, epoch)

    # ------------------------------------------------------------------
    # Trainer hook interface  (after_val / after_test)
    # ------------------------------------------------------------------

    def after_val(self, trainer: Any, log_buffer: Any = None) -> None:
        metrics = getattr(trainer, "last_eval_metrics", {}) or {}
        self._run_viz(
            metrics,
            trainer,
            mode="val",
            epoch=getattr(trainer, "epoch", None),
        )

    def after_test(self, trainer: Any, log_buffer: Any = None) -> None:
        metrics = getattr(trainer, "last_eval_metrics", {}) or {}
        self._run_viz(metrics, trainer, mode="test", epoch=None)
