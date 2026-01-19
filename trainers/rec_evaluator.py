from __future__ import annotations

from typing import Any, Optional

import torch

from trainers.metrics import (
    translation_error,
    rotation_error,
    se3_error,
    cumulative_drift,
    loop_closure_error,
    volume_ncc,
    volume_ssim,
    volume_dice,
)
from trainers.utils.forward_utils import (
    unpack_batch,
    build_pred_transforms,
    points_from_transforms,
    convpose_if_needed,
)
from trainers.utils.loss import compute_loss
from trainers.utils.interp_reg import scatter_pts_interpolation, scatter_pts_registration
from utils.funcs import wrapped_pred_dist


class RecEvaluator:
    """
    Evaluation loop for reconstruction/registration models.

    Expects batches with keys: frames, tforms, tforms_inv.
    """

    def __init__(self, *, device: str | torch.device):
        self.device = device

    @torch.no_grad()
    def run(
        self,
        *,
        model,
        voxel_morph_net=None,
        loader,
        cfg: Any,
        trainer: Any,
        mode: str = "val",
        epoch: Optional[int] = None,
        ctx: Optional[Any] = None,
        callbacks: Optional[list[Any]] = None,
    ) -> dict[str, float]:
        if loader is None:
            raise ValueError(f"RecEvaluator.run() requires a loader (mode={mode})")

        model.eval()
        if voxel_morph_net is not None:
            voxel_morph_net.eval()

        callbacks = list(callbacks or [])
        for cb in callbacks:
            fn = getattr(cb, "on_start", None)
            if callable(fn):
                fn(mode=mode, epoch=epoch, ctx=ctx, loader=loader)

        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        loss_sums = {
            "loss": 0.0,
            "loss_rec": 0.0,
            "loss_reg": 0.0,
            "dist": 0.0,
            "wrap_dist": 0.0,
        }
        loss_count = 0

        for batch in loader:
            frames, tforms, tforms_inv = unpack_batch(batch)
            frames = frames.to(self.device)
            tforms = tforms.to(self.device)
            tforms_inv = tforms_inv.to(self.device)

            tforms_each_frame2frame0 = trainer.transform_label(tforms, tforms_inv)
            frames = frames / 255

            outputs = model(frames)
            pred_transfs = build_pred_transforms(trainer.transform_prediction, outputs, self.device)
            labels, pred_pts = points_from_transforms(
                img_pro_coord=trainer.img_pro_coord,
                tform_calib_R_T=trainer.tform_calib_R_T,
                tform_calib=trainer.tform_calib,
                image_points=trainer.image_points,
                tforms_each_frame2frame0=tforms_each_frame2frame0,
                pred_transfs=pred_transfs,
            )
            labels, pred_pts, convR_batched, minxyz_all = convpose_if_needed(
                conv_coords=trainer.Conv_Coords,
                img_pro_coord=trainer.img_pro_coord,
                tforms_each_frame2frame0=tforms_each_frame2frame0,
                pred_transfs=pred_transfs,
                tform_calib=trainer.tform_calib,
                image_points=trainer.image_points,
                labels=labels,
                pred_pts=pred_pts,
                device=self.device,
            )

            loss, loss1, loss2, dist, wrap_dist, extras = compute_loss(
                loss_type=trainer.Loss_type,
                labels=labels,
                pred_pts=pred_pts,
                frames=frames,
                step=0,
                criterion=trainer.criterion,
                img_loss=trainer.img_loss,
                regularization=trainer.regularization,
                reg_loss_weight=trainer.reg_loss_weight,
                ddf_dirc=trainer.ddf_dirc,
                conv_coords=trainer.Conv_Coords,
                option=trainer.option,
                device=self.device,
                scatter_pts_registration=lambda labels, pred_pts, frames, step: scatter_pts_registration(
                    labels=labels,
                    pred_pts=pred_pts,
                    frames=frames,
                    step=step,
                    device=self.device,
                    option=trainer.option,
                    intepoletion_method=trainer.intepoletion_method,
                    intepoletion_volume=trainer.intepoletion_volume,
                    voxel_morph_net=voxel_morph_net or trainer.VoxelMorph_net,
                    use_deform=bool(getattr(trainer, "use_deform", False)),
                    enable_voxel=bool(getattr(trainer, "enable_voxel", False)),
                ),
                scatter_pts_interpolation=lambda labels, pred_pts, frames, step: scatter_pts_interpolation(
                    labels=labels,
                    pred_pts=pred_pts,
                    frames=frames,
                    step=step,
                    device=self.device,
                    option=trainer.option,
                    intepoletion_method=trainer.intepoletion_method,
                    intepoletion_volume=trainer.intepoletion_volume,
                    enable_voxel=bool(getattr(trainer, "enable_voxel", False)),
                ),
                wrapped_pred_dist_fn=wrapped_pred_dist,
                convR_batched=convR_batched,
                minxyz_all=minxyz_all,
                rigid_only=not bool(getattr(trainer, "nonrigid_enabled", False)),
            )
            loss_sums["loss"] += float(loss.item())
            loss_sums["loss_rec"] += float(loss1.item())
            loss_sums["loss_reg"] += float(loss2.item())
            loss_sums["dist"] += float(dist.item())
            loss_sums["wrap_dist"] += float(wrap_dist.item())
            loss_count += 1

            metrics = {}
            trans_err = translation_error(pred_transfs[..., :3, 3], tforms_each_frame2frame0[..., :3, 3])
            rot_err = rotation_error(pred_transfs[..., :3, :3], tforms_each_frame2frame0[..., :3, :3])
            se3_err = se3_error(pred_transfs, tforms_each_frame2frame0)
            metrics["translation_error"] = trans_err.mean().item()
            metrics["rotation_error"] = rot_err.mean().item()
            metrics["se3_error"] = se3_err.mean().item()

            drift = cumulative_drift(pred_transfs, tforms_each_frame2frame0)
            loop_err = loop_closure_error(pred_transfs, tforms_each_frame2frame0)
            metrics["cumulative_drift"] = float(drift.mean().item())
            metrics["loop_closure_error"] = float(loop_err.mean().item())

            if extras.get("pred_volume") is not None and extras.get("gt_volume") is not None:
                pred_volume = extras["pred_volume"]
                gt_volume = extras["gt_volume"]
                metrics["volume_ncc"] = float(volume_ncc(pred_volume, gt_volume).mean().item())
                metrics["volume_ssim"] = float(volume_ssim(pred_volume, gt_volume).mean().item())
                metrics["volume_dice"] = float(volume_dice(pred_volume, gt_volume).mean().item())

            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_counts[key] = metric_counts.get(key, 0) + 1

            if callbacks:
                dataset = getattr(loader, "dataset", None)
                for cb in callbacks:
                    fn = getattr(cb, "on_batch", None)
                    if callable(fn):
                        fn(
                            batch=batch,
                            outputs=pred_transfs,
                            masks=None,
                            dataset=dataset,
                            mode=mode,
                            epoch=epoch,
                        )
                if any(
                    bool(getattr(cb, "request_stop", False)) and bool(getattr(cb, "done", False))
                    for cb in callbacks
                ):
                    break

        avg_metrics = {
            key: (metric_sums[key] / max(metric_counts.get(key, 1), 1))
            for key in metric_sums
        }
        if loss_count > 0:
            avg_metrics[f"{mode}_loss"] = loss_sums["loss"] / loss_count
            avg_metrics[f"{mode}_loss_rec"] = loss_sums["loss_rec"] / loss_count
            avg_metrics[f"{mode}_loss_reg"] = loss_sums["loss_reg"] / loss_count
            avg_metrics[f"{mode}_dist"] = loss_sums["dist"] / loss_count
            avg_metrics[f"{mode}_wrap_dist"] = loss_sums["wrap_dist"] / loss_count

        for cb in callbacks:
            fn = getattr(cb, "on_end", None)
            if callable(fn):
                fn(metrics=avg_metrics, mode=mode, epoch=epoch, ctx=ctx, loader=loader)
        return avg_metrics
