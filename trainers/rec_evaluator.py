from __future__ import annotations

from typing import Any, Optional
import time

import torch

from trainers.metrics import (
    end_to_start_rpe_rotation_deg,
    end_to_start_rpe_translation_mm,
    endpoint_rpe_rotation_deg,
    endpoint_rpe_translation_mm,
    rotation_error_deg,
    se3_rotation_error_deg,
    se3_translation_error,
    translation_error_mm,
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
from utils.metrics.tusrec_metrics import compute_tusrec_metrics
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
        tusrec_sums: dict[str, float] = {}
        tusrec_counts: dict[str, int] = {}
        tusrec_rows: list[dict[str, Any]] = []
        loss_sums = {
            "loss": 0.0,
            "loss_rec": 0.0,
            "loss_reg": 0.0,
            "dist": 0.0,
            "wrap_dist": 0.0,
        }
        loss_count = 0

        sample_index = 0
        for batch in loader:
            frames, tforms, tforms_inv = unpack_batch(batch)
            frames = frames.to(self.device)
            tforms = tforms.to(self.device)
            tforms_inv = tforms_inv.to(self.device)

            tforms_each_frame2frame0 = trainer.transform_label(tforms, tforms_inv)
            frames = frames / 255

            start_time = time.perf_counter()
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
            trans_err = translation_error_mm(
                pred_transfs[..., :3, 3], tforms_each_frame2frame0[..., :3, 3]
            )
            rot_err = rotation_error_deg(pred_transfs[..., :3, :3], tforms_each_frame2frame0[..., :3, :3])
            se3_trans = se3_translation_error(pred_transfs, tforms_each_frame2frame0)
            se3_rot = se3_rotation_error_deg(pred_transfs, tforms_each_frame2frame0)
            metrics["translation_error_mm"] = float(trans_err.mean().item())
            metrics["rotation_error_deg"] = float(rot_err.mean().item())
            metrics["se3_trans_mm"] = float(se3_trans.mean().item())
            metrics["se3_rot_deg"] = float(se3_rot.mean().item())

            drift_t = endpoint_rpe_translation_mm(pred_transfs, tforms_each_frame2frame0)
            drift_r = endpoint_rpe_rotation_deg(pred_transfs, tforms_each_frame2frame0)
            loop_t = end_to_start_rpe_translation_mm(pred_transfs, tforms_each_frame2frame0)
            loop_r = end_to_start_rpe_rotation_deg(pred_transfs, tforms_each_frame2frame0)
            metrics["endpoint_rpe_mm"] = float(drift_t.mean().item())
            metrics["endpoint_rpe_deg"] = float(drift_r.mean().item())
            metrics["end_to_start_rpe_mm"] = float(loop_t.mean().item())
            metrics["end_to_start_rpe_deg"] = float(loop_r.mean().item())

            if extras.get("pred_volume") is not None and extras.get("gt_volume") is not None:
                pred_volume = extras["pred_volume"]
                gt_volume = extras["gt_volume"]
                metrics["volume_ncc"] = float(volume_ncc(pred_volume, gt_volume).mean().item())
                metrics["volume_ssim"] = float(volume_ssim(pred_volume, gt_volume).mean().item())
                metrics["volume_dice"] = float(volume_dice(pred_volume, gt_volume).mean().item())

            batch_size = int(frames.shape[0]) if frames.ndim >= 4 else 1
            batch_start_index = len(tusrec_rows)
            for idx in range(batch_size):
                frame_seq = frames[idx] if frames.ndim >= 4 else frames
                gt_seq = tforms_each_frame2frame0[idx] if tforms_each_frame2frame0.ndim == 4 else tforms_each_frame2frame0
                pred_seq = pred_transfs[idx] if pred_transfs.ndim == 4 else pred_transfs
                landmarks = None
                scan_id = None
                if isinstance(batch, dict):
                    if "landmarks" in batch:
                        lm = batch["landmarks"]
                        if torch.is_tensor(lm):
                            landmarks = lm[idx] if lm.ndim >= 3 else lm
                    if "scan_id" in batch:
                        scan_id = batch["scan_id"][idx] if isinstance(batch["scan_id"], list) else batch["scan_id"]
                    elif "meta" in batch:
                        meta = batch["meta"]
                        if isinstance(meta, list) and idx < len(meta) and isinstance(meta[idx], dict):
                            scan_id = meta[idx].get("scan_id") or meta[idx].get("scan_name")
                        elif isinstance(meta, dict):
                            scan_id = meta.get("scan_id") or meta.get("scan_name")
                tusrec = compute_tusrec_metrics(
                    frames=frame_seq,
                    gt_transforms=gt_seq,
                    pred_transforms=pred_seq,
                    calib={
                        "tform_calib": trainer.tform_calib,
                        "spacing_mm": getattr(trainer, "tform_calib_scale", None),
                    },
                    landmarks=landmarks,
                )
                row = {"scan_id": scan_id or f"sample_{sample_index}"}
                row.update({k: v for k, v in tusrec.items() if v is not None})
                tusrec_rows.append(row)
                for key, value in tusrec.items():
                    if value is None:
                        continue
                    tusrec_sums[key] = tusrec_sums.get(key, 0.0) + float(value)
                    tusrec_counts[key] = tusrec_counts.get(key, 0) + 1
                sample_index += 1

            elapsed = time.perf_counter() - start_time
            runtime_per_scan = elapsed / max(batch_size, 1)
            for row in tusrec_rows[batch_start_index:]:
                row["runtime_s_per_scan"] = runtime_per_scan
            tusrec_sums["runtime_s_per_scan"] = tusrec_sums.get("runtime_s_per_scan", 0.0) + float(runtime_per_scan) * batch_size
            tusrec_counts["runtime_s_per_scan"] = tusrec_counts.get("runtime_s_per_scan", 0) + int(batch_size)

            trans_err_flat = trans_err.reshape(-1)
            rot_err_flat = rot_err.reshape(-1)
            se3_trans_flat = se3_trans.reshape(-1)
            se3_rot_flat = se3_rot.reshape(-1)
            metric_sums["translation_error_mm"] = metric_sums.get("translation_error_mm", 0.0) + float(trans_err_flat.sum().item())
            metric_counts["translation_error_mm"] = metric_counts.get("translation_error_mm", 0) + int(trans_err_flat.numel())
            metric_sums["rotation_error_deg"] = metric_sums.get("rotation_error_deg", 0.0) + float(rot_err_flat.sum().item())
            metric_counts["rotation_error_deg"] = metric_counts.get("rotation_error_deg", 0) + int(rot_err_flat.numel())
            metric_sums["se3_trans_mm"] = metric_sums.get("se3_trans_mm", 0.0) + float(se3_trans_flat.sum().item())
            metric_counts["se3_trans_mm"] = metric_counts.get("se3_trans_mm", 0) + int(se3_trans_flat.numel())
            metric_sums["se3_rot_deg"] = metric_sums.get("se3_rot_deg", 0.0) + float(se3_rot_flat.sum().item())
            metric_counts["se3_rot_deg"] = metric_counts.get("se3_rot_deg", 0) + int(se3_rot_flat.numel())
            metric_sums["endpoint_rpe_mm"] = metric_sums.get("endpoint_rpe_mm", 0.0) + float(drift_t.sum().item())
            metric_counts["endpoint_rpe_mm"] = metric_counts.get("endpoint_rpe_mm", 0) + int(drift_t.numel())
            metric_sums["endpoint_rpe_deg"] = metric_sums.get("endpoint_rpe_deg", 0.0) + float(drift_r.sum().item())
            metric_counts["endpoint_rpe_deg"] = metric_counts.get("endpoint_rpe_deg", 0) + int(drift_r.numel())
            metric_sums["end_to_start_rpe_mm"] = metric_sums.get("end_to_start_rpe_mm", 0.0) + float(loop_t.sum().item())
            metric_counts["end_to_start_rpe_mm"] = metric_counts.get("end_to_start_rpe_mm", 0) + int(loop_t.numel())
            metric_sums["end_to_start_rpe_deg"] = metric_sums.get("end_to_start_rpe_deg", 0.0) + float(loop_r.sum().item())
            metric_counts["end_to_start_rpe_deg"] = metric_counts.get("end_to_start_rpe_deg", 0) + int(loop_r.numel())

            if extras.get("pred_volume") is not None and extras.get("gt_volume") is not None:
                pred_volume = extras["pred_volume"]
                gt_volume = extras["gt_volume"]
                vol_ncc = float(volume_ncc(pred_volume, gt_volume).mean().item())
                vol_ssim = float(volume_ssim(pred_volume, gt_volume).mean().item())
                vol_dice = float(volume_dice(pred_volume, gt_volume).mean().item())
                metric_sums["volume_ncc"] = metric_sums.get("volume_ncc", 0.0) + vol_ncc
                metric_counts["volume_ncc"] = metric_counts.get("volume_ncc", 0) + 1
                metric_sums["volume_ssim"] = metric_sums.get("volume_ssim", 0.0) + vol_ssim
                metric_counts["volume_ssim"] = metric_counts.get("volume_ssim", 0) + 1
                metric_sums["volume_dice"] = metric_sums.get("volume_dice", 0.0) + vol_dice
                metric_counts["volume_dice"] = metric_counts.get("volume_dice", 0) + 1

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
        for key, value in tusrec_sums.items():
            avg_metrics[key] = value / max(tusrec_counts.get(key, 1), 1)
        if loss_count > 0:
            avg_metrics[f"{mode}_loss"] = loss_sums["loss"] / loss_count
            avg_metrics[f"{mode}_loss_rec"] = loss_sums["loss_rec"] / loss_count
            avg_metrics[f"{mode}_loss_reg"] = loss_sums["loss_reg"] / loss_count
            avg_metrics[f"{mode}_dist"] = loss_sums["dist"] / loss_count
            avg_metrics[f"{mode}_wrap_dist"] = loss_sums["wrap_dist"] / loss_count
        if tusrec_rows:
            avg_metrics["tusrec_per_scan"] = tusrec_rows

        for cb in callbacks:
            fn = getattr(cb, "on_end", None)
            if callable(fn):
                fn(metrics=avg_metrics, mode=mode, epoch=epoch, ctx=ctx, loader=loader)
        return avg_metrics
