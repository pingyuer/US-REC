from __future__ import annotations

from typing import Any, Optional
import time
from collections import defaultdict

import torch
from omegaconf import OmegaConf

from trainers.metrics import (
    compose_global_from_local,
    compute_tusrec_metrics,
    end_to_start_rpe_rotation_deg,
    end_to_start_rpe_translation_mm,
    endpoint_rpe_rotation_deg,
    endpoint_rpe_translation_mm,
    rotation_error_deg,
    se3_rotation_error_deg,
    se3_translation_error_mm,
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
from utils.funcs import wrapped_pred_dist


class RecEvaluator:
    """
    Evaluation loop for reconstruction/registration models.

    Expects batches with keys: frames, tforms, tforms_inv.

    Transform convention
    --------------------
    * The dataset yields **pairs** of consecutive frames (frame_idx0, frame_idx1).
    * ``LabelTransform.to_transform_t2t`` produces per-pair local transforms:
      ``T_prev_from_curr[i] = T_{i-1 <- i}`` — maps frame *i* into frame *i-1*.
    * These are collected in ``scan_state[sid]["locals_gt"]`` /
      ``scan_state[sid]["locals_pred"]``, keyed by *frame_idx1*.
    * Global transforms are accumulated using the **single authoritative**
      :func:`compose_global_from_local`:
      ``global[0] = I``,  ``global[i] = global[i-1] @ local[i]``.
    * Each scan resets ``global[0] = I`` — no cross-scan leakage.
    """

    def __init__(self, *, device: str | torch.device):
        self.device = device

    @staticmethod
    def _canonical_scan_id(scan_id: Any, sample_index: int) -> str:
        if isinstance(scan_id, (list, tuple)):
            if len(scan_id) == 1:
                return str(scan_id[0])
            return "/".join(str(x) for x in scan_id)
        if scan_id is None:
            return f"sample_{sample_index}"
        return str(scan_id)

    @staticmethod
    def _aggregate_tusrec_rows_by_scan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sums: dict[str, dict[str, float]] = defaultdict(dict)
        counts: dict[str, dict[str, int]] = defaultdict(dict)
        pair_counts: dict[str, int] = defaultdict(int)

        for row in rows:
            sid = str(row.get("scan_id", "unknown"))
            for key, value in row.items():
                if key == "scan_id" or value is None:
                    continue
                if key == "num_pairs" and isinstance(value, (int, float)):
                    pair_counts[sid] += int(value)
                    continue
                if isinstance(value, (int, float)):
                    sums[sid][key] = sums[sid].get(key, 0.0) + float(value)
                    counts[sid][key] = counts[sid].get(key, 0) + 1

        out: list[dict[str, Any]] = []
        for sid in sorted(sums.keys()):
            entry = {"scan_id": sid}
            for key, total in sums[sid].items():
                cnt = max(1, counts[sid].get(key, 1))
                entry[key] = float(total / cnt)
            entry["num_pairs"] = int(pair_counts.get(sid, 0))
            out.append(entry)
        return out

    @staticmethod
    def _meta_value(meta: Any, key: str, idx: int):
        if isinstance(meta, dict):
            value = meta.get(key)
            if isinstance(value, list):
                if not value:
                    return None
                return value[idx] if idx < len(value) else value[0]
            return value
        if isinstance(meta, list):
            if idx < len(meta) and isinstance(meta[idx], dict):
                return meta[idx].get(key)
            return None
        return None

    @staticmethod
    def _compose_global_from_locals(
        local_by_frame: dict[int, torch.Tensor],
        *,
        dtype,
        device,
    ) -> torch.Tensor:
        """Build global transforms ``T_{0<-i}`` from per-frame locals ``T_{i-1<-i}``.

        Uses the **single authoritative** :func:`compose_global_from_local`.
        Each call represents ONE scan — global[0] is always reset to I.

        Parameters
        ----------
        local_by_frame : dict mapping frame_idx (int) -> (4, 4) Tensor
            Local transforms ``T_prev_from_curr`` keyed by frame number.
            Key 0 is absent (frame 0 has no predecessor).
        """
        if not local_by_frame:
            eye = torch.eye(4, dtype=dtype, device=device)
            return eye.unsqueeze(0)
        max_frame = max(int(k) for k in local_by_frame.keys())
        eye = torch.eye(4, dtype=dtype, device=device)
        # Assemble a dense (T, 4, 4) tensor from the sparse dict.
        # Missing frames are filled with I (no motion).
        local_list = [eye]  # frame 0 = I
        for frame_idx in range(1, max_frame + 1):
            local_tf = local_by_frame.get(frame_idx)
            if local_tf is None:
                local_tf = eye
            local_list.append(local_tf)
        local_dense = torch.stack(local_list, dim=0)  # (T, 4, 4)
        # Convention: local_dense[i] = T_{i-1 <- i} (prev_from_curr)
        return compose_global_from_local(local_dense, convention="prev_from_curr")

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
        scan_state: dict[str, dict[str, Any]] = {}
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
            batch_start_time = time.perf_counter()
            frames, tforms, tforms_inv = unpack_batch(batch)
            frames = frames.to(self.device)
            tforms = tforms.to(self.device)
            tforms_inv = tforms_inv.to(self.device)

            tforms_each_frame2frame0 = trainer.transform_label(tforms, tforms_inv)
            frames = frames / 255

            forward_start_time = time.perf_counter()
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
            se3_trans = se3_translation_error_mm(pred_transfs, tforms_each_frame2frame0)
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
            metrics["wrap_dist_enabled"] = 1.0 if bool(extras.get("wrap_enabled", False)) else 0.0

            if extras.get("pred_volume") is not None and extras.get("gt_volume") is not None:
                pred_volume = extras["pred_volume"]
                gt_volume = extras["gt_volume"]
                metrics["volume_ncc"] = float(volume_ncc(pred_volume, gt_volume).mean().item())
                metrics["volume_ssim"] = float(volume_ssim(pred_volume, gt_volume).mean().item())
                metrics["volume_dice"] = float(volume_dice(pred_volume, gt_volume).mean().item())

            batch_size = int(frames.shape[0]) if frames.ndim >= 4 else 1
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
                        scan_field = batch["scan_id"]
                        if isinstance(scan_field, list):
                            scan_id = scan_field[idx] if idx < len(scan_field) else scan_field[0]
                        else:
                            scan_id = scan_field
                    elif "meta" in batch:
                        meta = batch["meta"]
                        if isinstance(meta, list) and idx < len(meta) and isinstance(meta[idx], dict):
                            scan_id = meta[idx].get("scan_id") or meta[idx].get("scan_name")
                        elif isinstance(meta, dict):
                            meta_scan_id = meta.get("scan_id")
                            if isinstance(meta_scan_id, list):
                                scan_id = meta_scan_id[idx] if idx < len(meta_scan_id) else meta_scan_id[0]
                            else:
                                scan_id = meta_scan_id or meta.get("scan_name")

                sid = self._canonical_scan_id(scan_id, sample_index)
                meta_obj = batch.get("meta") if isinstance(batch, dict) else None
                frame_idx1 = self._meta_value(meta_obj, "frame_idx1", idx)
                if frame_idx1 is None:
                    frame_idx0 = self._meta_value(meta_obj, "frame_idx0", idx)
                    frame_idx1 = int(frame_idx0) + 1 if frame_idx0 is not None else 1
                frame_idx1 = int(frame_idx1)

                if sid not in scan_state:
                    h = int(frame_seq.shape[-2])
                    w = int(frame_seq.shape[-1])
                    scan_state[sid] = {
                        "locals_gt": {},
                        "locals_pred": {},
                        "image_size": (h, w),
                        "landmarks": landmarks,
                        "runtime_forward": [],
                        "runtime_e2e": [],
                    }
                scan_state[sid]["locals_gt"][frame_idx1] = gt_seq[1].detach()
                scan_state[sid]["locals_pred"][frame_idx1] = pred_seq[1].detach()
                # Convention note: gt_seq[1] / pred_seq[1] are local transforms
                # T_{prev_from_curr} = T_{frame_idx0 <- frame_idx1} produced by
                # LabelTransform.to_transform_t2t / PredictionTransform.
                if landmarks is not None and scan_state[sid]["landmarks"] is None:
                    scan_state[sid]["landmarks"] = landmarks
                sample_index += 1

            runtime_forward_per_scan = (time.perf_counter() - forward_start_time) / max(batch_size, 1)
            runtime_e2e_per_scan = (time.perf_counter() - batch_start_time) / max(batch_size, 1)
            if isinstance(batch, dict) and "meta" in batch:
                meta_obj = batch["meta"]
                for idx in range(batch_size):
                    sid = self._canonical_scan_id(
                        self._meta_value(meta_obj, "scan_id", idx)
                        or self._meta_value(meta_obj, "scan_name", idx),
                        sample_index + idx,
                    )
                    if sid in scan_state:
                        scan_state[sid]["runtime_forward"].append(float(runtime_forward_per_scan))
                        scan_state[sid]["runtime_e2e"].append(float(runtime_e2e_per_scan))

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
            metric_sums["wrap_dist_enabled"] = metric_sums.get("wrap_dist_enabled", 0.0) + float(metrics["wrap_dist_enabled"]) * batch_size
            metric_counts["wrap_dist_enabled"] = metric_counts.get("wrap_dist_enabled", 0) + int(batch_size)

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

        # Paper-aligned protocol: evaluate GPE/LPE on scan-level trajectories.
        # Each scan resets global[0] = I — no cross-scan accumulation.
        from trainers.metrics.tusrec import local_from_global as _lfg

        for sid in sorted(scan_state.keys()):
            state = scan_state[sid]
            locals_gt = state["locals_gt"]
            locals_pred = state["locals_pred"]
            if not locals_gt or not locals_pred:
                continue
            dtype = next(iter(locals_gt.values())).dtype
            device_scan = next(iter(locals_gt.values())).device
            # --- Frame-gap check: missing frames are filled with identity, which
            # causes the global trajectory to be wrong and GPE to explode.
            # Warn loudly so the user knows shuffle/IO settings need fixing.
            def _check_frame_gaps(local_by_frame: dict, scan_id: str, role: str) -> None:
                if not local_by_frame:
                    return
                max_f = max(int(k) for k in local_by_frame.keys())
                expected = set(range(1, max_f + 1))
                present = set(int(k) for k in local_by_frame.keys())
                missing = sorted(expected - present)
                if missing:
                    pct = 100.0 * len(missing) / max_f if max_f > 0 else 0.0
                    gap_str = str(missing[:10]) + ("..." if len(missing) > 10 else "")
                    print(
                        f"[WARNING] scan={scan_id!r} {role}: {len(missing)}/{max_f} frames"
                        f" ({pct:.1f}%) have no local transform → filled with identity (no-motion)."
                        f" Missing frame indices: {gap_str}."
                        f" This will corrupt the global trajectory and inflate GPE."
                        f" Check shuffle_slices/shuffle_pairs or IO errors.",
                        flush=True,
                    )

            _check_frame_gaps(locals_gt, sid, "GT")
            _check_frame_gaps(locals_pred, sid, "pred")
            gt_global = self._compose_global_from_locals(locals_gt, dtype=dtype, device=device_scan)
            pred_global = self._compose_global_from_locals(locals_pred, dtype=dtype, device=device_scan)
            num_frames = int(min(gt_global.shape[0], pred_global.shape[0]))
            if num_frames <= 1:
                continue
            gt_global = gt_global[:num_frames]
            pred_global = pred_global[:num_frames]

            # Store for downstream callbacks (e.g. VizHook)
            state["globals_gt"] = gt_global
            state["globals_pred"] = pred_global

            # ---- diagnostic logging (3 sample frames per scan) ----
            _sample_idxs = [1, 2, num_frames - 1] if num_frames > 2 else [1]
            _local_pred_dense = _lfg(pred_global)
            _local_gt_dense = _lfg(gt_global)
            for _si in _sample_idxs:
                if _si >= num_frames:
                    continue
                _lp_norm = float(torch.linalg.norm(_local_pred_dense[_si, :3, 3]).item())
                _lg_norm = float(torch.linalg.norm(_local_gt_dense[_si, :3, 3]).item())
                _gp_norm = float(torch.linalg.norm(pred_global[_si, :3, 3]).item())
                _gg_norm = float(torch.linalg.norm(gt_global[_si, :3, 3]).item())
                # Sanity: local should equal inv(global[i-1]) @ global[i].
                _recon = torch.matmul(
                    torch.linalg.inv(pred_global[_si - 1]),
                    pred_global[_si],
                )
                _sanity = float(torch.abs(_local_pred_dense[_si] - _recon).mean().item())
                print(
                    f"[diag] scan={sid} frame={_si}/{num_frames-1}  "
                    f"local_pred_t_norm={_lp_norm:.4f}mm  local_gt_t_norm={_lg_norm:.4f}mm  "
                    f"global_pred_t_norm={_gp_norm:.4f}mm  global_gt_t_norm={_gg_norm:.4f}mm  "
                    f"sanity(local-inv(g[i-1])@g[i])={_sanity:.2e}"
                )
            h, w = state["image_size"]
            dummy_frames = torch.zeros((1, int(h), int(w)), dtype=dtype, device=device_scan)
            tusrec = compute_tusrec_metrics(
                frames=dummy_frames,
                gt_transforms=gt_global,
                pred_transforms=pred_global,
                calib={
                    "tform_calib": trainer.tform_calib,
                    "spacing_mm": getattr(trainer, "tform_calib_scale", None),
                },
                landmarks=state.get("landmarks"),
                image_points=trainer.image_points,
                enforce_lp_gp_distinct=bool(
                    OmegaConf.select(cfg, "metrics.tusrec.enforce_lp_gp_distinct")
                    if OmegaConf.is_config(cfg)
                    else False
                ),
            )
            row = {"scan_id": sid}
            row.update({k: v for k, v in tusrec.items() if v is not None})
            if state["runtime_forward"]:
                row["runtime_s_per_scan"] = float(sum(state["runtime_forward"]) / len(state["runtime_forward"]))
                row["runtime_forward_s_per_scan"] = row["runtime_s_per_scan"]
            if state["runtime_e2e"]:
                row["runtime_e2e_s_per_scan"] = float(sum(state["runtime_e2e"]) / len(state["runtime_e2e"]))
            row["num_pairs"] = int(min(len(locals_gt), len(locals_pred)))
            tusrec_rows.append(row)
            for key, value in row.items():
                if key in {"scan_id", "num_pairs"} or value is None:
                    continue
                if isinstance(value, (int, float)):
                    tusrec_sums[key] = tusrec_sums.get(key, 0.0) + float(value)
                    tusrec_counts[key] = tusrec_counts.get(key, 0) + 1

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
            avg_metrics["tusrec_per_scan"] = self._aggregate_tusrec_rows_by_scan(tusrec_rows)

        # Expose per-scan global transforms for downstream hooks (e.g. VizHook)
        scan_globals = {
            sid: {
                "pred": state["globals_pred"],
                "gt": state["globals_gt"],
            }
            for sid, state in scan_state.items()
            if "globals_pred" in state and "globals_gt" in state
        }
        if scan_globals:
            avg_metrics["scan_globals"] = scan_globals

        for cb in callbacks:
            fn = getattr(cb, "on_end", None)
            if callable(fn):
                fn(metrics=avg_metrics, mode=mode, epoch=epoch, ctx=ctx, loader=loader)
        return avg_metrics
