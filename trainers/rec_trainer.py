"""
Reconstruction/registration trainer.

Moved out of utils/utils_ete.py to match the refactored trainer layout.
Helper ops now live in utils/rec_ops.py.
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.utils.data import IterableDataset
from omegaconf import OmegaConf

from utils.network import build_model
from trainers.hooks.base_hook import Hook
from utils.utils_ori import (
    reference_image_points,
    add_scalars_rec_volume,
    add_scalars_reg,
    save_best_network,
    save_best_network_reg,
    add_scalars_wrap_dist,
)
from utils.utils_grid_data import *
from utils.monai.networks.nets import VoxelMorph
from utils.monai.losses import BendingEnergyLoss
from utils.funcs import *

from trainers.utils.config import parse_rec_cfg
from trainers.utils.data import init_datasets, build_dataloaders
from trainers.utils.calibration import load_calibration
from trainers.utils.transforms import build_transforms
from trainers.utils.forward_utils import (
    unpack_batch,
    build_pred_transforms,
    points_from_transforms,
    convpose_if_needed,
)
from trainers.utils.loss import compute_loss
from trainers.utils.interp_reg import scatter_pts_interpolation, scatter_pts_registration
from trainers.utils.model_io import save_rec_model, save_reg_model, load_model, save_best_models
from trainers.utils.bn_utils import switch_off_batch_norm
from utils.rec_ops import compute_dimention, data_pairs_adjacent, ConvPose
from trainers.rec_evaluator import RecEvaluator
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


class Train_Rec_Reg_Model:

    def __init__(
        self,
        cfg,
        save_path,
        non_improve_maxmum,
        reg_loss_weight,
        val_loss_min,
        val_dist_min,
        val_loss_min_reg,
        dset_train,
        dset_val,
        dset_train_reg,
        dset_val_reg,
        device,
        writer,
        option,
    ):
        self.non_improve_maxmum = non_improve_maxmum
        self.val_loss_min = val_loss_min
        self.val_dist_min = val_dist_min
        self.val_loss_min_reg = val_loss_min_reg

        self.val_dist_min_T = val_loss_min
        self.val_dist_min_R = val_loss_min

        self.device = device
        self.writer = writer
        self.option = option
        self.cfg = cfg
        self.save_path = save_path
        self.multi_gpu = bool(OmegaConf.select(cfg, "runtime.multi_gpu") or False)

        self._load_cfg_fields()
        self._init_datasets(dset_train, dset_val)
        self._init_dataloaders()
        self._init_calibration()
        self._init_example_state()
        self._init_transforms()
        self._init_models()
        self._init_optimizer(reg_loss_weight)

        self.hooks: list[Hook] = []
        self.epoch = 0
        self.global_step = 0
        self.ctx = None

    def _load_cfg_fields(self) -> None:
        fields = parse_rec_cfg(self.cfg)
        for key, value in fields.items():
            setattr(self, key, value)

    def _init_datasets(self, dset_train, dset_val) -> None:
        self.dset_train = dset_train
        self.dset_val = dset_val
        self.NUM_SAMPLES, self.data_pairs = init_datasets(
            self.dset_train, self.dset_val, self.NUM_SAMPLES
        )

    def _init_dataloaders(self) -> None:
        self.train_loader_rec, self.val_loader_rec = build_dataloaders(
            self.dset_train,
            self.dset_val,
            self.MINIBATCH_SIZE_rec,
        )

    def _init_calibration(self) -> None:
        self.tform_calib_scale, self.tform_calib_R_T, self.tform_calib = load_calibration(
            self.FILENAME_CALIB,
            self.RESAMPLE_FACTOR,
            self.device,
        )

    def _init_example_state(self) -> None:
        example = None
        if hasattr(self.dset_train, "get_example"):
            example = self.dset_train.get_example()
        else:
            try:
                example = self.dset_train[0]
            except (TypeError, NotImplementedError):
                example = next(iter(self.dset_train))
        if isinstance(example, dict):
            frames_sample = example["frames"]
        else:
            frames_sample = example[0]

        # Align NUM_SAMPLES to actual sample length (pair-based datasets use 2).
        self.NUM_SAMPLES = int(frames_sample.shape[0])
        self.data_pairs = data_pairs_adjacent(self.NUM_SAMPLES)
        self.image_points = reference_image_points(
            (frames_sample.shape[1], frames_sample.shape[2]),
            (frames_sample.shape[1], frames_sample.shape[2]),
        ).to(self.device)
        self.pred_dim = compute_dimention(
            self.PRED_TYPE, self.image_points.shape[1], self.NUM_SAMPLES, "pred"
        )
        self.label_dim = compute_dimention(
            self.LABEL_TYPE, self.image_points.shape[1], self.NUM_SAMPLES, "label"
        )

    def _init_transforms(self) -> None:
        self.transform_label, self.transform_prediction = build_transforms(
            label_type=self.LABEL_TYPE,
            pred_type=self.PRED_TYPE,
            data_pairs=self.data_pairs,
            image_points=self.image_points,
            tform_calib=self.tform_calib,
            tform_calib_R_T=self.tform_calib_R_T,
        )

        self.criterion = torch.nn.MSELoss()
        self.img_loss = MSELoss()
        self.regularization = BendingEnergyLoss()

    def _init_models(self) -> None:
        self.model = build_model(
            self,
            in_frames=self.NUM_SAMPLES,
            pred_dim=self.pred_dim,
            label_dim=self.label_dim,
            image_points=self.image_points,
            tform_calib=self.tform_calib,
            tform_calib_R_T=self.tform_calib_R_T,
        ).to(self.device)
        self.nonrigid_enabled = bool(
            self.use_voxelmorph
            or self.use_deform
            or self.use_backward
            or self.enable_voxel
            or self.alpha_def
            or self.alpha_voxel
        )
        if self.nonrigid_enabled:
            raise NotImplementedError("Nonrigid disabled")
        self.VoxelMorph_net = None

    def _init_optimizer(self, reg_loss_weight: float) -> None:
        self.current_epoch = 0
        self.reg_loss_weight = reg_loss_weight

        self.optimiser_rec_reg = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE_rec)

    def register_hook(self, hook: Hook) -> None:
        self.hooks.append(hook)
        self.hooks.sort(key=lambda h: int(getattr(h, "priority", 50)), reverse=True)

    def register_hooks(self, hooks: list[Hook]) -> None:
        for hook in hooks:
            self.register_hook(hook)

    def call_hooks(self, event: str, **kwargs) -> None:
        for hook in self.hooks:
            fn = getattr(hook, event, None)
            if callable(fn):
                fn(self, **kwargs)

    def _run_step(self, batch, step):
        frames, tforms, tforms_inv = unpack_batch(batch)
        frames = frames.to(self.device)
        tforms = tforms.to(self.device)
        tforms_inv = tforms_inv.to(self.device)

        tforms_each_frame2frame0 = self.transform_label(tforms, tforms_inv)
        frames = frames / 255

        outputs = self.model(frames)
        pred_transfs = build_pred_transforms(self.transform_prediction, outputs, self.device)
        labels, pred_pts = points_from_transforms(
            img_pro_coord=self.img_pro_coord,
            tform_calib_R_T=self.tform_calib_R_T,
            tform_calib=self.tform_calib,
            image_points=self.image_points,
            tforms_each_frame2frame0=tforms_each_frame2frame0,
            pred_transfs=pred_transfs,
        )
        labels, pred_pts, convR_batched, minxyz_all = convpose_if_needed(
            conv_coords=self.Conv_Coords,
            img_pro_coord=self.img_pro_coord,
            tforms_each_frame2frame0=tforms_each_frame2frame0,
            pred_transfs=pred_transfs,
            tform_calib=self.tform_calib,
            image_points=self.image_points,
            labels=labels,
            pred_pts=pred_pts,
            device=self.device,
        )

        loss, loss1, loss2, dist, wrap_dist, extras = compute_loss(
            loss_type=self.Loss_type,
            labels=labels,
            pred_pts=pred_pts,
            frames=frames,
            step=step,
            criterion=self.criterion,
            img_loss=self.img_loss,
            regularization=self.regularization,
            reg_loss_weight=self.reg_loss_weight,
            ddf_dirc=self.ddf_dirc,
            conv_coords=self.Conv_Coords,
            option=self.option,
            device=self.device,
            scatter_pts_registration=self._scatter_pts_registration,
            scatter_pts_interpolation=self._scatter_pts_interpolation,
            wrapped_pred_dist_fn=wrapped_pred_dist,
            convR_batched=convR_batched,
            minxyz_all=minxyz_all,
            rigid_only=not self.nonrigid_enabled,
        )
        metrics = {}
        trans_err = translation_error_mm(
            pred_transfs[..., :3, 3], tforms_each_frame2frame0[..., :3, 3]
        )
        rot_err = rotation_error_deg(pred_transfs[..., :3, :3], tforms_each_frame2frame0[..., :3, :3])
        se3_trans = se3_translation_error(pred_transfs, tforms_each_frame2frame0)
        se3_rot = se3_rotation_error_deg(pred_transfs, tforms_each_frame2frame0)
        metrics["translation_error_mm"] = trans_err.mean().item()
        metrics["rotation_error_deg"] = rot_err.mean().item()
        metrics["se3_trans_mm"] = se3_trans.mean().item()
        metrics["se3_rot_deg"] = se3_rot.mean().item()

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

        return loss, loss1, loss2, dist, wrap_dist, metrics

    def train_rec_model(self):
        # train reconstruction network
        self.call_hooks("before_run", mode="train")
        try:
            self.model.train(True)
            if self.VoxelMorph_net is not None:
                self.VoxelMorph_net.train(True)
            switch_off_batch_norm(self.model, self.VoxelMorph_net, self.BatchNorm)

            max_iters = self.MAX_ITERS
            max_iters = int(max_iters) if max_iters is not None else None
            val_fre = int(self.val_fre) if self.val_fre is not None else 0

            self.call_hooks("before_train")
            for epoch in range(int(self.retain_epoch), int(self.retain_epoch) + self.NUM_EPOCHS):
                self.epoch = epoch
                if hasattr(self.dset_train, "set_epoch"):
                    self.dset_train.set_epoch(epoch)
                self.call_hooks("before_epoch")

                train_epoch_loss = 0
                train_epoch_dist, train_epoch_wrap_dist = 0, 0
                train_epoch_loss_reg = 0
                train_epoch_loss_rec = 0
                for step, batch in enumerate(self.train_loader_rec):
                    if max_iters is not None and max_iters > 0 and step >= max_iters:
                        break
                    self.call_hooks("before_step")
                    self.global_step += 1

                    self.optimiser_rec_reg.zero_grad()
                    loss, loss1, loss2, dist, wrap_dist, _metrics = self._run_step(batch, step)

                    train_epoch_loss += loss.item()
                    train_epoch_dist += dist.item()
                    train_epoch_wrap_dist += wrap_dist.item()
                    train_epoch_loss_rec = train_epoch_loss_rec + loss1.item()
                    train_epoch_loss_reg += loss2.item()

                    loss.backward()
                    self.optimiser_rec_reg.step()

                    lr = float(self.optimiser_rec_reg.param_groups[0].get("lr", 0.0))
                    self.call_hooks(
                        "after_step",
                        log_buffer={
                            "mode": "train",
                            "epoch": epoch + 1,
                            "iter": step + 1,
                            "global_step": self.global_step,
                            "loss": float(loss.item()),
                            "lr": lr,
                        },
                    )

                train_epoch_loss /= (step + 1)
                train_epoch_dist /= (step + 1)
                train_epoch_wrap_dist /= (step + 1)
                train_epoch_loss_reg /= (step + 1)
                train_epoch_loss_rec /= (step + 1)

                if epoch in range(0, self.NUM_EPOCHS, self.FREQ_INFO):
                    print('[Rec - Epoch %d] train-loss-rec=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss_rec, train_epoch_dist))

                lr = float(self.optimiser_rec_reg.param_groups[0].get("lr", 0.0))
                self.call_hooks(
                    "after_epoch",
                    log_buffer={
                        "epoch": epoch + 1,
                        "max_epochs": self.NUM_EPOCHS,
                        "train_loss": float(train_epoch_loss),
                        "train_loss_rec": float(train_epoch_loss_rec),
                        "train_loss_reg": float(train_epoch_loss_reg),
                        "train_dist": float(train_epoch_dist),
                        "train_wrap_dist": float(train_epoch_wrap_dist),
                        "lr": lr,
                    },
                )

                # validation
                if val_fre > 0 and epoch in range(0, self.NUM_EPOCHS, val_fre):
                    self.call_hooks("before_val")
                    self.model.train(False)
                    if self.VoxelMorph_net is not None:
                        self.VoxelMorph_net.train(False)
                    switch_off_batch_norm(self.model, self.VoxelMorph_net, self.BatchNorm)

                    rec_eval = RecEvaluator(device=self.device)
                    metrics = rec_eval.run(
                        model=self.model,
                        voxel_morph_net=self.VoxelMorph_net,
                        loader=self.val_loader_rec,
                        cfg=self.cfg,
                        trainer=self,
                        mode="val",
                        epoch=epoch + 1,
                        ctx=self.ctx,
                    )
                    epoch_loss_val = float(metrics.get("val_loss", 0.0))
                    epoch_loss_val_rec = float(metrics.get("val_loss_rec", 0.0))
                    epoch_loss_val_reg = float(metrics.get("val_loss_reg", 0.0))
                    epoch_dist_val = float(metrics.get("val_dist", 0.0))
                    epoch_wrap_dist_val = float(metrics.get("val_wrap_dist", 0.0))

                    # save model
                    save_rec_model(
                        self.model,
                        epoch=epoch,
                        num_epochs=self.NUM_EPOCHS,
                        freq_save=self.FREQ_SAVE,
                        save_path=self.save_path,
                        multi_gpu=self.multi_gpu,
                    )
                    if self.VoxelMorph_net is not None:
                        save_reg_model(
                            self.VoxelMorph_net,
                            epoch=epoch,
                            num_epochs=self.NUM_EPOCHS,
                            freq_save=self.FREQ_SAVE,
                            save_path=self.save_path,
                            multi_gpu=self.multi_gpu,
                        )

                    if self.VoxelMorph_net is not None:
                        self.val_dist_min_T = save_best_models(
                            epoch=epoch,
                            running_dist=epoch_dist_val,
                            best_dist=self.val_dist_min_T,
                            save_path=self.save_path,
                            multi_gpu=self.multi_gpu,
                            model_T=self.model,
                            model_R=self.VoxelMorph_net,
                            tag="val_dist_T",
                        )
                        self.val_dist_min_R = save_best_models(
                            epoch=epoch,
                            running_dist=epoch_wrap_dist_val,
                            best_dist=self.val_dist_min_R,
                            save_path=self.save_path,
                            multi_gpu=self.multi_gpu,
                            model_T=self.model,
                            model_R=self.VoxelMorph_net,
                            tag="val_dist_R",
                        )

                    if epoch in range(0, self.NUM_EPOCHS, self.FREQ_INFO):
                        print('[Rec - Epoch %d] val-loss-rec=%.3f, val-dist=%.3f' % (epoch, epoch_loss_val_rec, epoch_dist_val))

                    loss_dists = {
                        'train_epoch_loss_all': train_epoch_loss,
                        'train_epoch_dist': train_epoch_dist,
                        'train_epoch_loss_reg': train_epoch_loss_reg,
                        'train_epoch_loss_rec': train_epoch_loss_rec,
                        'epoch_loss_val_all': epoch_loss_val,
                        'epoch_dist_val': epoch_dist_val,
                        'epoch_loss_val_reg': epoch_loss_val_reg,
                        'epoch_loss_val_rec': epoch_loss_val_rec,
                    }
                    add_scalars_rec_volume(self.writer, epoch, loss_dists)

                    dist_wrap = {
                        'train_wrap_dist': train_epoch_wrap_dist,
                        'val_wrap_dist': epoch_wrap_dist_val,
                    }
                    add_scalars_wrap_dist(self.writer, epoch, dist_wrap, 'rec_reg')

                    self.call_hooks(
                        "after_val",
                        log_buffer={
                            "epoch": epoch + 1,
                            "max_epochs": self.NUM_EPOCHS,
                            "val_loss": float(epoch_loss_val),
                            "val_loss_rec": float(epoch_loss_val_rec),
                            "val_loss_reg": float(epoch_loss_val_reg),
                            "val_dist": float(epoch_dist_val),
                            "val_wrap_dist": float(epoch_wrap_dist_val),
                            **{
                                k: float(v)
                                for k, v in metrics.items()
                                if k
                                not in {
                                    "val_loss",
                                    "val_loss_rec",
                                    "val_loss_reg",
                                    "val_dist",
                                    "val_wrap_dist",
                                    "tusrec_per_scan",
                                }
                                and isinstance(v, (int, float))
                            },
                            **(
                                {"tusrec_per_scan": metrics.get("tusrec_per_scan")}
                                if "tusrec_per_scan" in metrics
                                else {}
                            ),
                        },
                    )

                    self.model.train(True)
                    if self.VoxelMorph_net is not None:
                        self.VoxelMorph_net.train(True)
                    switch_off_batch_norm(self.model, self.VoxelMorph_net, self.BatchNorm)
            self.call_hooks("after_train")
        except BaseException as exc:
            self.call_hooks("on_exception", exc=exc)
            raise
        finally:
            self.call_hooks("after_run", mode="train")

    def _scatter_pts_registration(self, labels, pred_pts, frames, step):
        return scatter_pts_registration(
            labels=labels,
            pred_pts=pred_pts,
            frames=frames,
            step=step,
            device=self.device,
            option=self.option,
            intepoletion_method=self.intepoletion_method,
            intepoletion_volume=self.intepoletion_volume,
            voxel_morph_net=self.VoxelMorph_net,
            use_deform=self.use_deform,
            enable_voxel=self.enable_voxel,
        )

    def _scatter_pts_interpolation(self, labels, pred_pts, frames, step):
        return scatter_pts_interpolation(
            labels=labels,
            pred_pts=pred_pts,
            frames=frames,
            step=step,
            device=self.device,
            option=self.option,
            intepoletion_method=self.intepoletion_method,
            intepoletion_volume=self.intepoletion_volume,
            enable_voxel=self.enable_voxel,
        )

    def load_best_rec_model(self):
        try:
            load_model(
                self.model,
                path=os.path.join(self.save_path, 'saved_model', 'best_validation_loss_model'),
                device=self.device,
            )
        except Exception:
            print('No best rec model saved at the moment...')

    def load_best_reg_model(self):
        if self.VoxelMorph_net is None:
            print("Nonrigid disabled: skipping reg model load.")
            return
        try:
            load_model(
                self.VoxelMorph_net,
                path=os.path.join(self.save_path, 'saved_model', 'best_validation_loss_model_reg'),
                device=self.device,
            )
        except Exception:
            print('No best reg model saved at the moment...')

    def load_recon_model_initial(self):
        try:
            load_model(
                self.model,
                path=os.path.join(self.save_path, 'saved_model', 'ete_initial_recon'),
                device=self.device,
            )
        except Exception:
            raise RuntimeError('No best model saved at the moment...')

    def load_def_model_initial(self):
        if self.VoxelMorph_net is None:
            raise RuntimeError("Nonrigid disabled")
        try:
            load_model(
                self.VoxelMorph_net,
                path=os.path.join(self.save_path, 'saved_model', 'ete_initial_def'),
                device=self.device,
            )
        except Exception:
            raise RuntimeError('No best model saved at the moment...')

    def multi_model(self):
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)
            if self.VoxelMorph_net is not None:
                self.VoxelMorph_net = nn.DataParallel(self.VoxelMorph_net)
            print('multi-gpu')
            print(os.environ["CUDA_VISIBLE_DEVICES"])
