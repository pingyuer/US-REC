"""Config helpers for rec/rec-reg trainer."""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import OmegaConf


def get_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely read a value from OmegaConf using a dotted path."""
    value = OmegaConf.select(cfg, path)
    return default if value is None else value


def parse_rec_cfg(cfg: Any) -> Dict[str, Any]:
    """Extract trainer-relevant fields from cfg.

    Returns a dict of values used by Train_Rec_Reg_Model.
    """
    out: Dict[str, Any] = {}

    out["model_name"] = get_cfg_value(cfg, "model.name")
    out["PRED_TYPE"] = get_cfg_value(cfg, "model.pred_type")
    out["LABEL_TYPE"] = get_cfg_value(cfg, "model.label_type")
    out["in_ch_reg"] = get_cfg_value(cfg, "model.in_ch_reg")
    out["ddf_dirc"] = get_cfg_value(cfg, "model.ddf_dirc")
    out["Conv_Coords"] = get_cfg_value(cfg, "model.conv_coords")
    out["img_pro_coord"] = get_cfg_value(cfg, "model.img_pro_coord")
    out["BatchNorm"] = get_cfg_value(cfg, "model.batch_norm")
    out["intepoletion_method"] = get_cfg_value(cfg, "model.interpolation.method")
    out["intepoletion_volume"] = get_cfg_value(cfg, "model.interpolation.volume")

    out["FILENAME_CALIB"] = get_cfg_value(cfg, "dataset.calib_file")
    out["RESAMPLE_FACTOR"] = int(get_cfg_value(cfg, "dataset.resample_factor", 1))

    out["NUM_SAMPLES"] = int(get_cfg_value(cfg, "dataset.sampling.num_samples", 0))
    out["SAMPLE_RANGE"] = int(get_cfg_value(cfg, "dataset.sampling.sample_range", 0))

    out["MINIBATCH_SIZE_rec"] = int(get_cfg_value(cfg, "dataloader.train.batch_size", 1))
    out["MINIBATCH_SIZE_reg"] = int(
        get_cfg_value(cfg, "dataloader.train.batch_size_reg", out["MINIBATCH_SIZE_rec"])
    )

    out["LEARNING_RATE_rec"] = float(
        get_cfg_value(cfg, "optimizer.lr_rec", get_cfg_value(cfg, "optimizer.lr", 1e-4))
    )
    out["LEARNING_RATE_reg"] = float(
        get_cfg_value(cfg, "optimizer.lr_reg", get_cfg_value(cfg, "optimizer.lr", 1e-4))
    )

    out["Loss_type"] = get_cfg_value(cfg, "loss.type")

    out["inter"] = get_cfg_value(cfg, "trainer.mode.inter")
    out["meta"] = get_cfg_value(cfg, "trainer.mode.meta")
    out["retain_epoch"] = get_cfg_value(cfg, "trainer.retain_epoch", 0)
    out["NUM_EPOCHS"] = int(get_cfg_value(cfg, "trainer.max_epochs", 1))
    out["MAX_ITERS"] = get_cfg_value(cfg, "trainer.max_steps")
    out["val_fre"] = get_cfg_value(cfg, "trainer.validate_every", 1)
    out["FREQ_INFO"] = int(get_cfg_value(cfg, "trainer.info_interval", get_cfg_value(cfg, "trainer.log_interval", 50)))
    out["FREQ_SAVE"] = int(get_cfg_value(cfg, "trainer.save_interval", 100))

    return out
