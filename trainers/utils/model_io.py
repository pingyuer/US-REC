"""Model IO helpers for rec/rec-reg trainer."""

from __future__ import annotations

import os
import torch


def save_rec_model(model, *, epoch, num_epochs, freq_save, save_path, multi_gpu):
    """Save reconstruction model checkpoint."""
    if epoch in range(0, num_epochs, freq_save):
        path = os.path.join(save_path, "saved_model", f"model_epoch{epoch:08d}")
        if multi_gpu:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)


def save_reg_model(model, *, epoch, num_epochs, freq_save, save_path, multi_gpu):
    """Save registration model checkpoint."""
    if epoch in range(0, num_epochs, freq_save):
        path = os.path.join(save_path, "saved_model", f"model_reg_epoch{epoch:08d}")
        if multi_gpu:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)


def load_model(model, *, path, device):
    """Load model weights from path."""
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))


def save_best_models(
    *,
    epoch,
    running_dist,
    best_dist,
    save_path,
    multi_gpu,
    model_T,
    model_R,
    tag,
):
    """Save best model pair and update best distance.

    Returns updated best_dist.
    """
    if running_dist >= best_dist:
        return best_dist

    best_dist = running_dist
    file_name = os.path.join(save_path, "config.txt")
    with open(file_name, "a") as opt_file:
        opt_file.write(
            f"------------ best {tag} - epoch {epoch}: dist = {running_dist} -------------\n"
        )

    if multi_gpu:
        torch.save(model_T.module.state_dict(), os.path.join(save_path, "saved_model", f"best_{tag}_T_T"))
        torch.save(model_R.module.state_dict(), os.path.join(save_path, "saved_model", f"best_{tag}_T_R"))
    else:
        torch.save(model_T.state_dict(), os.path.join(save_path, "saved_model", f"best_{tag}_T_T"))
        torch.save(model_R.state_dict(), os.path.join(save_path, "saved_model", f"best_{tag}_T_R"))

    return best_dist
