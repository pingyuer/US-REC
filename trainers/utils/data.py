"""Dataset and DataLoader helpers for rec/rec-reg trainer."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import IterableDataset

from trainers.utils.rec_ops import data_pairs_adjacent


def init_datasets(dset_train, dset_val, num_samples: int) -> Tuple[int, torch.Tensor]:
    """Resolve NUM_SAMPLES and build pair indices.

    Returns:
        (num_samples, data_pairs)
    """
    num_frames = getattr(dset_train, "num_samples", None)
    if num_frames is None or int(num_frames) == 0:
        num_frames = num_samples
    if num_frames == -1:
        try:
            sample = dset_train[0]
            if isinstance(sample, (list, tuple)) and len(sample) > 0:
                num_frames = int(sample[0].shape[0])
        except Exception:
            num_frames = num_samples
    if num_frames is None:
        num_frames = num_samples

    num_samples = int(num_frames)
    data_pairs = data_pairs_adjacent(num_samples)
    return num_samples, data_pairs


def build_dataloaders(dset_train, dset_val, batch_size_rec: int):
    """Create train/val dataloaders with IterableDataset-aware shuffling."""
    train_shuffle = not isinstance(dset_train, IterableDataset)
    if hasattr(dset_train, "set_epoch"):
        try:
            is_empty = len(dset_train) == 0
        except TypeError:
            is_empty = True
        if is_empty:
            dset_train.set_epoch(0)
    train_loader = torch.utils.data.DataLoader(
        dset_train,
        batch_size=batch_size_rec,
        shuffle=train_shuffle,
        num_workers=0,
    )

    val_shuffle = not isinstance(dset_val, IterableDataset)
    if hasattr(dset_val, "set_epoch"):
        try:
            is_empty = len(dset_val) == 0
        except TypeError:
            is_empty = True
        if is_empty:
            dset_val.set_epoch(0)
    val_loader = torch.utils.data.DataLoader(
        dset_val,
        batch_size=1,
        shuffle=val_shuffle,
        num_workers=0,
    )

    return train_loader, val_loader
