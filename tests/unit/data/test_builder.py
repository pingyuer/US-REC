"""Covers data.builder factory helpers."""

import pytest
pytestmark = pytest.mark.unit


import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, SequentialSampler

from data.builder import build_dataset, build_dataloader, build_pipeline
from data.transforms.registry import register_transform


class _BuilderDummyDataset(Dataset):
    def __init__(self, data, flag=False, split=None, mode=None):
        self.data = list(data)
        self.flag = flag
        self.received_split = split
        self.received_mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


def test_build_dataset_selects_split_specific_values():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": f"{_BuilderDummyDataset.__module__}.{_BuilderDummyDataset.__name__}",
                "data": {"train": [1, 2, 3], "val": [10, 20]},
                "flag": {"train": True, "val": False},
            }
        }
    )

    dataset = build_dataset(cfg, split="val")

    assert isinstance(dataset, _BuilderDummyDataset)
    assert list(dataset.data) == [10, 20]
    assert dataset.flag is False
    # build_dataset should inject split/mode when constructor accepts them
    assert dataset.received_split == "val"
    assert dataset.received_mode == "val"


@register_transform("_AddOffset")
class _AddOffset:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, sample):
        sample["value"] += self.delta
        return sample


def test_build_pipeline_uses_registry_transform():
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "train": [
                    {"type": "_AddOffset", "delta": 5},
                ]
            }
        }
    )

    pipeline = build_pipeline(cfg, split="train")

    sample = {"value": 3}
    result = pipeline(sample)
    assert result["value"] == 8


def test_build_pipeline_supports_albumentations():
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "type": "Albumentations",
                "train": [
                    {"type": "HorizontalFlip", "p": 1.0},
                ],
            }
        }
    )

    pipeline = build_pipeline(cfg, split="train")

    image = np.arange(9, dtype=np.uint8).reshape(3, 3)
    mask = np.arange(9, dtype=np.uint8).reshape(3, 3)
    transformed = pipeline(image=image, mask=mask)

    assert set(transformed.keys()) >= {"image", "mask"}
    assert transformed["image"].shape == image.shape
    assert transformed["mask"].shape == mask.shape


def test_build_dataloader_honors_config_and_sampler():
    dataset = _BuilderDummyDataset([0, 1, 2, 3])
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "train": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "shuffle": True,
                    "drop_last": False,
                }
            }
        }
    )

    sampler = SequentialSampler(dataset)
    loader = build_dataloader(dataset, cfg, split="train", sampler=sampler)

    assert loader.batch_size == 2
    assert loader.num_workers == 0
    assert loader.drop_last is False
    # When sampler is provided, dataloader should use it directly
    assert loader.batch_sampler.sampler is sampler
    batches = list(loader)
    assert len(batches) == 2
    assert torch.equal(batches[0], torch.tensor([0.0, 1.0]))
    assert torch.equal(batches[1], torch.tensor([2.0, 3.0]))
