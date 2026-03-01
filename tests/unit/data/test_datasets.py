"""Covers dataset base classes and segmentation dataset utilities."""

import pytest
pytestmark = pytest.mark.unit


import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

from data.datasets.dataset import BaseDataset
from data.datasets.seg_dataset import SegmentationDataset
from data.transforms.registry import register_transform


class _BaseDatasetExample(BaseDataset):
    def load_sample(self, item):
        return {"value": item["value"]}


def test_base_dataset_pipeline_is_applied():
    called = {"count": 0}

    def pipeline(sample):
        called["count"] += 1
        sample["value"] += 10
        return sample

    dataset = _BaseDatasetExample(
        data_list=[{"value": 1}, {"value": 2}],
        pipeline=pipeline,
    )

    assert len(dataset) == 2
    assert dataset[0]["value"] == 11
    assert called["count"] == 1


@register_transform("_ReturnWithFlag")
class _ReturnWithFlag:
    def __call__(self, sample):
        sample["processed"] = True
        return sample


def _make_temp_image(path: Path, value: int) -> None:
    array = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_segmentation_dataset_pairs_and_sorts_files(tmp_path: Path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    _make_temp_image(img_dir / "img_b.png", 80)
    _make_temp_image(img_dir / "img_a.png", 20)
    _make_temp_image(mask_dir / "img_a.png", 30)
    _make_temp_image(mask_dir / "img_b.png", 40)

    pipeline_cfg = OmegaConf.create(
        {
            "pipeline": {
                "train": [
                    {"type": "_ReturnWithFlag"},
                ]
            }
        }
    )

    dataset = SegmentationDataset(
        img_dir=str(img_dir),
        mask_dir=str(mask_dir),
        pipeline_cfg=pipeline_cfg,
    )

    assert len(dataset) == 2
    first_sample = dataset[0]
    assert first_sample["img_path"].endswith("img_a.png")
    assert first_sample["mask_path"].endswith("img_a.png")
    assert first_sample["processed"] is True


def test_segmentation_dataset_handles_missing_mask(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _make_temp_image(img_dir / "image.png", 50)

    pipeline_cfg = OmegaConf.create(
        {
            "pipeline": {
                "train": [
                    {"type": "_ReturnWithFlag"},
                ]
            }
        }
    )

    dataset = SegmentationDataset(
        img_dir=str(img_dir),
        mask_dir=None,
        pipeline_cfg=pipeline_cfg,
    )

    sample = dataset[0]
    assert sample["mask_path"] is None
    assert sample["processed"] is True
