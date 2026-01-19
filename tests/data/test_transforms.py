"""Covers transform registry and Synapse-style pipelines."""

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("torchvision.transforms.v2")

from data.transforms import registry
from data.transforms.dataset_synapse import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)


def test_register_transform_decorator_adds_transform():
    @registry.register_transform("_TestTransform")
    class _TestTransform:
        pass

    resolved = registry.get_transform_cls("_TestTransform")
    assert resolved is _TestTransform


def test_get_transform_cls_falls_back_to_torchvision():
    torchvision_cls = registry.get_transform_cls("Resize")
    assert torchvision_cls.__name__ == "Resize"


def _make_image(shape, value):
    array = np.full(shape, value, dtype=np.uint8)
    return Image.fromarray(array)


def test_dataset_synapse_resize_and_flip(tmp_path):
    image = _make_image((6, 6, 3), 100)
    mask = _make_image((6, 6, 3), 50)
    edge = _make_image((6, 6, 3), 25)
    masked = _make_image((6, 6, 3), 75)

    resize = Resize((4, 4))
    cropped_image, cropped_mask, cropped_edge, cropped_masked = resize(image, mask, edge, masked)
    assert cropped_image.size == (4, 4)
    assert cropped_mask.size == (4, 4)
    assert cropped_edge.size == (4, 4)
    assert cropped_masked.size == (4, 4)

    flip = RandomHorizontalFlip(1.0)
    flipped_image, flipped_mask, flipped_edge, flipped_masked = flip(image, mask, edge, masked)
    assert flipped_image.size == image.size
    assert flipped_mask.size == mask.size
    assert flipped_edge.size == edge.size
    assert flipped_masked.size == masked.size


def test_dataset_synapse_compose_and_tensor_conversion():
    image = _make_image((4, 4, 3), 120)
    mask = _make_image((4, 4, 3), 10)
    edge = _make_image((4, 4, 3), 20)
    masked = _make_image((4, 4, 3), 30)

    pipeline = Compose(
        [
            Resize((2, 2)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
        ]
    )

    result = pipeline(image, mask, edge, masked)

    assert set(result.keys()) == {"image", "mask", "edge", "masked"}
    assert isinstance(result["image"], torch.Tensor)
    assert result["image"].shape == (3, 2, 2)
    assert result["mask"].shape == (1, 2, 2)
