"""Covers training-side sample normalization transforms."""

import pytest
pytestmark = pytest.mark.unit


import numpy as np
import torch

from data.transforms.finalize_ops import FinalizeSegSample


def test_finalize_seg_sample_converts_and_drops_extra_keys():
    sample = {
        "image": np.zeros((8, 8, 3), dtype=np.float32),
        "mask": np.ones((8, 8), dtype=np.uint8),
        "meta": {"img_file": "x.png"},
        "raw_bytes": b"123",
        "segments": [np.array([[0, 0], [0, 1], [1, 1], [1, 0]])],
        "s3": {"endpoint": "http://minio"},
    }

    out = FinalizeSegSample()(sample)
    assert set(out.keys()) == {"image", "mask", "meta"}
    assert torch.is_tensor(out["image"]) and out["image"].shape == (3, 8, 8)
    assert out["image"].dtype == torch.float32
    assert torch.is_tensor(out["mask"]) and out["mask"].shape == (8, 8)
    assert out["mask"].dtype == torch.long
    assert out["meta"]["img_file"] == "x.png"

