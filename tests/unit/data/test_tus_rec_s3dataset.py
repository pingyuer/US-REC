import pytest
pytestmark = pytest.mark.unit

import io

import numpy as np
import torch

from data.datasets.TUS_rec_s3 import TUSRecS3Dataset, TUSRecS3Iterable
from data.utils import s3_io


def _make_payload(frames, tforms):
    import h5py

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("frames", data=frames)
        f.create_dataset("tforms", data=tforms)
    buf.seek(0)
    return buf.read()


def _mock_keys():
    return [
        "train/frames/000/scan_a.h5",
        "train/frames/001/scan_b.h5",
    ]


def test_dataset_alias():
    assert TUSRecS3Dataset is TUSRecS3Iterable


@pytest.mark.xfail(
    reason="global_slice_keys attribute was removed from TUSRecS3Iterable; "
    "test documents the intended API for when it is re-added."
)
def test_global_slice_keys_stable(monkeypatch):
    frames = np.zeros((4, 2, 2), dtype=np.uint8)
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)

    keys = _mock_keys()
    monkeypatch.setattr(s3_io, "list_keys", lambda **kwargs: keys)
    monkeypatch.setattr(s3_io, "get_object", lambda **kwargs: payload)

    ds = TUSRecS3Dataset(
        bucket="dummy",
        prefix="train",
        frame_dir="frames",
        shuffle_slices=False,
    )
    assert ds.global_slice_keys == sorted(keys)


def test_worker_sharding(monkeypatch):
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload_a = _make_payload(np.full((4, 2, 2), 1, dtype=np.uint8), tforms)
    payload_b = _make_payload(np.full((4, 2, 2), 2, dtype=np.uint8), tforms)
    keys = _mock_keys()

    def get_object(**kwargs):
        key = kwargs["key"]
        if "scan_a" in key:
            return payload_a
        return payload_b

    monkeypatch.setattr(s3_io, "list_keys", lambda **kwargs: keys)
    monkeypatch.setattr(s3_io, "get_object", get_object)

    ds_worker0 = TUSRecS3Dataset(
        bucket="dummy",
        prefix="train",
        frame_dir="frames",
        shuffle_slices=False,
    )
    ds_worker1 = TUSRecS3Dataset(
        bucket="dummy",
        prefix="train",
        frame_dir="frames",
        shuffle_slices=False,
    )
    ds_worker0.set_epoch(0)
    ds_worker1.set_epoch(0)

    # Force distinct worker sharding by faking worker_info.
    class _WorkerInfo:
        def __init__(self, worker_id, num_workers):
            self.id = worker_id
            self.num_workers = num_workers

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _WorkerInfo(0, 2))
    sample0 = next(iter(ds_worker0))
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _WorkerInfo(1, 2))
    sample1 = next(iter(ds_worker1))
    assert sample0["frames"][0, 0, 0].item() != sample1["frames"][0, 0, 0].item()
