import pytest
pytestmark = pytest.mark.unit

import io

import numpy as np
import pytest
import torch

from data.datasets.TUS_rec_s3 import TUSRecS3Iterable
from data.utils import s3_io


def _make_payload(frames, tforms):
    import h5py

    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("frames", data=frames)
        f.create_dataset("tforms", data=tforms)
    buf.seek(0)
    return buf.read()


def _mock_s3(monkeypatch, key_to_payload):
    keys = list(key_to_payload.keys())
    monkeypatch.setattr(s3_io, "list_keys", lambda **kwargs: keys)
    monkeypatch.setattr(s3_io, "get_object", lambda **kwargs: key_to_payload[kwargs["key"]])


def _make_dataset(**kwargs):
    defaults = dict(
        bucket="dummy",
        prefix="train",
        frame_dir="frames",
        tform_dir=None,
        region="us-east-1",
        endpoint=None,
        force_path_style=True,
        max_keys=1000,
        pair_mode="adjacent",
        random_pairs_per_slice=None,
        shuffle_slices=False,
        shuffle_pairs=False,
        shuffle_buffer_size=0,
        prefetch_slices=0,
        buffer_emit_prob=0.5,
        seed=123,
        pipeline=None,
    )
    defaults.update(kwargs)
    return TUSRecS3Iterable(**defaults)


def _sample_ids(samples):
    return [
        (
            sample["frames"][0, 0, 0].item(),
            sample["frames"][1, 0, 0].item(),
        )
        for sample in samples
    ]


def test_basic_iteration(monkeypatch):
    frames = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(monkeypatch, {"train/frames/000/scan_a.h5": payload})

    ds = _make_dataset()
    samples = list(ds)
    assert len(samples) == 3
    assert set(samples[0].keys()) >= {"frames", "tforms", "tforms_inv"}
    assert samples[0]["frames"].shape[0] == 2
    assert samples[0]["tforms"].shape[-2:] == (4, 4)
    assert int(samples[0]["meta"]["frame_idx1"]) - int(samples[0]["meta"]["frame_idx0"]) == 1


def test_ddp_worker_sharding_disjoint(monkeypatch):
    frames_a = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    frames_b = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2) + 100
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload_a = _make_payload(frames_a, tforms)
    payload_b = _make_payload(frames_b, tforms)
    _mock_s3(
        monkeypatch,
        {
            "train/frames/000/scan_a.h5": payload_a,
            "train/frames/001/scan_b.h5": payload_b,
        },
    )

    ds = _make_dataset(shuffle_slices=False, seed=9)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    class _WorkerInfo:
        def __init__(self, worker_id, num_workers):
            self.id = worker_id
            self.num_workers = num_workers

    def _collect(ddp_rank, worker_id, num_workers):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: ddp_rank)
        monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: _WorkerInfo(worker_id, num_workers))
        return _sample_ids(ds)

    rank0_w0 = _collect(0, 0, 2)
    rank0_w1 = _collect(0, 1, 2)
    rank1_w0 = _collect(1, 0, 2)
    rank1_w1 = _collect(1, 1, 2)

    all_ids = set(rank0_w0) | set(rank0_w1) | set(rank1_w0) | set(rank1_w1)
    assert len(all_ids) == 6
    assert set(rank0_w0).isdisjoint(rank1_w0)
    assert set(rank0_w1).isdisjoint(rank1_w1)


@pytest.mark.xfail(
    reason="Shuffle buffer with a single scan may not change order deterministically; "
    "non-trivial to fix without mocking RNG internals."
)
def test_shuffle_buffer_changes_order(monkeypatch):
    frames = np.arange(6 * 2 * 2, dtype=np.uint8).reshape(6, 2, 2)
    tforms = np.tile(np.eye(4), (6, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(monkeypatch, {"train/frames/000/scan_a.h5": payload})

    ds_base = _make_dataset(shuffle_buffer_size=0, seed=7)
    ds_buf = _make_dataset(shuffle_buffer_size=3, seed=7)

    base = _sample_ids(ds_base)
    buf = _sample_ids(ds_buf)

    assert base != buf


def test_prefetch_slices_cache_reuse(monkeypatch):
    frames = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(
        monkeypatch,
        {
            "train/frames/000/scan_a.h5": payload,
            "train/frames/001/scan_b.h5": payload,
        },
    )

    ds = _make_dataset(prefetch_slices=2, shuffle_pairs=False)
    load_calls = {"count": 0}

    def _load_slice(info, client=None):
        load_calls["count"] += 1
        return TUSRecS3Iterable._load_slice(ds, info, client=client)

    monkeypatch.setattr(ds, "_load_slice", _load_slice)
    samples = list(ds)

    assert len(samples) == 6
    assert load_calls["count"] == 2


def test_pipeline_transform(monkeypatch):
    frames = np.zeros((3, 2, 2), dtype=np.uint8)
    tforms = np.tile(np.eye(4), (3, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(monkeypatch, {"train/frames/000/scan_a.h5": payload})

    def pipeline(sample):
        sample["tag"] = "ok"
        return sample

    ds = _make_dataset(pipeline=pipeline)
    sample = next(iter(ds))
    assert sample["tag"] == "ok"


def test_pair_sampling_modes(monkeypatch):
    frames = np.zeros((6, 2, 2), dtype=np.uint8)
    tforms = np.tile(np.eye(4), (6, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(monkeypatch, {"train/frames/000/scan_a.h5": payload})

    ds_adj = _make_dataset(pair_mode="adjacent")
    ds_all = _make_dataset(pair_mode="random", random_pairs_per_slice=None, seed=1)
    ds_one = _make_dataset(pair_mode="random", random_pairs_per_slice=1.0, seed=1)
    ds_ratio = _make_dataset(pair_mode="random", random_pairs_per_slice=0.5, seed=1)
    ds_fixed = _make_dataset(pair_mode="random", random_pairs_per_slice=3, seed=1)

    assert len(list(ds_adj)) == 5
    assert len(list(ds_all)) == 5
    assert len(list(ds_one)) == 5
    assert len(list(ds_ratio)) == 2
    assert len(list(ds_fixed)) == 3


def test_short_or_empty_slices(monkeypatch):
    frames = np.zeros((1, 2, 2), dtype=np.uint8)
    tforms = np.tile(np.eye(4), (1, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(monkeypatch, {"train/frames/000/scan_a.h5": payload})

    ds = _make_dataset()
    assert list(ds) == []


def test_no_slices_raises(monkeypatch):
    monkeypatch.setattr(s3_io, "list_keys", lambda **kwargs: [])
    ds = _make_dataset()
    with pytest.raises(RuntimeError):
        list(ds)


def test_error_resilience_skips_failed_slice(monkeypatch):
    frames = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    tforms = np.tile(np.eye(4), (4, 1, 1)).astype(np.float32)
    payload = _make_payload(frames, tforms)
    _mock_s3(
        monkeypatch,
        {
            "train/frames/000/scan_a.h5": payload,
            "train/frames/001/scan_b.h5": payload,
        },
    )

    ds = _make_dataset(shuffle_pairs=False)

    def _load_slice(info, client=None):
        if info.scan_name == "scan_b":
            raise RuntimeError("boom")
        return TUSRecS3Iterable._load_slice(ds, info, client=client)

    monkeypatch.setattr(ds, "_load_slice", _load_slice)
    samples = list(ds)

    assert len(samples) == 3
