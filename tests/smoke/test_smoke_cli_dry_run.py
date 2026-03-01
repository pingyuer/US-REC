"""Smoke test 5: CLI dry-run.

Verifies that ``python main_rec.py --config ... --dry-run`` exits
successfully (return code 0) when invoked programmatically via
``main_rec.main(argv=...)``.

Because the real dataset requires S3 access, we test with a patched
``build_datasets`` that returns tiny synthetic datasets.

Note: This test is marked xfail because the full execution path requires
the ``monai`` package (via ``trainers.rec_trainer``).  Run manually in an
environment where monai is installed:

    python main_rec.py --config configs/demo_rec24_ete.yml --dry-run
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from torch.utils.data import IterableDataset

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "demo_rec24_ete.yml")


class _FakeDataset(IterableDataset):
    """Minimal iterable dataset that yields one pair-sample."""

    def __init__(self, num_samples=2, H=64, W=64):
        self.num_samples = num_samples
        self.H = H
        self.W = W

    def __iter__(self):
        for _ in range(2):
            frames = torch.randn(self.num_samples, self.H, self.W)
            tforms = torch.eye(4).unsqueeze(0).repeat(self.num_samples, 1, 1)
            tforms_inv = tforms.clone()
            yield {
                "frames": frames,
                "tforms": tforms,
                "tforms_inv": tforms_inv,
                "meta": {"scan_id": "fake_scan", "frame_idx0": 0, "frame_idx1": 1},
            }

    def get_example(self):
        return next(iter(self))


def _fake_build_datasets(cfg):
    ds = _FakeDataset()
    return ds, ds, ds


@pytest.mark.smoke
@pytest.mark.xfail(
    reason=(
        "Requires the full rec_trainer/monai chain. "
        "Run manually: python main_rec.py --config configs/demo_rec24_ete.yml --dry-run"
    ),
    strict=False,
)
class TestSmokeCliDryRun:
    """Invoke main_rec in dry-run mode with mocked data."""

    def test_dry_run_returns_zero(self):
        """Dry-run should build components, check 1 batch, and exit 0."""
        with patch("main_rec.build_datasets", side_effect=_fake_build_datasets):
            from main_rec import main  # noqa: PLC0415

            rc = main(["--config", CONFIG_PATH, "--dry-run"])
        assert rc == 0, f"Dry-run exited with code {rc}"
