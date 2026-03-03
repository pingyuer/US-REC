"""Integration test: verify main_rec.py can be imported and parsed."""
import pytest
pytestmark = pytest.mark.integration

import torch


class DummySeq(torch.utils.data.Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        frames = torch.zeros((2, 4, 4))
        tforms = torch.eye(4).repeat(2, 1, 1)
        tforms_inv = torch.eye(4).repeat(2, 1, 1)
        return frames, tforms, tforms_inv


def test_main_rec_is_importable():
    """main_rec module should import without side effects."""
    import main_rec  # noqa: F401
    assert hasattr(main_rec, "main")
    assert hasattr(main_rec, "load_config")


def test_main_rec_parse_args():
    """CLI parser should accept --config flag."""
    import main_rec
    args = main_rec._parse_args(["--config", "configs/demo_rec24_ete.yml"])
    assert args.config == "configs/demo_rec24_ete.yml"
    assert not args.eval_only
    assert not args.dry_run


def test_dummy_dataset_basic_contract():
    """Verify dummy dataset contract: returns (frames, tforms, tforms_inv)."""
    ds = DummySeq()
    assert len(ds) == 3
    frames, tforms, tforms_inv = ds[0]
    assert frames.shape == (2, 4, 4)
    assert tforms.shape == (2, 4, 4)
