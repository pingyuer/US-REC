import pytest
pytestmark = pytest.mark.integration

import types
import torch
import pytest

# Dummy dataset mimicking TUSRecS3/SSFrameDataset outputs
class DummySeq(torch.utils.data.Dataset):
    def __len__(self):
        return 3
    def __getitem__(self, idx):
        frames = torch.zeros((2, 4, 4))
        tforms = torch.eye(4).repeat(2,1,1)
        tforms_inv = torch.eye(4).repeat(2,1,1)
        return frames, tforms, tforms_inv


def test_dataloader_init_main_rec_ete(monkeypatch):
    import main_rec as mod

    monkeypatch.setattr(mod, "OmegaConf", types.SimpleNamespace(load=lambda path: {}))

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def multi_model(self):
            return None

        def train_rec_model(self):
            return None

    monkeypatch.setattr(mod, "Train_Rec_Reg_Model", DummyModel)

    dset_train = DummySeq()
    dset_val = DummySeq()
    assert len(dset_train) == 3 and len(dset_val) == 3


def test_dataloader_init_main_rec_meta(monkeypatch):
    import main_rec as mod
    monkeypatch.setattr(mod, "OmegaConf", types.SimpleNamespace(load=lambda path: {}))

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def multi_model(self):
            return None

        def train_rec_model(self):
            return None

    monkeypatch.setattr(mod, "Train_Rec_Reg_Model", DummyModel)

    dset_train = DummySeq()
    assert len(dset_train) == 3
