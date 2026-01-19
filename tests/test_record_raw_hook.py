from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.builder import default_collate_fn
from trainers.context import TrainingContext
from trainers.hooks.record_raw_hook import RecordRawHook
from trainers.trainer import Trainer


class RawFileDataset(Dataset):
    def __init__(self, root: Path, n: int = 3):
        self.img_dir = str(root / "imgs")
        (root / "imgs").mkdir(parents=True, exist_ok=True)
        self.files: list[str] = []
        for idx in range(n):
            arr = np.full((16, 16, 3), fill_value=idx * 30, dtype=np.uint8)
            p = root / "imgs" / f"m{idx:03d}.png"
            Image.fromarray(arr).save(p)
            self.files.append(p.name)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Minimal sample for evaluator: image tensor + mask tensor + meta.
        image = torch.randn(3, 16, 16)
        mask = torch.zeros(16, 16, dtype=torch.long)
        meta = {"img_file": self.files[idx]}
        return {"image": image, "mask": mask, "meta": meta}


def make_trainer(*, cfg, ctx, val_loader):
    model = nn.Conv2d(3, 2, kernel_size=1)
    with torch.no_grad():
        model.weight.zero_()
        if model.bias is not None:
            model.bias.zero_()
    return Trainer(
        model=model,
        train_loader=None,
        val_loader=val_loader,
        cfg=cfg,
        device="cpu",
        ctx=ctx,
        test_loader=None,
    )


def test_record_raw_hook_saves_original_bytes(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "trainer": {"max_epochs": 1, "eval_interval": 1},
            "records": {
                "enabled": True,
                "splits": ["val"],
                "interval_epochs": 1,
                "num_samples": 2,
            },
        }
    )
    ctx = TrainingContext.create(exp_name="exp", run_name="run", root_dir=tmp_path)
    dataset = RawFileDataset(tmp_path, n=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=default_collate_fn)
    trainer = make_trainer(cfg=cfg, ctx=ctx, val_loader=loader)
    trainer.register_hook(
        RecordRawHook(
            enabled=True,
            splits=("val",),
            interval_epochs=1,
            num_samples=2,
            save_pred_mask=True,
            save_gt_mask=True,
            threshold=0.5,
        )
    )

    trainer.run("val")

    out_dir = ctx.run_dir / "records" / "raw" / "val" / "epoch_1"
    assert out_dir.exists()
    saved = sorted(p.name for p in out_dir.glob("*.png"))
    assert len(saved) == 6  # 2 raw + 2 pred + 2 gt

    # Ensure bytes match originals.
    raw_names = [n for n in saved if not (n.endswith("_pred.png") or n.endswith("_gt.png"))]
    assert len(raw_names) == 2
    for name in raw_names:
        assert (Path(dataset.img_dir) / name).read_bytes() == (out_dir / name).read_bytes()

    # With zero logits, argmax => class 0 everywhere, so pred mask should be all zero.
    for name in saved:
        if name.endswith("_pred.png") or name.endswith("_gt.png"):
            arr = np.asarray(Image.open(out_dir / name))
            assert set(np.unique(arr)).issubset({0})

    manifest = ctx.run_dir / "records" / "raw" / "manifest.jsonl"
    assert manifest.exists()
