import os

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from trainers.context import TrainingContext
from trainers.hooks.base_hook import Hook
from trainers.hooks.logger_hook import LoggerHook
from trainers.trainer import Trainer


class DummySegDataset(Dataset):
    def __init__(self, n: int = 4, h: int = 16, w: int = 16, num_classes: int = 2):
        self.n = n
        self.h = h
        self.w = w
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(3, self.h, self.w)
        mask = torch.randint(0, self.num_classes, (self.h, self.w), dtype=torch.long)
        return {"image": image, "mask": mask}


class SpyHook(Hook):
    priority = 999

    def __init__(self):
        self.events = []

    def before_run(self, trainer, mode: str = "train"):
        self.events.append(("before_run", {"mode": mode}))

    def after_run(self, trainer, mode: str = "train"):
        self.events.append(("after_run", {"mode": mode}))

    def before_train(self, trainer):
        self.events.append(("before_train", None))

    def before_epoch(self, trainer):
        self.events.append(("before_epoch", {"epoch": trainer.epoch}))

    def before_step(self, trainer):
        self.events.append(("before_step", {"global_step": trainer.global_step}))

    def after_step(self, trainer, log_buffer=None):
        self.events.append(("after_step", dict(log_buffer or {})))

    def after_epoch(self, trainer, log_buffer=None):
        self.events.append(("after_epoch", dict(log_buffer or {})))

    def before_val(self, trainer):
        self.events.append(("before_val", {"epoch": trainer.epoch}))

    def after_val(self, trainer, log_buffer=None):
        self.events.append(("after_val", dict(log_buffer or {})))

    def before_test(self, trainer):
        self.events.append(("before_test", {"epoch": trainer.epoch}))

    def after_test(self, trainer, log_buffer=None):
        self.events.append(("after_test", dict(log_buffer or {})))

    def after_train(self, trainer):
        self.events.append(("after_train", None))


def make_cfg(*, max_epochs=1, log_interval=1, eval_interval=1):
    return OmegaConf.create(
        {
            "trainer": {
                "max_epochs": max_epochs,
                "log_interval": log_interval,
                "eval_interval": eval_interval,
            },
        }
    )


def make_trainer(*, cfg, ctx=None, test_loader=None):
    model = nn.Conv2d(3, 2, kernel_size=1)
    train_loader = DataLoader(DummySegDataset(n=4), batch_size=2, shuffle=False)
    val_loader = DataLoader(DummySegDataset(n=4), batch_size=2, shuffle=False)
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device="cpu",
        ctx=ctx,
        test_loader=test_loader,
    )


def test_context_creates_expected_paths_and_saves_config(tmp_path):
    cfg = OmegaConf.create({"experiment": {"name": "expA"}})
    ctx = TrainingContext.from_cfg(cfg, root_dir=tmp_path, run_name="run1")
    ctx.save_config(cfg)

    assert ctx.run_dir.exists()
    assert ctx.run_dir == tmp_path / "expA" / "run1"
    assert ctx.config_file.exists()


def test_trainer_calls_hooks_with_buffers():
    cfg = make_cfg(max_epochs=1, log_interval=1, eval_interval=1)
    trainer = make_trainer(cfg=cfg)
    spy = SpyHook()
    trainer.register_hook(spy)
    trainer.train()

    names = [n for n, _ in spy.events]
    assert "before_train" in names
    assert "after_train" in names
    assert "after_step" in names
    assert "after_epoch" in names
    assert "after_val" in names

    after_step = next(payload for name, payload in spy.events if name == "after_step")
    assert "global_step" in after_step
    assert "loss" in after_step
    assert "lr" in after_step


def test_logger_hook_writes_local_log(tmp_path):
    cfg = make_cfg(max_epochs=1, log_interval=1, eval_interval=1)
    ctx = TrainingContext.create(exp_name="exp", run_name="run", root_dir=tmp_path)
    ctx.save_config(cfg)
    trainer = make_trainer(cfg=cfg, ctx=ctx)
    trainer.register_hook(
        LoggerHook(
            interval=1,
            log_file=str(ctx.log_file),
            console=False,
            mlflow_enabled=False,
            upload_run_dir=False,
            delete_local_run_dir=False,
        )
    )
    trainer.train()

    content = ctx.log_file.read_text(encoding="utf-8")
    assert "[run] log_file=" in content
    assert "[train]" in content
    assert "[epoch]" in content
    assert "[val]" in content


def test_logger_hook_uploads_and_deletes_run_dir_when_configured(tmp_path, monkeypatch):
    cfg = make_cfg(max_epochs=1, log_interval=1, eval_interval=1)
    ctx = TrainingContext.create(exp_name="exp", run_name="run", root_dir=tmp_path)
    ctx.save_config(cfg)
    trainer = make_trainer(cfg=cfg, ctx=ctx)

    calls = {"log_artifacts": 0}

    class FakeMlflow:
        def active_run(self):
            return object()

        def log_artifacts(self, local_dir, artifact_path=None):
            calls["log_artifacts"] += 1

        def log_metric(self, *args, **kwargs):
            pass

    monkeypatch.setattr("trainers.hooks.logger_hook.mlflow", FakeMlflow())

    trainer.register_hook(
        LoggerHook(
            interval=1,
            log_file=str(ctx.log_file),
            console=False,
            mlflow_enabled=True,
            upload_run_dir=True,
            delete_local_run_dir=True,
            artifact_path="run",
        )
    )
    trainer.run("train")

    assert calls["log_artifacts"] == 1
    assert not ctx.run_dir.exists()


def test_trainer_run_calls_hooks_for_test_mode(tmp_path):
    cfg = make_cfg(max_epochs=1, log_interval=1, eval_interval=1)
    test_loader = DataLoader(DummySegDataset(n=2), batch_size=1, shuffle=False)
    trainer = make_trainer(cfg=cfg, test_loader=test_loader)
    spy = SpyHook()
    trainer.register_hook(spy)
    trainer.run("test")

    names = [n for n, _ in spy.events]
    assert "before_run" in names
    assert "before_test" in names
    assert "after_test" in names
    assert "after_run" in names
