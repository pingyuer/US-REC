import argparse
import os
import tempfile
import time
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf

from data.builder import build_dataset, build_dataloader
from models import build_model
from trainers.context import TrainingContext
from trainers.hooks import CheckpointHook, LoggerHook, RecordRawHook
from trainers.trainer import Trainer


def load_dotenv(path: str = ".env", *, override: bool = False) -> None:
    """
    Minimal .env loader (no external dependency).

    - Ignores blank lines and comments.
    - Supports `KEY=value`, optional surrounding quotes.
    - Does not print values (to avoid leaking secrets).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def _load_checkpoint_into_model(trainer: Trainer, path: Path) -> None:
    payload = torch.load(str(path), map_location=trainer.device)
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    else:
        state = payload
    trainer.model.load_state_dict(state, strict=True)


def _maybe_load_checkpoint_for_test(trainer: Trainer, *, which: str) -> None:
    which = str(which).lower().strip()
    if which == "none":
        return
    ctx = getattr(trainer, "ctx", None)
    run_dir = getattr(ctx, "run_dir", None) if ctx is not None else None
    if not run_dir:
        return
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_path = ckpt_dir / ("best_model.pth" if which == "best" else "last_model.pth")
    if not ckpt_path.exists():
        print(f"[test] checkpoint missing: {ckpt_path} (skipped)")
        return
    _load_checkpoint_into_model(trainer, ckpt_path)
    print(f"[test] loaded checkpoint: {ckpt_path}")


def _load_best_model_from_mlflow(*, run_id: str, cfg, trainer: Trainer) -> None:
    # Download the best_model checkpoint from the given run_id and load it.
    mlflow_cfg = OmegaConf.to_container(cfg.get("mlflow") or {}, resolve=True) or {}
    base = str(mlflow_cfg.get("artifact_path") or "run").strip("/")
    artifact_path = f"{base}/checkpoints/best_model.pth"
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    _load_checkpoint_into_model(trainer, Path(local_path))
    print(f"[mlflow] loaded best_model from run_id={run_id} artifact={artifact_path}")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to main yaml config file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config, e.g. trainer.max_epochs=50 model.num_classes=3",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to build for training (defaults to train)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to build for validation (defaults to val)",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to build for testing (defaults to test)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Run mode: train (train+val), val (val only), test (test only)",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="When --mode train, also run test at the end (same run/artifacts).",
    )
    parser.add_argument(
        "--test-from",
        type=str,
        default="best",
        choices=["best", "last", "none"],
        help="When --run-test, which checkpoint to load before test.",
    )
    parser.add_argument(
        "--load-mlflow-best",
        type=str,
        default=None,
        help="Download and load best_model from an MLflow run_id (for val/test mode).",
    )
    return parser.parse_args()


def load_config(config_path: str, override_list: list):
    """Load and merge config from yaml file and command-line overrides."""
    base_cfg = OmegaConf.load(config_path)  # yaml → DictConfig

    if override_list:
        override_cfg = OmegaConf.from_dotlist(override_list)
        base_cfg = OmegaConf.merge(base_cfg, override_cfg)

    return base_cfg


def _flatten_config(cfg: OmegaConf) -> dict:
    """Flatten nested OmegaConf config into dot-delimited params."""

    def _recurse(value, prefix=""):
        entries = {}
        if isinstance(value, dict):
            for key, field in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                entries.update(_recurse(field, next_prefix))
        else:
            low = prefix.lower()
            if any(s in low for s in ("secret", "password", "token", "access_key")):
                entries[prefix] = "***"
                return entries
            entries[prefix] = "None" if value is None else str(value)
        return entries

    container = OmegaConf.to_container(cfg, resolve=True)
    return _recurse(container)


def _mlflow_run_active() -> bool:
    """Return True when an MLflow run exists."""
    return mlflow.active_run() is not None


def _log_params(params: dict, *, batch_size: int = 32):
    """Batch MLflow parameter logging to respect API limits."""
    if not _mlflow_run_active():
        return
    items = list(params.items())
    for idx in range(0, len(items), batch_size):
        batch = dict(items[idx : idx + batch_size])
        mlflow.log_params(batch)


def _log_config_artifact(cfg: OmegaConf):
    """Dump the active config as a YAML file and log it as an MLflow artifact."""
    if not _mlflow_run_active():
        return
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    temp_path = temp_file.name
    temp_file.close()
    OmegaConf.save(cfg, temp_path)

    try:
        mlflow.log_artifact(temp_path, artifact_path="configs")
    finally:
        os.remove(temp_path)


def main():
    """Main pipeline entry."""
    args = get_args()
    load_dotenv(".env", override=False)
    cfg = load_config(args.config, args.overrides)
    print("\n" + "=" * 60)
    print("CONFIG:")
    # Do not resolve env interpolations here (may contain secrets).
    print(OmegaConf.to_yaml(cfg, resolve=False))
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("[1/6] Building datasets...")
    try:
        train_loader = None
        val_loader = None
        test_loader = None

        if args.mode == "train":
            train_dataset = build_dataset(cfg, split=args.train_split)
            val_dataset = build_dataset(cfg, split=args.val_split)
            print("✅ Datasets built")
            train_loader = build_dataloader(train_dataset, cfg, split=args.train_split)
            val_loader = build_dataloader(val_dataset, cfg, split=args.val_split)
            print(f"✅ Train dataset: {len(train_dataset)} samples")
            print(f"✅ Val dataset: {len(val_dataset)} samples\n")
            if args.run_test:
                test_dataset = build_dataset(cfg, split=args.test_split)
                test_loader = build_dataloader(test_dataset, cfg, split=args.test_split)
                print(f"✅ Test dataset: {len(test_dataset)} samples\n")
        elif args.mode == "val":
            val_dataset = build_dataset(cfg, split=args.val_split)
            print("✅ Dataset built")
            val_loader = build_dataloader(val_dataset, cfg, split=args.val_split)
            print(f"✅ Val dataset: {len(val_dataset)} samples\n")
        else:
            test_dataset = build_dataset(cfg, split=args.test_split)
            print("✅ Dataset built")
            test_loader = build_dataloader(test_dataset, cfg, split=args.test_split)
            print(f"✅ Test dataset: {len(test_dataset)} samples\n")
    except Exception as e:
        print(f"⚠️  Dataset loading failed: {e}")
        print("   Skipping dataset building. Please check dataset config.\n")
        train_loader = None
        val_loader = None
        test_loader = None

    print("[2/6] Building model...")
    try:
        model = build_model(cfg.model)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model built with {total_params:,} parameters\n")
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        return

    if args.mode == "train" and (train_loader is None or val_loader is None):
        print("❌ Cannot proceed without datasets. Please configure dataset correctly.\n")
        return
    if args.mode == "val" and val_loader is None:
        print("❌ Cannot proceed without val dataset.\n")
        return
    if args.mode == "test" and test_loader is None:
        print("❌ Cannot proceed without test dataset.\n")
        return
    if args.mode == "train" and args.run_test and test_loader is None:
        print("❌ Cannot proceed without test dataset (--run-test).\n")
        return

    print("[3/6] Building trainer...")
    ctx = TrainingContext.from_cfg(cfg, root_dir="logs")
    ctx.save_config(cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        ctx=ctx,
        test_loader=test_loader,
    )

    # Checkpoints (best/last) saved under ctx.run_dir/checkpoints for archiving.
    trainer.register_hook(
        CheckpointHook(
            interval=int(cfg.trainer.get("save_interval", 0) or 0),
            metric_name="val_loss",
            mode="min",
            save_best=True,
            save_last=True,
            save_optimizer=False,
        )
    )
    trainer.register_hook(
        LoggerHook(
            interval=int(cfg.trainer.get("log_interval", 50)),
            log_file=str(ctx.log_file),
            console=True,
            mlflow_enabled=bool(cfg.get("mlflow")),
            upload_run_dir=bool(OmegaConf.select(cfg, "mlflow.archive_run_dir") or False),
            delete_local_run_dir=bool(OmegaConf.select(cfg, "mlflow.delete_local_run_dir") or False),
            artifact_path=str(OmegaConf.select(cfg, "mlflow.artifact_path") or "run"),
        )
    )

    records_cfg = cfg.get("records")
    if records_cfg and bool(records_cfg.get("enabled", False)):
        splits = tuple(records_cfg.get("splits", ["val", "test"]))
        trainer.register_hook(
            RecordRawHook(
                enabled=True,
                splits=splits,
                interval_epochs=int(records_cfg.get("interval_epochs", 1)),
                num_samples=int(records_cfg.get("num_samples", 16)),
                out_dir=str(records_cfg.get("out_dir", "records/raw")),
                save_pred_mask=bool(records_cfg.get("save_pred_mask", True)),
                save_gt_mask=bool(records_cfg.get("save_gt_mask", True)),
                threshold=float(records_cfg.get("threshold", 0.5)),
            )
        )
    print("✅ Trainer built\n")

    mlflow_cfg = OmegaConf.to_container(cfg.get("mlflow") or {}, resolve=True) or {}
    mlflow_enabled = bool(mlflow_cfg)

    params = _flatten_config(cfg)

    if not mlflow_enabled:
        print("[4/6] Starting run (MLflow disabled)...")
        if args.load_mlflow_best:
            print("❌ --load-mlflow-best requires mlflow config enabled.")
            return
        trainer.run(args.mode)
        if args.mode == "train" and args.run_test:
            _maybe_load_checkpoint_for_test(trainer, which=args.test_from)
            trainer.run("test")
        print("[5/6] Run completed!")
        return

    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name") or "default"
    run_name = mlflow_cfg.get("run_name") or f"{experiment_name}-{int(time.time())}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("[5/6] Starting run...")
    with mlflow.start_run(run_name=run_name, tags=mlflow_cfg.get("tags")) as active_run:
        run_id = active_run.info.run_id
        print(f"📡 MLflow run {run_id} (experiment={experiment_name})")
        trainer.ctx = ctx.with_mlflow_run_id(run_id)
        _log_params(params)
        mlflow.log_param("device", str(device))
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        if args.load_mlflow_best:
            _load_best_model_from_mlflow(run_id=args.load_mlflow_best, cfg=cfg, trainer=trainer)

        trainer.run(args.mode)
        if args.mode == "train" and args.run_test:
            _maybe_load_checkpoint_for_test(trainer, which=args.test_from)
            trainer.run("test")

        # Prefer metrics already produced during the run. Avoid doing an extra
        # full validation/test pass at the end (can be very expensive).
        final_metrics = getattr(trainer, "last_val_metrics", None) or getattr(trainer, "last_test_metrics", None) or {}

        if "val_loss" in final_metrics:
            mlflow.log_metric("final_val_loss", float(final_metrics["val_loss"]))
        if "test_loss" in final_metrics:
            mlflow.log_metric("final_test_loss", float(final_metrics["test_loss"]))
        mlflow.log_metric("epochs", trainer.epoch + 1)

        best_metric = trainer.best_metric
        best_epoch = trainer.best_epoch + 1
        if best_metric == float("inf") and "val_loss" in final_metrics:
            best_metric = float(final_metrics["val_loss"])
            best_epoch = trainer.epoch + 1

        mlflow.log_metric("best_val_loss", best_metric)
        mlflow.log_metric("best_epoch", best_epoch)

        _log_config_artifact(cfg)

    print("[6/6] Run completed!")


if __name__ == "__main__":
    main()
