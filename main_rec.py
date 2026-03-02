"""TUS-REC training / evaluation CLI.

Usage
-----
Train::

    python main_rec.py --config configs/demo_rec24_ete.yml

Evaluate only::

    python main_rec.py --config configs/demo_rec24_ete.yml --eval-only

Sampled evaluate::

    python main_rec.py --config configs/demo_rec24_ete.yml --eval-only \
        --max-scans 1 --max-frames 10

Dry-run (build components + read 1 batch, then exit)::

    python main_rec.py --config configs/demo_rec24_ete.yml --dry-run

All heavy lifting is delegated to ``trainers.builder`` / ``eval.builder``.
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

from trainers.builder import (
    build_datasets,
    build_hooks,
    build_rec_trainer,
    build_training_context,
    limit_dataset,
    load_dotenv,
    resolve_run_dirs,
)
from trainers.rec_evaluator import RecEvaluator
from trainers.utils.checkpoint_loader import load_checkpoint


def load_config(config_path: str, overrides: list[str]):
    base_cfg = OmegaConf.load(config_path)
    if overrides:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(overrides))
    return base_cfg


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="TUS-REC train / eval CLI")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--eval-only", action="store_true", help="Run evaluation only (no training)")
    p.add_argument(
        "--eval-smoke",
        action="store_true",
        help=(
            "Quick smoke eval: implies --eval-only with max_scans=1 and max_frames_per_scan=10."
            " Useful for checking the full eval pipeline end-to-end without waiting for all data."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Build components, read 1 batch, exit")
    p.add_argument("--max-scans", type=int, default=None, help="Limit scans (eval/data)")
    p.add_argument("--max-frames", type=int, default=None, help="Limit frames per scan")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="RUN_ID_OR_PATH",
        help=(
            "Checkpoint to load.  Prefix with 'mlflow:' for a MLflow run ID "
            "(e.g. 'mlflow:abc123'), or supply a local .pt file path.  "
            "Overrides checkpoint.mlflow_run_id / checkpoint.local_path in config."
        ),
    )
    p.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    # --eval-smoke is sugar for --eval-only --max-scans 1 --max-frames 10
    if args.eval_smoke:
        args.eval_only = True
        if args.max_scans is None:
            args.max_scans = 1
        if args.max_frames is None:
            args.max_frames = 10
    load_dotenv(".env", override=False)

    # 1. Load config + apply CLI overrides
    cfg = load_config(args.config, args.overrides)
    if args.max_scans is not None:
        OmegaConf.update(cfg, "data.max_scans", args.max_scans)
        OmegaConf.update(cfg, "eval.max_scans", args.max_scans)
    if args.max_frames is not None:
        OmegaConf.update(cfg, "data.max_frames_per_scan", args.max_frames)
    if args.seed is not None:
        OmegaConf.update(cfg, "seed", args.seed)

    # 2. Seed
    seed = OmegaConf.select(cfg, "seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    # 3. Context & paths
    ctx = build_training_context(cfg)
    ctx.save_config(cfg)
    dirs = resolve_run_dirs(cfg, ctx)
    save_path = str(dirs["run_dir"])

    runtime_cfg = OmegaConf.select(cfg, "runtime") or {}
    gpu_ids = runtime_cfg.get("gpu_ids")
    if gpu_ids:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Datasets (with optional sampling limits)
    dset_train, dset_val, dset_test = build_datasets(cfg)
    max_scans = OmegaConf.select(cfg, "data.max_scans")
    max_frames = OmegaConf.select(cfg, "data.max_frames_per_scan")
    if max_scans is not None or max_frames is not None:
        dset_val = limit_dataset(dset_val, max_scans=max_scans, max_frames_per_scan=max_frames)
        dset_test = limit_dataset(dset_test, max_scans=max_scans, max_frames_per_scan=max_frames)

    # 5. Build trainer (holds model, calibration, transforms)
    trainer = build_rec_trainer(
        cfg,
        save_path=save_path,
        dset_train=dset_train,
        dset_val=dset_val,
        device=device,
        writer=None,
    )
    trainer.ctx = ctx

    # ------------------------------------------------------------------
    # Apply --checkpoint CLI override into config before loading
    # ------------------------------------------------------------------
    if args.checkpoint:
        val = args.checkpoint
        if val.startswith("mlflow:"):
            OmegaConf.update(cfg, "checkpoint.mlflow_run_id", val[len("mlflow:"):])
        else:
            OmegaConf.update(cfg, "checkpoint.local_path", val)

    # --dry-run: verify build + 1 batch, then exit
    if args.dry_run:
        print("[dry-run] Components built successfully.")
        loader = trainer.val_loader_rec
        batch = next(iter(loader))
        if isinstance(batch, dict):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"  batch[{k!r}]: shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  batch[{k!r}]: type={type(v).__name__}")
        print("[dry-run] Data readable. Exiting.")
        return 0

    # --eval-only: run evaluation and exit
    if args.eval_only:
        from eval.export import export_results  # noqa: PLC0415

        ckpt_cfg = OmegaConf.select(cfg, "checkpoint") or {}
        if OmegaConf.is_config(ckpt_cfg):
            ckpt_cfg = OmegaConf.to_container(ckpt_cfg, resolve=True)
        if bool(ckpt_cfg.get("load_on_eval", True)):
            load_checkpoint(trainer.model, cfg, device, ctx=ctx)

        # Build hooks and start the MLflow run so VizHook can upload artifacts
        eval_hooks = build_hooks(cfg, ctx, trainer=trainer)
        mlflow_hook = next((h for h in eval_hooks if h.__class__.__name__ == "MLflowHook"), None)
        if mlflow_hook is not None:
            mlflow_hook.before_run(trainer, mode="test")
        eval_callbacks = [h for h in eval_hooks if hasattr(h, "on_end")]

        evaluator = RecEvaluator(device=device)
        metrics = evaluator.run(
            model=trainer.model,
            loader=trainer.val_loader_rec,
            cfg=cfg,
            trainer=trainer,
            mode="test",
            callbacks=eval_callbacks,
        )
        print("\n=== Evaluation results ===")
        for k, v in sorted(metrics.items()):
            if k in ("tusrec_per_scan", "scan_globals"):
                continue
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.6f}")
        out_dir = str(dirs.get("run_dir", "eval_output"))
        export_results(metrics, out_dir=out_dir, save_json=True)
        print(f"\nResults saved to {out_dir}/")
        if mlflow_hook is not None:
            mlflow_hook.after_run(trainer, mode="test")
        return 0

    # 6. Normal training path
    hooks = build_hooks(cfg, ctx, trainer=trainer)
    for h in hooks:
        trainer.register_hook(h)
    trainer.multi_model()

    # Load checkpoint when resuming (retain_epoch > 0) or --checkpoint given
    ckpt_cfg = OmegaConf.select(cfg, "checkpoint") or {}
    if OmegaConf.is_config(ckpt_cfg):
        ckpt_cfg = OmegaConf.to_container(ckpt_cfg, resolve=True)
    retain_epoch = int(getattr(trainer, "retain_epoch", 0) or 0)
    if bool(ckpt_cfg.get("load_on_resume", True)) and (
        retain_epoch > 0 or args.checkpoint
    ):
        load_checkpoint(trainer.model, cfg, device, ctx=ctx)

    trainer.train_rec_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())
