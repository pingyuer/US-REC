"""Unified evaluation entry-point.

Usage::

    # Evaluate only (no training)
    python -m eval.run_eval --config configs/demo_rec24_ete.yml --eval-only

    # With sampling limits
    python -m eval.run_eval --config configs/demo_rec24_ete.yml --eval-only \
        --max-scans 1 --max-frames 10

    # Dry-run: build components and verify one batch, then exit
    python -m eval.run_eval --config configs/demo_rec24_ete.yml --dry-run

This module delegates to :func:`eval.builder.build_evaluator` for
component construction and calls :meth:`RecEvaluator.run` for evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Ensure repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.builder import build_evaluator  # noqa: E402
from trainers.builder import load_dotenv  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TUS-REC unified evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--dry-run", action="store_true", help="Build components & check 1 batch, then exit")
    parser.add_argument("--max-scans", type=int, default=None, help="Limit evaluation scans")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per scan")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--export-json", action="store_true", help="Export per-scan JSON")
    parser.add_argument("--export-npz", action="store_true", help="Export pred/gt transforms as npz")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dot-list overrides")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    load_dotenv(".env", override=False)

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Apply CLI sampling limits into config
    if args.max_scans is not None:
        OmegaConf.update(cfg, "eval.max_scans", args.max_scans)
    if args.max_frames is not None:
        OmegaConf.update(cfg, "data.max_frames_per_scan", args.max_frames)
    if args.output_dir is not None:
        OmegaConf.update(cfg, "paths.output_dir", args.output_dir)

    components = build_evaluator(cfg)
    trainer = components["trainer"]
    evaluator = components["evaluator"]
    loader = components["loader"]
    model = components["model"]

    if args.dry_run:
        print("[dry-run] Components built successfully.")
        batch = next(iter(loader))
        if isinstance(batch, dict):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"  batch[{k!r}]: shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  batch[{k!r}]: type={type(v).__name__}")
        else:
            print(f"  batch type: {type(batch).__name__}")
        print("[dry-run] Data readable. Exiting.")
        return 0

    metrics = evaluator.run(
        model=model,
        loader=loader,
        cfg=cfg,
        trainer=trainer,
        mode="test",
    )

    # Print summary
    print("\n=== Evaluation results ===")
    for k, v in sorted(metrics.items()):
        if k == "tusrec_per_scan":
            continue
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

    # Export if requested
    if args.export_json or args.export_npz:
        from eval.export import export_results
        out_dir = OmegaConf.select(cfg, "paths.output_dir") or "eval_output"
        export_results(
            metrics,
            out_dir=out_dir,
            save_json=args.export_json,
            save_npz=args.export_npz,
        )
        print(f"\nExported to {out_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
