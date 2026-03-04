"""K-root stitch evaluation CLI.

Loads SHORT and LONG checkpoints, runs stitch on each scan, computes
metrics (GPE/LPE/drift for short_only / long_only / fused), and saves
debug CSV per scan.

Usage — separate checkpoints::

    python evaluate_kroot_stitch.py \\
        --short-ckpt logs/short_run/checkpoints/best.pt \\
        --long-ckpt  logs/long_run/checkpoints/best.pt \\
        --config     configs/demo_rec24_ete_long_kroot.yml \\
        --max-scans 2

Usage — unified dual checkpoint (from KRootDualTrainer)::

    python evaluate_kroot_stitch.py \\
        --ckpt logs/dual_run/checkpoints/best.pt \\
        --config configs/demo_rec24_ete_kroot_dual.yml \\
        --max-scans 2

Smoke test with synthetic data (no S3 needed)::

    python evaluate_kroot_stitch.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import torch
from omegaconf import OmegaConf


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="K-root stitch evaluation")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--ckpt", default=None, help="Unified dual checkpoint (KRootDualTrainer)")
    p.add_argument("--short-ckpt", default=None, help="Path to short model checkpoint (separate mode)")
    p.add_argument("--long-ckpt", default=None, help="Path to long model checkpoint (separate mode)")
    p.add_argument("--max-scans", type=int, default=None)
    p.add_argument("--out-dir", default="logs/kroot_eval")
    p.add_argument("--smoke", action="store_true", help="Run on synthetic data (no S3)")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--s", type=int, default=None)
    p.add_argument("--no-endpoint-interp", action="store_true")
    return p.parse_args(argv)


def _build_model(cfg, device):
    """Build a LongSeqPoseModel from config (no auxΔ)."""
    from models.temporal.model_longseq import LongSeqPoseModel

    def _get(path, default=None):
        v = OmegaConf.select(cfg, path)
        return default if v is None else v

    return LongSeqPoseModel(
        backbone=str(_get("model.encoder.backbone", "efficientnet_b0")),
        in_channels=1,
        token_dim=int(_get("model.transformer.d_model", 256)),
        n_heads=int(_get("model.transformer.n_heads", 4)),
        n_layers=int(_get("model.transformer.n_layers", 4)),
        dim_feedforward=int(_get("model.transformer.dim_feedforward", 1024)),
        window_size=int(_get("model.transformer.window_size", 64)),
        dropout=0.0,
        rotation_rep=str(_get("model.pose_head.rotation_rep", "rot6d")),
        aux_intervals=[],
        memory_size=0,
        pretrained_backbone=False,
    ).to(device)


def run_smoke(args):
    """Smoke test with synthetic data."""
    import math
    from data.datasets.dual_kroot_window import (
        SyntheticShortWindowDataset,
        SyntheticLongWindowDataset,
    )
    from trainers.kroot_trainer import KRootTrainer
    from torch.utils.data import DataLoader
    from eval.kroot_stitch import (
        stitch_long_base_short_refine,
        compute_stitch_metrics,
        export_debug_csv,
    )

    k = args.k
    s = args.s or int(round(math.sqrt(k)))
    H, W = 32, 32
    T_scan = 128
    device = torch.device("cpu")

    # Minimal config
    model_cfg = OmegaConf.create({
        "seed": 42,
        "kroot": {"branch": "short", "k": k, "s": s},
        "model": {
            "type": "kroot",
            "encoder": {"backbone": "efficientnet_b0", "pretrained": False},
            "transformer": {
                "d_model": 64, "n_heads": 2, "n_layers": 1,
                "dim_feedforward": 128, "window_size": k, "dropout": 0.0,
            },
            "pose_head": {"rotation_rep": "rot6d"},
        },
        "optimizer": {"lr_rec": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "loss": {"rot_weight": 1.0, "trans_weight": 1.0},
        "trainer": {"max_epochs": 1, "log_interval": 1, "validate_every": 99,
                     "grad_accum": 1, "max_grad_norm": 1.0},
        "train": {"max_steps": 3},
        "paths": {"output_dir": args.out_dir},
    })

    # Train short
    print("[smoke] Training short model...")
    ds_s = SyntheticShortWindowDataset(
        num_scans=2, frames_per_scan=T_scan,
        height=H, width=W, k=k, overlap=4, mode="train",
    )
    trainer_s = KRootTrainer(
        model_cfg, device="cpu",
        train_loader=DataLoader(ds_s, batch_size=1),
    )
    trainer_s.train()

    # Train long
    print("[smoke] Training long model...")
    cfg_long = model_cfg.copy()
    OmegaConf.update(cfg_long, "kroot.branch", "long")
    ds_l = SyntheticLongWindowDataset(
        num_scans=2, frames_per_scan=T_scan * 4,
        height=H, width=W, k=k, s=s, mode="train",
    )
    trainer_l = KRootTrainer(
        cfg_long, device="cpu",
        train_loader=DataLoader(ds_l, batch_size=1),
    )
    trainer_l.train()

    # Generate synthetic scans
    gen = torch.Generator().manual_seed(0)
    n_scans = args.max_scans or 2
    all_metrics = []

    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    for scan_idx in range(n_scans):
        frames = torch.rand(T_scan, H, W, generator=gen) * 255
        # Random walk GT
        eye = torch.eye(4)
        gt_list = [eye]
        for _ in range(T_scan - 1):
            angle = torch.randn(3, generator=gen) * 0.02
            t = torch.randn(3, generator=gen) * 0.5
            theta = angle.norm()
            if theta > 1e-8:
                kv = angle / theta
                Kmat = torch.tensor([[0, -kv[2], kv[1]], [kv[2], 0, -kv[0]], [-kv[1], kv[0], 0]])
                R = torch.eye(3) + torch.sin(theta) * Kmat + (1 - torch.cos(theta)) * (Kmat @ Kmat)
            else:
                R = torch.eye(3)
            step = torch.eye(4)
            step[:3, :3] = R
            step[:3, 3] = t
            gt_list.append(gt_list[-1] @ step)
        gt_global = torch.stack(gt_list, dim=0)

        print(f"[smoke] Stitching scan {scan_idx}...")
        result = stitch_long_base_short_refine(
            short_model=trainer_s.model,
            long_model=trainer_l.model,
            scan_frames=frames,
            k=k, s=s,
            device=device,
            short_overlap=4,
            enable_endpoint_interp=not args.no_endpoint_interp,
        )

        metrics = compute_stitch_metrics(
            fused_global=result["fused_global"],
            short_global=result["short_global"],
            long_global=result["long_global"],
            anchor_indices=result["anchor_indices"],
            gt_global=gt_global,
        )
        all_metrics.append(metrics)

        export_debug_csv(
            f"scan_{scan_idx}",
            result["fused_global"],
            result["short_global"],
            result["long_global"],
            result["anchor_indices"],
            gt_global,
            out_dir=viz_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("K-root Stitch Smoke Results")
    print("=" * 60)
    agg_keys = ["gpe_mm_short_only", "gpe_mm_long_only", "gpe_mm_fused",
                "drift_last_mm_short_only", "drift_last_mm_long_only", "drift_last_mm_fused",
                "lpe_mm_short_only", "lpe_mm_long_only", "lpe_mm_fused"]
    for key in agg_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            mean_v = sum(vals) / len(vals)
            print(f"  mean_{key:35s}: {mean_v:.6f}")

    # Save summary JSON
    summary = {}
    for key in agg_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            summary[f"mean_{key}"] = sum(vals) / len(vals)
    summary_path = os.path.join(args.out_dir, "kroot_stitch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Debug CSVs in {viz_dir}/")


def run_real(args):
    """Run stitch evaluation with real checkpoints."""
    import math
    from eval.kroot_stitch import (
        stitch_long_base_short_refine,
        compute_stitch_metrics,
        export_debug_csv,
    )

    cfg = OmegaConf.load(args.config)
    k = args.k or int(OmegaConf.select(cfg, "kroot.k") or 64)
    s = args.s or int(OmegaConf.select(cfg, "kroot.s") or int(round(math.sqrt(k))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build models
    short_model = _build_model(cfg, device)
    long_model = _build_model(cfg, device)

    # Load checkpoints — unified dual checkpoint or separate short/long
    if args.ckpt:
        # Unified checkpoint from KRootDualTrainer
        state = torch.load(args.ckpt, map_location=device)
        if "short_model" in state and "long_model" in state:
            short_model.load_state_dict(state["short_model"])
            long_model.load_state_dict(state["long_model"])
            print(f"[eval] Dual checkpoint loaded from {args.ckpt}")
        elif "model" in state and isinstance(state["model"], dict):
            # ModuleDict wrapper: model.short.* / model.long.*
            md = state["model"]
            short_state = {k.replace("short.", "", 1): v for k, v in md.items() if k.startswith("short.")}
            long_state = {k.replace("long.", "", 1): v for k, v in md.items() if k.startswith("long.")}
            if short_state and long_state:
                short_model.load_state_dict(short_state)
                long_model.load_state_dict(long_state)
                print(f"[eval] Dual checkpoint (ModuleDict) loaded from {args.ckpt}")
            else:
                warnings.warn(f"Checkpoint {args.ckpt} has 'model' key but no short/long keys.")
        else:
            warnings.warn(f"Unrecognised checkpoint format in {args.ckpt}; using random weights.")
    else:
        # Separate checkpoints
        if args.short_ckpt:
            state = torch.load(args.short_ckpt, map_location=device)
            model_state = state.get("model") or state.get("state_dict") or state
            short_model.load_state_dict(model_state)
            print(f"[eval] Short model loaded from {args.short_ckpt}")
        else:
            warnings.warn("No --short-ckpt or --ckpt provided; short model uses random weights.")

        if args.long_ckpt:
            state = torch.load(args.long_ckpt, map_location=device)
            model_state = state.get("model") or state.get("state_dict") or state
            long_model.load_state_dict(model_state)
            print(f"[eval] Long model loaded from {args.long_ckpt}")
        else:
            warnings.warn("No --long-ckpt or --ckpt provided; long model uses random weights.")

    # Build validation dataset → full scan loader
    from data.builder import build_dataset
    dset_val = build_dataset(cfg, split="val")
    from data.datasets.dual_kroot_window import ShortWindowDataset
    from torch.utils.data import DataLoader

    sw_val = ShortWindowDataset(
        base_dataset=dset_val, k=k, mode="val",
    )
    val_loader = DataLoader(sw_val, batch_size=1, num_workers=0)

    # Calibration
    tform_calib = None
    calib_file = OmegaConf.select(cfg, "dataset.calib_file")
    if calib_file:
        try:
            from trainers.utils.calibration import load_calibration
            resample_factor = float(OmegaConf.select(cfg, "dataset.resample_factor") or 1.0)
            _, _, tform_calib = load_calibration(calib_file, resample_factor, device=device)
            if not isinstance(tform_calib, torch.Tensor):
                import numpy as np
                tform_calib = torch.as_tensor(tform_calib, dtype=torch.float32)
            tform_calib = tform_calib.float().to(device)
        except Exception as exc:
            warnings.warn(f"Could not load calibration: {exc}")

    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    all_metrics = []
    scan_count = 0

    for batch in val_loader:
        if args.max_scans is not None and scan_count >= args.max_scans:
            break
        frames = batch["frames"].squeeze(0).to(device)
        gt_global = batch["gt_global_T"].squeeze(0).to(device)
        meta = batch.get("meta", {})
        scan_id = "unknown"
        if isinstance(meta, dict):
            sid = meta.get("scan_id")
            if isinstance(sid, (list, tuple)):
                scan_id = str(sid[0])
            elif sid is not None:
                scan_id = str(sid)

        result = stitch_long_base_short_refine(
            short_model, long_model, frames, k=k, s=s,
            device=device,
            short_overlap=int(OmegaConf.select(cfg, "stitch.short_overlap") or 8),
            long_window_stride=int(OmegaConf.select(cfg, "stitch.long_window_stride") or max(1, k - 1)),
            enable_endpoint_interp=not args.no_endpoint_interp,
        )

        metrics = compute_stitch_metrics(
            fused_global=result["fused_global"],
            short_global=result["short_global"],
            long_global=result["long_global"],
            anchor_indices=result["anchor_indices"],
            gt_global=gt_global,
            tform_calib=tform_calib,
            frames=frames,
        )
        all_metrics.append(metrics)

        export_debug_csv(
            scan_id,
            result["fused_global"],
            result["short_global"],
            result["long_global"],
            result["anchor_indices"],
            gt_global,
            out_dir=viz_dir,
        )
        scan_count += 1
        print(f"[eval] scan {scan_count}: {scan_id}  gpe_fused={metrics.get('gpe_mm_fused', 0):.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("K-root Stitch Evaluation Results")
    print("=" * 60)
    summary = {}
    agg_keys = [k for m in all_metrics for k in m if isinstance(m[k], float)]
    agg_keys = sorted(set(agg_keys))
    for key in agg_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            mean_v = sum(vals) / len(vals)
            summary[f"mean_{key}"] = mean_v
            print(f"  mean_{key:35s}: {mean_v:.6f}")

    summary_path = os.path.join(args.out_dir, "kroot_stitch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n Summary → {summary_path}")
    print(f" Debug CSVs → {viz_dir}/")


def main(argv=None):
    args = _parse_args(argv)
    if args.smoke:
        run_smoke(args)
    else:
        if args.config is None:
            print("ERROR: --config is required (or use --smoke for synthetic)")
            return 1
        run_real(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
