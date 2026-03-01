import os
import sys
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from data.builder import build_dataset


def _env_loaded() -> None:
    missing = [
        name
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "TUS_REC24_BUCKET",
            "TUS_REC24_REGION",
            "TUS_REC24_ENDPOINT",
        )
        if not os.environ.get(name)
    ]
    if missing:
        print("[warn] missing env:", ", ".join(missing))


def main() -> int:
    if "--config" not in sys.argv:
        print("Usage: python test.py --config <yaml>")
        return 1
    config_idx = sys.argv.index("--config") + 1
    config_path = sys.argv[config_idx]
    cfg = OmegaConf.load(config_path)

    _env_loaded()

    dset = build_dataset(cfg, split="train")
    print("dataset type:", type(dset))
    print("dataset len:", len(dset))

    sample = dset[0]
    if isinstance(sample, (list, tuple)):
        frames, tforms, tforms_inv = sample[:3]
    else:
        raise TypeError(f"Unexpected sample type: {type(sample)}")

    print("frames shape:", getattr(frames, "shape", None), "dtype:", getattr(frames, "dtype", None))
    print("tforms shape:", getattr(tforms, "shape", None), "dtype:", getattr(tforms, "dtype", None))
    print("tforms_inv shape:", getattr(tforms_inv, "shape", None), "dtype:", getattr(tforms_inv, "dtype", None))

    frames_np = np.asarray(frames)
    print("frames min/max:", float(frames_np.min()), float(frames_np.max()))

    # Basic sanity: tforms * tforms_inv ~= I
    tforms_t = torch.tensor(np.asarray(tforms))
    tforms_inv_t = torch.tensor(np.asarray(tforms_inv))
    eye = torch.eye(4).unsqueeze(0)
    recon = torch.matmul(tforms_t, tforms_inv_t)
    err = (recon - eye).abs().max().item()
    print("max |tform * inv - I|:", err)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
