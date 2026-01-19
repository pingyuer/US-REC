from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset

from data.builder import build_pipeline


class SegmentationDataset(Dataset):
    """
    Simple paired image/mask dataset driven by directories.

    This is primarily for local folder datasets where image and mask filenames
    match (e.g. img_a.png in both directories).
    """

    def __init__(
        self,
        *,
        img_dir: str,
        mask_dir: Optional[str] = None,
        pipeline_cfg: Optional[Any] = None,
        split: str = "train",
    ):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.split = split

        if pipeline_cfg is not None:
            self.pipeline = build_pipeline(pipeline_cfg, split=split)
        else:
            self.pipeline = None

        self._items = self._scan()

    def _scan(self):
        if not self.img_dir.exists():
            raise FileNotFoundError(f"img_dir not found: {self.img_dir}")

        img_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.is_file() and not p.name.startswith(".")],
            key=lambda p: p.name,
        )

        items = []
        for img_path in img_paths:
            mask_path = None
            if self.mask_dir is not None:
                candidate = self.mask_dir / img_path.name
                if candidate.exists():
                    mask_path = candidate
            items.append((img_path, mask_path))
        return items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path, mask_path = self._items[idx]
        sample = {
            "img_path": str(img_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
        }

        if self.pipeline is not None:
            sample = self.pipeline(sample)
        return sample

