"""VQ Memory Scan-Window Dataset — adds scan-level context for VQ cache.

Wraps ``ScanWindowDataset`` to include the *full* scan frames alongside each
local window sample.  The model/trainer is responsible for anchor extraction so
the anchor stride is applied exactly once and stays consistent with padding
masks after batching.

For each window sample it yields:
    * ``scan_frames``     : full scan frames
    * ``scan_gt_global_T``: corresponding full scan GT transforms

For val/test mode the full scan is already yielded as a single sample,
so scan_frames == frames.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from data.datasets.scan_window import ScanWindowDataset
from data.datasets.TUS_rec_s3 import SliceInfo
from data.utils import s3_io


class VQMemoryScanWindowDataset(IterableDataset):
    """Extends ScanWindowDataset to include full scan context.

    Each sample dict includes the ScanWindowDataset keys plus:
        * ``scan_frames``      : (T_scan, H, W) full scan frames
        * ``scan_gt_global_T`` : (T_scan, 4, 4) full scan GT transforms
        * ``scan_total_frames``: int total frames in the scan

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
        Underlying S3 dataset.
    anchor_stride : int
    Stored for metadata / downstream consistency with the model config.
    window_size : int
        Size of local windows (same as ScanWindowDataset).
    Other params : same as ScanWindowDataset.
    """

    def __init__(
        self,
        base_dataset,
        *,
        anchor_stride: int = 8,
        window_size: int = 128,
        windows_per_scan: int = 4,
        sampling_mode: str = "tiled",
        tile_overlap: int = 0,
        min_tile_size: int = 0,
        max_tiles_per_scan: int = 0,
        mode: str = "train",
        seed: int = 0,
        augment_flip: bool = False,
        flip_prob: float = 0.5,
        calib_matrix: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self._base = base_dataset
        self.anchor_stride = anchor_stride
        self.window_size = max(2, int(window_size))
        self.windows_per_scan = max(1, int(windows_per_scan))
        self.sampling_mode = str(sampling_mode).lower()
        self.tile_overlap = max(0, int(tile_overlap))
        self.min_tile_size = int(min_tile_size) if min_tile_size > 0 else self.window_size // 2
        self.max_tiles_per_scan = max(0, int(max_tiles_per_scan))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0
        self.augment_flip = bool(augment_flip) and self.mode == "train"
        self.flip_prob = float(flip_prob)
        self.calib_matrix = calib_matrix
        self.image_size = image_size

        # Delegate window logic to ScanWindowDataset
        self._scan_window = ScanWindowDataset(
            base_dataset=base_dataset,
            window_size=window_size,
            windows_per_scan=windows_per_scan,
            sampling_mode=sampling_mode,
            tile_overlap=tile_overlap,
            min_tile_size=min_tile_size,
            max_tiles_per_scan=max_tiles_per_scan,
            mode=mode,
            seed=seed,
            augment_flip=augment_flip,
            flip_prob=flip_prob,
            calib_matrix=calib_matrix,
            image_size=image_size,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._scan_window.set_epoch(epoch)

    @staticmethod
    def _normalise_global_transforms(tforms: torch.Tensor) -> torch.Tensor:
        inv_first = torch.linalg.inv(tforms[0:1])
        return inv_first @ tforms

    def _iter_scan_with_context(
        self,
        info: SliceInfo,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        """Yield window samples augmented with full scan context."""
        T_total = frames_t.shape[0]
        if T_total < 2:
            return

        # Keep scan frames in their compact source dtype on CPU. They are
        # converted to float only inside the trainer/model path.
        scan_frames_full = frames_t.contiguous()                        # (T, H, W)
        scan_gt_full = self._normalise_global_transforms(tforms_t.float())  # (T, 4, 4)

        # Yield each window sample with scan context attached
        for sample in self._scan_window._iter_scan_windows(
            info, frames_t, tforms_t, rng,
        ):
            sample["scan_frames"] = scan_frames_full      # (T, H, W) — full scan
            sample["scan_gt_global_T"] = scan_gt_full     # (T, 4, 4) — full scan GT
            sample["scan_total_frames"] = T_total
            yield sample

    def __iter__(self):
        ddp_rank, ddp_world = self._scan_window._resolve_rank_info()
        worker_id, num_workers = self._scan_window._resolve_worker_info()
        global_rank = ddp_rank * num_workers + worker_id
        global_world = ddp_world * num_workers

        client = s3_io.create_client(
            region=self._base.region,
            endpoint=self._base.endpoint,
            force_path_style=self._base.force_path_style,
        )

        global_rng = random.Random(self.seed + self.epoch)
        slices = self._base._list_slices(client=client)
        if self._base.shuffle_slices:
            global_rng.shuffle(slices)
        slices = slices[global_rank::global_world]

        worker_seed = global_rng.randint(0, 2**32 - 1) + global_rank
        rng = random.Random(worker_seed)

        if not slices:
            return iter(())

        for info, frames_t, tforms_t in self._base._iter_loaded_slices(slices, client=client):
            for sample in self._iter_scan_with_context(info, frames_t, tforms_t, rng):
                yield sample
