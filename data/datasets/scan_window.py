"""Scan-window dataset wrapper for long-sequence pose models.

Wraps ``TUSRecS3Iterable`` to yield contiguous windows of *T* frames
(with ground-truth global transforms) instead of pairwise samples.

Training: random windows of ``window_size`` frames from each scan.
Eval:     full scan (all frames) as a single sample.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Tuple

import torch

from data.utils import h5_io, s3_io
from data.datasets.TUS_rec_s3 import TUSRecS3Iterable, SliceInfo


class ScanWindowDataset(torch.utils.data.IterableDataset):
    """Yields contiguous windows of *T* frames from a scan.

    Each sample is a dict::

        {
            "frames":    (T, H, W)     float32     [0, 255]
            "gt_global_T": (T, 4, 4)   float32     global transforms (frame 0 = I)
            "meta": {
                "scan_id": str,
                "subject": str,
                "scan_name": str,
                "window_start": int,
                "window_size": int,
                "total_frames": int,
            }
        }

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
        The underlying S3 dataset (used only for its slice listing / loading).
    window_size : int
        Number of consecutive frames per sample (training).  Eval uses full scan.
    windows_per_scan : int
        Number of random windows to extract from each scan during training.
    mode : str
        "train" → random windows; "val"/"test" → full scan.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        base_dataset: TUSRecS3Iterable,
        window_size: int = 128,
        windows_per_scan: int = 1,
        mode: str = "train",
        seed: int = 0,
    ):
        super().__init__()
        self._base = base_dataset
        self.window_size = max(2, int(window_size))
        self.windows_per_scan = max(1, int(windows_per_scan))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_rank_info(self) -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _resolve_worker_info(self) -> Tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    @staticmethod
    def _normalise_global_transforms(tforms: torch.Tensor) -> torch.Tensor:
        """Make transforms relative to the first frame in the window.

        Given tforms (T, 4, 4) where each is T_{world<-i}, produce
        T_{0<-i} = inv(T_{world<-0}) @ T_{world<-i} so that frame 0 = I.
        """
        inv_first = torch.linalg.inv(tforms[0:1])  # (1, 4, 4)
        return inv_first @ tforms  # (T, 4, 4)

    def _iter_scan_windows(
        self,
        info: SliceInfo,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        """Yield dict samples from a single loaded scan."""
        T_total = frames_t.shape[0]
        if T_total < 2:
            return

        scan_id = f"{info.subject}/{info.scan_name}"

        if self.mode in ("val", "test"):
            # Full scan — normalise global transforms so frame 0 = I
            gt_global = self._normalise_global_transforms(tforms_t.float())
            yield {
                "frames": frames_t.float(),
                "gt_global_T": gt_global,
                "meta": {
                    "scan_id": scan_id,
                    "subject": info.subject,
                    "scan_name": info.scan_name,
                    "window_start": 0,
                    "window_size": T_total,
                    "total_frames": T_total,
                },
            }
            return

        # Training: random windows
        W = min(self.window_size, T_total)
        max_start = T_total - W
        for _ in range(self.windows_per_scan):
            start = rng.randint(0, max_start) if max_start > 0 else 0
            end = start + W
            win_frames = frames_t[start:end].float().clone()
            win_tforms = tforms_t[start:end].float().clone()
            gt_global = self._normalise_global_transforms(win_tforms)
            yield {
                "frames": win_frames,
                "gt_global_T": gt_global,
                "meta": {
                    "scan_id": scan_id,
                    "subject": info.subject,
                    "scan_name": info.scan_name,
                    "window_start": start,
                    "window_size": W,
                    "total_frames": T_total,
                },
            }

    def __iter__(self):
        ddp_rank, ddp_world = self._resolve_rank_info()
        worker_id, num_workers = self._resolve_worker_info()
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
            for sample in self._iter_scan_windows(info, frames_t, tforms_t, rng):
                yield sample


class SyntheticScanWindowDataset(torch.utils.data.IterableDataset):
    """Synthetic scan-window dataset for smoke testing (no S3).

    Yields random frames + random-walk SE(3) transforms.

    Parameters
    ----------
    num_scans : int
        Number of synthetic scans.
    frames_per_scan : int
        Frames per scan.
    height, width : int
        Frame dimensions.
    window_size : int
        Window size (training).
    mode : str
        "train" or "val".
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        num_scans: int = 4,
        frames_per_scan: int = 64,
        height: int = 128,
        width: int = 128,
        window_size: int = 32,
        mode: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.num_scans = num_scans
        self.frames_per_scan = frames_per_scan
        self.height = height
        self.width = width
        self.window_size = window_size
        self.mode = mode
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _random_walk_transforms(T: int, *, generator: torch.Generator) -> torch.Tensor:
        """Generate T global transforms via small random-walk SE(3)."""
        eye = torch.eye(4)
        globals_list = [eye]
        for _ in range(T - 1):
            # Small rotation perturbation
            angle = torch.randn(3, generator=generator) * 0.02  # ~1 degree
            t = torch.randn(3, generator=generator) * 0.5       # mm
            # Rodrigues for small angles
            theta = angle.norm()
            if theta > 1e-8:
                k = angle / theta
                K = torch.tensor([
                    [0, -k[2], k[1]],
                    [k[2], 0, -k[0]],
                    [-k[1], k[0], 0],
                ])
                R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
            else:
                R = torch.eye(3)
            step = torch.eye(4)
            step[:3, :3] = R
            step[:3, 3] = t
            globals_list.append(globals_list[-1] @ step)
        return torch.stack(globals_list, dim=0)  # (T, 4, 4)

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        rng = random.Random(self.seed + self.epoch)

        for scan_idx in range(self.num_scans):
            T = self.frames_per_scan
            frames = torch.rand(T, self.height, self.width, generator=gen) * 255
            gt_global = self._random_walk_transforms(T, generator=gen)

            if self.mode in ("val", "test"):
                yield {
                    "frames": frames,
                    "gt_global_T": gt_global,
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": 0,
                        "window_size": T,
                        "total_frames": T,
                    },
                }
            else:
                W = min(self.window_size, T)
                max_start = T - W
                start = rng.randint(0, max_start) if max_start > 0 else 0
                win_frames = frames[start : start + W]
                win_global = gt_global[start : start + W].clone()
                # Re-normalise to frame 0 = I
                inv0 = torch.linalg.inv(win_global[0:1])
                win_global = inv0 @ win_global
                yield {
                    "frames": win_frames,
                    "gt_global_T": win_global,
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": start,
                        "window_size": W,
                        "total_frames": T,
                    },
                }
