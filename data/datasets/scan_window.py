"""Scan-window dataset wrapper for long-sequence pose models.

Wraps ``TUSRecS3Iterable`` to yield contiguous windows of *T* frames
(with ground-truth global transforms) instead of pairwise samples.

Training modes:

* **tiled** (default) — deterministic non-overlapping tiling.  Each frame
  appears exactly once per epoch.  An epoch-dependent random offset (jitter)
  shifts the tiling grid so tile boundaries vary across epochs.
* **random** (legacy) — ``windows_per_scan`` random windows per scan; frames
  in overlapping regions are processed multiple times.

Eval:  full scan (all frames) as a single sample (unchanged).
"""

from __future__ import annotations

import math
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
        (legacy *random* mode) Number of random windows from each scan.
    sampling_mode : str
        ``"tiled"`` (default) — non-overlapping tiles with epoch jitter;
        ``"random"`` — legacy random-window sampling.
    tile_overlap : int
        Overlap between consecutive tiles (default 0 — no overlap).  A small
        overlap (e.g. 8–16) gives the transformer context at tile boundaries.
    min_tile_size : int
        Minimum acceptable tile length.  Remainder tiles shorter than this
        are merged into the preceding tile.  Default ``window_size // 2``.
    max_tiles_per_scan : int
        Maximum number of tiles per scan (0 = unlimited).  When a scan
        produces more tiles than this, tiles are evenly subsampled.  Useful
        for capping training cost on very long scans.
    mode : str
        "train" → tiled/random windows; "val"/"test" → full scan.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        base_dataset: TUSRecS3Iterable,
        window_size: int = 128,
        windows_per_scan: int = 1,
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
        self.window_size = max(2, int(window_size))
        self.windows_per_scan = max(1, int(windows_per_scan))
        self.sampling_mode = str(sampling_mode).lower()  # "tiled" or "random"
        self.tile_overlap = max(0, int(tile_overlap))
        self.min_tile_size = int(min_tile_size) if min_tile_size > 0 else self.window_size // 2
        self.max_tiles_per_scan = max(0, int(max_tiles_per_scan))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0
        # Time-reversal augmentation: randomly reverse frame order during training.
        # The local transforms in the reversed sequence at position j equal
        # inv(original local transform at position T-j), which is physically
        # equivalent to running the probe in the reverse direction.
        # After reversing, we normalise so that the new frame-0 = I, and the
        # normalisation step correctly handles the updated global transforms.
        self.augment_flip = bool(augment_flip) and self.mode == "train"
        self.flip_prob = float(flip_prob)
        # Optional calibration matrix (4x4) and image size for DDF metrics/loss.
        # When provided these are included in every sample's ``meta`` dict so
        # the trainer can compute TUS-REC DDF metrics without re-loading calib.
        self.calib_matrix = calib_matrix  # (4, 4) tensor or None
        self.image_size = image_size      # (H, W) tuple or None

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

    # ------------------------------------------------------------------
    # Tiling helpers
    # ------------------------------------------------------------------
    def _compute_tile_starts(
        self, T_total: int, rng: random.Random,
    ) -> List[Tuple[int, int]]:
        """Return a list of ``(start, end)`` for non-overlapping tiles.

        * Tiles are ``window_size`` long with ``stride = window_size - tile_overlap``.
        * An epoch-dependent random jitter shifts the grid so that tile
          boundaries move across epochs (prevents fixed boundary artefacts).
        * If the final tile is shorter than ``min_tile_size`` it is absorbed
          into the previous tile (that tile grows slightly beyond ``window_size``).
        """
        W = min(self.window_size, T_total)
        stride = max(1, W - self.tile_overlap)

        # ── Epoch jitter: shift grid start by a random value in [0, stride) ──
        max_jitter = min(stride, T_total - W) if T_total > W else 0
        jitter = rng.randint(0, max_jitter) if max_jitter > 0 else 0

        tiles: List[Tuple[int, int]] = []
        pos = jitter
        while pos + W <= T_total:
            tiles.append((pos, pos + W))
            pos += stride

        # Handle any remainder beyond the last full tile
        if pos < T_total:
            remainder = T_total - pos
            if remainder >= self.min_tile_size or not tiles:
                # Remainder is large enough → keep as its own tile
                tiles.append((pos, T_total))
            else:
                # Merge remainder into the previous tile (extend it)
                prev_start, _ = tiles[-1]
                tiles[-1] = (prev_start, T_total)

        # If jitter left frames before the first tile, extend tile 0 leftward
        if tiles and tiles[0][0] > 0:
            first_start, first_end = tiles[0]
            uncovered = first_start
            if uncovered < self.min_tile_size:
                # Extend tile 0 to start from 0
                tiles[0] = (0, first_end)
            else:
                # Prepend a tile for the left margin
                tiles.insert(0, (0, first_start + self.tile_overlap))

        return tiles

    # ------------------------------------------------------------------
    # Per-window sample builder (shared between tiled and random paths)
    # ------------------------------------------------------------------
    def _make_sample(
        self,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        start: int,
        end: int,
        scan_id: str,
        info: SliceInfo,
        T_total: int,
        rng: random.Random,
    ) -> dict:
        """Extract and normalise a single window → sample dict."""
        # Keep the original on-disk dtype (typically uint8) so CPU-side
        # dataloader / pinned-memory usage stays low. The trainer normalises
        # to float right before the model forward pass.
        win_frames = frames_t[start:end].clone()
        win_tforms = tforms_t[start:end].float().clone()

        # Time-reversal augmentation
        if self.augment_flip and rng.random() < self.flip_prob:
            win_frames = win_frames.flip(0).contiguous()
            win_tforms = win_tforms.flip(0).contiguous()

        gt_global = self._normalise_global_transforms(win_tforms)
        meta: dict = {
            "scan_id": scan_id,
            "subject": info.subject,
            "scan_name": info.scan_name,
            "window_start": start,
            "window_size": end - start,
            "total_frames": T_total,
        }
        if self.calib_matrix is not None:
            meta["tform_calib"] = self.calib_matrix
        if self.image_size is not None:
            meta["image_size"] = list(self.image_size)
        return {
            "frames": win_frames,
            "gt_global_T": gt_global,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------
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
            meta: dict = {
                "scan_id": scan_id,
                "subject": info.subject,
                "scan_name": info.scan_name,
                "window_start": 0,
                "window_size": T_total,
                "total_frames": T_total,
            }
            if self.calib_matrix is not None:
                meta["tform_calib"] = self.calib_matrix
            if self.image_size is not None:
                meta["image_size"] = list(self.image_size)
            yield {
                # Same rationale as training mode: defer float conversion to the
                # trainer to avoid materialising full-scan float32 tensors in CPU RAM.
                "frames": frames_t,
                "gt_global_T": gt_global,
                "meta": meta,
            }
            return

        # ── Training: tiled (default) or random ─────────────────────────
        if self.sampling_mode == "random":
            # Legacy random-window sampling
            W = min(self.window_size, T_total)
            max_start = T_total - W
            for _ in range(self.windows_per_scan):
                start = rng.randint(0, max_start) if max_start > 0 else 0
                yield self._make_sample(
                    frames_t, tforms_t, start, start + W,
                    scan_id, info, T_total, rng,
                )
        else:
            # Tiled sampling — each frame seen (approximately) once per epoch
            tiles = self._compute_tile_starts(T_total, rng)
            # Cap number of tiles for very long scans
            if self.max_tiles_per_scan > 0 and len(tiles) > self.max_tiles_per_scan:
                # Evenly subsample tiles to keep spatial diversity
                step = len(tiles) / self.max_tiles_per_scan
                selected = [tiles[int(i * step)] for i in range(self.max_tiles_per_scan)]
                tiles = selected
            # Shuffle tile order within this scan so that consecutive batches
            # come from different parts of the scan.
            rng.shuffle(tiles)
            for start, end in tiles:
                yield self._make_sample(
                    frames_t, tforms_t, start, end,
                    scan_id, info, T_total, rng,
                )

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
