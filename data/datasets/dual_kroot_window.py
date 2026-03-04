"""Dual K-root window datasets for Short / Long independent training.

Provides two ``IterableDataset`` implementations that wrap the base
``TUSRecS3Iterable`` and yield fixed-length (exactly *k* tokens) windows:

* **ShortWindowDataset** — contiguous frames ``[t, t+1, ..., t+k-1]``.
  Used to train the short-range dense transformer.

* **LongWindowDataset** — sparse frames at stride *s*:
  ``[t, t+s, t+2s, ..., t+(k-1)*s]``.  Covers a real span of ``k*s``
  frames, used to train the long-range anchor transformer.

Both datasets guarantee **exactly k tokens per sample** (no remainder
merging, no variable-length tiles).  Windows that would be shorter than
*k* are discarded.

Convention
----------
All GT transforms are **global** ``T_{0<-i}`` normalised so that the
first token in each window has ``T = I`` (identity).
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Tuple

import torch

from data.utils import s3_io
from data.datasets.TUS_rec_s3 import TUSRecS3Iterable, SliceInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_global_transforms(tforms: torch.Tensor) -> torch.Tensor:
    """Re-root global transforms so the first frame = I."""
    inv_first = torch.linalg.inv(tforms[0:1])  # (1, 4, 4)
    return inv_first @ tforms                   # (T, 4, 4)


def _make_meta(
    scan_id: str,
    info: SliceInfo,
    total_frames: int,
    window_start: int,
    stride: int,
    k: int,
    calib_matrix: Optional[torch.Tensor] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> dict:
    meta: dict = {
        "scan_id": scan_id,
        "subject": info.subject,
        "scan_name": info.scan_name,
        "window_start": window_start,
        "window_size": k,
        "stride": stride,
        "total_frames": total_frames,
    }
    if calib_matrix is not None:
        meta["tform_calib"] = calib_matrix
    if image_size is not None:
        meta["image_size"] = list(image_size)
    return meta


# ===========================================================================
# ShortWindowDataset — contiguous windows of length k
# ===========================================================================

class ShortWindowDataset(torch.utils.data.IterableDataset):
    """Yields contiguous windows of exactly *k* frames from each scan.

    Tiled sampling with configurable overlap.  Windows that would be
    shorter than *k* are **discarded** (never merged).

    Each sample is a dict::

        {
            "frames":        (k, H, W)   float32 [0, 255]
            "gt_global_T":   (k, 4, 4)   float32 — frame 0 = I
            "meta": { scan_id, subject, scan_name, window_start,
                      window_size, stride(=1), total_frames }
        }

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
    k : int
        Token count per sample (window length).
    overlap : int
        Overlap between consecutive tiles (default 8).
    mode : str
        ``"train"`` → tiled with jitter + shuffle;
        ``"val"`` / ``"test"`` → full scan *or* tiled deterministic.
    seed : int
    augment_flip : bool
        Time-reversal augmentation (train only).
    flip_prob : float
    calib_matrix : (4,4) Tensor or None
    image_size : (H, W) or None
    """

    def __init__(
        self,
        base_dataset: TUSRecS3Iterable,
        k: int = 64,
        overlap: int = 8,
        mode: str = "train",
        seed: int = 0,
        augment_flip: bool = False,
        flip_prob: float = 0.5,
        calib_matrix: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self._base = base_dataset
        self.k = max(2, int(k))
        self.overlap = max(0, min(int(overlap), self.k - 1))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0
        self.augment_flip = bool(augment_flip) and self.mode == "train"
        self.flip_prob = float(flip_prob)
        self.calib_matrix = calib_matrix
        self.image_size = image_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_worker_info(self) -> Tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    # ------------------------------------------------------------------
    # Tile computation — strict k, discard remainder
    # ------------------------------------------------------------------
    def _compute_tiles(self, T_total: int, rng: random.Random) -> List[Tuple[int, int]]:
        """Return (start, end) pairs of length exactly k; discard remainder."""
        W = self.k
        if T_total < W:
            return []  # scan too short — skip entirely

        stride = max(1, W - self.overlap)
        # Epoch jitter (train only)
        max_jitter = min(stride, T_total - W) if (T_total > W and self.mode == "train") else 0
        jitter = rng.randint(0, max_jitter) if max_jitter > 0 else 0

        tiles: List[Tuple[int, int]] = []
        pos = jitter
        while pos + W <= T_total:
            tiles.append((pos, pos + W))
            pos += stride
        # Do NOT merge remainder — strictly discard any leftover < k
        return tiles

    # ------------------------------------------------------------------
    # Per-window sample
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
        win_frames = frames_t[start:end].float().clone()
        win_tforms = tforms_t[start:end].float().clone()

        # Time-reversal augmentation
        if self.augment_flip and rng.random() < self.flip_prob:
            win_frames = win_frames.flip(0).contiguous()
            win_tforms = win_tforms.flip(0).contiguous()

        gt_global = _normalise_global_transforms(win_tforms)
        meta = _make_meta(
            scan_id, info, T_total,
            window_start=start, stride=1, k=end - start,
            calib_matrix=self.calib_matrix, image_size=self.image_size,
        )
        return {"frames": win_frames, "gt_global_T": gt_global, "meta": meta}

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def _iter_scan(
        self,
        info: SliceInfo,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        T_total = frames_t.shape[0]
        scan_id = f"{info.subject}/{info.scan_name}"

        if self.mode in ("val", "test"):
            # Eval: yield the full scan as a single sample (variable length)
            gt_global = _normalise_global_transforms(tforms_t.float())
            meta = _make_meta(
                scan_id, info, T_total,
                window_start=0, stride=1, k=T_total,
                calib_matrix=self.calib_matrix, image_size=self.image_size,
            )
            yield {"frames": frames_t.float(), "gt_global_T": gt_global, "meta": meta}
            return

        tiles = self._compute_tiles(T_total, rng)
        if self.mode == "train":
            rng.shuffle(tiles)
        for start, end in tiles:
            yield self._make_sample(frames_t, tforms_t, start, end, scan_id, info, T_total, rng)

    def __iter__(self):
        worker_id, num_workers = self._resolve_worker_info()
        client = s3_io.create_client(
            region=self._base.region,
            endpoint=self._base.endpoint,
            force_path_style=self._base.force_path_style,
        )
        global_rng = random.Random(self.seed + self.epoch)
        slices = self._base._list_slices(client=client)
        if self._base.shuffle_slices:
            global_rng.shuffle(slices)
        slices = slices[worker_id::num_workers]
        rng = random.Random(global_rng.randint(0, 2**32 - 1) + worker_id)

        for info, frames_t, tforms_t in self._base._iter_loaded_slices(slices, client=client):
            for sample in self._iter_scan(info, frames_t, tforms_t, rng):
                yield sample


# ===========================================================================
# LongWindowDataset — sparse windows of exactly k tokens at stride s
# ===========================================================================

class LongWindowDataset(torch.utils.data.IterableDataset):
    """Yields sparse windows of exactly *k* tokens at stride *s* from each scan.

    Token frame indices: ``[t, t+s, t+2s, ..., t+(k-1)*s]``
    covering a real span of ``k*s`` frames.

    Each sample is a dict::

        {
            "frames":        (k, H, W)   float32 [0, 255]
            "gt_global_T":   (k, 4, 4)   float32 — token 0 = I
            "idx_long":      (k,)         int64 — original frame indices
            "meta": { scan_id, subject, scan_name, window_start,
                      window_size(=k), stride(=s), total_frames }
        }

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
    k : int
        Token count per sample.
    s : int or None
        Sparse stride.  If None, computed as ``int(round(sqrt(k)))``.
    overlap_tokens : int
        Overlap in *token space* between consecutive tiled windows (default 0).
    mode : str
    seed : int
    augment_flip : bool
    flip_prob : float
    calib_matrix : (4,4) Tensor or None
    image_size : (H, W) or None
    """

    def __init__(
        self,
        base_dataset: TUSRecS3Iterable,
        k: int = 64,
        s: Optional[int] = None,
        overlap_tokens: int = 0,
        mode: str = "train",
        seed: int = 0,
        augment_flip: bool = False,
        flip_prob: float = 0.5,
        calib_matrix: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self._base = base_dataset
        self.k = max(2, int(k))
        self.s = int(s) if s is not None else int(round(math.sqrt(self.k)))
        self.overlap_tokens = max(0, min(int(overlap_tokens), self.k - 1))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0
        self.augment_flip = bool(augment_flip) and self.mode == "train"
        self.flip_prob = float(flip_prob)
        self.calib_matrix = calib_matrix
        self.image_size = image_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_worker_info(self) -> Tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    # ------------------------------------------------------------------
    # Sparse tile computation
    # ------------------------------------------------------------------
    def _compute_sparse_tiles(self, T_total: int, rng: random.Random) -> List[List[int]]:
        """Return list of index-lists; each has exactly k indices.

        A tile starting at origin ``t`` covers frames
        ``[t, t+s, t+2s, ..., t+(k-1)*s]``.  The last index must be
        ``< T_total``; windows that would exceed are discarded.
        """
        span = (self.k - 1) * self.s  # real-frame span of one window (excluding origin)
        if T_total < span + 1:
            return []  # scan is too short

        # Token-level stride for tiled sampling
        token_stride = max(1, self.k - self.overlap_tokens)
        # Convert token stride to frame stride
        frame_stride = token_stride * self.s

        # Epoch jitter (train only)
        max_jitter = min(frame_stride, T_total - span - 1) if self.mode == "train" else 0
        jitter = rng.randint(0, max(0, max_jitter)) if max_jitter > 0 else 0

        tiles: List[List[int]] = []
        origin = jitter
        while origin + span < T_total:
            idx = [origin + m * self.s for m in range(self.k)]
            tiles.append(idx)
            origin += frame_stride

        # No remainder merging — strictly discard
        return tiles

    # ------------------------------------------------------------------
    # Per-window sample
    # ------------------------------------------------------------------
    def _make_sample(
        self,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        idx: List[int],
        scan_id: str,
        info: SliceInfo,
        T_total: int,
        rng: random.Random,
    ) -> dict:
        idx_tensor = torch.tensor(idx, dtype=torch.long)
        win_frames = frames_t[idx_tensor].float().clone()     # (k, H, W)
        win_tforms = tforms_t[idx_tensor].float().clone()     # (k, 4, 4)

        # Time-reversal augmentation
        if self.augment_flip and rng.random() < self.flip_prob:
            win_frames = win_frames.flip(0).contiguous()
            win_tforms = win_tforms.flip(0).contiguous()
            idx_tensor = idx_tensor.flip(0).contiguous()

        gt_global = _normalise_global_transforms(win_tforms)
        meta = _make_meta(
            scan_id, info, T_total,
            window_start=idx[0], stride=self.s, k=len(idx),
            calib_matrix=self.calib_matrix, image_size=self.image_size,
        )
        return {
            "frames": win_frames,
            "gt_global_T": gt_global,
            "idx_long": idx_tensor,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def _iter_scan(
        self,
        info: SliceInfo,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        T_total = frames_t.shape[0]
        scan_id = f"{info.subject}/{info.scan_name}"

        if self.mode in ("val", "test"):
            # Eval: yield full scan with all frames + anchor index list
            gt_global = _normalise_global_transforms(tforms_t.float())
            # Also provide the sparse anchor indices for eval stitching
            anchor_idx = list(range(0, T_total, self.s))
            idx_tensor = torch.tensor(anchor_idx, dtype=torch.long)
            meta = _make_meta(
                scan_id, info, T_total,
                window_start=0, stride=self.s, k=T_total,
                calib_matrix=self.calib_matrix, image_size=self.image_size,
            )
            yield {
                "frames": frames_t.float(),
                "gt_global_T": gt_global,
                "idx_long": idx_tensor,
                "meta": meta,
            }
            return

        tiles = self._compute_sparse_tiles(T_total, rng)
        if self.mode == "train":
            rng.shuffle(tiles)
        for idx in tiles:
            yield self._make_sample(frames_t, tforms_t, idx, scan_id, info, T_total, rng)

    def __iter__(self):
        worker_id, num_workers = self._resolve_worker_info()
        client = s3_io.create_client(
            region=self._base.region,
            endpoint=self._base.endpoint,
            force_path_style=self._base.force_path_style,
        )
        global_rng = random.Random(self.seed + self.epoch)
        slices = self._base._list_slices(client=client)
        if self._base.shuffle_slices:
            global_rng.shuffle(slices)
        slices = slices[worker_id::num_workers]
        rng = random.Random(global_rng.randint(0, 2**32 - 1) + worker_id)

        for info, frames_t, tforms_t in self._base._iter_loaded_slices(slices, client=client):
            for sample in self._iter_scan(info, frames_t, tforms_t, rng):
                yield sample


# ===========================================================================
# Synthetic versions for smoke testing (no S3)
# ===========================================================================

class SyntheticShortWindowDataset(torch.utils.data.IterableDataset):
    """Synthetic short-window dataset for smoke testing."""

    def __init__(
        self,
        num_scans: int = 4,
        frames_per_scan: int = 256,
        height: int = 128,
        width: int = 128,
        k: int = 64,
        overlap: int = 8,
        mode: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.num_scans = num_scans
        self.frames_per_scan = frames_per_scan
        self.height = height
        self.width = width
        self.k = max(2, int(k))
        self.overlap = max(0, int(overlap))
        self.mode = str(mode).lower()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _random_walk_transforms(T: int, *, generator: torch.Generator) -> torch.Tensor:
        eye = torch.eye(4)
        out = [eye]
        for _ in range(T - 1):
            angle = torch.randn(3, generator=generator) * 0.02
            t = torch.randn(3, generator=generator) * 0.5
            theta = angle.norm()
            if theta > 1e-8:
                k = angle / theta
                K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
            else:
                R = torch.eye(3)
            step = torch.eye(4)
            step[:3, :3] = R
            step[:3, 3] = t
            out.append(out[-1] @ step)
        return torch.stack(out, dim=0)

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        rng = random.Random(self.seed + self.epoch)
        for scan_idx in range(self.num_scans):
            T = self.frames_per_scan
            frames = torch.rand(T, self.height, self.width, generator=gen) * 255
            gt_global = self._random_walk_transforms(T, generator=gen)

            if self.mode in ("val", "test"):
                inv0 = torch.linalg.inv(gt_global[0:1])
                yield {
                    "frames": frames,
                    "gt_global_T": inv0 @ gt_global,
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": 0,
                        "window_size": T,
                        "stride": 1,
                        "total_frames": T,
                    },
                }
            else:
                # Tiled sampling — discard remainder
                W = self.k
                stride = max(1, W - self.overlap)
                tiles = []
                pos = 0
                while pos + W <= T:
                    tiles.append((pos, pos + W))
                    pos += stride
                rng.shuffle(tiles)
                for start, end in tiles:
                    win_frames = frames[start:end]
                    win_global = gt_global[start:end].clone()
                    inv0 = torch.linalg.inv(win_global[0:1])
                    yield {
                        "frames": win_frames,
                        "gt_global_T": inv0 @ win_global,
                        "meta": {
                            "scan_id": f"synth_{scan_idx}",
                            "subject": "synth",
                            "scan_name": f"scan_{scan_idx}",
                            "window_start": start,
                            "window_size": W,
                            "stride": 1,
                            "total_frames": T,
                        },
                    }


class SyntheticLongWindowDataset(torch.utils.data.IterableDataset):
    """Synthetic long-window (sparse) dataset for smoke testing."""

    def __init__(
        self,
        num_scans: int = 4,
        frames_per_scan: int = 512,
        height: int = 128,
        width: int = 128,
        k: int = 64,
        s: Optional[int] = None,
        overlap_tokens: int = 0,
        mode: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.num_scans = num_scans
        self.frames_per_scan = frames_per_scan
        self.height = height
        self.width = width
        self.k = max(2, int(k))
        self.s = int(s) if s is not None else int(round(math.sqrt(self.k)))
        self.overlap_tokens = max(0, int(overlap_tokens))
        self.mode = str(mode).lower()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _random_walk_transforms(T: int, *, generator: torch.Generator) -> torch.Tensor:
        eye = torch.eye(4)
        out = [eye]
        for _ in range(T - 1):
            angle = torch.randn(3, generator=generator) * 0.02
            t = torch.randn(3, generator=generator) * 0.5
            theta = angle.norm()
            if theta > 1e-8:
                k = angle / theta
                K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
            else:
                R = torch.eye(3)
            step = torch.eye(4)
            step[:3, :3] = R
            step[:3, 3] = t
            out.append(out[-1] @ step)
        return torch.stack(out, dim=0)

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        rng = random.Random(self.seed + self.epoch)
        for scan_idx in range(self.num_scans):
            T = self.frames_per_scan
            frames = torch.rand(T, self.height, self.width, generator=gen) * 255
            gt_global = self._random_walk_transforms(T, generator=gen)

            if self.mode in ("val", "test"):
                inv0 = torch.linalg.inv(gt_global[0:1])
                anchor_idx = list(range(0, T, self.s))
                yield {
                    "frames": frames,
                    "gt_global_T": inv0 @ gt_global,
                    "idx_long": torch.tensor(anchor_idx, dtype=torch.long),
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": 0,
                        "window_size": T,
                        "stride": self.s,
                        "total_frames": T,
                    },
                }
            else:
                span = (self.k - 1) * self.s
                if T < span + 1:
                    continue
                token_stride = max(1, self.k - self.overlap_tokens)
                frame_stride = token_stride * self.s
                tiles = []
                origin = 0
                while origin + span < T:
                    idx = [origin + m * self.s for m in range(self.k)]
                    tiles.append(idx)
                    origin += frame_stride
                rng.shuffle(tiles)
                for idx in tiles:
                    idx_tensor = torch.tensor(idx, dtype=torch.long)
                    win_frames = frames[idx_tensor]
                    win_tforms = gt_global[idx_tensor].clone()
                    inv0 = torch.linalg.inv(win_tforms[0:1])
                    yield {
                        "frames": win_frames,
                        "gt_global_T": inv0 @ win_tforms,
                        "idx_long": idx_tensor,
                        "meta": {
                            "scan_id": f"synth_{scan_idx}",
                            "subject": "synth",
                            "scan_name": f"scan_{scan_idx}",
                            "window_start": idx[0],
                            "window_size": self.k,
                            "stride": self.s,
                            "total_frames": T,
                        },
                    }
