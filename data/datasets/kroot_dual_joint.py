"""Joint K-root dual dataset — one DataLoader yields both short & long windows.

Each iteration reads a scan **once**, then from the *same* start offset ``t``
constructs:

* **short window**: contiguous ``[t, t+1, …, t+k-1]``
* **long  window**: sparse    ``[t, t+s, …, t+(k-1)*s]``   (s = ⌊√k⌋)

Guarantees
----------
1. Short & long share the *same* scan and the *same* origin ``t`` — no
   granularity mismatch between the two branches.
2. Only **one** S3 read + decode per scan (``_iter_loaded_slices``).
3. Tensors are returned as **uint8** (frames) and **float32 views** (tforms)
   — no ``.float().clone()`` on CPU.  Dtype promotion happens on GPU.
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
# JointKRootDualDataset
# ===========================================================================

class JointKRootDualDataset(torch.utils.data.IterableDataset):
    """Single-loader dataset that yields ``{"short": …, "long": …, "meta": …}``.

    For each scan the loader reads frames + tforms **once**, then tiles both
    short (contiguous) and long (sparse) windows from a **shared** origin.

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
        Underlying S3-backed iterable that provides raw scans.
    k : int
        Token count per window (both short and long).
    s : int | None
        Sparse stride for the long branch.  ``None`` → ``round(√k)``.
    short_overlap : int
        Frame overlap for tiled short windows.
    long_overlap_tokens : int
        Token-space overlap for tiled long windows.
    mode : str
        ``"train"`` | ``"val"`` | ``"test"``.
    seed : int
    augment_flip : bool
        Time-reversal augmentation (train only).
    flip_prob : float
    calib_matrix : (4,4) Tensor | None
    image_size : (H, W) | None
    """

    def __init__(
        self,
        base_dataset: TUSRecS3Iterable,
        k: int = 64,
        s: Optional[int] = None,
        short_overlap: int = 8,
        long_overlap_tokens: int = 0,
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
        self.short_overlap = max(0, min(int(short_overlap), self.k - 1))
        self.long_overlap_tokens = max(0, min(int(long_overlap_tokens), self.k - 1))
        self.mode = str(mode).lower()
        self.seed = int(seed)
        self.epoch = 0
        self.augment_flip = bool(augment_flip) and self.mode == "train"
        self.flip_prob = float(flip_prob)
        self.calib_matrix = calib_matrix
        self.image_size = image_size

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_worker_info(self) -> Tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    # ------------------------------------------------------------------
    # Joint tile computation
    # ------------------------------------------------------------------
    def _compute_joint_tiles(
        self, T_total: int, rng: random.Random
    ) -> List[int]:
        """Return a list of valid start offsets ``t`` where both short and long
        windows of length *k* fit within [0, T_total).

        Short window covers ``[t .. t+k-1]``.
        Long  window covers ``[t, t+s, …, t+(k-1)*s]``, i.e. max index
        ``t + (k-1)*s``.

        The binding constraint is the long span, which is always ≥ short span.
        """
        long_span = (self.k - 1) * self.s  # real-frame span of long window
        # Need: t + long_span < T_total  ⟹  t ≤ T_total - long_span - 1
        max_start = T_total - long_span - 1
        if max_start < 0:
            return []  # scan too short

        # Tiled stride: use short stride (we want dense coverage)
        short_stride = max(1, self.k - self.short_overlap)

        # Also compute long token stride for minimum coverage
        long_token_stride = max(1, self.k - self.long_overlap_tokens)
        long_frame_stride = long_token_stride * self.s

        # Use the smaller stride so both branches get enough overlap
        tile_stride = min(short_stride, long_frame_stride)

        # Epoch jitter (train only)
        max_jitter = min(tile_stride, max_start) if (self.mode == "train" and max_start > 0) else 0
        jitter = rng.randint(0, max(0, max_jitter)) if max_jitter > 0 else 0

        starts: List[int] = []
        pos = jitter
        while pos <= max_start:
            starts.append(pos)
            pos += tile_stride

        return starts

    # ------------------------------------------------------------------
    # Build one sample from a shared start offset
    # ------------------------------------------------------------------
    def _make_joint_sample(
        self,
        frames_t: torch.Tensor,   # (T, H, W) uint8
        tforms_t: torch.Tensor,   # (T, 4, 4) float32
        t: int,
        scan_id: str,
        info: SliceInfo,
        T_total: int,
        rng: random.Random,
    ) -> dict:
        # ---- Short: contiguous [t .. t+k-1] ----------------------------
        s_end = t + self.k
        short_frames = frames_t[t:s_end]              # uint8 view, no clone
        short_tforms = tforms_t[t:s_end]              # float32 view

        # ---- Long: sparse [t, t+s, …, t+(k-1)*s] -----------------------
        long_idx = torch.arange(self.k, dtype=torch.long) * self.s + t
        long_frames = frames_t[long_idx]              # uint8 view
        long_tforms = tforms_t[long_idx]              # float32 view

        # ---- Time-reversal augmentation (coherent both branches) --------
        do_flip = self.augment_flip and rng.random() < self.flip_prob
        if do_flip:
            short_frames = short_frames.flip(0).contiguous()
            short_tforms = short_tforms.flip(0).contiguous()
            long_frames = long_frames.flip(0).contiguous()
            long_tforms = long_tforms.flip(0).contiguous()
            long_idx = long_idx.flip(0).contiguous()

        short_gt = _normalise_global_transforms(short_tforms)
        long_gt = _normalise_global_transforms(long_tforms)

        short_meta = _make_meta(
            scan_id, info, T_total,
            window_start=t, stride=1, k=self.k,
            calib_matrix=self.calib_matrix, image_size=self.image_size,
        )
        long_meta = _make_meta(
            scan_id, info, T_total,
            window_start=t, stride=self.s, k=self.k,
            calib_matrix=self.calib_matrix, image_size=self.image_size,
        )

        return {
            "short": {
                "frames": short_frames,         # uint8 (k, H, W) — no .float()
                "gt_global_T": short_gt,        # float32 (k, 4, 4)
            },
            "long": {
                "frames": long_frames,           # uint8 (k, H, W)
                "gt_global_T": long_gt,          # float32 (k, 4, 4)
                "idx_long": long_idx,            # int64 (k,)
            },
            "meta": {
                "scan_id": scan_id,
                "subject": info.subject,
                "scan_name": info.scan_name,
                "window_start": t,
                "k": self.k,
                "s": self.s,
                "total_frames": T_total,
            },
        }

    # ------------------------------------------------------------------
    # Per-scan iteration
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
            # Eval: yield full scan as single sample (variable length).
            # The trainer's evaluate() uses the stitch pipeline which
            # does its own windowing — we just pass everything.
            gt_global = _normalise_global_transforms(tforms_t)
            anchor_idx = list(range(0, T_total, self.s))
            yield {
                "short": {
                    "frames": frames_t,
                    "gt_global_T": gt_global,
                },
                "long": {
                    "frames": frames_t,
                    "gt_global_T": gt_global,
                    "idx_long": torch.tensor(anchor_idx, dtype=torch.long),
                },
                "meta": {
                    "scan_id": scan_id,
                    "subject": info.subject,
                    "scan_name": info.scan_name,
                    "window_start": 0,
                    "k": T_total,
                    "s": self.s,
                    "total_frames": T_total,
                },
            }
            return

        starts = self._compute_joint_tiles(T_total, rng)
        if self.mode == "train":
            rng.shuffle(starts)
        for t in starts:
            yield self._make_joint_sample(
                frames_t, tforms_t, t, scan_id, info, T_total, rng
            )

    # ------------------------------------------------------------------
    # __iter__
    # ------------------------------------------------------------------
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
# Synthetic version for smoke testing (no S3)
# ===========================================================================

class SyntheticJointKRootDualDataset(torch.utils.data.IterableDataset):
    """Synthetic joint dual-window dataset for smoke testing."""

    def __init__(
        self,
        num_scans: int = 4,
        frames_per_scan: int = 512,
        height: int = 128,
        width: int = 128,
        k: int = 64,
        s: Optional[int] = None,
        short_overlap: int = 8,
        long_overlap_tokens: int = 0,
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
        self.short_overlap = max(0, int(short_overlap))
        self.long_overlap_tokens = max(0, int(long_overlap_tokens))
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
                kk = angle / theta
                K = torch.tensor([
                    [0, -kk[2], kk[1]],
                    [kk[2], 0, -kk[0]],
                    [-kk[1], kk[0], 0],
                ])
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
            # uint8 to mimic real data (no .float())
            frames = (torch.rand(T, self.height, self.width, generator=gen) * 255).to(torch.uint8)
            gt_global = self._random_walk_transforms(T, generator=gen)

            if self.mode in ("val", "test"):
                inv0 = torch.linalg.inv(gt_global[0:1])
                normed = inv0 @ gt_global
                anchor_idx = list(range(0, T, self.s))
                yield {
                    "short": {"frames": frames, "gt_global_T": normed},
                    "long": {
                        "frames": frames,
                        "gt_global_T": normed,
                        "idx_long": torch.tensor(anchor_idx, dtype=torch.long),
                    },
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": 0,
                        "k": T,
                        "s": self.s,
                        "total_frames": T,
                    },
                }
            else:
                long_span = (self.k - 1) * self.s
                if T < long_span + 1:
                    continue
                max_start = T - long_span - 1
                short_stride = max(1, self.k - self.short_overlap)
                long_token_stride = max(1, self.k - self.long_overlap_tokens)
                long_frame_stride = long_token_stride * self.s
                tile_stride = min(short_stride, long_frame_stride)

                starts = []
                pos = 0
                while pos <= max_start:
                    starts.append(pos)
                    pos += tile_stride
                rng.shuffle(starts)

                for t in starts:
                    # Short
                    s_frames = frames[t : t + self.k]
                    s_tforms = gt_global[t : t + self.k]
                    inv0 = torch.linalg.inv(s_tforms[0:1])
                    s_gt = inv0 @ s_tforms

                    # Long
                    l_idx = torch.arange(self.k, dtype=torch.long) * self.s + t
                    l_frames = frames[l_idx]
                    l_tforms = gt_global[l_idx]
                    inv0l = torch.linalg.inv(l_tforms[0:1])
                    l_gt = inv0l @ l_tforms

                    yield {
                        "short": {"frames": s_frames, "gt_global_T": s_gt},
                        "long": {
                            "frames": l_frames,
                            "gt_global_T": l_gt,
                            "idx_long": l_idx,
                        },
                        "meta": {
                            "scan_id": f"synth_{scan_idx}",
                            "subject": "synth",
                            "scan_name": f"scan_{scan_idx}",
                            "window_start": t,
                            "k": self.k,
                            "s": self.s,
                            "total_frames": T,
                        },
                    }
