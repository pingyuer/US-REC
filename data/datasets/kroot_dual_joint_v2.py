"""V2 Joint K-root Dual Dataset — scientific stride computation + real frame indices.

Changes vs V1 :class:`JointKRootDualDataset`:

1. **Stride from target coverage**:
   ``s = ceil(L_target_frames / (k - 1))``
   instead of the heuristic ``round(√k)``.  ``L_target_frames`` is the
   desired *real-frame span* of the long window (e.g. 500 frames for
   a ~10 s scan at 50 fps).  Falls back to V1 heuristic when
   ``L_target_frames`` is not provided.

2. **Real frame indices** in the long branch metadata,
   enabling :class:`RealIndexSinusoidalPosEmb` in the V2 transformer.

3. **Scan-adaptive stride** (optional): when a scan has fewer than
   ``L_target_frames`` frames, ``s`` is re-clamped so the long window
   still fits.

Everything else — joint tile computation, augmentation, train/val
split, worker partitioning — is inherited from the V1 base class.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Tuple

import torch

from data.datasets.kroot_dual_joint import (
    JointKRootDualDataset,
    SyntheticJointKRootDualDataset,
    _normalise_global_transforms,
    _make_meta,
)


# ─── V2 real dataset ────────────────────────────────────────────────────────

class V2JointKRootDualDataset(JointKRootDualDataset):
    """Joint dual-window dataset with scientific stride computation.

    Parameters
    ----------
    base_dataset : TUSRecS3Iterable
    k : int
        Token window length (both branches).
    L_target_frames : int or None
        Desired real-frame span for the long window.  When provided:
        ``s = ceil(L_target / (k - 1))``.  When ``None``, falls back
        to explicit ``s`` or ``round(sqrt(k))``.
    s : int or None
        Explicit stride override.  Ignored when ``L_target_frames`` is set.
    scan_adaptive : bool
        When True and a scan has fewer than ``(k-1)*s + 1`` frames,
        re-compute ``s`` for that scan to make the long window fit.
    short_overlap : int
    long_overlap_tokens : int
    mode : str
    seed : int
    augment_flip : bool
    flip_prob : float
    calib_matrix : Tensor or None
    image_size : tuple or None
    """

    def __init__(
        self,
        base_dataset,
        k: int = 64,
        L_target_frames: Optional[int] = None,
        s: Optional[int] = None,
        scan_adaptive: bool = True,
        short_overlap: int = 8,
        long_overlap_tokens: int = 0,
        mode: str = "train",
        seed: int = 0,
        augment_flip: bool = False,
        flip_prob: float = 0.5,
        calib_matrix: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        # Compute s from L_target_frames if provided
        self.L_target_frames = L_target_frames
        self.scan_adaptive = bool(scan_adaptive)

        if L_target_frames is not None and L_target_frames > 0 and k > 1:
            computed_s = math.ceil(L_target_frames / (k - 1))
            computed_s = max(1, computed_s)
        elif s is not None:
            computed_s = int(s)
        else:
            computed_s = None  # will fall back to sqrt(k) in super().__init__

        super().__init__(
            base_dataset=base_dataset,
            k=k,
            s=computed_s,
            short_overlap=short_overlap,
            long_overlap_tokens=long_overlap_tokens,
            mode=mode,
            seed=seed,
            augment_flip=augment_flip,
            flip_prob=flip_prob,
            calib_matrix=calib_matrix,
            image_size=image_size,
        )

    # ------------------------------------------------------------------
    def _resolve_s_for_scan(self, T_total: int) -> int:
        """Possibly re-compute s so the long window fits in this scan."""
        s = self.s
        long_span = (self.k - 1) * s
        if self.scan_adaptive and long_span >= T_total and self.k > 1:
            # Clamp s so the long window fits
            s = max(1, (T_total - 1) // (self.k - 1))
        return s

    # ------------------------------------------------------------------
    # Override _make_joint_sample to pass real frame indices for long PE
    # ------------------------------------------------------------------
    def _make_joint_sample(
        self,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        t: int,
        scan_id: str,
        info,
        T_total: int,
        rng: random.Random,
    ) -> dict:
        s = self._resolve_s_for_scan(T_total)

        # ---- Short: contiguous [t .. t+k-1] ----------------------------
        s_end = t + self.k
        short_frames = frames_t[t:s_end]
        short_tforms = tforms_t[t:s_end]

        # Real position ids for short: contiguous starting at t
        short_pos_ids = torch.arange(t, t + self.k, dtype=torch.long)

        # ---- Long: sparse [t, t+s, …, t+(k-1)*s] -----------------------
        long_idx = torch.arange(self.k, dtype=torch.long) * s + t
        long_frames = frames_t[long_idx]
        long_tforms = tforms_t[long_idx]

        # ---- Time-reversal augmentation (coherent both branches) --------
        do_flip = self.augment_flip and rng.random() < self.flip_prob
        if do_flip:
            short_frames = short_frames.flip(0).contiguous()
            short_tforms = short_tforms.flip(0).contiguous()
            short_pos_ids = short_pos_ids.flip(0).contiguous()
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
            window_start=t, stride=s, k=self.k,
            calib_matrix=self.calib_matrix, image_size=self.image_size,
        )

        return {
            "short": {
                "frames": short_frames,
                "gt_global_T": short_gt,
                "position_ids": short_pos_ids,  # V2: real frame indices
            },
            "long": {
                "frames": long_frames,
                "gt_global_T": long_gt,
                "idx_long": long_idx,           # V1 compat + real frame indices
                "position_ids": long_idx,       # V2: real frame indices for PE
            },
            "meta": {
                "scan_id": scan_id,
                "subject": info.subject,
                "scan_name": info.scan_name,
                "window_start": t,
                "k": self.k,
                "s": s,
                "L_target_frames": self.L_target_frames or (self.k - 1) * s,
                "total_frames": T_total,
            },
        }

    # ------------------------------------------------------------------
    # Override _compute_joint_tiles to use scan-adaptive s
    # ------------------------------------------------------------------
    def _compute_joint_tiles(
        self, T_total: int, rng: random.Random
    ) -> List[int]:
        s = self._resolve_s_for_scan(T_total)
        long_span = (self.k - 1) * s
        max_start = T_total - long_span - 1
        if max_start < 0:
            return []

        short_stride = max(1, self.k - self.short_overlap)
        long_token_stride = max(1, self.k - self.long_overlap_tokens)
        long_frame_stride = long_token_stride * s
        tile_stride = min(short_stride, long_frame_stride)

        max_jitter = min(tile_stride, max_start) if (self.mode == "train" and max_start > 0) else 0
        jitter = rng.randint(0, max(0, max_jitter)) if max_jitter > 0 else 0

        starts: List[int] = []
        pos = jitter
        while pos <= max_start:
            starts.append(pos)
            pos += tile_stride
        return starts

    # ------------------------------------------------------------------
    # Override _iter_scan for val/test to include V2 fields
    # ------------------------------------------------------------------
    def _iter_scan(
        self,
        info,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        T_total = frames_t.shape[0]
        scan_id = f"{info.subject}/{info.scan_name}"
        s = self._resolve_s_for_scan(T_total)

        if self.mode in ("val", "test"):
            gt_global = _normalise_global_transforms(tforms_t)
            anchor_idx = list(range(0, T_total, s))

            yield {
                "short": {
                    "frames": frames_t,
                    "gt_global_T": gt_global,
                    "position_ids": torch.arange(T_total, dtype=torch.long),
                },
                "long": {
                    "frames": frames_t,
                    "gt_global_T": gt_global,
                    "idx_long": torch.tensor(anchor_idx, dtype=torch.long),
                    "position_ids": torch.tensor(anchor_idx, dtype=torch.long),
                },
                "meta": {
                    "scan_id": scan_id,
                    "subject": info.subject,
                    "scan_name": info.scan_name,
                    "window_start": 0,
                    "k": T_total,
                    "s": s,
                    "L_target_frames": self.L_target_frames or (self.k - 1) * s,
                    "total_frames": T_total,
                },
            }
            return

        starts = self._compute_joint_tiles(T_total, rng)
        if self.mode == "train":
            rng.shuffle(starts)
        for t in starts:
            yield self._make_joint_sample(
                frames_t, tforms_t, t, scan_id, info, T_total, rng,
            )


# ─── Synthetic V2 dataset for smoke testing ─────────────────────────────────

class SyntheticV2JointKRootDualDataset(torch.utils.data.IterableDataset):
    """Synthetic V2 joint dual-window dataset for smoke testing.

    Identical to :class:`SyntheticJointKRootDualDataset` but computes
    ``s = ceil(L_target / (k-1))`` and includes ``position_ids``.
    """

    def __init__(
        self,
        num_scans: int = 4,
        frames_per_scan: int = 512,
        height: int = 128,
        width: int = 128,
        k: int = 64,
        L_target_frames: Optional[int] = None,
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
        self.L_target_frames = L_target_frames

        # Compute s from L_target_frames
        if L_target_frames is not None and L_target_frames > 0 and k > 1:
            self.s = max(1, math.ceil(L_target_frames / (self.k - 1)))
        elif s is not None:
            self.s = int(s)
        else:
            self.s = int(round(math.sqrt(self.k)))

        self.short_overlap = max(0, int(short_overlap))
        self.long_overlap_tokens = max(0, int(long_overlap_tokens))
        self.mode = str(mode).lower()
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _random_walk_transforms(T: int, *, generator: torch.Generator) -> torch.Tensor:
        """Generate a random walk of SE(3) transforms."""
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
            frames = (torch.rand(T, self.height, self.width, generator=gen) * 255).to(torch.uint8)
            gt_global = self._random_walk_transforms(T, generator=gen)

            if self.mode in ("val", "test"):
                inv0 = torch.linalg.inv(gt_global[0:1])
                normed = inv0 @ gt_global
                anchor_idx = list(range(0, T, self.s))
                yield {
                    "short": {
                        "frames": frames,
                        "gt_global_T": normed,
                        "position_ids": torch.arange(T, dtype=torch.long),
                    },
                    "long": {
                        "frames": frames,
                        "gt_global_T": normed,
                        "idx_long": torch.tensor(anchor_idx, dtype=torch.long),
                        "position_ids": torch.tensor(anchor_idx, dtype=torch.long),
                    },
                    "meta": {
                        "scan_id": f"synth_{scan_idx}",
                        "subject": "synth",
                        "scan_name": f"scan_{scan_idx}",
                        "window_start": 0,
                        "k": T,
                        "s": self.s,
                        "L_target_frames": self.L_target_frames or (self.k - 1) * self.s,
                        "total_frames": T,
                    },
                }
            else:
                s_eff = self.s
                long_span = (self.k - 1) * s_eff
                # Scan-adaptive: clamp s if scan is too short
                if long_span >= T and self.k > 1:
                    s_eff = max(1, (T - 1) // (self.k - 1))
                    long_span = (self.k - 1) * s_eff
                if T < long_span + 1:
                    continue
                max_start = T - long_span - 1
                short_stride = max(1, self.k - self.short_overlap)
                tile_stride = short_stride

                starts = []
                pos = 0
                while pos <= max_start:
                    starts.append(pos)
                    pos += tile_stride
                rng.shuffle(starts)

                for t in starts:
                    # Short
                    s_frames = frames[t: t + self.k]
                    s_tforms = gt_global[t: t + self.k]
                    inv0 = torch.linalg.inv(s_tforms[0:1])
                    s_gt = inv0 @ s_tforms
                    short_pos_ids = torch.arange(t, t + self.k, dtype=torch.long)

                    # Long (use s_eff for scan-adaptive stride)
                    l_idx = torch.arange(self.k, dtype=torch.long) * s_eff + t
                    l_frames = frames[l_idx]
                    l_tforms = gt_global[l_idx]
                    inv0l = torch.linalg.inv(l_tforms[0:1])
                    l_gt = inv0l @ l_tforms

                    yield {
                        "short": {
                            "frames": s_frames,
                            "gt_global_T": s_gt,
                            "position_ids": short_pos_ids,
                        },
                        "long": {
                            "frames": l_frames,
                            "gt_global_T": l_gt,
                            "idx_long": l_idx,
                            "position_ids": l_idx,
                        },
                        "meta": {
                            "scan_id": f"synth_{scan_idx}",
                            "subject": "synth",
                            "scan_name": f"scan_{scan_idx}",
                            "window_start": t,
                            "k": self.k,
                            "s": s_eff,
                            "L_target_frames": self.L_target_frames or (self.k - 1) * s_eff,
                            "total_frames": T,
                        },
                    }
