import concurrent.futures
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch

from data.utils import h5_io, s3_io


@dataclass(frozen=True)
class SliceInfo:
    frame_key: str
    tform_key: str
    subject: str
    scan_name: str


class TUSRecS3Iterable(torch.utils.data.IterableDataset):
    """
    IterableDataset for S3-backed H5 slices optimized for high-latency storage.

    Behavior summary:
    - All S3/H5 I/O occurs in __iter__ (no I/O in __init__).
    - Shards slices by global_rank/global_world for DDP + workers.
    - Uses a producer-consumer buffer with optional async prefetch.
    - Local shuffle only: randomness is within the buffer, not global.
    - Yields dict samples with keys: frames, tforms, tforms_inv.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        *,
        frame_dir: str = "frames",
        tform_dir: Optional[str] = None,
        region: str = "us-east-1",
        endpoint: Optional[str] = None,
        force_path_style: bool = True,
        max_keys: int = 1000,
        pair_mode: str = "adjacent",
        random_pairs_per_slice: Optional[float] = None,
        shuffle_slices: bool = True,
        shuffle_pairs: bool = True,
        shuffle_buffer_size: int = 0,
        prefetch_slices: int = 0,
        buffer_emit_prob: float = 0.5,
        seed: int = 0,
        mode: str = "train",
        augment_direction_flip: bool = False,
        pipeline=None,
        **_unused,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.frame_dir = frame_dir.strip("/")
        self.tform_dir = tform_dir.strip("/") if tform_dir else None
        self.region = region
        self.endpoint = endpoint
        self.force_path_style = force_path_style
        self.max_keys = max_keys
        self.pair_mode = str(pair_mode).lower()
        self.random_pairs_per_slice = random_pairs_per_slice
        self.shuffle_slices = bool(shuffle_slices)
        self.shuffle_pairs = bool(shuffle_pairs)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size or 0))
        self.prefetch_slices = max(0, int(prefetch_slices or 0))
        self.buffer_emit_prob = float(buffer_emit_prob)
        self.seed = int(seed)
        self.mode = str(mode).lower()
        self.augment_direction_flip = bool(augment_direction_flip)
        self.pipeline = pipeline
        self.epoch = 0

        if self.pair_mode not in {"adjacent", "random"}:
            raise ValueError(f"Unsupported pair_mode: {self.pair_mode}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        raise TypeError("IterableDataset does not support len()")

    def _resolve_rank_info(self) -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _resolve_worker_info(self) -> Tuple[int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def _list_slices(self, client=None) -> List[SliceInfo]:
        frame_prefix = "/".join([p for p in [self.prefix, self.frame_dir] if p])
        keys = s3_io.list_keys(
            bucket=self.bucket,
            prefix=frame_prefix,
            region=self.region,
            endpoint=self.endpoint,
            force_path_style=self.force_path_style,
            max_keys=self.max_keys,
            client=client,
        )
        slices: List[SliceInfo] = []
        for key in sorted(keys):
            if not key.lower().endswith(".h5"):
                continue
            parts = key.split("/")
            if len(parts) < 4:
                continue
            subject = parts[-2]
            scan_name = parts[-1].rsplit(".", 1)[0]
            if self.tform_dir:
                tform_key = "/".join([self.prefix, self.tform_dir, subject, f"{scan_name}.h5"])
            else:
                tform_key = key
            slices.append(
                SliceInfo(
                    frame_key=key,
                    tform_key=tform_key,
                    subject=subject,
                    scan_name=scan_name,
                )
            )
        if not slices:
            raise RuntimeError("No H5 slices found for the given prefix/frame_dir.")
        return slices

    def _slice_pair_indices(self, num_pairs: int, rng: random.Random) -> List[int]:
        if num_pairs <= 0:
            return []
        indices = list(range(num_pairs))
        if self.pair_mode == "random":
            if self.random_pairs_per_slice is None:
                pass
            elif float(self.random_pairs_per_slice) == 1.0:
                pass
            elif 0 < float(self.random_pairs_per_slice) < 1:
                k = max(1, int(num_pairs * float(self.random_pairs_per_slice)))
                indices = rng.sample(indices, min(k, num_pairs))
            else:
                k = max(1, int(self.random_pairs_per_slice))
                indices = rng.sample(indices, min(k, num_pairs))
        if self.shuffle_pairs:
            rng.shuffle(indices)
        return indices

    def _load_slice(self, info: SliceInfo, client=None) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = s3_io.get_object(
            bucket=self.bucket,
            key=info.frame_key,
            region=self.region,
            endpoint=self.endpoint,
            force_path_style=self.force_path_style,
            client=client,
        )
        frames, tforms = h5_io.decode_frames_tforms(payload)
        if info.tform_key != info.frame_key:
            payload = s3_io.get_object(
                bucket=self.bucket,
                key=info.tform_key,
                region=self.region,
                endpoint=self.endpoint,
                force_path_style=self.force_path_style,
                client=client,
            )
            tforms = h5_io.decode_tforms_only(payload)
        frames_t = torch.from_numpy(frames)
        tforms_t = torch.from_numpy(tforms.astype("float32"))
        return frames_t, tforms_t

    def _iter_loaded_slices(
        self, slice_infos: List[SliceInfo], client=None
    ) -> Iterable[Tuple[SliceInfo, torch.Tensor, torch.Tensor]]:
        _strict = self.mode in ("val", "test")
        # In eval/test mode retry up to 3 times then propagate the error.
        # In train mode a single attempt is sufficient; failures are silently skipped.
        _max_retries = 3 if _strict else 1

        def _load(info: SliceInfo) -> Tuple[torch.Tensor, torch.Tensor]:
            last_exc: Optional[Exception] = None
            for attempt in range(_max_retries):
                try:
                    return self._load_slice(info, client=client)
                except Exception as exc:
                    last_exc = exc
                    if attempt < _max_retries - 1:
                        time.sleep(0.5 * (2 ** attempt))  # 0.5 s, 1 s, …
            if _strict:
                raise RuntimeError(
                    f"[eval] Failed to load {info.frame_key!r} after {_max_retries} attempt(s): {last_exc}"
                ) from last_exc
            raise last_exc  # type: ignore[misc]  # train path: caught below

        if self.prefetch_slices <= 0:
            for info in slice_infos:
                try:
                    frames_t, tforms_t = _load(info)
                except Exception:
                    if _strict:
                        raise
                    continue
                yield info, frames_t, tforms_t
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.prefetch_slices) as executor:
            info_iter = iter(slice_infos)
            futures: dict = {}
            for _ in range(self.prefetch_slices):
                info = next(info_iter, None)
                if info is None:
                    break
                futures[executor.submit(_load, info)] = info
            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    timeout=None,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    info = futures.pop(future)
                    try:
                        frames_t, tforms_t = future.result()
                    except Exception:
                        if _strict:
                            raise
                        frames_t, tforms_t = None, None
                    next_info = next(info_iter, None)
                    if next_info is not None:
                        try:
                            futures[executor.submit(_load, next_info)] = next_info
                        except RuntimeError:
                            break
                    if frames_t is None:
                        continue
                    yield info, frames_t, tforms_t

    def _iter_samples_for_slice(
        self,
        info: SliceInfo,
        frames_t: torch.Tensor,
        tforms_t: torch.Tensor,
        rng: random.Random,
    ) -> Iterable[dict]:
        tforms_inv = torch.linalg.inv(tforms_t)
        num_pairs = max(0, frames_t.shape[0] - 1)
        if num_pairs == 0:
            return
        indices = self._slice_pair_indices(num_pairs, rng)
        for idx in indices:
            # Materialize pair tensors so buffered samples don't retain the
            # storage of the full slice tensors in memory.
            frames_pair = frames_t[idx : idx + 2].clone()
            tforms_pair = tforms_t[idx : idx + 2].clone()
            tforms_inv_pair = tforms_inv[idx : idx + 2].clone()
            frame_idx0 = int(idx)
            frame_idx1 = int(idx + 1)

            # Direction-flip augmentation (training only): with p=0.5 swap the
            # pair order so the model learns motion in both scan directions.
            if self.augment_direction_flip and rng.random() < 0.5:
                frames_pair = frames_pair.flip(0).clone()
                tforms_pair = tforms_pair.flip(0).clone()
                tforms_inv_pair = torch.linalg.inv(tforms_pair)
                frame_idx0, frame_idx1 = frame_idx1, frame_idx0

            yield {
                "frames": frames_pair,
                "tforms": tforms_pair,
                "tforms_inv": tforms_inv_pair,
                "meta": {
                    "scan_id": f"{info.subject}/{info.scan_name}",
                    "scan_name": info.scan_name,
                    "subject": info.subject,
                    "frame_idx0": frame_idx0,
                    "frame_idx1": frame_idx1,
                    "pair_mode": self.pair_mode,
                },
            }

    def _buffered_iter(self, slice_infos: List[SliceInfo], producer_rng: random.Random, client=None):
        buffer: List[dict] = []
        lock = threading.Lock()
        not_empty = threading.Condition(lock)
        not_full = threading.Condition(lock)
        done = {"value": False}
        stop = {"value": False}

        def buffer_put(sample: dict) -> bool:
            if stop["value"]:
                return False
            if self.shuffle_buffer_size <= 0:
                buffer.append(sample)
                return True
            with not_full:
                while len(buffer) >= self.shuffle_buffer_size and not stop["value"]:
                    not_full.wait()
                if stop["value"]:
                    return False
                buffer.append(sample)
                not_empty.notify()
                return True

        def producer() -> None:
            try:
                for info, frames_t, tforms_t in self._iter_loaded_slices(slice_infos, client=client):
                    if stop["value"]:
                        break
                    for sample in self._iter_samples_for_slice(info, frames_t, tforms_t, producer_rng):
                        if stop["value"]:
                            break
                        if not buffer_put(sample):
                            break
                    if stop["value"]:
                        break
            finally:
                with not_empty:
                    done["value"] = True
                    not_empty.notify_all()
                with not_full:
                    not_full.notify_all()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                with not_empty:
                    while not buffer and not done["value"]:
                        not_empty.wait()
                    if not buffer and done["value"]:
                        break
                    idx = producer_rng.randrange(len(buffer))
                    sample = buffer.pop(idx)
                    not_full.notify()
                yield sample
        finally:
            with not_full:
                stop["value"] = True
                not_full.notify_all()
            with not_empty:
                done["value"] = True
                not_empty.notify_all()
            thread.join(timeout=1)

    def __iter__(self):
        ddp_rank, ddp_world = self._resolve_rank_info()
        worker_id, num_workers = self._resolve_worker_info()
        global_rank = ddp_rank * num_workers + worker_id
        global_world = ddp_world * num_workers

        # Initialize S3 client once per worker/iterator to reuse connections
        client = s3_io.create_client(
            region=self.region,
            endpoint=self.endpoint,
            force_path_style=self.force_path_style,
        )

        global_rng = random.Random(self.seed + self.epoch)
        slices = self._list_slices(client=client)
        if self.shuffle_slices:
            global_rng.shuffle(slices)
        slices = slices[global_rank::global_world]

        worker_seed = global_rng.randint(0, 2**32 - 1) + global_rank
        producer_rng = random.Random(worker_seed)

        if not slices:
            return iter(())

        if self.shuffle_buffer_size <= 0:
            for info, frames_t, tforms_t in self._iter_loaded_slices(slices, client=client):
                for sample in self._iter_samples_for_slice(info, frames_t, tforms_t, producer_rng):
                    if self.pipeline is None:
                        yield sample
                    else:
                        yield self.pipeline(sample)
            return

        for sample in self._buffered_iter(slices, producer_rng, client=client):
            if self.pipeline is None:
                yield sample
            else:
                yield self.pipeline(sample)


# Unified entrypoint for personal development.
TUSRecS3Dataset = TUSRecS3Iterable
TUSRecS3 = TUSRecS3Iterable
