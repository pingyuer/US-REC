from typing import Dict, List, Sequence, Tuple


def build_pair_index(num_frames: int) -> List[int]:
    if num_frames < 2:
        return []
    return list(range(num_frames - 1))


def build_epoch_index_map(
    *,
    slice_ids: Sequence[int],
    pair_indices_by_slice: Dict[int, List[int]],
    batch_size: int,
    iterations: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Build a stable index map for an epoch: List[(slice_id, pair_idx)].
    This stays pure CPU and does not touch any data buffers.
    """
    import random

    rng = random.Random(seed)
    index_map: List[Tuple[int, int]] = []
    per_slice = {sid: pair_indices_by_slice[sid][:] for sid in slice_ids}
    for sid in slice_ids:
        rng.shuffle(per_slice[sid])

    slice_cycle = list(slice_ids)
    for _ in range(iterations):
        rng.shuffle(slice_cycle)
        for b in range(batch_size):
            sid = slice_cycle[b % len(slice_cycle)]
            if not per_slice[sid]:
                per_slice[sid] = pair_indices_by_slice[sid][:]
                rng.shuffle(per_slice[sid])
            pair_idx = per_slice[sid].pop()
            index_map.append((sid, pair_idx))
    return index_map
