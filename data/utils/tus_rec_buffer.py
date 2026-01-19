from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class SliceBuffer:
    frames: torch.Tensor
    tforms: torch.Tensor
    tforms_inv: torch.Tensor
    pair_indices: List[int]

    def get_pair(self, pair_idx: int):
        i = pair_idx
        frames_pair = self.frames[i : i + 2]
        tforms_pair = self.tforms[i : i + 2]
        tforms_inv = self.tforms_inv[i : i + 2]
        return frames_pair, tforms_pair, tforms_inv


class SliceBufferManager:
    """
    Holds active slice buffers. No I/O here.
    """

    def __init__(self):
        self.buffers: Dict[int, SliceBuffer] = {}

    def set_buffers(self, buffers: Dict[int, SliceBuffer]) -> None:
        self.buffers = buffers

    def get_pair(self, slice_id: int, pair_idx: int):
        return self.buffers[slice_id].get_pair(pair_idx)
