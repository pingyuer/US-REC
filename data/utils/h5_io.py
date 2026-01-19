import io
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


def decode_frames_tforms(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(io.BytesIO(payload), "r") as f:
        frames = np.asarray(f.get("frames"))
        tforms = np.asarray(f.get("tforms"))
    if frames is None or tforms is None:
        raise KeyError("H5 missing required datasets 'frames'/'tforms'")
    return frames, tforms


def decode_tforms_only(payload: bytes) -> np.ndarray:
    with h5py.File(io.BytesIO(payload), "r") as f:
        tforms = np.asarray(f.get("tforms"))
    if tforms is None:
        raise KeyError("H5 missing required dataset 'tforms'")
    return tforms


def decode_landmarks(payload: bytes, key: str) -> Optional[np.ndarray]:
    with h5py.File(io.BytesIO(payload), "r") as f:
        if key not in f:
            return None
        return np.asarray(f[key])
