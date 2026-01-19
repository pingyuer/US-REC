from __future__ import annotations

from typing import Optional

from utils.loader import SSFrameDataset


class TUSRec(SSFrameDataset):
    """
    Local (filesystem) dataset for reconstruction/registration.

    This is a thin wrapper around SSFrameDataset to make it instantiable from
    the unified config reflection builder (data/builder.py).
    """

    def __init__(
        self,
        *,
        data_path: str,
        h5_file_name: str,
        min_scan_len: int,
        num_samples: int,
        sample_range: Optional[int] = None,
        indices_json: Optional[str] = None,
        pipeline=None,
    ):
        if indices_json:
            ds = SSFrameDataset.read_json(data_path, indices_json, h5_file_name, num_samples=num_samples)
            self.__dict__ = ds.__dict__
            self.pipeline = pipeline
            return

        super().__init__(
            min_scan_len=min_scan_len,
            data_path=data_path,
            h5_file_name=h5_file_name,
            indices_in_use=None,
            num_samples=num_samples,
            sample_range=sample_range,
        )
        self.pipeline = pipeline

