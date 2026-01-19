import os
import torch
from torch.utils.data import Dataset
from typing import Optional


class BaseDataset(Dataset):
    def __init__(self, data_list, pipeline=None):
        """Base dataset that applies an optional pipeline to samples."""
        self.data_list = data_list
        self.pipeline = pipeline

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.load_sample(self.data_list[idx])
        if self.pipeline:
            sample = self.pipeline(sample)
        return sample

    def load_sample(self, item):
        raise NotImplementedError()

    def get_raw_image_bytes(self, img_file: str) -> Optional[bytes]:
        """
        Optional public API for hooks/artifact recorders.

        Datasets may override this to provide the original image file bytes
        (e.g., from local disk or object storage). When not implemented, hooks
        should fall back to best-effort mechanisms.
        """
        return None

    def get_raw_image_ref(self, img_file: str) -> Optional[str]:
        """Optional stable source reference (path or s3://...)."""
        return None
