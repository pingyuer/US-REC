import os
import torch.utils.data
from PIL import Image
from .dataset import BaseDataset
import cv2
import numpy as np
import json
from tqdm import tqdm

class TUI(BaseDataset):
    """
    TUI annotations are stored as a list of dictionaries in a JSON file (one dict per image).

    The JSON file is expected to contain a list of annotation objects. Each
    annotation object corresponds to one image and may include multiple nodules.
    The minimal expected keys and their usage are described below.

    Example JSON entry:
    {
        "img_file": "m001578.png",
        "bn": "malignant",
        "artery": true,
        "machine": "Philips1",
        "hv": "Horizontal",
        "anchor": [[230, 104], [287, 56]],
        "nodule_location": [
            {
                "bbox": [50, 55, 230, 59],
                "segment": [[256, 59], [255, 60], ...]  # polygon points [x, y]
            },
            ...
        ],
        "crop_location": [145, 119],
        "remove_nodule": false,
        "remove_mark": true,
        "size": [7.66, 5.0]
    }

    Args:
        img_dir (str): Directory containing input images.
        label_path (str): Path to the JSON file described above (list of dicts).
        split_file (str, optional): A text file that defines the subset of data
            (e.g., train/val/test). Each line should contain a image file.
            If None, all files in `img_dir` will be used.
        pipeline (callable, optional): Data augmentation or preprocessing pipeline
            applied to each sample.
    """

    def __init__(self, img_dir, label_path, split_file=None, pipeline=None):
        self.img_dir = img_dir
        # Build sample index list
        annotations = self.load_label_json(label_path)
        data_list = self.load_data_list(img_dir, annotations, split_file)
        # Initialize BaseDataset with data_list and pipeline
        super().__init__(data_list, pipeline)


    def load_label_json(self, label_json):
        """
        Load annotation JSON file. Expected format:
        Returns:
            list[dict]: list of annotation dicts
        """
        with open(label_json, "r") as f:
            annotations = json.load(f)
        assert isinstance(annotations, list), "Annotation JSON must be a list of dicts."
        
        return annotations

    def load_data_list(self, img_dir, annotations, split_file):
        """
        Convert annotations into internal sample list.
        
        Each element in data_list will be:
            {
                "img": str,                  # full path to the image file
                "segments": list of ndarray, # list of polygons (Nx2, int32)
                "meta": dict                 # original annotation dict
            }
        """
        # optional: load split list (e.g. train.txt)
        allowed_ids = None
        if split_file is not None:
            with open(split_file, "r") as f:
                allowed_ids = set([line.strip() for line in f if line.strip()])
        data_list = []
        for ann in tqdm(annotations, desc="Building data list"):
            img_file = ann.get("img_file")
            if img_file is None:
                continue
            if allowed_ids is not None and img_file not in allowed_ids:
                continue

            img_path = os.path.join(img_dir, img_file)
            if not os.path.exists(img_path):
                continue

            # extract segmentation polygons
            segments = []
            for nodule in ann.get("nodule_location", []):
                seg = nodule.get("segment")
                if seg is not None and len(seg) > 2:  # valid polygon
                    seg = np.array(seg, dtype=np.int32)
                    segments.append(seg)

            sample = {
                "img_path": img_path,
                "segments": segments,
                "meta": ann  # keep full annotation for downstream use
            }
            data_list.append(sample)
        return data_list

    def load_sample(self, item):
        """
        Return an indexing-only sample; decoding/mask-building are handled by transforms.

        Args:
            item (dict): Dictionary containing "img" and "mask" file paths.

        Returns:
            dict: A sample dictionary with:
                - "image": numpy array of the image, normalized to [0,1], dtype=float32
                - "mask": numpy array of the mask, dtype=int64 (for CrossEntropyLoss)
                - "meta": metadata for downstream use
        
        Note: `pipeline` is expected to turn this into {"image": ..., "mask": ...}
        """
        return {
            "img_ref": item.get("img_path") or item.get("img"),
            "segments": item.get("segments") or [],
            "meta": item.get("meta") or {},
        }

    def get_raw_image_bytes(self, img_file: str):
        img_path = os.path.join(self.img_dir, str(img_file))
        if not os.path.exists(img_path):
            return None
        try:
            with open(img_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def get_raw_image_ref(self, img_file: str):
        return os.path.join(self.img_dir, str(img_file))
