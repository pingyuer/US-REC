from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import torch

from .registry import register_transform


def _read_bytes(sample: dict) -> Optional[bytes]:
    img_ref = sample.get("img_ref") or sample.get("img_file") or sample.get("img_path") or sample.get("img_key")
    if not img_ref:
        return None
    img_ref = str(img_ref)

    dataset = sample.get("__dataset__")
    reader = getattr(dataset, "get_raw_image_bytes", None) if dataset is not None else None
    if callable(reader):
        payload = reader(img_ref)
        return payload or None

    if img_ref.startswith("s3://"):
        return _read_s3_bytes(img_ref, sample.get("s3") or {})

    try:
        with open(img_ref, "rb") as f:
            return f.read()
    except Exception:
        return None


def _read_s3_bytes(uri: str, s3_cfg: dict) -> Optional[bytes]:
    try:
        from s3torchconnector._s3client import S3Client, S3ClientConfig  # type: ignore
        from s3torchconnector import S3ReaderConstructor  # type: ignore
    except Exception:
        return None

    try:
        _, without_scheme = uri.split("s3://", 1)
        bucket, key = without_scheme.split("/", 1)
    except Exception:
        return None

    region = s3_cfg.get("region") or "us-east-1"
    endpoint = s3_cfg.get("endpoint")
    force_path_style = bool(s3_cfg.get("force_path_style", True))

    client = S3Client(
        region=region,
        endpoint=endpoint,
        s3client_config=S3ClientConfig(force_path_style=force_path_style),
    )
    reader = client.get_object(
        bucket=bucket,
        key=key,
        reader_constructor=S3ReaderConstructor.default(),
    )
    payload = reader.read()
    if not payload:
        return None
    return payload


def _decode_image(payload: bytes) -> Optional[np.ndarray]:
    if not payload:
        return None

    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(payload)) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        return arr.astype(np.float32) / 255.0
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def _fill_polygons(mask: np.ndarray, segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        return mask

    try:
        import cv2  # type: ignore

        for seg in segments:
            if seg is None:
                continue
            seg = np.asarray(seg, dtype=np.int32)
            if seg.size < 6:
                continue
            cv2.fillPoly(mask, [seg], 1)
        return mask
    except Exception:
        pass

    try:
        from PIL import Image, ImageDraw  # type: ignore

        pil = Image.fromarray(mask, mode="L")
        draw = ImageDraw.Draw(pil)
        for seg in segments:
            seg = np.asarray(seg, dtype=np.int32)
            if seg.size < 6:
                continue
            pts = [(int(x), int(y)) for x, y in seg.reshape(-1, 2)]
            draw.polygon(pts, outline=1, fill=1)
        return np.asarray(pil, dtype=np.uint8)
    except Exception:
        return mask


@register_transform("LoadRawBytes")
class LoadRawBytes:
    """Load original image bytes into sample['raw_bytes']."""

    def __call__(self, sample: dict) -> dict:
        sample.pop("__dataset__", None)
        payload = _read_bytes(sample)
        sample["raw_bytes"] = payload
        return sample


@register_transform("DecodeImage")
class DecodeImage:
    """Decode sample['raw_bytes'] into sample['image'] (H,W,C float32 in [0,1])."""

    def __call__(self, sample: dict) -> dict:
        payload = sample.get("raw_bytes")
        if payload:
            img = _decode_image(payload)
            if img is not None:
                sample["image"] = img
        return sample


@register_transform("BuildMaskFromSegments")
class BuildMaskFromSegments:
    """Convert polygon segments into a binary mask aligned to sample['image']."""

    def __init__(self, foreground_value: int = 1):
        self.foreground_value = int(foreground_value)

    def __call__(self, sample: dict) -> dict:
        image = sample.get("image")
        if image is None:
            return sample
        h, w = int(image.shape[0]), int(image.shape[1])

        segments = sample.get("segments") or []
        seg_arrays: list[np.ndarray] = []
        for seg in segments:
            if seg is None:
                continue
            seg_arrays.append(np.asarray(seg, dtype=np.int32))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask = _fill_polygons(mask, seg_arrays)
        if self.foreground_value != 1:
            mask = (mask > 0).astype(np.uint8) * np.uint8(self.foreground_value)
        sample["mask"] = mask.astype(np.int64)
        return sample


@register_transform("ApplyAlbumentations")
class ApplyAlbumentations:
    """
    Apply an Albumentations pipeline to (image, mask).

    Config example in YAML:
      - type: ApplyAlbumentations
        transforms:
          - type: HorizontalFlip
            p: 0.5
          - type: Normalize
            mean: [...]
            std: [...]
          - type: ToTensorV2
    """

    def __init__(self, *, transforms: list[dict]):
        import albumentations as A  # local import for optional dependency
        from albumentations.pytorch import ToTensorV2

        ts = []
        for t_cfg in transforms or []:
            t_type = t_cfg["type"]
            params = {k: v for k, v in t_cfg.items() if k != "type"}
            if t_type == "ToTensorV2":
                ts.append(ToTensorV2(**params))
                continue
            if hasattr(A, t_type):
                ts.append(getattr(A, t_type)(**params))
                continue
            raise KeyError(f"Albumentations transform '{t_type}' not found")

        self._compose = A.Compose(ts)

    def __call__(self, sample: dict) -> dict:
        if "image" not in sample:
            return sample

        image = sample["image"]
        mask = sample.get("mask")

        if mask is None:
            out = self._compose(image=image)
        else:
            out = self._compose(image=image, mask=mask)

        sample["image"] = out["image"]
        if "mask" in out:
            sample["mask"] = out["mask"]
        return sample


@register_transform("ToTorchTensor")
class ToTorchTensor:
    """Convert numpy HWC image/mask into torch tensors (CHW float32 / HW long)."""

    def __call__(self, sample: dict) -> dict:
        image = sample.get("image")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            sample["image"] = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        mask = sample.get("mask")
        if isinstance(mask, np.ndarray):
            sample["mask"] = torch.from_numpy(mask).long()
        elif torch.is_tensor(mask):
            sample["mask"] = mask.long()
        return sample
