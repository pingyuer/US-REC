import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from typing import Callable, Iterable, List, Optional, Set, Tuple

from s3torchconnector._s3client import S3Client, S3ClientConfig
from s3torchconnector import S3ReaderConstructor
from s3torchconnector.s3reader import S3ReaderConstructorProtocol

from .dataset import BaseDataset


class TUIS3(BaseDataset):
    """TUI map-style dataset that streams images and lists directly from S3."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        label_path: str,
        *,
        region: str,
        img_dir: Optional[str] = None,
        endpoint: Optional[str] = None,
        force_path_style: bool = True,
        split_file: Optional[str] = None,
        pipeline: Optional[Callable] = None,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
        s3client_config: Optional[S3ClientConfig] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.img_dir = img_dir.strip("/") if img_dir else ""
        self.endpoint = endpoint
        self.force_path_style = bool(force_path_style)
        self.reader_constructor = reader_constructor or S3ReaderConstructor.default()
        self._s3client_config = s3client_config or S3ClientConfig(force_path_style=self.force_path_style)
        self._client: Optional[S3Client] = None
        self._img_prefix = self._join_paths(self.prefix, self.img_dir)
        self._apply_credentials(aws_access_key_id, aws_secret_access_key, aws_session_token)

        annotations = self._load_label_json(label_path)
        data_list = self._build_data_list(annotations, split_file)
        super().__init__(data_list, pipeline)

    def _apply_credentials(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
    ):
        if aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        if aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        if aws_session_token:
            os.environ["AWS_SESSION_TOKEN"] = aws_session_token

    @staticmethod
    def _join_paths(*segments: str) -> str:
        parts = [seg.strip("/") for seg in segments if seg]
        return "/".join(parts)

    def _load_label_json(self, label_path: str) -> List[dict]:
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                annotations = json.load(f)
        else:
            bucket, key = self._resolve_s3_location(label_path, root=self.prefix)
            raw = self._read_text(key, bucket)
            annotations = json.loads(raw)
        assert isinstance(annotations, list), "Annotation JSON must be a list of dicts."
        return annotations

    def _build_data_list(self, annotations: Iterable[dict], split_file: Optional[str]) -> List[dict]:
        allowed_ids = self._load_split_list(split_file)

        data_list = []
        for ann in tqdm(annotations, desc="Building S3 data list"):
            img_file = ann.get("img_file")
            if not img_file or (allowed_ids is not None and img_file not in allowed_ids):
                continue

            img_key = self._resolve_img_key(img_file)
            segments = self._parse_segments(ann)
            data_list.append({
                "img_key": img_key,
                "segments": segments,
                "meta": ann,
            })
        return data_list

    def _load_split_list(self, split_file: Optional[str]) -> Optional[Set[str]]:
        if not split_file:
            return None

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            bucket, key = self._resolve_s3_location(split_file, root=self.prefix)
            content = self._read_text(key, bucket)
            lines = [line.strip() for line in content.splitlines() if line.strip()]

        return set(lines)

    def _parse_segments(self, annotation: dict) -> List[np.ndarray]:
        segments = []
        for nodule in annotation.get("nodule_location", []):
            seg = nodule.get("segment")
            if seg is not None and len(seg) > 2:
                segments.append(np.array(seg, dtype=np.int32))
        return segments

    def _resolve_img_key(self, img_file: str) -> str:
        img_file = img_file.strip("/")
        if self._img_prefix:
            if img_file.startswith(f"{self._img_prefix}/") or img_file == self._img_prefix:
                return img_file
            return f"{self._img_prefix}/{img_file}"
        return img_file

    def _resolve_s3_location(self, path: str, *, root: Optional[str] = None) -> Tuple[str, str]:
        path = path.strip()
        if path.startswith("s3://"):
            _, without_scheme = path.split("s3://", 1)
            bucket, key = without_scheme.split("/", 1)
            return bucket, key
        normalized = path.lstrip("/")
        base = (root or self.prefix).strip("/") if (root or self.prefix) else ""
        if base:
            if normalized == base or normalized.startswith(f"{base}/"):
                key = normalized
            else:
                key = f"{base}/{normalized}"
        else:
            key = normalized
        return self.bucket, key

    def _get_client(self) -> S3Client:
        if self._client is None:
            self._client = S3Client(
                region=self.region,
                endpoint=self.endpoint,
                s3client_config=self._s3client_config,
            )
        return self._client

    def _read_bytes(self, key: str, bucket: Optional[str] = None) -> bytes:
        reader = self._get_client().get_object(
            bucket=bucket or self.bucket,
            key=key,
            reader_constructor=self.reader_constructor,
        )
        payload = reader.read()
        if payload is None:
            return b""
        return payload

    def _read_text(self, key: str, bucket: Optional[str] = None) -> str:
        return self._read_bytes(key, bucket).decode("utf-8")

    def _fetch_image(self, key: str) -> np.ndarray:
        raw = self._read_bytes(key)
        if not raw:
            raise RuntimeError(f"Empty payload for s3://{self.bucket}/{key}")
        img_array = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode s3://{self.bucket}/{key}")
        return img.astype(np.float32) / 255.0

    def load_sample(self, item: dict) -> dict:
        img_ref = item.get("img_key")
        if img_ref:
            img_ref = f"s3://{self.bucket}/{img_ref}"
        return {
            "img_ref": img_ref,
            "segments": item.get("segments") or [],
            "meta": item.get("meta") or {},
            "s3": {
                "region": self.region,
                "endpoint": self.endpoint,
                "force_path_style": self.force_path_style,
            },
        }

    def get_raw_image_bytes(self, img_file: str):
        try:
            key = self._resolve_img_key(str(img_file))
            payload = self._read_bytes(key)
            return payload or None
        except Exception:
            return None

    def get_raw_image_ref(self, img_file: str):
        try:
            key = self._resolve_img_key(str(img_file))
            return f"s3://{self.bucket}/{key}"
        except Exception:
            return None
