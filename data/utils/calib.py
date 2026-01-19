
import csv
import io
import os
from urllib.parse import urlparse

import numpy as np
import torch
from s3torchconnector._s3client import S3Client, S3ClientConfig


def _read_calib_bytes(path: str, *, endpoint=None, region="us-east-1", force_path_style=True):
    if path.startswith("s3://"):
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        client = S3Client(
            region=region,
            endpoint=endpoint,
            s3client_config=S3ClientConfig(force_path_style=force_path_style),
        )
        reader = client.get_object(bucket=bucket, key=key)
        return reader.read()
    with open(os.path.join(os.getcwd(), path), "rb") as f:
        return f.read()


def read_calib_matrices(
    filename_calib,
    resample_factor,
    device,
    *,
    endpoint=None,
    region="us-east-1",
    force_path_style=True,
):
    """
    T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm} * T{resampled_image_pix -> image_pix}
    Supports local path or s3://bucket/key (uses s3torchconnector).
    """
    raw = _read_calib_bytes(
        filename_calib,
        endpoint=endpoint or os.environ.get("TUS_REC24_ENDPOINT") or os.environ.get("S3_ENDPOINT"),
        region=region or os.environ.get("TUS_REC24_REGION") or "us-east-1",
        force_path_style=force_path_style,
    )
    csv_reader = csv.reader(io.StringIO(raw.decode()), delimiter=",")
    rows = []
    for row in csv_reader:
        if not row:
            continue
        if len(row) < 4:
            continue
        try:
            rows.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        except ValueError:
            continue
    if len(rows) < 8:
        raise ValueError(f"Calibration CSV requires 8 rows of 4 floats; got {len(rows)} rows")
    tform_calib = np.asarray(rows[:8], dtype=np.float32)
    resample = np.array(
        [[resample_factor, 0, 0, 0], [0, resample_factor, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.float32,
    )
    return (
        torch.tensor(tform_calib[0:4, :], device=device),
        torch.tensor(tform_calib[4:8, :], device=device),
        torch.tensor(tform_calib[4:8, :] @ tform_calib[0:4, :] @ resample, device=device),
    )
