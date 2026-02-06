from typing import List, Optional

from s3torchconnector._s3client import S3Client, S3ClientConfig
from s3torchconnector import S3ReaderConstructor


def create_client(
    region: str = "us-east-1",
    endpoint: Optional[str] = None,
    force_path_style: bool = True,
) -> S3Client:
    return S3Client(
        region=region,
        endpoint=endpoint,
        s3client_config=S3ClientConfig(force_path_style=force_path_style),
    )


def list_keys(
    bucket: str,
    prefix: str,
    *,
    region: str = "us-east-1",
    endpoint: Optional[str] = None,
    force_path_style: bool = True,
    max_keys: int = 1000,
    client: Optional[S3Client] = None,
) -> List[str]:
    if client is None:
        client = create_client(region=region, endpoint=endpoint, force_path_style=force_path_style)
    
    keys: List[str] = []
    stream = client.list_objects(
        bucket=bucket, prefix=prefix, delimiter="", max_keys=max_keys
    )
    for page in stream:
        for info in getattr(page, "object_info", []) or []:
            key = getattr(info, "key", None)
            if key:
                keys.append(key)
    return keys


def get_object(
    bucket: str,
    key: str,
    *,
    region: str = "us-east-1",
    endpoint: Optional[str] = None,
    force_path_style: bool = True,
    client: Optional[S3Client] = None,
) -> bytes:
    if client is None:
        client = create_client(region=region, endpoint=endpoint, force_path_style=force_path_style)

    reader = client.get_object(
        bucket=bucket, key=key, reader_constructor=S3ReaderConstructor.default()
    )
    return reader.read()
