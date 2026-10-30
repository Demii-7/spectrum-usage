"""Ray storage configuration, including S3-compatible MinIO."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


def spectrum_s3_client():
    """Client for the source data bucket, independent of Ray's MinIO store."""
    import boto3
    from pathlib import Path

    def credential(name: str) -> str:
        value = os.environ.get(name)
        if value:
            return value
        return Path(os.environ[f"{name}_FILE"]).read_text().strip()

    return boto3.client(
        "s3",
        endpoint_url=os.environ["SPECTRUM_S3_ENDPOINT_URL"],
        aws_access_key_id=credential("SPECTRUM_AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=credential("SPECTRUM_AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


@dataclass(frozen=True)
class MinIOConfig:
    bucket: str
    prefix: str = "spectrum-usage/ray"
    endpoint: str | None = None
    region: str = "us-east-1"

    @property
    def storage_path(self) -> str:
        prefix = self.prefix.strip("/")
        return f"s3://{self.bucket}/{prefix}" if prefix else f"s3://{self.bucket}"

    def environment(self, source: Mapping[str, str] | None = None) -> dict[str, str]:
        env = source or os.environ
        required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing = [key for key in required if not env.get(key)]
        if missing:
            raise ValueError(f"MinIO credentials are missing from environment: {', '.join(missing)}")
        result = {key: env[key] for key in required}
        result["AWS_DEFAULT_REGION"] = self.region
        if self.endpoint:
            result["AWS_ENDPOINT_URL"] = self.endpoint
            result["AWS_S3_ENDPOINT_URL"] = self.endpoint
        return result
