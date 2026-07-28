"""Ray storage configuration, including S3-compatible MinIO."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


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
