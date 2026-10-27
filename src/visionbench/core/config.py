"""Runtime configuration loaded from environment variables (prefix ``VISIONBENCH_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_SUBDIRS = ("datasets", "artifacts", "db", "experiments", "cache", "reports", "tmp")

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISIONBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Field(default=Path("./var"), description="Root for all runtime data.")
    max_upload_mb: int = Field(default=200, gt=0, description="Upper bound for uploads in MB.")
    log_level: LogLevel = "INFO"

    # Dataset ingestion / duplicate detection / splitting defaults (CLI flags override).
    max_image_pixels: int = Field(default=50_000_000, gt=0)
    min_images_per_class: int = Field(default=3, ge=1)
    dhash_threshold: int = Field(default=5, ge=0, le=64)
    low_info_std_threshold: float = Field(default=3.0, ge=0)
    low_info_popcount_min: int = Field(default=8, ge=0, le=64)
    low_info_popcount_max: int = Field(default=56, ge=0, le=64)
    report_samples_per_class: int = Field(default=8, ge=0)
    conflict_policy: Literal["exclude", "error"] = "exclude"

    # torch.hub directory for pretrained weights (VISIONBENCH_TORCH_HOME); None -> data-root cache.
    torch_home: Path | None = None

    # API + worker (M4). The API has no authentication: bind to localhost only.
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Host header names accepted besides localhost / 127.0.0.1 / [::1] (DNS-rebinding guard).
    extra_trusted_hosts: list[str] = []
    max_queued_jobs: int = Field(default=10, ge=1)
    worker_poll_seconds: float = Field(default=0.5, gt=0)
    worker_heartbeat_seconds: float = Field(default=2.0, gt=0)
    cancel_timeout_seconds: float = Field(default=30.0, gt=0)
    # Server-side path imports are disabled unless import_root is set.
    import_root: Path | None = None
    # ZIP upload limits (see visionbench.data.archive).
    zip_max_members: int = Field(default=50_000, ge=1)
    zip_max_uncompressed_mb: int = Field(default=4096, ge=1)
    zip_max_ratio: float = Field(default=100.0, gt=1)

    @property
    def resolved_data_root(self) -> Path:
        return self.data_root.expanduser().resolve()

    @property
    def resolved_torch_home(self) -> Path:
        """``torch_home`` if set, else ``<data_root>/cache/torch``."""
        if self.torch_home is not None:
            return self.torch_home.expanduser().resolve()
        return self.resolved_data_root / "cache" / "torch"

    def redaction_roots(self) -> list[tuple[Path | None, str]]:
        """Absolute roots that must never appear in API responses or job errors."""
        return [
            (self.resolved_data_root, "<data_root>"),
            (
                self.import_root.expanduser().resolve() if self.import_root else None,
                "<import_root>",
            ),
            (self.resolved_torch_home, "<torch_home>"),
        ]

    def norm_stats_dir(self, dataset_id: str, split_id: str) -> Path:
        """``<data_root>/cache/norm_stats/<dataset_id>/<split_id>`` (created on demand)."""
        path = self.data_dir("cache") / "norm_stats" / dataset_id / split_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def data_dir(self, name: str) -> Path:
        """Return ``data_root/<name>``, creating it on demand."""
        if name not in DATA_SUBDIRS:
            raise ValueError(f"unknown data subdir {name!r}; expected one of {DATA_SUBDIRS}")
        path = self.resolved_data_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_data_dirs(self) -> dict[str, Path]:
        return {name: self.data_dir(name) for name in DATA_SUBDIRS}


@lru_cache
def get_settings() -> Settings:
    return Settings()
