"""Runtime configuration loaded from environment variables (prefix ``VISIONBENCH_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_SUBDIRS = ("datasets", "artifacts", "db", "experiments", "cache", "reports")

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

    @property
    def resolved_data_root(self) -> Path:
        return self.data_root.expanduser().resolve()

    @property
    def resolved_torch_home(self) -> Path:
        """``torch_home`` if set, else ``<data_root>/cache/torch``."""
        if self.torch_home is not None:
            return self.torch_home.expanduser().resolve()
        return self.resolved_data_root / "cache" / "torch"

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
