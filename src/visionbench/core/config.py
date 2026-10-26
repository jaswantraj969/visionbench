"""Runtime configuration loaded from environment variables (prefix ``VISIONBENCH_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DATA_SUBDIRS = ("datasets", "artifacts", "db")

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

    @property
    def resolved_data_root(self) -> Path:
        return self.data_root.expanduser().resolve()

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
