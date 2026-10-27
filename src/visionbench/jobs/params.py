"""Validated parameters for each job kind (stored as JSON in ``jobs.params_json``).

Paths are never stored absolute: uploads are referenced by upload id, server-side imports by a
path relative to ``import_root``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from visionbench.training.config import TrainConfig

DATASET_ID = r"^ds-[0-9a-f]{12}$"
SPLIT_ID = r"^sp-[0-9a-f]{12}$"
EXPERIMENT_ID = r"^exp-\d{8}-\d{6}-[0-9a-f]{6}$"
UPLOAD_ID = r"^up-[0-9a-f]{32}$"


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetImportParams(_Params):
    source: Literal["upload", "path"]
    name: str = Field(min_length=1, max_length=100)
    upload_id: str | None = Field(default=None, pattern=UPLOAD_ID)
    relpath: str | None = None  # relative to import_root (source == "path")
    original_filename: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def _check(self) -> DatasetImportParams:
        if self.source == "upload" and not self.upload_id:
            raise ValueError("upload_id is required for source=upload")
        if self.source == "path" and self.relpath is None:
            raise ValueError("relpath is required for source=path")
        return self


class DatasetSplitParams(_Params):
    dataset_id: str = Field(pattern=DATASET_ID)
    seed: int = Field(ge=0, lt=2**32)
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    dhash_threshold: int | None = Field(default=None, ge=0, le=64)
    low_info_std: float | None = Field(default=None, ge=0)
    low_info_popcount_min: int | None = Field(default=None, ge=0, le=64)
    low_info_popcount_max: int | None = Field(default=None, ge=0, le=64)
    conflict_policy: Literal["exclude", "error"] | None = None
    allow_missing_classes: bool = False


class DatasetVerifyParams(_Params):
    dataset_id: str = Field(pattern=DATASET_ID)
    full: bool = False


class TrainParams(_Params):
    config: TrainConfig


class EvaluateParams(_Params):
    experiment_id: str = Field(pattern=EXPERIMENT_ID)
    split: Literal["val", "test"]
    checkpoint: Literal["best", "last"] = "best"
    force: bool = False
    reason: str | None = Field(default=None, max_length=1000)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    bootstrap: int = Field(default=1000, ge=10, le=100_000)
    seed: int = Field(default=0, ge=0)


class LatencyParams(_Params):
    experiment_id: str = Field(pattern=EXPERIMENT_ID)
    device: Literal["cpu", "cuda"]
    warmup: int = Field(default=10, ge=0, le=1000)
    iterations: int = Field(default=50, ge=1, le=10_000)


PARAMS: dict[str, type[_Params]] = {
    "dataset_import": DatasetImportParams,
    "dataset_split": DatasetSplitParams,
    "dataset_verify": DatasetVerifyParams,
    "train": TrainParams,
    "evaluate": EvaluateParams,
    "latency": LatencyParams,
}


def validate_params(kind: str, params: dict) -> _Params:
    return PARAMS[kind].model_validate(params)
