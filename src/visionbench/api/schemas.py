"""Pydantic request/response models of the HTTP API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from visionbench.api.deps import EVAL_ID, EXPERIMENT_ID


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class JobOut(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    params: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    progress: dict[str, Any] | None
    cancel_requested: bool
    heartbeat_at: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class JobAccepted(BaseModel):
    job: JobOut


class PathImportRequest(_Req):
    path: str = Field(min_length=1, max_length=1024, description="path below import_root")
    name: str = Field(min_length=1, max_length=100)


class SplitRequest(_Req):
    seed: int = Field(ge=0, lt=2**32)
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    dhash_threshold: int | None = Field(default=None, ge=0, le=64)
    low_info_std: float | None = Field(default=None, ge=0)
    low_info_popcount_min: int | None = Field(default=None, ge=0, le=64)
    low_info_popcount_max: int | None = Field(default=None, ge=0, le=64)
    conflict_policy: Literal["exclude", "error"] | None = None
    allow_missing_classes: bool = False


class VerifyRequest(_Req):
    full: bool = False


class EvaluationRequest(_Req):
    split: Literal["val", "test"]
    checkpoint: Literal["best", "last"] = "best"
    force: bool = False
    reason: str | None = Field(default=None, max_length=1000)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    bootstrap: int = Field(default=1000, ge=10, le=100_000)
    seed: int = Field(default=0, ge=0)


class LatencyRequest(_Req):
    device: Literal["cpu", "cuda"]
    warmup: int = Field(default=10, ge=0, le=1000)
    iterations: int = Field(default=50, ge=1, le=10_000)


ExperimentIdStr = Annotated[str, StringConstraints(pattern=EXPERIMENT_ID)]
EvalIdStr = Annotated[str, StringConstraints(pattern=EVAL_ID)]


class ComparisonRequest(_Req):
    experiment_ids: list[ExperimentIdStr] = Field(min_length=2, max_length=50)
    split: Literal["val", "test"]
    eval_ids: list[EvalIdStr] = Field(default_factory=list, max_length=50)
    reference: ExperimentIdStr | None = None
    latency_device: Literal["cpu", "cuda"] | None = None
    bootstrap: int = Field(default=1000, ge=10, le=100_000)
    seed: int = Field(default=0, ge=0)
    note: str | None = Field(default=None, max_length=500)


class DatasetSummary(BaseModel):
    id: str
    name: str
    created_at: str
    classes: list[str]
    class_counts: dict[str, int]
    total_images: int
    fingerprint: str
