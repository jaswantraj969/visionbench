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


class RobustnessRequest(_Req):
    suite: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    split: Literal["val", "test"]
    clean_eval_id: str | None = Field(default=None, pattern=r"^ev-\d{8}-\d{6}-[0-9a-f]{6}$")
    device: Literal["auto", "cpu", "cuda"] = "auto"


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


# --- read-only views for the frontend (M7) -------------------------------------------------


class HashFile(BaseModel):
    name: str
    expected: str
    actual: str | None
    ok: bool


class HashStatus(BaseModel):
    present: bool
    ok: bool
    files: list[HashFile]


class BenchmarkListItem(BaseModel):
    name: str
    protocol_hashes: HashStatus
    robustness_hashes: HashStatus
    runs: int


class Overview(BaseModel):
    version: str
    datasets: int
    experiments: int
    experiments_by_status: dict[str, int]
    jobs_by_status: dict[str, int]
    test_images_unlocked: bool
    inference_device: str
    benchmarks: list[BenchmarkListItem]


class BenchmarkDetail(BaseModel):
    name: str
    protocol_hashes: HashStatus
    robustness_hashes: HashStatus
    final_analysis_plan_hashes: HashStatus
    documents: dict[str, bool]
    runs: list[dict[str, Any]]
    val_label: str
    val_summary: list[dict[str, Any]]
    robustness_val: dict[str, Any] | None
    final_report: dict[str, Any] | None


class BenchmarkDocument(BaseModel):
    benchmark: str
    document: str
    path: str
    markdown: str


class SuiteCondition(BaseModel):
    corruption: str
    parameter: str
    severity: str


class SuiteOut(BaseModel):
    name: str
    sha256: str
    base_seed: int
    conditions: list[SuiteCondition]


class SampleImage(BaseModel):
    relpath: str
    label: str
    split: str


class SplitSample(BaseModel):
    dataset_id: str
    split_id: str
    split: str
    seed: int
    per_class: int
    rule: str
    images: list[SampleImage]
