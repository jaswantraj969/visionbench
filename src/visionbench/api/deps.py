"""Shared helpers for API routers (settings, stores, id patterns, job enqueueing)."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from pydantic import BaseModel

from visionbench.core.config import Settings
from visionbench.jobs.params import DATASET_ID, EXPERIMENT_ID, SPLIT_ID
from visionbench.jobs.store import JobStore
from visionbench.tracking.store import DB_FILENAME, ExperimentStore

EVAL_ID = r"^ev-\d{8}-\d{6}-[0-9a-f]{6}$"
REPORT_ID = r"^rp-\d{8}-\d{6}-[0-9a-f]{6}$"
JOB_ID = r"^job-\d{8}-\d{6}-[0-9a-f]{6}$"
__all__ = ["DATASET_ID", "EVAL_ID", "EXPERIMENT_ID", "JOB_ID", "REPORT_ID", "SPLIT_ID"]

PUBLIC_JOB_FIELDS = (
    "id",
    "kind",
    "status",
    "params",
    "result",
    "error",
    "progress",
    "cancel_requested",
    "heartbeat_at",
    "created_at",
    "started_at",
    "finished_at",
)


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def experiment_store(settings: Settings) -> ExperimentStore:
    return ExperimentStore(settings.data_dir("db") / DB_FILENAME)


def job_store(settings: Settings) -> JobStore:
    return JobStore(settings.data_dir("db") / DB_FILENAME)


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: job[k] for k in PUBLIC_JOB_FIELDS}


def enqueue(settings: Settings, kind: str, params: BaseModel) -> dict[str, Any]:
    """Queue a job (the API never runs ML work itself); raises QueueFullError -> 429."""
    with job_store(settings) as js:
        job_id = js.enqueue(
            kind, params.model_dump(mode="json"), max_queued=settings.max_queued_jobs
        )
        return public_job(js.get(job_id))
