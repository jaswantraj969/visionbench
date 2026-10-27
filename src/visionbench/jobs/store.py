"""Job store: the ``jobs`` table (tracking DB, schema v4) and its state machine.

Lifecycle: queued -> running -> completed | failed | cancelled; a queued job can also be
cancelled directly. The API process enqueues and reads; the single worker claims, heartbeats,
reports progress and finishes. Both write to the same SQLite file (WAL + busy_timeout, set in
:func:`visionbench.tracking.store.open_engine`).

Atomicity:
- enqueue: one ``INSERT ... SELECT ... WHERE (queued count) < limit`` statement, so the queue
  limit cannot be exceeded by concurrent requests.
- claim: one ``UPDATE ... WHERE id = (oldest queued) AND status = 'queued' RETURNING id``
  statement, so a job is claimed by at most one worker.
- finish: conditional on ``status = 'running'`` so a late child cannot overwrite a job that the
  worker already marked cancelled/failed.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Integer, String, Text, select, text
from sqlalchemy.orm import Mapped, Session, mapped_column

from visionbench.core.errors import VisionBenchError
from visionbench.tracking.store import _INSERTION_ORDER, Base, TrackingError, open_engine, utcnow

JOB_KINDS = (
    "dataset_import",
    "dataset_split",
    "dataset_verify",
    "train",
    "evaluate",
    "latency",
)
JOB_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
TERMINAL = ("completed", "failed", "cancelled")


class QueueFullError(VisionBenchError):
    pass


class JobNotFoundError(TrackingError):
    pass


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True)
    params_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)  # ids of created objects
    error: Mapped[str | None] = mapped_column(Text)  # "<Type>: <message>", no absolute paths
    progress_json: Mapped[str | None] = mapped_column(Text)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    worker_pid: Mapped[int | None] = mapped_column(Integer)
    child_pid: Mapped[int | None] = mapped_column(Integer)
    heartbeat_at: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[str | None] = mapped_column(String(32))
    finished_at: Mapped[str | None] = mapped_column(String(32))


def _public(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "params": json.loads(job.params_json),
        "result": json.loads(job.result_json) if job.result_json else None,
        "error": job.error,
        "progress": json.loads(job.progress_json) if job.progress_json else None,
        "cancel_requested": bool(job.cancel_requested),
        "worker_pid": job.worker_pid,
        "child_pid": job.child_pid,
        "heartbeat_at": job.heartbeat_at,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def new_job_id() -> str:
    return f"job-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.engine = open_engine(db_path)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- API side -------------------------------------------------------------------------

    def enqueue(self, kind: str, params: dict[str, Any], *, max_queued: int) -> str:
        if kind not in JOB_KINDS:
            raise TrackingError(f"unknown job kind {kind!r}")
        job_id = new_job_id()
        with self.engine.begin() as conn:
            inserted = conn.execute(
                text(
                    "INSERT INTO jobs (id, kind, status, params_json, cancel_requested, created_at)"
                    " SELECT :id, :kind, 'queued', :params, 0, :now"
                    " WHERE (SELECT COUNT(*) FROM jobs WHERE status = 'queued') < :limit"
                ),
                {
                    "id": job_id,
                    "kind": kind,
                    "params": json.dumps(params, sort_keys=True),
                    "now": utcnow(),
                    "limit": max_queued,
                },
            ).rowcount
        if inserted != 1:
            raise QueueFullError(f"queue is full ({max_queued} queued jobs); try again later")
        return job_id

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        """Queued -> cancelled immediately; running -> cancel_requested (cooperative, the worker
        terminates the child after a timeout); terminal jobs are returned unchanged."""
        with Session(self.engine) as s, s.begin():
            job = self._get(s, job_id)
            if job.status == "queued":
                job.status, job.finished_at = "cancelled", utcnow()
                job.error = "cancelled before start"
            elif job.status == "running":
                job.cancel_requested = True
            return _public(job)

    # --- worker side ----------------------------------------------------------------------

    def claim(self, worker_pid: int) -> dict[str, Any] | None:
        """Atomically move the oldest queued job to running; ``None`` if the queue is empty."""
        now = utcnow()
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    "UPDATE jobs SET status = 'running', started_at = :now, heartbeat_at = :now,"
                    " worker_pid = :pid"
                    " WHERE id = (SELECT id FROM jobs WHERE status = 'queued'"
                    "             ORDER BY created_at, rowid LIMIT 1)"
                    " AND status = 'queued' RETURNING id"
                ),
                {"now": now, "pid": worker_pid},
            ).first()
        return self.get(row[0]) if row else None

    def _update(self, job_id: str, **fields: Any) -> None:
        with Session(self.engine) as s, s.begin():
            job = self._get(s, job_id)
            for k, v in fields.items():
                setattr(job, k, v)

    def set_child_pid(self, job_id: str, pid: int) -> None:
        self._update(job_id, child_pid=pid)

    def heartbeat(self, job_id: str) -> None:
        self._update(job_id, heartbeat_at=utcnow())

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        self._update(job_id, progress_json=json.dumps(progress, sort_keys=True))

    def merge_result(self, job_id: str, result: dict[str, Any]) -> None:
        with Session(self.engine) as s, s.begin():
            job = self._get(s, job_id)
            merged = (json.loads(job.result_json) if job.result_json else {}) | result
            job.result_json = json.dumps(merged, sort_keys=True)

    def is_cancel_requested(self, job_id: str) -> bool:
        with Session(self.engine) as s:
            return bool(self._get(s, job_id).cancel_requested)

    def finish(
        self,
        job_id: str,
        status: str,
        *,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """running -> completed|failed|cancelled. Returns False (no change) if the job is no
        longer running, e.g. the worker already marked it after terminating the child."""
        if status not in TERMINAL:
            raise TrackingError(f"finish() needs a terminal status, got {status!r}")
        with Session(self.engine) as s, s.begin():
            job = self._get(s, job_id)
            if job.status != "running":
                return False
            job.status, job.finished_at = status, utcnow()
            if error is not None:
                job.error = error
            if result is not None:
                merged = (json.loads(job.result_json) if job.result_json else {}) | result
                job.result_json = json.dumps(merged, sort_keys=True)
            return True

    # --- reads ----------------------------------------------------------------------------

    @staticmethod
    def _get(s: Session, job_id: str) -> Job:
        job = s.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(f"job {job_id} not found")
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        with Session(self.engine) as s:
            return _public(self._get(s, job_id))

    def list(self, *, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            q = select(Job)
            if status is not None:
                q = q.where(Job.status == status)
            q = q.order_by(Job.created_at.desc(), _INSERTION_ORDER.desc()).limit(limit)
            return [_public(j) for j in s.scalars(q)]

    def count(self, status: str) -> int:
        with self.engine.connect() as conn:
            return int(
                conn.execute(
                    text("SELECT COUNT(*) FROM jobs WHERE status = :s"), {"s": status}
                ).scalar_one()
            )
