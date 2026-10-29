"""SQLite experiment store (SQLAlchemy) at ``<data_root>/db/visionbench.db``.

Schema is created with ``create_all``; a ``schema_version`` table records the version. All
upgrades so far are additive: new tables (``create_all``) and new nullable/defaulted columns
(``ALTER TABLE ... ADD COLUMN`` from :data:`_COLUMN_UPGRADES`). History: v1 (M2) experiments,
epoch_metrics, artifacts; v2 (M3) evaluations; v3 (M4 step 0) evaluations.forced/reason and
latency_measurements; v4 (M4) jobs table (``visionbench.jobs.store``) and
experiments.runner_pid; v5 (M4.1) process creation times next to every recorded pid
(jobs.worker_create_time/child_create_time, experiments.runner_create_time; see
:mod:`visionbench.core.process`); v6 (M5 step 0) experiments.stopped_reason. A stored version
newer than this code is an error. Status transitions are enforced: pending -> running ->
completed | failed | cancelled (pending may also go to failed/cancelled).
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    literal_column,
    select,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from visionbench.core.errors import VisionBenchError

SCHEMA_VERSION = 6
# Columns added to tables after they were first created (older DBs get ALTER TABLE ADD COLUMN).
_COLUMN_UPGRADES: dict[str, dict[str, str]] = {
    "evaluations": {"forced": "BOOLEAN NOT NULL DEFAULT 0", "reason": "TEXT"},
    "experiments": {
        "runner_pid": "INTEGER",
        "runner_create_time": "REAL",
        "stopped_reason": "VARCHAR(32)",
    },
    "jobs": {"worker_create_time": "REAL", "child_create_time": "REAL"},
}
DB_FILENAME = "visionbench.db"
STATUSES = ("pending", "running", "completed", "failed", "cancelled")
_TRANSITIONS = {
    "pending": {"running", "failed", "cancelled"},
    "running": {"completed", "failed", "cancelled"},
}


# Timestamps have 1 s resolution; ties are broken by SQLite insertion order so that
# "latest" is well defined (compare uses the latest evaluation).
_INSERTION_ORDER = literal_column("rowid")


class TrackingError(VisionBenchError):
    pass


class NotFoundError(TrackingError):
    pass


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[str] = mapped_column(String(32))


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[str | None] = mapped_column(String(32))
    finished_at: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    dataset_id: Mapped[str] = mapped_column(String(32))
    split_id: Mapped[str] = mapped_column(String(32))
    dataset_fingerprint: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(32))
    weights_source: Mapped[str | None] = mapped_column(String(16))
    weights_enum: Mapped[str | None] = mapped_column(String(64))
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[str] = mapped_column(Text)
    env_json: Mapped[str | None] = mapped_column(Text)
    seed: Mapped[int] = mapped_column(Integer)
    best_epoch: Mapped[int | None] = mapped_column(Integer)
    best_metric: Mapped[float | None] = mapped_column(Float)
    train_seconds: Mapped[float | None] = mapped_column(Float)
    run_dir: Mapped[str] = mapped_column(String(255))
    runner_pid: Mapped[int | None] = mapped_column(Integer)  # process running the training
    runner_create_time: Mapped[float | None] = mapped_column(Float)  # its identity (with pid)
    # Why training ended (early_stopping | epochs_completed | max_steps | baseline_fit); set when
    # the run completes, NULL for failed/cancelled runs.
    stopped_reason: Mapped[str | None] = mapped_column(String(32))


class EpochMetric(Base):
    __tablename__ = "epoch_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float | None] = mapped_column(Float)
    val_loss: Mapped[float] = mapped_column(Float)
    val_accuracy: Mapped[float] = mapped_column(Float)
    val_balanced_accuracy: Mapped[float] = mapped_column(Float)
    lr: Mapped[float | None] = mapped_column(Float)
    epoch_seconds: Mapped[float] = mapped_column(Float)
    steps: Mapped[int] = mapped_column(Integer)


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    relpath: Mapped[str] = mapped_column(String(255))  # relative to the data root, POSIX
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(32))


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    split: Mapped[str] = mapped_column(String(8))  # "val" | "test"
    checkpoint: Mapped[str] = mapped_column(String(16))  # "best" | "last" | "baseline"
    created_at: Mapped[str] = mapped_column(String(32))
    metrics_path: Mapped[str] = mapped_column(String(255))  # relative to the data root
    predictions_sha256: Mapped[str] = mapped_column(String(64))
    visionbench_version: Mapped[str] = mapped_column(String(32))
    forced: Mapped[bool] = mapped_column(Boolean, default=False)  # repeated test eval, forced
    reason: Mapped[str | None] = mapped_column(Text)  # required when forced


class LatencyMeasurement(Base):
    __tablename__ = "latency_measurements"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    checkpoint: Mapped[str] = mapped_column(String(16))
    device: Mapped[str] = mapped_column(String(8))  # "cpu" | "cuda"
    device_name: Mapped[str] = mapped_column(String(128))
    torch_threads: Mapped[int] = mapped_column(Integer)
    input_size: Mapped[int] = mapped_column(Integer)
    warmup: Mapped[int] = mapped_column(Integer)
    iterations: Mapped[int] = mapped_column(Integer)
    results_json: Mapped[str] = mapped_column(Text)  # {batch size: {median_ms, p95_ms, ...}}
    path: Mapped[str] = mapped_column(String(255))  # latency.json, relative to the data root
    created_at: Mapped[str] = mapped_column(String(32))
    visionbench_version: Mapped[str] = mapped_column(String(32))


_EPOCH_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "lr",
    "epoch_seconds",
    "steps",
)


def _row(obj: Base) -> dict[str, Any]:
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


def open_engine(db_path: Path) -> Engine:
    """Engine for the tracking DB with the schema created/upgraded to :data:`SCHEMA_VERSION`."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # The API process and the worker write concurrently: WAL lets readers proceed during a
    # write, and busy_timeout makes a writer wait for the lock instead of failing immediately.
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    # journal_mode=WAL is persistent in the database file, so it is set once here (with a short
    # retry) rather than on every connection: changing the journal mode needs a lock, and right
    # after a writer process was killed Windows can report a transient "disk I/O error".
    for attempt in range(5):
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            break
        except OperationalError:
            if attempt == 4:
                raise
            time.sleep(0.5)

    import visionbench.jobs.store  # noqa: F401  (registers the jobs table on Base.metadata)

    with engine.begin() as conn:
        versions = (
            [r[0] for r in conn.exec_driver_sql("SELECT version FROM schema_version")]
            if engine.dialect.has_table(conn, "schema_version")
            else []
        )
    if len(versions) > 1 or (versions and not 1 <= versions[0] <= SCHEMA_VERSION):
        engine.dispose()
        raise TrackingError(f"database schema version {versions} != supported {SCHEMA_VERSION}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:  # additive column upgrades for tables created by older code
        for table, columns in _COLUMN_UPGRADES.items():
            existing = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for column, ddl in columns.items():
                if column not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    with Session(engine) as s, s.begin():
        if versions != [SCHEMA_VERSION]:
            if versions:
                s.delete(s.get(SchemaVersion, versions[0]))
                s.flush()
            s.add(SchemaVersion(version=SCHEMA_VERSION, applied_at=utcnow()))
    return engine


class ExperimentStore:
    def __init__(self, db_path: Path) -> None:
        self.engine = open_engine(db_path)

    def close(self) -> None:
        self.engine.dispose()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes ---------------------------------------------------------------------------

    def create(
        self,
        *,
        dataset_id: str,
        split_id: str,
        dataset_fingerprint: str,
        model: str,
        seed: int,
        config: dict[str, Any],
        run_dir: str,
        runner: tuple[int, float] | None = None,
    ) -> str:
        """``runner`` is the (pid, creation time) of the process that will train it, recorded in
        the same transaction so crash recovery never sees a pending run without its owner."""
        exp_id = f"exp-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        with Session(self.engine) as s, s.begin():
            s.add(
                Experiment(
                    id=exp_id,
                    created_at=utcnow(),
                    status="pending",
                    dataset_id=dataset_id,
                    split_id=split_id,
                    dataset_fingerprint=dataset_fingerprint,
                    model=model,
                    seed=seed,
                    config_json=json.dumps(config, sort_keys=True),
                    run_dir=run_dir.format(exp_id=exp_id),
                    runner_pid=runner[0] if runner else None,
                    runner_create_time=runner[1] if runner else None,
                )
            )
        return exp_id

    def transition(self, exp_id: str, status: str, *, error: str | None = None) -> None:
        with Session(self.engine) as s, s.begin():
            exp = self._get(s, exp_id)
            if status not in _TRANSITIONS.get(exp.status, set()):
                raise TrackingError(f"illegal status transition {exp.status} -> {status}")
            exp.status = status
            if status == "running":
                exp.started_at = utcnow()
            else:
                exp.finished_at = utcnow()
            if error is not None:
                exp.error = error

    def update(self, exp_id: str, **fields: Any) -> None:
        allowed = {
            "weights_source",
            "weights_enum",
            "weights_sha256",
            "config_json",
            "env_json",
            "best_epoch",
            "best_metric",
            "train_seconds",
            "runner_pid",
            "runner_create_time",
            "stopped_reason",
        }
        if set(fields) - allowed:
            raise TrackingError(f"cannot update fields {sorted(set(fields) - allowed)}")
        with Session(self.engine) as s, s.begin():
            exp = self._get(s, exp_id)
            for k, v in fields.items():
                setattr(exp, k, v)

    def add_epoch(self, exp_id: str, record: dict[str, Any]) -> None:
        with Session(self.engine) as s, s.begin():
            s.add(EpochMetric(experiment_id=exp_id, **{k: record.get(k) for k in _EPOCH_FIELDS}))

    def add_artifact(
        self, exp_id: str, *, kind: str, relpath: str, sha256: str, size_bytes: int
    ) -> None:
        with Session(self.engine) as s, s.begin():
            s.add(
                Artifact(
                    experiment_id=exp_id,
                    kind=kind,
                    relpath=relpath,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    created_at=utcnow(),
                )
            )

    # --- reads ----------------------------------------------------------------------------

    @staticmethod
    def _get(s: Session, exp_id: str) -> Experiment:
        exp = s.get(Experiment, exp_id)
        if exp is None:
            raise NotFoundError(f"experiment {exp_id} not found")
        return exp

    def get(self, exp_id: str) -> dict[str, Any]:
        with Session(self.engine) as s:
            return _row(self._get(s, exp_id))

    def list(self) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Experiment).order_by(Experiment.created_at, _INSERTION_ORDER))
            return [_row(r) for r in rows]

    def epochs(self, exp_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            q = select(EpochMetric).where(EpochMetric.experiment_id == exp_id)
            return [_row(r) for r in s.scalars(q.order_by(EpochMetric.epoch))]

    def artifacts(self, exp_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            q = select(Artifact).where(Artifact.experiment_id == exp_id)
            return [_row(r) for r in s.scalars(q.order_by(Artifact.relpath))]

    # --- evaluations ----------------------------------------------------------------------

    def add_evaluation(
        self,
        *,
        eval_id: str,
        experiment_id: str,
        split: str,
        checkpoint: str,
        metrics_path: str,
        predictions_sha256: str,
        visionbench_version: str,
        forced: bool = False,
        reason: str | None = None,
    ) -> None:
        with Session(self.engine) as s, s.begin():
            self._get(s, experiment_id)
            s.add(
                Evaluation(
                    id=eval_id,
                    experiment_id=experiment_id,
                    split=split,
                    checkpoint=checkpoint,
                    created_at=utcnow(),
                    metrics_path=metrics_path,
                    predictions_sha256=predictions_sha256,
                    visionbench_version=visionbench_version,
                    forced=forced,
                    reason=reason,
                )
            )

    def get_evaluation(self, eval_id: str) -> dict[str, Any]:
        with Session(self.engine) as s:
            row = s.get(Evaluation, eval_id)
            if row is None:
                raise NotFoundError(f"evaluation {eval_id} not found")
            return _row(row)

    def evaluations(
        self,
        experiment_id: str | None = None,
        *,
        split: str | None = None,
        checkpoint: str | None = None,
    ) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            q = select(Evaluation)
            if experiment_id is not None:
                q = q.where(Evaluation.experiment_id == experiment_id)
            if split is not None:
                q = q.where(Evaluation.split == split)
            if checkpoint is not None:
                q = q.where(Evaluation.checkpoint == checkpoint)
            return [_row(r) for r in s.scalars(q.order_by(Evaluation.created_at, _INSERTION_ORDER))]

    # --- latency --------------------------------------------------------------------------

    def add_latency(self, **fields: Any) -> None:
        with Session(self.engine) as s, s.begin():
            self._get(s, fields["experiment_id"])
            s.add(LatencyMeasurement(created_at=utcnow(), **fields))

    def latencies(
        self, experiment_id: str | None = None, *, device: str | None = None
    ) -> list[dict[str, Any]]:
        with Session(self.engine) as s:
            q = select(LatencyMeasurement)
            if experiment_id is not None:
                q = q.where(LatencyMeasurement.experiment_id == experiment_id)
            if device is not None:
                q = q.where(LatencyMeasurement.device == device)
            order = (LatencyMeasurement.created_at, _INSERTION_ORDER)
            return [_row(r) for r in s.scalars(q.order_by(*order))]
