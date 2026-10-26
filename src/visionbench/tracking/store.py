"""SQLite experiment store (SQLAlchemy) at ``<data_root>/db/visionbench.db``.

Schema is created with ``create_all``; a ``schema_version`` table records the version. The only
supported upgrade is 1 -> 2 (M3 added the ``evaluations`` table, which ``create_all`` creates);
any other stored version is an error. Status transitions are enforced:
pending -> running -> completed | failed | cancelled (pending may also go to failed/cancelled).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    literal_column,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from visionbench.core.errors import VisionBenchError

SCHEMA_VERSION = 2
_ADDITIVE_UPGRADES = {1}  # versions upgraded by create_all alone (new tables only)
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


class ExperimentStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path.as_posix()}")
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as s, s.begin():
            versions = s.scalars(select(SchemaVersion.version)).all()
            if not versions:
                s.add(SchemaVersion(version=SCHEMA_VERSION, applied_at=utcnow()))
            elif len(versions) == 1 and versions[0] in _ADDITIVE_UPGRADES:
                s.delete(s.get(SchemaVersion, versions[0]))
                s.add(SchemaVersion(version=SCHEMA_VERSION, applied_at=utcnow()))
            elif versions != [SCHEMA_VERSION]:
                raise TrackingError(
                    f"database schema version {versions} != supported {SCHEMA_VERSION}"
                )

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
    ) -> str:
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
            raise TrackingError(f"experiment {exp_id} not found")
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
                )
            )

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
