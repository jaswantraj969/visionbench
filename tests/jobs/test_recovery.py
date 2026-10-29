"""Process identity, the OS worker lock and recovery under the lock (no worker process).

Every process these tests terminate was spawned by the test itself (the ``sleeper`` fixture).
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from visionbench.core.config import Settings
from visionbench.core.process import (
    own_identity,
    process_create_time,
    process_is,
    terminate_process,
)
from visionbench.jobs.store import JobStore
from visionbench.jobs.worker import WorkerError, WorkerLock, recover, worker_lock
from visionbench.tracking.store import DB_FILENAME, SCHEMA_VERSION, ExperimentStore

PARAMS = {"dataset_id": "ds-000000000000"}
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture
def sleeper():
    """Start live processes that are unrelated to VisionBench (and stop them afterwards)."""
    procs: list[subprocess.Popen] = []

    def start() -> tuple[subprocess.Popen, float]:
        proc = subprocess.Popen(SLEEP)
        procs.append(proc)
        create_time = process_create_time(proc.pid)
        assert create_time is not None
        return proc, create_time

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(30)


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_root=tmp_path / "var")


def _db(settings: Settings) -> Path:
    return settings.data_dir("db") / DB_FILENAME


def _dead_identity() -> tuple[int, float]:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    create_time = process_create_time(proc.pid) or 0.0
    proc.wait()
    return proc.pid, create_time


def _experiment(es: ExperimentStore, status: str, runner: tuple[int, float | None] | None) -> str:
    exp_id = es.create(
        dataset_id="ds",
        split_id="sp",
        dataset_fingerprint="f",
        model="m",
        seed=0,
        config={},
        run_dir="experiments/{exp_id}",
        runner=runner,  # type: ignore[arg-type]
    )
    if status == "running":
        es.transition(exp_id, "running")
    return exp_id


# --- process identity ----------------------------------------------------------------------


def test_process_identity_needs_pid_and_creation_time(sleeper) -> None:
    pid, create_time = own_identity()
    assert process_is(pid, create_time)
    assert not process_is(pid, create_time - 5.0)  # same pid, different process
    assert not process_is(pid, None)  # no recorded creation time never matches
    assert not process_is(None, create_time) and not process_is(0, create_time)
    assert not process_is(*_dead_identity())
    proc, real = sleeper()
    assert terminate_process(proc.pid, real - 5.0) is False  # pid matches, identity does not
    assert proc.poll() is None
    assert terminate_process(proc.pid, real) is True  # the recorded process itself
    proc.wait(30)


# --- recovery under the lock ---------------------------------------------------------------


def test_recovery_does_not_kill_a_reused_child_pid(tmp_path: Path, sleeper) -> None:
    """H1: the recorded child pid now belongs to an unrelated live process (pid reuse)."""
    settings = _settings(tmp_path)
    dummy, real = sleeper()
    with JobStore(_db(settings)) as s:
        job = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(*_dead_identity())
        s.set_child(job, dummy.pid, real - 100.0)  # identity of the earlier owner of the pid
    with worker_lock(settings) as lock:
        report = recover(settings, lock)
    with JobStore(_db(settings)) as s:
        recovered = s.get(job)
    assert recovered["status"] == "failed"
    assert recovered["error"].startswith("interrupted: worker ")
    assert f"child process {dummy.pid} not running" in recovered["error"]
    assert [j["job_id"] for j in report["jobs"]] == [job]
    assert dummy.poll() is None and process_is(dummy.pid, real)  # still alive, untouched


def test_recovery_terminates_a_child_whose_identity_matches(tmp_path: Path, sleeper) -> None:
    settings = _settings(tmp_path)
    child, real = sleeper()  # stands in for an orphaned child that is still running
    with JobStore(_db(settings)) as s:
        job = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(*_dead_identity())
        s.set_child(job, child.pid, real)
    with worker_lock(settings) as lock:
        recover(settings, lock)
    with JobStore(_db(settings)) as s:
        assert "was still running and was terminated" in s.get(job)["error"]
    assert child.wait(30) is not None and not process_is(child.pid, real)


def test_running_job_with_live_worker_pid_and_fresh_heartbeat_is_recovered(
    tmp_path: Path,
) -> None:
    """M1: under the lock no worker exists, so a live pid / fresh heartbeat proves nothing."""
    settings = _settings(tmp_path)
    with JobStore(_db(settings)) as s:
        job = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(os.getpid(), own_identity()[1])  # this live process; heartbeat set just now
        no_child = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(os.getpid())
        s.heartbeat(job)
    with worker_lock(settings) as lock:
        report = recover(settings, lock)
    with JobStore(_db(settings)) as s:
        assert s.get(job)["status"] == "failed"
        assert s.get(job)["error"].startswith(f"interrupted: worker {os.getpid()} is gone")
        assert "no child process was started" in s.get(no_child)["error"]
        assert s.count("running") == 0
    assert {j["job_id"] for j in report["jobs"]} == {job, no_child}


def test_recovery_closes_the_jobs_experiment_first(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with ExperimentStore(_db(settings)) as es:
        exp = _experiment(es, "running", own_identity())  # live runner, but owned by the job
    with JobStore(_db(settings)) as s:
        job = s.enqueue("train", {"config": {}}, max_queued=10)
        s.claim(*_dead_identity())
        s.merge_result(job, {"experiment_id": exp})
    with worker_lock(settings) as lock:
        recover(settings, lock)
    with ExperimentStore(_db(settings)) as es:
        assert es.get(exp)["status"] == "failed"
        assert es.get(exp)["error"].endswith(f"(job {job})")


def test_orphaned_experiments_use_runner_identity(tmp_path: Path, sleeper) -> None:
    settings = _settings(tmp_path)
    dummy, real = sleeper()
    with ExperimentStore(_db(settings)) as es:
        reused = _experiment(es, "running", (dummy.pid, real - 100.0))  # pid reused by dummy
        live = _experiment(es, "running", (dummy.pid, real))  # e.g. a live CLI train
        mine = _experiment(es, "running", own_identity())
        dead = _experiment(es, "running", _dead_identity())
        legacy = _experiment(es, "running", (dummy.pid, None))  # pre-v5 row: pid only
        pending = _experiment(es, "pending", None)
    with worker_lock(settings) as lock:
        report = recover(settings, lock)
    with ExperimentStore(_db(settings)) as es:
        status = {e: es.get(e)["status"] for e in (reused, live, mine, dead, legacy, pending)}
        assert es.get(reused)["error"] == f"interrupted: runner process {dummy.pid} not running"
    assert status == {
        reused: "failed",
        live: "running",
        mine: "running",
        dead: "failed",
        legacy: "failed",
        pending: "failed",
    }
    assert {e["experiment_id"] for e in report["experiments"]} == {reused, dead, legacy, pending}
    assert dummy.poll() is None  # never touched


def test_recover_requires_the_lock(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(WorkerError, match="requires the worker lock"):
        recover(settings, WorkerLock(tmp_path / "x", held=False))
    with worker_lock(settings) as lock:
        pass
    with pytest.raises(WorkerError, match="requires the worker lock"):
        recover(settings, lock)  # released


# --- OS lock -------------------------------------------------------------------------------


def test_lock_file_naming_a_live_unrelated_pid_does_not_block(tmp_path: Path, sleeper) -> None:
    """M1: the file content is informational; only the OS lock counts."""
    settings = _settings(tmp_path)
    dummy, real = sleeper()
    lock_file = settings.data_dir("db") / "worker.lock"
    for content in (str(dummy.pid), json.dumps({"pid": dummy.pid, "create_time": real})):
        lock_file.write_text(content, encoding="utf-8")  # M4 format, then the M4.1 format
        with worker_lock(settings):
            info = json.loads(lock_file.read_text(encoding="utf-8"))
            assert (info["pid"], info["create_time"]) == own_identity()
    assert dummy.poll() is None


def test_second_lock_holder_fails_fast(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with (
        worker_lock(settings),
        pytest.raises(WorkerError, match=f"another worker .* names pid {os.getpid()}"),
        worker_lock(settings),
    ):
        pass
    with worker_lock(settings):  # released on exit, so it can be taken again
        pass


def test_schema_upgrade_from_v4_adds_creation_time_columns(tmp_path: Path) -> None:
    db = tmp_path / DB_FILENAME
    with JobStore(db) as s:  # create, then strip the v5 columns to get an M4 (v4) database
        job = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(4242)
    con = sqlite3.connect(db)
    con.executescript(
        "ALTER TABLE jobs DROP COLUMN worker_create_time;"
        "ALTER TABLE jobs DROP COLUMN child_create_time;"
        "ALTER TABLE experiments DROP COLUMN runner_create_time;"
        "ALTER TABLE experiments DROP COLUMN stopped_reason;"
        "UPDATE schema_version SET version = 4;"
    )
    con.close()
    with JobStore(db) as s:
        row = s.get(job)
    assert row["worker_pid"] == 4242 and row["worker_create_time"] is None
    con = sqlite3.connect(db)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(SCHEMA_VERSION,)]
    columns = {r[1] for r in con.execute("PRAGMA table_info(experiments)")}
    con.close()
    assert {"runner_create_time", "stopped_reason"} <= columns
