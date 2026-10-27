"""Job store and worker bookkeeping (no worker process)."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from visionbench.core.config import Settings
from visionbench.jobs import worker as worker_mod
from visionbench.jobs.store import JobStore, QueueFullError
from visionbench.jobs.worker import WorkerError, execute_job, pid_alive, recover, worker_lock
from visionbench.tracking.store import DB_FILENAME, ExperimentStore, TrackingError

PARAMS = {"dataset_id": "ds-000000000000"}


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / DB_FILENAME)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not pid_alive(proc.pid)
    return proc.pid


def test_lifecycle_and_transitions(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        a = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        assert s.get(a)["status"] == "queued" and s.get(a)["params"] == PARAMS
        claimed = s.claim(worker_pid=4242)
        assert claimed["id"] == a and claimed["status"] == "running"
        assert claimed["worker_pid"] == 4242 and claimed["heartbeat_at"] is not None
        assert s.claim(worker_pid=4242) is None  # queue empty
        s.update_progress(a, {"epoch": 1})
        s.merge_result(a, {"experiment_id": "exp-x"})
        assert s.finish(a, "completed", result={"ok": True})
        done = s.get(a)
        assert done["status"] == "completed" and done["result"] == {
            "experiment_id": "exp-x",
            "ok": True,
        }
        assert done["progress"] == {"epoch": 1} and done["finished_at"] is not None
        assert s.finish(a, "failed", error="late") is False  # terminal: unchanged
        assert s.get(a)["status"] == "completed"
        with pytest.raises(TrackingError):
            s.finish(a, "running")
        with pytest.raises(TrackingError):
            s.enqueue("bogus", {}, max_queued=10)


def test_claim_order_is_fifo(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        ids = [s.enqueue("dataset_verify", PARAMS, max_queued=10) for _ in range(3)]
        assert [s.claim(1)["id"] for _ in range(3)] == ids


def test_queue_limit(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        s.enqueue("dataset_verify", PARAMS, max_queued=2)
        s.enqueue("dataset_verify", PARAMS, max_queued=2)
        with pytest.raises(QueueFullError, match="queue is full"):
            s.enqueue("dataset_verify", PARAMS, max_queued=2)
        s.claim(1)  # one leaves the queue
        s.enqueue("dataset_verify", PARAMS, max_queued=2)
        assert s.count("queued") == 2 and s.count("running") == 1


def test_cancel_flag(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        q = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        r = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        assert s.request_cancel(q)["status"] == "cancelled"  # queued: cancelled at once
        assert s.claim(1)["id"] == r  # the cancelled job is never claimed
        assert s.is_cancel_requested(r) is False
        out = s.request_cancel(r)
        assert out["status"] == "running" and out["cancel_requested"] is True
        assert s.is_cancel_requested(r)
        s.finish(r, "cancelled", error="cancelled by request")
        assert s.request_cancel(r)["status"] == "cancelled"  # terminal: unchanged


def test_atomic_claim_under_contention(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        job = s.enqueue("dataset_verify", PARAMS, max_queued=10)
    results: list = []
    barrier = threading.Barrier(8)

    def claimer(pid: int) -> None:
        with _store(tmp_path) as own:
            barrier.wait()
            results.append(own.claim(pid))

    threads = [threading.Thread(target=claimer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if r is not None]
    assert len(results) == 8 and len(winners) == 1 and winners[0]["id"] == job


def test_concurrent_enqueue_and_worker_updates(tmp_path: Path) -> None:
    with _store(tmp_path) as s:
        running = s.enqueue("train", {"config": {}}, max_queued=1000)
        s.claim(1)
    errors: list[BaseException] = []

    def api_side() -> None:
        try:
            with _store(tmp_path) as own:
                for _ in range(60):
                    own.enqueue("dataset_verify", PARAMS, max_queued=1000)
        except BaseException as exc:
            errors.append(exc)

    def worker_side() -> None:
        try:
            with _store(tmp_path) as own:
                for i in range(120):
                    own.heartbeat(running)
                    own.update_progress(running, {"step": i})
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=api_side), threading.Thread(target=worker_side)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with _store(tmp_path) as s:
        assert s.count("queued") == 60 and s.get(running)["progress"] == {"step": 119}


def _settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_root=tmp_path / "var")


def test_recover_stale_and_dead_jobs(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    db = settings.data_dir("db") / DB_FILENAME
    with JobStore(db) as s:
        dead = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        stale = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        healthy = s.enqueue("dataset_verify", PARAMS, max_queued=10)
        s.claim(_dead_pid())  # worker pid that no longer exists
        s.claim(os.getpid())
        s.claim(os.getpid())
        s._update(stale, heartbeat_at="2000-01-01T00:00:00+00:00")
    report = recover(settings, stale_after_seconds=30)
    with JobStore(db) as s:
        assert s.get(dead)["status"] == "failed" and "not alive" in s.get(dead)["error"]
        assert s.get(stale)["status"] == "failed" and "heartbeat stale" in s.get(stale)["error"]
        assert s.get(healthy)["status"] == "running"
        assert all(s.get(j)["error"].startswith("interrupted") for j in (dead, stale))
    assert {j["job_id"] for j in report["jobs"]} == {dead, stale}


def test_recover_orphaned_experiments(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    db = settings.data_dir("db") / DB_FILENAME
    with ExperimentStore(db) as es:
        ids = []
        for pid in (_dead_pid(), os.getpid(), None):
            e = es.create(
                dataset_id="ds",
                split_id="sp",
                dataset_fingerprint="f",
                model="m",
                seed=0,
                config={},
                run_dir="experiments/{exp_id}",
            )
            if pid:
                es.update(e, runner_pid=pid)
            ids.append(e)
        es.transition(ids[0], "running")
        es.transition(ids[1], "running")  # alive runner (this process): left alone
    report = recover(settings, stale_after_seconds=30)
    with ExperimentStore(db) as es:
        assert es.get(ids[0])["status"] == "failed"
        assert es.get(ids[0])["error"].startswith("interrupted")
        assert es.get(ids[1])["status"] == "running"
        assert es.get(ids[2])["status"] == "failed"  # pending with no runner process
    assert {e["experiment_id"] for e in report["experiments"]} == {ids[0], ids[2]}


def test_worker_lock(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    lock = settings.data_dir("db") / "worker.lock"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock.write_text(str(sleeper.pid), encoding="utf-8")
        with pytest.raises(WorkerError, match="another worker"), worker_lock(settings):
            pass
    finally:
        sleeper.kill()
        sleeper.wait()
    lock.write_text(str(_dead_pid()), encoding="utf-8")  # stale lock is taken over
    with worker_lock(settings):
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
    assert not lock.exists()


def test_execute_job_inline_success_and_failure(train_env) -> None:
    settings = train_env.settings
    db = settings.data_dir("db") / DB_FILENAME
    with JobStore(db) as s:
        ok = s.enqueue("dataset_verify", {"dataset_id": train_env.dataset_id}, max_queued=10)
        bad = s.enqueue("dataset_verify", {"dataset_id": "ds-0123456789ab"}, max_queued=10)
        invalid = s.enqueue("dataset_verify", {"dataset_id": "../etc"}, max_queued=10)
        for _ in range(3):
            s.claim(1)
    assert execute_job(ok, settings) == "completed"
    assert execute_job(bad, settings) == "failed"
    assert execute_job(invalid, settings) == "failed"
    with JobStore(db) as s:
        assert s.get(ok)["result"]["ok"] is True and s.get(ok)["result"]["files_checked"] == 36
        assert s.get(bad)["error"] == "DatasetNotFoundError: dataset ds-0123456789ab not found"
        assert s.get(invalid)["error"].startswith("ValidationError")
        root = str(settings.resolved_data_root)
        assert all(root.lower() not in (s.get(j)["error"] or "").lower() for j in (bad, invalid))


def test_pid_alive_self_and_dead() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(_dead_pid()) and not pid_alive(None) and not pid_alive(0)
    assert worker_mod.terminate_pid is not None
