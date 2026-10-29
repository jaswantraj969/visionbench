"""Real `visionbench worker` subprocess: progress, one-at-a-time, cancel, kill + recovery, lock.

The only processes these tests kill are worker processes they started themselves.
"""

import re
import subprocess
import sys
import time

import pytest
from vb_worker import WorkerProcess, get_job, job_store, wait_for

from visionbench.core.process import process_is
from visionbench.jobs.handlers import TEST_IGNORE_CANCEL_ENV
from visionbench.jobs.worker import read_lock_info
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.runner import run_training

JOB_TIMEOUT = 180  # generous: each job spawns a fresh Python + torch import


@pytest.fixture
def worker(train_env, tmp_path):
    procs: list[WorkerProcess] = []

    def start(max_jobs: int | None = None, **env: str) -> WorkerProcess:
        log = tmp_path / f"worker{len(procs)}.log"
        w = WorkerProcess(train_env.settings, log, max_jobs=max_jobs, **env)
        procs.append(w)
        return w

    yield start
    for w in procs:
        w.stop()


def _train_params(env, **overrides) -> dict:
    return {"config": env.config(**overrides).model_dump(mode="json")}


def _experiment(env, exp_id: str) -> dict:
    with ExperimentStore(env.settings.data_dir("db") / DB_FILENAME) as es:
        return es.get(exp_id) | {"epochs": es.epochs(exp_id)}


def test_jobs_run_one_at_a_time_with_progress(train_env, worker) -> None:
    with job_store(train_env.settings) as s:
        first = s.enqueue("train", _train_params(train_env, epochs=3), max_queued=10)
        second = s.enqueue("dataset_verify", {"dataset_id": train_env.dataset_id}, max_queued=10)
    worker(max_jobs=2)
    seen_progress, max_running = [], 0
    while get_job(train_env.settings, second)["status"] not in ("completed", "failed"):
        with job_store(train_env.settings) as s:
            max_running = max(max_running, s.count("running"))
            job = s.get(first)
        if job["status"] == "running" and job["progress"]:
            seen_progress.append(job["progress"])
        time.sleep(0.1)  # polling interval
    done, verify = get_job(train_env.settings, first), get_job(train_env.settings, second)
    assert done["status"] == "completed" and verify["status"] == "completed"
    assert max_running <= 1  # never two jobs at once
    assert verify["started_at"] >= done["finished_at"]  # strictly after the first job
    assert any(p.get("phase") == "training" for p in seen_progress)
    assert done["progress"]["epoch"] == 3 and done["progress"]["latest"]["val_loss"] > 0
    exp = _experiment(train_env, done["result"]["experiment_id"])
    assert exp["status"] == "completed" and len(exp["epochs"]) == 3


def test_cooperative_cancel_keeps_partial_history(train_env, worker) -> None:
    with job_store(train_env.settings) as s:
        job_id = s.enqueue("train", _train_params(train_env, epochs=500), max_queued=10)
    worker(max_jobs=1)
    wait_for(
        lambda: (get_job(train_env.settings, job_id)["progress"] or {}).get("epoch", 0) >= 1,
        JOB_TIMEOUT,
        what="first epoch",
    )
    with job_store(train_env.settings) as s:
        assert s.request_cancel(job_id)["cancel_requested"] is True
    job = wait_for(
        lambda: (j := get_job(train_env.settings, job_id))["status"] != "running" and j,
        60,
        what="cancel",
    )
    assert job["status"] == "cancelled" and job["error"] == "cancelled by request"
    exp = _experiment(train_env, job["result"]["experiment_id"])
    assert exp["status"] == "cancelled" and exp["error"] == "cancelled by request"
    assert 1 <= len(exp["epochs"]) < 500


def test_cancel_timeout_terminates_child(train_env, worker) -> None:
    exp_id = run_training(train_env.config(epochs=1), train_env.settings).experiment_id
    with job_store(train_env.settings) as s:  # latency ignores cancel: forces termination
        job_id = s.enqueue(
            "latency",
            {"experiment_id": exp_id, "device": "cpu", "iterations": 10_000_000},
            max_queued=10,
        )
    worker(max_jobs=1, VISIONBENCH_CANCEL_TIMEOUT_SECONDS="1")
    job = wait_for(
        lambda: (j := get_job(train_env.settings, job_id))["child_pid"] and j,
        JOB_TIMEOUT,
        what="child start",
    )
    child = (job["child_pid"], job["child_create_time"])
    with job_store(train_env.settings) as s:
        s.request_cancel(job_id)
    job = wait_for(
        lambda: (j := get_job(train_env.settings, job_id))["status"] != "running" and j,
        60,
        what="termination",
    )
    assert job["status"] == "cancelled" and "child terminated after 1 s" in job["error"]
    assert not process_is(*child)


def test_cancel_timeout_terminates_training_child(train_env, worker) -> None:
    """Training normally honours cancel; the test-only hook makes it ignore the request."""
    with job_store(train_env.settings) as s:
        # early stopping off: on a fast machine it would end the tiny run before the cancel
        params = _train_params(train_env, epochs=500, early_stopping={"enabled": False})
        job_id = s.enqueue("train", params, max_queued=10)
    worker(max_jobs=1, VISIONBENCH_CANCEL_TIMEOUT_SECONDS="1", **{TEST_IGNORE_CANCEL_ENV: "1"})
    job = wait_for(
        lambda: (
            ((j := get_job(train_env.settings, job_id))["progress"] or {}).get("epoch", 0) >= 1
            and j
        ),
        JOB_TIMEOUT,
        what="first epoch",
    )
    child, exp_id = (job["child_pid"], job["child_create_time"]), job["result"]["experiment_id"]
    with job_store(train_env.settings) as s:
        s.request_cancel(job_id)
    job = wait_for(
        lambda: (j := get_job(train_env.settings, job_id))["status"] != "running" and j,
        60,
        what="termination",
    )
    note = "cancelled: child terminated after 1 s (cooperative cancel not honoured)"
    assert job["status"] == "cancelled" and job["error"] == note
    exp = _experiment(train_env, exp_id)
    assert exp["status"] == "cancelled" and exp["error"] == note
    assert len(exp["epochs"]) >= 1  # history recorded before the termination is kept
    assert not process_is(*child)


def test_hard_kill_then_recovery_marks_interrupted(train_env, worker) -> None:
    with job_store(train_env.settings) as s:
        job_id = s.enqueue("train", _train_params(train_env, epochs=500), max_queued=10)
    w = worker()
    wait_for(
        lambda: (get_job(train_env.settings, job_id)["progress"] or {}).get("epoch", 0) >= 1,
        JOB_TIMEOUT,
        what="first epoch",
    )
    job = get_job(train_env.settings, job_id)
    exp_id, child = job["result"]["experiment_id"], (job["child_pid"], job["child_create_time"])
    w.kill()  # hard kill: no cleanup in the worker
    wait_for(lambda: not process_is(*child), 30, what="child exit via parent watchdog")
    assert get_job(train_env.settings, job_id)["status"] == "running"  # nothing cleaned up yet
    assert _experiment(train_env, exp_id)["status"] == "running"

    w2 = worker(max_jobs=1)  # restarted at once: no heartbeat age has to pass
    job = wait_for(
        lambda: (j := get_job(train_env.settings, job_id))["status"] != "running" and j,
        60,
        what="recovery",
    )
    assert job["status"] == "failed"
    assert job["error"].startswith(f"interrupted: worker {job['worker_pid']} is gone")
    exp = _experiment(train_env, exp_id)
    assert exp["status"] == "failed" and exp["error"].startswith("interrupted")
    wait_for(lambda: "recovered job" in w2.log(), 10, what="recovery log line")
    with job_store(train_env.settings) as s:  # one job so that w2 exits by itself
        s.enqueue("dataset_verify", {"dataset_id": train_env.dataset_id}, max_queued=10)


def _started_pid(w: WorkerProcess) -> int:
    match = wait_for(
        lambda: re.search(r"worker pid (\d+) started", w.log()), 60, what="worker start"
    )
    return int(match.group(1))


def test_worker_lock_second_worker_fails_fast_and_hard_kill_releases(train_env, worker) -> None:
    first = worker()
    first_pid = _started_pid(first)
    assert read_lock_info(train_env.settings)["pid"] == first_pid
    t0 = time.monotonic()
    second = subprocess.run(
        [sys.executable, "-m", "visionbench.cli", "worker"],
        env=first.env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.returncode == 1
    assert "another worker is running on this data root" in second.stderr
    assert f"names pid {first_pid}" in second.stderr
    assert time.monotonic() - t0 < 60  # no waiting for the lock: start-up time only
    first.kill()  # hard kill: the OS releases the lock; the file still names first_pid
    assert read_lock_info(train_env.settings)["pid"] == first_pid
    third = worker()
    third_pid = _started_pid(third)
    assert third_pid != first_pid and read_lock_info(train_env.settings)["pid"] == third_pid
