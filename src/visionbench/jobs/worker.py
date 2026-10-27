"""Single-worker job runner: claim -> run in a child process -> heartbeat/cancel -> recover.

- One worker per data root: an exclusive OS lock on ``db/worker.lock`` (``msvcrt.locking`` on
  Windows, ``fcntl.flock`` on POSIX) is held for the worker's whole lifetime. The OS drops it when
  the process dies, however it dies, so a crashed worker never blocks the next one. The file's
  content (pid, creation time) is informational only and never decides anything.
- Processes are identified by (pid, creation time) (:mod:`visionbench.core.process`); nothing is
  ever terminated, or treated as alive, on the strength of a pid alone.
- The oldest queued job is claimed atomically (:meth:`JobStore.claim`) and executed in a
  ``spawn`` child process, so a stuck job can be terminated without killing the worker.
- The worker writes the heartbeat while the child runs (informational: shown by the API). The
  child watches the worker's identity and exits immediately if the worker disappears, so a
  hard-killed worker leaves no job running.
- Cancellation is cooperative (training checks the flag via its step hook). If the child has
  not stopped ``cancel_timeout`` seconds after the request, it is terminated and both the job
  and its experiment are marked cancelled with a note.
- On startup, while holding the lock, :func:`recover` fails EVERY job left ``running`` (no other
  worker can exist, so all of them are orphaned) and every ``pending``/``running`` experiment
  whose recorded runner process is not alive, with the error "interrupted: ...".
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
from visionbench.core.process import (
    own_identity,
    process_create_time,
    process_is,
    terminate_process,
    wait_gone,
)
from visionbench.core.redact import scrub_paths
from visionbench.data.archive import cleanup_upload
from visionbench.jobs.handlers import HANDLERS, JobContext
from visionbench.jobs.params import validate_params
from visionbench.jobs.store import JobStore
from visionbench.tracking.store import DB_FILENAME, ExperimentStore, TrackingError
from visionbench.training.loop import RunCancelled

log = logging.getLogger("visionbench.worker")


class WorkerError(VisionBenchError):
    pass


@dataclass(frozen=True)
class WorkerConfig:
    poll_seconds: float = 0.5
    heartbeat_seconds: float = 2.0
    cancel_timeout_seconds: float = 30.0

    @classmethod
    def from_settings(cls, s: Settings) -> WorkerConfig:
        return cls(s.worker_poll_seconds, s.worker_heartbeat_seconds, s.cancel_timeout_seconds)


# --- worker lock (OS-level) ----------------------------------------------------------------

# msvcrt locks are mandatory byte-range locks: the locked byte sits far past the informational
# content so other processes can still read the file. Locking beyond end-of-file is allowed.
_LOCK_OFFSET = 1 << 30
LOCK_FILENAME = "worker.lock"


def _try_lock(fd: int) -> bool:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def read_lock_info(settings: Settings) -> dict[str, Any] | None:
    """The informational content of the lock file (who last held it), if readable."""
    try:
        text = (settings.data_dir("db") / LOCK_FILENAME).read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else None
    except (OSError, ValueError):
        return None


@dataclass
class WorkerLock:
    path: Path
    held: bool = True


@contextmanager
def worker_lock(settings: Settings) -> Iterator[WorkerLock]:
    """Hold the exclusive per-data-root worker lock; fail fast if another process holds it."""
    path = settings.data_dir("db") / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            info = read_lock_info(settings) or {}
            raise WorkerError(
                "another worker is running on this data root (the worker lock is held; "
                f"the lock file names pid {info.get('pid')})"
            )
        pid, create_time = own_identity()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({"pid": pid, "create_time": create_time}).encode() + b"\n")
        lock = WorkerLock(path)
        try:
            yield lock
        finally:
            lock.held = False
            _unlock(fd)
    finally:
        os.close(fd)


# --- recovery ------------------------------------------------------------------------------


def _close_experiment(settings: Settings, job: dict[str, Any], status: str, note: str) -> None:
    exp_id = (job.get("result") or {}).get("experiment_id")
    if not exp_id:
        return
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as es:
        if es.get(exp_id)["status"] in ("pending", "running"):
            es.transition(exp_id, status, error=note)


def _cleanup_job_temp(settings: Settings, job: dict[str, Any]) -> None:
    """Temp files a killed child could not remove (its ``finally`` never ran)."""
    upload_id = (job.get("params") or {}).get("upload_id")
    if job.get("kind") == "dataset_import" and upload_id:
        cleanup_upload(settings, upload_id)


def recover(settings: Settings, lock: WorkerLock) -> dict[str, list[dict[str, Any]]]:
    """Fail jobs and experiments orphaned by a dead worker/runner. Returns what was changed.

    Must be called while holding the worker lock: then no other worker exists, so every
    ``running`` job is orphaned, whatever its pids or heartbeat say.
    """
    if not lock.held:
        raise WorkerError("recover() requires the worker lock")
    report: dict[str, list[dict[str, Any]]] = {"jobs": [], "experiments": []}
    db = settings.data_dir("db") / DB_FILENAME
    with JobStore(db) as js:
        for job in js.list(status="running", limit=100_000):
            child, child_time = job["child_pid"], job["child_create_time"]
            if child is None:
                child_note = "no child process was started"
            elif terminate_process(child, child_time):  # normally it already exited by itself
                wait_gone(child, child_time, 10.0)
                child_note = f"child process {child} was still running and was terminated"
            else:
                child_note = f"child process {child} not running (pid gone or reused)"
            note = f"interrupted: worker {job['worker_pid']} is gone; {child_note}"
            # Close the job's own experiment FIRST, then fail the job: an observer that sees the
            # job failed sees the experiment closed.
            exp_id = (job.get("result") or {}).get("experiment_id")
            if exp_id:
                with ExperimentStore(db) as es:
                    if es.get(exp_id)["status"] in ("pending", "running"):
                        exp_note = f"{note} (job {job['id']})"
                        es.transition(exp_id, "failed", error=exp_note)
                        report["experiments"].append({"experiment_id": exp_id, "error": exp_note})
            _cleanup_job_temp(settings, job)
            js.finish(job["id"], "failed", error=note)
            report["jobs"].append({"job_id": job["id"], "kind": job["kind"], "error": note})
    with ExperimentStore(db) as es:
        for exp in es.list():
            if exp["status"] not in ("pending", "running"):
                continue
            if process_is(exp.get("runner_pid"), exp.get("runner_create_time")):
                continue  # e.g. a CLI `visionbench train` that is still running
            note = f"interrupted: runner process {exp.get('runner_pid')} not running"
            es.transition(exp["id"], "failed", error=note)
            report["experiments"].append({"experiment_id": exp["id"], "error": note})
    for item in report["jobs"]:
        log.warning("recovered job %s (%s): %s", item["job_id"], item["kind"], item["error"])
    for item in report["experiments"]:
        log.warning("recovered experiment %s: %s", item["experiment_id"], item["error"])
    return report


# --- execution (runs inside the child process, or inline in tests) -------------------------


def execute_job(job_id: str, settings: Settings) -> str:
    """Run the handler for a claimed job and record its outcome; returns the final status."""
    with JobStore(settings.data_dir("db") / DB_FILENAME) as store:
        job = store.get(job_id)
        try:
            params = validate_params(job["kind"], job["params"])
            handler = HANDLERS.get(job["kind"])
            if handler is None:
                raise WorkerError(f"no handler for job kind {job['kind']!r}")
            result = handler(params, JobContext(job_id, store, settings))
        except RunCancelled:
            store.finish(job_id, "cancelled", error="cancelled by request")
        except KeyboardInterrupt:
            store.finish(job_id, "cancelled", error="interrupted (KeyboardInterrupt)")
        except Exception as exc:  # recorded on the job; the child exits normally
            msg = scrub_paths(f"{type(exc).__name__}: {exc}", settings.redaction_roots())
            log.error("job %s failed: %s", job_id, msg)
            store.finish(job_id, "failed", error=msg)
        else:
            store.finish(job_id, "completed", result=result)
        return store.get(job_id)["status"]


def _watch_parent(parent_pid: int, parent_create_time: float) -> None:
    while True:
        time.sleep(1.0)
        if not process_is(parent_pid, parent_create_time):
            os._exit(3)  # worker is gone: stop now, leave the job for recovery


def _child_main(job_id: str, settings_payload: dict[str, Any], parent: tuple[int, float]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threading.Thread(target=_watch_parent, args=parent, daemon=True).start()
    execute_job(job_id, Settings(_env_file=None, **settings_payload))


# --- supervision (worker process) ----------------------------------------------------------


def run_job(store: JobStore, job: dict[str, Any], settings: Settings, wc: WorkerConfig) -> str:
    job_id = job["id"]
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_main,
        args=(job_id, settings.model_dump(mode="json"), own_identity()),
        name=f"visionbench-{job_id}",
    )
    proc.start()
    store.set_child(job_id, proc.pid, process_create_time(proc.pid))
    log.info("job %s (%s) started in child pid %s", job_id, job["kind"], proc.pid)
    cancel_seen: float | None = None
    try:
        while True:
            proc.join(timeout=wc.heartbeat_seconds)
            if not proc.is_alive():
                break
            store.heartbeat(job_id)
            if cancel_seen is None and store.is_cancel_requested(job_id):
                cancel_seen = time.monotonic()
                log.info("job %s: cancel requested", job_id)
            if (
                cancel_seen is not None
                and time.monotonic() - cancel_seen > wc.cancel_timeout_seconds
            ):
                # proc holds the child's process handle (Windows) / it is our unreaped child
                # (POSIX), so this cannot hit a reused pid.
                proc.terminate()
                proc.join(15)
                note = (
                    f"cancelled: child terminated after {wc.cancel_timeout_seconds:g} s "
                    "(cooperative cancel not honoured)"
                )
                _close_experiment(settings, store.get(job_id), "cancelled", note)
                _cleanup_job_temp(settings, job)
                store.finish(job_id, "cancelled", error=note)
                log.warning("job %s: %s", job_id, note)
                return "cancelled"
    except KeyboardInterrupt:
        proc.terminate()
        proc.join(15)
        note = "interrupted: worker stopped while the job was running"
        _close_experiment(settings, store.get(job_id), "failed", note)
        _cleanup_job_temp(settings, job)
        store.finish(job_id, "failed", error=note)
        raise
    final = store.get(job_id)
    if final["status"] == "running":  # child died without recording an outcome
        note = f"child process exited with code {proc.exitcode} before finishing"
        _close_experiment(settings, final, "failed", note)
        _cleanup_job_temp(settings, final)
        store.finish(job_id, "failed", error=note)
        final = store.get(job_id)
    log.info("job %s finished: %s", job_id, final["status"])
    return final["status"]


def run_worker(
    settings: Settings,
    wc: WorkerConfig | None = None,
    *,
    stop: threading.Event | None = None,
    max_jobs: int | None = None,
) -> None:
    """Run jobs one at a time until ``stop`` is set (or ``max_jobs`` have run)."""
    wc = wc or WorkerConfig.from_settings(settings)
    stop = stop or threading.Event()
    with worker_lock(settings) as lock:
        log.info("worker pid %s started on data root %s", os.getpid(), settings.resolved_data_root)
        recover(settings, lock)
        identity = own_identity()
        done = 0
        with JobStore(settings.data_dir("db") / DB_FILENAME) as store:
            while not stop.is_set():
                job = store.claim(*identity)
                if job is None:
                    stop.wait(wc.poll_seconds)
                    continue
                try:
                    run_job(store, job, settings, wc)
                except TrackingError as exc:  # keep the worker alive on bookkeeping races
                    log.error("job %s bookkeeping error: %s", job["id"], exc)
                done += 1
                if max_jobs is not None and done >= max_jobs:
                    break
