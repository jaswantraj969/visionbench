"""Single-worker job runner: claim -> run in a child process -> heartbeat/cancel -> recover.

- One worker per data root (lock file ``db/worker.lock``); it runs exactly one job at a time.
- The oldest queued job is claimed atomically (:meth:`JobStore.claim`) and executed in a
  ``spawn`` child process, so a stuck job can be terminated without killing the worker.
- The worker writes the heartbeat while the child runs. The child watches the worker's pid and
  exits immediately if the worker disappears, so a hard-killed worker leaves no job running.
- Cancellation is cooperative (training checks the flag via its step hook). If the child has
  not stopped ``cancel_timeout`` seconds after the request, it is terminated and both the job
  and its experiment are marked cancelled with a note.
- On startup, :func:`recover` fails jobs left ``running`` by a dead worker (pid not alive or
  heartbeat stale) and experiments left ``pending``/``running`` with no live runner process,
  with the error "interrupted: ...".
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
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
    stale_after_seconds: float = 30.0
    cancel_timeout_seconds: float = 30.0

    @classmethod
    def from_settings(cls, s: Settings) -> WorkerConfig:
        return cls(
            s.worker_poll_seconds,
            s.worker_heartbeat_seconds,
            s.heartbeat_stale_seconds,
            s.cancel_timeout_seconds,
        )


# --- process helpers (no third-party dependency; os.kill(pid, 0) is unsafe on Windows) ------


def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_pid(pid: int) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = k32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
        if handle:
            k32.TerminateProcess(handle, 1)
            k32.CloseHandle(handle)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _wait_dead(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    return not pid_alive(pid)


@contextmanager
def worker_lock(settings: Settings) -> Iterator[None]:
    """Exclusive per-data-root worker lock (stale locks of dead pids are taken over)."""
    path = settings.data_dir("db") / "worker.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                other = int(path.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                other = 0
            if other and other != os.getpid() and pid_alive(other):
                raise WorkerError(
                    f"another worker (pid {other}) is running on this data root"
                ) from None
            path.unlink(missing_ok=True)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        break
    else:
        raise WorkerError("could not acquire the worker lock")
    try:
        yield
    finally:
        try:
            if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                path.unlink()
        except OSError:
            pass


# --- recovery ------------------------------------------------------------------------------


def _age_seconds(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (datetime.now(UTC) - datetime.fromisoformat(stamp)).total_seconds()


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


def recover(settings: Settings, stale_after_seconds: float) -> dict[str, list[dict[str, Any]]]:
    """Fail jobs and experiments orphaned by a dead worker/runner. Returns what was changed."""
    report: dict[str, list[dict[str, Any]]] = {"jobs": [], "experiments": []}
    db = settings.data_dir("db") / DB_FILENAME
    with JobStore(db) as js:
        for job in js.list(status="running", limit=100_000):
            reasons = []
            if not pid_alive(job["worker_pid"]):
                reasons.append(f"worker pid {job['worker_pid']} not alive")
            age = _age_seconds(job["heartbeat_at"])
            if age is None or age > stale_after_seconds:
                reasons.append(f"heartbeat stale ({age if age is None else round(age)} s)")
            if not reasons:
                continue
            child = job["child_pid"]
            if child and pid_alive(child):  # the child normally exits by itself; make sure
                terminate_pid(child)
                _wait_dead(child, 10.0)
            note = "interrupted: " + "; ".join(reasons)
            # Close the job's own experiment FIRST (its child, the training process, was made
            # sure dead above; the runner pid is not consulted since it may have been reused),
            # then fail the job: an observer that sees the job failed sees the experiment closed.
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
            if pid_alive(exp.get("runner_pid")):
                continue  # e.g. a CLI `visionbench train` that is still running
            note = f"interrupted: no live process for this run (runner pid {exp.get('runner_pid')})"
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


def _watch_parent(parent_pid: int) -> None:
    while True:
        time.sleep(1.0)
        if not pid_alive(parent_pid):
            os._exit(3)  # worker is gone: stop now, leave the job for recovery


def _child_main(job_id: str, settings_payload: dict[str, Any], parent_pid: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    execute_job(job_id, Settings(_env_file=None, **settings_payload))


# --- supervision (worker process) ----------------------------------------------------------


def run_job(store: JobStore, job: dict[str, Any], settings: Settings, wc: WorkerConfig) -> str:
    job_id = job["id"]
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=_child_main,
        args=(job_id, settings.model_dump(mode="json"), os.getpid()),
        name=f"visionbench-{job_id}",
    )
    proc.start()
    store.set_child_pid(job_id, proc.pid)
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
    with worker_lock(settings):
        log.info("worker pid %s started on data root %s", os.getpid(), settings.resolved_data_root)
        recover(settings, wc.stale_after_seconds)
        done = 0
        with JobStore(settings.data_dir("db") / DB_FILENAME) as store:
            while not stop.is_set():
                job = store.claim(os.getpid())
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
