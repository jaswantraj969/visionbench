"""Helpers to run a real `visionbench worker` subprocess against a temporary data root."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from visionbench.core.config import Settings
from visionbench.jobs.store import JobStore
from visionbench.tracking.store import DB_FILENAME


class WorkerProcess:
    def __init__(self, settings: Settings, log_path: Path, **env_overrides: str) -> None:
        env = (
            os.environ
            | {
                "VISIONBENCH_DATA_ROOT": str(settings.resolved_data_root),
                "VISIONBENCH_WORKER_HEARTBEAT_SECONDS": "0.5",
                "VISIONBENCH_HEARTBEAT_STALE_SECONDS": "5",
                "VISIONBENCH_CANCEL_TIMEOUT_SECONDS": "30",
                "PYTHONUNBUFFERED": "1",
            }
            | env_overrides
        )
        self.log_path = log_path
        self._log = open(log_path, "ab")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "visionbench.cli", "worker", "--poll-seconds", "0.2"],
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    def kill(self) -> None:
        """Hard kill (TerminateProcess on Windows): no cleanup code runs in the worker."""
        self.proc.kill()
        self.proc.wait(30)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(30)
        self._log.close()

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")


def wait_for(
    predicate: Callable[[], Any], timeout: float, interval: float = 0.2, what: str = "condition"
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout} s waiting for {what}")


def job_store(settings: Settings) -> JobStore:
    return JobStore(settings.data_dir("db") / DB_FILENAME)


def get_job(settings: Settings, job_id: str) -> dict[str, Any]:
    with job_store(settings) as s:
        return s.get(job_id)
