"""Process identity: a process is (pid, creation time), never a pid alone.

Operating systems reuse pids (Windows does so quickly), so "a process with pid N exists" says
nothing about whether it is the process that was recorded. Everything that decides a recorded
process is still running, or terminates one, goes through :func:`process_is` /
:func:`terminate_process`, which compare the recorded creation time (psutil, seconds since the
epoch) with that of the live process.
"""

from __future__ import annotations

import os

import psutil

# Creation times are read from the same OS clock both times; the tolerance only absorbs float
# rounding. A reused pid gets a creation time of its own, normally seconds or more later.
CREATE_TIME_TOLERANCE_SECONDS = 0.05


def process_create_time(pid: int | None) -> float | None:
    """Creation time of the live process ``pid``, or ``None`` if there is none (or no access)."""
    if not pid or pid <= 0:
        return None
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def own_identity() -> tuple[int, float]:
    pid = os.getpid()
    return pid, psutil.Process(pid).create_time()


def process_is(pid: int | None, create_time: float | None) -> bool:
    """True only if a process with ``pid`` exists AND its creation time matches.

    A missing creation time (rows written before schema v5) never matches.
    """
    if create_time is None:
        return False
    actual = process_create_time(pid)
    return actual is not None and abs(actual - create_time) <= CREATE_TIME_TOLERANCE_SECONDS


def terminate_process(pid: int | None, create_time: float | None) -> bool:
    """Kill the recorded process if, and only if, it is still the same process.

    Returns True if a kill was sent. The identity check and the kill use one ``psutil.Process``
    handle, and psutil re-checks the creation time before signalling, so a pid reused between
    the two steps is not killed either.
    """
    if not process_is(pid, create_time):
        return False
    try:
        proc = psutil.Process(pid)
        if abs(proc.create_time() - create_time) > CREATE_TIME_TOLERANCE_SECONDS:  # type: ignore[operator]
            return False
        proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return True


def wait_gone(pid: int | None, create_time: float | None, timeout: float) -> bool:
    """Wait until the recorded process is gone; True if it is gone."""
    if not process_is(pid, create_time):
        return True
    try:
        psutil.Process(pid).wait(timeout)  # type: ignore[arg-type]
    except psutil.TimeoutExpired:
        return not process_is(pid, create_time)
    except psutil.NoSuchProcess:
        pass
    return True
