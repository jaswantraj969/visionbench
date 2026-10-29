"""Keep a long-running process from being suspended by Windows sleep / Modern Standby."""

from __future__ import annotations

import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> bool:
    """Ask Windows not to sleep while this process runs (released when it exits).

    Returns True if the request was made (Windows), False elsewhere. It does not stop a user
    from closing the lid or choosing Sleep, and Modern Standby may still be entered on battery
    by policy; long runs should still record timings as indicative.
    """
    if sys.platform != "win32":
        return False
    import ctypes

    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
