import os
from collections.abc import Callable

import pytest

_REPORT_LINES = pytest.StashKey[list[str]]()


@pytest.fixture
def report_line(request: pytest.FixtureRequest) -> Callable[[str], None]:
    """Append a line to the end-of-run summary (e.g. measured timings)."""
    return request.config.stash.setdefault(_REPORT_LINES, []).append


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Record the in-process accelerator state so CPU-only runs are evidenced in the log."""
    import torch

    for line in config.stash.get(_REPORT_LINES, []):
        terminalreporter.write_line(line)
    terminalreporter.write_line(
        f"torch {torch.__version__}; torch.cuda.is_available()={torch.cuda.is_available()}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )
