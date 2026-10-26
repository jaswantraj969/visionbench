import os

import pytest


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Record the in-process accelerator state so CPU-only runs are evidenced in the log."""
    import torch

    terminalreporter.write_line(
        f"torch {torch.__version__}; torch.cuda.is_available()={torch.cuda.is_available()}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    )
