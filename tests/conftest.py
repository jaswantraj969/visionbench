import os
from collections.abc import Callable
from pathlib import Path

import pytest
import vb_synth
import vb_trainenv

from visionbench.data.scan import ScanConfig

_REPORT_LINES = pytest.StashKey[list[str]]()


@pytest.fixture
def report_line(request: pytest.FixtureRequest) -> Callable[[str], None]:
    """Append a line to the end-of-run summary (e.g. measured timings)."""
    return request.config.stash.setdefault(_REPORT_LINES, []).append


@pytest.fixture
def messy(tmp_path: Path) -> vb_synth.Fixture:
    return vb_synth.build_messy(tmp_path / "src")


@pytest.fixture
def messy_config() -> ScanConfig:
    return ScanConfig(max_image_pixels=vb_synth.SMALL_MAX_PIXELS)


@pytest.fixture
def balanced_src(tmp_path: Path) -> Path:
    return vb_synth.build_balanced(tmp_path / "balanced")


@pytest.fixture
def make_images() -> Callable[..., None]:
    return vb_synth.make_images


@pytest.fixture
def train_env(tmp_path: Path) -> vb_trainenv.TrainEnv:
    """Shapes dataset (3 classes x 12 images, 32 px) imported and split 0.5/0.25/0.25."""
    return vb_trainenv.make_train_env(tmp_path)


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
