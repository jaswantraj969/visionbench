import random

import numpy as np
import pytest
import torch
from torch import nn

from visionbench.core.seeding import set_seed


def _seeded_forward(seed: int) -> torch.Tensor:
    set_seed(seed)
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        return model(x)


def test_identical_seeds_give_identical_outputs() -> None:
    a = _seeded_forward(123)
    b = _seeded_forward(123)
    assert torch.equal(a, b)


def test_different_seeds_give_different_outputs() -> None:
    assert not torch.equal(_seeded_forward(1), _seeded_forward(2))


def test_python_and_numpy_rngs_seeded() -> None:
    set_seed(7)
    first = (random.random(), np.random.rand())
    set_seed(7)
    assert (random.random(), np.random.rand()) == first


def test_report_describes_what_was_set() -> None:
    report = set_seed(42)
    assert report["seed"] == 42
    assert report["deterministic"] is True
    assert report["cudnn_deterministic"] is True
    assert report["cudnn_benchmark"] is False
    assert report["torch_deterministic_algorithms"] is True
    assert report["torch_cuda"] == torch.cuda.is_available()


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "1", True])
def test_invalid_seed_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        set_seed(bad)  # type: ignore[arg-type]
