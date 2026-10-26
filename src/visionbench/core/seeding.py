"""Seeding of all random number generators used by VisionBench.

Determinism notes:
- ``torch.backends.cudnn.deterministic = True`` and ``benchmark = False`` make cuDNN choose
  deterministic algorithms (slower, but reproducible on the same hardware/software).
- ``torch.use_deterministic_algorithms(True, warn_only=True)`` makes PyTorch warn when an op
  without a deterministic implementation runs. Some CUDA ops additionally require the env var
  ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, which must be set before CUDA is initialised.
- ``PYTHONHASHSEED`` only affects interpreters started after it is set (e.g. DataLoader workers
  using spawn); it cannot change hashing in the already-running process.
- Bitwise reproducibility is only expected on identical hardware, drivers, and library versions.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python ``random``, NumPy, and PyTorch (CPU and all CUDA devices).

    Returns a dict describing exactly what was set, suitable for experiment records.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch": True,
        "torch_cuda": torch.cuda.is_available(),
        "pythonhashseed_env": str(seed),
        "deterministic": deterministic,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def derived_rng(seed: int, key: str) -> np.random.Generator:
    """Return a NumPy Generator seeded from ``(seed, key)``.

    The key is mixed in via SHA-256 (not Python's salted ``hash()``), so the stream is the same
    in every process and on every platform for a given NumPy version.
    """
    key_int = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([seed, key_int]))
