"""Checkpoints: ``<name>.pt`` (state_dict + primitives) with a sibling ``<name>.json``.

Loading is restricted to files inside ``<data_root>/experiments`` and always uses
``torch.load(weights_only=True)``. User-supplied model files are never loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from visionbench.core.errors import VisionBenchError
from visionbench.core.jsonio import write_json
from visionbench.models.pretrained import file_sha256


class CheckpointError(VisionBenchError):
    pass


def save_checkpoint(
    path: Path, model: nn.Module, *, epoch: int, metrics: dict[str, float], meta: dict[str, Any]
) -> str:
    """Write ``path`` and ``path.with_suffix('.json')``; return the .pt file's SHA-256."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state": state,
            "epoch": int(epoch),
            "metrics": {k: float(v) for k, v in metrics.items()},
        },
        path,
    )
    sha = file_sha256(path)
    write_json(
        path.with_suffix(".json"),
        meta | {"epoch": int(epoch), "metrics": metrics, "file": path.name, "sha256": sha},
    )
    return sha


def load_checkpoint(path: Path, data_root: Path) -> dict[str, Any]:
    """Load an internal checkpoint (``weights_only=True``) from the experiments store only."""
    store = (data_root / "experiments").resolve()
    resolved = Path(path).resolve()
    if store not in resolved.parents:
        raise CheckpointError(f"refusing to load a checkpoint outside {store}: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=True)
