"""Pretrained ImageNet weights from torchvision, cached in the torch hub directory
(``VISIONBENCH_TORCH_HOME``, default ``<data_root>/cache/torch``).

Weights are fetched with torch.hub (``check_hash=True`` on download) and loaded with
``weights_only=True``. The cached file's SHA-256 is computed and must start with the hash prefix
embedded in its file name (torch.hub's naming convention); a mismatch is an error.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, WeightsEnum

from visionbench.core.errors import VisionBenchError

IMAGENET_WEIGHTS: dict[str, WeightsEnum] = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
}
_HASH_IN_NAME = re.compile(r"-([a-f0-9]+)\.")


class WeightsUnavailableError(VisionBenchError):
    pass


@dataclass(frozen=True)
class PretrainedInfo:
    enum_name: str  # e.g. "ResNet18_Weights.IMAGENET1K_V1"
    url: str
    path: Path
    sha256: str
    resize: int
    crop: int
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enum": self.enum_name,
            "url": self.url,
            "path": str(self.path),
            "sha256": self.sha256,
            "resize": self.resize,
            "crop": self.crop,
            "interpolation": self.interpolation,
            "mean": list(self.mean),
            "std": list(self.std),
        }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_state_dict(weights: WeightsEnum, hub_dir: Path) -> tuple[dict[str, Any], Path]:
    """Download (or reuse the cached copy of) ``weights`` into ``hub_dir/checkpoints``."""
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))  # set before any weight load
    model_dir = hub_dir / "checkpoints"
    state = torch.hub.load_state_dict_from_url(
        weights.url, model_dir=str(model_dir), progress=False, check_hash=True, weights_only=True
    )
    return state, model_dir / Path(weights.url).name


def load_imagenet_weights(model_name: str, hub_dir: Path) -> tuple[dict[str, Any], PretrainedInfo]:
    """Return (state_dict, info). Raises :class:`WeightsUnavailableError` on any failure."""
    weights = IMAGENET_WEIGHTS[model_name]
    try:
        state, path = fetch_state_dict(weights, hub_dir)
    except Exception as exc:  # network, hash, permission, corrupt file, ...
        raise WeightsUnavailableError(
            f"could not obtain {weights} from {weights.url}: {type(exc).__name__}: {exc}"
        ) from exc
    sha = file_sha256(path)
    match = _HASH_IN_NAME.search(path.name)
    if match and not sha.startswith(match.group(1)):
        raise WeightsUnavailableError(
            f"{path.name}: sha256 {sha} does not match the hash prefix in its name"
        )
    t = weights.transforms()
    info = PretrainedInfo(
        enum_name=f"{type(weights).__name__}.{weights.name}",
        url=weights.url,
        path=path,
        sha256=sha,
        resize=int(t.resize_size[0]),
        crop=int(t.crop_size[0]),
        interpolation=t.interpolation.value,
        mean=tuple(float(v) for v in t.mean),
        std=tuple(float(v) for v in t.std),
    )
    return state, info
