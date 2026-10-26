"""Transforms, train-split normalization statistics, and class weights.

- Pretrained models: the weights' own resize / center-crop / interpolation / mean / std.
- Scratch models: resize to ``input_size`` x ``input_size`` and normalize with per-channel
  mean/std computed on the TRAIN split only (no augmentation), cached per split and resolution.
- Augmentation is applied to training images only; validation transforms are deterministic.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from visionbench.core.jsonio import read_json, write_json
from visionbench.training import data as data_mod
from visionbench.training.config import AugmentationConfig

NORM_STATS_VERSION = 1
SCRATCH_INTERPOLATION = "bilinear"


@dataclass(frozen=True)
class Preprocessing:
    source: str  # "imagenet_weights" | "train_split"
    input_size: int
    resize: int | None  # pretrained: resize shorter side to this, then center-crop input_size
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"mean": list(self.mean), "std": list(self.std)}


def _interp(name: str) -> InterpolationMode:
    return InterpolationMode(name)


def _geometry(prep: Preprocessing) -> list[Any]:
    interp = _interp(prep.interpolation)
    if prep.resize is not None:
        return [
            T.Resize(prep.resize, interpolation=interp, antialias=True),
            T.CenterCrop(prep.input_size),
        ]
    return [T.Resize((prep.input_size, prep.input_size), interpolation=interp, antialias=True)]


def build_transforms(prep: Preprocessing, aug: AugmentationConfig) -> tuple[T.Compose, T.Compose]:
    normalize = [T.ToTensor(), T.Normalize(prep.mean, prep.std)]
    val_tf = T.Compose([*_geometry(prep), *normalize])
    train_ops: list[Any] = []
    if aug.random_resized_crop:
        train_ops.append(
            T.RandomResizedCrop(
                prep.input_size,
                scale=aug.rrc_scale,
                ratio=aug.rrc_ratio,
                interpolation=_interp(prep.interpolation),
                antialias=True,
            )
        )
    else:
        train_ops.extend(_geometry(prep))
    if aug.horizontal_flip:
        train_ops.append(T.RandomHorizontalFlip())
    return T.Compose([*train_ops, *normalize]), val_tf


def describe_transforms(tf: T.Compose) -> list[str]:
    return [repr(t) for t in tf.transforms]


def _train_fingerprint(manifest: dict[str, Any]) -> str:
    lines = sorted(
        f"{e['relpath']}\t{e['sha256']}\n" for e in data_mod.select_entries(manifest, "train")
    )
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def compute_train_norm_stats(
    manifest: dict[str, Any], raw_root: Path, input_size: int
) -> dict[str, Any]:
    """Per-channel mean/std over all TRAIN pixels at ``input_size`` (resize only, no augment)."""
    geometry = T.Compose(
        [
            T.Resize(
                (input_size, input_size),
                interpolation=_interp(SCRATCH_INTERPOLATION),
                antialias=True,
            ),
            T.ToTensor(),
        ]
    )
    entries = data_mod.select_entries(manifest, "train")
    if not entries:
        raise ValueError("train split is empty; cannot compute normalization statistics")
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    pixels = 0
    for e in entries:
        x = geometry(data_mod.load_image(raw_root.joinpath(*e["relpath"].split("/"))))
        x = x.to(torch.float64).reshape(3, -1)
        total += x.sum(dim=1)
        total_sq += (x * x).sum(dim=1)
        pixels += x.shape[1]
    mean = total / pixels
    std = (total_sq / pixels - mean * mean).clamp_min(1e-12).sqrt()
    return {
        "version": NORM_STATS_VERSION,
        "input_size": input_size,
        "resize": "square",
        "interpolation": SCRATCH_INTERPOLATION,
        "train_images": len(entries),
        "train_fingerprint": _train_fingerprint(manifest),
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
    }


def train_norm_stats(
    manifest: dict[str, Any], raw_root: Path, input_size: int, split_dir: Path
) -> tuple[dict[str, Any], bool]:
    """Cached :func:`compute_train_norm_stats`; returns (stats, loaded_from_cache).

    The cache file lives next to the manifest, keyed by resolution, and is only reused when its
    train fingerprint (train relpaths + sha256) matches.
    """
    cache = split_dir / f"norm_stats_train_{input_size}px.json"
    if cache.is_file():
        stats = read_json(cache)
        if stats.get("version") == NORM_STATS_VERSION and stats.get(
            "train_fingerprint"
        ) == _train_fingerprint(manifest):
            return stats, True
    stats = compute_train_norm_stats(manifest, raw_root, input_size)
    write_json(cache, stats)
    return stats, False


def class_weights_from_train(manifest: dict[str, Any]) -> list[float]:
    """Inverse-frequency weights ``N / (K * n_c)`` from TRAIN counts only (0 for absent classes)."""
    counts = data_mod.split_class_counts(manifest, "train")
    total, k = sum(counts.values()), len(counts)
    return [total / (k * n) if n else 0.0 for n in counts.values()]
