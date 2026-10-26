"""Data-loader factory for training. Only the "train" and "val" splits can ever be read here.

Every access to manifest entries goes through :func:`select_entries`, which rejects any split
other than train/val, so test images are never constructed, opened or iterated by training
code. Images are opened only via :func:`load_image` (a single choke point that tests patch).
"""

from __future__ import annotations

import os
import random
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from visionbench.core.errors import VisionBenchError

ALLOWED_SPLITS = ("train", "val")


class SplitAccessError(VisionBenchError):
    pass


def load_image(path: Path) -> Image.Image:
    """Open a snapshot image as RGB (fully decoded, file handle closed)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def select_entries(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Manifest entries of ``split`` ("train" or "val" only)."""
    if split not in ALLOWED_SPLITS:
        raise SplitAccessError(
            f"split {split!r} is not available to training code (allowed: {ALLOWED_SPLITS}); "
            "the test split is reserved for the final evaluation (M3)"
        )
    return [e for e in manifest["entries"] if e["split"] == split]


def class_names_from_manifest(manifest: dict[str, Any]) -> list[str]:
    """All labels in the manifest's per-class counts, sorted (index = class id)."""
    return sorted(manifest["counts"])


def split_class_counts(manifest: dict[str, Any], split: str) -> dict[str, int]:
    counts = Counter(e["label"] for e in select_entries(manifest, split))
    return {c: counts.get(c, 0) for c in class_names_from_manifest(manifest)}


class ManifestImageDataset(Dataset):
    def __init__(
        self, raw_root: Path, items: Sequence[tuple[str, int]], transform: Callable | None
    ) -> None:
        self.raw_root = raw_root
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        relpath, label = self.items[index]
        # global name lookup happens at call time, so a patched `load_image` is always used
        img = load_image(self.raw_root.joinpath(*relpath.split("/")))
        return (self.transform(img) if self.transform else img), label


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32  # distinct per worker, derived from the loader generator
    np.random.seed(seed)
    random.seed(seed)


def build_dataset(
    split: str, manifest: dict[str, Any], raw_root: Path, transform: Callable | None
) -> ManifestImageDataset:
    index = {c: i for i, c in enumerate(class_names_from_manifest(manifest))}
    items = [(e["relpath"], index[e["label"]]) for e in select_entries(manifest, split)]
    return ManifestImageDataset(raw_root, items, transform)


def build_dataloader(
    split: str,
    manifest: dict[str, Any],
    raw_root: Path,
    transform: Callable | None,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Train: shuffled with a seeded generator; a final batch of size 1 is dropped (BatchNorm
    cannot train on it). Val: fixed order, nothing dropped."""
    dataset = build_dataset(split, manifest, raw_root, transform)
    train = split == "train"
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train and len(dataset) % batch_size == 1,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers else None,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=False,
        multiprocessing_context="spawn" if num_workers and os.name == "nt" else None,
    )
