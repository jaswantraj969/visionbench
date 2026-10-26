"""Fixtures for training tests: a tiny learnable shapes dataset, imported and split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import vb_synth

from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json
from visionbench.data.split import SplitConfig, make_split
from visionbench.data.store import import_dataset
from visionbench.training.config import TrainConfig


@dataclass
class TrainEnv:
    settings: Settings
    dataset_id: str
    split_id: str
    split_dir: Path
    raw_root: Path

    @property
    def manifest(self) -> dict[str, Any]:
        return read_json(self.split_dir / "manifest.json")

    def config(self, **overrides: Any) -> TrainConfig:
        base: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "model": "small_cnn",
            "input_size": 32,
            "epochs": 2,
            "batch_size": 8,
            "lr": 0.01,
            "device": "cpu",
            "num_workers": 0,
            "seed": 0,
        }
        return TrainConfig.model_validate(base | overrides)


@pytest.fixture
def train_env(tmp_path: Path) -> TrainEnv:
    src = vb_synth.build_shapes(tmp_path / "src", per_class=12, size=32)
    settings = Settings(_env_file=None, data_root=tmp_path / "var")
    datasets = settings.data_dir("datasets")
    imported = import_dataset(src, "shapes", datasets)
    split = make_split(
        datasets, imported.dataset_id, SplitConfig(seed=0, fractions=(0.5, 0.25, 0.25))
    )
    return TrainEnv(
        settings, imported.dataset_id, split.split_id, split.path, imported.path / "raw"
    )
