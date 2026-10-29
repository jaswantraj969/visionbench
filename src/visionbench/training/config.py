"""Training configuration: a Pydantic model loaded from YAML and overridable from the CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from visionbench.core.errors import VisionBenchError


class ConfigError(VisionBenchError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EarlyStoppingConfig(_Strict):
    enabled: bool = True
    monitor: Literal["val_loss", "val_balanced_accuracy"] = "val_loss"
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.0, ge=0)


class AugmentationConfig(_Strict):
    """Train-only augmentation. Validation transforms are always deterministic."""

    horizontal_flip: bool = True
    random_resized_crop: bool = True
    rrc_scale: tuple[float, float] = (0.8, 1.0)
    rrc_ratio: tuple[float, float] = (0.75, 1.3333)


def _default_workers() -> int:
    return 0 if os.name == "nt" else 2


class TrainConfig(_Strict):
    dataset_id: str
    split_id: str
    model: Literal["majority_baseline", "small_cnn", "resnet18", "efficientnet_b0"]
    weights: Literal["imagenet", "scratch"] = "scratch"
    finetune: Literal["full", "head_only"] = "full"
    input_size: int | None = Field(default=None, ge=16)  # None -> model default
    epochs: int = Field(default=10, ge=1)
    max_steps: int | None = Field(default=None, ge=1)  # smoke runs: cap total optimizer steps
    batch_size: int = Field(default=32, ge=1)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(default=1e-3, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    momentum: float = Field(default=0.9, ge=0)  # sgd only
    scheduler: Literal["none", "cosine"] = "none"
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    class_weighted_loss: bool = False
    seed: int = Field(default=0, ge=0, lt=2**32)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = False
    num_workers: int = Field(default_factory=_default_workers, ge=0)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    allow_weight_fallback: bool = False

    @model_validator(mode="after")
    def _check(self) -> TrainConfig:
        if self.model == "small_cnn" and self.weights == "imagenet":
            raise ValueError("small_cnn has no imagenet weights; use weights: scratch")
        if self.finetune == "head_only" and self.weights != "imagenet":
            raise ValueError("finetune: head_only requires weights: imagenet")
        return self


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot set {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def parse_override(item: str) -> tuple[str, Any]:
    """``key.sub=value``; the value is parsed as YAML (so 3, 0.1, true, [a, b] work)."""
    if "=" not in item:
        raise ConfigError(f"override {item!r} must look like key=value")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_train_config(
    path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load YAML (optional), apply dotted-key overrides (``None`` values are ignored), validate."""
    data: dict[str, Any] = {}
    if path is not None:
        try:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"config {path} must be a YAML mapping")
        data = loaded or {}
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(data, key, value)
    try:
        return TrainConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid training config:\n{exc}") from exc


NOT_APPLICABLE = "n/a"
# Settings the majority baseline never uses (it is fitted once from TRAIN class counts and opens
# no image); its stored config still holds TrainConfig's defaults for them.
BASELINE_UNUSED_FIELDS = (
    "weights",
    "finetune",
    "input_size",
    "epochs",
    "max_steps",
    "batch_size",
    "optimizer",
    "lr",
    "weight_decay",
    "momentum",
    "scheduler",
    "early_stopping",
    "class_weighted_loss",
    "device",
    "amp",
    "num_workers",
    "augmentation",
    "allow_weight_fallback",
)


def display_config(config: dict[str, Any]) -> dict[str, Any]:
    """A training config for display. For ``majority_baseline`` every setting it does not use is
    shown as ``"n/a"`` instead of the irrelevant defaults; the stored config is not changed."""
    if config.get("model") != "majority_baseline":
        return dict(config)
    return {k: (NOT_APPLICABLE if k in BASELINE_UNUSED_FIELDS else v) for k, v in config.items()}
