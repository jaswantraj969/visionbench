"""Model registry: majority_baseline, small_cnn, resnet18, efficientnet_b0."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn
from torchvision import models as tv_models

from visionbench.core.errors import VisionBenchError
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import PretrainedInfo, WeightsUnavailableError
from visionbench.models.small_cnn import SmallCNN

MODEL_NAMES = ("majority_baseline", "small_cnn", "resnet18", "efficientnet_b0")
TORCHVISION_MODELS = ("resnet18", "efficientnet_b0")
DEFAULT_INPUT_SIZE = {"small_cnn": 64, "resnet18": 224, "efficientnet_b0": 224}
# Grad-CAM target layers (post-activation feature maps; used by visionbench.explain.gradcam).
TARGET_LAYERS = {
    "small_cnn": SmallCNN.TARGET_LAYER,
    "resnet18": "layer4",
    "efficientnet_b0": "features.8",  # last block of `features` (1x1 conv to 1280 channels)
}


class ModelConfigError(VisionBenchError):
    pass


@dataclass
class BuiltModel:
    name: str
    model: nn.Module | MajorityBaseline
    input_size: int | None  # None for the baseline (no images used)
    normalization_source: str  # "imagenet_weights" | "train_split" | "none"
    weights: str  # "imagenet" | "scratch" | "none"
    finetune: str  # "full" | "head_only" | "none"
    param_count: int
    trainable_param_count: int
    target_layer: str | None
    pretrained: PretrainedInfo | None = None
    fallback_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "normalization_source": self.normalization_source,
            "weights": self.weights,
            "finetune": self.finetune,
            "param_count": self.param_count,
            "trainable_param_count": self.trainable_param_count,
            "target_layer": self.target_layer,
            "pretrained": self.pretrained.to_dict() if self.pretrained else None,
            "weight_fallback_reason": self.fallback_reason,
        }


def _head(model: nn.Module, name: str) -> nn.Module:
    return model.fc if name == "resnet18" else model.classifier


def _torchvision_model(name: str, num_classes: int, state: dict[str, Any] | None) -> nn.Module:
    if name == "resnet18":
        model = tv_models.resnet18(weights=None)
        if state is not None:
            model.load_state_dict(state)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = tv_models.efficientnet_b0(weights=None)
        if state is not None:
            model.load_state_dict(state)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _counts(model: nn.Module) -> tuple[int, int]:
    params = list(model.parameters())
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)


def build_model(
    name: str,
    class_names: Sequence[str],
    *,
    weights: str = "scratch",
    finetune: str = "full",
    input_size: int | None = None,
    hub_dir: Path | None = None,
    allow_weight_fallback: bool = False,
) -> BuiltModel:
    """Construct a model. Consumes the torch RNG for initialization, so seed first."""
    if name not in MODEL_NAMES:
        raise ModelConfigError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    num_classes = len(class_names)
    if num_classes < 2:
        raise ModelConfigError("at least 2 classes are required")

    if name == "majority_baseline":
        return BuiltModel(
            name,
            MajorityBaseline(class_names),
            None,
            "none",
            "none",
            "none",
            num_classes,
            num_classes,
            None,
        )

    if weights not in ("imagenet", "scratch"):
        raise ModelConfigError(f"weights must be 'imagenet' or 'scratch', got {weights!r}")
    if finetune not in ("full", "head_only"):
        raise ModelConfigError(f"finetune must be 'full' or 'head_only', got {finetune!r}")
    if input_size is not None and input_size < 16:
        raise ModelConfigError("input_size must be >= 16")

    if name == "small_cnn":
        if weights != "scratch" or finetune != "full":
            raise ModelConfigError("small_cnn only supports weights='scratch', finetune='full'")
        model = SmallCNN(num_classes)
        total, trainable = _counts(model)
        return BuiltModel(
            name,
            model,
            input_size or DEFAULT_INPUT_SIZE[name],
            "train_split",
            "scratch",
            "full",
            total,
            trainable,
            TARGET_LAYERS[name],
        )

    pretrained: PretrainedInfo | None = None
    fallback_reason: str | None = None
    state = None
    if weights == "imagenet":
        if hub_dir is None:
            raise ModelConfigError("hub_dir is required for imagenet weights")
        from visionbench.models import pretrained as pretrained_mod

        try:
            state, pretrained = pretrained_mod.load_imagenet_weights(name, hub_dir)
        except WeightsUnavailableError as exc:
            if not allow_weight_fallback:
                raise WeightsUnavailableError(
                    f"{exc}. Training aborted: pass --allow-weight-fallback to train from "
                    "scratch instead (recorded as weights='scratch')."
                ) from exc
            if finetune == "head_only":
                raise ModelConfigError(
                    f"{exc}. Cannot fall back to scratch with finetune='head_only' "
                    "(a frozen random backbone is meaningless)."
                ) from exc
            fallback_reason = str(exc)
            weights = "scratch"
    elif finetune == "head_only":
        raise ModelConfigError("finetune='head_only' requires weights='imagenet'")

    if pretrained is not None and input_size not in (None, pretrained.crop):
        raise ModelConfigError(
            f"{name} with imagenet weights uses input size {pretrained.crop} "
            f"(resize {pretrained.resize}); got input_size={input_size}"
        )

    model = _torchvision_model(name, num_classes, state)
    if finetune == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in _head(model, name).parameters():
            p.requires_grad = True
    total, trainable = _counts(model)
    size = pretrained.crop if pretrained else (input_size or DEFAULT_INPUT_SIZE[name])
    return BuiltModel(
        name=name,
        model=model,
        input_size=size,
        normalization_source="imagenet_weights" if pretrained else "train_split",
        weights=weights,
        finetune=finetune,
        param_count=total,
        trainable_param_count=trainable,
        target_layer=TARGET_LAYERS[name],
        pretrained=pretrained,
        fallback_reason=fallback_reason,
    )
