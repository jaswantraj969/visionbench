"""Grad-CAM (Selvaraju et al., 2017) on the registry's target layer of a trained model.

For one image and one target class c:

1. a forward hook on the target layer captures its activation A (1 x K x h x w);
2. the gradient of the TARGET LOGIT y_c (pre-softmax) with respect to A is computed with
   ``torch.autograd.grad``. This gives the same gradient a backward hook would see, without
   writing ``.grad`` into the parameters of the (shared, cached) model;
3. channel weights a_k = spatial mean of dy_c/dA_k;
4. CAM = ReLU(sum_k a_k A_k), an h x w map in model-input space.

The model is in eval mode, gradients are enabled only inside this call, and the hook is
always removed (``try``/``finally``). The CAM is normalised by its maximum only when it holds
positive evidence; an all-zero CAM is returned as zeros with ``all_zero=True`` and is never
rescaled into a heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from visionbench.core.errors import VisionBenchError
from visionbench.models.registry import TARGET_LAYERS

LIMITATIONS = (
    "Grad-CAM shows which regions of the final convolutional feature map most increased the "
    "score of the target class. It is coarse (feature-map resolution stated above), is not a "
    "causal explanation, and does not show that the prediction is correct."
)
# A CAM whose positive peak is below this fraction of the largest |weighted activation| is
# numerically zero (no positive evidence); it is not normalised into a heatmap.
ZERO_RELATIVE_TOLERANCE = 1e-7


class ExplanationNotSupported(VisionBenchError):
    pass


@dataclass(frozen=True)
class CamResult:
    layer: str
    target_index: int
    logits: torch.Tensor  # (num_classes,), detached, CPU
    cam: torch.Tensor  # (h, w) ReLU(sum_k a_k A_k), unnormalised, float64, CPU
    normalized: torch.Tensor  # (h, w) in [0, 1]; all zeros when all_zero
    all_zero: bool

    @property
    def feature_map_size(self) -> tuple[int, int]:
        return int(self.cam.shape[0]), int(self.cam.shape[1])


def target_layer_name(model_name: str) -> str:
    if model_name not in TARGET_LAYERS:
        raise ExplanationNotSupported(
            f"Grad-CAM is not supported for {model_name!r} (no convolutional target layer)"
        )
    return TARGET_LAYERS[model_name]


def _module(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise ExplanationNotSupported(f"target layer {name!r} not found in the model")
    return modules[name]


def grad_cam(model: nn.Module, x: torch.Tensor, target_index: int, layer_name: str) -> CamResult:
    """Grad-CAM for a single preprocessed image ``x`` (1 x 3 x s x s) on the model's device."""
    if x.shape[0] != 1:
        raise ValueError("grad_cam expects a batch of exactly one image")
    if model.training:
        raise ValueError("grad_cam needs the model in eval mode")
    layer = _module(model, layer_name)
    captured: list[torch.Tensor] = []
    handle = layer.register_forward_hook(lambda _m, _i, out: captured.append(out))
    try:
        with torch.enable_grad():
            # Input requires grad so the activation is part of a graph even if the model's
            # parameters were frozen; only dy_c/dA is computed below.
            logits = model(x.detach().requires_grad_(True))
            if len(captured) != 1:
                raise ExplanationNotSupported(
                    f"target layer {layer_name!r} ran {len(captured)} times in one forward pass"
                )
            activation = captured[0]
            score = logits[0, target_index]  # IndexError for an invalid class index
            (grad,) = torch.autograd.grad(score, activation)
    finally:
        handle.remove()
    with torch.no_grad():
        weights = grad.mean(dim=(2, 3), keepdim=True)
        weighted = (weights * activation).sum(dim=1)[0].to(torch.float64).cpu()
        cam = weighted.clamp_min(0.0)
        peak = float(cam.max())
        scale = float(weighted.abs().max())
        all_zero = peak <= 0.0 or peak <= ZERO_RELATIVE_TOLERANCE * scale
        normalized = torch.zeros_like(cam) if all_zero else cam / peak
    return CamResult(
        layer_name,
        target_index,
        logits.detach()[0].cpu(),
        cam,
        normalized,
        all_zero,
    )
