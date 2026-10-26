"""Epoch loop, validation metrics and early stopping. Uses only train and val loaders.

Model selection and early stopping are driven exclusively by validation metrics.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

MONITOR_MODES = {"val_loss": "min", "val_balanced_accuracy": "max"}


@dataclass
class EarlyStopping:
    """Tracks the best value of ``monitor``; an epoch improves on the best only if it beats it by
    more than ``min_delta``. Stops after ``patience`` epochs without improvement."""

    monitor: str
    patience: int
    min_delta: float = 0.0
    enabled: bool = True
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.monitor not in MONITOR_MODES:
            raise ValueError(f"monitor must be one of {sorted(MONITOR_MODES)}")

    def update(self, epoch: int, value: float) -> bool:
        """Record ``value`` for ``epoch``; return True if it is the new best."""
        if math.isnan(value):
            improved = False
        elif self.best_value is None:
            improved = True
        elif MONITOR_MODES[self.monitor] == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        if improved:
            self.best_value, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.enabled and self.bad_epochs >= self.patience


def frozen_batchnorm_layers(model: nn.Module) -> list[nn.Module]:
    """BatchNorm layers whose affine parameters are all frozen (requires_grad False)."""
    out = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                out.append(m)
    return out


def train_mode(model: nn.Module) -> int:
    """``model.train()``, then put frozen BatchNorm layers back into eval mode so their running
    statistics stay fixed (head_only fine-tuning). Returns the number of such layers."""
    model.train()
    frozen = frozen_batchnorm_layers(model)
    for m in frozen:
        m.eval()
    return len(frozen)


def balanced_accuracy(confusion: torch.Tensor) -> float:
    """Mean recall over classes that occur in the targets (rows = true class)."""
    support = confusion.sum(dim=1)
    present = support > 0
    if not bool(present.any()):
        return float("nan")
    recall = confusion.diag()[present].to(torch.float64) / support[present].to(torch.float64)
    return float(recall.mean())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    max_steps: int | None = None,
) -> tuple[float, int, int]:
    """One pass over ``loader`` (stops early after ``max_steps`` optimizer steps).

    Returns (mean training loss per sample, optimizer steps taken, samples seen). With class
    weights in ``criterion`` the training loss is the class-weighted loss.
    """
    train_mode(model)  # re-applied every epoch: frozen BN layers stay in eval mode
    loss_sum, samples, steps = 0.0, 0, 0
    for inputs, targets in loader:
        if max_steps is not None and steps >= max_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(inputs), targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        steps += 1
        samples += targets.shape[0]
        loss_sum += float(loss.detach()) * targets.shape[0]
    return (loss_sum / samples if samples else float("nan")), steps, samples


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> dict[str, float]:
    """Unweighted mean cross-entropy, accuracy and balanced accuracy on a (val) loader."""
    model.eval()
    loss_sum, n = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs).float()
        loss_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
        preds = logits.argmax(dim=1)
        idx = (targets * num_classes + preds).cpu()
        confusion += torch.bincount(idx, minlength=num_classes**2).reshape(num_classes, -1)
        n += targets.shape[0]
    if n == 0:
        raise ValueError("validation loader is empty")
    return {
        "val_loss": loss_sum / n,
        "val_accuracy": float(confusion.diag().sum()) / n,
        "val_balanced_accuracy": balanced_accuracy(confusion),
    }


@dataclass
class FitResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int | None = None
    best_value: float | None = None
    stopped_reason: str = "epochs_completed"
    total_steps: int = 0


EpochCallback = Callable[[dict[str, Any], bool, nn.Module], None]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    num_classes: int,
    early_stopping: EarlyStopping,
    max_steps: int | None = None,
    amp: bool = False,
    on_epoch_end: EpochCallback | None = None,
) -> FitResult:
    """Train for up to ``epochs``; after every epoch evaluate on VAL, update early stopping and
    call ``on_epoch_end(record, is_best, model)`` (where checkpoints are written)."""
    scaler = torch.amp.GradScaler(device.type) if amp else None
    result = FitResult()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        lr = float(optimizer.param_groups[0]["lr"])
        remaining = None if max_steps is None else max_steps - result.total_steps
        train_loss, steps, samples = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, max_steps=remaining
        )
        result.total_steps += steps
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(model, val_loader, device, num_classes)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
            "lr": lr,
            "steps": result.total_steps,
            "train_samples": samples,
            "epoch_seconds": time.perf_counter() - start,
        }
        is_best = early_stopping.update(epoch, metrics[early_stopping.monitor])
        result.history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record, is_best, model)
        if max_steps is not None and result.total_steps >= max_steps:
            result.stopped_reason = "max_steps"
            break
        if early_stopping.should_stop:
            result.stopped_reason = "early_stopping"
            break
    result.best_epoch, result.best_value = early_stopping.best_epoch, early_stopping.best_value
    return result
