"""Efficiency: parameter counts, checkpoint size, training time, forward-pass latency.

Latency covers the model forward pass ONLY (preprocessing, image decoding and data loading are
excluded): random input of the model's input size, ``torch.inference_mode``, eval mode, warmup
iterations, then timed iterations with ``torch.cuda.synchronize`` around each on CUDA. A latency
number is valid only for the device (and threads) it was measured on.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

LATENCY_SCOPE = "model forward pass only; preprocessing, decoding and data loading excluded"
DEFAULT_BATCH_SIZES = (1, 32)
DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 50


def param_counts(model: nn.Module) -> dict[str, int]:
    params = list(model.parameters())
    return {
        "total": sum(p.numel() for p in params),
        "trainable": sum(p.numel() for p in params if p.requires_grad),
    }


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def not_applicable(reason: str) -> dict[str, Any]:
    return {"applicable": False, "reason": reason, "scope": LATENCY_SCOPE, "batch_sizes": None}


def measure_latency(
    model: nn.Module,
    input_size: int,
    device: torch.device,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    warmup: int = DEFAULT_WARMUP,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> dict[str, Any]:
    """Median and p95 forward-pass time (ms) per batch size on ``device``."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    model = model.to(device).eval()
    cuda = device.type == "cuda"
    gen = torch.Generator().manual_seed(seed)
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for bs in batch_sizes:
            x = torch.randn(bs, 3, input_size, input_size, generator=gen).to(device)
            for _ in range(warmup):
                model(x)
            if cuda:
                torch.cuda.synchronize(device)
            times = []
            for _ in range(iterations):
                if cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(x)
                if cuda:
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000.0)
            median = statistics.median(times)
            results[str(bs)] = {
                "median_ms": median,
                "p95_ms": float(np.percentile(times, 95)),
                "median_ms_per_image": median / bs,
            }
    return {
        "applicable": True,
        "scope": LATENCY_SCOPE,
        "device": str(device),
        "device_name": _device_name(device),
        "torch_threads": torch.get_num_threads(),
        "input_size": input_size,
        "warmup": warmup,
        "iterations": iterations,
        "batch_sizes": results,
    }
