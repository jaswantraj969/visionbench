"""Collect software/hardware environment information for experiment records."""

from __future__ import annotations

import platform
import sys
from typing import Any

import visionbench


def collect_env_info() -> dict[str, Any]:
    import numpy as np
    import torch
    import torchvision

    cuda_available = torch.cuda.is_available()
    return {
        "visionbench": visionbench.__version__,
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "torch_num_threads": torch.get_num_threads(),
    }
