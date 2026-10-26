"""SmallCNN: 4 x (conv3x3 -> BN -> ReLU -> maxpool2) -> global average pool -> linear."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)  # non-overlapping windows: deterministic backward on CUDA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class SmallCNN(nn.Module):
    """Default input 64x64; any input >= 16 px works (four 2x poolings)."""

    TARGET_LAYER = "features.block4.conv"  # last conv layer, for Grad-CAM (M5)

    def __init__(
        self, num_classes: int, channels: Sequence[int] = (32, 64, 128, 256), dropout: float = 0.2
    ) -> None:
        super().__init__()
        blocks: OrderedDict[str, nn.Module] = OrderedDict()
        in_ch = 3
        for i, out_ch in enumerate(channels, start=1):
            blocks[f"block{i}"] = ConvBlock(in_ch, out_ch)
            in_ch = out_ch
        self.features = nn.Sequential(blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pooling as a plain mean (deterministic backward)
        return self.classifier(self.dropout(x))
