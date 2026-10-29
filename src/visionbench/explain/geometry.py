"""Geometry of the recorded validation transform, and its exact inverse.

The val transform of every trainable model is one of (``training.preprocessing``):

- scratch models: ``Resize((s, s))`` - the whole original is stretched onto the s x s input;
- pretrained models: ``Resize(r)`` (shorter side to r, aspect kept) then ``CenterCrop(s)`` - the
  border outside the crop is never seen by the model.

``ToTensor`` / ``Normalize`` do not move pixels. Coordinates here are continuous pixel-edge
coordinates (pixel i covers [i, i + 1); its centre is i + 0.5), the convention of torchvision's
resize (``align_corners=False``): an original point x maps to x * (resized / original) in the
resized image, and the crop subtracts its integer offset. The sizes and offsets are computed
exactly as torchvision computes them, so the mapping matches the tensor the model received.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from visionbench.training.preprocessing import Preprocessing


@dataclass(frozen=True)
class ValGeometry:
    original_size: tuple[int, int]  # (width, height) of the decoded original
    resized_size: tuple[int, int]  # (width, height) after the resize step
    crop_offset: tuple[int, int]  # (left, top) of the centre crop in resized pixels; (0, 0) if none
    input_size: int  # side of the square model input

    @classmethod
    def from_preprocessing(cls, prep: Preprocessing, width: int, height: int) -> ValGeometry:
        s = prep.input_size
        if prep.resize is None:  # Resize((s, s))
            return cls((width, height), (s, s), (0, 0), s)
        # torchvision Resize(int): the shorter edge becomes `resize`, the longer one
        # int(resize * long / short) (functional._compute_resized_output_size).
        short, long = min(width, height), max(width, height)
        new_short, new_long = prep.resize, int(prep.resize * long / short)
        new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
        if new_w < s or new_h < s:
            raise ValueError("resize smaller than the centre crop (padding is not supported)")
        # torchvision CenterCrop: int(round((size - crop) / 2.0)).
        left, top = int(round((new_w - s) / 2.0)), int(round((new_h - s) / 2.0))
        return cls((width, height), (new_w, new_h), (left, top), s)

    @property
    def scale(self) -> tuple[float, float]:
        return (
            self.resized_size[0] / self.original_size[0],
            self.resized_size[1] / self.original_size[1],
        )

    def to_input(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple[np.ndarray, ...]:
        """Original coordinates -> model-input coordinates (may fall outside [0, s))."""
        sx, sy = self.scale
        return np.asarray(x) * sx - self.crop_offset[0], np.asarray(y) * sy - self.crop_offset[1]

    def to_original(self, xi: np.ndarray | float, yi: np.ndarray | float) -> tuple[np.ndarray, ...]:
        """Model-input coordinates -> original coordinates (exact inverse of :meth:`to_input`)."""
        sx, sy = self.scale
        return (np.asarray(xi) + self.crop_offset[0]) / sx, (
            np.asarray(yi) + self.crop_offset[1]
        ) / sy

    def seen_box(self) -> dict[str, float]:
        """The part of the original that the model input covers (original coordinates)."""
        x0, y0 = self.to_original(0.0, 0.0)
        x1, y1 = self.to_original(float(self.input_size), float(self.input_size))
        w, h = self.original_size
        return {
            "left": float(max(0.0, x0)),
            "top": float(max(0.0, y0)),
            "right": float(min(float(w), x1)),
            "bottom": float(min(float(h), y1)),
        }

    def seen_mask(self, out_w: int, out_h: int) -> np.ndarray:
        """(out_h, out_w) bool: is the centre of each display pixel inside the model input?"""
        xi, yi = self._display_to_input(out_w, out_h)
        s = self.input_size
        return (xi >= 0) & (xi < s) & (yi >= 0) & (yi < s)

    def unseen_fraction(self) -> float:
        """Fraction of the ORIGINAL pixels (by centre) that the model never saw."""
        w, h = self.original_size
        return float(1.0 - self.seen_mask(w, h).mean())

    def _display_to_input(self, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.original_size
        xs = (np.arange(out_w) + 0.5) * (w / out_w)  # display pixel centres -> original coords
        ys = (np.arange(out_h) + 0.5) * (h / out_h)
        xi, yi = self.to_input(xs, ys)
        return np.broadcast_to(xi[None, :], (out_h, out_w)), np.broadcast_to(
            yi[:, None], (out_h, out_w)
        )

    def map_to_original(self, cam: torch.Tensor, out_w: int, out_h: int) -> np.ndarray:
        """Sample a feature-map CAM (h_f, w_f), which spans the model input, at every display
        pixel of the original image. Pixels the model never saw get NaN.

        The feature map is treated as covering the s x s input uniformly (cell centres at
        (j + 0.5) * s / w_f), i.e. bilinear upsampling with ``align_corners=False``.
        """
        xi, yi = self._display_to_input(out_w, out_h)
        s = self.input_size
        grid = np.stack([2.0 * xi / s - 1.0, 2.0 * yi / s - 1.0], axis=-1)
        sampled = F.grid_sample(
            cam.detach().to(torch.float64)[None, None],
            torch.from_numpy(np.ascontiguousarray(grid))[None],
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, 0].numpy()
        return np.where(self.seen_mask(out_w, out_h), sampled, np.nan)

    def describe(self) -> dict[str, object]:
        return {
            "original_size": list(self.original_size),
            "resized_size": list(self.resized_size),
            "crop_offset": list(self.crop_offset),
            "input_size": self.input_size,
            "seen_box_in_original": self.seen_box(),
            "unseen_fraction": self.unseen_fraction(),
        }
