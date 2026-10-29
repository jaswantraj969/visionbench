"""The four corruptions of ROBUSTNESS_PROTOCOL.md section 3, as pure functions.

Input and output are RGB ``uint8`` arrays (H x W x 3). Noise, blur and brightness work in float64
on [0, 1] and are clipped and quantized back with ``rint(x * 255)`` (round half to even); JPEG
re-encodes the ``uint8`` image with Pillow and decodes it again.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

JPEG_SUBSAMPLING_420 = 2  # Pillow's code for 4:2:0


def _to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an H x W x 3 uint8 RGB array")
    return img.astype(np.float64) / 255.0


def quantize(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = _to_float(img)
    return quantize(x + rng.normal(0.0, sigma, size=x.shape))


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = math.ceil(3 * sigma)
    i = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(i**2) / (2 * sigma**2))
    return k / k.sum()


def _blur_axis(x: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = (len(k) - 1) // 2
    pad = [(0, 0)] * x.ndim
    pad[axis] = (r, r)
    padded = np.pad(x, pad, mode="reflect")  # d c b | a b c d | c b a
    n = x.shape[axis]
    out = np.zeros_like(x)
    for j, w in enumerate(k):  # fixed summation order: deterministic
        out += w * np.take(padded, np.arange(j, j + n), axis=axis)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    x = _to_float(img)
    k = gaussian_kernel(sigma)
    if (len(k) - 1) // 2 >= min(x.shape[0], x.shape[1]):
        raise ValueError("blur kernel radius must be smaller than the image")
    return quantize(_blur_axis(_blur_axis(x, k, axis=1), k, axis=0))


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """HSV V + delta (clipped to 1), H and S unchanged: every channel scales by V'/V."""
    x = _to_float(img)
    v = x.max(axis=2, keepdims=True)
    v_new = np.minimum(v + delta, 1.0)
    scale = np.divide(v_new, v, out=np.zeros_like(v), where=v > 0)
    out = np.where(v > 0, x * scale, v_new)  # V = 0 (black): S undefined -> grey V'
    return quantize(out)


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an H x W x 3 uint8 RGB array")
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(
        buf,
        format="JPEG",
        quality=int(quality),
        subsampling=JPEG_SUBSAMPLING_420,
        optimize=False,
        progressive=False,
    )
    with Image.open(io.BytesIO(buf.getvalue()), formats=("JPEG",)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def apply(img: np.ndarray, corruption: str, severity: float | int, seed: int) -> np.ndarray:
    if corruption == "gaussian_noise":
        return gaussian_noise(img, float(severity), np.random.default_rng(seed))
    if corruption == "gaussian_blur":
        return gaussian_blur(img, float(severity))
    if corruption == "brightness":
        return brightness(img, float(severity))
    if corruption == "jpeg":
        return jpeg(img, int(severity))
    raise ValueError(f"unknown corruption {corruption!r}")
