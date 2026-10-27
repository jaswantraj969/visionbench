"""64-bit DCT perceptual hash (pHash), implemented in-house for the split audit.

Algorithm ``phash-v1`` (input: the EXIF-transposed image, see ``visionbench.data.imageio``):

1. convert to 8-bit grayscale and resize to 32x32 (Lanczos);
2. 2-D DCT-II with orthonormal scaling (``scipy.fft.dctn``);
3. keep the top-left 8x8 block (lowest frequencies);
4. threshold every coefficient against the median of the 63 AC coefficients (the DC term,
   i.e. mean brightness, is excluded from the median);
5. the DC bit is always 0, so the hash carries 63 informative bits; bits are packed row-major
   (``[u, v]`` with u the vertical frequency), most significant bit first.

It is deliberately unrelated to the dHash used for duplicate grouping (gradient signs on a
9x8 thumbnail): the audit uses it as a second, independent detector.
"""

from __future__ import annotations

import numpy as np
import scipy.fft
from PIL import Image

PHASH_VERSION = "phash-v1"
PHASH_SIZE = 32
PHASH_LOW = 8


def phash_from_gray(gray: np.ndarray) -> int:
    """pHash of an already-resized 32x32 grayscale array."""
    a = np.asarray(gray, dtype=np.float64)
    if a.shape != (PHASH_SIZE, PHASH_SIZE):
        raise ValueError(f"expected a {PHASH_SIZE}x{PHASH_SIZE} array, got shape {a.shape}")
    low = scipy.fft.dctn(a, type=2, norm="ortho")[:PHASH_LOW, :PHASH_LOW].ravel()
    bits = low > np.median(low[1:])
    bits[0] = False  # DC: excluded
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def phash(image: Image.Image) -> int:
    gray = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
    return phash_from_gray(np.asarray(gray))
