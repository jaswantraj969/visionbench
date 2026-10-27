"""The one way image pixels are read: decode fully, then apply the EXIF orientation.

Orientation policy ``"exif_transposed"``: every consumer of pixels sees an image the way it is
meant to be displayed, i.e. after :func:`PIL.ImageOps.exif_transpose`. This covers:

- the dataset scan (recorded width/height, gray std, dHash);
- training and evaluation (``training.data.load_image``);
- the split audit (pHash, decoded-pixel hash, contact sheets).

A JPEG stored sideways with ``Orientation=6`` and a physically rotated copy without EXIF
therefore get the same dimensions, hashes and tensors. Snapshot files are still stored byte for
byte (their SHA-256 is of the original bytes, EXIF included).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

ORIENTATION_POLICY = "exif_transposed"
PILLOW_FORMATS = ("JPEG", "PNG", "BMP", "WEBP")


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image  # fully loaded, EXIF orientation applied, file handle closed
    mode: str  # mode as stored in the file
    format: str  # container format as detected by Pillow


def decode_image(
    source: bytes | Path, *, formats: tuple[str, ...] = PILLOW_FORMATS
) -> DecodedImage:
    """Open ``source`` (bytes or a path), decode it completely and apply EXIF orientation.

    Raises whatever Pillow raises for unreadable, truncated or oversized images; callers decide
    how to report that.
    """
    fp = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(fp, formats=formats) as im:
        im.load()
        mode, fmt = im.mode, im.format or ""
        oriented = ImageOps.exif_transpose(im)  # always a new, loaded image
    return DecodedImage(oriented, mode, fmt)


def load_rgb(path: Path) -> Image.Image:
    """A snapshot image as RGB, orientation applied (used by training, evaluation, audit)."""
    return decode_image(path).image.convert("RGB")
