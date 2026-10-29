"""A small synthetic demo dataset (``visionbench demo-data``): coloured shapes on noisy backgrounds.

Generated locally from a seed (no download, not EuroSAT) so that the Quick Start, the
end-to-end test and the container can be exercised without the benchmark data. Results on it
mean nothing about real imagery.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")


def shape_png(rng: np.random.Generator, shape: str, size: int = 32) -> bytes:
    bg = tuple(int(c) for c in rng.integers(0, 256, 3))
    arr = np.asarray(Image.new("RGB", (size, size), bg), dtype=np.float64)
    arr = arr + rng.normal(0, 10, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    r = int(rng.integers(size // 6, size // 3))
    cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, 2))
    fg = tuple(int((b + rng.integers(90, 166)) % 256) for b in bg)
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def demo_images(per_class: int, seed: int = 0) -> list[tuple[str, bytes]]:
    """``(relpath, png bytes)`` pairs, ``<class>/<class>_<i>.png``, deterministic for a seed."""
    rng = np.random.default_rng(seed)
    return [
        (f"{shape}/{shape}_{i:03d}.png", shape_png(rng, shape))
        for shape in SHAPES
        for i in range(per_class)
    ]


def write_zip(path: Path, per_class: int, seed: int = 0) -> None:
    """A ZIP with one top-level ``shapes/`` folder (the layout the upload endpoint accepts)."""
    with zipfile.ZipFile(path, "w") as zf:
        for rel, data in demo_images(per_class, seed):
            zf.writestr(f"shapes/{rel}", data)


def write_demo(out: Path, per_class: int, seed: int = 0) -> tuple[Path, Path]:
    """Write ``<out>/shapes/<class>/*.png`` (for ``dataset import``) and ``<out>/shapes.zip``
    (for the web upload). ``out/shapes`` must not exist yet."""
    folder = out / "shapes"
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    for rel, data in demo_images(per_class, seed):
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    zip_path = out / "shapes.zip"
    write_zip(zip_path, per_class, seed)
    return folder, zip_path
