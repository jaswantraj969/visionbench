"""Programmatic synthetic image datasets for tests and demos (no downloads, no binaries).

Shared by all test directories (tests/ is on sys.path via the root conftest); fixtures that
wrap these builders live in tests/conftest.py.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG = 64  # side of regular fixture images
SMALL_MAX_PIXELS = 10_000  # lowered limit used by the messy fixture (64*64 = 4096 passes)


def synthetic_image(rng: np.random.Generator, kind: str = "mixed", size: int = IMG) -> Image.Image:
    """A varied, information-rich image: random smooth field + gradient + shape + noise."""
    field_ = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    arr = np.asarray(field_, dtype=np.float64)
    if kind in ("gradient", "mixed"):
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:size, 0:size] / size
        arr = 0.6 * arr + 0.4 * 255 * ((np.cos(angle) * xx + np.sin(angle) * yy) % 1)[..., None]
    if kind in ("noise", "mixed"):
        arr = arr + rng.normal(0, 12, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind in ("shapes", "mixed"):
        draw = ImageDraw.Draw(img)
        x0, y0 = rng.integers(0, size // 2, 2)
        x1, y1 = x0 + rng.integers(8, size // 2), y0 + rng.integers(8, size // 2)
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=color)
    return img


def encode(img: Image.Image, fmt: str = "PNG", **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@dataclass
class Fixture:
    root: Path
    expected: dict[str, list[str]] = field(default_factory=dict)  # category -> relpaths
    file_link: bool = False  # alpha/link.png -> alpha/a3.png was created
    dir_link: str | None = None  # "symlink" | "junction" | None for linked_class -> gamma


def try_symlink(target: Path, link: Path, target_is_directory: bool = False) -> bool:
    try:
        os.symlink(target, link, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError):
        return False
    return True


def try_dir_link(target: Path, link: Path) -> str | None:
    """Directory symlink, or on Windows without symlink privilege an NTFS junction."""
    if try_symlink(target, link, target_is_directory=True):
        return "symlink"
    if os.name == "nt":
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError:
            return None
        return "junction"
    return None


def build_messy(root: Path, seed: int = 1234) -> Fixture:
    """3 classes with every issue category and every duplicate case the task lists."""
    rng = np.random.default_rng(seed)
    fx = Fixture(root=root)
    kinds = ["gradient", "shapes", "noise", "mixed", "mixed"]
    for label, prefix in (("alpha", "a"), ("beta", "b"), ("gamma", "g")):
        (root / label).mkdir(parents=True)
        for i, kind in enumerate(kinds):
            (root / label / f"{prefix}{i}.png").write_bytes(encode(synthetic_image(rng, kind)))

    alpha, beta = root / "alpha", root / "beta"
    # exact duplicate in the same class
    (alpha / "a0_copy.png").write_bytes((alpha / "a0.png").read_bytes())
    # near duplicate: resized + re-encoded as JPEG
    with Image.open(alpha / "a1.png") as im:
        near = im.convert("RGB").resize((48, 48), Image.Resampling.BILINEAR)
    (alpha / "a1_near.jpg").write_bytes(encode(near, "JPEG", quality=85))
    # cross-class exact duplicate -> label conflict
    (beta / "b_conflict.png").write_bytes((alpha / "a2.png").read_bytes())
    # two different uniform images (low information; must not be grouped perceptually)
    (alpha / "uniform_red.png").write_bytes(encode(Image.new("RGB", (IMG, IMG), (220, 20, 20))))
    (beta / "uniform_blue.png").write_bytes(encode(Image.new("RGB", (IMG, IMG), (20, 20, 220))))

    # rejected files
    jpeg = encode(synthetic_image(rng), "JPEG", quality=90)
    (alpha / "truncated.jpg").write_bytes(jpeg[: len(jpeg) * 6 // 10])
    (alpha / "empty.jpg").write_bytes(b"")
    (alpha / "notes.txt").write_text("not an image", encoding="utf-8")
    (alpha / "anim.gif").write_bytes(encode(synthetic_image(rng), "GIF"))
    (alpha / ".DS_Store").write_bytes(b"\x00\x01")
    (alpha / "Thumbs.db").write_bytes(b"\x00\x01")
    (alpha / ".hidden.png").write_bytes(encode(synthetic_image(rng)))
    (alpha / "big_warn.png").write_bytes(encode(synthetic_image(rng, size=110)))  # 12,100 px
    (alpha / "big_error.png").write_bytes(encode(synthetic_image(rng, size=150)))  # 22,500 px
    (alpha / "subdir").mkdir()
    (alpha / "subdir" / "inner.png").write_bytes(encode(synthetic_image(rng)))
    (root / "README.txt").write_text("stray file", encoding="utf-8")
    (root / ".DS_Store").write_bytes(b"\x00")
    (root / ".cache").mkdir()

    fx.expected = {
        "zero_byte": ["alpha/empty.jpg"],
        "unsupported_extension": ["alpha/anim.gif", "alpha/notes.txt"],
        "unreadable_or_corrupt": ["alpha/truncated.jpg"],
        "too_large_pixels": ["alpha/big_error.png", "alpha/big_warn.png"],
        "hidden_or_system_file": [
            ".DS_Store",
            "alpha/.DS_Store",
            "alpha/.hidden.png",
            "alpha/Thumbs.db",
        ],
        "nested_directory": ["alpha/subdir"],
        "invalid_class_name": [".cache"],
        "file_outside_class_folder": ["README.txt"],
        "symlink": [],
        "invalid_file_name": [],
        "path_collision": [],
    }

    fx.file_link = try_symlink(alpha / "a3.png", alpha / "link.png")
    fx.dir_link = try_dir_link(root / "gamma", root / "linked_class")
    if fx.file_link:
        fx.expected["symlink"].append("alpha/link.png")
    if fx.dir_link:
        fx.expected["symlink"].append("linked_class")
    return fx


def build_balanced(root: Path, per_class: int = 20, classes: int = 3, seed: int = 99) -> Path:
    rng = np.random.default_rng(seed)
    for c in range(classes):
        d = root / f"class_{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            (d / f"img_{i:03d}.png").write_bytes(encode(synthetic_image(rng)))
    return root


def make_images(folder: Path, count: int, seed: int = 0, prefix: str = "img") -> None:
    """Write ``count`` unique synthetic PNGs into ``folder`` (created if needed)."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (folder / f"{prefix}_{i:03d}.png").write_bytes(encode(synthetic_image(rng)))


def tilted_gradient(tilt: float, color: tuple[float, float, float], size: int = IMG) -> Image.Image:
    """Smooth vertical gradient with a slight horizontal tilt (sky-like).

    Its gray std is high (~38), so the std guard does not catch it, but its dHash is sparse
    (tilt 1.1 -> popcount 3; tilt 1.2 -> 3, two bits apart from tilt 1.1).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    g = 40 + 160 * yy / size + tilt * xx / size
    return Image.fromarray(np.clip(np.stack([g * c for c in color], -1), 0, 255).astype(np.uint8))


def near_flat_noise(seed: int, sd: float = 1.5, size: int = IMG) -> Image.Image:
    """Uniform gray plus faint noise: gray std ~1, dHash popcount ~15 (noise-driven bits)."""
    rng = np.random.default_rng(seed)
    arr = 128 + rng.normal(0, sd, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build_low_info(root: Path, seed: int = 7) -> Path:
    """Two classes with normal images plus two smooth gradients (one per class) and a
    near-flat noisy image. Without the popcount guard the gradients form a cross-class group."""
    make_images(root / "sky", 3, seed=seed)
    make_images(root / "sea", 3, seed=seed + 1)
    (root / "sky" / "gradient_a.png").write_bytes(encode(tilted_gradient(1.1, (1.0, 0.8, 0.6))))
    (root / "sea" / "gradient_b.png").write_bytes(encode(tilted_gradient(1.2, (0.6, 0.8, 1.0))))
    (root / "sky" / "flat_noise.png").write_bytes(encode(near_flat_noise(seed)))
    return root


SHAPES = ("circle", "square", "triangle")


def shape_image(rng: np.random.Generator, shape: str, size: int = IMG) -> Image.Image:
    """A filled shape with random size, position and colors on a noisy random background."""
    bg = tuple(int(c) for c in rng.integers(0, 256, 3))
    img = Image.new("RGB", (size, size), bg)
    arr = np.asarray(img, dtype=np.float64) + rng.normal(0, 10, (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    r = int(rng.integers(size // 6, size // 3))
    cx, cy = (int(v) for v in rng.integers(r + 2, size - r - 2, 2))
    fg = tuple(int((b + rng.integers(90, 166)) % 256) for b in bg)  # contrast with background
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fg)
    else:
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fg)
    return img


def build_shapes(root: Path, per_class: int, seed: int = 0, size: int = IMG) -> Path:
    """Learnable synthetic dataset: one class per shape in :data:`SHAPES`."""
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        d = root / shape
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            (d / f"{shape}_{i:04d}.png").write_bytes(encode(shape_image(rng, shape, size)))
    return root


def write_zip(
    path: Path,
    members: list[tuple[str, bytes]],
    *,
    compression: int = 0,
    raw_names: bool = False,
) -> Path:
    """Write a ZIP with exactly these member names (``raw_names`` bypasses zipfile's
    separator normalisation so backslashes etc. are stored verbatim)."""
    import zipfile

    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for name, data in members:
            info = zipfile.ZipInfo("placeholder")
            info.filename = name  # set after __init__: stored exactly as given
            info.compress_type = compression
            if not raw_names:
                info = zipfile.ZipInfo(name)
                info.compress_type = compression
            zf.writestr(info, data)
    return path


def shapes_zip(
    path: Path, per_class: int = 4, size: int = 32, top: str | None = None, seed: int = 0
) -> Path:
    """A valid dataset ZIP of synthetic shapes (optionally inside one top-level folder)."""
    rng = np.random.default_rng(seed)
    members = []
    for shape in SHAPES:
        for i in range(per_class):
            rel = f"{shape}/{shape}_{i:03d}.png"
            members.append((f"{top}/{rel}" if top else rel, encode(shape_image(rng, shape, size))))
    return write_zip(path, members)
