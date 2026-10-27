"""Scan and validate a ``<root>/<class>/<image>`` source folder.

Every problem is collected as an :class:`Issue`; scanning never stops at the first one.
Entries are visited in sorted (NFC-normalized) name order, and symlinks / reparse points are
reported and never followed.
"""

from __future__ import annotations

import hashlib
import io
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from visionbench.core.errors import DatasetValidationError
from visionbench.data.hashing import dhash, format_hash
from visionbench.data.imageio import PILLOW_FORMATS, decode_image
from visionbench.data.records import FileRecord, Issue, IssueCategory, normalize_name

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SYSTEM_FILE_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})

# Windows file attribute bits (st_file_attributes); absent on POSIX.
_ATTR_HIDDEN = 0x2
_ATTR_SYSTEM = 0x4
_ATTR_REPARSE_POINT = 0x400

GRAY_STD_THUMBNAIL = 256  # gray std is measured on a thumbnail with this max side


@dataclass(frozen=True)
class ScanConfig:
    max_image_pixels: int = 50_000_000
    min_images_per_class: int = 3
    dhash_threshold: int = 5
    low_info_std: float = 3.0
    low_info_popcount_min: int = 8
    low_info_popcount_max: int = 56
    samples_per_class: int = 8
    sample_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_image_pixels <= 0:
            raise DatasetValidationError("max_image_pixels must be > 0")
        if self.min_images_per_class < 1:
            raise DatasetValidationError("min_images_per_class must be >= 1")
        if not 0 <= self.dhash_threshold <= 64:
            raise DatasetValidationError("dhash_threshold must be in [0, 64]")
        if self.low_info_std < 0:
            raise DatasetValidationError("low_info_std must be >= 0")
        if not 0 <= self.low_info_popcount_min <= self.low_info_popcount_max <= 64:
            raise DatasetValidationError(
                "need 0 <= low_info_popcount_min <= low_info_popcount_max <= 64"
            )
        if self.samples_per_class < 0:
            raise DatasetValidationError("samples_per_class must be >= 0")


@dataclass
class ScanResult:
    classes: list[str]  # valid class folder names, sorted (may include classes with 0 images)
    records: list[FileRecord]  # accepted images, sorted by relpath
    issues: list[Issue]  # sorted by (category, relpath)

    def class_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(self.classes, 0)
        for r in self.records:
            counts[r.label] += 1
        return counts


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    mode: str
    format: str
    dhash: int
    gray_std: float


class ImageRejected(Exception):
    def __init__(self, category: IssueCategory, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


FileSink = Callable[[FileRecord, bytes], None]


@contextmanager
def _pillow_limits(max_pixels: int) -> Iterator[None]:
    """Apply our pixel limit explicitly (Pillow warns above it and errors above 2x; the warning
    is escalated to an error) and make sure truncated images are never silently accepted.

    Mutates Pillow module globals for the duration, so scanning is not thread-safe.
    """
    old_max, old_truncated = Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES
    Image.MAX_IMAGE_PIXELS = max_pixels
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            yield
    finally:
        Image.MAX_IMAGE_PIXELS, ImageFile.LOAD_TRUNCATED_IMAGES = old_max, old_truncated


def inspect_image(data: bytes, max_pixels: int) -> ImageInfo:
    """Fully decode ``data``; raise :class:`ImageRejected` if it is too large or corrupt.

    ``verify()`` alone misses truncation, so the image is then decoded completely as well,
    through :func:`visionbench.data.imageio.decode_image`: dimensions, gray std and dHash are
    those of the EXIF-transposed image (orientation policy ``exif_transposed``).
    """
    with _pillow_limits(max_pixels):
        try:
            with Image.open(io.BytesIO(data), formats=PILLOW_FORMATS) as im:
                im.verify()
            decoded = decode_image(data)
            width, height = decoded.image.size
            mode, fmt = decoded.mode, decoded.format
            gray = decoded.image.convert("L")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageRejected(IssueCategory.TOO_LARGE_PIXELS, str(exc)) from exc
        except Exception as exc:  # any decoder failure means the file is unusable
            raise ImageRejected(
                IssueCategory.UNREADABLE_OR_CORRUPT, f"{type(exc).__name__}: {exc}"
            ) from exc

    thumb = gray.copy()
    thumb.thumbnail((GRAY_STD_THUMBNAIL, GRAY_STD_THUMBNAIL))
    gray_std = round(float(np.asarray(thumb, dtype=np.float64).std()), 4)
    return ImageInfo(width, height, mode, fmt, dhash(gray), gray_std)


def _sorted_entries(path: str | os.PathLike[str]) -> list[os.DirEntry[str]]:
    with os.scandir(path) as it:
        return sorted(it, key=lambda e: (normalize_name(e.name), e.name))


def _attributes(entry: os.DirEntry[str]) -> int:
    return getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)


def _is_link(entry: os.DirEntry[str]) -> bool:
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)  # Python >= 3.12
    if is_junction is not None and is_junction():
        return True
    return bool(_attributes(entry) & _ATTR_REPARSE_POINT)


def _is_hidden_or_system(entry: os.DirEntry[str], name: str) -> bool:
    return (
        name.startswith(".")
        or name.casefold() in SYSTEM_FILE_NAMES
        or bool(_attributes(entry) & (_ATTR_HIDDEN | _ATTR_SYSTEM))
    )


def _has_bad_chars(name: str) -> bool:
    return "/" in name or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name)


def class_name_problem(name: str) -> str | None:
    if not name.strip():
        return "class name is empty"
    if name.startswith("."):
        return "class name starts with a dot"
    if _has_bad_chars(name):
        return "class name contains a path separator or control character"
    return None


def scan_source(src: Path, config: ScanConfig, sink: FileSink | None = None) -> ScanResult:
    """Validate ``src``; call ``sink(record, data)`` for each accepted image.

    ``data`` is exactly the bytes that were hashed and decoded, so a sink that writes them
    produces a snapshot whose content matches the recorded SHA-256 values.
    """
    src = Path(src)
    try:
        attrs = getattr(os.lstat(src), "st_file_attributes", 0)
    except OSError:
        attrs = 0
    if src.is_symlink() or attrs & _ATTR_REPARSE_POINT:  # is_symlink() misses NTFS junctions
        raise DatasetValidationError(f"source directory is a link (not followed): {src}")
    if not src.is_dir():
        raise DatasetValidationError(f"source directory does not exist: {src}")

    issues: list[Issue] = []
    records: list[FileRecord] = []
    classes: list[str] = []
    seen: dict[str, str] = {}  # casefolded NFC relpath -> first relpath claiming it

    def claim(rel: str) -> bool:
        """Reserve a snapshot path; names equal after NFC + casefold would collide on disk."""
        key = rel.casefold()
        if key in seen:
            detail = f"collides with {seen[key]!r} after Unicode NFC / case normalization"
            issues.append(Issue(IssueCategory.PATH_COLLISION, rel, detail))
            return False
        seen[key] = rel
        return True

    for entry in _sorted_entries(src):
        name = normalize_name(entry.name)
        if _is_link(entry):
            issues.append(Issue(IssueCategory.SYMLINK, name, "symlinks are not followed"))
        elif entry.is_dir(follow_symlinks=False):
            problem = class_name_problem(name)
            if problem:
                issues.append(Issue(IssueCategory.INVALID_CLASS_NAME, name, problem))
            elif claim(name):
                classes.append(name)
                _scan_class(entry.path, name, config, sink, issues, records, claim)
        elif _is_hidden_or_system(entry, name):
            issues.append(Issue(IssueCategory.HIDDEN_OR_SYSTEM_FILE, name, "skipped"))
        else:
            issues.append(
                Issue(
                    IssueCategory.FILE_OUTSIDE_CLASS_FOLDER, name, "files must be in class folders"
                )
            )

    records.sort(key=lambda r: r.relpath)
    issues.sort()
    return ScanResult(classes=classes, records=records, issues=issues)


def _scan_class(
    class_dir: str,
    label: str,
    config: ScanConfig,
    sink: FileSink | None,
    issues: list[Issue],
    records: list[FileRecord],
    claim: Callable[[str], bool],
) -> None:
    for entry in _sorted_entries(class_dir):
        name = normalize_name(entry.name)
        rel = f"{label}/{name}"

        def report(category: IssueCategory, detail: str, rel: str = rel) -> None:
            issues.append(Issue(category, rel, detail))

        if _is_link(entry):
            report(IssueCategory.SYMLINK, "symlinks are not followed")
        elif entry.is_dir(follow_symlinks=False):
            report(
                IssueCategory.NESTED_DIRECTORY, "sub-directories in class folders are unsupported"
            )
        elif _is_hidden_or_system(entry, name):
            report(IssueCategory.HIDDEN_OR_SYSTEM_FILE, "skipped")
        elif _has_bad_chars(name):
            report(IssueCategory.INVALID_FILE_NAME, "contains a backslash or control character")
        elif Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            report(IssueCategory.UNSUPPORTED_EXTENSION, f"extension {Path(name).suffix!r}")
        elif not entry.is_file(follow_symlinks=False):
            report(IssueCategory.UNREADABLE_OR_CORRUPT, "not a regular file")
        elif not claim(rel):
            continue
        elif entry.stat(follow_symlinks=False).st_size == 0:
            report(IssueCategory.ZERO_BYTE, "file is empty")
        else:
            try:
                data = Path(entry.path).read_bytes()
                info = inspect_image(data, config.max_image_pixels)
            except ImageRejected as exc:
                report(exc.category, exc.detail)
                continue
            except OSError as exc:
                report(IssueCategory.UNREADABLE_OR_CORRUPT, f"{type(exc).__name__}: {exc.strerror}")
                continue
            record = FileRecord(
                relpath=rel,
                label=label,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                width=info.width,
                height=info.height,
                mode=info.mode,
                format=info.format,
                dhash=format_hash(info.dhash),
                gray_std=info.gray_std,
            )
            records.append(record)
            if sink is not None:
                sink(record, data)


def check_requirements(result: ScanResult, config: ScanConfig) -> None:
    """Raise :class:`DatasetValidationError` unless there are >= 2 classes, each with at least
    ``min_images_per_class`` accepted images."""
    problems = []
    if len(result.classes) < 2:
        problems.append(
            f"at least 2 classes are required, found {len(result.classes)}: {result.classes}"
        )
    low = {c: n for c, n in result.class_counts().items() if n < config.min_images_per_class}
    if low:
        listed = ", ".join(f"{c!r} ({n})" for c, n in sorted(low.items()))
        problems.append(
            f"classes below the minimum of {config.min_images_per_class} valid images: {listed}"
        )
    if problems:
        raise DatasetValidationError("; ".join(problems))
