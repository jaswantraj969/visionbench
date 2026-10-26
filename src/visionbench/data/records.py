"""Per-file records, validation issues, and the dataset fingerprint."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class IssueCategory(StrEnum):
    ZERO_BYTE = "zero_byte"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNREADABLE_OR_CORRUPT = "unreadable_or_corrupt"
    TOO_LARGE_PIXELS = "too_large_pixels"
    SYMLINK = "symlink"
    HIDDEN_OR_SYSTEM_FILE = "hidden_or_system_file"
    NESTED_DIRECTORY = "nested_directory"
    INVALID_CLASS_NAME = "invalid_class_name"
    INVALID_FILE_NAME = "invalid_file_name"
    FILE_OUTSIDE_CLASS_FOLDER = "file_outside_class_folder"
    PATH_COLLISION = "path_collision"


@dataclass(frozen=True, order=True)
class Issue:
    category: IssueCategory
    relpath: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"category": str(self.category), "relpath": self.relpath, "detail": self.detail}


@dataclass(frozen=True)
class FileRecord:
    """One accepted image. ``relpath`` is ``<label>/<file name>``, POSIX-style, NFC."""

    relpath: str
    label: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    mode: str
    format: str
    dhash: str  # 16 hex chars (64-bit dHash); a string so JSON consumers never lose precision
    gray_std: float

    @property
    def dhash_int(self) -> int:
        return int(self.dhash, 16)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FileRecord:
        return cls(**d)


def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def compute_fingerprint(records: Iterable[FileRecord]) -> str:
    """SHA-256 over sorted ``relpath\\tlabel\\tsha256`` lines.

    Independent of listing order; relpaths are already POSIX/NFC, so independent of OS
    separators too. Tabs and newlines are rejected in names, so lines are unambiguous.
    """
    lines = sorted(f"{r.relpath}\t{r.label}\t{r.sha256}\n" for r in records)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
