"""Hardened ZIP dataset archives: validate everything, then extract, then import, then clean up.

:func:`validate_archive` reads only the central directory and rejects the WHOLE archive (raising
:class:`ArchiveRejected` with a machine-readable ``code``) if any member has:

- an absolute path, a drive letter, a UNC prefix, a ``..`` component, a backslash, an empty or
  ``.`` component, a NUL/control character or one of ``<>:"|?*`` (``invalid_name``);
- a Windows-reserved name (CON, PRN, AUX, NUL, COM1-9, LPT1-9; with or without extension) or a
  component ending in a dot or space;
- a symlink type in ``external_attr``; the encryption flag;
- a case-insensitive (NFC) path collision with another member, including file-vs-directory;
- more than ``max_members`` members, more than ``max_uncompressed_bytes`` in total, or a
  compression ratio above ``max_ratio`` (per member >= 1 MiB, and overall >= 1 MiB).

Layout: the archive root holds the class folders, or a single top-level folder holds them;
an archive whose files all sit at the root is rejected (``bad_layout``). ``__MACOSX/`` entries
are validated and then skipped. Nested archives are extracted as plain files and reported by the
dataset scan as unsupported (they are never opened).

:func:`extract_archive` writes into a fresh staging directory, checks that every target resolves
inside it, and enforces the declared and total sizes while streaming. :func:`import_archive`
runs the normal dataset import (all M1 validation) and always deletes the staging directory and
the uploaded file.
"""

from __future__ import annotations

import re
import shutil
import stat
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
from visionbench.data.scan import ScanConfig
from visionbench.data.store import ImportResult, import_dataset

ZIP_MAGIC = b"PK\x03\x04"
EMPTY_ZIP_MAGIC = b"PK\x05\x06"
RATIO_MIN_BYTES = 1 << 20  # ratios are only meaningful for reasonably large data
IGNORED_PREFIX = "__macosx"
WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_INVALID_CHARS = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_DRIVE = re.compile(r"^[A-Za-z]:")
UPLOAD_ID = re.compile(r"^up-[0-9a-f]{32}$")


class ArchiveRejected(VisionBenchError):
    def __init__(self, code: str, message: str, member: str | None = None) -> None:
        super().__init__(message if member is None else f"{message}: {member!r}")
        self.code = code
        self.member = member


class ImportPathError(VisionBenchError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 50_000
    max_uncompressed_bytes: int = 4096 << 20
    max_ratio: float = 100.0

    @classmethod
    def from_settings(cls, s: Settings) -> ArchiveLimits:
        return cls(s.zip_max_members, s.zip_max_uncompressed_mb << 20, s.zip_max_ratio)


@dataclass(frozen=True)
class Member:
    info: zipfile.ZipInfo
    parts: tuple[str, ...]


@dataclass
class ArchivePlan:
    files: list[Member]
    layout: str  # "class_root" | "single_top_folder"
    strip_prefix: str | None
    total_uncompressed: int
    ignored: list[str] = field(default_factory=list)


def check_magic(path: Path) -> None:
    with open(path, "rb") as f:
        head = f.read(4)
    if head == EMPTY_ZIP_MAGIC:
        raise ArchiveRejected("empty_archive", "the ZIP archive contains no files")
    if head != ZIP_MAGIC:
        raise ArchiveRejected("not_a_zip", "the upload is not a ZIP archive (bad magic bytes)")


def _check_name(raw: str) -> tuple[tuple[str, ...], bool]:
    """Return (path components, is_directory) for a member name, or raise."""
    if "\x00" in raw:
        raise ArchiveRejected("invalid_name", "member name contains a NUL byte", raw)
    if "\\" in raw:
        raise ArchiveRejected("backslash", "member name contains a backslash", raw)
    if raw.startswith("//"):
        raise ArchiveRejected("unc_path", "member name is a UNC path", raw)
    if raw.startswith("/"):
        raise ArchiveRejected("absolute_path", "member name is an absolute path", raw)
    if _DRIVE.match(raw):
        raise ArchiveRejected("drive_letter", "member name starts with a drive letter", raw)
    is_dir = raw.endswith("/")
    parts = tuple(raw[:-1].split("/") if is_dir else raw.split("/"))
    for part in parts:
        if part == "..":
            raise ArchiveRejected("parent_traversal", "member name contains '..'", raw)
        if part in ("", "."):
            raise ArchiveRejected("invalid_name", "member name has an empty or '.' component", raw)
        if _INVALID_CHARS.search(part):
            raise ArchiveRejected("invalid_name", "member name has an invalid character", raw)
        if part[-1] in ". ":
            raise ArchiveRejected(
                "trailing_dot_or_space", "a name component ends with a dot or space", raw
            )
        if part.split(".")[0].rstrip().upper() in WINDOWS_RESERVED:
            raise ArchiveRejected("reserved_name", "member name is a Windows-reserved name", raw)
    return parts, is_dir


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def validate_archive(path: Path, limits: ArchiveLimits) -> ArchivePlan:
    check_magic(path)
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArchiveRejected("corrupt_zip", f"the ZIP archive cannot be read ({exc})") from exc
    with zf:
        infos = zf.infolist()
    if len(infos) > limits.max_members:
        raise ArchiveRejected(
            "too_many_members", f"{len(infos)} members exceed the limit of {limits.max_members}"
        )
    seen: dict[str, str] = {}
    dir_keys: set[str] = set()
    file_keys: dict[str, str] = {}
    files: list[Member] = []
    ignored: list[str] = []
    total = total_compressed = 0
    for info in infos:
        raw = info.orig_filename
        parts, is_dir = _check_name(raw)
        if _is_symlink(info):
            raise ArchiveRejected("symlink", "member is a symbolic link", raw)
        if info.flag_bits & 0x1:
            raise ArchiveRejected("encrypted", "member is encrypted", raw)
        key = unicodedata.normalize("NFC", "/".join(parts)).casefold()
        if key in seen:
            raise ArchiveRejected(
                "case_collision", f"member collides with {seen[key]!r} (case/Unicode)", raw
            )
        seen[key] = raw
        for i in range(1, len(parts)):  # every ancestor is an implied directory
            dir_keys.add(unicodedata.normalize("NFC", "/".join(parts[:i])).casefold())
        if is_dir:
            dir_keys.add(key)
            continue
        file_keys[key] = raw
        if info.file_size >= RATIO_MIN_BYTES and (
            info.file_size / max(info.compress_size, 1) > limits.max_ratio
        ):
            raise ArchiveRejected(
                "compression_ratio",
                f"compression ratio above {limits.max_ratio:g} (possible zip bomb)",
                raw,
            )
        total += info.file_size
        total_compressed += info.compress_size
        if total > limits.max_uncompressed_bytes:
            raise ArchiveRejected(
                "too_large",
                f"uncompressed size exceeds {limits.max_uncompressed_bytes} bytes",
            )
        if parts[0].casefold() == IGNORED_PREFIX:
            ignored.append(raw)
            continue
        files.append(Member(info, parts))
    clash = sorted(set(file_keys) & dir_keys)
    if clash:
        raise ArchiveRejected(
            "case_collision", "a file collides with a directory path", file_keys[clash[0]]
        )
    if total >= RATIO_MIN_BYTES and total / max(total_compressed, 1) > limits.max_ratio:
        raise ArchiveRejected(
            "compression_ratio",
            f"overall compression ratio above {limits.max_ratio:g} (possible zip bomb)",
        )
    if not files:
        raise ArchiveRejected("empty_archive", "the ZIP archive contains no files")

    tops = {m.parts[0] for m in files}
    root_files = [m for m in files if len(m.parts) == 1]
    if len(tops) == 1 and not root_files and any(len(m.parts) >= 3 for m in files):
        return ArchivePlan(files, "single_top_folder", files[0].parts[0], total, ignored)
    if any(len(m.parts) >= 2 for m in files):
        return ArchivePlan(files, "class_root", None, total, ignored)
    raise ArchiveRejected(
        "bad_layout",
        "expected class folders at the archive root (or one top-level folder containing them); "
        "all files are at the root",
    )


def extract_archive(path: Path, plan: ArchivePlan, staging: Path, limits: ArchiveLimits) -> Path:
    """Extract ``plan.files`` into a NEW ``staging`` dir; return the dataset root inside it."""
    staging.mkdir(parents=True, exist_ok=False)
    root = staging.resolve()
    written_total = 0
    with zipfile.ZipFile(path) as zf:
        for m in plan.files:
            target = staging.joinpath(*m.parts)
            resolved = target.resolve()
            if root not in resolved.parents:
                raise ArchiveRejected(
                    "extraction_escape",
                    "member would be written outside the staging directory",
                    m.info.orig_filename,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            try:
                with zf.open(m.info) as src, open(target, "xb") as dst:
                    while chunk := src.read(1 << 20):
                        written += len(chunk)
                        written_total += len(chunk)
                        if written > m.info.file_size:
                            raise ArchiveRejected(
                                "size_mismatch",
                                "member is larger than declared",
                                m.info.orig_filename,
                            )
                        if written_total > limits.max_uncompressed_bytes:
                            raise ArchiveRejected(
                                "too_large", "extracted data exceeds the uncompressed limit"
                            )
                        dst.write(chunk)
            except FileExistsError as exc:
                raise ArchiveRejected(
                    "case_collision",
                    "member collides with an extracted file",
                    m.info.orig_filename,
                ) from exc
            except zipfile.BadZipFile as exc:
                raise ArchiveRejected(
                    "corrupt_zip", f"member cannot be decompressed ({exc})", m.info.orig_filename
                ) from exc
    return staging / plan.strip_prefix if plan.strip_prefix else staging


def new_upload_id() -> str:
    return f"up-{uuid.uuid4().hex}"


def upload_path(settings: Settings, upload_id: str) -> Path:
    if not UPLOAD_ID.match(upload_id):
        raise ArchiveRejected("invalid_upload_id", "invalid upload id")
    return settings.data_dir("tmp") / "uploads" / f"{upload_id}.zip"


def staging_path(settings: Settings, upload_id: str) -> Path:
    upload_path(settings, upload_id)  # validates the id
    return settings.data_dir("tmp") / f"staging-{upload_id}"


def cleanup_upload(settings: Settings, upload_id: str) -> None:
    """Remove an upload's ZIP and staging directory (idempotent)."""
    shutil.rmtree(staging_path(settings, upload_id), ignore_errors=True)
    upload_path(settings, upload_id).unlink(missing_ok=True)


def import_archive(
    settings: Settings,
    upload_id: str,
    name: str,
    config: ScanConfig,
    *,
    original_filename: str | None = None,
) -> ImportResult:
    """Validate -> extract -> import an uploaded ZIP; the upload and staging dir are always
    removed afterwards (success or failure)."""
    try:
        zip_path = upload_path(settings, upload_id)
        limits = ArchiveLimits.from_settings(settings)
        plan = validate_archive(zip_path, limits)
        root = extract_archive(zip_path, plan, staging_path(settings, upload_id), limits)
        label = f"upload:{original_filename}" if original_filename else "upload"
        return import_dataset(
            root, name, settings.data_dir("datasets"), config, source_description=label
        )
    finally:
        cleanup_upload(settings, upload_id)


def resolve_import_path(settings: Settings, relpath: str) -> Path:
    """Resolve a server-side import path; it must lie inside ``import_root`` after resolving
    symlinks/junctions. Raises :class:`ImportPathError` otherwise."""
    if settings.import_root is None:
        raise ImportPathError("path_import_disabled", "server-side path imports are disabled")
    root = settings.import_root.expanduser().resolve(strict=True)
    try:
        candidate = (root / relpath).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImportPathError("path_not_found", "import path does not exist") from exc
    if candidate != root and root not in candidate.parents:
        raise ImportPathError("path_outside_import_root", "import path is outside import_root")
    if not candidate.is_dir():
        raise ImportPathError("path_not_a_directory", "import path is not a directory")
    return candidate


def import_relpath(settings: Settings, path: str) -> str:
    """Normalize a user-supplied server path to a POSIX path relative to ``import_root``."""
    if settings.import_root is None:
        raise ImportPathError("path_import_disabled", "server-side path imports are disabled")
    root = settings.import_root.expanduser().resolve(strict=True)
    resolved = resolve_import_path(settings, str(Path(path)))
    return resolved.relative_to(root).as_posix() if resolved != root else "."
