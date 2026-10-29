"""On-disk dataset snapshots under ``<data_root>/datasets/<dataset_id>/``.

Layout::

    <dataset_id>/
      dataset.json   metadata (name, id, fingerprint, classes, counts, validation summary)
      report.json    statistics, issues, duplicate groups, samples
      files.json     one record per accepted image (sha256, dHash, gray std, size, ...)
      audits/<split_id>/   split audits (``visionbench.data.audit``; never inside splits/)
      raw/<class>/<file>   byte-for-byte copies of the accepted images
      splits/<split_id>/{manifest.json, split_info.json}

Imports are staged in a temporary sibling directory and renamed into place only after
validation succeeds; an existing dataset directory is never overwritten.

Format history: 1 (M1); 2 (D1) dimensions, gray std and dHash are computed after EXIF
orientation (``orientation: "exif_transposed"``, ``hash_version: 2`` in dataset.json and
files.json). Snapshots in an older format are refused with a request to re-import.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionbench import __version__
from visionbench.core.errors import DatasetExistsError, DatasetNotFoundError, DatasetValidationError
from visionbench.core.jsonio import read_json, write_json
from visionbench.data.grouping import build_groups
from visionbench.data.hashing import HASH_VERSION
from visionbench.data.imageio import ORIENTATION_POLICY
from visionbench.data.records import FileRecord, compute_fingerprint
from visionbench.data.report import build_report
from visionbench.data.scan import ScanConfig, check_requirements, scan_source

DATASET_FORMAT_VERSION = 2
DATASET_ID_RE = re.compile(r"^ds-[0-9a-f]{12}$")
MAX_NAME_LENGTH = 100


@dataclass(frozen=True)
class ImportResult:
    dataset_id: str
    path: Path
    dataset: dict[str, Any]
    report: dict[str, Any]


def make_dataset_id(fingerprint: str, name: str) -> str:
    return "ds-" + hashlib.sha256(f"{fingerprint}\n{name}".encode()).hexdigest()[:12]


def validate_name(name: str) -> str:
    name = name.strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        raise DatasetValidationError(f"dataset name must be 1-{MAX_NAME_LENGTH} characters")
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise DatasetValidationError("dataset name must not contain control characters")
    return name


def dataset_dir(datasets_dir: Path, dataset_id: str) -> Path:
    if not DATASET_ID_RE.match(dataset_id):
        raise DatasetNotFoundError(f"invalid dataset id {dataset_id!r} (expected ds-<12 hex>)")
    path = datasets_dir / dataset_id
    if not (path / "dataset.json").is_file():
        raise DatasetNotFoundError(f"dataset {dataset_id} not found")
    return path


class DatasetFormatError(DatasetValidationError):
    pass


def format_header() -> dict[str, Any]:
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "orientation": ORIENTATION_POLICY,
        "hash_version": HASH_VERSION,
    }


def check_format(meta: dict[str, Any], dataset_id: str, what: str) -> None:
    """Refuse snapshots written with another format / orientation policy / hash version."""
    found = {k: meta.get(k) for k in format_header()}
    if found != format_header():
        raise DatasetFormatError(
            f"dataset {dataset_id}: {what} has format_version={found['format_version']}, "
            f"orientation={found['orientation']!r}, hash_version={found['hash_version']}; this "
            f"version of VisionBench needs format_version={DATASET_FORMAT_VERSION}, "
            f"orientation={ORIENTATION_POLICY!r}, hash_version={HASH_VERSION}. Re-import the "
            "source folder (remove the old snapshot directory first, or import under a "
            "different name)"
        )


def load_dataset(datasets_dir: Path, dataset_id: str) -> dict[str, Any]:
    meta = read_json(dataset_dir(datasets_dir, dataset_id) / "dataset.json")
    check_format(meta, dataset_id, "dataset.json")
    return meta


def load_report(datasets_dir: Path, dataset_id: str) -> dict[str, Any]:
    return read_json(dataset_dir(datasets_dir, dataset_id) / "report.json")


def load_records(datasets_dir: Path, dataset_id: str) -> list[FileRecord]:
    data = read_json(dataset_dir(datasets_dir, dataset_id) / "files.json")
    check_format(data, dataset_id, "files.json")
    return [FileRecord.from_dict(d) for d in data["records"]]


@lru_cache(maxsize=16)
def _records_by_relpath(files_json: str, mtime_ns: int) -> dict[str, FileRecord]:
    data = read_json(Path(files_json))
    return {d["relpath"]: FileRecord.from_dict(d) for d in data["records"]}


def recorded_image(
    datasets_dir: Path, dataset_id: str, relpath: str
) -> tuple[FileRecord, Path] | None:
    """The record and file of ``relpath`` if it is a recorded member of the dataset, else None.

    ``relpath`` is only used as a dictionary key, never joined onto a filesystem path: the path
    is built from the stored record and must stay inside the snapshot's ``raw/`` directory.
    """
    root = dataset_dir(datasets_dir, dataset_id)
    files_json = root / "files.json"
    record = _records_by_relpath(str(files_json), files_json.stat().st_mtime_ns).get(relpath)
    if record is None:
        return None
    raw = (root / "raw").resolve()
    path = raw.joinpath(*record.relpath.split("/")).resolve()
    if raw not in path.parents or not path.is_file():
        return None
    return record, path


def list_datasets(datasets_dir: Path) -> list[dict[str, Any]]:
    if not datasets_dir.is_dir():
        return []
    out = []
    for path in sorted(datasets_dir.iterdir()):
        if DATASET_ID_RE.match(path.name) and (path / "dataset.json").is_file():
            out.append(read_json(path / "dataset.json"))
    return sorted(out, key=lambda d: (d["created_at"], d["id"]))


@dataclass(frozen=True)
class VerifyResult:
    dataset_id: str
    full: bool
    files_checked: int
    problems: list[tuple[str, str]]  # (relpath or "<dataset>", description), sorted

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_dataset(datasets_dir: Path, dataset_id: str, *, full: bool = False) -> VerifyResult:
    """Check a snapshot against its stored records.

    Fast mode: the fingerprint recomputed from ``files.json`` must equal ``dataset.json``;
    every recorded file must exist in ``raw/`` with the recorded size; no unrecorded files.
    ``full=True`` additionally re-hashes every file (SHA-256).
    """
    root = dataset_dir(datasets_dir, dataset_id)
    meta = load_dataset(datasets_dir, dataset_id)
    records = load_records(datasets_dir, dataset_id)
    problems: list[tuple[str, str]] = []
    if compute_fingerprint(records) != meta["fingerprint"]:
        problems.append(("<dataset>", "fingerprint of files.json does not match dataset.json"))

    raw = root / "raw"
    recorded = {r.relpath for r in records}
    for r in records:
        path = raw.joinpath(*r.relpath.split("/"))
        if not path.is_file():
            problems.append((r.relpath, "missing"))
            continue
        size = path.stat().st_size
        if size != r.size_bytes:
            problems.append((r.relpath, f"size {size} != recorded {r.size_bytes}"))
        elif full and hashlib.sha256(path.read_bytes()).hexdigest() != r.sha256:
            problems.append((r.relpath, "sha256 differs from recorded value"))
    if raw.is_dir():
        for path in raw.rglob("*"):
            rel = path.relative_to(raw).as_posix()
            if path.is_file() and rel not in recorded:
                problems.append((rel, "unexpected file not in files.json"))
    return VerifyResult(dataset_id, full, len(records), sorted(problems))


def import_dataset(
    src: Path,
    name: str,
    datasets_dir: Path,
    config: ScanConfig | None = None,
    *,
    source_description: str | None = None,
) -> ImportResult:
    """Validate ``src`` and snapshot it. ``source_description`` replaces the recorded absolute
    source path (e.g. ``"upload:<file name>"`` for API uploads, whose source is a staging dir)."""
    config = config or ScanConfig()
    name = validate_name(name)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    staging = datasets_dir / f".tmp-import-{uuid.uuid4().hex}"
    raw = staging / "raw"
    raw.mkdir(parents=True)

    def write_file(record: FileRecord, data: bytes) -> None:
        dest = raw.joinpath(*record.relpath.split("/"))
        dest.parent.mkdir(exist_ok=True)
        with open(dest, "xb") as f:  # "x": never overwrite
            f.write(data)

    try:
        scan = scan_source(src, config, sink=write_file)
        check_requirements(scan, config)
        fingerprint = compute_fingerprint(scan.records)
        dataset_id = make_dataset_id(fingerprint, name)
        final = datasets_dir / dataset_id
        if final.exists():
            raise DatasetExistsError(f"dataset {dataset_id} already exists; refusing to overwrite")

        grouping = build_groups(
            scan.records,
            dhash_threshold=config.dhash_threshold,
            low_info_std=config.low_info_std,
            popcount_min=config.low_info_popcount_min,
            popcount_max=config.low_info_popcount_max,
        )
        report = build_report(
            dataset_id=dataset_id,
            name=name,
            fingerprint=fingerprint,
            scan=scan,
            grouping=grouping,
            config=config,
        )
        counts = scan.class_counts()
        dataset = {
            **format_header(),
            "id": dataset_id,
            "name": name,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            # CLI-only metadata; never exposed via the API
            "source_path": source_description or str(Path(src).resolve()),
            "fingerprint": fingerprint,
            "classes": scan.classes,
            "class_counts": counts,
            "total_images": len(scan.records),
            "visionbench_version": __version__,
            "import_config": {
                "max_image_pixels": config.max_image_pixels,
                "min_images_per_class": config.min_images_per_class,
                "dhash_threshold": config.dhash_threshold,
                "low_info_std_threshold": config.low_info_std,
                "low_info_popcount_min": config.low_info_popcount_min,
                "low_info_popcount_max": config.low_info_popcount_max,
            },
            "validation": {
                "accepted": len(scan.records),
                "issue_counts": {k: v["count"] for k, v in report["issues"].items()},
            },
        }
        write_json(
            staging / "files.json",
            {**format_header(), "records": [r.to_dict() for r in scan.records]},
        )
        write_json(staging / "report.json", report)
        write_json(staging / "dataset.json", dataset)
        try:
            os.rename(staging, final)
        except OSError as exc:  # lost a race with a concurrent import of the same id
            raise DatasetExistsError(f"dataset {dataset_id} already exists") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ImportResult(dataset_id=dataset_id, path=final, dataset=dataset, report=report)
