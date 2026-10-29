"""Materialize every condition of a suite once, as lossless PNGs, with a hashed manifest.

Layout: ``<data_root>/cache/corruptions/<dataset_id>/<split_id>/<suite>/<corruption>/<severity>/``
with ``condition.json`` and one ``<class>/<stem>.png`` per image. Every model evaluates these
exact files.

- A condition that already exists is not rewritten: every image is corrupted again and its bytes
  are compared with the stored file and with the manifest, so a rerun proves the files are
  byte-identical (or fails).
- New conditions are written to a temporary directory and renamed into place.
- ``split="test"`` is refused unless ``test_images_unlocked`` (M8).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image
from PIL import features as pil_features

from visionbench import __version__
from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json, write_json
from visionbench.data.imageio import load_rgb
from visionbench.data.store import dataset_dir, load_dataset, verify_dataset
from visionbench.evaluation.data import select_eval_entries
from visionbench.robustness import corruptions
from visionbench.robustness.suite import (
    Condition,
    RobustnessError,
    Suite,
    TestSplitLockedError,
    image_seed,
)

CONDITION_FORMAT = 1
UNHASHED_FIELDS = ("created_at", "condition_hash")


@dataclass(frozen=True)
class SplitSource:
    dataset_id: str
    split_id: str
    split: str
    manifest_sha256: str
    entries: list[dict[str, Any]]  # manifest entries of the split, manifest order
    raw_root: Path


@dataclass(frozen=True)
class ConditionResult:
    key: str
    status: str  # "written" | "verified"
    condition_hash: str
    images: int
    path: Path


def check_split_allowed(split: str, settings: Settings) -> None:
    if split == "test" and not settings.test_images_unlocked:
        raise TestSplitLockedError(
            "robustness on the TEST split is locked until the final test evaluation (M8); "
            "set VISIONBENCH_TEST_IMAGES_UNLOCKED=true only then"
        )


def open_split(settings: Settings, dataset_id: str, split_id: str, split: str) -> SplitSource:
    check_split_allowed(split, settings)
    datasets = settings.data_dir("datasets")
    meta = load_dataset(datasets, dataset_id)
    check = verify_dataset(datasets, dataset_id, full=False)
    if not check.ok:
        raise RobustnessError(f"dataset {dataset_id} failed verification: {check.problems[:5]}")
    path = dataset_dir(datasets, dataset_id) / "splits" / split_id / "manifest.json"
    if not path.is_file():
        raise RobustnessError(f"split {split_id} not found for dataset {dataset_id}")
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("dataset_fingerprint") != meta["fingerprint"]:
        raise RobustnessError("split manifest fingerprint differs from the dataset's")
    return SplitSource(
        dataset_id,
        split_id,
        split,
        hashlib.sha256(raw).hexdigest(),
        select_eval_entries(manifest, split),
        dataset_dir(datasets, dataset_id) / "raw",
    )


def suite_root(settings: Settings, src: SplitSource, suite: Suite) -> Path:
    return settings.data_dir("cache") / "corruptions" / src.dataset_id / src.split_id / suite.name


def condition_dir(root: Path, cond: Condition) -> Path:
    return root / cond.corruption / cond.severity_str


def corrupted_relpath(relpath: str) -> str:
    label, name = relpath.split("/", 1)
    return f"{label}/{name.rsplit('.', 1)[0]}.png"


def library_versions() -> dict[str, Any]:
    return {
        "visionbench": __version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "libjpeg": pil_features.version("jpg"),
        "libjpeg_turbo": pil_features.check_feature("libjpeg_turbo"),
        "zlib": pil_features.version("zlib"),
    }


def condition_hash(doc: dict[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k not in UNHASHED_FIELDS}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corrupt_one(src: SplitSource, suite: Suite, cond: Condition, entry: dict[str, Any]) -> bytes:
    """PNG bytes of one corrupted image (the source file's sha256 is checked first)."""
    path = src.raw_root.joinpath(*entry["relpath"].split("/"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise RobustnessError(f"source image changed: {entry['relpath']}")
    img = np.asarray(load_rgb(path), dtype=np.uint8)
    seed = image_seed(
        suite.name, cond.corruption, cond.severity_str, suite.base_seed, entry["relpath"]
    )
    out = corruptions.apply(img, cond.corruption, cond.severity, seed)
    buf = io.BytesIO()
    Image.fromarray(out, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _doc(src: SplitSource, suite: Suite, cond: Condition, entries: list[dict[str, Any]]) -> dict:
    doc = {
        "format": CONDITION_FORMAT,
        "suite": suite.name,
        "suite_sha256": suite.sha256,
        "corruption": cond.corruption,
        "parameter": cond.parameter,
        "severity": cond.severity_str,
        "base_seed": suite.base_seed,
        "dataset_id": src.dataset_id,
        "split_id": src.split_id,
        "split": src.split,
        "manifest_sha256": src.manifest_sha256,
        "applied_to": "original decoded RGB image (EXIF applied), before the val transform",
        "quantization": "clip [0,1], rint(x*255) uint8 (jpeg: Pillow decode)",
        "storage": "PNG (Pillow defaults)",
        "versions": library_versions(),
        "entries": sorted(entries, key=lambda e: e["relpath"]),
    }
    return doc | {"condition_hash": condition_hash(doc)}


def _rename_into_place(tmp: Path, target: Path, attempts: int = 10) -> None:
    """``os.replace`` with retries: on Windows a directory of freshly written files can be
    briefly locked by the search indexer or antivirus (WinError 5 / 32)."""
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.5 * (i + 1))


def materialize_condition(
    src: SplitSource, suite: Suite, cond: Condition, root: Path
) -> ConditionResult:
    target = condition_dir(root, cond)
    manifest = target / "condition.json"
    if manifest.is_file():  # rerun: prove byte identity
        stored = read_json(manifest)
        expected = {e["relpath"]: e for e in stored["entries"]}
        if set(expected) != {e["relpath"] for e in src.entries}:
            raise RobustnessError(f"{cond.key}: stored condition covers different images")
        for entry in src.entries:
            data = corrupt_one(src, suite, cond, entry)
            sha = hashlib.sha256(data).hexdigest()
            rec = expected[entry["relpath"]]
            on_disk = (target / rec["corrupted_file"]).read_bytes()
            if sha != rec["corrupted_sha256"] or on_disk != data:
                raise RobustnessError(f"{cond.key}: {entry['relpath']} is not byte-identical")
        return ConditionResult(
            cond.key, "verified", stored["condition_hash"], len(src.entries), target
        )
    tmp = target.with_name(f".tmp-{target.name}-{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        rows = []
        for entry in src.entries:
            data = corrupt_one(src, suite, cond, entry)
            rel = corrupted_relpath(entry["relpath"])
            out = tmp.joinpath(*rel.split("/"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            rows.append(
                {
                    "relpath": entry["relpath"],
                    "label": entry["label"],
                    "source_sha256": entry["sha256"],
                    "corrupted_file": rel,
                    "corrupted_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        doc = _doc(src, suite, cond, rows)
        write_json(
            tmp / "condition.json",
            doc | {"created_at": datetime.now(UTC).isoformat(timespec="seconds")},
        )
        _rename_into_place(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return ConditionResult(cond.key, "written", doc["condition_hash"], len(rows), target)


def materialize(
    settings: Settings,
    dataset_id: str,
    split_id: str,
    suite: Suite,
    split: str,
    *,
    out_root: Path | None = None,
) -> list[ConditionResult]:
    src = open_split(settings, dataset_id, split_id, split)
    root = out_root or suite_root(settings, src, suite)
    return [materialize_condition(src, suite, c, root) for c in suite.conditions]


def load_verified_condition(root: Path, suite: Suite, cond: Condition, src: SplitSource) -> dict:
    """The condition manifest, after checking it matches this suite and split and that every
    corrupted file has its recorded sha256. Raises :class:`RobustnessError` otherwise."""
    path = condition_dir(root, cond) / "condition.json"
    if not path.is_file():
        raise RobustnessError(f"condition {cond.key} is not materialized (run materialize first)")
    doc = read_json(path)
    if condition_hash(doc) != doc.get("condition_hash"):
        raise RobustnessError(f"{cond.key}: condition.json does not match its condition hash")
    expected = {
        "suite": suite.name,
        "suite_sha256": suite.sha256,
        "corruption": cond.corruption,
        "severity": cond.severity_str,
        "dataset_id": src.dataset_id,
        "split_id": src.split_id,
        "split": src.split,
        "manifest_sha256": src.manifest_sha256,
    }
    for k, v in expected.items():
        if doc.get(k) != v:
            raise RobustnessError(f"{cond.key}: condition {k}={doc.get(k)!r}, expected {v!r}")
    if [e["relpath"] for e in doc["entries"]] != sorted(e["relpath"] for e in src.entries):
        raise RobustnessError(f"{cond.key}: condition covers different images than the split")
    base = condition_dir(root, cond)
    for e in doc["entries"]:
        data = base.joinpath(*e["corrupted_file"].split("/")).read_bytes()
        if hashlib.sha256(data).hexdigest() != e["corrupted_sha256"]:
            raise RobustnessError(f"{cond.key}: {e['corrupted_file']} does not match its sha256")
    return doc
