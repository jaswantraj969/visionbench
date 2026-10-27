"""Duplicate-group-aware, class-stratified train/val/test splits.

Algorithm ``vb-split-v2``:
1. Rebuild duplicate groups from the stored file records with this split's thresholds.
2. Label-conflict groups (members with different labels) are excluded ("exclude") or abort the
   split ("error"). They are never relabelled.
3. Per class: order its groups by ``sha256(f"{seed}|{class}|{group_id}")`` (pure hashing, so the
   order does not depend on any RNG implementation or library version), then give each group, in
   that order, to the split with the largest remaining deficit ``target - assigned`` (in images).
   Ties go train, then val, then test. Targets use exact rational arithmetic, so tie-breaking is
   platform-independent.
4. Leakage is computed from the final entries, not assumed.

Group-size warnings (D1, :func:`group_size_stats`): a multi-member group larger than its
class's val target, or larger than 2% of its class, is listed in ``split_info.json`` and in the
CLI output. They are not part of ``manifest.json``, whose bytes stay unchanged.

``vb-split-v1`` (M1) used a NumPy Generator permutation instead of step 3's hash order.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from visionbench import __version__
from visionbench.core.errors import SplitError
from visionbench.core.jsonio import canonical_json_bytes, write_json
from visionbench.data.grouping import DuplicateGroup, build_groups
from visionbench.data.records import FileRecord, compute_fingerprint
from visionbench.data.store import dataset_dir, load_dataset, load_records

ALGORITHM = "vb-split-v2"
SPLIT_NAMES = ("train", "val", "test")
CONFLICT_POLICIES = ("exclude", "error")
MANIFEST_FORMAT_VERSION = 1
OVERSIZED_SHARE = Fraction(2, 100)  # a group above 2% of its class is reported


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    dhash_threshold: int = 5
    low_info_std: float = 3.0
    low_info_popcount_min: int = 8
    low_info_popcount_max: int = 56
    conflict_policy: str = "exclude"
    allow_missing_classes: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SplitError(f"seed must be an integer, got {self.seed!r}")
        if not 0 <= self.seed < 2**32:
            raise SplitError(f"seed must be in [0, 2**32), got {self.seed}")
        if len(self.fractions) != 3 or any(f <= 0 for f in self.fractions):
            raise SplitError(f"fractions must be three values > 0, got {list(self.fractions)}")
        if abs(sum(self.exact_fractions()) - 1) > Fraction(1, 10**9):
            raise SplitError(f"fractions must sum to 1, got {list(self.fractions)}")
        if not 0 <= self.dhash_threshold <= 64:
            raise SplitError("dhash_threshold must be in [0, 64]")
        if self.low_info_std < 0:
            raise SplitError("low_info_std must be >= 0")
        if not 0 <= self.low_info_popcount_min <= self.low_info_popcount_max <= 64:
            raise SplitError("need 0 <= low_info_popcount_min <= low_info_popcount_max <= 64")
        if self.conflict_policy not in CONFLICT_POLICIES:
            raise SplitError(f"conflict_policy must be one of {CONFLICT_POLICIES}")

    def exact_fractions(self) -> list[Fraction]:
        # repr() gives the shortest decimal that round-trips, so 0.15 -> Fraction(3, 20).
        return [Fraction(repr(float(f))) for f in self.fractions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fractions": [float(f) for f in self.fractions],
            "dhash_threshold": self.dhash_threshold,
            "low_info_std_threshold": float(self.low_info_std),
            "low_info_popcount_min": self.low_info_popcount_min,
            "low_info_popcount_max": self.low_info_popcount_max,
            "conflict_policy": self.conflict_policy,
            "allow_missing_classes": self.allow_missing_classes,
        }


@dataclass(frozen=True)
class SplitResult:
    split_id: str
    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    already_existed: bool
    group_warnings: list[dict[str, Any]]


def make_split_id(fingerprint: str, config: SplitConfig) -> str:
    key = {"algorithm": ALGORITHM, "dataset_fingerprint": fingerprint, "config": config.to_dict()}
    return "sp-" + hashlib.sha256(canonical_json_bytes(key)).hexdigest()[:12]


def group_order_key(seed: int, label: str, group_id: str) -> str:
    return hashlib.sha256(f"{seed}|{label}|{group_id}".encode()).hexdigest()


def assign_class_groups(
    groups: list[DuplicateGroup], fractions: list[Fraction], seed: int, label: str
) -> tuple[dict[str, str], list[int], list[Fraction]]:
    """Assign one class's groups; return (group_id -> split, achieved counts, targets)."""
    ordered = sorted(groups, key=lambda g: group_order_key(seed, label, g.group_id))
    total = sum(len(g.members) for g in ordered)
    targets = [f * total for f in fractions]
    assigned = [0, 0, 0]
    assignment: dict[str, str] = {}
    for group in ordered:
        deficits = [targets[k] - assigned[k] for k in range(3)]
        best = 0  # strict ">" keeps the earliest split on ties: train, val, test
        for k in (1, 2):
            if deficits[k] > deficits[best]:
                best = k
        assigned[best] += len(group.members)
        assignment[group.group_id] = SPLIT_NAMES[best]
    return assignment, assigned, targets


def compute_manifest(
    records: list[FileRecord], *, dataset_id: str, fingerprint: str, config: SplitConfig
) -> dict[str, Any]:
    grouping = build_groups(
        records,
        dhash_threshold=config.dhash_threshold,
        low_info_std=config.low_info_std,
        popcount_min=config.low_info_popcount_min,
        popcount_max=config.low_info_popcount_max,
    )
    conflicts = grouping.conflicts
    if conflicts and config.conflict_policy == "error":
        shown = "; ".join(f"{g.group_id} labels={list(g.labels)}" for g in conflicts[:10])
        raise SplitError(f"{len(conflicts)} label-conflict group(s) (policy 'error'): {shown}")
    conflict_ids = {g.group_id for g in conflicts}

    by_class: dict[str, list[DuplicateGroup]] = defaultdict(list)
    for g in grouping.groups:
        if g.group_id not in conflict_ids:
            by_class[g.labels[0]].append(g)

    fractions = config.exact_fractions()
    labels = sorted({r.label for r in records})
    assignment: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    targets: dict[str, dict[str, float]] = {}
    for label in labels:
        assigned_ids, achieved, class_targets = assign_class_groups(
            by_class.get(label, []), fractions, config.seed, label
        )
        assignment.update(assigned_ids)
        counts[label] = dict(zip(SPLIT_NAMES, achieved, strict=True)) | {"total": sum(achieved)}
        targets[label] = {
            s: round(float(t), 4) for s, t in zip(SPLIT_NAMES, class_targets, strict=True)
        }

    missing = [
        f"class {label!r} has 0 items in {s} (train={c['train']}, val={c['val']}, test={c['test']})"
        for label, c in counts.items()
        for s in SPLIT_NAMES
        if c[s] == 0
    ]
    if missing and not config.allow_missing_classes:
        raise SplitError(
            "; ".join(missing)
            + " — set allow_missing_classes (--allow-missing-classes) to record a warning instead"
        )

    entries: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for r in sorted(records, key=lambda r: r.relpath):
        gid = grouping.group_of[r.relpath]
        if gid in conflict_ids:
            excluded.append(
                {
                    "relpath": r.relpath,
                    "label": r.label,
                    "group_id": gid,
                    "reason": "label_conflict",
                }
            )
        else:
            entries.append(
                {
                    "relpath": r.relpath,
                    "label": r.label,
                    "sha256": r.sha256,
                    "group_id": gid,
                    "split": assignment[gid],
                }
            )

    splits_of_group: dict[str, set[str]] = defaultdict(set)
    splits_of_sha: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        splits_of_group[e["group_id"]].add(e["split"])
        splits_of_sha[e["sha256"]].add(e["split"])

    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "algorithm": ALGORITHM,
        "visionbench_version": __version__,
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "config": config.to_dict(),
        "counts": counts,
        "split_totals": {s: sum(c[s] for c in counts.values()) for s in SPLIT_NAMES},
        "target_counts": targets,
        "excluded": excluded,
        "excluded_groups": len({e["group_id"] for e in excluded}),
        "low_information_files": len(grouping.low_information),
        "warnings": missing,
        "leakage_check": {
            "groups_checked": len(splits_of_group),
            "sha256_checked": len(splits_of_sha),
            "groups_in_multiple_splits": sum(len(s) > 1 for s in splits_of_group.values()),
            "sha256_in_multiple_splits": sum(len(s) > 1 for s in splits_of_sha.values()),
        },
        "entries": entries,
    }


def group_size_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    """Group sizes from a manifest's entries: distribution, largest group per class, and the
    oversized groups (multi-member groups larger than the class's val target or larger than
    2% of the class). Deterministic: computed from the manifest alone."""
    size: Counter[str] = Counter(e["group_id"] for e in manifest["entries"])
    label_of = {e["group_id"]: e["label"] for e in manifest["entries"]}
    by_class: dict[str, list[str]] = defaultdict(list)
    for gid in size:
        by_class[label_of[gid]].append(gid)
    per_class: dict[str, dict[str, Any]] = {}
    oversized: list[dict[str, Any]] = []
    for label in sorted(manifest["counts"]):
        total = manifest["counts"][label]["total"]
        targets = manifest["target_counts"][label]
        gids = sorted(by_class.get(label, []), key=lambda g: (-size[g], g))
        per_class[label] = {
            "class_total": total,
            "largest_group_id": gids[0] if gids else None,
            "largest_size": size[gids[0]] if gids else 0,
            "val_target": targets["val"],
            "test_target": targets["test"],
        }
        for gid in gids:
            n = size[gid]
            if n < 2:
                break
            reasons = []
            if n > targets["val"]:
                reasons.append("larger than the class's val target")
            if n > OVERSIZED_SHARE * total:
                reasons.append("larger than 2% of the class")
            if reasons:
                oversized.append(
                    {
                        "label": label,
                        "group_id": gid,
                        "size": n,
                        "class_total": total,
                        "share_of_class": round(n / total, 4),
                        "val_target": targets["val"],
                        "test_target": targets["test"],
                        "reasons": reasons,
                    }
                )
    return {
        "size_distribution": {str(k): v for k, v in sorted(Counter(size.values()).items())},
        "per_class": per_class,
        "oversized": oversized,
    }


def make_split(datasets_dir: Path, dataset_id: str, config: SplitConfig) -> SplitResult:
    meta = load_dataset(datasets_dir, dataset_id)
    records = load_records(datasets_dir, dataset_id)
    fingerprint = meta["fingerprint"]
    if compute_fingerprint(records) != fingerprint:
        raise SplitError(f"dataset {dataset_id}: files.json does not match the fingerprint")

    manifest = compute_manifest(
        records, dataset_id=dataset_id, fingerprint=fingerprint, config=config
    )
    data = canonical_json_bytes(manifest)
    digest = hashlib.sha256(data).hexdigest()
    warnings = group_size_stats(manifest)["oversized"]
    split_id = make_split_id(fingerprint, config)
    splits_dir = dataset_dir(datasets_dir, dataset_id) / "splits"
    final = splits_dir / split_id

    if final.exists():
        existing = (final / "manifest.json").read_bytes()
        if existing != data:
            raise SplitError(
                f"split {split_id} exists but its manifest differs from the recomputed one; "
                "refusing to overwrite"
            )
        return SplitResult(split_id, final, manifest, digest, True, warnings)

    staging = splits_dir / f".tmp-split-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        (staging / "manifest.json").write_bytes(data)
        write_json(
            staging / "split_info.json",
            {
                "split_id": split_id,
                "dataset_id": dataset_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "manifest_sha256": digest,
                "numpy_version": np.__version__,
                "python_version": sys.version.split()[0],
                "group_size_warnings": warnings,
            },
        )
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return SplitResult(split_id, final, manifest, digest, False, warnings)
