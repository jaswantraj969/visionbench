"""Dataset statistics report (``report.json``)."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from visionbench.core.seeding import derived_rng
from visionbench.data.grouping import LOW_INFO_REASONS, Grouping, low_information_reasons
from visionbench.data.records import IssueCategory
from visionbench.data.scan import ScanConfig, ScanResult

REPORT_FORMAT_VERSION = 2  # 2 (D1): low-information breakdown per class and reason


def _summary(values: list[int]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def sample_relpaths(relpaths: list[str], k: int, seed: int, label: str) -> list[str]:
    """Seeded sample of up to ``k`` relpaths, independent of listing order."""
    ordered = sorted(relpaths)
    if len(ordered) <= k:
        return ordered
    idx = derived_rng(seed, label).choice(len(ordered), size=k, replace=False)
    return sorted(ordered[i] for i in idx)


def build_report(
    *,
    dataset_id: str,
    name: str,
    fingerprint: str,
    scan: ScanResult,
    grouping: Grouping,
    config: ScanConfig,
) -> dict[str, Any]:
    records = scan.records
    counts = scan.class_counts()
    by_class: dict[str, list[str]] = {c: [] for c in scan.classes}
    for r in records:
        by_class[r.label].append(r.relpath)

    issues: dict[str, dict[str, Any]] = {}
    for category in IssueCategory:
        items = [i for i in scan.issues if i.category == category]
        issues[str(category)] = {
            "count": len(items),
            "items": [{"relpath": i.relpath, "detail": i.detail} for i in items],
        }

    multi = grouping.multi_member
    low_info = set(grouping.low_information)
    low_by_class: dict[str, dict[str, int]] = {
        c: dict.fromkeys(("total", *LOW_INFO_REASONS), 0) for c in scan.classes
    }
    for r in records:
        if r.relpath in low_info:
            entry = low_by_class[r.label]
            entry["total"] += 1
            for reason in low_information_reasons(
                r,
                config.low_info_std,
                config.low_info_popcount_min,
                config.low_info_popcount_max,
            ):
                entry[reason] += 1
    return {
        "format_version": REPORT_FORMAT_VERSION,
        "dataset_id": dataset_id,
        "name": name,
        "fingerprint": fingerprint,
        "classes": scan.classes,
        "class_counts": counts,
        "total_images": len(records),
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 4),
        "width": _summary([r.width for r in records]),
        "height": _summary([r.height for r in records]),
        "mode_counts": dict(sorted(Counter(r.mode for r in records).items())),
        "format_counts": dict(sorted(Counter(r.format for r in records).items())),
        "total_bytes": sum(r.size_bytes for r in records),
        "issues": issues,
        "duplicates": {
            "dhash_threshold": config.dhash_threshold,
            "low_info_std_threshold": config.low_info_std,
            "low_info_popcount_min": config.low_info_popcount_min,
            "low_info_popcount_max": config.low_info_popcount_max,
            "groups_total": len(grouping.groups),
            "multi_member_groups": len(multi),
            "files_in_multi_member_groups": sum(len(g.members) for g in multi),
            "exact_duplicate_files": len(records) - len({r.sha256 for r in records}),
            "perceptual_edges": grouping.near_edges,
            "groups": [g.to_dict() for g in multi],
            "label_conflict_groups": [g.to_dict() for g in grouping.conflicts],
            "perceptual_check_skipped_low_information": {
                "count": len(grouping.low_information),
                # reasons overlap: one image can fail several tests
                "by_class": low_by_class,
                "by_reason": {
                    k: sum(c[k] for c in low_by_class.values()) for k in LOW_INFO_REASONS
                },
                "relpaths": list(grouping.low_information),
            },
        },
        "samples": {
            "seed": config.sample_seed,
            "per_class": config.samples_per_class,
            "relpaths": {
                c: sample_relpaths(paths, config.samples_per_class, config.sample_seed, c)
                for c, paths in by_class.items()
            },
        },
    }
