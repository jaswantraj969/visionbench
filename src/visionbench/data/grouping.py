"""Duplicate groups: union-find over exact (SHA-256) and near (dHash) edges."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from visionbench.data.hashing import ALL_ONES, near_duplicate_edges
from visionbench.data.records import FileRecord


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    members: tuple[str, ...]  # sorted relpaths
    labels: tuple[str, ...]  # sorted distinct labels
    kind: str  # "single" | "exact" | "near" | "mixed"

    @property
    def conflict(self) -> bool:
        return len(self.labels) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "kind": self.kind,
            "labels": list(self.labels),
            "members": list(self.members),
            "conflict": self.conflict,
        }


@dataclass(frozen=True)
class Grouping:
    groups: tuple[DuplicateGroup, ...]  # sorted by group_id; every record is in exactly one
    group_of: dict[str, str]  # relpath -> group_id
    low_information: tuple[str, ...]  # relpaths excluded from perceptual matching
    near_edges: int

    @property
    def multi_member(self) -> list[DuplicateGroup]:
        return [g for g in self.groups if len(g.members) > 1]

    @property
    def conflicts(self) -> list[DuplicateGroup]:
        return [g for g in self.groups if g.conflict]


DEFAULT_POPCOUNT_MIN = 8
DEFAULT_POPCOUNT_MAX = 56


def is_low_information(
    record: FileRecord,
    low_info_std: float,
    popcount_min: int = DEFAULT_POPCOUNT_MIN,
    popcount_max: int = DEFAULT_POPCOUNT_MAX,
) -> bool:
    """True if the dHash carries too little information for perceptual matching.

    Any of: near-uniform pixels (gray std below threshold); a degenerate all-0/all-1 hash; or a
    hash with fewer than ``popcount_min`` / more than ``popcount_max`` set bits. Smooth
    gradients pass the std test but give such sparse/dense hashes, which then sit within a few
    bits of each other and would form false groups.
    """
    h = record.dhash_int
    popcount = h.bit_count()
    return (
        record.gray_std < low_info_std
        or h in (0, ALL_ONES)
        or popcount < popcount_min
        or popcount > popcount_max
    )


def make_group_id(members: Sequence[str]) -> str:
    return "g-" + hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()[:16]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(
    records: Sequence[FileRecord],
    *,
    dhash_threshold: int,
    low_info_std: float,
    popcount_min: int = DEFAULT_POPCOUNT_MIN,
    popcount_max: int = DEFAULT_POPCOUNT_MAX,
) -> Grouping:
    records = sorted(records, key=lambda r: r.relpath)
    uf = _UnionFind(len(records))

    first_by_sha: dict[str, int] = {}
    for i, r in enumerate(records):
        uf.union(first_by_sha.setdefault(r.sha256, i), i)

    low_info = [
        i
        for i, r in enumerate(records)
        if is_low_information(r, low_info_std, popcount_min, popcount_max)
    ]
    low_info_set = set(low_info)
    eligible = np.array([i for i in range(len(records)) if i not in low_info_set], dtype=np.int64)
    hashes = np.array([records[i].dhash_int for i in eligible], dtype=np.uint64)
    edges = near_duplicate_edges(hashes, dhash_threshold)
    for a, b in edges:
        uf.union(int(eligible[a]), int(eligible[b]))

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        members_by_root[uf.find(i)].append(i)

    groups = []
    for idxs in members_by_root.values():
        members = tuple(sorted(records[i].relpath for i in idxs))
        shas = [records[i].sha256 for i in idxs]
        if len(idxs) == 1:
            kind = "single"
        elif len(set(shas)) == 1:
            kind = "exact"
        elif len(set(shas)) == len(shas):
            kind = "near"
        else:
            kind = "mixed"
        labels = tuple(sorted({records[i].label for i in idxs}))
        groups.append(DuplicateGroup(make_group_id(members), members, labels, kind))
    groups.sort(key=lambda g: g.group_id)

    return Grouping(
        groups=tuple(groups),
        group_of={m: g.group_id for g in groups for m in g.members},
        low_information=tuple(records[i].relpath for i in low_info),
        near_edges=int(len(edges)),
    )
