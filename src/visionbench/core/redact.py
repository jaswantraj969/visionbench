"""Remove absolute filesystem paths from text that leaves the process (API bodies, job errors)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


def _variants(path: Path) -> set[str]:
    p = str(path)
    forward = p.replace("\\", "/")
    out = {p, forward, p.replace("\\", "\\\\")}
    return {v for v in out if len(v) > 3}


def scrub_paths(text: str, roots: Iterable[tuple[Path | None, str]]) -> str:
    """Replace every occurrence of each root (in its usual spellings, case-insensitive) with its
    placeholder, e.g. ``(data_root, "<data_root>")``. Longest roots first."""
    pairs = [(v, label) for root, label in roots if root is not None for v in _variants(root)]
    for value, label in sorted(pairs, key=lambda x: -len(x[0])):
        text = re.sub(re.escape(value), label, text, flags=re.IGNORECASE)
    return text
