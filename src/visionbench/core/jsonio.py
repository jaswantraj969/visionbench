"""Canonical JSON serialization: identical objects always produce identical bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    """Sorted keys, 2-space indent, UTF-8, LF line endings, trailing newline, no NaN."""
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_json(path: Path, obj: Any) -> bytes:
    """Write canonical JSON as bytes (no platform newline translation); return the bytes."""
    data = canonical_json_bytes(obj)
    path.write_bytes(data)
    return data


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))
