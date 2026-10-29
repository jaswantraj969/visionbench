"""Corruption suites: a frozen YAML file (suite name, base seed, corruptions x severities).

The suite file's sha256 identifies it; every materialized condition and every robustness
evaluation records it. Severities are written as the YAML values print in Python (``0.04``,
``50``): that string is part of the per-image seed and of the cache path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from visionbench.core.errors import VisionBenchError

CORRUPTIONS = {
    "gaussian_noise": "sigma",
    "gaussian_blur": "sigma_px",
    "brightness": "delta_v",
    "jpeg": "quality",
}
SUITE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class RobustnessError(VisionBenchError):
    pass


class SuiteNotFoundError(RobustnessError):
    pass


class TestSplitLockedError(RobustnessError):
    """split=test before the final test evaluation (M8)."""

    __test__ = False  # not a pytest class


@dataclass(frozen=True)
class Condition:
    corruption: str
    parameter: str
    severity: float | int

    @property
    def severity_str(self) -> str:
        return str(self.severity)

    @property
    def key(self) -> str:
        return f"{self.corruption}/{self.severity_str}"


@dataclass(frozen=True)
class Suite:
    name: str
    base_seed: int
    conditions: tuple[Condition, ...]
    sha256: str  # of the YAML file bytes
    path: Path

    def condition(self, corruption: str, severity: str) -> Condition:
        for c in self.conditions:
            if c.corruption == corruption and c.severity_str == severity:
                return c
        raise RobustnessError(f"suite {self.name} has no condition {corruption}/{severity}")


def load_suite(path: Path) -> Suite:
    raw = Path(path).read_bytes()
    doc = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(doc, dict) or set(doc) != {"suite", "base_seed", "corruptions"}:
        raise RobustnessError("suite YAML needs exactly: suite, base_seed, corruptions")
    name = doc["suite"]
    if not isinstance(name, str) or not SUITE_NAME_RE.match(name):
        raise RobustnessError(f"invalid suite name {name!r}")
    if not isinstance(doc["base_seed"], int) or doc["base_seed"] < 0:
        raise RobustnessError("base_seed must be a non-negative integer")
    conditions = []
    for corruption, spec in doc["corruptions"].items():
        if corruption not in CORRUPTIONS:
            raise RobustnessError(f"unknown corruption {corruption!r}; known: {list(CORRUPTIONS)}")
        if spec.get("parameter") != CORRUPTIONS[corruption]:
            raise RobustnessError(f"{corruption}: parameter must be {CORRUPTIONS[corruption]!r}")
        severities = spec.get("severities")
        if not severities or len(set(map(str, severities))) != len(severities):
            raise RobustnessError(f"{corruption}: severities must be a non-empty unique list")
        for s in severities:
            if isinstance(s, bool) or not isinstance(s, int | float):
                raise RobustnessError(f"{corruption}: severity {s!r} is not a number")
            if corruption == "jpeg" and not (isinstance(s, int) and 1 <= s <= 95):
                raise RobustnessError("jpeg quality must be an integer in [1, 95]")
            if corruption != "jpeg" and not s > 0:
                raise RobustnessError(f"{corruption}: severity must be > 0")
            conditions.append(Condition(corruption, CORRUPTIONS[corruption], s))
    return Suite(
        name, doc["base_seed"], tuple(conditions), hashlib.sha256(raw).hexdigest(), Path(path)
    )


def find_suite(name: str, root: Path) -> Suite:
    """The suite named ``name`` among ``root/**/robustness_suite*.yaml`` (exactly one)."""
    if not SUITE_NAME_RE.match(name):
        raise RobustnessError(f"invalid suite name {name!r}")
    found = []
    for path in sorted(Path(root).rglob("robustness_suite*.yaml")):
        try:
            suite = load_suite(path)
        except (RobustnessError, OSError, yaml.YAMLError):
            continue
        if suite.name == name:
            found.append(suite)
    if len(found) != 1:
        raise SuiteNotFoundError(f"suite {name!r}: {len(found)} definitions found (need exactly 1)")
    return found[0]


def image_seed(suite: str, corruption: str, severity: str, base_seed: int, relpath: str) -> int:
    """Per-image seed: first 8 bytes of sha256("suite|corruption|severity|base_seed|relpath")."""
    text = f"{suite}|{corruption}|{severity}|{base_seed}|{relpath}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
