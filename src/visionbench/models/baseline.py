"""Majority-class baseline: predicts the most frequent TRAIN class; probabilities = train priors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

PRIOR_FLOOR = 1e-12  # avoids log(0) for classes absent from the train split


class MajorityBaseline:
    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = list(class_names)
        self.priors: list[float] | None = None
        self.train_counts: dict[str, int] | None = None

    def fit(self, train_counts: Mapping[str, int]) -> MajorityBaseline:
        """Fit from per-class TRAIN counts (no images are needed)."""
        unknown = set(train_counts) - set(self.class_names)
        if unknown:
            raise ValueError(f"unknown classes in train counts: {sorted(unknown)}")
        counts = {c: int(train_counts.get(c, 0)) for c in self.class_names}
        total = sum(counts.values())
        if total == 0:
            raise ValueError("cannot fit a majority baseline on an empty train split")
        self.train_counts = counts
        self.priors = [counts[c] / total for c in self.class_names]
        return self

    def _require_fit(self) -> list[float]:
        if self.priors is None:
            raise RuntimeError("MajorityBaseline is not fitted")
        return self.priors

    @property
    def majority_index(self) -> int:
        priors = self._require_fit()
        return max(range(len(priors)), key=lambda i: (priors[i], -i))  # ties -> lowest index

    def predict(self, n: int) -> list[int]:
        return [self.majority_index] * n

    def predict_proba(self, n: int) -> list[list[float]]:
        return [list(self._require_fit()) for _ in range(n)]

    def nll(self, label_index: int) -> float:
        return -math.log(max(self._require_fit()[label_index], PRIOR_FLOOR))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "majority_baseline",
            "class_names": self.class_names,
            "train_counts": self.train_counts,
            "priors": self._require_fit(),
            "majority_class": self.class_names[self.majority_index],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MajorityBaseline:
        model = cls(data["class_names"])
        return model.fit(data["train_counts"])
