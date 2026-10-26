"""Classification metrics from saved predictions, with explicit handling of undefined values.

Policy: a metric that is mathematically undefined for the evaluated set is reported as ``None``
(JSON null) together with a machine-readable reason in ``undefined`` — never as 0 or a made-up
number. Per-class definitions (tp/fp/fn from the confusion matrix):

- precision_k = tp / (tp + fp)        undefined if the class is never predicted
- recall_k    = tp / (tp + fn)        undefined if the class is absent from the evaluated set
- f1_k        = 2tp / (2tp + fp + fn) undefined only if the class is absent AND never predicted
  (same formula as scikit-learn >= 1.3, so a present-but-never-predicted class has F1 = 0)

Macro averages are means over the classes whose value is defined; the excluded classes are
listed. Weighted averages weight defined values by support. Balanced accuracy = macro recall
(mean recall over classes present in the evaluated set). Log loss uses scikit-learn's clipping
of probabilities to [eps, 1-eps]; the number of clipped samples is reported, and if it is > 0
the value is a lower bound.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

ECE_BINS = 15
DEFAULT_BOOTSTRAP = 1000
CI_LEVEL = 0.95

ABSENT = "class_absent_from_eval_set"
NEVER_PREDICTED = "class_never_predicted"
ABSENT_AND_NEVER_PREDICTED = "class_absent_and_never_predicted"


def _num(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """k x k counts; rows = true class, columns = predicted class."""
    codes = np.asarray(y_true, dtype=np.int64) * k + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(codes, minlength=k * k).reshape(k, k)


def per_class(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Per-class precision/recall/F1/support via scikit-learn with ``zero_division=np.nan``."""
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), zero_division=np.nan, average=None
    )
    return {"precision": p, "recall": r, "f1": f, "support": s}


def _nan_reason(metric: str, support: int, predicted: int) -> str:
    if metric == "precision":
        return NEVER_PREDICTED
    if metric == "recall":
        return ABSENT
    return ABSENT_AND_NEVER_PREDICTED if support == 0 and predicted == 0 else "undefined"


def _averages(values: np.ndarray, support: np.ndarray) -> tuple[float | None, float | None]:
    defined = ~np.isnan(values)
    if not defined.any():
        return None, None
    macro = float(values[defined].mean())
    w = support[defined].astype(np.float64)
    weighted = float((values[defined] * w).sum() / w.sum()) if w.sum() > 0 else None
    return macro, weighted


def balanced_accuracy_value(confusion: np.ndarray) -> float:
    support = confusion.sum(axis=1)
    present = support > 0
    if not present.any():
        return float("nan")
    return float((np.diag(confusion)[present] / support[present]).mean())


def macro_f1_value(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    defined = denom > 0
    if not defined.any():
        return float("nan")
    return float((2 * tp[defined] / denom[defined]).mean())


def accuracy_value(confusion: np.ndarray) -> float:
    n = confusion.sum()
    return float(np.trace(confusion) / n) if n else float("nan")


def calibration(proba: np.ndarray, y_true: np.ndarray, bins: int = ECE_BINS) -> dict[str, Any]:
    """Top-label ECE with ``bins`` equal-width confidence bins on (0, 1]."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    idx = np.clip(np.ceil(conf * bins).astype(np.int64) - 1, 0, bins - 1)
    n = len(y_true)
    ece = 0.0
    table = []
    for b in range(bins):
        mask = idx == b
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        mean_conf = float(conf[mask].mean()) if count else None
        if count:
            ece += abs(acc - mean_conf) * count / n
        table.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "count": count,
                "accuracy": acc,
                "mean_confidence": mean_conf,
            }
        )
    return {
        "ece": float(ece),
        "bins": bins,
        "binning": "equal-width on (0, 1], top-label confidence",
        "note": f"calibration estimate from {n} samples; noisy on small sets",
        "reliability": table,
    }


def _auc_and_ap(
    y_true: np.ndarray, proba: np.ndarray, k: int, class_names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    undefined: dict[str, str] = {}
    pos = np.bincount(y_true, minlength=k)
    neg = len(y_true) - pos
    bad = [class_names[i] for i in range(k) if pos[i] == 0 or neg[i] == 0]
    out: dict[str, Any] = {
        "roc_auc_ovr_macro": None,
        "average_precision_macro": None,
        "average_precision_per_class": {},
    }
    for i, name in enumerate(class_names):
        if pos[i] and neg[i]:
            out["average_precision_per_class"][name] = float(
                average_precision_score(y_true == i, proba[:, i])
            )
        else:
            out["average_precision_per_class"][name] = None
            reason = ABSENT if pos[i] == 0 else "class_has_no_negatives_in_eval_set"
            undefined[f"average_precision_per_class.{name}"] = reason
    if bad:
        reason = "classes_without_positive_and_negative_examples:" + ",".join(bad)
        undefined["roc_auc_ovr_macro"] = reason
        undefined["average_precision_macro"] = reason
        return out, undefined
    if k == 2:
        out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba[:, 1]))
    else:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(k)))
        )
    out["average_precision_macro"] = float(
        np.mean(list(out["average_precision_per_class"].values()))
    )
    return out, undefined


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """All point metrics for one evaluated set (no bootstrap)."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    if len(yt) == 0:
        raise ValueError("cannot compute metrics on an empty set")
    cm = confusion_counts(yt, yp, k)
    pc = per_class(yt, yp, k)
    support = pc["support"]
    predicted = cm.sum(axis=0)
    undefined: dict[str, str] = {}

    per_class_out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(class_names):
        row: dict[str, Any] = {"support": int(support[i]), "predicted": int(predicted[i])}
        for m in ("precision", "recall", "f1"):
            row[m] = _num(pc[m][i])
            if row[m] is None:
                undefined[f"per_class.{name}.{m}"] = _nan_reason(m, support[i], predicted[i])
        per_class_out[name] = row

    averages: dict[str, Any] = {}
    excluded: dict[str, list[str]] = {}
    for m in ("precision", "recall", "f1"):
        macro, weighted = _averages(pc[m], support)
        averages[f"macro_{m}"] = macro
        averages[f"weighted_{m}"] = weighted
        excluded[m] = [class_names[i] for i in range(k) if math.isnan(pc[m][i])]
        if macro is None:
            undefined[f"macro_{m}"] = f"no_class_with_defined_{m}"
            undefined[f"weighted_{m}"] = f"no_class_with_defined_{m}"

    absent = [class_names[i] for i in range(k) if support[i] == 0]
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = [
        [None if row_sums[i, 0] == 0 else float(cm[i, j] / row_sums[i, 0]) for j in range(k)]
        for i in range(k)
    ]
    for i in range(k):
        if row_sums[i, 0] == 0:
            undefined[f"confusion_matrix.row_normalized.{i}"] = f"{ABSENT}:{class_names[i]}"

    out: dict[str, Any] = {
        "n": int(len(yt)),
        "num_classes": k,
        "class_names": list(class_names),
        "accuracy": accuracy_value(cm),
        "balanced_accuracy": _num(balanced_accuracy_value(cm)),
        "balanced_accuracy_excluded_classes": absent,
        **averages,
        "macro_excluded_classes": excluded,
        "per_class": per_class_out,
        "confusion_matrix": {
            "labels": list(class_names),
            "counts": cm.tolist(),
            "row_normalized": normalized,
        },
    }
    if out["balanced_accuracy"] is None:
        undefined["balanced_accuracy"] = "no_class_present"

    if proba is None:
        for key in ("log_loss", "roc_auc_ovr_macro", "average_precision_macro", "calibration"):
            out[key] = None
            undefined[key] = "no_probabilities"
        out["average_precision_per_class"] = None
    else:
        p = np.asarray(proba, dtype=np.float64)
        if p.shape != (len(yt), k):
            raise ValueError(f"proba must have shape {(len(yt), k)}, got {p.shape}")
        # scikit-learn clips probabilities to [eps, 1 - eps] (float64 eps); when the true-class
        # probability underflows, the reported value is a lower bound of the exact loss.
        eps = np.finfo(np.float64).eps
        clipped = int((p[np.arange(len(yt)), yt] < eps).sum())
        out["log_loss"] = float(log_loss(yt, p, labels=list(range(k))))
        out["log_loss_clipped_samples"] = clipped
        out["log_loss_note"] = (
            f"lower bound: {clipped} sample(s) had true-class probability < {eps:.3g} and were "
            "clipped"
            if clipped
            else None
        )
        auc, auc_undefined = _auc_and_ap(yt, p, k, class_names)
        out.update(auc)
        undefined.update(auc_undefined)
        out["calibration"] = calibration(p, yt)
    out["undefined"] = dict(sorted(undefined.items()))
    return out


# --- bootstrap ---------------------------------------------------------------------------

BOOTSTRAP_METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "accuracy": accuracy_value,
    "balanced_accuracy": balanced_accuracy_value,
    "macro_f1": macro_f1_value,
}


def bootstrap_indices(n: int, resamples: int, seed: int) -> np.ndarray:
    """(resamples, n) indices drawn with replacement from a seeded NumPy Generator."""
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def metric_samples(
    y_true: np.ndarray, y_pred: np.ndarray, k: int, indices: np.ndarray, metric: str
) -> np.ndarray:
    """Value of ``metric`` on each resample (NaN where undefined on that resample)."""
    fn = BOOTSTRAP_METRICS[metric]
    codes = np.asarray(y_true, dtype=np.int64) * k + np.asarray(y_pred, dtype=np.int64)
    return np.array([fn(np.bincount(codes[idx], minlength=k * k).reshape(k, k)) for idx in indices])


def percentile_ci(samples: np.ndarray, level: float = CI_LEVEL) -> dict[str, Any]:
    defined = samples[~np.isnan(samples)]
    alpha = (1 - level) / 2
    if len(defined) == 0:
        return {"low": None, "high": None, "resamples_defined": 0}
    low, high = np.percentile(defined, [100 * alpha, 100 * (1 - alpha)])
    return {"low": float(low), "high": float(high), "resamples_defined": int(len(defined))}


def bootstrap_cis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    k: int,
    *,
    resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    idx = bootstrap_indices(len(yt), resamples, seed)
    return {
        "method": "percentile bootstrap over evaluated samples",
        "level": CI_LEVEL,
        "resamples": resamples,
        "seed": seed,
        "rng": f"numpy.random.default_rng (numpy {np.__version__})",
        "metrics": {m: percentile_ci(metric_samples(yt, yp, k, idx, m)) for m in BOOTSTRAP_METRICS},
    }


def paired_bootstrap_difference(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    k: int,
    *,
    metrics: Sequence[str] = ("balanced_accuracy", "macro_f1"),
    resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """CI of metric(A) - metric(B), using the SAME resampled indices for both models."""
    yt = np.asarray(y_true)
    idx = bootstrap_indices(len(yt), resamples, seed)
    out: dict[str, Any] = {
        "resamples": resamples,
        "seed": seed,
        "level": CI_LEVEL,
        "method": "paired percentile bootstrap (shared indices)",
        "metrics": {},
    }
    ca = confusion_counts(yt, np.asarray(pred_a), k)
    cb = confusion_counts(yt, np.asarray(pred_b), k)
    for m in metrics:
        fn = BOOTSTRAP_METRICS[m]
        diff = metric_samples(yt, np.asarray(pred_a), k, idx, m) - metric_samples(
            yt, np.asarray(pred_b), k, idx, m
        )
        out["metrics"][m] = {"point": _num(fn(ca) - fn(cb)), **percentile_ci(diff)}
    return out
