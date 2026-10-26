"""Metrics against hand-computed fixtures. Every expected number below is worked out by hand
from the confusion matrix in the comment above it."""

import math

import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support

from visionbench.evaluation.metrics import (
    ABSENT,
    NEVER_PREDICTED,
    bootstrap_cis,
    bootstrap_indices,
    compute_metrics,
    metric_samples,
    paired_bootstrap_difference,
    percentile_ci,
)

C3 = ["c0", "c1", "c2"]


def _walk_undefined_are_null(m: dict) -> None:
    """Every key listed in `undefined` must resolve to None, or to a list of Nones for a confusion
    row (never a number)."""
    for key in m["undefined"]:
        node = m
        for part in key.split("."):
            node = node[int(part)] if isinstance(node, list) else node[part]
        assert node is None or (isinstance(node, list) and all(v is None for v in node)), key


def test_absent_class_and_never_predicted_class() -> None:
    # y_true [0,0,1,1], y_pred [0,0,0,2] ->  rows=true, cols=pred
    #   c0: [2,0,0]  c1: [1,0,1]  c2: [0,0,0]
    # c0: tp2 fp1 fn0 -> P 2/3, R 1, F1 4/5
    # c1: tp0 fp0 fn2 -> P undefined (never predicted), R 0, F1 0/(0+0+2)=0
    # c2: tp0 fp1 fn0 -> P 0, R undefined (absent), F1 0/(0+1+0)=0
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 2], C3)
    assert m["accuracy"] == 0.5
    pc = m["per_class"]
    assert pc["c0"] == {
        "support": 2,
        "predicted": 3,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
    }
    assert pc["c1"]["precision"] is None and pc["c1"]["recall"] == 0.0 and pc["c1"]["f1"] == 0.0
    assert pc["c2"]["precision"] == 0.0 and pc["c2"]["recall"] is None and pc["c2"]["f1"] == 0.0
    assert m["undefined"]["per_class.c1.precision"] == NEVER_PREDICTED
    assert m["undefined"]["per_class.c2.recall"] == ABSENT
    # balanced accuracy = mean recall over present classes (c0, c1) = (1 + 0) / 2
    assert m["balanced_accuracy"] == 0.5 and m["balanced_accuracy_excluded_classes"] == ["c2"]
    assert m["macro_precision"] == pytest.approx((2 / 3 + 0) / 2)  # c1 excluded
    assert m["macro_recall"] == pytest.approx(0.5)  # c2 excluded
    assert m["macro_f1"] == pytest.approx((0.8 + 0 + 0) / 3)
    assert m["macro_excluded_classes"] == {"precision": ["c1"], "recall": ["c2"], "f1": []}
    # weighted by support (2, 2, 0) over defined values
    assert m["weighted_precision"] == pytest.approx(2 / 3)  # (2*2/3 + 0*0) / 2
    assert m["weighted_recall"] == pytest.approx(0.5)  # (2*1 + 2*0) / 4
    assert m["weighted_f1"] == pytest.approx(0.4)  # (2*0.8 + 2*0 + 0*0) / 4
    cm = m["confusion_matrix"]
    assert cm["counts"] == [[2, 0, 0], [1, 0, 1], [0, 0, 0]]
    assert cm["row_normalized"] == [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [None, None, None]]
    # no probabilities -> probability metrics are null with a reason
    for key in ("log_loss", "roc_auc_ovr_macro", "average_precision_macro", "calibration"):
        assert m[key] is None and m["undefined"][key] == "no_probabilities"
    _walk_undefined_are_null(m)


def test_macro_averages_match_sklearn_nan_policy() -> None:
    yt, yp = [0, 0, 1, 1, 2], [0, 1, 1, 1, 1]
    m = compute_metrics(yt, yp, C3)
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            yt, yp, labels=[0, 1, 2], zero_division=np.nan, average=avg
        )
        assert m[f"{avg}_precision"] == pytest.approx(p)
        assert m[f"{avg}_recall"] == pytest.approx(r)
        assert m[f"{avg}_f1"] == pytest.approx(f)


def test_auc_null_with_reason_when_class_absent() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]])
    m = compute_metrics([0, 0, 1, 1], [0, 0, 1, 2], C3, proba)
    assert m["roc_auc_ovr_macro"] is None and m["average_precision_macro"] is None
    assert m["undefined"]["roc_auc_ovr_macro"] == (
        "classes_without_positive_and_negative_examples:c2"
    )
    assert m["average_precision_per_class"]["c2"] is None
    assert m["undefined"]["average_precision_per_class.c2"] == ABSENT
    assert m["average_precision_per_class"]["c0"] == 1.0  # c0 scores .7,.6 > all others
    assert m["log_loss"] is not None
    _walk_undefined_are_null(m)


def test_perfect_predictions() -> None:
    proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.2, 0.1, 0.7]])
    m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2], C3, proba)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "average_precision_macro",
    ):
        assert m[key] == pytest.approx(1.0), key
    expected_ll = -(math.log(0.9) + math.log(0.8) + math.log(0.8) + math.log(0.7)) / 4
    assert m["log_loss"] == pytest.approx(expected_ll)
    # every prediction correct -> ECE = mean(1 - confidence) = (0.1 + 0.2 + 0.2 + 0.3) / 4
    assert m["calibration"]["ece"] == pytest.approx(0.2)
    assert sum(b["count"] for b in m["calibration"]["reliability"]) == 4
    assert m["undefined"] == {}


def test_constant_majority_predictor() -> None:
    # y_true [0,0,0,1,2], always predict 0 with priors (.6,.2,.2)
    # c0: tp3 fp2 fn0 -> P .6 R 1 F1 6/8 ; c1, c2: never predicted -> P undefined, R 0, F1 0
    proba = np.tile([0.6, 0.2, 0.2], (5, 1))
    m = compute_metrics([0, 0, 0, 1, 2], [0] * 5, C3, proba)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["macro_precision"] == pytest.approx(0.6)
    assert m["macro_excluded_classes"]["precision"] == ["c1", "c2"]
    assert m["macro_f1"] == pytest.approx(0.25)  # (0.75 + 0 + 0) / 3
    assert m["roc_auc_ovr_macro"] == pytest.approx(0.5)  # constant scores
    assert m["average_precision_per_class"] == pytest.approx({"c0": 0.6, "c1": 0.2, "c2": 0.2})
    assert m["log_loss"] == pytest.approx(-(3 * math.log(0.6) + 2 * math.log(0.2)) / 5)
    assert m["calibration"]["ece"] == pytest.approx(0.0)  # confidence .6, accuracy .6
    _walk_undefined_are_null(m)


def test_binary_vs_multiclass_auc() -> None:
    # binary: positives score .4,.9 ; negatives .1,.6 -> 3 of 4 pairs ordered -> AUC .75
    p1 = np.array([0.1, 0.6, 0.4, 0.9])
    proba = np.stack([1 - p1, p1], axis=1)
    m = compute_metrics([0, 0, 1, 1], proba.argmax(1), ["neg", "pos"], proba)
    assert m["roc_auc_ovr_macro"] == pytest.approx(0.75)
    assert m["accuracy"] == 0.5
    assert m["per_class"]["pos"]["precision"] == 0.5 and m["per_class"]["pos"]["recall"] == 0.5


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [], C3)
    with pytest.raises(ValueError, match="shape"):
        compute_metrics([0, 1], [0, 1], C3, np.ones((2, 2)) / 2)


def test_bootstrap_reproducible_and_seed_dependent() -> None:
    rng = np.random.default_rng(0)
    yt, yp = rng.integers(0, 3, 60), rng.integers(0, 3, 60)
    a = bootstrap_cis(yt, yp, 3, resamples=200, seed=7)
    b = bootstrap_cis(yt, yp, 3, resamples=200, seed=7)
    c = bootstrap_cis(yt, yp, 3, resamples=200, seed=8)
    assert a == b and a["metrics"] != c["metrics"]
    for ci in a["metrics"].values():
        assert ci["low"] <= ci["high"] and ci["resamples_defined"] == 200


def test_paired_bootstrap_shares_indices() -> None:
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 3, 50)
    pa, pb = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    res = paired_bootstrap_difference(yt, pa, pb, 3, resamples=300, seed=5)
    idx = bootstrap_indices(50, 300, 5)
    for metric in ("balanced_accuracy", "macro_f1"):
        diff = metric_samples(yt, pa, 3, idx, metric) - metric_samples(yt, pb, 3, idx, metric)
        expected = percentile_ci(diff)
        assert res["metrics"][metric]["low"] == expected["low"]
        assert res["metrics"][metric]["high"] == expected["high"]
    same = paired_bootstrap_difference(yt, pa, pa, 3, resamples=100, seed=5)
    assert same["metrics"]["macro_f1"] == {
        "point": 0.0,
        "low": 0.0,
        "high": 0.0,
        "resamples_defined": 100,
    }


def test_percentile_ci_all_undefined() -> None:
    assert percentile_ci(np.array([np.nan, np.nan])) == {
        "low": None,
        "high": None,
        "resamples_defined": 0,
    }


def test_log_loss_reports_clipping() -> None:
    proba = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
    m = compute_metrics([0, 1, 2], [0, 1, 0], C3, proba)  # true class 2 has probability 0
    assert m["log_loss_clipped_samples"] == 1 and m["log_loss_note"].startswith("lower bound")
    assert math.isfinite(m["log_loss"])
