"""scripts/final_report.py: seed-matched comparisons, rank groups and labels on synthetic runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from visionbench.evaluation.metrics import (
    balanced_accuracy_value,
    confusion_counts,
    macro_f1_value,
)

_spec = importlib.util.spec_from_file_location(
    "final_report", Path(__file__).resolve().parents[2] / "scripts" / "final_report.py"
)
FR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FR)

CLASSES = ["a", "b"]
N = 400
# accuracy level per model (per seed); efficientnet and resnet18_imagenet are almost equal,
# resnet18_imagenet is far ahead of resnet18_scratch.
LEVEL = {
    "majority_baseline": (0.5, 0.5, 0.5),
    "resnet18_scratch_64": (0.70, 0.72, 0.71),
    "small_cnn_scratch_64": (0.71, 0.74, 0.70),
    "resnet18_imagenet_224": (0.95, 0.95, 0.96),
    "efficientnet_b0_imagenet_224": (0.955, 0.945, 0.96),
}


def _run(model: str, seed: int) -> dict:
    rng = np.random.default_rng(1000 * seed + len(model))
    y = np.arange(N) % 2
    if model == "majority_baseline":
        p = np.zeros(N, dtype=int)
    else:
        wrong = rng.permutation(N)[: int(round((1 - LEVEL[model][seed]) * N))]
        p = y.copy()
        p[wrong] = 1 - p[wrong]
    c = confusion_counts(y, p, 2)
    preds = {f"img{i:04d}.png": (CLASSES[y[i]], CLASSES[p[i]]) for i in range(N)}
    ba, f1 = balanced_accuracy_value(c), macro_f1_value(c)
    return {
        "model": model,
        "seed": seed,
        "experiment_id": f"exp-{model}-{seed}",
        "evaluation_id": f"ev-{model}-{seed}",
        "n": N,
        "class_names": CLASSES,
        "balanced_accuracy": ba,
        "macro_f1": f1,
        "accuracy": float((y == p).mean()),
        "ci": {m: {"low": 0.0, "high": 1.0} for m in FR.METRICS},
        "val": {"balanced_accuracy": ba + 0.01, "macro_f1": f1 + 0.01},
        "predictions": preds,
    }


def _analysis(split: str = "test"):
    runs = [_run(m, s) for m in FR.MODELS for s in FR.SEEDS]
    return runs, FR.analyse(runs, split)


def test_direction_requires_every_ci_on_the_same_side() -> None:
    up, down = {"low": 0.01, "high": 0.02}, {"low": -0.02, "high": -0.01}
    across = {"low": -0.01, "high": 0.01}
    assert FR.direction([up, up, up]) == 1
    assert FR.direction([down, down, down]) == -1
    assert FR.direction([up, up, across]) == 0
    assert FR.direction([up, down, up]) == 0


def test_preregistered_comparisons_and_rank_groups() -> None:
    _, doc = _analysis()
    verdicts = [c["verdict"]["balanced_accuracy"] for c in doc["comparisons"]]
    assert [(c["a"], c["b"]) for c in doc["comparisons"]] == list(FR.COMPARISONS)
    assert verdicts[0] == "not distinguishable"  # efficientnet vs resnet18_imagenet: ~equal
    assert verdicts[2] == "distinguishable: resnet18_imagenet_224 > resnet18_scratch_64"
    order = [x["model"] for x in doc["ranking"]]
    assert order[-1] == "majority_baseline" and set(order[:2]) == {
        "resnet18_imagenet_224",
        "efficientnet_b0_imagenet_224",
    }
    groups = {x["model"]: x["rank_group"] for x in doc["ranking"]}
    assert groups[order[0]] == groups[order[1]] == 1  # not separated
    assert groups["majority_baseline"] == max(groups.values())
    # the per-model mean is the mean over the three seeds
    m = doc["models"]["resnet18_scratch_64"]["balanced_accuracy"]
    assert abs(m["mean"] - sum(m["per_seed"]) / 3) < 1e-12 and len(m["per_seed"]) == 3


def test_seed_matched_pairs_use_the_same_images() -> None:
    runs, _ = _analysis()
    by = {(r["model"], r["seed"]): r for r in runs}
    d = FR.paired(by[("resnet18_imagenet_224", 0)], by[("resnet18_scratch_64", 0)])
    point = (
        by[("resnet18_imagenet_224", 0)]["balanced_accuracy"]
        - by[("resnet18_scratch_64", 0)]["balanced_accuracy"]
    )
    assert abs(d["balanced_accuracy"]["point"] - point) < 1e-12
    assert d["balanced_accuracy"]["low"] <= point <= d["balanced_accuracy"]["high"]


def test_render_labels_every_table_and_makes_no_claims() -> None:
    runs, analysis = _analysis()
    doc = {
        "benchmark": "x",
        "label": FR.TEST_LABEL,
        "final_analysis_plan_sha256": "0" * 64,
        **analysis,
    }
    text = FR.render(doc, runs, None)
    assert text.startswith("# Final report: x — TEST")
    assert "balanced accuracy (TEST)" in text and "VALIDATION" in text
    for code, _ in FR.CONSTRAINTS:
        assert f"**{code}**" in text
    assert "state-of-the-art." not in text.replace("robust or state-of-the-art.", "")
    val_doc = {**doc, "label": FR.VAL_LABEL, **FR.analyse(runs, "val")}
    assert FR.render(val_doc, runs, None).startswith("# DRY RUN of the final report on VALIDATION")
