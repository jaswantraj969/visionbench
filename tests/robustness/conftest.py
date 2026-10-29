"""Shared fixture for the robustness tests: a synthetic dataset, two runs, a materialized suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import vb_trainenv

from visionbench.core.config import Settings
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.robustness import materialize as M
from visionbench.robustness.suite import load_suite
from visionbench.training.runner import run_training

SUITE_FILE = Path("benchmarks/eurosat-v1/robustness/robustness_suite.yaml")


@pytest.fixture(scope="package")
def env(tmp_path_factory):
    """Shapes dataset (32 px), a small_cnn and a majority baseline, each with a clean VAL eval."""
    root = tmp_path_factory.mktemp("robust")
    e = vb_trainenv.make_train_env(root)
    suites = root / "suites" / "x" / "robustness"
    suites.mkdir(parents=True)
    shutil.copy(SUITE_FILE, suites / "robustness_suite.yaml")
    settings = Settings(
        _env_file=None, data_root=e.settings.data_root, robustness_suites_dir=root / "suites"
    )
    cnn = run_training(e.config(epochs=2), settings).experiment_id
    base = run_training(e.config(model="majority_baseline"), settings).experiment_id
    clean = {
        x: evaluate_experiment(x, "val", settings, device="cpu", bootstrap_resamples=50).eval_id
        for x in (cnn, base)
    }
    suite = load_suite(suites / "robustness_suite.yaml")
    results = M.materialize(settings, e.dataset_id, e.split_id, suite, "val")
    return {
        "env": e,
        "settings": settings,
        "cnn": cnn,
        "base": base,
        "clean": clean,
        "suite": suite,
        "first": results,
        "root": root,
    }
