"""The TEST path of the robustness report (M8), on a synthetic dataset with the TEST unlock set.

The report pairs each run with its single clean TEST evaluation; zero or more than one is an
error, and the locked default still refuses.
"""

from __future__ import annotations

import json
import shutil

import pytest
import vb_trainenv

from visionbench.core.config import Settings
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.robustness import materialize as M
from visionbench.robustness.evaluate import evaluate_robustness
from visionbench.robustness.report import single_test_evaluation, write_report
from visionbench.robustness.suite import RobustnessError, TestSplitLockedError, load_suite
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.runner import run_training

SUITE_FILE = "benchmarks/eurosat-v1/robustness/robustness_suite.yaml"


@pytest.fixture(scope="module")
def unlocked(tmp_path_factory):
    root = tmp_path_factory.mktemp("robust_test")
    e = vb_trainenv.make_train_env(root)
    suites = root / "suites" / "x" / "robustness"
    suites.mkdir(parents=True)
    shutil.copy(SUITE_FILE, suites / "robustness_suite.yaml")
    settings = Settings(
        _env_file=None,
        data_root=e.settings.data_root,
        robustness_suites_dir=root / "suites",
        test_images_unlocked=True,
    )
    base = run_training(e.config(model="majority_baseline"), settings).experiment_id
    val = evaluate_experiment(base, "val", settings, device="cpu", bootstrap_resamples=50).eval_id
    suite = load_suite(suites / "robustness_suite.yaml")
    return {"env": e, "settings": settings, "base": base, "val": val, "suite": suite, "root": root}


def _runs(path, exp, val) -> None:
    path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "majority_baseline",
                        "seed": 0,
                        "status": "completed",
                        "experiment_id": exp,
                        "val_eval_id": val,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_test_report_pairs_the_single_test_evaluation(unlocked) -> None:
    s, e, suite, base = unlocked["settings"], unlocked["env"], unlocked["suite"], unlocked["base"]
    runs = unlocked["root"] / "runs.json"
    _runs(runs, base, unlocked["val"])
    with (
        ExperimentStore(s.data_dir("db") / DB_FILENAME) as store,
        pytest.raises(RobustnessError, match="0 clean TEST evaluations"),
    ):
        single_test_evaluation(store, base)
    test_id = evaluate_experiment(base, "test", s, device="cpu", bootstrap_resamples=50).eval_id
    with ExperimentStore(s.data_dir("db") / DB_FILENAME) as store:
        assert single_test_evaluation(store, base) == test_id
    M.materialize(s, e.dataset_id, e.split_id, suite, "test")
    out = evaluate_robustness(
        base, suite, "test", s, clean_eval_id=test_id, device="cpu", bootstrap_resamples=50
    )
    assert [o.status for o in out] == ["evaluated"] * 12
    doc = write_report(s, runs, suite, "test", unlocked["root"] / "out" / "robustness_test.md")
    text = (unlocked["root"] / "out" / "robustness_test.md").read_text(encoding="utf-8")
    assert doc["label"] == "TEST" and doc["missing"] == []
    assert doc["runs"][0]["clean_evaluation_id"] == test_id
    assert "single clean TEST evaluation of each run" in text
    assert "selection-biased" not in text.split("## Confounds")[0]
    assert "the TEST numbers chose nothing" in text
    assert "equal for all 12 run x condition pairs" in text
    # the locked default still refuses a TEST report
    locked = Settings(_env_file=None, data_root=s.data_root)
    with pytest.raises(TestSplitLockedError):
        write_report(locked, runs, suite, "test", unlocked["root"] / "out2" / "r.md")


def test_more_than_one_test_evaluation_is_an_error(unlocked) -> None:
    s, base = unlocked["settings"], unlocked["base"]
    with ExperimentStore(s.data_dir("db") / DB_FILENAME) as store:
        if not store.evaluations(base, split="test"):
            evaluate_experiment(base, "test", s, device="cpu", bootstrap_resamples=50)
    evaluate_experiment(
        base, "test", s, device="cpu", bootstrap_resamples=50, force=True, reason="test of rule"
    )
    with (
        ExperimentStore(s.data_dir("db") / DB_FILENAME) as store,
        pytest.raises(RobustnessError, match="2 clean TEST evaluations"),
    ):
        single_test_evaluation(store, base)


def test_val_then_test_materialization_on_one_split_stays_separate(unlocked) -> None:
    """Regression (final analysis plan, amendment 1): the cache path carries the split name, so a
    TEST pass after a VAL pass on the same split_id no longer collides with the VAL cache."""
    s, e, suite = unlocked["settings"], unlocked["env"], unlocked["suite"]
    val = M.materialize(s, e.dataset_id, e.split_id, suite, "val")
    test = M.materialize(s, e.dataset_id, e.split_id, suite, "test")
    again = M.materialize(s, e.dataset_id, e.split_id, suite, "val")
    assert [r.condition_hash for r in again] == [r.condition_hash for r in val]
    roots = {
        split: M.suite_root(s, M.open_split(s, e.dataset_id, e.split_id, split), suite)
        for split in ("val", "test")
    }
    assert roots["val"] != roots["test"]
    assert roots["val"].parent.name == "val" and roots["test"].parent.name == "test"
    assert {r.condition_hash for r in val}.isdisjoint({r.condition_hash for r in test})
