"""Materialization, evaluation, storage, test lock, report, API and schema v7 (synthetic data)."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from visionbench.cli import main as cli_main
from visionbench.core.jsonio import read_json, write_json
from visionbench.robustness import materialize as M
from visionbench.robustness.evaluate import evaluate_robustness
from visionbench.robustness.report import write_report
from visionbench.robustness.suite import (
    RobustnessError,
    TestSplitLockedError,
)
from visionbench.tracking.store import DB_FILENAME, SCHEMA_VERSION, ExperimentStore

SUITE_FILE = Path("benchmarks/eurosat-v1/robustness/robustness_suite.yaml")


def _files(cond_dir: Path) -> dict[str, str]:
    return {
        p.relative_to(cond_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(cond_dir.rglob("*.png"))
    }


def test_materialize_twice_and_in_another_process_is_byte_identical(env, tmp_path) -> None:
    s, e, suite = env["settings"], env["env"], env["suite"]
    assert [r.status for r in env["first"]] == ["written"] * 12
    again = M.materialize(s, e.dataset_id, e.split_id, suite, "val")
    assert [r.status for r in again] == ["verified"] * 12
    assert [r.condition_hash for r in again] == [r.condition_hash for r in env["first"]]
    other = tmp_path / "proc2"
    code = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from visionbench.core.config import Settings;"
            "from visionbench.robustness.materialize import materialize;"
            "from visionbench.robustness.suite import load_suite;"
            "s = Settings(_env_file=None, data_root=Path(sys.argv[1]));"
            "r = materialize(s, sys.argv[2], sys.argv[3], load_suite(Path(sys.argv[4])), 'val',"
            " out_root=Path(sys.argv[5]));"
            "print(len(r))",
            str(s.data_root),
            e.dataset_id,
            e.split_id,
            str(SUITE_FILE),
            str(other),
        ],
        capture_output=True,
        text=True,
    )
    assert code.returncode == 0, code.stderr
    for r in env["first"]:
        cond = suite.condition(*r.key.split("/"))
        assert _files(r.path) == _files(M.condition_dir(other, cond))
        assert (
            read_json(M.condition_dir(other, cond) / "condition.json")["condition_hash"]
            == r.condition_hash
        )


def test_order_independence(env, tmp_path) -> None:
    s, e, suite = env["settings"], env["env"], env["suite"]
    src = M.open_split(s, e.dataset_id, e.split_id, "val")
    shuffled = M.SplitSource(**{**src.__dict__, "entries": list(reversed(src.entries))})
    cond = suite.condition("gaussian_noise", "0.12")
    res = M.materialize_condition(shuffled, suite, cond, tmp_path / "rev")
    first = next(r for r in env["first"] if r.key == cond.key)
    assert _files(res.path) == _files(first.path) and res.condition_hash == first.condition_hash


def test_different_relpaths_get_different_noise(env) -> None:
    s, e, suite = env["settings"], env["env"], env["suite"]
    src = M.open_split(s, e.dataset_id, e.split_id, "val")
    a, b = src.entries[0], src.entries[1]
    import io

    from PIL import Image

    from visionbench.data.imageio import load_rgb

    def resid(entry, sev):
        cond = suite.condition("gaussian_noise", sev)
        out = np.asarray(
            Image.open(io.BytesIO(M.corrupt_one(src, suite, cond, entry))), dtype=float
        )
        orig = np.asarray(load_rgb(src.raw_root / entry["relpath"]), dtype=float)
        return (out - orig).ravel()

    assert abs(np.corrcoef(resid(a, "0.12"), resid(b, "0.12"))[0, 1]) < 0.2
    assert abs(np.corrcoef(resid(a, "0.12"), resid(a, "0.08"))[0, 1]) < 0.2


def test_test_split_is_locked(env, capsys) -> None:
    s, e, suite = env["settings"], env["env"], env["suite"]
    with pytest.raises(TestSplitLockedError):
        M.materialize(s, e.dataset_id, e.split_id, suite, "test")
    with pytest.raises(TestSplitLockedError):
        evaluate_robustness(env["cnn"], suite, "test", s, device="cpu")
    code = cli_main(
        [
            "robustness",
            "materialize",
            e.dataset_id,
            e.split_id,
            "--suite",
            str(SUITE_FILE),
            "--split",
            "test",
        ]
    )
    assert code == 1 and "locked" in capsys.readouterr().err


def test_evaluate_stores_separately_pins_clean_and_resumes(env) -> None:
    s, suite, cnn, base = env["settings"], env["suite"], env["cnn"], env["base"]
    with ExperimentStore(s.data_dir("db") / DB_FILENAME) as es:
        n_eval_before = len(es.evaluations())
    out = evaluate_robustness(
        cnn, suite, "val", s, clean_eval_id=env["clean"][cnn], device="cpu", bootstrap_resamples=50
    )
    assert [o.status for o in out] == ["evaluated"] * 12
    again = evaluate_robustness(cnn, suite, "val", s, clean_eval_id=env["clean"][cnn], device="cpu")
    assert [o.status for o in again] == ["skipped"] * 12
    assert [o.robustness_id for o in again] == [o.robustness_id for o in out]
    with ExperimentStore(s.data_dir("db") / DB_FILENAME) as es:
        assert len(es.evaluations()) == n_eval_before  # nothing written to `evaluations`
        rows = es.robustness_evaluations(cnn)
    assert len(rows) == 12 and {r["clean_evaluation_id"] for r in rows} == {env["clean"][cnn]}
    assert {r["suite_sha256"] for r in rows} == {suite.sha256}
    doc = read_json(s.resolved_data_root / rows[0]["metrics_path"])
    diff = doc["paired_difference_corrupted_minus_clean"]["metrics"]["balanced_accuracy"]
    assert diff["point"] == pytest.approx(
        doc["metrics"]["balanced_accuracy"] - doc["clean"]["balanced_accuracy"]
    )
    assert doc["paired_difference_corrupted_minus_clean"]["method"].startswith(
        "paired percentile bootstrap"
    )
    assert doc["condition_hash"] == rows[0]["condition_hash"]
    # majority baseline: corrupted == clean exactly, paired CI degenerate at 0
    evaluate_robustness(
        base,
        suite,
        "val",
        s,
        clean_eval_id=env["clean"][base],
        device="cpu",
        bootstrap_resamples=50,
    )
    with ExperimentStore(s.data_dir("db") / DB_FILENAME) as es:
        brows = es.robustness_evaluations(base)
    for r in brows:
        d = read_json(s.resolved_data_root / r["metrics_path"])
        for m in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert d["metrics"][m] == d["clean"][m]
        p = d["paired_difference_corrupted_minus_clean"]["metrics"]["balanced_accuracy"]
        assert p["point"] == 0.0 and p["low"] == 0.0 and p["high"] == 0.0
    # a clean evaluation of another experiment cannot be pinned
    with pytest.raises(RobustnessError, match="expected"):
        evaluate_robustness(cnn, suite, "val", s, clean_eval_id=env["clean"][base], device="cpu")
    # report over a runs.json for these two runs
    runs = env["root"] / "runs.json"
    runs.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "model": "small_cnn_x",
                        "seed": 0,
                        "status": "completed",
                        "experiment_id": cnn,
                        "val_eval_id": env["clean"][cnn],
                    },
                    {
                        "model": "majority_baseline",
                        "seed": 0,
                        "status": "completed",
                        "experiment_id": base,
                        "val_eval_id": env["clean"][base],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = write_report(s, runs, suite, "val", env["root"] / "out" / "robustness_val.md")
    text = (env["root"] / "out" / "robustness_val.md").read_text(encoding="utf-8")
    assert (
        "VALIDATION (selection-biased) — not test results" in text
        and "THESE corruptions only" in text
    )
    assert "equal for all 12 run x condition pairs" in text and report["missing"] == []
    assert "resolution_upsampling" in text and " robust." not in text.replace("called robust.", "")


def test_tampered_corrupted_set_is_refused(env, tmp_path) -> None:
    s, e, suite, cnn = env["settings"], env["env"], env["suite"], env["cnn"]
    src = M.open_split(s, e.dataset_id, e.split_id, "val")
    root = tmp_path / "copy"
    shutil.copytree(M.suite_root(s, src, suite), root)
    cond = suite.condition("jpeg", "10")
    png = next(M.condition_dir(root, cond).rglob("*.png"))
    png.write_bytes(png.read_bytes() + b"\0")
    with pytest.raises(RobustnessError, match="does not match its sha256"):
        M.load_verified_condition(root, suite, cond, src)
    cond2 = suite.condition("brightness", "0.1")
    meta = M.condition_dir(root, cond2) / "condition.json"
    doc = read_json(meta)
    doc["entries"][0]["corrupted_sha256"] = "0" * 64
    write_json(meta, doc)
    with pytest.raises(RobustnessError, match="condition hash"):
        M.load_verified_condition(root, suite, cond2, src)
    with pytest.raises(RobustnessError):
        evaluate_robustness(cnn, suite, "val", s, device="cpu", cache_root=root)


def test_schema_upgrade_from_v6_adds_the_robustness_table(tmp_path) -> None:
    db = tmp_path / DB_FILENAME
    ExperimentStore(db).close()
    con = sqlite3.connect(db)
    con.executescript("DROP TABLE robustness_evaluations; UPDATE schema_version SET version = 6;")
    con.close()
    with ExperimentStore(db) as s:
        assert s.robustness_evaluations() == []
    con = sqlite3.connect(db)
    assert (
        con.execute("SELECT version FROM schema_version").fetchall()
        == [(SCHEMA_VERSION,)]
        == [(7,)]
    )
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert "robustness_evaluations" in tables
