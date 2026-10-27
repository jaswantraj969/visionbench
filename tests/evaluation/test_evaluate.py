import csv
import hashlib
import io
import json
import sqlite3
from pathlib import Path

import pytest
import torch
import vb_trainenv

from visionbench.cli import main
from visionbench.core.jsonio import read_json
from visionbench.data.split import SplitConfig, make_split
from visionbench.evaluation import efficiency
from visionbench.evaluation.compare import compare_experiments
from visionbench.evaluation.data import (
    EvaluationError,
    RepeatedTestEvaluationError,
    select_eval_entries,
)
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.evaluation.latency import measure_experiment_latency
from visionbench.models.small_cnn import SmallCNN
from visionbench.tracking.store import DB_FILENAME, ExperimentStore, TrackingError
from visionbench.training import data as data_mod
from visionbench.training.data import SplitAccessError, select_entries
from visionbench.training.runner import TrainingError, run_training

FAST = {"bootstrap_resamples": 200, "device": "cpu"}


@pytest.fixture
def trained(train_env):
    cnn = run_training(train_env.config(epochs=2), train_env.settings).experiment_id
    base = run_training(
        train_env.config(model="majority_baseline"), train_env.settings
    ).experiment_id
    return train_env, cnn, base


def _store(env) -> ExperimentStore:
    return ExperimentStore(env.settings.data_dir("db") / DB_FILENAME)


def _relpaths_by_split(env) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for e in env.manifest["entries"]:
        out[e["split"]].add(e["relpath"])
    return out


def test_selectors_both_ways(train_env) -> None:
    m = train_env.manifest
    with pytest.raises(EvaluationError, match="not a supported report"):
        select_eval_entries(m, "train")
    with pytest.raises(SplitAccessError):
        select_entries(m, "test")  # training selector unchanged: still train/val only
    assert {e["split"] for e in select_eval_entries(m, "test")} == {"test"}


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_opens_only_the_requested_split(trained, monkeypatch, split: str) -> None:
    env, cnn, base = trained
    opened: list[Path] = []
    real = data_mod.load_image
    monkeypatch.setattr(data_mod, "load_image", lambda p: opened.append(Path(p)) or real(p))
    evaluate_experiment(cnn, split, env.settings, **FAST)
    rel = {p.relative_to(env.raw_root).as_posix() for p in opened}
    by_split = _relpaths_by_split(env)
    assert rel == by_split[split]
    assert rel.isdisjoint(by_split["train"])
    assert rel.isdisjoint(by_split["test" if split == "val" else "val"])
    opened.clear()
    evaluate_experiment(base, split, env.settings, **FAST)
    assert opened == []  # the baseline needs labels only


def test_val_evaluation_reproduces_training_val_loss(train_env) -> None:
    # lr 1e-3 keeps probabilities away from float64 eps, so log loss is not clipped and must equal
    # the training val_loss (same checkpoint, same deterministic val transform).
    env = train_env
    cnn = run_training(env.config(epochs=2, lr=0.001), env.settings).experiment_id
    r = evaluate_experiment(cnn, "val", env.settings, **FAST)
    with _store(env) as s:
        exp = s.get(cnn)
        best = next(e for e in s.epochs(cnn) if e["epoch"] == exp["best_epoch"])
    m = r.metrics["metrics"]
    assert m["log_loss_clipped_samples"] == 0 and m["log_loss_note"] is None
    assert m["log_loss"] == pytest.approx(best["val_loss"], rel=1e-5)
    assert m["balanced_accuracy"] == pytest.approx(best["val_balanced_accuracy"])
    assert r.metrics["checkpoint"]["epoch"] == exp["best_epoch"]


def test_outputs_artifacts_and_db_row(trained) -> None:
    env, cnn, _ = trained
    r = evaluate_experiment(cnn, "test", env.settings, **FAST)
    files = sorted(p.name for p in r.eval_dir.iterdir())
    assert files == ["confusion_matrix.json", "efficiency.json", "metrics.json", "predictions.csv"]
    data_root = env.settings.resolved_data_root
    with _store(env) as s:
        arts = {a["relpath"]: a for a in s.artifacts(cnn)}
        (row,) = s.evaluations(cnn)
    for name in files:
        rel = (r.eval_dir / name).relative_to(data_root).as_posix()
        assert arts[rel]["sha256"] == hashlib.sha256((r.eval_dir / name).read_bytes()).hexdigest()
    pred = (r.eval_dir / "predictions.csv").read_bytes()
    assert (
        row["predictions_sha256"]
        == hashlib.sha256(pred).hexdigest()
        == r.metrics["predictions_sha256"]
    )
    assert row["split"] == "test" and row["checkpoint"] == "best"
    rows = list(csv.DictReader(io.StringIO(pred.decode())))
    assert [x["relpath"] for x in rows] == sorted(_relpaths_by_split(env)["test"])
    assert set(rows[0]) == {
        "relpath",
        "true_label",
        "pred_label",
        "prob_circle",
        "prob_square",
        "prob_triangle",
    }
    acc = sum(x["true_label"] == x["pred_label"] for x in rows) / len(rows)
    assert r.metrics["metrics"]["accuracy"] == pytest.approx(acc)
    eff = read_json(r.eval_dir / "efficiency.json")
    assert eff["params"]["total"] > 0 and eff["checkpoint_size_bytes"] > 0
    assert eff["latency"].startswith("measured separately")
    assert eff["train_seconds"] > 0 and eff["input_size"] == 32


def test_repeat_test_evaluation_requires_force_and_reason(trained) -> None:
    env, cnn, _ = trained
    warnings: list[str] = []
    first = evaluate_experiment(cnn, "test", env.settings, on_warning=warnings.append, **FAST)
    assert first.warning is None and warnings == []
    assert first.metrics["test_use"]["test_evaluation_number"] == 1
    with pytest.raises(RepeatedTestEvaluationError, match="requires --force --reason"):
        evaluate_experiment(cnn, "test", env.settings, **FAST)
    with pytest.raises(EvaluationError, match="non-empty --reason"):
        evaluate_experiment(cnn, "test", env.settings, force=True, reason="  ", **FAST)
    second = evaluate_experiment(
        cnn,
        "test",
        env.settings,
        force=True,
        reason="re-run after bug fix",
        on_warning=warnings.append,
        **FAST,
    )
    assert second.prior_test_evaluations == 1 and "FORCED" in warnings[0]
    assert second.metrics["test_use"] == {
        "prior_test_evaluations_same_checkpoint": 1,
        "prior_test_evaluations_experiment": 1,
        "test_evaluation_number": 2,
        "forced": True,
        "reason": "re-run after bug fix",
    }
    last = evaluate_experiment(cnn, "test", env.settings, checkpoint="last", **FAST)
    assert last.checkpoint == "last" and last.metrics["checkpoint"]["file"] == "last.pt"
    assert last.metrics["test_use"]["forced"] is False  # first test use of "last"
    val = evaluate_experiment(cnn, "val", env.settings, **FAST)
    val_again = evaluate_experiment(cnn, "val", env.settings, **FAST)  # val repeats are free
    assert val.warning is None and val_again.metrics["test_use"]["forced"] is False
    with _store(env) as s:
        rows = s.evaluations(cnn)
    assert [(e["split"], e["forced"], e["reason"]) for e in rows] == [
        ("test", False, None),
        ("test", True, "re-run after bug fix"),
        ("test", False, None),
        ("val", False, None),
        ("val", False, None),
    ]


def test_baseline_last_checkpoint_is_an_error(trained) -> None:
    env, _, base = trained
    with pytest.raises(EvaluationError, match="majority_baseline has no 'last' checkpoint"):
        evaluate_experiment(base, "val", env.settings, checkpoint="last", **FAST)


def test_checkpoint_sha_mismatch_aborts(trained) -> None:
    env, cnn, _ = trained
    with _store(env) as s:
        run_dir = env.settings.resolved_data_root / s.get(cnn)["run_dir"]
    best = run_dir / "checkpoints" / "best.pt"
    best.write_bytes(best.read_bytes() + b"tamper")
    with pytest.raises(EvaluationError, match="sha256 mismatch"):
        evaluate_experiment(cnn, "test", env.settings, **FAST)
    with _store(env) as s:
        assert s.evaluations(cnn) == []


def test_fingerprint_mismatch_aborts(trained) -> None:
    env, cnn, _ = trained
    path = env.split_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_fingerprint"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrainingError, match="fingerprint mismatch"):
        evaluate_experiment(cnn, "test", env.settings, **FAST)


def test_baseline_efficiency_and_latency_not_applicable(trained) -> None:
    env, _, base = trained
    r = evaluate_experiment(base, "test", env.settings, device="cpu", bootstrap_resamples=50)
    eff = read_json(r.eval_dir / "efficiency.json")
    assert eff["params"] == {"total": None, "trainable": None}
    assert r.checkpoint == "baseline"
    with pytest.raises(EvaluationError, match="not applicable to majority_baseline"):
        measure_experiment_latency(base, env.settings, device="cpu", iterations=1, warmup=0)


def test_latency_command_touches_no_split(trained, monkeypatch) -> None:
    env, cnn, _ = trained
    opened: list[Path] = []
    monkeypatch.setattr(data_mod, "load_image", lambda p: opened.append(Path(p)))
    r = measure_experiment_latency(
        cnn, env.settings, device="cpu", warmup=1, iterations=2, batch_sizes=(1, 2)
    )
    assert opened == []  # random input only; no dataset image is read
    m = r.measurement
    assert m["device"] == "cpu" and m["input_size"] == 32 and m["checkpoint"]["name"] == "best"
    assert set(m["batch_sizes"]) == {"1", "2"}
    data_root = env.settings.resolved_data_root
    with _store(env) as s:
        (row,) = s.latencies(cnn)
        arts = {a["relpath"]: a for a in s.artifacts(cnn)}
    rel = r.path.relative_to(data_root).as_posix()
    assert row["id"] == r.latency_id and row["path"] == rel and arts[rel]["kind"] == "latency"
    assert arts[rel]["sha256"] == hashlib.sha256(r.path.read_bytes()).hexdigest()
    assert json.loads(row["results_json"]) == m["batch_sizes"]


def test_measure_latency_fields() -> None:
    model = SmallCNN(3)
    lat = efficiency.measure_latency(
        model, 32, torch.device("cpu"), batch_sizes=(1, 4), warmup=1, iterations=3
    )
    assert lat["applicable"] and lat["device"] == "cpu" and lat["input_size"] == 32
    assert lat["torch_threads"] == torch.get_num_threads() and lat["iterations"] == 3
    assert "preprocessing" in lat["scope"] and set(lat["batch_sizes"]) == {"1", "4"}
    for v in lat["batch_sizes"].values():
        assert 0 < v["median_ms"] <= v["p95_ms"] or v["median_ms"] == v["p95_ms"]
    with pytest.raises(ValueError):
        efficiency.measure_latency(model, 32, torch.device("cpu"), iterations=0)


def test_compare_refuses_mismatched_splits(trained) -> None:
    env, cnn, _ = trained
    other = make_split(
        env.settings.data_dir("datasets"),
        env.dataset_id,
        SplitConfig(seed=1, fractions=(0.5, 0.25, 0.25)),
    ).split_id
    exp2 = run_training(
        env.config(model="majority_baseline", split_id=other), env.settings
    ).experiment_id
    for exp in (cnn, exp2):
        evaluate_experiment(exp, "test", env.settings, **FAST)
    with pytest.raises(EvaluationError, match="refusing to compare"):
        compare_experiments([cnn, exp2], "test", env.settings)


def test_compare_requires_evaluations(trained) -> None:
    env, cnn, base = trained
    with pytest.raises(EvaluationError, match="run `visionbench evaluate"):
        compare_experiments([cnn, base], "test", env.settings)


def test_compare_report_flags_and_paired_differences(trained, tmp_path, monkeypatch) -> None:
    env, cnn, base = trained
    vb_trainenv.fake_resnet18_weights(tmp_path, monkeypatch)
    rn = run_training(
        env.config(model="resnet18", weights="imagenet", input_size=None, epochs=1, max_steps=1),
        env.settings,
    ).experiment_id
    for exp in (cnn, base, rn):
        evaluate_experiment(exp, "test", env.settings, **FAST)
    evaluate_experiment(cnn, "test", env.settings, force=True, reason="demo repeat", **FAST)
    measure_experiment_latency(cnn, env.settings, device="cpu", warmup=0, iterations=1)
    report = compare_experiments(
        [cnn, rn, base],
        "test",
        env.settings,
        resamples=200,
        note="pipeline test on synthetic data",
        latency_device="cpu",
    )
    assert report["reference_experiment"] == base  # majority baseline by default
    codes = {c["code"] for c in report["confounds"]}
    assert {
        "input_resolution_differs",
        "pretrained_vs_scratch",
        "single_seed_per_model",
        "training_budget_differs",
        "repeated_test_evaluation",
        "forced_test_reevaluation",
    } <= codes
    assert "latency_devices_differ" not in codes
    res = next(c for c in report["confounds"] if c["code"] == "input_resolution_differs")
    assert res["values"] == {cnn: 32, rn: 224}
    rows = {r["experiment_id"]: r for r in report["rows"]}
    assert "paired_difference_vs_reference" not in rows[base]
    diff = rows[cnn]["paired_difference_vs_reference"]["metrics"]["balanced_accuracy"]
    assert diff["low"] <= diff["point"] <= diff["high"] or diff["low"] == diff["high"]
    assert rows[cnn]["test"]["balanced_accuracy_ci"]["low"] is not None
    assert rows[cnn]["test_evaluations"] == 2
    assert rows[cnn]["forced_test_reasons"] == ["demo repeat"]
    assert rows[cnn]["latency"]["applicable"] and rows[cnn]["latency_device"] == "cpu"
    assert rows[rn]["latency_bs1_median_ms"] is None
    assert rows[rn]["latency_note"].startswith("no latency measured on cpu")
    assert rows[base]["latency_note"] == "majority_baseline has no forward pass"
    assert "winner_on_split" not in report
    assert report["val_ranking"]["basis"].startswith("validation balanced accuracy")
    data_root = env.settings.resolved_data_root
    md = (data_root / report["paths"]["markdown"]).read_text(encoding="utf-8")
    assert "**pipeline test on synthetic data**" in md
    assert "without a winner label" in md and "Paired differences" in md
    assert "input_resolution_differs" in md and "single_seed_per_model" in md
    assert '(forced: "demo repeat")' in md and "measured on **cpu** only" in md
    assert "Selection bias" not in md
    saved = read_json(data_root / report["paths"]["json"])
    assert saved["confounds"] == report["confounds"]


def test_schema_upgrade_from_v1(tmp_path: Path) -> None:
    db = tmp_path / DB_FILENAME
    ExperimentStore(db).close()
    con = sqlite3.connect(db)
    con.execute("DROP TABLE evaluations")
    con.execute("UPDATE schema_version SET version = 1")
    con.commit()
    con.close()
    with ExperimentStore(db) as s:
        assert s.evaluations() == []
    con = sqlite3.connect(db)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
    con.execute("UPDATE schema_version SET version = 99")
    con.commit()
    con.close()
    with pytest.raises(TrackingError, match="schema version"):
        ExperimentStore(db)


def test_cli_evaluate_compare_list(trained, monkeypatch, capsys) -> None:
    env, cnn, base = trained
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(env.settings.resolved_data_root))
    args = ["--split", "test", "--device", "cpu", "--bootstrap", "100"]
    assert main(["evaluate", cnn, *args]) == 0
    out = capsys.readouterr().out
    assert "on TEST with checkpoint 'best'" in out and "balanced_accuracy=" in out
    assert main(["evaluate", cnn, *args]) == 1
    assert "requires --force --reason" in capsys.readouterr().err
    assert main(["evaluate", cnn, *args, "--force", "--reason", "cli check"]) == 0
    assert "FORCED repeated TEST evaluation #2" in capsys.readouterr().err
    assert main(["evaluate", base, *args]) == 0
    capsys.readouterr()
    assert main(["compare", cnn, base, "--split", "test", "--bootstrap", "100"]) == 0
    assert "confound flags:" in capsys.readouterr().out
    assert main(["evaluations", "list", cnn]) == 0
    assert capsys.readouterr().out.count("split=test") == 2
    with pytest.raises(SystemExit) as exc:  # argparse rejects --split train (usage error)
        main(["evaluate", cnn, "--split", "train"])
    assert exc.value.code == 2
    assert main(["compare", cnn, "--split", "test"]) == 1


def test_competition_ranks_mark_ties() -> None:
    from visionbench.evaluation.compare import competition_ranks

    assert competition_ranks({"a": 1.0, "b": 1.0, "c": 0.8, "d": 0.3}) == {
        "a": 1,
        "b": 1,
        "c": 3,
        "d": 4,
    }


def test_evaluations_ordered_by_insertion_within_same_timestamp(tmp_path, monkeypatch) -> None:
    from visionbench.tracking import store as store_mod

    monkeypatch.setattr(store_mod, "utcnow", lambda: "2026-01-01T00:00:00+00:00")
    with ExperimentStore(tmp_path / DB_FILENAME) as s:
        exp = s.create(
            dataset_id="ds",
            split_id="sp",
            dataset_fingerprint="f",
            model="m",
            seed=0,
            config={},
            run_dir="experiments/{exp_id}",
        )
        for eid in ("ev-c", "ev-a", "ev-b"):  # not alphabetical: insertion order must win
            s.add_evaluation(
                eval_id=eid,
                experiment_id=exp,
                split="test",
                checkpoint="best",
                metrics_path="m.json",
                predictions_sha256="0" * 64,
                visionbench_version="t",
            )
        assert [e["id"] for e in s.evaluations(exp)] == ["ev-c", "ev-a", "ev-b"]


def test_schema_upgrade_from_v2_adds_forced_columns(tmp_path: Path) -> None:
    db = tmp_path / DB_FILENAME
    con = sqlite3.connect(db)  # a v2 database as written by M3 (no forced/reason columns)
    con.executescript(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at VARCHAR(32));"
        "INSERT INTO schema_version VALUES (2, 'x');"
        "CREATE TABLE evaluations (id VARCHAR(40) PRIMARY KEY, experiment_id VARCHAR(40),"
        " split VARCHAR(8), checkpoint VARCHAR(16), created_at VARCHAR(32),"
        " metrics_path VARCHAR(255), predictions_sha256 VARCHAR(64),"
        " visionbench_version VARCHAR(32));"
        "INSERT INTO evaluations VALUES ('ev-old', 'exp-x', 'test', 'best', 't', 'm', 'h', 'v');"
    )
    con.commit()
    con.close()
    with ExperimentStore(db) as s:
        (row,) = s.evaluations()
    assert row["id"] == "ev-old" and row["forced"] is False and row["reason"] is None
    con = sqlite3.connect(db)
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(3,)]
    con.close()


def test_compare_pins_eval_ids_and_val_bias_note(trained) -> None:
    env, cnn, base = trained
    first = evaluate_experiment(cnn, "val", env.settings, **FAST)
    evaluate_experiment(cnn, "val", env.settings, **FAST)  # newer: used unless pinned
    b = evaluate_experiment(base, "val", env.settings, **FAST)
    t = evaluate_experiment(cnn, "test", env.settings, **FAST)
    last = evaluate_experiment(cnn, "val", env.settings, checkpoint="last", **FAST)
    report = compare_experiments(
        [cnn, base], "val", env.settings, resamples=50, eval_ids=[first.eval_id], write=False
    )
    rows = {r["experiment_id"]: r for r in report["rows"]}
    assert rows[cnn]["eval_id"] == first.eval_id and rows[cnn]["eval_pinned"] is True
    assert rows[base]["eval_id"] == b.eval_id and rows[base]["eval_pinned"] is False
    assert "selection-biased" in report["selection_bias_note"]
    for bad, match in (
        ([t.eval_id], "is on split 'test'"),
        ([last.eval_id], "not the val-selected one"),
        ([first.eval_id, first.eval_id], "more than one evaluation pinned"),
    ):
        with pytest.raises(EvaluationError, match=match):
            compare_experiments([cnn, base], "val", env.settings, eval_ids=bad, write=False)
