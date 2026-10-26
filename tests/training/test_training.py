import hashlib
import json
from pathlib import Path

import pytest
import torch

from visionbench.cli import main
from visionbench.core.jsonio import read_json
from visionbench.data.split import SplitConfig, make_split
from visionbench.models import pretrained as pretrained_mod
from visionbench.models.pretrained import WeightsUnavailableError
from visionbench.models.registry import build_model
from visionbench.tracking.store import DB_FILENAME, ExperimentStore, TrackingError
from visionbench.training import loop
from visionbench.training.checkpoints import CheckpointError, load_checkpoint
from visionbench.training.config import ConfigError, load_train_config
from visionbench.training.data import build_dataloader
from visionbench.training.preprocessing import (
    Preprocessing,
    build_transforms,
    class_weights_from_train,
)
from visionbench.training.runner import TrainingError, run_training

METRICS = ("train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy")


def _store(env) -> ExperimentStore:
    return ExperimentStore(env.settings.data_dir("db") / DB_FILENAME)


def _rows(env, exp_id: str) -> tuple[dict, list[dict], list[dict]]:
    with _store(env) as s:
        return s.get(exp_id), s.epochs(exp_id), s.artifacts(exp_id)


@pytest.mark.parametrize(
    ("model", "extra"),
    [
        ("majority_baseline", {}),
        ("small_cnn", {}),
        ("resnet18", {"max_steps": 2}),
        ("efficientnet_b0", {"max_steps": 2}),
    ],
)
def test_smoke_each_model(train_env, model: str, extra: dict) -> None:
    result = run_training(train_env.config(model=model, **extra), train_env.settings)
    assert result.status == "completed" and result.history
    exp, epochs, artifacts = _rows(train_env, result.experiment_id)
    assert exp["status"] == "completed" and exp["error"] is None
    assert exp["best_epoch"] == result.best_epoch and exp["train_seconds"] > 0
    assert len(epochs) == len(result.history)
    ckpt = result.run_dir / "checkpoints"
    if model == "majority_baseline":
        assert sorted(p.name for p in ckpt.iterdir()) == ["baseline.json"]
        assert exp["weights_source"] == "none"
    else:
        assert {"best.pt", "best.json", "last.pt", "last.json"} <= {p.name for p in ckpt.iterdir()}
        assert exp["weights_source"] == "scratch"
    data_root = train_env.settings.resolved_data_root
    kinds = {a["kind"] for a in artifacts}
    assert {"config", "env", "history", "log"} <= kinds
    for a in artifacts:
        data = (data_root / a["relpath"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == a["sha256"] and len(data) == a["size_bytes"]
    config = read_json(result.run_dir / "config.json")
    assert config["resolved"]["dataset_fingerprint"] == exp["dataset_fingerprint"]
    env = read_json(result.run_dir / "env.json")
    assert env["device"] == "cpu" and env["determinism"]["torch_deterministic_algorithms"]


def test_scratch_model_records_train_norm_stats_and_resolution(train_env) -> None:
    result = run_training(train_env.config(epochs=1), train_env.settings)
    resolved = read_json(result.run_dir / "config.json")["resolved"]
    norm = resolved["normalization"]
    assert resolved["input_size"] == 32 and norm["source"] == "train_split"
    assert norm["stats"]["train_images"] == sum(resolved["train_class_counts"].values())
    assert resolved["preprocessing"]["mean"] == norm["stats"]["mean"]
    assert norm["cache_file"].startswith("cache/norm_stats/")
    assert any("RandomHorizontalFlip" in t for t in resolved["train_transforms"])
    assert not any("Random" in t for t in resolved["val_transforms"])
    second = run_training(train_env.config(epochs=1), train_env.settings)
    assert read_json(second.run_dir / "config.json")["resolved"]["normalization"][
        "loaded_from_cache"
    ]


def test_cpu_determinism_identical_runs(train_env) -> None:
    cfg = train_env.config(epochs=3)
    a = run_training(cfg, train_env.settings)
    b = run_training(cfg, train_env.settings)
    assert [[r[k] for k in METRICS] for r in a.history] == [
        [r[k] for k in METRICS] for r in b.history
    ]
    data_root = train_env.settings.resolved_data_root
    sa = load_checkpoint(a.run_dir / "checkpoints" / "best.pt", data_root)["model_state"]
    sb = load_checkpoint(b.run_dir / "checkpoints" / "best.pt", data_root)["model_state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_early_stopping_on_plateau_and_best_epoch(train_env) -> None:
    cfg = train_env.config(
        epochs=10, early_stopping={"monitor": "val_loss", "patience": 2, "min_delta": 100.0}
    )
    result = run_training(cfg, train_env.settings)
    assert result.stopped_reason == "early_stopping"
    assert len(result.history) == 3 and result.best_epoch == 1  # 1 best + 2 without improvement
    best = read_json(result.run_dir / "checkpoints" / "best.json")
    assert best["epoch"] == 1 and best["metrics"]["val_loss"] == result.history[0]["val_loss"]
    assert read_json(result.run_dir / "checkpoints" / "last.json")["epoch"] == 3


def test_early_stopping_unit() -> None:
    es = loop.EarlyStopping("val_balanced_accuracy", patience=2, min_delta=0.01)
    assert es.update(1, 0.5) and not es.update(2, 0.505) and es.update(3, 0.6)
    assert not es.update(4, 0.6) and not es.should_stop and not es.update(5, float("nan"))
    assert es.should_stop and (es.best_epoch, es.best_value) == (3, 0.6)
    with pytest.raises(ValueError):
        loop.EarlyStopping("val_accuracy", patience=1)


def test_best_checkpoint_reload_reproduces_val_loss(train_env) -> None:
    result = run_training(train_env.config(epochs=3), train_env.settings)
    data_root = train_env.settings.resolved_data_root
    ckpt = load_checkpoint(result.run_dir / "checkpoints" / "best.pt", data_root)
    assert ckpt["epoch"] == result.best_epoch
    config = read_json(result.run_dir / "config.json")
    torch.manual_seed(123)
    model = build_model("small_cnn", config["resolved"]["class_names"], input_size=32).model
    model.load_state_dict(ckpt["model_state"])
    prep = Preprocessing(**config["resolved"]["preprocessing"])
    _, val_tf = build_transforms(prep, train_env.config().augmentation)
    loader = build_dataloader(
        "val", train_env.manifest, train_env.raw_root, val_tf, batch_size=8, seed=0
    )
    metrics = loop.evaluate(model, loader, torch.device("cpu"), 3)
    expected = result.history[result.best_epoch - 1]["val_loss"]
    assert metrics["val_loss"] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    with pytest.raises(CheckpointError, match="outside"):
        load_checkpoint(result.run_dir / "checkpoints" / "best.pt", data_root / "datasets")


def test_injected_failure_marks_failed_keeps_partial_history(train_env, monkeypatch) -> None:
    real = loop.train_one_epoch
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure in epoch 2")
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "train_one_epoch", flaky)
    with pytest.raises(RuntimeError, match="injected failure"):
        run_training(train_env.config(epochs=4), train_env.settings)
    with _store(train_env) as s:
        (exp,) = s.list()
        epochs, artifacts = s.epochs(exp["id"]), s.artifacts(exp["id"])
    assert exp["status"] == "failed" and exp["finished_at"] is not None
    assert exp["error"] == "RuntimeError: injected failure in epoch 2"
    assert [e["epoch"] for e in epochs] == [1]
    run_dir = train_env.settings.resolved_data_root / exp["run_dir"]
    assert len((run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert "failed: RuntimeError" in (run_dir / "train.log").read_text(encoding="utf-8")
    assert {"history", "checkpoint", "log"} <= {a["kind"] for a in artifacts}


def test_keyboard_interrupt_marks_cancelled(train_env, monkeypatch) -> None:
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(loop, "train_one_epoch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_training(train_env.config(), train_env.settings)
    with _store(train_env) as s:
        (exp,) = s.list()
    assert exp["status"] == "cancelled" and "KeyboardInterrupt" in exp["error"]


def test_fingerprint_mismatch_aborts_before_any_record(train_env) -> None:
    path = train_env.split_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["dataset_fingerprint"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TrainingError, match="fingerprint mismatch"):
        run_training(train_env.config(), train_env.settings)
    with _store(train_env) as s:
        assert s.list() == []


def test_snapshot_tampering_aborts_training(train_env) -> None:
    victim = next(p for p in train_env.raw_root.rglob("*.png"))
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(TrainingError, match="failed verification"):
        run_training(train_env.config(), train_env.settings)


def test_unknown_split_and_amp_on_cpu(train_env) -> None:
    with pytest.raises(TrainingError, match="not found"):
        run_training(train_env.config(split_id="sp-000000000000"), train_env.settings)
    with pytest.raises(TrainingError, match="invalid split id"):
        run_training(train_env.config(split_id="../x"), train_env.settings)
    with pytest.raises(TrainingError, match="amp requires a CUDA device"):
        run_training(train_env.config(amp=True), train_env.settings)
    with _store(train_env) as s:
        assert [e["status"] for e in s.list()] == ["failed"]  # amp check runs inside the run


def test_weight_fallback_in_runner(train_env, monkeypatch) -> None:
    def offline(weights, hub_dir):
        raise OSError("simulated: no network")

    monkeypatch.setattr(pretrained_mod, "fetch_state_dict", offline)
    cfg = train_env.config(model="resnet18", weights="imagenet", max_steps=1, epochs=1)
    with pytest.raises(WeightsUnavailableError, match="--allow-weight-fallback"):
        run_training(cfg, train_env.settings)
    ok = run_training(cfg.model_copy(update={"allow_weight_fallback": True}), train_env.settings)
    with _store(train_env) as s:
        failed, done = s.list()
    assert failed["status"] == "failed" and "allow-weight-fallback" in failed["error"]
    assert done["status"] == "completed" and done["weights_source"] == "scratch"
    model = read_json(ok.run_dir / "config.json")["resolved"]["model"]
    assert model["weights"] == "scratch" and "simulated" in model["weight_fallback_reason"]
    assert "WEIGHT FALLBACK" in (ok.run_dir / "train.log").read_text(encoding="utf-8")


def test_class_weighted_loss_recorded(train_env) -> None:
    result = run_training(train_env.config(class_weighted_loss=True, epochs=1), train_env.settings)
    resolved = read_json(result.run_dir / "config.json")["resolved"]
    assert resolved["class_weights"] == class_weights_from_train(train_env.manifest)


def test_scheduler_and_sgd(train_env) -> None:
    cfg = train_env.config(optimizer="sgd", scheduler="cosine", epochs=2)
    result = run_training(cfg, train_env.settings)
    assert result.history[0]["lr"] == 0.01 and result.history[1]["lr"] < 0.01


def test_config_yaml_overrides_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "dataset_id: ds-x\nsplit_id: sp-y\nmodel: small_cnn\nearly_stopping:\n  patience: 7\n",
        encoding="utf-8",
    )
    cfg = load_train_config(
        path, {"epochs": 3, "early_stopping.monitor": "val_balanced_accuracy", "lr": None}
    )
    assert (cfg.epochs, cfg.early_stopping.patience, cfg.lr) == (3, 7, 1e-3)
    assert cfg.early_stopping.monitor == "val_balanced_accuracy"
    for bad in (
        {"unknown_key": 1},
        {"weights": "imagenet"},
        {"finetune": "head_only"},
        {"epochs": 0},
    ):
        with pytest.raises(ConfigError):
            load_train_config(path, bad)


def test_tracking_state_machine(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "db" / DB_FILENAME) as s:
        exp = s.create(
            dataset_id="ds",
            split_id="sp",
            dataset_fingerprint="f",
            model="m",
            seed=0,
            config={},
            run_dir="experiments/{exp_id}",
        )
        with pytest.raises(TrackingError, match="illegal status transition pending -> completed"):
            s.transition(exp, "completed")
        s.transition(exp, "running")
        s.transition(exp, "completed")
        with pytest.raises(TrackingError):
            s.transition(exp, "running")
        with pytest.raises(TrackingError):
            s.update(exp, status="running")
        assert s.get(exp)["run_dir"] == f"experiments/{exp}"


def test_cli_train_and_experiments(train_env, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(train_env.settings.resolved_data_root))
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        f"dataset_id: {train_env.dataset_id}\nsplit_id: {train_env.split_id}\n"
        "model: small_cnn\ninput_size: 32\nbatch_size: 8\ndevice: cpu\n"
        "num_workers: 0\n",
        encoding="utf-8",
    )
    assert (
        main(["train", "--config", str(cfg), "--epochs", "2", "--set", "early_stopping.patience=1"])
        == 0
    )
    out = capsys.readouterr().out
    exp_id = out.split()[1]
    assert "epoch   1" in out and ": completed" in out
    assert main(["experiments", "list"]) == 0
    assert exp_id in capsys.readouterr().out
    assert main(["experiments", "show", exp_id]) == 0
    shown = capsys.readouterr().out
    assert "status: completed" in shown and "checkpoint" in shown
    assert main(["experiments", "show", exp_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["experiment"]["id"] == exp_id
    assert main(["train", "--config", str(cfg), "--set", "bogus=1"]) == 1
    assert "invalid training config" in capsys.readouterr().err
    assert main(["experiments", "show", "exp-nope"]) == 1


def test_split_reuse_unaffected_by_training(train_env) -> None:
    """Norm-stats cache files live in the split dir but never change the manifest bytes."""
    before = (train_env.split_dir / "manifest.json").read_bytes()
    run_training(train_env.config(epochs=1), train_env.settings)
    again = make_split(
        train_env.settings.data_dir("datasets"),
        train_env.dataset_id,
        SplitConfig(seed=0, fractions=(0.5, 0.25, 0.25)),
    )
    assert again.already_existed and (again.path / "manifest.json").read_bytes() == before
