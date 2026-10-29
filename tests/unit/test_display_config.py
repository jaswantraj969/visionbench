"""The majority baseline's unused training defaults are shown as "n/a" (stored config unchanged)."""

import json

from vb_api import RecordingClient

from visionbench.cli import main
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.config import BASELINE_UNUSED_FIELDS, display_config
from visionbench.training.runner import run_training


def test_display_config_only_changes_the_baseline() -> None:
    cfg = {
        "model": "majority_baseline",
        "seed": 0,
        "epochs": 10,
        "batch_size": 32,
        "weights": "scratch",
    }
    shown = display_config(cfg)
    assert shown == {
        "model": "majority_baseline",
        "seed": 0,
        "epochs": "n/a",
        "batch_size": "n/a",
        "weights": "n/a",
    }
    assert cfg["epochs"] == 10  # input not modified
    cnn = {"model": "small_cnn", "epochs": 10, "batch_size": 32, "weights": "scratch"}
    assert display_config(cnn) == cnn
    assert {"epochs", "batch_size", "weights", "lr", "augmentation"} <= set(BASELINE_UNUSED_FIELDS)


def test_api_and_cli_show_na_for_the_baseline(train_env, monkeypatch, capsys) -> None:
    exp = run_training(
        train_env.config(model="majority_baseline"), train_env.settings
    ).experiment_id
    api = RecordingClient(train_env.settings)
    body = api.get(f"/api/experiments/{exp}").json()
    assert body["config_display"]["epochs"] == "n/a" and body["config_display"]["weights"] == "n/a"
    assert body["config_display"]["model"] == "majority_baseline"
    assert body["config"]["config"]["epochs"] == 2  # stored config unchanged
    with ExperimentStore(train_env.settings.data_dir("db") / DB_FILENAME) as es:
        assert json.loads(es.get(exp)["config_json"])["config"]["batch_size"] == 8
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(train_env.settings.data_root))
    assert main(["experiments", "show", exp]) == 0
    out = capsys.readouterr().out
    assert (
        "  epochs: n/a" in out
        and "  batch_size: n/a" in out
        and "  model: majority_baseline" in out
    )
