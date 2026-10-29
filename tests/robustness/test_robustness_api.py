"""Robustness API and worker job (synthetic data); fixture in conftest.py."""

from __future__ import annotations

import shutil
from pathlib import Path

import vb_trainenv
from vb_api import RecordingClient, run_jobs_inline

from visionbench.core.config import Settings
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.training.runner import run_training

SUITE_FILE = Path("benchmarks/eurosat-v1/robustness/robustness_suite.yaml")


def test_api_test_split_is_locked(env) -> None:
    api = RecordingClient(env["settings"])
    r = api.post(
        f"/api/experiments/{env['cnn']}/robustness",
        json={"suite": env["suite"].name, "split": "test"},
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "test_split_locked"


def test_api_robustness_job_and_reads(env) -> None:
    s, suite = env["settings"], env["suite"]
    exp = run_training(env["env"].config(epochs=1), s).experiment_id
    evaluate_experiment(exp, "val", s, device="cpu", bootstrap_resamples=20)
    api = RecordingClient(s)
    r = api.post(
        f"/api/experiments/{exp}/robustness",
        json={"suite": suite.name, "split": "val"},
        client_header=False,
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "missing_client_header"
    r = api.post(
        f"/api/experiments/{exp}/robustness", json={"suite": "no-such-suite", "split": "val"}
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "suite_not_found"
    r = api.post(
        f"/api/experiments/{exp}/robustness",
        json={"suite": suite.name, "split": "val", "path": "x"},
    )
    assert r.status_code == 422
    r = api.post(
        f"/api/experiments/{exp}/robustness",
        json={"suite": suite.name, "split": "val", "device": "cpu"},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job"]["id"]
    run_jobs_inline(s)
    job = api.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    assert job["result"]["evaluated"] == 12 and job["progress"]["conditions_done"] == 12
    listing = api.get("/api/robustness", params={"experiment_id": exp}).json()
    assert len(listing) == 12
    one = api.get(f"/api/robustness/{listing[0]['id']}").json()
    assert one["metrics"]["robustness_id"] == listing[0]["id"]
    assert api.get("/api/robustness/rb-20000101-000000-abcdef").status_code == 404
    assert api.absolute_root_leaks() == []


def test_api_refuses_unmaterialized_suite(env, tmp_path) -> None:
    other = vb_trainenv.make_train_env(tmp_path)  # different dataset: nothing materialized
    suites = tmp_path / "suites" / "robustness"
    suites.mkdir(parents=True)
    shutil.copy(SUITE_FILE, suites / "robustness_suite.yaml")
    s = Settings(
        _env_file=None,
        data_root=other.settings.data_root,
        robustness_suites_dir=tmp_path / "suites",
    )
    exp = run_training(other.config(epochs=1), s).experiment_id
    r = RecordingClient(s).post(
        f"/api/experiments/{exp}/robustness", json={"suite": "eurosat-c-v1", "split": "val"}
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "corruptions_not_materialized"
