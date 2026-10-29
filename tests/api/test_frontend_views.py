"""Read-only views for the frontend (overview, benchmarks, sample, suites) and static serving."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from vb_api import LOCAL_BASE_URL, RecordingClient

from visionbench.api.main import create_app
from visionbench.core.config import Settings
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.training.runner import run_training

SUITE = Path("benchmarks/eurosat-v1/robustness/robustness_suite.yaml")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    import vb_trainenv

    root = tmp_path_factory.mktemp("views")
    env = vb_trainenv.make_train_env(root)
    exps = []
    for seed in (0, 1):
        exp = run_training(env.config(epochs=1, seed=seed), env.settings).experiment_id
        ev = evaluate_experiment(exp, "val", env.settings, device="cpu", bootstrap_resamples=20)
        exps.append((exp, ev.eval_id))
    bdir = root / "benchmarks" / "toy-v1"
    (bdir / "robustness").mkdir(parents=True)
    (bdir / "BENCHMARK_PROTOCOL.md").write_text("# Toy protocol\n", encoding="utf-8")
    (bdir / "small_cnn.yaml").write_text("model: small_cnn\n", encoding="utf-8")
    hashes = "# comment line\n" + "".join(
        f"{_sha(bdir / n)} *{n}\n" for n in ("BENCHMARK_PROTOCOL.md", "small_cnn.yaml")
    )
    (bdir / "PROTOCOL_HASHES.txt").write_text(hashes, encoding="utf-8")
    shutil.copy(SUITE, bdir / "robustness" / "robustness_suite.yaml")
    (bdir / "robustness" / "ROBUSTNESS_HASHES.txt").write_text(
        f"{_sha(bdir / 'robustness' / 'robustness_suite.yaml')} *robustness_suite.yaml\n",
        encoding="utf-8",
    )
    runs = [
        {"model": "small_cnn", "seed": s, "status": "completed", "attempt": 1,
         "experiment_id": e, "val_eval_id": v}
        for s, (e, v) in enumerate(exps)
    ]  # fmt: skip
    (bdir / "runs.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    (bdir / "pilot.json").write_text(json.dumps({"pilot_experiment_ids": ["exp-x"]}), "utf-8")
    settings = Settings(
        _env_file=None,
        data_root=env.settings.data_root,
        benchmarks_dir=root / "benchmarks",
        robustness_suites_dir=root / "benchmarks",
        frontend_dist=root / "no-dist",
    )
    return {"env": env, "settings": settings, "exps": exps, "dir": bdir, "root": root}


def test_overview_and_benchmark_views(bench) -> None:
    api = RecordingClient(bench["settings"])
    o = api.get("/api/overview").json()
    assert o["datasets"] == 1 and o["experiments"] == 2 and o["test_images_unlocked"] is False
    assert [b["name"] for b in o["benchmarks"]] == ["toy-v1"]
    b0 = o["benchmarks"][0]
    assert b0["protocol_hashes"]["ok"] and b0["robustness_hashes"]["ok"] and b0["runs"] == 2
    assert len(b0["protocol_hashes"]["files"]) == 2  # the comment line is not an entry
    d = api.get("/api/benchmarks/toy-v1").json()
    assert [r["seed"] for r in d["runs"]] == [0, 1]
    assert d["runs"][0]["stopped_reason"] == "epochs_completed"
    assert d["runs"][0]["test_evaluations"] == 0 and d["runs"][0]["val"]["balanced_accuracy"] >= 0
    s = d["val_summary"][0]
    assert s["model"] == "small_cnn" and s["balanced_accuracy"]["n"] == 2
    assert d["val_label"].startswith("VALIDATION") and d["final_report"] is None
    assert d["documents"]["benchmark_protocol"] and not d["documents"]["final_report"]
    doc = api.get("/api/benchmarks/toy-v1/documents/benchmark_protocol").json()
    assert doc["markdown"] == "# Toy protocol\n"
    assert api.get("/api/benchmarks/toy-v1/documents/final_report").status_code == 404
    assert api.get("/api/benchmarks/toy-v1/documents/..%2Fsecret").status_code in (404, 422)
    assert api.get("/api/benchmarks/nope").status_code == 404
    assert api.get("/api/benchmarks/Bad Name!").status_code == 422
    tags = {e["id"]: e["tags"] for e in api.get("/api/experiments").json()}
    assert tags[bench["exps"][0][0]] == ["benchmark:toy-v1"]
    one = api.get(f"/api/experiments/{bench['exps'][1][0]}").json()
    assert one["tags"] == ["benchmark:toy-v1"]
    suites = api.get("/api/robustness-suites").json()
    assert suites[0]["name"] == "eurosat-c-v1" and len(suites[0]["conditions"]) == 12
    assert api.absolute_root_leaks() == []


def test_tampered_protocol_file_shows_hash_failure(bench, tmp_path) -> None:
    root = tmp_path / "benchmarks"
    shutil.copytree(bench["dir"], root / "toy-v1")
    (root / "toy-v1" / "small_cnn.yaml").write_text("model: resnet18\n", encoding="utf-8")
    s = bench["settings"].model_copy(update={"benchmarks_dir": root})
    b = RecordingClient(s).get("/api/benchmarks/toy-v1").json()
    assert b["protocol_hashes"]["ok"] is False
    bad = [f for f in b["protocol_hashes"]["files"] if not f["ok"]]
    assert [f["name"] for f in bad] == ["small_cnn.yaml"]


def test_seeded_sample_and_test_lock(bench) -> None:
    env, s = bench["env"], bench["settings"]
    api = RecordingClient(s)
    url = f"/api/datasets/{env.dataset_id}/splits/{env.split_id}/sample"
    a = api.get(url, params={"split": "val", "per_class": 2, "seed": 0}).json()
    b = api.get(url, params={"split": "val", "per_class": 2, "seed": 0}).json()
    c = api.get(url, params={"split": "val", "per_class": 2, "seed": 1}).json()
    assert a == b and a["images"] and all(i["split"] == "val" for i in a["images"])
    val = {e["relpath"] for e in env.manifest["entries"] if e["split"] == "val"}
    assert {i["relpath"] for i in a["images"]} <= val
    assert len(a["images"]) <= 2 * 3 and a["rule"].startswith("numpy default_rng")
    assert c["seed"] == 1
    r = api.get(url, params={"split": "test"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "test_split_locked"
    unlocked = RecordingClient(s.model_copy(update={"test_images_unlocked": True}))
    assert unlocked.get(url, params={"split": "test"}).status_code == 200
    bad = f"/api/datasets/{env.dataset_id}/splits/sp-000000000000/sample"
    assert api.get(bad, params={"split": "val"}).status_code == 404
    assert api.get(f"/api/datasets/{env.dataset_id}/splits/{env.split_id}/audit").status_code == 404


def test_static_frontend_serving_and_route_precedence(bench, tmp_path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    s = bench["settings"].model_copy(update={"frontend_dist": dist})
    client = TestClient(create_app(s), base_url=LOCAL_BASE_URL)
    assert client.get("/").text.startswith("<!doctype html>")
    spa = client.get("/experiments/exp-20000101-000000-abcdef")  # client-side route
    assert spa.status_code == 200 and "id=root" in spa.text
    asset = client.get("/assets/app.js")
    assert asset.status_code == 200 and asset.text == "console.log(1)"
    assert client.get("/api/health").json()["status"] == "ok"  # API keeps precedence
    missing = client.get("/api/no-such-endpoint")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "http_404"
    trav = client.get("/..%2Fsecret.txt")
    assert "outside" not in trav.text
    assert client.post("/experiments").status_code == 403  # client-header rule unchanged
    bad_host = client.get("/", headers={"host": "evil.example"})
    assert bad_host.status_code == 400 and bad_host.json()["error"]["code"] == "invalid_host"
    assert client.get("/").headers["x-content-type-options"] == "nosniff"
    # no dist folder: no SPA route at all
    plain = TestClient(create_app(bench["settings"]), base_url=LOCAL_BASE_URL)
    assert plain.get("/").status_code == 404
