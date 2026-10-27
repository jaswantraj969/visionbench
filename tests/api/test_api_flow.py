"""Whole API flow with jobs executed inline (same execute_job as the worker child)."""

from pathlib import Path
from urllib.parse import quote

import pytest
import vb_synth
from vb_api import RecordingClient, make_settings, run_jobs_inline, upload


@pytest.fixture
def api(tmp_path: Path) -> RecordingClient:
    return RecordingClient(make_settings(tmp_path))


def _job(api: RecordingClient, response) -> dict:
    assert response.status_code == 202, response.text
    job_id = response.json()["job"]["id"]
    run_jobs_inline(api.settings)
    job = api.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job


def _train(api, dataset_id, split_id, **kw) -> str:
    cfg = {
        "dataset_id": dataset_id,
        "split_id": split_id,
        "model": "small_cnn",
        "input_size": 32,
        "epochs": 1,
        "batch_size": 8,
        "device": "cpu",
        "num_workers": 0,
    } | kw
    return _job(api, api.post("/api/experiments", json=cfg))["result"]["experiment_id"]


def test_full_api_flow_and_path_hygiene(api: RecordingClient, tmp_path: Path) -> None:
    zip_bytes = vb_synth.shapes_zip(tmp_path / "s.zip", per_class=8, top="mydata").read_bytes()
    job = _job(api, upload(api, zip_bytes, name="api-shapes"))
    ds = job["result"]["dataset_id"]

    listing = api.get("/api/datasets").json()
    assert [d["id"] for d in listing] == [ds]
    detail = api.get(f"/api/datasets/{ds}").json()
    assert detail["class_counts"] == {"circle": 8, "square": 8, "triangle": 8}
    assert "source_path" not in detail and detail["report"]["issues"]

    # --- image endpoint: only recorded relpaths are served ---
    rel = detail["report"]["samples"]["relpaths"]["circle"][0]
    img = api.get(f"/api/datasets/{ds}/images", params={"relpath": rel})
    raw = api.settings.data_dir("datasets") / ds / "raw" / Path(rel)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert img.content == raw.read_bytes()
    (raw.parent / "unrecorded.png").write_bytes(raw.read_bytes())  # exists, not in records
    bad = [
        "../dataset.json",
        "circle/../../dataset.json",
        "..%2Fdataset.json",
        str(raw),
        raw.as_posix(),
        "/etc/passwd",
        "circle\\..\\..\\dataset.json",
        "circle/unrecorded.png",
        "CIRCLE/" + rel.split("/")[1],
    ]
    for relpath in bad:
        r = api.get(f"/api/datasets/{ds}/images?relpath={quote(relpath, safe='')}")
        assert r.status_code == 404, relpath
        assert r.json()["error"]["code"] == "image_not_found"
    r = api.get(f"/api/datasets/{ds}/images?relpath=%2e%2e%2fdataset.json")
    assert r.status_code == 404
    assert api.get("/api/datasets/ds-nothexvalue!/images?relpath=x").status_code == 422
    assert api.get("/api/datasets/ds-0123456789ab").status_code == 404

    # --- split, verify ---
    split = _job(
        api,
        api.post(f"/api/datasets/{ds}/splits", json={"seed": 0, "fractions": [0.5, 0.25, 0.25]}),
    )
    sp = split["result"]["split_id"]
    assert split["result"]["leakage_check"]["groups_in_multiple_splits"] == 0
    assert [s["split_id"] for s in api.get(f"/api/datasets/{ds}/splits").json()] == [sp]
    sd = api.get(f"/api/datasets/{ds}/splits/{sp}").json()
    assert "entries" not in sd and sd["entries_total"] == 24 and sd["split_totals"]["test"] == 6
    assert api.get(f"/api/datasets/{ds}/splits/sp-000000000000").status_code == 404
    verify = _job(api, api.post(f"/api/datasets/{ds}/verify", json={"full": True}))
    assert verify["result"]["ok"] is False  # the planted unrecorded.png is detected
    assert verify["result"]["problems"] == [
        {"problem": "unexpected file not in files.json", "relpath": "circle/unrecorded.png"}
    ]
    (raw.parent / "unrecorded.png").unlink()
    verify = _job(api, api.post(f"/api/datasets/{ds}/verify", json={"full": True}))
    assert verify["result"]["ok"] is True and verify["result"]["files_checked"] == 24

    # --- training (queued, run by the "worker") ---
    cnn = _train(api, ds, sp)
    base = _train(api, ds, sp, model="majority_baseline")
    exps = {e["id"]: e for e in api.get("/api/experiments").json()}
    assert exps[cnn]["status"] == "completed" and exps[base]["model"] == "majority_baseline"
    ed = api.get(f"/api/experiments/{cnn}").json()
    assert ed["run_dir"] == f"experiments/{cnn}" and len(ed["epochs"]) == 1
    assert all(not Path(a["relpath"]).is_absolute() for a in ed["artifacts"])
    r = api.post(
        "/api/experiments",
        json={"dataset_id": ds, "split_id": "sp-000000000000", "model": "small_cnn"},
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "split_not_found"

    # --- evaluations + test-use rules (identical to the CLI) ---
    _job(
        api,
        api.post(
            f"/api/experiments/{cnn}/evaluations",
            json={"split": "val", "device": "cpu", "bootstrap": 50},
        ),
    )
    first = _job(
        api,
        api.post(
            f"/api/experiments/{cnn}/evaluations",
            json={"split": "test", "device": "cpu", "bootstrap": 50},
        ),
    )
    again = api.post(f"/api/experiments/{cnn}/evaluations", json={"split": "test"})
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "test_reevaluation_requires_force"
    no_reason = api.post(
        f"/api/experiments/{cnn}/evaluations", json={"split": "test", "force": True}
    )
    assert no_reason.status_code == 422 and "reason" in no_reason.json()["error"]["message"]
    forced = _job(
        api,
        api.post(
            f"/api/experiments/{cnn}/evaluations",
            json={
                "split": "test",
                "force": True,
                "reason": "api check",
                "device": "cpu",
                "bootstrap": 50,
            },
        ),
    )
    assert forced["result"]["forced"] is True and first["result"]["forced"] is False
    ev = api.get(f"/api/evaluations/{forced['result']['eval_id']}").json()
    assert ev["evaluation"]["reason"] == "api check"
    assert ev["metrics"]["test_use"]["test_evaluation_number"] == 2
    assert len(api.get("/api/evaluations", params={"experiment_id": cnn}).json()) == 3
    r = api.post(
        f"/api/experiments/{base}/evaluations", json={"split": "val", "checkpoint": "last"}
    )
    assert r.status_code == 422 and "no 'last' checkpoint" in r.json()["error"]["message"]
    _job(
        api,
        api.post(
            f"/api/experiments/{base}/evaluations",
            json={"split": "test", "device": "cpu", "bootstrap": 50},
        ),
    )

    # --- latency + comparison ---
    lat = _job(
        api,
        api.post(
            f"/api/experiments/{cnn}/latency", json={"device": "cpu", "warmup": 1, "iterations": 3}
        ),
    )
    assert lat["result"]["device"] == "cpu"
    r = api.post(f"/api/experiments/{base}/latency", json={"device": "cpu"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "latency_not_applicable"
    report = api.post(
        "/api/comparisons",
        json={
            "experiment_ids": [cnn, base],
            "split": "test",
            "latency_device": "cpu",
            "bootstrap": 50,
            "note": "pipeline test",
        },
    )
    assert report.status_code == 201, report.text
    rep = report.json()
    rows = {r["experiment_id"]: r for r in rep["rows"]}
    assert rows[cnn]["latency_device"] == "cpu" and rows[cnn]["forced_test_reasons"] == [
        "api check"
    ]
    assert api.get(f"/api/comparisons/{rep['report_id']}").json()["report_id"] == rep["report_id"]
    assert api.get("/api/comparisons/rp-20000101-000000-abcdef").status_code == 404
    r = api.post("/api/comparisons", json={"experiment_ids": [cnn, "nope"], "split": "test"})
    assert r.status_code == 422

    # --- jobs ---
    jobs = api.get("/api/jobs").json()
    assert len(jobs) == 11 and all(j["status"] == "completed" for j in jobs)
    assert "worker_pid" not in jobs[0]
    queued = api.post(f"/api/datasets/{ds}/verify", json={}).json()["job"]
    assert api.post(f"/api/jobs/{queued['id']}/cancel").json()["status"] == "cancelled"
    assert api.get("/api/jobs", params={"status": "cancelled"}).json()[0]["id"] == queued["id"]
    assert api.get("/api/jobs/job-20000101-000000-abcdef").status_code == 404
    assert api.get("/api/jobs/not-a-job").status_code == 422
    assert api.get("/api/experiments/exp-bad").status_code == 422

    # --- path hygiene over every JSON body of the whole flow ---
    assert len(api.bodies) > 40
    assert api.absolute_root_leaks() == []
