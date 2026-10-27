"""API + real `visionbench worker` subprocess: queued training, progress, cancel, health."""

import time

import pytest
from vb_api import RecordingClient
from vb_worker import WorkerProcess, wait_for

JOB_TIMEOUT = 180  # generous: each job spawns a fresh Python + torch import


@pytest.fixture
def api_worker(train_env, tmp_path, request):
    api = RecordingClient(train_env.settings)
    max_jobs = request.node.get_closest_marker("max_jobs").args[0]  # then it exits by itself
    w = WorkerProcess(train_env.settings, tmp_path / "worker.log", max_jobs=max_jobs)
    yield api, w
    w.stop()


def _train(api, env, **overrides) -> dict:
    r = api.post("/api/experiments", json=env.config(**overrides).model_dump(mode="json"))
    assert r.status_code == 202, r.text
    return r.json()["job"]


def _wait_done(api, job_id: str, timeout: float = JOB_TIMEOUT) -> dict:
    return wait_for(
        lambda: (
            (j := api.get(f"/api/jobs/{job_id}").json())["status"]
            in ("completed", "failed", "cancelled")
            and j
        ),
        timeout,
        what=f"job {job_id}",
    )


@pytest.mark.max_jobs(2)
def test_worker_runs_api_jobs_with_progress_and_health_stays_fast(
    train_env, api_worker, report_line
) -> None:
    api, _ = api_worker
    job = _train(api, train_env, epochs=40)
    progress, health_times = [], []
    while (j := api.get(f"/api/jobs/{job['id']}").json())["status"] in ("queued", "running"):
        if j["status"] == "running" and j["progress"]:
            progress.append(j["progress"])
            t0 = time.perf_counter()
            assert api.get("/api/health").status_code == 200
            health_times.append(time.perf_counter() - t0)
        time.sleep(0.1)  # polling interval
    assert j["status"] == "completed", j
    assert any(p.get("phase") == "training" for p in progress)
    exp_id = j["result"]["experiment_id"]
    epochs = api.get(f"/api/experiments/{exp_id}").json()["epochs"]  # may stop early
    assert j["progress"]["epoch"] == len(epochs) >= 2
    report_line(
        f"[measured] GET /api/health while training: n={len(health_times)}, "
        f"max={max(health_times) * 1000:.1f} ms"
    )
    assert len(health_times) >= 5 and max(health_times) < 1.0  # API is never blocked by training
    ev = _wait_done(
        api,
        api.post(
            f"/api/experiments/{exp_id}/evaluations",
            json={"split": "val", "device": "cpu", "bootstrap": 50},
        ).json()["job"]["id"],
    )
    assert ev["status"] == "completed" and ev["result"]["split"] == "val"
    assert api.absolute_root_leaks() == []


@pytest.mark.max_jobs(1)
def test_cancel_via_api_keeps_partial_history(train_env, api_worker) -> None:
    api, _ = api_worker
    job = _train(api, train_env, epochs=500)
    wait_for(
        lambda: (api.get(f"/api/jobs/{job['id']}").json()["progress"] or {}).get("epoch", 0) >= 1,
        JOB_TIMEOUT,
        what="first epoch",
    )
    r = api.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    done = _wait_done(api, job["id"], 60)
    assert done["status"] == "cancelled" and done["error"] == "cancelled by request"
    exp = api.get(f"/api/experiments/{done['result']['experiment_id']}").json()
    assert exp["status"] == "cancelled" and 1 <= len(exp["epochs"]) < 500
    again = api.post(f"/api/jobs/{job['id']}/cancel")  # terminal: returned unchanged
    assert again.status_code == 200 and again.json()["finished_at"] == done["finished_at"]
