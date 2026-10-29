"""Prediction and Grad-CAM endpoints (synchronous, in the API process)."""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import vb_synth
from fastapi.testclient import TestClient
from PIL import Image
from vb_api import LOCAL_BASE_URL, RecordingClient
from vb_trainenv import TrainEnv, fake_resnet18_weights

from visionbench.api.main import create_app
from visionbench.api.routes import inference as inference_routes
from visionbench.api.security import CLIENT_HEADER
from visionbench.core.config import Settings
from visionbench.inference.service import InferenceService
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.runner import run_training

CLIENT = {CLIENT_HEADER: "1"}


def _settings(env: TrainEnv, **kw) -> Settings:
    base = {"inference_wait_seconds": 0.2, "inference_max_upload_mb": 1}
    return Settings(_env_file=None, data_root=env.settings.data_root, **(base | kw))


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[TrainEnv, str, str]:
    import vb_trainenv

    env = vb_trainenv.make_train_env(tmp_path_factory.mktemp("inference"))
    cnn = run_training(env.config(epochs=2), env.settings).experiment_id
    base = run_training(env.config(model="majority_baseline"), env.settings).experiment_id
    return env, cnn, base


def _rel(env: TrainEnv, split: str) -> str:
    return sorted(e["relpath"] for e in env.manifest["entries"] if e["split"] == split)[0]


def _png(img: Image.Image) -> bytes:
    return vb_synth.encode(img)


def _multipart(data: bytes, filename: str = "x.png", **fields: str) -> tuple[bytes, str]:
    from vb_api import multipart

    return multipart(fields, filename, data)


def _post_image(api: RecordingClient, url: str, data: bytes, **fields: str):
    body, ctype = _multipart(data, **fields)
    return api.post(url, content=body, headers={"content-type": ctype})


def test_predict_val_image_and_baseline(trained) -> None:
    env, cnn, base = trained
    api = RecordingClient(_settings(env))
    rel = _rel(env, "val")
    r = api.post(
        f"/api/experiments/{cnn}/predict", json={"dataset_id": env.dataset_id, "relpath": rel}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert probs == sorted(probs, reverse=True) and sum(probs) == pytest.approx(1.0)
    assert body["predicted_class"] == body["probabilities"][0]["class"]
    assert {p["class"] for p in body["probabilities"]} == {"circle", "square", "triangle"}
    assert body["input"]["split"] == "val"
    assert body["input"]["original_size"] == [32, 32] and body["warnings"] == []
    assert any("VALIDATION" in n for n in body["notes"])
    assert body["cache"] == "miss" and body["timing_ms"]["model_load"] > 0
    assert body["timing_ms"]["forward"] > 0 and body["device"] == "cpu"
    with ExperimentStore(env.settings.data_dir("db") / DB_FILENAME) as es:
        arts = {a["relpath"].rsplit("/", 1)[-1]: a["sha256"] for a in es.artifacts(cnn)}
    assert body["checkpoint"]["name"] == "best" and body["checkpoint"]["sha256"] == arts["best.pt"]
    assert body["preprocessing"]["val_transforms"][0].startswith("Resize(size=(32, 32)")
    again = api.post(
        f"/api/experiments/{cnn}/predict", json={"dataset_id": env.dataset_id, "relpath": rel}
    )
    assert again.json()["cache"] == "hit" and again.json()["timing_ms"]["model_load"] is None
    assert again.json()["probabilities"] == body["probabilities"]  # deterministic

    r = api.post(
        f"/api/experiments/{base}/predict", json={"dataset_id": env.dataset_id, "relpath": rel}
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["model"] == "majority_baseline" and b["timing_ms"]["forward"] is None
    assert any("does not look at the image" in n for n in b["notes"])
    train = _rel(env, "train")
    r = api.post(
        f"/api/experiments/{cnn}/predict", json={"dataset_id": env.dataset_id, "relpath": train}
    )
    assert r.status_code == 200 and any("TRAIN image" in n for n in r.json()["notes"])
    assert api.absolute_root_leaks() == []


def test_test_split_lock(trained) -> None:
    env, cnn, base = trained
    api = RecordingClient(_settings(env))
    rel = _rel(env, "test")
    ref = {"dataset_id": env.dataset_id, "relpath": rel}
    for exp in (cnn, base):
        for kind in ("predict", "gradcam"):
            r = api.post(f"/api/experiments/{exp}/{kind}", json=ref)
            if exp == base and kind == "gradcam":
                assert r.status_code == 422  # not supported, before any image is read
                continue
            assert r.status_code == 403, (exp, kind, r.text)
            assert r.json()["error"]["code"] == "test_split_locked"
    # the same bytes uploaded are locked too (identical file = test image)
    test_bytes = (env.raw_root / Path(rel)).read_bytes()
    r = _post_image(api, f"/api/experiments/{cnn}/predict", test_bytes)
    assert r.status_code == 403 and r.json()["error"]["code"] == "test_split_locked"
    unlocked = RecordingClient(_settings(env, test_images_unlocked=True))
    r = unlocked.post(f"/api/experiments/{cnn}/predict", json=ref)
    assert r.status_code == 200 and r.json()["input"]["split"] == "test"
    assert any("TEST image" in w for w in r.json()["warnings"])


def test_uploads(trained) -> None:
    env, cnn, _ = trained
    api = RecordingClient(_settings(env))
    rng = np.random.default_rng(5)
    r = _post_image(
        api, f"/api/experiments/{cnn}/predict", _png(vb_synth.synthetic_image(rng, size=64))
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input"]["type"] == "upload" and body["input"]["original_size"] == [64, 64]
    assert any("out-of-distribution" in w and "32x32" in w for w in body["warnings"])
    same_size = _png(vb_synth.synthetic_image(rng, size=32))
    r = _post_image(api, f"/api/experiments/{cnn}/predict", same_size)
    assert r.status_code == 200 and r.json()["warnings"] == []
    rgba = Image.new("RGBA", (32, 32), (10, 200, 30, 128))
    r = _post_image(api, f"/api/experiments/{cnn}/predict", _png(rgba))
    assert r.status_code == 200 and any("RGBA" in n for n in r.json()["notes"])

    big = np.random.default_rng(0).integers(0, 256, (700, 700, 3), dtype=np.uint8)
    data = _png(Image.fromarray(big))
    assert len(data) > 1 << 20
    r = _post_image(api, f"/api/experiments/{cnn}/predict", data)
    assert r.status_code == 413 and r.json()["error"]["code"] == "upload_too_large"
    r = _post_image(api, f"/api/experiments/{cnn}/predict", b"this is not an image")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_image"
    gif = vb_synth.encode(vb_synth.synthetic_image(rng, size=32), "GIF")
    assert _post_image(api, f"/api/experiments/{cnn}/predict", gif).status_code == 415
    body, ctype = _multipart(b"", filename="")
    r = api.post(
        f"/api/experiments/{cnn}/predict",
        content=body.replace(b'name="file"', b'name="x"'),
        headers={"content-type": ctype},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_file"
    r = api.post(
        f"/api/experiments/{cnn}/predict", content=b"abc", headers={"content-type": "text/plain"}
    )
    assert r.status_code == 415
    small = RecordingClient(_settings(env, max_image_pixels=500))
    r = _post_image(small, f"/api/experiments/{cnn}/predict", same_size)  # 32x32 = 1024 px
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"
    r = _post_image(api, f"/api/experiments/{cnn}/predict", same_size, target_class="circle")
    assert r.status_code == 422  # target_class belongs to gradcam
    assert api.absolute_root_leaks() == []


def test_gradcam_scratch_model(trained) -> None:
    env, cnn, base = trained
    api = RecordingClient(_settings(env))
    ref = {"dataset_id": env.dataset_id, "relpath": _rel(env, "val")}
    r = api.post(f"/api/experiments/{cnn}/gradcam", json=ref)
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["cam"]["feature_map_size"] == [4, 4] and g["cam"]["resolution"] == "4x4"
    values = np.array(g["cam"]["values"])
    assert values.shape == (4, 4) and values.min() >= 0 and values.max() <= 1
    assert g["cam"]["all_zero"] or values.max() == pytest.approx(1.0)
    assert g["target_class"] == g["predicted_class"]
    assert g["geometry"]["unseen_fraction"] == 0.0 and g["geometry"]["input_size"] == 32
    assert "not a causal explanation" in g["limitations"]
    png = base64.b64decode(g["overlay"]["png"].removeprefix("data:image/png;base64,"))
    overlay = Image.open(io.BytesIO(png))
    assert overlay.format == "PNG" and overlay.width >= 256
    other = next(c for c in ("circle", "square", "triangle") if c != g["predicted_class"])
    r2 = api.post(f"/api/experiments/{cnn}/gradcam", json=ref | {"target_class": other})
    assert r2.status_code == 200 and r2.json()["target_class"] == other
    r = api.post(f"/api/experiments/{cnn}/gradcam", json=ref | {"target_class": "hexagon"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_class"
    r = api.post(f"/api/experiments/{base}/gradcam", json=ref)
    assert r.status_code == 422 and r.json()["error"]["code"] == "explanation_not_supported"
    body, ctype = _multipart(
        _png(vb_synth.synthetic_image(np.random.default_rng(1), size=32)), target_class=other
    )
    r = api.post(f"/api/experiments/{cnn}/gradcam", content=body, headers={"content-type": ctype})
    assert r.status_code == 200 and r.json()["target_class"] == other
    r = api.post(f"/api/experiments/{cnn}/gradcam", json=ref | {"extra": 1})
    assert r.status_code == 422
    assert api.absolute_root_leaks() == []


def test_gradcam_center_crop_model_marks_the_unseen_border(trained, tmp_path, monkeypatch) -> None:
    env, _, _ = trained
    fake_resnet18_weights(tmp_path, monkeypatch)
    cfg = env.config(
        model="resnet18", weights="imagenet", input_size=None, epochs=1, max_steps=1, batch_size=4
    )
    exp = run_training(cfg, env.settings).experiment_id
    api = RecordingClient(_settings(env))
    r = api.post(
        f"/api/experiments/{exp}/gradcam",
        json={"dataset_id": env.dataset_id, "relpath": _rel(env, "val")},
    )
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["cam"]["feature_map_size"] == [7, 7] and g["geometry"]["input_size"] == 224
    # 32 px -> resize 256 -> centre crop 224: a 2 px border of the original is never seen
    assert g["geometry"]["resized_size"] == [256, 256] and g["geometry"]["crop_offset"] == [16, 16]
    assert g["geometry"]["seen_box_in_original"] == {
        "left": 2.0,
        "top": 2.0,
        "right": 30.0,
        "bottom": 30.0,
    }
    assert g["geometry"]["unseen_fraction"] == pytest.approx(1 - (28 / 32) ** 2)
    png = base64.b64decode(g["overlay"]["png"].split(",", 1)[1])
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    shown = g["overlay"]["display_size"]
    assert shown == {"width": 256, "height": 256}  # 8x nearest-neighbour enlargement
    border = arr[:16, :256]  # display rows 0..15 = original rows 0..1: unseen
    assert (border[..., 0] == border[..., 1]).all() and (border[..., 1] == border[..., 2]).all()
    assert (arr[0, 0] == [230, 230, 230]).all()  # hatch line
    centre = arr[100:156, 100:156]
    assert not ((centre[..., 0] == centre[..., 1]) & (centre[..., 1] == centre[..., 2])).all()
    p = api.post(
        f"/api/experiments/{exp}/predict",
        json={"dataset_id": env.dataset_id, "relpath": _rel(env, "val")},
    )
    assert p.status_code == 200
    assert (
        p.json()["preprocessing"]["resize"] == 256
        and p.json()["preprocessing"]["input_size"] == 224
    )


def test_busy_cache_and_client_header(trained) -> None:
    env, cnn, base = trained
    settings = _settings(env, inference_cache_size=1)
    api = RecordingClient(settings)
    service: InferenceService = api.client.app.state.inference
    ref = {"dataset_id": env.dataset_id, "relpath": _rel(env, "val")}
    assert service._slots.acquire(timeout=1)  # simulate a request holding the only slot
    try:
        t0 = time.perf_counter()
        r = api.post(f"/api/experiments/{cnn}/predict", json=ref)
        assert r.status_code == 429 and r.json()["error"]["code"] == "inference_busy"
        assert time.perf_counter() - t0 < 5
    finally:
        service._slots.release()
    assert api.post(f"/api/experiments/{cnn}/predict", json=ref).json()["cache"] == "miss"
    assert api.post(f"/api/experiments/{cnn}/predict", json=ref).json()["cache"] == "hit"
    assert api.post(f"/api/experiments/{base}/predict", json=ref).json()["cache"] == "miss"
    assert [k[0] for k in service.cached_keys()] == [base]  # LRU size 1: cnn evicted
    assert api.post(f"/api/experiments/{cnn}/predict", json=ref).json()["cache"] == "miss"
    r = api.post(f"/api/experiments/{cnn}/predict", json=ref, client_header=False)
    assert r.status_code == 403 and r.json()["error"]["code"] == "missing_client_header"
    r = api.post("/api/experiments/exp-20000101-000000-abcdef/predict", json=ref)
    assert r.status_code == 404


def test_tampered_checkpoint_is_not_served_from_the_cache(trained, tmp_path) -> None:
    env, _, _ = trained
    exp = run_training(env.config(epochs=1), env.settings).experiment_id
    api = RecordingClient(_settings(env))
    ref = {"dataset_id": env.dataset_id, "relpath": _rel(env, "val")}
    assert api.post(f"/api/experiments/{exp}/predict", json=ref).status_code == 200
    best = env.settings.resolved_data_root / "experiments" / exp / "checkpoints" / "best.pt"
    best.write_bytes(best.read_bytes() + b"\0")
    r = api.post(f"/api/experiments/{exp}/predict", json=ref)
    assert r.status_code == 422 and "sha256 mismatch" in r.json()["error"]["message"]


def test_inference_does_not_block_the_event_loop(trained, monkeypatch, report_line) -> None:
    """A slow inference request runs in the thread pool: /api/health still answers at once.
    Control: with the thread pool bypassed, the same request blocks the loop."""
    env, cnn, _ = trained
    ref = {"dataset_id": env.dataset_id, "relpath": _rel(env, "val")}
    original = InferenceService.predict

    def slow_predict(self, *a, **kw):
        time.sleep(1.5)
        return original(self, *a, **kw)

    monkeypatch.setattr(InferenceService, "predict", slow_predict)

    def health_latency_during_predict(client: TestClient) -> float:
        result = {}
        t = threading.Thread(
            target=lambda: result.setdefault(
                "r", client.post(f"/api/experiments/{cnn}/predict", json=ref, headers=CLIENT)
            )
        )
        t.start()
        time.sleep(0.3)
        t0 = time.perf_counter()
        assert client.get("/api/health").status_code == 200
        latency = time.perf_counter() - t0
        t.join()
        assert result["r"].status_code == 200
        return latency

    app = create_app(_settings(env))
    with TestClient(app, base_url=LOCAL_BASE_URL) as client:
        threadpool = health_latency_during_predict(client)
    assert threadpool < 0.5

    async def inline(func, *args):  # the anti-pattern: run the model on the event loop
        return func(*args)

    monkeypatch.setattr(inference_routes, "run_in_threadpool", inline)
    with TestClient(create_app(_settings(env)), base_url=LOCAL_BASE_URL) as client:
        blocking = health_latency_during_predict(client)
    report_line(
        f"[measured] GET /api/health during a 1.5 s predict: {threadpool * 1000:.1f} ms "
        f"(thread pool); control with the model run on the event loop: {blocking * 1000:.1f} ms"
    )
    assert blocking > 0.9  # control: the loop was blocked
