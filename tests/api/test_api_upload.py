"""Upload endpoint: streaming limit, archive rejection, cleanup, imports, CORS."""

from pathlib import Path

import pytest
import vb_synth
from vb_api import RecordingClient, chunked, make_settings, multipart, upload

from visionbench.jobs.store import JobStore
from visionbench.tracking.store import DB_FILENAME


@pytest.fixture
def api(tmp_path: Path) -> RecordingClient:
    return RecordingClient(make_settings(tmp_path, max_upload_mb=1))


def _uploads_dir_files(api: RecordingClient) -> list[str]:
    tmp = api.settings.data_dir("tmp")
    return sorted(p.name for p in tmp.rglob("*") if p.is_file())


def test_valid_upload_is_queued(api: RecordingClient, tmp_path: Path) -> None:
    data = vb_synth.shapes_zip(tmp_path / "s.zip").read_bytes()
    r = upload(api, data)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["kind"] == "dataset_import" and job["status"] == "queued"
    assert (
        job["params"]["source"] == "upload" and job["params"]["original_filename"] == "shapes.zip"
    )
    assert _uploads_dir_files(api) == [f"{job['params']['upload_id']}.zip"]


@pytest.mark.parametrize("use_chunked", [False, True])
def test_oversized_upload_413_while_streaming(api: RecordingClient, use_chunked: bool) -> None:
    body, ctype = multipart({"name": "big"}, "big.zip", b"PK\x03\x04" + b"x" * (3 << 20))
    content = chunked(body) if use_chunked else body
    r = api.post("/api/datasets/uploads", content=content, headers={"content-type": ctype})
    assert r.status_code == 413 and r.json()["error"]["code"] == "upload_too_large"
    assert _uploads_dir_files(api) == []


def test_non_zip_rejected(api: RecordingClient) -> None:
    r = upload(api, b"\x89PNG\r\n\x1a\n not a zip")
    assert r.status_code == 422 and r.json()["error"]["code"] == "archive_not_a_zip"
    assert _uploads_dir_files(api) == []


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("../evil.png", "archive_parent_traversal"),
        ("C:/evil.png", "archive_drive_letter"),
        ("cats/CON.png", "archive_reserved_name"),
    ],
)
def test_unsafe_archive_rejected_before_queueing(api, tmp_path, name, code) -> None:
    data = vb_synth.write_zip(
        tmp_path / "bad.zip", [("cats/a.png", b"x"), (name, b"x")], raw_names=True
    ).read_bytes()
    r = upload(api, data)
    assert r.status_code == 422 and r.json()["error"]["code"] == code
    assert _uploads_dir_files(api) == []
    with JobStore(api.settings.data_dir("db") / DB_FILENAME) as js:
        assert js.list() == []


def test_bad_requests(api: RecordingClient, tmp_path: Path) -> None:
    data = vb_synth.shapes_zip(tmp_path / "s.zip").read_bytes()
    body, ctype = multipart({"name": "x"}, None, None)
    r = api.post("/api/datasets/uploads", content=body, headers={"content-type": ctype})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_file"
    r = api.post("/api/datasets/uploads", content=data, headers={"content-type": "application/zip"})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_media_type"
    body, ctype = multipart({}, "s.zip", data)
    r = api.post("/api/datasets/uploads", content=body, headers={"content-type": ctype})
    assert r.status_code == 422 and r.json()["error"]["code"] == "dataset_invalid"
    assert _uploads_dir_files(api) == []


def test_queue_full_429_and_upload_removed(tmp_path: Path) -> None:
    api = RecordingClient(make_settings(tmp_path, max_queued_jobs=1))
    data = vb_synth.shapes_zip(tmp_path / "s.zip").read_bytes()
    assert upload(api, data).status_code == 202
    r = upload(api, data)
    assert r.status_code == 429 and r.json()["error"]["code"] == "queue_full"
    assert len(_uploads_dir_files(api)) == 1  # only the first (queued) upload remains


def test_cancelling_queued_import_removes_upload(api: RecordingClient, tmp_path: Path) -> None:
    job = upload(api, vb_synth.shapes_zip(tmp_path / "s.zip").read_bytes()).json()["job"]
    r = api.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert _uploads_dir_files(api) == []


def test_path_imports(tmp_path: Path) -> None:
    disabled = RecordingClient(make_settings(tmp_path))
    r = disabled.post("/api/datasets/imports", json={"path": "x", "name": "x"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "path_import_disabled"
    root = tmp_path / "imports"
    vb_synth.build_shapes(root / "ds", per_class=3, size=32)
    api = RecordingClient(make_settings(tmp_path, import_root=root))
    r = api.post("/api/datasets/imports", json={"path": str(root / "ds"), "name": "ds"})
    assert r.status_code == 202 and r.json()["job"]["params"]["relpath"] == "ds"
    for bad in ("..", str(tmp_path)):
        r = api.post("/api/datasets/imports", json={"path": bad, "name": "x"})
        assert r.status_code == 403 and r.json()["error"]["code"] == "path_outside_import_root"
    assert api.absolute_root_leaks() == []
    assert all(str(root).lower() not in b.lower() for b in api.bodies)


def test_cors_only_configured_origins(api: RecordingClient) -> None:
    ok = api.client.options(
        "/api/datasets",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    evil = api.client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
