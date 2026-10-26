import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visionbench import __version__
from visionbench.api import main as api_main
from visionbench.core.config import Settings


def _client(data_root: Path) -> TestClient:
    return TestClient(api_main.create_app(Settings(_env_file=None, data_root=data_root)))


def test_health(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": __version__}


def test_ready_with_writable_data_root(tmp_path: Path) -> None:
    root = tmp_path / "var"
    r = _client(root).get("/api/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "checks": {"data_root": "ok", "torch": "ok"}}
    assert all((root / d).is_dir() for d in ("datasets", "artifacts", "db"))
    assert not any(p.name.startswith(".write-probe-") for p in root.rglob("*"))


def test_ready_with_unwritable_data_root(tmp_path: Path) -> None:
    # A path beneath a regular file can never be created as a directory, on any OS
    # (directory chmod is not enforced on Windows, so permissions are not used here).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    r = _client(blocker / "var").get("/api/ready")
    assert r.status_code == 503
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "not_ready"
    assert "not writable" in body["error"]["message"]


def test_ready_when_torch_import_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path)
    # A None entry in sys.modules makes `import torch` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    r = client.get("/api/ready")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "not_ready"
    assert "torch import failed" in r.json()["error"]["message"]


def test_unknown_route_uses_error_schema(tmp_path: Path) -> None:
    r = _client(tmp_path).get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "http_404", "message": "Not Found"}}


def test_method_not_allowed_uses_error_schema(tmp_path: Path) -> None:
    r = _client(tmp_path).post("/api/health")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "http_405"


def test_unhandled_exception_uses_error_schema(tmp_path: Path) -> None:
    app = api_main.create_app(Settings(_env_file=None, data_root=tmp_path))

    @app.get("/api/_boom")
    def boom() -> None:
        raise RuntimeError("secret detail")

    r = TestClient(app, raise_server_exceptions=False).get("/api/_boom")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "internal server error"}}
