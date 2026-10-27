"""API test helpers: settings, client, inline job runner, multipart bodies, path hygiene."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from visionbench.api.main import create_app
from visionbench.api.security import CLIENT_HEADER, UNSAFE_METHODS
from visionbench.core.config import Settings
from visionbench.jobs.store import JobStore
from visionbench.jobs.worker import execute_job
from visionbench.tracking.store import DB_FILENAME

LOCAL_BASE_URL = "http://127.0.0.1"


class RecordingClient:
    """TestClient wrapper that keeps every response body (for the path-hygiene check).

    Requests go to a local Host (127.0.0.1) and state-changing ones carry the required
    ``X-VisionBench-Request: 1`` header unless ``client_header=False``.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = TestClient(create_app(settings), base_url=LOCAL_BASE_URL)
        self.bodies: list[str] = []

    def request(self, method: str, url: str, *, client_header: bool = True, **kw: Any):
        if client_header and method.upper() in UNSAFE_METHODS:
            kw["headers"] = {CLIENT_HEADER: "1"} | dict(kw.get("headers") or {})
        r = self.client.request(method, url, **kw)
        if not r.headers.get("content-type", "").startswith("image/"):
            self.bodies.append(r.text)
        return r

    def get(self, url: str, **kw: Any):
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw: Any):
        return self.request("POST", url, **kw)

    def absolute_root_leaks(self) -> list[str]:
        root = str(self.settings.resolved_data_root)
        forms = {root, root.replace("\\", "/"), root.replace("\\", "\\\\")}
        return [b for b in self.bodies for f in forms if f.lower() in b.lower()]


def make_settings(tmp_path: Path, **kw: Any) -> Settings:
    return Settings(_env_file=None, data_root=tmp_path / "var", **kw)


def run_jobs_inline(settings: Settings) -> list[str]:
    """Run every queued job in this process (the same execute_job the worker child uses)."""
    done = []
    with JobStore(settings.data_dir("db") / DB_FILENAME) as js:
        while (job := js.claim(os.getpid())) is not None:
            execute_job(job["id"], settings)
            done.append(job["id"])
    return done


def multipart(
    fields: dict[str, str], filename: str | None, data: bytes | None, boundary: str | None = None
) -> tuple[bytes, str]:
    boundary = boundary or uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    if data is not None:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"'
            f"\r\nContent-Type: application/zip\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def chunked(data: bytes, size: int = 64 << 10) -> Iterator[bytes]:
    """A body without Content-Length: forces the streaming byte limit (not the header check)."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


def upload(
    client: RecordingClient,
    zip_bytes: bytes,
    name: str = "shapes",
    filename: str = "shapes.zip",
    **kw: Any,
):
    body, ctype = multipart({"name": name}, filename, zip_bytes)
    headers = {"content-type": ctype} | kw.pop("headers", {})
    return client.post("/api/datasets/uploads", content=body, headers=headers, **kw)
