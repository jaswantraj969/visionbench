"""Local-API protection: Host allow-list, client header + Origin on unsafe methods, nosniff."""

import pytest
from vb_api import RecordingClient, make_settings, upload
from vb_synth import shapes_zip

from visionbench.api.security import host_of

ALLOWED_ORIGIN = "http://localhost:5173"


@pytest.fixture
def zip_data(tmp_path) -> bytes:
    return shapes_zip(tmp_path / "s.zip").read_bytes()


@pytest.fixture
def api(tmp_path) -> RecordingClient:
    return RecordingClient(make_settings(tmp_path, extra_trusted_hosts=["visionbench.test"]))


def _uploads_dir_empty(api: RecordingClient) -> bool:
    tmp = api.settings.data_dir("tmp")
    return not any(p.is_file() for p in tmp.rglob("*"))


def _code(r) -> str:
    return r.json()["error"]["code"]


def test_upload_needs_client_header_and_allowed_origin(api: RecordingClient, zip_data) -> None:
    data = zip_data
    r = upload(api, data, client_header=False)  # what a cross-site <form> POST looks like
    assert r.status_code == 403 and _code(r) == "missing_client_header"
    assert _uploads_dir_empty(api)  # rejected before a byte was stored
    r = upload(api, data, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and _code(r) == "origin_not_allowed"
    r = upload(api, data, headers={"Origin": "null"})  # sandboxed iframe / file://
    assert r.status_code == 403 and _code(r) == "origin_not_allowed"
    r = upload(api, data, headers={"Origin": ALLOWED_ORIGIN})
    assert r.status_code == 202, r.text
    assert r.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert upload(api, data, name="no-origin").status_code == 202  # curl / scripts
    r = api.post(
        "/api/datasets/uploads", client_header=False, headers={"X-VisionBench-Request": "yes"}
    )
    assert r.status_code == 403 and _code(r) == "missing_client_header"  # value must be "1"


def test_cancel_without_client_header_is_rejected(api: RecordingClient, zip_data) -> None:
    job_id = upload(api, zip_data).json()["job"]["id"]
    r = api.post(f"/api/jobs/{job_id}/cancel", client_header=False)  # body-less simple POST
    assert r.status_code == 403 and _code(r) == "missing_client_header"
    assert api.get(f"/api/jobs/{job_id}").json()["status"] == "queued"  # nothing happened
    r = api.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_other_unsafe_methods_need_the_header_too(api: RecordingClient, method: str) -> None:
    r = api.request(method, "/api/datasets", client_header=False)
    assert r.status_code == 403 and _code(r) == "missing_client_header"
    assert api.request(method, "/api/datasets").status_code == 405  # header ok -> routing


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:8000", "127.0.0.1.evil.example", "localhost.", ""]
)
def test_foreign_host_header_is_rejected(api: RecordingClient, zip_data, host: str) -> None:
    """DNS rebinding: the browser sends the attacker's hostname as Host."""
    for r in (
        api.get("/api/health", headers={"Host": host}),
        api.get("/api/datasets", headers={"Host": host}),
        upload(api, zip_data, headers={"Host": host}),
    ):
        assert r.status_code == 400 and _code(r) == "invalid_host"
        assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.1:8000", "localhost:8000", "LOCALHOST", "[::1]:8000", "[::1]"]
    + ["visionbench.test:8000"],  # from extra_trusted_hosts
)
def test_local_host_headers_are_accepted(api: RecordingClient, host: str) -> None:
    assert api.get("/api/health", headers={"Host": host}).status_code == 200


def test_get_endpoints_need_no_header_and_all_responses_are_nosniff(api: RecordingClient) -> None:
    cases = [
        (api.get("/api/health", client_header=False), 200),
        (api.get("/api/datasets", client_header=False), 200),
        (api.get("/api/jobs", client_header=False), 200),
        (api.get("/api/jobs", headers={"Origin": "https://evil.example"}), 200),
        (api.get("/api/datasets/ds-0123456789ab"), 404),
        (api.get("/api/jobs/not-a-job"), 422),
        (api.post("/api/jobs/job-20260101-000000-abcdef/cancel", client_header=False), 403),
    ]
    for r, status in cases:
        assert r.status_code == status, r.text
        assert r.headers["x-content-type-options"] == "nosniff"
    assert "access-control-allow-origin" not in cases[3][0].headers  # cannot be read


def test_cors_preflight_allows_the_header_only_for_configured_origins(api) -> None:
    def preflight(origin: str):
        return api.client.options(
            "/api/datasets/uploads",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-visionbench-request",
            },
        )

    ok = preflight(ALLOWED_ORIGIN)
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "x-visionbench-request" in ok.headers["access-control-allow-headers"].lower()
    assert ok.headers["x-content-type-options"] == "nosniff"
    evil = preflight("https://evil.example")
    assert evil.status_code == 400 and "access-control-allow-origin" not in evil.headers


def test_host_of() -> None:
    assert host_of("127.0.0.1:8000") == "127.0.0.1"
    assert host_of("[::1]:8000") == "[::1]" and host_of("[::1]") == "[::1]"
    assert host_of("LocalHost") == "localhost"
    assert host_of("::1") == "::1"  # unbracketed IPv6 is not a valid Host: never matches
