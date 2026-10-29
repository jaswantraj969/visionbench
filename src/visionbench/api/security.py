"""Protection for a local API without authentication (CSRF and DNS rebinding).

Threat: a web page open in the user's browser can make the browser send requests to
``http://127.0.0.1:<port>``. CORS only stops that page from READING responses. "Simple" requests
(e.g. a multipart form POST, or a body-less POST) are still sent and executed, and a DNS
rebinding attack (attacker hostname re-pointed at 127.0.0.1) makes the browser treat the API as
same-origin, so it can read responses too.

Defences (pure ASGI, so upload bodies keep streaming):

- **Host check**: the ``Host`` header must be ``localhost``, ``127.0.0.1`` or ``[::1]`` (any
  port), plus ``Settings.extra_trusted_hosts``; otherwise 400 ``invalid_host``. A rebinding
  page sends its own hostname as ``Host``, so it is rejected for every method, including GET.
- **Client header**: POST/PUT/PATCH/DELETE require ``X-VisionBench-Request: 1``; otherwise 403
  ``missing_client_header``. A browser only sends a custom header cross-origin after a CORS
  preflight, which a foreign origin fails, so a foreign page cannot produce such a request.
- **Origin check**: if an unsafe request carries ``Origin``, it must be one of
  ``Settings.cors_origins`` or the API's own origin (``http(s)://`` + the request's Host, which
  has already passed the local-host check: the built frontend served by ``visionbench serve``
  is same-origin, and browsers send ``Origin`` on same-origin POSTs too); otherwise 403
  ``origin_not_allowed`` (defence in depth). A foreign page cannot forge ``Origin``, and a DNS
  rebinding page fails the Host check first.
- ``X-Content-Type-Options: nosniff`` on every response.

GET requests need no header (they change nothing). Non-browser clients (curl, scripts) just
send the header.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CLIENT_HEADER = "x-visionbench-request"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def host_of(host_header: str) -> str:
    """Hostname part of a Host header, lower-cased (IPv6 literals keep their brackets)."""
    value = host_header.strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


class LocalRequestGuard:
    """Host allow-list, client header + Origin check for unsafe methods, nosniff."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: Iterable[str],
        allowed_origins: Iterable[str],
        error_response: Callable[[int, str, str], Any],
    ) -> None:
        self.app = app
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.allowed_origins = set(allowed_origins)
        self.error_response = error_response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_nosniff(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Content-Type-Options"] = "nosniff"
            await send(message)

        rejection = self.check(scope)
        if rejection is not None:
            await self.error_response(*rejection)(scope, receive, send_nosniff)
            return
        await self.app(scope, receive, send_nosniff)

    def check(self, scope: Scope) -> tuple[int, str, str] | None:
        headers = Headers(scope=scope)
        if host_of(headers.get("host", "")) not in self.allowed_hosts:
            return 400, "invalid_host", "Host header is not an allowed local host name"
        if scope["method"] in UNSAFE_METHODS:
            if headers.get(CLIENT_HEADER) != "1":
                return (
                    403,
                    "missing_client_header",
                    "state-changing requests need the header 'X-VisionBench-Request: 1'",
                )
            origin = headers.get("origin")
            host = headers.get("host", "").strip().lower()
            own = {f"http://{host}", f"https://{host}"}
            if (
                origin is not None
                and origin not in self.allowed_origins
                and origin.lower() not in own
            ):
                return 403, "origin_not_allowed", "request Origin is not an allowed CORS origin"
        return None
