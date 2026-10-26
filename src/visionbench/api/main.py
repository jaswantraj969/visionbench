"""FastAPI application factory.

Every error response uses the schema ``{"error": {"code": str, "message": str}}``.
"""

from __future__ import annotations

import importlib
import logging
import tempfile

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from visionbench import __version__
from visionbench.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def check_data_root(settings: Settings) -> str | None:
    """Return ``None`` if data_root and its subdirs are writable, else a reason string."""
    try:
        dirs = settings.ensure_data_dirs()
        for path in dirs.values():
            with tempfile.NamedTemporaryFile(dir=path, prefix=".write-probe-"):
                pass
    except OSError as exc:
        return f"data_root {settings.resolved_data_root} is not writable: {exc}"
    return None


def check_torch() -> str | None:
    try:
        importlib.import_module("torch")
    except Exception as exc:  # any import failure means not ready
        return f"torch import failed: {exc!r}"
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="VisionBench", version=__version__)
    app.state.settings = settings

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(422, "validation_error", str(exc.errors()))

    @app.exception_handler(Exception)
    async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return error_response(500, "internal_error", "internal server error")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/ready")
    def ready() -> JSONResponse:
        for check in (lambda: check_data_root(settings), check_torch):
            reason = check()
            if reason is not None:
                return error_response(503, "not_ready", reason)
        return JSONResponse({"status": "ready", "checks": {"data_root": "ok", "torch": "ok"}})

    return app


app = create_app()
