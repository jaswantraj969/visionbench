"""FastAPI application factory.

The API is a thin layer: it validates input, reads state, and enqueues jobs for the single
worker (``visionbench worker``); no training, evaluation or import runs inside a request.
There is no authentication: bind to localhost only (``visionbench serve`` refuses other hosts).

Every error response uses ``{"error": {"code": str, "message": str}}``; every JSON body is
scrubbed of absolute paths (data root, import root, torch home).
"""

from __future__ import annotations

import importlib
import logging
import tempfile

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from visionbench import __version__
from visionbench.api.errors import ApiError, make_response_class, map_exception
from visionbench.core.config import Settings, get_settings
from visionbench.core.errors import VisionBenchError
from visionbench.core.redact import scrub_paths

logger = logging.getLogger(__name__)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def check_data_root(settings: Settings) -> str | None:
    """Return ``None`` if data_root and its subdirs are writable, else a generic reason.

    The reason is returned to clients, so it never contains paths; details are logged.
    """
    try:
        dirs = settings.ensure_data_dirs()
        for path in dirs.values():
            with tempfile.NamedTemporaryFile(dir=path, prefix=".write-probe-"):
                pass
    except OSError as exc:
        logger.error("data root %s is not writable: %s", settings.resolved_data_root, exc)
        return "data root is not writable"
    return None


def check_torch() -> str | None:
    try:
        importlib.import_module("torch")
    except Exception as exc:  # any import failure means not ready
        return f"torch import failed: {exc!r}"
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    roots = settings.redaction_roots()
    response_class = make_response_class(roots)
    app = FastAPI(
        title="VisionBench",
        version=__version__,
        default_response_class=response_class,
        description="Local benchmarking API. No authentication: bind to 127.0.0.1 only.",
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    def err(status: int, code: str, message: str) -> JSONResponse:
        body = {"error": {"code": code, "message": scrub_paths(message, roots)}}
        return response_class(status_code=status, content=body)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return err(exc.status, exc.code, exc.message)

    @app.exception_handler(VisionBenchError)
    async def _library_error(_: Request, exc: VisionBenchError) -> JSONResponse:
        status, code = map_exception(exc)
        return err(status, code, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return err(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return err(422, "validation_error", "; ".join(parts))

    @app.exception_handler(Exception)
    async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return err(500, "internal_error", "internal server error")

    @app.get("/api/health", tags=["service"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/ready", tags=["service"])
    def ready() -> JSONResponse:
        for check in (lambda: check_data_root(settings), check_torch):
            reason = check()
            if reason is not None:
                return err(503, "not_ready", reason)
        return response_class({"status": "ready", "checks": {"data_root": "ok", "torch": "ok"}})

    from visionbench.api.routes import datasets, experiments

    for router in (
        datasets.router,
        experiments.experiments,
        experiments.evaluations,
        experiments.comparisons,
        experiments.jobs,
    ):
        app.include_router(router)
    return app


app = create_app()
