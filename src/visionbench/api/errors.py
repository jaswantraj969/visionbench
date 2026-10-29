"""API error mapping and path-scrubbing JSON responses.

Every error body is ``{"error": {"code": str, "message": str}}``. Every JSON body (success or
error) passes through :class:`ScrubbedJSONResponse`, which replaces the absolute data root,
import root and torch home with placeholders, so no response reveals filesystem layout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

from visionbench.core.errors import (
    DatasetExistsError,
    DatasetNotFoundError,
    DatasetValidationError,
    SplitError,
    VisionBenchError,
)
from visionbench.core.redact import scrub_paths
from visionbench.data.archive import ArchiveRejected, ImportPathError
from visionbench.evaluation.data import EvaluationError, RepeatedTestEvaluationError
from visionbench.explain.gradcam import ExplanationNotSupported
from visionbench.inference.service import InferenceError
from visionbench.jobs.store import QueueFullError
from visionbench.tracking.store import NotFoundError, TrackingError
from visionbench.training.config import ConfigError


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _finite(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    return obj


def make_response_class(roots: list[tuple[Path | None, str]]) -> type[JSONResponse]:
    class ScrubbedJSONResponse(JSONResponse):
        def render(self, content: Any) -> bytes:
            text = json.dumps(
                _finite(content),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                default=str,
            )
            return scrub_paths(text, roots).encode("utf-8")

    return ScrubbedJSONResponse


def map_exception(exc: Exception) -> tuple[int, str]:
    """(HTTP status, error code) for library exceptions; order matters (subclasses first)."""
    table: list[tuple[type[Exception], int, str]] = [
        (RepeatedTestEvaluationError, 409, "test_reevaluation_requires_force"),
        (ExplanationNotSupported, 422, "explanation_not_supported"),
        (QueueFullError, 429, "queue_full"),
        (DatasetNotFoundError, 404, "dataset_not_found"),
        (NotFoundError, 404, "not_found"),
        (DatasetExistsError, 409, "dataset_exists"),
        (DatasetValidationError, 422, "dataset_invalid"),
        (SplitError, 422, "split_invalid"),
        (EvaluationError, 422, "evaluation_invalid"),
        (ConfigError, 422, "invalid_config"),
        (TrackingError, 409, "conflict"),
        (VisionBenchError, 400, "bad_request"),
    ]
    if isinstance(exc, InferenceError):
        return exc.status, exc.code
    if isinstance(exc, ArchiveRejected):
        return 422, f"archive_{exc.code}"
    if isinstance(exc, ImportPathError):
        return (404 if exc.code == "path_not_found" else 403), exc.code
    for cls, status, code in table:
        if isinstance(exc, cls):
            return status, code
    return 500, "internal_error"
