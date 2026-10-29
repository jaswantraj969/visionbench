"""Synchronous prediction and Grad-CAM for one image (served from the API process).

Both endpoints take EITHER a multipart upload (``file``: the image; ``target_class``: optional,
Grad-CAM only) OR a JSON body ``{"dataset_id", "relpath"[, "target_class"]}`` naming a recorded
dataset image. The request body is read on the event loop (bounded), and the model work runs in
the thread pool via :func:`run_in_threadpool`, so the event loop is never blocked; at most
``inference_max_concurrent`` requests run at once (429 ``inference_busy`` otherwise). See
:mod:`visionbench.inference.service` for loading, the cache and the TEST-split lock.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool

from visionbench.api.deps import DATASET_ID, EXPERIMENT_ID, settings_of
from visionbench.api.errors import ApiError
from visionbench.api.upload import receive_image_upload
from visionbench.inference.service import (
    DatasetSource,
    ImageSource,
    InferenceService,
    UploadSource,
)

router = APIRouter(prefix="/api/experiments", tags=["inference"])
ExperimentId = PathParam(pattern=EXPERIMENT_ID)
MAX_JSON_BYTES = 8 << 10


class DatasetImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=DATASET_ID)
    relpath: str = Field(min_length=1, max_length=1024)


class DatasetGradcamRequest(DatasetImageRequest):
    target_class: str | None = Field(default=None, min_length=1, max_length=200)


def _body_doc(gradcam: bool) -> dict[str, Any]:
    extra = {"target_class": {"type": "string"}} if gradcam else {}
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}} | extra,
                    }
                },
                "application/json": {
                    "schema": (
                        DatasetGradcamRequest if gradcam else DatasetImageRequest
                    ).model_json_schema()
                },
            },
        }
    }


async def _read_source(request: Request, *, gradcam: bool) -> tuple[ImageSource, str | None]:
    settings = settings_of(request)
    ctype, _ = parse_options_header(request.headers.get("content-type", ""))
    if ctype == b"multipart/form-data":
        up = await receive_image_upload(request, settings.inference_max_upload_mb)
        target = up.fields.get("target_class") or None
        if target is not None and not gradcam:
            raise ApiError(422, "validation_error", "target_class is only accepted by gradcam")
        return UploadSource(up.data, up.filename), target
    if ctype == b"application/json":
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_JSON_BYTES:
                raise ApiError(413, "body_too_large", "JSON body is too large")
        model = DatasetGradcamRequest if gradcam else DatasetImageRequest
        try:
            body = model.model_validate(json.loads(raw or b"null"))
        except (ValidationError, ValueError) as exc:
            raise ApiError(422, "validation_error", f"invalid request body: {exc}") from exc
        return DatasetSource(body.dataset_id, body.relpath), getattr(body, "target_class", None)
    raise ApiError(
        415, "unsupported_media_type", "send multipart/form-data (file) or application/json"
    )


def _service(request: Request) -> InferenceService:
    return request.app.state.inference


@router.post("/{experiment_id}/predict", openapi_extra=_body_doc(False))
async def predict(request: Request, experiment_id: str = ExperimentId) -> dict[str, Any]:
    """Predicted class and the probabilities of all classes (sorted) for one image."""
    source, _ = await _read_source(request, gradcam=False)
    return await run_in_threadpool(_service(request).predict, experiment_id, source)


@router.post("/{experiment_id}/gradcam", openapi_extra=_body_doc(True))
async def gradcam(request: Request, experiment_id: str = ExperimentId) -> dict[str, Any]:
    """Grad-CAM of the target class (default: the predicted class) for one image, with an
    overlay on the original image that marks the regions the model never saw."""
    source, target = await _read_source(request, gradcam=True)
    return await run_in_threadpool(_service(request).gradcam, experiment_id, source, target)
