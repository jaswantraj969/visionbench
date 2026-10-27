"""Streaming multipart upload: the ZIP is written to ``<data_root>/tmp/uploads/<upload_id>.zip``
chunk by chunk while counting bytes, so an oversized upload is rejected (413) without ever being
buffered in memory (Starlette's ``UploadFile`` would spool the whole body first, unbounded).

Accepted form fields: ``file`` (the archive) and ``name`` (dataset name, small). Anything else
is ignored without being stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO, Any

from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header

from visionbench.api.errors import ApiError
from visionbench.core.config import Settings
from visionbench.data.archive import cleanup_upload, new_upload_id, upload_path

MULTIPART_OVERHEAD = 64 << 10  # boundaries, headers and the small "name" field
MAX_FIELD_BYTES = 1024


class _TooLarge(Exception):
    pass


@dataclass
class ReceivedUpload:
    upload_id: str
    filename: str | None
    size: int
    fields: dict[str, str]


async def receive_zip_upload(request: Request, settings: Settings) -> ReceivedUpload:
    ctype, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise ApiError(415, "unsupported_media_type", "expected multipart/form-data")
    limit = settings.max_upload_mb << 20
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD:
        raise ApiError(413, "upload_too_large", f"upload exceeds {settings.max_upload_mb} MB")

    upload_id = new_upload_id()
    target = upload_path(settings, upload_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "headers": {},
        "field": None,
        "buf": bytearray(),
        "file": None,
        "filename": None,
        "size": 0,
        "seen_file": False,
        "header": b"",
    }
    fields: dict[str, str] = {}
    out: IO[bytes] | None = None

    def on_part_begin() -> None:
        state.update(headers={}, field=None, buf=bytearray(), file=None)

    def on_header_field(data: bytes, start: int, end: int) -> None:
        state["header"] = data[start:end].lower()

    def on_header_value(data: bytes, start: int, end: int) -> None:
        state["headers"][state["header"]] = data[start:end]

    def on_headers_finished() -> None:
        nonlocal out
        _, opts = parse_options_header(state["headers"].get(b"content-disposition", b""))
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        state["field"] = name
        if name == "file":
            if state["seen_file"]:
                raise ApiError(422, "multiple_files", "exactly one file part is allowed")
            state["seen_file"] = True
            raw = opts.get(b"filename")
            state["filename"] = raw.decode("utf-8", "replace")[-255:] if raw else None
            out = open(target, "xb")  # noqa: SIM115 - closed in on_part_end / finally

    def on_part_data(data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if state["field"] == "file" and out is not None:
            state["size"] += len(chunk)
            if state["size"] > limit:
                raise _TooLarge
            out.write(chunk)
        elif state["field"] == "name":
            state["buf"] += chunk
            if len(state["buf"]) > MAX_FIELD_BYTES:
                raise ApiError(422, "field_too_large", "form field 'name' is too long")

    def on_part_end() -> None:
        nonlocal out
        if state["field"] == "file" and out is not None:
            out.close()
            out = None
        elif state["field"] == "name":
            fields["name"] = state["buf"].decode("utf-8", "replace")

    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > limit + MULTIPART_OVERHEAD:
                raise _TooLarge
            parser.write(chunk)
        parser.finalize()
    except _TooLarge:
        _abort(out, settings, upload_id)
        raise ApiError(
            413, "upload_too_large", f"upload exceeds {settings.max_upload_mb} MB"
        ) from None
    except ApiError:
        _abort(out, settings, upload_id)
        raise
    except Exception as exc:  # malformed multipart body
        _abort(out, settings, upload_id)
        raise ApiError(422, "malformed_multipart", "could not parse the multipart body") from exc
    if out is not None:
        out.close()
    if not state["seen_file"]:
        cleanup_upload(settings, upload_id)
        raise ApiError(422, "missing_file", "form field 'file' is required")
    return ReceivedUpload(upload_id, state["filename"], state["size"], fields)


def _abort(out: IO[bytes] | None, settings: Settings, upload_id: str) -> None:
    if out is not None:
        out.close()
    cleanup_upload(settings, upload_id)
