"""Dataset endpoints: uploads, path imports, listing, images, splits, verify."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from visionbench.api.deps import DATASET_ID, SPLIT_ID, enqueue, settings_of
from visionbench.api.errors import ApiError
from visionbench.api.schemas import (
    DatasetSummary,
    JobAccepted,
    PathImportRequest,
    SplitRequest,
    SplitSample,
    VerifyRequest,
)
from visionbench.api.upload import receive_zip_upload
from visionbench.core.jsonio import read_json
from visionbench.data.archive import (
    ArchiveLimits,
    cleanup_upload,
    import_relpath,
    upload_path,
    validate_archive,
)
from visionbench.data.store import (
    dataset_dir,
    list_datasets,
    load_dataset,
    recorded_image,
    validate_name,
)
from visionbench.jobs.params import DatasetImportParams, DatasetSplitParams, DatasetVerifyParams

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "BMP": "image/bmp", "WEBP": "image/webp"}
DatasetId = PathParam(pattern=DATASET_ID)
SplitId = PathParam(pattern=SPLIT_ID)


def _summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in DatasetSummary.model_fields}


_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file", "name"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "ZIP archive of class folders",
                        },
                        "name": {"type": "string", "maxLength": 100},
                    },
                }
            }
        },
    }
}


@router.post("/uploads", status_code=202, response_model=JobAccepted, openapi_extra=_UPLOAD_BODY)
async def upload_dataset(request: Request) -> dict[str, Any]:
    """Multipart form: ``file`` (ZIP) and ``name``. Streamed to disk with a byte limit; the
    archive is validated (central directory only) before the import job is queued."""
    settings = settings_of(request)
    up = await receive_zip_upload(request, settings)
    try:
        name = validate_name(up.fields.get("name", ""))
        validate_archive(upload_path(settings, up.upload_id), ArchiveLimits.from_settings(settings))
        params = DatasetImportParams(
            source="upload", upload_id=up.upload_id, name=name, original_filename=up.filename
        )
        return {"job": enqueue(settings, "dataset_import", params)}
    except BaseException:
        cleanup_upload(settings, up.upload_id)
        raise


@router.post("/imports", status_code=202, response_model=JobAccepted)
def import_from_path(body: PathImportRequest, request: Request) -> dict[str, Any]:
    """Server-side import; only when ``import_root`` is configured (otherwise 403)."""
    settings = settings_of(request)
    if settings.import_root is None:
        raise ApiError(403, "path_import_disabled", "server-side path imports are disabled")
    relpath = import_relpath(settings, body.path)
    params = DatasetImportParams(source="path", relpath=relpath, name=validate_name(body.name))
    return {"job": enqueue(settings, "dataset_import", params)}


@router.get("", response_model=list[DatasetSummary])
def list_all(request: Request) -> list[dict[str, Any]]:
    return [_summary(d) for d in list_datasets(settings_of(request).data_dir("datasets"))]


@router.get("/{dataset_id}")
def get_dataset(request: Request, dataset_id: str = DatasetId) -> dict[str, Any]:
    datasets = settings_of(request).data_dir("datasets")
    meta = load_dataset(datasets, dataset_id)
    report = read_json(dataset_dir(datasets, dataset_id) / "report.json")
    dup = report["duplicates"]
    return {
        **_summary(meta),
        "visionbench_version": meta["visionbench_version"],
        "import_config": meta["import_config"],
        "validation": meta["validation"],
        "report": {
            "imbalance_ratio": report["imbalance_ratio"],
            "width": report["width"],
            "height": report["height"],
            "mode_counts": report["mode_counts"],
            "format_counts": report["format_counts"],
            "total_bytes": report["total_bytes"],
            "issues": report["issues"],
            "duplicates": {k: v for k, v in dup.items() if k != "groups"},
            "samples": report["samples"],
        },
    }


@router.get("/{dataset_id}/images", response_class=FileResponse)
def get_image(
    request: Request,
    dataset_id: str = DatasetId,
    relpath: str = Query(min_length=1, max_length=1024),
) -> FileResponse:
    """Serve an image ONLY if ``relpath`` is a recorded member of the dataset: the user value
    is used as a dictionary key, never joined onto a filesystem path."""
    found = recorded_image(settings_of(request).data_dir("datasets"), dataset_id, relpath)
    if found is None:
        raise ApiError(404, "image_not_found", "no such image in this dataset")
    record, path = found
    return FileResponse(path, media_type=MEDIA_TYPES.get(record.format, "application/octet-stream"))


@router.post("/{dataset_id}/splits", status_code=202, response_model=JobAccepted)
def create_split(
    body: SplitRequest, request: Request, dataset_id: str = DatasetId
) -> dict[str, Any]:
    settings = settings_of(request)
    load_dataset(settings.data_dir("datasets"), dataset_id)  # 404 if unknown
    params = DatasetSplitParams(dataset_id=dataset_id, **body.model_dump())
    return {"job": enqueue(settings, "dataset_split", params)}


def _split_summary(split_dir: Path) -> dict[str, Any]:
    m = read_json(split_dir / "manifest.json")
    info = read_json(split_dir / "split_info.json")
    return {
        "split_id": split_dir.name,
        "created_at": info["created_at"],
        "manifest_sha256": info["manifest_sha256"],
        "algorithm": m["algorithm"],
        "config": m["config"],
        "split_totals": m["split_totals"],
        "leakage_check": m["leakage_check"],
        "_manifest": m,
    }


@router.get("/{dataset_id}/splits")
def list_splits(request: Request, dataset_id: str = DatasetId) -> list[dict[str, Any]]:
    root = dataset_dir(settings_of(request).data_dir("datasets"), dataset_id) / "splits"
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "manifest.json").is_file() and not d.name.startswith("."):
            s = _split_summary(d)
            s.pop("_manifest")
            out.append(s)
    return out


@router.get("/{dataset_id}/splits/{split_id}")
def get_split(
    request: Request, dataset_id: str = DatasetId, split_id: str = SplitId
) -> dict[str, Any]:
    d = dataset_dir(settings_of(request).data_dir("datasets"), dataset_id) / "splits" / split_id
    if not (d / "manifest.json").is_file():
        raise ApiError(404, "split_not_found", f"split {split_id} not found")
    s = _split_summary(d)
    m = s.pop("_manifest")
    return s | {
        "counts": m["counts"],
        "target_counts": m["target_counts"],
        "warnings": m["warnings"],
        "excluded": m["excluded"],
        "excluded_groups": m["excluded_groups"],
        "low_information_files": m["low_information_files"],
        "entries_total": len(m["entries"]),
    }


SAMPLE_RULE = (
    "numpy default_rng(seed); per_class entries of the split drawn without replacement from each "
    "class's entries sorted by relpath"
)


@router.get("/{dataset_id}/splits/{split_id}/sample", response_model=SplitSample)
def split_sample(
    request: Request,
    dataset_id: str = DatasetId,
    split_id: str = SplitId,
    split: str = Query(pattern="^(train|val|test)$"),
    per_class: int = Query(default=2, ge=1, le=50),
    seed: int = Query(default=0, ge=0, lt=2**32),
) -> dict[str, Any]:
    """A seeded, non-hand-picked sample of a split's images (for the image browser). TEST
    images are not listed while the test split is locked (403 test_split_locked)."""
    import numpy as np

    settings = settings_of(request)
    from visionbench.benchmarks import split_test_images_allowed

    if split == "test" and not split_test_images_allowed(settings, split_id):
        raise ApiError(
            403,
            "test_split_locked",
            "TEST images are locked until the final test evaluation "
            "(VISIONBENCH_TEST_IMAGES_UNLOCKED)",
        )
    d = dataset_dir(settings.data_dir("datasets"), dataset_id) / "splits" / split_id
    if not (d / "manifest.json").is_file():
        raise ApiError(404, "split_not_found", f"split {split_id} not found")
    manifest = read_json(d / "manifest.json")
    rng = np.random.default_rng(seed)
    images = []
    for label in sorted(manifest["counts"]):
        entries = sorted(
            (e for e in manifest["entries"] if e["split"] == split and e["label"] == label),
            key=lambda e: e["relpath"],
        )
        k = min(per_class, len(entries))
        if k:
            for i in sorted(rng.choice(len(entries), size=k, replace=False).tolist()):
                images.append({"relpath": entries[i]["relpath"], "label": label, "split": split})
    return {
        "dataset_id": dataset_id,
        "split_id": split_id,
        "split": split,
        "seed": seed,
        "per_class": per_class,
        "rule": SAMPLE_RULE,
        "images": images,
    }


@router.get("/{dataset_id}/splits/{split_id}/audit")
def split_audit(
    request: Request, dataset_id: str = DatasetId, split_id: str = SplitId
) -> dict[str, Any]:
    """The D1 split audit (``audit.json``) if one was run; 404 otherwise."""
    root = dataset_dir(settings_of(request).data_dir("datasets"), dataset_id)
    path = root / "audits" / split_id / "audit.json"
    if not path.is_file():
        raise ApiError(404, "audit_not_found", f"no audit for split {split_id}")
    return read_json(path)


@router.post("/{dataset_id}/verify", status_code=202, response_model=JobAccepted)
def verify(body: VerifyRequest, request: Request, dataset_id: str = DatasetId) -> dict[str, Any]:
    settings = settings_of(request)
    load_dataset(settings.data_dir("datasets"), dataset_id)
    return {
        "job": enqueue(
            settings, "dataset_verify", DatasetVerifyParams(dataset_id=dataset_id, full=body.full)
        )
    }
