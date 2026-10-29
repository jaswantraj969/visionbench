"""Experiment, evaluation, latency, comparison and job endpoints (enqueue + read only)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi import Path as PathParam

from visionbench.api.deps import (
    EVAL_ID,
    EXPERIMENT_ID,
    JOB_ID,
    REPORT_ID,
    enqueue,
    experiment_store,
    job_store,
    public_job,
    settings_of,
)
from visionbench.api.errors import ApiError
from visionbench.api.schemas import (
    ComparisonRequest,
    EvaluationRequest,
    JobAccepted,
    JobOut,
    LatencyRequest,
)
from visionbench.core.jsonio import read_json
from visionbench.data.archive import cleanup_upload
from visionbench.data.store import dataset_dir, load_dataset
from visionbench.evaluation.compare import compare_experiments
from visionbench.evaluation.data import check_test_reuse, checkpoint_name
from visionbench.jobs.params import EvaluateParams, LatencyParams, TrainParams
from visionbench.training.config import TrainConfig

experiments = APIRouter(prefix="/api/experiments", tags=["experiments"])
evaluations = APIRouter(prefix="/api/evaluations", tags=["evaluations"])
comparisons = APIRouter(prefix="/api/comparisons", tags=["comparisons"])
jobs = APIRouter(prefix="/api/jobs", tags=["jobs"])

ExperimentId = PathParam(pattern=EXPERIMENT_ID)


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "status",
        "error",
        "model",
        "weights_source",
        "dataset_id",
        "split_id",
        "seed",
        "best_epoch",
        "best_metric",
        "stopped_reason",
        "train_seconds",
        "created_at",
        "started_at",
        "finished_at",
    )
    return {k: row[k] for k in keys}


@experiments.post("", status_code=202, response_model=JobAccepted)
def create_experiment(config: TrainConfig, request: Request) -> dict[str, Any]:
    """Queue a training job (never trained inside the request)."""
    settings = settings_of(request)
    datasets = settings.data_dir("datasets")
    load_dataset(datasets, config.dataset_id)  # 404 if unknown
    if not (
        dataset_dir(datasets, config.dataset_id) / "splits" / config.split_id / "manifest.json"
    ).is_file():
        raise ApiError(404, "split_not_found", f"split {config.split_id} not found")
    return {"job": enqueue(settings, "train", TrainParams(config=config))}


@experiments.get("")
def list_experiments(request: Request) -> list[dict[str, Any]]:
    with experiment_store(settings_of(request)) as es:
        return [_summary(r) for r in es.list()]


@experiments.get("/{experiment_id}")
def get_experiment(request: Request, experiment_id: str = ExperimentId) -> dict[str, Any]:
    with experiment_store(settings_of(request)) as es:
        row = es.get(experiment_id)
        lat = es.latencies(experiment_id)
        return {
            **_summary(row),
            "run_dir": row["run_dir"],
            "config": json.loads(row["config_json"]),
            "env": json.loads(row["env_json"]) if row["env_json"] else None,
            "weights_enum": row["weights_enum"],
            "weights_sha256": row["weights_sha256"],
            "epochs": es.epochs(experiment_id),
            "artifacts": es.artifacts(experiment_id),
            "evaluations": es.evaluations(experiment_id),
            "latency": [
                {
                    **{k: v for k, v in r.items() if k != "results_json"},
                    "batch_sizes": json.loads(r["results_json"]),
                }
                for r in lat
            ],
        }


@experiments.post("/{experiment_id}/evaluations", status_code=202, response_model=JobAccepted)
def create_evaluation(
    body: EvaluationRequest, request: Request, experiment_id: str = ExperimentId
) -> dict[str, Any]:
    """Queue an evaluation. The repeated-TEST rule is checked here (409) and again in the
    worker by the same library function, so CLI and API behave identically."""
    settings = settings_of(request)
    with experiment_store(settings) as es:
        exp = es.get(experiment_id)
        if exp["status"] != "completed":
            raise ApiError(
                409,
                "experiment_not_completed",
                f"experiment {experiment_id} has status {exp['status']!r}",
            )
        name = checkpoint_name(exp, body.checkpoint)
        check_test_reuse(es, experiment_id, body.split, name, force=body.force, reason=body.reason)
    params = EvaluateParams(experiment_id=experiment_id, **body.model_dump())
    return {"job": enqueue(settings, "evaluate", params)}


@experiments.post("/{experiment_id}/latency", status_code=202, response_model=JobAccepted)
def create_latency(
    body: LatencyRequest, request: Request, experiment_id: str = ExperimentId
) -> dict[str, Any]:
    settings = settings_of(request)
    with experiment_store(settings) as es:
        exp = es.get(experiment_id)
    if exp["model"] == "majority_baseline":
        raise ApiError(422, "latency_not_applicable", "majority_baseline has no forward pass")
    if exp["status"] != "completed":
        raise ApiError(
            409,
            "experiment_not_completed",
            f"experiment {experiment_id} has status {exp['status']!r}",
        )
    params = LatencyParams(experiment_id=experiment_id, **body.model_dump())
    return {"job": enqueue(settings, "latency", params)}


@evaluations.get("")
def list_evaluations(
    request: Request,
    experiment_id: str | None = Query(default=None, pattern=EXPERIMENT_ID),
) -> list[dict[str, Any]]:
    with experiment_store(settings_of(request)) as es:
        return es.evaluations(experiment_id)


@evaluations.get("/{eval_id}")
def get_evaluation(request: Request, eval_id: str = PathParam(pattern=EVAL_ID)) -> dict[str, Any]:
    settings = settings_of(request)
    with experiment_store(settings) as es:
        row = es.get_evaluation(eval_id)
    return {
        "evaluation": row,
        "metrics": read_json(settings.resolved_data_root / row["metrics_path"]),
    }


@comparisons.post("", status_code=201)
def create_comparison(body: ComparisonRequest, request: Request) -> dict[str, Any]:
    """Synchronous: reads recorded evaluations only (no model is run)."""
    return compare_experiments(
        body.experiment_ids,
        body.split,
        settings_of(request),
        reference=body.reference,
        resamples=body.bootstrap,
        seed=body.seed,
        note=body.note,
        eval_ids=body.eval_ids,
        latency_device=body.latency_device,
    )


@comparisons.get("/{report_id}")
def get_comparison(
    request: Request, report_id: str = PathParam(pattern=REPORT_ID)
) -> dict[str, Any]:
    path = settings_of(request).data_dir("reports") / report_id / "comparison.json"
    if not path.is_file():
        raise ApiError(404, "report_not_found", f"report {report_id} not found")
    return read_json(path)


@jobs.get("", response_model=list[JobOut])
def list_jobs(
    request: Request,
    status: str | None = Query(
        default=None, pattern="^(queued|running|completed|failed|cancelled)$"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    with job_store(settings_of(request)) as js:
        return [public_job(j) for j in js.list(status=status, limit=limit)]


@jobs.get("/{job_id}", response_model=JobOut)
def get_job(request: Request, job_id: str = PathParam(pattern=JOB_ID)) -> dict[str, Any]:
    with job_store(settings_of(request)) as js:
        return public_job(js.get(job_id))


@jobs.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(request: Request, job_id: str = PathParam(pattern=JOB_ID)) -> dict[str, Any]:
    """Queued -> cancelled now; running -> cancel requested (the worker stops it)."""
    settings = settings_of(request)
    with job_store(settings) as js:
        job = js.request_cancel(job_id)
    if job["kind"] == "dataset_import" and job["status"] == "cancelled" and not job["started_at"]:
        upload_id = job["params"].get("upload_id")
        if upload_id:
            cleanup_upload(settings, upload_id)  # never started: its upload is not needed
    return public_job(job)
