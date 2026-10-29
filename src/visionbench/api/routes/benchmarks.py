"""Read-only views for the frontend (M7): overview, frozen benchmarks, robustness suites.

Nothing here runs a model or writes state; benchmark documents are read by fixed names only.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Request
from fastapi import Path as PathParam

from visionbench import __version__
from visionbench.api.deps import experiment_store, job_store, settings_of
from visionbench.api.schemas import (
    BenchmarkDetail,
    BenchmarkDocument,
    BenchmarkListItem,
    Overview,
    SuiteOut,
)
from visionbench.benchmarks import (
    BENCHMARK_NAME_RE,
    DOCUMENTS,
    _final_runs,
    _read,
    benchmark_dir,
    benchmark_info,
    final_test_splits,
    list_benchmarks,
    read_document,
    verify_hashes,
)
from visionbench.data.store import list_datasets
from visionbench.robustness.suite import RobustnessError, load_suite

router = APIRouter(prefix="/api", tags=["overview"])
BenchmarkName = PathParam(pattern=BENCHMARK_NAME_RE.pattern)


def _list_item(settings: Any, name: str) -> dict[str, Any]:
    d = benchmark_dir(settings.benchmarks_dir, name)
    return {
        "name": name,
        "protocol_hashes": verify_hashes(d, "PROTOCOL_HASHES.txt"),
        "robustness_hashes": verify_hashes(d / "robustness", "ROBUSTNESS_HASHES.txt"),
        "runs": len(_final_runs(_read(d / "runs.json"))),
    }


@router.get("/overview", response_model=Overview)
def overview(request: Request) -> dict[str, Any]:
    settings = settings_of(request)
    with experiment_store(settings) as es:
        exps = es.list()
    with job_store(settings) as js:
        jobs = js.list(limit=100_000)
    return {
        "version": __version__,
        "datasets": len(list_datasets(settings.data_dir("datasets"))),
        "experiments": len(exps),
        "experiments_by_status": dict(Counter(e["status"] for e in exps)),
        "jobs_by_status": dict(Counter(j["status"] for j in jobs)),
        "test_images_unlocked": settings.test_images_unlocked,
        "final_test_splits": final_test_splits(settings),
        "inference_device": settings.inference_device,
        "benchmarks": [_list_item(settings, n) for n in list_benchmarks(settings.benchmarks_dir)],
    }


@router.get("/benchmarks", response_model=list[BenchmarkListItem])
def benchmarks(request: Request) -> list[dict[str, Any]]:
    settings = settings_of(request)
    return [_list_item(settings, n) for n in list_benchmarks(settings.benchmarks_dir)]


@router.get("/benchmarks/{name}", response_model=BenchmarkDetail)
def benchmark(request: Request, name: str = BenchmarkName) -> dict[str, Any]:
    return benchmark_info(settings_of(request), name)


@router.get("/benchmarks/{name}/documents/{document}", response_model=BenchmarkDocument)
def benchmark_document(
    request: Request,
    name: str = BenchmarkName,
    document: str = PathParam(pattern="^(" + "|".join(sorted(DOCUMENTS)) + ")$"),
) -> dict[str, Any]:
    return read_document(settings_of(request), name, document)


@router.get("/robustness-suites", response_model=list[SuiteOut])
def robustness_suites(request: Request) -> list[dict[str, Any]]:
    root = settings_of(request).robustness_suites_dir
    out = []
    for path in sorted(root.rglob("robustness_suite*.yaml")) if root.is_dir() else []:
        try:
            s = load_suite(path)
        except (RobustnessError, OSError, ValueError):
            continue
        out.append(
            {
                "name": s.name,
                "sha256": s.sha256,
                "base_seed": s.base_seed,
                "conditions": [
                    {
                        "corruption": c.corruption,
                        "parameter": c.parameter,
                        "severity": c.severity_str,
                    }
                    for c in s.conditions
                ],
            }
        )
    return out
