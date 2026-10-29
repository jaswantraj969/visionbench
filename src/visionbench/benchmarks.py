"""Read-only views of frozen benchmarks (``benchmarks/<name>/``) for the API and the frontend.

A benchmark directory holds ``BENCHMARK_PROTOCOL.md``, its YAMLs and ``PROTOCOL_HASHES.txt``,
``runs.json`` (the driver record), optionally ``pilot.json``, ``repro_check.json``,
``robustness/`` (protocol, suite, hashes) and ``results/``. Nothing here writes or runs a model:
it verifies hashes, joins ``runs.json`` with the experiment store, and reads recorded metrics.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
from visionbench.core.jsonio import read_json
from visionbench.tracking.store import DB_FILENAME, ExperimentStore, NotFoundError

BENCHMARK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
VAL_LABEL = "VALIDATION (selection-biased) — not test results"
TEST_LABEL = "TEST"
# Documents that may be read through the API (fixed names; never a client-supplied path).
DOCUMENTS = {
    "benchmark_protocol": "BENCHMARK_PROTOCOL.md",
    "robustness_protocol": "robustness/ROBUSTNESS_PROTOCOL.md",
    "final_analysis_plan": "FINAL_ANALYSIS_PLAN.md",
    "robustness_val": "results/robustness_val.md",
    "final_report": "results/FINAL_REPORT.md",
}


class BenchmarkNotFoundError(VisionBenchError):
    pass


def benchmark_dir(root: Path, name: str) -> Path:
    if not BENCHMARK_NAME_RE.match(name):
        raise BenchmarkNotFoundError(f"invalid benchmark name {name!r}")
    path = Path(root) / name
    if not (path / "BENCHMARK_PROTOCOL.md").is_file():
        raise BenchmarkNotFoundError(f"benchmark {name} not found")
    return path


def list_benchmarks(root: Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and BENCHMARK_NAME_RE.match(d.name)
        and (d / "BENCHMARK_PROTOCOL.md").is_file()
    )


def verify_hashes(directory: Path, hash_file: str) -> dict[str, Any]:
    """Check ``<sha256> *<name>`` lines of ``hash_file`` against the files next to it."""
    path = directory / hash_file
    if not path.is_file():
        return {"present": False, "ok": False, "files": []}
    files = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):  # blank or comment
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        target = directory / name
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        files.append({"name": name, "expected": digest, "actual": actual, "ok": actual == digest})
    return {"present": True, "ok": bool(files) and all(f["ok"] for f in files), "files": files}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _final_runs(runs_doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    final: dict[tuple[str, int], dict[str, Any]] = {}
    for r in (runs_doc or {}).get("runs", []):
        final[(r["model"], r["seed"])] = r
    return [final[k] for k in sorted(final)]


def experiment_tags(root: Path) -> dict[str, list[str]]:
    """experiment id -> tags such as ``benchmark:eurosat-v1``, ``pilot:eurosat-v1``,
    ``repro-check:eurosat-v1`` (from the benchmark records; never from the experiment itself)."""
    tags: dict[str, list[str]] = {}
    for name in list_benchmarks(root):
        d = Path(root) / name
        for r in _final_runs(_read(d / "runs.json")):
            if r.get("experiment_id"):
                tags.setdefault(r["experiment_id"], []).append(f"benchmark:{name}")
        for exp in (_read(d / "pilot.json") or {}).get("pilot_experiment_ids", []):
            tags.setdefault(exp, []).append(f"pilot:{name}")
        repro = _read(d / "repro_check.json")
        if repro and repro.get("repro_check_experiment_id"):
            tags.setdefault(repro["repro_check_experiment_id"], []).append(f"repro-check:{name}")
    return tags


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "values": values,
    }


def benchmark_info(settings: Settings, name: str) -> dict[str, Any]:
    d = benchmark_dir(settings.benchmarks_dir, name)
    data_root = settings.resolved_data_root
    runs = []
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        for r in _final_runs(_read(d / "runs.json")):
            row: dict[str, Any] = {
                "model": r["model"],
                "seed": r["seed"],
                "experiment_id": r.get("experiment_id"),
                "driver_status": r.get("status"),
                "attempt": r.get("attempt"),
                "val_eval_id": r.get("val_eval_id"),
            }
            try:
                exp = store.get(r["experiment_id"]) if r.get("experiment_id") else None
            except NotFoundError:
                exp = None
            if exp is not None:
                row |= {
                    "status": exp["status"],
                    "best_epoch": exp["best_epoch"],
                    "epochs_trained": len(store.epochs(exp["id"])),
                    "stopped_reason": exp["stopped_reason"],
                    "train_seconds": exp["train_seconds"],
                    "test_evaluations": len(store.evaluations(exp["id"], split="test")),
                    "robustness_val": len(store.robustness_evaluations(exp["id"], split="val")),
                    "robustness_test": len(store.robustness_evaluations(exp["id"], split="test")),
                }
            if r.get("val_eval_id"):
                try:
                    ev = store.get_evaluation(r["val_eval_id"])
                    m = read_json(data_root / ev["metrics_path"])["metrics"]
                    row["val"] = {
                        "balanced_accuracy": m["balanced_accuracy"],
                        "macro_f1": m["macro_f1"],
                        "accuracy": m["accuracy"],
                    }
                except (NotFoundError, OSError):
                    row["val"] = None
            runs.append(row)
    models = list(dict.fromkeys(r["model"] for r in runs))
    val_summary = []
    for m in models:
        rs = [r for r in runs if r["model"] == m and r.get("val")]
        val_summary.append(
            {
                "model": m,
                "balanced_accuracy": _mean_sd([r["val"]["balanced_accuracy"] for r in rs]),
                "macro_f1": _mean_sd([r["val"]["macro_f1"] for r in rs]),
                "best_epochs": [r.get("best_epoch") for r in rs],
                "seeds": [r["seed"] for r in rs],
            }
        )
    rob = _read(d / "results" / "robustness_val.json")
    final = _read(d / "results" / "FINAL_REPORT.json")
    return {
        "name": name,
        "protocol_hashes": verify_hashes(d, "PROTOCOL_HASHES.txt"),
        "robustness_hashes": verify_hashes(d / "robustness", "ROBUSTNESS_HASHES.txt"),
        "final_analysis_plan_hashes": verify_hashes(d, "FINAL_ANALYSIS_PLAN_HASHES.txt"),
        "documents": {k: (d / v).is_file() for k, v in DOCUMENTS.items()},
        "runs": runs,
        "val_label": VAL_LABEL,
        "val_summary": val_summary,
        "robustness_val": None
        if rob is None
        else {
            "label": rob["label"],
            "scope": rob["scope"],
            "confounds": rob["confounds"],
            "suite": rob["suite"],
            "suite_sha256": rob["suite_sha256"],
            "summary": rob["summary"],
            "runs": [
                {
                    "model": r["model"],
                    "seed": r["seed"],
                    "experiment_id": r["experiment_id"],
                    "clean_balanced_accuracy": r["clean_balanced_accuracy"],
                    "conditions": {
                        k: None
                        if c is None
                        else {"balanced_accuracy": c["balanced_accuracy"], "paired": c["paired"]}
                        for k, c in r["conditions"].items()
                    },
                }
                for r in rob["runs"]
            ],
        },
        "final_report": final,
    }


def read_document(settings: Settings, name: str, doc: str) -> dict[str, Any]:
    d = benchmark_dir(settings.benchmarks_dir, name)
    if doc not in DOCUMENTS:
        raise BenchmarkNotFoundError(f"unknown document {doc!r}; known: {sorted(DOCUMENTS)}")
    path = d / DOCUMENTS[doc]
    if not path.is_file():
        raise BenchmarkNotFoundError(f"{doc} does not exist for benchmark {name}")
    return {
        "benchmark": name,
        "document": doc,
        "path": DOCUMENTS[doc],
        "markdown": path.read_text(encoding="utf-8"),
    }
