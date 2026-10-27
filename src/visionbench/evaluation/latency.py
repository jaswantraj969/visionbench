"""`visionbench latency`: forward-pass latency of a trained model, independent of any split.

Uses random input of the model's input size (no dataset image is read), the val-selected
checkpoint (verified against its artifact sha256), and :func:`efficiency.measure_latency`.
Each measurement is stored as ``experiments/<exp>/latency/<lat_id>/latency.json`` (artifact)
plus a ``latency_measurements`` row, so comparisons can report latency for one device only.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from visionbench import __version__
from visionbench.core.config import Settings
from visionbench.core.jsonio import write_json
from visionbench.evaluation import efficiency
from visionbench.evaluation.data import EvaluationError, load_weights, verify_checkpoint
from visionbench.models.pretrained import file_sha256
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.runner import resolve_device

LATENCY_DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class LatencyResult:
    latency_id: str
    experiment_id: str
    path: Path
    measurement: dict[str, Any]


def measure_experiment_latency(
    exp_id: str,
    settings: Settings,
    *,
    device: str,
    warmup: int = efficiency.DEFAULT_WARMUP,
    iterations: int = efficiency.DEFAULT_ITERATIONS,
    batch_sizes: tuple[int, ...] = efficiency.DEFAULT_BATCH_SIZES,
) -> LatencyResult:
    if device not in LATENCY_DEVICES:
        raise EvaluationError(f"device must be one of {LATENCY_DEVICES}, got {device!r}")
    dev = resolve_device(device)  # "cuda" without CUDA is an error, never a silent CPU run
    data_root = settings.resolved_data_root
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        exp = store.get(exp_id)
        if exp["model"] == "majority_baseline":
            raise EvaluationError(
                "latency is not applicable to majority_baseline (no forward pass)"
            )
        ck = verify_checkpoint(store, settings, exp_id, "best")
        model, epoch = load_weights(
            exp["model"],
            ck.run_config["resolved"]["class_names"],
            ck.run_config,
            ck.path,
            data_root,
        )
        input_size = int(ck.run_config["resolved"]["input_size"])
        lat = efficiency.measure_latency(
            model, input_size, dev, batch_sizes=batch_sizes, warmup=warmup, iterations=iterations
        )
        latency_id = f"lat-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        out_dir = ck.run_dir / "latency" / latency_id
        out_dir.mkdir(parents=True)
        doc = {
            "latency_id": latency_id,
            "experiment_id": exp_id,
            "model": exp["model"],
            "checkpoint": {"name": ck.checkpoint, "sha256": ck.sha256, "epoch": epoch},
            "visionbench_version": __version__,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **lat,
        }
        path = out_dir / "latency.json"
        write_json(path, doc)
        relpath = path.relative_to(data_root).as_posix()
        store.add_artifact(
            exp_id,
            kind="latency",
            relpath=relpath,
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        )
        store.add_latency(
            id=latency_id,
            experiment_id=exp_id,
            checkpoint=ck.checkpoint,
            device=dev.type,
            device_name=lat["device_name"],
            torch_threads=lat["torch_threads"],
            input_size=input_size,
            warmup=warmup,
            iterations=iterations,
            results_json=json.dumps(lat["batch_sizes"], sort_keys=True),
            path=relpath,
            visionbench_version=__version__,
        )
    return LatencyResult(latency_id, exp_id, path, doc)
