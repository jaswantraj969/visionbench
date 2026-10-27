"""`visionbench evaluate`: predictions, metrics, CIs, efficiency, test-use accounting."""

from __future__ import annotations

import csv
import hashlib
import io
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from visionbench import __version__
from visionbench.core.config import Settings
from visionbench.core.jsonio import write_json
from visionbench.evaluation import efficiency
from visionbench.evaluation.data import (
    EvaluationError,
    LoadedModel,
    build_eval_dataloader,
    check_test_reuse,
    checkpoint_name,
    load_for_evaluation,
    open_for_evaluation,
    select_eval_entries,
)
from visionbench.evaluation.metrics import DEFAULT_BOOTSTRAP, bootstrap_cis, compute_metrics
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import file_sha256
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.runner import resolve_device


@dataclass(frozen=True)
class EvalResult:
    eval_id: str
    experiment_id: str
    split: str
    checkpoint: str
    eval_dir: Path
    metrics: dict[str, Any]
    prior_test_evaluations: int  # same experiment + checkpoint, before this one
    warning: str | None


def _predict(
    loaded: LoadedModel,
    split: str,
    source_manifest: dict[str, Any],
    raw_root: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """(relpaths, y_true, y_pred, proba) in manifest relpath order."""
    entries = select_eval_entries(source_manifest, split)
    if not entries:
        raise EvaluationError(f"split {split!r} has no entries")
    index = {c: i for i, c in enumerate(class_names)}
    relpaths = [e["relpath"] for e in entries]
    y_true = np.array([index[e["label"]] for e in entries], dtype=np.int64)
    if isinstance(loaded.model, MajorityBaseline):  # labels only; no image is opened
        proba = np.array(loaded.model.predict_proba(len(entries)), dtype=np.float64)
        y_pred = np.array(loaded.model.predict(len(entries)), dtype=np.int64)
        return relpaths, y_true, y_pred, proba
    model = loaded.model.to(device).eval()
    loader = build_eval_dataloader(
        split, source_manifest, raw_root, loaded.val_transform, batch_size
    )
    probs = []
    with torch.inference_mode():
        for inputs, _ in loader:
            logits = model(inputs.to(device))
            probs.append(torch.softmax(logits.double(), dim=1).cpu().numpy())
    proba = np.concatenate(probs)
    return relpaths, y_true, proba.argmax(axis=1), proba


def predictions_csv(
    relpaths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["relpath", "true_label", "pred_label", *[f"prob_{c}" for c in class_names]])
    for rel, t, p, pr in zip(relpaths, y_true, y_pred, proba, strict=True):
        w.writerow([rel, class_names[t], class_names[p], *[f"{v:.8f}" for v in pr]])
    return buf.getvalue().encode("utf-8")


def evaluate_experiment(
    exp_id: str,
    split: str,
    settings: Settings,
    *,
    checkpoint: str = "best",
    device: str = "auto",
    batch_size: int = 64,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    force: bool = False,
    reason: str | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> EvalResult:
    """Evaluate a completed experiment on val or test.

    A repeated TEST evaluation of the same experiment + checkpoint raises
    :class:`RepeatedTestEvaluationError` unless ``force`` and a non-empty ``reason`` are given;
    forced repeats are recorded (``forced``, ``reason``) and shown in every report.
    Latency is not measured here (see :mod:`visionbench.evaluation.latency`).
    """
    data_root = settings.resolved_data_root
    dev = resolve_device(device)
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        exp_row = store.get(exp_id)
        name = checkpoint_name(exp_row, checkpoint)
        select_eval_entries({"entries": []}, split)  # validates the split name before any work
        prior_same = check_test_reuse(store, exp_id, split, name, force=force, reason=reason)
        prior_all = len(store.evaluations(exp_id, split="test"))
        forced = split == "test" and prior_same > 0
        source = open_for_evaluation(store, settings, exp_id, checkpoint)
        warning = None
        if forced:
            warning = (
                f"FORCED repeated TEST evaluation #{prior_same + 1} of experiment {exp_id} "
                f"checkpoint '{source.checkpoint}'; reason recorded: {reason}"
            )
            if on_warning is not None:
                on_warning(warning)

        loaded = load_for_evaluation(source, data_root)
        relpaths, y_true, y_pred, proba = _predict(
            loaded,
            split,
            source.pre.manifest,
            source.pre.raw_root,
            source.class_names,
            dev,
            batch_size,
        )
        point = compute_metrics(y_true, y_pred, source.class_names, proba)
        cis = bootstrap_cis(
            y_true, y_pred, len(source.class_names), resamples=bootstrap_resamples, seed=seed
        )

        resolved = source.run_config["resolved"]
        if isinstance(loaded.model, MajorityBaseline):
            params = {"total": None, "trainable": None}
        else:
            params = {
                "total": resolved["model"]["param_count"],
                "trainable": resolved["model"]["trainable_param_count"],
            }
            if efficiency.param_counts(loaded.model)["total"] != params["total"]:
                raise EvaluationError("rebuilt model parameter count differs from training")
        eff = {
            "params": params,
            "checkpoint_file": source.checkpoint_path.name,
            "checkpoint_size_bytes": source.checkpoint_path.stat().st_size,
            "train_seconds": source.experiment["train_seconds"],
            "input_size": loaded.input_size,
            "latency": "measured separately: `visionbench latency <exp_id> --device cpu|cuda`",
        }

        eval_id = f"ev-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        eval_dir = source.run_dir / "evaluations" / eval_id
        eval_dir.mkdir(parents=True)
        pred_bytes = predictions_csv(relpaths, y_true, y_pred, proba, source.class_names)
        (eval_dir / "predictions.csv").write_bytes(pred_bytes)
        pred_sha = hashlib.sha256(pred_bytes).hexdigest()
        exp = source.experiment
        doc = {
            "eval_id": eval_id,
            "experiment_id": exp_id,
            "model": exp["model"],
            "weights_source": exp["weights_source"],
            "split": split,
            "checkpoint": {
                "name": source.checkpoint,
                "file": source.checkpoint_path.name,
                "sha256": source.checkpoint_sha256,
                "epoch": loaded.epoch,
            },
            "selection": {
                "selected_on": "val",
                "monitor": source.run_config["config"]["early_stopping"]["monitor"],
                "best_epoch": exp["best_epoch"],
                "best_value": exp["best_metric"],
            },
            "dataset_id": exp["dataset_id"],
            "split_id": exp["split_id"],
            "dataset_fingerprint": exp["dataset_fingerprint"],
            "manifest_sha256": source.pre.manifest_sha256,
            "device": str(dev),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "visionbench_version": __version__,
            "predictions_sha256": pred_sha,
            "test_use": {
                "prior_test_evaluations_same_checkpoint": prior_same,
                "prior_test_evaluations_experiment": prior_all,
                "test_evaluation_number": prior_same + 1 if split == "test" else None,
                "forced": forced,
                "reason": reason if forced else None,
            },
            "metrics": point,
            "bootstrap": cis,
        }
        write_json(eval_dir / "metrics.json", doc)
        write_json(eval_dir / "confusion_matrix.json", point["confusion_matrix"])
        write_json(eval_dir / "efficiency.json", eff)
        kinds = {
            "metrics.json": "eval_metrics",
            "predictions.csv": "eval_predictions",
            "confusion_matrix.json": "eval_confusion",
            "efficiency.json": "eval_efficiency",
        }
        for name, kind in kinds.items():
            path = eval_dir / name
            store.add_artifact(
                exp_id,
                kind=kind,
                relpath=path.relative_to(data_root).as_posix(),
                sha256=file_sha256(path),
                size_bytes=path.stat().st_size,
            )
        store.add_evaluation(
            eval_id=eval_id,
            experiment_id=exp_id,
            split=split,
            checkpoint=source.checkpoint,
            metrics_path=(eval_dir / "metrics.json").relative_to(data_root).as_posix(),
            predictions_sha256=pred_sha,
            visionbench_version=__version__,
            forced=forced,
            reason=reason if forced else None,
        )
    return EvalResult(eval_id, exp_id, split, source.checkpoint, eval_dir, doc, prior_same, warning)
