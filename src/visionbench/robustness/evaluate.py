"""Evaluate a run's val-selected checkpoint on every materialized condition of a suite.

- The model loads through the verified evaluation path (``open_for_evaluation``) and uses its
  recorded val transform, applied to the decoded corrupted PNG.
- The clean reference is an existing clean evaluation of the same checkpoint on the same split,
  pinned by id (B1's ``val_eval_id``). Its ``predictions.csv`` is checked against its recorded
  sha256 and must list the same images, in the same order, as the condition.
- Each condition's manifest and every corrupted file are verified by hash before use.
- Results go to ``robustness_evaluations`` and ``<run_dir>/robustness/<id>/`` only (never to
  ``evaluations``).
- Resumable: a condition already evaluated with the same suite hash and condition hash is
  skipped.
- ``split="test"`` is refused unless ``test_images_unlocked`` (M8).
"""

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
from torch.utils.data import DataLoader

from visionbench import __version__
from visionbench.core.config import Settings
from visionbench.core.jsonio import read_json, write_json
from visionbench.evaluation.data import (
    EvalSource,
    load_for_evaluation,
    open_for_evaluation,
)
from visionbench.evaluation.evaluate import predictions_csv
from visionbench.evaluation.metrics import (
    DEFAULT_BOOTSTRAP,
    bootstrap_cis,
    compute_metrics,
    paired_bootstrap_difference,
)
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import file_sha256
from visionbench.robustness.materialize import (
    SplitSource,
    check_split_allowed,
    condition_dir,
    load_verified_condition,
    open_split,
    suite_root,
)
from visionbench.robustness.suite import Condition, RobustnessError, Suite
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.data import ManifestImageDataset
from visionbench.training.runner import resolve_device

LABEL = "VALIDATION (selection-biased) — not test results"


@dataclass(frozen=True)
class ConditionOutcome:
    key: str
    status: str  # "evaluated" | "skipped"
    robustness_id: str


def _clean_reference(
    store: ExperimentStore, source: EvalSource, split: str, clean_eval_id: str | None
) -> dict[str, Any]:
    exp_id = source.experiment["id"]
    if clean_eval_id is None:
        rows = store.evaluations(exp_id, split=split, checkpoint=source.checkpoint)
        if not rows:
            raise RobustnessError(f"{exp_id} has no clean {split} evaluation to pair with")
        row = rows[-1]
    else:
        row = store.get_evaluation(clean_eval_id)
    if (row["experiment_id"], row["split"], row["checkpoint"]) != (
        exp_id,
        split,
        source.checkpoint,
    ):
        raise RobustnessError(
            f"clean evaluation {row['id']} is {row['experiment_id']}/{row['split']}/"
            f"{row['checkpoint']}, expected {exp_id}/{split}/{source.checkpoint}"
        )
    return row


def _read_clean_predictions(
    settings: Settings, row: dict[str, Any], class_names: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    metrics_path = settings.resolved_data_root / row["metrics_path"]
    pred_path = metrics_path.parent / "predictions.csv"
    data = pred_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["predictions_sha256"]:
        raise RobustnessError(f"clean predictions of {row['id']} do not match their sha256")
    index = {c: i for i, c in enumerate(class_names)}
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    relpaths = [r["relpath"] for r in rows]
    y_true = np.array([index[r["true_label"]] for r in rows], dtype=np.int64)
    y_pred = np.array([index[r["pred_label"]] for r in rows], dtype=np.int64)
    return relpaths, y_true, y_pred, read_json(metrics_path)


def _predict(
    loaded: Any, base: Path, files: list[str], labels: np.ndarray, device: torch.device, bs: int
) -> np.ndarray:
    if isinstance(loaded.model, MajorityBaseline):  # does not look at the image
        return np.array(loaded.model.predict_proba(len(files)), dtype=np.float64)
    loader = DataLoader(
        ManifestImageDataset(
            base, list(zip(files, labels.tolist(), strict=True)), loaded.val_transform
        ),
        batch_size=bs,
        shuffle=False,
        num_workers=0,
    )
    model = loaded.model.to(device).eval()
    probs = []
    with torch.inference_mode():
        for inputs, _ in loader:
            probs.append(torch.softmax(model(inputs.to(device)).double(), dim=1).cpu().numpy())
    return np.concatenate(probs)


def evaluate_robustness(
    exp_id: str,
    suite: Suite,
    split: str,
    settings: Settings,
    *,
    clean_eval_id: str | None = None,
    device: str = "auto",
    batch_size: int = 64,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    cache_root: Path | None = None,
    on_condition: Callable[[int, int, str, str], None] | None = None,
) -> list[ConditionOutcome]:
    """Evaluate every condition of ``suite``; ``on_condition(done, total, key, status)``."""
    check_split_allowed(split, settings)
    data_root = settings.resolved_data_root
    dev = resolve_device(device)
    outcomes: list[ConditionOutcome] = []
    with ExperimentStore(settings.data_dir("db") / DB_FILENAME) as store:
        source = open_for_evaluation(store, settings, exp_id, "best")
        exp = source.experiment
        clean_row = _clean_reference(store, source, split, clean_eval_id)
        relpaths, y_true, clean_pred, clean_doc = _read_clean_predictions(
            settings, clean_row, source.class_names
        )
        src: SplitSource = open_split(settings, exp["dataset_id"], exp["split_id"], split)
        if src.manifest_sha256 != source.pre.manifest_sha256:
            raise RobustnessError("split manifest differs from the one used for training")
        if relpaths != [e["relpath"] for e in src.entries]:
            raise RobustnessError("clean predictions do not list the split's images in order")
        index = {c: i for i, c in enumerate(source.class_names)}
        if not np.array_equal(y_true, [index[e["label"]] for e in src.entries]):
            raise RobustnessError("clean predictions' labels differ from the split manifest")
        k = len(source.class_names)
        clean_check = compute_metrics(y_true, clean_pred, source.class_names)
        if clean_check["balanced_accuracy"] != clean_doc["metrics"]["balanced_accuracy"]:
            raise RobustnessError("clean predictions do not reproduce the clean metrics")
        root = cache_root or suite_root(settings, src, suite)
        loaded = load_for_evaluation(source, data_root)
        done_hashes = {
            (r["corruption"], r["severity"], r["condition_hash"]): r["id"]
            for r in store.robustness_evaluations(exp_id, split=split, suite_sha256=suite.sha256)
        }
        total = len(suite.conditions)
        for n, cond in enumerate(suite.conditions, start=1):
            doc = load_verified_condition(root, suite, cond, src)
            key = (cond.corruption, cond.severity_str, doc["condition_hash"])
            if key in done_hashes:
                outcomes.append(ConditionOutcome(cond.key, "skipped", done_hashes[key]))
                if on_condition:
                    on_condition(n, total, cond.key, "skipped")
                continue
            rob_id = _evaluate_condition(
                store,
                settings,
                source,
                loaded,
                cond,
                doc,
                root,
                src,
                y_true,
                clean_pred,
                clean_row,
                clean_doc,
                split,
                dev,
                batch_size,
                bootstrap_resamples,
                seed,
                k,
            )
            outcomes.append(ConditionOutcome(cond.key, "evaluated", rob_id))
            if on_condition:
                on_condition(n, total, cond.key, "evaluated")
    return outcomes


def _evaluate_condition(
    store: ExperimentStore,
    settings: Settings,
    source: EvalSource,
    loaded: Any,
    cond: Condition,
    doc: dict[str, Any],
    root: Path,
    src: SplitSource,
    y_true: np.ndarray,
    clean_pred: np.ndarray,
    clean_row: dict[str, Any],
    clean_doc: dict[str, Any],
    split: str,
    dev: torch.device,
    batch_size: int,
    resamples: int,
    seed: int,
    k: int,
) -> str:
    data_root = settings.resolved_data_root
    by_rel = {e["relpath"]: e["corrupted_file"] for e in doc["entries"]}
    relpaths = [e["relpath"] for e in src.entries]
    files = [by_rel[r] for r in relpaths]
    proba = _predict(loaded, condition_dir(root, cond), files, y_true, dev, batch_size)
    y_pred = proba.argmax(axis=1)
    point = compute_metrics(y_true, y_pred, source.class_names, proba)
    cis = bootstrap_cis(y_true, y_pred, k, resamples=resamples, seed=seed)
    paired = paired_bootstrap_difference(
        y_true, y_pred, clean_pred, k, resamples=resamples, seed=seed
    )
    clean_metrics = clean_doc["metrics"]
    rob_id = f"rb-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out_dir = data_root / source.experiment["run_dir"] / "robustness" / rob_id
    out_dir.mkdir(parents=True)
    pred_bytes = predictions_csv(relpaths, y_true, y_pred, proba, source.class_names)
    (out_dir / "predictions.csv").write_bytes(pred_bytes)
    pred_sha = hashlib.sha256(pred_bytes).hexdigest()
    result = {
        "label": LABEL if split == "val" else "TEST",
        "robustness_id": rob_id,
        "experiment_id": source.experiment["id"],
        "model": source.experiment["model"],
        "split": split,
        "checkpoint": {
            "name": source.checkpoint,
            "sha256": source.checkpoint_sha256,
            "epoch": loaded.epoch,
        },
        "suite": doc["suite"],
        "suite_sha256": doc["suite_sha256"],
        "corruption": cond.corruption,
        "parameter": cond.parameter,
        "severity": cond.severity_str,
        "condition_hash": doc["condition_hash"],
        "materialization_versions": doc["versions"],
        "manifest_sha256": src.manifest_sha256,
        "clean_evaluation_id": clean_row["id"],
        "images": len(relpaths),
        "device": str(dev),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "visionbench_version": __version__,
        "predictions_sha256": pred_sha,
        "metrics": point,
        "bootstrap": cis,
        "clean": {
            "accuracy": clean_metrics["accuracy"],
            "balanced_accuracy": clean_metrics["balanced_accuracy"],
            "macro_f1": clean_metrics["macro_f1"],
            "per_class_recall": {c: v["recall"] for c, v in clean_metrics["per_class"].items()},
        },
        "paired_difference_corrupted_minus_clean": paired,
    }
    write_json(out_dir / "metrics.json", result)
    for name, kind in (
        ("metrics.json", "robustness_metrics"),
        ("predictions.csv", "robustness_predictions"),
    ):
        path = out_dir / name
        store.add_artifact(
            source.experiment["id"],
            kind=kind,
            relpath=path.relative_to(data_root).as_posix(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        )
    store.add_robustness(
        id=rob_id,
        experiment_id=source.experiment["id"],
        split=split,
        checkpoint=source.checkpoint,
        suite=doc["suite"],
        suite_sha256=doc["suite_sha256"],
        corruption=cond.corruption,
        severity=cond.severity_str,
        condition_hash=doc["condition_hash"],
        clean_evaluation_id=clean_row["id"],
        metrics_path=(out_dir / "metrics.json").relative_to(data_root).as_posix(),
        predictions_sha256=pred_sha,
        visionbench_version=__version__,
    )
    return rob_id
