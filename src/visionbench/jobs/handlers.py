"""Job handlers: thin adapters from validated job params to library calls.

Each handler returns a JSON-serialisable result dict of created ids (no absolute paths).
Training reports progress and checks for cancellation through the training step hook.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from visionbench.core.config import Settings
from visionbench.data.split import SplitConfig, make_split
from visionbench.data.store import verify_dataset
from visionbench.evaluation.evaluate import evaluate_experiment
from visionbench.evaluation.latency import measure_experiment_latency
from visionbench.jobs import params as P
from visionbench.jobs.store import JobStore
from visionbench.training.loop import RunCancelled
from visionbench.training.runner import run_training

CANCEL_POLL_SECONDS = 0.5  # how often a training step may hit the DB for cancel/progress


@dataclass
class JobContext:
    job_id: str
    store: JobStore
    settings: Settings
    progress_state: dict[str, Any] = field(default_factory=dict)
    _last_poll: float = 0.0

    def progress(self, **values: Any) -> None:
        self.progress_state.update(values)
        self.store.update_progress(self.job_id, self.progress_state)

    def check_cancel(self) -> None:
        if self.store.is_cancel_requested(self.job_id):
            raise RunCancelled(f"job {self.job_id} cancel requested")

    def step_hook(self, step: int) -> None:
        """Called after every optimizer step (and at epoch ends); throttled DB access."""
        now = time.monotonic()
        if now - self._last_poll < CANCEL_POLL_SECONDS:
            return
        self._last_poll = now
        self.progress(step=step)
        self.check_cancel()


def handle_dataset_split(p: P.DatasetSplitParams, ctx: JobContext) -> dict[str, Any]:
    s = ctx.settings
    config = SplitConfig(
        seed=p.seed,
        fractions=p.fractions,
        dhash_threshold=s.dhash_threshold if p.dhash_threshold is None else p.dhash_threshold,
        low_info_std=s.low_info_std_threshold if p.low_info_std is None else p.low_info_std,
        low_info_popcount_min=(
            s.low_info_popcount_min if p.low_info_popcount_min is None else p.low_info_popcount_min
        ),
        low_info_popcount_max=(
            s.low_info_popcount_max if p.low_info_popcount_max is None else p.low_info_popcount_max
        ),
        conflict_policy=p.conflict_policy or s.conflict_policy,
        allow_missing_classes=p.allow_missing_classes,
    )
    r = make_split(s.data_dir("datasets"), p.dataset_id, config)
    return {
        "dataset_id": p.dataset_id,
        "split_id": r.split_id,
        "already_existed": r.already_existed,
        "manifest_sha256": r.manifest_sha256,
        "leakage_check": r.manifest["leakage_check"],
    }


def handle_dataset_verify(p: P.DatasetVerifyParams, ctx: JobContext) -> dict[str, Any]:
    r = verify_dataset(ctx.settings.data_dir("datasets"), p.dataset_id, full=p.full)
    return {
        "dataset_id": p.dataset_id,
        "full": r.full,
        "ok": r.ok,
        "files_checked": r.files_checked,
        "problem_count": len(r.problems),
        "problems": [{"relpath": a, "problem": b} for a, b in r.problems[:100]],
    }


def handle_train(p: P.TrainParams, ctx: JobContext) -> dict[str, Any]:
    cfg = p.config
    ctx.progress(phase="preparing", epochs=cfg.epochs, max_steps=cfg.max_steps, step=0)

    def on_start(exp_id: str, _run_dir: Any) -> None:
        ctx.store.merge_result(ctx.job_id, {"experiment_id": exp_id})
        ctx.progress(phase="training", experiment_id=exp_id, epoch=0)

    def on_epoch(r: dict[str, Any]) -> None:
        latest = {
            k: r[k] for k in ("train_loss", "val_loss", "val_accuracy", "val_balanced_accuracy")
        }
        ctx.progress(epoch=r["epoch"], step=r["steps"], latest=latest)

    result = run_training(
        cfg, ctx.settings, on_start=on_start, on_epoch=on_epoch, on_step=ctx.step_hook
    )
    ctx.progress(phase="finished")
    return {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "best_epoch": result.best_epoch,
        "best_value": result.best_value,
        "stopped_reason": result.stopped_reason,
    }


def handle_evaluate(p: P.EvaluateParams, ctx: JobContext) -> dict[str, Any]:
    r = evaluate_experiment(
        p.experiment_id,
        p.split,
        ctx.settings,
        checkpoint=p.checkpoint,
        device=p.device,
        bootstrap_resamples=p.bootstrap,
        seed=p.seed,
        force=p.force,
        reason=p.reason,
    )
    return {
        "eval_id": r.eval_id,
        "experiment_id": r.experiment_id,
        "split": r.split,
        "checkpoint": r.checkpoint,
        "forced": r.metrics["test_use"]["forced"],
    }


def handle_latency(p: P.LatencyParams, ctx: JobContext) -> dict[str, Any]:
    r = measure_experiment_latency(
        p.experiment_id, ctx.settings, device=p.device, warmup=p.warmup, iterations=p.iterations
    )
    return {"latency_id": r.latency_id, "experiment_id": r.experiment_id, "device": p.device}


HANDLERS: dict[str, Callable[[Any, JobContext], dict[str, Any]]] = {
    "dataset_split": handle_dataset_split,
    "dataset_verify": handle_dataset_verify,
    "train": handle_train,
    "evaluate": handle_evaluate,
    "latency": handle_latency,
}
