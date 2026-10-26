"""Training orchestration: pre-flight checks, tracked run, checkpoints, artifacts.

Pre-flight (before any experiment row exists): the dataset must exist and pass a fast verify, the
split manifest must belong to it and carry the same fingerprint. Invalid inputs therefore never
create experiment records. Everything after that runs inside a tracked run whose status always
ends as completed, failed (error stored, partial history kept) or cancelled (KeyboardInterrupt).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from visionbench.core.config import Settings
from visionbench.core.env_info import collect_env_info
from visionbench.core.errors import VisionBenchError
from visionbench.core.jsonio import read_json, write_json
from visionbench.core.seeding import set_seed
from visionbench.data.store import dataset_dir, load_dataset, verify_dataset
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import file_sha256
from visionbench.models.registry import BuiltModel, build_model
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training import loop
from visionbench.training.checkpoints import save_checkpoint
from visionbench.training.config import TrainConfig
from visionbench.training.data import (
    build_dataloader,
    class_names_from_manifest,
    select_entries,
    split_class_counts,
)
from visionbench.training.preprocessing import (
    Preprocessing,
    build_transforms,
    class_weights_from_train,
    describe_transforms,
    train_norm_stats,
)

SPLIT_ID_RE = re.compile(r"^sp-[0-9a-f]{12}$")
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

logger = logging.getLogger(__name__)


class TrainingError(VisionBenchError):
    pass


@dataclass(frozen=True)
class Preflight:
    dataset: dict[str, Any]
    manifest: dict[str, Any]
    manifest_sha256: str
    split_dir: Path
    raw_root: Path


@dataclass(frozen=True)
class TrainingResult:
    experiment_id: str
    status: str
    run_dir: Path
    best_epoch: int | None
    best_value: float | None
    stopped_reason: str
    history: list[dict[str, Any]]


def preflight(config: TrainConfig, datasets_dir: Path) -> Preflight:
    meta = load_dataset(datasets_dir, config.dataset_id)
    check = verify_dataset(datasets_dir, config.dataset_id, full=False)
    if not check.ok:
        listed = "; ".join(f"{p}: {d}" for p, d in check.problems[:10])
        raise TrainingError(
            f"dataset {config.dataset_id} failed verification ({len(check.problems)} problem(s)): "
            f"{listed}"
        )
    if not SPLIT_ID_RE.match(config.split_id):
        raise TrainingError(f"invalid split id {config.split_id!r} (expected sp-<12 hex>)")
    split_dir = dataset_dir(datasets_dir, config.dataset_id) / "splits" / config.split_id
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"split {config.split_id} not found for dataset {config.dataset_id}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("dataset_id") != config.dataset_id:
        raise TrainingError(f"manifest belongs to dataset {manifest.get('dataset_id')!r}")
    if manifest.get("dataset_fingerprint") != meta["fingerprint"]:
        raise TrainingError(
            f"fingerprint mismatch: manifest {manifest.get('dataset_fingerprint')} != "
            f"dataset.json {meta['fingerprint']}; refusing to train"
        )
    return Preflight(
        meta,
        manifest,
        hashlib.sha256(raw).hexdigest(),
        split_dir,
        dataset_dir(datasets_dir, config.dataset_id) / "raw",
    )


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainingError("device 'cuda' requested but torch.cuda.is_available() is False")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_determinism(seed: int) -> dict[str, Any]:
    """Seed everything and require deterministic algorithms (errors on ops without one)."""
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG  # before any cuBLAS work
    report = set_seed(seed, deterministic=True)
    torch.use_deterministic_algorithms(True)  # strict: unsupported ops raise instead of warn
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    return report | {
        "torch_deterministic_algorithms_warn_only": warn_only,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "cublas_workspace_config_previous": previous,
    }


def _env(device: torch.device, determinism: dict[str, Any]) -> dict[str, Any]:
    info = collect_env_info()
    info["device"] = str(device)
    info["device_name"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else info["processor"]
    )
    info["determinism"] = determinism
    return info


def _artifact_kind(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    if name.endswith(".pt"):
        return "checkpoint"
    if rel.startswith("checkpoints/") and name == "baseline.json":
        return "baseline_model"
    if rel.startswith("checkpoints/"):
        return "checkpoint_meta"
    return {
        "config.json": "config",
        "env.json": "env",
        "history.jsonl": "history",
        "train.log": "log",
    }.get(name, "other")


def _register_artifacts(
    store: ExperimentStore, exp_id: str, run_dir: Path, data_root: Path
) -> None:
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        store.add_artifact(
            exp_id,
            kind=_artifact_kind(path.relative_to(run_dir).as_posix()),
            relpath=path.relative_to(data_root).as_posix(),
            sha256=file_sha256(path),
            size_bytes=path.stat().st_size,
        )


class _RunContext:
    """Per-run outputs: history.jsonl, DB epoch rows, train.log, user callback."""

    def __init__(
        self,
        store: ExperimentStore,
        exp_id: str,
        run_dir: Path,
        run_log: logging.Logger,
        on_epoch: Callable[[dict[str, Any]], None] | None,
    ):
        self.store, self.exp_id, self.run_dir = store, exp_id, run_dir
        self.log, self.on_epoch = run_log, on_epoch

    def record_epoch(self, record: dict[str, Any]) -> None:
        with open(self.run_dir / "history.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self.store.add_epoch(self.exp_id, record)
        self.log.info(
            "epoch %d train_loss=%s val_loss=%.6f val_acc=%.4f val_bal_acc=%.4f lr=%s "
            "steps=%d time=%.2fs",
            record["epoch"],
            record["train_loss"],
            record["val_loss"],
            record["val_accuracy"],
            record["val_balanced_accuracy"],
            record["lr"],
            record["steps"],
            record["epoch_seconds"],
        )
        if self.on_epoch is not None:
            self.on_epoch(record)


def _run_baseline(
    ctx: _RunContext, config: TrainConfig, pre: Preflight, class_names: list[str]
) -> loop.FitResult:
    """Fit priors on TRAIN counts; score VAL labels. No image is opened."""
    start = time.perf_counter()
    model = MajorityBaseline(class_names).fit(split_class_counts(pre.manifest, "train"))
    index = {c: i for i, c in enumerate(class_names)}
    val_labels = [index[e["label"]] for e in select_entries(pre.manifest, "val")]
    if not val_labels:
        raise TrainingError("validation split is empty")
    confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.int64)
    for y, p in zip(val_labels, model.predict(len(val_labels)), strict=True):
        confusion[y, p] += 1
    train_counts = model.train_counts or {}
    train_n = sum(train_counts.values())
    record = {
        "epoch": 1,
        "train_loss": sum(n * model.nll(index[c]) for c, n in train_counts.items()) / train_n,
        "val_loss": sum(model.nll(y) for y in val_labels) / len(val_labels),
        "val_accuracy": float(confusion.diag().sum()) / len(val_labels),
        "val_balanced_accuracy": loop.balanced_accuracy(confusion),
        "lr": None,
        "steps": 0,
        "train_samples": train_n,
        "epoch_seconds": time.perf_counter() - start,
    }
    write_json(
        ctx.run_dir / "checkpoints" / "baseline.json",
        model.to_dict()
        | {
            "epoch": 1,
            "metrics": {
                k: record[k] for k in ("val_loss", "val_accuracy", "val_balanced_accuracy")
            },
        },
    )
    ctx.record_epoch(record)
    monitor = config.early_stopping.monitor
    return loop.FitResult(
        history=[record],
        best_epoch=1,
        best_value=record[monitor],
        stopped_reason="baseline_fit",
        total_steps=0,
    )


def _preprocessing(
    built: BuiltModel, pre: Preflight, cache_dir: Path, data_root: Path
) -> tuple[Preprocessing, dict[str, Any]]:
    if built.pretrained is not None:
        p = built.pretrained
        prep = Preprocessing("imagenet_weights", p.crop, p.resize, p.interpolation, p.mean, p.std)
        return prep, {"source": "imagenet_weights", "weights_enum": p.enum_name}
    assert built.input_size is not None
    stats, cached, cache_file = train_norm_stats(
        pre.manifest, pre.raw_root, built.input_size, cache_dir
    )
    prep = Preprocessing(
        "train_split",
        built.input_size,
        None,
        stats["interpolation"],
        tuple(stats["mean"]),
        tuple(stats["std"]),
    )
    return prep, {
        "source": "train_split",
        "stats": stats,
        "cache_file": cache_file.relative_to(data_root).as_posix(),
        "loaded_from_cache": cached,
    }


def _make_optimizer(config: TrainConfig, params: list[nn.Parameter]) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )


def _execute(
    ctx: _RunContext, config: TrainConfig, pre: Preflight, settings: Settings
) -> loop.FitResult:
    data_root = settings.resolved_data_root
    device = resolve_device(config.device)
    if config.amp and device.type != "cuda":
        raise TrainingError("amp requires a CUDA device")
    determinism = configure_determinism(config.seed)
    env = _env(device, determinism)
    write_json(ctx.run_dir / "env.json", env)
    ctx.store.update(ctx.exp_id, env_json=json.dumps(env, sort_keys=True))

    class_names = class_names_from_manifest(pre.manifest)
    built = build_model(
        config.model,
        class_names,
        weights=config.weights,
        finetune=config.finetune,
        input_size=config.input_size,
        hub_dir=settings.resolved_torch_home,
        allow_weight_fallback=config.allow_weight_fallback,
    )
    ctx.store.update(
        ctx.exp_id,
        weights_source=built.weights,
        weights_enum=built.pretrained.enum_name if built.pretrained else None,
        weights_sha256=built.pretrained.sha256 if built.pretrained else None,
    )
    if built.fallback_reason:
        ctx.log.warning("WEIGHT FALLBACK: training from scratch: %s", built.fallback_reason)

    resolved: dict[str, Any] = {
        "model": built.describe(),
        "class_names": class_names,
        "dataset_fingerprint": pre.dataset["fingerprint"],
        "manifest_sha256": pre.manifest_sha256,
        "split_algorithm": pre.manifest.get("algorithm"),
        "train_class_counts": split_class_counts(pre.manifest, "train"),
        "val_class_counts": split_class_counts(pre.manifest, "val"),
        "device": str(device),
        "torch_hub_dir": str(settings.resolved_torch_home),
    }

    def write_config() -> None:
        doc = {"config": config.model_dump(mode="json"), "resolved": resolved}
        write_json(ctx.run_dir / "config.json", doc)
        ctx.store.update(ctx.exp_id, config_json=json.dumps(doc, sort_keys=True))

    if isinstance(built.model, MajorityBaseline):
        write_config()
        return _run_baseline(ctx, config, pre, class_names)

    norm_cache = settings.norm_stats_dir(config.dataset_id, config.split_id)
    prep, norm = _preprocessing(built, pre, norm_cache, data_root)
    train_tf, val_tf = build_transforms(prep, config.augmentation)
    pin = device.type == "cuda"
    train_loader = build_dataloader(
        "train",
        pre.manifest,
        pre.raw_root,
        train_tf,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    val_loader = build_dataloader(
        "val",
        pre.manifest,
        pre.raw_root,
        val_tf,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    class_weights = class_weights_from_train(pre.manifest) if config.class_weighted_loss else None
    resolved |= {
        "input_size": prep.input_size,
        "preprocessing": prep.to_dict(),
        "normalization": norm,
        "train_transforms": describe_transforms(train_tf),
        "val_transforms": describe_transforms(val_tf),
        "class_weights": class_weights,
        "train_drop_last": train_loader.drop_last,
        "train_loss": "class-weighted cross-entropy" if class_weights else "mean cross-entropy",
        "val_loss": "unweighted mean cross-entropy",
        "frozen_bn": bool(loop.frozen_batchnorm_layers(built.model)),
        "frozen_bn_layers": len(loop.frozen_batchnorm_layers(built.model)),
    }
    write_config()

    model = built.model.to(device)
    weight = (
        torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = _make_optimizer(config, [p for p in model.parameters() if p.requires_grad])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )
    es = config.early_stopping
    early = loop.EarlyStopping(es.monitor, es.patience, es.min_delta, es.enabled)
    ckpt_meta = {
        "model": config.model,
        "class_names": class_names,
        "input_size": prep.input_size,
        "weights": built.weights,
        "monitor": es.monitor,
    }

    def on_epoch_end(record: dict[str, Any], is_best: bool, m: nn.Module) -> None:
        metrics = {k: record[k] for k in ("val_loss", "val_accuracy", "val_balanced_accuracy")}
        ckpt_dir = ctx.run_dir / "checkpoints"
        save_checkpoint(
            ckpt_dir / "last.pt", m, epoch=record["epoch"], metrics=metrics, meta=ckpt_meta
        )
        if is_best:
            save_checkpoint(
                ckpt_dir / "best.pt",
                m,
                epoch=record["epoch"],
                metrics=metrics,
                meta=ckpt_meta | {"selected_by": f"best {es.monitor} on val"},
            )
        ctx.record_epoch(record | {"is_best": is_best})

    return loop.fit(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.epochs,
        num_classes=len(class_names),
        early_stopping=early,
        max_steps=config.max_steps,
        amp=config.amp,
        on_epoch_end=on_epoch_end,
    )


def run_training(
    config: TrainConfig,
    settings: Settings,
    *,
    on_start: Callable[[str, Path], None] | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    data_root = settings.resolved_data_root
    pre = preflight(config, settings.data_dir("datasets"))
    store = ExperimentStore(settings.data_dir("db") / DB_FILENAME)
    exp_id = store.create(
        dataset_id=config.dataset_id,
        split_id=config.split_id,
        dataset_fingerprint=pre.dataset["fingerprint"],
        model=config.model,
        seed=config.seed,
        config={"config": config.model_dump(mode="json")},
        run_dir="experiments/{exp_id}",
    )
    run_dir = settings.data_dir("experiments") / exp_id
    (run_dir / "checkpoints").mkdir(parents=True)
    run_log = logging.getLogger(f"visionbench.training.run.{exp_id}")
    run_log.setLevel(logging.INFO)
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    run_log.addHandler(handler)
    ctx = _RunContext(store, exp_id, run_dir, run_log, on_epoch)

    start = time.perf_counter()
    status, result = "failed", loop.FitResult()
    try:
        store.transition(exp_id, "running")
        run_log.info("experiment %s started: %s", exp_id, config.model_dump_json())
        if on_start is not None:
            on_start(exp_id, run_dir)
        result = _execute(ctx, config, pre, settings)
        best = result.best_value
        store.update(
            exp_id,
            best_epoch=result.best_epoch,
            best_metric=None if best is None or math.isnan(best) else best,
        )
        store.transition(exp_id, "completed")
        status = "completed"
        run_log.info(
            "completed: best_epoch=%s best_%s=%s stopped=%s",
            result.best_epoch,
            config.early_stopping.monitor,
            result.best_value,
            result.stopped_reason,
        )
    except KeyboardInterrupt:
        status = "cancelled"
        run_log.warning("cancelled by user (KeyboardInterrupt)")
        store.transition(exp_id, "cancelled", error="cancelled by user (KeyboardInterrupt)")
        raise
    except BaseException as exc:
        run_log.error("failed: %s: %s", type(exc).__name__, exc)
        store.transition(exp_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        store.update(exp_id, train_seconds=time.perf_counter() - start)
        run_log.removeHandler(handler)
        handler.close()
        _register_artifacts(store, exp_id, run_dir, data_root)
        store.close()

    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return TrainingResult(
        exp_id,
        status,
        run_dir,
        result.best_epoch,
        result.best_value,
        result.stopped_reason,
        history,
    )


def load_run_config(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "config.json")
