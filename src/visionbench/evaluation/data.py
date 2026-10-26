"""Evaluation data access: the ONLY code allowed to read test images.

- :func:`select_eval_entries` allows "val" and "test" (never "train": evaluating on the training
  split is not a supported report). Training code keeps its own train/val-only selector.
- Images are read through the same choke point as training (``training.data.load_image``), using
  the run's recorded deterministic validation transform, so evaluation preprocessing is identical
  to validation preprocessing (checked against the transform list stored in config.json).
- Before a checkpoint is loaded: fast dataset verify + manifest fingerprint check (the training
  pre-flight), the manifest must be the one recorded at training time, and the checkpoint file's
  sha256 must equal its artifact row. Checkpoints load with ``weights_only=True`` from the
  experiments directory only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
from visionbench.core.jsonio import read_json
from visionbench.models.baseline import MajorityBaseline
from visionbench.models.pretrained import file_sha256
from visionbench.models.registry import build_model
from visionbench.tracking.store import ExperimentStore
from visionbench.training.checkpoints import load_checkpoint
from visionbench.training.config import AugmentationConfig, TrainConfig
from visionbench.training.data import ManifestImageDataset, class_names_from_manifest
from visionbench.training.preprocessing import Preprocessing, build_transforms, describe_transforms
from visionbench.training.runner import Preflight, preflight

EVAL_SPLITS = ("val", "test")
CHECKPOINTS = ("best", "last")


class EvaluationError(VisionBenchError):
    pass


def select_eval_entries(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Manifest entries of ``split`` ("val" or "test" only)."""
    if split not in EVAL_SPLITS:
        raise EvaluationError(
            f"split {split!r} cannot be evaluated (allowed: {EVAL_SPLITS}); "
            "evaluating on the training split is not a supported report"
        )
    return [e for e in manifest["entries"] if e["split"] == split]


def build_eval_dataloader(
    split: str, manifest: dict[str, Any], raw_root: Path, transform: T.Compose, batch_size: int
) -> DataLoader:
    """Fixed-order loader over ``split`` (manifest relpath order), no shuffling, no workers."""
    index = {c: i for i, c in enumerate(class_names_from_manifest(manifest))}
    items = [(e["relpath"], index[e["label"]]) for e in select_eval_entries(manifest, split)]
    return DataLoader(
        ManifestImageDataset(raw_root, items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


@dataclass(frozen=True)
class EvalSource:
    experiment: dict[str, Any]
    run_dir: Path
    run_config: dict[str, Any]  # config.json ({"config", "resolved"})
    pre: Preflight
    class_names: list[str]
    checkpoint: str  # "best" | "last" | "baseline"
    checkpoint_path: Path
    checkpoint_sha256: str


def open_for_evaluation(
    store: ExperimentStore, settings: Settings, exp_id: str, checkpoint: str = "best"
) -> EvalSource:
    if checkpoint not in CHECKPOINTS:
        raise EvaluationError(f"checkpoint must be one of {CHECKPOINTS}, got {checkpoint!r}")
    exp = store.get(exp_id)
    if exp["status"] != "completed":
        raise EvaluationError(
            f"experiment {exp_id} has status {exp['status']!r}; only "
            "completed experiments can be evaluated"
        )
    data_root = settings.resolved_data_root
    run_dir = data_root / exp["run_dir"]
    run_config = read_json(run_dir / "config.json")
    train_cfg = TrainConfig.model_validate(run_config["config"])
    pre = preflight(train_cfg, settings.data_dir("datasets"))  # fast verify + fingerprint check
    if pre.dataset["fingerprint"] != exp["dataset_fingerprint"]:
        raise EvaluationError("dataset fingerprint differs from the one recorded at training")
    if run_config["resolved"].get("manifest_sha256") != pre.manifest_sha256:
        raise EvaluationError("split manifest differs from the one used for training")

    if exp["model"] == "majority_baseline":
        name, rel = "baseline", "checkpoints/baseline.json"
    else:
        name, rel = checkpoint, f"checkpoints/{checkpoint}.pt"
    path = run_dir / rel
    relpath = path.relative_to(data_root).as_posix()
    rows = [a for a in store.artifacts(exp_id) if a["relpath"] == relpath]
    if len(rows) != 1:
        raise EvaluationError(f"no artifact record for {relpath}")
    if not path.is_file():
        raise EvaluationError(f"checkpoint file missing: {relpath}")
    sha = file_sha256(path)
    if sha != rows[0]["sha256"]:
        raise EvaluationError(
            f"checkpoint sha256 mismatch for {relpath}: file {sha} != recorded {rows[0]['sha256']}"
        )
    return EvalSource(
        exp, run_dir, run_config, pre, class_names_from_manifest(pre.manifest), name, path, sha
    )


@dataclass
class LoadedModel:
    model: nn.Module | MajorityBaseline
    epoch: int
    input_size: int | None
    val_transform: T.Compose | None


def load_for_evaluation(source: EvalSource, data_root: Path) -> LoadedModel:
    """Load the verified checkpoint and rebuild the recorded validation transform."""
    resolved = source.run_config["resolved"]
    if source.checkpoint == "baseline":
        data = read_json(source.checkpoint_path)
        return LoadedModel(MajorityBaseline.from_dict(data), int(data["epoch"]), None, None)

    ckpt = load_checkpoint(source.checkpoint_path, data_root)  # weights_only=True
    built = build_model(
        source.experiment["model"],
        source.class_names,
        weights="scratch",
        input_size=resolved["input_size"],
    )  # architecture only; weights below
    built.model.load_state_dict(ckpt["model_state"])
    p = resolved["preprocessing"]  # JSON lists -> tuples, exactly as built at training time
    prep = Preprocessing(**{**p, "mean": tuple(p["mean"]), "std": tuple(p["std"])})
    _, val_tf = build_transforms(prep, AugmentationConfig())
    if describe_transforms(val_tf) != resolved["val_transforms"]:
        raise EvaluationError(
            "rebuilt validation transform differs from the recorded one: "
            f"{describe_transforms(val_tf)} != {resolved['val_transforms']}"
        )
    return LoadedModel(built.model, int(ckpt["epoch"]), int(resolved["input_size"]), val_tf)
