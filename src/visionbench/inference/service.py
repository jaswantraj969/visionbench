"""Single-image prediction and Grad-CAM, run synchronously in the API process.

Why not a worker job: a job needs a fresh process (10-15 s to import torch and start), which is
unusable for interactive requests. Inference is cheap enough to run in the API process, so it is
bounded instead:

- **Verified loading only.** A model is loaded through the evaluation path
  (:func:`~visionbench.evaluation.data.open_for_evaluation`): completed experiment, dataset fast
  verify + fingerprint, the split manifest recorded at training, the checkpoint's sha256 equal to
  its artifact row, ``torch.load(weights_only=True)`` from the experiments directory only. The
  validation transform is rebuilt and must equal the recorded one.
- **LRU cache** (``inference_cache_size``, default 2) keyed by (experiment id, checkpoint
  sha256). The checkpoint file is re-hashed on every request, so a changed file is never served
  from the cache.
- **Device** ``inference_device`` = cpu (default) | cuda | auto. CPU by default so inference never
  competes with a training job for the GPU. ``inference_threads`` sets torch's intra-op threads
  for the API process.
- **Concurrency**: at most ``inference_max_concurrent`` requests (default 1) run at once; a request
  that gets no slot within ``inference_wait_seconds`` fails with 429 ``inference_busy``. Each cached
  model also has its own lock (Grad-CAM registers hooks on it). The API calls this module from
  the thread pool, never on the event loop.
- **Inputs**: an uploaded image (decoded in memory with the shared decoder: EXIF orientation,
  pixel limit; never written to disk) or a recorded dataset image (lookup by relpath only).
- **Test lock**: an image of the experiment's TEST split - by relpath, or by identical bytes
  (sha256), which also catches uploads and other datasets holding the same file - is refused
  with 403 ``test_split_locked`` until ``test_images_unlocked`` is set (only after the final test
  evaluation, M8). Otherwise a test image could be inspected one by one before M8, which is
  exactly the informal test-set use the protocol forbids.
"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms as T

from visionbench.core.config import Settings
from visionbench.core.errors import VisionBenchError
from visionbench.data.imageio import PILLOW_FORMATS, ImageTooLargeError, decode_image
from visionbench.data.store import load_dataset, load_report, recorded_image
from visionbench.evaluation.data import (
    load_for_evaluation,
    open_for_evaluation,
    verify_checkpoint,
)
from visionbench.explain.geometry import ValGeometry
from visionbench.explain.gradcam import (
    GRADCAM_VERSION,
    LIMITATIONS,
    ExplanationNotSupported,
    grad_cam,
    target_layer_name,
)
from visionbench.explain.overlay import render_overlay
from visionbench.models.baseline import MajorityBaseline
from visionbench.tracking.store import DB_FILENAME, ExperimentStore
from visionbench.training.preprocessing import Preprocessing
from visionbench.training.runner import resolve_device

BASELINE_NOTE = (
    "majority_baseline does not look at the image: it always returns the class priors of the "
    "TRAIN split"
)


class InferenceError(VisionBenchError):
    """An inference request that cannot be served; carries the HTTP status and error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


@dataclass(frozen=True)
class UploadSource:
    data: bytes
    filename: str | None = None


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    relpath: str


ImageSource = UploadSource | DatasetSource


@dataclass(frozen=True)
class InputImage:
    image: Image.Image  # RGB, EXIF orientation applied
    sha256: str  # of the file bytes
    format: str
    mode: str
    source: dict[str, Any]  # what the response reports about the input

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


@dataclass
class LoadedModel:
    experiment_id: str
    model_name: str
    weights_source: str | None
    class_names: list[str]
    checkpoint: dict[str, Any]  # name, sha256, epoch
    model: nn.Module | MajorityBaseline
    val_transform: T.Compose | None
    preprocessing: Preprocessing | None
    val_transforms: list[str] | None  # recorded description
    dataset_id: str
    dataset_name: str
    split_id: str
    split_of_relpath: dict[str, str]  # experiment's split manifest: relpath -> train|val|test
    test_sha256: frozenset[str]
    training_image_size: tuple[int, int] | None  # when every dataset image has one size
    load_seconds: float
    lock: threading.Lock = field(default_factory=threading.Lock)


def _preprocessing(run_config: dict[str, Any]) -> Preprocessing:
    p = run_config["resolved"]["preprocessing"]
    return Preprocessing(**{**p, "mean": tuple(p["mean"]), "std": tuple(p["std"])})


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.inference_device)
        if settings.inference_threads is not None:
            torch.set_num_threads(settings.inference_threads)  # process-wide (API process)
        self._cache: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(settings.inference_max_concurrent)

    # --- concurrency and cache ------------------------------------------------------------

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._slots.acquire(timeout=self.settings.inference_wait_seconds):
            raise InferenceError(
                429,
                "inference_busy",
                f"all {self.settings.inference_max_concurrent} inference slot(s) are busy; "
                "retry shortly",
            )
        try:
            yield
        finally:
            self._slots.release()

    def cached_keys(self) -> list[tuple[str, str]]:
        with self._cache_lock:
            return list(self._cache)

    def get_model(self, experiment_id: str) -> tuple[LoadedModel, bool]:
        """(loaded model, cache hit). Verifies the checkpoint sha256 on every call."""
        with ExperimentStore(self.settings.data_dir("db") / DB_FILENAME) as store:
            exp = store.get(experiment_id)
            if exp["status"] != "completed":
                raise InferenceError(
                    409,
                    "experiment_not_completed",
                    f"experiment {experiment_id} has status {exp['status']!r}",
                )
            ck = verify_checkpoint(store, self.settings, experiment_id, "best")
            key = (experiment_id, ck.sha256)
            with self._cache_lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key], True
                loaded = self._load(store, experiment_id)
                if loaded.checkpoint["sha256"] != ck.sha256:  # changed while loading
                    raise InferenceError(409, "checkpoint_changed", "checkpoint changed; retry")
                self._cache[key] = loaded
                while len(self._cache) > self.settings.inference_cache_size:
                    self._cache.popitem(last=False)
                return loaded, False

    def _load(self, store: ExperimentStore, experiment_id: str) -> LoadedModel:
        start = time.perf_counter()
        source = open_for_evaluation(store, self.settings, experiment_id, "best")
        loaded = load_for_evaluation(source, self.settings.resolved_data_root)
        model = loaded.model
        if isinstance(model, nn.Module):
            model = model.to(self.device).eval()
        exp = source.experiment
        datasets = self.settings.data_dir("datasets")
        report = load_report(datasets, exp["dataset_id"])
        w, h = report["width"], report["height"]
        uniform = w["min"] == w["max"] and h["min"] == h["max"]
        entries = source.pre.manifest["entries"]
        resolved = source.run_config["resolved"]
        is_baseline = isinstance(model, MajorityBaseline)
        return LoadedModel(
            experiment_id=experiment_id,
            model_name=exp["model"],
            weights_source=exp["weights_source"],
            class_names=source.class_names,
            checkpoint={
                "name": source.checkpoint,
                "sha256": source.checkpoint_sha256,
                "epoch": loaded.epoch,
            },
            model=model,
            val_transform=loaded.val_transform,
            preprocessing=None if is_baseline else _preprocessing(source.run_config),
            val_transforms=None if is_baseline else resolved["val_transforms"],
            dataset_id=exp["dataset_id"],
            dataset_name=load_dataset(datasets, exp["dataset_id"])["name"],
            split_id=exp["split_id"],
            split_of_relpath={e["relpath"]: e["split"] for e in entries},
            test_sha256=frozenset(e["sha256"] for e in entries if e["split"] == "test"),
            training_image_size=(w["min"], h["min"]) if uniform else None,
            load_seconds=time.perf_counter() - start,
        )

    # --- inputs ---------------------------------------------------------------------------

    def read_input(self, source: ImageSource) -> InputImage:
        if isinstance(source, UploadSource):
            data = source.data
            meta: dict[str, Any] = {"type": "upload", "filename": source.filename}
        else:
            found = recorded_image(
                self.settings.data_dir("datasets"), source.dataset_id, source.relpath
            )
            if found is None:
                raise InferenceError(404, "image_not_found", "no such image in this dataset")
            record, path = found
            data = path.read_bytes()
            meta = {
                "type": "dataset",
                "dataset_id": source.dataset_id,
                "relpath": record.relpath,
                "label": record.label,
            }
        try:
            decoded = decode_image(
                data, formats=PILLOW_FORMATS, max_pixels=self.settings.max_image_pixels
            )
        except ImageTooLargeError as exc:
            raise InferenceError(413, "image_too_large", str(exc)) from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise InferenceError(
                415,
                "unsupported_image",
                f"not a readable image (allowed: {', '.join(PILLOW_FORMATS)})",
            ) from exc
        return InputImage(
            decoded.image.convert("RGB"),
            hashlib.sha256(data).hexdigest(),
            decoded.format,
            decoded.mode,
            meta,
        )

    def check_input(self, lm: LoadedModel, img: InputImage) -> tuple[list[str], list[str]]:
        """Apply the test lock (raises 403); return (warnings, notes) for the response."""
        split = None
        if img.source["type"] == "dataset" and img.source["dataset_id"] == lm.dataset_id:
            split = lm.split_of_relpath.get(img.source["relpath"])
        identical_to_test = img.sha256 in lm.test_sha256
        img.source["split"] = split
        warnings: list[str] = []
        notes: list[str] = []
        if split == "test" or identical_to_test:
            if not self.settings.test_images_unlocked:
                how = "is in" if split == "test" else "has the same bytes as an image of"
                raise InferenceError(
                    403,
                    "test_split_locked",
                    f"this image {how} the TEST split of {lm.split_id}; test images are locked "
                    "until the final test evaluation (VISIONBENCH_TEST_IMAGES_UNLOCKED)",
                )
            warnings.append("TEST image: allowed only because test images are unlocked (M8)")
        elif split == "val":
            notes.append("VALIDATION image: val chose this model's checkpoint (selection-biased)")
        elif split == "train":
            notes.append("TRAIN image: the model was fitted on it; says nothing about new data")
        elif img.source["type"] == "dataset":
            notes.append(f"image is not part of this experiment's split {lm.split_id}")
        size = lm.training_image_size
        if size is not None and img.size != size:
            warnings.append(
                f"input differs from training data ({size[0]}x{size[1]} images of dataset "
                f"'{lm.dataset_name}'); prediction may be unreliable (out-of-distribution)"
            )
        if img.source["type"] == "upload":
            notes.append(
                "uploaded image: only its size is compared with the training data; it may "
                "still be out-of-distribution"
            )
        if img.mode not in ("RGB", "L"):
            notes.append(f"converted from {img.mode} to RGB")
        return warnings, notes

    # --- requests -------------------------------------------------------------------------

    def _common(self, lm: LoadedModel, img: InputImage, hit: bool) -> dict[str, Any]:
        prep = lm.preprocessing
        return {
            "experiment_id": lm.experiment_id,
            "model": lm.model_name,
            "weights": lm.weights_source,
            "checkpoint": lm.checkpoint,
            "input": img.source
            | {"original_size": list(img.size), "format": img.format, "mode": img.mode},
            "preprocessing": None
            if prep is None
            else {
                "val_transforms": lm.val_transforms,
                "input_size": prep.input_size,
                "resize": prep.resize,
                "interpolation": prep.interpolation,
                "normalization": prep.source,
            },
            "device": str(self.device),
            "cache": "hit" if hit else "miss",
        }

    def _tensor(self, lm: LoadedModel, img: InputImage) -> torch.Tensor:
        assert lm.val_transform is not None
        return lm.val_transform(img.image).unsqueeze(0).to(self.device)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, experiment_id: str, source: ImageSource) -> dict[str, Any]:
        start = time.perf_counter()
        with self.slot():
            lm, hit = self.get_model(experiment_id)
            img = self.read_input(source)
            warnings, notes = self.check_input(lm, img)
            forward_ms = None
            if isinstance(lm.model, MajorityBaseline):
                probs = lm.model.predict_proba(1)[0]
                notes.append(BASELINE_NOTE)
            else:
                x = self._tensor(lm, img)
                with lm.lock, torch.inference_mode():
                    self._sync()
                    t0 = time.perf_counter()
                    logits = lm.model(x)
                    self._sync()
                    forward_ms = (time.perf_counter() - t0) * 1000
                probs = torch.softmax(logits.double(), dim=1)[0].cpu().tolist()
        ranked = sorted(zip(lm.class_names, probs, strict=True), key=lambda p: (-p[1], p[0]))
        top = max(range(len(probs)), key=lambda i: (probs[i], -i))
        return self._common(lm, img, hit) | {
            "predicted_class": lm.class_names[top],
            "probabilities": [{"class": c, "probability": p} for c, p in ranked],
            "timing_ms": {
                "forward": forward_ms,
                "model_load": None if hit else lm.load_seconds * 1000,
                "total": (time.perf_counter() - start) * 1000,
            },
            "warnings": warnings,
            "notes": notes,
        }

    def gradcam(
        self, experiment_id: str, source: ImageSource, target_class: str | None = None
    ) -> dict[str, Any]:
        start = time.perf_counter()
        with self.slot():
            lm, hit = self.get_model(experiment_id)
            if isinstance(lm.model, MajorityBaseline):
                raise ExplanationNotSupported(f"Grad-CAM is not supported for {lm.model_name}")
            layer = target_layer_name(lm.model_name)
            if target_class is not None and target_class not in lm.class_names:
                raise InferenceError(
                    422,
                    "unknown_class",
                    f"unknown target_class {target_class!r}; classes: {lm.class_names}",
                )
            img = self.read_input(source)
            warnings, notes = self.check_input(lm, img)
            x = self._tensor(lm, img)
            with lm.lock:
                with torch.inference_mode():
                    probs = torch.softmax(lm.model(x).double(), dim=1)[0].cpu()
                predicted = int(probs.argmax())
                target = predicted if target_class is None else lm.class_names.index(target_class)
                self._sync()
                t0 = time.perf_counter()
                cam = grad_cam(lm.model, x, target, layer)
                self._sync()
                cam_ms = (time.perf_counter() - t0) * 1000
        assert lm.preprocessing is not None
        geometry = ValGeometry.from_preprocessing(lm.preprocessing, *img.size)
        if cam.all_zero:
            warnings.append(
                "the CAM is all zeros: no positive evidence for the target class at this layer "
                "(nothing is highlighted; the map is not rescaled)"
            )
        fh, fw = cam.feature_map_size
        where = (
            f"{img.source['dataset_id']}/{img.source['relpath']}"
            if img.source["type"] == "dataset"
            else "upload"
        )
        png, shown = render_overlay(
            img.image,
            cam.normalized,
            geometry,
            all_zero=cam.all_zero,
            caption=[
                f"target: {lm.class_names[target]} (p={float(probs[target]):.3f}); "
                f"predicted: {lm.class_names[predicted]}",
                f"{lm.model_name} {lm.experiment_id}; CAM {fh}x{fw} at {cam.layer}",
                f"source: {where}",
            ],
        )
        return self._common(lm, img, hit) | {
            "gradcam_version": GRADCAM_VERSION,
            "target_class": lm.class_names[target],
            "target_probability": float(probs[target]),
            "predicted_class": lm.class_names[predicted],
            "predicted_probability": float(probs[predicted]),
            "cam": {
                "layer": cam.layer,
                "feature_map_size": [fh, fw],
                "resolution": f"{fh}x{fw}",
                "values": [[round(float(v), 6) for v in row] for row in cam.normalized],
                "normalization": "divided by the CAM maximum (0 = none, 1 = max)",
                "all_zero": cam.all_zero,
            },
            "geometry": geometry.describe(),
            "overlay": {
                "png": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                "display_size": shown,
            },
            "timing_ms": {
                "gradcam": cam_ms,
                "model_load": None if hit else lm.load_seconds * 1000,
                "total": (time.perf_counter() - start) * 1000,
            },
            "warnings": warnings,
            "notes": notes,
            "limitations": LIMITATIONS,
        }
