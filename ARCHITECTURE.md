# Architecture

## Layering (library-first)

```
  FastAPI (visionbench.api)    CLI (visionbench.cli)    Job runner (visionbench.jobs)
                 \                    |                      /
                  `----------- visionbench library ---------'
          core · data · models · training · evaluation · explain · robustness · tracking
```

All ML and data logic lives in the importable `visionbench` package. The API, CLI, and job
runner are thin layers: they parse input, call library functions, and serialize output. They
contain no ML logic. Single-image prediction and Grad-CAM (M5) are the only model work that runs
inside the API process (through `visionbench.inference`, see "Inference and Grad-CAM").

## Package map

| Package | Status | Responsibility |
|---|---|---|
| `core.config` | implemented (M0) | `Settings` from env (`VISIONBENCH_` prefix / `.env`); data-root subdirs created on demand |
| `core.seeding` | implemented (M0) | `set_seed()` for random / NumPy / torch; returns a record of what was set |
| `core.env_info` | implemented (M0) | Python / torch / torchvision / NumPy / platform / CUDA versions for experiment records |
| `core.errors` | implemented (M1) | `VisionBenchError` hierarchy, the expected user-facing errors |
| `core.jsonio` | implemented (M1) | canonical JSON (sorted keys, LF, UTF-8) for byte-identical files |
| `core.redact` | implemented (M4) | `scrub_paths()`: replaces data root / import root / torch home spellings with placeholders |
| `api.main` | implemented (M0, M4) | `create_app()` factory; CORS; exception handlers; scrubbed JSON responses; routers |
| `api.security` | implemented (M4.1) | Host allow-list, client header + Origin check on unsafe methods, `nosniff` |
| `core.process` | implemented (M4.1) | process identity (pid + creation time, psutil); identity-checked terminate |
| `api.routes.*` | implemented (M4) | datasets, experiments, evaluations, comparisons, jobs (see "API layer") |
| `api.upload` | implemented (M4) | streaming multipart receiver with a byte limit (no spooling) |
| `api.errors` / `api.deps` / `api.schemas` | implemented (M4) | error mapping, id patterns, enqueue helper, request/response models |
| `jobs.store` | implemented (M4) | `jobs` table, atomic enqueue (queue limit) and claim, cancel, heartbeat, progress |
| `jobs.params` | implemented (M4) | strict per-kind parameter models (`extra="forbid"`) |
| `jobs.handlers` | implemented (M4) | job kind → library call (import, split, verify, train, evaluate, latency) |
| `jobs.worker` | implemented (M4, M4.1) | OS lock, recovery, claim loop, child process, heartbeat, cancel, timeouts |
| `data.archive` | implemented (M4) | ZIP validation (central directory), safe extraction, upload import, `import_root` confinement |
| `evaluation.latency` | implemented (M4 step 0) | split-independent latency measurement (`latency_measurements` table) |
| `cli` | implemented (M0–M4, D1) | `env`; `dataset import\|report\|verify\|split\|list\|audit-split`; `train`; `experiments list\|show`; `evaluate`; `latency`; `compare`; `evaluations list`; `worker`; `serve` (`--allow-non-loopback` in the container only); `demo-data`; `robustness` |
| `data.records` | implemented (M1) | `FileRecord`, `Issue`/`IssueCategory`, NFC names, fingerprint |
| `data.imageio` | implemented (D1) | the one image decoder: full decode + EXIF orientation (`exif_transposed`) |
| `data.phash` | implemented (D1) | in-house 64-bit DCT pHash (audit only) |
| `data.audit` | implemented (D1) | independent split audit: nearest-train search, histograms, pair samples, contact sheets |
| `data.scan` | implemented (M1) | source validation, Pillow decode checks, symlink/junction refusal |
| `data.hashing` | implemented (M1) | 64-bit dHash, vectorized near-duplicate edge search |
| `data.grouping` | implemented (M1) | union-find duplicate groups, low-information guard, conflicts |
| `data.report` | implemented (M1) | `report.json` statistics |
| `data.store` | implemented (M1–M2) | snapshot import (staged + atomic rename), load/list, `verify_dataset` |
| `data.split` | implemented (M1–M2) | group-aware stratified splits (`vb-split-v2`), manifests, leakage check |
| `models.registry` | implemented (M2) | `build_model()` → model, input size, normalization source, param counts, Grad-CAM target layer |
| `models.baseline` | implemented (M2) | majority-class baseline (train priors; JSON) |
| `models.small_cnn` | implemented (M2) | 4-block CNN, Grad-CAM target `features.block4.relu` (post-ReLU, pre-pool; since M6 step 0) |
| `models.pretrained` | implemented (M2) | torchvision ImageNet weights via torch.hub into `VISIONBENCH_TORCH_HOME` (default `<data_root>/cache/torch`); sha256 check |
| `training.config` | implemented (M2) | `TrainConfig` (Pydantic, YAML + dotted overrides) |
| `training.data` | implemented (M2) | loader factory; **train/val only** (`select_entries`); `load_image` choke point |
| `training.preprocessing` | implemented (M2) | transforms, train-only norm stats (cached), class weights |
| `training.loop` | implemented (M2) | train/val epochs, metrics, `EarlyStopping`, `fit` |
| `training.checkpoints` | implemented (M2) | `save_checkpoint` / `load_checkpoint` (`weights_only=True`, experiments store only) |
| `training.runner` | implemented (M2) | pre-flight, determinism, tracked run, artifacts |
| `tracking.store` | implemented (M2) | SQLite experiment store (SQLAlchemy), status state machine |
| `evaluation.data` | implemented (M3) | **only** selector that reaches test (`val`\|`test`, never `train`); verified checkpoint + recorded val transform |
| `evaluation.metrics` | implemented (M3) | metrics with explicit undefined values, ECE, bootstrap and paired-bootstrap CIs |
| `evaluation.efficiency` | implemented (M3) | params, checkpoint size, forward-pass latency |
| `evaluation.evaluate` | implemented (M3) | `evaluate_experiment`: predictions, metrics, outputs, test-use accounting |
| `evaluation.compare` | implemented (M3) | comparison report, VAL-only ranking, confound flags |
| `explain.gradcam` | implemented (M5) | Grad-CAM on the registry target layer (target logit, autograd.grad, hook removed in `finally`, all-zero reported) |
| `explain.geometry` | implemented (M5) | `ValGeometry`: the recorded val transform's mapping and its exact inverse; unseen fraction |
| `explain.overlay` | implemented (M5) | overlay PNG on the original image with "not seen by the model" marking and a legend |
| `inference.service` | implemented (M5) | `InferenceService`: verified loading, LRU model cache, concurrency slots, inputs, test lock, predict / gradcam |
| `api.routes.inference` | implemented (M5) | `POST /api/experiments/{id}/predict` and `/gradcam` (body read on the loop, model work in the thread pool) |
| `robustness.suite` | implemented (M6) | suite YAML (corruptions x fixed severities), suite sha256, lookup by name, per-image seeds |
| `robustness.corruptions` | implemented (M6) | gaussian_noise, gaussian_blur, brightness, jpeg as pure uint8 -> uint8 functions |
| `robustness.materialize` | implemented (M6) | corrupted PNGs written once per condition with a hashed manifest; reruns prove byte identity; test lock |
| `robustness.evaluate` | implemented (M6) | val-selected checkpoint on every verified condition, paired with the pinned clean evaluation; resumable |
| `robustness.report` | implemented (M6) | per-model / per-condition report over a benchmark's `runs.json` (Markdown + JSON) |
| `core.power` | implemented (M6 step 0) | `keep_awake()` for long-running scripts (Windows) |
| `benchmarks` | implemented (M7) | read-only views of frozen benchmark folders: hash status, runs joined with the database, fixed documents, VAL summary, final report |
| `api.routes.benchmarks` | implemented (M7) | `GET /api/overview`, `/api/benchmarks[/{name}[/documents/{document}]]`, `/api/robustness-suites` |
| `data.demo` | implemented (B3) | seeded synthetic shapes dataset for `visionbench demo-data` and the e2e test (not EuroSAT) |
| `frontend/` | implemented (M7) | React + TypeScript SPA over the API (see "Frontend") |

## Data root layout

Default `./var/` (gitignored), overridable with `VISIONBENCH_DATA_ROOT`:

```
var/
  datasets/                          (M1, implemented)
    <dataset_id>/                    ds-<12 hex>; immutable once renamed into place
      dataset.json                   format_version (2), orientation, hash_version, name, id,
                                     created_at, source_path, fingerprint, classes, counts,
                                     import config, validation summary
      report.json                    statistics, issues (+relpaths), duplicate groups, samples
      files.json                     format header + per accepted image: relpath, label,
                                     sha256, size, w/h (after EXIF orientation), mode, format,
                                     dhash (hex), gray_std
      raw/<class>/<file>             byte-identical copies of accepted images
      splits/<split_id>/             sp-<12 hex>
        manifest.json                canonical, byte-reproducible (no timestamps)
        split_info.json              created_at, manifest sha256, numpy/python versions,
                                     group_size_warnings (D1)
      audits/<split_id>/             (D1) audit.json, audit.md, pairs.csv, sheets/*.png;
                                     replaced on re-run; never inside splits/
    .tmp-import-*/  .tmp-split-*/    staging dirs; renamed into place on success, removed on error
  experiments/                       (M2, implemented)
    <exp_id>/                        exp-<YYYYMMDD-HHMMSS>-<6 hex>
      config.json                    {"config": as given, "resolved": model, preprocessing,
                                     transforms, class names/counts, weights, fingerprints}
      env.json                       env_info + device/GPU name + determinism report
      history.jsonl                  one JSON line per epoch
      train.log                      run log (start, epochs, warnings, outcome)
      checkpoints/                   best.pt/.json, last.pt/.json (or baseline.json)
      evaluations/<eval_id>/         (M3) ev-<YYYYMMDD-HHMMSS>-<6 hex>: metrics.json,
                                     predictions.csv, confusion_matrix.json, efficiency.json
      latency/<latency_id>/          (M4) lat-...: latency.json (one device per measurement)
      robustness/<robustness_id>/    (M6) rb-...: metrics.json, predictions.csv (one corruption
                                     condition; never mixed with evaluations/)
  db/visionbench.db                  (M2) SQLite experiments + jobs (WAL: also -wal, -shm)
  db/worker.lock                     (M4.1) OS-locked while a worker runs; content (pid, creation
                                     time of the last holder) is informational only
  tmp/uploads/<upload_id>.zip        (M4) streamed upload, removed after import/rejection
  tmp/staging-<upload_id>/           (M4) extraction staging, always removed
  cache/torch/checkpoints/           (M2) torchvision weights (default torch hub dir;
                                     VISIONBENCH_TORCH_HOME overrides)
  cache/norm_stats/<dataset_id>/<split_id>/norm_stats_train_<N>px.json
                                     (M3 step 0) train-only mean/std cache per resolution;
                                     split directories are never written after creation
  cache/corruptions/<dataset_id>/<split_id>/<suite>/<corruption>/<severity>/
                                     (M6) condition.json (hashed manifest) + <class>/<stem>.png;
                                     written once (temp dir + rename), re-verified on rerun
  reports/<report_id>/               (M3) rp-...: comparison.json, comparison.md
  downloads/                         (D1) official dataset archives + their extraction
                                     (not managed by Settings; see DATASETS.md)
  artifacts/                         reserved (M5+: Grad-CAM maps)
```

Subdirectories are created on first use (`Settings.data_dir()`), never at import time.

## Data layer (M1)

`visionbench dataset import <src> --name N` runs the following:

1. `scan_source` (sorted, NFC). Each accepted file is read **once**. Those bytes are
   SHA-256 hashed, then decoded twice: `verify()`, then a full decode through
   `data.imageio.decode_image`, which applies EXIF orientation. Dimensions, gray std and dHash
   are taken from the oriented image. The bytes are then passed to a sink that writes them
   into a staging directory.
2. `check_requirements`: at least 2 classes, and at least `min_images_per_class` per class.
3. Fingerprint, then `dataset_id`. The import is refused if the id already exists.
4. `build_groups` + `build_report`, then `files.json` / `report.json` / `dataset.json`.
5. `os.rename(staging, datasets/<id>)`. On any error the staging directory is deleted, so no
   partial dataset is ever visible.

`visionbench dataset split <id> --seed S` loads `files.json` and checks it against the
fingerprint. It then rebuilds the groups with the split's thresholds and assigns them
(see EXPERIMENT_PROTOCOL.md §4). Finally it writes `splits/<split_id>/` using the same
staging + rename pattern. If that split already exists, it only verifies it byte for byte.

Notes:
- The scan temporarily sets Pillow's module-level `MAX_IMAGE_PIXELS` and
  `LOAD_TRUNCATED_IMAGES`, so scanning is **not thread-safe**. The M4 job runner is
  single-worker and runs each job in its own process.
- `dataset.json.source_path` is local metadata. The API never returns it (uploads record
  `upload:<filename>`, path imports `import_root:<relpath>`).
- `data.imageio.decode_image` is the only function that turns image bytes into pixels.
  `training.data.load_image` (the patched choke point in tests), the audit and the scan all
  call it.
- `load_dataset` / `load_records` refuse snapshots whose format header (`format_version`,
  `orientation`, `hash_version`) differs from the current one (`DatasetFormatError`, which
  asks for a re-import).
- `split.group_size_stats(manifest)` gives the group-size distribution, the largest group per
  class and the oversized groups. `make_split` writes the oversized groups to
  `split_info.json`; the audit reuses the same function.
- `dataset audit-split` (`data.audit.audit_split`):
  1. reads the manifest and records;
  2. decodes every entry (pHash + decoded-pixel digest);
  3. runs chunked nearest-neighbour searches (val/test against train, NumPy XOR + popcount);
  4. samples pairs with `derived_rng(seed, ...)`;
  5. renders the contact sheets with Pillow (tiles upscaled with NEAREST);
  6. writes everything to a staging directory that replaces `audits/<split_id>/`.

  Cost on EuroSAT (27,000 images of 64x64): about 30 s per audit.
- The snapshot is "immutable" because it is never overwritten and its hashes are recorded. Files
  are not made read-only. `dataset verify [--full]` (M2) detects missing, resized, unrecorded and
  (with `--full`) altered files.

## Training (M2)

`visionbench train --config cfg.yaml [flags] [--set key=value]` → `training.runner.run_training`:

1. **Pre-flight**, before any DB row exists:
   - the dataset exists and passes a fast `verify_dataset`;
   - the split id is valid and its manifest exists;
   - the manifest's `dataset_id` and fingerprint match `dataset.json`.

   Any failure here aborts with an error, and no experiment is recorded.
2. Experiment row `pending`, run directory created, `train.log` handler attached, then `running`.
3. Inside the tracked run:
   - resolve the device;
   - configure determinism and write `env.json`;
   - `build_model`, which fetches pretrained weights if requested;
   - preprocessing (pretrained transform, or train-split norm stats);
   - train and val loaders;
   - optimizer, scheduler, loss;
   - write `config.json`;
   - `loop.fit`.
4. After every epoch: append to `history.jsonl`, insert a DB `epoch_metrics` row, save
   `last.pt`, and save `best.pt` when the monitored val metric improves.
5. Outcome:
   - success → `completed` with `best_epoch`/`best_metric`;
   - `KeyboardInterrupt` → `cancelled`;
   - any other exception → `failed`, with `"<Type>: <message>"` stored and the partial history and
     checkpoints kept.

   In all cases `train_seconds` is stored and every file in the run directory is registered as an
   artifact (path, sha256, size).

The majority baseline uses the same path but needs only labels. It fits train priors, scores the
val labels, and writes `checkpoints/baseline.json`. It opens no images.

### Database schema (`db/visionbench.db`, schema version 7)

| Table | Columns |
|---|---|
| `schema_version` | `version` (PK), `applied_at` |
| `experiments` | `id` (PK), `created_at`, `started_at`, `finished_at`, `status`, `error`, `dataset_id`, `split_id`, `dataset_fingerprint`, `model`, `weights_source`, `weights_enum`, `weights_sha256`, `config_json`, `env_json`, `seed`, `best_epoch`, `best_metric`, `train_seconds`, `run_dir`, `runner_pid` (v4), `runner_create_time` (v5), `stopped_reason` (v6: `early_stopping`\|`epochs_completed`\|`max_steps`\|`baseline_fit`, set when a run completes) |
| `epoch_metrics` | `id`, `experiment_id` (FK), `epoch`, `train_loss`, `val_loss`, `val_accuracy`, `val_balanced_accuracy`, `lr`, `epoch_seconds`, `steps` |
| `artifacts` | `id`, `experiment_id` (FK), `kind`, `relpath` (relative to data root), `sha256`, `size_bytes`, `created_at` |
| `evaluations` (M3) | `id` (PK), `experiment_id` (FK), `split` (`val`\|`test`), `checkpoint` (`best`\|`last`\|`baseline`), `created_at`, `metrics_path`, `predictions_sha256`, `visionbench_version`, `forced`, `reason` (v3) |
| `latency_measurements` (v3) | `id` (PK), `experiment_id` (FK), `checkpoint`, `device`, `device_name`, `torch_threads`, `input_size`, `warmup`, `iterations`, `results_json`, `path`, `created_at`, `visionbench_version` |
| `robustness_evaluations` (v7) | `id` (PK, `rb-...`), `experiment_id` (FK), `split`, `checkpoint`, `suite`, `suite_sha256`, `corruption`, `severity`, `condition_hash`, `clean_evaluation_id` (FK → `evaluations`), `metrics_path`, `predictions_sha256`, `created_at`, `visionbench_version` |
| `jobs` (v4) | `id` (PK), `kind`, `status`, `params_json`, `result_json`, `error`, `progress_json`, `cancel_requested`, `worker_pid`, `worker_create_time` (v5), `child_pid`, `child_create_time` (v5), `heartbeat_at`, `created_at`, `started_at`, `finished_at` |

- Upgrades are additive only: `create_all` adds missing tables, and a fixed list of missing
  columns is added with `ALTER TABLE ... ADD COLUMN` (v3: `evaluations.forced/reason`; v4:
  `experiments.runner_pid`; v5: `experiments.runner_create_time`,
  `jobs.worker_create_time/child_create_time`; v6: `experiments.stopped_reason`; v7: the
  `robustness_evaluations` table). v1 → … → v7
  databases upgrade in place. Experiments completed before v6 were backfilled once from their
  own `train.log` (`scripts/backfill_stopped_reason.py`, which changes no other field). A stored
  version newer than the code is an error; there are no migrations (Alembic) yet.
- Every connection sets `PRAGMA busy_timeout=30000`. `journal_mode=WAL` is set once when the
  engine is opened (retried, because a database whose last writer was hard-killed can briefly
  report `disk I/O error`). WAL lets the API read while the worker child writes.
- Listings are ordered by `created_at` and then SQLite insertion order (`rowid`). Timestamps
  have 1-second resolution, and `compare` relies on "latest evaluation" being well defined.
- Allowed status transitions: `pending → running | failed | cancelled` and
  `running → completed | failed | cancelled`. Anything else raises.
- Rows are never left `running` by a process that exits through Python. A hard kill (power loss,
  `taskkill /F`) can still leave a `running` row; the worker's startup recovery (below) marks
  those `failed` with `interrupted: ...`. A user stop inside the API worker (cancel) ends as
  `cancelled` with the partial history kept.

## Evaluation (M3)

`visionbench evaluate <exp> --split val|test [--checkpoint best|last]` → `evaluation.evaluate.evaluate_experiment`:

1. `open_for_evaluation`:
   - the experiment must be `completed`;
   - the training pre-flight runs (fast verify + manifest fingerprint);
   - the manifest sha256 must equal the one recorded at training;
   - the checkpoint file's sha256 must equal its artifact row.
2. The previous test evaluations of this experiment + checkpoint are counted, and a warning
   is printed if there are any.
3. `load_for_evaluation`: `weights_only=True` load from `experiments/`. The architecture is
   rebuilt from the registry, and the val transform is rebuilt from `config.json` and must
   match the recorded transform list exactly.
4. Predictions run in manifest relpath order through `evaluation.data.build_eval_dataloader`
   (`select_eval_entries` + the shared `load_image` choke point). The majority baseline uses
   labels only and opens no images.
5. Metrics, bootstrap CIs and efficiency (params, checkpoint size, training time) are
   computed. The four output files are written and registered as artifacts, and an
   `evaluations` row is inserted. Since M4 step 0 evaluation does **not** measure latency.

A repeated test evaluation of the same experiment + checkpoint is refused unless
`force=True` with a non-empty `reason` (`check_test_reuse`, used by both CLI and API); the
row records `forced` and `reason`.

`visionbench latency <exp> --device cpu|cuda` (`evaluation.latency`) measures forward-pass
latency on random input at the model's input size. It verifies the checkpoint (sha256 vs its
artifact row) but reads **no split and no image**. Results go to
`experiments/<exp>/latency/<lat_id>/latency.json` and a `latency_measurements` row.

`visionbench compare <exp> <exp> ... --split val|test [--eval-ids ...] [--latency-device D]`
reads the latest recorded evaluation of each experiment's val-selected checkpoint, or the
pinned `--eval-ids`. It never evaluates implicitly. It verifies each predictions file against
its recorded sha256, requires identical samples and labels, takes latency from the latest
measurement on the chosen device, and writes `reports/<report_id>/comparison.{json,md}`.
Comparisons on val state the selection-bias note.

## Inference and Grad-CAM (M5)

`POST /api/experiments/{id}/predict` and `POST /api/experiments/{id}/gradcam` serve one image
synchronously from the API process. A worker job is not used for this: starting a job process
takes 10-15 s (torch import), which is unusable interactively.

**Loading (verified path only).** `InferenceService.get_model` loads a model through
`evaluation.data.open_for_evaluation` + `load_for_evaluation`: the experiment must be completed;
the dataset passes a fast verify and its fingerprint matches; the split manifest is the one
recorded at training; the checkpoint's sha256 equals its artifact row; `torch.load` runs with
`weights_only=True` on a file inside `<data_root>/experiments/`; the rebuilt val transform must
equal the recorded one. Nothing else is ever loaded.

**Model cache.** An LRU `OrderedDict` of `inference_cache_size` entries (default 2), keyed by
(experiment id, checkpoint sha256). Every request re-hashes the checkpoint file and checks its
artifact row first, so a changed file is never served from the cache (it fails with the
evaluation path's sha256 error). The re-hash dominates the warm request time of the larger
models (about 45 MB for resnet18).

**Device and threads.** `VISIONBENCH_INFERENCE_DEVICE` = `cpu` (default) | `cuda` | `auto`. CPU
is the default so that interactive requests never compete with a training job for the 6 GB
GPU. `VISIONBENCH_INFERENCE_THREADS` sets torch's intra-op threads for the whole API process.

**Concurrency.** A `BoundedSemaphore(inference_max_concurrent)` (default 1) guards every
request; a request that gets no slot within `inference_wait_seconds` (default 2 s) fails with
`429 inference_busy`. Each cached model also has its own lock, because Grad-CAM registers a hook
on it. The route reads the request body on the event loop (bounded) and runs all model work
with `run_in_threadpool`, so the event loop is never blocked. A test measures `GET /api/health`
during a slow prediction and, as a control, shows that running the model on the loop does block
it.

**Inputs.** Either a multipart upload (`file`; at most `inference_max_upload_mb`, default 10 MB;
JPEG/PNG/BMP/WEBP; parsed in memory, never written to disk; decoded by the shared decoder with
EXIF orientation and the `max_image_pixels` limit checked on the header before decoding), or a
JSON body `{dataset_id, relpath}` naming a recorded dataset image (lookup-only, the same
`data.store.recorded_image` as the image endpoint). Errors: `413 upload_too_large` /
`image_too_large`, `415 unsupported_image` / `unsupported_media_type`, `422 missing_file`,
`404 image_not_found`.

**Test lock.** If the image is in the TEST split of the experiment's manifest (by relpath), or
its bytes are identical (sha256) to a test image (which also covers uploads and other datasets
holding the same file), the request fails with `403 test_split_locked` unless
`VISIONBENCH_TEST_IMAGES_UNLOCKED=true` (default false; to be set only at M8, after the final
test evaluation). Reason: looking at test images one by one before M8 is informal test-set use,
which the protocol forbids (EXPERIMENT_PROTOCOL.md section 6). Re-encoded or edited copies of a
test image are not detected. Val and train images are allowed and labelled in `notes`; an image
whose size differs from the training images gets an out-of-distribution warning.

**Grad-CAM** (`explain.gradcam`). A forward hook on the registry's target layer (`small_cnn`
`features.block4.relu`, `resnet18` `layer4`, `efficientnet_b0` `features.8`, all post-activation maps) captures the
activation A. The gradient of the target logit (pre-softmax) with respect to A is computed with
`torch.autograd.grad`: the same gradient a backward hook sees, without writing `.grad` into the
shared cached model (a test checks equality with a backward-hook implementation). Channel
weights are the spatial mean of the gradients; CAM = ReLU(sum_k w_k A_k). The model is in eval
mode, gradients are enabled only inside the call, and the hook is removed in `finally`. A CAM
with no positive evidence is returned as zeros with `all_zero: true` and a warning, never
rescaled. The feature-map resolution depends on the input size: 8x8 for small_cnn at 64 px,
7x7 for resnet18 / efficientnet_b0 at 224 px, and only **2x2 for resnet18 at 64 px**.

**Geometry** (`explain.geometry.ValGeometry`). The CAM lives in model-input space. It is mapped
onto the ORIGINAL image through the exact inverse of the recorded val transform: for
`Resize((s, s))` a per-axis scale; for `Resize(r)` + `CenterCrop(s)` the resized size and crop
offset computed with torchvision's own arithmetic (`int(r * long / short)`,
`int(round((size - s) / 2))`), in continuous pixel-edge coordinates (`align_corners=False`).
Each display pixel centre is mapped to input coordinates and the CAM sampled bilinearly. Pixels
outside the input, i.e. the border removed by the centre crop (23.4 % of a 64x64 EuroSAT tile),
are "not seen by the model": the overlay greys and hatches them, outlines the seen region and
states the unseen fraction in the legend, and the response reports `seen_box_in_original` and
`unseen_fraction`. Tests check the mapping against the real transforms (marker positions,
cropped border markers, round trips).

**Response** (gradcam): target and predicted class with probabilities, `cam.feature_map_size`
and `resolution`, the normalised low-resolution CAM values, `geometry`, the overlay PNG as a
base64 data URI, timings, warnings, notes, the fixed `limitations` text, and
`gradcam_version` (`vb-gradcam-2` since M6 step 0, when small_cnn's target moved from the pre-BN
conv output `features.block4.conv` to the post-ReLU map; `vb-gradcam-1` before). Shared overlays of
dataset images need the dataset's attribution (DATASETS.md).

## Robustness (M6)

Pre-registered corruptions at fixed severities, evaluated identically for every model; the
protocol for EuroSAT is `benchmarks/eurosat-v1/robustness/ROBUSTNESS_PROTOCOL.md`.

- **Suite** (`robustness.suite`): a YAML file with `suite`, `base_seed` and corruptions x
  severities; its sha256 identifies it. The API and the worker look suites up **by name** under
  `VISIONBENCH_ROBUSTNESS_SUITES_DIR` (default `benchmarks`); clients never send paths.
- **Corruptions** (`robustness.corruptions`): pure functions on the decoded original `uint8` RGB
  image (noise, separable Gaussian blur with reflect padding, HSV-V brightness, Pillow JPEG
  4:2:0), clipped and quantized back to `uint8`. Per-image seeds come from
  `sha256("suite|corruption|severity|base_seed|relpath")`, so results do not depend on order,
  batching, model or process.
- **Materialization** (`robustness.materialize`): each condition is written once to
  `cache/corruptions/...` as lossless PNG in a temp directory renamed into place (with retries:
  Windows can briefly lock fresh files). `condition.json` records the suite hash, the split
  manifest hash, library versions and, per image, the source and corrupted sha256; the condition
  hash covers all of it. A rerun re-derives every image and compares bytes; it never rewrites.
- **Evaluation** (`robustness.evaluate`): the val-selected checkpoint through the verified
  evaluation path, its recorded val transform applied to the corrupted PNG. Before use, every
  condition manifest and corrupted file is re-hashed. The clean reference is a pinned clean
  evaluation of the same checkpoint and split; its `predictions.csv` must match its sha256 and
  list the same images in the same order, so the paired bootstrap uses identical indices.
  Results go only to `robustness_evaluations` + `<run_dir>/robustness/<id>/`. Conditions already
  evaluated (same suite hash and condition hash) are skipped.
- **Test lock:** materialization, evaluation, the report and the API refuse `split=test`
  (`TestSplitLockedError`; API `403 test_split_locked`) unless
  `VISIONBENCH_TEST_IMAGES_UNLOCKED=true` (M8).
- **API / job:** `POST /api/experiments/{id}/robustness {suite, split}` checks the test lock,
  that the experiment is completed (409), that the suite exists (404) and that every condition is
  materialized (409 `corruptions_not_materialized`), then queues a `robustness` job (suite sha256
  re-checked in the worker; progress per condition; cooperative cancel between conditions).
  `GET /api/robustness?experiment_id=...` and `GET /api/robustness/{id}` read results.
- **Report** (`robustness.report`): reads recorded results only; scope statement, confound flags
  and the VAL label in every report; no model is called robust.

## Error schema

Every API error response: `{"error": {"code": "<machine code>", "message": "<human text>"}}`.
Unhandled exceptions return `500 internal_error` without leaking exception details.

## API layer (M4)

`create_app(settings)` builds the app; `visionbench serve` runs it with uvicorn on
`127.0.0.1` (non-loopback hosts are refused: there is **no authentication**).

- **No ML work in a request, except bounded single-image inference.** Every endpoint that
  imports, splits, verifies, trains, evaluates or measures latency validates its input, runs
  cheap pre-checks (dataset/split/experiment exists, experiment completed, test-reuse rule,
  baseline-has-no-latency), and enqueues a job: `202 {"job": {...}}`. Synchronous
  computation: `POST /api/comparisons` (reads recorded predictions; no model is loaded) and,
  since M5, `predict` / `gradcam` for one image (see "Inference and Grad-CAM").
- **Ids** are validated with strict patterns (`ds-[0-9a-f]{12}`, `sp-...`, `exp-YYYYMMDD-
  HHMMSS-[0-9a-f]{6}`, `ev-`, `rp-`, `job-`); a mismatch is `422 validation_error` before
  any filesystem access.
- **Errors** map library exceptions to codes (`api.errors.map_exception`): not found → 404,
  `RepeatedTestEvaluationError` → `409 test_reevaluation_requires_force`, `QueueFullError`
  → `429 queue_full`, archive rejections → `422 archive_<code>`, path-import refusals →
  403, validation → 422, unexpected → `500 internal_error` (no details).
- **Path hygiene.** Responses never contain absolute paths: records store paths relative to
  the data root, `source_path` is not returned, job errors are scrubbed by the worker, and
  the default response class (`ScrubbedJSONResponse`) replaces any spelling of the data root,
  import root or torch home with `<data_root>` / `<import_root>` / `<torch_home>` as a last
  line of defence. Process ids are not returned either.
- **CORS** allows only `VISIONBENCH_CORS_ORIGINS` (default the Vite dev server on
  `localhost:5173` / `127.0.0.1:5173`), methods GET/POST, headers `content-type` and
  `x-visionbench-request`.
- **Browser protection** (`api.security.LocalRequestGuard`, outermost middleware): see the
  threat model below.
- **Image endpoint** `GET /api/datasets/{id}/images?relpath=...` treats `relpath` as a key
  into the dataset's `files.json` records; it never joins user input onto a path. The file
  served is built from the stored record, resolved, and must lie inside `raw/`. Anything
  else (traversal, encoded traversal, absolute paths, unrecorded files) is
  `404 image_not_found`.

### Threat model: a local API without authentication (M4.1)

The API trusts whoever can reach `127.0.0.1:<port>`. That is the user, plus any program they
run. Authentication is out of scope. What the API must withstand is **web pages open in the
user's browser**, which can make the browser send requests to localhost:

| Attack | Without protection | Defence |
|---|---|---|
| CSRF with a "simple" request: a cross-site `<form>` multipart upload, or a body-less `fetch(..., {method: "POST", mode: "no-cors"})` to `/api/jobs/{id}/cancel` | executed; CORS only hides the response | POST/PUT/PATCH/DELETE need `X-VisionBench-Request: 1` → otherwise `403 missing_client_header`. A browser sends a custom header cross-origin only after a CORS preflight, which a foreign origin fails, and a `<form>` cannot set headers. |
| A cross-origin request that carries the header anyway (defence in depth) | — | an `Origin` header that is present but not in `VISIONBENCH_CORS_ORIGINS` (including `null`) → `403 origin_not_allowed` |
| DNS rebinding: `attacker.example` re-resolved to 127.0.0.1, so the page is "same-origin" and can also read responses | executed and readable | the `Host` header must be `localhost`, `127.0.0.1` or `[::1]` (any port; plus `VISIONBENCH_EXTRA_TRUSTED_HOSTS`) → otherwise `400 invalid_host`, for every method including GET |
| MIME sniffing of served images/JSON | — | `X-Content-Type-Options: nosniff` on every response |

GET endpoints change nothing and need no header. A rebinding page cannot read them either,
because of the Host check. Not covered: other local programs or users on the same machine,
and a malicious browser extension. These need authentication, which is out of scope. The
guard is a pure ASGI middleware, so upload bodies still stream through the byte limit.
Starlette's `TrustedHostMiddleware` was not used because its 400 is plain text. The guard
returns the standard JSON error body instead.

## Job lifecycle and worker (M4)

```
             POST (202)            claim (atomic)          child outcome
  client ──► queued ───────────► running ───────────────► completed | failed | cancelled
               │ cancel              │ cancel → cancel_requested
               ▼                     │   (child stops cooperatively, or is terminated
           cancelled                 │    after cancel_timeout → cancelled + note)
     "cancelled before start"        │ worker/child dies → (next worker start) failed
                                     ▼                     "interrupted: ..."
```

- **Enqueue** is one `INSERT ... SELECT ... WHERE (queued count) < :limit`, so the queue
  limit (`VISIONBENCH_MAX_QUEUED_JOBS`, default 10) holds under concurrent requests.
- **Process identity (M4.1).** A process is identified by **(pid, creation time)**, never by
  its pid alone. Pids are reused, and quickly on Windows. `core.process` (psutil) records the
  creation time next to every stored pid (`jobs.worker_*`, `jobs.child_*`,
  `experiments.runner_*`). `process_is(pid, create_time)` is true only if that pid exists
  **and** its creation time matches (±0.05 s). A missing creation time never matches.
  `terminate_process` kills only after `process_is` is true, using the same psutil handle, which
  re-checks the creation time. The other kill sites terminate a `Popen`/`multiprocessing`
  handle of a child the caller started itself, which cannot refer to a reused pid.
- **One worker per data root (M4.1)**: the worker holds an exclusive **OS lock** on
  `db/worker.lock` for its whole lifetime: `msvcrt.locking` on one byte at offset 2^30 on
  Windows, so the content stays readable, and `fcntl.flock` on POSIX. A second worker fails at
  once with `another worker is running on this data root`. The OS releases the lock when the
  holder dies, even by `taskkill /F`, so the next worker acquires it immediately. The file
  content (`{"pid", "create_time"}` of the last holder) is informational and is never used to
  decide anything.
- **Claim** is `UPDATE jobs SET status='running' ... WHERE id = (oldest queued) AND status=
  'queued' RETURNING id`, a single atomic statement. The worker runs one job at a time.
- **Child process.** Each job runs in a fresh `multiprocessing` *spawn* child calling the same
  library functions as the CLI (`execute_job` → `jobs.handlers`). The worker polls the child,
  writes `heartbeat_at` every `worker_heartbeat_seconds` (2 s), and watches
  `cancel_requested`. The child runs a watchdog thread that calls `os._exit` as soon as the
  worker (pid + creation time) is gone, so killing the worker never leaves training running.
  `heartbeat_at` is informational (shown by the API). Recovery does not use it.
- **Progress.** Training updates `progress_json` after every epoch (epoch, latest metrics)
  and, throttled to 0.5 s, the step counter; the API returns it with the job.
- **Cancel.** `POST /api/jobs/{id}/cancel`: a queued job is cancelled at once (a queued
  upload's file is deleted); a running job gets `cancel_requested`. Training checks the flag
  in its step hook, raises `RunCancelled`, marks the experiment `cancelled` (history and
  checkpoints kept) and the job `cancelled by request`. If the child has not stopped
  `cancel_timeout_seconds` (30 s) after the request, it is terminated and the job and
  experiment are marked cancelled with a note.
- **Recovery** (`recover(settings, lock)`) runs when a worker starts, and only while it
  holds the lock. It refuses to run otherwise. At that moment no other worker can exist,
  so:
  1. **every** `running` job is orphaned, whatever its pids or heartbeat say. Recovery
     terminates its child only if `process_is(child_pid, child_create_time)`; otherwise it
     records that the child was not running. It marks the job's experiment `failed` first,
     deletes its upload temp files, then marks the job `failed` with
     `interrupted: worker <pid> is gone; <child note>`;
  2. every `pending`/`running` experiment is marked `failed`
     (`interrupted: runner process <pid> not running`) **unless**
     `process_is(runner_pid, runner_create_time)` is true, e.g. a live CLI
     `visionbench train`. The runner identity is written in the same transaction that creates
     the experiment.

  Recovery therefore does not wait for any heartbeat age. The former
  `VISIONBENCH_HEARTBEAT_STALE_SECONDS` setting was removed.

  Jobs are never resumed automatically. Each recovery is logged as a WARNING.
- `visionbench serve` starts the API and **one** `visionbench worker` subprocess.
  `visionbench worker` can also be run alone (e.g. with `serve --no-worker`).

## Upload pipeline and limits (M4)

`POST /api/datasets/uploads` (multipart: `file`, `name`):

1. **Receive.** Only `multipart/form-data` (else 415). A `Content-Length` larger than the
   limit is rejected before reading; the body is then parsed in a streaming way
   (`python-multipart`) and the file part is written to `tmp/uploads/<upload_id>.zip`
   (opened with `"xb"`) while counting bytes. Exceeding `VISIONBENCH_MAX_UPLOAD_MB`
   (default 200) aborts with `413 upload_too_large`. Starlette's `UploadFile` is not used
   because it spools the whole body first.
2. **Validate** (synchronously, central directory only; no member is decompressed):
   - magic bytes `PK\x03\x04` (`PK\x05\x06` → `archive_empty_archive`, anything else →
     `archive_not_a_zip`), readable central directory (else `archive_corrupt_zip`);
   - member names checked on the raw stored name: NUL/control characters, `<>:"|?*`, empty
     or `.` components, backslashes, UNC, absolute paths, drive letters, `..`, trailing
     dots/spaces, Windows reserved names (`CON`, `NUL`, `COM1`, ...);
   - symlinks (Unix mode in `external_attr`) and encrypted members are rejected;
   - duplicate names after NFC + casefold (`case_collision`);
   - at most `VISIONBENCH_ZIP_MAX_MEMBERS` (50 000) members, total uncompressed size at most
     `VISIONBENCH_ZIP_MAX_UNCOMPRESSED_MB` (4096), compression ratio at most
     `VISIONBENCH_ZIP_MAX_RATIO` (100) for each member of at least 1 MiB and for the whole
     archive once it is at least 1 MiB;
   - layout: class folders at the root, or one top folder containing them (`__MACOSX/` is
     ignored); anything else is `422 archive_bad_layout`.
   Any rejection deletes the upload and returns `422 archive_<code>` naming the member.
3. **Queue** a `dataset_import` job (429 if the queue is full; the upload is deleted).
4. **Import** (in the worker): extract into `tmp/staging-<upload_id>/`, creating each file
   with `"xb"` after checking the resolved path is inside the staging dir, and counting the
   bytes actually written against the declared size and the total limit. The staging folder
   is then imported with the normal dataset import (all M1 validation applies). The upload
   and staging folder are deleted in a `finally`, by recovery if the child was killed, and
   by cancel if the job never started.

**Path imports** (`POST /api/datasets/imports`) are disabled unless `VISIONBENCH_IMPORT_ROOT`
is set (403 `path_import_disabled`); the resolved path must be a directory inside the
import root (403 `path_outside_import_root`). Only the relative path is stored.

## Model-loading policy

- Only checkpoints produced internally by VisionBench are loaded. `load_checkpoint` refuses any
  path outside `<data_root>/experiments/`.
- Checkpoints contain only a state_dict plus primitives, and are loaded with
  `torch.load(..., weights_only=True)`.
- User-uploaded models or pickles are never accepted or loaded.
- Pretrained weights come only from torchvision's weights API. They are fetched through torch.hub
  with `check_hash=True` into the torch hub dir (`VISIONBENCH_TORCH_HOME`, default
  `<data_root>/cache/torch`), loaded with `weights_only=True`, and the
  file's sha256 is checked against the hash prefix in its name and recorded.

## Frontend (M7)

`frontend/` is a single-page app: React 18, TypeScript (strict), Vite, React Router, TanStack
Query and Recharts.

**Serving:**
- **Production:** `visionbench serve` serves `frontend/dist` from the API's own origin.
  - A catch-all GET, registered after every router, returns files from inside `dist`, and
    otherwise `index.html` (SPA fallback, `Cache-Control: no-cache`).
  - `/api/...` paths that match no route keep the JSON 404.
  - The Host check and the client-header rule apply to these responses too.
- **Development:** Vite serves the app on `127.0.0.1:5173` and proxies `/api` to the API. That
  origin is in `cors_origins` by default.

**API layer:**
- **Types:** `src/api/schema.d.ts` is generated from `create_app().openapi()`
  (`npm run gen:api`). `npm run check:api` fails when it is stale.
- **Client:** `src/api/client.ts` is the one client.
  - It sends `X-VisionBench-Request: 1` on every mutation.
  - It maps the error schema to user messages, with distinct texts for `test_split_locked`,
    `inference_busy`, `queue_full`, 409, 413, 415, 422 and 429.
  - Uploads use XHR, for progress.
- **Hooks:** `src/api/hooks.ts` holds the TanStack Query hooks. Active jobs are polled.

**Same-origin POSTs:** browsers send `Origin` on same-origin POSTs. The Origin check therefore
accepts the API's own origin (`http(s)://` + the validated Host), as well as `cors_origins`.

**What the pages guarantee:**
- Every page has loading, empty, error and success states.
- Every metric carries its split label ("VALIDATION (selection-biased)" or "TEST").
- Undefined metrics are shown as undefined, never as 0.
- TEST actions are not offered while the test split is locked.
- EuroSAT attribution appears wherever dataset tiles do.
- The app has no state of its own: everything comes from the API.

**Tests:**
- Vitest + Testing Library + MSW cover components and pages.
- Playwright runs one end-to-end flow against a real backend: `scripts/e2e_backend.py`, a
  temporary data root and the synthetic shapes dataset.

## Packaging and CI

**Docker:**
- **Image (`Dockerfile`):** a Node build stage (frontend), then `python:3.11-slim`. Both are
  pinned by digest.
  - Dependencies come from `requirements/lock-cpu-linux.txt` (CPU torch).
  - The image runs as a non-root user, with data in the `/data` volume and a healthcheck on
    `/api/health`.
  - It sets `VISIONBENCH_CONTAINER=1`, and only then does `serve --allow-non-loopback` bind to
    `0.0.0.0` (`cli.bind_refusal`).
- **`compose.yaml`** publishes the port on the host's `127.0.0.1` only. Inside the container the
  Host allow-list still applies.

**Lock files:**
- `lock-cpu-linux.txt` (runtime) and `lock-dev-cpu-linux.txt` (runtime + dev tools) come from
  `scripts/lock_cpu_linux.sh` in the pinned image.
- `lock-dev-cu130.txt` is the Windows CUDA development environment.

**CI:**
- `scripts/linux_tests.sh` runs ruff and the full suite in the pinned image as a non-root user.
  This exercises the POSIX branches: the `fcntl` worker lock and psutil process handling.
- `.github/workflows/ci.yml` has four jobs: backend, frontend, e2e and docker.

