# Architecture

## Layering (library-first)

```
  FastAPI (visionbench.api)    CLI (visionbench.cli)    Job runner (visionbench.jobs, planned)
                 \                    |                      /
                  `----------- visionbench library ---------'
          core · data · models · training · evaluation · explain · robustness · tracking
```

All ML and data logic lives in the importable `visionbench` package. The API, CLI, and job
runner are thin layers: they parse input, call library functions, and serialize output. They
contain no ML logic.

## Package map

| Package | Status | Responsibility |
|---|---|---|
| `core.config` | implemented (M0) | `Settings` from env (`VISIONBENCH_` prefix / `.env`); data-root subdirs created on demand |
| `core.seeding` | implemented (M0) | `set_seed()` for random / NumPy / torch; returns a record of what was set |
| `core.env_info` | implemented (M0) | Python / torch / torchvision / NumPy / platform / CUDA versions for experiment records |
| `core.errors` | implemented (M1) | `VisionBenchError` hierarchy, the expected user-facing errors |
| `core.jsonio` | implemented (M1) | canonical JSON (sorted keys, LF, UTF-8) for byte-identical files |
| `api.main` | implemented (M0) | `create_app()` factory; `/api/health`, `/api/ready`; uniform error schema |
| `cli` | implemented (M0–M2) | `env`; `dataset import\|report\|verify\|split\|list`; `train`; `experiments list\|show` |
| `data.records` | implemented (M1) | `FileRecord`, `Issue`/`IssueCategory`, NFC names, fingerprint |
| `data.scan` | implemented (M1) | source validation, Pillow decode checks, symlink/junction refusal |
| `data.hashing` | implemented (M1) | 64-bit dHash, vectorized near-duplicate edge search |
| `data.grouping` | implemented (M1) | union-find duplicate groups, low-information guard, conflicts |
| `data.report` | implemented (M1) | `report.json` statistics |
| `data.store` | implemented (M1–M2) | snapshot import (staged + atomic rename), load/list, `verify_dataset` |
| `data.split` | implemented (M1–M2) | group-aware stratified splits (`vb-split-v2`), manifests, leakage check |
| `models.registry` | implemented (M2) | `build_model()` → model, input size, normalization source, param counts, Grad-CAM target layer |
| `models.baseline` | implemented (M2) | majority-class baseline (train priors; JSON) |
| `models.small_cnn` | implemented (M2) | 4-block CNN, target layer `features.block4.conv` |
| `models.pretrained` | implemented (M2) | torchvision ImageNet weights via torch.hub into `VISIONBENCH_TORCH_HOME` (default `<data_root>/cache/torch`); sha256 check |
| `training.config` | implemented (M2) | `TrainConfig` (Pydantic, YAML + dotted overrides) |
| `training.data` | implemented (M2) | loader factory; **train/val only** (`select_entries`); `load_image` choke point |
| `training.preprocessing` | implemented (M2) | transforms, train-only norm stats (cached), class weights |
| `training.loop` | implemented (M2) | train/val epochs, metrics, `EarlyStopping`, `fit` |
| `training.checkpoints` | implemented (M2) | `save_checkpoint` / `load_checkpoint` (`weights_only=True`, experiments store only) |
| `training.runner` | implemented (M2) | pre-flight, determinism, tracked run, artifacts |
| `tracking.store` | implemented (M2) | SQLite experiment store (SQLAlchemy), status state machine |
| `evaluation` | empty | M3: test-set evaluation (once), metrics, comparison |
| `explain` | empty | M5: Grad-CAM |
| `robustness` | empty | M6: corruption benchmarks |
| `jobs` | empty | M4: single-worker job runner |

## Data root layout

Default `./var/` (gitignored), overridable with `VISIONBENCH_DATA_ROOT`:

```
var/
  datasets/                          (M1, implemented)
    <dataset_id>/                    ds-<12 hex>; immutable once renamed into place
      dataset.json                   name, id, created_at, source_path, fingerprint, classes,
                                     counts, import config, validation summary
      report.json                    statistics, issues (+relpaths), duplicate groups, samples
      files.json                     per accepted image: relpath, label, sha256, size, w/h,
                                     mode, format, dhash (hex), gray_std
      raw/<class>/<file>             byte-identical copies of accepted images
      splits/<split_id>/             sp-<12 hex>
        manifest.json                canonical, byte-reproducible (no timestamps)
        split_info.json              created_at, manifest sha256, numpy/python versions
    .tmp-import-*/  .tmp-split-*/    staging dirs; renamed into place on success, removed on error
  experiments/                       (M2, implemented)
    <exp_id>/                        exp-<YYYYMMDD-HHMMSS>-<6 hex>
      config.json                    {"config": as given, "resolved": model, preprocessing,
                                     transforms, class names/counts, weights, fingerprints}
      env.json                       env_info + device/GPU name + determinism report
      history.jsonl                  one JSON line per epoch
      train.log                      run log (start, epochs, warnings, outcome)
      checkpoints/                   best.pt/.json, last.pt/.json (or baseline.json)
  db/visionbench.db                  (M2) SQLite experiment store
  cache/torch/checkpoints/           (M2) torchvision weights (default torch hub dir;
                                     VISIONBENCH_TORCH_HOME overrides)
  cache/norm_stats/<dataset_id>/<split_id>/norm_stats_train_<N>px.json
                                     (M3 step 0) train-only mean/std cache per resolution;
                                     split directories are never written after creation
  artifacts/                         reserved (M3+: reports, Grad-CAM maps)
```

Subdirectories are created on first use (`Settings.data_dir()`), never at import time.

## Data layer (M1)

`visionbench dataset import <src> --name N` runs the following:

1. `scan_source` (sorted, NFC). Each accepted file is read **once**. Those bytes are SHA-256
   hashed, decoded twice (`verify()`, then reopen and `load()`), dHashed, and passed to a sink that
   writes them into a staging directory.
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
  `LOAD_TRUNCATED_IMAGES`, so scanning is **not thread-safe**. The planned M4 job runner is
  single-worker.
- `dataset.json.source_path` is local CLI metadata. The M4 API must not expose it.
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

### Database schema (`db/visionbench.db`, schema version 1)

| Table | Columns |
|---|---|
| `schema_version` | `version` (PK), `applied_at` |
| `experiments` | `id` (PK), `created_at`, `started_at`, `finished_at`, `status`, `error`, `dataset_id`, `split_id`, `dataset_fingerprint`, `model`, `weights_source`, `weights_enum`, `weights_sha256`, `config_json`, `env_json`, `seed`, `best_epoch`, `best_metric`, `train_seconds`, `run_dir` |
| `epoch_metrics` | `id`, `experiment_id` (FK), `epoch`, `train_loss`, `val_loss`, `val_accuracy`, `val_balanced_accuracy`, `lr`, `epoch_seconds`, `steps` |
| `artifacts` | `id`, `experiment_id` (FK), `kind`, `relpath` (relative to data root), `sha256`, `size_bytes`, `created_at` |

- Tables are created with `create_all`. A stored schema version other than 1 is an error, since
  there are no migrations (Alembic) yet.
- Allowed status transitions: `pending → running | failed | cancelled` and
  `running → completed | failed | cancelled`. Anything else raises.
- Rows are never left `running` by a process that exits through Python. A hard kill (power loss,
  `taskkill /F`) can still leave a `running` row, and M4's startup recovery must mark those
  `failed`.

## Error schema

Every API error response: `{"error": {"code": "<machine code>", "message": "<human text>"}}`.
Unhandled exceptions return `500 internal_error` without leaking exception details.

## Planned job-runner design (M4, not implemented)

- One worker process executes jobs serially; the API process only enqueues and reads status.
- Jobs are rows in a SQLite table (`id, kind, params JSON, state, created/started/finished, error, cancel_requested`).
- Cancellation is cooperative: the worker checks `cancel_requested` between steps (e.g. per batch/epoch).
- On worker startup, any job still in `running` state is marked `failed` with reason "interrupted" (crash recovery); jobs are not auto-resumed.

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
