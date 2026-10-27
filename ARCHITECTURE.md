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
contain no ML logic.

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
| `cli` | implemented (M0–M4, D1) | `env`; `dataset import\|report\|verify\|split\|list\|audit-split`; `train`; `experiments list\|show`; `evaluate`; `latency`; `compare`; `evaluations list`; `worker`; `serve` |
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
| `models.small_cnn` | implemented (M2) | 4-block CNN, target layer `features.block4.conv` |
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
| `explain` | empty | M5: Grad-CAM |
| `robustness` | empty | M6: corruption benchmarks |

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

### Database schema (`db/visionbench.db`, schema version 5)

| Table | Columns |
|---|---|
| `schema_version` | `version` (PK), `applied_at` |
| `experiments` | `id` (PK), `created_at`, `started_at`, `finished_at`, `status`, `error`, `dataset_id`, `split_id`, `dataset_fingerprint`, `model`, `weights_source`, `weights_enum`, `weights_sha256`, `config_json`, `env_json`, `seed`, `best_epoch`, `best_metric`, `train_seconds`, `run_dir`, `runner_pid` (v4), `runner_create_time` (v5) |
| `epoch_metrics` | `id`, `experiment_id` (FK), `epoch`, `train_loss`, `val_loss`, `val_accuracy`, `val_balanced_accuracy`, `lr`, `epoch_seconds`, `steps` |
| `artifacts` | `id`, `experiment_id` (FK), `kind`, `relpath` (relative to data root), `sha256`, `size_bytes`, `created_at` |
| `evaluations` (M3) | `id` (PK), `experiment_id` (FK), `split` (`val`\|`test`), `checkpoint` (`best`\|`last`\|`baseline`), `created_at`, `metrics_path`, `predictions_sha256`, `visionbench_version`, `forced`, `reason` (v3) |
| `latency_measurements` (v3) | `id` (PK), `experiment_id` (FK), `checkpoint`, `device`, `device_name`, `torch_threads`, `input_size`, `warmup`, `iterations`, `results_json`, `path`, `created_at`, `visionbench_version` |
| `jobs` (v4) | `id` (PK), `kind`, `status`, `params_json`, `result_json`, `error`, `progress_json`, `cancel_requested`, `worker_pid`, `worker_create_time` (v5), `child_pid`, `child_create_time` (v5), `heartbeat_at`, `created_at`, `started_at`, `finished_at` |

- Upgrades are additive only: `create_all` adds missing tables, and a fixed list of missing
  columns is added with `ALTER TABLE ... ADD COLUMN` (v3: `evaluations.forced/reason`; v4:
  `experiments.runner_pid`; v5: `experiments.runner_create_time`,
  `jobs.worker_create_time/child_create_time`). v1 → … → v5 databases upgrade in place. A stored
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

## Error schema

Every API error response: `{"error": {"code": "<machine code>", "message": "<human text>"}}`.
Unhandled exceptions return `500 internal_error` without leaking exception details.

## API layer (M4)

`create_app(settings)` builds the app; `visionbench serve` runs it with uvicorn on
`127.0.0.1` (non-loopback hosts are refused: there is **no authentication**).

- **No ML work in a request.** Every endpoint that imports, splits, verifies, trains,
  evaluates or measures latency validates its input, runs cheap pre-checks (dataset/split/
  experiment exists, experiment completed, test-reuse rule, baseline-has-no-latency), and
  enqueues a job: `202 {"job": {...}}`. The only synchronous computation is
  `POST /api/comparisons`, which reads recorded predictions (no model is loaded).
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
