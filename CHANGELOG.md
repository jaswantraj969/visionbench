# Changelog

## Unreleased: B1 eurosat-v1 benchmark, validation only (2026-10-28)

### Added
- `benchmarks/eurosat-v1/`: frozen `BENCHMARK_PROTOCOL.md`, one YAML per model
  (majority_baseline, small_cnn_scratch_64, resnet18_imagenet_224,
  efficientnet_b0_imagenet_224, resnet18_scratch_64 control) and `PROTOCOL_HASHES.txt`.
  Committed before the first experiment.
- `scripts/pilot_benchmark.py`: 1-epoch timing and peak GPU memory per model, a bitwise GPU
  determinism check (two identical 50-step runs), and a budget upper bound (`pilot.json`).
- `scripts/run_benchmark.py`: `train` (hash check, resumable, interruptions recorded and rerun,
  at most one retry for run-caused failures, stop on CUDA OOM), `annotate`, `evaluate-val`,
  `latency`, `summary` (mean ± sd, per-seed values, best epoch, epochs trained,
  stopped_reason, indicative train time, per-class recall with SeaLake/Forest flagged, list of
  runs that reached the epoch cap) and `check` (no test evaluations).
- `benchmarks/eurosat-v1/runs.json` and `results/` (seed-0 VAL comparison, multi-seed VAL
  summary, seed-0 latency). Labelled "VALIDATION (selection-biased) — not test results".
- EXPERIMENT_PROTOCOL.md links the benchmark protocol; README has the reproduction commands.

### Changed
- `dataset audit-split`: field renamed to `phash_nearest_within_dhash_threshold_value`
  (different hash scales; indicative only), audit format version 2.
- DATASETS.md: Copernicus notice without a year (none is stated by the dataset authors), the
  frozen split, and the low-information limitation.

### Runs (not committed: `var/`)
- 15 benchmark runs, all completed on the first attempt; 12 pilot runs; 15 VAL evaluations;
  8 latency measurements. No test evaluation.

## Unreleased: D1 EuroSAT onboarding (2026-10-27/28)

### Added
- `visionbench.data.imageio`: the single image decoder. It decodes fully and applies
  `ImageOps.exif_transpose`. Used by the scan, training/evaluation `load_image` and the audit.
- `visionbench.data.phash`: in-house 64-bit DCT pHash (`phash-v1`; 32x32 gray, DCT-II, top-left
  8x8, median of the 63 AC coefficients, DC bit 0).
- `visionbench dataset audit-split <dataset_id> <split_id> [--seed]`
  (`visionbench.data.audit`), an independent leakage audit:
  - nearest train image of every val/test image by pHash and by dHash, over all images;
  - histograms, the consistency check, near-miss band, cross-split exact and decoded-pixel
    matches, group statistics;
  - `pairs.csv` and labelled contact sheets, under `datasets/<id>/audits/<split_id>/`.
- Oversized-group warnings (`split.group_size_stats`): multi-member groups larger than the
  class val target or 2 % of the class. Written to `split_info.json`
  (`group_size_warnings`) and printed by `dataset split`.
- `report.json` format 2: low-information images per class and per reason (std / degenerate /
  popcount), printed by `dataset import` and `dataset report`.
- `DATASETS.md`: EuroSAT RGB source, MIT license, Copernicus attribution, citations, download
  record.
- Dependency: `scipy` declared (DCT for pHash; it was already installed through scikit-learn).

### Changed
- Dataset format version 2, hash version 2. Dimensions, gray std and dHash are computed after
  EXIF orientation. `dataset.json` and `files.json` carry
  `format_version` / `orientation` / `hash_version`. Format-1 snapshots are refused with a
  request to re-import (`DatasetFormatError`).
- `SplitResult` has a `group_warnings` field.

### Data (not committed)
- EuroSAT RGB downloaded from Zenodo record 7711810. The MD5 matches the record.
- Imported as `ds-61112928e412`; `verify --full` OK.
- Nine calibration splits and two audits (thresholds 3 and 5) created under `var/`. No
  threshold was chosen; no training or evaluation was run.

## Unreleased: M4.1 process-safety and local-API security hardening (2026-10-27)

### Fixed
- **H1** Recovery could terminate an unrelated process whose pid reused a recorded child
  pid. Processes are now identified by (pid, creation time) (`visionbench.core.process`,
  psutil). Every terminate checks that identity first, and nothing is ever killed by pid alone.
- **H2** Web pages in the user's browser could send state-changing requests to the local API
  (cross-site multipart upload, body-less cancel) and, via DNS rebinding, read responses.
  New `visionbench.api.security.LocalRequestGuard`:
  - `Host` must be `localhost`/`127.0.0.1`/`[::1]` (+ `VISIONBENCH_EXTRA_TRUSTED_HOSTS`),
    otherwise `400 invalid_host`;
  - POST/PUT/PATCH/DELETE need `X-VisionBench-Request: 1`, otherwise
    `403 missing_client_header`;
  - an `Origin` not in the CORS origins gets `403 origin_not_allowed`;
  - `X-Content-Type-Options: nosniff` on every response.
- **M1** The worker lock and recovery relied on pid liveness:
  - the pid-file lock is now an exclusive OS lock (`msvcrt.locking` / `fcntl.flock`) held for
    the worker's lifetime; the file content is informational;
  - `recover()` runs only under that lock and fails every `running` job unconditionally;
  - orphaned experiments are kept only if their runner's (pid, creation time) is alive.

### Changed
- `recover(settings, lock)` requires the held `WorkerLock`, and its `interrupted:` messages
  changed (`worker <pid> is gone; child process <pid> not running`, `runner process <pid>
  not running`).
- `JobStore.claim(pid, create_time)`, `JobStore.set_child(job, pid, create_time)`, and
  `ExperimentStore.create(..., runner=(pid, create_time))` (the runner is recorded in the same
  transaction).
- The child watchdog compares the worker's identity, not just its pid.
- CORS `allow_headers` includes `x-visionbench-request`.
- Removed the `VISIONBENCH_HEARTBEAT_STALE_SECONDS` setting (recovery no longer uses heartbeat
  age).
- Coverage now measures `visionbench worker` subprocesses and their spawn children
  (`patch = ["subprocess", "_exit"]`, `concurrency = ["thread", "multiprocessing"]`). Real-worker
  tests let the worker exit by itself (`--max-jobs`) so that its data is saved.

### Added
- Dependency: `psutil` (process creation times; replaces the ctypes Win32 helpers).
- Test-only hook `VISIONBENCH_TEST_IGNORE_COOPERATIVE_CANCEL=1`, which makes a job ignore
  cooperative cancel so that the training terminate-after-timeout path can be tested.
- Tests: `tests/jobs/test_recovery.py`, `tests/api/test_api_security.py`, the training
  cancel-timeout test and the real-process lock test in `tests/jobs/test_worker_process.py`.

### Database
- Schema v5: `jobs.worker_create_time`, `jobs.child_create_time`,
  `experiments.runner_create_time`. Upgraded in place (additive).

## Unreleased: M4 FastAPI backend, single-worker job runner, secure upload (2026-10-27)

### Added
- `visionbench.jobs`:
  - `store`: `jobs` table; atomic enqueue with a queue limit (`QueueFullError` → 429) and
    atomic claim (`UPDATE ... RETURNING`); cancel, heartbeat, progress, result.
  - `params`: strict per-kind parameter models.
  - `handlers`: dataset import/split/verify, train (progress per epoch and step), evaluate,
    latency, each calling the same library functions as the CLI.
  - `worker`: per-data-root lock, one job at a time in a spawn child, heartbeat, child
    watchdog (exits when the worker dies), cooperative cancel with terminate-after-timeout,
    startup recovery of interrupted jobs and orphaned experiments.
- `visionbench.data.archive`: ZIP validation from the central directory (names, symlinks,
  encryption, case collisions, member count, total size, compression ratio, layout), safe
  extraction into a staging dir, upload import with cleanup, `import_root` confinement.
- `visionbench.api`: 23 operations under `/api` (datasets, uploads, imports, images, splits,
  verify, experiments, evaluations, latency, comparisons, jobs); streaming multipart upload
  with byte limit; strict id patterns; error mapping; scrubbed JSON responses (no absolute
  paths); CORS limited to configured origins.
- `visionbench.core.redact.scrub_paths`.
- CLI: `worker` and `serve` (API + one worker; non-loopback hosts refused).
- Training: step hook (`RunCancelled`) for cooperative cancel; `experiments.runner_pid`.
- Settings: API host/port, CORS origins, queue limit, worker timings, `import_root`, ZIP
  limits.
- Dependency: `python-multipart` (streaming multipart parser).

### Changed (M4 step 0)
- `visionbench latency <exp> --device cpu|cuda` measures latency without touching any split;
  `evaluate` no longer measures latency (`--no-latency` removed).
- `compare --eval-ids` pins evaluations; `compare --latency-device` selects one device.
- A repeated test evaluation requires `--force --reason`; `forced`/`reason` are stored and
  reported (`forced_test_reevaluation` flag).
- Comparisons on val include the selection-bias note.
- `--checkpoint last` on a majority baseline is an error.
- Comparison report format version 2; report paths are relative to the data root.

### Database
- Schema v3: `evaluations.forced`, `evaluations.reason`, `latency_measurements` table.
- Schema v4: `jobs` table, `experiments.runner_pid`. WAL journal mode and a 30 s busy
  timeout. Older databases are upgraded in place (additive only).

## Unreleased: M3 evaluation, test-use accounting, comparison (2026-10-27)

### Added
- `visionbench.evaluation.data`:
  - the only selector that reaches test images (`val|test`, never `train`);
  - evaluation re-uses the recorded val transform, which is checked exactly;
  - pre-load checks: fast verify, manifest fingerprint, manifest sha256, checkpoint sha256 vs its
    artifact row; checkpoints load with `weights_only=True`.
- `visionbench.evaluation.metrics`:
  - accuracy; balanced accuracy (absent classes listed); macro/weighted/per-class P/R/F1;
    confusion matrix (counts + row-normalized);
  - log loss, with the clipped-sample count reported;
  - ROC-AUC OvR macro and AP only when every class has positives and negatives;
  - ECE (15 bins) with a reliability table;
  - undefined values are `null` + reason;
  - percentile bootstrap CIs and a paired bootstrap of differences with shared indices.
- `visionbench.evaluation.efficiency`: parameter counts, checkpoint size, training seconds, and
  forward-pass latency (batch 1/32, median/p95, device info). The baseline reports latency as not
  applicable.
- `visionbench.evaluation.evaluate`: per-evaluation `metrics.json`, `predictions.csv`,
  `confusion_matrix.json` and `efficiency.json`, all registered as artifacts. An `evaluations`
  table (DB schema v2, additive upgrade from v1) records every run, and repeated test evaluations
  are warned and counted.
- `visionbench.evaluation.compare`:
  - refuses mismatched datasets or splits;
  - table with efficiency, val selection metric, and split metrics with CIs;
  - paired differences vs a reference;
  - VAL-only ranking with ties marked;
  - confound flags;
  - writes `reports/<id>/comparison.{json,md}`.
- CLI: `evaluate`, `compare`, `evaluations list`.
- Dependency: `scikit-learn` (planned since M0).

### Changed (M3 step 0)
- `VISIONBENCH_TORCH_HOME` sets the torch hub dir (recorded as `resolved.torch_hub_dir`).
- head_only fine-tuning keeps frozen backbone BatchNorm layers in eval mode (`frozen_bn`).
- The train norm-stats cache moved to `cache/norm_stats/<dataset_id>/<split_id>/`; split
  directories are no longer written after creation.
- The protocol documents that `train_loss` is class-weighted under weighting and `val_loss` is
  always unweighted.

### Fixed
- Listings of experiments and evaluations break same-second timestamp ties by insertion order.
  Before this, `compare` could treat an older evaluation as the latest.

## Unreleased: M2 model registry, training, tracking (2026-10-27)

### Added
- `visionbench.models`:
  - majority baseline (train priors, JSON);
  - SmallCNN (Grad-CAM target `features.block4.conv`);
  - ResNet18 and EfficientNet-B0 with scratch or ImageNet weights and full or head_only fine-tuning
    (target layers `layer4` / `features.8`, metadata only).
- Pretrained weights are fetched via torch.hub into `<data_root>/cache/torch`, loaded with
  `weights_only=True`, sha256 recorded and checked against the file-name hash. Training fails
  unless `--allow-weight-fallback` is given.
- `visionbench.training`:
  - YAML/Pydantic config with CLI overrides;
  - train/val-only loader factory;
  - train-only normalization stats (cached per split and resolution);
  - train-only augmentation and class weights;
  - val-only early stopping and best checkpoint;
  - strict deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG`;
  - run directory and status handling (completed/failed/cancelled).
- `visionbench.tracking`: SQLite experiment store (SQLAlchemy) with `experiments`,
  `epoch_metrics`, `artifacts` and `schema_version` tables.
- CLI: `visionbench train`, `visionbench experiments list|show`.
- Dependencies: `sqlalchemy` (new, the planned DB layer); `pyyaml` declared (it was already
  installed via `uvicorn[standard]`).
- Tests: model registry, overfit sanity, 4-model smoke, test-split isolation, CPU determinism,
  train-only norm stats and class weights, early stopping, failure/cancel paths, fingerprint and
  tamper aborts, checkpoint reload, weight fallback, config and CLI.

### Changed (M2 step 0)
- Splits are `vb-split-v2`: groups are ordered by `sha256(seed|class|group_id)`, with no RNG.
  `split_id`s differ from v1.
- Low-information guard: a dHash popcount < 8 or > 56 (configurable) is also excluded from
  perceptual matching.
- `.gitattributes` (LF, binary rules). Test builders moved to `tests/vb_synth.py`.

### Added (M2 step 0)
- `visionbench dataset verify <id> [--full]`.

## Unreleased: M1 dataset layer (2026-10-27)

### Added
- `visionbench dataset import|report|split|list` CLI. Errors exit with code 1 without a traceback;
  `--debug` shows it.
- `data.scan`: per-file validation with 11 issue categories. Truncation is detected with
  `verify()` followed by a reopen and full `load()`. The pixel limit is explicit, and
  symlinks/junctions are never followed.
- `data.hashing`: in-house 64-bit dHash and vectorized chunked near-duplicate search.
- `data.grouping`: union-find duplicate groups, a low-information guard, and label-conflict
  detection.
- `data.store`: staged, atomic snapshot import (`dataset.json`, `report.json`, `files.json`,
  `raw/`). It never overwrites.
- `data.split`: group-aware, class-stratified largest-deficit splits (`vb-split-v1`) with
  byte-identical canonical manifests and a computed leakage check.
- `core.errors`, `core.jsonio`, `core.seeding.derived_rng`; new `VISIONBENCH_*` dataset settings.
- `requirements/lock-dev-cu130.txt` (local CUDA dev environment freeze).
- Tests: generated fixtures covering every issue category, duplicates, splits, CLI, and a
  20k-hash performance test.

### Changed
- `/api/ready` 503 message is now generic (`data root is not writable`); the path is only logged.
- `pillow` is declared as a direct dependency (it was already installed via torchvision).
- README claims only Python 3.11 / Windows 11 as tested.

### Fixed
- `--debug` placed before a subcommand was ignored. Argparse parent parsers share action
  objects, so the root's default overwrote the subparsers' SUPPRESS default.

## 0.0.1 — M0 foundation (2026-10-27)

### Added
- `visionbench` package skeleton (src layout) with `pyproject.toml` and dev extras.
- `core.config.Settings` (pydantic-settings, `VISIONBENCH_` prefix, data root default `./var`, subdirs on demand).
- `core.seeding.set_seed` (random / NumPy / torch, deterministic flags, returns report).
- `core.env_info.collect_env_info`.
- FastAPI app factory: `GET /api/health`, `GET /api/ready` (200/503), uniform error schema.
- CLI `visionbench env`.
- Empty subpackages: data, models, training, evaluation, explain, robustness, tracking, jobs.
- Tests for config, seeding, CLI, health/ready/error schema (23 tests).
- `.gitignore`, `.env.example`, GitHub Actions CI workflow (not yet run).
- Continuity docs: IMPLEMENTATION_PLAN, PROJECT_STATUS, ARCHITECTURE, EXPERIMENT_PROTOCOL, README, REVIEW_PACKET.
