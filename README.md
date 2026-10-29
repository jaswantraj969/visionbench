# VisionBench

Reproducible computer-vision benchmarking and explainability platform. **Current state: milestone
M5 (single-image prediction and Grad-CAM served from the API, with a model cache, bounded
concurrency and a TEST-image lock), on top of M4 (FastAPI backend, single-worker job runner,
secure dataset upload), D1 (EuroSAT RGB imported and audited) and B1 (frozen `eurosat-v1`
benchmark protocol; 15 runs trained and evaluated on VAL only; test untouched until M8)**. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for what comes next.

## What exists now

- Importable package `visionbench` (src layout).
- `visionbench.core`: settings (`VISIONBENCH_` env vars), `set_seed()`, environment info.
- `visionbench.data` (library and CLI): dataset import into immutable snapshots, validation,
  statistics, exact and near duplicate groups, and leakage-aware train/val/test split manifests.
- `visionbench.models`: majority baseline, SmallCNN, ResNet18 and EfficientNet-B0 (scratch or
  ImageNet weights).
- `visionbench.training` and `visionbench.tracking`: CLI training from a split manifest. Training
  uses the train split only, and model selection and early stopping use val only. Checkpoints and
  a SQLite experiment store are written.
- `visionbench.evaluation`: the only code that reads test images. It provides metrics with
  explicit undefined values, bootstrap CIs, test-use accounting, comparison reports with
  confound flags, and split-independent latency measurement.
- FastAPI backend under `/api` (datasets, uploads, splits, experiments, evaluations, latency,
  comparisons, jobs) and a single-worker job runner: every training/evaluation/import runs
  as a queued job in a separate process, with progress, cancel and crash recovery.
- CLI: `visionbench env`, `dataset import|report|verify|split|list|audit-split`, `train`,
  `experiments list|show`, `evaluate`, `latency`, `compare`, `evaluations list`, `worker`,
  `serve`.
- Tests (pytest) and lint (ruff).

- `visionbench.inference` and `visionbench.explain` (M5): prediction and Grad-CAM for one image,
  served synchronously by the API, with overlays that mark what the model never saw.

There is **no** robustness suite, authentication, or frontend yet.

## Setup

Tested on Python 3.11 (Windows 11). `pyproject.toml` declares `>=3.11,<3.14`, but 3.12 and 3.13
have not been tested. Install PyTorch from the official wheel index first, then the package.
The exact versions of the verified local CUDA environment are in
[requirements/lock-dev-cu130.txt](requirements/lock-dev-cu130.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# EITHER: CUDA 13.0 build (what was used on this machine: RTX 3050, driver CUDA 13.4)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# OR: CPU-only build (used by CI; not run on this machine yet)
# .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On Linux/macOS use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Configuration

Copy `.env.example` to `.env` or set environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VISIONBENCH_DATA_ROOT` | `./var` | Runtime data root; `datasets/`, `artifacts/`, `db/` created on demand |
| `VISIONBENCH_MAX_UPLOAD_MB` | `200` | ZIP upload size limit (413 above it, enforced while streaming) |
| `VISIONBENCH_LOG_LEVEL` | `INFO` | Log level |
| `VISIONBENCH_MAX_IMAGE_PIXELS` | `50000000` | Images with more pixels are rejected |
| `VISIONBENCH_MIN_IMAGES_PER_CLASS` | `3` | Minimum valid images per class |
| `VISIONBENCH_DHASH_THRESHOLD` | `5` | Max dHash Hamming distance for near duplicates |
| `VISIONBENCH_LOW_INFO_STD_THRESHOLD` | `3.0` | Gray std below which perceptual matching is skipped |
| `VISIONBENCH_LOW_INFO_POPCOUNT_MIN` / `_MAX` | `8` / `56` | dHash popcount outside this range skips perceptual matching |
| `VISIONBENCH_REPORT_SAMPLES_PER_CLASS` | `8` | Seeded sample size per class in `report.json` |
| `VISIONBENCH_CONFLICT_POLICY` | `exclude` | Label-conflict duplicate groups: `exclude` or `error` |
| `VISIONBENCH_TORCH_HOME` | `<data_root>/cache/torch` | torch hub dir for pretrained weights |
| `VISIONBENCH_API_HOST` / `_API_PORT` | `127.0.0.1` / `8000` | `serve` bind address (loopback only) |
| `VISIONBENCH_CORS_ORIGINS` | `["http://localhost:5173","http://127.0.0.1:5173"]` | Allowed browser origins (JSON list); also the only `Origin` values accepted on POST |
| `VISIONBENCH_EXTRA_TRUSTED_HOSTS` | `[]` | `Host` names accepted besides `localhost`, `127.0.0.1`, `[::1]` (JSON list) |
| `VISIONBENCH_MAX_QUEUED_JOBS` | `10` | Queued jobs before `429 queue_full` |
| `VISIONBENCH_WORKER_POLL_SECONDS` | `0.5` | Worker queue poll interval |
| `VISIONBENCH_WORKER_HEARTBEAT_SECONDS` | `2.0` | Heartbeat interval while a job runs |
| `VISIONBENCH_CANCEL_TIMEOUT_SECONDS` | `30.0` | After a cancel request, terminate the job's process after this long |
| `VISIONBENCH_IMPORT_ROOT` | unset | Enables `POST /api/datasets/imports`, confined to this folder |
| `VISIONBENCH_ZIP_MAX_MEMBERS` | `50000` | Max ZIP members |
| `VISIONBENCH_ZIP_MAX_UNCOMPRESSED_MB` | `4096` | Max total uncompressed ZIP size |
| `VISIONBENCH_ZIP_MAX_RATIO` | `100` | Max compression ratio (zip-bomb guard) |
| `VISIONBENCH_INFERENCE_DEVICE` | `cpu` | `cpu`, `cuda` or `auto` for predict / gradcam (CPU keeps the GPU free for training) |
| `VISIONBENCH_INFERENCE_THREADS` | unset | torch intra-op threads in the API process |
| `VISIONBENCH_INFERENCE_CACHE_SIZE` | `2` | Loaded models kept (LRU) |
| `VISIONBENCH_INFERENCE_MAX_CONCURRENT` | `1` | Inference requests running at once |
| `VISIONBENCH_INFERENCE_WAIT_SECONDS` | `2.0` | Wait for a free slot, then `429 inference_busy` |
| `VISIONBENCH_INFERENCE_MAX_UPLOAD_MB` | `10` | Image upload limit for predict / gradcam |
| `VISIONBENCH_ROBUSTNESS_SUITES_DIR` | `benchmarks` | Where robustness suites (`robustness_suite*.yaml`) are looked up by name |
| `VISIONBENCH_TEST_IMAGES_UNLOCKED` | `false` | Allow TEST-split images in predict / gradcam; set only at M8, after the final test evaluation |

## Datasets

Source folder format: exactly one level of class folders. The folder name is the label.

```
my_dataset/
  cats/  img001.jpg  img002.png ...
  dogs/  a.webp  b.bmp ...
```

Supported extensions (case-insensitive): `.jpg .jpeg .png .bmp .webp`. At least 2 classes are
required, each with at least 3 valid images (configurable). Anything else is reported, never
silently dropped: unsupported, corrupt, truncated, empty, oversized, hidden/system files,
symlinks/junctions (never followed), nested folders, and name collisions.

```powershell
.\.venv\Scripts\visionbench.exe dataset import C:\path\to\my_dataset --name my-dataset
.\.venv\Scripts\visionbench.exe dataset list
.\.venv\Scripts\visionbench.exe dataset verify ds-xxxxxxxxxxxx          # add --full to re-hash files
.\.venv\Scripts\visionbench.exe dataset report ds-xxxxxxxxxxxx          # add --json for everything
.\.venv\Scripts\visionbench.exe dataset split ds-xxxxxxxxxxxx --seed 42 --fractions 0.7 0.15 0.15
.\.venv\Scripts\visionbench.exe dataset audit-split ds-xxxxxxxxxxxx sp-xxxxxxxxxxxx   # leakage audit
```

- `import` copies accepted images into `<data_root>/datasets/<id>/raw/` and writes
  `dataset.json`, `report.json` and `files.json`. It refuses to overwrite an existing dataset.
  Optional flags: `--max-image-pixels`, `--min-images-per-class`, `--dhash-threshold`,
  `--low-info-std`, `--low-info-popcount-min`, `--low-info-popcount-max`.
- `verify` checks the snapshot against `files.json`: fingerprint, file existence and sizes, and
  no unrecorded files. `--full` also re-hashes every file. Any mismatch is listed and the command
  exits with code 1.
- `split` writes `<data_root>/datasets/<id>/splits/<split_id>/manifest.json`. The same dataset and
  config always produce a byte-identical manifest. Duplicate groups never span splits, and the
  manifest records a computed leakage check. Optional flags: `--conflict-policy exclude|error`,
  `--allow-missing-classes`, `--dhash-threshold`, `--low-info-std`, `--low-info-popcount-min`,
  `--low-info-popcount-max`. It also prints **oversized-group warnings**: multi-member groups
  larger than their class's val target or larger than 2 % of the class. These are recorded in
  `split_info.json`, not in the manifest.
- `audit-split` re-checks a split **without trusting its duplicate grouping**. It decodes every
  image and uses an independent DCT pHash plus the dHash over all images, low-information
  ones included. For each val/test image it finds the nearest train image and reports
  distance histograms, a consistency check, near misses, cross-split exact and
  pixel-identical matches, and group statistics. It writes `audit.json`, `audit.md`,
  `pairs.csv` and labelled contact sheets to `<data_root>/datasets/<id>/audits/<split_id>/`;
  the split directory is never touched. `--seed` (default 0) controls the sampled pairs. The
  contact sheets contain dataset images, so do not commit them.
- Images are read with their EXIF orientation applied everywhere: scan, hashes, training,
  evaluation and audit. Datasets imported before this change (format 1) must be re-imported.
- Errors print `error: ...` and exit with code 1. Add `--debug` for a traceback.

Real datasets used by this project, with their licenses, citations and download records, are
listed in [DATASETS.md](DATASETS.md). Dataset files are never committed.

Algorithm details and limitations: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Training

Training is always started explicitly. A YAML config sets any `TrainConfig` field, and CLI flags or
`--set key=value` override it (dotted keys, YAML values):

```yaml
# small_cnn.yaml
model: small_cnn          # majority_baseline | small_cnn | resnet18 | efficientnet_b0
weights: scratch          # imagenet (resnet18/efficientnet_b0 only) | scratch
input_size: 64
epochs: 15
batch_size: 32
optimizer: adamw          # adamw | sgd
lr: 0.001
scheduler: cosine         # none | cosine
early_stopping: {monitor: val_loss, patience: 5, min_delta: 0.0}
seed: 0
num_workers: 0
```

```powershell
.\.venv\Scripts\visionbench.exe train --config small_cnn.yaml --dataset-id ds-xxxxxxxxxxxx --split-id sp-xxxxxxxxxxxx --device cuda
.\.venv\Scripts\visionbench.exe train --dataset-id ds-... --split-id sp-... --model majority_baseline --device cpu
.\.venv\Scripts\visionbench.exe train --config cfg.yaml --set early_stopping.patience=3 --max-steps 10
.\.venv\Scripts\visionbench.exe experiments list
.\.venv\Scripts\visionbench.exe experiments show exp-YYYYMMDD-HHMMSS-xxxxxx   # add --json
```

- Before training, the dataset is fast-verified and the manifest fingerprint is checked. Any
  mismatch aborts the run.
- Each run writes `<data_root>/experiments/<exp_id>/`: `config.json`, `env.json`,
  `history.jsonl`, `train.log` and `checkpoints/`. It also adds rows to
  `<data_root>/db/visionbench.db`.
- `--device auto|cpu|cuda`. `--amp` requires CUDA and is untested.
- With `weights: imagenet`, torchvision weights are downloaded into `VISIONBENCH_TORCH_HOME` (default
  `<data_root>/cache/torch`). If
  they can't be obtained the run fails. `--allow-weight-fallback` trains from scratch instead and
  records `weights=scratch`.
- Synthetic demo results are **pipeline sanity checks on synthetic data, not benchmark results**.
  See [REVIEW_PACKET.md](REVIEW_PACKET.md).

## Evaluation and comparison

```powershell
.\.venv\Scripts\visionbench.exe evaluate exp-... --split test --device cpu        # val-selected best.pt
.\.venv\Scripts\visionbench.exe evaluate exp-... --split val --device cuda
.\.venv\Scripts\visionbench.exe evaluate exp-... --split test --checkpoint last   # recorded as 'last'
.\.venv\Scripts\visionbench.exe evaluate exp-... --split test --force --reason "fixed eval bug #12"
.\.venv\Scripts\visionbench.exe latency exp-... --device cpu                      # no split, no images
.\.venv\Scripts\visionbench.exe evaluations list [exp-...]
.\.venv\Scripts\visionbench.exe compare exp-A exp-B --split test --latency-device cpu --note "..."
.\.venv\Scripts\visionbench.exe compare exp-A exp-B --split test --eval-ids ev-... ev-...
```

- `evaluate` writes
  `<data_root>/experiments/<exp>/evaluations/<eval_id>/{metrics.json, predictions.csv,
  confusion_matrix.json, efficiency.json}`.
- Undefined metrics are `null` with a reason, never 0.
- CIs are 95% percentile bootstrap (default 1000 resamples, `--bootstrap`, `--seed`).
- A **second test evaluation** of the same experiment and checkpoint is refused unless
  `--force --reason "..."` is given; the reason is stored and shown in comparison reports.
  The API applies the same rule (409 without force).
- `latency` measures the forward pass only (random input, batch 1 and 32), on one device per
  run, and needs no dataset split. The majority baseline has no latency.
- `compare` needs an existing evaluation of each experiment on the chosen split (latest, or
  pinned with `--eval-ids`). It refuses experiments on different datasets or splits, and
  writes `<data_root>/reports/<report_id>/comparison.{json,md}`. Comparisons on val state
  that val metrics are selection-biased.
- Ranking uses validation metrics only, and test results get no winner label.

## Reproduce the eurosat-v1 benchmark runs (validation only)

Protocol: [benchmarks/eurosat-v1/BENCHMARK_PROTOCOL.md](benchmarks/eurosat-v1/BENCHMARK_PROTOCOL.md). Data source: [DATASETS.md](DATASETS.md).

```powershell
.\.venv\Scripts\visionbench.exe dataset import var\downloads\extracted\EuroSAT_RGB --name eurosat-rgb
.\.venv\Scripts\visionbench.exe dataset split ds-61112928e412 --seed 0 --dhash-threshold 5 --low-info-std 3.0 --low-info-popcount-min 8 --low-info-popcount-max 56 --conflict-policy exclude
.\.venv\Scripts\python.exe scripts\pilot_benchmark.py benchmarks\eurosat-v1
.\.venv\Scripts\python.exe scripts\run_benchmark.py train benchmarks\eurosat-v1
.\.venv\Scripts\python.exe scripts\run_benchmark.py evaluate-val benchmarks\eurosat-v1
.\.venv\Scripts\python.exe scripts\run_benchmark.py latency benchmarks\eurosat-v1
.\.venv\Scripts\visionbench.exe compare <seed-0 exp ids, protocol order> --split val --eval-ids <their val eval ids> --latency-device cuda --note "VALIDATION (selection-biased) — not test results"
.\.venv\Scripts\python.exe scripts\run_benchmark.py summary benchmarks\eurosat-v1
.\.venv\Scripts\python.exe scripts\run_benchmark.py check benchmarks\eurosat-v1
```

## Robustness of the eurosat-v1 runs (validation only)

Protocol: [benchmarks/eurosat-v1/robustness/ROBUSTNESS_PROTOCOL.md](benchmarks/eurosat-v1/robustness/ROBUSTNESS_PROTOCOL.md).
These commands were run for M6:

```powershell
.\.venv\Scripts\visionbench.exe robustness materialize ds-61112928e412 sp-55320e11726c --suite benchmarks/eurosat-v1/robustness/robustness_suite.yaml --split val
.\.venv\Scripts\python.exe scripts\run_robustness.py benchmarks/eurosat-v1 --split val
.\.venv\Scripts\visionbench.exe robustness report --runs benchmarks/eurosat-v1/runs.json --suite benchmarks/eurosat-v1/robustness/robustness_suite.yaml --split val --out benchmarks/eurosat-v1/results/robustness_val.md
.\.venv\Scripts\python.exe scripts\run_benchmark.py check benchmarks\eurosat-v1
```

- Running `materialize` again verifies every corrupted file byte for byte (it never rewrites).
- `run_robustness.py` runs `visionbench robustness evaluate <exp_id> --suite ... --split val
  --clean-eval-id <pinned B1 val evaluation> --device cuda` for each run; it is resumable.
- `--split test` is refused until M8 (`VISIONBENCH_TEST_IMAGES_UNLOCKED`).

## Run tests and lint

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\pytest.exe -q --cov=visionbench
```

To run with the GPU hidden in Windows PowerShell 5.1, use `$env:CUDA_VISIBLE_DEVICES="-1"`.
`$env:CUDA_VISIBLE_DEVICES=""` deletes the variable instead of setting it, so the GPU stays
visible. The pytest summary prints `torch.cuda.is_available()` for every run.

## Run the API and worker

> **Security: local use only.** The API has **no authentication**. `serve` binds to
> `127.0.0.1` by default and refuses any non-loopback host. Do not expose it on a network or
> put it behind a public proxy. Anyone who can reach the port can upload data and start jobs.
>
> **Every POST needs the header `X-VisionBench-Request: 1`** (403 `missing_client_header`
> otherwise). Web pages open in your browser can send requests to `127.0.0.1` too; this header
> is what keeps them out: a browser only sends a custom header to another origin after a CORS
> preflight, which a foreign site fails. The API also rejects a POST whose `Origin` is not in
> `VISIONBENCH_CORS_ORIGINS` (403 `origin_not_allowed`) and any request whose `Host` is not a
> local name (400 `invalid_host`; this stops DNS-rebinding pages, which send their own hostname).
> GET requests need no header. All responses carry `X-Content-Type-Options: nosniff`.

```powershell
.\.venv\Scripts\visionbench.exe serve                 # API on http://127.0.0.1:8000 + one worker
.\.venv\Scripts\visionbench.exe serve --port 8765
.\.venv\Scripts\visionbench.exe serve --no-worker     # API only ...
.\.venv\Scripts\visionbench.exe worker                # ... and the worker in another terminal
```

```text
> visionbench serve --host 0.0.0.0
error: refusing to bind to '0.0.0.0': the API has no authentication; use one of ('127.0.0.1', 'localhost', '::1')
```

Interactive API docs are generated by FastAPI at `http://127.0.0.1:8000/docs` (Swagger UI),
`/redoc` and `/openapi.json`. Endpoints (all under `/api`):

| Method | Path | What it does |
|---|---|---|
| GET | `/api/health`, `/api/ready` | liveness; readiness (data root writable, torch imports) |
| POST | `/api/datasets/uploads` | multipart `file` (ZIP) + `name` → validated, queued import job (202) |
| POST | `/api/datasets/imports` | `{"path","name"}` import from inside `VISIONBENCH_IMPORT_ROOT` (403 if unset) |
| GET | `/api/datasets`, `/api/datasets/{id}` | list; detail with report (no source path) |
| GET | `/api/datasets/{id}/images?relpath=...` | one recorded image (lookup by record, 404 otherwise) |
| POST/GET | `/api/datasets/{id}/splits` | create split (job); list splits |
| GET | `/api/datasets/{id}/splits/{split_id}` | split detail (counts, leakage check) |
| POST | `/api/datasets/{id}/verify` | verify job (`{"full": true}` re-hashes) |
| POST/GET | `/api/experiments` | train (job; body = `TrainConfig`); list |
| GET | `/api/experiments/{exp_id}` | detail: config, env, epochs, artifacts, evaluations, latency |
| POST | `/api/experiments/{exp_id}/evaluations` | evaluate job (`split`, `checkpoint`, `force`, `reason`, ...) |
| POST | `/api/experiments/{exp_id}/latency` | latency job (`device`, `warmup`, `iterations`) |
| GET | `/api/evaluations`, `/api/evaluations/{eval_id}` | list (`?experiment_id=`); detail with metrics |
| POST | `/api/comparisons` | comparison report (synchronous, reads recorded predictions; 201) |
| GET | `/api/comparisons/{report_id}` | stored report |
| GET | `/api/jobs`, `/api/jobs/{job_id}` | jobs with status, progress, result, error |
| POST | `/api/jobs/{job_id}/cancel` | cancel (queued: at once; running: cooperative, then terminate) |
| POST | `/api/experiments/{exp_id}/robustness` | robustness job on every condition of a suite (`{"suite","split"}`; test locked; 202) |
| GET | `/api/robustness`, `/api/robustness/{robustness_id}` | robustness results (`?experiment_id=`); detail with metrics |
| POST | `/api/experiments/{exp_id}/predict` | one image (upload or `{"dataset_id","relpath"}`) → class probabilities (synchronous) |
| POST | `/api/experiments/{exp_id}/gradcam` | the same, plus optional `target_class` → Grad-CAM, overlay PNG (synchronous) |

Example (PowerShell, `curl.exe`):

```powershell
curl.exe -s -H "X-VisionBench-Request: 1" -F "name=my-data" -F "file=@my_data.zip;type=application/zip" http://127.0.0.1:8000/api/datasets/uploads
curl.exe -s http://127.0.0.1:8000/api/jobs/job-YYYYMMDD-HHMMSS-xxxxxx
curl.exe -s -H "X-VisionBench-Request: 1" -H "content-type: application/json" -d '{\"seed\":0}' http://127.0.0.1:8000/api/datasets/ds-xxxxxxxxxxxx/splits
curl.exe -s -X POST -H "X-VisionBench-Request: 1" http://127.0.0.1:8000/api/jobs/job-YYYYMMDD-HHMMSS-xxxxxx/cancel
```

Without the header:

```text
> curl.exe -s -X POST http://127.0.0.1:8000/api/jobs/job-YYYYMMDD-HHMMSS-xxxxxx/cancel
{"error":{"code":"missing_client_header","message":"state-changing requests need the header 'X-VisionBench-Request: 1'"}}
```

- Long operations return `202 {"job": {...}}`; poll `GET /api/jobs/{id}` for `status`
  (`queued|running|completed|failed|cancelled`), `progress` and `result`.
- Only one job runs at a time. More than `VISIONBENCH_MAX_QUEUED_JOBS` queued jobs → 429.
- A ZIP must contain class folders at its root or inside one top folder. Unsafe archives
  (traversal, absolute paths, symlinks, encryption, case collisions, zip bombs, too many
  members) are rejected with `422 archive_<code>` before anything is queued.
- One worker per data root: it holds an OS file lock on `db/worker.lock` while it runs, so a
  second worker exits at once with `another worker is running on this data root`. The lock is
  released by the OS when the worker dies, however it dies.
- If the worker is killed, the next worker start (holding the lock) marks every job still
  `running` and its experiment `failed` (`interrupted: ...`) at once. Jobs are not resumed
  automatically.
- All errors use `{"error": {"code": ..., "message": ...}}`. Responses never contain absolute
  paths.

### Prediction and Grad-CAM (one image, synchronous)

These commands were run against the real eurosat-v1 seed-0 models (CPU inference). The image
must be a recorded dataset image (`dataset_id` + `relpath`) or an upload (`file`). TEST-split
images are refused until M8.

```powershell
# a recorded VAL image
curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/exp-20261027-204957-d42310/predict -H "X-VisionBench-Request: 1" -H "Content-Type: application/json" -d '{\"dataset_id\":\"ds-61112928e412\",\"relpath\":\"AnnualCrop/AnnualCrop_264.jpg\"}'
# an upload; gradcam with an explicit target class
curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/exp-20261027-204957-d42310/predict -H "X-VisionBench-Request: 1" -F "file=@blue.png"
curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/exp-20261027-204957-d42310/gradcam -H "X-VisionBench-Request: 1" -F "file=@blue.png" -F "target_class=SeaLake"
```

A TEST image:

```text
> curl.exe -s -X POST http://127.0.0.1:8000/api/experiments/exp-20261027-214915-e0d372/predict -H "X-VisionBench-Request: 1" -H "Content-Type: application/json" -d '{\"dataset_id\":\"ds-61112928e412\",\"relpath\":\"AnnualCrop/AnnualCrop_100.jpg\"}'
{"error":{"code":"test_split_locked","message":"this image is in the TEST split of sp-55320e11726c; test images are locked until the final test evaluation (VISIONBENCH_TEST_IMAGES_UNLOCKED)"}}
```

- `predict` returns the predicted class, all class probabilities (sorted), the checkpoint name
  and sha256, the input's original size, the preprocessing (recorded val transforms), timings
  (`forward`, `model_load` on a cache miss, `total`), `warnings` and `notes`.
- `gradcam` adds the target class and its probability, the CAM's feature-map resolution and
  values (0-1), the geometry (`unseen_fraction`, `seen_box_in_original`), an overlay PNG as a
  `data:` URI, and a fixed limitations text. Grad-CAM is coarse, not causal, and does not show
  that a prediction is correct (EXPERIMENT_PROTOCOL.md section 13).
- Errors: `403 test_split_locked`, `413 upload_too_large` / `image_too_large`,
  `415 unsupported_image`, `422 unknown_class` / `explanation_not_supported` (majority
  baseline), `429 inference_busy`.
- Demo (VALIDATION images, not results): `python scripts/demo_inference.py benchmarks/eurosat-v1
  --out var/demos/m5` against a running `serve`.

## CLI

```powershell
.\.venv\Scripts\visionbench.exe env
```

## Docs

[ARCHITECTURE.md](ARCHITECTURE.md) · [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) ·
[PROJECT_STATUS.md](PROJECT_STATUS.md) · [CHANGELOG.md](CHANGELOG.md)
