# Review Packet — VB-M4

**Task:** VB-M4: FastAPI backend, single-worker job runner, secure dataset upload
**Date:** 2026-10-27
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130; torchvision
0.29.1+cu130; FastAPI 0.162.0; Starlette 1.10.0; python-multipart 0.0.32 (new); SQLAlchemy
2.1.6; SQLite 3.53.1; NVIDIA GeForce RTX 3050 6GB Laptop GPU.

**Objective:** expose the library through an HTTP API in which no request runs ML work, with:
- a single-worker job runner (one job at a time) with progress, cancellation and crash
  recovery;
- a hardened ZIP upload path.

Step 0 first closed the M3 follow-ups (latency independent of splits, pinned evaluations,
forced test re-evaluation).

Every output below came from commands executed in this session. Scratchpad paths are shortened
to `<scratchpad>`.

> **All model numbers in this packet are a pipeline demo on synthetic data — not benchmark
> results.**

---

## 1. Commits (local only; no remote, nothing pushed)

```
3c1955e M4 docs: API layer, job lifecycle, recovery, upload pipeline, protocol
f349f5d serve: say which pid is printed for the worker process
5d56963 M4: FastAPI endpoints under /api, serve command, API tests
3a54115 M4: hardened ZIP dataset upload (validate all, extract, import, clean up)
51cad45 M4: job store (schema v4, WAL) and single-worker runner with recovery
7936bda M4 step 0: split-independent latency, pinned evals, forced test re-evaluation
a70fe97 M3 review packet
```

The stages ran in the specified order: Step 0 → job store + worker → upload security →
endpoints → docs. This packet is the final commit. `f349f5d` is a one-line message fix found
during the live demo (§6).

## 2. Step 0: M3 follow-ups (commit `7936bda`)

| Item | Result | Tests (`tests/evaluation/test_evaluate.py`) |
|---|---|---|
| 0a | `visionbench latency <exp> --device cpu\|cuda` (`evaluation/latency.py`) loads the verified checkpoint and times random input. It reads no split and no image, and writes `latency/<lat_id>/latency.json` plus a `latency_measurements` row. `evaluate` no longer measures latency (`--no-latency` removed). `compare --latency-device` reports only measurements from that device. | `test_latency_command_touches_no_split` (the image loader and split selectors are patched to fail), `test_baseline_efficiency_and_latency_not_applicable` |
| 0b | `compare --eval-ids` pins one evaluation per experiment. Each pinned id must be on the split, use the val-selected checkpoint, and belong to a compared experiment. | `test_compare_pins_eval_ids_and_val_bias_note` |
| 0c | A repeated TEST evaluation of the same experiment + checkpoint is refused unless `--force --reason "..."`. `force` without a reason is an error. `forced`/`reason` are stored in `evaluations`, `metrics.json` `test_use`, and compare reports (`forced_test_reevaluation` flag). | `test_repeat_test_evaluation_requires_force_and_reason`, `test_schema_upgrade_from_v2_adds_forced_columns` |
| 0d | `compare --split val` carries `selection_bias_note` in JSON and in Markdown. | `test_compare_pins_eval_ids_and_val_bias_note` |
| 0e | `--checkpoint last` on a majority baseline is an explicit error. | `test_baseline_last_checkpoint_is_an_error` |

The API enforces the same rule through the same function (`evaluation.data.check_test_reuse`).
It is checked before queueing (409) and again inside the job. See §5 and the demo in §6.

## 3. Files changed since the M3 packet (`git diff --stat a70fe97 3c1955e`)

```
 ARCHITECTURE.md                           | 195 ++++++++++++++---
 CHANGELOG.md                              |  41 ++++
 EXPERIMENT_PROTOCOL.md                    |  49 ++++-
 IMPLEMENTATION_PLAN.md                    |  15 +-
 PROJECT_STATUS.md                         |  62 +++---
 README.md                                 | 113 ++++++++--
 pyproject.toml                            |   1 +
 requirements/lock-dev-cu130.txt           |   3 +-
 src/visionbench/api/deps.py               |  58 +++++
 src/visionbench/api/errors.py             |  85 ++++++++
 src/visionbench/api/main.py               |  66 +++++-
 src/visionbench/api/routes/__init__.py    |   1 +
 src/visionbench/api/routes/datasets.py    | 222 +++++++++++++++++++
 src/visionbench/api/routes/experiments.py | 224 +++++++++++++++++++
 src/visionbench/api/schemas.py            | 102 +++++++++
 src/visionbench/api/upload.py             | 146 +++++++++++++
 src/visionbench/cli.py                    | 127 ++++++++++-
 src/visionbench/core/config.py            |  29 ++-
 src/visionbench/core/redact.py            |  23 ++
 src/visionbench/data/archive.py           | 333 ++++++++++++++++++++++++++++
 src/visionbench/data/store.py             |  12 +-
 src/visionbench/evaluation/compare.py     | 167 +++++++++++---
 src/visionbench/evaluation/data.py        | 125 +++++++++--
 src/visionbench/evaluation/evaluate.py    |  50 ++---
 src/visionbench/evaluation/latency.py     | 105 +++++++++
 src/visionbench/jobs/__init__.py          |   1 +
 src/visionbench/jobs/handlers.py          | 192 ++++++++++++++++
 src/visionbench/jobs/params.py            |  91 ++++++++
 src/visionbench/jobs/store.py             | 243 +++++++++++++++++++++
 src/visionbench/jobs/worker.py            | 352 ++++++++++++++++++++++++++++++
 src/visionbench/tracking/store.py         | 150 +++++++++++--
 src/visionbench/training/loop.py          |  32 ++-
 src/visionbench/training/runner.py        |  16 +-
 tests/api/test_api_flow.py                | 207 ++++++++++++++++++
 tests/api/test_api_upload.py              | 121 ++++++++++
 tests/api/test_api_worker.py              |  87 ++++++++
 tests/data/test_archive.py                | 280 ++++++++++++++++++++++++
 tests/evaluation/test_evaluate.py         | 166 +++++++++++---
 tests/jobs/test_job_store.py              | 233 ++++++++++++++++++++
 tests/jobs/test_worker_process.py         | 131 +++++++++++
 tests/vb_api.py                           |  90 ++++++++
 tests/vb_synth.py                         |  36 +++
 tests/vb_worker.py                        |  77 +++++++
 43 files changed, 4615 insertions(+), 244 deletions(-)
```

(PROJECT_STATUS.md changed again by one line in the packet commit: the final health timings.)

## 4. Lint and tests at `3c1955e` (verbatim)

```
> .\.venv\Scripts\ruff.exe check .
All checks passed!
> .\.venv\Scripts\ruff.exe format --check .
90 files already formatted
```

**Normal run** (`.\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 29%]
....................s................................................... [ 59%]
........................................................................ [ 88%]
...........................                                              [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=16, max=12.7 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.15 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 6.51 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________
[... per-file table; M4 rows below ...]
TOTAL                                        4157    292    938    115    91%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
242 passed, 1 skipped, 1 warning in 258.11s (0:04:18)
```

**GPU hidden** (`$env:CUDA_VISIBLE_DEVICES="-1"; .\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 29%]
....................s................................................... [ 59%]
........................................................................ [ 88%]
...........................                                              [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=21, max=19.8 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.04 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 10.31 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________
[... per-file table ...]
TOTAL                                        4157    292    938    115    91%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
242 passed, 1 skipped, 1 warning in 356.68s (0:05:56)
```

Coverage rows for M4 code (normal run; the GPU-hidden run has the same TOTAL):

```
Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
src\visionbench\api\deps.py                    25      0      0      0   100%
src\visionbench\api\errors.py                  41      2     14      2    93%
src\visionbench\api\main.py                    76      1      8      0    99%
src\visionbench\api\routes\datasets.py        105      2     14      3    96%
src\visionbench\api\routes\experiments.py     100      2     14      3    96%
src\visionbench\api\schemas.py                 71      0      0      0   100%
src\visionbench\api\upload.py                  95     11     30      7    86%
src\visionbench\cli.py                        353     46     60      6    85%
src\visionbench\core\redact.py                 14      0      2      0   100%
src\visionbench\data\archive.py               207     12     76      7    93%
src\visionbench\evaluation\compare.py         215     13     90     17    90%
src\visionbench\evaluation\data.py            116      7     28      7    90%
src\visionbench\evaluation\latency.py          45      1      4      1    96%
src\visionbench\jobs\handlers.py               74      3      6      2    94%
src\visionbench\jobs\params.py                 55      2      4      2    93%
src\visionbench\jobs\store.py                 121      1     22      0    99%
src\visionbench\jobs\worker.py                253    119     72     10    51%
src\visionbench\tracking\store.py             233      5     40      3    96%
```

`jobs/worker.py` shows 51% because `run_job`, `_child_main`, `_watch_parent` and the
worker loop run inside `visionbench worker` subprocesses, and coverage does not measure
subprocesses. Those paths are exercised by `tests/jobs/test_worker_process.py` and
`tests/api/test_api_worker.py`, which start a real worker. Coverage went from 95% (M3) to 91%
overall for the same reason.

## 5. Requirement → evidence

### Job store and worker

| Requirement | Evidence |
|---|---|
| jobs table, WAL + busy_timeout | `tracking/store.py::open_engine`, `jobs/store.py::Job`; `test_concurrent_enqueue_and_worker_updates` (API-style writers and a worker writing concurrently) |
| queue limit → 429 | `test_queue_limit` (store), `test_queue_full_429_and_upload_removed` (API) |
| one job at a time, atomic claim | `test_atomic_claim_under_contention` (8 threads with separate engines race for one queued job; exactly one wins); `test_jobs_run_one_at_a_time_with_progress` (real worker: running count ≤ 1, the second job starts after the first finishes) |
| child process, heartbeat, training progress | `test_jobs_run_one_at_a_time_with_progress`, `test_worker_runs_api_jobs_with_progress_and_health_stays_fast`; demo §6 step 3 |
| cooperative cancel | `test_cooperative_cancel_keeps_partial_history` (store), `test_cancel_via_api_keeps_partial_history` (API); demo §6 step 8 |
| terminate after timeout | `test_cancel_timeout_terminates_child`: a latency job (10 000 000 iterations; latency has no cancel hook) is cancelled with `VISIONBENCH_CANCEL_TIMEOUT_SECONDS=1`; the child is terminated, the job is `cancelled` with `cancelled: child terminated after 1 s (cooperative cancel not honoured)`, and the child pid is dead. For training jobs the same path also marks the experiment `cancelled`, but no test covers that |
| crash recovery | `test_hard_kill_then_recovery_marks_interrupted` (real worker killed during training: the child exits, the next worker marks the job and the experiment `failed: interrupted`); `test_recover_stale_and_dead_jobs`; `test_recover_orphaned_experiments`; demo §6 step 9 |
| evaluations through the same `evaluate_experiment` with force/reason | `jobs/handlers.py::handle_evaluate` calls `evaluate_experiment(..., force, reason)`; the API flow test checks the 409, then 422 (force without reason), then 202 + recorded reason |
| health stays fast while training | `[measured] GET /api/health while training: n=16, max=12.7 ms` (normal), `n=21, max=19.8 ms` (GPU hidden); assertion < 1 s |

### Upload security

Every rule has a passing test:

| Rule | Test |
|---|---|
| byte limit 413 while streaming | `test_oversized_upload_413_while_streaming[False]` (Content-Length) and `[True]` (chunked, no Content-Length) |
| magic check | `test_non_zip_rejected` (API), `test_magic_bytes_and_corrupt_zip` |
| absolute / drive / UNC / `..` / `.`/empty / invalid chars / reserved / trailing dot or space | `test_name_rules` (17 cases), `test_unsafe_archive_rejected_before_queueing` (3 via API) |
| backslashes | `test_backslash_names_rejected` (3 cases) |
| symlink member | `test_symlink_member` |
| encrypted member | `test_encrypted_member` |
| member count | `test_member_count_limit` |
| total uncompressed size | `test_total_uncompressed_limit` |
| compression ratio, per member and overall | `test_member_compression_ratio`, `test_overall_compression_ratio` |
| case-insensitive collisions (NFC + casefold) | `test_case_insensitive_collisions` |
| layouts → 422 `bad_layout` | `test_layouts` |
| extraction confined to staging | `test_extraction_escape_guard` |
| staging and upload cleanup in all cases | `test_import_archive_success_and_cleanup`, `test_import_archive_cleans_up_after_rejection_and_failed_import`, `test_cancelling_queued_import_removes_upload`, and every API rejection test asserts `tmp/` is empty |
| `import_root` confinement | `test_path_import_confinement`, `test_path_imports` (403 disabled, 403 outside, 202 inside, no root in bodies) |
| bad multipart / 415 / missing file / missing name | `test_bad_requests` |

### API

| Requirement | Evidence |
|---|---|
| 23 operations under `/api` | `app.openapi()` listing (README endpoint table) |
| image endpoint is lookup-only | `test_full_api_flow_and_path_hygiene`: a recorded image → 200 `image/png`, bytes equal to the file. Each of these → 404 `image_not_found`: `../dataset.json`, `circle/../../dataset.json`, `..%2Fdataset.json`, `%2e%2e%2fdataset.json`, the absolute Windows path, the absolute POSIX path, `/etc/passwd`, a backslash traversal, the wrong case of a class folder, and `circle/unrecorded.png` (exists on disk, not in `files.json`; verify flags it as `unexpected file not in files.json`). An invalid dataset id → 422 |
| strict id regexes → 422 | flow test: `ds-nothexvalue!`, `not-a-job`, `exp-bad` → 422; well-formed unknown ids → 404 |
| CORS restricted | `test_cors_only_configured_origins` |
| consistent error schema | every error assertion reads `error.code` |
| no absolute paths in responses | flow test `api.absolute_root_leaks() == []` over more than 40 bodies; demo: 550 responses, 0 containing the data root |
| no ML in requests | every ML endpoint returns 202 + job; the flow test runs the jobs with `run_jobs_inline` (the same `execute_job` the worker child calls) |
| test-use via API | flow test: 409 `test_reevaluation_requires_force`; force without reason → 422; force + reason → 202, recorded (`reason == "api check"`, `test_evaluation_number == 2`); compare row lists `forced_test_reasons == ["api check"]` |
| `serve` on loopback only | `visionbench serve --host 0.0.0.0` → exit 2 with `error: refusing to bind to '0.0.0.0': the API has no authentication; use one of ('127.0.0.1', 'localhost', '::1')` |

## 6. Live demo: `visionbench serve` driven over HTTP

**Pipeline demo on synthetic data — not benchmark results.**

`<scratchpad>/demo/m4_demo.py` (not committed) did the following:
- deleted and recreated `<scratchpad>/demo/work/var`;
- started `python -m visionbench.cli serve --port 8765` (API + one worker) as a subprocess;
- built a synthetic shapes ZIP (`tests/vb_synth.shapes_zip`, 3 classes × 40 PNGs, 64 px);
- drove every step with `httpx` and printed each request and response.

To shorten this listing, the `params` field of job bodies is replaced by a marker string,
because it repeats the request (for training, the full resolved `TrainConfig`). This was done
with `<scratchpad>/demo/compress.py`, which re-renders all other fields unchanged. Everything
else is the demo's output, unchanged.

The same API was also exercised with `curl.exe`: upload → poll job → split, using the README
commands on port 8766 against a separate data root. All three returned the expected JSON.

**Problems the demo uncovered and how they were handled:**
- **Wrong pid.** The first demo run killed the pid printed by `serve`. With a Windows venv that
  pid is the `python.exe` launcher, not the worker. Recovery then did happen, but the demo had
  only waited 2 s for the restarted worker. The demo now reads the real worker pid from
  `db/worker.lock` and the job row, kills that, and polls until recovery. The serve message
  was reworded in `f349f5d`.
- **Training length.** The CUDA training stopped early at epoch 10 of 25 (early stopping,
  patience 5). That is expected behaviour, not a fault.

```text
=== VisionBench M4 live demo — pipeline demo on synthetic data — not benchmark results ===
data root: <scratchpad>/demo/work/var (fresh); API: http://127.0.0.1:8765
[t+  12.9s] `visionbench serve --port 8765` up (serve pid 2492, worker pid 2296); log serve1.log

[t+  12.9s] >>> GET /api/health
<<< 200
{
 "status": "ok",
 "version": "0.0.1"
}

### 1. upload a synthetic ZIP (multipart, streamed)
shapes_demo.zip: 940615 bytes, 3 classes x 40 synthetic PNGs

[t+  13.3s] >>> POST /api/datasets/uploads  (multipart: name=shapes-demo, file=shapes_demo.zip)
<<< 202
{
 "job": {
  "id": "job-20261027-153953-d19406",
  "kind": "dataset_import",
  "status": "queued",
  "params": "[7 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:39:53+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+  37.8s] >>> GET /api/jobs/job-20261027-153953-d19406
<<< 200
{
 "id": "job-20261027-153953-d19406",
 "kind": "dataset_import",
 "status": "completed",
 "params": "[7 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "class_counts": {
   "circle": 40,
   "square": 40,
   "triangle": 40
  },
  "dataset_id": "ds-274cbece8d45",
  "issue_counts": {
   "file_outside_class_folder": 0,
   "hidden_or_system_file": 0,
   "invalid_class_name": 0,
   "invalid_file_name": 0,
   "nested_directory": 0,
   "path_collision": 0,
   "symlink": 0,
   "too_large_pixels": 0,
   "unreadable_or_corrupt": 0,
   "unsupported_extension": 0,
   "zero_byte": 0
  },
  "name": "shapes-demo",
  "total_images": 120
 },
 "error": null,
 "progress": {
  "phase": "importing",
  "source": "upload"
 },
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:40:15+00:00",
 "created_at": "2026-10-27T15:39:53+00:00",
 "started_at": "2026-10-27T15:40:03+00:00",
 "finished_at": "2026-10-27T15:40:17+00:00"
}

[t+  37.9s] >>> GET /api/datasets
<<< 200
[
 {
  "id": "ds-274cbece8d45",
  "name": "shapes-demo",
  "created_at": "2026-10-27T15:40:17+00:00",
  "classes": [
   "circle",
   "square",
   "triangle"
  ],
  "class_counts": {
   "circle": 40,
   "square": 40,
   "triangle": 40
  },
  "total_images": 120,
  "fingerprint": "f77312bc79901e81ecb9a427409d085cc4c8c7e4dc876f0cdeb6a40754959789"
 }
]

### 2. split

[t+  37.9s] >>> POST /api/datasets/ds-274cbece8d45/splits
{
 "seed": 0,
 "fractions": [
  0.6,
  0.2,
  0.2
 ]
}
<<< 202
{
 "job": {
  "id": "job-20261027-154017-ac3946",
  "kind": "dataset_split",
  "status": "queued",
  "params": "[15 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:40:17+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+  52.0s] >>> GET /api/jobs/job-20261027-154017-ac3946
<<< 200
{
 "id": "job-20261027-154017-ac3946",
 "kind": "dataset_split",
 "status": "completed",
 "params": "[15 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "already_existed": false,
  "dataset_id": "ds-274cbece8d45",
  "leakage_check": {
   "groups_checked": 120,
   "groups_in_multiple_splits": 0,
   "sha256_checked": 120,
   "sha256_in_multiple_splits": 0
  },
  "manifest_sha256": "bc6b5241ed065394d3abe17bb6b18fa38fae68c01ea3a3488f0d3148fc81f546",
  "split_id": "sp-6dbfe703a2b8"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:40:31+00:00",
 "created_at": "2026-10-27T15:40:17+00:00",
 "started_at": "2026-10-27T15:40:19+00:00",
 "finished_at": "2026-10-27T15:40:31+00:00"
}

### 3. train small_cnn on CUDA (job progress polled while it runs) — pipeline demo on synthetic data — not benchmark results

[t+  52.0s] >>> POST /api/experiments
{
 "dataset_id": "ds-274cbece8d45",
 "split_id": "sp-6dbfe703a2b8",
 "model": "small_cnn",
 "input_size": 64,
 "epochs": 25,
 "batch_size": 16,
 "lr": 0.01,
 "device": "cuda",
 "num_workers": 0,
 "seed": 0
}
<<< 202
{
 "job": {
  "id": "job-20261027-154031-01576c",
  "kind": "train",
  "status": "queued",
  "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:40:31+00:00",
  "started_at": null,
  "finished_at": null
 }
}
[t+  65.4s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 0, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "max_steps": null, "phase": "training", "step": 0}
[t+  67.7s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 0, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "max_steps": null, "phase": "training", "step": 1}
[t+  68.2s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 1, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 2.0040274461110434, "val_accuracy": 0.3333333333333333, "val_balanced_accuracy": 0.3333333333333333, "val_loss": 6.894384066263835}, "max_steps": null, "phase": "training", "step": 5}
[t+  68.5s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 2, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.1976794136895075, "val_accuracy": 0.375, "val_balanced_accuracy": 0.375, "val_loss": 6.933880885442098}, "max_steps": null, "phase": "training", "step": 10}
[t+  68.8s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 3, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.2634715106752183, "val_accuracy": 0.4166666666666667, "val_balanced_accuracy": 0.4166666666666667, "val_loss": 4.037823498249054}, "max_steps": null, "phase": "training", "step": 15}
[t+  69.1s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 4, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.1219837798012628, "val_accuracy": 0.25, "val_balanced_accuracy": 0.25, "val_loss": 1.6077161232630413}, "max_steps": null, "phase": "training", "step": 20}
[t+  69.3s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 5, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.1264962090386286, "val_accuracy": 0.2916666666666667, "val_balanced_accuracy": 0.2916666666666667, "val_loss": 1.3522382378578186}, "max_steps": null, "phase": "training", "step": 25}
[t+  69.6s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 6, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.0582421355777316, "val_accuracy": 0.20833333333333334, "val_balanced_accuracy": 0.20833333333333334, "val_loss": 1.4396714369455974}, "max_steps": null, "phase": "training", "step": 30}
[t+  69.9s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 6, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.0582421355777316, "val_accuracy": 0.20833333333333334, "val_balanced_accuracy": 0.20833333333333334, "val_loss": 1.4396714369455974}, "max_steps": null, "phase": "training", "step": 32}
[t+  70.2s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 8, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.0065867900848389, "val_accuracy": 0.20833333333333334, "val_balanced_accuracy": 0.20833333333333334, "val_loss": 1.5818219582239788}, "max_steps": null, "phase": "training", "step": 40}
[t+  70.5s] GET /api/jobs/job-20261027-154031-01576c -> status=running progress={"epoch": 9, "epochs": 25, "experiment_id": "exp-20261027-154045-b5862d", "latest": {"train_loss": 1.0147285328971014, "val_accuracy": 0.20833333333333334, "val_balanced_accuracy": 0.20833333333333334, "val_loss": 1.8836629390716553}, "max_steps": null, "phase": "training", "step": 45}

[t+  71.1s] >>> GET /api/jobs/job-20261027-154031-01576c
<<< 200
{
 "id": "job-20261027-154031-01576c",
 "kind": "train",
 "status": "completed",
 "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "best_epoch": 5,
  "best_value": 1.3522382378578186,
  "experiment_id": "exp-20261027-154045-b5862d",
  "status": "completed",
  "stopped_reason": "early_stopping"
 },
 "error": null,
 "progress": {
  "epoch": 10,
  "epochs": 25,
  "experiment_id": "exp-20261027-154045-b5862d",
  "latest": {
   "train_loss": 1.0120509531762865,
   "val_accuracy": 0.25,
   "val_balanced_accuracy": 0.25,
   "val_loss": 1.5093722343444824
  },
  "max_steps": null,
  "phase": "finished",
  "step": 50
 },
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:40:49+00:00",
 "created_at": "2026-10-27T15:40:31+00:00",
 "started_at": "2026-10-27T15:40:32+00:00",
 "finished_at": "2026-10-27T15:40:50+00:00"
}
[t+  71.1s] GET /api/health during training: n=69, max=4.5 ms

[t+  71.1s] >>> POST /api/experiments
{
 "dataset_id": "ds-274cbece8d45",
 "split_id": "sp-6dbfe703a2b8",
 "model": "majority_baseline",
 "input_size": 64,
 "epochs": 25,
 "batch_size": 16,
 "lr": 0.01,
 "device": "cuda",
 "num_workers": 0,
 "seed": 0
}
<<< 202
{
 "job": {
  "id": "job-20261027-154051-e1a5b7",
  "kind": "train",
  "status": "queued",
  "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:40:51+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+  85.3s] >>> GET /api/jobs/job-20261027-154051-e1a5b7
<<< 200
{
 "id": "job-20261027-154051-e1a5b7",
 "kind": "train",
 "status": "completed",
 "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "best_epoch": 1,
  "best_value": 1.09861228866811,
  "experiment_id": "exp-20261027-154104-62cc94",
  "status": "completed",
  "stopped_reason": "baseline_fit"
 },
 "error": null,
 "progress": {
  "epoch": 1,
  "epochs": 25,
  "experiment_id": "exp-20261027-154104-62cc94",
  "latest": {
   "train_loss": 1.0986122886681098,
   "val_accuracy": 0.3333333333333333,
   "val_balanced_accuracy": 0.3333333333333333,
   "val_loss": 1.09861228866811
  },
  "max_steps": null,
  "phase": "finished",
  "step": 0
 },
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:41:04+00:00",
 "created_at": "2026-10-27T15:40:51+00:00",
 "started_at": "2026-10-27T15:40:52+00:00",
 "finished_at": "2026-10-27T15:41:04+00:00"
}

GET /api/experiments/exp-20261027-154045-b5862d -> status=completed device=cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU) epochs=10 run_dir=experiments/exp-20261027-154045-b5862d

### 4. validation evaluation — pipeline demo on synthetic data — not benchmark results

[t+  85.4s] >>> POST /api/experiments/exp-20261027-154045-b5862d/evaluations
{
 "split": "val",
 "device": "cuda",
 "bootstrap": 200
}
<<< 202
{
 "job": {
  "id": "job-20261027-154105-2f9090",
  "kind": "evaluate",
  "status": "queued",
  "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:41:05+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+ 100.1s] >>> GET /api/jobs/job-20261027-154105-2f9090
<<< 200
{
 "id": "job-20261027-154105-2f9090",
 "kind": "evaluate",
 "status": "completed",
 "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "checkpoint": "best",
  "eval_id": "ev-20261027-154119-fa73d9",
  "experiment_id": "exp-20261027-154045-b5862d",
  "forced": false,
  "split": "val"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:41:18+00:00",
 "created_at": "2026-10-27T15:41:05+00:00",
 "started_at": "2026-10-27T15:41:06+00:00",
 "finished_at": "2026-10-27T15:41:19+00:00"
}

[t+ 100.1s] >>> POST /api/experiments/exp-20261027-154104-62cc94/evaluations
{
 "split": "val",
 "device": "cuda",
 "bootstrap": 200
}
<<< 202
{
 "job": {
  "id": "job-20261027-154119-e5f56c",
  "kind": "evaluate",
  "status": "queued",
  "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:41:19+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+ 114.3s] >>> GET /api/jobs/job-20261027-154119-e5f56c
<<< 200
{
 "id": "job-20261027-154119-e5f56c",
 "kind": "evaluate",
 "status": "completed",
 "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "checkpoint": "baseline",
  "eval_id": "ev-20261027-154133-f52b52",
  "experiment_id": "exp-20261027-154104-62cc94",
  "forced": false,
  "split": "val"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:41:33+00:00",
 "created_at": "2026-10-27T15:41:19+00:00",
 "started_at": "2026-10-27T15:41:21+00:00",
 "finished_at": "2026-10-27T15:41:33+00:00"
}

### 5. latency (cpu and cuda) — pipeline demo on synthetic data — not benchmark results

[t+ 114.3s] >>> POST /api/experiments/exp-20261027-154045-b5862d/latency
{
 "device": "cpu",
 "warmup": 5,
 "iterations": 30
}
<<< 202
{
 "job": {
  "id": "job-20261027-154134-736b7c",
  "kind": "latency",
  "status": "queued",
  "params": "[6 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:41:34+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+ 130.5s] >>> GET /api/jobs/job-20261027-154134-736b7c
<<< 200
{
 "id": "job-20261027-154134-736b7c",
 "kind": "latency",
 "status": "completed",
 "params": "[6 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "device": "cpu",
  "experiment_id": "exp-20261027-154045-b5862d",
  "latency_id": "lat-20261027-154149-a2af91"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:41:49+00:00",
 "created_at": "2026-10-27T15:41:34+00:00",
 "started_at": "2026-10-27T15:41:35+00:00",
 "finished_at": "2026-10-27T15:41:49+00:00"
}
latency job result: {"device": "cpu", "experiment_id": "exp-20261027-154045-b5862d", "latency_id": "lat-20261027-154149-a2af91"}

[t+ 130.6s] >>> POST /api/experiments/exp-20261027-154045-b5862d/latency
{
 "device": "cuda",
 "warmup": 5,
 "iterations": 30
}
<<< 202
{
 "job": {
  "id": "job-20261027-154150-f4ac0c",
  "kind": "latency",
  "status": "queued",
  "params": "[6 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:41:50+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+ 144.8s] >>> GET /api/jobs/job-20261027-154150-f4ac0c
<<< 200
{
 "id": "job-20261027-154150-f4ac0c",
 "kind": "latency",
 "status": "completed",
 "params": "[6 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "device": "cuda",
  "experiment_id": "exp-20261027-154045-b5862d",
  "latency_id": "lat-20261027-154204-388e89"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:42:03+00:00",
 "created_at": "2026-10-27T15:41:50+00:00",
 "started_at": "2026-10-27T15:41:51+00:00",
 "finished_at": "2026-10-27T15:42:04+00:00"
}
latency job result: {"device": "cuda", "experiment_id": "exp-20261027-154045-b5862d", "latency_id": "lat-20261027-154204-388e89"}

### 6. compare on val — pipeline demo on synthetic data — not benchmark results

[t+ 144.9s] >>> POST /api/comparisons
{
 "experiment_ids": [
  "exp-20261027-154045-b5862d",
  "exp-20261027-154104-62cc94"
 ],
 "split": "val",
 "latency_device": "cuda",
 "bootstrap": 200,
 "note": "pipeline demo on synthetic data \u2014 not benchmark results"
}
<<< 201
{
 "format_version": 2,
 "report_id": "rp-20261027-154204-6d7c7f",
 "created_at": "2026-10-27T15:42:04+00:00",
 "visionbench_version": "0.0.1",
 "note": "pipeline demo on synthetic data \u2014 not benchmark results",
 "split": "val",
 "dataset_id": "ds-274cbece8d45",
 "dataset_fingerprint": "f77312bc79901e81ecb9a427409d085cc4c8c7e4dc876f0cdeb6a40754959789",
 "split_id": "sp-6dbfe703a2b8",
 "n_samples": 24,
 "class_names": [
  "circle",
  "square",
  "triangle"
 ],
 "reference_experiment": "exp-20261027-154104-62cc94",
 "latency_device": "cuda",
 "selection_bias_note": "validation metrics are selection-biased: the same validation data chose each model's checkpoint (early stopping / best epoch), so they overestimate performance on new data",
 "paired_bootstrap": {
  "resamples": 200,
  "seed": 0,
  "difference": "experiment minus reference"
 },
 "rows": [
  {
   "experiment_id": "exp-20261027-154045-b5862d",
   "model": "small_cnn",
   "weights_source": "scratch",
   "input_size": 64,
   "params_total": 389667,
   "checkpoint_size_bytes": 1569717,
   "latency": {
    "applicable": true,
    "latency_id": "lat-20261027-154204-388e89",
    "device": "cuda",
    "device_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
    "torch_threads": 8,
    "input_size": 64,
    "batch_sizes": {
     "1": {
      "median_ms": 1.1231999997107778,
      "median_ms_per_image": 1.1231999997107778,
      "p95_ms": 1.3467250000758213
     },
     "32": {
      "median_ms": 3.0540499997187,
      "median_ms_per_image": 0.09543906249120937,
      "p95_ms": 3.2095049996769376
     }
    }
   },
   "latency_bs1_median_ms": 1.1231999997107778,
   "latency_device": "cuda",
   "latency_note": null,
   "train_seconds": 5.353469100000439,
   "best_epoch": 5,
   "selection_monitor": "val_loss",
   "selection_value": 1.3522382378578186,
   "val_balanced_accuracy_at_selected_epoch": 0.2916666666666667,
   "checkpoint": "best",
   "eval_id": "ev-20261027-154119-fa73d9",
   "eval_pinned": false,
   "test_evaluations": 0,
   "forced_test_reasons": [],
   "bootstrap": {
    "resamples": 200,
    "seed": 0
   },
   "val": {
    "accuracy": 0.2916666666666667,
    "accuracy_ci": {
     "high": 0.5,
     "low": 0.125,
     "resamples_defined": 200
    },
    "balanced_accuracy": 0.2916666666666667,
    "balanced_accuracy_ci": {
     "high": 0.4907855339105339,
     "low": 0.14254960317460322,
     "resamples_defined": 200
    },
    "macro_f1": 0.2800402212166918,
    "macro_f1_ci": {
     "high": 0.47416925180083075,
     "low": 0.12376390322210759,
     "resamples_defined": 200
    }
   },
   "paired_difference_vs_reference": {
    "resamples": 200,
    "seed": 0,
    "level": 0.95,
    "method": "paired percentile bootstrap (shared indices)",
    "metrics": {
     "balanced_accuracy": {
      "point": -0.04166666666666663,
      "low": -0.1907837301587301,
      "high": 0.15745220057720058,
      "resamples_defined": 200
     },
     "macro_f1": {
      "point": 0.11337355455002515,
      "low": -0.06009676597911892,
      "high": 0.28396376692622827,
      "resamples_defined": 200
     }
    }
   }
  },
  {
   "experiment_id": "exp-20261027-154104-62cc94",
   "model": "majority_baseline",
   "weights_source": "none",
   "input_size": null,
   "params_total": null,
   "checkpoint_size_bytes": 458,
   "latency": {
    "applicable": false,
    "reason": "majority_baseline has no forward pass"
   },
   "latency_bs1_median_ms": null,
   "latency_device": null,
   "latency_note": "majority_baseline has no forward pass",
   "train_seconds": 0.14664920000086568,
   "best_epoch": 1,
   "selection_monitor": "val_loss",
   "selection_value": 1.09861228866811,
   "val_balanced_accuracy_at_selected_epoch": 0.3333333333333333,
   "checkpoint": "baseline",
   "eval_id": "ev-20261027-154133-f52b52",
   "eval_pinned": false,
   "test_evaluations": 0,
   "forced_test_reasons": [],
   "bootstrap": {
    "resamples": 200,
    "seed": 0
   },
   "val": {
    "accuracy": 0.3333333333333333,
    "accuracy_ci": {
     "high": 0.5010416666666669,
     "low": 0.16666666666666666,
     "resamples_defined": 200
    },
    "balanced_accuracy": 0.3333333333333333,
    "balanced_accuracy_ci": {
     "high": 0.3333333333333333,
     "low": 0.3333333333333333,
     "resamples_defined": 200
    },
    "macro_f1": 0.16666666666666666,
    "macro_f1_ci": {
     "high": 0.22252252252252258,
     "low": 0.09523809523809523,
     "resamples_defined": 200
    }
   }
  }
 ],
 "val_ranking": {
  "basis": "validation balanced accuracy at the val-selected epoch (VAL only)",
  "order": [
   "exp-20261027-154104-62cc94",
   "exp-20261027-154045-b5862d"
  ],
  "ranks": {
   "exp-20261027-154104-62cc94": 1,
   "exp-20261027-154045-b5862d": 2
  }
 },
 "confounds": [
  {
   "code": "single_seed_per_model",
   "message": "one run (one seed) per model: seed-to-seed variance is unmeasured, so differences may be within run-to-run noise",
   "values": {
    "small_cnn": 1,
    "majority_baseline": 1
   }
  }
 ],
 "paths": {
  "json": "reports/rp-20261027-154204-6d7c7f/comparison.json",
  "markdown": "reports/rp-20261027-154204-6d7c7f/comparison.md"
 }
}

### 7. test-use rules via the API

[t+ 145.0s] >>> POST /api/experiments/exp-20261027-154045-b5862d/evaluations
{
 "split": "test",
 "device": "cuda",
 "bootstrap": 200
}
<<< 202
{
 "job": {
  "id": "job-20261027-154204-f6400e",
  "kind": "evaluate",
  "status": "queued",
  "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:42:04+00:00",
  "started_at": null,
  "finished_at": null
 }
}

[t+ 159.7s] >>> GET /api/jobs/job-20261027-154204-f6400e
<<< 200
{
 "id": "job-20261027-154204-f6400e",
 "kind": "evaluate",
 "status": "completed",
 "params": "[10 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "checkpoint": "best",
  "eval_id": "ev-20261027-154219-5595dc",
  "experiment_id": "exp-20261027-154045-b5862d",
  "forced": false,
  "split": "test"
 },
 "error": null,
 "progress": null,
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:42:18+00:00",
 "created_at": "2026-10-27T15:42:04+00:00",
 "started_at": "2026-10-27T15:42:06+00:00",
 "finished_at": "2026-10-27T15:42:19+00:00"
}

[t+ 159.7s] >>> POST /api/experiments/exp-20261027-154045-b5862d/evaluations
{
 "split": "test"
}
<<< 409
{
 "error": {
  "code": "test_reevaluation_requires_force",
  "message": "experiment exp-20261027-154045-b5862d checkpoint 'best' has already been evaluated on TEST 1 time(s); repeating it requires --force --reason \"<why>\" (recorded)"
 }
}

[t+ 159.7s] >>> POST /api/experiments/exp-20261027-154045-b5862d/evaluations
{
 "split": "test",
 "force": true
}
<<< 422
{
 "error": {
  "code": "evaluation_invalid",
  "message": "--force requires a non-empty --reason"
 }
}

### 8. cancel demo (cooperative)

[t+ 159.7s] >>> POST /api/experiments
{
 "dataset_id": "ds-274cbece8d45",
 "split_id": "sp-6dbfe703a2b8",
 "model": "small_cnn",
 "input_size": 64,
 "epochs": 500,
 "batch_size": 16,
 "lr": 0.01,
 "device": "cuda",
 "num_workers": 0,
 "seed": 0,
 "early_stopping": {
  "enabled": false
 }
}
<<< 202
{
 "job": {
  "id": "job-20261027-154219-cdb00f",
  "kind": "train",
  "status": "queued",
  "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:42:19+00:00",
  "started_at": null,
  "finished_at": null
 }
}
[t+ 175.5s] GET /api/jobs/job-20261027-154219-cdb00f -> status=running progress={"epoch": 2, "epochs": 500, "experiment_id": "exp-20261027-154233-c44f9c", "latest": {"train_loss": 1.1976794136895075, "val_accuracy": 0.375, "val_balanced_accuracy": 0.375, "val_loss": 6.933880885442098}, "max_steps": null, "phase": "training", "step": 10}

[t+ 175.5s] >>> POST /api/jobs/job-20261027-154219-cdb00f/cancel
<<< 200
{
 "id": "job-20261027-154219-cdb00f",
 "kind": "train",
 "status": "running",
 "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "experiment_id": "exp-20261027-154233-c44f9c"
 },
 "error": null,
 "progress": {
  "epoch": 2,
  "epochs": 500,
  "experiment_id": "exp-20261027-154233-c44f9c",
  "latest": {
   "train_loss": 1.1976794136895075,
   "val_accuracy": 0.375,
   "val_balanced_accuracy": 0.375,
   "val_loss": 6.933880885442098
  },
  "max_steps": null,
  "phase": "training",
  "step": 10
 },
 "cancel_requested": true,
 "heartbeat_at": "2026-10-27T15:42:35+00:00",
 "created_at": "2026-10-27T15:42:19+00:00",
 "started_at": "2026-10-27T15:42:20+00:00",
 "finished_at": null
}

[t+ 176.1s] >>> GET /api/jobs/job-20261027-154219-cdb00f
<<< 200
{
 "id": "job-20261027-154219-cdb00f",
 "kind": "train",
 "status": "cancelled",
 "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "experiment_id": "exp-20261027-154233-c44f9c"
 },
 "error": "cancelled by request",
 "progress": {
  "epoch": 3,
  "epochs": 500,
  "experiment_id": "exp-20261027-154233-c44f9c",
  "latest": {
   "train_loss": 1.2634715106752183,
   "val_accuracy": 0.4166666666666667,
   "val_balanced_accuracy": 0.4166666666666667,
   "val_loss": 4.037823498249054
  },
  "max_steps": null,
  "phase": "training",
  "step": 15
 },
 "cancel_requested": true,
 "heartbeat_at": "2026-10-27T15:42:35+00:00",
 "created_at": "2026-10-27T15:42:19+00:00",
 "started_at": "2026-10-27T15:42:20+00:00",
 "finished_at": "2026-10-27T15:42:35+00:00"
}
experiment exp-20261027-154233-c44f9c: status=cancelled error='cancelled by request' epochs kept=3

### 9. hard-kill + recovery demo

[t+ 176.1s] >>> POST /api/experiments
{
 "dataset_id": "ds-274cbece8d45",
 "split_id": "sp-6dbfe703a2b8",
 "model": "small_cnn",
 "input_size": 64,
 "epochs": 500,
 "batch_size": 16,
 "lr": 0.01,
 "device": "cuda",
 "num_workers": 0,
 "seed": 0,
 "early_stopping": {
  "enabled": false
 }
}
<<< 202
{
 "job": {
  "id": "job-20261027-154236-f5cafa",
  "kind": "train",
  "status": "queued",
  "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
  "result": null,
  "error": null,
  "progress": null,
  "cancel_requested": false,
  "heartbeat_at": null,
  "created_at": "2026-10-27T15:42:36+00:00",
  "started_at": null,
  "finished_at": null
 }
}
[t+ 191.6s] GET /api/jobs/job-20261027-154236-f5cafa -> status=running progress={"epoch": 3, "epochs": 500, "experiment_id": "exp-20261027-154249-da1c34", "latest": {"train_loss": 1.2634715106752183, "val_accuracy": 0.4166666666666667, "val_balanced_accuracy": 0.4166666666666667, "val_loss": 4.037823498249054}, "max_steps": null, "phase": "training", "step": 15}
(read from db/worker.lock + jobs table: worker pid 2528, job worker_pid 2528, training child pid 17156; serve printed launcher pid 2296)
[t+ 191.6s] taskkill /F /PID 2528   (hard kill of the worker; no cleanup code runs in it)
SUCCESS: The process with PID 2528 has been terminated.
[t+ 195.8s] after kill, the API still answers: job status=running heartbeat_at=2026-10-27T15:42:51+00:00
[t+ 195.8s] worker 2528 alive: False; training child 17156 alive: False (child watchdog exits when its parent dies)
[t+ 196.0s] serve stopped; restarting `visionbench serve` (its worker recovers)
[t+ 209.8s] `visionbench serve --port 8765` up (serve pid 8380, worker pid 9816); log serve2.log

[t+ 221.8s] >>> GET /api/jobs/job-20261027-154236-f5cafa
<<< 200
{
 "id": "job-20261027-154236-f5cafa",
 "kind": "train",
 "status": "failed",
 "params": "[42 lines omitted in the packet: the job's echo of its parameters]",
 "result": {
  "experiment_id": "exp-20261027-154249-da1c34"
 },
 "error": "interrupted: worker pid 2528 not alive; heartbeat stale (30 s)",
 "progress": {
  "epoch": 5,
  "epochs": 500,
  "experiment_id": "exp-20261027-154249-da1c34",
  "latest": {
   "train_loss": 1.1264962090386286,
   "val_accuracy": 0.2916666666666667,
   "val_balanced_accuracy": 0.2916666666666667,
   "val_loss": 1.3522382378578186
  },
  "max_steps": null,
  "phase": "training",
  "step": 25
 },
 "cancel_requested": false,
 "heartbeat_at": "2026-10-27T15:42:51+00:00",
 "created_at": "2026-10-27T15:42:36+00:00",
 "started_at": "2026-10-27T15:42:37+00:00",
 "finished_at": "2026-10-27T15:43:21+00:00"
}
experiment exp-20261027-154249-da1c34: status=failed error='interrupted: worker pid 2528 not alive; heartbeat stale (30 s) (job job-20261027-154236-f5cafa)' epochs kept=5

worker log lines from the restarted serve:
  worker process started (pid 9816)
  2026-10-27 20:43:21,115 INFO visionbench.worker worker pid 14448 started on data root <scratchpad>\demo\work\var
  2026-10-27 20:43:21,259 WARNING visionbench.worker recovered job job-20261027-154236-f5cafa (train): interrupted: worker pid 2528 not alive; heartbeat stale (30 s)
  2026-10-27 20:43:21,260 WARNING visionbench.worker recovered experiment exp-20261027-154249-da1c34: interrupted: worker pid 2528 not alive; heartbeat stale (30 s) (job job-20261027-154236-f5cafa)

GET /api/jobs (summary): dataset_import=completed, dataset_split=completed, train=completed, train=completed, evaluate=completed, evaluate=completed, latency=completed, latency=completed, evaluate=completed, train=cancelled, train=failed
responses received: 550; containing the absolute data root: 0
[t+ 222.0s] serve stopped
```

The `latency.json` files written by the two latency jobs in step 5 are shown verbatim below. The
transcript shows only their ids, and the CUDA numbers also appear in the comparison. Paths are
relative to the data root.

`experiments/exp-20261027-154045-b5862d/latency/lat-20261027-154149-a2af91/latency.json`:

```json
{
  "applicable": true,
  "batch_sizes": {
    "1": {
      "median_ms": 4.393999999592779,
      "median_ms_per_image": 4.393999999592779,
      "p95_ms": 16.274875000317476
    },
    "32": {
      "median_ms": 63.45519999968019,
      "median_ms_per_image": 1.982974999990006,
      "p95_ms": 80.75701999946432
    }
  },
  "checkpoint": {
    "epoch": 5,
    "name": "best",
    "sha256": "a425279da887a1ca0ae2b3ef19d1eb015b0b522f2322b4fe709c4987493a8793"
  },
  "created_at": "2026-10-27T15:41:49+00:00",
  "device": "cpu",
  "device_name": "Intel64 Family 6 Model 186 Stepping 2, GenuineIntel",
  "experiment_id": "exp-20261027-154045-b5862d",
  "input_size": 64,
  "iterations": 30,
  "latency_id": "lat-20261027-154149-a2af91",
  "model": "small_cnn",
  "scope": "model forward pass only; preprocessing, decoding and data loading excluded",
  "torch_threads": 8,
  "visionbench_version": "0.0.1",
  "warmup": 5
}
```

`experiments/exp-20261027-154045-b5862d/latency/lat-20261027-154204-388e89/latency.json`:

```json
{
  "applicable": true,
  "batch_sizes": {
    "1": {
      "median_ms": 1.1231999997107778,
      "median_ms_per_image": 1.1231999997107778,
      "p95_ms": 1.3467250000758213
    },
    "32": {
      "median_ms": 3.0540499997187,
      "median_ms_per_image": 0.09543906249120937,
      "p95_ms": 3.2095049996769376
    }
  },
  "checkpoint": {
    "epoch": 5,
    "name": "best",
    "sha256": "a425279da887a1ca0ae2b3ef19d1eb015b0b522f2322b4fe709c4987493a8793"
  },
  "created_at": "2026-10-27T15:42:04+00:00",
  "device": "cuda",
  "device_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "experiment_id": "exp-20261027-154045-b5862d",
  "input_size": 64,
  "iterations": 30,
  "latency_id": "lat-20261027-154204-388e89",
  "model": "small_cnn",
  "scope": "model forward pass only; preprocessing, decoding and data loading excluded",
  "torch_threads": 8,
  "visionbench_version": "0.0.1",
  "warmup": 5
}
```

What the demo shows (**pipeline demo on synthetic data — not benchmark results**):
- **Upload.** A 940 615-byte multipart upload returned 202 immediately. Validation ran during
  the request, and the import ran in the worker. 120 images were imported.
- **Training.** The training job's progress was visible through `GET /api/jobs/{id}` while it
  ran on CUDA. `GET /api/health` answered in at most 4.5 ms (69 calls) during training.
- **Ordering.** Every job started only after the previous one finished: one worker, one job.
  Each job pays about 10–15 s of child-process start-up (spawn + torch import).
- **Test-use rules.** The second test evaluation returned 409 without force, and 422 with force
  but no reason.
- **Cancel.** The job and the experiment both became `cancelled by request` within about
  0.6 s, with 3 epochs kept.
- **Hard kill.** Worker pid 2528 was killed with `taskkill /F`:
  - the training child 17156 exited by itself (parent watchdog);
  - the API kept answering, and the job stayed `running`, with its heartbeat frozen;
  - after `serve` restarted, the new worker logged `recovered job` and `recovered experiment`
    lines, and both were `failed` with `interrupted: worker pid 2528 not alive; heartbeat
    stale (30 s)`. 5 epochs of history were kept.
- **Path hygiene.** 0 of the 550 responses contained the absolute data root. The only
  absolute path printed is in the worker's own log line on the console.

## 7. Deviations and decisions to review

1. **Schema version numbers.** The task said "schema v3" for the jobs table. Step 0 needed its
   own additive schema change first (`evaluations.forced/reason`, `latency_measurements`),
   so that became v3 and the jobs table plus `experiments.runner_pid` became **v4**. The
   upgrade v1 → v2 → v3 → v4 is additive and in place.
2. **New dependency `python-multipart`.** It is used directly (`MultipartParser`) to parse the
   upload as a stream. Starlette's `request.form()`/`UploadFile` spools the whole body before
   the handler runs, so it cannot enforce the limit while reading. Pinned `>=0.0.9,<1`; the
   lock file gained one line.
3. **Status names.** These are `queued|running|completed|failed|cancelled`. The plan said
   "succeeded"; `completed` matches the experiment states.
4. **Comparisons are synchronous** (`POST /api/comparisons` → 201). They read recorded
   predictions and never load a model, so no ML work runs in the request. That matches "POST
   (synchronous)" in the task.
5. **Recovery criterion.** A `running` job is recovered when its worker pid is dead **or** its
   heartbeat is older than 30 s. Recovery runs only at worker start, under the worker lock, so
   a live worker's job is never touched. Orphaned experiments (pending/running with no live
   `runner_pid`, e.g. a killed CLI `train`) are also marked failed.
6. **Windows process handling.** `pid_alive` and `terminate_pid` use `OpenProcess`,
   `GetExitCodeProcess` and `TerminateProcess` through `ctypes`, because `os.kill(pid, 0)`
   terminates the target on Windows. The POSIX branches have not been run.
7. **Screenshot.** None is included. The README documents `/docs`, `/redoc` and
   `/openapi.json` (each checked to return 200) and lists all 23 operations.
8. **Test warning.** The 1 warning is Starlette's deprecation of `httpx` in `TestClient`. It
   does not affect the API at runtime.

## 8. Not verified / limitations

- The CI workflow has never run. Linux/macOS process handling was not run.
- No authentication by design. Loopback-only binding is enforced by `serve`, but nothing stops
  someone from running uvicorn by hand on another interface; the README warns against it.
- One API process and one worker are tested. Multiple uvicorn workers are not supported or
  tested; the SQLite queue would still be atomic, but this is not verified.
- Upload limits are tested with small configured values (1 MB upload, a few members). The
  defaults (200 MB, 50 000 members, 4 GB) are not exercised.
- The cancel timeout is tested only with a latency job (above). The training variant of the
  same code path is untested.
- Job start-up costs about 10–15 s per job on this machine (spawn + torch import).
- All data is synthetic. Every number here is a **pipeline demo on synthetic data — not
  benchmark results**.

## 9. Code printed for review

The final message of this task prints the full source of:
- `src/visionbench/jobs/store.py`, the job store;
- `src/visionbench/jobs/worker.py`, the worker (claim/run/cancel/recovery);
- `src/visionbench/data/archive.py`, ZIP validation and extraction;
- the image-serving endpoint `get_image` in `src/visionbench/api/routes/datasets.py`, with the
  `_records` helper and constants it uses.
