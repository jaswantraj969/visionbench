# Project Status

_Last updated: 2026-10-29, end of M5 (prediction and Grad-CAM served from the API process; pending supervisor review)._

## Current state
- **M0:** foundation (config, seeding, env info, health/ready API).
- **M1 (+ M2 step 0):** dataset layer.
  - Import, validation, stats, duplicate groups with the popcount low-information guard.
  - `vb-split-v2` splits.
  - `dataset verify`.
- **M2 (+ M3 step 0):**
  - model registry;
  - CLI training (train/val only), checkpoints, SQLite tracking;
  - configurable torch hub dir; frozen BatchNorm for head_only;
  - norm-stats cache under `cache/norm_stats/`, so split directories are write-once.
- **M3:**
  - `evaluate`, the only code path that reads test images (val/test, never train);
  - metrics with explicit undefined values (null + reason);
  - bootstrap and paired-bootstrap CIs;
  - test-use accounting (`evaluations` table, counts in reports);
  - `compare` reports with VAL-only ranking and confound flags.
- **M4 step 0:**
  - `visionbench latency` (split-independent; evaluate no longer measures latency);
  - `compare --eval-ids` and `--latency-device`;
  - repeated test evaluation requires `--force --reason` (stored, reported);
  - val comparisons state the selection-bias note;
  - `--checkpoint last` on the baseline is an error.
- **M4:**
  - job store (`jobs` table, schema v4, WAL + busy_timeout, queue limit → 429);
  - single worker: lock file, atomic claim, spawn child per job, heartbeat, training progress,
    cooperative cancel with terminate-after-timeout, startup crash recovery;
  - hardened ZIP upload (streamed with byte limit, central-directory validation, safe
    extraction, cleanup in every path) and `import_root`-confined path imports;
  - 23 API operations under `/api`, strict id patterns, uniform errors, scrubbed responses,
    CORS restricted to configured origins;
  - `visionbench worker` and `visionbench serve` (loopback only).
- **M4.1** (fixes from the M4 review, no new features):
  - processes are identified by (pid, creation time) via psutil (schema v5 stores creation
    times); nothing is terminated or treated as alive by pid alone;
  - the worker lock is an exclusive OS lock (`msvcrt.locking` / `fcntl.flock`), not a pid
    file;
  - recovery runs under that lock and fails every `running` job unconditionally; orphaned
    experiments are kept only while their runner's identity is alive;
  - the API rejects foreign `Host` headers (400), and POST/PUT/PATCH/DELETE without
    `X-VisionBench-Request: 1` (403) or with a foreign `Origin` (403); `nosniff` on every
    response.
- **D1** (EuroSAT onboarding, no training or evaluation):
  - EXIF orientation policy: one shared decoder; dataset format 2, hash version 2; format-1
    snapshots are refused with a request to re-import;
  - `dataset audit-split`: an independent leakage audit (in-house DCT pHash + dHash over all
    images, histograms, consistency check, near misses, exact and decoded-pixel matches,
    group statistics, pairs.csv, contact sheets);
  - oversized-group warnings in `split_info.json` and the `split` output;
  - low-information breakdown per class and reason in `report.json`;
  - EuroSAT RGB license verified (MIT + Copernicus attribution; DATASETS.md), downloaded from
    Zenodo (MD5 matches), imported as `ds-61112928e412` (27,000 images, `verify --full` OK);
  - 9 calibration splits and audits for dHash thresholds 3 and 5. The supervisor chose
    `sp-55320e11726c` (dHash 5, std 3.0, popcount [8, 56]) as the frozen benchmark split
    (DATASETS.md).
- **B1** (eurosat-v1 benchmark, VALIDATION only):
  - protocol + 5 YAMLs frozen and hashed (`benchmarks/eurosat-v1/`, commit `c7d32c0`)
    before the first experiment; no amendments;
  - pilot (`pilot.json`): batch 64 fits all models; GPU training bitwise deterministic for
    all four trainable models;
  - `scripts/run_benchmark.py`: resumable driver (hash check, interruption handling, OOM
    stop), VAL evaluation, latency, multi-seed summary, no-test check;
  - 15/15 runs completed on the first attempt (no reruns, no interruptions); 15 VAL
    evaluations, 8 latency measurements, the seed-0 VAL comparison and the multi-seed
    summary in `benchmarks/eurosat-v1/results/`. All numbers there are VALIDATION
    (selection-biased), not test results;
  - **0 test evaluations exist in the whole database.** Test is evaluated once, at M8.
- **M5 step 0** (B1 follow-ups): `experiments.stopped_reason` (schema v6), backfilled for the
  27 existing experiments from their own `train.log`; full-length repro check of
  resnet18_scratch_64 seed 1 (`repro_check.json`): history (10 epochs), `best.pt` and `last.pt`
  bitwise identical to the benchmark run.
- **M5** (prediction and Grad-CAM, no frontend):
  - `POST /api/experiments/{id}/predict` and `/gradcam`, synchronous in the API process; body
    read on the event loop, model work in the thread pool;
  - `inference.service`: verified loading only (evaluation path), LRU cache keyed by
    (experiment, checkpoint sha256) and re-hashed per request, CPU by default, bounded
    concurrency (429), in-memory uploads through the shared decoder, TEST-image lock (403;
    relpath or identical bytes), out-of-distribution warning;
  - `explain`: Grad-CAM (target logit, autograd.grad, hook removed in `finally`, all-zero
    reported), exact val-transform geometry with "not seen by the model" marking, overlay PNG;
  - sanity checks: localization 75/75, target sensitivity mean r -0.606, model randomization
    mean |rho| 0.129, equality with a backward-hook reference, geometry against real
    transforms, no hook leaks.
- There is no authentication, robustness suite, or frontend.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130, RTX 3050 6GB)
- M5 (2026-10-29, after the last code change): `ruff check .` / `ruff format --check .` pass;
  `pytest -q --cov=visionbench`: 316 passed and 1 skipped in 630.53 s, coverage 94 %
  (`inference/service.py` 96 %, `explain/gradcam.py` 94 %, `explain/geometry.py` 97 %,
  `explain/overlay.py` 91 %, `api/routes/inference.py` 97 %); with `CUDA_VISIBLE_DEVICES=-1`:
  316 passed and 1 skipped in 495.29 s.
- M5 demos on the real eurosat-v1 seed-0 models through `visionbench serve` (CPU inference;
  VALIDATION images, demo, not results): 20 seeded VAL images x 4 models, predict + gradcam;
  cold model load 6.1-7.1 s, warm predict 27.5-146.2 ms (median, total), warm gradcam
  78.6-252.2 ms; one synthetic upload got the OOD warning; a TEST image request returned
  `403 test_split_locked` (curl.exe). README curl lines were run in PowerShell 5.1.
- `inference_device=cuda` checked once on resnet18_imagenet_224 (same prediction as CPU;
  probabilities within 6.7e-7, normalised CAM within 1.1e-4).
- B1 (2026-10-28, after the last code change): `ruff check .` / `ruff format --check .` pass;
  `pytest -q`: 285 passed and 1 skipped in 386.02 s; with `CUDA_VISIBLE_DEVICES=-1`: 285
  passed and 1 skipped in 215.66 s (no coverage run in B1).
- B1 benchmark: `run_benchmark.py check` reports 0 test evaluations for the 15 benchmark
  and 12 pilot experiments, and 0 in the whole database.
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench` (D1): 285 passed and 1 skipped (file symlink) in 511.76 s,
  coverage 94%. With `CUDA_VISIBLE_DEVICES=-1`: 285 passed and 1 skipped in 539.21 s
  (in-process `torch.cuda.is_available()=False`). New modules: `data/audit.py` 94%,
  `data/phash.py` 100%, `data/imageio.py` 100%.
- M4.1 (unchanged since): 270 passed / 1 skipped in both modes.
  Coverage now includes the worker subprocesses and their spawn children: `jobs/worker.py`
  87% (was 51% in M4).
- Real-worker tests: one job at a time, progress, cooperative cancel, terminate-after-timeout
  for latency and training jobs, hard kill of the worker + immediate recovery, second worker
  fails fast on the OS lock, API health while training (max 22.4 ms normal run /
  15.7 ms GPU-hidden run).
- Every M4.1 defect scenario was run against the pre-fix code (`bcd6674`) and reproduced there
  (REVIEW_PACKET.md §6).
- Every upload rejection rule has a test; image endpoint traversal cases return 404; a full
  API flow test asserts that no response body contains the absolute data root.
- **Live demo with `visionbench serve`** (pipeline demo on synthetic data, not benchmark
  results): upload → split → CUDA training with live progress → val evaluations → CPU and
  CUDA latency → val comparison → test-use 409/422 → cancel → hard kill + recovery. 550
  responses, none containing the absolute data root (M4 packet, commit `bcd6674`).
- M4.1 demos (REVIEW_PACKET.md): hard kill + `serve` restart 0.4 s later → recovery logged
  0.08 s after the new worker started; curl.exe 403/403/400/2xx; pid-reuse child survives a
  real `visionbench worker` recovery.

## Not verified
- The CI workflow has never run. The file-symlink test is skipped on this machine.
- Linux/macOS process handling: psutil and the `fcntl.flock` branch of the worker lock were
  not run on POSIX.
- Concurrency beyond one API process + one worker; very large uploads (limits are tested with
  small values).
- Python 3.12/3.13.
- Any TEST-split number for EuroSAT: none exists by design (M8).
- Inference with `inference_max_concurrent` > 1 under real parallel load (only the busy path
  and per-model locks are tested); CUDA inference in the API tests (they run on CPU).
- The TEST-image lock does not recognise re-encoded, resized or edited copies of test images.
- Grad-CAM sanity checks use a synthetic task and SmallCNN only; they do not establish that the
  EuroSAT CAMs are faithful.
- Training times are indicative (one shared laptop GPU; throughput changed during the
  session, see REVIEW_PACKET.md).
- The audit does not search near-duplicates within val/test or between val and test by
  perceptual hash, only val/test against train. Near-uniform EuroSAT tiles (3357
  low-information images, mostly SeaLake and Forest) cannot be checked by hashing.

## Blockers
- None. M5 awaits supervisor review (REVIEW_PACKET.md).

## Next task
M6 (per IMPLEMENTATION_PLAN.md): robustness suite. A frontend is not started.
