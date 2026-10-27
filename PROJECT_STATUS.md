# Project Status

_Last updated: 2026-10-28, end of D1 (EuroSAT onboarding; pending supervisor review and the threshold decision)._

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
  - 9 calibration splits and audits for dHash thresholds 3 and 5. **The thresholds have not
    been chosen**; the supervisor decides from REVIEW_PACKET.md.
- There is no authentication, Grad-CAM, robustness suite, or frontend.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130, RTX 3050 6GB)
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
- Training or evaluation on a real dataset: not started (EuroSAT is imported and audited only).
- The audit does not search near-duplicates within val/test or between val and test by
  perceptual hash, only val/test against train. Near-uniform EuroSAT tiles (3357
  low-information images, mostly SeaLake and Forest) cannot be checked by hashing.

## Blockers
- Waiting for the supervisor's choice of the dHash and low-information thresholds for EuroSAT.
- The Copernicus attribution `[Year]` is not stated by the dataset source (DATASETS.md).

## Next task
M5: Grad-CAM for registry CNNs (overlays as artifacts, randomized-weights sanity check), run
as worker jobs and exposed through the API.
