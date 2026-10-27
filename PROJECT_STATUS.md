# Project Status

_Last updated: 2026-10-27, end of M4 (pending supervisor review)._

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
- There is no authentication, Grad-CAM, robustness suite, or frontend.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130, RTX 3050 6GB)
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench`: 242 passed and 1 skipped (file symlink), coverage 91%. The same
  holds with `CUDA_VISIBLE_DEVICES=-1` (in-process `torch.cuda.is_available()=False`).
  `jobs/worker.py` reports 51% because most of it runs in worker subprocesses, which coverage
  does not measure; those paths are exercised by the real-worker tests.
- Real-worker tests: one job at a time, progress, cooperative cancel, hard kill of the worker
  + recovery (0 failures in 10 repeated runs after the WAL/ordering fixes), API health while
  training (max 20.4 ms normal run / 24.3 ms GPU-hidden run).
- Every upload rejection rule has a test; image endpoint traversal cases return 404; a full
  API flow test asserts that no response body contains the absolute data root.
- **Live demo with `visionbench serve`** (pipeline demo on synthetic data, not benchmark
  results): upload → split → CUDA training with live progress → val evaluations → CPU and
  CUDA latency → val comparison → test-use 409/422 → cancel → hard kill + recovery. 550
  responses, none containing the absolute data root. See REVIEW_PACKET.md.

## Not verified
- The CI workflow has never run. The file-symlink test is skipped on this machine.
- Linux/macOS process handling (`pid_alive`/terminate have POSIX branches that were not run).
- Concurrency beyond one API process + one worker; very large uploads (limits are tested with
  small values).
- Real (non-synthetic) datasets; Python 3.12/3.13.

## Blockers
None.

## Next task
M5: Grad-CAM for registry CNNs (overlays as artifacts, randomized-weights sanity check), run
as worker jobs and exposed through the API.
