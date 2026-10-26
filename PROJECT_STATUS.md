# Project Status

_Last updated: 2026-10-27, end of M3 (pending supervisor review)._

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
  - forward-pass latency;
  - test-use accounting (`evaluations` table, warnings, counts in reports);
  - `compare` reports with VAL-only ranking and confound flags.
- There are no API endpoints beyond health/ready, no job runner, Grad-CAM, robustness, or
  frontend.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130, scikit-learn 1.9.1, RTX 3050 6GB)
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench`: 174 passed and 1 skipped (file symlink), coverage 95%. The same
  holds with `CUDA_VISIBLE_DEVICES=-1`, where in-process `torch.cuda.is_available()=False`.
- **Isolation both ways (tests):** `evaluate --split val` opens exactly the val images and
  `--split test` exactly the test images. Training still cannot reach test, and the evaluation
  selector rejects `train`.
- **Metric fixtures (hand-computed):** absent class, never-predicted class, perfect predictions,
  constant majority predictor, and binary vs multiclass. Undefined values are always `null` with a
  reason.
- **Demos on the M2 synthetic shapes data** (pipeline demo on synthetic data, not benchmark
  results):
  - test evaluations of 4 experiments;
  - a repeat-evaluation warning (#2);
  - a `compare` report with CIs, paired differences and 6 confound flags;
  - CPU and CUDA latency for small_cnn and resnet18.
  - The M2 database (schema v1) was upgraded to v2 in place.

## Not verified
- The CI workflow has never run. The file-symlink test is skipped on this machine.
- Latency variance across runs and thermal states (one measurement per configuration).
- Real (non-synthetic) datasets; Python 3.12/3.13.

## Blockers
None.

## Next task
M4: FastAPI endpoints over the library and a single-worker job runner (SQLite job table, cancel,
mark interrupted `running` jobs failed on startup), plus secure uploads. Test evaluation must stay
behind the same accounting.
