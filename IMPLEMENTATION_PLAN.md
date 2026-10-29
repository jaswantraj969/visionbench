# Implementation Plan

Each milestone ends with a supervisor review of evidence (`REVIEW_PACKET.md`). A milestone is
done only when every acceptance criterion has been demonstrated by commands executed in-session.

## M0 — Foundation (done; follow-ups addressed in M1)
- Package `visionbench` (src layout) installs into a fresh venv with documented commands.
- `core.config` (pydantic-settings, `VISIONBENCH_` prefix, data root default `./var`, subdirs created on demand), `core.seeding.set_seed`, `core.env_info`.
- FastAPI factory with `GET /api/health` and `GET /api/ready` (200 / 503 with reason), uniform error schema `{"error":{"code","message"}}`.
- CLI `visionbench env`.
- ruff clean; tests cover config, seed reproducibility, health, ready (writable and unwritable data root).
- Continuity files present; CI workflow file present (validated later).

## M1 — Dataset ingestion & validation (done; follow-ups in M2 step 0)
- Import an image-folder dataset (class-per-directory) into the data root; reject corrupt/unsupported files with a per-file report.
- Statistics: class counts, image size / mode distribution, corrupt-file list.
- Duplicate groups: exact (SHA-256) and near (perceptual hash, documented threshold); label-conflict groups flagged.
- Group-aware stratified train/val/test splits: a whole duplicate group goes to one split; class proportions within a stated tolerance; deterministic given a seed.
- Versioned JSON manifest (files, hashes, labels, split, group IDs); tests prove no group leaks across splits.

## M2 — Model registry & training (CLI) (done; follow-ups in M3 step 0)
- Registry: majority-class baseline, SmallCNN, ResNet18, EfficientNet-B0 (pretrained optional, recorded).
- `visionbench train` with config file + seed; checkpoints under `experiments/<exp_id>/checkpoints/`; early stopping and model selection on validation only.
- Run record: config, seed report, env info, manifest hash, per-epoch metrics.
- Tests: tiny synthetic dataset trains end-to-end on CPU; same seed → same metrics on same machine.

## M3 — Evaluation & comparison (done; follow-ups in M4 step 0)
- Test-split evaluation of the val-selected checkpoint; every test evaluation recorded and counted: accuracy, macro-F1, per-class precision/recall, confusion matrix, calibration (ECE), bootstrap CIs.
- Comparison report across runs, including the majority baseline; results stored as JSON.

## M4 — API, job runner, uploads (done; hardened in M4.1)
- Step 0: split-independent latency; `compare --eval-ids/--latency-device`; repeated test evaluation requires `--force --reason`; val selection-bias note; baseline `--checkpoint last` is an error.
- FastAPI endpoints for datasets, uploads, splits, experiments, evaluations, latency, comparisons, jobs over the library; no ML work inside a request.
- Single-worker job runner: SQLite job table (queued/running/completed/failed/cancelled), atomic claim, child process per job, heartbeat, progress, cooperative cancel + terminate timeout, startup recovery of interrupted jobs and orphaned experiments.
- Secure uploads: streamed byte limit, magic check, member-name/symlink/encryption/collision/bomb/layout rules, safe extraction, cleanup in every path; `import_root` confinement.
- Tests for job lifecycle, cancel, hard kill + recovery, every upload rejection rule, image traversal, path hygiene.

## M5 — Grad-CAM (done: prediction + Grad-CAM in the API, sanity checks)
- Grad-CAM for CNN models in the registry (target layer per architecture), overlays saved as artifacts; sanity check (map changes when weights are randomized).

## M6 — Robustness (done: suite `eurosat-c-v1`, VALIDATION evaluations)
- Corruption suite (noise, blur, brightness, contrast, JPEG) at graded severities; accuracy-vs-severity curves per model.

## M7 — Frontend (done: 9 pages, unit tests, Playwright e2e)
- React + TypeScript + Vite UI over the API: datasets, runs, comparison, Grad-CAM viewer, robustness charts; API-typed client; basic tests.

## M8 — Real benchmark & packaging (in progress: packaging done in `v1.0.0-rc1`; the single TEST evaluation follows `FINAL_ANALYSIS_PLAN.md`)
- Benchmark on a real public dataset with documented license; results reported per protocol.
- Dockerfile(s), passing CI, user/developer docs, screenshots.
