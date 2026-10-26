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

## M1 — Dataset ingestion & validation (implemented; pending review)
- Import an image-folder dataset (class-per-directory) into the data root; reject corrupt/unsupported files with a per-file report.
- Statistics: class counts, image size / mode distribution, corrupt-file list.
- Duplicate groups: exact (SHA-256) and near (perceptual hash, documented threshold); label-conflict groups flagged.
- Group-aware stratified train/val/test splits: a whole duplicate group goes to one split; class proportions within a stated tolerance; deterministic given a seed.
- Versioned JSON manifest (files, hashes, labels, split, group IDs); tests prove no group leaks across splits.

## M2 — Model registry & training (CLI)
- Registry: majority-class baseline, SmallCNN, ResNet18, EfficientNet-B0 (pretrained optional, recorded).
- `visionbench train` with config file + seed; checkpoints under `artifacts/`; early stopping and model selection on validation only.
- Run record: config, seed report, env info, manifest hash, per-epoch metrics.
- Tests: tiny synthetic dataset trains end-to-end on CPU; same seed → same metrics on same machine.

## M3 — Evaluation & comparison
- Test-split evaluation once per final model: accuracy, macro-F1, per-class precision/recall, confusion matrix, calibration (ECE).
- Comparison report across runs, including the majority baseline; results stored as JSON.

## M4 — API, job runner, uploads
- FastAPI endpoints for datasets, runs, evaluations over the library.
- Single-worker job runner: SQLite job table, queued/running/succeeded/failed/cancelled states, cancellation, jobs found `running` at startup marked `failed`.
- Secure uploads: size limit (`max_upload_mb`), archive path-traversal and bomb protection, content-type sniffing.
- Tests for job lifecycle, cancel, crash recovery, upload rejection cases.

## M5 — Grad-CAM
- Grad-CAM for CNN models in the registry (target layer per architecture), overlays saved as artifacts; sanity check (map changes when weights are randomized).

## M6 — Robustness
- Corruption suite (noise, blur, brightness, contrast, JPEG) at graded severities; accuracy-vs-severity curves per model.

## M7 — Frontend
- React + TypeScript + Vite UI over the API: datasets, runs, comparison, Grad-CAM viewer, robustness charts; API-typed client; basic tests.

## M8 — Real benchmark & packaging
- Benchmark on a real public dataset with documented license; results reported per protocol.
- Dockerfile(s), passing CI, user/developer docs, screenshots.
