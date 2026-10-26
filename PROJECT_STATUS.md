# Project Status

_Last updated: 2026-10-27 — end of M0 (pending supervisor review)._

## Current state
Foundation only: package skeleton, config, seeding, env info, health/ready API, `visionbench env`
CLI, tests, CI workflow file, continuity docs. No ML features.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130)
- Fresh venv + `pip install -e ".[dev]"` succeeded.
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench`: 23 passed, total coverage 97%.
- `visionbench env` prints JSON; CUDA available (RTX 3050 6GB).
- Live uvicorn: `/api/health` 200; `/api/ready` 200 with default `./var`; `/api/ready` 503 with a data root beneath a regular file.
- CPU resnet18 forward (1×3×224×224 → 1×1000) works; pretrained ResNet18 and EfficientNet-B0 weights downloaded and loaded.
- Identical seeds give bitwise-identical forward outputs (test + script).

## Not verified
- CI workflow (`.github/workflows/ci.yml`) has never run; no remote exists.
- CPU-only torch build (`/whl/cpu`) not installed on this machine; only the cu130 build was tested locally.
- Python 3.12/3.13 compatibility (declared in `requires-python`, not tested).

## Blockers
None.

## Git
Not a git repository; no commits made (commits were not authorized for this task).

## Next task
M1 — dataset ingestion, validation, stats, duplicate groups, group-aware stratified splits, manifests.
