# Changelog

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
