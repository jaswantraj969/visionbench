# Changelog

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
