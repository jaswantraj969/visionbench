# Changelog

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
