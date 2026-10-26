# VisionBench

Reproducible computer-vision benchmarking and explainability platform — **currently at milestone
M0 (foundation only)**. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for what comes next.

## What exists now

- Importable package `visionbench` (src layout).
- `visionbench.core`: settings (`VISIONBENCH_` env vars), `set_seed()`, environment info.
- FastAPI app with `GET /api/health` and `GET /api/ready`.
- CLI: `visionbench env` prints environment info as JSON.
- Tests (pytest) and lint (ruff).

There is **no** dataset handling, training, evaluation, explainability, robustness, or
frontend yet.

## Setup

Tested on Python 3.11 (Windows 11). `pyproject.toml` declares `>=3.11,<3.14`, but 3.12 and 3.13
have not been tested. Install PyTorch from the official wheel index first, then the package.
The exact versions of the verified local CUDA environment are in
[requirements/lock-dev-cu130.txt](requirements/lock-dev-cu130.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# EITHER: CUDA 13.0 build (what was used on this machine: RTX 3050, driver CUDA 13.4)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# OR: CPU-only build (used by CI; not run on this machine yet)
# .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On Linux/macOS use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Configuration

Copy `.env.example` to `.env` or set environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VISIONBENCH_DATA_ROOT` | `./var` | Runtime data root; `datasets/`, `artifacts/`, `db/` created on demand |
| `VISIONBENCH_MAX_UPLOAD_MB` | `200` | Upload size limit (enforced once uploads exist, M4) |
| `VISIONBENCH_LOG_LEVEL` | `INFO` | Log level |

## Run tests and lint

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\pytest.exe -q --cov=visionbench
```

## Run the API

```powershell
.\.venv\Scripts\uvicorn.exe visionbench.api.main:app --host 127.0.0.1 --port 8000
```

- `GET /api/health` → `{"status":"ok","version":"0.0.1"}`
- `GET /api/ready` → `200 {"status":"ready",...}` if the data root is writable and torch imports,
  otherwise `503 {"error":{"code":"not_ready","message":...}}`

All errors use `{"error": {"code": ..., "message": ...}}`.

## CLI

```powershell
.\.venv\Scripts\visionbench.exe env
```

## Docs

[ARCHITECTURE.md](ARCHITECTURE.md) · [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) ·
[PROJECT_STATUS.md](PROJECT_STATUS.md) · [CHANGELOG.md](CHANGELOG.md)
