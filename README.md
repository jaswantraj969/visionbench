# VisionBench

Reproducible computer-vision benchmarking and explainability platform. **Current state: milestone
M1 (dataset layer)**. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for what comes next.

## What exists now

- Importable package `visionbench` (src layout).
- `visionbench.core`: settings (`VISIONBENCH_` env vars), `set_seed()`, environment info.
- `visionbench.data` (library and CLI): dataset import into immutable snapshots, validation,
  statistics, exact and near duplicate groups, and leakage-aware train/val/test split manifests.
- FastAPI app with `GET /api/health` and `GET /api/ready`. There are no dataset endpoints yet
  (planned for M4).
- CLI: `visionbench env` and `visionbench dataset import|report|split|list`.
- Tests (pytest) and lint (ruff).

There is **no** training, evaluation, explainability, robustness, or frontend yet.

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
| `VISIONBENCH_MAX_IMAGE_PIXELS` | `50000000` | Images with more pixels are rejected |
| `VISIONBENCH_MIN_IMAGES_PER_CLASS` | `3` | Minimum valid images per class |
| `VISIONBENCH_DHASH_THRESHOLD` | `5` | Max dHash Hamming distance for near duplicates |
| `VISIONBENCH_LOW_INFO_STD_THRESHOLD` | `3.0` | Gray std below which perceptual matching is skipped |
| `VISIONBENCH_REPORT_SAMPLES_PER_CLASS` | `8` | Seeded sample size per class in `report.json` |
| `VISIONBENCH_CONFLICT_POLICY` | `exclude` | Label-conflict duplicate groups: `exclude` or `error` |

## Datasets

Source folder format: exactly one level of class folders. The folder name is the label.

```
my_dataset/
  cats/  img001.jpg  img002.png ...
  dogs/  a.webp  b.bmp ...
```

Supported extensions (case-insensitive): `.jpg .jpeg .png .bmp .webp`. At least 2 classes are
required, each with at least 3 valid images (configurable). Anything else is reported, never
silently dropped: unsupported, corrupt, truncated, empty, oversized, hidden/system files,
symlinks/junctions (never followed), nested folders, and name collisions.

```powershell
.\.venv\Scripts\visionbench.exe dataset import C:\path\to\my_dataset --name my-dataset
.\.venv\Scripts\visionbench.exe dataset list
.\.venv\Scripts\visionbench.exe dataset report ds-xxxxxxxxxxxx          # add --json for everything
.\.venv\Scripts\visionbench.exe dataset split ds-xxxxxxxxxxxx --seed 42 --fractions 0.7 0.15 0.15
```

- `import` copies accepted images into `<data_root>/datasets/<id>/raw/` and writes
  `dataset.json`, `report.json` and `files.json`. It refuses to overwrite an existing dataset.
  Optional flags: `--max-image-pixels`, `--min-images-per-class`, `--dhash-threshold`,
  `--low-info-std`.
- `split` writes `<data_root>/datasets/<id>/splits/<split_id>/manifest.json`. The same dataset and
  config always produce a byte-identical manifest. Duplicate groups never span splits, and the
  manifest records a computed leakage check. Optional flags: `--conflict-policy exclude|error`,
  `--allow-missing-classes`, `--dhash-threshold`, `--low-info-std`.
- Errors print `error: ...` and exit with code 1. Add `--debug` for a traceback.

Algorithm details and limitations: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## Run tests and lint

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\pytest.exe -q --cov=visionbench
```

To run with the GPU hidden in Windows PowerShell 5.1, use `$env:CUDA_VISIBLE_DEVICES="-1"`.
`$env:CUDA_VISIBLE_DEVICES=""` deletes the variable instead of setting it, so the GPU stays
visible. The pytest summary prints `torch.cuda.is_available()` for every run.

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
