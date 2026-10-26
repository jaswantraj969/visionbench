# Architecture

## Layering (library-first)

```
  FastAPI (visionbench.api)    CLI (visionbench.cli)    Job runner (visionbench.jobs, planned)
                 \                    |                      /
                  `----------- visionbench library ---------'
          core · data · models · training · evaluation · explain · robustness · tracking
```

All ML and data logic lives in the importable `visionbench` package. The API, CLI, and job
runner are thin layers: they parse input, call library functions, and serialize output. They
contain no ML logic.

## Package map

| Package | Status | Responsibility |
|---|---|---|
| `core.config` | implemented (M0) | `Settings` from env (`VISIONBENCH_` prefix / `.env`); data-root subdirs created on demand |
| `core.seeding` | implemented (M0) | `set_seed()` for random / NumPy / torch; returns a record of what was set |
| `core.env_info` | implemented (M0) | Python / torch / torchvision / NumPy / platform / CUDA versions for experiment records |
| `api.main` | implemented (M0) | `create_app()` factory; `/api/health`, `/api/ready`; uniform error schema |
| `cli` | implemented (M0) | `visionbench env` |
| `data` | empty | M1: ingestion, validation, stats, duplicate groups, splits, manifests |
| `models` | empty | M2: model registry |
| `training` | empty | M2: training loop, checkpoints, early stopping |
| `evaluation` | empty | M3: metrics, comparison |
| `explain` | empty | M5: Grad-CAM |
| `robustness` | empty | M6: corruption benchmarks |
| `tracking` | empty | run records / experiment metadata (SQLite via SQLAlchemy, later) |
| `jobs` | empty | M4: single-worker job runner |

## Data root layout

Default `./var/` (gitignored), overridable with `VISIONBENCH_DATA_ROOT`:

```
var/
  datasets/    imported datasets + manifests      (M1)
  artifacts/   checkpoints, reports, Grad-CAM maps (M2+)
  db/          SQLite database                    (M4)
```

Subdirectories are created on first use (`Settings.data_dir()`), never at import time.

## Error schema

Every API error response: `{"error": {"code": "<machine code>", "message": "<human text>"}}`.
Unhandled exceptions return `500 internal_error` without leaking exception details.

## Planned job-runner design (M4, not implemented)

- One worker process executes jobs serially; the API process only enqueues and reads status.
- Jobs are rows in a SQLite table (`id, kind, params JSON, state, created/started/finished, error, cancel_requested`).
- Cancellation is cooperative: the worker checks `cancel_requested` between steps (e.g. per batch/epoch).
- On worker startup, any job still in `running` state is marked `failed` with reason "interrupted" (crash recovery); jobs are not auto-resumed.

## Model-loading policy

- Only checkpoints produced internally by VisionBench (under `artifacts/`) are ever loaded.
- Checkpoints are loaded with `torch.load(..., weights_only=True)`.
- User-uploaded models or pickles are never accepted or loaded.
- Pretrained weights come only from torchvision's weights API (official URLs, hash-checked by torchvision).
