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
| `core.errors` | implemented (M1) | `VisionBenchError` hierarchy, the expected user-facing errors |
| `core.jsonio` | implemented (M1) | canonical JSON (sorted keys, LF, UTF-8) for byte-identical files |
| `api.main` | implemented (M0) | `create_app()` factory; `/api/health`, `/api/ready`; uniform error schema |
| `cli` | implemented (M0–M1) | `visionbench env`, `visionbench dataset import\|report\|split\|list` |
| `data.records` | implemented (M1) | `FileRecord`, `Issue`/`IssueCategory`, NFC names, fingerprint |
| `data.scan` | implemented (M1) | source validation, Pillow decode checks, symlink/junction refusal |
| `data.hashing` | implemented (M1) | 64-bit dHash, vectorized near-duplicate edge search |
| `data.grouping` | implemented (M1) | union-find duplicate groups, low-information guard, conflicts |
| `data.report` | implemented (M1) | `report.json` statistics |
| `data.store` | implemented (M1) | snapshot import (staged + atomic rename), load/list |
| `data.split` | implemented (M1) | group-aware stratified splits, manifests, leakage check |
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
  datasets/                          (M1, implemented)
    <dataset_id>/                    ds-<12 hex>; immutable once renamed into place
      dataset.json                   name, id, created_at, source_path, fingerprint, classes,
                                     counts, import config, validation summary
      report.json                    statistics, issues (+relpaths), duplicate groups, samples
      files.json                     per accepted image: relpath, label, sha256, size, w/h,
                                     mode, format, dhash (hex), gray_std
      raw/<class>/<file>             byte-identical copies of accepted images
      splits/<split_id>/             sp-<12 hex>
        manifest.json                canonical, byte-reproducible (no timestamps)
        split_info.json              created_at, manifest sha256, numpy/python versions
    .tmp-import-*/  .tmp-split-*/    staging dirs; renamed into place on success, removed on error
  artifacts/   checkpoints, reports, Grad-CAM maps (M2+)
  db/          SQLite database                    (M4)
```

Subdirectories are created on first use (`Settings.data_dir()`), never at import time.

## Data layer (M1)

`visionbench dataset import <src> --name N` runs the following:

1. `scan_source` (sorted, NFC). Each accepted file is read **once**. Those bytes are SHA-256
   hashed, decoded twice (`verify()`, then reopen and `load()`), dHashed, and passed to a sink that
   writes them into a staging directory.
2. `check_requirements`: at least 2 classes, and at least `min_images_per_class` per class.
3. Fingerprint, then `dataset_id`. The import is refused if the id already exists.
4. `build_groups` + `build_report`, then `files.json` / `report.json` / `dataset.json`.
5. `os.rename(staging, datasets/<id>)`. On any error the staging directory is deleted, so no
   partial dataset is ever visible.

`visionbench dataset split <id> --seed S` loads `files.json` and checks it against the
fingerprint. It then rebuilds the groups with the split's thresholds and assigns them
(see EXPERIMENT_PROTOCOL.md §4). Finally it writes `splits/<split_id>/` using the same
staging + rename pattern. If that split already exists, it only verifies it byte for byte.

Notes:
- The scan temporarily sets Pillow's module-level `MAX_IMAGE_PIXELS` and
  `LOAD_TRUNCATED_IMAGES`, so scanning is **not thread-safe**. The planned M4 job runner is
  single-worker.
- `dataset.json.source_path` is local CLI metadata. The M4 API must not expose it.
- The snapshot is "immutable" because it is never overwritten and its hashes are recorded. Files
  are not made read-only, and there is no `verify` command yet that re-hashes `raw/`.

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
