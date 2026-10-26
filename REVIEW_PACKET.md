# Review Packet — VB-M2

**Task:** VB-M2: model registry, CLI training from split manifests, experiment tracking
**Date:** 2026-10-27
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130,
torchvision 0.29.1+cu130; NumPy 2.4.6; Pillow 12.3.0; SQLAlchemy 2.1.6; PyYAML 6.0.3;
NVIDIA GeForce RTX 3050 6GB Laptop GPU (driver supports CUDA 13.4).

**Objective:** model definitions plus a reproducible CLI training pipeline. It consumes an M1 split
manifest, trains on TRAIN, selects and early-stops on VAL only, writes checkpoints, and records
every run in a SQLite experiment store. There is no API, no test-set evaluation, no Grad-CAM,
robustness or frontend.

Every output below came from commands executed in this session. Long scratchpad paths are shortened
to `<scratchpad>`.

> **All model metrics in this packet are pipeline sanity checks on synthetic data, not benchmark
> results.**

---

## 1. Step 0: dataset-layer follow-ups (commit `a321521`)

| Item | Result |
|---|---|
| 0a | `data/split.py`: within each class, groups are ordered by `sha256(f"{seed}\|{class}\|{group_id}")`. The largest-deficit assignment is unchanged. The algorithm is now `vb-split-v2`, and `split_info.json` still records `numpy_version`. Test `test_split_does_not_use_numpy_rng` replaces `np.random.default_rng` / `np.random.Generator` with functions that raise, and the split still succeeds. All M1 split tests still pass, including byte-identical manifests for the same seed. |
| 0b | `is_low_information` now also flags dHash popcount < `low_info_popcount_min` (8) or > `low_info_popcount_max` (56). Both bounds are configurable through settings, `ScanConfig`/`SplitConfig` and CLI flags, and are recorded in the report and manifest. New fixture `vb_synth.build_low_info`: two sky-like gradients (std ≈ 38.6, popcount 3, 2 bits apart) plus a near-flat noisy image (std ≈ 1). The test asserts all three are excluded and reported. A control with the popcount rule disabled shows the two gradients forming a false cross-class group. |
| 0c | `.gitattributes`: `* text=auto eol=lf`, plus `binary` for png/jpg/jpeg/bmp/webp/pt/pth/zip. The tree was renormalized (no content changes), and the CRLF warnings from earlier commits are gone. |
| 0d | `visionbench dataset verify <id> [--full]`. Fast mode: fingerprint recomputed from `files.json` vs `dataset.json`; existence and size of every recorded file; unrecorded files under `raw/`. `--full` also re-hashes every file. Mismatches are listed and the command exits 1. Tests cover size change, missing file, extra file, same-size content change (detected only by `--full`), fingerprint mismatch, and CLI exit codes. |

Two existing tests used hand-made hashes with popcount 6–8. The new rule correctly treats those as
low-information, so I changed them to informative hashes (popcount 32). This was a fixture change;
behaviour did not change.

## 2. Files created and changed (since M1 packet `9581b24`)
```
 .env.example                              |   3 +
 .gitattributes                            |  11 +
 ARCHITECTURE.md                           | 100 +++++-
 CHANGELOG.md                              |  38 +++
 EXPERIMENT_PROTOCOL.md                    | 138 ++++++--
 IMPLEMENTATION_PLAN.md                    |   4 +-
 PROJECT_STATUS.md                         |  77 ++---
 README.md                                 |  68 +++-
 pyproject.toml                            |   2 +
 requirements/lock-dev-cu130.txt           |   3 +-
 src/visionbench/cli.py                    | 178 ++++++++++-
 src/visionbench/core/config.py            |   4 +-
 src/visionbench/data/grouping.py          |  41 ++-
 src/visionbench/data/report.py            |   2 +
 src/visionbench/data/scan.py              |   6 +
 src/visionbench/data/split.py             |  37 ++-
 src/visionbench/data/store.py             |  54 +++-
 src/visionbench/models/__init__.py        |   1 +
 src/visionbench/models/baseline.py        |  62 ++++
 src/visionbench/models/pretrained.py      | 108 +++++++
 src/visionbench/models/registry.py        | 192 +++++++++++
 src/visionbench/models/small_cnn.py       |  45 +++
 src/visionbench/tracking/__init__.py      |   1 +
 src/visionbench/tracking/store.py         | 243 ++++++++++++++
 src/visionbench/training/__init__.py      |   1 +
 src/visionbench/training/checkpoints.py   |  51 +++
 src/visionbench/training/config.py        | 114 +++++++
 src/visionbench/training/data.py          | 117 +++++++
 src/visionbench/training/loop.py          | 193 +++++++++++
 src/visionbench/training/preprocessing.py | 151 +++++++++
 src/visionbench/training/runner.py        | 514 ++++++++++++++++++++++++++++++
 tests/conftest.py                         |  24 ++
 tests/data/test_grouping.py               |  12 +-
 tests/data/test_lowinfo_verify.py         | 146 +++++++++
 tests/data/test_scan.py                   |   2 +-
 tests/data/test_store.py                  |   2 +-
 tests/training/conftest.py                |  58 ++++
 tests/training/test_isolation.py          | 103 ++++++
 tests/training/test_models.py             | 143 +++++++++
 tests/training/test_training.py           | 321 +++++++++++++++++++
 tests/{data/conftest.py => vb_synth.py}   |  88 +++--
 41 files changed, 3317 insertions(+), 141 deletions(-)
```
Plus this packet and a one-line IMPLEMENTATION_PLAN fix (checkpoints live under `experiments/`, not
`artifacts/`), both in the final commit.

**No API endpoints, test-set evaluation, Grad-CAM, robustness or frontend code was added.**
`api/main.py` is unchanged since M1.

**Dependencies:**
- `sqlalchemy>=2.0,<3` is new (2.1.6 installed). It is the DB layer planned since M0 for the
  SQLite store.
- `pyyaml>=6.0,<7` is declared but was already installed through `uvicorn[standard]`. It is needed
  for the YAML training configs, and the stdlib has no YAML parser.
- The lock file was refreshed.

## 3. Design summary
- **Test split unreachable.** All manifest access in training goes through
  `training.data.select_entries(manifest, split)`, which allows only `train`/`val`. The dataset
  class, loader factory, normalization stats, class weights and baseline all use it. Images are
  opened only via `training.data.load_image`, which tests patch. The majority baseline opens no
  images.
- **Pre-flight before any DB row:** dataset exists, fast verify passes, split id is well-formed,
  manifest exists, and the manifest's `dataset_id` and fingerprint match `dataset.json`.
- **Determinism:**
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before building the model;
  - `set_seed(seed)` with cudnn deterministic, then strict `torch.use_deterministic_algorithms(True)`;
  - the DataLoader `Generator` is seeded, with a `worker_init_fn`;
  - SmallCNN uses mean-GAP and non-overlapping max-pool, so its backward passes have deterministic
    kernels.
- **Normalization:**
  - Pretrained models use their weights' transforms (ResNet18: 256→224 bilinear;
    EfficientNet-B0: 256→224 **bicubic**; ImageNet mean/std).
  - Scratch models use train-only mean/std at the model's resolution, with no augmentation. These
    are cached as `splits/<id>/norm_stats_train_<N>px.json`, keyed by resolution and a train
    fingerprint.
- **Checkpoints:** `best.pt` (best val metric) and `last.pt` each hold a state_dict, epoch and
  metrics, with a sibling `.json` (metadata and sha256). `load_checkpoint` uses `weights_only=True`
  and refuses paths outside `<data_root>/experiments/`.
- **Weights:** torch.hub with hub dir `<data_root>/cache/torch`, `check_hash=True` on download and
  `weights_only=True`. The file's sha256 must match the hash prefix in its name, and the enum, path
  and sha256 are recorded. Failure is an error. `--allow-weight-fallback` trains from scratch and
  records `weights=scratch` plus the reason, with a `WEIGHT FALLBACK` warning in `train.log`.
- **Status:** the store enforces `pending → running → completed | failed | cancelled`. Exceptions
  store `"<Type>: <msg>"` and keep the partial history and checkpoints. `KeyboardInterrupt` gives
  `cancelled` (CLI exit 130). Every run-dir file is registered as an artifact with sha256 and size.

## 4. Lint and tests (final run on the committed code)
```
> ruff check .
All checks passed!
exit=0
> ruff format --check .
60 files already formatted
exit=0
```
`pytest -q --cov=visionbench` (normal):
```
...........................................s............................ [ 51%]
.....................................................................    [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.14 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 9.85 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
src\visionbench\__init__.py                     1      0      0      0   100%
src\visionbench\api\__init__.py                 0      0      0      0   100%
src\visionbench\api\main.py                    55      1      6      0    98%
src\visionbench\cli.py                        230      5     44      4    97%
src\visionbench\core\__init__.py                0      0      0      0   100%
src\visionbench\core\config.py                 35      0      2      0   100%
src\visionbench\core\env_info.py               11      0      0      0   100%
src\visionbench\core\errors.py                  9      0      0      0   100%
src\visionbench\core\jsonio.py                 13      0      0      0   100%
src\visionbench\core\seeding.py                22      0      4      1    96%
src\visionbench\data\__init__.py                0      0      0      0   100%
src\visionbench\data\grouping.py               84      1     20      1    98%
src\visionbench\data\hashing.py                52      0     18      0   100%
src\visionbench\data\records.py                51      1      0      0    98%
src\visionbench\data\report.py                 29      0      6      0   100%
src\visionbench\data\scan.py                  204      9     68      7    94%
src\visionbench\data\split.py                 145      5     44      2    96%
src\visionbench\data\store.py                 126      2     32      2    97%
src\visionbench\evaluation\__init__.py          0      0      0      0   100%
src\visionbench\explain\__init__.py             0      0      0      0   100%
src\visionbench\jobs\__init__.py                0      0      0      0   100%
src\visionbench\models\__init__.py              0      0      0      0   100%
src\visionbench\models\baseline.py             41      1      6      1    96%
src\visionbench\models\pretrained.py           53      6      4      0    89%
src\visionbench\models\registry.py             98      4     40      4    94%
src\visionbench\models\small_cnn.py            30      0      2      0   100%
src\visionbench\robustness\__init__.py          0      0      0      0   100%
src\visionbench\tracking\__init__.py            0      0      0      0   100%
src\visionbench\tracking\store.py             137      1     16      1    99%
src\visionbench\training\__init__.py            0      0      0      0   100%
src\visionbench\training\checkpoints.py        22      0      2      0   100%
src\visionbench\training\config.py             83      5     18      4    91%
src\visionbench\training\data.py               52      3      2      0    94%
src\visionbench\training\loop.py              113      5     32      4    94%
src\visionbench\training\preprocessing.py      75      3     14      5    91%
src\visionbench\training\runner.py            232      8     42      5    95%
-----------------------------------------------------------------------------
TOTAL                                        2003     60    422     41    96%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
140 passed, 1 skipped, 1 warning in 99.08s (0:01:39)
```
`$env:CUDA_VISIBLE_DEVICES="-1"; pytest -q --cov=visionbench` (GPU hidden). The output is identical
apart from these lines (checked with `diff` after removing the PERF, torch and summary lines):
```
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.98 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 7.92 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
140 passed, 1 skipped, 1 warning in 81.48s (0:01:21)
```
**Not covered by tests:**
- the real network download in `models/pretrained.py` (by design: no downloads in pytest; it was
  exercised in demo b);
- `num_workers > 0`;
- AMP on CUDA;
- the sgd/head_only branches beyond what is listed below.

### Requirement → test map
| Required test | Test(s) |
|---|---|
| Test isolation (patched loader, full smoke run, zero test relpaths opened) | `training/test_isolation.py::test_training_never_opens_test_images[small_cnn\|majority_baseline]`. It asserts no test relpath was opened, every train and val image was opened (small_cnn), and nothing was opened (baseline). |
| Loader factory rejects "test" | `::test_loader_factory_rejects_non_train_val[test\|TEST\|all\|]` (also covers `build_dataset`, `select_entries`, `split_class_counts`) |
| Overfit sanity | `training/test_models.py::test_small_cnn_overfits_single_batch` (8 images, 150 Adam steps, loss < 0.05) |
| Smoke: 4 models, history, checkpoints/JSON, DB rows, artifacts with sha256 | `training/test_training.py::test_smoke_each_model[...]`. It re-hashes every registered artifact. The torchvision models use scratch weights, 32 px, `max_steps=2`. |
| CPU determinism (identical losses and `best.pt` tensors) | `::test_cpu_determinism_identical_runs` (3 epochs, exact equality) |
| Train-only normalization (val/test changes: same; train changes: different) | `test_isolation.py::test_norm_stats_depend_on_train_only`. The val/test entries are pointed at non-existent files, which proves they are never opened. |
| Class weights from train counts only | `test_isolation.py::test_class_weights_from_train_counts_only`, `test_training.py::test_class_weighted_loss_recorded` |
| Early stopping on plateau, best = best val epoch | `::test_early_stopping_on_plateau_and_best_epoch` (`min_delta=100` makes every epoch after the first a non-improvement; stops after epoch 3, `best.json` epoch 1), `::test_early_stopping_unit` |
| Failure path | `::test_injected_failure_marks_failed_keeps_partial_history` (status failed, error stored, 1 epoch row, 1 history line, `last.pt` and log kept, artifacts registered); `::test_keyboard_interrupt_marks_cancelled` |
| Fingerprint mismatch aborts | `::test_fingerprint_mismatch_aborts_before_any_record` (no DB rows); `::test_snapshot_tampering_aborts_training` (fast verify) |
| Checkpoint reload (`weights_only=True`) reproduces val loss | `::test_best_checkpoint_reload_reproduces_val_loss` (fresh model, different seed, rel 1e-6); also refuses paths outside `experiments/` |
| Weight fallback | `test_models.py::test_weight_fallback_requires_explicit_flag`, `test_training.py::test_weight_fallback_in_runner` (default: failed with message; flag: completed, `weights_source=scratch`, reason in config, `WEIGHT FALLBACK` in log) |
| Pretrained load path without network | `test_models.py::test_pretrained_load_path_records_sha_and_preprocessing` (fake cached file whose name hash matches), `::test_pretrained_hash_mismatch_is_rejected` |

## 5. Demos (outside pytest; data root `<scratchpad>/m2/var`)

### a. Learnable synthetic dataset → baseline vs small_cnn (CPU and CUDA)
The data comes from `vb_synth.build_shapes`: 3 classes (circle/square/triangle) × 120 images,
64 px. Each image has a random shape size, position, and colour on a noisy random background.

```
$ python make_shapes.py <scratchpad>/m2/src
generated 360 images in 3 classes ('circle', 'square', 'triangle') at <scratchpad>\m2\src

$ visionbench dataset import <scratchpad>/m2/src --name shapes-demo
imported dataset ds-001fffaa50a2 (shapes-demo)
  path: <scratchpad>\m2\var\datasets\ds-001fffaa50a2
  fingerprint: 10c842287c954667a97ad80d27903e9bf0cdfafe5576c8f9fd71c34710cf365c
  classes: 3 | images: 360 | bytes: 2787502
  class counts: circle=120, square=120, triangle=120 | imbalance ratio: 1.0
  issues: none
  duplicate groups (>1 member): 2 (4 files) | exact duplicate files: 0 | label-conflict groups: 0
  perceptual check skipped (low information): 0 (std < 3.0, degenerate dHash, or popcount outside [8, 56]; dHash threshold 5)
[exit 0]

$ visionbench dataset verify ds-001fffaa50a2 --full
verify ds-001fffaa50a2: full (sizes + sha256); 360 files checked
  OK: snapshot matches its records
[exit 0]

$ visionbench dataset split ds-001fffaa50a2 --seed 0
split sp-5c3e3bf5d61e for dataset ds-001fffaa50a2: new
  manifest: <scratchpad>\m2\var\datasets\ds-001fffaa50a2\splits\sp-5c3e3bf5d61e\manifest.json
  manifest sha256: 7b090995026cfc57f90dd402354eea474f19c248b6acc549174b7c066f38fed6
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    circle: 84/18/18/120
    square: 84/18/18/120
    triangle: 84/18/18/120
  totals: train=252, val=54, test=54
  excluded (label conflicts): 0 files in 0 groups
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 358 groups, 360 sha256 values)
[exit 0]
```
The import found 2 near-duplicate pairs among the randomly generated shapes. Each group is assigned
whole to one split.

Config `small_cnn.yaml`: `model: small_cnn, weights: scratch, input_size: 64, epochs: 15,
batch_size: 32, optimizer: adamw, lr: 0.001, weight_decay: 0.0001, scheduler: cosine,
early_stopping: {monitor: val_loss, patience: 5, min_delta: 0.0}, seed: 0, num_workers: 0`.
```
$ visionbench train --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e --model majority_baseline --device cpu
experiment exp-20261026-202532-8a255a running (majority_baseline, seed=0)
  epoch   1  train_loss=1.0986  val_loss=1.0986  val_acc=0.3333  val_bal_acc=0.3333  lr=-  0.00s
experiment exp-20261026-202532-8a255a: completed (stopped: baseline_fit; best epoch 1, best val_loss=1.0986)
[exit 0]

$ visionbench train --config small_cnn.yaml --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e --device cpu
experiment exp-20261026-202542-1c7e11 running (small_cnn, seed=0)
  epoch   1  train_loss=1.2071  val_loss=1.1079  val_acc=0.4074  val_bal_acc=0.4074  lr=1.00e-03  2.23s  *best
  epoch   2  train_loss=1.0040  val_loss=1.0995  val_acc=0.4074  val_bal_acc=0.4074  lr=9.89e-04  1.98s  *best
  epoch   3  train_loss=0.8621  val_loss=0.9236  val_acc=0.6111  val_bal_acc=0.6111  lr=9.57e-04  2.12s  *best
  epoch   4  train_loss=0.6667  val_loss=0.8155  val_acc=0.5926  val_bal_acc=0.5926  lr=9.05e-04  2.02s  *best
  epoch   5  train_loss=0.5350  val_loss=0.5694  val_acc=0.8333  val_bal_acc=0.8333  lr=8.35e-04  2.04s  *best
  epoch   6  train_loss=0.4073  val_loss=0.3924  val_acc=0.8889  val_bal_acc=0.8889  lr=7.50e-04  2.05s  *best
  epoch   7  train_loss=0.2772  val_loss=0.2497  val_acc=0.9630  val_bal_acc=0.9630  lr=6.55e-04  2.05s  *best
  epoch   8  train_loss=0.2265  val_loss=0.2022  val_acc=0.9630  val_bal_acc=0.9630  lr=5.52e-04  2.07s  *best
  epoch   9  train_loss=0.1615  val_loss=0.2609  val_acc=0.8704  val_bal_acc=0.8704  lr=4.48e-04  2.11s
  epoch  10  train_loss=0.1239  val_loss=0.0810  val_acc=1.0000  val_bal_acc=1.0000  lr=3.45e-04  2.04s  *best
  epoch  11  train_loss=0.1029  val_loss=0.1188  val_acc=0.9630  val_bal_acc=0.9630  lr=2.50e-04  2.06s
  epoch  12  train_loss=0.0930  val_loss=0.0665  val_acc=1.0000  val_bal_acc=1.0000  lr=1.65e-04  2.07s  *best
  epoch  13  train_loss=0.0957  val_loss=0.0592  val_acc=1.0000  val_bal_acc=1.0000  lr=9.55e-05  2.05s  *best
  epoch  14  train_loss=0.0909  val_loss=0.0535  val_acc=1.0000  val_bal_acc=1.0000  lr=4.32e-05  2.02s  *best
  epoch  15  train_loss=0.0673  val_loss=0.0539  val_acc=1.0000  val_bal_acc=1.0000  lr=1.09e-05  2.06s
experiment exp-20261026-202542-1c7e11: completed (stopped: epochs_completed; best epoch 14, best val_loss=0.0535)
[exit 0]

$ visionbench train --config small_cnn.yaml --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e --device cuda   (run 1)
experiment exp-20261026-202624-fef3d3 running (small_cnn, seed=0)
  epoch   1  train_loss=1.2297  val_loss=1.1029  val_acc=0.4259  val_bal_acc=0.4259  lr=1.00e-03  1.57s  *best
  epoch   2  train_loss=1.0563  val_loss=1.0984  val_acc=0.4444  val_bal_acc=0.4444  lr=9.89e-04  0.63s  *best
  epoch   3  train_loss=0.9024  val_loss=0.9230  val_acc=0.6111  val_bal_acc=0.6111  lr=9.57e-04  0.63s  *best
  epoch   4  train_loss=0.7309  val_loss=0.8524  val_acc=0.5741  val_bal_acc=0.5741  lr=9.05e-04  0.62s  *best
  epoch   5  train_loss=0.6078  val_loss=0.5671  val_acc=0.8333  val_bal_acc=0.8333  lr=8.35e-04  0.65s  *best
  epoch   6  train_loss=0.4461  val_loss=0.4794  val_acc=0.8519  val_bal_acc=0.8519  lr=7.50e-04  0.62s  *best
  epoch   7  train_loss=0.3433  val_loss=0.2627  val_acc=0.9630  val_bal_acc=0.9630  lr=6.55e-04  0.62s  *best
  epoch   8  train_loss=0.2540  val_loss=0.2121  val_acc=0.9815  val_bal_acc=0.9815  lr=5.52e-04  0.65s  *best
  epoch   9  train_loss=0.1817  val_loss=0.1396  val_acc=1.0000  val_bal_acc=1.0000  lr=4.48e-04  0.67s  *best
  epoch  10  train_loss=0.1400  val_loss=0.1035  val_acc=1.0000  val_bal_acc=1.0000  lr=3.45e-04  0.70s  *best
  epoch  11  train_loss=0.1141  val_loss=0.0927  val_acc=1.0000  val_bal_acc=1.0000  lr=2.50e-04  0.75s  *best
  epoch  12  train_loss=0.0966  val_loss=0.0733  val_acc=1.0000  val_bal_acc=1.0000  lr=1.65e-04  0.76s  *best
  epoch  13  train_loss=0.1021  val_loss=0.0670  val_acc=1.0000  val_bal_acc=1.0000  lr=9.55e-05  0.76s  *best
  epoch  14  train_loss=0.1106  val_loss=0.0643  val_acc=1.0000  val_bal_acc=1.0000  lr=4.32e-05  0.80s  *best
  epoch  15  train_loss=0.0862  val_loss=0.0669  val_acc=1.0000  val_bal_acc=1.0000  lr=1.09e-05  0.80s
experiment exp-20261026-202624-fef3d3: completed (stopped: epochs_completed; best epoch 14, best val_loss=0.0643)
[exit 0]
```
(The `run dir:` line printed after each `running` line is omitted here.)

**Result (pipeline sanity check on synthetic data, not a benchmark result):** val balanced accuracy
at the best epoch was:
- majority baseline: **0.3333**;
- small_cnn on CPU: **1.0000** (epoch 14);
- small_cnn on CUDA: **1.0000** (epoch 14).

small_cnn > baseline holds on both devices. The CPU and CUDA histories differ from each other
(e.g. epoch 1 train_loss 1.2071 vs 1.2297) because the kernels differ. The val split has only 54
images, so these numbers are coarse. **The test split was not evaluated** (that is M3).

### b. Pretrained weights on CUDA (2 epochs)
Config `pretrained.yaml`: `weights: imagenet, finetune: full, epochs: 2, batch_size: 32,
optimizer: adamw, lr: 0.0001, weight_decay: 0.0001, seed: 0, device: cuda, num_workers: 0`.
```
$ visionbench train --config pretrained.yaml --model resnet18 --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e
experiment exp-20261026-202742-757de7 running (resnet18, seed=0)
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to <scratchpad>\m2\var\cache\torch\checkpoints\resnet18-f37072fd.pth
  epoch   1  train_loss=0.6174  val_loss=0.0882  val_acc=1.0000  val_bal_acc=1.0000  lr=1.00e-04  3.04s  *best
  epoch   2  train_loss=0.0322  val_loss=0.0036  val_acc=1.0000  val_bal_acc=1.0000  lr=1.00e-04  2.32s  *best
experiment exp-20261026-202742-757de7: completed (stopped: epochs_completed; best epoch 2, best val_loss=0.0036)
[exit 0]
  weights=imagenet enum=ResNet18_Weights.IMAGENET1K_V1 sha256=f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
  weight file: <scratchpad>\m2\var\cache\torch\checkpoints\resnet18-f37072fd.pth
  input_size=224 resize=256 crop=224 interpolation=bilinear
  normalization source=imagenet_weights mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
  params=11,178,051 trainable=11,178,051 target_layer=layer4
  val transforms: ['Resize(size=256, interpolation=bilinear, max_size=None, antialias=True)', 'CenterCrop(size=(224, 224))', 'ToTensor()', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))']

$ visionbench train --config pretrained.yaml --model efficientnet_b0 --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e
experiment exp-20261026-202808-8c6076 running (efficientnet_b0, seed=0)
Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth" to <scratchpad>\m2\var\cache\torch\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth
  epoch   1  train_loss=1.0242  val_loss=0.9889  val_acc=0.6667  val_bal_acc=0.6667  lr=1.00e-04  4.28s  *best
  epoch   2  train_loss=0.7785  val_loss=0.7722  val_acc=0.8519  val_bal_acc=0.8519  lr=1.00e-04  3.39s  *best
experiment exp-20261026-202808-8c6076: completed (stopped: epochs_completed; best epoch 2, best val_loss=0.7722)
[exit 0]
  weights=imagenet enum=EfficientNet_B0_Weights.IMAGENET1K_V1 sha256=7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934
  weight file: <scratchpad>\m2\var\cache\torch\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth
  input_size=224 resize=256 crop=224 interpolation=bicubic
  normalization source=imagenet_weights mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
  params=4,011,391 trainable=4,011,391 target_layer=features.8
  val transforms: ['Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)', 'CenterCrop(size=(224, 224))', 'ToTensor()', 'Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))']

$ sha256sum <scratchpad>/m2/var/cache/torch/checkpoints/*.pth
7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934 *.../efficientnet_b0_rwightman-7f5810bc.pth
f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec *.../resnet18-f37072fd.pth
```
(The indented lines after each run are read from that run's `config.json` by a small script.)

- The pretrained load path works for both models. The recorded sha256 matches an independent
  `sha256sum` and the hash prefix in each file name.
- **Discrepancy with the task wording ("already cached"):** the weights were cached in
  `~/.cache/torch` (from M0), but the new policy uses `<data_root>/cache/torch`. They were therefore
  **downloaded again** from download.pytorch.org.
- Input resolution is 224 for both. That is a resolution confound against small_cnn at 64 px, so
  any comparison must flag it.
- EfficientNet-B0 reached val balanced accuracy 0.8519 after 2 epochs at lr 1e-4. That is a sanity
  check on synthetic data, not a tuned result.

### c. GPU determinism: the same small_cnn CUDA config run twice
```
$ visionbench train --config small_cnn.yaml --dataset-id ds-001fffaa50a2 --split-id sp-5c3e3bf5d61e --device cuda   (run 2)
experiment exp-20261026-202702-168e29 running (small_cnn, seed=0)
  epoch   1  train_loss=1.2297  val_loss=1.1029  val_acc=0.4259  val_bal_acc=0.4259  lr=1.00e-03  1.39s  *best
  ... (epochs 2–15 print the same metric values as run 1 above; only the times differ)
  epoch  15  train_loss=0.0862  val_loss=0.0669  val_acc=1.0000  val_bal_acc=1.0000  lr=1.09e-05  0.78s
experiment exp-20261026-202702-168e29: completed (stopped: epochs_completed; best epoch 14, best val_loss=0.0643)
[exit 0]

$ python compare_runs.py exp-20261026-202624-fef3d3 exp-20261026-202702-168e29
run A exp-20261026-202624-fef3d3: 15 epochs | run B exp-20261026-202702-168e29: 15 epochs
per-epoch metrics bitwise identical: True
  max |A-B| train_loss             = 0.000e+00
  max |A-B| val_loss               = 0.000e+00
  max |A-B| val_accuracy           = 0.000e+00
  max |A-B| val_balanced_accuracy  = 0.000e+00
best.pt epoch A=14 B=14; tensors equal: 26/26
```
**Result:** for this configuration the two CUDA runs were **bitwise identical**. They ran in separate
processes on the same GPU, driver and library versions, with strict deterministic algorithms on, and
no op raised. This is one observation. It does not establish GPU determinism for resnet18,
efficientnet_b0, AMP, `num_workers > 0`, or other hardware, and none of those were run twice.

### d. `visionbench experiments list`
```
$ visionbench experiments list
exp-20261026-202532-8a255a  completed  majority_baseline  weights=none      seed=0  best_epoch=1  best=1.0986  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:25:32+00:00
exp-20261026-202542-1c7e11  completed  small_cnn          weights=scratch   seed=0  best_epoch=14  best=0.0535  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:25:42+00:00
exp-20261026-202624-fef3d3  completed  small_cnn          weights=scratch   seed=0  best_epoch=14  best=0.0643  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:26:24+00:00
exp-20261026-202702-168e29  completed  small_cnn          weights=scratch   seed=0  best_epoch=14  best=0.0643  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:27:02+00:00
exp-20261026-202742-757de7  completed  resnet18           weights=imagenet  seed=0  best_epoch=2  best=0.0036  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:27:42+00:00
exp-20261026-202808-8c6076  completed  efficientnet_b0    weights=imagenet  seed=0  best_epoch=2  best=0.7722  ds-001fffaa50a2/sp-5c3e3bf5d61e  2026-10-26T20:28:08+00:00
[exit 0]
```
(`best` is the monitored metric, `val_loss`.)

`visionbench experiments show exp-20261026-202624-fef3d3` printed the full record: status
completed, `best_epoch` 14, `best_metric` 0.06430718192347774, `train_seconds` 13.13, all 15 epoch
rows, and 8 artifacts with sha256 (best/last `.pt` + `.json`, `config.json`, `env.json`,
`history.jsonl`, `train.log`). The unabridged transcripts are in `<scratchpad>/m2/demo_*.txt`.

## 6. git log --oneline
```
899248f M2 docs: protocol, architecture, README, status, changelog
42f638b M2: CLI training from split manifests with SQLite experiment tracking
3548826 M2: model registry (baseline, SmallCNN, ResNet18, EfficientNet-B0)
a321521 M2 step 0: hash-ordered splits, popcount low-info guard, dataset verify
9581b24 M1 review packet
e049491 M1 docs: protocol, architecture, README, status, changelog
34930b6 M1: `visionbench dataset import|report|split|list` CLI
57d7b8f M1: dataset library - validation, dHash duplicate groups, snapshots, splits
d3d52b5 M0 follow-ups: generic ready error, dev lock file, CUDA state in test log
4165e71 M0: foundation baseline
```
One more commit adds this packet and the IMPLEMENTATION_PLAN fix. Everything is local only: no
remote, nothing pushed.

## 7. Known limitations
- **GPU determinism:** shown for one small_cnn configuration only. Strict mode will make a run
  *fail* on a CUDA op without a deterministic kernel; none occurred in these demos.
- **Untested settings:** `num_workers > 0`, AMP, and SGD + head_only on real data have no tests or
  demos. With workers on Windows, a monkeypatched `load_image` would not apply inside spawned
  workers. The isolation guarantee there still rests on `select_entries`.
- **Hard kills:** a hard kill can leave an experiment row `running`. M4's startup recovery is
  required.
- **head_only:** BatchNorm running statistics in the frozen backbone still update.
- **Metrics:** balanced accuracy ignores classes absent from val. The val splits in the demos are
  small (54 images), so the metrics are coarse.
- **Weights cache:** weights live in a per-data-root cache, so each data root downloads its own copy
  (65 MB here).
- **Norm-stats cache location:** the cache file lives in the split directory, which is otherwise
  write-once. The manifest bytes are unaffected, and a test checks this.
- **Pretrained resolution:** pretrained models must use their weights' 224 crop. A run that falls
  back to scratch uses the configured or default size, so a fallback run may differ in resolution
  from the intended pretrained run. Both are recorded.
- **Duplicates in the shapes demo:** 2 near-duplicate groups were found among randomly generated
  shapes, showing dHash merging visually similar synthetic images. They are kept within one split.

## 8. Open questions for the supervisor
1. Should the pretrained-weight cache be shared across data roots, or pre-seeded from
   `~/.cache/torch` with a verified sha256 copy, to avoid re-downloading per data root?
2. Is `val_loss` the right default early-stopping monitor, or should `val_balanced_accuracy` be the
   default for imbalanced datasets?
3. Should GPU determinism be a CI-style check (two runs per model on CUDA) in M3/M8, given that it
   is currently a one-off demo?
4. Should head_only fine-tuning also freeze BatchNorm statistics, by putting the backbone in eval
   mode?
5. Should the norm-stats cache move out of the split directory, e.g. to
   `cache/norm_stats/<split_id>/`?

## 9. Recommended next task
**M3:** a one-time test-set evaluation of selected checkpoints, through a separate evaluation entry
point that is the *only* code allowed to open test images. It needs a metrics module (per-class
precision/recall/F1, confusion matrix, ECE, bootstrap CIs) and comparison reports that include the
majority baseline and flag input-resolution confounds.
