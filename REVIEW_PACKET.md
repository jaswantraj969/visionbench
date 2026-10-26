# Review Packet — VB-M3

**Task:** VB-M3: evaluation module, test-set protocol, model comparison
**Date:** 2026-10-27
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130; torchvision
0.29.1+cu130; NumPy 2.4.6; scikit-learn 1.9.1 (new); SQLAlchemy 2.1.6; NVIDIA GeForce RTX 3050 6GB
Laptop GPU.

**Objective:** an evaluation layer with these parts:
- a separate entry point that is the only code allowed to read test images;
- metrics with explicit handling of undefined values;
- efficiency measurement;
- bootstrap CIs;
- comparison reports that make confounds visible.

It was developed and demonstrated on synthetic data only. There is no API, Grad-CAM, robustness or
frontend.

Every output below came from commands executed in this session. Scratchpad paths are shortened to
`<scratchpad>`.

> **All model numbers in this packet are a pipeline demo on synthetic data, not benchmark
> results.**

---

## 0. EXIF question (answered from the code; not changed in this task)

**No.** M1's dHash computation does not apply EXIF orientation.
`src/visionbench/data/scan.py::inspect_image` does `Image.open` → `verify()` → reopen → `load()` →
`im.convert("L")` → `dhash(gray)`, and never calls `ImageOps.exif_transpose`. The training and
evaluation loader `src/visionbench/training/data.py::load_image` (`Image.open(path)` →
`im.convert("RGB")`) does not call it either. So duplicate detection, training and evaluation all
consistently use the **stored pixel orientation** and ignore the EXIF Orientation tag. This is now
listed under the protocol's known limitations.

## 1. Step 0: M2 follow-ups (commit `c508d5f`)

| Item | Result |
|---|---|
| 0a | `VISIONBENCH_TORCH_HOME` → `Settings.torch_home` / `resolved_torch_home` (default unchanged: `<data_root>/cache/torch`). The runner passes it as the hub dir and records it as `resolved.torch_hub_dir`. Tests: the setting resolves both ways, and a run with a custom torch home calls the weight loader with exactly that directory and records it. |
| 0b | `loop.train_mode(model)` calls `model.train()` and then puts every BatchNorm layer whose parameters are all frozen back into eval mode. `train_one_epoch` uses it every epoch. Recorded as `resolved.frozen_bn: true` and `frozen_bn_layers` (20 for ResNet18). Test: after a head_only epoch, the backbone BN `running_mean`/`running_var`/`num_batches_tracked` are bitwise unchanged while the head's weights changed. The full-fine-tune control shows the statistics *do* change without freezing. |
| 0c | The norm-stats cache moved to `<data_root>/cache/norm_stats/<dataset_id>/<split_id>/norm_stats_train_<N>px.json`. Test: the split directory's file list and every file's sha256 are identical before and after a training run, and contain only `manifest.json` + `split_info.json`. |
| 0d | EXPERIMENT_PROTOCOL §7 says that `train_loss` is class-weighted when weighting is on, while `val_loss` is always unweighted. `config.json` now records both definitions (`resolved.train_loss`, `resolved.val_loss`). |

## 2. Files created and changed (since M2 packet `bc983ee`)
```
 .env.example                                   |   4 +
 ARCHITECTURE.md                                |  61 +++-
 CHANGELOG.md                                   |  45 +++
 EXPERIMENT_PROTOCOL.md                         | 130 +++++++-
 IMPLEMENTATION_PLAN.md                         |   6 +-
 PROJECT_STATUS.md                              |  75 ++---
 README.md                                      |  35 ++-
 pyproject.toml                                 |   1 +
 requirements/lock-dev-cu130.txt                |   8 +-
 src/visionbench/cli.py                         | 131 ++++++++
 src/visionbench/core/config.py                 |  18 +-
 src/visionbench/evaluation/__init__.py         |   1 +
 src/visionbench/evaluation/compare.py          | 402 +++++++++++++++++++++++++
 src/visionbench/evaluation/data.py             | 154 ++++++++++
 src/visionbench/evaluation/efficiency.py       |  94 ++++++
 src/visionbench/evaluation/evaluate.py         | 245 +++++++++++++++
 src/visionbench/evaluation/metrics.py          | 350 +++++++++++++++++++++
 src/visionbench/models/pretrained.py           |   7 +-
 src/visionbench/tracking/store.py              |  86 +++++-
 src/visionbench/training/loop.py               |  26 +-
 src/visionbench/training/preprocessing.py      |  19 +-
 src/visionbench/training/runner.py             |  22 +-
 tests/conftest.py                              |   7 +
 tests/evaluation/test_evaluate.py              | 337 +++++++++++++++++++++
 tests/evaluation/test_metrics.py               | 198 ++++++++++++
 tests/training/test_m3_step0.py                | 111 +++++++
 tests/training/test_training.py                |   2 +-
 tests/{training/conftest.py => vb_trainenv.py} |  32 +-
 28 files changed, 2507 insertions(+), 100 deletions(-)
```
Plus this packet in the final commit.

- **Scope:** no API, Grad-CAM, robustness or frontend code was added. `training.data.select_entries`
  is unchanged (train/val only).
- **Dependency:** `scikit-learn>=1.3,<2` (1.9.1 installed), planned since M0. It pulled in scipy
  1.17.1, joblib, threadpoolctl, narwhals and cloudpickle. The lock file was refreshed.
- **Commit hygiene note:** the pure rename `tests/training/conftest.py → tests/vb_trainenv.py` was
  already staged when I made commit `9c4112d`, so it landed there while its content change landed
  in `56b33fc`. The tree at `9c4112d` alone therefore lacks the `train_env` fixture, and training
  tests fail at that exact commit. HEAD is correct. I did not rewrite history.

## 3. Design summary
- **Test access:**
  - `evaluation.data.select_eval_entries` is the only selector that returns test entries. It
    allows `val` and `test` and raises on `train` or anything else.
  - The eval loader reuses `ManifestImageDataset`, and therefore the single `load_image` choke
    point.
  - The baseline evaluates from labels alone.
- **Identical preprocessing:**
  - The val transform is rebuilt from `config.json` (`Preprocessing` + `build_transforms`), and its
    `repr` list must equal the recorded `val_transforms`, otherwise evaluation errors.
  - This check caught a real list-vs-tuple difference in my first version, which I fixed by
    restoring tuples.
  - A test shows that evaluation on val reproduces the training `val_loss` (rel 1e-5) and
    `val_balanced_accuracy` exactly.
- **Integrity before loading:**
  - the training pre-flight (fast verify + manifest fingerprint);
  - the manifest sha256 must equal the value recorded at training;
  - the checkpoint sha256 must equal its `artifacts` row;
  - `weights_only=True` load restricted to `experiments/`.
- **Checkpoint choice:** `best` (val-selected) by default. `--checkpoint last` is recorded as
  `last`, and the baseline is recorded as `baseline`.
- **Test-use accounting:** `evaluations` table (DB schema v2). A v1 database is upgraded additively;
  this was exercised on the real M2 demo DB (`schema before: [(1,)]` → `after: [(2,)]`). Repeated
  test evaluation of the same experiment + checkpoint prints a WARNING with the count. Counts appear
  in `metrics.json.test_use` and in the comparison table, plus a `repeated_test_evaluation` flag.
- **Metric policy:** undefined → `null` + machine-readable reason under its JSON path in
  `metrics.undefined`. A test walks every listed path and asserts it is null. Macro and weighted
  averages are taken over defined classes, and the excluded classes are listed.
- **Log loss:** this is sklearn's clipped log loss. During development I found that, for a diverged
  model, clipped log loss (15.25) differs from the training cross-entropy (20.00). I checked
  directly that cross-entropy computed from the evaluation logits equals the training val_loss
  exactly (19.998664 vs 19.998664), so preprocessing was identical and only clipping differed.
  `log_loss_clipped_samples` and a "lower bound" note are now reported.
- **Comparison:**
  - picks the latest evaluation of each experiment's val-selected checkpoint on the split;
  - verifies predictions against their sha256 and requires identical samples and labels;
  - never evaluates implicitly;
  - ranking uses VAL balanced accuracy at the selected epoch (competition ranks, ties marked `=`);
  - test results get no winner label.
- **Bug found and fixed (`ede73ff`):** listings were ordered by `created_at` (1 s resolution) and
  then the random-hex id. Two evaluations in the same second could therefore come back in random
  order, so `compare` could treat an older evaluation as "latest". The GPU-hidden test run happened
  to expose this. Ordering now falls back to SQLite insertion order, with a deterministic
  regression test. It did not affect the demo: the two small_cnn test evaluations were made 3.5
  minutes apart.

## 4. Lint and tests (final run on the committed code, after the fixes)
```
> ruff check .
All checks passed!
exit=0
> ruff format --check .
68 files already formatted
exit=0
```
`pytest -q --cov=visionbench` (normal):
```
...........................................s............................ [ 41%]
........................................................................ [ 82%]
...............................                                          [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.99 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 10.05 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
src\visionbench\__init__.py                     1      0      0      0   100%
src\visionbench\api\__init__.py                 0      0      0      0   100%
src\visionbench\api\main.py                    55      1      6      0    98%
src\visionbench\cli.py                        291      6     50      6    96%
src\visionbench\core\__init__.py                0      0      0      0   100%
src\visionbench\core\config.py                 45      0      4      0   100%
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
src\visionbench\evaluation\compare.py         172     10     62     15    89%
src\visionbench\evaluation\data.py             90      7     22      7    88%
src\visionbench\evaluation\efficiency.py       48      4     16      4    88%
src\visionbench\evaluation\evaluate.py        102      2     18      2    97%
src\visionbench\evaluation\metrics.py         177      6     50      5    95%
src\visionbench\explain\__init__.py             0      0      0      0   100%
src\visionbench\jobs\__init__.py                0      0      0      0   100%
src\visionbench\models\__init__.py              0      0      0      0   100%
src\visionbench\models\baseline.py             41      1      6      1    96%
src\visionbench\models\pretrained.py           51      5      4      0    91%
src\visionbench\models\registry.py             98      4     40      4    94%
src\visionbench\models\small_cnn.py            30      0      2      0   100%
src\visionbench\robustness\__init__.py          0      0      0      0   100%
src\visionbench\tracking\__init__.py            0      0      0      0   100%
src\visionbench\tracking\store.py             166      0     24      0   100%
src\visionbench\training\__init__.py            0      0      0      0   100%
src\visionbench\training\checkpoints.py        22      0      2      0   100%
src\visionbench\training\config.py             83      5     18      4    91%
src\visionbench\training\data.py               52      3      2      0    94%
src\visionbench\training\loop.py              127      5     40      4    95%
src\visionbench\training\preprocessing.py      76      2     14      4    93%
src\visionbench\training\runner.py            232      5     42      4    97%
-----------------------------------------------------------------------------
TOTAL                                        2705     84    614     73    95%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
174 passed, 1 skipped, 1 warning in 146.16s (0:02:26)
```
`$env:CUDA_VISIBLE_DEVICES="-1"; pytest -q --cov=visionbench` (GPU hidden). The output is identical
apart from these lines (checked with `diff` after removing the PERF, torch and summary lines):
```
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.11 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 10.56 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
174 passed, 1 skipped, 1 warning in 147.29s (0:02:27)
```
The **previous** GPU-hidden run, before the ordering fix, ended with
`FAILED tests/evaluation/test_evaluate.py::test_repeat_test_evaluation_is_counted_and_warned`
(`['test', 'val', ...] != ['test', 'test', ..., 'val']`). That failure is the ordering bug described
in §3, and it is reported here rather than hidden.

### Requirement → test map
| Required test | Test(s) |
|---|---|
| Training still cannot reach test | all M2 isolation tests unchanged and passing; `test_evaluate.py::test_selectors_both_ways` |
| Eval selector rejects "train" | `::test_selectors_both_ways`; the CLI's `--split train` is an argparse usage error (exit 2) |
| `--split val` opens zero test images; `--split test` opens zero train images | `::test_evaluation_opens_only_the_requested_split[val\|test]`, which asserts the opened set **equals** the split's relpaths. Baseline: zero images opened. |
| Checkpoint sha mismatch aborts; fingerprint mismatch aborts | `::test_checkpoint_sha_mismatch_aborts` (no evaluation row written), `::test_fingerprint_mismatch_aborts` |
| Metric fixtures: absent class, never-predicted class, perfect, constant majority, binary vs multiclass | `test_metrics.py::test_absent_class_and_never_predicted_class`, `::test_perfect_predictions`, `::test_constant_majority_predictor`, `::test_binary_vs_multiclass_auc`, `::test_macro_averages_match_sklearn_nan_policy` |
| AUC null with reason when a class is absent | `::test_auc_null_with_reason_when_class_absent` |
| No undefined metric reported as a number | `_walk_undefined_are_null` in every fixture test |
| Bootstrap reproducibility; paired bootstrap shares indices | `::test_bootstrap_reproducible_and_seed_dependent`, `::test_paired_bootstrap_shares_indices` (recomputes the CI from `bootstrap_indices` + per-model samples; identical models → diff CI [0, 0]) |
| Test-evaluation counting and warning | `test_evaluate.py::test_repeat_test_evaluation_is_counted_and_warned`, `::test_cli_evaluate_compare_list` |
| Compare refuses mismatched splits | `::test_compare_refuses_mismatched_splits`, `::test_compare_requires_evaluations` |
| Confound flags for 64 px scratch vs 224 px imagenet | `::test_compare_report_flags_and_paired_differences` (ResNet18 with faked "imagenet" weights, no network): flags `input_resolution_differs` ({cnn: 32, rn: 224}), `pretrained_vs_scratch`, `single_seed_per_model`, `training_budget_differs`, `repeated_test_evaluation` |
| Latency function fields; baseline handling | `::test_measure_latency_fields`, `::test_baseline_efficiency_not_applicable` |
| Evaluation preprocessing = validation preprocessing | `::test_val_evaluation_reproduces_training_val_loss` |
| Step 0 | `training/test_m3_step0.py` (6 tests) |
| Other | `::test_schema_upgrade_from_v1`, `::test_evaluations_ordered_by_insertion_within_same_timestamp`, `::test_competition_ranks_mark_ties`, `test_metrics.py::test_log_loss_reports_clipping` |

## 5. Demos (pipeline demo on synthetic data — not benchmark results)
These use the M2 synthetic shapes dataset `ds-001fffaa50a2` (3 classes × 120 images, 64 px; split
`sp-5c3e3bf5d61e`, with 54 test images, 18 per class) and the M2 experiments. The data root is
`<scratchpad>/m2/var`, whose DB was schema v1 and was upgraded to v2 on first open.

### a. Test evaluation of 4 experiments (CPU)
```
schema before: [(1,)]
$ visionbench evaluate exp-20261026-202532-8a255a --split test --device cpu
evaluation ev-20261026-211143-ac2902: experiment exp-20261026-202532-8a255a (majority_baseline) on TEST with checkpoint 'baseline' (epoch 1, sha256 d87f9cc619c8...)
  n=54  accuracy=0.3333 [0.2037, 0.4630]
  balanced_accuracy=0.3333 [0.3333, 0.3333]  (excluded absent classes: none)
  macro_f1=0.1667 [0.1128, 0.2110]  weighted_f1=0.1667  log_loss=1.0986
  roc_auc_ovr_macro=0.5000  ap_macro=0.3333  ece=0.0000
  undefined (null): per_class.square.precision=class_never_predicted, per_class.triangle.precision=class_never_predicted
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 0
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202532-8a255a\evaluations\ev-20261026-211143-ac2902
[exit 0]

$ visionbench evaluate exp-20261026-202542-1c7e11 --split test --device cpu
evaluation ev-20261026-211200-222430: experiment exp-20261026-202542-1c7e11 (small_cnn) on TEST with checkpoint 'best' (epoch 14, sha256 1242e24090c2...)
  n=54  accuracy=1.0000 [1.0000, 1.0000]
  balanced_accuracy=1.0000 [1.0000, 1.0000]  (excluded absent classes: none)
  macro_f1=1.0000 [1.0000, 1.0000]  weighted_f1=1.0000  log_loss=0.0339
  roc_auc_ovr_macro=1.0000  ap_macro=1.0000  ece=0.0325
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 0
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202542-1c7e11\evaluations\ev-20261026-211200-222430
[exit 0]

$ visionbench evaluate exp-20261026-202742-757de7 --split test --device cpu
evaluation ev-20261026-211312-1ea6b4: experiment exp-20261026-202742-757de7 (resnet18) on TEST with checkpoint 'best' (epoch 2, sha256 e0e68ef98488...)
  n=54  accuracy=1.0000 [1.0000, 1.0000]
  balanced_accuracy=1.0000 [1.0000, 1.0000]  (excluded absent classes: none)
  macro_f1=1.0000 [1.0000, 1.0000]  weighted_f1=1.0000  log_loss=0.0016
  roc_auc_ovr_macro=1.0000  ap_macro=1.0000  ece=0.0016
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 0
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202742-757de7\evaluations\ev-20261026-211312-1ea6b4
[exit 0]

$ visionbench evaluate exp-20261026-202808-8c6076 --split test --device cpu
evaluation ev-20261026-211449-9cdadc: experiment exp-20261026-202808-8c6076 (efficientnet_b0) on TEST with checkpoint 'best' (epoch 2, sha256 30da4054a3d6...)
  n=54  accuracy=0.7407 [0.6111, 0.8519]
  balanced_accuracy=0.7407 [0.6303, 0.8380]  (excluded absent classes: none)
  macro_f1=0.7293 [0.5963, 0.8375]  weighted_f1=0.7293  log_loss=0.8403
  roc_auc_ovr_macro=0.9599  ap_macro=0.9432  ece=0.2736
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 0
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202808-8c6076\evaluations\ev-20261026-211449-9cdadc
[exit 0]

schema after: [(2,)]
```
The following excerpts of `metrics.json` (selected fields) were read from the files by a small
script.
```
{"eval_id": "ev-20261026-211143-ac2902", "model": "majority_baseline", "split": "test", "checkpoint": {"epoch": 1, "file": "baseline.json", "name": "baseline", "sha256": "d87f9cc619c8d839b5233328b9e87040e4234e6055c9a13949906e5e5c9eeb5d"}, "selection": {"best_epoch": 1, "best_value": 1.098612288668109, "monitor": "val_loss", "selected_on": "val"}, "test_use": {"prior_test_evaluations_experiment": 0, "prior_test_evaluations_same_checkpoint": 0, "test_evaluation_number": 1}, "n": 54, "accuracy": 0.3333333333333333, "balanced_accuracy": 0.3333333333333333, "macro_f1": 0.16666666666666666, "macro_precision": 0.3333333333333333, "macro_excluded_classes": {"f1": [], "precision": ["square", "triangle"], "recall": []}, "log_loss": 1.0986122886681096, "log_loss_clipped_samples": 0, "roc_auc_ovr_macro": 0.5, "ece": 1.1102230246251565e-16, "confusion_counts": [[18, 0, 0], [18, 0, 0], [18, 0, 0]], "undefined": {"per_class.square.precision": "class_never_predicted", "per_class.triangle.precision": "class_never_predicted"}, "bootstrap_ci": {"accuracy": {"high": 0.46296296296296297, "low": 0.2037037037037037, "resamples_defined": 1000}, "balanced_accuracy": {"high": 0.3333333333333333, "low": 0.3333333333333333, "resamples_defined": 1000}, "macro_f1": {"high": 0.2109704641350211, "low": 0.11282051282051282, "resamples_defined": 1000}}}
  efficiency: latency not applicable: majority_baseline has no forward pass (constant prediction from train priors)

{"eval_id": "ev-20261026-211200-222430", "model": "small_cnn", "split": "test", "checkpoint": {"epoch": 14, "file": "best.pt", "name": "best", "sha256": "1242e24090c21cba82c062ce07f2334a842b15f6335cdd39e56c240445ee4cb4"}, "selection": {"best_epoch": 14, "best_value": 0.053498791323767766, "monitor": "val_loss", "selected_on": "val"}, "test_use": {"prior_test_evaluations_experiment": 0, "prior_test_evaluations_same_checkpoint": 0, "test_evaluation_number": 1}, "n": 54, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "macro_precision": 1.0, "macro_excluded_classes": {"f1": [], "precision": [], "recall": []}, "log_loss": 0.03392406424282924, "log_loss_clipped_samples": 0, "roc_auc_ovr_macro": 1.0, "ece": 0.032480326895021235, "confusion_counts": [[18, 0, 0], [0, 18, 0], [0, 0, 18]], "undefined": {}, "bootstrap_ci": {"accuracy": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}, "balanced_accuracy": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}, "macro_f1": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}}}
  efficiency: params=389,667 ckpt=1,569,717 B train_s=32.6 | latency on cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 8 threads, 64 px): bs1 median 3.846 ms p95 13.600 ms | bs32 median 56.669 ms p95 67.367 ms

{"eval_id": "ev-20261026-211312-1ea6b4", "model": "resnet18", "split": "test", "checkpoint": {"epoch": 2, "file": "best.pt", "name": "best", "sha256": "e0e68ef98488f2a9c7b8a178d155aece5f01dd7fb7bf7528e466a99ce423773b"}, "selection": {"best_epoch": 2, "best_value": 0.0036070705736400903, "monitor": "val_loss", "selected_on": "val"}, "test_use": {"prior_test_evaluations_experiment": 0, "prior_test_evaluations_same_checkpoint": 0, "test_evaluation_number": 1}, "n": 54, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "macro_precision": 1.0, "macro_excluded_classes": {"f1": [], "precision": [], "recall": []}, "log_loss": 0.0015924800337523257, "log_loss_clipped_samples": 0, "roc_auc_ovr_macro": 1.0, "ece": 0.0015878712462101818, "confusion_counts": [[18, 0, 0], [0, 18, 0], [0, 0, 18]], "undefined": {}, "bootstrap_ci": {"accuracy": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}, "balanced_accuracy": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}, "macro_f1": {"high": 1.0, "low": 1.0, "resamples_defined": 1000}}}
  efficiency: params=11,178,051 ckpt=44,780,555 B train_s=15.1 | latency on cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 8 threads, 224 px): bs1 median 52.130 ms p95 75.003 ms | bs32 median 922.032 ms p95 974.450 ms

{"eval_id": "ev-20261026-211449-9cdadc", "model": "efficientnet_b0", "split": "test", "checkpoint": {"epoch": 2, "file": "best.pt", "name": "best", "sha256": "30da4054a3d606a8e00d3d1b849c0c150b6907ac95d91acceecf79621f905bd5"}, "selection": {"best_epoch": 2, "best_value": 0.7721684420550311, "monitor": "val_loss", "selected_on": "val"}, "test_use": {"prior_test_evaluations_experiment": 0, "prior_test_evaluations_same_checkpoint": 0, "test_evaluation_number": 1}, "n": 54, "accuracy": 0.7407407407407407, "balanced_accuracy": 0.7407407407407408, "macro_f1": 0.7292827945001857, "macro_precision": 0.8134920634920636, "macro_excluded_classes": {"f1": [], "precision": [], "recall": []}, "log_loss": 0.840279810084572, "log_loss_clipped_samples": 0, "roc_auc_ovr_macro": 0.9598765432098766, "ece": 0.27355003642024844, "confusion_counts": [[17, 1, 0], [3, 15, 0], [8, 2, 8]], "undefined": {}, "bootstrap_ci": {"accuracy": {"high": 0.8518518518518519, "low": 0.6111111111111112, "resamples_defined": 1000}, "balanced_accuracy": {"high": 0.8379802447294706, "low": 0.6303330928330928, "resamples_defined": 1000}, "macro_f1": {"high": 0.8374983518659989, "low": 0.5962656131635043, "resamples_defined": 1000}}}
  efficiency: params=4,011,391 ckpt=16,304,357 B train_s=14.1 | latency on cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 8 threads, 224 px): bs1 median 52.898 ms p95 57.343 ms | bs32 median 1315.398 ms p95 1363.920 ms
```
Notes:
- **Baseline undefined values:** the baseline's precision for `square` and `triangle` (never
  predicted) is `null` with reason `class_never_predicted`, not 0. Its macro precision (0.3333) is
  over `circle` only, and the excluded classes are listed.
- **Baseline ECE:** reported as 1.1e-16 because of floating-point rounding. Confidence 1/3 equals
  accuracy 1/3, so the exact value is 0.
- **Degenerate CIs:** CIs of [1, 1] for perfect predictions are degenerate. They reflect test-set
  resampling only and do not show that the error rate is zero.
- **EfficientNet-B0:** test balanced accuracy is 0.7407, while its val-selected epoch had val
  balanced accuracy 0.8519. Its test ECE (0.27) indicates under-confidence or poor calibration
  after only 2 epochs.

### b. Repeat test evaluation (warning + count)
```
$ visionbench evaluate exp-20261026-202542-1c7e11 --split test --device cuda    (repeat)
WARNING: experiment exp-20261026-202542-1c7e11 checkpoint 'best' has already been evaluated on TEST 1 time(s); this is test evaluation #2. Repeated test evaluation is recorded and reported.
evaluation ev-20261026-211527-85714a: experiment exp-20261026-202542-1c7e11 (small_cnn) on TEST with checkpoint 'best' (epoch 14, sha256 1242e24090c2...)
  n=54  accuracy=1.0000 [1.0000, 1.0000]
  balanced_accuracy=1.0000 [1.0000, 1.0000]  (excluded absent classes: none)
  macro_f1=1.0000 [1.0000, 1.0000]  weighted_f1=1.0000  log_loss=0.0339
  roc_auc_ovr_macro=1.0000  ap_macro=1.0000  ece=0.0325
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 1
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202542-1c7e11\evaluations\ev-20261026-211527-85714a
[exit 0]

$ visionbench evaluate exp-20261026-202742-757de7 --split val --device cuda     (val: CUDA latency, no test use)
evaluation ev-20261026-211544-16a8b5: experiment exp-20261026-202742-757de7 (resnet18) on VAL with checkpoint 'best' (epoch 2, sha256 e0e68ef98488...)
  n=54  accuracy=1.0000 [1.0000, 1.0000]
  balanced_accuracy=1.0000 [1.0000, 1.0000]  (excluded absent classes: none)
  macro_f1=1.0000 [1.0000, 1.0000]  weighted_f1=1.0000  log_loss=0.0036
  roc_auc_ovr_macro=1.0000  ap_macro=1.0000  ece=0.0036
  CIs: 1000 bootstrap resamples, seed 0
  test evaluations of this experiment+checkpoint before this one: 1
  outputs: <scratchpad>\m2\var\experiments\exp-20261026-202742-757de7\evaluations\ev-20261026-211544-16a8b5
[exit 0]

$ visionbench evaluations list
ev-20261026-211143-ac2902  exp-20261026-202532-8a255a  split=test  checkpoint=baseline  2026-10-26T21:11:43+00:00  experiments/exp-20261026-202532-8a255a/evaluations/ev-20261026-211143-ac2902/metrics.json
ev-20261026-211200-222430  exp-20261026-202542-1c7e11  split=test  checkpoint=best      2026-10-26T21:12:00+00:00  experiments/exp-20261026-202542-1c7e11/evaluations/ev-20261026-211200-222430/metrics.json
ev-20261026-211312-1ea6b4  exp-20261026-202742-757de7  split=test  checkpoint=best      2026-10-26T21:13:12+00:00  experiments/exp-20261026-202742-757de7/evaluations/ev-20261026-211312-1ea6b4/metrics.json
ev-20261026-211449-9cdadc  exp-20261026-202808-8c6076  split=test  checkpoint=best      2026-10-26T21:14:49+00:00  experiments/exp-20261026-202808-8c6076/evaluations/ev-20261026-211449-9cdadc/metrics.json
ev-20261026-211527-85714a  exp-20261026-202542-1c7e11  split=test  checkpoint=best      2026-10-26T21:15:27+00:00  experiments/exp-20261026-202542-1c7e11/evaluations/ev-20261026-211527-85714a/metrics.json
ev-20261026-211544-16a8b5  exp-20261026-202742-757de7  split=val   checkpoint=best      2026-10-26T21:15:44+00:00  experiments/exp-20261026-202742-757de7/evaluations/ev-20261026-211544-16a8b5/metrics.json
[exit 0]
```
The repeat on CUDA gave the same metrics as the first evaluation on CPU (same checkpoint, same
predictions at reported precision). The resnet18 **val** evaluation was done only to measure CUDA
latency without adding a test use. Its output line "test evaluations … before this one: 1" counts
the earlier *test* evaluation.

### c. `visionbench compare` across the 4 experiments on test
```
$ visionbench compare exp-20261026-202532-8a255a exp-20261026-202542-1c7e11 exp-20261026-202742-757de7 exp-20261026-202808-8c6076 --split test --note "pipeline demo on synthetic data — not benchmark results"
comparison rp-20261026-211751-a21364 on TEST: 4 experiments, reference exp-20261026-202532-8a255a
  json: <scratchpad>\m2\var\reports\rp-20261026-211751-a21364\comparison.json
  markdown: <scratchpad>\m2\var\reports\rp-20261026-211751-a21364\comparison.md
  confound flags: input_resolution_differs, pretrained_vs_scratch, training_budget_differs, single_seed_per_model, latency_devices_differ, repeated_test_evaluation
[exit 0]
```
The first `compare` run (`rp-20261026-211617-b83e71`) listed tied validation values as ranks 1 and
2. I changed the ranking to mark ties and re-ran `compare`, which reads existing evaluations and
adds no test use. The DB still holds 5 test evaluations. Full `comparison.md` of the re-run:

```markdown
# Model comparison (test split)

> **pipeline demo on synthetic data — not benchmark results**

- Report: `rp-20261026-211751-a21364` (2026-10-26T21:17:51+00:00, visionbench 0.0.1)
- Dataset: `ds-001fffaa50a2` (fingerprint `10c842287c954667…`), split `sp-5c3e3bf5d61e`, 54 test samples, classes: circle, square, triangle
- Reference for paired differences: `exp-20261026-202532-8a255a`

Test-set metrics are reported for every model **without a winner label**. Model ranking uses validation metrics only (see *Validation ranking*).

## Models

| experiment | model | weights | input px | params | ckpt bytes | latency bs1 (median ms, device) | train s | best epoch | val selection | val bal. acc |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp-20261026-202532-8a255a` | majority_baseline | none | n/a | n/a | 459 | n/a: majority_baseline has no forward pass (constant prediction from train priors) | 0.2 | 1 | val_loss=1.0986 | 0.3333 |
| `exp-20261026-202542-1c7e11` | small_cnn | scratch | 64 | 389667 | 1569717 | 1.205 (cuda) | 32.6 | 14 | val_loss=0.0535 | 1.0000 |
| `exp-20261026-202742-757de7` | resnet18 | imagenet | 224 | 11178051 | 44780555 | 52.130 (cpu) | 15.1 | 2 | val_loss=0.0036 | 1.0000 |
| `exp-20261026-202808-8c6076` | efficientnet_b0 | imagenet | 224 | 4011391 | 16304357 | 52.898 (cpu) | 14.1 | 2 | val_loss=0.7722 | 0.8519 |

## Test metrics

Point value [95% percentile bootstrap CI] from each evaluation (1000 resamples, seed 0).

| experiment | model | checkpoint | accuracy | balanced accuracy | macro F1 | test evaluations so far |
|---|---|---|---|---|---|---|
| `exp-20261026-202532-8a255a` | majority_baseline | baseline | 0.3333 [0.2037, 0.4630] | 0.3333 [0.3333, 0.3333] | 0.1667 [0.1128, 0.2110] | 1 |
| `exp-20261026-202542-1c7e11` | small_cnn | best | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 2 |
| `exp-20261026-202742-757de7` | resnet18 | best | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 1 |
| `exp-20261026-202808-8c6076` | efficientnet_b0 | best | 0.7407 [0.6111, 0.8519] | 0.7407 [0.6303, 0.8380] | 0.7293 [0.5963, 0.8375] | 1 |

## Paired differences vs reference `exp-20261026-202532-8a255a`

Paired percentile bootstrap: 1000 resamples, seed 0, the same resampled indices for both models; difference = experiment − reference.

| experiment | model | Δ balanced accuracy [95% CI] | Δ macro F1 [95% CI] |
|---|---|---|---|
| `exp-20261026-202542-1c7e11` | small_cnn | 0.6667 [0.6667, 0.6667] | 0.8333 [0.7890, 0.8872] |
| `exp-20261026-202742-757de7` | resnet18 | 0.6667 [0.6667, 0.6667] | 0.8333 [0.7890, 0.8872] |
| `exp-20261026-202808-8c6076` | efficientnet_b0 | 0.4074 [0.2970, 0.5046] | 0.5626 [0.4327, 0.6720] |

## Validation ranking

Basis: validation balanced accuracy at the val-selected epoch (VAL only).

- rank 1=: `exp-20261026-202542-1c7e11` small_cnn: 1.0000
- rank 1=: `exp-20261026-202742-757de7` resnet18: 1.0000
- rank 3: `exp-20261026-202808-8c6076` efficientnet_b0: 0.8519
- rank 4: `exp-20261026-202532-8a255a` majority_baseline: 0.3333

`=` marks tied validation values; tied models are not ordered.

## Confound flags

- **input_resolution_differs**: models see different input resolutions; architecture and resolution are confounded. Values: `{"exp-20261026-202542-1c7e11": 64, "exp-20261026-202742-757de7": 224, "exp-20261026-202808-8c6076": 224}`
- **pretrained_vs_scratch**: ImageNet-pretrained and scratch models are compared; pretraining is a confound. Values: `{"exp-20261026-202542-1c7e11": "scratch", "exp-20261026-202742-757de7": "imagenet", "exp-20261026-202808-8c6076": "imagenet"}`
- **training_budget_differs**: configured epoch/step budgets differ. Values: `{"exp-20261026-202542-1c7e11": {"epochs": 15, "epochs_trained": 15, "max_steps": null}, "exp-20261026-202742-757de7": {"epochs": 2, "epochs_trained": 2, "max_steps": null}, "exp-20261026-202808-8c6076": {"epochs": 2, "epochs_trained": 2, "max_steps": null}}`
- **single_seed_per_model**: one run (one seed) per model: seed-to-seed variance is unmeasured, so differences may be within run-to-run noise. Values: `{"efficientnet_b0": 1, "majority_baseline": 1, "resnet18": 1, "small_cnn": 1}`
- **latency_devices_differ**: latency was measured on different devices. Values: `{"exp-20261026-202542-1c7e11": "cuda", "exp-20261026-202742-757de7": "cpu", "exp-20261026-202808-8c6076": "cpu"}`
- **repeated_test_evaluation**: at least one experiment has been evaluated on TEST more than once. Values: `{"exp-20261026-202532-8a255a": 1, "exp-20261026-202542-1c7e11": 2, "exp-20261026-202742-757de7": 1, "exp-20261026-202808-8c6076": 1}`
```
Reading notes:
- **Mixed latency devices:** the latency column mixes devices because `compare` uses each
  experiment's latest evaluation, and small_cnn's latest is the CUDA repeat. The report flags this
  (`latency_devices_differ`); demo d shows like-for-like numbers.
- **Degenerate paired CI:** Δ balanced accuracy for small_cnn/resnet18 vs the baseline has a
  degenerate CI [0.6667, 0.6667]. On every resample the perfect model scores 1, and the constant
  baseline scores exactly 1/3 (all three classes appear in every resample of 54 balanced samples).
  This is a property of the data, not a bug.

### d. Latency on CPU and CUDA (small_cnn, resnet18)
Read from the four `efficiency.json` files.
```
| model | eval | device (name) | threads | input px | bs1 median / p95 ms | bs32 median / p95 ms | bs32 median ms/image |
|---|---|---|---|---|---|---|---|
| small_cnn | ev-20261026-211200-222430 (test) | cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel) | 8 | 64 | 3.846 / 13.600 | 56.669 / 67.367 | 1.771 |
| small_cnn | ev-20261026-211527-85714a (test) | cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU) | 8 | 64 | 1.205 / 1.521 | 2.938 / 3.313 | 0.092 |
| resnet18 | ev-20261026-211312-1ea6b4 (test) | cpu (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel) | 8 | 224 | 52.130 / 75.003 | 922.032 / 974.450 | 28.813 |
| resnet18 | ev-20261026-211544-16a8b5 (val) | cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU) | 8 | 224 | 3.872 / 4.920 | 37.968 / 38.738 | 1.186 |

scope: model forward pass only; preprocessing, decoding and data loading excluded; warmup 10, 50 timed iterations per batch size
```
Each value is a single measurement session on a laptop: no repeated sessions, and thermal and
power state were not controlled. The 64 px vs 224 px input sizes make small_cnn and resnet18
latency incomparable as architectures.

## 6. git log --oneline
```
a2db7bf M3 docs: evaluation protocol, architecture, README, status, changelog
ede73ff M3 fixes: deterministic "latest evaluation"; tie-aware val ranking
56b33fc M3: evaluation entry point, test-use accounting, model comparison
9c4112d M3: metrics with explicit undefined values; bootstrap CIs; latency
c508d5f M3 step 0: torch home setting, frozen BN for head_only, norm-stats cache
bc983ee M2 review packet
899248f M2 docs: protocol, architecture, README, status, changelog
42f638b M2: CLI training from split manifests with SQLite experiment tracking
3548826 M2: model registry (baseline, SmallCNN, ResNet18, EfficientNet-B0)
a321521 M2 step 0: hash-ordered splits, popcount low-info guard, dataset verify
...
```
One more commit adds this packet. Everything is local only: no remote, nothing pushed.

## 7. Known limitations
- **CI scope:** CIs cover test-set sampling only, not seed or training variance. Each model has one
  run, which is flagged. With 54 test images the CIs are coarse, and perfect scores give degenerate
  intervals.
- **Macro precision with never-predicted classes:** macro precision excludes never-predicted
  classes. This is consistent with sklearn's NaN policy and stated in the report, but it can look
  favourable for degenerate predictors. Macro F1 (0 for those classes) and balanced accuracy are the
  primary headline metrics.
- **Latency:** forward pass only, a single session, no thermal control. The `p95` on CPU for
  small_cnn (13.6 ms vs a 3.8 ms median) shows scheduling noise.
- **Comparison selection:** `compare` uses the latest evaluation per experiment, which can mix
  devices for latency (flagged). It has no option yet to pin specific evaluation ids.
- **EXIF:** not applied anywhere (§0).
- **Bisect:** commit `9c4112d` is not test-clean on its own (§2).
- **Test-use enforcement:** test evaluations are counted but not limited (by design for M3). In M4
  the API must apply the same accounting.

## 8. Open questions for the supervisor
1. Should `compare` accept explicit evaluation ids, so that latency and device can be held fixed?
2. Should macro precision be reported as `null` (rather than over defined classes) whenever any
   class is never predicted? This is stricter and arguably clearer for degenerate predictors.
3. Should a limit on test re-evaluations be enforced, for example by requiring `--force` after
   N = 1, once real benchmarks start (M8)?
4. Should EXIF orientation be applied, consistently in duplicate detection, training and
   evaluation, before real photo datasets are used? That would change dataset fingerprints for
   affected images, and would need a dataset format version bump.

## 9. Recommended next task
**M4:** FastAPI endpoints over the library, and a single-worker job runner. The runner needs a
SQLite job table, cancellation, startup recovery of `running` rows (training experiments
included), and secure uploads. API-triggered test evaluations must go through the same
`evaluate_experiment` accounting.
