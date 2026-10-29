> **Commit hashes:** the history was rewritten before publishing (author identity, local paths).
> Hashes cited below are the pre-rewrite ones; see [COMMIT_MAP.md](COMMIT_MAP.md).

# Review Packet — VB-M6

**Task:** VB-M6, "Robustness evaluation under fixed corruptions (validation only)".
**Date:** 2026-10-29.
**Environment:**
- Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130; NVIDIA GeForce RTX 3050 6GB
  Laptop GPU.
- Pillow 12.3.0 with libjpeg-turbo (JPEG library 8.0, zlib-ng 1.3.1); NumPy 2.4.6.

All numbers come from commands run in this session. Paths under the session scratchpad are
written as `<scratchpad>`.

> **Every robustness number in this packet is VALIDATION (selection-biased) — not test
> results.** It describes sensitivity to the four synthetic corruptions of `eurosat-c-v1` at
> their three severities only. No model is called robust.

## Summary

1. **Step 0a:** the SmallCNN Grad-CAM target moved from `features.block4.conv` (pre-BN,
   pre-ReLU) to `features.block4.relu` (post-ReLU, pre-pool, 8x8 at 64 px).
   - All 12 trainable benchmark checkpoints give **bitwise-identical logits** on the 20 demo
     images before and after the change.
   - All-zero CAMs for small_cnn seed 0 went from **1 → 0** of 20.
   - The CAM still equals the reference implementation, and all sanity checks pass.
   - `gradcam_version` is now `vb-gradcam-2`.
2. **Step 0b:** cascading randomization on the four seed-0 models. No model is flagged. The
   chance level of |ρ| is reported, because it is 0.5 for a 2x2 CAM.
3. **Step 0c:** `keep_awake()` is used in every long-running script.
4. **Protocol before the first run:** commit `bf1de6e` at **06:01:25Z**; the first
   `robustness_evaluations` row was created at **06:55:15Z**.
5. **Materialization:** the corrupted sets are **byte-identical** across reruns and across a
   fresh output root written by a new process: 48,600 PNGs, all 12 condition hashes equal.
   Every model was evaluated from the same files.
6. **Evaluation:** **15 runs x 12 conditions = 180 VAL robustness evaluations**, all on the first
   attempt, with no failures or interruptions.
7. **Separate storage:** clean and corrupted results are stored apart (the `evaluations` table
   still holds 15 val rows). Clean evaluation ids are pinned. The majority baseline's corrupted
   metrics equal its clean metrics in **36/36** pairs.
8. **Zero test evaluations of any kind** in the whole database.
9. **Tests:** lint clean; 338 passed and 1 skipped in both modes; coverage 92 %.
10. **Main finding:**
    - Gaussian noise reduces every trainable model to near chance.
    - The two pretrained 224 px models fall the furthest under noise, blur and JPEG, and the
      least under brightness.
    - The direction under noise, blur and JPEG is opposite to the "upsampling smooths
      high-frequency corruptions" mechanism described in the protocol's confound note
      (section 6.2).
    - An identity check confirms that the image path is not the cause.

---

## 1. Commits (local only; nothing pushed)

```
597c4dc 2026-10-29T12:49:53+05:00 M6: VAL robustness results (15 runs x 12 conditions) and docs
25baac5 2026-10-29T12:49:53+05:00 M6: robustness driver, identity check, rename retry, check covers robustness
d3b6f64 2026-10-29T11:16:19+05:00 M6: robustness worker job and API endpoints
7590c03 2026-10-29T11:12:01+05:00 M6: robustness library (corruptions, materialization, evaluation, report), schema v7, CLI
bf1de6e 2026-10-29T11:01:25+05:00 M6: freeze the eurosat-c-v1 robustness protocol (before any robustness evaluation)
67347c3 2026-10-29T11:01:25+05:00 M6 step 0: SmallCNN Grad-CAM on the post-ReLU map; cascading randomization; keep-awake
```

Order: Step 0 → robustness protocol → library + tests → API/job → driver/fix → results +
docs (this packet last). The library and API/job trees were each exported and tested alone
(76 passed; 94 passed).

## 2. Step 0

### 2a. SmallCNN target layer: before/after

| | before (`vb-gradcam-1`) | after (`vb-gradcam-2`) |
|---|---|---|
| `SmallCNN.TARGET_LAYER` | `features.block4.conv` (conv output, before BN and ReLU) | `features.block4.relu` (post-ReLU, pre-pool) |
| feature map at 64 px | 8x8 | 8x8 |
| logits of all 12 trainable benchmark checkpoints, 20 demo VAL images | — | **bitwise identical** to before (12/12 checkpoints, shape 20x10) |
| all-zero CAMs, small_cnn seed 0, 20 demo VAL images | **1** (image 3, Forest_1946) | **0** |

- **No model code changed.** The ReLU is inplace, so the module's forward output is the
  activated tensor, and nothing modifies it afterwards.
- **New test:** the hooked tensor equals `ReLU(BN(conv(x)))` of block4, shape (1, 256, 8, 8),
  ≥ 0.
- **Reference test:** it now uses a tensor backward hook plus `.backward()`. PyTorch forbids a
  module `register_full_backward_hook` on an inplace ReLU; the first attempt failed with exactly
  that RuntimeError. Our CAM equals the reference on the new layer.
- **Overlays:** the small_cnn overlays were regenerated (`var/demos/m5/overlays/small_cnn_scratch_64/`)
  and the old ones moved to `var/demos/m5/superseded_vb-gradcam-1/`. The regenerated overlays of
  the other three models are byte-identical to M5 (60/60), and all 80 predictions are identical.

**Grad-CAM sanity checks, old layer vs new** (synthetic task; thresholds unchanged):

| check | threshold | old (`block4.conv`, M5) | new (`block4.relu`) |
|---|---|---|---|
| localization (peak in the disk's quadrant) | ≥ 80 % | 75/75 (100 %) | 75/75 (100 %) |
| target sensitivity: mean Pearson r (true vs other class CAM) | < 0.5 | −0.606 over 100 pairs (50 skipped: all-zero) | −0.676 over 50 pairs (100 skipped: all-zero) |
| model randomization: mean \|Spearman ρ\| | < 0.5 | 0.129 (8 images) | 0.160 (7 images; 1 all-zero) |
| equals the reference implementation | allclose | yes | yes |

On the post-ReLU map, other-class CAMs are all-zero more often (100 of 150 pairs). That is
expected: a ReLU'd map has no positive evidence for a class the image does not show.

### 2b. Cascading randomization on the real seed-0 models (report only)

**Setup:**
- Images: the 20 seeded M5 demo VAL images.
- Target: the trained model's predicted class.
- Re-initialisation: `reset_parameters()` per stage, from the top, cumulative.
- Measure: Spearman ρ between the trained CAM and the CAM after each stage.
- Full table: `benchmarks/eurosat-v1/results/gradcam_cascading_randomization.md`.

| model (CAM) | after stage 1 (head) mean \|ρ\| | after full randomization: mean ρ / mean \|ρ\| | chance \|ρ\| | flag (> 0.5) |
|---|---|---|---|---|
| resnet18_scratch_64 (2x2) | 0.658 | 0.059 / 0.430 | 0.500 | no |
| small_cnn_scratch_64 (8x8) | 0.260 | −0.002 / 0.252 | 0.102 | no |
| resnet18_imagenet_224 (7x7) | 0.370 | −0.036 / 0.194 | 0.116 | no |
| efficientnet_b0_imagenet_224 (7x7) | 0.370 | 0.066 / 0.258 | 0.116 | no |

- **Chance level:** under independence E|ρ| is 0.5 for 4 cells (exact, all 24 permutations) and
  about 0.10–0.12 for 49 or 64 cells (simulated).
  - resnet18_scratch_64's |ρ| is at chance for a 2x2 map at every stage, so the 0.5 flag cannot
    be judged for it.
  - The other models end at 0.19–0.26: above chance, below the flag. The CAMs keep some
    input-driven structure even with random weights, as Adebayo et al. also observed.

### 2c. Keep-awake

`visionbench.core.power.keep_awake()` (`SetThreadExecutionState`) is now called by
`run_benchmark.py`, `repro_check.py`, `pilot_benchmark.py`, `cascading_randomization.py` and
`run_robustness.py`.

## 3. Protocol before the first robustness evaluation

| event | time (UTC) |
|---|---|
| robustness protocol commit `bf1de6e` (committer date `2026-10-29T11:01:25+05:00`) | **2026-10-29 06:01:25** |
| library commit `7590c03` (schema v7 introduced) | 06:12:01 |
| API/job commit `d3b6f64` | 06:16:19 |
| first materialization | 06:16:36 |
| **first `robustness_evaluations` row** (`created_at`) | **2026-10-29 06:55:15** |
| last of the 180 rows | 07:34:24 |

- **DB state at the protocol commit:** the real DB was still at schema v6 and had no
  `robustness_evaluations` table (checked right after the commit, and again after the API/job
  commit).
- **Protocol files unchanged:** `sha256sum -c ROBUSTNESS_HASHES.txt` reported OK for both files
  before materialization, and `run_robustness.py` re-checked them before evaluating. Git shows
  `bf1de6e` as the only commit touching them.

## 4. Materialization and the hash comparison

```
visionbench robustness materialize ds-61112928e412 sp-55320e11726c --suite benchmarks/eurosat-v1/robustness/robustness_suite.yaml --split val
```

| condition | condition hash (identical in runs 2, 3 and 4) |
|---|---|
| gaussian_noise/0.04 | `0e06fd815f1bcca79b2a9ae37557f2949af16b31fec16a9c0abd8827c4806f26` |
| gaussian_noise/0.08 | `82576046d19e65c15e7ab8d5f628439cf5089f4924c8612c5d0fa7ced6aa830f` |
| gaussian_noise/0.12 | `5f9cbd3fef043fa149764a33fc1e1c6981ba7b5335e4dc2f9c1ba9cb978e6c66` |
| gaussian_blur/0.5 | `2ba756d9d7a812eb07eb6fda49c18d3364d37423aac43ba5b2fe4459b12593d3` |
| gaussian_blur/1.0 | `94a9ef7d449a1c532680779f9e6b24127ef461ed405465f5e707064efe8af058` |
| gaussian_blur/1.5 | `0880e258e486d3fa68cf9d207a9101b7fd043ec374fd947f0cc19c44454ed535` |
| brightness/0.1 | `2d49319f2121beccc4c2f186248d9803c2650cc81c374a870ccac59cb29c8ab0` |
| brightness/0.2 | `f69f4755ece42118358ddb3567b52ca4b6e5103cc06fa2772380c230c47b0baf` |
| brightness/0.3 | `5575ce41652f6617848bfb56fbce4211c67f34a00f8acdb3740b9b0a1e844a6f` |
| jpeg/50 | `58356494e134259dba31807aacbc0fb13395999dd53e0d886dcda478e51e9daf` |
| jpeg/25 | `eed6d8f66049b94a901c3c2037cf85225e6ed034fd6d2ddc0225d9b6d8164ee4` |
| jpeg/10 | `5030987a2145865a3129028793c36191f196b3c6541d27c66c0689f1cc1bc5fb` |

- **Run 1 (06:16:36Z) failed:** `PermissionError [WinError 5]` while renaming the temporary
  directory of `gaussian_blur/1.0` into place, after 4 conditions had been written.
  - The most likely cause is the Windows indexer or antivirus briefly holding the freshly
    written files.
  - The temporary directory was removed by the error path, and nothing partial remained.
  - Fix: `_rename_into_place` retries `os.replace` up to 10 times with backoff. It is in the
    driver/fix commit and was made before any evaluation.
- **Run 2:** verified the 4 existing conditions byte for byte and wrote the other 8.
- **Run 3:** a new process writing to a fresh output root (`<scratchpad>/materialize_copy`):
  all 12 conditions written, with the same 12 condition hashes.
- **Run 4:** the canonical root, all 12 conditions **verified** (every image re-derived and
  compared with the stored bytes).
- **Direct comparison:** 48,600 PNG files in the canonical cache vs 48,600 in the fresh copy,
  **identical file set and bytes: True**.
- **During evaluation:** every condition manifest and every corrupted file was re-hashed for
  every run before use (`load_verified_condition`).

## 5. Evaluation runs (`benchmarks/eurosat-v1/robustness/robustness_runs.json`)

- **Command:** `scripts/run_robustness.py benchmarks/eurosat-v1 --split val` ran from 06:54:48Z
  to 07:34:24Z.
- **Execution:** sequential, one process per run, on CUDA (CPU for the baseline). Windows was
  kept awake, and no other GPU work ran meanwhile; the GPU-hidden test run used the CPU only.
- **Outcome:** all 15 runs `completed` on attempt 1, each "12 evaluated, 0 skipped". No
  interruption or rerun.
- **Clean reference:** each run's B1 `val_eval_id`, pinned.

**Identity check of the image path** (`scripts/robustness_identity_check.py`, CUDA):

The original val images, re-encoded as lossless PNG and read through the same code as the
corrupted sets, reproduce the clean predictions exactly.

```
resnet18_scratch_64 predictions identical to clean: True | max |prob diff|: 4.999948528041373e-09
small_cnn_scratch_64 predictions identical to clean: True | max |prob diff|: 4.999956594586852e-09
resnet18_imagenet_224 predictions identical to clean: True | max |prob diff|: 4.999977012561235e-09
efficientnet_b0_imagenet_224 predictions identical to clean: True | max |prob diff|: 4.999999969612645e-09
```

The remaining 5e-9 difference is the 8-decimal rounding of `predictions.csv`. I ran this check
because of the finding below: the collapse is not caused by the PNG path, and the majority
baseline check alone could not show that, since it never looks at an image.

## 6. Results (`benchmarks/eurosat-v1/results/robustness_val.md`, printed in full in the handover)

### 6.1 Headline numbers

**VALIDATION (selection-biased) — not test results.** Balanced accuracy, mean ± sd over 3
seeds:

| model | clean | mean over 12 conditions | relative drop |
|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0.9871 ± 0.0027 | 0.4395 ± 0.0025 | 0.5548 ± 0.0016 |
| resnet18_imagenet_224 | 0.9862 ± 0.0035 | 0.5178 ± 0.0048 | 0.4749 ± 0.0063 |
| resnet18_scratch_64 | 0.9125 ± 0.0485 | 0.5857 ± 0.0292 | 0.3580 ± 0.0124 |
| small_cnn_scratch_64 | 0.9556 ± 0.0027 | 0.5532 ± 0.0042 | 0.4211 ± 0.0029 |
| majority_baseline | 0.1000 | 0.1000 | 0 |

### 6.2 Observations

These are stated only for these corruptions. Each is backed by seed-0 paired CIs that exclude
zero, unless noted.

- **Gaussian noise:**
  - Every trainable model drops strongly at every severity.
  - The pretrained models are near the 0.10 chance level already at σ = 0.04 (0.21 and 0.19
    mean); the scratch models reach 0.57 and 0.36.
  - Under σ = 0.12 the models predict almost a single class. resnet18_imagenet's recall is 1.00
    for Residential and ≤ 0.07 for every other class.
- **Blur:**
  - σ = 0.5 changes little. For small_cnn seed 0 it slightly *raises* balanced accuracy
    (+0.0076 [+0.0033, +0.0120]). For resnet18_scratch_64 seed 0 the CI includes zero
    (−0.0001 [−0.0049, +0.0048]).
  - At σ = 1.0 and 1.5, the pretrained models fall further than the scratch models.
- **JPEG:** the pretrained models fall further than the scratch models at all three qualities.
- **Brightness:** the order reverses. At +0.3 the pretrained models keep 0.75–0.76; the scratch
  models keep 0.33–0.44.
- **Against the protocol's confound note:**
  - The note describes a mechanism: upsampling 64 → 256 smooths high-frequency corruptions
    before pretrained models see them.
  - The measured direction under noise, blur and JPEG is the opposite.
  - The upsampled corruption is itself a different input (for example, bilinear-upsampled noise
    becomes spatially correlated blobs), and pretraining and resolution remain confounded.
  - Nothing here identifies a cause. The difference between the groups must not be read as an
    architecture or pretraining effect.
  - The frozen protocol text is unchanged, and no amendment was made.
- **Severity strength (a finding, not acted on):** at 64 px the lowest noise severity already
  reduces two models to near chance, so the suite resolves little between models for noise.
  Changing severities would require a dated amendment and rerunning every evaluation; I did not
  do that.
- **Low-information classes:** in the worst condition (noise 0.12 for every trainable seed-0
  model), SeaLake keeps 0.36 recall for efficientnet_b0 and small_cnn, and Forest drops to 0.00
  for all four. Both are flagged as largely low-information.
- **No model is called robust,** and no model, seed or checkpoint was selected using these
  results.

## 7. No-test-evaluation check

```
$ python scripts/run_benchmark.py check benchmarks/eurosat-v1
benchmark experiments: 15; pilot + repro-check experiments: 13
val evaluations of benchmark experiments: 15
test evaluations of benchmark or pilot experiments: 0
test evaluations in the whole database: 0
robustness val evaluations of benchmark experiments: 180
robustness test evaluations in the whole database: 0
OK: zero test evaluations of any kind (clean or robustness) in the whole database
```

A direct SQLite query gives:
- `robustness_evaluations` by split: `[('val', 180)]`;
- `evaluations` by split: `[('val', 15)]`;
- schema version 7.

## 8. Tests (after the last code change)

```
$ ruff check . ; ruff format --check .
All checks passed!
134 files already formatted

$ pytest -q -rs -p no:cacheprovider --cov=visionbench
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
338 passed, 1 skipped, 1 warning in 340.31s (0:05:40)
TOTAL 92 %; robustness: corruptions 89 %, evaluate 91 %, materialize 85 %, report 85 %, suite 79 %

$ CUDA_VISIBLE_DEVICES=-1 pytest -q -rs -p no:cacheprovider
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
338 passed, 1 skipped, 1 warning in 421.14s (0:07:01)
```

The skipped test is the file-symlink test, which cannot run on this machine without symlink
privilege.

**New tests** (`tests/robustness/`, 21 tests, synthetic data only):
- **Corruption behaviour:**
  - noise variance equals σ² within 3 %;
  - blur reduces high-frequency energy monotonically, and reflect padding keeps a flat image
    flat;
  - brightness raises mean V by δ and preserves the dominant channel; clipping; black → grey;
  - lower JPEG quality → larger error and different bytes, deterministic;
  - dtype, shape and range for every corruption; identity at near-zero severity.
- **Seeds:** they differ by relpath, corruption, severity and base seed; noise is independent
  across relpaths and severities.
- **Determinism:** materializing twice gives "verified"; a second process into another root
  gives identical files and hashes; reversed entry order gives identical files.
- **Tampering:** a tampered PNG or condition manifest is refused.
- **Test lock:** in the library, the CLI (exit 1, "locked") and the API (403).
- **Storage and pairing:**
  - the `evaluations` table is untouched;
  - the pinned clean id is stored; another experiment's clean evaluation is refused;
  - the paired difference point equals corrupted − clean.
- **Pipeline behaviour:**
  - majority baseline exactly equal, with a degenerate paired CI at 0;
  - resumable (the second run skips all 12);
  - report (label, scope, confounds, baseline check).
- **API and job:** client header 403, unknown suite 404, extra field 422, not materialized 409;
  the job runs inline, with progress 12/12 and list/detail endpoints; no path leaks.
- **Schema:** upgrade from v6 to v7.

## 9. Limitations and what was not done

- **Scope:** only four synthetic corruptions at three severities, on one dataset and one split,
  with 3 seeds. No statement about real degradations or distribution shift. TEST is untouched;
  M8 runs it once with the same suite.
- **JPEG reproducibility:** JPEG bytes depend on the Pillow/libjpeg build. Byte identity was
  shown on this machine only; the versions are recorded in every condition manifest.
- **Severity calibration:** severities were fixed before any result and were not tuned. At
  64 px the noise severities are strong (section 6.2).
- **No normalization:** no ImageNet-C mCE normalization; relative drops are against each run's
  own clean VAL score.
- **Unverified README commands:** the README's robustness commands are the ones run here. The
  API endpoints were exercised in tests only, not with curl against a live server.
- **Not done:** no frontend, no change to the benchmark protocol or checkpoints, no push.

---

# VB-FINAL — top summary (STOPPED in Phase C, step C3)

Written 2026-10-29 at the stop. Evidence for each phase follows under "VB-FINAL — evidence by
phase".

**Delivered and verified in this session:**
- **Phase 0:** the M6 follow-ups and Gate 0.
- **Phase A:** the M7 frontend (9 pages, 42 component tests, Playwright e2e) and Gate A. That
  includes a page-by-page walk on the real data root.
- **Phase B:** Docker image and compose; Linux container test run; CI workflow file; demo-data;
  native and Docker Quick Start; Linux CPU locks; full documentation; Gate B; tag
  **`v1.0.0-rc1`**.
- **Phase C preparation and plan:**
  - `FINAL_ANALYSIS_PLAN.md` committed and hashed before any TEST access;
  - TEST drivers, `freeze-check` and `final_report.py`, committed in rc1;
  - VAL dry run of the report;
  - a synthetic end-to-end rehearsal.
- **C2:** freeze checklist, all items a-g pass.
- **C3, partly:**
  - **the 15 clean TEST evaluations are done**, each on its first attempt;
  - one per run, on the val-selected checkpoint;
  - none forced;
  - recorded in `benchmarks/eurosat-v1/results/test_runs.json`.

**Stopped:**
- **Where:** C3, "materialize TEST pass 1".
- **Error:** `error: gaussian_noise/0.04: stored condition covers different images`.
- **Cause, from reading the code:** the corrupted-image cache is laid out as
  `cache/corruptions/<dataset>/<split_id>/<suite>/<corruption>/<severity>/`
  (`src/visionbench/robustness/materialize.py:3`, `:104`). The path has no val/test component, so
  the TEST pass finds the M6 VAL `condition.json` in the same folder. It refuses because the image
  sets differ (`materialize.py:191-192`).
- **Classification:** a code defect, not an infrastructure failure. STOP condition 2 and plan
  §3.3 therefore apply: nothing was changed or rerun.
- **State, checked read-only:**
  - nothing was written by the failed pass: the newest cache file is from 06:25:54 UTC (M6); the
    pass ran at 11:32 UTC;
  - the 12 VAL conditions still verify (manifest hash and every file's sha256);
  - 0 robustness TEST rows.
- **Why the rehearsal missed it:** its throwaway data root never had a VAL materialization before
  the TEST one, and neither did the new TEST-split test. That gap in the rehearsal is mine.

**Not produced because of the stop:**
- the 180 robustness TEST evaluations;
- `results/FINAL_REPORT.md/.json` and `results/robustness_test.md/.json`;
- unlocking the app and the Benchmark TEST section (C6);
- screenshots (D1) and the README results update (D2);
- tag `v1.0.0`.

No TEST metric has been computed into a report or read for this summary.

**Owner decisions needed:**
1. **How to continue the robustness TEST evaluation.** Any route deviates from C4 ("no code
   changes between evaluations") or from the plan's commands, so it must be approved.
   - **(a) Recommended:** a minimal code fix.
     - TEST conditions get their own cache path (for example `<suite>/test/...`), with tests
       covering a VAL-then-TEST materialization.
     - The VAL path stays unchanged, so the 180 VAL rows and the VAL cache stay valid.
     - The fix would be committed and recorded as a dated deviation.
     - The 15 clean TEST evaluations are not affected by this code and would **not** be redone.
   - **(b)** Move the VAL cache aside, materialize and evaluate TEST in its place, then restore
     it. This needs no code change, but it is a manual data-layout operation.
   - **(c)** Stop at the clean TEST results and report robustness TEST as not done.
   - Not viable: `materialize --out` exists, but `robustness evaluate` cannot read from another
     root.
2. **Definition of done:** confirm the 14-item list below (reconstructed; the original is not
   in the repository or this session).
3. **License:** choose one. There is no LICENSE file, and `pyproject.toml` says "Proprietary".

**Deviations and readings:**
- **Ordering:**
  - Phase C code and the plan were committed before rc1. That is stricter than the task order.
  - Freeze item f was rerun on rc1 itself.
- **Docker Desktop** was started as an installed user-level app.
- **Gate A suite output** was captured with `tail -4`, so the one warning's text was first
  recorded at Gate B. It is a TestClient deprecation.
- **Linux run, attempt 1:** a wheel download failed pip's hash check (treated as an
  infrastructure failure) and was rerun.
- **Responsive layout:** at 768 px, wide tables scroll inside their own box.

**Blocked items:**
- **GitHub Actions** has not run (no remote; pushing is prohibited).
- **Phase C after "materialize TEST pass 1", and all of Phase D except this summary**, wait for
  the owner (decision 1).

### Definition-of-done audit (14 items, reconstructed; status at the stop)

| # | item | status | evidence |
|---|---|---|---|
| 1 | Installs from a fresh clone with documented commands | done | Phase B: native fresh-clone Quick Start (Windows) and Docker Quick Start, every command recorded |
| 2 | Dataset import, validation and statistics report | done | M1/D1; EuroSAT `verify --full` OK at C2 |
| 3 | Leakage-aware, deterministic splits with an independent audit | done | M1/D1 `audit-split`; split `sp-55320e11726c` frozen |
| 4 | Model registry incl. majority baseline and pretrained models | done | M2; 5 protocol models |
| 5 | Training with tracking; selection on VAL only | done | M2/B1; 15 runs in `runs.json` |
| 6 | Evaluation with CIs, explicit undefined metrics, TEST-use accounting | done | M3; `check` counts every TEST row |
| 7 | Comparison reports with confound flags; no TEST winner label | done | M3/M4 |
| 8 | API, single-worker jobs, secure uploads, local-only guards | done | M4/M4.1; same-origin fix found by e2e (9d1678a) |
| 9 | Prediction and Grad-CAM with sanity checks and stated limitations | done | M5 / M6 step 0 |
| 10 | Robustness suite with VAL and TEST evaluations | **partial** | VAL 180/180 done (M6); TEST 0/180 (stopped) |
| 11 | Frontend over the API with tests and an e2e | done | Phase A; Gate A/B |
| 12 | Frozen benchmark evaluated once on TEST per a committed plan, with a final report | **partial** | plan committed before TEST access; clean TEST 15/15 done; FINAL_REPORT not produced (stopped) |
| 13 | Packaging: Docker, lock files, CI, documentation | done, except CI on GitHub | Phase B; ci.yml not run on GitHub Actions (no remote) |
| 14 | Screenshots and an honest results summary in the README | **not done** | blocked by item 12 (Phase D needs C) |

# VB-FINAL — evidence by phase

(The VB-FINAL top summary is written at the top of this file in Phase D.)

## Phase 0 — M6 follow-ups (done)

Commits: `b5ee799` (0a), `07b2000` (0b), `9a088bf` (0d), `6cf7de2` (0c).

**0a. Robustness protocol amendment 1 (text only, dated 2026-10-29).**
- Appended under Amendments.
- Upsampling 64 → 256 → crop 224 scales the corruption together with the content; it does not
  smooth the corruption relative to the content.
- The task's "≈ 3.5 px" figure was checked and corrected. The resize scales by 256/64 = 4 and the
  centre crop does not rescale, so a 1 px blur becomes about 4 px in the model input; 224/64 = 3.5
  would apply only to a direct resize to 224.
- The VAL direction (pretrained models fall further under noise, blur and JPEG) is stated as
  consistent with the correction but not explained by it.
- No procedure, suite, severity or evaluation changed, and nothing was rerun.
- Old ROBUSTNESS_PROTOCOL.md sha256: `3d13075f2783c232f577f5d031fd91113922b8d01056501861cc5d3aab41794a`.
- New sha256: `0b738cfad455673b184eee30d02aeba138a573a984b74aee3129ad8cc1f1785b`.
- `robustness_suite.yaml` unchanged: `6d8d647f…cfbad2`.
- The report generator's confound sentence was corrected the same way, and `results/robustness_val.md/.json`
  were regenerated from the recorded results. The diff is exactly that one line in each file.

**0b. Documentation.**
- **pHash on dark low-texture tiles:** audit pair `a-001` was checked visually, side by side and
  contrast-stretched. `Forest/Forest_1552.jpg` (val, recorded gray std 4.51) and
  `River/River_1476.jpg` (train, 12.67) have pHash distance 0 and dHash distance 26. Neither is
  low-information, and they are clearly different images (canopy vs a river with a bank).
- **Noise severities:** every VAL noise condition is below 0.65 balanced accuracy (maximum
  0.6495), so the noise results resolve little between models.

**0c. README robustness API commands, run in PowerShell 5.1 against `visionbench serve`
(API + worker), real data root:**

```
queued: job-20261029-080711-4d0eed
status=completed evaluated=0 skipped=12
rows=12
detail: gaussian_noise/0.04 bal_acc=0.2948444444444444
{"error":{"code":"test_split_locked","message":"robustness on the TEST split is locked until the final test evaluation (M8); set VISIONBENCH_TEST_IMAGES_UNLOCKED=true only then"}}
```

- No new rows were written: `robustness_evaluations` still holds `[('val', 180)]` and
  `evaluations` `[('val', 15)]`.
- The server and its worker were stopped afterwards (no python process left).

**0d. Majority baseline display.**
- `training.config.display_config` shows `"n/a"` for every TrainConfig field the baseline does
  not use.
- It is used by the experiment detail API (`config_display`) and by `visionbench experiments show`.
- Stored `config_json` is unchanged (tested).

**Gate 0.**
- **Hashes:** `sha256sum -c` reports OK for all 6 benchmark protocol files and both robustness
  files.
- **Lint:** `ruff check` and `ruff format --check` pass.
- **Test run 1 (working tree):** 340 passed, 1 skipped in both modes (normal 508.16 s; GPU hidden
  410.72 s). Phase A files were being written meanwhile, so this run is not clean evidence for the
  Phase 0 commit.
- **Test run 2 (clean):** repeated on a `git archive` export of `6cf7de2` (the import resolved to
  the export): normal **340 passed, 1 skipped** (334.05 s); `CUDA_VISIBLE_DEVICES=-1`
  **340 passed, 1 skipped** (512.21 s).

## Phase A — M7 frontend (done)

Commits: `7b53c81` (backend read-only views + same-origin static serving), `9d1678a` (security
fix found by the e2e test), `99da9d4` (frontend, tests, e2e).

**Stack and dependencies.**
- React 18.3.1, TypeScript 5.6.3 (strict, `noUncheckedIndexedAccess`), Vite 5.4.11, React Router
  6.28.0, TanStack Query 5.62.7, Recharts 2.15.0.
- Tests: Vitest 2.1.8, Testing Library, MSW 2.6.8, Playwright 1.49.1.
- Lint and format: ESLint 9.16.0 and Prettier 3.4.2.
- All versions are pinned exactly, with `package-lock.json`.
- `npm install` printed deprecation notices for recharts 2.x, eslint 9.16 and glob@10.5.0. They
  are noted here and were not acted on.
- Playwright downloaded Chromium headless shell 131 to `%LOCALAPPDATA%\ms-playwright` (user
  profile, not the repo).

**API types.**
- `frontend/src/api/schema.d.ts` is generated from `create_app().openapi()` by
  `npm run gen:api`.
- `npm run check:api` fails if the file is stale. It reported "up to date" at 99da9d4.

**Backend additions for the pages** (all read-only, GET):
- `/api/overview`, `/api/benchmarks`, `/api/benchmarks/{name}`,
  `/api/benchmarks/{name}/documents/{document}` (a fixed list of documents, no paths) and
  `/api/robustness-suites`;
- a seeded split sample, where a TEST sample gets 403 `test_split_locked`;
- dataset audit;
- experiment tags.

`visionbench serve` serves `frontend/dist` with an SPA fallback. `/api/...` keeps JSON 404s, and
the Host and header guards are unchanged (backend test: `tests/api/test_frontend_views.py`).

**Bug found by the e2e test.** Browsers send `Origin` on same-origin POSTs, so the served
frontend got 403 `origin_not_allowed`. The guard now also accepts the API's own origin
(`http(s)://` + the already-validated Host). Test:
`test_same_origin_post_from_the_served_frontend_is_allowed`. A foreign Host still fails first
with `invalid_host`.

**Gate A** (all run at 99da9d4 in this session):

| check | result |
|---|---|
| `npm run lint` (ESLint + Prettier, src and e2e) | pass |
| `npm run typecheck` | pass |
| `npm run check:api` | up to date |
| `npm run test:coverage` | 4 files, **42 passed**; lines 87.67 %, branches 82.13 %, functions 89.27 % |
| `npm run build` | pass (JS 711.29 kB, gzip 206.33 kB; Vite's > 500 kB chunk notice, not acted on) |
| Playwright e2e (real backend, temp data root, synthetic shapes) | **1 passed** (2.7 min) |
| `ruff check .`, `ruff format --check .` | pass (140 files) |
| full pytest, clean `git archive` export of 99da9d4, normal | **345 passed, 1 skipped, 1 warning** (620.72 s) |
| same, `CUDA_VISIBLE_DEVICES=-1` | **345 passed, 1 skipped, 1 warning** (593.15 s) |
| `run_benchmark.py check benchmarks/eurosat-v1` | 0 test evaluations, 0 robustness test evaluations (whole DB) |

- The pytest output was captured with `tail -4`, so the text of the one warning was not recorded
  at Gate A. It is recorded with the Gate B run.
- The skipped test is the file-symlink case (no symlink privilege on this Windows account).

**The e2e flow** runs in the browser against `scripts/e2e_backend.py`, with a fresh data root and
generated shapes, never EuroSAT:
1. upload a ZIP, then the import job;
2. create a split;
3. train `small_cnn` on CPU with `max_steps`, with visible progress, plus a majority baseline;
4. curves;
5. VAL evaluations;
6. compare, showing confound flags;
7. confusion matrix;
8. predict and Grad-CAM on a VAL image;
9. upload of a TEST image's bytes, which shows the locked message;
10. robustness on a small synthetic suite (VAL), 2 evaluated;
11. cancel a running training job, then confirmed `cancelled`;
12. Overview shows "Locked.".

**Manual walk of the built app on the real data root.**
- Ran `visionbench serve` on `./var` and drove Chromium with a Playwright script (outside the
  repo).
- Every page was visited; recorded: h1, any error box, console errors.

| page | URL | h1 | error boxes |
|---|---|---|---|
| Overview | `/` | Overview | 0 |
| Datasets | `/datasets` | Datasets | 0 |
| Dataset detail | `/datasets/ds-61112928e412` | Dataset: eurosat-rgb | 0 |
| Training | `/training` | Training | 0 |
| Jobs | `/jobs` | Jobs | 0 |
| Job detail | `/jobs/job-20261029-080711-4d0eed` | Job | 0 |
| Experiments | `/experiments` | Experiments | 0 |
| Experiment detail | `/experiments/exp-20261027-201053-5f2809` | Experiment exp-20261027-201053-5f2809 | 0 |
| Evaluation view | `/evaluate?evaluation=ev-20261028-041748-ef337a` | Evaluate & Compare | 0 |
| Compare (two benchmark runs, VAL) | `/evaluate` | Evaluate & Compare | 0 |
| Explain (seed-0 VAL browser, Predict + Grad-CAM, efficientnet_b0 seed 0) | `/explain` | Explain | 0 |
| Robustness | `/robustness` | Robustness | 0 |
| Benchmark | `/benchmark` | Benchmark | 0 |
| unknown route | `/no-such-page` | Page not found | 0 |

- Console errors: none.
- Page horizontal overflow at 1024 px and 768 px: 0 px. At 768 px the wide runs table scrolls
  inside its own box.
- Explain (checked visually):
  - Prediction: AnnualCrop, p = 0.9988, "VALIDATION (selection-biased)" image label, forward
    88.9 ms, total 10 212.2 ms (first, cold request);
  - Grad-CAM: 7×7 at `features.8`, `vb-gradcam-2`, "not seen by the model 23.4 %";
  - the limitations note and attribution are visible.
- Benchmark page: both hash groups "all hashes OK", final analysis plan "not present", TEST
  section "Not evaluated yet".
- Predict and Grad-CAM are not recorded, so no DB rows were added. Server and worker were
  stopped afterwards, and port 8000 was free.

## Phase B — Release engineering (done; tag `v1.0.0-rc1`)

Commits: `d411106` (Docker, compose, CI, demo-data, locks), `4c9f83f` (docs, version 1.0.0rc1,
linux_tests.sh), `b1a6ef3` (Phase C preparation code), `3677583` (C1 plan), then the Gate B
docs commit (tagged).

**B1 Docker.**
- Docker Desktop was installed per user but stopped. It was started as an app (no install, no
  admin change), and the daemon came up: 29.7.2, linux/amd64.
- `docker build` succeeded at d411106. It took about 14 min, mostly the torch download at about
  245 kB/s. Image `visionbench:local` is 1.87 GB.
- `docker compose up -d` gave:
  - health `healthy` after start;
  - `uid=10001(visionbench)`;
  - port `127.0.0.1:8000->8000/tcp` only;
  - `/api/health` 200; `/` and `/benchmark` 200 (SPA);
  - `Host: evil.example` → 400 `invalid_host`; POST without the header → 403
    `missing_client_header`.
- Container torch is `2.14.1+cpu`, with CUDA unavailable.
- A host script drove the API: demo ZIP upload → import 8.2 s → split 9.2 s → `small_cnn` 2-epoch
  CPU training 44.0 s → VAL evaluation 9.2 s. VAL balanced accuracy was 0.444 on 18 synthetic
  images: a pipeline sanity check, not a result.
- After `compose down` / `up` the dataset and experiment were still in the volume. The volume was
  then removed (`down -v`).
- At 3677583 the image was rebuilt from cache (under 1 min). The Quick Start A commands, run in
  PowerShell:
  - `demo-data /data/demo` → 120 images;
  - `dataset import /data/demo/shapes` → `ds-161c17e14fbf`, which the API lists;
  - the plan hashes are OK inside the container.
- In Git Bash the same commands failed, because Git Bash rewrote `/data/demo` into
  `C:/Program Files/Git/data/demo`. This is documented under Troubleshooting (use
  `MSYS_NO_PATHCONV=1` or PowerShell).

**B2 Linux + CI.**
- `scripts/linux_tests.sh` runs in `python:3.11-slim@sha256:6737…` as a non-root user, on a
  `git archive` export.
- Attempt 1 (export of d411106) never reached the tests. `pip install` failed with "THESE
  PACKAGES DO NOT MATCH THE HASHES" for one wheel: a download did not match the index hash, on a
  slow, shared link. It was treated as an infrastructure failure. The next run used a pip-cache
  volume, and the export was refreshed to the newer HEAD.
- At 3677583: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41 python 3.11.18
  torch 2.14.1+cpu`; ruff check and format pass; **360 passed, 1 warning** in 411.68 s. This
  includes the file-symlink test that is skipped on Windows.
- `.github/workflows/ci.yml` has backend, frontend, e2e and docker jobs, and keeps the previous
  workflow's coverage step. **It has not run on GitHub Actions; there is no remote.**

**B3.**
- **`demo-data`:** deterministic pixels for a seed. PNG bytes differ by platform: the container
  (zlib 1.3.1) and Windows (zlib 1.3.2) gave identical pixels but different bytes (2237 vs 2234
  bytes for `circle_000.png`), so the dataset ids differ. This is documented in DATASETS.md.
- **Lock files:** `lock-cpu-linux.txt` (45 lines) and `lock-dev-cpu-linux.txt` (56 lines), both
  from `scripts/lock_cpu_linux.sh` in the pinned image.
- **Fresh-clone Quick Start B** (native Windows, PowerShell 5.1):
  - **Clone:** `git clone` of the local repo at 4c9f83f.
  - **First attempt:** it failed in `python -m venv`, because `ensurepip` hit the Windows
    260-character path limit inside the long scratchpad path. Long paths were not enabled (that
    is an admin change).
  - **Second attempt:** rerun in `<home>\vbqs`, with the commands as in the README.
    `py -3.11` was replaced by the uv-managed Python 3.11.15 path, because `py -3.11` is not
    registered on this machine.

| step | time |
|---|---|
| `venv` | 34.8 s |
| `pip install --upgrade pip` | 16.9 s |
| `pip install torch==2.14.1 torchvision==0.29.1 --index-url .../cpu` | 312.7 s |
| `pip install -e ".[dev]"` | 331.1 s |
| `npm ci` | 54.3 s |
| `npm run build` | 18.1 s |
| `visionbench demo-data var\demo` | 6.3 s |
| `visionbench dataset import var\demo\shapes --name demo-shapes` → `ds-f7fd8da0fba3` | 12.1 s |
| import check | `2.14.1+cpu 1.0.0rc1` |

  - **Then `visionbench serve`:**
    - health was `1.0.0rc1`, and `GET /` returned 200 text/html;
    - split job 28.8 s, CPU training 18.7 s, VAL evaluation 16.5 s, then predict 200;
    - VAL balanced accuracy 0.389 is a synthetic sanity check only.
  - **Clean-up:** the server was stopped, port 8000 was free, and the clone folder was deleted.

**B4 docs.**
- README: rewritten as described in CHANGELOG 1.0.0-rc1.
- Commands in the README that were run in this session:
  - the Quick Start A and B commands (above);
  - the Docker commands;
  - the Linux test command;
  - the frontend commands;
  - the curl lines carried over from 0c.
- Two troubleshooting rows came from what happened here: Windows long paths and the Git Bash
  path rewrite.
- No LICENSE file: owner TODO.

**Phase C preparation (b1a6ef3, before any TEST access).**
- **Changes:**
  - `run_benchmark.py freeze-check` and `evaluate-test`;
  - `run_robustness.py --split test`;
  - TEST support in `robustness.report`: the clean reference is the single TEST evaluation,
    0 or more than 1 is an error, with TEST wording. The VAL report regenerated from the real
    database **byte-identical** (md and json `cmp`);
  - `scripts/final_report.py`.
- **Tests:** `tests/robustness/test_robustness_test_split.py` (2) and
  `tests/unit/test_final_report.py` (4).
- **freeze-check trial on the real data root (read-only, 50 s):** hashes OK; plan sha256
  `11de6e1c…`; 0 TEST evaluations of any kind; `dataset verify --full` exit 0 with the
  fingerprint matching the protocol; 15/15 checkpoints matching their artifact rows and VAL
  evaluation records.
- **Dry run of `final_report.py --split val`** (scratchpad, not committed), labelled
  "VALIDATION (selection-biased) — not test results":
  - it reproduces the known VAL multi-seed summary exactly (for example efficientnet 0.9871 ±
    0.0027, resnet18_scratch 0.9125 ± 0.0485);
  - the plan was written and hashed before this dry run and was not changed after it.
- **Rehearsal of the whole C3 sequence on SYNTHETIC data:**
  - **Setup:**
    - a throwaway data root, with no contact with `./var`;
    - 15 tiny shapes runs mapped onto the five protocol model names;
    - a copy of the frozen protocol files, so the hashes verify;
    - the scripts unmodified.
  - **Results:**
    - `evaluate-test`: 15 TEST evaluations. A second call skipped all 15.
    - `robustness materialize --split test` without the unlock exited 1 with the locked message.
      With it, pass 1 wrote and pass 2 verified.
    - `run_robustness.py --split test`: 15 × 12 evaluated, all attempts completed.
    - `final_report.py --split test` wrote FINAL_REPORT.md/.json labelled TEST.
    - `check` then failed as expected, with 15 TEST evaluations and 180 TEST robustness
      evaluations. C6 changes it.
  - **Clean-up:** the throwaway root was deleted.

**Gate B** (code at 3677583; the tag is on a docs-only commit after it):

| check | result |
|---|---|
| `ruff check .`, `ruff format --check .` | pass (147 files) |
| full pytest, clean export of 3677583, Windows normal | **359 passed, 1 skipped, 1 warning** (590.24 s) |
| same, `CUDA_VISIBLE_DEVICES=-1` | **359 passed, 1 skipped, 1 warning** (626.96 s) |
| same, Linux container (non-root, CPU torch) | **360 passed, 1 warning** (411.68 s) |
| frontend lint / typecheck / check:api / test:coverage / build | pass / pass / up to date / 42 passed, 87.67 % lines / pass |
| Playwright e2e (after the e2e_backend change) | **1 passed** (2.7 min) |
| `run_benchmark.py check` | 0 clean TEST, 0 robustness TEST evaluations (whole DB) |
| `sha256sum -c` protocol (6) + plan (1) + robustness (2) | 9/9 OK |

- The one warning, in every run, is a `StarletteDeprecationWarning` from
  `fastapi/testclient.py`: "Using `httpx` with `starlette.testclient` is deprecated; install
  `httpx2` instead". It concerns the test client only, and was not acted on.
- The skipped Windows test is the file-symlink case.

## Phase C — one-time TEST evaluation (STOPPED in C3)

**C1** (3677583): `FINAL_ANALYSIS_PLAN.md`, sha256
`11de6e1ccae6866b604a708d948253714862a924b5585e0af976377169eb59fc`, in
`FINAL_ANALYSIS_PLAN_HASHES.txt`. At that commit the database had 0 TEST evaluations of any kind
(`check`).

**C2 freeze checklist:**

| item | result |
|---|---|
| a. clean tree; HEAD = rc1 or a docs-only descendant | clean; HEAD 288dbe1 = v1.0.0-rc1 + MASTER_PROGRESS.md only |
| b. hashes verify; plan commit precedes any TEST access | protocol 6/6, robustness 2/2, plan 1/1 OK; plan committed at 3677583 with 0 TEST rows |
| c. check: 0 TEST evaluations, 0 TEST materializations | 0 clean, 0 robustness TEST evaluations; no TEST condition in the cache |
| d. `dataset verify --full` passes; fingerprint matches | exit 0; `53a853ae…` = protocol |
| e. 15 checkpoint sha256 = artifact rows | 15/15 (file = artifact row = VAL evaluation record) |
| f. full suite on this commit in both modes | `git archive v1.0.0-rc1`: 359 passed / 1 skipped, normal 555.24 s, GPU hidden 554.73 s |
| g. authorization line | "proceed automatically" |

Items b-e ran as the first step of the C3 runner (`freeze-check`, exit 0 in 44.2 s, 11:24:33
UTC).

**C3, as run.**
- **Runner:** a scratchpad script that runs the plan's §2 commands in order, with keep-awake, and
  stops at the first non-zero exit. `VISIONBENCH_TEST_IMAGES_UNLOCKED=true` was set only for the
  materialize and report processes, and inside `evaluate-test` only for its evaluation children.
- **`run_benchmark.py evaluate-test`:** exit 0 in 466.2 s. All 15 runs on attempt 1:
  - majority_baseline: `ev-20261029-112504-ec73cf`, `ev-20261029-112523-9b58fd`,
    `ev-20261029-112543-28d2fa`;
  - resnet18_scratch_64: `ev-20261029-112608-c4d668`, `ev-20261029-112633-e4bace`,
    `ev-20261029-112658-df5146`;
  - small_cnn_scratch_64: `ev-20261029-112722-d17c6a`, `ev-20261029-112747-79a389`,
    `ev-20261029-112811-931ebd`;
  - resnet18_imagenet_224: `ev-20261029-112850-a00cad`, `ev-20261029-112930-d32a72`,
    `ev-20261029-113009-789459`;
  - efficientnet_b0_imagenet_224: `ev-20261029-113052-84ed06`, `ev-20261029-113134-691d5f`,
    `ev-20261029-113216-4c6760`.
  - Database afterwards: 15 TEST evaluations, 15 distinct experiments, checkpoints `best` and
    `baseline`, none forced.
- **`robustness materialize ... --split test` (pass 1):** exit 1 in 16.4 s, with
  `error: gaussian_noise/0.04: stored condition covers different images`. **STOP.**
- **Not run:** materialize pass 2, `run_robustness.py --split test`, `final_report.py`.

**After the stop (read-only checks):**
- The 12 VAL conditions verify through `load_verified_condition` (condition hash and every file's
  sha256), all with `split: val`.
- The newest file in the cache is from 06:25:54 UTC (M6).
- `check` now raises its assertion, listing the 15 TEST evaluation ids. That is expected:
  extending it is C6.
- The app is still locked.

The diagnosis and the owner options are in the top summary.

