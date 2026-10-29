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

