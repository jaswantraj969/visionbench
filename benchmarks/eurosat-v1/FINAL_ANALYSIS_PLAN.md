# eurosat-v1 — final analysis plan (TEST split)

**Status:** this plan is committed, and hashed in `FINAL_ANALYSIS_PLAN_HASHES.txt`, **before any
access to the TEST split**. It fixes what is evaluated, how, what is reported, and how the result
may be read. It does not change the frozen protocol (`BENCHMARK_PROTOCOL.md`, the five YAMLs,
`robustness/ROBUSTNESS_PROTOCOL.md`, `robustness/robustness_suite.yaml`) or any recorded run.

Nothing in this plan was chosen from a result:
- no model, seed, checkpoint, hyperparameter or severity is selected from TEST or robustness
  numbers;
- the 15 runs and their val-selected checkpoints were fixed when training finished (`runs.json`).

## 1. What is evaluated

**Runs:** all 15 runs of `runs.json`, i.e. 5 models × seeds 0, 1, 2:
- `majority_baseline`;
- `resnet18_scratch_64`;
- `small_cnn_scratch_64`;
- `resnet18_imagenet_224`;
- `efficientnet_b0_imagenet_224`.

**Checkpoint:** each run uses its val-selected checkpoint (`best.pt`; the baseline's fitted
priors). The checkpoint's sha256 is checked against its artifact row before the first evaluation.

**Data:** dataset `ds-61112928e412` and split `sp-55320e11726c`, TEST part (the 15 % test share
of the frozen split). The dataset is checked with `dataset verify --full`, and its fingerprint
must match the frozen value.

**Clean evaluation:** exactly **one** clean TEST evaluation per run, `visionbench evaluate <exp>
--split test` (device `cuda` for trainable models, `cpu` for the baseline, as on VAL).

**Robustness:**
- the 12 conditions of suite `eurosat-c-v1` (gaussian_noise, gaussian_blur, brightness and jpeg,
  each at its 3 fixed severities) are materialized on the TEST images **twice**, and the second
  pass must confirm byte identity;
- then **15 × 12** robustness TEST evaluations follow, each paired with that run's single clean
  TEST evaluation.

## 2. Execution (commands)

All commands run sequentially on the CUDA machine, with keep-awake and no other GPU work.
`VISIONBENCH_TEST_IMAGES_UNLOCKED=true` is set in the environment of these processes only; the
app stays locked until all of this is done.

```
python scripts/run_benchmark.py evaluate-test benchmarks/eurosat-v1
visionbench robustness materialize ds-61112928e412 sp-55320e11726c --suite benchmarks/eurosat-v1/robustness/robustness_suite.yaml --split test   # pass 1: writes
visionbench robustness materialize ds-61112928e412 sp-55320e11726c --suite benchmarks/eurosat-v1/robustness/robustness_suite.yaml --split test   # pass 2: verifies byte identity
python scripts/run_robustness.py benchmarks/eurosat-v1 --split test
python scripts/final_report.py benchmarks/eurosat-v1 --split test
```

**Driver checks** (`run_benchmark.py evaluate-test`):
- it refuses to start unless the protocol hashes and this plan's hash verify;
- it records each attempt in `results/test_runs.json`;
- it never evaluates a run again that already has a TEST evaluation row.

## 3. Failure policy

1. **Infrastructure failure with no metrics written** (the process or session died; no
   evaluation row exists): rerun the identical command.
2. **A row was already written:** no re-evaluation. The only exception is `--force --reason`,
   and only if the row is provably incomplete (for example missing or hash-mismatched outputs).
   The reason is stored and reported.
3. **Any non-infrastructure failure** (an error raised by the evaluation itself, a hash or
   fingerprint mismatch, a refused precondition): **stop**. Nothing is changed or rerun until the
   owner decides.
4. **No code changes** between the first TEST evaluation and the last robustness TEST
   evaluation.

## 4. Metrics and statistics

**Primary metrics:**
- **Primary:** balanced accuracy.
- **Secondary:** macro F1.
- Accuracy is reported, not interpreted.

**Per run:**
- the point value;
- a 95 % percentile bootstrap CI over the TEST images, with 1000 resamples and seed 0. These are
  the values stored by `evaluate`, and the report script checks that resamples = 1000 and
  seed = 0.

**Per model:** mean ± sample sd over the 3 seeds, with the per-seed values. The split is fixed,
so the sd reflects training variance only.

**Ranking:** models are ranked by 3-seed mean TEST balanced accuracy (competition ranks).

**Rank groups** are descriptive:
- Walk the ranking from the top.
- Two neighbours fall into different rank groups only if all three seed-matched paired CIs
  (§5) of their balanced-accuracy difference exclude zero, with the higher-ranked model ahead.
- Otherwise they share a group.

## 5. Pre-registered comparisons (confirmatory)

These three comparisons are the only confirmatory ones. Each is A - B:

1. `efficientnet_b0_imagenet_224` - `resnet18_imagenet_224`
2. `small_cnn_scratch_64` - `resnet18_scratch_64`
3. `resnet18_imagenet_224` - `resnet18_scratch_64`

**Method:**
- For each seed s ∈ {0, 1, 2}, run A seed s is paired with run B seed s.
- The test is a paired percentile bootstrap of metric(A) - metric(B) on the same TEST images
  (aligned by image), with 1000 resamples, seed 0, and resample indices shared by A and B.

**Verdict, per metric (balanced accuracy primary, macro F1 secondary):**
- **"distinguishable: A > B"** if all 3 seed-matched CIs lie above zero;
- **"distinguishable: A < B"** if all 3 lie below zero;
- **"not distinguishable"** otherwise.

There is no multiplicity correction beyond this strict all-three-seeds rule, and there are no
p-values. Comparison 3 contrasts pretraining *and* input resolution (224 vs 64 px), so it cannot
attribute a difference to either (§8).

## 6. Robustness tables (TEST)

These use the same report as the VAL robustness report (`visionbench.robustness.report`), on the
TEST split, written to `results/robustness_test.md` and `.json`.

**Per model:**
- clean balanced accuracy;
- the mean over the 12 conditions;
- drop and relative drop;
- mean ± sd over seeds, with per-seed values.

**Per corruption and per condition:** the same tables as on VAL.

**Seed 0:**
- the paired corrupted - clean CIs;
- per-class recall in the worst condition.

**Pipeline check:** the majority baseline must equal its clean metrics exactly.

## 7. VALIDATION next to TEST

For every model the report shows, side by side, the 3-seed mean ± sd on VALIDATION
(selection-biased) and on TEST, for balanced accuracy and macro F1. The VAL numbers are the
pinned B1 val evaluations, unchanged. The difference between them is described, not tested.

## 8. Fixed interpretation constraints

The report states these, whatever the numbers turn out to be:

- **Epoch cap:**
  - every trainable run had at most 25 epochs, with early stopping on `val_loss` (patience 5);
  - several runs stopped at the cap;
  - so the numbers describe these training budgets, not each architecture's best accuracy.
- **Untuned learning rates:**
  - learning rates and the other hyperparameters were fixed in the protocol, not tuned per model;
  - another setting could change any ranking.
- **Single split:** one fixed split of one dataset. There is no split-to-split variance, and
  nothing generalises beyond EuroSAT RGB.
- **n = 3:** three seeds per model. The sd is itself very uncertain, and the all-three-seeds
  rule is strict by design.
- **Low-information classes:**
  - many SeaLake (about 78 %) and Forest (about 34 %) tiles are near-uniform;
  - their near-duplicates cannot be detected by hashing, so leakage is bounded less tightly for
    those classes (DATASETS.md).
- **Resolution / pretraining confound:**
  - the ImageNet-pretrained models use 224 px inputs (upsampled from 64 px) and the scratch models
    64 px;
  - pretraining and input resolution are confounded, so group differences have no attributed
    cause.
- **Corruption scaling (corrected confound note,** robustness protocol amendment 1**):**
  - corruptions are applied at 64 px before each model's transform;
  - for the 224 px models corruption and content are upsampled together (a 1 px blur becomes
    about 4 px);
  - this is not a smoothing of the corruption relative to the content.
- **No robustness or state-of-the-art claim:**
  - the robustness numbers describe 4 synthetic corruptions at 3 severities only;
  - no model is called robust or state-of-the-art.

## 9. Outputs

- **Script:** `results/FINAL_REPORT.md` and `results/FINAL_REPORT.json` come from the committed
  `scripts/final_report.py`, which implements §4-§8.
  - It was dry-run on VALIDATION before any TEST access; that dry-run output is not committed.
  - TEST numbers are labelled "TEST".
- **Recorded files:**
  - `results/test_runs.json` holds the TEST evaluation attempts;
  - `robustness/robustness_runs_test.json` holds the robustness TEST attempts;
  - `results/robustness_test.md/.json` hold the TEST robustness tables.
- **After the report** (the only changes):
  1. the TEST lock is lifted for the app;
  2. the Benchmark page shows the TEST section;
  3. `run_benchmark.py check` asserts exactly 1 clean TEST evaluation and 12 robustness TEST
     evaluations per run.

## Amendment 1 (procedural), 2026-10-29

**The defect:** the first TEST materialization pass (§2) stopped with
`gaussian_noise/0.04: stored condition covers different images`.
- The corrupted-image cache path,
  `cache/corruptions/<dataset_id>/<split_id>/<suite>/<corruption>/<severity>/`, had no
  val/test component.
- The TEST pass therefore found the VAL conditions of the same split in the same folder, and
  refused.
- Nothing was written by that pass.

**The fix:**
- **Path:** the cache path now includes the split name,
  `cache/corruptions/<dataset_id>/<split_id>/<split>/<suite>/<corruption>/<severity>/`. This
  supersedes the storage path in `robustness/ROBUSTNESS_PROTOCOL.md` §3, step 4. That frozen file is
  not edited.
- **VAL cache:** the existing VAL cache was *moved* (not regenerated) into `.../val/`. All 12
  VAL conditions then verified file by file, and their condition hashes equal the M6 values
  recorded in the database.
- **Test:** a regression test materializes VAL and then TEST on the same split, and checks that
  both succeed and stay separate.

**What did not change:**
- the corruption functions and bytes;
- the seeds, the suite and its hashes;
- the evaluation logic;
- this plan's analysis (§4-§8).

**State and consequences:**
- No robustness TEST result existed when the defect was found and fixed: 0 robustness TEST rows.
- The 15 clean TEST evaluations (`results/test_runs.json`) were completed before the defect
  surfaced. They do not depend on the cache, and they are **not redone**.
- This amendment departs from §3.4 ("no code changes between the first TEST evaluation and the
  last robustness TEST evaluation"). The owner authorized it.
