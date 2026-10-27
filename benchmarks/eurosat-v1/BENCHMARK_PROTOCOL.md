# Benchmark protocol: EuroSAT v1 (`eurosat-v1`)

**Status:** frozen. Written and committed **before** any benchmark training run (task VB-B1).
This file and the YAML configs next to it are hashed in `PROTOCOL_HASHES.txt`. Any later
change is appended under [Amendments](#amendments) with its date and reason; nothing above that
section is edited after the freeze.

General rules that apply here too: [EXPERIMENT_PROTOCOL.md](../../EXPERIMENT_PROTOCOL.md).
Dataset provenance and license: [DATASETS.md](../../DATASETS.md).

## 1. Data

| | |
|---|---|
| Dataset | `ds-61112928e412` ("eurosat-rgb"), EuroSAT RGB, 27,000 images, 10 classes, 64x64 |
| Fingerprint | `53a853ae82ee65dd0359263e5a76d81889bd46f770f6224e58965c42637b852f` |
| Split | `sp-55320e11726c` (`vb-split-v2`) |

Split settings: dHash threshold 5, low-info std 3.0, popcount [8, 56], split seed 0,
fractions 0.7 / 0.15 / 0.15, label conflicts excluded (1 group, 2 files). That gives
train 18,898 / val 4,050 / test 4,050.

The split is chosen by the supervisor and is **fixed** for every run. The other D1 calibration
splits are not used.

Known data limitation: 3,357 images (12.4 %) are low-information, namely SeaLake 78 % and
Forest 34 % of their classes. Near-duplicates among them cannot be assessed by hashing
(DATASETS.md). Per-class results for these two classes are flagged in every report.

## 2. Models

One YAML per model in this folder:

| Run name (YAML) | Model | Weights | Fine-tune | Input | LR | Purpose |
|---|---|---|---|---|---|---|
| `majority_baseline.yaml` | majority_baseline | none | — | — | — | floor: predicts the train prior |
| `small_cnn_scratch_64.yaml` | small_cnn | scratch | full | 64 px | 1e-3 | small model at native resolution |
| `resnet18_imagenet_224.yaml` | resnet18 | ImageNet (IMAGENET1K_V1) | full | 224 px | 1e-4 | pretrained CNN |
| `efficientnet_b0_imagenet_224.yaml` | efficientnet_b0 | ImageNet (IMAGENET1K_V1) | full | 224 px | 1e-4 | pretrained efficient CNN |
| `resnet18_scratch_64.yaml` | resnet18 | scratch | full | 64 px | 1e-3 | **control** (see below) |

**Purpose of the control.** `resnet18_scratch_64` has the same architecture as
`resnet18_imagenet_224` but neither ImageNet pretraining nor upscaling to 224 px. Comparing it
with `small_cnn_scratch_64` (same input, same training recipe) isolates the effect of the
**architecture**. Comparing it with `resnet18_imagenet_224` (same architecture) shows the
combined effect of **pretraining + resolution**. These two factors stay confounded with each
other; the protocol does not separate them.

**Shared settings for all trainable models** (fixed now; no tuning on test, and no tuning loop
on val in this task):

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 1e-3 (scratch: small_cnn, resnet18-scratch-64); 1e-4 (pretrained fine-tunes) |
| Schedule | cosine over the configured epochs |
| Epochs | max 25 |
| Early stopping | on `val_loss`, patience 5, min_delta 0 |
| Batch size | 64 (reduced only if it does not fit in memory, then recorded as an amendment) |
| Augmentation (train only) | horizontal flip + RandomResizedCrop, scale (0.8, 1.0), ratio (0.75, 1.3333) |
| Class-weighted loss | off |
| Precision | fp32 (AMP off) |
| Determinism | on: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic, `torch.use_deterministic_algorithms(True)` in strict mode (always on in VisionBench training) |
| Data loading | `num_workers: 0` |
| Device | CUDA |
| Weight fallback | off: a pretrained run fails rather than silently training from scratch |

**Preprocessing**, as implemented by VisionBench:
- **Scratch models:** square resize to the input size; mean/std computed on the TRAIN split at
  that resolution.
- **Pretrained models:** the weights' own preprocessing (resize 256 → center-crop 224; ImageNet
  mean/std). Their training augmentation is RandomResizedCrop(224) applied to the 64-px
  original.
- **Validation:** deterministic (no augmentation).

## 3. Seeds and runs

- **Training seeds 0, 1, 2** for every model: 5 models × 3 seeds = 15 runs. The split is fixed,
  so **seed variation measures training variance only** (initialisation, batch order,
  augmentation draws). It does not measure split or data variance.
- The majority baseline is deterministic and uses no randomness. Its three seeds are
  identical; they are run anyway so that every model has the same reporting shape.
- Runs are started by `scripts/run_benchmark.py`, which reads these YAMLs and runs them
  sequentially through the `visionbench` CLI. Experiment ids and statuses are recorded in
  `runs.json`.
- **Failure policy:** a failed run is recorded and rerun **at most once** with the identical
  config. Both attempts are reported.
- **Pilot runs** (timing, memory, determinism checks) are listed in `pilot.json`, labelled
  "pilot", and **excluded** from all benchmark results.

## 4. Checkpoint selection

Identical for all trainable models: the **best checkpoint by `val_loss`** (`best.pt`), with
early stopping on `val_loss` (patience 5, min_delta 0, max 25 epochs). The majority baseline
has a single fitted checkpoint (`baseline`). `last.pt` is never used for results.

## 5. Metrics

- **Primary:** balanced accuracy and macro F1, with 95 % bootstrap CIs (1000 resamples,
  bootstrap seed 0).
- **Per class:** recall and F1 for every class.
- **Secondary:** accuracy, efficiency (parameters, checkpoint size, training time) and latency.
  Latency is forward pass only, at batch sizes 1 and 32, on CPU and CUDA, measured separately
  from any split on the seed-0 run of each trainable model. Seeds do not change the
  architecture.
- **Across seeds:** mean ± sample standard deviation (n = 3) of balanced accuracy and macro F1,
  plus the per-seed values.

## 6. Validation use (this task) and test policy

- **Now (VB-B1): validation only.**
  - Every completed run's selected checkpoint is evaluated on VAL.
  - `visionbench compare --split val` across the seed-0 runs; it carries the selection-bias
    note.
  - A multi-seed VAL summary.
  - Every validation number is labelled **"VALIDATION (selection-biased) — not test
    results"**: val chose each checkpoint, so it overestimates performance on new data.
- **Test policy:**
  - The TEST split is evaluated **exactly once**, at release (milestone M8), for **all seeds of
    all protocol models**.
  - It is reported as mean ± sd across seeds plus per-run bootstrap CIs.
  - **No model selection on test:** no seed, model, checkpoint or hyperparameter is chosen by
    looking at test.
  - Until M8 no test evaluation of any benchmark experiment may exist, and the driver checks
    this.
- **UI/demo model rule:** the model shown in the UI or demos is **seed 0** of each protocol
  model. This is fixed now so that no seed can be picked for looking good.

## 7. Confounds that remain, and how they are reported

| Confound | Reporting |
|---|---|
| Pretraining and input resolution are confounded with each other (ImageNet models at 224 px vs scratch at 64 px) | The control (`resnet18_scratch_64`) separates architecture from pretraining+resolution; the two remain joint. Stated in every comparison write-up. |
| Different learning rates for scratch vs pretrained (1e-3 vs 1e-4), not tuned per model | Listed with the results. Results are "default-recipe" results, not tuned optima. |
| Pretrained models: val/test use resize-256 + center-crop-224 (the border is cropped), while training uses RandomResizedCrop of the full tile | Listed. Inherent to using the weights' preprocessing. |
| Checkpoints selected on val (selection bias) | Every val number is labelled; `compare` prints the selection-bias note. |
| One fixed split: no split-to-split variance | Stated. Seeds give training variance only. |
| Three seeds only: sd from n = 3 is itself uncertain | The per-seed values are always shown next to mean ± sd. |
| Low-information classes (SeaLake 78 %, Forest 34 %): possible undetected near-duplicates across splits | Per-class results flag SeaLake and Forest. |
| Early stopping gives different epoch counts; training time measured on one shared laptop GPU (RTX 3050 6GB), `num_workers: 0` | Best epoch and train time are reported per run; times are machine-specific. |
| Latency measured on this machine only (forward pass, batch 1 and 32) | Device names are recorded in latency.json and printed. |
| GPU determinism verified only for the configurations checked in the pilot | Pilot results are reported; any failure is reported, not hidden. |
| ImageNet overlap with EuroSAT is not assessed (different domain; not checked) | Listed. |

`visionbench compare` additionally prints its own confound flags, such as a single seed per
model or different input sizes.

## Amendments

Each amendment is appended here with the date, what changed, why, and which runs were redone.
Any hyperparameter change made after seeing validation results requires redoing the affected
runs for **all seeds** of that model.

_None._
