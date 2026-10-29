# Robustness protocol: EuroSAT v1, corruption suite `eurosat-c-v1`

**Status:** frozen. Written and committed **before** any robustness evaluation (task VB-M6).
This file and `robustness_suite.yaml` are hashed in `ROBUSTNESS_HASHES.txt`. Any later change
is appended under [Amendments](#amendments) with its date and reason, and every affected
evaluation is redone for all runs. Nothing above that section is edited after the freeze.

It extends, and does not change, the benchmark protocol
[`../BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) (dataset `ds-61112928e412`, split
`sp-55320e11726c`, 15 runs in `../runs.json`, val-selected `best.pt` checkpoints).

## 1. Scope

The suite contains **synthetic corruptions at 3 severities, on one dataset**. Results describe
each model's sensitivity to **these corruptions only**. They do not show general robustness,
robustness to real sensor degradation (atmosphere, clouds, haze, sensor noise of a real
instrument), or robustness to distribution shift (other regions, seasons, sensors). No model is
called "robust" on the basis of this suite.

## 2. Corruptions (`robustness_suite.yaml`)

4 corruptions x 3 severities = **12 conditions**:

| corruption | parameter | severities |
|---|---|---|
| `gaussian_noise` | sigma of additive N(0, sigma) per pixel and channel (pixel range [0, 1]) | 0.04, 0.08, 0.12 |
| `gaussian_blur` | sigma in pixels (at 64 px) | 0.5, 1.0, 1.5 |
| `brightness` | delta added to V of HSV (V in [0, 1]), clipped to 1 | +0.1, +0.2, +0.3 |
| `jpeg` | Pillow JPEG quality (4:2:0 chroma subsampling) | 50, 25, 10 |

Severities are fixed now and never tuned on results.

## 3. Exact procedure

Each corruption is applied to the **original decoded 64x64 RGB image** (the shared decoder:
full decode, EXIF orientation applied), **before** each model's own val transform. Every
model is evaluated on the same corrupted files.

1. Decode the snapshot image to RGB `uint8`, H x W x 3.
2. `gaussian_noise`, `gaussian_blur`, `brightness`: convert to float64 in [0, 1] (`x / 255`).
   - noise: `x + rng.normal(0, sigma, size=(H, W, 3))` (NumPy `default_rng`, PCG64).
   - blur: separable Gaussian, kernel radius `r = ceil(3 * sigma)`, weights
     `exp(-i^2 / (2 sigma^2))` for `i = -r..r`, normalised to sum 1; horizontal pass then
     vertical pass; padding NumPy mode `"reflect"` (mirror without repeating the edge pixel:
     `d c b | a b c d | c b a`).
   - brightness: `V = max(R, G, B)`, `V' = min(V + delta, 1)`; every channel is scaled by
     `V' / V` (pixels with `V = 0` become grey `V'`). This is exactly an HSV round trip with H
     and S unchanged and V replaced by `V'`.
   - then clip to [0, 1] and quantize: `uint8 = rint(x * 255)` (round half to even).
3. `jpeg`: the `uint8` image is encoded with Pillow (`quality=q`, `subsampling=2` = 4:2:0,
   `optimize=False`, `progressive=False`) and decoded back to RGB `uint8`. Bytes depend on the
   Pillow / libjpeg build; materialization records Pillow 12.3.0 with libjpeg-turbo (JPEG
   library version 8.0) for this run.
4. The result is stored once as a **lossless PNG** (Pillow defaults, no metadata) at
   `<data_root>/cache/corruptions/<dataset_id>/<split_id>/<suite>/<corruption>/<severity>/<class>/<file stem>.png`.

**Randomness.** Per image and condition:
`seed = int.from_bytes(sha256(f"{suite}|{corruption}|{severity}|{base_seed}|{relpath}").digest()[:8], "big")`,
with `severity` written as the YAML value prints in Python (`0.04`, `50`), `base_seed = 0` and
`relpath` the snapshot relpath. The seed is independent of batch order, model and process. Only
`gaussian_noise` draws random numbers.

**Condition manifest.** Each condition directory holds `condition.json`: suite name and
sha256 of `robustness_suite.yaml`, corruption, parameter, severity, base seed, dataset, split id,
split-manifest sha256, split name, library versions, and one entry per image (relpath, label,
source sha256, corrupted-file sha256). The **condition hash** is the sha256 of the canonical
JSON of these fields (entries sorted by relpath). Materializing again must reproduce every
file byte for byte (checked by hash), and evaluation re-hashes every corrupted file and refuses
a condition whose hashes do not match.

## 4. Evaluation

- **Checkpoint:** the val-selected checkpoint of every run (`best.pt`; `baseline` for the
  majority baseline), loaded through the verified evaluation path (dataset verify, manifest
  fingerprint, checkpoint sha256, `weights_only=True`, recorded val transform).
- **Clean reference:** the run's existing clean VAL evaluation from B1, **pinned by evaluation
  id** (`val_eval_id` in `../runs.json`). Clean and corrupted predictions cover the same 4,050
  val images in the same order.
- **Per condition:**
  - accuracy, balanced accuracy and macro F1, with 95 % percentile bootstrap CIs (1000
    resamples, seed 0);
  - per-class recall;
  - the **paired difference corrupted - clean** in balanced accuracy and macro F1: paired
    percentile bootstrap on the same resampled image indices (1000 resamples, seed 0).
- **Aggregates per run:**
  - mean corrupted balanced accuracy over all 12 conditions;
  - mean corrupted balanced accuracy per corruption (over its 3 severities);
  - relative drop `(clean - corrupted) / clean`.
  - **No reference-model normalization is used** (this is not ImageNet-C's mCE).
- **Majority baseline:** included. Its predictions do not depend on the image, so its corrupted
  metrics must equal its clean metrics exactly. This is a check on the pipeline.
- **Storage separation:** results go to a separate table, `robustness_evaluations`, and to
  files under `<run_dir>/robustness/<id>/`. They are never written to the `evaluations` table
  and never mixed with clean metrics.

## 5. Reporting

- **Per model:**
  - mean ± sample sd across the 3 seeds of clean VAL balanced accuracy and of corrupted
    balanced accuracy;
  - the drop, per corruption and severity, with the per-seed values.
- **Seed 0:** paired differences with their CIs.
- **Worst condition:** per model, the per-class recall drop for the worst condition. SeaLake
  and Forest are flagged as largely low-information (DATASETS.md).
- **Labelling:** every number is labelled **"VALIDATION (selection-biased) — not test
  results"**.
- **Comparisons:**
  - a difference is called distinguishable from zero only where its paired CI excludes zero;
  - statements are only about these corruptions;
  - no model is called "robust".

## 6. Confounds stated with every result

| Confound | Effect |
|---|---|
| **Resolution / upsampling** | Corruptions are applied at 64 px. Pretrained models then upsample to 256 and centre-crop 224, which smooths high-frequency corruptions (especially noise and JPEG artefacts) before the model sees them; scratch models see them at native resolution. Differences between the two groups may reflect this, not the networks. |
| **Pretrained vs scratch** | ImageNet pretraining and input resolution are confounded with each other (benchmark protocol §7). |
| **Centre crop** | Pretrained models never see the 4 px border of a tile, corrupted or not. |
| **Single split** | One fixed split: no split-to-split variance. |
| **n = 3 seeds** | sd from three values is itself uncertain; per-seed values are shown. |
| **Selection bias** | VAL chose each checkpoint, so clean VAL numbers are optimistic; the corrupted numbers share that bias. |
| **Low-information classes** | SeaLake (78 %) and Forest (34 %) are largely near-uniform tiles: noise and brightness change them strongly in relative terms. |

## 7. Test policy and plan

- **M6 (now):** VAL only. All 15 runs x all 12 conditions.
- **M8:** TEST exactly once, with the same suite and the same procedure, for all 15 runs,
  together with the clean test evaluation.
- **Until M8:** the robustness code refuses `split=test`, by both materialization and
  evaluation (CLI: an error; API: `403 test_split_locked`), unless
  `VISIONBENCH_TEST_IMAGES_UNLOCKED=true`.
- **No selection on robustness:** no model, seed, checkpoint or hyperparameter is chosen from
  robustness results.

## Amendments

Each amendment is appended here with the date, what changed, why, and which evaluations were
redone (all runs).

### Amendment 1 — 2026-10-29 — text only: correction of the resolution/upsampling confound note (section 6)

**What changed:** only the explanation of the "Resolution / upsampling" confound. No
procedure, suite, corruption, severity, seed, storage rule or evaluation changes; nothing is
rerun. Section 6 above is left as frozen; this amendment supersedes its first row.

**Why:** the note says that upsampling 64 -> 256 -> centre crop 224 "smooths high-frequency
corruptions ... before the model sees them". That was a reasoning error. Resizing scales the
corruption **together with the image content**: the resize scales by 256 / 64 = 4 and the
centre crop does not rescale, so a blur of sigma = 1 px at 64 px becomes a blur of about 4 px in
the 224 px model input (the often-quoted 224 / 64 = 3.5 would apply to a direct resize to 224),
noise grains become blobs of about 4x4 pixels, and JPEG 8x8 blocks become 32x32 blocks. Relative
to the content, the corruption is not smoothed away.

**Corrected note:** corruptions are applied at 64 px; pretrained models see an upsampled
(256, centre crop 224) version in which corruption and content are scaled together, while
scratch models see both at native resolution. The input statistics therefore differ between the
two groups, but not by a smoothing of the corruption relative to the content.

**Relation to the VAL results:** the measured VAL direction (the pretrained 224 px models fall
further than the scratch 64 px models under noise, blur and JPEG; `results/robustness_val.md`) is
consistent with this correction but **not explained by it**: pretraining and input resolution
remain confounded with each other, so no cause can be attributed.

**Evaluations redone:** none (text-only amendment).
