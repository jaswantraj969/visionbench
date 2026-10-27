# Experiment Protocol

This protocol governs all benchmarks. Status per section: **implemented (M1)**, or
**planned** (rules that later milestones must follow).

## 1. Dataset format (implemented, M1)

```
<root>/<class_name>/<image files>
```

- Exactly one level of class folders directly under `<root>`. The folder name is the label.
- Supported extensions (case-insensitive): `.jpg .jpeg .png .bmp .webp`. Only the JPEG, PNG, BMP
  and WEBP decoders are allowed, so a file whose content is another format is rejected as corrupt.
- Nothing is silently dropped. Every skipped entry is reported in `report.json` under one of these
  categories:

| Category | Meaning |
|---|---|
| `zero_byte` | empty file |
| `unsupported_extension` | extension not in the list (e.g. `.txt`, `.gif`) |
| `unreadable_or_corrupt` | Pillow `verify()` fails, or the full `load()` of a reopened image fails (catches truncation, which `verify()` alone misses); also non-regular files |
| `too_large_pixels` | width × height > `max_image_pixels` (default 50,000,000) |
| `symlink` | symlink or Windows reparse point (incl. NTFS junction); never followed |
| `hidden_or_system_file` | dotfiles, `.DS_Store`, `Thumbs.db`, `desktop.ini`, Windows hidden/system attribute |
| `nested_directory` | a sub-directory inside a class folder (not descended into) |
| `invalid_class_name` | empty, leading dot, path separator or control character |
| `invalid_file_name` | backslash or control character in a file name |
| `file_outside_class_folder` | regular file directly under `<root>` |
| `path_collision` | two names that are equal after Unicode NFC + case folding |

- The pixel limit is applied explicitly per scan: Pillow's `MAX_IMAGE_PIXELS` is set to the
  configured value, and `DecompressionBombWarning` is escalated to an error. Pillow's global
  default is not relied on. `LOAD_TRUNCATED_IMAGES` is forced off.
- Requirements: at least 2 classes, and at least `min_images_per_class` (default 3) valid images
  per class. If either fails, import stops with an error naming the offending classes and nothing
  is written.
- Paths are stored as `<class>/<file>`: POSIX separators, Unicode NFC. Directories are visited in
  sorted order, so no result depends on filesystem listing order.
- **EXIF orientation policy (D1, dataset format 2, hash version 2).** Pixels are read in exactly
  one place, `visionbench.data.imageio.decode_image`. It decodes fully and applies
  `PIL.ImageOps.exif_transpose`, so every consumer sees the image as meant to be displayed:
  - the scan (recorded width/height, gray std, dHash);
  - training and evaluation (`training.data.load_image`);
  - the split audit (pHash, decoded-pixel digest, contact sheets).

  A JPEG stored sideways with `Orientation=6` and a physically rotated copy without EXIF get
  identical dimensions, hashes and pixels (tested). SHA-256 values remain those of the original
  file bytes. `dataset.json` and `files.json` record `format_version: 2`,
  `orientation: "exif_transposed"` and `hash_version: 2`. Snapshots written in another format
  are refused with a request to re-import; only synthetic data existed in format 1.

## 2. Snapshot and fingerprint (implemented, M1)

- Each accepted file is read once. The same bytes are hashed (SHA-256), decoded, and written to
  `datasets/<id>/raw/<class>/<file>`, so the snapshot matches its recorded hashes.
- **Fingerprint** = SHA-256 over the sorted lines `relpath \t label \t sha256 \n`. It is
  independent of creation/listing order and OS path separators. Tabs and newlines are rejected in
  names.
- **dataset_id** = `ds-` + first 12 hex of SHA-256(`fingerprint \n name`). An existing id is never
  overwritten.

## 3. Duplicate groups (implemented, M1)

- **Exact duplicates:** identical SHA-256 of file bytes.
- **Near duplicates:** an in-house 64-bit dHash. Steps: grayscale, LANCZOS resize to 9×8, then
  per row 8 bits `pixel[x+1] > pixel[x]`, packed row-major with the MSB first. Two images are
  near duplicates if the Hamming distance is ≤ `dhash_threshold` (default **5**).
- **Low-information guard:** an image is excluded from perceptual matching when any of these holds:
  - its grayscale standard deviation (0–255 scale, measured on a ≤256 px thumbnail) is below
    `low_info_std_threshold` (default **3.0**);
  - its dHash is degenerate (all 0 or all 1 bits);
  - its dHash popcount is below `low_info_popcount_min` (default **8**) or above
    `low_info_popcount_max` (default **56**). This rule was added in M2.

  Such images are listed as `perceptual_check_skipped_low_information`. Exact matching still
  applies to them. Without this guard, distinct uniform images (e.g. blank sky or sea tiles), which
  all hash to `0x0`, would merge into one large false group.

  The popcount rule catches smooth gradients. These have a high pixel std, so the std rule passes
  them, but their hashes are sparse. In the test fixture, two different sky-like gradients have
  std ≈ 38.6 and popcount 3, and sit 2 bits apart. Without the rule they form a false
  cross-class group (`tests/data/test_lowinfo_verify.py`).
- Pairwise comparison is vectorized: chunked NumPy XOR plus popcount, using `np.bitwise_count`
  when available, otherwise a 256-entry byte lookup table. Identical hashes are collapsed first.
- Groups are the connected components (union-find) of exact plus near edges. Every image belongs to
  exactly one group (a singleton if it has no duplicates).
  `group_id` = `g-` + first 16 hex of SHA-256 of the sorted member relpaths.
- **Label conflicts:** a group whose members have more than one label.
  - Policy `exclude` (default): the whole group is removed from every split and listed in
    `manifest.json` → `excluded`.
  - Policy `error`: the split is refused.
  - Labels are never changed.
- **Same-class duplicates** stay in the dataset, and their group is always assigned to one split.
- `report.json` (format 2) lists the low-information images per class with the reasons that
  apply: `std`, `degenerate`, `popcount`. The reasons overlap; one image can fail several tests.

## 4. Splits (implemented; algorithm `vb-split-v2` since M2)

1. Rebuild duplicate groups from the stored per-file records, using this split's thresholds.
2. Apply the conflict policy.
3. For each class: order its groups by `sha256(f"{seed}|{class}|{group_id}")`. This is pure
   hashing, so the order does not depend on any RNG implementation or NumPy version. (`vb-split-v1`
   in M1 used a NumPy `Generator` permutation instead. The algorithm string and `split_id` changed
   accordingly.)
4. Walk the groups in that order, giving each one to the split with the largest remaining deficit,
   where deficit = `fraction × class_total − assigned` (in images). Ties go to train, then val,
   then test.
5. Fractions are converted to exact rationals (`Fraction(repr(f))`), so tie-breaking is identical
   on every platform.
6. If any class has 0 items in train, val or test, the split errors and names the class and its
   counts. `--allow-missing-classes` turns this into a recorded warning instead.
7. **Leakage check:** computed from the final entries. It records how many groups and SHA-256 values
   appear in more than one split, and must be 0 by construction.
8. `split_id` = `sp-` + 12 hex of SHA-256 over (algorithm, dataset fingerprint, full split config).
9. `manifest.json` is canonical JSON: sorted keys, LF line endings, entries sorted by relpath, and no
   timestamp. The same inputs therefore give byte-identical manifests. `created_at` and the NumPy
   and Python versions are kept in `split_info.json`.
10. Re-running an existing split recomputes the manifest and compares it byte for byte. If it
    differs, the split is refused rather than overwritten.

Manifest fields:
- `entries`: one per file, with `relpath`, `label`, `sha256`, `group_id`, `split`.
- Metadata: `config`, per-class `counts` and `target_counts`, `split_totals`, `excluded`,
  `warnings`, `leakage_check`, `algorithm`, `visionbench_version`, `dataset_fingerprint`.

**Oversized-group warnings (D1).** A multi-member group is reported if it is larger than its
class's **val target** (`fraction_val x class_total`) or larger than **2 % of its class**. A
group is never split, so such a group can unbalance a split or concentrate one scene in one
split. The warnings are computed from the manifest (`split.group_size_stats`). They are written
to `split_info.json` (`group_size_warnings`) and printed by `dataset split`. They are **not**
part of `manifest.json`, so manifest bytes and split ids of existing configurations do not
change. Singletons are never reported.

## 4a. Independent split audit (implemented, D1)

`visionbench dataset audit-split <dataset_id> <split_id>` checks a split for cross-split
leakage **without trusting the duplicate grouping that produced it**. Outputs go to
`<data_root>/datasets/<id>/audits/<split_id>/` (`audit.json`, `audit.md`, `pairs.csv`,
`sheets/*.png`). Nothing is written into the write-once split directory.

1. Every image of the split is decoded again (orientation policy above).
2. **Independent detector, pHash (`phash-v1`, in-house):**
   - grayscale, Lanczos resize to 32x32;
   - 2-D DCT-II (`scipy.fft.dctn`, orthonormal), keeping the top-left 8x8;
   - each coefficient is compared with the median of the 63 AC coefficients (the DC term is
     excluded and its bit is always 0).

   It is computed for **all** images, low-information ones included. Because a median splits
   the 63 AC values into exactly 31 above and 32 below, every pHash has popcount 31, and **all
   pHash distances are even**. This is a property of the method, not an error.
3. The dHash is used again, over **all** images (low-information ones included and flagged).
4. For every **val and test** image, the audit finds the nearest **train** image by pHash and
   by dHash. Ties go to the first train image by relpath.
5. Reported:
   - distance histograms 0..16 (and >16) for both hashes, separately for informative and
     low-information query images;
   - the **consistency check**: informative val/test images whose nearest *informative* train
     image is within the dHash threshold. This must be 0 by construction; a non-zero value
     means a split bug;
   - val/test images whose nearest train image (low-info included) is within the threshold,
     broken down by which side is low-information. These are pairs the grouping could not see;
   - near-miss counts in the band threshold+1 .. threshold+3 (informative images);
   - cross-split exact SHA-256 matches (must be 0) and cross-split decoded-pixel matches
     (SHA-256 of the decoded RGB bytes, which catches re-encoded identical images);
   - group statistics (size distribution, largest group per class vs its val/test targets,
     oversized groups as above).
6. Pairs for visual inspection (`pairs.csv` and labelled contact sheets, at most 20 pairs per
   PNG; each pair shows pair id, classes, splits, low-info flags and both distances):
   - (a) the 60 closest cross-split pairs by pHash (each val/test image with its nearest
     train image);
   - (b) up to 40 seeded within-group pairs per dHash distance 1..threshold (informative
     members), to judge false positives;
   - (c) up to 40 seeded cross-split pairs from the near-miss band, to judge false negatives;
   - (d) up to 20 seeded low-information images per class.

   Contact sheets contain dataset images and are never committed.

## 4b. Threshold calibration procedure (D1)

The duplicate thresholds for a real dataset are chosen by the supervisor from evidence, not by
the implementer:

1. Import with the current defaults and record the import statistics (low-information counts
   per class and reason, groups).
2. Create splits with seed 0 and fractions 0.7/0.15/0.15 for dHash thresholds {0, 2, 3, 5, 8}
   at the default low-information settings. Add threshold 5 with popcount bounds {[4,60],
   [12,52]} and std thresholds {1.5, 6.0}; [8,56] and 3.0 are the defaults, so those runs
   would be duplicates.
3. Tabulate for each run:
   - multi-member groups;
   - largest group per class;
   - low-information images;
   - label-conflict groups;
   - oversized-group warnings;
   - split counts.
4. Run `audit-split` on the candidate thresholds and inspect the contact sheets.
5. The supervisor records the decision and its reasons. The chosen configuration is then fixed
   for all benchmark splits of that dataset.

The EuroSAT evidence is in REVIEW_PACKET.md (task VB-D1). No threshold has been chosen yet.

## 5. Seeds and determinism (implemented, M2)
Each training run does the following before building the model:
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, before any cuBLAS work;
- calls `set_seed(seed)`, which seeds Python `random`, NumPy, and torch (CPU and all CUDA devices)
  and sets `cudnn.deterministic=True` and `cudnn.benchmark=False`;
- calls `torch.use_deterministic_algorithms(True)` in **strict** mode, so an op without a
  deterministic implementation raises instead of warning.

The DataLoader shuffle uses its own `torch.Generator` seeded with the run seed. Workers get a
`worker_init_fn` that seeds NumPy and `random`; the default is 0 workers on Windows. Random
augmentations draw from the torch RNG in the main process (0 workers), so they follow the seed. The
seed, the `set_seed` report and `env_info` (including device and GPU name) are stored in `env.json`
and in the experiment row.

What has been verified, and only this:
- **CPU:** two identical runs give identical per-epoch metrics and identical `best.pt` tensors.
  This is enforced by a test.
- **CUDA:** for the M2 demo, `small_cnn` at 64 px, 15 epochs on an RTX 3050 6GB with
  torch 2.14.1+cu130, two identical runs in separate processes gave bitwise-identical histories and
  `best.pt` tensors. This is a single observation for one configuration. It is not a guarantee for
  other models, GPUs, drivers, or library versions, and it is not covered by the test suite.

Limits:
- The CPU and CUDA results of the same configuration differ, because they use different kernels.
- `num_workers > 0` has not been checked for determinism.
- AMP (`amp: true`, CUDA only) has not been tested and may break determinism.
- Ops that lack deterministic CUDA kernels make the run fail, with the error recorded.

## 6. Model selection and test use (implemented, M2–M3)
- Training builds data loaders for **train** and **val** only. Every access to manifest entries
  goes through `training.data.select_entries`, which rejects any other split name. A test patches the
  image loader and asserts that no test-split file is ever opened during a full training run.
- Early stopping and best-checkpoint selection use **validation metrics only**: `val_loss`
  (default, unweighted mean cross-entropy) or `val_balanced_accuracy`, with `patience` and
  `min_delta`. `best.pt` is the epoch with the best monitored value, and its epoch is recorded.
- **Test images are reachable only through `evaluation.data.select_eval_entries`.** It allows `val`
  and `test` and never `train`, because evaluating on the training split is not a supported report.
  Tests check isolation both ways: `evaluate --split val` opens no test image, and
  `--split test` opens no train or val image.
- The checkpoint reported on test is the **val-selected** `best.pt` by default. `last.pt` is
  evaluated only with `--checkpoint last`, and the result is recorded as `checkpoint: last`.
- **Test-use accounting:** every evaluation is a row in the `evaluations` table. The counts
  appear in every `metrics.json` (`test_use`) and in every comparison report (`test evaluations
  so far` column, plus a `repeated_test_evaluation` confound flag).
- **Forced test re-evaluation (since M4).** The first test evaluation of an experiment +
  checkpoint is free. A **second or later one is refused** unless it is explicitly forced with
  a non-empty reason (`--force --reason "..."` on the CLI, `"force": true, "reason": "..."` in
  the API). `force` without a reason is an error. The rule is one function
  (`evaluation.data.check_test_reuse`), called by the CLI, by the API before queueing (409
  `test_reevaluation_requires_force`), and again inside the job. A forced evaluation records
  `forced=true` and the reason in the `evaluations` row and in `metrics.json`
  (`test_use.forced`, `test_use.reason`); comparison reports list the reasons per
  experiment and raise a `forced_test_reevaluation` confound flag. Legitimate reasons are
  bug fixes in evaluation code, not "trying again". Val evaluations are never restricted.
- **Pinned evaluations.** `compare --eval-ids ...` (API: `eval_ids`) pins exactly which
  recorded evaluation is used per experiment, so a report can be reproduced after later
  evaluations exist; otherwise the latest one is used. Pinned ids are listed in the report.
- **Ranking uses validation only.** `compare` ranks models by validation balanced accuracy at the
  val-selected epoch, using competition ranks with ties marked `=`. Test metrics are reported for
  every model, with CIs and paired differences, **without a winner label**.
- **Validation comparisons are selection-biased.** A comparison on `--split val` states that
  the same validation data chose each model's checkpoint, so val metrics overestimate
  performance on new data (`selection_bias_note` in JSON, a note in Markdown).

## 7. Preprocessing (implemented, M2)
- **Pretrained weights:** use the weights' own transform. ResNet18: resize 256 (bilinear), then
  center-crop 224. EfficientNet-B0: resize 256 (**bicubic**), then center-crop 224. Both use ImageNet
  mean/std. The exact values are recorded in `config.json`. Overriding the input size of a
  pretrained model is an error.
- **Scratch models:** resize to `input_size × input_size` (bilinear) and normalize with per-channel
  mean/std computed on the **train split only**. The statistics use the model's resolution and no
  augmentation. They are cached as
  `<data_root>/cache/norm_stats/<dataset_id>/<split_id>/norm_stats_train_<N>px.json`, keyed by
  resolution and a fingerprint of the train relpaths and hashes, and recorded in `config.json`.
  Split directories stay write-once (since M3: a test checks that their files and hashes are
  unchanged by training). Tests check that the stats do not change when only val/test change, and
  do change when train changes.
- **Augmentation (train only):** default `RandomResizedCrop(scale 0.8–1.0)` plus horizontal flip.
  Validation transforms are deterministic. Both transform lists are recorded.
- **Class-weighted loss** (optional) uses `N / (K · n_c)` from **train counts only**.
- **Loss reporting:** when class weighting is on, the logged `train_loss` is the **class-weighted**
  cross-entropy (PyTorch's weighted mean per batch, averaged over samples). `val_loss` is
  **always unweighted** mean cross-entropy, so it stays comparable across runs with and without
  weighting. `config.json` records both definitions (`resolved.train_loss`, `resolved.val_loss`).
  Because the two can be defined differently, don't compare train and val loss directly in
  weighted runs.
- **Resolution confound:** `input_size` is recorded for every run (SmallCNN default 64, torchvision
  models 224). Comparisons across models at different resolutions confound architecture with input
  resolution, and M3 comparison reports must flag this.

## 8. Pretrained weight policy (implemented, M2)
- Weights come only from torchvision's weights API. They are downloaded through torch.hub into
  `<torch_home>/checkpoints/` with `check_hash=True`, and loaded with `weights_only=True`.
  `<torch_home>` is `VISIONBENCH_TORCH_HOME` if set, otherwise `<data_root>/cache/torch`. The
  resolved directory is recorded as `resolved.torch_hub_dir` in `config.json`.
- The file's SHA-256 must start with the hash prefix in its file name. The weights enum name, file
  path and SHA-256 are recorded in `config.json` and in the experiment row.
- If weights cannot be obtained, training **fails** with a clear message by default. Only with
  `--allow-weight-fallback` does it train from scratch instead. Then `weights="scratch"` is recorded
  everywhere, the reason is recorded, and `train.log` contains a `WEIGHT FALLBACK` warning. There is
  never a silent fallback.
- `finetune: head_only` freezes all parameters except the classifier head. Since M3, every
  BatchNorm layer whose parameters are all frozen is put back into eval mode after each
  `model.train()`. Its running statistics therefore stay at their pretrained values, and this is
  recorded as `resolved.frozen_bn: true` with `frozen_bn_layers` (20 for ResNet18). A test checks
  that the backbone running mean/var are unchanged after a head_only epoch, with a full-fine-tune
  control.

## 9. Baselines
- The majority-class baseline (implemented, M2) predicts the most frequent **train** class. Its
  probabilities are the train priors. It is saved as JSON and opens no images.
- It must be reported with every comparison (M3).

## 10. Evaluation metrics, CIs, efficiency and confounds (implemented, M3)

**Inputs.** Metrics are computed from saved predictions: true label, predicted label (the argmax
of the probabilities) and per-class probabilities. Evaluation re-uses the run's recorded
deterministic validation transform; it is rebuilt from `config.json` and checked string-for-string.
Before a checkpoint is loaded, these checks run:
- the fast dataset verify;
- the manifest fingerprint check;
- the manifest sha256 must equal the one recorded at training;
- the checkpoint sha256 must equal its artifact row.

The checkpoint is then loaded with `weights_only=True` from `experiments/` only.

**Point metrics** (`evaluation/metrics.py`):

| Metric | Definition | Undefined (→ `null` + reason) |
|---|---|---|
| accuracy | correct / n | never (n > 0 is required) |
| per-class precision | tp / (tp + fp) | class never predicted → `class_never_predicted` |
| per-class recall | tp / (tp + fn) | class absent from the evaluated set → `class_absent_from_eval_set` |
| per-class F1 | 2tp / (2tp + fp + fn), the same formula as scikit-learn ≥ 1.3 | class absent **and** never predicted |
| macro P/R/F1 | mean over classes whose value is defined; excluded classes listed in `macro_excluded_classes` | no class defined |
| weighted P/R/F1 | support-weighted mean over defined values | no class defined |
| balanced accuracy | mean recall over classes present in the set (= macro recall); absent classes listed | no class present |
| confusion matrix | counts (rows = true) + row-normalized | row of an absent class → `null` row |
| log loss | scikit-learn `log_loss` (clips probabilities to [eps, 1−eps], float64 eps) | no probabilities. `log_loss_clipped_samples` > 0 means the value is a **lower bound** |
| ROC-AUC (OvR macro; binary uses p(class 1)) | scikit-learn `roc_auc_score` | probabilities missing, or any class without ≥1 positive and ≥1 negative |
| average precision | per class + macro | per class: no positive/negative; macro: same rule as ROC-AUC |
| ECE | 15 equal-width bins on (0, 1], top-label confidence; Σ \|acc_b − conf_b\| · n_b / n; reliability-bin table included | no probabilities. **A calibration estimate that is noisy on small sets.** |

Undefined values are never reported as 0 or as an invented number. scikit-learn is called with
`zero_division=np.nan`, NaN is converted to `null`, and the reason goes into `metrics.undefined`
under the JSON path of the value. All metrics are validated against hand-computed fixtures:
absent class, never-predicted class, perfect predictions, constant majority predictor, and binary
vs multiclass.

**Confidence intervals.** These are 95% percentile bootstrap CIs over the evaluated samples, for
accuracy, balanced accuracy and macro F1. The defaults are 1000 resamples and seed 0; both are
recorded, along with the NumPy version, since the Generator stream is version-dependent. Resamples
where a metric is undefined are excluded and counted (`resamples_defined`). The CIs reflect
**test-set sampling variation only**, not training-seed variance.

**Paired differences** (in `compare`): metric(experiment) − metric(reference) on each resample,
using the **same resampled indices** for both models. The reference is the majority baseline if
present, otherwise the first experiment, or `--reference`.

**Latency** (`evaluation/efficiency.py` timing, `evaluation/latency.py` since M4):
- **Independent of splits (since M4).** Latency is its own operation (`visionbench latency
  <exp> --device cpu|cuda`, `POST /api/experiments/{id}/latency`). It loads the verified
  checkpoint and times random input; it reads **no split and no image**, so measuring
  latency never counts as a use of val or test. `evaluate` no longer measures latency.
- Measures the **model forward pass only**; preprocessing, decoding and data loading are excluded.
- Uses random input at the model's input size, `torch.inference_mode`, eval mode, 10 warmup
  iterations, then 50 timed iterations with `torch.cuda.synchronize` before and after each on CUDA
  (both configurable).
- Batch sizes are 1 and 32. Reported: median and p95 ms, plus device, GPU or CPU name, torch
  threads and input size. Each measurement is one device, stored as
  `experiments/<exp>/latency/<lat_id>/latency.json` and a `latency_measurements` row.
- A latency number is valid only for the device it was measured on. `compare
  --latency-device cpu|cuda` reports, for every model, the latest measurement on that one
  device (or "no latency measured on <device>"; without `--latency-device` no latency is
  reported), so a report never mixes devices; `latency_devices_differ` remains as a guard
  flag.
- The majority baseline has no forward pass: latency is not applicable (`null` + reason) and
  requesting it is an error.
- Efficiency also records parameter counts (total/trainable, from training), `best.pt` size in
  bytes, and training seconds.

**Confound flags** (computed automatically by `compare`):
- `input_resolution_differs`
- `pretrained_vs_scratch`
- `early_stopping_monitor_differs`
- `training_budget_differs` (configured epochs/max_steps; actual epochs trained shown)
- `seeds_differ`
- `single_seed_per_model` (seed variance unmeasured)
- `class_weighting_differs`
- `latency_devices_differ`
- `forced_test_reevaluation` (with the recorded reasons)
- `repeated_test_evaluation` (with per-experiment test counts)

`compare` refuses experiments with different dataset fingerprints or split_ids; there is no
override.

## 11. Synthetic sanity checks are not benchmarks
The M2, M3 and M4 demos use a generated shapes dataset to show that the pipeline works end to end. Their numbers are
**pipeline sanity checks on synthetic data, not benchmark results**, and must never be cited as
model performance.

## 12. Known limitations
- dHash finds resizes and re-encodes. It **misses** crops, rotations, flips, heavy color or contrast
  edits, overlays, and semantic near-duplicates (the same object photographed twice).
- The threshold trades false positives (distinct but similar images merged) against false negatives.
  The default of 5/64 is a heuristic and has not been tuned on real data yet.
- Low-information thresholds (std 3.0, popcount 8–56) are heuristics checked on synthetic images
  only and are not calibrated on real photos. The popcount bounds also exclude genuinely
  informative images whose hash happens to be sparse or dense. Those images still get exact
  matching but lose near-duplicate detection.
- Duplicate detection runs within one dataset only. Overlap with pretraining data (e.g. ImageNet)
  is not detected.
- Split assignment uses no RNG (`vb-split-v2`). The NumPy version is still recorded in
  `split_info.json`.
- GPU determinism has been observed for only one configuration (§5). `set_seed` alone uses
  `warn_only=True`, but training enables strict mode.
- Balanced accuracy averages recall over the classes **present in val**. A class with no val images
  is left out of the mean.
- The majority baseline floors zero priors at 1e-12 when computing loss.
- Since D1 every pixel consumer applies EXIF orientation (section 1). Other EXIF/ICC metadata
  (color profiles) is not applied.
- pHash (audit only) and dHash both miss crops, flips and strong color edits. pHash distances
  are always even (section 4a).
- Near-uniform images (open water, dense canopy) defeat both perceptual hashes: their hashes are
  driven by noise and compression artefacts. On EuroSAT many Forest and SeaLake images are
  low-information (REVIEW_PACKET.md), so near-duplicates among them are not detectable by
  hashing.
- Bootstrap CIs on small evaluation sets are coarse; with 54 test images a single error moves
  accuracy by ~0.019. Perfect predictions give a degenerate CI [1, 1], which does not mean the
  true error rate is zero.
