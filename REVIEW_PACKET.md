# Review Packet — VB-M5

**Task:** VB-M5, "Prediction and Grad-CAM served from the API process (with sanity checks)".
**Dates:** 2026-10-28/29.
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130; NVIDIA GeForce
RTX 3050 6GB Laptop GPU; CPU Intel64 Family 6 Model 186.

All numbers come from commands run in this session. Scratchpad paths are written `<scratchpad>`.
Demo numbers are **"VALIDATION images — demo, not results"**: 20 val images, which also chose
each checkpoint, say nothing about model quality. No test-split image was used in any demo or
check.

## Summary

1. **Step 0a:** `experiments.stopped_reason` stored (additive schema v6) and set when a run
   completes. 27 of 27 existing experiments were backfilled from their own `train.log`; no other
   field changed (checked against a dump taken before the backfill).
2. **Step 0b:** the full-length rerun of resnet18_scratch_64 seed 1 (`exp-20261028-090206-78eeb8`,
   "repro-check") is **bitwise identical** to `exp-20261027-202124-e89a98`: all 10 epochs of
   history (every field except `epoch_seconds`), all 122 tensors of `best.pt` and of `last.pt`,
   best epoch, best value and stopped_reason. Its wall time (20.7 h) is not representative: the
   laptop was in Modern Standby and then asleep for most of the run (section 2).
3. **Predict and Grad-CAM work end to end** on the four real seed-0 models through
   `visionbench serve` (CPU inference): 80 predict + gradcam pairs, an out-of-distribution
   upload, and a TEST request refused with `403 test_split_locked`.
4. **Geometry:** the CAM is mapped onto the original through the exact inverse of the val
   transform. The border removed by resize 256 + centre crop 224 (23.4 % of a 64x64 tile) is
   hatched and labelled "not seen by the model" in the overlay and reported in the response.
5. **Sanity checks pass**, with the thresholds fixed beforehand: localization 75/75 (100 %);
   target sensitivity mean Pearson r = −0.606; model randomization mean |Spearman ρ| = 0.129;
   identical to a backward-hook reference implementation; no hook leaks. **The first
   localization design failed (24/80, chance level) and was replaced** (section 4.1).
6. **Event loop:** `GET /api/health` took 13.6 ms during a 1.5 s prediction (8.6 ms with the GPU
   hidden). A control that runs the model on the loop took 1321 ms. Concurrency is bounded
   (429 tested).
7. **Lint and tests:** ruff passes; 316 passed and 1 skipped in both modes; coverage 94 %.

---

## 1. Commits (local only; nothing pushed)

```
5136b1c 2026-10-29T09:44:11+05:00 M5: predict and gradcam API endpoints, image upload parser, tests and demo script
e5e9a6f 2026-10-29T09:24:33+05:00 M5: Grad-CAM with exact val-transform geometry, overlay and sanity checks
ea64ec1 2026-10-29T09:23:39+05:00 M5: inference service (verified loading, LRU cache, concurrency slots, test lock)
14305b1 2026-10-29T09:12:57+05:00 M5 step 0: stopped_reason in the DB (schema v6, backfilled); full-length repro check
```

The docs and this packet are committed after these four commits.

- **Order:** Step 0 → inference service → Grad-CAM → API → docs.
- **`ea64ec1`:** contains `service.py` without the `gradcam` method; `e5e9a6f` adds it, so each
  commit holds one layer.
- **Step 0 tree checked alone:** exported with `git checkout-index` and run with `PYTHONPATH`
  pointing at it (the import resolved to the exported copy): 285 passed, 1 skipped.
- **Commits 2 and 3 checked alone:** lint, the related tests (commit 2: 29 passed, 1 skipped;
  commit 3: 23 passed) and a predict smoke test on the real EuroSAT model.
- **Not committed:** overlays, images, weights and `var/`.

## 2. Step 0

### 2a. stopped_reason (schema v6)

- **Schema:** `experiments.stopped_reason VARCHAR(32)` is an additive column (`_COLUMN_UPGRADES`).
  It is set in `run_training` when a run completes and stays NULL for failed or cancelled runs.
  The experiment summary in the API shows it.
- **Tests:** the v4 → v6 upgrade test now drops the column too and checks that it comes back; the
  early-stopping test asserts the stored value.
- **Backfill:** `scripts/backfill_stopped_reason.py` reads the line `completed: ... stopped=<reason>`
  in each completed run's own `train.log` and sets only `stopped_reason`.

```
backfilled 27; already set 0; not completed 0; no usable log line 0
```

- **Values:** epochs_completed 9, max_steps 8 (the pilot determinism runs), early_stopping 7,
  baseline_fit 3.
- **Other fields unchanged:** every other column of all 27 rows is equal to the dump taken before
  the backfill.
- **Consistency:** for all 15 benchmark runs the DB value equals the value in the driver log used
  in B1.

### 2b. Full-length reproducibility check (`benchmarks/eurosat-v1/repro_check.json`)

`scripts/repro_check.py` reruns the frozen `resnet18_scratch_64.yaml` (sha256 checked against
`PROTOCOL_HASHES.txt`) with seed 1 on CUDA through `visionbench train`, as a new experiment. No
other GPU work ran meanwhile. The rerun is excluded from all benchmark results: it is not in
`runs.json`. `run_benchmark.py check` now also covers it.

| compared | reference `exp-20261027-202124-e89a98` | repro-check `exp-20261028-090206-78eeb8` |
|---|---|---|
| status / stopped_reason | completed / early_stopping | completed / early_stopping |
| best epoch / best val_loss | 5 / 0.39823326600922476 | 5 / 0.39823326600922476 |
| history, 10 epochs: train_loss, val_loss, val_accuracy, val_balanced_accuracy, lr, steps, is_best, train_samples, epoch | | **0 differing values** |
| best.pt (epoch 5): 122 tensors | | **0 differing**, max abs diff 0.0, metrics identical |
| last.pt (epoch 10): 122 tensors | | **0 differing**, max abs diff 0.0, metrics identical |
| train_seconds | 371.7 | 68442.0 (not representative) |

**Timing anomaly, reported and not acted on.** The Windows System log (Kernel-Power 506/507/42/107,
Power-Troubleshooter 1) shows the following:
- Before 09:02:04Z the laptop was in Modern Standby. The experiment was created at 09:02:06Z;
  the script had started at 07:18:35Z.
- It entered Modern Standby again at 09:08:17Z, during epoch 3.
- It slept from 11:41:38Z to 03:49:34Z on 10-29.

So epoch 3 took 67,390 s. The other epochs took 90–154 s, against 30–52 s in the benchmark run,
which suggests a lower power state after resume (not measured). `repro_check.py` does not ask
Windows to stay awake; the benchmark driver does. Determinism was unaffected.

## 3. Design

- **Inference service** (`src/visionbench/inference/service.py`, printed in full in the
  handover):
  - **Loading:** only through `open_for_evaluation` + `load_for_evaluation`. The experiment must
    be completed; then come the dataset fast verify and fingerprint, the recorded manifest, the
    checkpoint sha256 against its artifact row, `weights_only=True`, the experiments directory
    only, and a rebuilt val transform that must equal the recorded one.
  - **Cache:** LRU of size `inference_cache_size` (default 2), keyed by (experiment, checkpoint
    sha256). The checkpoint is re-hashed on every request, so a tampered file is refused and
    never served from the cache (tested).
  - **Device:** `cpu` by default.
  - **Concurrency:** a semaphore with `inference_max_concurrent` slots (default 1) and a wait of
    `inference_wait_seconds`, then `429 inference_busy`. Each model also has its own lock.
- **Routes** (`src/visionbench/api/routes/inference.py`): `async` handlers. The body is read on
  the event loop (multipart in memory with a byte limit, or JSON of at most 8 KiB); the model
  work runs through `run_in_threadpool`.
- **Test lock:** an image is refused if its relpath is in the TEST split of the experiment's
  manifest, **or if its bytes are identical (sha256) to a test image**.
  - The second rule is my addition. Without it, the lock would be bypassed by uploading the test
    file or reading it from another dataset that holds the same file.
  - Uploads are still allowed.
  - The lock is lifted only by `VISIONBENCH_TEST_IMAGES_UNLOCKED=true` (M8).
  - Edited or re-encoded copies are not detected (documented).
- **Grad-CAM** (`src/visionbench/explain/gradcam.py`, printed in full in the handover):
  - A forward hook on the registry's target layer, and the gradient of the target logit via
    `torch.autograd.grad`.
  - **Deviation:** the task says "forward/backward hooks". `autograd.grad` gives the same
    gradient, a test proves equality with a backward-hook + `.backward()` implementation, and it
    writes no `.grad` into the cached shared model.
  - Channel weights are the spatial mean of the gradients, followed by ReLU, in eval mode.
  - The hook is removed in `finally`.
  - An all-zero CAM is flagged and never rescaled. In the overlay it shows the plain image with
    the notice; there is no zero-tint.
- **Geometry and overlay:** `explain/geometry.py` and `explain/overlay.py` (section 4.2).

## 4. Sanity checks (`tests/explain/`)

### 4.1 Grad-CAM (thresholds fixed before running; EXPERIMENT_PROTOCOL.md section 13)

Output from the final test run (identical in both modes):

```
GRADCAM localization: peak in the disk's quadrant for 75/75 held-out images (100.0%); model accuracy 100.0%
GRADCAM target sensitivity: mean Pearson r(true-class CAM, other-class CAM) = -0.606 over 100 pairs (50 pairs with an all-zero CAM skipped)
GRADCAM model randomization: Spearman rho(trained CAM, random-weights CAM) per image = -0.055, -0.146, 0.130, -0.003, -0.184, 0.312, 0.194, -0.007; mean |rho| = 0.129; 0 image(s) with an all-zero CAM
```

- **Localization:** SmallCNN (8x8 CAM) on a synthetic task.
  - The class is decided only by the colour of a disk (red, green or blue) placed in a random,
    known quadrant.
  - A white square distractor sits in a different quadrant in every image.
  - Training: 360 images; held out: 75.
  - Threshold: ≥ 80 %. Result: **100 %**.
- **Target sensitivity:** threshold mean r < 0.5. Result: **−0.606**.
  - 50 of the 150 (true, other) pairs had an all-zero CAM for the other class: no positive
    evidence for a class the image does not show.
  - Those pairs are skipped (correlation undefined) and counted in the output.
- **Model randomization:** the same architecture with re-initialised weights, on 8 images.
  Threshold: mean |ρ| < 0.5. Result: **0.129**.
- **Reference implementation:** on 5 images our CAM equals a textbook backward-hook +
  `.backward()` CAM (allclose).
- **Hooks:** after 100 successful and 100 failing calls (an invalid class index fails after the
  hook fired), no module has any hook and no parameter has a `.grad`.
- **All-zero CAM:** a hand-built model with only negative evidence returns zeros with
  `all_zero=True`; the positive class normalises to a maximum of 1.

**The first localization design failed, and that result is reported here.**
- **First design:** the class was the disk's QUADRANT (4 classes) and the distractor was a dark
  square.
- **Result:** test accuracy 100 %, but the peak hit the quadrant for only **24/80 = 30 %**
  (chance is 25 %).
- **Not a CAM bug:** I compared our CAMs with the independent backward-hook implementation, and
  the maximum difference was **0.0**.
- **Cause:** SmallCNN ends in global average pooling, so it can only read position from padding
  effects at the borders. Its class evidence is spread across the map (visible in the printed
  CAMs). That design tested the architecture, not the explanation.
- **Fix:** I replaced it with the standard design above (the class depends on what the disk is;
  the quadrant is only known) and kept the 80 % threshold. The reference-equality check is now a
  permanent test.
- **Also fixed in the same run:** a test bug. The hook test expected `.grad is None`, but the
  fixture's training had left gradients; the fixture now clears them.

### 4.2 Geometry (16 tests, against the real torchvision val transforms)

- **Configurations:** 64 px square resize (64x64 and 100x60 originals) and resize 256 + centre
  crop 224 (64x64, 100x60, 60x100).
- **Round trips:** original → input → original and back are exact (1e-9).
- **Real transform:** a 3x3 white marker in the original lands in the transformed tensor where
  the mapping predicts it, within 1 px, at 3 positions per configuration. The resized size equals
  torchvision's.
- **Cropped border:** for 64x64 with resize 256 + crop 224, the resized size is 256x256, the
  offset (16, 16), the seen box (4, 4)–(60, 60) and the unseen fraction 0.234375. A marker on
  pixels 0–2 never reaches the tensor (max 0.0) and is reported unseen. With a scratch 64 px
  model the same marker is seen and the unseen fraction is 0.
- **CAM-cell mapping:** a single hot cell of a 7x7 CAM maps to the right place in the original
  (within 1 px). The NaN fraction of the mapping equals the unseen fraction.
- **End to end:** an API test trains a resize-256/crop-224 resnet18 (fake ImageNet weights) on
  32 px images.
  - The response reports resized size [256, 256], offset [16, 16], seen box (2, 2)–(30, 30) and
    unseen fraction 1 − (28/32)².
  - In the overlay PNG the top border rows are greyscale (unseen) with hatch pixels, and the
    centre is coloured.

### 4.3 API tests (`tests/api/test_inference_api.py`, 8 tests)

**Dataset images:**
- VAL image: 200, probabilities sorted with sum 1, the checkpoint sha256 equals the artifact row,
  no warnings, and the note "VALIDATION". A cache miss and then a hit give deterministic results.
- TRAIN image: allowed, with a note.
- Baseline predict: returns the priors, with the note "does not look at the image".

**Test lock:**
- 403 `test_split_locked` for predict and gradcam of the CNN and for baseline predict. Baseline
  gradcam gives 422 first.
- An upload of the identical test file bytes also gives 403.
- With `test_images_unlocked=True`: 200 and a "TEST image" warning.

**Uploads:**
- A 64x64 upload to a 32 px model gets the OOD warning; a 32x32 upload has no warning; an RGBA
  upload gets a "converted" note.
- Errors: a body over the byte limit gives 413 `upload_too_large`; non-image bytes and a GIF give
  415 `unsupported_image`; a missing file part gives 422 `missing_file`; `text/plain` gives 415;
  the pixel limit gives 413 `image_too_large`; `target_class` on predict gives 422.

**Grad-CAM:**
- Scratch model: CAM 4x4, values in [0, 1], unseen fraction 0, the limitations text present and
  the overlay PNG ≥ 256 px. Explicit `target_class` works through both JSON and multipart.
- `unknown_class` 422; baseline `explanation_not_supported` 422; an extra JSON field gives 422.

**Service behaviour:**
- Busy: holding the only slot gives 429 `inference_busy`.
- LRU: with size 1, a cache miss, then a hit; a second model evicts the first; the first then
  misses again.
- POST without the client header: 403. Unknown experiment: 404.
- Tampered `best.pt` after caching: 422 "sha256 mismatch", not served from the cache.
- Event loop: see the summary. The control (`run_in_threadpool` replaced by a direct call)
  proves that the test detects blocking.

**Path hygiene:** no response body contains the absolute data root.

## 5. Demos on real EuroSAT seed-0 models (CPU) — "VALIDATION images — demo, not results"

**Server:** `visionbench serve --no-worker`, default settings (CPU inference, cache size 2,
1 slot). Each demo ran on a freshly started server, so the first request per model was cold.

**Image selection:**
- `scripts/demo_inference.py` draws 2 val entries per class with `numpy.random.default_rng(0)`
  from each class's val entries sorted by relpath.
- The list is written to `var/demos/m5/selection.json` before any request is sent.
- Selected images: AnnualCrop_264, AnnualCrop_545, Forest_185, Forest_1946,
  HerbaceousVegetation_104, HerbaceousVegetation_1212, Highway_250, Highway_650,
  Industrial_2157, Industrial_2386, Pasture_365, Pasture_536, PermanentCrop_2265,
  PermanentCrop_870, Residential_2845, Residential_58, River_192, River_699, SeaLake_11,
  SeaLake_39 (each `<Class>/<Class>_<n>.jpg`).

**The demo ran three times on the same selection.**
- Run 1: the legend text was cut off at the right edge.
- Run 2: after widening the canvas to fit the text, an all-zero CAM still showed as a dark
  tint.
- Run 3: after showing the plain image for an all-zero CAM. This is the run reported below.
- The predictions, confidences, CAM resolutions, unseen fractions and all-zero flags were
  **identical in all three runs**; only the rendering changed.
- Runs 1 and 2 remain in `var/demos/m5_run1_legend_cut/` and `var/demos/m5_run2_zero_tint/`
  (superseded).

### 5.1 Per model

| model | prediction = label (of 20) | CAM resolution | unseen fraction | all-zero CAMs |
|---|---|---|---|---|
| resnet18_scratch_64 | 17 | **2x2** | 0.0000 | 0 |
| small_cnn_scratch_64 | 18 | 8x8 | 0.0000 | 1 |
| resnet18_imagenet_224 | 20 | 7x7 | 0.2344 | 0 |
| efficientnet_b0_imagenet_224 | 19 | 7x7 | 0.2344 | 0 |

The counts above are 20 val images and are not a measure of accuracy.

### 5.2 Per image (predicted class and its probability; ✗ = differs from the label)

| image | true | resnet18_scratch_64 | small_cnn_scratch_64 | resnet18_imagenet_224 | efficientnet_b0_imagenet_224 |
|---|---|---|---|---|---|
| AnnualCrop/AnnualCrop_264.jpg | AnnualCrop | AnnualCrop (0.997) | AnnualCrop (0.989) | AnnualCrop (1.000) | AnnualCrop (1.000) |
| AnnualCrop/AnnualCrop_545.jpg | AnnualCrop | Pasture (0.956) ✗ | AnnualCrop (0.606) | AnnualCrop (0.985) | Pasture (0.974) ✗ |
| Forest/Forest_185.jpg | Forest | Forest (0.995) | Forest (0.998) | Forest (1.000) | Forest (0.999) |
| Forest/Forest_1946.jpg | Forest | Forest (0.995) | Forest (0.997) [CAM all zero] | Forest (0.999) | Forest (0.985) |
| HerbaceousVegetation/HerbaceousVegetation_104.jpg | HerbaceousVegetation | HerbaceousVegetation (0.630) | HerbaceousVegetation (0.981) | HerbaceousVegetation (1.000) | HerbaceousVegetation (0.998) |
| HerbaceousVegetation/HerbaceousVegetation_1212.jpg | HerbaceousVegetation | HerbaceousVegetation (0.519) | HerbaceousVegetation (0.889) | HerbaceousVegetation (1.000) | HerbaceousVegetation (1.000) |
| Highway/Highway_250.jpg | Highway | Highway (0.999) | Highway (0.999) | Highway (1.000) | Highway (1.000) |
| Highway/Highway_650.jpg | Highway | Highway (0.824) | River (0.569) ✗ | Highway (0.999) | Highway (0.997) |
| Industrial/Industrial_2157.jpg | Industrial | Industrial (0.998) | Industrial (0.963) | Industrial (1.000) | Industrial (1.000) |
| Industrial/Industrial_2386.jpg | Industrial | Industrial (0.976) | Industrial (0.967) | Industrial (1.000) | Industrial (1.000) |
| Pasture/Pasture_365.jpg | Pasture | Pasture (0.658) | Pasture (0.859) | Pasture (1.000) | Pasture (0.997) |
| Pasture/Pasture_536.jpg | Pasture | Pasture (0.998) | Pasture (1.000) | Pasture (1.000) | Pasture (1.000) |
| PermanentCrop/PermanentCrop_2265.jpg | PermanentCrop | PermanentCrop (0.982) | PermanentCrop (0.980) | PermanentCrop (1.000) | PermanentCrop (0.999) |
| PermanentCrop/PermanentCrop_870.jpg | PermanentCrop | PermanentCrop (0.585) | PermanentCrop (1.000) | PermanentCrop (1.000) | PermanentCrop (0.991) |
| Residential/Residential_2845.jpg | Residential | Residential (1.000) | Residential (1.000) | Residential (1.000) | Residential (1.000) |
| Residential/Residential_58.jpg | Residential | Residential (1.000) | Residential (1.000) | Residential (1.000) | Residential (1.000) |
| River/River_192.jpg | River | River (0.999) | River (1.000) | River (1.000) | River (1.000) |
| River/River_699.jpg | River | PermanentCrop (0.756) ✗ | PermanentCrop (0.699) ✗ | River (0.917) | River (0.999) |
| SeaLake/SeaLake_11.jpg | SeaLake | Forest (0.625) ✗ | SeaLake (0.705) | SeaLake (1.000) | SeaLake (0.999) |
| SeaLake/SeaLake_39.jpg | SeaLake | SeaLake (0.960) | SeaLake (0.979) | SeaLake (1.000) | SeaLake (0.999) |

- **Wrong and confident:** two wrong predictions came with probabilities of 0.956 and 0.974
  (AnnualCrop_545 → Pasture). Confidence is not correctness.
- **All-zero CAM:** small_cnn's CAM for Forest_1946 is all zero (no positive evidence at the
  target layer), although the prediction has probability 0.997. The response warns and the
  overlay highlights nothing.
- **Low-information classes:** SeaLake and Forest are largely low-information
  (DATASETS.md).

### 5.3 Timing on CPU (run 3; indicative; one laptop)

`model load` = `timing_ms.model_load` of the first request. That covers the verified loading
path: dataset fast verify of 27,000 files, manifest, checkpoint hash, weights. `total` covers
the whole service call; `forward` / `CAM` cover the model work only. Warm figures are medians of
19 requests.

| model | cold: model load ms | cold: predict total ms | cold: gradcam total ms | warm: predict total ms (forward) | warm: gradcam total ms (CAM only) |
|---|---|---|---|---|---|
| resnet18_scratch_64 | 6712 | 7140 | 288 | 127.4 (35.8) | 182.9 (25.9) |
| small_cnn_scratch_64 | 6700 | 6756 | 75 | 27.5 (4.5) | 78.6 (6.3) |
| resnet18_imagenet_224 | 6089 | 6271 | 256 | 146.2 (57.6) | 252.2 (62.1) |
| efficientnet_b0_imagenet_224 | 7079 | 7240 | 259 | 114.1 (64.8) | 240.5 (85.3) |

- **Re-hash cost:** the gap between warm total and forward is mostly the per-request checkpoint
  re-hash (45 MB for resnet18, 1.6 MB for small_cnn) plus the DB lookups.
- **Run-to-run variation:** times varied between demo runs; for example, warm predict for
  resnet18_imagenet_224 was 118.6 ms in run 1 and 146.2 ms in run 3.
- **CUDA, checked once outside the demo:** `inference_device=cuda` on resnet18_imagenet_224 and
  Highway_250 gave the same prediction as CPU, with probabilities within 6.7e-7 and normalised
  CAM values within 1.1e-4. Warm forward took 5.2 ms on CUDA against 52.6 ms on CPU.

### 5.4 Out-of-distribution upload

- **Input:** a generated 300x200 PNG with gradients, a yellow disk and a blue rectangle, not
  EuroSAT, sent to resnet18_imagenet_224.
- **Prediction:** **SeaLake, probability 0.995**.
- **Warnings:** `input differs from training data (64x64 images of dataset 'eurosat-rgb');
  prediction may be unreliable (out-of-distribution)`, plus the note that only the size is
  compared.
- **Unseen fraction:** 0.4896 (300x200 → 384x256 → crop 224). The overlay is
  `var/demos/m5/overlays/ood_upload_resnet18_imagenet_224.png`.

### 5.5 TEST image request (curl.exe, with the client header)

```
> curl.exe -s -i -X POST http://127.0.0.1:8000/api/experiments/exp-20261027-214915-e0d372/predict -H "X-VisionBench-Request: 1" -H "Content-Type: application/json" -d '{"dataset_id":"ds-61112928e412","relpath":"AnnualCrop/AnnualCrop_100.jpg"}'
HTTP/1.1 403 Forbidden
date: Thu, 29 Oct 2026 04:21:49 GMT
server: uvicorn
content-length: 192
content-type: application/json
vary: Origin
x-content-type-options: nosniff

{"error":{"code":"test_split_locked","message":"this image is in the TEST split of sp-55320e11726c; test images are locked until the final test evaluation (VISIONBENCH_TEST_IMAGES_UNLOCKED)"}}
```

- **Which image:** the alphabetically first test relpath. The same request to `/gradcam` was
  also 403 (run 1). Without the client header the request was `403 missing_client_header`.
- **README commands:** the curl lines in the README (a val image, an upload, an upload with
  `target_class`, the TEST request) were run as written in PowerShell 5.1 against a live server.

### 5.6 Overlay files (under `var/`, not committed; 81 PNGs)

Every dataset overlay has a footer with the EuroSAT/Copernicus attribution (DATASETS.md) and
"VALIDATION images - demo, not results".

- `var/demos/m5/overlays/<model>/<Class>__<Class>_<n>.png` for each of the 4 models
  (`resnet18_scratch_64`, `small_cnn_scratch_64`, `resnet18_imagenet_224`,
  `efficientnet_b0_imagenet_224`) and each of the 20 images:
  - AnnualCrop__AnnualCrop_264, AnnualCrop__AnnualCrop_545
  - Forest__Forest_185, Forest__Forest_1946
  - HerbaceousVegetation__HerbaceousVegetation_104, HerbaceousVegetation__HerbaceousVegetation_1212
  - Highway__Highway_250, Highway__Highway_650
  - Industrial__Industrial_2157, Industrial__Industrial_2386
  - Pasture__Pasture_365, Pasture__Pasture_536
  - PermanentCrop__PermanentCrop_2265, PermanentCrop__PermanentCrop_870
  - Residential__Residential_2845, Residential__Residential_58
  - River__River_192, River__River_699
  - SeaLake__SeaLake_11, SeaLake__SeaLake_39
- `var/demos/m5/overlays/ood_upload_resnet18_imagenet_224.png`

**Suggested for supervisor review.** Each one shows one property; none was chosen for looking
good:

1. `var/demos/m5/overlays/resnet18_imagenet_224/Highway__Highway_250.png`: the hatched,
   unseen 4 px border (23.4 %).
2. `var/demos/m5/overlays/small_cnn_scratch_64/Highway__Highway_250.png`: the same image with a
   64 px scratch model (nothing unseen, 8x8 CAM).
3. `var/demos/m5/overlays/resnet18_scratch_64/Highway__Highway_250.png`: the same image with a
   **2x2** CAM, too coarse to follow the road.
4. `var/demos/m5/overlays/small_cnn_scratch_64/Forest__Forest_1946.png`: an all-zero CAM,
   reported rather than rescaled.
5. `var/demos/m5/overlays/efficientnet_b0_imagenet_224/AnnualCrop__AnnualCrop_545.png`: a
   confident wrong prediction (Pasture, 0.974) with a focused-looking CAM.
6. `var/demos/m5/overlays/ood_upload_resnet18_imagenet_224.png`: the out-of-distribution
   upload (SeaLake 0.995; 49 % unseen).

## 6. Lint and full test suite (after the last code change)

```
$ ruff check . ; ruff format --check .
All checks passed!
118 files already formatted

$ pytest -q -rs -p no:cacheprovider --cov=visionbench
[measured] GET /api/health while training: n=31, max=19.0 ms
[measured] GET /api/health during a 1.5 s predict: 13.6 ms (thread pool); control with the model run on the event loop: 1321.1 ms
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
316 passed, 1 skipped, 1 warning in 630.53s (0:10:30)
TOTAL 94 % (inference/service.py 96 %, explain/gradcam.py 94 %, explain/geometry.py 97 %, explain/overlay.py 91 %, api/routes/inference.py 97 %)

$ CUDA_VISIBLE_DEVICES=-1 pytest -q -rs -p no:cacheprovider
[measured] GET /api/health while training: n=32, max=13.5 ms
[measured] GET /api/health during a 1.5 s predict: 8.6 ms (thread pool); control with the model run on the event loop: 1279.6 ms
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
316 passed, 1 skipped, 1 warning in 495.29s (0:08:15)
```

After this run only docs and this packet changed. The no-test check still holds:
`run_benchmark.py check` reports 0 test evaluations for the benchmark, pilot and repro-check
experiments and 0 in the whole database.

## 7. Limitations and what was not done

- **Coarse CAMs:** 2x2 for resnet18_scratch_64, 7x7 at 224 px and 8x8 for small_cnn. Grad-CAM is
  not causal and does not show correctness (the fixed text is in every response).
- **Synthetic sanity checks only:** the sanity checks use a synthetic task and SmallCNN. They do
  not show that the EuroSAT CAMs are faithful.
- **Test-lock gap:** the lock misses re-encoded, resized or edited copies of test images.
- **Concurrency:** more than one concurrent slot under real load was not tested, only the busy
  path and the per-model locks.
- **CUDA inference:** checked once by hand, not in the test suite.
- **Warm request cost:** the per-request checkpoint re-hash dominates warm request time for the
  larger models. That is the price of never serving a changed checkpoint.
- **Overlays not stored:** overlays are returned in the response and saved only by the demo
  script under `var/`. They are not stored as experiment artifacts; the task's design replaces
  the plan's "overlays as artifacts".
- **No frontend, no protocol change, no push.**
