# Project Status

_Last updated: 2026-10-27, end of M2 (pending supervisor review)._

## Current state
- **M0:** foundation (config, seeding, env info, health/ready API).
- **M1 (+ M2 step 0):** dataset layer.
  - Import, validation, stats, duplicate groups (with the popcount low-information guard).
  - `vb-split-v2` hash-ordered leakage-aware splits.
  - `dataset verify [--full]`.
- **M2:** model registry (majority baseline, SmallCNN, ResNet18, EfficientNet-B0), CLI training
  from split manifests (train/val only), checkpoints, and a SQLite experiment store with
  `experiments list|show`.
- There is no test-set evaluation (M3), no API endpoints beyond health/ready, no job runner,
  Grad-CAM, robustness, or frontend.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130, RTX 3050 6GB)
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench`: 140 passed and 1 skipped (file symlink), total coverage 96%. The
  same result holds with `CUDA_VISIBLE_DEVICES=-1`, where in-process
  `torch.cuda.is_available()=False`.
- **Test isolation:** a full training run, with the image loader patched, opens no test-split
  image, and the loader factory rejects `"test"`.
- **CPU determinism:** two identical runs give identical per-epoch metrics and identical `best.pt`
  tensors (test).
- **CUDA determinism (demo, one configuration):** two `small_cnn` runs in separate processes were
  bitwise identical (max abs per-epoch difference 0; 26/26 `best.pt` tensors equal).
- **Synthetic shapes demo (pipeline sanity checks on synthetic data, not benchmark results):**
  best-epoch val balanced accuracy was
  - majority baseline 0.3333;
  - small_cnn CPU 1.0000;
  - small_cnn CUDA 1.0000;
  - resnet18 (ImageNet, 2 epochs) 1.0000;
  - efficientnet_b0 (ImageNet, 2 epochs) 0.8519.
- **Pretrained weights:** downloaded into `<data_root>/cache/torch` (re-downloaded, because the
  earlier copies were in `~/.cache`). Recorded sha256 values match the files and the hash prefixes
  in their names.

## Not verified
- The CI workflow has never run. The file-symlink test is skipped on this machine.
- `num_workers > 0` and AMP (no test or demo); GPU determinism for resnet18, efficientnet_b0, or
  any other configuration than the one demo.
- Behaviour after a hard kill (`running` rows would remain until M4's startup recovery exists).
- Python 3.12/3.13; real (non-synthetic) datasets.

## Blockers
None.

## Next task
M3: one-time test-set evaluation of selected checkpoints, plus a metrics module. Metrics include
per-class precision/recall, confusion matrix, calibration (ECE) and confidence intervals. Add
comparison reports that include the majority baseline and flag the input-resolution confound.
