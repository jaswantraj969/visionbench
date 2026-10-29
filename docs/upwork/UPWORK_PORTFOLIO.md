# Upwork portfolio entry: VisionBench

Profile: **Jaswant R.** (AI & Machine Learning Developer | Python, Deep Learning & Research).
All numbers come from `benchmarks/eurosat-v1/results/FINAL_REPORT.md` (TEST, mean over 3 seeds).

## Project title (68 characters; limit 70, as given; Upwork's current limit not verified)

VisionBench: Reproducible CV Benchmarking with Grad-CAM & Robustness

## Your role

Solo developer

## Project description (544 characters; limit 600, as given; not verified against Upwork)

Built a local platform for benchmarking image classifiers without test leakage: dataset audits, leakage-aware splits, PyTorch training, evaluation with bootstrap CIs, Grad-CAM and corruption-robustness tests, behind a FastAPI API and a React/TypeScript UI. On EuroSAT (5 models x 3 seeds, one pre-registered TEST evaluation), EfficientNet-B0 reached 0.984 balanced accuracy (ResNet18-ImageNet 0.984, majority baseline 0.100); under 12 synthetic corruptions mean accuracy fell to 0.44-0.59. Tested with pytest, Vitest and Playwright; Dockerized.

## Skills (up to 5)

1. PyTorch
2. Computer Vision
3. Deep Learning
4. FastAPI
5. React

(Exact skill names must be picked from Upwork's skill list; not verified.)

## Project URL

https://github.com/jaswantraj969/visionbench
(**Not live yet:** the repository has not been published; see the release notes.)

## Images (1600 x 1200, from `docs/screenshots/`)

1. `09-benchmark-test.png`: TEST results, rank groups and the pre-registered comparisons
2. `07-explain-gradcam.png`: prediction and Grad-CAM with stated limitations
3. `08-robustness.png`: corruption suite and accuracy-vs-severity charts (VALIDATION)
4. `02-dataset.png`: dataset sample tiles with EuroSAT attribution

The images contain EuroSAT tiles (MIT license, modified Copernicus Sentinel data); keep the
attribution line visible. Upwork's recommended image size was not verified.

## Profile overview (2 sentences)

I build reproducible ML systems end to end: from leakage-aware data splits and PyTorch training to evaluation with confidence intervals, explainability and robustness testing. My VisionBench project benchmarks image classifiers on EuroSAT with a pre-registered test protocol, a FastAPI backend and a React dashboard.

## Claims checklist (all verified in the repository)

- 0.984 balanced accuracy: EfficientNet-B0 0.9844 ± 0.0014, ResNet18-ImageNet 0.9836 ± 0.0033
  (TEST); the two are **not distinguishable** under the pre-registered rule, so do not claim
  EfficientNet "beat" ResNet18.
- Majority baseline 0.1000.
- Corrupted mean balanced accuracy 0.4372-0.5869 across the four trained models (TEST).
- No "state-of-the-art" or "robust" claims; one dataset, one split, 3 seeds.
