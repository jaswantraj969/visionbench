# Definition of done: audit (v1.0.0)

Each item has the evidence run in the development sessions (details in REVIEW_PACKET.md) or is
marked **not met**.

| # | item | status | evidence |
|---|---|---|---|
| 1 | Documented setup works for a new developer | met | README Quick Start B was run from a fresh clone on Windows (Python 3.11, CPU torch); every command and its time is recorded (REVIEW_PACKET "Phase B"). Quick Start A (Docker) was run too. Not run on macOS or on a Linux host. |
| 2 | App starts locally via the documented workflow | met | `visionbench serve` from the fresh clone served the API and the built frontend on 127.0.0.1:8000 (health 200, `/` 200); the same in the container via `docker compose up`. |
| 3 | User can inspect a real dataset | met | EuroSAT RGB (27,000 images) imported, verified (`verify --full`) and audited; the Dataset page shows the class chart, issues, seeded sample tiles with attribution, splits and the leakage audit (`docs/screenshots/02-dataset.png`). |
| 4 | Baseline + deep models train/evaluate through the supported workflow | met | 15 benchmark runs (majority baseline, SmallCNN, ResNet18 scratch/ImageNet, EfficientNet-B0) trained and evaluated with `scripts/run_benchmark.py` / the CLI; training and evaluation through the API and UI in the e2e test and the Quick Starts. |
| 5 | Experiments preserve config and results | met | Each run stores config, env, history, checkpoints (sha256 in artifact rows), evaluations and predictions; the freeze check confirmed 15/15 checkpoints against their artifact rows before the TEST evaluation. |
| 6 | Comparisons use consistent leakage-aware splits | met | One frozen group-aware split (`sp-55320e11726c`) with a computed leakage check and an independent pHash/dHash audit; `compare` refuses experiments on different datasets or splits; the final report pairs runs on the same TEST images. Limitation: near-uniform tiles cannot be checked by hashing (DATASETS.md). |
| 7 | Predictions and Grad-CAM work end to end | met | API + Explain page on the real models (`docs/screenshots/07-explain-gradcam.png`), sanity checks in the test suite, the e2e test; TEST images of the finalised split only. |
| 8 | Robustness tests are reproducible | met | Every corrupted condition is materialized once with a hashed manifest and verified byte for byte on rerun (VAL 12/12 in M6, again after the cache move; TEST 12/12 on the second pass); 180 VAL + 180 TEST robustness evaluations, each paired with its clean evaluation. |
| 9 | Frontend connected to working endpoints | met | 9 pages over the real API, generated API types with a stale check, Playwright e2e against a real backend, a page-by-page walk and screenshots on the real data root. |
| 10 | Unit/integration/frontend tests pass; limitations reported | met | Backend suite (Windows normal and GPU-hidden; Linux container), 43 frontend tests, e2e; limitations in README, EXPERIMENT_PROTOCOL.md, DATASETS.md and FINAL_REPORT.md. |
| 11 | Docker and CI work | met | Docker: image built, ran healthy, Quick Start A verified (Phase B). CI: GitHub Actions run [40917916209](https://github.com/jaswantraj969/visionbench/actions/runs/40917916209) green on all four jobs (backend, frontend, e2e, docker). |
| 12 | Docs cover install, architecture, usage, reproducibility, limitations, troubleshooting | met | README (Quick Start, architecture diagram, app, API, Docker, reproduction, troubleshooting, limitations, license), ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md, DATASETS.md, the frozen protocols and FINAL_ANALYSIS_PLAN.md. |
| 13 | No known defect concealed | met | Known issues are listed below and in REVIEW_PACKET.md, including the cache-path defect found during the TEST run and its amendment. |
| 14 | Presentable repo with genuine screenshots and results | met | `docs/screenshots/` (taken from the real app on the real data), README results summary linked to FINAL_REPORT.md. |

## Known issues and limitations (not hidden)

- **Cache-path defect (fixed, amendment 1):** the corrupted-image cache had no val/test path
  component, so the first TEST materialization pass collided with the VAL cache. Fixed before any
  robustness TEST result existed; recorded in FINAL_ANALYSIS_PLAN.md amendment 1.
- **Pretraining and input resolution are confounded** in the benchmark design; one split, n = 3
  seeds, an epoch cap and untuned learning rates (FINAL_REPORT.md constraints).
- **No authentication:** local, single-user use only (loopback bind, Host/header/Origin guards).
- **Test tooling warning:** every backend test run shows one `StarletteDeprecationWarning` from
  FastAPI's TestClient (httpx); not acted on.
- **Not exercised:** macOS, Python 3.12/3.13, a Linux host with a GPU.
- **Demo dataset:** pixels are deterministic, PNG bytes (and so its id) differ by zlib build.
- **Screenshots run:** the first run logged one browser "404" console message that was not
  identified; a second run of every page logged none.

## CI

GitHub Actions passed on `main`: https://github.com/jaswantraj969/visionbench/actions/runs/40917916209 (backend, frontend, e2e, docker all green).
The first run on publishing failed in one backend test
(`test_cancel_timeout_terminates_training_child`): on the faster runner the tiny training run
early-stopped before the cancel request. Early stopping was disabled in that test; its intent is
unchanged.
