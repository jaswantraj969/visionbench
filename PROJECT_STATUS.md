# Project Status

_Last updated: 2026-10-27, end of M1 (pending supervisor review)._

## Current state
- **M0 foundation:** package, config, seeding, env info, health/ready API, `visionbench env`.
- **M1 dataset layer** (library + CLI only):
  - import into immutable snapshots, with validation and every issue reported;
  - statistics;
  - exact (SHA-256) and near (dHash) duplicate groups, with a low-information guard and
    label-conflict handling;
  - group-aware, class-stratified, byte-reproducible split manifests with a computed leakage check.
- There are no API dataset endpoints and no training or frontend code.

## Verified in session (2026-10-27, Windows 11, Python 3.11.15, torch 2.14.1+cu130)
- `ruff check .` and `ruff format --check .` pass.
- `pytest -q --cov=visionbench`: 94 passed, 1 skipped (file symlink; no symlink privilege), 97% total
  coverage. The same result holds with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`; in-process
  `torch.cuda.is_available()=False`).
- Near-duplicate search over 20,000 random hashes: 2.12–2.14 s with `np.bitwise_count` and
  8.7–9.5 s with the byte-LUT fallback in the final runs. An earlier run measured 1.33 s / 6.64 s.
- CLI demo on a generated fixture:
  - import, then report, then split with seed 42 twice. The second run started from a deleted split
    directory and produced a byte-identical `manifest.json` (sha256 `97fb340c…`).
  - Seed 43 gives a different assignment (7 of 18 entries differ).
  - The leakage check was 0/0 for every split.
  - The two uniform images have Hamming distance 0, were not grouped, and were reported as
    low-information.
- M0 follow-ups:
  - the 503 response from `/api/ready` contains no paths;
  - `requirements/lock-dev-cu130.txt` was added;
  - the README claims only that Python 3.11 was tested.

## Not verified
- CI workflow (never run; no remote).
- File-symlink handling: the test is skipped on this machine. Directory links are tested via an
  NTFS junction.
- File names with backslashes or control characters, non-regular files, and read errors. These
  code paths can't be created on Windows here and are untested.
- Python 3.12 and 3.13.
- dHash and low-information thresholds on real photographs (tested on synthetic images only).

## Blockers
None.

## Git
Local repository only, no remote. See `git log --oneline` in REVIEW_PACKET.md.

## Next task
M2: model registry (majority baseline, SmallCNN, ResNet18, EfficientNet-B0), plus CLI training that
consumes a split manifest, with seeds, checkpoints, and validation-only early stopping.
