# Review Packet — VB-M1

**Task:** VB-M1: dataset ingestion, validation, statistics, duplicate groups, leakage-aware splits
**Date:** 2026-10-27
**Environment:** Windows 11 Home 10.0.26300, Python 3.11.15, torch 2.14.1+cu130, NumPy 2.4.6,
Pillow 12.3.0 (full details in the VB-M0 section of git history: `git show 4165e71:REVIEW_PACKET.md`)

**Objective:** the dataset layer of the `visionbench` library plus a CLI, covering:
- immutable snapshots with full validation;
- statistics;
- exact and near duplicate groups with a low-information guard;
- reproducible, stratified, duplicate-group-aware split manifests.

It is library and CLI only: no API endpoints, no training, no frontend.

Every output quoted below came from commands executed in this session. The full CLI demo
transcript is reproduced in §6.

---

## 1. Step 0: M0 follow-ups

| Item | Result |
|---|---|
| 0a | `/api/ready` 503 now returns `{"error":{"code":"not_ready","message":"data root is not writable"}}`. The full path is logged server-side with `logger.error`. The test asserts that no `tmp_path`, `blocker`, or escaped path appears in the body, and that the path *does* appear in the captured log. |
| 0b | `requirements/lock-dev-cu130.txt` from `pip freeze --exclude-editable` (43 pins; the `visionbench` line is absent, grep count 0). The header states Python 3.11.15, Windows 11 10.0.26300, and that it is the local CUDA dev environment. Its install recipe is marked as not yet exercised. CI is unchanged (CPU index). |
| 0c | The README says "Tested on Python 3.11 (Windows 11)" and explicitly says 3.12/3.13 are untested. |
| 0d | See the finding below. With the GPU hidden, `torch.cuda.is_available()` is `False` in-process and all tests pass. |

**0d finding (important):** the instructed command `$env:CUDA_VISIBLE_DEVICES=""` does **not** hide the
GPU in Windows PowerShell 5.1. Assigning an empty string *removes* the variable. The first attempt
showed it, via the new pytest summary line:
```
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
23 passed, 1 warning in 4.96s
```
With `$env:CUDA_VISIBLE_DEVICES="-1"` (an invalid device index, which hides all devices):
```
'-1' False 0                      <- python -c: env value, cuda.is_available(), device_count()
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
23 passed, 1 warning in 4.35s
```
`tests/conftest.py` now prints this line at the end of every pytest run. The value is measured
inside the test process, not inferred. The README documents the `-1` form.

## 2. Files created and changed (M0 baseline `4165e71` → HEAD)
```
 .env.example                     |  14 ++
 ARCHITECTURE.md                  |  51 ++++++-
 CHANGELOG.md                     |  29 ++++
 EXPERIMENT_PROTOCOL.md           | 179 ++++++++++++++++------
 IMPLEMENTATION_PLAN.md           |   4 +-
 PROJECT_STATUS.md                |  49 +++++--
 README.md                        |  65 +++++++-
 pyproject.toml                   |   1 +
 requirements/lock-dev-cu130.txt  |  49 +++++++
 src/visionbench/api/main.py      |   8 +-
 src/visionbench/cli.py           | 211 +++++++++++++++++++++++++-
 src/visionbench/core/config.py   |   8 +
 src/visionbench/core/errors.py   |  21 +++
 src/visionbench/core/jsonio.py   |  24 +++
 src/visionbench/core/seeding.py  |  11 ++
 src/visionbench/data/grouping.py | 123 ++++++++++++++++
 src/visionbench/data/hashing.py  |  94 ++++++++++++
 src/visionbench/data/records.py  |  75 ++++++++++
 src/visionbench/data/report.py   |  92 ++++++++++++
 src/visionbench/data/scan.py     | 310 +++++++++++++++++++++++++++++++++++++++
 src/visionbench/data/split.py    | 261 ++++++++++++++++++++++++++++++++
 src/visionbench/data/store.py    | 163 ++++++++++++++++++++
 tests/api/test_health.py         |  15 +-
 tests/conftest.py                |  26 ++++
 tests/data/conftest.py           | 183 +++++++++++++++++++++++
 tests/data/test_cli_dataset.py   | 110 ++++++++++++++
 tests/data/test_grouping.py      |  70 +++++++++
 tests/data/test_hashing.py       | 145 ++++++++++++++++++
 tests/data/test_scan.py          | 157 ++++++++++++++++++++
 tests/data/test_split.py         | 172 ++++++++++++++++++++++
 tests/data/test_store.py         | 142 ++++++++++++++++++
 31 files changed, 2783 insertions(+), 79 deletions(-)
```
Plus this `REVIEW_PACKET.md`, rewritten for M1 in the final commit. API changes are limited to item
0a. No training or frontend code was added. No binaries were committed: every fixture image is
generated in `tmp_path`.

**Dependencies:** none added. `pillow` is now declared explicitly in `pyproject.toml` because
`visionbench.data` imports it directly. It was already installed as a torchvision dependency
(12.3.0, unchanged). dHash is implemented in-house (`data/hashing.py`, about 30 lines), so no
imagehash or SciPy dependency is needed.

## 3. Design summary (details in EXPERIMENT_PROTOCOL.md §1–4, ARCHITECTURE.md "Data layer")
- **Single read per file.** The same bytes are hashed (SHA-256), validated (Pillow `verify()`, then a
  reopen and full `load()`), dHashed, and written to a staging directory. The snapshot therefore
  matches `files.json`. The staging directory is renamed into `datasets/<id>/` only after the
  class requirements pass, so a failed import leaves nothing behind.
- **Pixel limit:** Pillow's `MAX_IMAGE_PIXELS` is set to the configured value inside a context
  manager, and `DecompressionBombWarning` is escalated to an error. Both the warning path (>limit)
  and the error path (>2×limit) are tested. `LOAD_TRUNCATED_IMAGES` is forced off.
- **Links:** `DirEntry.is_symlink()` plus the Windows `FILE_ATTRIBUTE_REPARSE_POINT`, because on
  Python 3.11 `is_symlink()` is False for NTFS junctions. This applies to entries *and* to the
  source directory itself. That second case was a gap found through the coverage review and was
  fixed before the library commit.
- **Split thresholds belong to the split config.** `files.json` stores each file's dHash and gray
  std, so `split` rebuilds the groups with its own thresholds. Those thresholds are recorded in the
  manifest and are part of `split_id`.
- **Exact rationals:** the targets use `Fraction(repr(f))`, so largest-deficit tie-breaking is
  platform-independent.
- **Per-class RNG:** `derived_rng(seed, class)` uses `SeedSequence([seed, sha256(class)[:8]])`,
  never Python's salted `hash()`.
- **Re-running an existing split** recomputes the manifest and compares it byte for byte. It
  reports "already existed (identical)" or refuses on a mismatch, and never overwrites.

## 4. Lint and tests (final run on the committed tree)
```
> ruff check .
All checks passed!
exit=0
> ruff format --check .
44 files already formatted
exit=0
```
`pytest -q --cov=visionbench` (normal):
```
...................................s.................................... [ 75%]
.......................                                                  [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.12 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 8.73 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

Name                                     Stmts   Miss Branch BrPart  Cover
--------------------------------------------------------------------------
src\visionbench\__init__.py                  1      0      0      0   100%
src\visionbench\api\__init__.py              0      0      0      0   100%
src\visionbench\api\main.py                 55      1      6      0    98%
src\visionbench\cli.py                     138      2     22      3    97%
src\visionbench\core\__init__.py             0      0      0      0   100%
src\visionbench\core\config.py              33      0      2      0   100%
src\visionbench\core\env_info.py            11      0      0      0   100%
src\visionbench\core\errors.py               9      0      0      0   100%
src\visionbench\core\jsonio.py              13      0      0      0   100%
src\visionbench\core\seeding.py             22      0      4      1    96%
src\visionbench\data\__init__.py             0      0      0      0   100%
src\visionbench\data\grouping.py            80      1     20      1    98%
src\visionbench\data\hashing.py             52      0     18      0   100%
src\visionbench\data\records.py             51      1      0      0    98%
src\visionbench\data\report.py              29      0      6      0   100%
src\visionbench\data\scan.py               200      8     66      6    95%
src\visionbench\data\split.py              141      4     42      1    97%
src\visionbench\data\store.py               92      2     16      1    97%
src\visionbench\evaluation\__init__.py       0      0      0      0   100%
src\visionbench\explain\__init__.py          0      0      0      0   100%
src\visionbench\jobs\__init__.py             0      0      0      0   100%
src\visionbench\models\__init__.py           0      0      0      0   100%
src\visionbench\robustness\__init__.py       0      0      0      0   100%
src\visionbench\tracking\__init__.py         0      0      0      0   100%
src\visionbench\training\__init__.py         0      0      0      0   100%
--------------------------------------------------------------------------
TOTAL                                      927     19    202     13    97%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
94 passed, 1 skipped, 1 warning in 47.68s
```
`$env:CUDA_VISIBLE_DEVICES="-1"; pytest -q --cov=visionbench` (GPU hidden). The output is identical
apart from these lines (checked with `diff` after removing the PERF, torch and summary lines):
```
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.14 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 9.50 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
94 passed, 1 skipped, 1 warning in 48.52s
```
**Performance timings vary between runs.** An earlier full run on the same code measured 1.33 s
(`bitwise_count`) and 6.64 s (LUT). The asserted bound is < 60 s. The test plants one pair at
distance 3 and asserts that it is found.

**Uncovered lines:** these paths can't be produced on this machine:
- file symlinks (no privilege);
- `DirEntry.is_junction` (Python ≥ 3.12 only);
- names with a backslash or control character (illegal on NTFS);
- non-regular files and read `OSError`s;
- `/api` 422 handler and `deterministic=False` (both from M0);
- the concurrent-import rename race.

### Requirement → test map
| Required assertion | Test(s) |
|---|---|
| every issue category with correct relpaths | `test_scan.py::test_every_issue_category_detected_with_relpaths`, which compares all 11 categories to the expected relpaths |
| symlink / junction never followed | `test_directory_link_reported_not_followed`, `test_linked_source_directory_rejected`, `test_store.py::test_import_never_follows_links` (junction). The file symlink test is **skipped** here. |
| exact/near grouped; uniform not grouped; conflict excluded+reported | `test_grouping.py::test_exact_and_near_duplicates_grouped`, `::test_uniform_images_not_grouped_despite_identical_hash` (asserts Hamming 0 between them), `test_split.py::test_conflict_group_excluded_and_reported`, `::test_conflict_policy_error` |
| zero groups/sha256 across splits | `test_split.py::test_no_leakage_and_groups_kept_together` (recomputed independently in the test, and also checks the manifest's `leakage_check`) |
| same seed byte-identical; different seed different | `::test_same_seed_byte_identical_manifest` (deletes the split dir, then recomputes), `::test_different_seed_different_assignment` |
| ±1 group of target (balanced) | `::test_balanced_counts_within_one_group_of_target` (5 seeds × 3 classes × 3 splits) |
| missing-class error + allow flag | `::test_missing_class_error_and_allow_flag`, `test_cli_dataset.py::test_allow_missing_classes_flag` |
| fingerprint invariant to creation order | `test_store.py::test_fingerprint_invariant_to_creation_order` (files written in forward vs reverse order) |
| import refuses overwrite | `test_store.py::test_import_refuses_overwrite` (bytes unchanged, no staging left behind) |
| CLI exit codes | `test_cli_dataset.py::*`, including `test_process_exit_codes` (real subprocess: 0 / 1 / 2 for argparse usage errors) |
| dHash known inputs | `test_hashing.py` (constant → 0, increasing → all ones, MSB/LSB bit positions, alternating pattern) |
| 20k-hash performance | `test_hashing.py::test_near_duplicate_search_20k_performance[False/True]` |
| verify() misses truncation | `test_scan.py::test_truncated_jpeg_passes_verify_but_fails_load` |

## 5. Bugs found and fixed during this task
1. **`--debug` before a subcommand was ignored.** Parent parsers share `Action` objects, so the
   root's `set_defaults(debug=False)` also reset the subparsers' `SUPPRESS` default, and the
   subparser then overwrote `True`. A test caught it. The fix gives the root parser its own action.
2. **A junction used as the source directory would have been followed** (`Path.is_symlink()` is
   False for junctions on 3.11). The source check now also tests the reparse-point attribute, and
   a test was added.
3. An editing slip briefly put a literal NUL byte into `tests/data/conftest.py`. I detected it,
   replaced it with the `b"\x00"` escape, and verified no NUL bytes remain, before any commit.

## 6. End-to-end CLI demo (generated fixture, scratchpad data root; verbatim)
The fixture is generated by the same builder as the tests (`tests/data/conftest.py::build_messy`). It
contains 3 classes and every case from the task, including a root-level `.DS_Store` and a
`linked_class` NTFS junction → `gamma`. The file symlink could not be created.
```
### 0. generate fixture folder
fixture at <home>\AppData\Local\Temp\claude\...\scratchpad\demo\src; file symlink created: False; dir link: junction

### 1. import
$ visionbench dataset import .../scratchpad/demo/src --name demo --max-image-pixels 10000
imported dataset ds-6d2e5e7dbe20 (demo)
  path: <home>\AppData\Local\Temp\claude\...\scratchpad\demo\var\datasets\ds-6d2e5e7dbe20
  fingerprint: 9942c1dd5d58baf85015ae1df5aaeaf9602f0e01123c42522e7b816c50c2ae13
  classes: 3 | images: 20 | bytes: 134285
  class counts: alpha=8, beta=7, gamma=5 | imbalance ratio: 1.6
  issues: file_outside_class_folder=1, hidden_or_system_file=4, invalid_class_name=1, nested_directory=1, symlink=1, too_large_pixels=2, unreadable_or_corrupt=1, unsupported_extension=2, zero_byte=1
  duplicate groups (>1 member): 3 (6 files) | exact duplicate files: 2 | label-conflict groups: 1
  perceptual check skipped (low information): 2 (std < 3.0 or degenerate dHash; dHash threshold 5)
[exit code: 0]

### 2. import again (must refuse)
$ visionbench dataset import .../scratchpad/demo/src --name demo --max-image-pixels 10000
error: dataset ds-6d2e5e7dbe20 already exists; refusing to overwrite
[exit code: 1]

### 3. list + report
$ visionbench dataset list
ds-6d2e5e7dbe20  demo  classes=3  images=20  created=2026-10-26T19:54:28+00:00
[exit code: 0]

$ visionbench dataset report ds-6d2e5e7dbe20
dataset ds-6d2e5e7dbe20 (demo)
  fingerprint: 9942c1dd5d58baf85015ae1df5aaeaf9602f0e01123c42522e7b816c50c2ae13
  classes: 3 | images: 20 | bytes: 134285
  class counts: alpha=8, beta=7, gamma=5 | imbalance ratio: 1.6
  issues: file_outside_class_folder=1, hidden_or_system_file=4, invalid_class_name=1, nested_directory=1, symlink=1, too_large_pixels=2, unreadable_or_corrupt=1, unsupported_extension=2, zero_byte=1
  duplicate groups (>1 member): 3 (6 files) | exact duplicate files: 2 | label-conflict groups: 1
  perceptual check skipped (low information): 2 (std < 3.0 or degenerate dHash; dHash threshold 5)
  width min/median/max: 48/64.0/64 | height: 48/64.0/64
  modes: {'RGB': 20} | formats: {'JPEG': 1, 'PNG': 19}
[exit code: 0]

### 4. low-information guard evidence (from report.json / files.json)
uniform_red  dhash 0000000000000000 gray_std 0.0 sha256 c0fea7b3397b
uniform_blue dhash 0000000000000000 gray_std 0.0 sha256 dd44fcb116f3
hamming(red, blue) = 0
perceptual_check_skipped_low_information: {"count": 2, "relpaths": ["alpha/uniform_red.png", "beta/uniform_blue.png"]}
multi-member group: g-a6d1bbd613c714fb exact ['alpha', 'beta'] ['alpha/a2.png', 'beta/b_conflict.png'] conflict
multi-member group: g-ac9ff932f983ee6d exact ['alpha'] ['alpha/a0.png', 'alpha/a0_copy.png']
multi-member group: g-e3bbd86f057a2cba near ['alpha'] ['alpha/a1.png', 'alpha/a1_near.jpg']

### 5. split seed 42 (run 1)
$ visionbench dataset split ds-6d2e5e7dbe20 --seed 42
split sp-268fa95d6c45 for dataset ds-6d2e5e7dbe20: new
  manifest: ...\demo\var\datasets\ds-6d2e5e7dbe20\splits\sp-268fa95d6c45\manifest.json
  manifest sha256: 97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77
  seed=42 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    alpha: 5/1/1/7
    beta: 4/1/1/6
    gamma: 3/1/1/5
  totals: train=12, val=3, test=3
  excluded (label conflicts): 2 files in 1 groups
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 16 groups, 17 sha256 values)
[exit code: 0]

97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77 *.../scratchpad/demo/manifest_run1.json
split_info.json:
{
  "created_at": "2026-10-26T19:54:39+00:00",
  "dataset_id": "ds-6d2e5e7dbe20",
  "manifest_sha256": "97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77",
  "numpy_version": "2.4.6",
  "python_version": "3.11.15",
  "split_id": "sp-268fa95d6c45"
}

### 6. delete the split directory and recompute seed 42 from scratch (run 2)
$ visionbench dataset split ds-6d2e5e7dbe20 --seed 42
split sp-268fa95d6c45 for dataset ds-6d2e5e7dbe20: new
  manifest sha256: 97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77
  (counts, totals, exclusions and leakage check identical to run 1)
[exit code: 0]

97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77 *.../scratchpad/demo/manifest_run1.json
97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77 *.../scratchpad/demo/manifest_run2.json
cmp: manifests are byte-identical

### 7. seed 42 again with the split present (verifies, does not rewrite)
$ visionbench dataset split ds-6d2e5e7dbe20 --seed 42
split sp-268fa95d6c45 for dataset ds-6d2e5e7dbe20: already existed (recomputed manifest identical)
  manifest sha256: 97fb340c71daedd24587ddc95f02a0ec928c92a40bd583d3f7b6f9d9a95dbe77
[exit code: 0]

### 8. split seed 43
$ visionbench dataset split ds-6d2e5e7dbe20 --seed 43
split sp-8fda1b0e11fc for dataset ds-6d2e5e7dbe20: new
  manifest sha256: 9b7a7b0d4698ededcdf1fef386985effe0d6af7cb6e7fe8da3588b2789eec816
  counts (train/val/test/total):
    alpha: 5/1/1/7
    beta: 4/1/1/6
    gamma: 3/1/1/5
  totals: train=12, val=3, test=3
  excluded (label conflicts): 2 files in 1 groups
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 16 groups, 17 sha256 values)
[exit code: 0]

seed 42 vs 43: 7 of 18 entries assigned differently
seed 43 leakage_check: {"groups_checked": 16, "groups_in_multiple_splits": 0, "sha256_checked": 17, "sha256_in_multiple_splits": 0}

### 9. conflict policy 'error' and bad fractions (must fail)
$ visionbench dataset split ds-6d2e5e7dbe20 --seed 42 --conflict-policy error
error: 1 label-conflict group(s) (policy 'error'): g-a6d1bbd613c714fb labels=['alpha', 'beta']
[exit code: 1]

$ visionbench dataset split ds-6d2e5e7dbe20 --seed 42 --fractions 0.8 0.1 0.2
error: fractions must sum to 1, got [0.8, 0.1, 0.2]
[exit code: 1]
```
Abbreviations: the long scratchpad path is shortened to `...`. In runs 2 and 7, the repeated
counts and leakage lines are summarized because they were identical to run 1, and run 8's
`manifest:`/`seed=` lines are omitted. The unabridged transcript is in the session scratchpad
(`demo_output.txt`).

"16 groups, 17 sha256": the 18 included files contain one exact-duplicate pair (`a0`/`a0_copy`),
which gives 17 distinct hashes. The near pair `a1`/`a1_near` has two hashes but one group, so there
are 16 groups.

## 7. git log --oneline
```
e049491 M1 docs: protocol, architecture, README, status, changelog
34930b6 M1: `visionbench dataset import|report|split|list` CLI
57d7b8f M1: dataset library - validation, dHash duplicate groups, snapshots, splits
d3d52b5 M0 follow-ups: generic ready error, dev lock file, CUDA state in test log
4165e71 M0: foundation baseline
```
One more commit adds this REVIEW_PACKET.md. Everything is local only: no remote, nothing pushed.

## 8. Known limitations
- **dHash scope:** catches resizes and re-encodes. It misses crops, rotations, flips, heavy edits
  and semantic near-duplicates. The defaults (threshold 5, low-info std 3.0) were tested only on
  synthetic images and are not calibrated on real photos.
- **Low-information guard gap:** smooth low-texture images can produce *sparse, non-zero* dHashes
  that are close to each other. These pass the std/degenerate guard and could still form false
  groups. See open question 1.
- **NumPy streams:** split reproducibility depends on the NumPy Generator stream, which NumPy does
  not guarantee across versions. The version is recorded in `split_info.json` but not enforced.
- **Thread safety:** the scan mutates Pillow module globals, so it is not thread-safe. This is fine
  for the planned single-worker runner.
- **Memory:** each file is read fully into memory. There is no per-file byte limit yet; uploads in
  M4 will need one.
- **Immutability is by convention:** snapshots are never overwritten, but files are not read-only,
  and there is no command yet that re-hashes `raw/` against `files.json`.
- **Untested paths:** the file-symlink test is skipped on this machine, and the paths listed in §4
  could not be exercised. CI (Linux) would exercise the symlink test, but CI has not run yet.
- **Line endings:** the global git setting `core.autocrlf=true` converts tracked text files to CRLF
  on checkout. This does not affect generated manifests, which are written as bytes, but the repo
  has no `.gitattributes`.
- **Starlette warning:** the httpx deprecation warning remains (M0).

## 9. Open questions for the supervisor
1. Should the low-information guard also use a "dHash signal strength" measure? For example: mean
   absolute horizontal gradient of the 9×8 thumbnail below a threshold, or a popcount close to 0 or
   64. This would catch smooth sky and sea gradients that the std test misses. It would add one
   config value.
2. Should the dHash and low-info thresholds be calibrated on a real dataset before M8? If so, which
   one?
3. Should NumPy be pinned (or the version checked at split time) to guarantee manifest
   reproducibility across environments?
4. Add a `dataset verify <id>` command that re-hashes `raw/`, and/or make snapshot files read-only?
5. Add a `.gitattributes` (`* text=auto eol=lf`) for consistent line endings?
6. Enable Windows Developer Mode on this machine so the file-symlink test runs locally, or rely on
   Linux CI for it?

## 10. Recommended next task
**M2:** model registry (majority baseline, SmallCNN, ResNet18, EfficientNet-B0) and CLI training
that consumes a split manifest by `split_id`. It should cover seeds, checkpoints under `artifacts/`,
validation-only early stopping, run records (config, `set_seed` report, `env_info`, dataset
fingerprint, manifest sha256), and train-split-only normalization statistics for scratch models.
Training should run end-to-end on CPU on a tiny synthetic dataset.
