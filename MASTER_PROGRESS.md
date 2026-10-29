# VB-FINAL master progress

Owner authorization line in the task: **"proceed automatically"** (Phase C may run only if every
freeze-checklist item passes).

Resume rule: re-read this file, the latest REVIEW_PACKET.md section and `git log`; verify the
last step; continue. **Never redo the TEST evaluation** — check the DB first
(`python scripts/run_benchmark.py check benchmarks/eurosat-v1`).

Status values: todo / in progress / done / blocked.

## Phase 0 — M6 follow-ups

| step | status | commit | notes |
|---|---|---|---|
| 0a robustness protocol amendment (text only) + hashes | done | b5ee799 | old 3d13075f… new 0b738cfa… |
| 0b pHash dark-tile limitation; strong noise severities | done | 07b2000 | pair a-001 Forest_1552 (val) / River_1476 (train), pHash 0, dHash 26: checked visually, different images |
| 0c README robustness API commands verified with curl | done | 6cf7de2 | PowerShell 5.1, live serve |
| 0d baseline: no irrelevant training defaults in summaries | done | 9a088bf | |
| GATE 0 | done | 6cf7de2 | hashes OK; ruff OK; 340 passed / 1 skipped both modes on a clean export |

## Phase A — M7 frontend

| step | status | commit | notes |
|---|---|---|---|
| A1 scaffold, API types, client, proxy, static serving, read-only endpoints | done | 7b53c81, 99da9d4 | |
| A2 pages (Overview, Datasets, Training, Jobs, Experiments, Evaluate & Compare, Explain, Robustness, Benchmark) | done | 99da9d4 | |
| A3 component tests, Playwright e2e, coverage | done | 9d1678a, 99da9d4 | 42 unit tests, 87.67 % lines; e2e 1 passed; e2e found a real bug (same-origin POST refused), fixed in 9d1678a |
| GATE A | done | 99da9d4 | lint/typecheck/tests/build OK; backend 345 passed / 1 skipped both modes (clean export); walk of all pages on the real data root, 0 errors; check: 0 test evaluations |

## Phase B — Release engineering

| step | status | commit | notes |
|---|---|---|---|
| B1 Docker (files; build/run if available) | done | d411106 | Docker Desktop started (user-level app); image built (1.87 GB), compose up healthy, uid 10001, 127.0.0.1-only publish, guards hold, import/split/train/eval jobs in the container, volume survives recreate |
| B2 Linux test run + CI workflow | done | d411106, 3677583 | Linux container (python 3.11.18, CPU torch, non-root): ruff OK, 360 passed at 3677583; ci.yml written, NOT run on GitHub Actions (no remote) |
| B3 demo-data, fresh-clone Quick Start, lock files | done | d411106, 4c9f83f | native Quick Start from a fresh clone of 4c9f83f (Windows, CPU torch) and Docker Quick Start at 3677583 (PowerShell) both worked; Linux CPU locks generated in the pinned image |
| B4 documentation | done | 4c9f83f, b1a6ef3, Gate B commit | README rewrite, ARCHITECTURE, DATASETS, CHANGELOG, PROJECT_STATUS, IMPLEMENTATION_PLAN |
| GATE B + tag v1.0.0-rc1 | done | Gate B commit (tag `v1.0.0-rc1`) | suite at 3677583: Windows 359 passed / 1 skipped both modes, Linux 360 passed; frontend lint/typecheck/42 tests/build; e2e 1 passed; check 0 TEST evaluations; hashes OK |

## Phase C — One-time TEST evaluation

| step | status | commit | notes |
|---|---|---|---|
| C1 FINAL_ANALYSIS_PLAN.md + hashes (before any test access) | done | 3677583 | plan sha256 11de6e1c…; committed before rc1 (earlier than the task order); 0 TEST evaluations at that commit |
| C2 freeze checklist (a-g) | done | 288dbe1 | a: clean tree, HEAD = docs-only descendant of v1.0.0-rc1; b-e: `freeze-check` exit 0 (hashes OK, plan 11de6e1c…, 0 TEST evaluations, verify --full exit 0 + fingerprint match, 15/15 checkpoints); f: suite on v1.0.0-rc1 359 passed / 1 skipped both modes; g: "proceed automatically" |
| C3 clean TEST x15, materialize TEST x2, robustness TEST 15x12 | **STOPPED** | (records: this commit) | clean TEST x15: done, all first attempts (`results/test_runs.json`). Materialize TEST pass 1: **non-infrastructure failure** `gaussian_noise/0.04: stored condition covers different images` (the corruption cache path has no val/test component, so TEST collides with the VAL cache); nothing written. Robustness TEST: not run (0 rows). STOP condition 2 / plan §3.3: owner decision needed |
| C5 FINAL_REPORT.md (+JSON) from a committed script | blocked (C3 stopped) | | the plan runs the report after the robustness TEST evaluations; not run |
| C6 unlock for the app, Benchmark TEST section, check counts | blocked (C3 stopped) | | the app stays locked; `check` now fails by design (it asserts 0 TEST evaluations; 15 exist) |
| GATE C | blocked | | |

## Phase D — Screenshots, docs, release

| step | status | commit | notes |
|---|---|---|---|
| D1 screenshots (Playwright, real data root) | blocked (needs C) | | |
| D2 README results summary + screenshots | blocked (needs C) | | |
| D3 definition-of-done audit | done (as of the stop) | this commit | REVIEW_PACKET top summary |
| D4 final checks, tag v1.0.0, final packet summary | partial | this commit | packet top summary written; **no `v1.0.0` tag** (the release is not complete) |
| GATE D | blocked | | |

## Readings and decisions recorded (task says: take the most reasonable reading and record it)

- **Definition of done (14 items):** the list is not in the repository, nor in this session's
  messages. D3 uses a reconstruction from the task GOAL and IMPLEMENTATION_PLAN.md (M0-M8),
  listed in REVIEW_PACKET.md. **Owner to confirm or supply the original list.**
- **LICENSE:** the owner's choice is not known; no LICENSE file is added. **Owner TODO.**
- **Responsive layout:** at 768 px wide tables (e.g. Benchmark runs) scroll horizontally inside
  their own box (`.table-scroll`); the page itself has no horizontal overflow (measured 0 px at
  1024 and 768 px).
- **Docker Desktop:** it was installed but not running. Starting the installed per-user app is
  not a system-software install or admin change, so it was started (Phase B).
- **Phase C code before rc1:** freeze item (a) requires HEAD to be rc1 or a docs-only
  descendant, and C4 forbids code changes between evaluations. Everything the TEST run needs
  was therefore written, tested and committed before rc1 (b1a6ef3): `freeze-check`,
  `evaluate-test`, `run_robustness.py --split test`, the TEST robustness report and
  `final_report.py`. The whole pipeline was rehearsed on synthetic data in a throwaway data
  root. Only C6 (unlocking the app, the Benchmark TEST section, the extended check) changes
  code after the TEST evaluation.
- **C1 order:** the plan was committed (3677583) before Gate B rather than after it. It is
  still before any TEST access, which is what the rule protects. Its hash file is
  `FINAL_ANALYSIS_PLAN_HASHES.txt` (the name the API view already expected).
- **Freeze item (f):** the full suite is run again in both modes on the exact HEAD at C2,
  even though HEAD is a docs-only descendant of the tested 3677583.

## VB-RELEASE (owner decisions: fix option (a), MIT license by Jaswant Raj, publish to GitHub)

| step | status | commit | notes |
|---|---|---|---|
| S1 split-aware cache path + plan Amendment 1 + tag v1.0.0-rc2 | done | 28f97a7 | VAL cache moved to .../val/, 12/12 conditions = M6 hashes (48,600 files verified); plan sha256 ff41ad85…; Windows suite 360 passed / 1 skipped; Linux container skipped (Docker daemon not running); DB: 15 clean TEST (ids = test_runs.json), 0 robustness TEST |
| S2 robustness TEST + check update | done | (this commit) | materialize TEST pass 1 written 12/12 (199.6 s), pass 2 verified 12/12 with identical condition hashes (750.7 s); robustness TEST 15 runs x 12 = 180, all first attempts (2567.6 s, CUDA, keep-awake; an owner's game was also using the GPU); check: exactly 1 clean + 12 robustness TEST per run, 0 elsewhere |
| S3 final report + app unlock + Benchmark TEST section | done | (this commit) | FINAL_REPORT.md/.json and robustness_test.md/.json from scripts/final_report.py (rank groups {efficientnet, resnet18_imagenet} > {small_cnn, resnet18_scratch} > {baseline}; comparisons 1 and 2 not distinguishable, 3 distinguishable: resnet18_imagenet > resnet18_scratch); amendment 1 added to the constraints; TEST images unlocked per finalised split only; Benchmark TEST section |
| S4 screenshots + README + DoD + LICENSE | done | (this commit) | 10 screenshots 1600x1200 from the real app on the real data root (docs/screenshots); README results/screenshots/license; DEFINITION_OF_DONE.md (14 items); LICENSE MIT (Jaswant Raj); image captions no longer overlap |
| S5 pre-publish audit | done | (this commit; hashes per COMMIT_MAP.md) | author+committer of all commits and taggers -> Jaswant Raj <214715227+jaswantraj969@users.noreply.github.com> (git filter-repo, dates kept); local paths with the Windows user name replaced by <repo>/<home> in every commit; scans: no secrets, no .env, no var/ content, no weights, images only in docs/screenshots, no personal emails; backend 362 passed / 1 skipped both modes, frontend lint/typecheck/43 tests/build, e2e 1 passed; tag v1.0.0 |
| S6 publish | done | | public repo https://github.com/jaswantraj969/visionbench, tags pushed, topics set; first CI run failed (flaky cancel test, fixed); CI green: https://github.com/jaswantraj969/visionbench/actions/runs/40917916209 |
| S7 Upwork package | done | (this commit) | docs/upwork/UPWORK_PORTFOLIO.md |

## Next action

None: released.
