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
| C2 freeze checklist (a-g) | todo | | |
| C3 clean TEST x15, materialize TEST x2, robustness TEST 15x12 | todo | | |
| C5 FINAL_REPORT.md (+JSON) from a committed script | todo | | |
| C6 unlock for the app, Benchmark TEST section, check counts | todo | | |
| GATE C | todo | | |

## Phase D — Screenshots, docs, release

| step | status | commit | notes |
|---|---|---|---|
| D1 screenshots (Playwright, real data root) | todo | | |
| D2 README results summary + screenshots | todo | | |
| D3 definition-of-done audit | todo | | |
| D4 final checks, tag v1.0.0, final packet summary | todo | | |
| GATE D | todo | | |

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

## Next action

Phase C, C2: freeze checklist on the rc1 commit (a: clean tree, HEAD = rc1; b-e:
`run_benchmark.py freeze-check`; f: full suite in both modes on HEAD; g: authorization).
Then C3 per FINAL_ANALYSIS_PLAN.md §2.
