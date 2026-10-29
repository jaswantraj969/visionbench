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
| B1 Docker (files; build/run if available) | in progress | | Docker CLI 29.7.2 installed; daemon not running at start; started the installed Docker Desktop (user-level app, no install/admin): daemon 29.7.2 linux/amd64 up |
| B2 Linux test run + CI workflow | todo | | GitHub Actions cannot run: no remote |
| B3 demo-data, fresh-clone Quick Start, lock files | todo | | |
| B4 documentation | todo | | |
| GATE B + tag v1.0.0-rc1 | todo | | |

## Phase C — One-time TEST evaluation

| step | status | commit | notes |
|---|---|---|---|
| C1 FINAL_ANALYSIS_PLAN.md + hashes (before any test access) | todo | | |
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

## Next action

Phase B: build and run the image (B1); Linux test run in a container (B2); fresh-clone Quick
Start (B3); documentation (B4); Gate B and tag `v1.0.0-rc1`.
