# Review Packet — VB-M4.1

**Task:** VB-M4.1: process-safety and local-API security hardening (fixes from the M4 review;
no new features)
**Date:** 2026-10-27
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; torch 2.14.1+cu130; FastAPI
0.162.0; Starlette 1.10.0; psutil 7.2.2 (new); coverage 7.17.0; pytest-cov 7.1.0; SQLite
3.53.1; NVIDIA GeForce RTX 3050 6GB Laptop GPU.

Every output below came from commands executed in this session. Scratchpad paths are shortened
to `<scratchpad>`. The demos use synthetic data only. Nothing here is a benchmark result.

---

## 1. Commits (local only; no remote, nothing pushed)

```
<this packet>  M4.1 review packet
fd78b72 M4.1 docs: process identity, OS lock, recovery rule, local-API threat model
d23e318 M4.1: protect the local API against cross-site requests and DNS rebinding
c7f620a M4.1: process identity, OS worker lock, recovery under the lock
bcd6674 M4 review packet
```

`c7f620a` contains H1 + M1 with their tests, the training cancel-timeout test and the
subprocess-coverage setup. `d23e318` contains H2 with its tests. The full suite was run on the
final tree (`fd78b72`, §5). It was not re-run on the intermediate commit `c7f620a`.

## 2. Defects and fixes

| Defect | Fix | Where |
|---|---|---|
| **H1** `recover()` checked `child_pid` with `pid_alive()` and then terminated it, so a reused pid could kill any user process | A process is (pid, creation time). `terminate_process(pid, create_time)` kills only if `process_is()` is true, through one psutil handle that re-checks the creation time. Creation times are recorded next to every stored pid (schema v5). The child watchdog also compares the worker's identity, not its pid. | `core/process.py`, `jobs/worker.py::recover`, `_watch_parent`, `run_job` |
| **H2** a web page could POST to the localhost API (multipart upload, body-less cancel); DNS rebinding could also read responses | `api/security.py::LocalRequestGuard` (outermost ASGI middleware): Host allow-list → 400 `invalid_host`; POST/PUT/PATCH/DELETE need `X-VisionBench-Request: 1` → 403 `missing_client_header`; `Origin` present but not configured → 403 `origin_not_allowed`; `nosniff` on every response. The header is in CORS `allow_headers`. | `api/security.py`, `api/main.py`, `core/config.py` (`extra_trusted_hosts`) |
| **M1** pid-file lock + pid-liveness recovery: a reused pid could block worker start forever or leave a job `running` forever | Exclusive OS lock on `db/worker.lock` for the worker's lifetime (`msvcrt.locking`, one byte at offset 2^30 so the content stays readable / `fcntl.flock`); content informational. `recover(settings, lock)` refuses to run without the held lock and fails **every** `running` job. Experiments are kept only if `process_is(runner_pid, runner_create_time)`. | `jobs/worker.py::worker_lock`, `recover` |

### Decisions

1. **psutil instead of ctypes `GetProcessTimes`.** I chose psutil for these reasons:
   - It gives a creation time on Windows *and* POSIX through one API. The ctypes route would
     have needed `GetProcessTimes` plus separate `/proc/<pid>/stat` (Linux) and `sysctl`
     (macOS) code, none of which can be run here.
   - It replaces both hand-written Win32 helpers (`pid_alive`, `terminate_pid`), which are
     deleted.
   - `Process.kill()` re-checks the creation time it captured, so the identity check and the
     kill cannot be split by a pid reuse.

   Costs: one compiled dependency (binary wheels for all supported platforms), pinned
   `psutil>=5.9,<8`, and `psutil==7.2.2` added to the lock file.
2. **Tolerance** is 0.05 s. Both values come from the same OS clock, so the tolerance only
   absorbs float rounding. A missing creation time (a pre-v5 row) never matches, so a legacy
   row is treated as "not running": it is recovered, and nothing is killed.
3. **Own guard instead of Starlette's `TrustedHostMiddleware`.** The guard does the same Host
   check, but its 400 uses the project's JSON error schema (`invalid_host`); Starlette's is a
   plain-text body. It is pure ASGI, so uploads still stream through the byte limit.
4. **The runner identity is written in the same transaction that creates the experiment**
   (`ExperimentStore.create(..., runner=...)`). Before, `create` and `update(runner_pid)` were
   separate, so a recovery between them would have seen an owner-less pending experiment.
5. **`VISIONBENCH_HEARTBEAT_STALE_SECONDS` was removed.** Recovery no longer reads heartbeat
   age, and keeping the setting would have left a knob that does nothing. The heartbeat is
   still written and returned by the API.
6. **Test-only hook** `VISIONBENCH_TEST_IGNORE_COOPERATIVE_CANCEL=1` (`jobs/handlers.py`)
   makes `check_cancel` a no-op, so a training child ignores cancel and the worker's
   terminate-after-timeout path runs. It is an environment variable because the job runs in a
   spawned child, where a monkeypatch would not reach. It is not documented as a setting.

## 3. Every terminate/kill call site

```
$ git grep -n -E "\.terminate\(|\.kill\(|os\.kill|taskkill|TerminateProcess|SIGKILL|SIGTERM|terminate_process\(|_exit\(" -- src tests
src/visionbench/cli.py:427:            worker.terminate()
src/visionbench/cli.py:431:                worker.kill()
src/visionbench/core/process.py:47:def terminate_process(pid: int | None, create_time: float | None) -> bool:
src/visionbench/core/process.py:60:        proc.kill()
src/visionbench/jobs/worker.py:185:            elif terminate_process(child, child_time):  # normally it already exited by itself
src/visionbench/jobs/worker.py:249:            os._exit(3)  # worker is gone: stop now, leave the job for recovery
src/visionbench/jobs/worker.py:288:                proc.terminate()
src/visionbench/jobs/worker.py:300:        proc.terminate()
tests/jobs/test_recovery.py:45:            proc.kill()
tests/jobs/test_recovery.py:91:    assert terminate_process(proc.pid, real - 5.0) is False  # pid matches, identity does not
tests/jobs/test_recovery.py:93:    assert terminate_process(proc.pid, real) is True  # the recorded process itself
tests/jobs/test_worker_process.py:158:    w.kill()  # hard kill: no cleanup in the worker
tests/jobs/test_worker_process.py:201:    first.kill()  # hard kill: the OS releases the lock; the file still names first_pid
tests/vb_worker.py:57:        """Hard kill (TerminateProcess on Windows): no cleanup code runs in the worker."""
tests/vb_worker.py:58:        self.proc.kill()
tests/vb_worker.py:66:            self.proc.kill()
```

| Site | What is killed | Why it cannot hit an unrelated process |
|---|---|---|
| `core/process.py:60` `proc.kill()` | the recorded process | only reached after `process_is()`, then the same `psutil.Process` re-checks the creation time before signalling |
| `jobs/worker.py:185` `terminate_process(child, child_time)` | an orphaned job's child (recovery) | identity check above; mismatch → "not running (pid gone or reused)" |
| `jobs/worker.py:288`, `:300` `proc.terminate()` | the job child the worker itself spawned (cancel timeout, Ctrl-C) | `multiprocessing.Process` holds the child's process **handle** on Windows, and on POSIX the child is unreaped, so its pid cannot be reused yet |
| `jobs/worker.py:249` `os._exit(3)` | the child exits itself when the worker's identity is gone | not a kill of another process |
| `cli.py:427`, `:431` `worker.terminate()/kill()` | the `serve` command's own worker `Popen` | `Popen` handle (Windows) / unreaped child (POSIX), same argument |
| tests: `test_recovery.py:45`, `vb_worker.py:58,66`, `test_worker_process.py:158,201` (via `WorkerProcess.kill`) | `Popen` objects the test started (sleepers, worker processes) | started by the test itself, as required |
| tests: `test_recovery.py:91,93` | `terminate_process` on a sleeper the test started: wrong identity → `False`, process alive; right identity → killed | positive and negative control of the identity check |

There is no `os.kill`, `taskkill` or `TerminateProcess` in `src/` any more. The demos use
`taskkill` only on processes they started (demo a checks `process_is` and ancestry first).

## 4. Files changed (`git diff --stat bcd6674 fd78b72`)

```
 ARCHITECTURE.md                    |  84 +++++++++---
 CHANGELOG.md                       |  47 +++++++
 PROJECT_STATUS.md                  |  37 ++++--
 README.md                          |  32 ++++-
 pyproject.toml                     |   9 ++
 requirements/lock-dev-cu130.txt    |   3 +-
 src/visionbench/api/main.py        |  23 +++-
 src/visionbench/api/security.py    |  92 ++++++++++++++
 src/visionbench/cli.py             |   7 +-
 src/visionbench/core/config.py     |   3 +-
 src/visionbench/core/process.py    |  76 +++++++++++
 src/visionbench/jobs/handlers.py   |   6 +
 src/visionbench/jobs/store.py      |  22 +++-
 src/visionbench/jobs/worker.py     | 237 +++++++++++++++++-----------------
 src/visionbench/tracking/store.py  |  20 ++-
 src/visionbench/training/runner.py |   3 +-
 tests/api/test_api_security.py     | 128 +++++++++++++++++++
 tests/api/test_api_worker.py       |   7 +-
 tests/api/test_health.py           |   8 +-
 tests/evaluation/test_evaluate.py  |   4 +-
 tests/jobs/test_job_store.py       |  92 +-------------
 tests/jobs/test_recovery.py        | 254 +++++++++++++++++++++++++++++++++++++
 tests/jobs/test_worker_process.py  | 102 +++++++++++++--
 tests/vb_api.py                    |  24 +++-
 tests/vb_worker.py                 |  21 ++-
 25 files changed, 1041 insertions(+), 300 deletions(-)
```

## 5. Lint and tests at `fd78b72` (verbatim)

```
> .\.venv\Scripts\ruff.exe check .
All checks passed!
> .\.venv\Scripts\ruff.exe format --check .
94 files already formatted
```

**Normal run** (`.\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 26%]
........................................s............................... [ 53%]
........................................................................ [ 79%]
.......................................................                  [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=24, max=22.4 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.10 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 5.97 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

[... per-file coverage table; touched modules below ...]
TOTAL                                        4223    196    946    115    94%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
270 passed, 1 skipped, 1 warning in 369.49s (0:06:09)
```

**GPU hidden** (`$env:CUDA_VISIBLE_DEVICES="-1"; .\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 26%]
........................................s............................... [ 53%]
........................................................................ [ 79%]
.......................................................                  [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=34, max=15.7 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.40 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 7.35 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

[... per-file coverage table; touched modules below ...]
TOTAL                                        4223    196    946    115    94%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
270 passed, 1 skipped, 1 warning in 354.62s (0:05:54)
```

Coverage rows for touched modules (normal run; the GPU-hidden rows for `worker.py`,
`process.py` and `security.py` are identical):

```
Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
src\visionbench\api\main.py                    78      1      8      0    99%
src\visionbench\api\security.py                43      2     16      1    95%
src\visionbench\cli.py                        353     36     60      5    88%
src\visionbench\core\config.py                 59      0      4      0   100%
src\visionbench\core\process.py                40      8     10      2    80%
src\visionbench\jobs\handlers.py               78      3      8      2    94%
src\visionbench\jobs\params.py                 55      2      4      2    93%
src\visionbench\jobs\store.py                 123      0     22      0   100%
src\visionbench\jobs\worker.py                227     28     52      8    87%
src\visionbench\tracking\store.py             234      5     40      3    96%
src\visionbench\training\runner.py            238      5     42      4    97%
```

**Subprocess coverage works.** With `parallel = true`, `patch = ["subprocess", "_exit"]` and
`concurrency = ["thread", "multiprocessing"]` in `[tool.coverage.run]`, coverage starts itself
in every `visionbench worker` subprocess and in its spawn children, and saves data before
`os._exit`. pytest-cov combines the data files. **`jobs/worker.py`: 51% (M4) → 87%.**

For this to work, real-worker tests must let the worker exit normally. A hard-killed process
writes no data, so `WorkerProcess(max_jobs=n)` runs `visionbench worker --max-jobs n` and
`stop()` waits for it to exit. Deliberately hard-killed workers (the kill/recovery and lock
tests) still contribute nothing.

Overall coverage went from 91% to 94%. The suite went from 258 s to 369 s: there are 28 more
tests, and coverage now runs in the subprocesses.

## 6. Tests → defects, and proof that they fail on the old code

| Requirement | Test(s) | Fails on `bcd6674`? How confirmed |
|---|---|---|
| PID reuse, child: running job whose `child_pid` is a live unrelated `python -c "import time; time.sleep(60)"` with a non-matching creation time → job failed, dummy alive | `test_recovery.py::test_recovery_does_not_kill_a_reused_child_pid` (+ positive control `test_recovery_terminates_a_child_whose_identity_matches`) | **Yes, run.** The old `recover` killed the dummy (exit code 1) |
| PID reuse, worker: `worker_pid` = own pid, fresh heartbeat → still recovered | `test_running_job_with_live_worker_pid_and_fresh_heartbeat_is_recovered` | **Yes, run.** The old code left the job `running` |
| Lock file naming a live unrelated pid, no OS lock held → a new worker starts | `test_lock_file_naming_a_live_unrelated_pid_does_not_block` (old and new file formats) | **Yes, run.** Old: `WorkerError: another worker (pid …) is running` |
| Two workers at once → second fails fast | `test_second_lock_holder_fails_fast` (in-process), `test_worker_process.py::test_worker_lock_second_worker_fails_fast_and_hard_kill_releases` (real processes) | In-process: **yes, run.** The old lock let a second holder *in the same process* take over (its own pid matched). Real processes: the old code also failed fast here, so this part passes on old code too (by reasoning: a live, different pid in the file made it refuse) |
| Hard-kill the holder → a new worker acquires | same real-process test | **No (by reasoning):** the old pid file of a dead pid was also taken over. The test still guards the new OS-lock path |
| Runner identity: live pid + wrong creation time → recovered; matching → left alone | `test_orphaned_experiments_use_runner_identity` (also dead, legacy pid-only, pending-without-runner, own process) | Wrong identity: **yes, run.** Old left it `running`. Matching identity: passes on old code too (pid alive) |
| Security: upload without header 403; header + `Origin: https://evil.example` 403; header + allowed origin 202; `Host: evil.example` 400; cancel without header 403; GET unaffected; nosniff | `tests/api/test_api_security.py` (20 tests) | **Yes, run.** Old: 202, 202, 200, 200 (`cancelled`), no nosniff header |
| Cancel timeout for TRAINING | `test_worker_process.py::test_cancel_timeout_terminates_training_child` | **By reasoning:** the hook does not exist in the old code, so training honours the cancel and ends `cancelled by request`, which fails the note assertion. This closes a coverage gap; it is not a defect reproduction. The new test passes; the old code path itself was correct |
| Recovery requires the lock | `test_recover_requires_the_lock` | New API (old `recover` took no lock) |
| Schema v4 → v5 upgrade | `test_schema_upgrade_from_v4_adds_creation_time_columns` | New behaviour |

**How "run" was done.** `git archive bcd6674 src tests` was extracted into the scratchpad.
`<scratchpad>/m41/old_repro.py` was then run with `PYTHONPATH` pointing at that old source,
against the same scenarios. The new test files import new APIs (`process_is`, `set_child`,
`recover(settings, lock)`), so the scenarios were rewritten to the old signatures
(`set_child_pid`, `recover(settings, 30)`). The script killed only the dummies it had started.
Verbatim output:

```
<repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
recovered job job-20261027-174530-9a255b (dataset_verify): interrupted: worker pid 19340 not alive
code under test: <scratchpad>\m41\old\src\visionbench

[H1 pid reuse, child]
  new test expects: job failed AND the unrelated dummy process still alive
  old code:         dummy pid 1412 alive after recover: False (exit code 1)
  => new test on old code: FAILS (defect reproduced)

[M1 pid reuse, worker]
  new test expects: job recovered (failed) at startup
  old code:         job status after recover: running
  => new test on old code: FAILS (defect reproduced)

[M1 runner identity (pid reused by a live unrelated process)]
  new test expects: experiment recovered (failed)
  old code:         experiment status after recover: running
  => new test on old code: FAILS (defect reproduced)

[M1 lock file names a live unrelated pid]
  new test expects: a new worker starts
  old code:         WorkerError: another worker (pid 18780) is running on this data root
  => new test on old code: FAILS (defect reproduced)

[lock: second holder (in-process, as in test_second_lock_holder_fails_fast)]
  new test expects: WorkerError for the second holder
  old code:         second holder ALSO acquired the lock (same pid is taken over)
  => new test on old code: FAILS (defect reproduced)

[H2 upload without header]
  new test expects: 403 missing_client_header
  old code:         202 {"job": {"id": "job-20261027-174530-4e081e", "kind": "datase...
  => new test on old code: FAILS (defect reproduced)

[H2 upload with Origin https://evil.example]
  new test expects: 403 origin_not_allowed
  old code:         202
  => new test on old code: FAILS (defect reproduced)

[H2 Host: evil.example (DNS rebinding)]
  new test expects: 400 invalid_host
  old code:         200 {"status":"ok","version":"0.0.1"}
  => new test on old code: FAILS (defect reproduced)

[H2 body-less POST /jobs/{id}/cancel without header]
  new test expects: 403 missing_client_header
  old code:         200 status=cancelled
  => new test on old code: FAILS (defect reproduced)

[nosniff]
  new test expects: X-Content-Type-Options: nosniff on every response
  old code:         header on GET /api/health: None
  => new test on old code: FAILS (defect reproduced)
```

## 7. Demos (verbatim; synthetic data, not benchmark results)

### a. Hard kill of the worker, `serve` restarted within 5 s

`<scratchpad>/m41/demo_a.py`:
- starts `visionbench serve --port 8765`, then uploads, splits, and starts a 500-epoch CUDA
  training through the API (with the header);
- reads the worker identity from `db/worker.lock` and the jobs row, verifies it with
  `process_is` and that the worker descends from the serve it started, and kills it with
  `taskkill /F`;
- stops that serve and starts a new one at once.

```
=== M4.1 demo a: hard kill + restart within 5 s (synthetic data; not benchmark results)
[t+   0.0s] started `visionbench serve --port 8765` (serve pid 12160)
[t+   8.2s] API answers GET /api/health
[t+  32.8s] POST /api/experiments -> 202, job job-20261027-175158-a2bc0f (500 epochs)
[t+  44.5s] job job-20261027-175158-a2bc0f: status=running epoch=2
db/worker.lock (informational): {"pid": 17164, "create_time": 1793123493.4721398}
jobs row: worker=(17164, 1793123493.4721398) child=(13364, 1793123519.4313717)
process_is(17164, 1793123493.4721398) = True; it descends from our serve pid 12160
[t+  44.8s] taskkill /F /PID 17164 -> SUCCESS: The process with PID 17164 has been terminated.
[t+  44.9s] old serve stopped; job status in the DB is still 'running' -> running
[t+  45.0s] started `visionbench serve --port 8765` (serve pid 4704)
[t+  45.0s] restart issued 0.4 s after the kill (old stale window: 30 s)
[t+  55.4s] API answers GET /api/health
[t+  62.4s] recovered 17.8 s after the kill

>>> GET /api/jobs/job-20261027-175158-a2bc0f
<<< 200
{
 "id": "job-20261027-175158-a2bc0f",
 "status": "failed",
 "error": "interrupted: worker 17164 is gone; child process 13364 not running (pid gone or reused)",
 "heartbeat_at": "2026-10-27T17:52:09+00:00",
 "finished_at": "2026-10-27T17:52:28+00:00"
}
experiment exp-20261027-175208-0e6efc: status=failed error='interrupted: worker 17164 is gone; child process 13364 not running (pid gone or reused) (job job-20261027-175158-a2bc0f)'
  epochs kept=7
child (13364, 1793123519.4313717) running: False

log lines of the restarted serve's worker:
  worker process started (pid 3488)
  2026-10-27 22:52:27,970 INFO visionbench.worker worker pid 9708 started on data root <scratchpad>\m41\work\var
  2026-10-27 22:52:28,053 WARNING visionbench.worker recovered job job-20261027-175158-a2bc0f (train): interrupted: worker 17164 is gone; child process 13364 not running (pid gone or reused)
  2026-10-27 22:52:28,053 WARNING visionbench.worker recovered experiment exp-20261027-175208-0e6efc: interrupted: worker 17164 is gone; child process 13364 not running (pid gone or reused) (job job-20261027-175158-a2bc0f)
[t+  62.6s] serve stopped
```

- The restart was issued **0.4 s** after the kill.
- The new worker logged `recovered job` **0.08 s** after it logged its start (22:52:27.970 →
  22:52:28.053).
- The 17.8 s from kill to recovery is the restarted `serve` and worker starting up (Python +
  torch import). Recovery waited for nothing.

For honesty: in this exact scenario the old code would also have recovered at worker start,
because the dead worker pid was not reused. The difference only shows when a pid **is**
reused, which demo c and §6 cover.

### b. curl.exe against `serve --no-worker` (port 8766)

`<scratchpad>/m41/demo_b.py` prints each command and curl's status line, the security-relevant
headers and the body. Long bodies are cut at 400 characters, marked `...`.

```
=== M4.1 demo b: `visionbench serve --port 8766 --no-worker` (curl.exe) ===
(headers shown: status line, content-type, nosniff, CORS allow-origin)

## 1. upload WITHOUT the client header -> 403

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/datasets/uploads -F name=shapes -F file=@s.zip
HTTP/1.1 403 Forbidden
content-type: application/json
x-content-type-options: nosniff
{"error":{"code":"missing_client_header","message":"state-changing requests need the header 'X-VisionBench-Request: 1'"}}

## 2. upload with the header but a foreign Origin -> 403

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/datasets/uploads -H "X-VisionBench-Request: 1" -H "Origin: https://evil.example" -F name=shapes -F file=@s.zip
HTTP/1.1 403 Forbidden
content-type: application/json
x-content-type-options: nosniff
{"error":{"code":"origin_not_allowed","message":"request Origin is not an allowed CORS origin"}}

## 3. bad Host header (DNS rebinding) -> 400, even for GET

> curl.exe --no-progress-meter -i -H "Host: evil.example" http://127.0.0.1:8766/api/health
HTTP/1.1 400 Bad Request
content-type: application/json
x-content-type-options: nosniff
{"error":{"code":"invalid_host","message":"Host header is not an allowed local host name"}}

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/datasets/uploads -H "Host: evil.example:8766" -H "X-VisionBench-Request: 1" -F name=shapes -F file=@s.zip
HTTP/1.1 400 Bad Request
content-type: application/json
x-content-type-options: nosniff
{"error":{"code":"invalid_host","message":"Host header is not an allowed local host name"}}

## 4. correct requests -> 2xx

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/datasets/uploads -H "X-VisionBench-Request: 1" -F name=shapes -F file=@s.zip
HTTP/1.1 202 Accepted
content-type: application/json
x-content-type-options: nosniff
{"job":{"id":"job-20261027-175024-43a5b2","kind":"dataset_import","status":"queued","params":{"name":"shapes","original_filename":"s.zip","relpath":null,"source":"upload","upload_id":"up-1a1751cf84104f5a9f5d66efc43ce4da"},"result":null,"error":null,"progress":null,"cancel_requested":false,"heartbeat_at":null,"created_at":"2026-10-27T17:50:24+00:00","started_at":null,"finished_at":null}}

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/datasets/uploads -H "X-VisionBench-Request: 1" -H "Origin: http://localhost:5173" -F name=shapes2 -F file=@s.zip
HTTP/1.1 202 Accepted
content-type: application/json
access-control-allow-origin: http://localhost:5173
x-content-type-options: nosniff
{"job":{"id":"job-20261027-175024-bcc5ca","kind":"dataset_import","status":"queued","params":{"name":"shapes2","original_filename":"s.zip","relpath":null,"source":"upload","upload_id":"up-dcb4921aec0544fa8d223aae00b135d7"},"result":null,"error":null,"progress":null,"cancel_requested":false,"heartbeat_at":null,"created_at":"2026-10-27T17:50:24+00:00","started_at":null,"finished_at":null}}

## 5. body-less cancel: without the header -> 403; with it -> 200

> curl.exe --no-progress-meter -i -X POST http://127.0.0.1:8766/api/jobs/job-20261027-175024-43a5b2/cancel
HTTP/1.1 403 Forbidden
content-type: application/json
x-content-type-options: nosniff
{"error":{"code":"missing_client_header","message":"state-changing requests need the header 'X-VisionBench-Request: 1'"}}

> curl.exe --no-progress-meter -i -X POST -H "X-VisionBench-Request: 1" http://127.0.0.1:8766/api/jobs/job-20261027-175024-43a5b2/cancel
HTTP/1.1 200 OK
content-type: application/json
x-content-type-options: nosniff
{"id":"job-20261027-175024-43a5b2","kind":"dataset_import","status":"cancelled","params":{"name":"shapes","original_filename":"s.zip","relpath":null,"source":"upload","upload_id":"up-1a1751cf84104f5a9f5d66efc43ce4da"},"result":null,"error":"cancelled before start","progress":null,"cancel_requested":false,"heartbeat_at":null,"created_at":"2026-10-27T17:50:24+00:00","started_at":null,"finished_at":" ...

## 6. GET needs no header

> curl.exe --no-progress-meter -i http://127.0.0.1:8766/api/health
HTTP/1.1 200 OK
content-type: application/json
x-content-type-options: nosniff
{"status":"ok","version":"0.0.1"}

serve stopped
```

The README `curl.exe` examples (split with the header, cancel without and with it) were also
run verbatim in PowerShell against `serve --no-worker` on port 8000. The outputs matched the
README: 202 job, `missing_client_header`, then `cancelled`.

### c. PID reuse of a job's child, outside pytest, with the real `visionbench worker`

```
=== M4.1 demo c: pid reuse of a job's child process ===
dummy process: pid 1272, creation time 1793123461.3188045 (unrelated to VisionBench)
job row before: {"id": "job-20261027-175101-dec561", "status": "running", "worker_pid": 3808, "child_pid": 1272} child_create_time=1793119861.3188045
process_is(1272, 1793119861.3188045) = False
pid 1272 exists: True

> visionbench worker --max-jobs 1   (a queued job of an unknown dataset makes it exit)
  2026-10-27 22:51:07,245 INFO visionbench.worker worker pid 16632 started on data root <workdir>/var_c
  2026-10-27 22:51:07,300 WARNING visionbench.worker recovered job job-20261027-175101-dec561 (dataset_verify): interrupted: worker 3808 is gone; child process 1272 not running (pid gone or reused)
  2026-10-27 22:51:07,337 INFO visionbench.worker job job-20261027-175102-7dcb5a (dataset_verify) started in child pid 17680
  2026-10-27 22:51:15,046 ERROR job job-20261027-175102-7dcb5a failed: DatasetNotFoundError: dataset ds-0123456789ab not found
  2026-10-27 22:51:16,921 INFO visionbench.worker job job-20261027-175102-7dcb5a finished: failed
worker exit code: 0

job after: status=failed error='interrupted: worker 3808 is gone; child process 1272 not running (pid gone or reused)'
dummy pid 1272: still running = True; process_is(pid, real creation time) = True
dummy stopped by this script
```

## 8. Deviations

1. **Host check is not Starlette's `TrustedHostMiddleware`** (decision 3 in §2). The behaviour
   matches; the 400 body uses the JSON error schema.
2. **`recover()` signature** is now `recover(settings, lock)`. It requires the `WorkerLock`
   yielded by `worker_lock()`, which enforces "runs only while holding the lock".
3. **`VISIONBENCH_HEARTBEAT_STALE_SECONDS` removed** (decision 5).
4. **The process-identity helpers live in `core/process.py`**, not in `jobs/worker.py`. The
   training runner also needs `own_identity()`, and importing `jobs.worker` from
   `training.runner` would be circular (`worker → handlers → runner`). The lock and recovery
   code is in `jobs/worker.py`.
5. **Recovery messages changed**: `interrupted: worker <pid> is gone; child process <pid> not
   running (pid gone or reused)` and `interrupted: runner process <pid> not running`.
6. Apart from the tests listed in §6, existing tests changed only to follow these changes:
   - the client sends the header and a local Host;
   - the 405 test sends the header;
   - schema version assertions now expect 5;
   - real-worker tests use `max_jobs`;
   - recovery and lock tests moved from `test_job_store.py` to `test_recovery.py`.

## 9. Not verified / limitations

- POSIX: the psutil paths and the `fcntl.flock` lock were not run (Windows only here). The CI
  workflow has never run.
- The OS lock is per data root on a **local** filesystem. Lock semantics on network shares
  were not tested.
- A pid recorded in a pre-v5 row has no creation time and is treated as not running.
- Threat model limits: no protection against other local programs or users, or a malicious
  browser extension. That needs authentication, which is out of scope. A user who adds a
  hostname to `VISIONBENCH_EXTRA_TRUSTED_HOSTS` re-opens rebinding for that name.
- The intermediate commit `c7f620a` was not tested on its own; the full suite ran on the final
  tree.
- Hard-killed processes write no coverage data, so recovery code that runs only in a
  restarted worker is covered by the in-process recovery tests, not by those subprocesses.

## 10. Code printed for review

The final message of this task prints the full source of the process-identity, lock and
recovery part of `src/visionbench/jobs/worker.py` (and `src/visionbench/core/process.py`, which
holds the identity helpers; see §8.4), and of the security middleware module
`src/visionbench/api/security.py`. §3 contains the grep of all terminate/kill call sites.
