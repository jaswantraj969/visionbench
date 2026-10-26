# Review Packet — VB-M0

**Task:** VB-M0 — Environment audit, project skeleton, and continuity files
**Date:** 2026-10-27
**Objective:** Verified foundation for VisionBench: installable `visionbench` package (src layout),
config / seeding / env-info core, FastAPI health + ready endpoints, `visionbench env` CLI, tests, CI
workflow file, continuity docs. No dataset, training, evaluation, explainability, robustness, or
frontend features.

All outputs below were produced by commands executed in this session.

## 1. Initial directory state
`ls -la` of `<repo>` before any change: **empty** (only `.` and
`..`, no hidden files). Not an existing VisionBench attempt and no unrelated code.

## 2. Environment report

| Item | Value |
|---|---|
| OS | Microsoft Windows 11 Home 10.0.26300 (build 26300) |
| Shells | Windows PowerShell 5.1.26100.9711; Git Bash (GNU bash 5.3.15, MINGW64) |
| Python on PATH | 3.13.5 (`C:\Program Files\Python313`), pip 25.2 |
| Other Pythons | uv-managed CPython 3.11.15, 3.12.13, 3.14.6. The `py` launcher lists a 3.12 at `...\Programs\Python\Python312\python.exe`, but that file does not exist (stale registry entry; `py -3.12 -m venv` failed with "The system cannot find the file specified"). |
| Python used | **3.11.15** (uv-managed, `<home>\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe`) |
| venv | works (`python -m venv .venv`); venv pip 24.0, upgraded to 26.2.2 |
| uv | 0.11.29 (only used to locate interpreters) |
| Node / npm | v22.19.0 / 10.9.3 (reported only; nothing scaffolded) |
| git | 2.55.0.windows.3; directory is **not** a git repository |
| Disk (C:) | 146.55 GB free |
| CPU | 13th Gen Intel Core i5-13420H, 8 cores / 12 logical processors (torch uses 8 threads) |
| RAM | 15.64 GB |
| GPU | **NVIDIA GeForce RTX 3050 6GB Laptop GPU**, nvidia-smi 617.42, CUDA UMD 13.4, WDDM. Also Intel UHD Graphics (iGPU). |
| torch CUDA | `torch.cuda.is_available()` = True; torch CUDA build 13.0; cuDNN 92400 |

## 3. Exact install commands (run from the project root, PowerShell)

```powershell
& "<home>\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Why cu130: a working NVIDIA driver was detected (CUDA 13.4 driver support). I checked the
indexes: cu130 and cpu both offer torch 2.14.1; cu128 tops out at 2.11.0.
Result: `Successfully installed ... torch-2.14.1+cu130 torchvision-0.29.1+cu130` (numpy 2.4.6,
pillow 12.3.0). Then `visionbench-0.0.1` with fastapi 0.162.0, starlette 1.10.0, pydantic 2.14.1,
pydantic-settings 2.16.1, uvicorn 0.54.0, pytest 9.1.1, pytest-cov 7.1.0, ruff 0.17.2, httpx 0.28.1.
CI uses `--index-url https://download.pytorch.org/whl/cpu` instead. That command was **not** run locally.

## 4. Files created
```
.env.example
.github/workflows/ci.yml
.gitignore
ARCHITECTURE.md
CHANGELOG.md
EXPERIMENT_PROTOCOL.md
IMPLEMENTATION_PLAN.md
PROJECT_STATUS.md
README.md
REVIEW_PACKET.md
pyproject.toml
src/visionbench/__init__.py
src/visionbench/cli.py
src/visionbench/api/__init__.py
src/visionbench/api/main.py
src/visionbench/core/__init__.py
src/visionbench/core/config.py
src/visionbench/core/env_info.py
src/visionbench/core/seeding.py
src/visionbench/{data,models,training,evaluation,explain,robustness,tracking,jobs}/__init__.py  (empty)
tests/api/test_health.py
tests/unit/test_cli.py
tests/unit/test_config.py
tests/unit/test_seeding.py
```
No pre-existing files were changed. Generated (gitignored) items: `.venv/`, `var/{datasets,artifacts,db}/`
(created by the live `/api/ready` call), `.coverage`, `.pytest_cache/`, `.ruff_cache/`, `src/visionbench.egg-info/`.
Pretrained weights went to the user-level torch cache, outside the repo.

## 5. Feasibility checks
Script: `feasibility.py` in the session scratchpad (not part of the repo). Output, excluding the
download progress lines:

```
== (a) CPU resnet18(weights=None) forward ==
input tensor: (1, 3, 224, 224) cpu torch.float32
output shape: (1, 1000)
first forward: 41.6 ms; mean of next 5: 28.7 ms
torch threads: 8
(extra) CUDA forward on NVIDIA GeForce RTX 3050 6GB Laptop GPU: mean of 5 after warmup 3.1 ms

== (b) pretrained weights via torchvision weights API ==
torch hub dir: <home>/.cache\torch\hub
-- resnet18 ResNet18_Weights.IMAGENET1K_V1 url=https://download.pytorch.org/models/resnet18-f37072fd.pth
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to <home>/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
   SUCCESS load 10.6s; output (1, 1000); params=11,689,512
-- efficientnet_b0 EfficientNet_B0_Weights.IMAGENET1K_V1 url=https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth" to <home>/.cache\torch\hub\checkpoints\efficientnet_b0_rwightman-7f5810bc.pth
   SUCCESS load 3.9s; output (1, 1000); params=5,288,548

== (c) seeded reproducibility ==
seed 123 vs 123 identical: True
seed 123 vs 124 identical: False
first logits (seed 123): [0.998841404914856, -0.7886704206466675, 0.7243298888206482, -0.7158678770065308]
```
- (a) CPU forward works. Timings are single-sample wall-clock figures on this laptop, not a benchmark.
- (b) Both downloads succeeded. Cache: `<home>\.cache\torch\hub\checkpoints\`. Load times include the download.
- (c) Also asserted in `tests/unit/test_seeding.py::test_identical_seeds_give_identical_outputs` (`torch.equal`, bitwise), with a counter-test for different seeds.

## 6. Lint and tests (final run)

```
> ruff check .
All checks passed!
> ruff format --check .
26 files already formatted
> pytest -q --cov=visionbench
.......................                                                  [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

Name                                     Stmts   Miss Branch BrPart  Cover
--------------------------------------------------------------------------
src\visionbench\__init__.py                  1      0      0      0   100%
src\visionbench\api\__init__.py              0      0      0      0   100%
src\visionbench\api\main.py                 54      1      6      0    98%
src\visionbench\cli.py                      21      1      2      1    91%
src\visionbench\core\__init__.py             0      0      0      0   100%
src\visionbench\core\config.py              27      0      2      0   100%
src\visionbench\core\env_info.py            11      0      0      0   100%
src\visionbench\core\seeding.py             18      0      4      1    95%
src\visionbench\data\__init__.py             0      0      0      0   100%
src\visionbench\evaluation\__init__.py       0      0      0      0   100%
src\visionbench\explain\__init__.py          0      0      0      0   100%
src\visionbench\jobs\__init__.py             0      0      0      0   100%
src\visionbench\models\__init__.py           0      0      0      0   100%
src\visionbench\robustness\__init__.py       0      0      0      0   100%
src\visionbench\tracking\__init__.py         0      0      0      0   100%
src\visionbench\training\__init__.py         0      0      0      0   100%
--------------------------------------------------------------------------
TOTAL                                      132      2     14      2    97%
23 passed, 1 warning in 5.90s
```
The first ruff run failed with `E501` (line 103 > 100 chars in `api/main.py`) plus the matching
format diff. I wrapped the line and reran. Uncovered lines: the 422 validation handler (no endpoint
takes parameters yet), the `if __name__ == "__main__"` CLI guard, and the `deterministic=False`
branch of `set_seed`. The 26 files in the format check are 20 `.py` plus 6 `.md`: this ruff version
also checks Markdown. After REVIEW_PACKET.md was added, a rerun reported `27 files already
formatted` and `ruff check .` still passed.

Ready tests: `test_ready_with_writable_data_root` (200) and `test_ready_with_unwritable_data_root`
(503, data root beneath a regular file). Directory chmod is not enforced on Windows, so this
portable method is used instead of permissions. There is also `test_ready_when_torch_import_fails` (503).

## 7. CLI: `visionbench env`
```json
{
  "visionbench": "0.0.1",
  "python": "3.11.15",
  "python_implementation": "CPython",
  "platform": "Windows-10-10.0.26300-SP0",
  "machine": "AMD64",
  "processor": "Intel64 Family 6 Model 186 Stepping 2, GenuineIntel",
  "numpy": "2.4.6",
  "torch": "2.14.1+cu130",
  "torchvision": "0.29.1+cu130",
  "torch_cuda_build": "13.0",
  "cuda_available": true,
  "cuda_device_count": 1,
  "cuda_devices": [
    "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
  ],
  "cudnn": 92400,
  "torch_num_threads": 8
}
```
(`platform.platform()` reports "Windows-10" on Windows 11; that is how Python reports it.)

## 8. Live API (uvicorn, 127.0.0.1)
Command: `uvicorn visionbench.api.main:app --host 127.0.0.1 --port 8765`; requests made with curl; server killed afterwards (confirmed: port no longer responds).
```
== GET /api/health
HTTP/1.1 200 OK
content-type: application/json
{"status":"ok","version":"0.0.1"}

== GET /api/ready
HTTP/1.1 200 OK
content-type: application/json
{"status":"ready","checks":{"data_root":"ok","torch":"ok"}}

== GET /api/nope
HTTP/1.1 404 Not Found
{"error":{"code":"http_404","message":"Not Found"}}
```
Second server on port 8766 with `VISIONBENCH_DATA_ROOT=<scratchpad>/blocker/var`, where `blocker` is a regular file:
```
HTTP/1.1 503 Service Unavailable
{"error":{"code":"not_ready","message":"data_root C:\\...\\scratchpad\\blocker\\var is not writable: [WinError 183] Cannot create a file when that file already exists: '...\\scratchpad\\blocker'"}}
```
(The paths are shortened here; the full response contained the absolute scratchpad path.)

## 9. Git
Not a git repository. **No `git init` and no commits:** the task makes commits conditional on user
authorization, and none was given. `git log`: n/a.

## 10. Known limitations
- The CI workflow has never run. The CPU-only torch build was not installed locally; only cu130 was tested.
- Only Python 3.11.15 was tested; `requires-python = ">=3.11,<3.14"` is declared, not verified for 3.12/3.13.
- The `/api/ready` 503 message includes the absolute data-root path and the OS error text. This is fine for a local tool, but it is information disclosure if the API is ever exposed.
- `/api/ready` creates the data-root subdirectories as a side effect, and writes then deletes a probe file.
- `set_seed` sets `PYTHONHASHSEED`, which only affects child processes, and enables deterministic algorithms in `warn_only` mode. Bitwise reproducibility was shown only on CPU in one process. GPU determinism is untested.
- The torch version ranges in `pyproject.toml` (`torch>=2.4,<3`, `torchvision>=0.19,<1`) are compatible ranges. Exact pins and a lock file are not provided yet.
- Test warning: Starlette deprecates `httpx` in its TestClient in favor of `httpx2`. Not acted on, because the task specifies httpx.
- Pretrained weights are cached in the user profile (`~/.cache/torch/hub`), not under `var/`.

## 11. Open questions for the supervisor
1. Should local development standardize on the CUDA build (cu130) or the CPU build for parity with CI? Both are documented.
2. Should pretrained-weight caching move under the data root (set `TORCH_HOME=<data_root>/cache/torch`) so that runs are self-contained?
3. May I `git init` and make local commits in the next task?
4. Should the `/api/ready` 503 message omit absolute paths?
5. Should a lock file be added (e.g. `uv pip compile` / pip-tools) before M2, when exact versions start to matter for experiment records?

## 12. Recommended next task
**M1:** dataset ingestion and validation (image-folder import, corrupt-file report), stats, SHA-256 +
perceptual-hash duplicate groups with label-conflict flagging, group-aware stratified splits, and a
versioned manifest, with tests proving no group leaks across splits. Use a small synthetic test
fixture; do not download any dataset.
