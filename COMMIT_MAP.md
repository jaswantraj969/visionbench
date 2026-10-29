# Commit map (history rewrite before publishing)

Before the repository was published, every commit's author and committer were rewritten to
`Jaswant Raj <214715227+jaswantraj969@users.noreply.github.com>` (dates kept), and local
absolute paths containing the Windows user name were replaced by `<repo>` / `<home>` in the
files of every commit (`git filter-repo --mailmap --replace-text`). Commit hashes therefore
changed. Older documents (REVIEW_PACKET.md, MASTER_PROGRESS.md, CHANGELOG, commit messages)
cite the **old** hashes; this table maps them to the published ones.

| old | new | subject |
|---|---|---|
| `4165e71` | `e6e0480` | M0: foundation baseline |
| `d3d52b5` | `0283031` | M0 follow-ups: generic ready error, dev lock file, CUDA state in test log |
| `57d7b8f` | `82ba904` | M1: dataset library - validation, dHash duplicate groups, snapshots, splits |
| `34930b6` | `184aabc` | M1: `visionbench dataset import/report/split/list` CLI |
| `e049491` | `c5bd77e` | M1 docs: protocol, architecture, README, status, changelog |
| `9581b24` | `ac7e0b9` | M1 review packet |
| `a321521` | `913a2ea` | M2 step 0: hash-ordered splits, popcount low-info guard, dataset verify |
| `3548826` | `55aff99` | M2: model registry (baseline, SmallCNN, ResNet18, EfficientNet-B0) |
| `42f638b` | `2890d0f` | M2: CLI training from split manifests with SQLite experiment tracking |
| `899248f` | `7b3b682` | M2 docs: protocol, architecture, README, status, changelog |
| `bc983ee` | `3f967a6` | M2 review packet |
| `c508d5f` | `bf72406` | M3 step 0: torch home setting, frozen BN for head_only, norm-stats cache |
| `9c4112d` | `2864155` | M3: metrics with explicit undefined values; bootstrap CIs; latency |
| `56b33fc` | `fb81d99` | M3: evaluation entry point, test-use accounting, model comparison |
| `ede73ff` | `ae18a6b` | M3 fixes: deterministic "latest evaluation"; tie-aware val ranking |
| `a2db7bf` | `20f0f20` | M3 docs: evaluation protocol, architecture, README, status, changelog |
| `a70fe97` | `a788bf4` | M3 review packet |
| `7936bda` | `42d5e0a` | M4 step 0: split-independent latency, pinned evals, forced test re-evaluation |
| `51cad45` | `4e93942` | M4: job store (schema v4, WAL) and single-worker runner with recovery |
| `3a54115` | `e8793d2` | M4: hardened ZIP dataset upload (validate all, extract, import, clean up) |
| `5d56963` | `655718d` | M4: FastAPI endpoints under /api, serve command, API tests |
| `f349f5d` | `5d5f959` | serve: say which pid is printed for the worker process |
| `3c1955e` | `d4fcf66` | M4 docs: API layer, job lifecycle, recovery, upload pipeline, protocol |
| `bcd6674` | `6480557` | M4 review packet |
| `c7f620a` | `6b01950` | M4.1: process identity, OS worker lock, recovery under the lock |
| `d23e318` | `0106c0c` | M4.1: protect the local API against cross-site requests and DNS rebinding |
| `fd78b72` | `e7045eb` | M4.1 docs: process identity, OS lock, recovery rule, local-API threat model |
| `6166ce4` | `6cfdc3e` | M4.1 review packet |
| `052ef34` | `9959b32` | D1 step 0: EXIF orientation policy (one shared image decoder) |
| `b9c8e6c` | `dfd80cc` | D1: independent split audit (pHash), oversized-group warnings, low-info breakdown |
| `297879d` | `5620c52` | D1 docs: DATASETS.md (EuroSAT), EXIF policy, audit method, calibration procedure |
| `6aa11e0` | `baf7e03` | D1 review packet |
| `29ba89c` | `b737653` | B1 step 0: Copernicus year note, audit field rename, frozen split in DATASETS.md |
| `c7d32c0` | `ae8319d` | B1: freeze the eurosat-v1 benchmark protocol (before any benchmark run) |
| `da78fe2` | `e441819` | B1: benchmark driver, pilot script and pilot record (eurosat-v1) |
| `960c0a4` | `03aaf69` | B1: driver follows the supervisor's budget decision (option 3, no amendment) |
| `9eba5e3` | `b171b16` | B1: summary reports stopped_reason, epoch cap and indicative time; relative log paths |
| `ddbdae1` | `a350cb5` | B1: runs.json, VAL results and docs (eurosat-v1; validation only, no test) |
| `d8acd2b` | `5b116eb` | B1 review packet |
| `14305b1` | `14cdf34` | M5 step 0: stopped_reason in the DB (schema v6, backfilled); full-length repro check |
| `ea64ec1` | `f9ac483` | M5: inference service (verified loading, LRU cache, concurrency slots, test lock) |
| `e5e9a6f` | `61bac34` | M5: Grad-CAM with exact val-transform geometry, overlay and sanity checks |
| `5136b1c` | `b17194d` | M5: predict and gradcam API endpoints, image upload parser, tests and demo script |
| `76f4016` | `3282fee` | M5 docs and review packet |
| `67347c3` | `55bc78f` | M6 step 0: SmallCNN Grad-CAM on the post-ReLU map; cascading randomization; keep-awake |
| `bf1de6e` | `feca6bc` | M6: freeze the eurosat-c-v1 robustness protocol (before any robustness evaluation) |
| `7590c03` | `1145e68` | M6: robustness library (corruptions, materialization, evaluation, report), schema v7, CLI |
| `d3b6f64` | `ac2a603` | M6: robustness worker job and API endpoints |
| `25baac5` | `33fe16e` | M6: robustness driver, identity check, rename retry, check covers robustness |
| `597c4dc` | `3310614` | M6: VAL robustness results (15 runs x 12 conditions) and docs |
| `9f981c0` | `29d9512` | M6 review packet |
| `b5ee799` | `b30a7ad` | Phase 0a: robustness protocol amendment 1 (text only): corrected resolution/upsampling note |
| `07b2000` | `6f7994e` | Phase 0b: document pHash unreliability on dark low-texture tiles; strong noise severities |
| `9a088bf` | `2649066` | Phase 0d: show "n/a" for training settings the majority baseline does not use |
| `6cf7de2` | `76d6953` | Phase 0c: README robustness API curl commands, verified against a live serve |
| `3e15d55` | `a691279` | VB-FINAL: MASTER_PROGRESS.md; Phase 0 evidence in REVIEW_PACKET.md (Gate 0 passed) |
| `7b53c81` | `ed92d8f` | Phase A1 (backend): read-only views for the frontend; same-origin static serving |
| `9d1678a` | `f892935` | Fix: accept the API's own origin on state-changing requests (served frontend was refused) |
| `99da9d4` | `89233cc` | Phase A1-A3 (frontend): React + TypeScript app over the real API, tests, Playwright e2e |
| `bfdd07d` | `f1d5361` | Gate A passed: MASTER_PROGRESS and REVIEW_PACKET Phase A evidence |
| `d411106` | `cb50ccb` | Phase B1-B3 (code): Docker image, compose, CI workflow, demo-data command, Linux CPU locks |
| `4c9f83f` | `cb5c97b` | Phase B4: documentation for 1.0.0rc1; scripts/linux_tests.sh; version 1.0.0rc1 |
| `b1a6ef3` | `239d566` | Phase C preparation (before any TEST access): TEST drivers, freeze check, final report script |
| `3677583` | `887bba4` | C1: eurosat-v1 FINAL_ANALYSIS_PLAN.md + FINAL_ANALYSIS_PLAN_HASHES.txt (before any TEST access) |
| `c2aeec4` | `135efa1` | Gate B passed: release candidate 1.0.0rc1 (docs only) |
| `288dbe1` | `7a9a033` | C2: freeze checklist items a, f, g recorded (b-e run first in C3) |
| `eb3d9b7` | `3782f39` | C3 STOPPED: 15 clean TEST evaluations recorded; TEST materialization refused (owner decision) |
| `28f97a7` | `71e564c` | S1: split-aware corruption cache path; FINAL_ANALYSIS_PLAN amendment 1 (procedural); 1.0.0rc2 |
| `fe52f1c` | `f396433` | S1 gate recorded (MASTER_PROGRESS) |
| `9a613df` | `b606e1a` | S2 (before the robustness TEST run): check asserts 1 clean + 12 robustness TEST per run after M8 |
| `c55e40c` | `74a969b` | S2: robustness TEST evaluation done (15 x 12, first attempts); TEST conditions byte-identical on rerun |
| `895703b` | `b1d948b` | S3: final report (TEST) per the plan; TEST images unlocked per finalised split; Benchmark TEST section |
| `fb18dfb` | `be52ad6` | S4: screenshots from the real app, README results summary, DEFINITION_OF_DONE.md, MIT LICENSE |

## Second rewrite: commit-message trailers removed

After publishing, a trailer line was removed from every commit message
(`git filter-repo --message-callback`); authors, dates and file contents are unchanged. Commit hashes changed again. This table maps the first published hashes (above,
"new") to the final ones, in history order.

| published (pre-trailer) | final | subject |
|---|---|---|
| `4165e71` | `4125b15` | M0: foundation baseline |
| `d3d52b5` | `0701081` | M0 follow-ups: generic ready error, dev lock file, CUDA state in test log |
| `57d7b8f` | `b6c1bd1` | M1: dataset library - validation, dHash duplicate groups, snapshots, splits |
| `34930b6` | `9405eee` | M1: `visionbench dataset import|report|split|list` CLI |
| `e049491` | `d795b3b` | M1 docs: protocol, architecture, README, status, changelog |
| `9581b24` | `c1b93d3` | M1 review packet |
| `a321521` | `ceaba50` | M2 step 0: hash-ordered splits, popcount low-info guard, dataset verify |
| `3548826` | `67a47c5` | M2: model registry (baseline, SmallCNN, ResNet18, EfficientNet-B0) |
| `42f638b` | `f56b856` | M2: CLI training from split manifests with SQLite experiment tracking |
| `899248f` | `eab0384` | M2 docs: protocol, architecture, README, status, changelog |
| `bc983ee` | `6dfa64b` | M2 review packet |
| `c508d5f` | `a889399` | M3 step 0: torch home setting, frozen BN for head_only, norm-stats cache |
| `9c4112d` | `0950edd` | M3: metrics with explicit undefined values; bootstrap CIs; latency |
| `56b33fc` | `8e7fbe0` | M3: evaluation entry point, test-use accounting, model comparison |
| `ede73ff` | `d42a6c9` | M3 fixes: deterministic "latest evaluation"; tie-aware val ranking |
| `a2db7bf` | `67199d4` | M3 docs: evaluation protocol, architecture, README, status, changelog |
| `a70fe97` | `b858761` | M3 review packet |
| `7936bda` | `68444c7` | M4 step 0: split-independent latency, pinned evals, forced test re-evaluation |
| `51cad45` | `73b8dee` | M4: job store (schema v4, WAL) and single-worker runner with recovery |
| `3a54115` | `ada39f8` | M4: hardened ZIP dataset upload (validate all, extract, import, clean up) |
| `5d56963` | `b9d35cf` | M4: FastAPI endpoints under /api, serve command, API tests |
| `f349f5d` | `06af777` | serve: say which pid is printed for the worker process |
| `3c1955e` | `8ae3125` | M4 docs: API layer, job lifecycle, recovery, upload pipeline, protocol |
| `bcd6674` | `5fc1795` | M4 review packet |
| `c7f620a` | `6287372` | M4.1: process identity, OS worker lock, recovery under the lock |
| `d23e318` | `764818d` | M4.1: protect the local API against cross-site requests and DNS rebinding |
| `fd78b72` | `cda9783` | M4.1 docs: process identity, OS lock, recovery rule, local-API threat model |
| `6166ce4` | `8ba7f23` | M4.1 review packet |
| `052ef34` | `229e705` | D1 step 0: EXIF orientation policy (one shared image decoder) |
| `b9c8e6c` | `de2e7c2` | D1: independent split audit (pHash), oversized-group warnings, low-info breakdown |
| `297879d` | `7eae4c8` | D1 docs: DATASETS.md (EuroSAT), EXIF policy, audit method, calibration procedure |
| `6aa11e0` | `c8b8fbf` | D1 review packet |
| `29ba89c` | `28b113b` | B1 step 0: Copernicus year note, audit field rename, frozen split in DATASETS.md |
| `c7d32c0` | `5d9805f` | B1: freeze the eurosat-v1 benchmark protocol (before any benchmark run) |
| `da78fe2` | `135407c` | B1: benchmark driver, pilot script and pilot record (eurosat-v1) |
| `960c0a4` | `df42ef6` | B1: driver follows the supervisor's budget decision (option 3, no amendment) |
| `9eba5e3` | `601ef46` | B1: summary reports stopped_reason, epoch cap and indicative time; relative log paths |
| `ddbdae1` | `ae15679` | B1: runs.json, VAL results and docs (eurosat-v1; validation only, no test) |
| `d8acd2b` | `d257eda` | B1 review packet |
| `14305b1` | `194f09f` | M5 step 0: stopped_reason in the DB (schema v6, backfilled); full-length repro check |
| `ea64ec1` | `cae1b4e` | M5: inference service (verified loading, LRU cache, concurrency slots, test lock) |
| `e5e9a6f` | `9b1fd24` | M5: Grad-CAM with exact val-transform geometry, overlay and sanity checks |
| `5136b1c` | `861e61c` | M5: predict and gradcam API endpoints, image upload parser, tests and demo script |
| `76f4016` | `2c045f9` | M5 docs and review packet |
| `67347c3` | `ceaf79f` | M6 step 0: SmallCNN Grad-CAM on the post-ReLU map; cascading randomization; keep-awake |
| `bf1de6e` | `4650a5a` | M6: freeze the eurosat-c-v1 robustness protocol (before any robustness evaluation) |
| `7590c03` | `ccb4308` | M6: robustness library (corruptions, materialization, evaluation, report), schema v7, CLI |
| `d3b6f64` | `d745345` | M6: robustness worker job and API endpoints |
| `25baac5` | `a73892f` | M6: robustness driver, identity check, rename retry, check covers robustness |
| `597c4dc` | `e4ca2b8` | M6: VAL robustness results (15 runs x 12 conditions) and docs |
| `9f981c0` | `4080c6b` | M6 review packet |
| `b5ee799` | `d14b9a4` | Phase 0a: robustness protocol amendment 1 (text only): corrected resolution/upsampling note |
| `07b2000` | `7a96f39` | Phase 0b: document pHash unreliability on dark low-texture tiles; strong noise severities |
| `9a088bf` | `d5984be` | Phase 0d: show "n/a" for training settings the majority baseline does not use |
| `6cf7de2` | `094b042` | Phase 0c: README robustness API curl commands, verified against a live serve |
| `3e15d55` | `9ef5c75` | VB-FINAL: MASTER_PROGRESS.md; Phase 0 evidence in REVIEW_PACKET.md (Gate 0 passed) |
| `7b53c81` | `57683d2` | Phase A1 (backend): read-only views for the frontend; same-origin static serving |
| `9d1678a` | `217dc19` | Fix: accept the API's own origin on state-changing requests (served frontend was refused) |
| `99da9d4` | `4878ad6` | Phase A1-A3 (frontend): React + TypeScript app over the real API, tests, Playwright e2e |
| `bfdd07d` | `db4c122` | Gate A passed: MASTER_PROGRESS and REVIEW_PACKET Phase A evidence |
| `d411106` | `d506d60` | Phase B1-B3 (code): Docker image, compose, CI workflow, demo-data command, Linux CPU locks |
| `4c9f83f` | `679e3e1` | Phase B4: documentation for 1.0.0rc1; scripts/linux_tests.sh; version 1.0.0rc1 |
| `b1a6ef3` | `d029fef` | Phase C preparation (before any TEST access): TEST drivers, freeze check, final report script |
| `3677583` | `6fc9e5f` | C1: eurosat-v1 FINAL_ANALYSIS_PLAN.md + FINAL_ANALYSIS_PLAN_HASHES.txt (before any TEST access) |
| `c2aeec4` | `00f17da` | Gate B passed: release candidate 1.0.0rc1 (docs only) |
| `288dbe1` | `d5ce30b` | C2: freeze checklist items a, f, g recorded (b-e run first in C3) |
| `eb3d9b7` | `dbf5d09` | C3 STOPPED: 15 clean TEST evaluations recorded; TEST materialization refused (owner decision) |
| `28f97a7` | `b07b151` | S1: split-aware corruption cache path; FINAL_ANALYSIS_PLAN amendment 1 (procedural); 1.0.0rc2 |
| `fe52f1c` | `dd7d506` | S1 gate recorded (MASTER_PROGRESS) |
| `9a613df` | `bf01575` | S2 (before the robustness TEST run): check asserts 1 clean + 12 robustness TEST per run after M8 |
| `c55e40c` | `fc79f3f` | S2: robustness TEST evaluation done (15 x 12, first attempts); TEST conditions byte-identical on rerun |
| `895703b` | `c495083` | S3: final report (TEST) per the plan; TEST images unlocked per finalised split; Benchmark TEST section |
| `fb18dfb` | `316dc76` | S4: screenshots from the real app, README results summary, DEFINITION_OF_DONE.md, MIT LICENSE |
| `3832f9a` | `8d54ac4` | S5: version 1.0.0, CHANGELOG, COMMIT_MAP.md (history rewritten: identity, local paths) |
| `ea26be7` | `3116076` | S5: pre-publish audit recorded |
| `b2f7f8d` | `493d4f9` | S6 blocked (publishing not permitted here); S7 Upwork portfolio package |
| `75c317b` | `d3ca860` | Fix flaky CI test: disable early stopping in the cancel-timeout training test |
| `ac5b22b` | `fba3671` | DoD item 11 met (CI green); Upwork package links the live repo |
