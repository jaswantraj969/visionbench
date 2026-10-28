# Review Packet — VB-B1

## MORNING SUMMARY

1. Completed: **all 15 protocol runs** (5 models × seeds 0/1/2), each on its **first attempt**. There were no reruns, no interruptions, no OOM and no blocked model.
2. Also completed: 15 VAL evaluations; 8 latency measurements (cpu + cuda, seed 0); the seed-0 VAL comparison; the multi-seed VAL summary; ruff; the full suite in both modes (285 passed, 1 skipped in each).
3. VAL balanced accuracy, mean ± sd. **VALIDATION (selection-biased) — not test results:**
   - efficientnet_b0 0.9871 ± 0.0027
   - resnet18 (ImageNet) 0.9862 ± 0.0035
   - small_cnn 0.9556 ± 0.0027
   - resnet18 (scratch, control) 0.9125 ± 0.0485
   - majority baseline 0.1000
4. Finding (not acted on): **5 runs reached the 25-epoch cap** (`epochs_completed`): resnet18_scratch_64 s2, small_cnn s0/s1/s2 and resnet18_imagenet_224 s2.
5. Anomaly: resnet18_scratch_64 has a large seed spread. Seed 1 early-stopped with best epoch 5, at 0.8578, against 0.9300 and 0.9499 for the other seeds.
6. Anomaly: training throughput dropped mid-session, from about 30 s to about 50 s per resnet18-scratch epoch from roughly 20:25Z. This is why seed 2 took 22 min. Training times are indicative only.
7. The protocol was not amended. No config, hyperparameter, batch size, `num_workers` or AMP setting was changed, and the protocol hashes still verify.
8. **Zero TEST evaluations exist in the whole database** (`run_benchmark.py check`; SQLite count by split: `val` 15 and nothing else).

---

**Task:** VB-B1, "Freeze the EuroSAT benchmark protocol and run validation-only training (3 seeds per model)".
**Dates:** 2026-10-27/28. Training ran unattended overnight, under the supervisor's rules.
**Environment:**
- Windows 11 Home 10.0.26300, Python 3.11.15, torch 2.14.1+cu130;
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU;
- CPU: Intel64 Family 6 Model 186.

All numbers below come from commands run in this session. Paths under the session scratchpad are written as `<scratchpad>`.

> **Every model metric in this packet is VALIDATION (selection-biased) — not test results.**
> Val chose each checkpoint (best `val_loss`), so these numbers overestimate performance on new data.

---

## 1. Commits (local only; no remote, nothing pushed)

```
ddbdae1 2026-10-28T09:53:42+05:00 B1: runs.json, VAL results and docs (eurosat-v1; validation only, no test)
9eba5e3 2026-10-28T09:53:42+05:00 B1: summary reports stopped_reason, epoch cap and indicative time; relative log paths
960c0a4 2026-10-28T01:10:43+05:00 B1: driver follows the supervisor's budget decision (option 3, no amendment)
da78fe2 2026-10-28T01:07:06+05:00 B1: benchmark driver, pilot script and pilot record (eurosat-v1)
c7d32c0 2026-10-28T00:40:35+05:00 B1: freeze the eurosat-v1 benchmark protocol (before any benchmark run)
29ba89c 2026-10-28T00:37:51+05:00 B1 step 0: Copernicus year note, audit field rename, frozen split in DATASETS.md
```

- `29ba89c` step 0: no year in the Copernicus notice, audit field rename, frozen split in DATASETS.md.
- `c7d32c0`: protocol, YAMLs and `PROTOCOL_HASHES.txt`, **before any experiment** (§2).
- `da78fe2`: pilot script, driver and `pilot.json`.
- `960c0a4`: driver follows the supervisor's budget decision (run order, resumable, interruption and OOM handling). No protocol change.
- `9eba5e3`: driver reporting only, made after the runs and changing no run.
  - `summary` reports epochs trained, `stopped_reason` (parsed from each run's CLI log, because the DB does not store it), indicative train time, and the list of runs at the epoch cap.
  - `runs.json` stores log paths relative to the repo, so there is no user path.
- `ddbdae1`: `runs.json`, `results/` and docs (README reproduction commands, EXPERIMENT_PROTOCOL link, PROJECT_STATUS, CHANGELOG).
- This packet is committed last.

## 2. Evidence that the protocol precedes the first run

| Event | Time (UTC) |
|---|---|
| Protocol commit `c7d32c0` (author = committer date `2026-10-28T00:40:35+05:00`) | **2026-10-27 19:40:35** |
| Earliest experiment in the DB (`exp-20261027-195104-d067aa`, a pilot run) | 2026-10-27 19:51:04 |
| First benchmark run (`exp-20261027-201053-5f2809`, majority_baseline seed 0) | 2026-10-27 20:10:53 |

- Just before `c7d32c0`, the experiment DB had no tables.
- `git log` lists `c7d32c0` as the **only** commit touching `BENCHMARK_PROTOCOL.md`, `PROTOCOL_HASHES.txt` and the YAMLs. `git diff HEAD` on them is empty.
- `sha256sum -c PROTOCOL_HASHES.txt` reports OK for all 6 files.
- The driver re-verifies these hashes before training.

```
BENCHMARK_PROTOCOL.md: OK
majority_baseline.yaml: OK
small_cnn_scratch_64.yaml: OK
resnet18_imagenet_224.yaml: OK
efficientnet_b0_imagenet_224.yaml: OK
resnet18_scratch_64.yaml: OK
```

Amendments: **none.**

## 3. Pilot (excluded from all results; `benchmarks/eurosat-v1/pilot.json`)

The pilot ran from 19:50:46Z to 20:06:11Z, with 12 pilot experiments.

**Timing:** 1 epoch, seed 0, batch 64, 296 steps per epoch, each in a fresh process.

| model | epoch s (train + val) | peak GPU MiB (`max_memory_allocated`) | batch 64 fits | process wall s | experiment |
|---|---|---|---|---|---|
| small_cnn_scratch_64 | 56.30 | 242.0 | yes | 117.77 | exp-20261027-195104-d067aa |
| resnet18_imagenet_224 | 178.43 | 1594.6 | yes | 203.58 | exp-20261027-195257-71ff25 |
| efficientnet_b0_imagenet_224 | 256.59 | 5597.0 | yes | 281.81 | exp-20261027-195620-f1a5b1 |
| resnet18_scratch_64 | 31.55 | 345.1 | yes | 39.24 | exp-20261027-200055-e5454a |

- The budget upper bound, with every run at 25 epochs, was 10.99 h, over the 6 h limit.
- The supervisor chose option 3: the longer budget, no amendment.
- The actual training wall time was 20:10:44Z → 04:15:10Z, **8 h 04 min**.

**GPU determinism:** two identical CLI runs per model (seed 0, `max_steps` 50, separate processes), compared bitwise.

| model | runs | history (train_loss, val_loss, val_acc, val_bal_acc, steps) identical | best.pt tensors | differing | max abs diff |
|---|---|---|---|---|---|
| small_cnn_scratch_64 | 200135-3899f9 / 200150-c2cda6 | yes | 26 | 0 | 0.0 |
| resnet18_imagenet_224 | 200206-c48f45 / 200247-2922d3 | yes | 122 | 0 | 0.0 |
| efficientnet_b0_imagenet_224 | 200329-f8e6f2 / 200432-7e3622 | yes | 360 | 0 | 0.0 |
| resnet18_scratch_64 | 200540-23c1f7 / 200559-c3d495 | yes | 122 | 0 | 0.0 |

The pilot verified determinism **bitwise**, so a rerun with the identical config reproduces a run exactly. No rerun was needed in B1.

Caveat: bitwise identity was verified on 50-step runs. Full 25-epoch runs use the same deterministic settings but were not run twice.

## 4. Runs (`benchmarks/eurosat-v1/runs.json`)

| # | model | seed | attempt | experiment | status | epochs trained | stopped_reason | best epoch | started → finished (UTC) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | majority_baseline | 0 | 1 | exp-20261027-201053-5f2809 | completed | 1 | baseline_fit | 1 | 10-27 20:10:44 → 20:10:54 |
| 1 | majority_baseline | 1 | 1 | exp-20261027-201101-632b02 | completed | 1 | baseline_fit | 1 | 10-27 20:10:55 → 20:11:02 |
| 2 | majority_baseline | 2 | 1 | exp-20261027-201108-703436 | completed | 1 | baseline_fit | 1 | 10-27 20:11:02 → 20:11:09 |
| 3 | resnet18_scratch_64 | 0 | 1 | exp-20261027-201115-e384ed | completed | 17 | early_stopping | 12 | 10-27 20:11:09 → 20:21:17 |
| 4 | resnet18_scratch_64 | 1 | 1 | exp-20261027-202124-e89a98 | completed | 10 | early_stopping | 5 | 10-27 20:21:17 → 20:27:38 |
| 5 | resnet18_scratch_64 | 2 | 1 | exp-20261027-202749-e42275 | completed | 25 | epochs_completed | 25 | 10-27 20:27:38 → 20:49:45 |
| 6 | small_cnn_scratch_64 | 0 | 1 | exp-20261027-204957-d42310 | completed | 25 | epochs_completed | 25 | 10-27 20:49:45 → 21:10:53 |
| 7 | small_cnn_scratch_64 | 1 | 1 | exp-20261027-211106-cc7c83 | completed | 25 | epochs_completed | 25 | 10-27 21:10:53 → 21:32:28 |
| 8 | small_cnn_scratch_64 | 2 | 1 | exp-20261027-213240-d42c79 | completed | 25 | epochs_completed | 23 | 10-27 21:32:28 → 21:48:49 |
| 9 | resnet18_imagenet_224 | 0 | 1 | exp-20261027-214915-e0d372 | completed | 25 | early_stopping | 20 | 10-27 21:49:02 → 22:56:57 |
| 10 | resnet18_imagenet_224 | 1 | 1 | exp-20261027-225711-6db256 | completed | 13 | early_stopping | 8 | 10-27 22:56:58 → 23:34:39 |
| 11 | resnet18_imagenet_224 | 2 | 1 | exp-20261027-233452-43080e | completed | 25 | epochs_completed | 21 | 10-27 23:34:39 → 00:47:01 |
| 12 | efficientnet_b0_imagenet_224 | 0 | 1 | exp-20261028-004715-3da317 | completed | 10 | early_stopping | 5 | 10-28 00:47:01 → 01:38:13 |
| 13 | efficientnet_b0_imagenet_224 | 1 | 1 | exp-20261028-013827-6cbab1 | completed | 17 | early_stopping | 12 | 10-28 01:38:13 → 03:01:37 |
| 14 | efficientnet_b0_imagenet_224 | 2 | 1 | exp-20261028-030154-fe104a | completed | 15 | early_stopping | 10 | 10-28 03:01:37 → 04:15:10 |

- Every run has `attempt` 1 and `status` completed, with no errors.
- No interruption, failure or rerun is recorded.
- `config_sha256` matches `PROTOCOL_HASHES.txt` for every run.
- `stopped_reason` is not a DB column. It is parsed from the CLI's final line in each run's log under `var/benchmarks/eurosat-v1/logs/`.

### 4.1 Runs that ended at the 25-epoch cap: a finding, not acted on

| run | experiment | best epoch |
|---|---|---|
| resnet18_scratch_64 seed 2 | exp-20261027-202749-e42275 | 25 |
| small_cnn_scratch_64 seed 0 | exp-20261027-204957-d42310 | 25 |
| small_cnn_scratch_64 seed 1 | exp-20261027-211106-cc7c83 | 25 |
| small_cnn_scratch_64 seed 2 | exp-20261027-213240-d42c79 | 23 |
| resnet18_imagenet_224 seed 2 | exp-20261027-233452-43080e | 21 |

- In 3 of these 5 runs the best epoch is the last epoch (25), and in a 4th it is epoch 23. Their val loss was still falling at the cap, so a larger epoch cap might have helped them. Testing that would require a protocol amendment and reruns of all seeds; per the instructions, I took no action.
- `resnet18_imagenet_224` seed 0 also trained 25 epochs, but it stopped through **early stopping**: its best epoch was 20, and patience 5 ran out exactly at epoch 25. It is not counted as capped.

### 4.2 Why resnet18_scratch_64 seed 2 took ~22 min (vs 6–10 min)

This comes from the epoch histories in its CLI log. There are two causes:

1. **More epochs: 25, against 17 (seed 0) and 10 (seed 1).**
   - Seed 2's val_loss kept setting new bests late in training, at epochs 16, 17, 18, 19, 23, 24 and 25. Five non-improving epochs in a row never happened, so it ran to the cap with `epochs_completed`.
   - Seed 0 stopped after epoch 17 (best epoch 12).
   - Seed 1 stopped after epoch 10 (best epoch 5).
2. **Slower epochs: about 51 s, against about 30 s.**
   - Every seed 2 epoch took 50–53 s (sum 1305.8 s; mean 52.2 s).
   - Most epochs of seeds 0 and 1 took 30–32 s.
   - The slowdown is not specific to the run: it began mid-session, around 20:25Z. Seed 0's epochs 13–15 (49–53 s) and seed 1's epochs 8–10 (50–52 s) were already slow, and seed 2 started at 20:27:38Z.
   - The likely cause is the laptop's thermal or power state, or another load on the machine; I did not measure which.
   - Training is deterministic, so this affects time only, not results.

Together: 25 epochs × about 52 s = 1306 s of epochs, which gives the **1314 s** `train_seconds`.

## 5. VAL comparison, seed 0 of every model (`results/val_comparison_seed0.md`)

**VALIDATION (selection-biased) — not test results.**
- Command: `visionbench compare <5 seed-0 experiments> --split val --eval-ids <their val evaluations> --latency-device cuda --note "VALIDATION (selection-biased) — not test results. …"`
- Report: `rp-20261028-042730-bddd0b`.
- Confound flags: `input_resolution_differs`, `pretrained_vs_scratch`.

### Model comparison (val split)

> **VALIDATION (selection-biased) — not test results. eurosat-v1 benchmark, seed 0 of each protocol model (UI/demo rule); checkpoints were selected on this val split. Pretraining and input resolution are confounded (control: resnet18_scratch_64).**

- Report: `rp-20261028-042730-bddd0b` (2026-10-28T04:27:30+00:00, visionbench 0.0.1)
- Dataset: `ds-61112928e412` (fingerprint `53a853ae82ee65dd…`), split `sp-55320e11726c`, 4050 val samples, classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Reference for paired differences: `exp-20261027-201053-5f2809`

**Selection bias:** validation metrics are selection-biased: the same validation data chose each model's checkpoint (early stopping / best epoch), so they overestimate performance on new data

Latency: model forward pass only, measured on **cuda** only (`visionbench latency`); other devices are never mixed in.

#### Models

| experiment | model | weights | input px | params | ckpt bytes | latency bs1 (median ms, device) | train s | best epoch | val selection | val bal. acc |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp-20261027-201053-5f2809` | majority_baseline | none | n/a | n/a | 929 | n/a: majority_baseline has no forward pass | 0.2 | 1 | val_loss=2.2948 | 0.1000 |
| `exp-20261027-201115-e384ed` | resnet18 | scratch | 64 | 11181642 | 44794891 | 4.288 (cuda) | 600.9 | 12 | val_loss=0.1906 | 0.9300 |
| `exp-20261027-204957-d42310` | small_cnn | scratch | 64 | 391466 | 1576885 | 1.730 (cuda) | 1254.7 | 25 | val_loss=0.1166 | 0.9588 |
| `exp-20261027-214915-e0d372` | resnet18 | imagenet | 224 | 11181642 | 44794891 | 4.061 (cuda) | 4060.2 | 20 | val_loss=0.0383 | 0.9869 |
| `exp-20261028-004715-3da317` | efficientnet_b0 | imagenet | 224 | 4020358 | 16340197 | 13.526 (cuda) | 3056.0 | 5 | val_loss=0.0430 | 0.9840 |

#### Val metrics

Point value [95% percentile bootstrap CI] from each evaluation (1000 resamples, seed 0).

| experiment | model | checkpoint | accuracy | balanced accuracy | macro F1 | test evaluations so far |
|---|---|---|---|---|---|---|
| `exp-20261027-201053-5f2809` | majority_baseline | baseline (pinned `ev-20261028-041748-ef337a`) | 0.1111 [0.1012, 0.1207] | 0.1000 [0.1000, 0.1000] | 0.0200 [0.0184, 0.0215] | 0 |
| `exp-20261027-201115-e384ed` | resnet18 | best (pinned `ev-20261028-042026-b5cb93`) | 0.9321 [0.9247, 0.9395] | 0.9300 [0.9222, 0.9376] | 0.9305 [0.9227, 0.9381] | 0 |
| `exp-20261027-204957-d42310` | small_cnn | best (pinned `ev-20261028-042134-cf3fb5`) | 0.9605 [0.9546, 0.9664] | 0.9588 [0.9526, 0.9649] | 0.9596 [0.9536, 0.9656] | 0 |
| `exp-20261027-214915-e0d372` | resnet18 | best (pinned `ev-20261028-041854-e1bd88`) | 0.9877 [0.9842, 0.9911] | 0.9869 [0.9829, 0.9905] | 0.9873 [0.9836, 0.9908] | 0 |
| `exp-20261028-004715-3da317` | efficientnet_b0 | best (pinned `ev-20261028-041610-12527c`) | 0.9847 [0.9807, 0.9884] | 0.9840 [0.9798, 0.9879] | 0.9844 [0.9803, 0.9881] | 0 |

#### Paired differences vs reference `exp-20261027-201053-5f2809`

Paired percentile bootstrap: 1000 resamples, seed 0, the same resampled indices for both models; difference = experiment − reference.

| experiment | model | Δ balanced accuracy [95% CI] | Δ macro F1 [95% CI] |
|---|---|---|---|
| `exp-20261027-201115-e384ed` | resnet18 | 0.8300 [0.8222, 0.8376] | 0.9105 [0.9023, 0.9181] |
| `exp-20261027-204957-d42310` | small_cnn | 0.8588 [0.8526, 0.8649] | 0.9396 [0.9333, 0.9458] |
| `exp-20261027-214915-e0d372` | resnet18 | 0.8869 [0.8829, 0.8905] | 0.9673 [0.9632, 0.9709] |
| `exp-20261028-004715-3da317` | efficientnet_b0 | 0.8840 [0.8798, 0.8879] | 0.9644 [0.9601, 0.9685] |

#### Validation ranking

Basis: validation balanced accuracy at the val-selected epoch (VAL only).

- rank 1: `exp-20261027-214915-e0d372` resnet18: 0.9869
- rank 2: `exp-20261028-004715-3da317` efficientnet_b0: 0.9840
- rank 3: `exp-20261027-204957-d42310` small_cnn: 0.9588
- rank 4: `exp-20261027-201115-e384ed` resnet18: 0.9300
- rank 5: `exp-20261027-201053-5f2809` majority_baseline: 0.1000

#### Confound flags

- **input_resolution_differs**: models see different input resolutions; architecture and resolution are confounded. Values: `{"exp-20261027-201115-e384ed": 64, "exp-20261027-204957-d42310": 64, "exp-20261027-214915-e0d372": 224, "exp-20261028-004715-3da317": 224}`
- **pretrained_vs_scratch**: ImageNet-pretrained and scratch models are compared; pretraining is a confound. Values: `{"exp-20261027-201115-e384ed": "scratch", "exp-20261027-204957-d42310": "scratch", "exp-20261027-214915-e0d372": "imagenet", "exp-20261028-004715-3da317": "imagenet"}`

## 6. Multi-seed VAL summary (`results/val_summary.md`)

### eurosat-v1: multi-seed summary — VALIDATION (selection-biased) — not test results

**VALIDATION (selection-biased) — not test results.** Checkpoints were selected on this same validation split (best val_loss), so these numbers are optimistic. Mean ± sample sd over seeds 0/1/2 (n = 3); the split is fixed, so the sd reflects training variance only.

| model | val balanced accuracy (mean ± sd) | val macro F1 (mean ± sd) | per-seed bal. acc. (s0 / s1 / s2) | per-seed macro F1 (s0 / s1 / s2) | best epoch (s0/s1/s2) | epochs trained (s0/s1/s2) | stopped_reason (s0/s1/s2) | train time s, indicative (s0/s1/s2) |
|---|---|---|---|---|---|---|---|---|
| majority_baseline | 0.1000 ± 0.0000 | 0.0200 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 | 0.0200 / 0.0200 / 0.0200 | 1 / 1 / 1 | 1 / 1 / 1 | baseline_fit / baseline_fit / baseline_fit | 0 / 0 / 0 |
| resnet18_scratch_64 | 0.9125 ± 0.0485 | 0.9127 ± 0.0488 | 0.9300 / 0.8578 / 0.9499 | 0.9305 / 0.8575 / 0.9502 | 12 / 5 / 25 | 17 / 10 / 25 | early_stopping / early_stopping / epochs_completed | 601 / 372 / 1314 |
| small_cnn_scratch_64 | 0.9556 ± 0.0027 | 0.9563 ± 0.0029 | 0.9588 / 0.9542 / 0.9538 | 0.9596 / 0.9547 / 0.9545 | 25 / 25 / 23 | 25 / 25 / 25 | epochs_completed / epochs_completed / epochs_completed | 1255 / 1280 / 969 |
| resnet18_imagenet_224 | 0.9862 ± 0.0035 | 0.9864 ± 0.0036 | 0.9869 / 0.9824 / 0.9892 | 0.9873 / 0.9825 / 0.9894 | 20 / 8 / 21 | 25 / 13 / 25 | early_stopping / early_stopping / epochs_completed | 4060 / 2245 / 4327 |
| efficientnet_b0_imagenet_224 | 0.9871 ± 0.0027 | 0.9873 ± 0.0026 | 0.9840 / 0.9892 / 0.9880 | 0.9844 / 0.9892 / 0.9882 | 5 / 12 / 10 | 10 / 17 / 15 | early_stopping / early_stopping / early_stopping | 3056 / 4988 / 4394 |

Train time is indicative only: it depends on the session and on the laptop's thermal state.

Runs that reached the 25-epoch cap (stopped_reason `epochs_completed`) instead of early stopping — a finding, not acted on: resnet18_scratch_64 seed 2 (exp-20261027-202749-e42275; best epoch 25); small_cnn_scratch_64 seed 0 (exp-20261027-204957-d42310; best epoch 25); small_cnn_scratch_64 seed 1 (exp-20261027-211106-cc7c83; best epoch 25); small_cnn_scratch_64 seed 2 (exp-20261027-213240-d42c79; best epoch 23); resnet18_imagenet_224 seed 2 (exp-20261027-233452-43080e; best epoch 21).

Per-class VAL recall, seed 0 of each model — VALIDATION (selection-biased) — not test results. Classes marked † are largely low-information (near-duplicates among them cannot be assessed by hashing; DATASETS.md).

| class | majority_baseline | resnet18_scratch_64 | small_cnn_scratch_64 | resnet18_imagenet_224 | efficientnet_b0_imagenet_224 |
|---|---|---|---|---|---|
| AnnualCrop | 0.0000 | 0.8889 | 0.9356 | 0.9778 | 0.9822 |
| Forest † (34 % low-information) | 1.0000 | 0.9822 | 0.9911 | 0.9956 | 0.9911 |
| HerbaceousVegetation | 0.0000 | 0.9222 | 0.9444 | 0.9933 | 0.9911 |
| Highway | 0.0000 | 0.9360 | 0.9547 | 0.9893 | 0.9813 |
| Industrial | 0.0000 | 0.9520 | 0.9413 | 0.9920 | 1.0000 |
| Pasture | 0.0000 | 0.9067 | 0.9233 | 0.9700 | 0.9733 |
| PermanentCrop | 0.0000 | 0.8773 | 0.9707 | 0.9733 | 0.9573 |
| Residential | 0.0000 | 0.9933 | 1.0000 | 0.9933 | 0.9889 |
| River | 0.0000 | 0.8853 | 0.9467 | 0.9840 | 0.9813 |
| SeaLake † (78 % low-information) | 0.0000 | 0.9556 | 0.9800 | 1.0000 | 0.9933 |

Runs (experiment id / val evaluation id):

- majority_baseline seed 0: exp-20261027-201053-5f2809 / ev-20261028-041748-ef337a
- majority_baseline seed 1: exp-20261027-201101-632b02 / ev-20261028-041804-166ee7
- majority_baseline seed 2: exp-20261027-201108-703436 / ev-20261028-041820-d87583
- resnet18_scratch_64 seed 0: exp-20261027-201115-e384ed / ev-20261028-042026-b5cb93
- resnet18_scratch_64 seed 1: exp-20261027-202124-e89a98 / ev-20261028-042048-83be42
- resnet18_scratch_64 seed 2: exp-20261027-202749-e42275 / ev-20261028-042111-ebe183
- small_cnn_scratch_64 seed 0: exp-20261027-204957-d42310 / ev-20261028-042134-cf3fb5
- small_cnn_scratch_64 seed 1: exp-20261027-211106-cc7c83 / ev-20261028-042156-27fe9e
- small_cnn_scratch_64 seed 2: exp-20261027-213240-d42c79 / ev-20261028-042217-2ae5cf
- resnet18_imagenet_224 seed 0: exp-20261027-214915-e0d372 / ev-20261028-041854-e1bd88
- resnet18_imagenet_224 seed 1: exp-20261027-225711-6db256 / ev-20261028-041929-b260a5
- resnet18_imagenet_224 seed 2: exp-20261027-233452-43080e / ev-20261028-042003-26e664
- efficientnet_b0_imagenet_224 seed 0: exp-20261028-004715-3da317 / ev-20261028-041610-12527c
- efficientnet_b0_imagenet_224 seed 1: exp-20261028-013827-6cbab1 / ev-20261028-041651-df21cd
- efficientnet_b0_imagenet_224 seed 2: exp-20261028-030154-fe104a / ev-20261028-041731-f7b236

## 7. Latency, seed 0 (`results/latency_seed0.md`)

- Forward pass only, on random input: 10 warm-up and 50 timed iterations.
- Measured on its own, with no split and no images read, so it is neither a validation nor a test number.

| model | device | device name | bs1 median ms | bs1 p95 ms | bs32 median ms | bs32 ms/image | latency id |
|---|---|---|---|---|---|---|---|
| resnet18_scratch_64 | cpu | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 14.000 | 18.742 | 111.436 | 3.482 | lat-20261028-042615-15f43a |
| resnet18_scratch_64 | cuda | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 4.288 | 4.645 | 4.930 | 0.154 | lat-20261028-042627-c1d08e |
| small_cnn_scratch_64 | cpu | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 4.745 | 5.957 | 58.508 | 1.828 | lat-20261028-042641-8a733a |
| small_cnn_scratch_64 | cuda | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 1.730 | 1.868 | 3.018 | 0.094 | lat-20261028-042653-f82a13 |
| resnet18_imagenet_224 | cpu | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 52.563 | 55.270 | 913.155 | 28.536 | lat-20261028-042542-f0cc3e |
| resnet18_imagenet_224 | cuda | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 4.061 | 4.735 | 37.113 | 1.160 | lat-20261028-042557-9554bb |
| efficientnet_b0_imagenet_224 | cpu | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 66.395 | 73.569 | 1552.687 | 48.521 | lat-20261028-042417-fae778 |
| efficientnet_b0_imagenet_224 | cuda | NVIDIA GeForce RTX 3050 6GB Laptop GPU | 13.526 | 18.960 | 56.039 | 1.751 | lat-20261028-042433-5fcdbd |

## 8. No-test-evaluation check

```
$ python scripts/run_benchmark.py check benchmarks/eurosat-v1
benchmark experiments: 15; pilot experiments: 12
val evaluations of benchmark experiments: 15
test evaluations of benchmark or pilot experiments: 0
test evaluations in the whole database: 0
OK: no test evaluation exists for any benchmark or pilot experiment
```

A direct SQLite query on `var/db/visionbench.db` gives:
- evaluations by split: `[('val', 15)]`;
- experiments: `[('completed', 27)]` (15 benchmark + 12 pilot);
- latency measurements: 8.

## 9. Lint and tests (after the last code change)

```
$ ruff check . ; ruff format --check .
All checks passed!
106 files already formatted

$ pytest -q -rs
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
285 passed, 1 skipped, 1 warning in 386.02s (0:06:26)

$ CUDA_VISIBLE_DEVICES=-1 pytest -q -rs
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
285 passed, 1 skipped, 1 warning in 215.66s (0:03:35)
```

- The driver scripts are not covered by the test suite. Their behaviour is shown by the real runs above.
- The interruption logic was exercised during B1 on a temporary copy of the protocol folder, before the runs started.

## 10. Rules followed, and what was not done

- **No test evaluation of any kind.** Test is evaluated once, at M8, for all seeds of all models (protocol §6).
- **No changes after seeing validation results:** no hyperparameter, batch size, epoch, `num_workers` or AMP change, and no amendment. The epoch-cap finding is reported only.
- **No cherry-picking:** every seed is reported. The comparison uses seed 0, fixed by the protocol's UI/demo rule before any run.
- **Training times are indicative.** They depend on the session and on the laptop's thermal state (§4.2).
- **Confounds remain as listed in protocol §7.**
  - Pretraining and resolution are joint. Comparing the control resnet18_scratch_64 with resnet18_imagenet_224 shows their combined effect only.
  - Learning rates differ between scratch and pretrained models and were not tuned.
  - n = 3 seeds; one fixed split.
  - SeaLake and Forest are largely low-information (78 % and 34 %), and their near-duplicates cannot be assessed by hashing.
- **Not committed:** `var/` (dataset, DB, checkpoints, logs, reports). The committed `results/*.md` files are text copies of the reports in `var/`.
