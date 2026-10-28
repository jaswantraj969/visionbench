# eurosat-v1: multi-seed summary — VALIDATION (selection-biased) — not test results

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
