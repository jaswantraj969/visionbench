# Model comparison (val split)

> **VALIDATION (selection-biased) — not test results. eurosat-v1 benchmark, seed 0 of each protocol model (UI/demo rule); checkpoints were selected on this val split. Pretraining and input resolution are confounded (control: resnet18_scratch_64).**

- Report: `rp-20261028-042730-bddd0b` (2026-10-28T04:27:30+00:00, visionbench 0.0.1)
- Dataset: `ds-61112928e412` (fingerprint `53a853ae82ee65dd…`), split `sp-55320e11726c`, 4050 val samples, classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Reference for paired differences: `exp-20261027-201053-5f2809`

**Selection bias:** validation metrics are selection-biased: the same validation data chose each model's checkpoint (early stopping / best epoch), so they overestimate performance on new data

Latency: model forward pass only, measured on **cuda** only (`visionbench latency`); other devices are never mixed in.

## Models

| experiment | model | weights | input px | params | ckpt bytes | latency bs1 (median ms, device) | train s | best epoch | val selection | val bal. acc |
|---|---|---|---|---|---|---|---|---|---|---|
| `exp-20261027-201053-5f2809` | majority_baseline | none | n/a | n/a | 929 | n/a: majority_baseline has no forward pass | 0.2 | 1 | val_loss=2.2948 | 0.1000 |
| `exp-20261027-201115-e384ed` | resnet18 | scratch | 64 | 11181642 | 44794891 | 4.288 (cuda) | 600.9 | 12 | val_loss=0.1906 | 0.9300 |
| `exp-20261027-204957-d42310` | small_cnn | scratch | 64 | 391466 | 1576885 | 1.730 (cuda) | 1254.7 | 25 | val_loss=0.1166 | 0.9588 |
| `exp-20261027-214915-e0d372` | resnet18 | imagenet | 224 | 11181642 | 44794891 | 4.061 (cuda) | 4060.2 | 20 | val_loss=0.0383 | 0.9869 |
| `exp-20261028-004715-3da317` | efficientnet_b0 | imagenet | 224 | 4020358 | 16340197 | 13.526 (cuda) | 3056.0 | 5 | val_loss=0.0430 | 0.9840 |

## Val metrics

Point value [95% percentile bootstrap CI] from each evaluation (1000 resamples, seed 0).

| experiment | model | checkpoint | accuracy | balanced accuracy | macro F1 | test evaluations so far |
|---|---|---|---|---|---|---|
| `exp-20261027-201053-5f2809` | majority_baseline | baseline (pinned `ev-20261028-041748-ef337a`) | 0.1111 [0.1012, 0.1207] | 0.1000 [0.1000, 0.1000] | 0.0200 [0.0184, 0.0215] | 0 |
| `exp-20261027-201115-e384ed` | resnet18 | best (pinned `ev-20261028-042026-b5cb93`) | 0.9321 [0.9247, 0.9395] | 0.9300 [0.9222, 0.9376] | 0.9305 [0.9227, 0.9381] | 0 |
| `exp-20261027-204957-d42310` | small_cnn | best (pinned `ev-20261028-042134-cf3fb5`) | 0.9605 [0.9546, 0.9664] | 0.9588 [0.9526, 0.9649] | 0.9596 [0.9536, 0.9656] | 0 |
| `exp-20261027-214915-e0d372` | resnet18 | best (pinned `ev-20261028-041854-e1bd88`) | 0.9877 [0.9842, 0.9911] | 0.9869 [0.9829, 0.9905] | 0.9873 [0.9836, 0.9908] | 0 |
| `exp-20261028-004715-3da317` | efficientnet_b0 | best (pinned `ev-20261028-041610-12527c`) | 0.9847 [0.9807, 0.9884] | 0.9840 [0.9798, 0.9879] | 0.9844 [0.9803, 0.9881] | 0 |

## Paired differences vs reference `exp-20261027-201053-5f2809`

Paired percentile bootstrap: 1000 resamples, seed 0, the same resampled indices for both models; difference = experiment − reference.

| experiment | model | Δ balanced accuracy [95% CI] | Δ macro F1 [95% CI] |
|---|---|---|---|
| `exp-20261027-201115-e384ed` | resnet18 | 0.8300 [0.8222, 0.8376] | 0.9105 [0.9023, 0.9181] |
| `exp-20261027-204957-d42310` | small_cnn | 0.8588 [0.8526, 0.8649] | 0.9396 [0.9333, 0.9458] |
| `exp-20261027-214915-e0d372` | resnet18 | 0.8869 [0.8829, 0.8905] | 0.9673 [0.9632, 0.9709] |
| `exp-20261028-004715-3da317` | efficientnet_b0 | 0.8840 [0.8798, 0.8879] | 0.9644 [0.9601, 0.9685] |

## Validation ranking

Basis: validation balanced accuracy at the val-selected epoch (VAL only).

- rank 1: `exp-20261027-214915-e0d372` resnet18: 0.9869
- rank 2: `exp-20261028-004715-3da317` efficientnet_b0: 0.9840
- rank 3: `exp-20261027-204957-d42310` small_cnn: 0.9588
- rank 4: `exp-20261027-201115-e384ed` resnet18: 0.9300
- rank 5: `exp-20261027-201053-5f2809` majority_baseline: 0.1000

## Confound flags

- **input_resolution_differs**: models see different input resolutions; architecture and resolution are confounded. Values: `{"exp-20261027-201115-e384ed": 64, "exp-20261027-204957-d42310": 64, "exp-20261027-214915-e0d372": 224, "exp-20261028-004715-3da317": 224}`
- **pretrained_vs_scratch**: ImageNet-pretrained and scratch models are compared; pretraining is a confound. Values: `{"exp-20261027-201115-e384ed": "scratch", "exp-20261027-204957-d42310": "scratch", "exp-20261027-214915-e0d372": "imagenet", "exp-20261028-004715-3da317": "imagenet"}`
