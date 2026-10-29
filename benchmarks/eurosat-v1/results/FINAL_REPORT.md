# Final report: eurosat-v1 — TEST

**TEST.** One clean TEST evaluation per run (15 runs), following `FINAL_ANALYSIS_PLAN.md` (sha256 `ff41ad85e0b95278805ad2e6653bf7c7a065d83d23b6fb622e72994bb2258a79`). The checkpoints were chosen on VALIDATION before any TEST access.

Balanced accuracy is primary, macro F1 secondary. Per-run CIs: 95 % percentile bootstrap, 1000 resamples, seed 0. Mean ± sample sd over seeds 0/1/2 (the split is fixed, so the sd reflects training variance only).

## Per model (mean ± sd over 3 seeds)

| model | balanced accuracy (TEST) | macro F1 (TEST) | per-seed balanced accuracy (TEST) |
|---|---|---|---|
| majority_baseline | 0.1000 ± 0.0000 | 0.0200 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| resnet18_scratch_64 | 0.9133 ± 0.0457 | 0.9137 ± 0.0455 | 0.9241 / 0.8632 / 0.9526 |
| small_cnn_scratch_64 | 0.9561 ± 0.0027 | 0.9570 ± 0.0027 | 0.9577 / 0.9530 / 0.9577 |
| resnet18_imagenet_224 | 0.9836 ± 0.0033 | 0.9839 ± 0.0032 | 0.9844 / 0.9800 / 0.9864 |
| efficientnet_b0_imagenet_224 | 0.9844 ± 0.0014 | 0.9846 ± 0.0015 | 0.9844 / 0.9829 / 0.9858 |

## Per run

| model | seed | experiment | evaluation | n | balanced accuracy [95 % CI] (TEST) | macro F1 [95 % CI] (TEST) |
|---|---|---|---|---|---|---|
| majority_baseline | 0 | exp-20261027-201053-5f2809 | ev-20261029-112504-ec73cf | 4050 | 0.1000 [0.1000, 0.1000] | 0.0200 [0.0184, 0.0215] |
| majority_baseline | 1 | exp-20261027-201101-632b02 | ev-20261029-112523-9b58fd | 4050 | 0.1000 [0.1000, 0.1000] | 0.0200 [0.0184, 0.0215] |
| majority_baseline | 2 | exp-20261027-201108-703436 | ev-20261029-112543-28d2fa | 4050 | 0.1000 [0.1000, 0.1000] | 0.0200 [0.0184, 0.0215] |
| resnet18_scratch_64 | 0 | exp-20261027-201115-e384ed | ev-20261029-112608-c4d668 | 4050 | 0.9241 [0.9157, 0.9318] | 0.9247 [0.9162, 0.9323] |
| resnet18_scratch_64 | 1 | exp-20261027-202124-e89a98 | ev-20261029-112633-e4bace | 4050 | 0.8632 [0.8533, 0.8736] | 0.8638 [0.8533, 0.8740] |
| resnet18_scratch_64 | 2 | exp-20261027-202749-e42275 | ev-20261029-112658-df5146 | 4050 | 0.9526 [0.9454, 0.9592] | 0.9527 [0.9456, 0.9592] |
| small_cnn_scratch_64 | 0 | exp-20261027-204957-d42310 | ev-20261029-112722-d17c6a | 4050 | 0.9577 [0.9512, 0.9644] | 0.9586 [0.9524, 0.9648] |
| small_cnn_scratch_64 | 1 | exp-20261027-211106-cc7c83 | ev-20261029-112747-79a389 | 4050 | 0.9530 [0.9464, 0.9597] | 0.9539 [0.9475, 0.9604] |
| small_cnn_scratch_64 | 2 | exp-20261027-213240-d42c79 | ev-20261029-112811-931ebd | 4050 | 0.9577 [0.9511, 0.9641] | 0.9586 [0.9520, 0.9650] |
| resnet18_imagenet_224 | 0 | exp-20261027-214915-e0d372 | ev-20261029-112850-a00cad | 4050 | 0.9844 [0.9800, 0.9881] | 0.9848 [0.9807, 0.9885] |
| resnet18_imagenet_224 | 1 | exp-20261027-225711-6db256 | ev-20261029-112930-d32a72 | 4050 | 0.9800 [0.9754, 0.9840] | 0.9804 [0.9758, 0.9844] |
| resnet18_imagenet_224 | 2 | exp-20261027-233452-43080e | ev-20261029-113009-789459 | 4050 | 0.9864 [0.9824, 0.9899] | 0.9866 [0.9825, 0.9900] |
| efficientnet_b0_imagenet_224 | 0 | exp-20261028-004715-3da317 | ev-20261029-113052-84ed06 | 4050 | 0.9844 [0.9804, 0.9882] | 0.9848 [0.9808, 0.9882] |
| efficientnet_b0_imagenet_224 | 1 | exp-20261028-013827-6cbab1 | ev-20261029-113134-691d5f | 4050 | 0.9829 [0.9787, 0.9868] | 0.9830 [0.9789, 0.9869] |
| efficientnet_b0_imagenet_224 | 2 | exp-20261028-030154-fe104a | ev-20261029-113216-4c6760 | 4050 | 0.9858 [0.9820, 0.9893] | 0.9860 [0.9822, 0.9895] |

## Ranking by 3-seed mean balanced accuracy, with rank groups

Neighbours are in different rank groups only if all three seed-matched paired CIs of the balanced-accuracy difference exclude zero in the same direction. Rank groups are descriptive; only the three pre-registered comparisons below are confirmatory.

| rank | model | mean balanced accuracy | rank group |
|---|---|---|---|
| 1 | efficientnet_b0_imagenet_224 | 0.9844 | 1 |
| 2 | resnet18_imagenet_224 | 0.9836 | 1 |
| 3 | small_cnn_scratch_64 | 0.9561 | 2 |
| 4 | resnet18_scratch_64 | 0.9133 | 2 |
| 5 | majority_baseline | 0.1000 | 3 |

| higher | lower | Δ seed 0 | Δ seed 1 | Δ seed 2 | separated |
|---|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | resnet18_imagenet_224 | -0.0000 [-0.0041, +0.0039] | +0.0029 [-0.0010, +0.0071] | -0.0007 [-0.0042, +0.0030] | no |
| resnet18_imagenet_224 | small_cnn_scratch_64 | +0.0267 [+0.0201, +0.0330] | +0.0270 [+0.0204, +0.0346] | +0.0287 [+0.0217, +0.0350] | yes |
| small_cnn_scratch_64 | resnet18_scratch_64 | +0.0336 [+0.0262, +0.0412] | +0.0898 [+0.0793, +0.0999] | +0.0051 [-0.0014, +0.0122] | no |
| resnet18_scratch_64 | majority_baseline | +0.8241 [+0.8157, +0.8318] | +0.7632 [+0.7533, +0.7736] | +0.8526 [+0.8454, +0.8592] | yes |

## Pre-registered comparisons (seed-matched, A - B)

Paired percentile bootstrap of A - B on the same images (1000 resamples, seed 0, shared indices), one per seed. **Distinguishable** only if all three CIs exclude zero in the same direction.

| # | A - B | metric | Δ seed 0 | Δ seed 1 | Δ seed 2 | verdict |
|---|---|---|---|---|---|---|
| 1 | efficientnet_b0_imagenet_224 - resnet18_imagenet_224 | balanced_accuracy | -0.0000 [-0.0041, +0.0039] | +0.0029 [-0.0010, +0.0071] | -0.0007 [-0.0042, +0.0030] | not distinguishable |
| 1 | efficientnet_b0_imagenet_224 - resnet18_imagenet_224 | macro_f1 | -0.0000 [-0.0039, +0.0039] | +0.0027 [-0.0011, +0.0067] | -0.0006 [-0.0041, +0.0032] | not distinguishable |
| 2 | small_cnn_scratch_64 - resnet18_scratch_64 | balanced_accuracy | +0.0336 [+0.0262, +0.0412] | +0.0898 [+0.0793, +0.0999] | +0.0051 [-0.0014, +0.0122] | not distinguishable |
| 2 | small_cnn_scratch_64 - resnet18_scratch_64 | macro_f1 | +0.0339 [+0.0265, +0.0416] | +0.0901 [+0.0797, +0.1003] | +0.0059 [-0.0005, +0.0128] | not distinguishable |
| 3 | resnet18_imagenet_224 - resnet18_scratch_64 | balanced_accuracy | +0.0603 [+0.0524, +0.0687] | +0.1168 [+0.1058, +0.1273] | +0.0338 [+0.0268, +0.0410] | distinguishable: resnet18_imagenet_224 > resnet18_scratch_64 |
| 3 | resnet18_imagenet_224 - resnet18_scratch_64 | macro_f1 | +0.0601 [+0.0522, +0.0684] | +0.1166 [+0.1057, +0.1271] | +0.0339 [+0.0270, +0.0412] | distinguishable: resnet18_imagenet_224 > resnet18_scratch_64 |

## TEST next to VALIDATION (selection-biased)

| model | balanced accuracy: VALIDATION | balanced accuracy: TEST | macro F1: VALIDATION | macro F1: TEST |
|---|---|---|---|---|
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.0200 ± 0.0000 | 0.0200 ± 0.0000 |
| resnet18_scratch_64 | 0.9125 ± 0.0485 | 0.9133 ± 0.0457 | 0.9127 ± 0.0488 | 0.9137 ± 0.0455 |
| small_cnn_scratch_64 | 0.9556 ± 0.0027 | 0.9561 ± 0.0027 | 0.9563 ± 0.0029 | 0.9570 ± 0.0027 |
| resnet18_imagenet_224 | 0.9862 ± 0.0035 | 0.9836 ± 0.0033 | 0.9864 ± 0.0036 | 0.9839 ± 0.0032 |
| efficientnet_b0_imagenet_224 | 0.9871 ± 0.0027 | 0.9844 ± 0.0014 | 0.9873 ± 0.0026 | 0.9846 ± 0.0015 |

## Robustness (eurosat-c-v1) on TEST — TEST

Full tables: `results/robustness_test.md`. Mean balanced accuracy over the 12 conditions; drop = clean - corrupted (mean ± sd over seeds).

| model | clean | mean over 12 conditions | drop | relative drop |
|---|---|---|---|---|
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| resnet18_scratch_64 | 0.9133 ± 0.0457 | 0.5869 ± 0.0271 | 0.3264 ± 0.0239 | 0.3572 ± 0.0131 |
| small_cnn_scratch_64 | 0.9561 ± 0.0027 | 0.5545 ± 0.0030 | 0.4016 ± 0.0018 | 0.4201 ± 0.0021 |
| resnet18_imagenet_224 | 0.9836 ± 0.0033 | 0.5134 ± 0.0052 | 0.4702 ± 0.0074 | 0.4780 ± 0.0064 |
| efficientnet_b0_imagenet_224 | 0.9844 ± 0.0014 | 0.4372 ± 0.0022 | 0.5472 ± 0.0032 | 0.5559 ± 0.0026 |

## Interpretation constraints (fixed in the plan)

- **epoch_cap**: Every trainable run had a fixed epoch cap (25) and early stopping on validation loss; several runs stopped at the cap. The numbers describe these training budgets, not each architecture's best achievable accuracy.
- **untuned_learning_rates**: Learning rates and other hyperparameters were fixed in the protocol, not tuned per model. A different setting could change any ranking.
- **single_split**: One fixed train/val/test split of one dataset: no split-to-split variance is included, and nothing here generalises beyond EuroSAT RGB.
- **three_seeds**: n = 3 seeds per model: the seed sd is itself very uncertain, and the seed-matched comparison rule is strict by design.
- **low_information_classes**: Many SeaLake (about 78 %) and Forest (about 34 %) tiles are near-uniform; their near-duplicates cannot be detected by hashing, so leakage for those classes is bounded less tightly (DATASETS.md).
- **resolution_pretraining_confound**: The two ImageNet-pretrained models use 224 px inputs (upsampled from 64 px) and the scratch models 64 px: pretraining and input resolution are confounded, so a difference between the groups has no attributed cause.
- **corruption_scaling**: Corruptions are applied at 64 px before each model's transform; for the 224 px models the corruption is upsampled with the content (a 1 px blur becomes about 4 px; robustness protocol amendment 1). This scales corruption and content together; it is not a smoothing of the corruption relative to the content.
- **amendment_1**: Final analysis plan amendment 1 (procedural): after the first TEST materialization pass collided with the VAL cache, the corrupted-image cache path gained the split name. The corruption bytes, the evaluation logic and this analysis are unchanged; no robustness TEST result existed at the time, and the 15 clean TEST evaluations were not redone.
- **no_robustness_claim**: The robustness numbers describe sensitivity to 4 synthetic corruptions at 3 fixed severities only. No model is called robust.

No model is called robust or state-of-the-art. A difference is called distinguishable only under the rule above.
