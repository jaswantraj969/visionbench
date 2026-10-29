# Robustness (eurosat-c-v1) on VAL — VALIDATION (selection-biased) — not test results

**VALIDATION (selection-biased) — not test results.** Scope: synthetic corruptions at 3 fixed severities on one dataset (EuroSAT RGB, one split). The numbers describe sensitivity to THESE corruptions only; they do not show general robustness, robustness to real sensor degradation, or robustness to distribution shift. No model is called robust.

Suite `eurosat-c-v1` (robustness_suite.yaml sha256 `6d8d647f68021d21d6cb619f6e0cc637b6b3d97a5e20d25f3d3693addfcbfad2`), 12 conditions, applied to the 64x64 originals before each model's val transform. Clean numbers are the pinned B1 clean VAL evaluations; corrupted and clean predictions cover the same images. Drop = clean - corrupted balanced accuracy; relative drop = drop / clean. No reference-model normalization (this is not ImageNet-C's mCE). Mean ± sample sd over seeds 0/1/2.

## Summary per model (balanced accuracy)

| model | clean (mean ± sd) | mean over 12 conditions (mean ± sd) | drop (mean ± sd) | relative drop (mean ± sd) | per seed: clean / mean corrupted |
|---|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0.9871 ± 0.0027 | 0.4395 ± 0.0025 | 0.5476 ± 0.0014 | 0.5548 ± 0.0016 | s0 0.9840 / 0.4374; s1 0.9892 / 0.4423; s2 0.9880 / 0.4388 |
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | s0 0.1000 / 0.1000; s1 0.1000 / 0.1000; s2 0.1000 / 0.1000 |
| resnet18_imagenet_224 | 0.9862 ± 0.0035 | 0.5178 ± 0.0048 | 0.4684 ± 0.0076 | 0.4749 ± 0.0063 | s0 0.9869 / 0.5131; s1 0.9824 / 0.5227; s2 0.9892 / 0.5176 |
| resnet18_scratch_64 | 0.9125 ± 0.0485 | 0.5857 ± 0.0292 | 0.3268 ± 0.0240 | 0.3580 ± 0.0124 | s0 0.9300 / 0.6075; s1 0.8578 / 0.5525; s2 0.9499 / 0.5972 |
| small_cnn_scratch_64 | 0.9556 ± 0.0027 | 0.5532 ± 0.0042 | 0.4024 ± 0.0020 | 0.4211 ± 0.0029 | s0 0.9588 / 0.5578; s1 0.9542 / 0.5496; s2 0.9538 / 0.5523 |

## Mean corrupted balanced accuracy per corruption (over its 3 severities; mean ± sd over seeds)

| model | gaussian_noise | gaussian_blur | brightness | jpeg |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0.1282 ± 0.0147 | 0.4468 ± 0.0117 | 0.8740 ± 0.0026 | 0.3090 ± 0.0093 |
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 |
| resnet18_imagenet_224 | 0.1351 ± 0.0109 | 0.5668 ± 0.0330 | 0.8745 ± 0.0129 | 0.4948 ± 0.0365 |
| resnet18_scratch_64 | 0.3526 ± 0.0493 | 0.7488 ± 0.0671 | 0.5098 ± 0.0936 | 0.7317 ± 0.0141 |
| small_cnn_scratch_64 | 0.2178 ± 0.0108 | 0.7514 ± 0.0085 | 0.5929 ± 0.0081 | 0.6508 ± 0.0105 |

## Per condition: corrupted balanced accuracy and drop (mean ± sd over seeds)

| model | condition | corrupted bal. acc. | drop | per seed corrupted (s0 / s1 / s2) |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | gaussian_noise/0.04 | 0.2062 ± 0.0425 | 0.7808 ± 0.0435 | 0.2327 / 0.2288 / 0.1572 |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.08 | 0.1073 ± 0.0109 | 0.8797 ± 0.0135 | 0.1198 / 0.1024 / 0.0998 |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.12 | 0.0710 ± 0.0088 | 0.9161 ± 0.0098 | 0.0709 / 0.0622 / 0.0798 |
| efficientnet_b0_imagenet_224 | gaussian_blur/0.5 | 0.9455 ± 0.0070 | 0.0416 ± 0.0052 | 0.9382 / 0.9460 / 0.9522 |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.0 | 0.3070 ± 0.0353 | 0.6801 ± 0.0335 | 0.2738 / 0.3030 / 0.3441 |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.5 | 0.0878 ± 0.0075 | 0.8992 ± 0.0090 | 0.0931 / 0.0912 / 0.0792 |
| efficientnet_b0_imagenet_224 | brightness/0.1 | 0.9716 ± 0.0012 | 0.0155 ± 0.0030 | 0.9713 / 0.9705 / 0.9729 |
| efficientnet_b0_imagenet_224 | brightness/0.2 | 0.8954 ± 0.0043 | 0.0916 ± 0.0065 | 0.8997 / 0.8955 / 0.8911 |
| efficientnet_b0_imagenet_224 | brightness/0.3 | 0.7549 ± 0.0061 | 0.2322 ± 0.0059 | 0.7542 / 0.7614 / 0.7491 |
| efficientnet_b0_imagenet_224 | jpeg/50 | 0.4792 ± 0.0167 | 0.5079 ± 0.0141 | 0.4617 / 0.4950 / 0.4809 |
| efficientnet_b0_imagenet_224 | jpeg/25 | 0.2918 ± 0.0084 | 0.6952 ± 0.0060 | 0.2824 / 0.2948 / 0.2983 |
| efficientnet_b0_imagenet_224 | jpeg/10 | 0.1559 ± 0.0049 | 0.8311 ± 0.0032 | 0.1508 / 0.1564 / 0.1606 |
| majority_baseline | gaussian_noise/0.04 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | gaussian_noise/0.08 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | gaussian_noise/0.12 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | gaussian_blur/0.5 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | gaussian_blur/1.0 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | gaussian_blur/1.5 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | brightness/0.1 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | brightness/0.2 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | brightness/0.3 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | jpeg/50 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | jpeg/25 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| majority_baseline | jpeg/10 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 / 0.1000 / 0.1000 |
| resnet18_imagenet_224 | gaussian_noise/0.04 | 0.1879 ± 0.0196 | 0.7983 ± 0.0172 | 0.2054 / 0.1667 / 0.1916 |
| resnet18_imagenet_224 | gaussian_noise/0.08 | 0.1134 ± 0.0098 | 0.8728 ± 0.0067 | 0.1190 / 0.1021 / 0.1192 |
| resnet18_imagenet_224 | gaussian_noise/0.12 | 0.1040 ± 0.0036 | 0.8822 ± 0.0024 | 0.1071 / 0.1000 / 0.1048 |
| resnet18_imagenet_224 | gaussian_blur/0.5 | 0.9656 ± 0.0086 | 0.0206 ± 0.0052 | 0.9687 / 0.9559 / 0.9721 |
| resnet18_imagenet_224 | gaussian_blur/1.0 | 0.5060 ± 0.0816 | 0.4801 ± 0.0818 | 0.4127 / 0.5415 / 0.5640 |
| resnet18_imagenet_224 | gaussian_blur/1.5 | 0.2288 ± 0.0186 | 0.7574 ± 0.0193 | 0.2073 / 0.2383 / 0.2407 |
| resnet18_imagenet_224 | brightness/0.1 | 0.9664 ± 0.0055 | 0.0198 ± 0.0046 | 0.9722 / 0.9612 / 0.9658 |
| resnet18_imagenet_224 | brightness/0.2 | 0.8928 ± 0.0104 | 0.0933 ± 0.0080 | 0.9019 / 0.8816 / 0.8950 |
| resnet18_imagenet_224 | brightness/0.3 | 0.7643 ± 0.0257 | 0.2218 ± 0.0223 | 0.7700 / 0.7362 / 0.7868 |
| resnet18_imagenet_224 | jpeg/50 | 0.6463 ± 0.0129 | 0.3399 ± 0.0164 | 0.6439 / 0.6603 / 0.6347 |
| resnet18_imagenet_224 | jpeg/25 | 0.4982 ± 0.0526 | 0.4880 ± 0.0561 | 0.4938 / 0.5529 / 0.4479 |
| resnet18_imagenet_224 | jpeg/10 | 0.3399 ± 0.0459 | 0.6462 ± 0.0490 | 0.3552 / 0.3763 / 0.2884 |
| resnet18_scratch_64 | gaussian_noise/0.04 | 0.5744 ± 0.1052 | 0.3381 ± 0.1457 | 0.6195 / 0.6495 / 0.4542 |
| resnet18_scratch_64 | gaussian_noise/0.08 | 0.2872 ± 0.0362 | 0.6253 ± 0.0795 | 0.3031 / 0.3128 / 0.2458 |
| resnet18_scratch_64 | gaussian_noise/0.12 | 0.1960 ± 0.0139 | 0.7165 ± 0.0470 | 0.2121 / 0.1889 / 0.1871 |
| resnet18_scratch_64 | gaussian_blur/0.5 | 0.9106 ± 0.0563 | 0.0019 ± 0.0079 | 0.9299 / 0.8472 / 0.9547 |
| resnet18_scratch_64 | gaussian_blur/1.0 | 0.7539 ± 0.0908 | 0.1587 ± 0.0437 | 0.8024 / 0.6491 / 0.8102 |
| resnet18_scratch_64 | gaussian_blur/1.5 | 0.5819 ± 0.0645 | 0.3306 ± 0.0447 | 0.6476 / 0.5187 / 0.5794 |
| resnet18_scratch_64 | brightness/0.1 | 0.7294 ± 0.0853 | 0.1831 ± 0.0369 | 0.7609 / 0.6328 / 0.7946 |
| resnet18_scratch_64 | brightness/0.2 | 0.4718 ± 0.1028 | 0.4407 ± 0.0567 | 0.4822 / 0.3642 / 0.5690 |
| resnet18_scratch_64 | brightness/0.3 | 0.3282 ± 0.0944 | 0.5844 ± 0.0491 | 0.3347 / 0.2307 / 0.4191 |
| resnet18_scratch_64 | jpeg/50 | 0.8379 ± 0.0222 | 0.0746 ± 0.0263 | 0.8452 / 0.8130 / 0.8556 |
| resnet18_scratch_64 | jpeg/25 | 0.7664 ± 0.0104 | 0.1461 ± 0.0581 | 0.7678 / 0.7761 / 0.7554 |
| resnet18_scratch_64 | jpeg/10 | 0.5909 ± 0.0533 | 0.3217 ± 0.1012 | 0.5850 / 0.6469 / 0.5407 |
| small_cnn_scratch_64 | gaussian_noise/0.04 | 0.3590 ± 0.0574 | 0.5966 ± 0.0600 | 0.2948 / 0.4054 / 0.3769 |
| small_cnn_scratch_64 | gaussian_noise/0.08 | 0.1690 ± 0.0291 | 0.7866 ± 0.0290 | 0.1739 / 0.1378 / 0.1954 |
| small_cnn_scratch_64 | gaussian_noise/0.12 | 0.1254 ± 0.0277 | 0.8303 ± 0.0251 | 0.1565 / 0.1033 / 0.1163 |
| small_cnn_scratch_64 | gaussian_blur/0.5 | 0.9653 ± 0.0011 | -0.0096 ± 0.0018 | 0.9663 / 0.9652 / 0.9642 |
| small_cnn_scratch_64 | gaussian_blur/1.0 | 0.7571 ± 0.0114 | 0.1985 ± 0.0094 | 0.7683 / 0.7455 / 0.7575 |
| small_cnn_scratch_64 | gaussian_blur/1.5 | 0.5320 ± 0.0139 | 0.4237 ± 0.0124 | 0.5427 / 0.5163 / 0.5369 |
| small_cnn_scratch_64 | brightness/0.1 | 0.7928 ± 0.0046 | 0.1629 ± 0.0073 | 0.7877 / 0.7941 / 0.7965 |
| small_cnn_scratch_64 | brightness/0.2 | 0.5465 ± 0.0177 | 0.4091 ± 0.0150 | 0.5664 / 0.5407 / 0.5324 |
| small_cnn_scratch_64 | brightness/0.3 | 0.4393 ± 0.0117 | 0.5163 ± 0.0095 | 0.4503 / 0.4406 / 0.4270 |
| small_cnn_scratch_64 | jpeg/50 | 0.8028 ± 0.0157 | 0.1528 ± 0.0132 | 0.8190 / 0.8017 / 0.7877 |
| small_cnn_scratch_64 | jpeg/25 | 0.6903 ± 0.0080 | 0.2653 ± 0.0055 | 0.6985 / 0.6899 / 0.6825 |
| small_cnn_scratch_64 | jpeg/10 | 0.4594 ± 0.0084 | 0.4962 ± 0.0057 | 0.4691 / 0.4543 / 0.4548 |

## Seed 0: paired difference corrupted - clean [95 % paired bootstrap CI]

Paired percentile bootstrap, 1000 resamples, seed 0, the same image indices for clean and corrupted predictions. ✓ = the CI excludes zero (the change is distinguishable from zero for this run, for this corruption only).

| model | condition | Δ balanced accuracy | Δ macro F1 | CI excludes 0 (bal. acc.) |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | gaussian_noise/0.04 | -0.7513 [-0.7604, -0.7425] | -0.7900 [-0.8009, -0.7796] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.08 | -0.8642 [-0.8696, -0.8587] | -0.9328 [-0.9399, -0.9256] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.12 | -0.9131 [-0.9203, -0.9054] | -0.9471 [-0.9527, -0.9411] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/0.5 | -0.0458 [-0.0526, -0.0392] | -0.0444 [-0.0514, -0.0377] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.0 | -0.7102 [-0.7201, -0.6994] | -0.7157 [-0.7266, -0.7041] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.5 | -0.8909 [-0.8985, -0.8820] | -0.9046 [-0.9131, -0.8954] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.1 | -0.0127 [-0.0171, -0.0084] | -0.0139 [-0.0186, -0.0092] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.2 | -0.0843 [-0.0924, -0.0757] | -0.0921 [-0.1016, -0.0827] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.3 | -0.2298 [-0.2409, -0.2190] | -0.2472 [-0.2594, -0.2354] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/50 | -0.5223 [-0.5349, -0.5103] | -0.5738 [-0.5873, -0.5614] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/25 | -0.7016 [-0.7133, -0.6908] | -0.7560 [-0.7686, -0.7445] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/10 | -0.8332 [-0.8436, -0.8222] | -0.8657 [-0.8755, -0.8559] | ✓ |
| majority_baseline | gaussian_noise/0.04 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | gaussian_noise/0.08 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | gaussian_noise/0.12 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | gaussian_blur/0.5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | gaussian_blur/1.0 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | gaussian_blur/1.5 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | brightness/0.1 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | brightness/0.2 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | brightness/0.3 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | jpeg/50 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | jpeg/25 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| majority_baseline | jpeg/10 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | — |
| resnet18_imagenet_224 | gaussian_noise/0.04 | -0.7814 [-0.7909, -0.7723] | -0.8032 [-0.8145, -0.7925] | ✓ |
| resnet18_imagenet_224 | gaussian_noise/0.08 | -0.8678 [-0.8732, -0.8622] | -0.9325 [-0.9401, -0.9244] | ✓ |
| resnet18_imagenet_224 | gaussian_noise/0.12 | -0.8798 [-0.8841, -0.8753] | -0.9538 [-0.9598, -0.9478] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/0.5 | -0.0181 [-0.0231, -0.0136] | -0.0175 [-0.0225, -0.0129] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/1.0 | -0.5742 [-0.5854, -0.5643] | -0.6180 [-0.6320, -0.6052] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/1.5 | -0.7796 [-0.7849, -0.7745] | -0.8820 [-0.8903, -0.8741] | ✓ |
| resnet18_imagenet_224 | brightness/0.1 | -0.0147 [-0.0197, -0.0104] | -0.0158 [-0.0211, -0.0114] | ✓ |
| resnet18_imagenet_224 | brightness/0.2 | -0.0849 [-0.0930, -0.0771] | -0.0937 [-0.1031, -0.0849] | ✓ |
| resnet18_imagenet_224 | brightness/0.3 | -0.2169 [-0.2268, -0.2067] | -0.2552 [-0.2675, -0.2431] | ✓ |
| resnet18_imagenet_224 | jpeg/50 | -0.3430 [-0.3548, -0.3318] | -0.3828 [-0.3956, -0.3700] | ✓ |
| resnet18_imagenet_224 | jpeg/25 | -0.4930 [-0.5057, -0.4802] | -0.5443 [-0.5579, -0.5303] | ✓ |
| resnet18_imagenet_224 | jpeg/10 | -0.6317 [-0.6411, -0.6221] | -0.7366 [-0.7469, -0.7259] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.04 | -0.3105 [-0.3232, -0.2980] | -0.3139 [-0.3268, -0.3010] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.08 | -0.6268 [-0.6389, -0.6156] | -0.6569 [-0.6716, -0.6433] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.12 | -0.7179 [-0.7273, -0.7075] | -0.7755 [-0.7861, -0.7640] | ✓ |
| resnet18_scratch_64 | gaussian_blur/0.5 | -0.0001 [-0.0049, +0.0048] | +0.0002 [-0.0047, +0.0049] | — |
| resnet18_scratch_64 | gaussian_blur/1.0 | -0.1276 [-0.1381, -0.1181] | -0.1401 [-0.1512, -0.1293] | ✓ |
| resnet18_scratch_64 | gaussian_blur/1.5 | -0.2823 [-0.2945, -0.2703] | -0.3034 [-0.3160, -0.2900] | ✓ |
| resnet18_scratch_64 | brightness/0.1 | -0.1690 [-0.1808, -0.1571] | -0.1888 [-0.2025, -0.1753] | ✓ |
| resnet18_scratch_64 | brightness/0.2 | -0.4478 [-0.4606, -0.4343] | -0.4709 [-0.4845, -0.4571] | ✓ |
| resnet18_scratch_64 | brightness/0.3 | -0.5952 [-0.6070, -0.5830] | -0.6470 [-0.6593, -0.6345] | ✓ |
| resnet18_scratch_64 | jpeg/50 | -0.0848 [-0.0937, -0.0756] | -0.0899 [-0.0998, -0.0801] | ✓ |
| resnet18_scratch_64 | jpeg/25 | -0.1622 [-0.1734, -0.1504] | -0.1746 [-0.1874, -0.1621] | ✓ |
| resnet18_scratch_64 | jpeg/10 | -0.3450 [-0.3585, -0.3312] | -0.3694 [-0.3839, -0.3534] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.04 | -0.6639 [-0.6754, -0.6522] | -0.6783 [-0.6924, -0.6652] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.08 | -0.7849 [-0.7939, -0.7765] | -0.8557 [-0.8643, -0.8473] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.12 | -0.8023 [-0.8111, -0.7942] | -0.8765 [-0.8855, -0.8676] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/0.5 | +0.0076 [+0.0033, +0.0120] | +0.0073 [+0.0031, +0.0116] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/1.0 | -0.1905 [-0.2000, -0.1801] | -0.2209 [-0.2324, -0.2085] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/1.5 | -0.4161 [-0.4276, -0.4049] | -0.4623 [-0.4746, -0.4499] | ✓ |
| small_cnn_scratch_64 | brightness/0.1 | -0.1711 [-0.1822, -0.1594] | -0.1792 [-0.1920, -0.1665] | ✓ |
| small_cnn_scratch_64 | brightness/0.2 | -0.3924 [-0.4039, -0.3806] | -0.4373 [-0.4507, -0.4228] | ✓ |
| small_cnn_scratch_64 | brightness/0.3 | -0.5084 [-0.5197, -0.4975] | -0.5851 [-0.5988, -0.5706] | ✓ |
| small_cnn_scratch_64 | jpeg/50 | -0.1398 [-0.1498, -0.1290] | -0.1512 [-0.1630, -0.1391] | ✓ |
| small_cnn_scratch_64 | jpeg/25 | -0.2603 [-0.2718, -0.2483] | -0.3063 [-0.3198, -0.2924] | ✓ |
| small_cnn_scratch_64 | jpeg/10 | -0.4896 [-0.5028, -0.4765] | -0.5523 [-0.5678, -0.5377] | ✓ |

## Seed 0: per-class recall in the worst condition per model

Worst condition = lowest corrupted balanced accuracy of the seed-0 run. Classes marked † are largely low-information (near-uniform tiles; DATASETS.md).

**efficientnet_b0_imagenet_224** (seed 0, exp-20261028-004715-3da317): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9840 clean → 0.0709.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9822 | 0.2311 | +0.7511 |
| Forest † (34 % low-information) | 0.9911 | 0.0000 | +0.9911 |
| HerbaceousVegetation | 0.9911 | 0.0000 | +0.9911 |
| Highway | 0.9813 | 0.0000 | +0.9813 |
| Industrial | 1.0000 | 0.0133 | +0.9867 |
| Pasture | 0.9733 | 0.0000 | +0.9733 |
| PermanentCrop | 0.9573 | 0.0000 | +0.9573 |
| Residential | 0.9889 | 0.0000 | +0.9889 |
| River | 0.9813 | 0.1093 | +0.8720 |
| SeaLake † (78 % low-information) | 0.9933 | 0.3556 | +0.6378 |

**majority_baseline** (seed 0, exp-20261027-201053-5f2809): worst condition `gaussian_noise/0.04`, balanced accuracy 0.1000 clean → 0.1000.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.0000 | 0.0000 | +0.0000 |
| Forest † (34 % low-information) | 1.0000 | 1.0000 | +0.0000 |
| HerbaceousVegetation | 0.0000 | 0.0000 | +0.0000 |
| Highway | 0.0000 | 0.0000 | +0.0000 |
| Industrial | 0.0000 | 0.0000 | +0.0000 |
| Pasture | 0.0000 | 0.0000 | +0.0000 |
| PermanentCrop | 0.0000 | 0.0000 | +0.0000 |
| Residential | 0.0000 | 0.0000 | +0.0000 |
| River | 0.0000 | 0.0000 | +0.0000 |
| SeaLake † (78 % low-information) | 0.0000 | 0.0000 | +0.0000 |

**resnet18_imagenet_224** (seed 0, exp-20261027-214915-e0d372): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9869 clean → 0.1071.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9778 | 0.0044 | +0.9733 |
| Forest † (34 % low-information) | 0.9956 | 0.0000 | +0.9956 |
| HerbaceousVegetation | 0.9933 | 0.0000 | +0.9933 |
| Highway | 0.9893 | 0.0000 | +0.9893 |
| Industrial | 0.9920 | 0.0667 | +0.9253 |
| Pasture | 0.9700 | 0.0000 | +0.9700 |
| PermanentCrop | 0.9733 | 0.0000 | +0.9733 |
| Residential | 0.9933 | 1.0000 | -0.0067 |
| River | 0.9840 | 0.0000 | +0.9840 |
| SeaLake † (78 % low-information) | 1.0000 | 0.0000 | +1.0000 |

**resnet18_scratch_64** (seed 0, exp-20261027-201115-e384ed): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9300 clean → 0.2121.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.8889 | 0.0133 | +0.8756 |
| Forest † (34 % low-information) | 0.9822 | 0.0000 | +0.9822 |
| HerbaceousVegetation | 0.9222 | 0.0089 | +0.9133 |
| Highway | 0.9360 | 0.2000 | +0.7360 |
| Industrial | 0.9520 | 0.7973 | +0.1547 |
| Pasture | 0.9067 | 0.0100 | +0.8967 |
| PermanentCrop | 0.8773 | 0.0160 | +0.8613 |
| Residential | 0.9933 | 0.9978 | -0.0044 |
| River | 0.8853 | 0.0773 | +0.8080 |
| SeaLake † (78 % low-information) | 0.9556 | 0.0000 | +0.9556 |

**small_cnn_scratch_64** (seed 0, exp-20261027-204957-d42310): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9588 clean → 0.1565.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9356 | 0.0000 | +0.9356 |
| Forest † (34 % low-information) | 0.9911 | 0.0000 | +0.9911 |
| HerbaceousVegetation | 0.9444 | 0.0000 | +0.9444 |
| Highway | 0.9547 | 0.0000 | +0.9547 |
| Industrial | 0.9413 | 0.2160 | +0.7253 |
| Pasture | 0.9233 | 0.0000 | +0.9233 |
| PermanentCrop | 0.9707 | 0.0000 | +0.9707 |
| Residential | 1.0000 | 0.9911 | +0.0089 |
| River | 0.9467 | 0.0000 | +0.9467 |
| SeaLake † (78 % low-information) | 0.9800 | 0.3578 | +0.6222 |

## Pipeline check: majority baseline

The majority baseline does not look at the image, so every corrupted metric must equal its clean metric exactly: **equal for all 36 run x condition pairs**.

## Confounds

- **resolution_upsampling**: Corruptions are applied at 64 px. Pretrained models see an upsampled version (resize 256, centre crop 224) in which corruption and content are scaled together (a blur of 1 px becomes about 4 px); scratch models see both at native resolution. This is not a smoothing of the corruption relative to the content (robustness protocol, amendment 1); pretraining and resolution stay confounded, so group differences have no attributed cause.
- **pretrained_vs_scratch**: ImageNet pretraining and input resolution are confounded with each other.
- **centre_crop**: Pretrained models never see the 4 px border of a tile.
- **single_split**: One fixed split: no split-to-split variance.
- **three_seeds**: n = 3 seeds: the sd is itself uncertain; per-seed values are shown.
- **selection_bias**: VAL chose each checkpoint; clean and corrupted VAL numbers share that bias.

## Runs

| model | seed | experiment | clean evaluation (pinned) |
|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0 | exp-20261028-004715-3da317 | ev-20261028-041610-12527c |
| efficientnet_b0_imagenet_224 | 1 | exp-20261028-013827-6cbab1 | ev-20261028-041651-df21cd |
| efficientnet_b0_imagenet_224 | 2 | exp-20261028-030154-fe104a | ev-20261028-041731-f7b236 |
| majority_baseline | 0 | exp-20261027-201053-5f2809 | ev-20261028-041748-ef337a |
| majority_baseline | 1 | exp-20261027-201101-632b02 | ev-20261028-041804-166ee7 |
| majority_baseline | 2 | exp-20261027-201108-703436 | ev-20261028-041820-d87583 |
| resnet18_imagenet_224 | 0 | exp-20261027-214915-e0d372 | ev-20261028-041854-e1bd88 |
| resnet18_imagenet_224 | 1 | exp-20261027-225711-6db256 | ev-20261028-041929-b260a5 |
| resnet18_imagenet_224 | 2 | exp-20261027-233452-43080e | ev-20261028-042003-26e664 |
| resnet18_scratch_64 | 0 | exp-20261027-201115-e384ed | ev-20261028-042026-b5cb93 |
| resnet18_scratch_64 | 1 | exp-20261027-202124-e89a98 | ev-20261028-042048-83be42 |
| resnet18_scratch_64 | 2 | exp-20261027-202749-e42275 | ev-20261028-042111-ebe183 |
| small_cnn_scratch_64 | 0 | exp-20261027-204957-d42310 | ev-20261028-042134-cf3fb5 |
| small_cnn_scratch_64 | 1 | exp-20261027-211106-cc7c83 | ev-20261028-042156-27fe9e |
| small_cnn_scratch_64 | 2 | exp-20261027-213240-d42c79 | ev-20261028-042217-2ae5cf |
