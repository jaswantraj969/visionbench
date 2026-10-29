# Robustness (eurosat-c-v1) on TEST — TEST

**TEST.** Scope: synthetic corruptions at 3 fixed severities on one dataset (EuroSAT RGB, one split). The numbers describe sensitivity to THESE corruptions only; they do not show general robustness, robustness to real sensor degradation, or robustness to distribution shift. No model is called robust.

Suite `eurosat-c-v1` (robustness_suite.yaml sha256 `6d8d647f68021d21d6cb619f6e0cc637b6b3d97a5e20d25f3d3693addfcbfad2`), 12 conditions, applied to the 64x64 originals before each model's val transform. Clean numbers are the single clean TEST evaluation of each run; corrupted and clean predictions cover the same images. Drop = clean - corrupted balanced accuracy; relative drop = drop / clean. No reference-model normalization (this is not ImageNet-C's mCE). Mean ± sample sd over seeds 0/1/2.

## Summary per model (balanced accuracy)

| model | clean (mean ± sd) | mean over 12 conditions (mean ± sd) | drop (mean ± sd) | relative drop (mean ± sd) | per seed: clean / mean corrupted |
|---|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0.9844 ± 0.0014 | 0.4372 ± 0.0022 | 0.5472 ± 0.0032 | 0.5559 ± 0.0026 | s0 0.9844 / 0.4351; s1 0.9829 / 0.4395; s2 0.9858 / 0.4370 |
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | s0 0.1000 / 0.1000; s1 0.1000 / 0.1000; s2 0.1000 / 0.1000 |
| resnet18_imagenet_224 | 0.9836 ± 0.0033 | 0.5134 ± 0.0052 | 0.4702 ± 0.0074 | 0.4780 ± 0.0064 | s0 0.9844 / 0.5077; s1 0.9800 / 0.5179; s2 0.9864 / 0.5146 |
| resnet18_scratch_64 | 0.9133 ± 0.0457 | 0.5869 ± 0.0271 | 0.3264 ± 0.0239 | 0.3572 ± 0.0131 | s0 0.9241 / 0.6056; s1 0.8632 / 0.5558; s2 0.9526 / 0.5993 |
| small_cnn_scratch_64 | 0.9561 ± 0.0027 | 0.5545 ± 0.0030 | 0.4016 ± 0.0018 | 0.4201 ± 0.0021 | s0 0.9577 / 0.5577; s1 0.9530 / 0.5517; s2 0.9577 / 0.5542 |

## Mean corrupted balanced accuracy per corruption (over its 3 severities; mean ± sd over seeds)

| model | gaussian_noise | gaussian_blur | brightness | jpeg |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0.1253 ± 0.0165 | 0.4486 ± 0.0112 | 0.8680 ± 0.0024 | 0.3069 ± 0.0135 |
| majority_baseline | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 |
| resnet18_imagenet_224 | 0.1322 ± 0.0116 | 0.5626 ± 0.0307 | 0.8676 ± 0.0119 | 0.4912 ± 0.0371 |
| resnet18_scratch_64 | 0.3483 ± 0.0449 | 0.7540 ± 0.0637 | 0.5156 ± 0.0906 | 0.7297 ± 0.0136 |
| small_cnn_scratch_64 | 0.2200 ± 0.0105 | 0.7540 ± 0.0076 | 0.5926 ± 0.0060 | 0.6514 ± 0.0093 |

## Per condition: corrupted balanced accuracy and drop (mean ± sd over seeds)

| model | condition | corrupted bal. acc. | drop | per seed corrupted (s0 / s1 / s2) |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | gaussian_noise/0.04 | 0.1953 ± 0.0432 | 0.7891 ± 0.0442 | 0.2274 / 0.2122 / 0.1462 |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.08 | 0.1063 ± 0.0088 | 0.8780 ± 0.0090 | 0.1164 / 0.1021 / 0.1004 |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.12 | 0.0744 ± 0.0100 | 0.9100 ± 0.0088 | 0.0809 / 0.0629 / 0.0793 |
| efficientnet_b0_imagenet_224 | gaussian_blur/0.5 | 0.9411 ± 0.0083 | 0.0432 ± 0.0081 | 0.9319 / 0.9433 / 0.9481 |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.0 | 0.3135 ± 0.0283 | 0.6709 ± 0.0275 | 0.2871 / 0.3099 / 0.3434 |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.5 | 0.0911 ± 0.0026 | 0.8932 ± 0.0034 | 0.0939 / 0.0908 / 0.0887 |
| efficientnet_b0_imagenet_224 | brightness/0.1 | 0.9680 ± 0.0017 | 0.0163 ± 0.0003 | 0.9681 / 0.9663 / 0.9697 |
| efficientnet_b0_imagenet_224 | brightness/0.2 | 0.8874 ± 0.0015 | 0.0970 ± 0.0017 | 0.8890 / 0.8861 / 0.8871 |
| efficientnet_b0_imagenet_224 | brightness/0.3 | 0.7485 ± 0.0081 | 0.2359 ± 0.0093 | 0.7530 / 0.7534 / 0.7392 |
| efficientnet_b0_imagenet_224 | jpeg/50 | 0.4700 ± 0.0187 | 0.5144 ± 0.0192 | 0.4492 / 0.4856 / 0.4751 |
| efficientnet_b0_imagenet_224 | jpeg/25 | 0.2921 ± 0.0143 | 0.6922 ± 0.0148 | 0.2763 / 0.3041 / 0.2960 |
| efficientnet_b0_imagenet_224 | jpeg/10 | 0.1585 ± 0.0109 | 0.8259 ± 0.0102 | 0.1484 / 0.1569 / 0.1702 |
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
| resnet18_imagenet_224 | gaussian_noise/0.04 | 0.1828 ± 0.0240 | 0.8009 ± 0.0213 | 0.2014 / 0.1556 / 0.1912 |
| resnet18_imagenet_224 | gaussian_noise/0.08 | 0.1117 ± 0.0090 | 0.8719 ± 0.0060 | 0.1168 / 0.1013 / 0.1171 |
| resnet18_imagenet_224 | gaussian_noise/0.12 | 0.1022 ± 0.0020 | 0.8814 ± 0.0013 | 0.1029 / 0.1000 / 0.1037 |
| resnet18_imagenet_224 | gaussian_blur/0.5 | 0.9644 ± 0.0097 | 0.0193 ± 0.0064 | 0.9680 / 0.9534 / 0.9717 |
| resnet18_imagenet_224 | gaussian_blur/1.0 | 0.4995 ± 0.0778 | 0.4841 ± 0.0779 | 0.4115 / 0.5280 / 0.5590 |
| resnet18_imagenet_224 | gaussian_blur/1.5 | 0.2239 ± 0.0150 | 0.7598 ± 0.0147 | 0.2079 / 0.2260 / 0.2377 |
| resnet18_imagenet_224 | brightness/0.1 | 0.9627 ± 0.0037 | 0.0209 ± 0.0032 | 0.9666 / 0.9593 / 0.9622 |
| resnet18_imagenet_224 | brightness/0.2 | 0.8841 ± 0.0110 | 0.0996 ± 0.0082 | 0.8921 / 0.8716 / 0.8885 |
| resnet18_imagenet_224 | brightness/0.3 | 0.7560 ± 0.0236 | 0.2276 ± 0.0203 | 0.7591 / 0.7310 / 0.7779 |
| resnet18_imagenet_224 | jpeg/50 | 0.6379 ± 0.0141 | 0.3458 ± 0.0173 | 0.6309 / 0.6542 / 0.6286 |
| resnet18_imagenet_224 | jpeg/25 | 0.4895 ± 0.0550 | 0.4942 ± 0.0583 | 0.4787 / 0.5491 / 0.4406 |
| resnet18_imagenet_224 | jpeg/10 | 0.3463 ± 0.0450 | 0.6373 ± 0.0481 | 0.3564 / 0.3855 / 0.2971 |
| resnet18_scratch_64 | gaussian_noise/0.04 | 0.5649 ± 0.0984 | 0.3484 ± 0.1400 | 0.5994 / 0.6413 / 0.4539 |
| resnet18_scratch_64 | gaussian_noise/0.08 | 0.2836 ± 0.0357 | 0.6298 ± 0.0794 | 0.2934 / 0.3133 / 0.2440 |
| resnet18_scratch_64 | gaussian_noise/0.12 | 0.1966 ± 0.0115 | 0.7168 ± 0.0415 | 0.2093 / 0.1869 / 0.1935 |
| resnet18_scratch_64 | gaussian_blur/0.5 | 0.9145 ± 0.0539 | -0.0012 ± 0.0085 | 0.9298 / 0.8546 / 0.9592 |
| resnet18_scratch_64 | gaussian_blur/1.0 | 0.7661 ± 0.0879 | 0.1473 ± 0.0456 | 0.8126 / 0.6647 / 0.8208 |
| resnet18_scratch_64 | gaussian_blur/1.5 | 0.5814 ± 0.0609 | 0.3319 ± 0.0482 | 0.6441 / 0.5226 / 0.5774 |
| resnet18_scratch_64 | brightness/0.1 | 0.7327 ± 0.0884 | 0.1807 ± 0.0432 | 0.7632 / 0.6330 / 0.8017 |
| resnet18_scratch_64 | brightness/0.2 | 0.4815 ± 0.0933 | 0.4318 ± 0.0480 | 0.4944 / 0.3824 / 0.5677 |
| resnet18_scratch_64 | brightness/0.3 | 0.3327 ± 0.0922 | 0.5807 ± 0.0485 | 0.3319 / 0.2408 / 0.4252 |
| resnet18_scratch_64 | jpeg/50 | 0.8287 ± 0.0236 | 0.0847 ± 0.0237 | 0.8271 / 0.8059 / 0.8530 |
| resnet18_scratch_64 | jpeg/25 | 0.7613 ± 0.0130 | 0.1520 ± 0.0581 | 0.7631 / 0.7733 / 0.7476 |
| resnet18_scratch_64 | jpeg/10 | 0.5991 ± 0.0515 | 0.3142 ± 0.0967 | 0.5986 / 0.6509 / 0.5478 |
| small_cnn_scratch_64 | gaussian_noise/0.04 | 0.3631 ± 0.0558 | 0.5930 ± 0.0579 | 0.3019 / 0.4112 / 0.3763 |
| small_cnn_scratch_64 | gaussian_noise/0.08 | 0.1706 ± 0.0329 | 0.7856 ± 0.0304 | 0.1751 / 0.1356 / 0.2010 |
| small_cnn_scratch_64 | gaussian_noise/0.12 | 0.1264 ± 0.0276 | 0.8297 ± 0.0258 | 0.1573 / 0.1041 / 0.1179 |
| small_cnn_scratch_64 | gaussian_blur/0.5 | 0.9693 ± 0.0006 | -0.0132 ± 0.0022 | 0.9700 / 0.9688 / 0.9692 |
| small_cnn_scratch_64 | gaussian_blur/1.0 | 0.7623 ± 0.0096 | 0.1938 ± 0.0069 | 0.7693 / 0.7514 / 0.7663 |
| small_cnn_scratch_64 | gaussian_blur/1.5 | 0.5302 ± 0.0125 | 0.4259 ± 0.0099 | 0.5407 / 0.5164 / 0.5335 |
| small_cnn_scratch_64 | brightness/0.1 | 0.7895 ± 0.0068 | 0.1667 ± 0.0073 | 0.7827 / 0.7896 / 0.7962 |
| small_cnn_scratch_64 | brightness/0.2 | 0.5510 ± 0.0122 | 0.4052 ± 0.0112 | 0.5651 / 0.5443 / 0.5435 |
| small_cnn_scratch_64 | brightness/0.3 | 0.4375 ± 0.0153 | 0.5187 ± 0.0167 | 0.4478 / 0.4448 / 0.4198 |
| small_cnn_scratch_64 | jpeg/50 | 0.8004 ± 0.0146 | 0.1557 ± 0.0149 | 0.8150 / 0.8005 / 0.7857 |
| small_cnn_scratch_64 | jpeg/25 | 0.6834 ± 0.0114 | 0.2727 ± 0.0126 | 0.6917 / 0.6882 / 0.6705 |
| small_cnn_scratch_64 | jpeg/10 | 0.4704 ± 0.0053 | 0.4857 ± 0.0033 | 0.4757 / 0.4652 / 0.4703 |

## Seed 0: paired difference corrupted - clean [95 % paired bootstrap CI]

Paired percentile bootstrap, 1000 resamples, seed 0, the same image indices for clean and corrupted predictions. ✓ = the CI excludes zero (the change is distinguishable from zero for this run, for this corruption only).

| model | condition | Δ balanced accuracy | Δ macro F1 | CI excludes 0 (bal. acc.) |
|---|---|---|---|---|
| efficientnet_b0_imagenet_224 | gaussian_noise/0.04 | -0.7570 [-0.7656, -0.7480] | -0.7897 [-0.8015, -0.7781] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.08 | -0.8680 [-0.8729, -0.8625] | -0.9370 [-0.9440, -0.9298] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_noise/0.12 | -0.9035 [-0.9113, -0.8954] | -0.9419 [-0.9480, -0.9356] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/0.5 | -0.0525 [-0.0597, -0.0457] | -0.0505 [-0.0576, -0.0438] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.0 | -0.6973 [-0.7078, -0.6867] | -0.7031 [-0.7144, -0.6922] | ✓ |
| efficientnet_b0_imagenet_224 | gaussian_blur/1.5 | -0.8905 [-0.8983, -0.8823] | -0.9038 [-0.9125, -0.8944] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.1 | -0.0163 [-0.0211, -0.0114] | -0.0179 [-0.0229, -0.0127] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.2 | -0.0954 [-0.1034, -0.0870] | -0.1036 [-0.1125, -0.0951] | ✓ |
| efficientnet_b0_imagenet_224 | brightness/0.3 | -0.2314 [-0.2421, -0.2203] | -0.2461 [-0.2579, -0.2346] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/50 | -0.5352 [-0.5489, -0.5218] | -0.5826 [-0.5962, -0.5691] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/25 | -0.7081 [-0.7198, -0.6969] | -0.7623 [-0.7746, -0.7508] | ✓ |
| efficientnet_b0_imagenet_224 | jpeg/10 | -0.8360 [-0.8467, -0.8250] | -0.8658 [-0.8757, -0.8554] | ✓ |
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
| resnet18_imagenet_224 | gaussian_noise/0.04 | -0.7830 [-0.7922, -0.7736] | -0.8047 [-0.8168, -0.7937] | ✓ |
| resnet18_imagenet_224 | gaussian_noise/0.08 | -0.8677 [-0.8731, -0.8618] | -0.9336 [-0.9413, -0.9253] | ✓ |
| resnet18_imagenet_224 | gaussian_noise/0.12 | -0.8815 [-0.8856, -0.8767] | -0.9591 [-0.9642, -0.9532] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/0.5 | -0.0164 [-0.0210, -0.0122] | -0.0156 [-0.0201, -0.0116] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/1.0 | -0.5729 [-0.5835, -0.5625] | -0.6190 [-0.6326, -0.6068] | ✓ |
| resnet18_imagenet_224 | gaussian_blur/1.5 | -0.7765 [-0.7818, -0.7707] | -0.8779 [-0.8860, -0.8694] | ✓ |
| resnet18_imagenet_224 | brightness/0.1 | -0.0178 [-0.0231, -0.0130] | -0.0191 [-0.0246, -0.0142] | ✓ |
| resnet18_imagenet_224 | brightness/0.2 | -0.0923 [-0.1012, -0.0834] | -0.1010 [-0.1111, -0.0911] | ✓ |
| resnet18_imagenet_224 | brightness/0.3 | -0.2253 [-0.2354, -0.2149] | -0.2635 [-0.2757, -0.2518] | ✓ |
| resnet18_imagenet_224 | jpeg/50 | -0.3536 [-0.3654, -0.3402] | -0.3908 [-0.4033, -0.3771] | ✓ |
| resnet18_imagenet_224 | jpeg/25 | -0.5057 [-0.5185, -0.4940] | -0.5555 [-0.5700, -0.5416] | ✓ |
| resnet18_imagenet_224 | jpeg/10 | -0.6280 [-0.6373, -0.6181] | -0.7318 [-0.7426, -0.7212] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.04 | -0.3247 [-0.3373, -0.3123] | -0.3281 [-0.3415, -0.3151] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.08 | -0.6307 [-0.6422, -0.6171] | -0.6608 [-0.6744, -0.6453] | ✓ |
| resnet18_scratch_64 | gaussian_noise/0.12 | -0.7148 [-0.7242, -0.7054] | -0.7723 [-0.7828, -0.7610] | ✓ |
| resnet18_scratch_64 | gaussian_blur/0.5 | +0.0057 [+0.0006, +0.0107] | +0.0060 [+0.0011, +0.0111] | ✓ |
| resnet18_scratch_64 | gaussian_blur/1.0 | -0.1115 [-0.1210, -0.1017] | -0.1250 [-0.1362, -0.1141] | ✓ |
| resnet18_scratch_64 | gaussian_blur/1.5 | -0.2800 [-0.2916, -0.2675] | -0.3013 [-0.3133, -0.2889] | ✓ |
| resnet18_scratch_64 | brightness/0.1 | -0.1609 [-0.1728, -0.1493] | -0.1802 [-0.1930, -0.1675] | ✓ |
| resnet18_scratch_64 | brightness/0.2 | -0.4297 [-0.4419, -0.4153] | -0.4586 [-0.4723, -0.4453] | ✓ |
| resnet18_scratch_64 | brightness/0.3 | -0.5922 [-0.6040, -0.5800] | -0.6455 [-0.6588, -0.6331] | ✓ |
| resnet18_scratch_64 | jpeg/50 | -0.0970 [-0.1064, -0.0880] | -0.1042 [-0.1149, -0.0937] | ✓ |
| resnet18_scratch_64 | jpeg/25 | -0.1610 [-0.1719, -0.1502] | -0.1774 [-0.1896, -0.1646] | ✓ |
| resnet18_scratch_64 | jpeg/10 | -0.3255 [-0.3402, -0.3106] | -0.3499 [-0.3658, -0.3339] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.04 | -0.6558 [-0.6672, -0.6448] | -0.6693 [-0.6825, -0.6563] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.08 | -0.7826 [-0.7916, -0.7733] | -0.8557 [-0.8645, -0.8467] | ✓ |
| small_cnn_scratch_64 | gaussian_noise/0.12 | -0.8004 [-0.8093, -0.7915] | -0.8761 [-0.8854, -0.8669] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/0.5 | +0.0123 [+0.0077, +0.0174] | +0.0121 [+0.0077, +0.0170] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/1.0 | -0.1884 [-0.1991, -0.1777] | -0.2202 [-0.2324, -0.2073] | ✓ |
| small_cnn_scratch_64 | gaussian_blur/1.5 | -0.4170 [-0.4297, -0.4049] | -0.4599 [-0.4736, -0.4463] | ✓ |
| small_cnn_scratch_64 | brightness/0.1 | -0.1750 [-0.1860, -0.1636] | -0.1839 [-0.1961, -0.1715] | ✓ |
| small_cnn_scratch_64 | brightness/0.2 | -0.3926 [-0.4044, -0.3823] | -0.4405 [-0.4547, -0.4277] | ✓ |
| small_cnn_scratch_64 | brightness/0.3 | -0.5099 [-0.5211, -0.4994] | -0.5893 [-0.6032, -0.5760] | ✓ |
| small_cnn_scratch_64 | jpeg/50 | -0.1428 [-0.1531, -0.1321] | -0.1550 [-0.1669, -0.1429] | ✓ |
| small_cnn_scratch_64 | jpeg/25 | -0.2660 [-0.2774, -0.2550] | -0.3137 [-0.3271, -0.3006] | ✓ |
| small_cnn_scratch_64 | jpeg/10 | -0.4820 [-0.4958, -0.4691] | -0.5457 [-0.5624, -0.5305] | ✓ |

## Seed 0: per-class recall in the worst condition per model

Worst condition = lowest corrupted balanced accuracy of the seed-0 run. Classes marked † are largely low-information (near-uniform tiles; DATASETS.md).

**efficientnet_b0_imagenet_224** (seed 0, exp-20261028-004715-3da317): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9844 clean → 0.0809.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9911 | 0.3178 | +0.6733 |
| Forest † (34 % low-information) | 0.9956 | 0.0000 | +0.9956 |
| HerbaceousVegetation | 0.9844 | 0.0000 | +0.9844 |
| Highway | 0.9893 | 0.0000 | +0.9893 |
| Industrial | 0.9920 | 0.0133 | +0.9787 |
| Pasture | 0.9733 | 0.0000 | +0.9733 |
| PermanentCrop | 0.9547 | 0.0000 | +0.9547 |
| Residential | 0.9867 | 0.0000 | +0.9867 |
| River | 0.9813 | 0.1333 | +0.8480 |
| SeaLake † (78 % low-information) | 0.9956 | 0.3444 | +0.6511 |

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

**resnet18_imagenet_224** (seed 0, exp-20261027-214915-e0d372): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9844 clean → 0.1029.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9733 | 0.0000 | +0.9733 |
| Forest † (34 % low-information) | 0.9933 | 0.0000 | +0.9933 |
| HerbaceousVegetation | 0.9933 | 0.0000 | +0.9933 |
| Highway | 0.9947 | 0.0000 | +0.9947 |
| Industrial | 0.9893 | 0.0293 | +0.9600 |
| Pasture | 0.9733 | 0.0000 | +0.9733 |
| PermanentCrop | 0.9573 | 0.0000 | +0.9573 |
| Residential | 0.9933 | 1.0000 | -0.0067 |
| River | 0.9787 | 0.0000 | +0.9787 |
| SeaLake † (78 % low-information) | 0.9978 | 0.0000 | +0.9978 |

**resnet18_scratch_64** (seed 0, exp-20261027-201115-e384ed): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9241 clean → 0.2093.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.8844 | 0.0044 | +0.8800 |
| Forest † (34 % low-information) | 0.9822 | 0.0000 | +0.9822 |
| HerbaceousVegetation | 0.9222 | 0.0022 | +0.9200 |
| Highway | 0.9093 | 0.1733 | +0.7360 |
| Industrial | 0.9413 | 0.7813 | +0.1600 |
| Pasture | 0.8833 | 0.0067 | +0.8767 |
| PermanentCrop | 0.8827 | 0.0107 | +0.8720 |
| Residential | 0.9911 | 0.9978 | -0.0067 |
| River | 0.8800 | 0.1120 | +0.7680 |
| SeaLake † (78 % low-information) | 0.9644 | 0.0044 | +0.9600 |

**small_cnn_scratch_64** (seed 0, exp-20261027-204957-d42310): worst condition `gaussian_noise/0.12`, balanced accuracy 0.9577 clean → 0.1573.

| class | clean recall | corrupted recall | drop |
|---|---|---|---|
| AnnualCrop | 0.9467 | 0.0000 | +0.9467 |
| Forest † (34 % low-information) | 0.9978 | 0.0000 | +0.9978 |
| HerbaceousVegetation | 0.9400 | 0.0000 | +0.9400 |
| Highway | 0.9360 | 0.0000 | +0.9360 |
| Industrial | 0.9333 | 0.1973 | +0.7360 |
| Pasture | 0.9433 | 0.0000 | +0.9433 |
| PermanentCrop | 0.9467 | 0.0000 | +0.9467 |
| Residential | 0.9978 | 0.9889 | +0.0089 |
| River | 0.9600 | 0.0000 | +0.9600 |
| SeaLake † (78 % low-information) | 0.9756 | 0.3867 | +0.5889 |

## Pipeline check: majority baseline

The majority baseline does not look at the image, so every corrupted metric must equal its clean metric exactly: **equal for all 36 run x condition pairs**.

## Confounds

- **resolution_upsampling**: Corruptions are applied at 64 px. Pretrained models see an upsampled version (resize 256, centre crop 224) in which corruption and content are scaled together (a blur of 1 px becomes about 4 px); scratch models see both at native resolution. This is not a smoothing of the corruption relative to the content (robustness protocol, amendment 1); pretraining and resolution stay confounded, so group differences have no attributed cause.
- **pretrained_vs_scratch**: ImageNet pretraining and input resolution are confounded with each other.
- **centre_crop**: Pretrained models never see the 4 px border of a tile.
- **single_split**: One fixed split: no split-to-split variance.
- **three_seeds**: n = 3 seeds: the sd is itself uncertain; per-seed values are shown.
- **selection_bias**: VAL chose each checkpoint; the TEST numbers chose nothing (one clean TEST evaluation per run, following the committed final analysis plan).

## Runs

| model | seed | experiment | clean evaluation (pinned) |
|---|---|---|---|
| efficientnet_b0_imagenet_224 | 0 | exp-20261028-004715-3da317 | ev-20261029-113052-84ed06 |
| efficientnet_b0_imagenet_224 | 1 | exp-20261028-013827-6cbab1 | ev-20261029-113134-691d5f |
| efficientnet_b0_imagenet_224 | 2 | exp-20261028-030154-fe104a | ev-20261029-113216-4c6760 |
| majority_baseline | 0 | exp-20261027-201053-5f2809 | ev-20261029-112504-ec73cf |
| majority_baseline | 1 | exp-20261027-201101-632b02 | ev-20261029-112523-9b58fd |
| majority_baseline | 2 | exp-20261027-201108-703436 | ev-20261029-112543-28d2fa |
| resnet18_imagenet_224 | 0 | exp-20261027-214915-e0d372 | ev-20261029-112850-a00cad |
| resnet18_imagenet_224 | 1 | exp-20261027-225711-6db256 | ev-20261029-112930-d32a72 |
| resnet18_imagenet_224 | 2 | exp-20261027-233452-43080e | ev-20261029-113009-789459 |
| resnet18_scratch_64 | 0 | exp-20261027-201115-e384ed | ev-20261029-112608-c4d668 |
| resnet18_scratch_64 | 1 | exp-20261027-202124-e89a98 | ev-20261029-112633-e4bace |
| resnet18_scratch_64 | 2 | exp-20261027-202749-e42275 | ev-20261029-112658-df5146 |
| small_cnn_scratch_64 | 0 | exp-20261027-204957-d42310 | ev-20261029-112722-d17c6a |
| small_cnn_scratch_64 | 1 | exp-20261027-211106-cc7c83 | ev-20261029-112747-79a389 |
| small_cnn_scratch_64 | 2 | exp-20261027-213240-d42c79 | ev-20261029-112811-931ebd |
