# Grad-CAM cascading randomization on the seed-0 EuroSAT models - VALIDATION images - diagnostic, not results

vb-gradcam-2; 20 seeded VAL images (M5 demo selection); Spearman rho between the trained CAM and the CAM after each cascading stage (same target class: the trained model's prediction). Report only; flag if mean |rho| > 0.5 after full randomization.

## resnet18_scratch_64 (exp-20261027-201115-e384ed; layer layer4; CAM 2x2)

| stage (cumulative from the top) | mean rho | mean abs rho | defined / undefined |
|---|---|---|---|
| fc | 0.033 | 0.658 | 16 / 4 |
| layer4 | 0.092 | 0.491 | 15 / 5 |
| layer3 | 0.109 | 0.374 | 14 / 6 |
| layer2 | -0.083 | 0.570 | 17 / 3 |
| layer1 | -0.098 | 0.505 | 18 / 2 |
| stem (conv1, bn1) | 0.059 | 0.430 | 15 / 5 |

Flag (mean |rho| > 0.5 after full randomization): **False**

## small_cnn_scratch_64 (exp-20261027-204957-d42310; layer features.block4.relu; CAM 8x8)

| stage (cumulative from the top) | mean rho | mean abs rho | defined / undefined |
|---|---|---|---|
| classifier | 0.079 | 0.260 | 20 / 0 |
| block4 | 0.005 | 0.139 | 18 / 2 |
| block3 | 0.003 | 0.183 | 18 / 2 |
| block2 | -0.042 | 0.176 | 19 / 1 |
| block1 | -0.002 | 0.252 | 16 / 4 |

Flag (mean |rho| > 0.5 after full randomization): **False**

## resnet18_imagenet_224 (exp-20261027-214915-e0d372; layer layer4; CAM 7x7)

| stage (cumulative from the top) | mean rho | mean abs rho | defined / undefined |
|---|---|---|---|
| fc | 0.128 | 0.370 | 14 / 6 |
| layer4 | -0.043 | 0.165 | 20 / 0 |
| layer3 | 0.051 | 0.179 | 20 / 0 |
| layer2 | -0.017 | 0.172 | 20 / 0 |
| layer1 | -0.046 | 0.227 | 18 / 2 |
| stem (conv1, bn1) | -0.036 | 0.194 | 17 / 3 |

Flag (mean |rho| > 0.5 after full randomization): **False**

## efficientnet_b0_imagenet_224 (exp-20261028-004715-3da317; layer features.8; CAM 7x7)

| stage (cumulative from the top) | mean rho | mean abs rho | defined / undefined |
|---|---|---|---|
| classifier | -0.076 | 0.370 | 20 / 0 |
| features.8 | -0.065 | 0.399 | 18 / 2 |
| features.7 | -0.102 | 0.181 | 20 / 0 |
| features.6 | 0.070 | 0.184 | 20 / 0 |
| features.5 | -0.051 | 0.148 | 20 / 0 |
| features.4 | 0.008 | 0.167 | 20 / 0 |
| features.3 | -0.022 | 0.129 | 20 / 0 |
| features.2 | -0.001 | 0.164 | 20 / 0 |
| features.1 | 0.079 | 0.210 | 20 / 0 |
| features.0 | 0.066 | 0.258 | 20 / 0 |

Flag (mean |rho| > 0.5 after full randomization): **False**

## Chance level of mean |rho| (independent maps)

| CAM cells | E\|Spearman rho\| under independence |
|---|---|
| 4 (2x2, resnet18_scratch_64) | 0.500 (exact, all 24 permutations) |
| 49 (7x7) | 0.116 (simulated, 4000 draws) |
| 64 (8x8) | 0.102 (simulated, 4000 draws) |

resnet18_scratch_64's values are at chance level for a 4-cell map (its mean signed rho stays
near 0), so its |rho| cannot be judged against the 0.5 flag. The 7x7 / 8x8 models end at
0.19-0.26 after full randomization: above chance (about 0.10-0.12), below the 0.5 flag.
