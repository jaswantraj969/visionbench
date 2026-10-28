# eurosat-v1: latency, seed 0 of each trainable model

Model forward pass only (random input; preprocessing, decoding and data loading excluded), 10 warm-up + 50 timed iterations, measured with `visionbench latency` on this laptop on 2026-10-28. Split-independent (no images are read), so this is not a validation or test number. The majority baseline has no forward pass.

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
