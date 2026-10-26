# Experiment Protocol (initial draft — M0)

This protocol governs all benchmarks. Nothing below is implemented yet beyond `set_seed`;
it defines the rules later milestones must follow.

## Splits
- **Duplicate groups** are formed before splitting: exact duplicates via SHA-256 of file bytes;
  near duplicates via a perceptual hash with a documented Hamming-distance threshold.
- An entire group is assigned to exactly one split (train / val / test), so no duplicate leaks
  across splits.
- Splits are **stratified by class** at the group level; achieved per-split class proportions are
  recorded in the manifest.
- **Label-conflict groups** (same/near-identical image with different labels) are flagged in the
  dataset report and excluded by default; the choice is recorded.
- Splits are deterministic given (manifest, seed) and stored in the manifest.

## Seeds
- Every run records its seed and the `set_seed()` report plus `env_info`.
- Reported results state the number of seeds; multi-seed results report mean and spread.
- Bitwise reproducibility is only claimed for the same hardware and software versions.

## Model selection
- Hyperparameter choice, checkpoint selection, and early stopping use the **validation split only**.
- The **test split is evaluated once** per final model, for final reporting only. Test results never
  feed back into any choice.

## Preprocessing
- Models trained from scratch use normalization statistics computed on the **train split only**.
- Pretrained models use the normalization of their pretrained weights (recorded).
- Input resolution is recorded per model (e.g. 224×224 for ResNet18/EfficientNet-B0).
- Training-time augmentation is recorded in the run config; evaluation uses deterministic transforms.

## Baselines
- A majority-class baseline is reported alongside every comparison.

## Known limitations
- Perceptual hashing misses heavy crops, rotations, and strong color edits, and can group distinct
  but visually similar images (false positives). The threshold is a trade-off and is reported.
- Duplicate detection operates within one dataset; overlap with pretraining data (e.g. ImageNet) is
  not detected.
- GPU nondeterminism: some CUDA ops lack deterministic implementations; `set_seed` warns rather than
  fails on such ops.
