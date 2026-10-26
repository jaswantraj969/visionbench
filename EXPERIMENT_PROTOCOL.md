# Experiment Protocol

This protocol governs all benchmarks. Status per section: **implemented (M1)**, or
**planned** (rules that later milestones must follow).

## 1. Dataset format (implemented, M1)

```
<root>/<class_name>/<image files>
```

- Exactly one level of class folders directly under `<root>`. The folder name is the label.
- Supported extensions (case-insensitive): `.jpg .jpeg .png .bmp .webp`. Only the JPEG, PNG, BMP
  and WEBP decoders are allowed, so a file whose content is another format is rejected as corrupt.
- Nothing is silently dropped. Every skipped entry is reported in `report.json` under one of these
  categories:

| Category | Meaning |
|---|---|
| `zero_byte` | empty file |
| `unsupported_extension` | extension not in the list (e.g. `.txt`, `.gif`) |
| `unreadable_or_corrupt` | Pillow `verify()` fails, or the full `load()` of a reopened image fails (catches truncation, which `verify()` alone misses); also non-regular files |
| `too_large_pixels` | width × height > `max_image_pixels` (default 50,000,000) |
| `symlink` | symlink or Windows reparse point (incl. NTFS junction); never followed |
| `hidden_or_system_file` | dotfiles, `.DS_Store`, `Thumbs.db`, `desktop.ini`, Windows hidden/system attribute |
| `nested_directory` | a sub-directory inside a class folder (not descended into) |
| `invalid_class_name` | empty, leading dot, path separator or control character |
| `invalid_file_name` | backslash or control character in a file name |
| `file_outside_class_folder` | regular file directly under `<root>` |
| `path_collision` | two names that are equal after Unicode NFC + case folding |

- The pixel limit is applied explicitly per scan: Pillow's `MAX_IMAGE_PIXELS` is set to the
  configured value, and `DecompressionBombWarning` is escalated to an error. Pillow's global
  default is not relied on. `LOAD_TRUNCATED_IMAGES` is forced off.
- Requirements: at least 2 classes, and at least `min_images_per_class` (default 3) valid images
  per class. If either fails, import stops with an error naming the offending classes and nothing
  is written.
- Paths are stored as `<class>/<file>`: POSIX separators, Unicode NFC. Directories are visited in
  sorted order, so no result depends on filesystem listing order.

## 2. Snapshot and fingerprint (implemented, M1)

- Each accepted file is read once. The same bytes are hashed (SHA-256), decoded, and written to
  `datasets/<id>/raw/<class>/<file>`, so the snapshot matches its recorded hashes.
- **Fingerprint** = SHA-256 over the sorted lines `relpath \t label \t sha256 \n`. It is
  independent of creation/listing order and OS path separators. Tabs and newlines are rejected in
  names.
- **dataset_id** = `ds-` + first 12 hex of SHA-256(`fingerprint \n name`). An existing id is never
  overwritten.

## 3. Duplicate groups (implemented, M1)

- **Exact duplicates:** identical SHA-256 of file bytes.
- **Near duplicates:** an in-house 64-bit dHash. Steps: grayscale, LANCZOS resize to 9×8, then
  per row 8 bits `pixel[x+1] > pixel[x]`, packed row-major with the MSB first. Two images are
  near duplicates if the Hamming distance is ≤ `dhash_threshold` (default **5**).
- **Low-information guard:** an image is excluded from perceptual matching when either holds:
  - its grayscale standard deviation (0–255 scale, measured on a ≤256 px thumbnail) is below
    `low_info_std_threshold` (default **3.0**);
  - its dHash is degenerate (all 0 or all 1 bits).

  Such images are listed as `perceptual_check_skipped_low_information`. Exact matching still
  applies to them. Without this guard, distinct uniform images (e.g. blank sky or sea tiles), which
  all hash to `0x0`, would merge into one large false group.
- Pairwise comparison is vectorized: chunked NumPy XOR plus popcount, using `np.bitwise_count`
  when available, otherwise a 256-entry byte lookup table. Identical hashes are collapsed first.
- Groups are the connected components (union-find) of exact plus near edges. Every image belongs to
  exactly one group (a singleton if it has no duplicates).
  `group_id` = `g-` + first 16 hex of SHA-256 of the sorted member relpaths.
- **Label conflicts:** a group whose members have more than one label.
  - Policy `exclude` (default): the whole group is removed from every split and listed in
    `manifest.json` → `excluded`.
  - Policy `error`: the split is refused.
  - Labels are never changed.
- **Same-class duplicates** stay in the dataset, and their group is always assigned to one split.

## 4. Splits (implemented, M1; algorithm `vb-split-v1`)

1. Rebuild duplicate groups from the stored per-file records, using this split's thresholds.
2. Apply the conflict policy.
3. For each class: sort its groups by `group_id`, then permute them with a NumPy `Generator`
   seeded from `SeedSequence([seed, sha256(class name)[:8]])`.
4. Walk the groups in that order, giving each one to the split with the largest remaining deficit,
   where deficit = `fraction × class_total − assigned` (in images). Ties go to train, then val,
   then test.
5. Fractions are converted to exact rationals (`Fraction(repr(f))`), so tie-breaking is identical
   on every platform.
6. If any class has 0 items in train, val or test, the split errors and names the class and its
   counts. `--allow-missing-classes` turns this into a recorded warning instead.
7. **Leakage check:** computed from the final entries. It records how many groups and SHA-256 values
   appear in more than one split, and must be 0 by construction.
8. `split_id` = `sp-` + 12 hex of SHA-256 over (algorithm, dataset fingerprint, full split config).
9. `manifest.json` is canonical JSON: sorted keys, LF line endings, entries sorted by relpath, and no
   timestamp. The same inputs therefore give byte-identical manifests. `created_at` and the NumPy
   and Python versions are kept in `split_info.json`.
10. Re-running an existing split recomputes the manifest and compares it byte for byte. If it
    differs, the split is refused rather than overwritten.

Manifest fields:
- `entries`: one per file, with `relpath`, `label`, `sha256`, `group_id`, `split`.
- Metadata: `config`, per-class `counts` and `target_counts`, `split_totals`, `excluded`,
  `warnings`, `leakage_check`, `algorithm`, `visionbench_version`, `dataset_fingerprint`.

## 5. Seeds (partly implemented)
- `set_seed()` (M0) seeds Python, NumPy and torch. Split and report sampling use `derived_rng`
  (seed + SHA-256 of a key), never Python's salted `hash()`.
- Every training run must record its seed, the `set_seed()` report, and `env_info` (planned, M2).
- Bitwise reproducibility is only claimed for identical hardware and software versions.

## 6. Model selection (planned)
- Hyperparameters, checkpoint selection and early stopping use the **validation split only**.
- The **test split is evaluated once** per final model, for final reporting. Test results never feed
  back into any choice.

## 7. Preprocessing (planned)
- Models trained from scratch normalize with statistics computed on the **train split only**.
- Pretrained models use their weights' normalization (recorded).
- Input resolution is recorded per model (e.g. 224×224 for ResNet18 and EfficientNet-B0).
- Training augmentation is recorded; evaluation uses deterministic transforms.

## 8. Baselines (planned)
- A majority-class baseline is reported with every comparison.

## 9. Known limitations
- dHash finds resizes and re-encodes. It **misses** crops, rotations, flips, heavy color or contrast
  edits, overlays, and semantic near-duplicates (the same object photographed twice).
- The threshold trades false positives (distinct but similar images merged) against false negatives.
  The default of 5/64 is a heuristic and has not been tuned on real data yet.
- The low-information guard covers near-uniform images and exactly degenerate hashes. Low-texture
  images with smooth gradients can still produce *sparse but non-zero* hashes that sit close to
  each other. The default std threshold of 3.0 has not been calibrated on real data yet.
- Duplicate detection runs within one dataset only. Overlap with pretraining data (e.g. ImageNet)
  is not detected.
- Split determinism depends on the NumPy `Generator` stream, which NumPy does not guarantee across
  versions. The NumPy version is recorded in `split_info.json`.
- GPU nondeterminism: some CUDA ops have no deterministic implementation, so `set_seed` warns
  rather than fails.
