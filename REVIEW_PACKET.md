# Review Packet — VB-D1

**Task:** VB-D1: EuroSAT onboarding: license, EXIF policy, independent split audit, threshold
calibration
**Dates:** 2026-10-27/28
**Environment:** Windows 11 Home 10.0.26300; Python 3.11.15; Pillow 12.3.0; NumPy 2.4.6;
SciPy 1.17.1; torch 2.14.1+cu130 (not used for this task's data work).

All outputs below come from commands run in this session; paths under the session scratchpad
are shortened to `<scratchpad>`.

**Not done, by instruction:**
- no training and no evaluation;
- no test-split use: the audit compares val/test images with train images, but it is a data
  check, not a model evaluation;
- no threshold decision.

---

## 1. Commits (local only; no remote, nothing pushed)

```
297879d D1 docs: DATASETS.md (EuroSAT), EXIF policy, audit method, calibration procedure
b9c8e6c D1: independent split audit (pHash), oversized-group warnings, low-info breakdown
052ef34 D1 step 0: EXIF orientation policy (one shared image decoder)
6166ce4 M4.1 review packet
```

- `052ef34`: step 0, committed on its own. The staged tree alone passed
  `pytest tests/data tests/training` (163 passed, 1 skipped). The full suite before the commit
  had 276 passed, 1 skipped.
- `b9c8e6c`: audit, warnings and low-info breakdown. Full suite before the commit: 285 passed,
  1 skipped.
- The docs and this packet are the final commits.
- Nothing under `var/` is committed: downloads, the snapshot, splits, audits and contact sheets
  are all gitignored.

## 2. Acceptance criteria → evidence

| Criterion | Evidence |
|---|---|
| EXIF policy implemented and tested | §3; `tests/data/test_exif.py` (6 tests) |
| audit-split with an independent hash, tested | §4; `tests/data/test_audit.py` (9 tests) |
| oversized-group warning in split_info.json + CLI, manifest unchanged | §4; `test_oversized_group_warnings_in_split_info_not_manifest`, `test_cli_split_prints_group_warnings_and_audit_runs` |
| license verified from primary sources | §5, DATASETS.md |
| dataset imported and fully verified | §6 (verbatim), `verify --full` OK |
| calibration table + contact sheets for thresholds 3 and 5 | §7, §8, §10 |
| no training, no test evaluation | no `train`/`evaluate`/`latency` command was run; `var/experiments` and the experiment DB were not touched by this task |

## 3. Step 0: EXIF orientation policy (`052ef34`)

- **One shared decoder.** `visionbench.data.imageio.decode_image(bytes | path)` opens the image
  (JPEG/PNG/BMP/WEBP decoders only), calls `load()`, records the stored mode and format, and
  returns `ImageOps.exif_transpose(im)`. `load_rgb(path)` is the RGB variant.
- **Callers.** Every pixel read goes through it, and no other `Image.open` that reads pixels
  remains in `src/`:
  - the scan's `inspect_image` (width/height, gray std, dHash);
  - `training.data.load_image`, which is also used by evaluation and the norm-stats cache;
  - the audit.

  The scan keeps its separate `verify()` pass, which reads no pixels. The API image endpoint
  serves the original file bytes without decoding them, and a browser applies EXIF
  orientation itself.
- **Versions.** Dataset format 1 → **2**; new `HASH_VERSION = 2`. `dataset.json` and
  `files.json` both start with `{"format_version": 2, "orientation": "exif_transposed",
  "hash_version": 2}`.
- **Old snapshots.** `load_dataset`, `load_records` and everything built on them (`verify`,
  `split`, training, the audit, the API) raise `DatasetFormatError` (a
  `DatasetValidationError`, so API 422) for any other header. The message says: "Re-import the
  source folder (remove the old snapshot directory first, or import under a different name)".
- **Tests:**
  - a 48x32 JPEG stored sideways with `Orientation=6`, versus the displayed pixels saved as a
    PNG without EXIF: identical recorded size (32x48), identical dHash and gray std, and
    identical `load_image` pixels;
  - a lossy (JPEG q95) rotated copy: same size, dHash within 4 bits;
  - an image without EXIF: unchanged pixels and size;
  - an import records the transposed size;
  - `dataset.json`/`files.json` in format 1 are refused by every reader.

  The pixel-identical comparison uses a lossless PNG for the rotated copy, because two JPEG
  encodings never decode to identical pixels.

## 4. Audit, pHash, oversized-group warnings (`b9c8e6c`)

- **pHash** (`visionbench.data.phash`, in-house):
  1. grayscale, Lanczos resize to 32x32;
  2. `scipy.fft.dctn` (type II, orthonormal);
  3. keep the top-left 8x8;
  4. compare each coefficient with the median of the **63 AC coefficients**; the DC bit is
     forced to 0.

  **Consequence, found on real data:** a median of 63 values has exactly 31 values above it,
  so every hash has popcount 31 and **every pHash distance is even**. This is visible in the
  histograms in §8 and documented in EXPERIMENT_PROTOCOL §4a.
- **Audit:**
  - re-decodes every image of the split;
  - nearest train image of each val/test image by pHash and by dHash, over all train images
    with low-information ones included and flagged; chunked XOR + popcount;
  - a separate nearest search among *informative* images only, for the consistency check;
  - all informative cross-split pairs in the near-miss band;
  - cross-split exact SHA-256 matches and decoded-pixel matches (SHA-256 of the decoded RGB
    bytes and size);
  - group statistics and seeded pair samples (a)–(d);
  - contact sheets: two columns, at most 20 pairs per PNG, 128-px tiles (64-px images
    upscaled with NEAREST), each labelled with pair id, both distances, splits, classes,
    low-info flags and same-group;
  - output is written to a staging directory and renamed to `audits/<split_id>/`.
- **Oversized-group warnings:** `split.group_size_stats(manifest)` reports multi-member groups
  larger than the class's val target or larger than 2 % of the class. They go to
  `split_info.json` (`group_size_warnings`) and to the `dataset split` output, and are
  recomputed when a split already exists. The manifest is unchanged: same keys and bytes,
  tested with its sha256.
- **Synthetic tests** (`tests/data/test_audit.py`):
  - a planted re-scaled and re-encoded near-duplicate pair, made low-information by the
    split's std threshold, ends up in separate groups in different splits. The audit lists it
    as a top-2 closest pHash pair (distance ≤ 6, both flagged low-info, "same group: no");
  - a pixel-identical pair with different PNG bytes is counted as 1 decoded-pixel match and 0
    exact matches;
  - pHash matches JPEG-q60, downscaled and upscaled-JPEG copies within 6 bits, while 28
    unrelated pairs are ≥ 14 bits apart;
  - the audit (and a re-run) leaves every byte of the split directory unchanged;
  - sheets hold ≤ 20 pairs;
  - the CLI output.
- **Low-information breakdown** (report format 2): per class and per reason (std / degenerate /
  popcount, overlapping), printed by `dataset import` and `dataset report`. It was added
  because step 4 asks for that breakdown and the report did not have it.

## 5. License verification (before the download)

Done before downloading. Full details, quotes and URLs are in DATASETS.md (printed separately).
Summary:
- **Official source:** Zenodo record 10.5281/zenodo.7711810, linked from the authors'
  repository github.com/phelber/EuroSAT. That README lists the old DFKI host under
  "Deprecated Hosting".
- **Data license:** MIT. Sources:
  - the Zenodo API license field `mit-license`;
  - the README: "The dataset is licensed under the MIT license.";
  - the repo's `LICENSE` file is MIT.

  The repository holds no code, so there is no code-vs-data distinction to make. A third-party
  RDF index lists CC BY 4.0; the record's own metadata was treated as authoritative. This is
  noted in DATASETS.md.
- **Sentinel-2:** Copernicus Sentinel data licence (rev. 1). It permits reproduction,
  distribution and adaptation, and requires the notice "Contains modified Copernicus Sentinel
  data [Year]". The authors' link to the legal notice on `sentinel.esa.int` no longer resolves
  (DNS failure), so the licence text was read at `ads.atmosphere.copernicus.eu/licences/ec-sentinel`.
- **Citations:** IEEE JSTARS 2019 (DOI 10.1109/JSTARS.2019.2918242) and IGARSS 2018 (DOI
  10.1109/IGARSS.2018.8519248).
- **Verdict:** the license permits this use, so there was no STOP. Images are not redistributed
  in the repo.
- **Open point for the supervisor:** the `[Year]` of the Copernicus notice is not stated in the
  record or README.

## 6. Download and import (verbatim)

Download: `curl.exe -sS -L --fail -o var/downloads/EuroSAT_RGB.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"`

```
http 200 | 94658721 bytes | 696.617068 s | final URL https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1
-rw-r--r-- 1 ~JR 197609 94658721 Oct 27 23:37 EuroSAT_RGB.zip
-rw-r--r-- 1 ~JR 197609       21 Oct 27 23:37 EuroSAT_RGB.zip.finished
-rw-r--r-- 1 ~JR 197609       21 Oct 27 23:25 EuroSAT_RGB.zip.started
f46e308c4d50d4bf32fedad2d3d62f3b *var/downloads/EuroSAT_RGB.zip
b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90 *var/downloads/EuroSAT_RGB.zip
started 2026-10-27T18:25:58Z, finished 2026-10-27T18:37:35Z (UTC)
```

- The MD5 equals Zenodo's published `md5:f46e308c4d50d4bf32fedad2d3d62f3b`; the size equals
  the API size.
- The archive was checked with VisionBench's hardened validator:
  `members 27011`, `layout single_top_folder strip EuroSAT_RGB files 27000 uncompressed
  91844360 ignored 0`, 27,000 `.jpg`. Class counts: AnnualCrop 3000, Forest 3000,
  HerbaceousVegetation 3000, Highway 2500, Industrial 2500, Pasture 2000, PermanentCrop 2500,
  Residential 3000, River 2500, SeaLake 3000.
- It was then extracted with `extract_archive` into `var/downloads/extracted/` in 52.0 s
  (27,000 files).

Import with current defaults (dHash 5, std 3.0, popcount [8,56]), then report and full
verification. Each command was run through `<scratchpad>/d1/run_cli.py`, which prints the
command, its output and the wall time:

```
> visionbench dataset import var/downloads/extracted/EuroSAT_RGB --name eurosat-rgb
imported dataset ds-61112928e412 (eurosat-rgb)
  path: <repo>\var\datasets\ds-61112928e412
  fingerprint: 53a853ae82ee65dd0359263e5a76d81889bd46f770f6224e58965c42637b852f
  classes: 10 | images: 27000 | bytes: 91844360
  class counts: AnnualCrop=3000, Forest=3000, HerbaceousVegetation=3000, Highway=2500, Industrial=2500, Pasture=2000, PermanentCrop=2500, Residential=3000, River=2500, SeaLake=3000 | imbalance ratio: 1.5
  issues: none
  duplicate groups (>1 member): 8 (16 files) | exact duplicate files: 0 | label-conflict groups: 1
  perceptual check skipped (low information): 3357 (std < 3.0, degenerate dHash, or popcount outside [8, 56]; dHash threshold 5)
    low information per class (total; std/degenerate/popcount, overlapping):
      AnnualCrop: 3; 3/0/0
      Forest: 1019; 1019/0/5
      HerbaceousVegetation: 0; 0/0/0
      Highway: 0; 0/0/0
      Industrial: 0; 0/0/0
      Pasture: 8; 8/0/0
      PermanentCrop: 0; 0/0/0
      Residential: 0; 0/0/0
      River: 0; 0/0/0
      SeaLake: 2327; 2272/211/1059
[exit code 0; wall time 466.2 s]

> visionbench dataset report ds-61112928e412
dataset ds-61112928e412 (eurosat-rgb)
  fingerprint: 53a853ae82ee65dd0359263e5a76d81889bd46f770f6224e58965c42637b852f
  classes: 10 | images: 27000 | bytes: 91844360
  class counts: AnnualCrop=3000, Forest=3000, HerbaceousVegetation=3000, Highway=2500, Industrial=2500, Pasture=2000, PermanentCrop=2500, Residential=3000, River=2500, SeaLake=3000 | imbalance ratio: 1.5
  issues: none
  duplicate groups (>1 member): 8 (16 files) | exact duplicate files: 0 | label-conflict groups: 1
  perceptual check skipped (low information): 3357 (std < 3.0, degenerate dHash, or popcount outside [8, 56]; dHash threshold 5)
    low information per class (total; std/degenerate/popcount, overlapping):
      AnnualCrop: 3; 3/0/0
      Forest: 1019; 1019/0/5
      HerbaceousVegetation: 0; 0/0/0
      Highway: 0; 0/0/0
      Industrial: 0; 0/0/0
      Pasture: 8; 8/0/0
      PermanentCrop: 0; 0/0/0
      Residential: 0; 0/0/0
      River: 0; 0/0/0
      SeaLake: 2327; 2272/211/1059
  width min/median/max: 64/64.0/64 | height: 64/64.0/64
  modes: {'RGB': 27000} | formats: {'JPEG': 27000}
[exit code 0; wall time 3.2 s]

> visionbench dataset verify ds-61112928e412 --full
verify ds-61112928e412: full (sizes + sha256); 27000 files checked
  OK: snapshot matches its records
[exit code 0; wall time 427.6 s]
```

Duplicate groups at the import default (from `report.json`):

```
near  labels=['River'] members=['River/River_322.jpg', 'River/River_815.jpg']
near  labels=['SeaLake'] members=['SeaLake/SeaLake_1437.jpg', 'SeaLake/SeaLake_2683.jpg']
near  labels=['Highway'] members=['Highway/Highway_291.jpg', 'Highway/Highway_513.jpg']
near  labels=['AnnualCrop'] members=['AnnualCrop/AnnualCrop_2735.jpg', 'AnnualCrop/AnnualCrop_474.jpg']
near  labels=['River'] members=['River/River_492.jpg', 'River/River_597.jpg']
near  labels=['SeaLake'] members=['SeaLake/SeaLake_1569.jpg', 'SeaLake/SeaLake_2723.jpg']
near  labels=['AnnualCrop', 'River'] members=['AnnualCrop/AnnualCrop_2470.jpg', 'River/River_1922.jpg']
near  labels=['River'] members=['River/River_111.jpg', 'River/River_1680.jpg']
```

**Low-information summary:** 3357 images. By reason (overlapping): std 3302, degenerate 211,
popcount 1064. They are concentrated in **SeaLake (2327 of 3000)** and **Forest (1019 of
3000)**; Pasture has 8, AnnualCrop 3, and every other class 0. These images are excluded from
dHash grouping.

## 7. Calibration evidence (no decision)

Nine splits were made, all with seed 0 and fractions 0.7/0.15/0.15. Two requested variants
were **skipped as redundant**: popcount [8,56] and std 3.0 at threshold 5 are exactly the
defaults, i.e. run "t5 (default)". The table is built from the written manifests by
`<scratchpad>/d1/calibration_table.py`. Notes on the columns:
- "multi-member groups" counts kept groups of size ≥ 2 plus label-conflict groups;
- "files" counts their members;
- "oversized" uses the rule in §4.

| run | dHash thr | std | popcount | multi-member groups (files) | low-info images | label-conflict groups (files excluded) | oversized groups | train / val / test |
|---|---|---|---|---|---|---|---|---|
| t0 (`sp-de74e9eb6716`) | 0 | 3.0 | [8, 56] | 0 (0) | 3357 | 0 (0) | 0 | 18900 / 4050 / 4050 |
| t2 (`sp-46faf9061e5d`) | 2 | 3.0 | [8, 56] | 3 (6) | 3357 | 0 (0) | 0 | 18900 / 4050 / 4050 |
| t3 (`sp-c62c555820af`) | 3 | 3.0 | [8, 56] | 3 (6) | 3357 | 0 (0) | 0 | 18900 / 4050 / 4050 |
| t5 (default) (`sp-55320e11726c`) | 5 | 3.0 | [8, 56] | 8 (16) | 3357 | 1 (2) | 0 | 18898 / 4050 / 4050 |
| t8 (`sp-9212cd21f7c8`) | 8 | 3.0 | [8, 56] | 38 (92) | 3357 | 14 (38) | 0 | 18872 / 4046 / 4044 |
| t5 pop[4,60] (`sp-6946a445b35d`) | 5 | 3.0 | [4, 60] | 13 (27) | 3320 | 1 (2) | 0 | 18898 / 4050 / 4050 |
| t5 pop[12,52] (`sp-d3bbfdff4ffc`) | 5 | 3.0 | [12, 52] | 7 (14) | 3463 | 1 (2) | 0 | 18898 / 4050 / 4050 |
| t5 std1.5 (`sp-328ba84bdf47`) | 5 | 1.5 | [8, 56] | 8 (16) | 1787 | 1 (2) | 0 | 18898 / 4050 / 4050 |
| t5 std6.0 (`sp-dd53c5615dad`) | 5 | 6.0 | [8, 56] | 6 (12) | 5880 | 1 (2) | 0 | 18898 / 4050 / 4050 |

Largest group per class (size; val target in parentheses):

| run | AnnualCrop | Forest | HerbaceousVegetation | Highway | Industrial | Pasture | PermanentCrop | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | 1 (450.0) | 1 (450.0) | 1 (450.0) | 1 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 1 (375.0) | 1 (450.0) |
| t2 | 2 (450.0) | 1 (450.0) | 1 (450.0) | 1 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (375.0) | 2 (450.0) |
| t3 | 2 (450.0) | 1 (450.0) | 1 (450.0) | 1 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (375.0) | 2 (450.0) |
| t5 (default) | 2 (449.85) | 1 (450.0) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (374.85) | 2 (450.0) |
| t8 | 2 (448.8) | 1 (449.4) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (299.55) | 1 (375.0) | 1 (449.85) | 2 (373.95) | 4 (447.75) |
| t5 pop[4,60] | 2 (449.85) | 1 (450.0) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (374.85) | 3 (450.0) |
| t5 pop[12,52] | 2 (449.85) | 1 (450.0) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (374.85) | 2 (450.0) |
| t5 std1.5 | 2 (449.85) | 1 (450.0) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (374.85) | 2 (450.0) |
| t5 std6.0 | 2 (449.85) | 1 (450.0) | 1 (450.0) | 2 (375.0) | 1 (375.0) | 1 (300.0) | 1 (375.0) | 1 (450.0) | 2 (374.85) | 1 (450.0) |

Per-class split counts (train/val/test):

| run | AnnualCrop | Forest | HerbaceousVegetation | Highway | Industrial | Pasture | PermanentCrop | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | 2100/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1750/375/375 | 2100/450/450 |
| t2 | 2100/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1750/375/375 | 2100/450/450 |
| t3 | 2100/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1750/375/375 | 2100/450/450 |
| t5 (default) | 2099/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1749/375/375 | 2100/450/450 |
| t8 | 2094/449/449 | 2097/450/449 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1398/300/299 | 1750/375/375 | 2099/450/450 | 1745/374/374 | 2089/448/448 |
| t5 pop[4,60] | 2099/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1749/375/375 | 2100/450/450 |
| t5 pop[12,52] | 2099/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1749/375/375 | 2100/450/450 |
| t5 std1.5 | 2099/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1749/375/375 | 2100/450/450 |
| t5 std6.0 | 2099/450/450 | 2100/450/450 | 2100/450/450 | 1750/375/375 | 1750/375/375 | 1400/300/300 | 1750/375/375 | 2100/450/450 | 1749/375/375 | 2100/450/450 |

The full `dataset split` output of all nine runs is in Appendix A.

## 8. Audits for thresholds 3 and 5 (verbatim)

```
> visionbench dataset audit-split ds-61112928e412 sp-c62c555820af
audit of ds-61112928e412 / sp-c62c555820af (seed 0)
decoding 27000 images (pHash + decoded-pixel digest)
  decoded 5000/27000 images
  decoded 10000/27000 images
  decoded 15000/27000 images
  decoded 20000/27000 images
  decoded 25000/27000 images
rendering contact sheets
  output: <repo>\var\datasets\ds-61112928e412\audits\sp-c62c555820af
  images: 27000 (train 18900, val 4050, test 4050); low information 3357 (val/test 1005)
  dHash threshold 3:
    consistency check: OK (0 informative val/test images within the threshold of an informative train image)
    val/test images whose nearest train image (low-info included) is within the threshold: 236 {'query_informative_neighbour_informative': 0, 'query_informative_neighbour_low_info': 0, 'query_low_info_neighbour_informative': 0, 'query_low_info_neighbour_low_info': 236}
    near-miss band 4..6: 3 val/test images; 3 cross-split pairs
  nearest-train dhash distance, informative: 0:0 1:0 2:0 3:0 4:1 5:4 6:7 7:8 8:31 9:33 10:61 11:124 12:241 13:602 14:1165 15:1802 16:1882 >16:1134
  nearest-train dhash distance, low_information: 0:99 1:45 2:42 3:50 4:50 5:64 6:29 7:50 8:47 9:46 10:50 11:59 12:60 13:78 14:87 15:83 16:53 >16:13
  nearest-train phash distance, informative: 0:2 1:0 2:1 3:0 4:4 5:0 6:9 7:0 8:30 9:0 10:90 11:0 12:376 13:0 14:1855 15:0 16:4131 >16:597
  nearest-train phash distance, low_information: 0:0 1:0 2:1 3:0 4:6 5:0 6:9 7:0 8:13 9:0 10:22 11:0 12:48 13:0 14:221 15:0 16:594 >16:91
  exact sha256 matches across splits: 0; decoded-pixel-identical: 0
  group sizes: {'1': 26994, '2': 3}
  oversized groups: 0
  pairs: {'a_closest_phash': 60, 'b_within_group_by_distance': {'1': 1, '2': 2, '3': 0}, 'c_near_miss': 3, 'd_low_info_samples': {'AnnualCrop': 3, 'Forest': 20, 'HerbaceousVegetation': 0, 'Highway': 0, 'Industrial': 0, 'Pasture': 8, 'PermanentCrop': 0, 'Residential': 0, 'River': 0, 'SeaLake': 20}}
  contact sheets: 10 in <repo>\var\datasets\ds-61112928e412\audits\sp-c62c555820af\sheets
[exit code 0; wall time 31.1 s]

> visionbench dataset audit-split ds-61112928e412 sp-55320e11726c
audit of ds-61112928e412 / sp-55320e11726c (seed 0)
decoding 26998 images (pHash + decoded-pixel digest)
  decoded 5000/26998 images
  decoded 10000/26998 images
  decoded 15000/26998 images
  decoded 20000/26998 images
  decoded 25000/26998 images
rendering contact sheets
  output: <repo>\var\datasets\ds-61112928e412\audits\sp-55320e11726c
  images: 26998 (train 18898, val 4050, test 4050); low information 3357 (val/test 994)
  dHash threshold 5:
    consistency check: OK (0 informative val/test images within the threshold of an informative train image)
    val/test images whose nearest train image (low-info included) is within the threshold: 356 {'query_informative_neighbour_informative': 0, 'query_informative_neighbour_low_info': 4, 'query_low_info_neighbour_informative': 0, 'query_low_info_neighbour_low_info': 352}
    near-miss band 6..8: 11 val/test images; 12 cross-split pairs
  nearest-train dhash distance, informative: 0:0 1:0 2:0 3:0 4:0 5:4 6:4 7:4 8:22 9:40 10:62 11:118 12:249 13:616 14:1196 15:1774 16:1885 >16:1132
  nearest-train dhash distance, low_information: 0:110 1:52 2:35 3:48 4:51 5:56 6:39 7:53 8:44 9:53 10:43 11:52 12:65 13:69 14:84 15:76 16:51 >16:13
  nearest-train phash distance, informative: 0:1 1:0 2:1 3:0 4:2 5:0 6:10 7:0 8:22 9:0 10:88 11:0 12:398 13:0 14:1852 15:0 16:4121 >16:611
  nearest-train phash distance, low_information: 0:0 1:0 2:3 3:0 4:4 5:0 6:12 7:0 8:12 9:0 10:21 11:0 12:41 13:0 14:243 15:0 16:561 >16:97
  exact sha256 matches across splits: 0; decoded-pixel-identical: 0
  group sizes: {'1': 26984, '2': 7}
  oversized groups: 0
  pairs: {'a_closest_phash': 60, 'b_within_group_by_distance': {'1': 1, '2': 2, '3': 0, '4': 1, '5': 3}, 'c_near_miss': 12, 'd_low_info_samples': {'AnnualCrop': 3, 'Forest': 20, 'HerbaceousVegetation': 0, 'Highway': 0, 'Industrial': 0, 'Pasture': 8, 'PermanentCrop': 0, 'Residential': 0, 'River': 0, 'SeaLake': 20}}
  contact sheets: 12 in <repo>\var\datasets\ds-61112928e412\audits\sp-55320e11726c\sheets
[exit code 0; wall time 30.0 s]
```

### audit.md, threshold 3 (`sp-c62c555820af`)

````markdown
# Split audit: ds-61112928e412 / sp-c62c555820af

Created 2026-10-27T18:56:53+00:00 by VisionBench 0.0.1; seed 0.
Split config: dHash threshold 3, low-info std 3.0, popcount [8, 56], fractions [0.7, 0.15, 0.15], seed 0.
Methods: pHash phash-v1 (independent of the grouping), dHash (recomputed over all images), orientation exif_transposed.

## Counts

- entries 27000 (train 18900, val 4050, test 4050); excluded label-conflict files 0
- low-information images 3357 (of which val/test 1005)

## Nearest train image of each val/test image

dHash distance:

| query images | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | >16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| informative | 0 | 0 | 0 | 0 | 1 | 4 | 7 | 8 | 31 | 33 | 61 | 124 | 241 | 602 | 1165 | 1802 | 1882 | 1134 |
| low_information | 99 | 45 | 42 | 50 | 50 | 64 | 29 | 50 | 47 | 46 | 50 | 59 | 60 | 78 | 87 | 83 | 53 | 13 |

pHash distance:

| query images | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | >16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| informative | 2 | 0 | 1 | 0 | 4 | 0 | 9 | 0 | 30 | 0 | 90 | 0 | 376 | 0 | 1855 | 0 | 4131 | 597 |
| low_information | 0 | 0 | 1 | 0 | 6 | 0 | 9 | 0 | 13 | 0 | 22 | 0 | 48 | 0 | 221 | 0 | 594 | 91 |

## Within the dHash threshold (3)

- consistency check (informative vs informative, must be 0): **0** -> OK
- val/test images whose nearest train image (low-info included) is within the threshold: 236
  - query informative neighbour informative: 0
  - query informative neighbour low info: 0
  - query low info neighbour informative: 0
  - query low info neighbour low info: 236
- val/test images whose nearest pHash train distance is <= 3: 4

## Near-miss band (dHash 4..6, informative images)

- val/test images whose nearest informative train image is in the band: 3
- all cross-split (val/test x train) pairs in the band: 3 (by distance: {'4': 1, '6': 2})

## Exact and pixel-identical matches across splits

- exact SHA-256 matches across splits: **0** (must be 0)
- decoded-pixel-identical matches across splits: **0**

## Groups

Group-size distribution (size: groups): 1: 26994, 2: 3

| class | images | largest group | val target | test target |
|---|---|---|---|---|
| AnnualCrop | 3000 | 2 | 450.0 | 450.0 |
| Forest | 3000 | 1 | 450.0 | 450.0 |
| HerbaceousVegetation | 3000 | 1 | 450.0 | 450.0 |
| Highway | 2500 | 1 | 375.0 | 375.0 |
| Industrial | 2500 | 1 | 375.0 | 375.0 |
| Pasture | 2000 | 1 | 300.0 | 300.0 |
| PermanentCrop | 2500 | 1 | 375.0 | 375.0 |
| Residential | 3000 | 1 | 450.0 | 450.0 |
| River | 2500 | 2 | 375.0 | 375.0 |
| SeaLake | 3000 | 2 | 450.0 | 450.0 |

Oversized groups (> val target or > 2% of the class): 0

## Within-group pairs by dHash distance (informative members)

d=1: 1, d=2: 2, d=3: 0

## Pairs for visual inspection

- (a) closest cross-split pairs by pHash: 60
- (b) within-group pairs per dHash distance: {'1': 1, '2': 2, '3': 0}
- (c) near-miss cross-split pairs: 3
- (d) low-information samples per class: {'AnnualCrop': 3, 'Forest': 20, 'HerbaceousVegetation': 0, 'Highway': 0, 'Industrial': 0, 'Pasture': 8, 'PermanentCrop': 0, 'Residential': 0, 'River': 0, 'SeaLake': 20}

Contact sheets: sheets/a_closest_phash_01.png, sheets/a_closest_phash_02.png, sheets/a_closest_phash_03.png, sheets/b_within_group_d1_01.png, sheets/b_within_group_d2_01.png, sheets/c_near_miss_01.png, sheets/d_low_info_AnnualCrop.png, sheets/d_low_info_Forest.png, sheets/d_low_info_Pasture.png, sheets/d_low_info_SeaLake.png
````

### audit.md, threshold 5 (`sp-55320e11726c`)

````markdown
# Split audit: ds-61112928e412 / sp-55320e11726c

Created 2026-10-27T18:57:23+00:00 by VisionBench 0.0.1; seed 0.
Split config: dHash threshold 5, low-info std 3.0, popcount [8, 56], fractions [0.7, 0.15, 0.15], seed 0.
Methods: pHash phash-v1 (independent of the grouping), dHash (recomputed over all images), orientation exif_transposed.

## Counts

- entries 26998 (train 18898, val 4050, test 4050); excluded label-conflict files 2
- low-information images 3357 (of which val/test 994)

## Nearest train image of each val/test image

dHash distance:

| query images | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | >16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| informative | 0 | 0 | 0 | 0 | 0 | 4 | 4 | 4 | 22 | 40 | 62 | 118 | 249 | 616 | 1196 | 1774 | 1885 | 1132 |
| low_information | 110 | 52 | 35 | 48 | 51 | 56 | 39 | 53 | 44 | 53 | 43 | 52 | 65 | 69 | 84 | 76 | 51 | 13 |

pHash distance:

| query images | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | >16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| informative | 1 | 0 | 1 | 0 | 2 | 0 | 10 | 0 | 22 | 0 | 88 | 0 | 398 | 0 | 1852 | 0 | 4121 | 611 |
| low_information | 0 | 0 | 3 | 0 | 4 | 0 | 12 | 0 | 12 | 0 | 21 | 0 | 41 | 0 | 243 | 0 | 561 | 97 |

## Within the dHash threshold (5)

- consistency check (informative vs informative, must be 0): **0** -> OK
- val/test images whose nearest train image (low-info included) is within the threshold: 356
  - query informative neighbour informative: 0
  - query informative neighbour low info: 4
  - query low info neighbour informative: 0
  - query low info neighbour low info: 352
- val/test images whose nearest pHash train distance is <= 5: 11

## Near-miss band (dHash 6..8, informative images)

- val/test images whose nearest informative train image is in the band: 11
- all cross-split (val/test x train) pairs in the band: 12 (by distance: {'6': 2, '7': 2, '8': 8})

## Exact and pixel-identical matches across splits

- exact SHA-256 matches across splits: **0** (must be 0)
- decoded-pixel-identical matches across splits: **0**

## Groups

Group-size distribution (size: groups): 1: 26984, 2: 7

| class | images | largest group | val target | test target |
|---|---|---|---|---|
| AnnualCrop | 2999 | 2 | 449.85 | 449.85 |
| Forest | 3000 | 1 | 450.0 | 450.0 |
| HerbaceousVegetation | 3000 | 1 | 450.0 | 450.0 |
| Highway | 2500 | 2 | 375.0 | 375.0 |
| Industrial | 2500 | 1 | 375.0 | 375.0 |
| Pasture | 2000 | 1 | 300.0 | 300.0 |
| PermanentCrop | 2500 | 1 | 375.0 | 375.0 |
| Residential | 3000 | 1 | 450.0 | 450.0 |
| River | 2499 | 2 | 374.85 | 374.85 |
| SeaLake | 3000 | 2 | 450.0 | 450.0 |

Oversized groups (> val target or > 2% of the class): 0

## Within-group pairs by dHash distance (informative members)

d=1: 1, d=2: 2, d=3: 0, d=4: 1, d=5: 3

## Pairs for visual inspection

- (a) closest cross-split pairs by pHash: 60
- (b) within-group pairs per dHash distance: {'1': 1, '2': 2, '3': 0, '4': 1, '5': 3}
- (c) near-miss cross-split pairs: 12
- (d) low-information samples per class: {'AnnualCrop': 3, 'Forest': 20, 'HerbaceousVegetation': 0, 'Highway': 0, 'Industrial': 0, 'Pasture': 8, 'PermanentCrop': 0, 'Residential': 0, 'River': 0, 'SeaLake': 20}

Contact sheets: sheets/a_closest_phash_01.png, sheets/a_closest_phash_02.png, sheets/a_closest_phash_03.png, sheets/b_within_group_d1_01.png, sheets/b_within_group_d2_01.png, sheets/b_within_group_d4_01.png, sheets/b_within_group_d5_01.png, sheets/c_near_miss_01.png, sheets/d_low_info_AnnualCrop.png, sheets/d_low_info_Forest.png, sheets/d_low_info_Pasture.png, sheets/d_low_info_SeaLake.png
````

## 9. My observations from the contact sheets (observations only, not a decision)

**Sheets inspected:** for threshold 5, `a_closest_phash_01`, `b_within_group_d1/d2/d4/d5`,
`c_near_miss_01`, `d_low_info_Forest` and `d_low_info_SeaLake`. Also the excluded
label-conflict pair, rendered separately into the scratchpad (not in the audit directory).
`a_closest_phash_02/03` and the threshold-3 sheets were **not** opened by me. The threshold-3
b-sheets contain the same d1/d2 pairs as threshold 5.

What I saw:
- **Pairs grouped by dHash, (b):**
  - Four look like true near-duplicates of the same scene: River_111 / River_1680 (d=2),
    River_322 / River_815 (d=4), Highway_291 / Highway_513 (d=5) and River_492 / River_597
    (d=5). All four have pHash 2–8.
  - Three look like different scenes:
    - AnnualCrop_2735 / AnnualCrop_474 (d=1; pHash 28; striped fields);
    - SeaLake_1569 / SeaLake_2723 (d=2; pHash 20);
    - SeaLake_1437 / SeaLake_2683 (d=5; pHash 34; plain water).
  - So in my view false positives appear even at distance 1–2 on smooth or striped scenes.
- **The only label-conflict group at threshold 5** (AnnualCrop_2470 / River_1922; dHash 4,
  pHash 30) looks like a field next to a river: different scenes, which I read as a dHash
  false positive. It is excluded from the threshold-5 split. At threshold 3 the same two
  images sit in different splits (near-miss pair c-002).
- **Threshold 3:** the four true near-duplicates at d=4–5 are not grouped. With seed 0 they
  happened to land in the same split: they do not appear among the 3 cross-split near-miss
  pairs. Nothing at threshold 3 guarantees that for other seeds.
- **Near misses at threshold 5 (dHash 6–8, sheet c):** none of the 12 pairs looked to me like
  the same location. Most are different scenes with similar flat texture.
- **Closest cross-split pHash pairs (sheet a_01):** dominated by near-uniform dark water and
  forest tiles from different classes or places, e.g. Forest_1552 vs River_1476 at pHash 0. I
  saw no pair that looked like the same image. pHash distance ≤ 4 here reflects low texture,
  not duplication.
- **Low-information samples:** Forest and SeaLake samples are nearly uniform canopy or water
  at 64 px, so their exclusion from dHash matching looks justified. Their dHash is noise-driven:
  110 of the 994 low-info val/test images at threshold 5 have a dHash distance of **0** to some
  low-info train image. Their pHash is not near: no low-info query has pHash 0.

These are visual impressions from a limited number of sheets, not measurements. The threshold
choice is the supervisor's.

## 10. Contact sheets (for the owner to attach)

All under `var/datasets/ds-61112928e412/audits/` (gitignored; they contain dataset images):

```
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/a_closest_phash_01.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/a_closest_phash_02.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/a_closest_phash_03.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/b_within_group_d1_01.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/b_within_group_d2_01.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/c_near_miss_01.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/d_low_info_AnnualCrop.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/d_low_info_Forest.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/d_low_info_Pasture.png
var/datasets/ds-61112928e412/audits/sp-c62c555820af/sheets/d_low_info_SeaLake.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/a_closest_phash_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/a_closest_phash_02.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/a_closest_phash_03.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/b_within_group_d1_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/b_within_group_d2_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/b_within_group_d4_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/b_within_group_d5_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/c_near_miss_01.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/d_low_info_AnnualCrop.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/d_low_info_Forest.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/d_low_info_Pasture.png
var/datasets/ds-61112928e412/audits/sp-55320e11726c/sheets/d_low_info_SeaLake.png
```

## 11. Lint and tests (verbatim)

```
> .\.venv\Scripts\ruff.exe check .
All checks passed!
> .\.venv\Scripts\ruff.exe format --check .
99 files already formatted
```

Normal run (`.\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 25%]
.......................................................s................ [ 50%]
........................................................................ [ 75%]
......................................................................   [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=24, max=16.6 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 2.74 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 11.09 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=True; CUDA_VISIBLE_DEVICES=None
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

[... per-file coverage table; data modules in the next block ...]
TOTAL                                        4670    212   1070    129    94%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
285 passed, 1 skipped, 1 warning in 511.76s (0:08:31)
```

GPU hidden (`$env:CUDA_VISIBLE_DEVICES="-1"; .\.venv\Scripts\pytest.exe -q --cov=visionbench`):

```
........................................................................ [ 25%]
.......................................................s................ [ 50%]
........................................................................ [ 75%]
......................................................................   [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  <repo>\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[measured] GET /api/health while training: n=45, max=55.0 ms
PERF near-duplicate search, 20,000 random hashes, threshold 5, np.bitwise_count: 1.64 s, 1 edge(s)
PERF near-duplicate search, 20,000 random hashes, threshold 5, byte LUT: 9.30 s, 1 edge(s)
torch 2.14.1+cu130; torch.cuda.is_available()=False; CUDA_VISIBLE_DEVICES='-1'
=============================== tests coverage ================================
______________ coverage: platform win32, python 3.11.15-final-0 _______________

[... per-file coverage table; data modules in the next block ...]
TOTAL                                        4670    212   1070    129    94%
=========================== short test summary info ===========================
SKIPPED [1] tests\data\test_scan.py:34: cannot create file symlinks here (Windows without symlink privilege / Developer Mode); directory links are covered via an NTFS junction
285 passed, 1 skipped, 1 warning in 539.21s (0:08:59)
```

Coverage of the new and changed data modules (normal run):

```
Name                                        Stmts   Miss Branch BrPart  Cover
-----------------------------------------------------------------------------
src\visionbench\cli.py                        394     37     72      7    89%
src\visionbench\data\__init__.py                0      0      0      0   100%
src\visionbench\data\archive.py               207     12     76      7    93%
src\visionbench\data\audit.py                 308     15     82      9    94%
src\visionbench\data\grouping.py               94      1     26      1    98%
src\visionbench\data\hashing.py                53      0     18      0   100%
src\visionbench\data\imageio.py                21      0      0      0   100%
src\visionbench\data\phash.py                  18      0      2      0   100%
src\visionbench\data\records.py                51      1      0      0    98%
src\visionbench\data\report.py                 37      0     12      0   100%
src\visionbench\data\scan.py                  203      9     68      7    94%
src\visionbench\data\split.py                 173      5     58      5    96%
src\visionbench\data\store.py                 139      2     34      2    98%
src\visionbench\training\data.py               52      3      2      0    94%
-----------------------------------------------------------------------------
```

## 12. Deviations and decisions to review

1. **Low-information breakdown added to the report** (report format 2). Step 4 asks for it and
   the M1 report did not have it. The reasons overlap, so per-reason counts do not sum to the
   total.
2. **Oversized groups count only multi-member groups.** Taken literally, ">2 % of its class"
   would flag every singleton in classes under 50 images.
3. **pHash median excludes DC** (63 AC values), and the DC bit is fixed at 0. The task said
   "top-left 8x8 without DC → median threshold", which leaves 63 informative bits in a 64-bit
   hash. A consequence is that all distances are even (§4).
4. **Pair set (a) uses each val/test image's nearest train image.** Val-vs-test pairs are not
   searched for near-duplicates, but they are included in the exact and pixel-match counts.
5. **Within-group pairs (b) are all informative member pairs at distance d**, not only the
   union-find edges. With pairs (no larger groups) on EuroSAT, these are the same.
6. **A dataset re-imported in format 2 keeps its id** if the files and name are unchanged,
   because the id is derived from the fingerprint and name. The old snapshot must be removed
   first, and the error message says so.
7. **Rotated-copy test uses a lossless PNG** for exact pixel equality (§3).
8. **`scipy` declared as a direct dependency.** It was already installed via scikit-learn and
   is in the lock file.
9. **Import timing.** 466.2 s, and `verify --full` took 427.6 s for 92 MB. Both are dominated
   by per-file I/O on this Windows machine. I did not investigate further.

## 13. Not verified / limitations

- POSIX was not run. CI has never run.
- The audit compares val/test against train only. It does not detect near-duplicates *within*
  val or test, or between val and test, by perceptual hash.
- pHash and dHash both miss crops, flips, rotations and strong colour changes. On near-uniform
  tiles both hashes are dominated by noise. Low-information Forest and SeaLake images therefore
  cannot be checked for near-duplicates by hashing; only by exact and pixel matches.
- Contact-sheet observations (§9) cover a subset of sheets and are subjective.
- The Copernicus `[Year]` in the attribution is unresolved (§5).

## 14. Code printed for review

The final message of this task prints:
- this packet (it contains the calibration table and both audit.md files);
- DATASETS.md;
- the full source of `src/visionbench/data/phash.py` and `src/visionbench/data/audit.py`.

## Appendix A. `dataset split` output of the nine calibration runs (verbatim)

```
> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 0
split sp-de74e9eb6716 for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-de74e9eb6716\manifest.json
  manifest sha256: 5b77dea1facba7df0bebe439aaa554228f299ecd441f34d2993f65745b2b6c3a
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2100/450/450/3000
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1750/375/375/2500
    SeaLake: 2100/450/450/3000
  totals: train=18900, val=4050, test=4050
  excluded (label conflicts): 0 files in 0 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 27000 groups, 27000 sha256 values)
[exit code 0; wall time 7.1 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 2
split sp-46faf9061e5d for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-46faf9061e5d\manifest.json
  manifest sha256: 886ea36979c861e08c231c72c3a07f3908e8b49fcfe764e5c2fed95105332419
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2100/450/450/3000
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1750/375/375/2500
    SeaLake: 2100/450/450/3000
  totals: train=18900, val=4050, test=4050
  excluded (label conflicts): 0 files in 0 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26997 groups, 27000 sha256 values)
[exit code 0; wall time 7.1 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 3
split sp-c62c555820af for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-c62c555820af\manifest.json
  manifest sha256: 5ebf2da5c7f37b216f29228a8c4def3e8bca0aeaa4a8d91eaf316c4a8f6ccfea
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2100/450/450/3000
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1750/375/375/2500
    SeaLake: 2100/450/450/3000
  totals: train=18900, val=4050, test=4050
  excluded (label conflicts): 0 files in 0 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26997 groups, 27000 sha256 values)
[exit code 0; wall time 6.3 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 5
split sp-55320e11726c for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-55320e11726c\manifest.json
  manifest sha256: b0058d9eba3e86e8bf2d68bd719599327e0a43860fbd88b9e56a56e5e41a79fb
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2099/450/450/2999
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1749/375/375/2499
    SeaLake: 2100/450/450/3000
  totals: train=18898, val=4050, test=4050
  excluded (label conflicts): 2 files in 1 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26991 groups, 26998 sha256 values)
[exit code 0; wall time 5.9 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 8
split sp-9212cd21f7c8 for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-9212cd21f7c8\manifest.json
  manifest sha256: a7a1da8dad0ef742620e5b72d0ee3268c0bc624d64bd1ad1adc9ce1ba5ddc374
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2094/449/449/2992
    Forest: 2097/450/449/2996
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1398/300/299/1997
    PermanentCrop: 1750/375/375/2500
    Residential: 2099/450/450/2999
    River: 1745/374/374/2493
    SeaLake: 2089/448/448/2985
  totals: train=18872, val=4046, test=4044
  excluded (label conflicts): 38 files in 14 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26932 groups, 26962 sha256 values)
[exit code 0; wall time 6.2 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 5 --low-info-popcount-min 4 --low-info-popcount-max 60
split sp-6946a445b35d for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-6946a445b35d\manifest.json
  manifest sha256: 4c43c76e697dc2aa11ba879da924cd1b9834b58141d1b89df3019f873aa2a4cb
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2099/450/450/2999
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1749/375/375/2499
    SeaLake: 2100/450/450/3000
  totals: train=18898, val=4050, test=4050
  excluded (label conflicts): 2 files in 1 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26985 groups, 26998 sha256 values)
[exit code 0; wall time 5.8 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 5 --low-info-popcount-min 12 --low-info-popcount-max 52
split sp-d3bbfdff4ffc for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-d3bbfdff4ffc\manifest.json
  manifest sha256: 419d487678c6e31091ea7a8701cf1eaf8a4d57d87ae1cdeea38bdbbe14df99a9
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2099/450/450/2999
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1749/375/375/2499
    SeaLake: 2100/450/450/3000
  totals: train=18898, val=4050, test=4050
  excluded (label conflicts): 2 files in 1 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26992 groups, 26998 sha256 values)
[exit code 0; wall time 5.8 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 5 --low-info-std 1.5
split sp-328ba84bdf47 for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-328ba84bdf47\manifest.json
  manifest sha256: 5af68f525c55352fdd67de06ee07e5c27673eb280139b1444d910b880392034b
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2099/450/450/2999
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1749/375/375/2499
    SeaLake: 2100/450/450/3000
  totals: train=18898, val=4050, test=4050
  excluded (label conflicts): 2 files in 1 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26991 groups, 26998 sha256 values)
[exit code 0; wall time 6.1 s]

> visionbench dataset split ds-61112928e412 --seed 0 --fractions 0.7 0.15 0.15 --dhash-threshold 5 --low-info-std 6.0
split sp-dd53c5615dad for dataset ds-61112928e412: new
  manifest: <repo>\var\datasets\ds-61112928e412\splits\sp-dd53c5615dad\manifest.json
  manifest sha256: cffae732ffa7844cb967ac4e669b64a227eaf18bcba601c0cfb8692b64b6a314
  seed=0 fractions=[0.7, 0.15, 0.15] conflict_policy=exclude
  counts (train/val/test/total):
    AnnualCrop: 2099/450/450/2999
    Forest: 2100/450/450/3000
    HerbaceousVegetation: 2100/450/450/3000
    Highway: 1750/375/375/2500
    Industrial: 1750/375/375/2500
    Pasture: 1400/300/300/2000
    PermanentCrop: 1750/375/375/2500
    Residential: 2100/450/450/3000
    River: 1749/375/375/2499
    SeaLake: 2100/450/450/3000
  totals: train=18898, val=4050, test=4050
  excluded (label conflicts): 2 files in 1 groups
  oversized groups: 0 (multi-member groups > the class's val target or > 2% of the class; recorded in split_info.json)
  leakage check: groups_in_multiple_splits=0 sha256_in_multiple_splits=0 (checked 26993 groups, 26998 sha256 values)
[exit code 0; wall time 5.5 s]
```

