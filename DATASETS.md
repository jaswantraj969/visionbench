# Datasets

This file records every real dataset used by VisionBench: where it came from, under which
terms, and how it was obtained. **Dataset files are never committed.** They live under the data
root (`var/`, gitignored).

## EuroSAT (RGB)

### Source

| | |
|---|---|
| Dataset | EuroSAT: Land Use and Land Cover Classification with Sentinel-2, RGB version |
| Authors | Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth |
| Official distribution | Zenodo record [10.5281/zenodo.7711810](https://zenodo.org/records/7711810) (published 2018-07-22, labelled v2), linked from the authors' repository |
| Authors' repository | <https://github.com/phelber/EuroSAT> |
| File used | `EuroSAT_RGB.zip`, from `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1` |

The authors' README lists the former host, `madm.dfki.de`, under "Deprecated Hosting". It was
not used.

### License: dataset files

- **MIT License.** The Zenodo record's license field is `mit-license`, read from
  `https://zenodo.org/api/records/7711810` on 2026-10-27. The record page displays "MIT
  License" and links to <https://opensource.org/licenses/MIT>.
- The authors' README states: "The dataset is licensed under the MIT license."
- The repository's `LICENSE` file is the MIT License ("Copyright (c) 2023 Patrick Helber").

**Code vs data:** the GitHub repository contains documentation, not code. Both the README and
the Zenodo record apply MIT to the dataset itself. No separate data license was found.

One third-party metadata index (Knowledge Pixels) lists CC BY 4.0 for the same DOI. I treated
the record's own metadata as authoritative.

### Underlying Sentinel-2 data: Copernicus terms

The images are derived from Copernicus Sentinel-2 imagery. The authors ask users to "consider
the Copernicus Sentinel Data Terms and Conditions". Their link
(`sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice`) no longer resolves
(DNS failure on 2026-10-27). The same licence is published as the "Copernicus Sentinel data
licence (rev. 1)" at <https://ads.atmosphere.copernicus.eu/licences/ec-sentinel>.

The licence grants free use for "reproduction; distribution; communication to the public;
adaptation, modification and combination with other data and information". It requires one of
two notices:
- `Copernicus Sentinel data [Year]` when the data is communicated or distributed;
- `Contains modified Copernicus Sentinel data [Year]` when it has been adapted or modified.

**Acquisition year: not stated by the dataset authors.** The authors' paper (arXiv:1709.00029v2,
section III-A "Satellite Image Acquisition", p. 4) says only that the images were taken by
Sentinel-2A and downloaded via Amazon S3. It also says: "we have extracted images recorded all
over the year". It gives no year or acquisition period, and neither do the Zenodo record or the
README. The notice is therefore used **without a year** (below). No year is inferred from the
launch date or the submission date.

### Required citation

The authors ask users to cite both papers:

1. P. Helber, B. Bischke, A. Dengel, D. Borth, "EuroSAT: A Novel Dataset and Deep Learning
   Benchmark for Land Use and Land Cover Classification", IEEE Journal of Selected Topics in
   Applied Earth Observations and Remote Sensing, 2019. DOI 10.1109/JSTARS.2019.2918242.
2. P. Helber, B. Bischke, A. Dengel, D. Borth, "Introducing EuroSAT: A Novel Dataset and Deep
   Learning Benchmark for Land Use and Land Cover Classification", IGARSS 2018. DOI
   10.1109/IGARSS.2018.8519248.

Both DOIs are the record's `isDescribedBy` relations on Zenodo.

### Attribution text to use

> EuroSAT dataset © Patrick Helber, used under the MIT License
> (<https://zenodo.org/records/7711810>). Contains modified Copernicus Sentinel data
> (acquisition year not stated by the dataset authors). Please cite Helber et al., IEEE JSTARS
> 2019 and IGARSS 2018.

### What this project does and does not do with it (public portfolio)

- **Allowed and done:**
  - download from the official source into the gitignored data root;
  - train and evaluate locally;
  - publish derived results (metrics, plots, model comparisons) with the citation and the
    attribution above;
  - show a few images in screenshots (for example of the UI or of a contact sheet) with the
    attribution. The MIT license permits copying and publishing with the copyright and
    permission notice; the Copernicus licence permits distribution with the notice.
- **Not done:**
  - **the images are not redistributed in this repository**: no image files, archives,
    contact sheets or snapshot directories are committed;
  - no mirrors or unofficial copies are used.
- Trained model weights are not committed either (project rule).

### Download record

| | |
|---|---|
| URL | `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1` (HTTP 200, no redirect) |
| Local path | `<data_root>/downloads/EuroSAT_RGB.zip` (`var/downloads/`, gitignored) |
| Size | 94,658,721 bytes (equal to the Zenodo API `size`) |
| MD5 | `f46e308c4d50d4bf32fedad2d3d62f3b`, **equal** to the checksum published by Zenodo |
| SHA-256 | `b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90` |
| Downloaded | 2026-10-27, 18:25:58Z → 18:37:35Z (`curl.exe`, 696.6 s) |
| Contents | 27,000 JPEG files (64x64 RGB) in 10 class folders under `EuroSAT_RGB/` |

The archive passed VisionBench's hardened ZIP validator (`visionbench.data.archive`). It was
extracted with the same module into `var/downloads/extracted/EuroSAT_RGB/` and imported from
there as dataset `ds-61112928e412` ("eurosat-rgb"). The fingerprint is
`53a853ae82ee65dd0359263e5a76d81889bd46f770f6224e58965c42637b852f`.

### Frozen benchmark split (decided by the supervisor, task VB-B1)

| | |
|---|---|
| Dataset | `ds-61112928e412` (fingerprint `53a853ae82ee65dd0359263e5a76d81889bd46f770f6224e58965c42637b852f`) |
| Split | **`sp-55320e11726c`**: dHash threshold 5, low-info std 3.0, popcount [8, 56], seed 0, fractions 0.7/0.15/0.15, label conflicts excluded (1 group, 2 files) |

**Rationale.** The supervisor chose the default configuration (dHash 5, std 3.0, popcount
[8, 56]) from the D1 calibration evidence. Among the configurations compared, it keeps the
pairs that looked like true near-duplicates of the same scene (dHash 2–5) in one group. It
excludes the single label-conflict pair and needs no oversized-group handling. Lower
thresholds leave the d=4–5 near-duplicates ungrouped. Threshold 8 adds 14 label-conflict
groups.

**The other eight D1 calibration splits are not for benchmarking:** `sp-de74e9eb6716`,
`sp-46faf9061e5d`, `sp-c62c555820af`, `sp-9212cd21f7c8`, `sp-6946a445b35d`, `sp-d3bbfdff4ffc`,
`sp-328ba84bdf47` and `sp-dd53c5615dad`. They exist only as calibration evidence.

### Known limitation: low-information images

At the frozen settings, **3,357 of 27,000 images (12.4 %)** are low-information (std, degenerate
or popcount test). They are concentrated in two classes:
- **SeaLake: 2,327 of 3,000 (78 %)**;
- **Forest: 1,019 of 3,000 (34 %)**;
- Pasture has 8, AnnualCrop 3, and every other class 0.

These images are nearly uniform water or canopy at 64 px. They are excluded from dHash
grouping, and the D1 audit showed that both perceptual hashes are noise-driven on them.
**Near-duplicates among them therefore cannot be assessed by hashing.** Only exact (SHA-256)
and decoded-pixel duplicates are ruled out (0 across splits). Results for SeaLake and Forest
should be read with this in mind.

**pHash is also unreliable on dark, low-texture tiles that pass the low-information test.**
In the D1 audit of the frozen split (`audits/sp-55320e11726c/pairs.csv`, pair `a-001`),
`Forest/Forest_1552.jpg` (val) and `River/River_1476.jpg` (train) have pHash distance **0** but
dHash distance 26, and they are visibly different images (dense canopy vs a river with a bank;
checked side by side, contrast-stretched, on 2026-10-29). Neither is flagged low-information
(recorded gray std 4.51 and 12.67; the threshold is 3.0). A pHash distance of 0 therefore does not establish a duplicate on such
tiles; grouping uses dHash only, and the audit's pHash search is evidence for review, not a
decision.
