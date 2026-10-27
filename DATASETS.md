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

**Open point:** neither the Zenodo record nor the README states the acquisition year(s). The
`[Year]` in the notice is left for the supervisor to fill from the paper.

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
> (<https://zenodo.org/records/7711810>). Contains modified Copernicus Sentinel data [Year].
> Please cite Helber et al., IEEE JSTARS 2019 and IGARSS 2018.

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
