"""Independent split audit (pHash + recomputed dHash), and oversized-group warnings."""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vb_synth import encode, make_images, synthetic_image

from visionbench.cli import main
from visionbench.core.jsonio import read_json
from visionbench.data.audit import (
    PER_SHEET,
    audit_split,
    histogram,
    nearest,
    pairs_in_band,
    pixel_digest,
)
from visionbench.data.hashing import hamming
from visionbench.data.imageio import decode_image
from visionbench.data.phash import phash, phash_from_gray
from visionbench.data.scan import inspect_image
from visionbench.data.split import SplitConfig, make_split
from visionbench.data.store import import_dataset, load_records

# --- pHash ----------------------------------------------------------------------------------


def _img(seed: int) -> Image.Image:
    return synthetic_image(np.random.default_rng(seed), size=64)


def _reload(data: bytes) -> Image.Image:
    return decode_image(data).image


def test_phash_matches_reencoded_and_resized_copies() -> None:
    for seed in range(8):
        img = _img(seed)
        h = phash(img)
        assert h >> 63 == 0  # the DC bit is never set
        jpeg = _reload(encode(img, "JPEG", quality=60))
        small = img.resize((40, 40), Image.Resampling.BILINEAR)
        big_jpeg = _reload(encode(img.resize((96, 96)), "JPEG", quality=80))
        for copy in (jpeg, small, big_jpeg):
            assert hamming(h, phash(copy)) <= 6, seed
    unrelated = [hamming(phash(_img(a)), phash(_img(b))) for a in range(8) for b in range(a)]
    assert min(unrelated) >= 14  # different images are far apart


def test_phash_input_validation() -> None:
    with pytest.raises(ValueError, match="32x32"):
        phash_from_gray(np.zeros((8, 8)))


def test_nearest_band_and_histogram() -> None:
    refs = np.array([0b0000, 0b1111, 0b0011], dtype=np.uint64)
    queries = np.array([0b0001, 0b0111, 0b1111_0000_0000], dtype=np.uint64)
    dist, idx = nearest(queries, refs, chunk_elements=2)  # forces several chunks
    assert dist.tolist() == [1, 1, 4] and idx.tolist() == [0, 1, 0]
    band = pairs_in_band(queries, refs, 2, 3, chunk_elements=2)
    assert sorted(map(tuple, band.tolist())) == [(0, 1, 3), (1, 0, 3)]  # 0111^0011 = 1: out
    empty_d, empty_i = nearest(queries, np.array([], dtype=np.uint64))
    assert set(empty_d.tolist()) == {255} and set(empty_i.tolist()) == {-1}
    h = histogram(np.array([0, 0, 3, 16, 17, 40]))
    assert h["0"] == 2 and h["3"] == 1 and h["16"] == 1 and h[">16"] == 2


# --- the audit on a synthetic dataset -------------------------------------------------------


def _low_contrast(img: Image.Image, factor: float = 0.06) -> Image.Image:
    arr = np.asarray(img, dtype=np.float64)
    return Image.fromarray(np.clip(128 + (arr - arr.mean()) * factor, 0, 255).astype(np.uint8))


@pytest.fixture
def audited(tmp_path: Path):
    """A dataset with two planted pairs that dHash grouping cannot see (both low-information
    by the std test), split so that each pair straddles splits, then audited."""
    src = tmp_path / "src"
    for k, label in enumerate(("a", "b", "c")):
        make_images(src / label, 16, seed=40 + k)
    x = _low_contrast(_img(101))  # near duplicate: re-scaled + re-encoded copy
    (src / "a" / "x_orig.png").write_bytes(encode(x, "PNG"))
    x_copy = x.resize((56, 56), Image.Resampling.BILINEAR).resize((64, 64))
    (src / "a" / "x_copy.jpg").write_bytes(encode(x_copy, "JPEG", quality=95))
    y = _low_contrast(_img(202))  # pixel-identical, different bytes
    (src / "b" / "y_fast.png").write_bytes(encode(y, "PNG", compress_level=1))
    (src / "b" / "y_small.png").write_bytes(encode(y, "PNG", compress_level=9))

    datasets = tmp_path / "datasets"
    ds = import_dataset(src, "audit", datasets).dataset_id
    recs = {r.relpath: r for r in load_records(datasets, ds)}
    planted = ["a/x_orig.png", "a/x_copy.jpg", "b/y_fast.png", "b/y_small.png"]
    assert recs["b/y_fast.png"].sha256 != recs["b/y_small.png"].sha256
    std_cut = max(recs[p].gray_std for p in planted) + 0.5
    assert min(r.gray_std for p, r in recs.items() if p not in planted) > std_cut

    for seed in range(300):
        cfg = SplitConfig(seed=seed, low_info_std=std_cut)
        result = make_split(datasets, ds, cfg)
        where = {e["relpath"]: e["split"] for e in result.manifest["entries"]}
        x_splits = {where["a/x_orig.png"], where["a/x_copy.jpg"]}
        y_splits = {where["b/y_fast.png"], where["b/y_small.png"]}
        if len(x_splits) == 2 and "train" in x_splits and len(y_splits) == 2:
            break
    else:  # pragma: no cover - would mean the split is not mixing groups at all
        pytest.fail("no seed separated the planted pairs")
    split_dir = result.path
    before = {p.name: p.read_bytes() for p in split_dir.iterdir()}
    audit = audit_split(datasets, ds, result.split_id, seed=0)
    return datasets, ds, result, where, split_dir, before, audit


def test_audit_finds_near_duplicate_hidden_from_grouping(audited) -> None:
    _, _, result, where, _, _, audit = audited
    a = audit.audit
    # the grouping did not see it: separate groups, in different splits
    groups = {e["relpath"]: e["group_id"] for e in result.manifest["entries"]}
    assert groups["a/x_orig.png"] != groups["a/x_copy.jpg"]
    query = next(p for p in ("a/x_orig.png", "a/x_copy.jpg") if where[p] != "train")
    other = "a/x_copy.jpg" if query == "a/x_orig.png" else "a/x_orig.png"
    with open(audit.path / "pairs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    closest = [r for r in rows if r["set"] == "a"]
    hit = next(r for r in closest if r["relpath_a"] == query)
    assert hit["relpath_b"] == other and int(hit["phash_distance"]) <= 6
    assert hit["low_info_a"] == "True" and hit["low_info_b"] == "True"
    assert hit["same_group"] == "False"
    assert closest.index(hit) < 2  # among the very closest cross-split pairs
    # the low-information row of the pHash histogram shows it
    low_hist = a["nearest_train_histograms"]["phash"]["low_information"]
    assert sum(low_hist[str(d)] for d in range(7)) >= 1
    assert a["within_dhash_threshold"]["consistency_check"] == {
        "description": a["within_dhash_threshold"]["consistency_check"]["description"],
        "count": 0,
        "ok": True,
    }


def test_audit_counts_pixel_identical_but_not_exact(audited) -> None:
    *_, audit = audited
    a = audit.audit
    assert a["cross_split_exact_sha256"]["count"] == 0
    pix = a["cross_split_pixel_identical"]
    assert pix["count"] == 1
    assert {m["relpath"] for m in pix["items"][0]["members"]} == {"b/y_fast.png", "b/y_small.png"}


def test_audit_writes_only_into_audits_dir(audited) -> None:
    datasets, ds, result, _, split_dir, before, audit = audited
    after = {p.name: p.read_bytes() for p in split_dir.iterdir()}
    assert after == before  # the write-once split directory is untouched
    assert audit.path == datasets / ds / "audits" / result.split_id
    names = {p.name for p in audit.path.iterdir()}
    assert {"audit.json", "audit.md", "pairs.csv", "sheets"} <= names
    assert not list((datasets / ds / "audits").glob(".tmp-*"))
    assert read_json(audit.path / "audit.json")["split_id"] == result.split_id
    with open(audit.path / "pairs.csv", newline="", encoding="utf-8") as f:
        per_sheet: dict[str, int] = {}
        for row in csv.DictReader(f):
            per_sheet[row["sheet"]] = per_sheet.get(row["sheet"], 0) + 1
    assert per_sheet and max(per_sheet.values()) <= PER_SHEET
    for sheet in audit.sheets:
        with Image.open(sheet) as im:
            assert im.format == "PNG" and im.width >= 600
    md = (audit.path / "audit.md").read_text(encoding="utf-8")
    assert "consistency check" in md and "different scales, indicative only" in md
    field = audit.audit["within_dhash_threshold"]["phash_nearest_within_dhash_threshold_value"]
    assert set(field) == {"count", "description"} and field["count"] >= 0
    assert "phash_nearest_within_same_value" not in (audit.path / "audit.json").read_text()
    # re-running replaces the audit and still leaves the split alone
    audit_split(datasets, ds, result.split_id, seed=0)
    assert {p.name: p.read_bytes() for p in split_dir.iterdir()} == before


def test_pixel_digest_ignores_encoding() -> None:
    img = _img(7)
    png = _reload(encode(img, "PNG", compress_level=0))
    png9 = _reload(encode(img, "PNG", compress_level=9))
    assert pixel_digest(png) == pixel_digest(png9)
    assert pixel_digest(_reload(encode(img, "JPEG", quality=90))) != pixel_digest(png)
    assert (
        inspect_image(encode(img, "PNG"), 10**7).dhash
        == inspect_image(encode(img, "PNG", compress_level=9), 10**7).dhash
    )


# --- oversized-group warnings ---------------------------------------------------------------


@pytest.fixture
def oversized_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    make_images(src / "big", 96, seed=1)
    copy = encode(_img(500))
    for i in range(4):  # one exact-duplicate group of 4: > 2% of 100, but < val target 15
        (src / "big" / f"dup_{i}.png").write_bytes(copy)
    make_images(src / "small", 24, seed=2)
    copy = encode(_img(600))
    for i in range(6):  # 6 of 30: > val target 4.5 and > 2%
        (src / "small" / f"dup_{i}.png").write_bytes(copy)
    return src


def test_oversized_group_warnings_in_split_info_not_manifest(
    oversized_src: Path, tmp_path: Path
) -> None:
    datasets = tmp_path / "datasets"
    ds = import_dataset(oversized_src, "big", datasets).dataset_id
    result = make_split(datasets, ds, SplitConfig(seed=0))
    warnings = result.group_warnings
    assert [(w["label"], w["size"]) for w in warnings] == [("big", 4), ("small", 6)]
    big, small = warnings
    assert big["reasons"] == ["larger than 2% of the class"]
    assert small["reasons"] == ["larger than the class's val target", "larger than 2% of the class"]
    assert small["val_target"] == 4.5 and small["class_total"] == 30
    info = read_json(result.path / "split_info.json")
    assert info["group_size_warnings"] == warnings
    manifest_bytes = (result.path / "manifest.json").read_bytes()
    assert b"group_size_warnings" not in manifest_bytes and b"oversized" not in manifest_bytes
    assert hashlib.sha256(manifest_bytes).hexdigest() == result.manifest_sha256
    again = make_split(datasets, ds, SplitConfig(seed=0))  # existing split: same warnings
    assert again.already_existed and again.group_warnings == warnings


def test_cli_split_prints_group_warnings_and_audit_runs(
    oversized_src: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(tmp_path / "var"))
    assert main(["dataset", "import", str(oversized_src), "--name", "big"]) == 0
    ds = re.search(r"imported dataset (ds-[0-9a-f]{12})", capsys.readouterr().out).group(1)
    assert main(["dataset", "split", ds, "--seed", "0"]) == 0
    out = capsys.readouterr().out
    sp = re.search(r"split (sp-[0-9a-f]{12})", out).group(1)
    assert "oversized groups: 2" in out
    assert re.search(r"small g-[0-9a-f]{16}: 6 images .*val target 4\.5", out)
    assert main(["dataset", "audit-split", ds, sp]) == 0
    out = capsys.readouterr().out
    assert f"audit of {ds} / {sp}" in out and "consistency check: OK" in out
    assert "exact sha256 matches across splits: 0" in out
    assert main(["dataset", "audit-split", ds, "sp-000000000000"]) == 1
    assert "not found" in capsys.readouterr().err
