import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from vb_synth import encode, synthetic_image

from visionbench.core.errors import DatasetExistsError, DatasetNotFoundError, DatasetValidationError
from visionbench.core.jsonio import read_json
from visionbench.data.records import compute_fingerprint
from visionbench.data.scan import ScanConfig, scan_source
from visionbench.data.store import (
    dataset_dir,
    import_dataset,
    list_datasets,
    load_records,
    load_report,
    validate_name,
)


def test_import_writes_snapshot_metadata_and_report(messy, messy_config, tmp_path: Path) -> None:
    datasets = tmp_path / "var" / "datasets"
    result = import_dataset(messy.root, "messy", datasets, messy_config)
    root = result.path
    assert root == datasets / result.dataset_id and result.dataset_id.startswith("ds-")

    meta = read_json(root / "dataset.json")
    assert meta["name"] == "messy" and meta["classes"] == ["alpha", "beta", "gamma"]
    assert meta["class_counts"] == {"alpha": 8, "beta": 7, "gamma": 5}
    assert meta["source_path"] == str(messy.root.resolve())
    assert meta["validation"]["issue_counts"]["zero_byte"] == 1

    # snapshot holds exactly the accepted files, byte-identical to their recorded sha256
    records = load_records(datasets, result.dataset_id)
    snap = sorted(
        p.relative_to(root / "raw").as_posix() for p in (root / "raw").rglob("*") if p.is_file()
    )
    assert snap == [r.relpath for r in records]
    for r in records:
        data = (root / "raw" / r.relpath).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r.sha256
    assert compute_fingerprint(records) == meta["fingerprint"]

    report = load_report(datasets, result.dataset_id)
    assert report["total_images"] == 20 and report["imbalance_ratio"] == 1.6
    assert report["width"] == {"min": 48, "median": 64, "max": 64}
    assert report["mode_counts"] == {"RGB": 20}
    assert report["format_counts"] == {"JPEG": 1, "PNG": 19}
    assert report["issues"]["too_large_pixels"]["count"] == 2
    assert [i["relpath"] for i in report["issues"]["nested_directory"]["items"]] == ["alpha/subdir"]
    dup = report["duplicates"]
    assert dup["multi_member_groups"] == 3 and dup["exact_duplicate_files"] == 2
    assert dup["perceptual_check_skipped_low_information"] == {
        "count": 2,
        "relpaths": ["alpha/uniform_red.png", "beta/uniform_blue.png"],
    }
    assert [g["members"] for g in dup["label_conflict_groups"]] == [
        ["alpha/a2.png", "beta/b_conflict.png"]
    ]
    samples = report["samples"]["relpaths"]
    assert {c: len(v) for c, v in samples.items()} == {"alpha": 8, "beta": 7, "gamma": 5}
    assert list_datasets(datasets)[0]["id"] == result.dataset_id
    assert not list(datasets.glob(".tmp-*"))


def test_samples_are_seeded_and_capped(tmp_path: Path, make_images) -> None:
    src = tmp_path / "src"
    make_images(src / "a", 12, seed=1)
    make_images(src / "b", 12, seed=2)
    cfg = ScanConfig(samples_per_class=4)
    r1 = import_dataset(src, "one", tmp_path / "d1", cfg).report["samples"]["relpaths"]
    r2 = import_dataset(src, "one", tmp_path / "d2", cfg).report["samples"]["relpaths"]
    assert r1 == r2 and all(len(v) == 4 for v in r1.values())


def test_import_refuses_overwrite(messy, messy_config, tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    first = import_dataset(messy.root, "messy", datasets, messy_config)
    before = (first.path / "dataset.json").read_bytes()
    with pytest.raises(DatasetExistsError, match="refusing to overwrite"):
        import_dataset(messy.root, "messy", datasets, messy_config)
    assert (first.path / "dataset.json").read_bytes() == before
    assert not list(datasets.glob(".tmp-*"))
    other = import_dataset(messy.root, "messy-v2", datasets, messy_config)
    assert other.dataset_id != first.dataset_id


def test_failed_validation_leaves_nothing(tmp_path: Path, make_images) -> None:
    src = tmp_path / "src"
    make_images(src / "a", 5, seed=1)
    make_images(src / "b", 1, seed=2)
    datasets = tmp_path / "datasets"
    with pytest.raises(DatasetValidationError, match="'b' \\(1\\)"):
        import_dataset(src, "bad", datasets)
    assert list(datasets.iterdir()) == []


def test_import_never_follows_links(messy, messy_config, tmp_path: Path) -> None:
    if not (messy.file_link or messy.dir_link):
        pytest.skip("no symlink or junction could be created on this system")
    result = import_dataset(messy.root, "messy", tmp_path / "datasets", messy_config)
    raw = result.path / "raw"
    assert not (raw / "linked_class").exists()
    assert not (raw / "alpha" / "link.png").exists()
    assert all(not p.is_symlink() for p in raw.rglob("*"))
    assert result.dataset["classes"] == ["alpha", "beta", "gamma"]


def test_fingerprint_invariant_to_creation_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    files = {f"{c}/img_{i}.png": encode(synthetic_image(rng)) for c in "xyz" for i in range(3)}
    fingerprints = []
    for name, order in (("fwd", sorted(files)), ("rev", sorted(files, reverse=True))):
        root = tmp_path / name
        for rel in order:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(files[rel])
        fingerprints.append(compute_fingerprint(scan_source(root, ScanConfig()).records))
    assert fingerprints[0] == fingerprints[1]
    records = scan_source(tmp_path / "fwd", ScanConfig()).records
    assert compute_fingerprint(list(reversed(records))) == fingerprints[0]
    shutil.copy(tmp_path / "fwd" / "x" / "img_0.png", tmp_path / "fwd" / "y" / "extra.png")
    assert (
        compute_fingerprint(scan_source(tmp_path / "fwd", ScanConfig()).records)
        != (fingerprints[0])
    )


def test_dataset_lookup_rejects_bad_ids(tmp_path: Path) -> None:
    for bad in ("../etc", "ds-XYZ", "ds-0123456789ab"):
        with pytest.raises(DatasetNotFoundError):
            dataset_dir(tmp_path, bad)
    assert list_datasets(tmp_path / "missing") == []


def test_validate_name() -> None:
    assert validate_name("  cifar small ") == "cifar small"
    for bad in ("", "   ", "x" * 101, "a\nb"):
        with pytest.raises(DatasetValidationError):
            validate_name(bad)
