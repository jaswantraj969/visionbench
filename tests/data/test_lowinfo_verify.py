"""M2 step 0: popcount low-information guard, hash-ordered splits, `dataset verify`."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import vb_synth

from visionbench.cli import main
from visionbench.data.grouping import build_groups, is_low_information
from visionbench.data.hashing import format_hash, hamming
from visionbench.data.records import FileRecord
from visionbench.data.scan import ScanConfig, scan_source
from visionbench.data.split import SplitConfig, group_order_key, make_split
from visionbench.data.store import import_dataset, verify_dataset

# --- 0b: popcount guard -------------------------------------------------------------------


def test_popcount_rule_unit() -> None:
    base = FileRecord("a/x.png", "a", "0" * 64, 1, 8, 8, "RGB", "PNG", format_hash(0), 40.0)
    for bits, expected in ((7, True), (8, False), (56, False), (57, True)):
        rec = replace(base, dhash=format_hash((1 << bits) - 1))
        assert is_low_information(rec, 3.0) is expected, bits
    rec = replace(base, dhash=format_hash((1 << 7) - 1))
    assert not is_low_information(rec, 3.0, popcount_min=0, popcount_max=64)


def test_gradient_and_flat_noise_excluded_and_reported(tmp_path: Path) -> None:
    src = vb_synth.build_low_info(tmp_path / "src")
    records = {r.relpath: r for r in scan_source(src, ScanConfig()).records}
    grad_a, grad_b = records["sky/gradient_a.png"], records["sea/gradient_b.png"]
    flat = records["sky/flat_noise.png"]
    # the gradients are only caught by the popcount rule: high std, sparse non-zero hash
    for g in (grad_a, grad_b):
        assert g.gray_std > 3.0 and 0 < g.dhash_int.bit_count() < 8
    assert hamming(grad_a.dhash_int, grad_b.dhash_int) <= 5  # would match without the guard
    assert flat.gray_std < 3.0  # caught by the std rule

    result = import_dataset(src, "lowinfo", tmp_path / "datasets")
    dup = result.report["duplicates"]
    assert dup["perceptual_check_skipped_low_information"]["relpaths"] == [
        "sea/gradient_b.png",
        "sky/flat_noise.png",
        "sky/gradient_a.png",
    ]
    assert dup["low_info_popcount_min"] == 8 and dup["low_info_popcount_max"] == 56
    assert dup["multi_member_groups"] == 0 and dup["label_conflict_groups"] == []

    # Control: with the popcount rule disabled the two gradients form a false cross-class group.
    unguarded = build_groups(
        list(records.values()),
        dhash_threshold=5,
        low_info_std=3.0,
        popcount_min=0,
        popcount_max=64,
    )
    assert [g.members for g in unguarded.conflicts] == [
        ("sea/gradient_b.png", "sky/gradient_a.png")
    ]


# --- 0a: hash-based split ordering ---------------------------------------------------------


def test_group_order_key_is_plain_sha256() -> None:
    expected = hashlib.sha256(b"7|cats|g-0123").hexdigest()
    assert group_order_key(7, "cats", "g-0123") == expected


def test_split_does_not_use_numpy_rng(balanced_src: Path, tmp_path: Path, monkeypatch) -> None:
    datasets = tmp_path / "datasets"
    ds = import_dataset(balanced_src, "b", datasets).dataset_id

    def _forbidden(*args, **kwargs):
        raise AssertionError("split ordering must not use a NumPy RNG")

    monkeypatch.setattr(np.random, "default_rng", _forbidden)
    monkeypatch.setattr(np.random, "Generator", _forbidden)
    m = make_split(datasets, ds, SplitConfig(seed=3)).manifest
    assert m["algorithm"] == "vb-split-v2"
    assert m["config"]["low_info_popcount_min"] == 8
    assert m["leakage_check"]["groups_in_multiple_splits"] == 0


# --- 0d: dataset verify --------------------------------------------------------------------


@pytest.fixture
def imported(balanced_src: Path, tmp_path: Path) -> tuple[Path, str, Path]:
    datasets = tmp_path / "datasets"
    result = import_dataset(balanced_src, "b", datasets)
    return datasets, result.dataset_id, result.path


def test_verify_ok_fast_and_full(imported) -> None:
    datasets, ds, _ = imported
    for full in (False, True):
        result = verify_dataset(datasets, ds, full=full)
        assert result.ok and result.files_checked == 60 and result.full is full


def test_verify_detects_size_missing_extra_and_content(imported) -> None:
    datasets, ds, root = imported
    raw = root / "raw" / "class_0"
    data = (raw / "img_000.png").read_bytes()
    (raw / "img_000.png").write_bytes(data + b"x")  # size change
    (raw / "img_001.png").unlink()  # missing
    (raw / "stray.png").write_bytes(b"stray")  # unrecorded
    same_size = bytearray((raw / "img_002.png").read_bytes())
    same_size[-1] ^= 0xFF
    (raw / "img_002.png").write_bytes(bytes(same_size))  # same size, different content

    fast = verify_dataset(datasets, ds)
    assert [p[0] for p in fast.problems] == [
        "class_0/img_000.png",
        "class_0/img_001.png",
        "class_0/stray.png",
    ]
    full = verify_dataset(datasets, ds, full=True)
    assert ("class_0/img_002.png", "sha256 differs from recorded value") in full.problems
    assert len(full.problems) == 4


def test_verify_detects_fingerprint_mismatch(imported) -> None:
    datasets, ds, root = imported
    files = json.loads((root / "files.json").read_text(encoding="utf-8"))
    files["records"][0]["sha256"] = "0" * 64
    (root / "files.json").write_text(json.dumps(files), encoding="utf-8")
    result = verify_dataset(datasets, ds)
    assert ("<dataset>", "fingerprint of files.json does not match dataset.json") in result.problems


def test_verify_cli_exit_codes(imported, monkeypatch, capsys) -> None:
    datasets, ds, root = imported
    monkeypatch.setenv("VISIONBENCH_DATA_ROOT", str(datasets.parent))
    assert main(["dataset", "verify", ds]) == 0
    assert "OK: snapshot matches its records" in capsys.readouterr().out
    (root / "raw" / "class_1" / "img_005.png").unlink()
    assert main(["dataset", "verify", ds, "--full"]) == 1
    captured = capsys.readouterr()
    assert "MISMATCH class_1/img_005.png: missing" in captured.out
    assert "1 problem(s) found" in captured.err
