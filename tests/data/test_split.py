import hashlib
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from visionbench.core.errors import SplitError
from visionbench.core.jsonio import read_json
from visionbench.data.split import SplitConfig, assign_class_groups, make_split
from visionbench.data.store import import_dataset


@pytest.fixture
def messy_ds(messy, messy_config, tmp_path: Path) -> tuple[Path, str]:
    datasets = tmp_path / "datasets"
    return datasets, import_dataset(messy.root, "messy", datasets, messy_config).dataset_id


@pytest.fixture
def balanced_ds(balanced_src: Path, tmp_path: Path) -> tuple[Path, str]:
    datasets = tmp_path / "datasets"
    return datasets, import_dataset(balanced_src, "balanced", datasets).dataset_id


def test_no_leakage_and_groups_kept_together(messy_ds) -> None:
    datasets, ds = messy_ds
    m = make_split(datasets, ds, SplitConfig(seed=0)).manifest
    group_splits, sha_splits = defaultdict(set), defaultdict(set)
    for e in m["entries"]:
        group_splits[e["group_id"]].add(e["split"])
        sha_splits[e["sha256"]].add(e["split"])
    assert all(len(s) == 1 for s in group_splits.values())
    assert all(len(s) == 1 for s in sha_splits.values())
    assert m["leakage_check"] == {
        "groups_checked": len(group_splits),
        "sha256_checked": len(sha_splits),
        "groups_in_multiple_splits": 0,
        "sha256_in_multiple_splits": 0,
    }
    split_of = {e["relpath"]: e["split"] for e in m["entries"]}
    assert split_of["alpha/a0.png"] == split_of["alpha/a0_copy.png"]
    assert split_of["alpha/a1.png"] == split_of["alpha/a1_near.jpg"]
    assert [e["relpath"] for e in m["entries"]] == sorted(split_of)


def test_conflict_group_excluded_and_reported(messy_ds) -> None:
    datasets, ds = messy_ds
    m = make_split(datasets, ds, SplitConfig(seed=0)).manifest
    relpaths = {e["relpath"] for e in m["entries"]}
    assert {"alpha/a2.png", "beta/b_conflict.png"}.isdisjoint(relpaths)
    assert [(e["relpath"], e["reason"]) for e in m["excluded"]] == [
        ("alpha/a2.png", "label_conflict"),
        ("beta/b_conflict.png", "label_conflict"),
    ]
    assert m["excluded_groups"] == 1
    assert len(relpaths) + len(m["excluded"]) == 20


def test_conflict_policy_error(messy_ds) -> None:
    datasets, ds = messy_ds
    with pytest.raises(SplitError, match="1 label-conflict group"):
        make_split(datasets, ds, SplitConfig(seed=0, conflict_policy="error"))


def test_same_seed_byte_identical_manifest(messy_ds) -> None:
    datasets, ds = messy_ds
    first = make_split(datasets, ds, SplitConfig(seed=7))
    data = (first.path / "manifest.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == first.manifest_sha256
    shutil.rmtree(first.path)
    second = make_split(datasets, ds, SplitConfig(seed=7))
    assert not second.already_existed and second.split_id == first.split_id
    assert (second.path / "manifest.json").read_bytes() == data
    third = make_split(datasets, ds, SplitConfig(seed=7))  # existing, identical -> no rewrite
    assert third.already_existed and third.manifest_sha256 == first.manifest_sha256
    info = read_json(second.path / "split_info.json")
    assert info["manifest_sha256"] == first.manifest_sha256 and "created_at" in info
    assert "created_at" not in data.decode()


def test_existing_manifest_tampered_is_refused(messy_ds) -> None:
    datasets, ds = messy_ds
    result = make_split(datasets, ds, SplitConfig(seed=7))
    (result.path / "manifest.json").write_bytes(b"{}\n")
    with pytest.raises(SplitError, match="refusing to overwrite"):
        make_split(datasets, ds, SplitConfig(seed=7))


def test_different_seed_different_assignment(balanced_ds) -> None:
    datasets, ds = balanced_ds
    a = make_split(datasets, ds, SplitConfig(seed=1))
    b = make_split(datasets, ds, SplitConfig(seed=2))
    assert a.split_id != b.split_id
    assign_a = {e["relpath"]: e["split"] for e in a.manifest["entries"]}
    assign_b = {e["relpath"]: e["split"] for e in b.manifest["entries"]}
    assert assign_a.keys() == assign_b.keys() and assign_a != assign_b


def test_balanced_counts_within_one_group_of_target(balanced_ds) -> None:
    datasets, ds = balanced_ds
    for seed in range(5):
        m = make_split(datasets, ds, SplitConfig(seed=seed)).manifest
        for label, counts in m["counts"].items():
            assert counts["total"] == 20
            for split in ("train", "val", "test"):
                target = m["target_counts"][label][split]
                assert abs(counts[split] - target) <= 1, (seed, label, split)
        assert m["split_totals"] == {"train": 42, "val": 9, "test": 9}


def test_missing_class_error_and_allow_flag(tmp_path: Path, make_images) -> None:
    src = tmp_path / "src"
    make_images(src / "big", 20, seed=1)
    make_images(src / "tiny", 3, seed=2)
    datasets = tmp_path / "datasets"
    ds = import_dataset(src, "skewed", datasets).dataset_id
    with pytest.raises(SplitError) as exc:
        make_split(datasets, ds, SplitConfig(seed=0))
    assert "class 'tiny' has 0 items in test (train=2, val=1, test=0)" in str(exc.value)
    assert "--allow-missing-classes" in str(exc.value)
    assert not (datasets / ds / "splits").exists() or not any((datasets / ds / "splits").iterdir())

    m = make_split(datasets, ds, SplitConfig(seed=0, allow_missing_classes=True)).manifest
    assert m["warnings"] == ["class 'tiny' has 0 items in test (train=2, val=1, test=0)"]
    assert m["counts"]["tiny"] == {"train": 2, "val": 1, "test": 0, "total": 3}


def test_largest_deficit_assignment_rule() -> None:
    from visionbench.data.grouping import DuplicateGroup

    groups = [DuplicateGroup(f"g-{i}", (f"c/{i}.png",), ("c",), "single") for i in range(3)]
    fractions = [Fraction(7, 10), Fraction(3, 20), Fraction(3, 20)]
    assignment, achieved, targets = assign_class_groups(groups, fractions, seed=0, label="c")
    # targets 2.1/0.45/0.45: train, train, then val beats test on the tie
    assert achieved == [2, 1, 0] and targets == [Fraction(21, 10), Fraction(9, 20), Fraction(9, 20)]
    sizes = [DuplicateGroup("g-big", ("c/a", "c/b", "c/c", "c/d"), ("c",), "exact")] + groups
    _, achieved, _ = assign_class_groups(sizes, fractions, seed=0, label="c")
    assert sum(achieved) == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"seed": True},
        {"seed": 0, "fractions": (0.5, 0.5, 0.0)},
        {"seed": 0, "fractions": (0.7, 0.2, 0.2)},
        {"seed": 0, "fractions": (0.5, 0.5)},
        {"seed": 0, "dhash_threshold": -1},
        {"seed": 0, "low_info_std": -0.1},
        {"seed": 0, "conflict_policy": "relabel"},
    ],
)
def test_split_config_validation(kwargs) -> None:
    with pytest.raises(SplitError):
        SplitConfig(**kwargs)


def test_split_id_depends_on_full_config(messy_ds) -> None:
    datasets, ds = messy_ds
    ids = {
        make_split(datasets, ds, cfg).split_id
        for cfg in (
            SplitConfig(seed=0),
            SplitConfig(seed=0, dhash_threshold=4),
            SplitConfig(seed=0, fractions=(0.6, 0.2, 0.2)),
        )
    }
    assert len(ids) == 3
