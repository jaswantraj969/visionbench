from dataclasses import replace

from visionbench.data.grouping import build_groups, is_low_information, make_group_id
from visionbench.data.hashing import ALL_ONES, format_hash, hamming
from visionbench.data.records import FileRecord
from visionbench.data.scan import scan_source


def _groups(messy, messy_config, **kw):
    records = scan_source(messy.root, messy_config).records
    params = {"dhash_threshold": 5, "low_info_std": 3.0} | kw
    return records, build_groups(records, **params)


def test_exact_and_near_duplicates_grouped(messy, messy_config) -> None:
    _, grouping = _groups(messy, messy_config)
    by_members = {g.members: g for g in grouping.multi_member}
    assert by_members[("alpha/a0.png", "alpha/a0_copy.png")].kind == "exact"
    assert by_members[("alpha/a1.png", "alpha/a1_near.jpg")].kind == "near"
    conflict = by_members[("alpha/a2.png", "beta/b_conflict.png")]
    assert conflict.conflict and conflict.labels == ("alpha", "beta")
    assert [g.members for g in grouping.conflicts] == [conflict.members]
    assert len(grouping.multi_member) == 3
    assert all(len(g.members) == 1 for g in grouping.groups if g not in grouping.multi_member)


def test_uniform_images_not_grouped_despite_identical_hash(messy, messy_config) -> None:
    records, grouping = _groups(messy, messy_config)
    rec = {r.relpath: r for r in records}
    red, blue = rec["alpha/uniform_red.png"], rec["beta/uniform_blue.png"]
    assert red.sha256 != blue.sha256
    assert hamming(red.dhash_int, blue.dhash_int) == 0  # would match without the guard
    assert grouping.low_information == ("alpha/uniform_red.png", "beta/uniform_blue.png")
    assert grouping.group_of[red.relpath] != grouping.group_of[blue.relpath]


def test_low_information_guard_std_and_degenerate_hash() -> None:
    base = FileRecord(
        "a/x.png", "a", "0" * 64, 1, 8, 8, "RGB", "PNG", format_hash(0x0F0F_0F0F_0F0F_0F0F), 40.0
    )
    assert not is_low_information(base, 3.0)
    assert is_low_information(replace(base, gray_std=2.9), 3.0)
    assert is_low_information(replace(base, dhash=format_hash(0)), 3.0)
    assert is_low_information(replace(base, dhash=format_hash(ALL_ONES)), 3.0)


def test_low_info_images_still_exact_matched() -> None:
    a = FileRecord("a/x.png", "a", "f" * 64, 1, 8, 8, "RGB", "PNG", format_hash(0), 0.0)
    b = replace(a, relpath="a/y.png")
    grouping = build_groups([a, b], dhash_threshold=5, low_info_std=3.0)
    assert len(grouping.groups) == 1 and grouping.groups[0].kind == "exact"


def test_threshold_zero_still_groups_identical_hashes(messy, messy_config) -> None:
    _, grouping = _groups(messy, messy_config, dhash_threshold=0)
    assert ("alpha/a1.png", "alpha/a1_near.jpg") in [g.members for g in grouping.multi_member]


def test_group_ids_stable_and_order_independent(messy, messy_config) -> None:
    records, grouping = _groups(messy, messy_config)
    again = build_groups(list(reversed(records)), dhash_threshold=5, low_info_std=3.0)
    assert again == grouping
    assert make_group_id(["b", "a"]) == make_group_id(["a", "b"])
    assert len({g.group_id for g in grouping.groups}) == len(grouping.groups)


def test_group_kind_mixed() -> None:
    a = FileRecord(
        "a/x.png", "a", "1" * 64, 1, 8, 8, "RGB", "PNG", format_hash(0x0F0F_0F0F_0F0F_0F0F), 40.0
    )
    copy = replace(a, relpath="a/x_copy.png")
    near = replace(
        a, relpath="a/x_near.jpg", sha256="2" * 64, dhash=format_hash(0x0F0F_0F0F_0F0F_0F0E)
    )
    grouping = build_groups([a, copy, near], dhash_threshold=5, low_info_std=3.0)
    assert [g.kind for g in grouping.groups] == ["mixed"]
