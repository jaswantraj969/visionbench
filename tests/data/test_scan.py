import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vb_synth import SMALL_MAX_PIXELS, encode, synthetic_image, try_dir_link

from visionbench.core.errors import DatasetValidationError
from visionbench.data.records import IssueCategory
from visionbench.data.scan import (
    ImageRejected,
    ScanConfig,
    check_requirements,
    class_name_problem,
    inspect_image,
    scan_source,
)


def test_every_issue_category_detected_with_relpaths(messy, messy_config) -> None:
    result = scan_source(messy.root, messy_config)
    found: dict[str, list[str]] = {str(c): [] for c in IssueCategory}
    for issue in result.issues:
        found[str(issue.category)].append(issue.relpath)
    assert {k: sorted(v) for k, v in found.items()} == {
        k: sorted(v) for k, v in messy.expected.items()
    }


def test_file_symlink_reported_not_followed(messy, messy_config) -> None:
    if not messy.file_link:
        pytest.skip(
            "cannot create file symlinks here (Windows without symlink privilege / "
            "Developer Mode); directory links are covered via an NTFS junction"
        )
    result = scan_source(messy.root, messy_config)
    assert "alpha/link.png" in [i.relpath for i in result.issues if i.category == "symlink"]
    assert "alpha/link.png" not in [r.relpath for r in result.records]


def test_directory_link_reported_not_followed(messy, messy_config) -> None:
    if messy.dir_link is None:
        pytest.skip("could not create a directory symlink or junction")
    result = scan_source(messy.root, messy_config)
    assert "linked_class" in [i.relpath for i in result.issues if i.category == "symlink"]
    assert "linked_class" not in result.classes
    assert not any(r.relpath.startswith("linked_class/") for r in result.records)


def test_accepted_images_and_classes(messy, messy_config) -> None:
    result = scan_source(messy.root, messy_config)
    assert result.classes == ["alpha", "beta", "gamma"]
    assert result.class_counts() == {"alpha": 8, "beta": 7, "gamma": 5}
    relpaths = [r.relpath for r in result.records]
    assert relpaths == sorted(relpaths)
    assert all("\\" not in p and p.count("/") == 1 for p in relpaths)
    assert "alpha/a1_near.jpg" in relpaths and "alpha/uniform_red.png" in relpaths


def test_truncated_jpeg_passes_verify_but_fails_load(messy) -> None:
    data = (messy.root / "alpha" / "truncated.jpg").read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.verify()  # verify() alone does NOT detect this truncation
    with pytest.raises(ImageRejected) as exc:
        inspect_image(data, 50_000_000)
    assert exc.value.category == IssueCategory.UNREADABLE_OR_CORRUPT
    assert "truncated" in exc.value.detail


def test_pixel_limit_is_explicit_and_restored(messy) -> None:
    before = Image.MAX_IMAGE_PIXELS
    data = (messy.root / "alpha" / "big_warn.png").read_bytes()
    with pytest.raises(ImageRejected) as exc:
        inspect_image(data, SMALL_MAX_PIXELS)
    assert exc.value.category == IssueCategory.TOO_LARGE_PIXELS
    assert before == Image.MAX_IMAGE_PIXELS
    assert inspect_image(data, 12_100).width == 110  # exactly at the limit is accepted


def test_wrong_content_for_extension_is_corrupt() -> None:
    gif = encode(synthetic_image(np.random.default_rng(0)), "GIF")
    with pytest.raises(ImageRejected) as exc:
        inspect_image(gif, 50_000_000)  # GIF bytes are not an allowed format
    assert exc.value.category == IssueCategory.UNREADABLE_OR_CORRUPT


def test_unicode_nfc_and_case_collisions(tmp_path: Path, make_images) -> None:
    src = tmp_path / "src"
    make_images(src / "a", 3, seed=1)
    make_images(src / "b", 3, seed=2)
    img = encode(Image.new("RGB", (8, 8), (1, 2, 3)))
    (src / "a" / "café.png").write_bytes(img)  # NFC
    (src / "a" / "café.png").write_bytes(img)  # NFD spelling of the same name
    if len(os.listdir(src / "a")) != 5:
        pytest.skip("filesystem normalizes Unicode names; cannot create both spellings")
    result = scan_source(src, ScanConfig())
    collisions = [i for i in result.issues if i.category == IssueCategory.PATH_COLLISION]
    assert [i.relpath for i in collisions] == ["a/café.png"]
    assert sum(r.relpath == "a/café.png" for r in result.records) == 1


def test_class_name_rules() -> None:
    assert class_name_problem("cats") is None
    assert class_name_problem("  ") == "class name is empty"
    assert class_name_problem(".git") == "class name starts with a dot"
    assert class_name_problem("a\\b") is not None
    assert class_name_problem("a\tb") is not None


def test_requires_two_classes(tmp_path: Path, make_images) -> None:
    make_images(tmp_path / "src" / "only", 5)
    result = scan_source(tmp_path / "src", ScanConfig())
    with pytest.raises(DatasetValidationError, match="at least 2 classes"):
        check_requirements(result, ScanConfig())


def test_min_images_per_class_lists_offenders(tmp_path: Path, make_images) -> None:
    src = tmp_path / "src"
    make_images(src / "big", 5, seed=1)
    make_images(src / "small", 2, seed=2)
    (src / "empty").mkdir()
    result = scan_source(src, ScanConfig())
    with pytest.raises(DatasetValidationError) as exc:
        check_requirements(result, ScanConfig(min_images_per_class=3))
    assert "'empty' (0)" in str(exc.value) and "'small' (2)" in str(exc.value)
    assert "'big'" not in str(exc.value)


def test_missing_source(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="does not exist"):
        scan_source(tmp_path / "nope", ScanConfig())


def test_linked_source_directory_rejected(tmp_path: Path, make_images) -> None:
    make_images(tmp_path / "real" / "a", 3)
    kind = try_dir_link(tmp_path / "real", tmp_path / "linked")
    if kind is None:
        pytest.skip("could not create a directory symlink or junction")
    with pytest.raises(DatasetValidationError, match="is a link"):
        scan_source(tmp_path / "linked", ScanConfig())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_image_pixels": 0},
        {"min_images_per_class": 0},
        {"dhash_threshold": 65},
        {"low_info_std": -1},
        {"samples_per_class": -1},
    ],
)
def test_scan_config_validation(kwargs) -> None:
    with pytest.raises(DatasetValidationError):
        ScanConfig(**kwargs)
