"""EXIF orientation policy: every pixel consumer sees the EXIF-transposed image."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vb_synth import encode, synthetic_image

from visionbench.core.errors import DatasetValidationError
from visionbench.data.hashing import hamming
from visionbench.data.imageio import decode_image, load_rgb
from visionbench.data.scan import inspect_image
from visionbench.data.split import SplitConfig, make_split
from visionbench.data.store import (
    DatasetFormatError,
    import_dataset,
    load_dataset,
    load_records,
    verify_dataset,
)
from visionbench.training.data import load_image

ORIENTATION_TAG = 0x0112
MAX_PIXELS = 10_000_000


def _stored_sideways(width: int = 48, height: int = 32, seed: int = 3) -> tuple[bytes, Image.Image]:
    """A JPEG whose stored pixels are sideways, with EXIF Orientation=6 ("rotate 90° CW to
    display"), and the image as it is meant to be displayed."""
    stored = synthetic_image(np.random.default_rng(seed), size=64).resize((width, height))
    exif = Image.Exif()
    exif[ORIENTATION_TAG] = 6
    data = encode(stored, "JPEG", quality=95, exif=exif.tobytes())
    with Image.open(io.BytesIO(data)) as im:
        assert im.getexif()[ORIENTATION_TAG] == 6 and im.size == (width, height)
    return data, decode_image(data).image


def test_exif_rotated_jpeg_equals_physically_rotated_copy(tmp_path: Path) -> None:
    data, displayed = _stored_sideways()
    # the physically rotated copy: the displayed pixels, stored losslessly without EXIF
    rotated = encode(displayed, "PNG")
    with Image.open(io.BytesIO(rotated)) as im:
        assert ORIENTATION_TAG not in im.getexif()

    a, b = inspect_image(data, MAX_PIXELS), inspect_image(rotated, MAX_PIXELS)
    assert (a.width, a.height) == (b.width, b.height) == (32, 48)  # transposed, not 48x32
    assert a.dhash == b.dhash and a.gray_std == b.gray_std

    (tmp_path / "exif.jpg").write_bytes(data)
    (tmp_path / "rotated.png").write_bytes(rotated)
    pa, pb = load_image(tmp_path / "exif.jpg"), load_image(tmp_path / "rotated.png")
    assert pa.size == pb.size == (32, 48)
    assert np.array_equal(np.asarray(pa), np.asarray(pb))
    assert np.array_equal(np.asarray(load_rgb(tmp_path / "exif.jpg")), np.asarray(pa))


def test_lossy_rotated_copy_is_a_near_match(tmp_path: Path) -> None:
    data, displayed = _stored_sideways(seed=11)
    lossy = encode(displayed, "JPEG", quality=95)  # re-encoded: pixels differ slightly
    a, b = inspect_image(data, MAX_PIXELS), inspect_image(lossy, MAX_PIXELS)
    assert (a.width, a.height) == (b.width, b.height)
    assert hamming(a.dhash, b.dhash) <= 4


def test_image_without_exif_is_unchanged() -> None:
    img = synthetic_image(np.random.default_rng(5), size=64).resize((40, 24))
    data = encode(img, "PNG")
    decoded = decode_image(data)
    assert decoded.image.size == (40, 24) and decoded.mode == "RGB" and decoded.format == "PNG"
    assert np.array_equal(np.asarray(decoded.image), np.asarray(img))
    info = inspect_image(data, MAX_PIXELS)
    assert (info.width, info.height) == (40, 24)


def _import(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "src"
    for label, seed in (("a", 1), ("b", 2)):
        (src / label).mkdir(parents=True)
        for i in range(4):
            img = synthetic_image(np.random.default_rng(seed * 10 + i), size=64)
            (src / label / f"{i}.png").write_bytes(encode(img, "PNG"))
    data, _ = _stored_sideways()
    (src / "a" / "sideways.jpg").write_bytes(data)
    datasets = tmp_path / "datasets"
    return datasets, import_dataset(src, "exif", datasets).dataset_id


def test_import_records_transposed_dimensions(tmp_path: Path) -> None:
    datasets, ds = _import(tmp_path)
    rec = {r.relpath: r for r in load_records(datasets, ds)}["a/sideways.jpg"]
    assert (rec.width, rec.height) == (32, 48)
    meta = load_dataset(datasets, ds)
    assert meta["orientation"] == "exif_transposed" and meta["hash_version"] == 2


@pytest.mark.parametrize("file_name", ["dataset.json", "files.json"])
def test_older_format_is_refused_with_reimport_hint(tmp_path: Path, file_name: str) -> None:
    datasets, ds = _import(tmp_path)
    path = datasets / ds / file_name
    data = json.loads(path.read_text(encoding="utf-8"))
    if file_name == "dataset.json":
        data["format_version"] = 1
        del data["orientation"], data["hash_version"]
    else:
        data = {"records": data["records"]}  # files.json as written by format 1
    path.write_text(json.dumps(data), encoding="utf-8")
    calls = [
        lambda: load_dataset(datasets, ds),
        lambda: load_records(datasets, ds),
        lambda: verify_dataset(datasets, ds),
        lambda: make_split(datasets, ds, SplitConfig(seed=0)),
    ]
    raised = 0
    for call in calls:
        try:
            call()
        except DatasetFormatError as exc:
            assert "Re-import the source folder" in str(exc) and file_name in str(exc)
            assert isinstance(exc, DatasetValidationError)
            raised += 1
    assert raised >= 3  # every reader of that file refuses it
